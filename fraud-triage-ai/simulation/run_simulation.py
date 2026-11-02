"""
Closed-Loop Multi-Day Simulation Harness
Answers the Main Research Question:
  "Can a feedback-based ranking system identify more fraud accounts than the
   existing paper's ranking method using the same investigation budget?"

Simulates 30 Days of Daily Triage (Budget K = 40 accounts/day):
  - Baseline (Static Paper Distance Ranking): 40 slots/day ranked by GMM FRC distance, no retraining.
  - Proposed System: LightGBM ranker with 32 Exploitation + 8 Exploration slots,
                     logging investigator labels and retraining every 3 days.
"""

import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT, "temp_mpl")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.gmm_clustering import GMMFraudClusterer
from src.rankers import OriginalPaperDistanceRanker, LightGBMClosedLoopRanker
from src.triage_engine import PrioritizationEngine
from src.feedback_loop import FeedbackManager, retrain_model_with_feedback

def run_simulation(
    days: int = 20,
    daily_budget: int = 40,
    retrain_frequency: int = 3,
    output_dir: str = None
):
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "simulation")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 75)
    print("CLOSED-LOOP MULTI-DAY INVESTIGATION SIMULATION")
    print(f"Investigation Days: {days} | Daily Capacity: {daily_budget} accounts/day")
    print(f"Total Budget Per System: {days * daily_budget} reviews")
    print("=" * 75)

    data_path = os.path.join(PROJECT_ROOT, "data", "accounts_data.csv")
    df_raw = pd.read_csv(data_path)
    total_fraud = int(df_raw["is_fraud"].sum())

    # Step 2: GMM Clustering
    gmm = GMMFraudClusterer(n_components=3, random_state=42)
    # Give cold-start initial 30 verified seeds
    initial_labels = np.zeros(len(df_raw))
    initial_labels[df_raw[df_raw["is_fraud"] == 1].index[:30]] = 1
    gmm.fit(df_raw, initial_fraud_labels=initial_labels)
    df_transformed = gmm.transform(df_raw)

    # -------------------------------------------------------------
    # SIMULATION 1: Baseline (Original Paper Static Distance Ranking)
    # -------------------------------------------------------------
    print("\n[1/3] Simulating Baseline: Original Paper Distance Ranking (Static)...")
    paper_ranker = OriginalPaperDistanceRanker()
    paper_scores = paper_ranker.score(df_transformed)

    paper_engine = PrioritizationEngine(
        daily_capacity=daily_budget,
        exploration_ratio=0.0, # 100% exploitation of static distance
        random_state=42
    )

    baseline_daily_logs = []
    baseline_cum_fraud = 0

    for day in range(1, days + 1):
        batch = paper_engine.select_daily_batch(
            df_transformed,
            risk_scores=paper_scores,
            exploration_mode="random"
        )
        frauds_found = int(batch["is_fraud"].sum())
        baseline_cum_fraud += frauds_found
        p_day = frauds_found / len(batch)

        baseline_daily_logs.append({
            "day": day,
            "method": "Original Paper (Static)",
            "budget_reviewed": len(batch),
            "frauds_found": frauds_found,
            "precision": round(p_day, 4),
            "cumulative_frauds": baseline_cum_fraud,
            "cumulative_recall": round(baseline_cum_fraud / total_fraud, 4)
        })

    # -------------------------------------------------------------
    # SIMULATION 2: Proposed Closed-Loop LightGBM (Budget-Aware: 80/20)
    # -------------------------------------------------------------
    print("[2/3] Simulating Proposed: LightGBM + 80% Exploit / 20% Explore + Closed-Loop Retraining...")
    proposed_ranker = LightGBMClosedLoopRanker(random_state=42)
    # Initial cold-start seed training
    seed_pool = df_transformed.iloc[:100]
    proposed_ranker.fit(seed_pool, seed_pool["is_fraud"].values)

    proposed_engine = PrioritizationEngine(
        daily_capacity=daily_budget,
        exploration_ratio=0.20, # 32 exploit + 8 explore
        random_state=42
    )

    # We will accumulate feedback dataframe for retraining
    feedback_records = []
    proposed_daily_logs = []
    proposed_cum_fraud = 0

    for day in range(1, days + 1):
        # 1. Score remaining accounts with current model
        current_scores = proposed_ranker.score(df_transformed)

        # 2. Select batch (32 exploit + 8 explore)
        batch = proposed_engine.select_daily_batch(
            df_transformed,
            risk_scores=current_scores,
            exploration_mode="uncertainty"
        )

        frauds_found = int(batch["is_fraud"].sum())
        proposed_cum_fraud += frauds_found
        p_day = frauds_found / len(batch)

        # 3. Simulate Investigator Review & Decision Logging
        for _, row in batch.iterrows():
            decision = "Fraud" if row["is_fraud"] == 1 else "Genuine"
            feedback_records.append({
                "account_id": row["account_id"],
                "business_name": row["business_name"],
                "day_number": day,
                "risk_score": row["risk_score"],
                "selection_type": row["selection_type"],
                "ranking_method": "Proposed LightGBM",
                "investigator_decision": decision,
                "actual_is_fraud": row["is_fraud"]
            })

        proposed_daily_logs.append({
            "day": day,
            "method": "Proposed LightGBM (Closed-Loop)",
            "budget_reviewed": len(batch),
            "frauds_found": frauds_found,
            "precision": round(p_day, 4),
            "cumulative_frauds": proposed_cum_fraud,
            "cumulative_recall": round(proposed_cum_fraud / total_fraud, 4)
        })

        # 4. Closed-loop retraining every retrain_frequency days
        if day % retrain_frequency == 0 and day < days:
            fb_df = pd.DataFrame(feedback_records)
            proposed_ranker = retrain_model_with_feedback(
                proposed_ranker,
                df_transformed,
                fb_df
            )
            print(f" -> Day {day}: Retrained LightGBM model with {len(fb_df)} verified investigator labels.")

    # Save to SQLite and CSV
    fb_manager = FeedbackManager()
    fb_manager.save_investigation_decisions(pd.DataFrame(feedback_records))

    df_base = pd.DataFrame(baseline_daily_logs)
    df_prop = pd.DataFrame(proposed_daily_logs)
    combined_sim = pd.concat([df_base, df_prop], ignore_index=True)
    sim_csv_path = os.path.join(output_dir, "simulation_results.csv")
    combined_sim.to_csv(sim_csv_path, index=False)

    print("\n" + "=" * 80)
    print("SIMULATION FINAL OUTCOME (Same Budget: 800 total account reviews)")
    print("=" * 80)
    print(f"Total True Fraud in System: {total_fraud}")
    print(f"Baseline (Original Paper Distance) Total Frauds Caught: {baseline_cum_fraud} ({baseline_cum_fraud/total_fraud:.1%} recall)")
    print(f"Proposed (LightGBM Closed-Loop)    Total Frauds Caught: {proposed_cum_fraud} ({proposed_cum_fraud/total_fraud:.1%} recall)")
    gain = proposed_cum_fraud - baseline_cum_fraud
    print(f"Delta: +{gain} additional fraud accounts caught using the EXACT same investigator capacity!")
    print("=" * 80)

    # [3/3] Plot Cumulative Discovery Curve
    print("\n[3/3] Generating simulation trajectory plot...")
    plt.figure(figsize=(10, 5.5))
    plt.plot(df_base["day"], df_base["cumulative_frauds"], marker="o", linewidth=2.5, label="Original Paper (Static Distance)", color="#e74c3c")
    plt.plot(df_prop["day"], df_prop["cumulative_frauds"], marker="s", linewidth=2.5, label="Proposed LightGBM (Closed-Loop + 80/20)", color="#27ae60")
    plt.axhline(y=total_fraud, color="gray", linestyle="--", label=f"Total Fraud Ground Truth ({total_fraud})")

    plt.title("Cumulative Fraud Accounts Caught Over 20 Days (Capacity: 40 reviews/day)", fontsize=13, fontweight="bold")
    plt.xlabel("Investigation Day", fontsize=11)
    plt.ylabel("Cumulative Fraud Accounts Detected", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    traj_img_path = os.path.join(output_dir, "simulation_trajectory.png")
    plt.savefig(traj_img_path, dpi=200)
    plt.close()
    print(f"Simulation plot saved to: {traj_img_path}")

    return combined_sim

if __name__ == "__main__":
    run_simulation(days=20, daily_budget=40, retrain_frequency=3)
