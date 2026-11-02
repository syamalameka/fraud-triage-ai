"""
Benchmark Suite: Head-to-Head Comparison of 7 Ranking Strategies
Evaluates:
  1. Random Ranking
  2. Original Paper Distance Ranking
  3. Corrected Distance Ranking
  4. Classifier Probability Ranking
  5. Isolation Forest
  6. Multi-center Distance Ranking
  7. Proposed Method: LightGBM / HistGradientBoosting with GMM features
"""

import os
import sys
# Ensure project root is in sys.path
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
from src.rankers import (
    RandomRanker,
    OriginalPaperDistanceRanker,
    CorrectedDistanceRanker,
    ClassifierProbabilityRanker,
    IsolationForestRanker,
    MultiCenterDistanceRanker,
    LightGBMClosedLoopRanker
)
from src.metrics import evaluate_ranking_performance

def run_full_benchmark(data_path=None, output_dir=None):
    if data_path is None:
        data_path = os.path.join(PROJECT_ROOT, "data", "accounts_data.csv")
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "benchmark")
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("FRAUD INVESTIGATION PRIORITIZATION - BENCHMARK SUITE")
    print("=" * 70)

    # 1. Load Data
    df = pd.read_csv(data_path)
    total_fraud = df["is_fraud"].sum()
    print(f"Loaded {len(df)} accounts with {total_fraud} true fraud accounts ({total_fraud/len(df):.1%})\n")

    # 2. Step 2: Fit GMM and transform dataset
    print("[1/3] Fitting Gaussian Mixture Model (GMM) for clustering & FRC detection...")
    # Seed 50 known fraud examples to represent initial cold-start historical dataset
    fraud_indices = df[df["is_fraud"] == 1].index[:50]
    initial_labels = np.zeros(len(df))
    initial_labels[fraud_indices] = 1

    gmm = GMMFraudClusterer(n_components=3, random_state=42)
    gmm.fit(df, initial_fraud_labels=initial_labels)
    df_transformed = gmm.transform(df)

    print(f" -> GMM Fraud Risk Cluster (FRC) identified: Cluster #{gmm.frc_cluster_id}")
    frc_count = (df_transformed['gmm_cluster'] == gmm.frc_cluster_id).sum()
    frc_frauds = df_transformed[df_transformed['gmm_cluster'] == gmm.frc_cluster_id]['is_fraud'].sum()
    print(f" -> FRC members: {frc_count} accounts ({frc_frauds} true frauds, {frc_frauds/frc_count:.1%} density)\n")

    # 3. Instantiate and train/score all 7 ranking algorithms
    print("[2/3] Evaluating 7 Ranking Strategies...")
    rankers = {
        "Random Ranking": RandomRanker(seed=42),
        "Original Paper Distance": OriginalPaperDistanceRanker(),
        "Corrected Distance (Mahalanobis)": CorrectedDistanceRanker(),
        "Classifier Probability": ClassifierProbabilityRanker(random_state=42),
        "Isolation Forest": IsolationForestRanker(random_state=42),
        "Multi-center Distance": MultiCenterDistanceRanker(),
        "Proposed LightGBM (Ours)": LightGBMClosedLoopRanker(random_state=42)
    }

    # Prepare initial training data for supervised models (small 200 account sample)
    train_seed_idx = list(range(0, 200))
    train_seed_df = df_transformed.iloc[train_seed_idx]
    y_seed = train_seed_df["is_fraud"].values

    summary_rows = []
    k_eval = 40  # Daily investigator capacity

    for name, ranker in rankers.items():
        # Fit rankers that need training
        if isinstance(ranker, (ClassifierProbabilityRanker, LightGBMClosedLoopRanker)):
            ranker.fit(train_seed_df, y_seed)
        elif isinstance(ranker, (CorrectedDistanceRanker, IsolationForestRanker)):
            ranker.fit(df_transformed)

        # Generate scores and ranks
        scores = ranker.score(df_transformed)
        ranked = df_transformed.copy()
        ranked["score"] = scores
        # Higher score = higher risk
        ranked = ranked.sort_values(by="score", ascending=False).reset_index(drop=True)

        eval_res = evaluate_ranking_performance(ranked, k_cutoffs=[10, 20, 40, 100])
        m40 = eval_res["metrics_by_k"][40]
        m100 = eval_res["metrics_by_k"][100]

        summary_rows.append({
            "Method": name,
            "Precision@10": eval_res["metrics_by_k"][10]["Precision@K"],
            "Precision@40": m40["Precision@K"],
            "Recall@40": m40["Recall@K"],
            "Frauds_Caught_in_40": m40["Frauds_Caught"],
            "Frauds_Per_100_in_40": m40["Frauds_Per_100"],
            "Precision@100": m100["Precision@K"],
            "Recall@100": m100["Recall@K"],
            "Frauds_Caught_in_100": m100["Frauds_Caught"]
        })

    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    summary_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 90)
    print("BENCHMARK COMPARISON TABLE (Daily Investigation Budget K = 40 accounts)")
    print("=" * 90)
    display_df = summary_df[["Method", "Precision@40", "Recall@40", "Frauds_Caught_in_40", "Frauds_Per_100_in_40", "Frauds_Caught_in_100"]]
    print(display_df.to_string(index=False))
    print("=" * 90)
    print(f"Results saved to: {csv_path}\n")

    # 4. Generate Visualization
    print("[3/3] Generating benchmark comparison chart...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart 1: Frauds Caught in Day 1 Budget (K=40)
    methods = summary_df["Method"]
    frauds_40 = summary_df["Frauds_Caught_in_40"]
    colors = ["#95a5a6", "#3498db", "#2980b9", "#f39c12", "#e67e22", "#9b59b6", "#2ecc71"]

    axes[0].barh(methods, frauds_40, color=colors)
    axes[0].set_xlabel("Number of Frauds Caught (Top 40 Reviews)")
    axes[0].set_title("Fraud Discovery in Daily Budget (K = 40)")
    axes[0].grid(axis="x", linestyle="--", alpha=0.7)
    for i, v in enumerate(frauds_40):
        axes[0].text(v + 0.5, i, str(v), va="center", fontweight="bold")

    # Bar chart 2: Precision@40 vs Precision@100
    x = np.arange(len(methods))
    width = 0.35
    axes[1].bar(x - width/2, summary_df["Precision@40"] * 100, width, label="Precision@40", color="#3498db")
    axes[1].bar(x + width/2, summary_df["Precision@100"] * 100, width, label="Precision@100", color="#2ecc71")
    axes[1].set_ylabel("Precision (%)")
    axes[1].set_title("Precision at Different Investigation Cutoffs")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=35, ha="right")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "benchmark_comparison.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"Benchmark chart saved to: {chart_path}")

    return summary_df

if __name__ == "__main__":
    run_full_benchmark()
