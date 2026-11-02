"""
Master Runner for the Fraud Investigation Prioritization Project
Runs all steps end-to-end:
1. Synthetic Data Generation (10,000 accounts, 200 fraud)
2. 7-Model Benchmark Evaluation (Precision@40, Recall@40, Frauds caught)
3. 20-Day Closed-Loop Multi-Day Simulation (Exploitation vs Exploration + Retraining)
4. Verification & Summary Output
"""

import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.generate_synthetic_data import generate_accounts_dataset, initialize_feedback_database
from benchmark.run_benchmark import run_full_benchmark
from simulation.run_simulation import run_simulation

def main():
    print("=" * 80)
    print("   CLOSED-LOOP, BUDGET-AWARE FRAUD TRIAGE SYSTEM   ")
    print("   Fraud Investigation Prioritization Pipeline    ")
    print("=" * 80)

    # Step 1: Data Generation & Database Init
    print("\n[STEP 1/3] Generating Invoicing Accounts Dataset & SQLite Feedback Store...")
    generate_accounts_dataset()
    initialize_feedback_database()

    # Step 2: Benchmark Evaluation
    print("\n[STEP 2/3] Running 7-Model Benchmark Evaluation...")
    summary_df = run_full_benchmark()

    # Step 3: Closed-Loop Multi-Day Simulation
    print("\n[STEP 3/3] Running 20-Day Closed-Loop Budget Simulation (K=40/day)...")
    sim_df = run_simulation(days=20, daily_budget=40, retrain_frequency=3)

    print("\n" + "=" * 80)
    print("   ALL EXPERIMENTS COMPLETED SUCCESSFULLY!   ")
    print("=" * 80)
    print(f"Project Directory : {PROJECT_ROOT}")
    print(f"Data Path         : {os.path.join(PROJECT_ROOT, 'data', 'accounts_data.csv')}")
    print(f"Benchmark Results : {os.path.join(PROJECT_ROOT, 'benchmark', 'benchmark_results.csv')}")
    print(f"Benchmark Chart   : {os.path.join(PROJECT_ROOT, 'benchmark', 'benchmark_comparison.png')}")
    print(f"Simulation Traj   : {os.path.join(PROJECT_ROOT, 'simulation', 'simulation_trajectory.png')}")
    print("\nTo launch the Investigator Dashboard:")
    print("  Option A (Streamlit) : streamlit run app_streamlit.py")
    print("  Option B (Standalone): python app_server.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
