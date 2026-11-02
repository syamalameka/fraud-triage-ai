"""
Evaluation Metrics Module
Computes:
  - Precision@K: Precision among top K investigated accounts
  - Recall@K: Proportion of total fraud detected within top K reviews
  - Frauds per 100 reviews: Yield rate per 100 investigated accounts
  - Cumulative fraud caught over investigation days
  - Mean detection time
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any

def compute_precision_at_k(actual_labels: np.ndarray, k: int) -> float:
    """Computes Precision@K: fraction of top K reviewed accounts that are true fraud."""
    if k <= 0 or len(actual_labels) == 0:
        return 0.0
    top_k = actual_labels[:k]
    return float(np.sum(top_k) / len(top_k))

def compute_recall_at_k(actual_labels: np.ndarray, total_positives: int, k: int) -> float:
    """Computes Recall@K: fraction of all fraud accounts detected in top K reviews."""
    if total_positives <= 0:
        return 0.0
    top_k = actual_labels[:k]
    return float(np.sum(top_k) / total_positives)

def compute_frauds_per_100_reviews(total_frauds_found: int, total_reviewed: int) -> float:
    """Yield metric: expected true fraud accounts discovered per 100 investigated."""
    if total_reviewed <= 0:
        return 0.0
    return float((total_frauds_found / total_reviewed) * 100.0)

def evaluate_ranking_performance(
    ranked_df: pd.DataFrame,
    label_col: str = "is_fraud",
    k_cutoffs: List[int] = [10, 20, 40, 100, 200, 500]
) -> Dict[str, Any]:
    """
    Evaluates a ranked dataframe across multiple standard cutoffs K.
    """
    total_fraud = int(ranked_df[label_col].sum())
    actuals = ranked_df[label_col].values

    results = {
        "total_accounts": len(ranked_df),
        "total_fraud": total_fraud,
        "metrics_by_k": {}
    }

    for k in k_cutoffs:
        if k > len(ranked_df):
            continue
        p_k = compute_precision_at_k(actuals, k)
        r_k = compute_recall_at_k(actuals, total_fraud, k)
        f_count = int(np.sum(actuals[:k]))
        f_per_100 = compute_frauds_per_100_reviews(f_count, k)

        results["metrics_by_k"][k] = {
            "Precision@K": round(p_k, 4),
            "Recall@K": round(r_k, 4),
            "Frauds_Caught": f_count,
            "Frauds_Per_100": round(f_per_100, 2)
        }

    return results

def compute_cumulative_discovery_curve(daily_logs: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generates day-by-day cumulative fraud discovery trajectory."""
    df = pd.DataFrame(daily_logs)
    if "frauds_found" in df.columns:
        df["cumulative_frauds"] = df["frauds_found"].cumsum()
        df["cumulative_inspections"] = df["budget_reviewed"].cumsum()
        df["cumulative_precision"] = (df["cumulative_frauds"] / df["cumulative_inspections"]).round(4)
    return df
