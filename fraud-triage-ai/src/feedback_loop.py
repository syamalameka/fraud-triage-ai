"""
Closed-Loop Feedback and Retraining Engine
Step 6: Ingests investigator decisions, logs to SQLite database,
and triggers model retraining to continuously sharpen prioritization.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Any, List

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "feedback_store.db"
)

class FeedbackManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path if db_path is not None else DEFAULT_DB_PATH

    def save_investigation_decisions(self, decisions_df: pd.DataFrame):
        """
        Saves investigator decisions to SQLite.
        Expected columns: account_id, business_name, day_number, risk_score,
                          selection_type, ranking_method, investigator_decision, notes, actual_is_fraud
        """
        conn = sqlite3.connect(self.db_path)
        decisions_df.to_sql("investigations", conn, if_exists="append", index=False)
        conn.close()

    def get_labeled_feedback(self) -> pd.DataFrame:
        """Retrieves all feedback submitted by investigators."""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM investigations WHERE investigator_decision IN ('Fraud', 'Genuine')"
        df_feedback = pd.read_sql_query(query, conn)
        conn.close()
        return df_feedback

    def get_reviewed_account_ids(self) -> List[str]:
        """Gets list of all account IDs already inspected."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT account_id FROM investigations")
        rows = cursor.fetchall()
        conn.close()
        return [r[0] for r in rows]

    def reset_feedback_db(self):
        """Clears feedback tables for fresh experiment runs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM investigations")
        cursor.execute("DELETE FROM daily_audit_summary")
        conn.commit()
        conn.close()

def retrain_model_with_feedback(
    ranker,
    df_all: pd.DataFrame,
    df_feedback: pd.DataFrame,
    initial_seed_size: int = 100
):
    """
    Retrains the LightGBM/HistGradientBoosting ranker using:
    - Base seed data
    - Cumulative verified labels from investigator feedback (Human-in-the-Loop)
    """
    # Merge feedback labels into the training pool
    feedback_dict = dict(zip(df_feedback["account_id"], df_feedback["investigator_decision"]))

    # Map human label: 'Fraud' -> 1, 'Genuine' -> 0
    feedback_labels = {
        acc_id: 1 if dec == "Fraud" else 0
        for acc_id, dec in feedback_dict.items()
        if dec in ["Fraud", "Genuine"]
    }

    # Find matching rows in df_all
    mask = df_all["account_id"].isin(feedback_labels.keys())
    train_df = df_all[mask].copy()
    y_train = np.array([feedback_labels[acc] for acc in train_df["account_id"]])

    # If feedback is small, augment with small initial random verified seed to avoid collapse
    if len(train_df) < 50:
        seed_df = df_all.iloc[:initial_seed_size].copy()
        y_seed = seed_df["is_fraud"].values
        train_df = pd.concat([seed_df, train_df], ignore_index=True).drop_duplicates(subset=["account_id"])
        y_train = train_df["is_fraud"].values

    ranker.fit(train_df, y_train)
    return ranker
