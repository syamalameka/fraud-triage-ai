"""
Prioritization Engine (Step 4: Rank and Select Accounts)
Allocates daily investigation budget (K = 40 accounts/day):
  - 80% (32 accounts): Exploitation (highest fraud risk score)
  - 20% (8 accounts): Exploration (uncertainty sampling / random discovery of novel fraud patterns)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Set

class PrioritizationEngine:
    def __init__(
        self,
        daily_capacity: int = 40,
        exploration_ratio: float = 0.20,
        random_state: int = 42
    ):
        self.daily_capacity = daily_capacity
        self.exploration_ratio = exploration_ratio
        self.n_explore = int(daily_capacity * exploration_ratio)       # e.g., 8 slots
        self.n_exploit = daily_capacity - self.n_explore              # e.g., 32 slots
        self.rng = np.random.RandomState(random_state)
        self.inspected_account_ids: Set[str] = set()

    def set_already_inspected(self, account_ids: List[str]):
        """Populates the set of accounts already reviewed by investigators."""
        self.inspected_account_ids = set(account_ids)

    def select_daily_batch(
        self,
        df_accounts: pd.DataFrame,
        risk_scores: np.ndarray,
        exploration_mode: str = "uncertainty"  # 'uncertainty' or 'random'
    ) -> pd.DataFrame:
        """
        Selects K accounts for the day's investigation budget:
        - 32 exploitation accounts (highest model risk score)
        - 8 exploration accounts (uncertainty closest to 0.5 or random exploration)
        """
        df = df_accounts.copy()
        df["risk_score"] = risk_scores

        # Exclude already inspected accounts
        uninspected_mask = ~df["account_id"].isin(self.inspected_account_ids)
        available_df = df[uninspected_mask].copy()

        if len(available_df) <= self.daily_capacity:
            available_df["selection_type"] = "budget_remainder"
            self.inspected_account_ids.update(available_df["account_id"].tolist())
            return available_df

        # 1. Exploitation Selection: Top n_exploit highest risk scores
        available_df = available_df.sort_values(by="risk_score", ascending=False)
        exploit_batch = available_df.iloc[:self.n_exploit].copy()
        exploit_batch["selection_type"] = "exploitation"

        # Remaining pool for exploration
        remaining_pool = available_df.iloc[self.n_exploit:].copy()

        # 2. Exploration Selection: Top n_explore from remaining pool
        if exploration_mode == "uncertainty":
            # Uncertainty sampling: scores closest to 0.50 (where model is most uncertain)
            remaining_pool["uncertainty_dist"] = np.abs(remaining_pool["risk_score"] - 0.50)
            explore_candidates = remaining_pool.sort_values(by="uncertainty_dist", ascending=True)
            explore_batch = explore_candidates.iloc[:self.n_explore].copy()
            explore_batch.drop(columns=["uncertainty_dist"], inplace=True)
        else:
            # Pure random exploration
            sample_indices = self.rng.choice(
                len(remaining_pool),
                size=min(self.n_explore, len(remaining_pool)),
                replace=False
            )
            explore_batch = remaining_pool.iloc[sample_indices].copy()

        explore_batch["selection_type"] = "exploration"

        # Combine batches
        daily_queue = pd.concat([exploit_batch, explore_batch], ignore_index=True)

        # Mark as inspected
        self.inspected_account_ids.update(daily_queue["account_id"].tolist())

        # Categorize risk level for investigator UI display
        def get_risk_level(score):
            if score >= 0.70:
                return "High"
            elif score >= 0.40:
                return "Medium"
            else:
                return "Low"

        daily_queue["risk_level"] = daily_queue["risk_score"].apply(get_risk_level)

        return daily_queue
