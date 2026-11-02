"""
Ranking Algorithms Module
Implements all 6 benchmark ranking strategies and the Proposed LightGBM Ranker:
1. Random Ranking
2. Original Paper Distance Ranking
3. Corrected Distance Ranking (Mahalanobis / Scaled Distance)
4. Classifier Probability Ranking (Standard Supervised Classifier)
5. Isolation Forest Ranking (Unsupervised Outlier Ranking)
6. Multi-center Distance Ranking (Distance to Multiple Cluster Centroids)
7. Proposed Method: LightGBM / HistGradientBoosting with GMM Features & Closed-Loop
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Attempt to import LightGBM, fallback seamlessly to HistGradientBoostingClassifier
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    from sklearn.ensemble import HistGradientBoostingClassifier

from src.gmm_clustering import engineer_features, FEATURE_COLUMNS

MODEL_FEATURES = FEATURE_COLUMNS + [
    "invoices_per_day",
    "emails_per_invoice",
    "decline_rate",
    "expense_ratio",
    "clients_per_invoice"
]

class BaseRanker:
    def __init__(self, name: str):
        self.name = name

    def fit(self, df_train: pd.DataFrame, y_train: np.ndarray = None):
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def rank(self, df: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
        """Adds a rank and risk score to the dataframe."""
        df_out = df.copy()
        scores = self.score(df_out)
        df_out[f"{self.name}_score"] = scores
        df_out = df_out.sort_values(by=f"{self.name}_score", ascending=ascending).reset_index(drop=True)
        df_out[f"{self.name}_rank"] = np.arange(1, len(df_out) + 1)
        return df_out

# 1. Random Ranking
class RandomRanker(BaseRanker):
    def __init__(self, seed: int = 42):
        super().__init__("random")
        self.rng = np.random.RandomState(seed)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.rng.uniform(0.0, 1.0, size=len(df))

# 2. Original Paper Distance Ranking
# Distance to FRC centroid in GMM. In the paper, smaller distance = closer to fraud cluster center = higher risk.
# To make higher score = higher risk, we use negative distance or similarity: 1 / (1 + distance).
class OriginalPaperDistanceRanker(BaseRanker):
    def __init__(self):
        super().__init__("orig_paper_dist")

    def score(self, df: pd.DataFrame) -> np.ndarray:
        if "distance_to_frc" not in df.columns:
            raise ValueError("Dataframe must be transformed by GMMFraudClusterer first.")
        # Score where closer to FRC center gives higher risk score
        return -df["distance_to_frc"].values

# 3. Corrected Distance Ranking (Covariance / Normalized Mahalanobis Distance)
class CorrectedDistanceRanker(BaseRanker):
    def __init__(self):
        super().__init__("corrected_dist")
        self.scaler = StandardScaler()
        self.inv_cov = None
        self.frc_mean = None

    def fit(self, df_train: pd.DataFrame, y_train: np.ndarray = None):
        df_feat = engineer_features(df_train)
        X = df_feat[MODEL_FEATURES].values
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Estimate covariance on FRC members or whole set with regularization
        cov = np.cov(X_scaled, rowvar=False) + np.eye(X.shape[1]) * 1e-4
        self.inv_cov = np.linalg.pinv(cov)

        if "is_in_frc" in df_train.columns and df_train["is_in_frc"].sum() > 0:
            self.frc_mean = X_scaled[df_train["is_in_frc"] == 1].mean(axis=0)
        else:
            self.frc_mean = X_scaled.mean(axis=0)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        df_feat = engineer_features(df)
        X = df_feat[MODEL_FEATURES].values
        X_scaled = self.scaler.transform(X)

        diff = X_scaled - self.frc_mean
        # Mahalanobis distance squared: d_M = sqrt(diff * inv_cov * diff^T)
        m_dist = np.sqrt(np.sum((diff @ self.inv_cov) * diff, axis=1))
        # Closer Mahalanobis distance to fraud cluster = higher risk score
        return -m_dist

# 4. Standalone Classifier Probability Ranking (No GMM cluster features, No Exploration)
class ClassifierProbabilityRanker(BaseRanker):
    def __init__(self, random_state: int = 42):
        super().__init__("classifier_prob")
        self.model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state)

    def fit(self, df_train: pd.DataFrame, y_train: np.ndarray):
        df_feat = engineer_features(df_train)
        X = df_feat[MODEL_FEATURES].values
        self.model.fit(X, y_train)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        df_feat = engineer_features(df)
        X = df_feat[MODEL_FEATURES].values
        # Probability of class 1 (Fraud)
        return self.model.predict_proba(X)[:, 1]

# 5. Isolation Forest Ranking (Unsupervised Anomaly Score)
class IsolationForestRanker(BaseRanker):
    def __init__(self, contamination: float = 0.02, random_state: int = 42):
        super().__init__("isolation_forest")
        self.model = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            random_state=random_state
        )

    def fit(self, df_train: pd.DataFrame, y_train: np.ndarray = None):
        df_feat = engineer_features(df_train)
        X = df_feat[MODEL_FEATURES].values
        self.model.fit(X)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        df_feat = engineer_features(df)
        X = df_feat[MODEL_FEATURES].values
        # Decision function: lower values mean more anomalous.
        # Negate so higher score = higher risk.
        return -self.model.decision_function(X)

# 6. Multi-center Distance Ranking
class MultiCenterDistanceRanker(BaseRanker):
    def __init__(self):
        super().__init__("multi_center_dist")

    def score(self, df: pd.DataFrame) -> np.ndarray:
        if "multi_center_score" in df.columns:
            return df["multi_center_score"].values
        # Fallback to inverse min distance
        if "min_distance_to_any_centroid" in df.columns:
            return -df["min_distance_to_any_centroid"].values
        return np.zeros(len(df))

# 7. Proposed LightGBM / Gradient Boosting Closed-Loop Ranker
class LightGBMClosedLoopRanker(BaseRanker):
    """
    Proposed Method:
    - Ingests Account Activity features + GMM Cluster FRC Flag + Distance to Centroid
    - Trains LightGBM / HistGradientBoosting model
    - Retrained iteratively in closed-loop with investigator feedback
    """
    def __init__(self, random_state: int = 42):
        super().__init__("lightgbm_closed_loop")
        self.random_state = random_state
        self.feature_cols = MODEL_FEATURES + ["is_in_frc", "distance_to_frc", "frc_posterior_prob"]

        if HAS_LIGHTGBM:
            self.model = lgb.LGBMClassifier(
                n_estimators=120,
                learning_rate=0.05,
                num_leaves=31,
                random_state=random_state,
                verbose=-1
            )
            self.engine_name = "LightGBM"
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=120,
                learning_rate=0.05,
                max_leaf_nodes=31,
                random_state=random_state
            )
            self.engine_name = "HistGradientBoosting (LightGBM equivalent)"

    def fit(self, df_train: pd.DataFrame, y_train: np.ndarray):
        df_feat = engineer_features(df_train)
        # Ensure GMM features are present
        for col in ["is_in_frc", "distance_to_frc", "frc_posterior_prob"]:
            if col not in df_feat.columns:
                df_feat[col] = 0.0

        X = df_feat[self.feature_cols].values
        self.model.fit(X, y_train)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        df_feat = engineer_features(df)
        for col in ["is_in_frc", "distance_to_frc", "frc_posterior_prob"]:
            if col not in df_feat.columns:
                df_feat[col] = 0.0

        X = df_feat[self.feature_cols].values
        # Predict probability of fraud
        return self.model.predict_proba(X)[:, 1]
