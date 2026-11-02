"""
GMM Clustering & Fraud Risk Cluster (FRC) Detection
Step 2 of the Prioritization Pipeline:
- Fits Gaussian Mixture Model (GMM) on account activity features.
- Identifies the Fraud Risk Cluster (FRC).
- Computes distances to FRC centroid and cluster posterior probabilities.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "account_age_days",
    "invoices_created",
    "emails_sent",
    "clients_added",
    "expenses_created",
    "declined_payments",
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds velocity and behavioral ratios for enhanced clustering and modeling."""
    df_feat = df.copy()
    age_safe = np.maximum(df_feat["account_age_days"], 1.0)
    invoices_safe = np.maximum(df_feat["invoices_created"], 1.0)

    df_feat["invoices_per_day"] = df_feat["invoices_created"] / age_safe
    df_feat["emails_per_invoice"] = df_feat["emails_sent"] / invoices_safe
    df_feat["decline_rate"] = df_feat["declined_payments"] / invoices_safe
    df_feat["expense_ratio"] = df_feat["expenses_created"] / invoices_safe
    df_feat["clients_per_invoice"] = df_feat["clients_added"] / invoices_safe
    return df_feat

class GMMFraudClusterer:
    def __init__(self, n_components: int = 3, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=random_state,
            max_iter=200
        )
        self.frc_cluster_id = None
        self.frc_centroid = None
        self.all_centroids = None
        self.feature_names = None

    def fit(self, df: pd.DataFrame, initial_fraud_labels: np.ndarray = None):
        """
        Fits GMM on scaled features and determines the Fraud Risk Cluster (FRC).
        """
        df_feat = engineer_features(df)
        cols = FEATURE_COLUMNS + ["invoices_per_day", "decline_rate", "expense_ratio"]
        self.feature_names = cols
        X = df_feat[cols].values
        X_scaled = self.scaler.fit_transform(X)

        self.gmm.fit(X_scaled)
        cluster_preds = self.gmm.predict(X_scaled)
        self.all_centroids = self.gmm.means_

        # Determine FRC (Fraud Risk Cluster)
        # If seed labels are provided, pick cluster with highest fraud concentration
        # Otherwise, pick cluster with highest decline rate and invoice velocity
        if initial_fraud_labels is not None and initial_fraud_labels.sum() > 0:
            cluster_fraud_rates = []
            for c in range(self.n_components):
                mask = (cluster_preds == c)
                fraud_rate = initial_fraud_labels[mask].mean() if mask.sum() > 0 else 0
                cluster_fraud_rates.append(fraud_rate)
            self.frc_cluster_id = int(np.argmax(cluster_fraud_rates))
        else:
            # Unsupervised heuristic: cluster with highest (decline_rate + invoices_per_day - expense_ratio)
            heuristic_scores = []
            for c in range(self.n_components):
                mean_vec = self.all_centroids[c]
                # Index in cols: invoices_per_day is -3, decline_rate is -2, expense_ratio is -1
                score = mean_vec[-2] * 2.0 + mean_vec[-3] * 1.5 - mean_vec[-1] * 1.0
                heuristic_scores.append(score)
            self.frc_cluster_id = int(np.argmax(heuristic_scores))

        self.frc_centroid = self.all_centroids[self.frc_cluster_id]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes GMM features:
        - gmm_cluster: assigned cluster index
        - is_in_frc: binary flag (1 if account is in FRC cluster)
        - frc_posterior_prob: P(cluster == FRC | features)
        - distance_to_frc: Euclidean distance to FRC centroid
        - min_cluster_distance: distance to nearest cluster
        """
        df_feat = engineer_features(df)
        X = df_feat[self.feature_names].values
        X_scaled = self.scaler.transform(X)

        cluster_labels = self.gmm.predict(X_scaled)
        probs = self.gmm.predict_proba(X_scaled)

        # Distances to FRC centroid
        diff = X_scaled - self.frc_centroid
        dist_to_frc = np.linalg.norm(diff, axis=1)

        # Distances to all centroids (for multi-center ranking)
        all_dists = np.zeros((len(X), self.n_components))
        for c in range(self.n_components):
            c_diff = X_scaled - self.all_centroids[c]
            all_dists[:, c] = np.linalg.norm(c_diff, axis=1)

        out = df.copy()
        out["gmm_cluster"] = cluster_labels
        out["is_in_frc"] = (cluster_labels == self.frc_cluster_id).astype(int)
        out["frc_posterior_prob"] = probs[:, self.frc_cluster_id]
        out["distance_to_frc"] = dist_to_frc
        out["min_distance_to_any_centroid"] = np.min(all_dists, axis=1)

        # Multi-center fraud risk: Distance from normal cluster centroid (cluster 0 / genuine clusters)
        # Accounts furthest from normal centers and closest to FRC/suspicious centers
        genuine_cluster_id = 0 if self.frc_cluster_id != 0 else 1
        dist_from_normal = all_dists[:, genuine_cluster_id]
        
        # Multi-center risk combines distance from normal + proximity to FRC
        out["multi_center_score"] = dist_from_normal / (dist_to_frc + 1.0)

        return out
