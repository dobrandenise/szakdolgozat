import numpy as np
import pandas as pd
import cupy as cp
from cuml.cluster import KMeans
from cuml.cluster.hdbscan import HDBSCAN, approximate_predict
from cuml.preprocessing import StandardScaler


class Clusterer:
    """GPU-backed clustering (KMeans + HDBSCAN) with proper fit/transform separation.

    Használat:
        clusterer = Clusterer()
        clusterer.fit(train_df)                       # csak a train adaton tanul
        train_out = clusterer.transform(train_df)
        val_out   = clusterer.transform(val_df)
        test_out  = clusterer.transform(test_df)

    Vagy egyben:
        train_out = clusterer.fit_transform(train_df)
    """

    FEATURE_COLUMNS = [
    "step",
    "amount_log",
    "age",
    "gender",
    "customer",
    "merchant",
    "category",
    "amount_zscore_customer",
    "amount_ratio_customer",
    "amount_zscore_category",
    "amount_ratio_merchant_avg",
    "customer_tx_count_hist",
    "customer_days_since_first_tx",
    "customer_category_diversity_hist",
    "is_new_category_for_customer",
    "merchant_tx_count_hist",
    "category_tx_count_hist",
    "category_is_high_risk",
    "tx_count_last_7d",
    "tx_count_last_30d",
    "days_since_last_tx_customer",
    "cumulative_spend_30d",
]

    def __init__(self, 
                kmeans_params={"n_clusters": 9, "random_state": 42, "n_init": 10}, 
                hdbscan_params={"min_cluster_size": 100, "min_samples": 20}
        ):
        self.kmeans_params = kmeans_params 
        self.hdbscan_params = hdbscan_params

        self.scaler = StandardScaler()
        self.kmeans_model = KMeans(**self.kmeans_params)
        self.hdbscan_model = HDBSCAN(**self.hdbscan_params, prediction_data=True)
        self.kmeans_centroids = None
        self._is_fitted = False

    def _build_raw_matrix(self, df: pd.DataFrame) -> cp.ndarray:
        missing = [c for c in self.FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Hiányzó klaszterezési oszlop(ok): {missing}")

        return cp.asarray(df[self.FEATURE_COLUMNS].to_numpy(dtype=np.float32))

    def fit(self, train_df: pd.DataFrame) -> "Clusterer":
        """Fit scaler, KMeans and HDBSCAN kizárólag a train adaton."""
        raw_matrix = self._build_raw_matrix(train_df)

        scaled_matrix = self.scaler.fit_transform(raw_matrix)

        print("=== KMeans fit (train) ===") 
        self.kmeans_model.fit(scaled_matrix)
        self.kmeans_centroids = cp.asarray(self.kmeans_model.cluster_centers_)

        print("=== HDBSCAN fit (train) ===")
        self.hdbscan_model.fit(scaled_matrix)

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """A betanított modellek alkalmazása bármely (train/val/test) adatra."""
        if not self._is_fitted:
            raise RuntimeError("A Clusterer nincs betanítva — hívd meg előbb a fit()-et a train adaton.")

        raw_matrix = self._build_raw_matrix(df)
        scaled_matrix = self.scaler.transform(raw_matrix)

        # --- KMeans ---
        cluster_id = self.kmeans_model.predict(scaled_matrix)
        assigned_centroids = self.kmeans_centroids[cluster_id]
        dist_to_centroid = cp.linalg.norm(scaled_matrix - assigned_centroids, axis=1)

        # --- HDBSCAN ---
        hdbscan_labels, _strengths = approximate_predict(self.hdbscan_model, scaled_matrix)
        is_noise = cp.asnumpy(hdbscan_labels) == -1

        out_df = df.copy()
        out_df["cluster_id"] = cp.asnumpy(cluster_id).astype(np.int16)
        out_df["dist_to_centroid"] = cp.asnumpy(dist_to_centroid)
        out_df["is_hdbscan_noise"] = is_noise.astype(np.int8)
        return out_df

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        self.fit(train_df)
        return self.transform(train_df)