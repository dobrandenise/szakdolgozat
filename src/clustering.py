import numpy as np
import pandas as pd
import cupy as cp
from cuml.cluster import HDBSCAN, KMeans


class Clusterer:
    """GPU-backed clustering methods provided by RAPIDS cuML."""

    @staticmethod
    def _prepare_matrix(cleaned_df) -> cp.ndarray:
        required_columns = ["step", "amount", "customer" , "age", "gender", "category", "merchant"]
        missing = [column for column in required_columns if column not in cleaned_df.columns]
        if missing:
            raise ValueError(f"Hiányzó klaszterezési oszlop(ok): {missing}")

        features = cleaned_df.copy()
        features["amount_log"] = np.log1p(features["amount"])
        feature_columns = [
            "step",
            "amount_log",
            "age",
            "gender",
            "customer",
            "merchant",
            "category",
        ]
        return cp.asarray(features[feature_columns].to_numpy(), dtype=cp.float32)

    def Kmeans(self, cleaned_df) -> tuple[np.ndarray, np.ndarray]:
        """Add KMeans cluster ID and centroid distance to ``cleaned_df``."""
        matrix = self._prepare_matrix(cleaned_df)
        model = KMeans(
            n_clusters= 9,
            random_state= 42,
            n_init= 10,
        )
        labels = model.fit_predict(matrix)
        centroids = cp.asarray(model.cluster_centers_)
        distances = cp.linalg.norm(matrix - centroids[labels], axis=1)

        cluster_id = cp.asnumpy(labels).astype(np.int32)
        dist_to_centroid = cp.asnumpy(distances)
        return cluster_id, dist_to_centroid

    def HDBScan(self, cleaned_df) -> np.ndarray:
        """Add HDBSCAN cluster ID and binary noise flag to ``cleaned_df``."""
        matrix = self._prepare_matrix(cleaned_df)
        model = HDBSCAN(
            min_cluster_size= 100,
            min_samples= 20,
        )
        labels = model.fit_predict(matrix)

        labels_numpy = cp.asnumpy(labels).astype(np.int32)
        is_noise = labels_numpy == -1
        return is_noise

    def run(self, cleaned_df) -> pd.DataFrame:
        print("=== Kmeans futtatása ===")
        cluster_id, dist_to_centroid = self.Kmeans(cleaned_df)
        print("=== HDBSCAN futtatása ===")
        is_noise = self.HDBScan(cleaned_df)
        cleaned_df["cluster_id"] = cluster_id
        cleaned_df["dist_to_centroid"] = dist_to_centroid
        cleaned_df["is_hdbscan_noise"] = is_noise.astype(np.int8)
        return cleaned_df

