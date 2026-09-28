import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OrdinalEncoder

from data_split import DataSplit
from models import RandomForestModel


class BaselineFraudExperiment:
    """Train and evaluate a model using caller-provided data splits."""

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    def __init__(
        self,
        model: RandomForestModel,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        json_path: str | Path | None = None,
        target_col: str = "fraud",
    ):
        self.model = model
        self.train_df = train_df.copy()
        self.val_df = val_df.copy()
        self.test_df = test_df.copy()
        self.target_col = target_col
        self.json_path = Path(json_path) if json_path else self.PROJECT_ROOT / "metrics" / "baseline_metrics_results.json"
        self.encoder = None
        self.categorical_cols = []
        self.feature_cols = []
        self.fit_time_sec = None
        self.validation_results = None
        self.test_results = None

        for split_name, split_df in (
            ("train", self.train_df),
            ("val", self.val_df),
            ("test", self.test_df),
        ):
            if self.target_col not in split_df.columns:
                raise ValueError(f"A {split_name} adathalmazból hiányzik a céloszlop: {self.target_col!r}")

        self.feature_cols = [col for col in self.train_df.columns if col != self.target_col]
        self.categorical_cols = [
            col
            for col in self.feature_cols
            if (
                pd.api.types.is_object_dtype(self.train_df[col])
                or isinstance(self.train_df[col].dtype, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(self.train_df[col])
            )
        ]

    def _prepare_features(self, df: pd.DataFrame, fit_encoder: bool = False) -> pd.DataFrame:
        features = df[self.feature_cols].copy()
        if not self.categorical_cols:
            return features

        if fit_encoder:
            self.encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            self.encoder.fit(features[self.categorical_cols])
        if self.encoder is None:
            raise RuntimeError("A kategóriakódoló még nincs betanítva; előbb hívd meg a train() metódust.")

        encoded = self.encoder.transform(features[self.categorical_cols])
        features.loc[:, self.categorical_cols] = encoded
        return features

    @staticmethod
    def _positive_class_probabilities(model, features):
        probabilities = model.predict_proba(features)
        if getattr(probabilities, "ndim", 1) == 2:
            return probabilities[:, 1]
        return probabilities

    def _calculate_metrics(self, df: pd.DataFrame, features: pd.DataFrame) -> dict:
        y_true = df[self.target_col]
        y_proba = self._positive_class_probabilities(self.model, features)
        y_pred = (y_proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        return {
            "n_samples": int(len(y_true)),
            "fraud_rate": float(y_true.mean()) if len(y_true) else 0.0,
            "auc_roc": float(roc_auc_score(y_true, y_proba)),
            "auc_pr": float(average_precision_score(y_true, y_proba)),
            "precision_fraud": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall_fraud": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_fraud": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }

    def train(self) -> dict:
        """Fit encoders and model on train, then report validation metrics."""
        x_train = self._prepare_features(self.train_df, fit_encoder=True)
        x_val = self._prepare_features(self.val_df)
        y_train = self.train_df[self.target_col]

        start = time.perf_counter()
        self.model.fit(x_train, y_train)
        self.fit_time_sec = time.perf_counter() - start
        self.validation_results = self._calculate_metrics(self.val_df, x_val)
        self.validation_results["fit_time_sec"] = self.fit_time_sec
        return self.validation_results

    def test(self) -> dict:
        """Evaluate the trained model on the held-out test split."""
        if self.fit_time_sec is None:
            raise RuntimeError("A modellt előbb tanítsd be a train() metódussal.")

        x_test = self._prepare_features(self.test_df)
        self.test_results = {
            "model_name": type(self.model).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fit_time_sec": self.fit_time_sec,
            "validation": self.validation_results,
            "test": self._calculate_metrics(self.test_df, x_test),
        }
        return self.test_results

    def save_results(self) -> None:
        """Append the latest test results to the configured JSON file."""
        if self.test_results is None:
            raise RuntimeError("Eredménymentés előtt hívd meg a test() metódust.")

        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        if self.json_path.exists():
            with self.json_path.open("r", encoding="utf-8") as results_file:
                existing = json.load(results_file)
            if not isinstance(existing, list):
                existing = [existing]
        else:
            existing = []

        existing.append(self.test_results)
        with self.json_path.open("w", encoding="utf-8") as results_file:
            json.dump(existing, results_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    data_path = BaselineFraudExperiment.PROJECT_ROOT / "data" / "processed" / "dataset_with_clusters_final.csv"
    data = pd.read_csv(data_path)
    train_df, val_df, test_df = DataSplit(data).time_split()

    experiment = BaselineFraudExperiment(
        model=RandomForestModel(),
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )
    experiment.train()
    experiment.test()
    experiment.save_results()