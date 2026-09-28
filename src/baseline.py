import os
import time
from datetime import datetime

import joblib
import json
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from preprocessor import Preprocessor
from data_split import DataSplit


class BaselineFraudExperiment:
    """
    A Preprocessor által megtisztított CSV-n tanítja és teszteli a
    RandomForest és HistGradientBoosting modelleket.

    OneHotEncoder helyett OrdinalEncoder-t használ a kategorikus oszlopok
    kódolására (egész számokká alakítja őket OHE-szerű oszlop-robbanás nélkül).

    """

    def __init__(
            self,
            data_path,
            split_method="time",
            feature_cols=None,
            categorical_cols=None,
            numeric_cols=None,
            csv_path=None,
            json_path=None,
            model_dir=None,
    ):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        self.data_path = data_path
        self.split_method = split_method
        self.feature_cols = feature_cols or [
            "step",
            "age",
            "gender",
            "category",
            "merchant",
            "amount",
        ]
        self.categorical_cols = categorical_cols or ["age", "gender", "category", "merchant"]
        self.numeric_cols = numeric_cols or ["step", "amount"]
        self.csv_path = csv_path or os.path.join(project_root, "metrics", "baseline_metrics_results.csv")
        self.json_path = json_path or os.path.join(project_root, "metrics", "baseline_metrics_results.json")
        self.model_dir = model_dir or os.path.join(project_root, "models")

    def run(self):
        train_df, val_df, test_df = self.get_split()
        # val_df jelenleg nincs felhasználva itt — később pl. threshold-hangoláshoz
        # vagy early stoppinghoz tartjuk fenn.
        self.print_dataset_summary(train_df)

        # >>> MÓDOSÍTVA: külön train/test feature-tábla, majd fit csak a train-en
        train_features, y_train, _ = self.prepare_target_and_features(train_df)
        test_features, y_test, _ = self.prepare_target_and_features(test_df)
        x_train, x_test = self.encode_features(train_features, test_features)

        self.print_split_summary(x_train, x_test, y_train, y_test)

        models = self.build_models()
        results = {}

        for name, model in models.items():
            results[name] = self.train_and_evaluate_model(model, x_train, x_test, y_train, y_test, name)

        print("\n=== Összefoglaló ===")
        for name, metrics in results.items():
            print(f"{name}: AUC-ROC={metrics['AUC-ROC']:.4f}, AUC-PR={metrics['AUC-PR']:.4f}")

        for name in models.keys():
            self.print_confusion_matrix(name, models[name], x_test, y_test)

        comparison_df = self.compare_models(models, x_test, y_test, results)
        self.print_model_comparison(comparison_df)
        self.save_results(comparison_df)
        self.save_models(models)

    def get_split(self):
        """
        A self.split_method alapján hívja meg a DataSplit két függvénye
        közül a megfelelőt, és visszaadja a (train_df, val_df, test_df)
        hármast.
        """
        splitter = DataSplit(csv_path=self.data_path)

        if self.split_method == "customer":
            return splitter.customer_split()
        elif self.split_method == "time":
            return splitter.time_split()
        else:
            raise ValueError(f"Ismeretlen split_method: {self.split_method!r} (csak 'time' vagy 'customer' lehet)")

    def load_data(self):
        return pd.read_csv(self.data_path)

    def print_dataset_summary(self, df):
        print("=== Alak ===")
        print(df.shape)

        print("\n=== Oszlopok és dtype-ok ===")
        print(df.dtypes)

        print("\n=== Első néhány sor ===")
        print(df.head())

        print("\n=== Hiányzó értékek oszloponként ===")
        print(df.isnull().sum())

        print("\n=== Fraud osztályeloszlás (darabszám) ===")
        print(df["fraud"].value_counts())

        print("\n=== Fraud osztályeloszlás (arány, %) ===")
        print(df["fraud"].value_counts(normalize=True) * 100)

    def prepare_target_and_features(self, df):
        df_features = df[self.feature_cols].copy()
        y = df["fraud"].copy()
        groups = df["customer"].copy() if "customer" in df.columns else None
        return df_features, y, groups

    def encode_features(self, train_features, test_features):
        # két df-et vár (train, test); az encoder csak a train-en
        # fit-el, a teszten csak transform — elkerülve, hogy a teszt kategóriái
        # befolyásolják a kódolást.
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(train_features[self.categorical_cols])

        def _transform(df_features):
            encoded_cats = encoder.transform(df_features[self.categorical_cols])
            encoded_df = pd.DataFrame(encoded_cats, columns=self.categorical_cols, index=df_features.index)
            return pd.concat([encoded_df, df_features[self.numeric_cols]], axis=1)

        x_train = _transform(train_features)
        x_test = _transform(test_features)

        print("x_train alakja:", x_train.shape)
        print("x_train oszlopai:\n", list(x_train.columns))
        return x_train, x_test

    def split_dataset(self, x, y):
        return train_test_split(x,y,test_size=0.25,stratify=y,random_state=42,)

    def print_split_summary(self, x_train, x_test, y_train, y_test):
        print("x_train alakja:", x_train.shape)
        print("x_test alakja:", x_test.shape)

        print("\nFraud arány train-ben:")
        print(y_train.value_counts(normalize=True) * 100)

        print("\nFraud arány test-ben:")
        print(y_test.value_counts(normalize=True) * 100)

    def build_models(self):
        rf_model = RandomForestClassifier(random_state=42, class_weight="balanced")
        hgb_model = HistGradientBoostingClassifier(random_state=42, class_weight="balanced")
        return {
            "RandomForest": rf_model,
            "HistGradientBoosting": hgb_model,
        }

    def train_and_evaluate_model(self, model, x_train, x_test, y_train, y_test, name):
        print(f"\n=== {name} ===")

        start = time.time()
        model.fit(x_train, y_train)
        elapsed = time.time() - start
        print(f"Betanítási idő: {elapsed:.1f} mp")

        y_proba = model.predict_proba(x_test)[:, 1]
        y_pred = model.predict(x_test)

        auc_roc = roc_auc_score(y_test, y_proba)
        auc_pr = average_precision_score(y_test, y_proba)

        print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"AUC-PR:  {auc_pr:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, target_names=["legit", "fraud"]))

        return {
            "AUC-ROC": auc_roc,
            "AUC-PR": auc_pr,
            "fit_time_sec": elapsed,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

    def print_confusion_matrix(self, name, model, x_test, y_test):
        print(f"\n=== {name} — Confusion Matrix ===")
        cm = confusion_matrix(y_test, model.predict(x_test))
        print(cm)
        print("(sorok: valós [legit, fraud], oszlopok: predikált [legit, fraud])")

        tn, fp, fn, tp = cm.ravel()
        print(f"True Negative:  {tn}")
        print(f"False Positive: {fp}")
        print(f"False Negative: {fn}")
        print(f"True Positive:  {tp}")

    def compare_models(self, models, x_test, y_test, results):
        comparison_rows = []

        for name, model in models.items():
            y_pred = model.predict(x_test)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test,
                y_pred,
                average="binary",
                pos_label=1,
            )
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            comparison_rows.append(
                {
                    "dataset": "BankSim",
                    "model": name,
                    "auc_roc": results[name]["AUC-ROC"],
                    "auc_pr": results[name]["AUC-PR"],
                    "precision_fraud": precision,
                    "recall_fraud": recall,
                    "f1_fraud": f1,
                    "fit_time_sec": results[name]["fit_time_sec"],
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                    "n_test": len(y_test),
                    "run_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "data_split": self.split_method,
                }
            )

        return pd.DataFrame(comparison_rows)

    def print_model_comparison(self, comparison_df):
        print("=== Modellek összevetése (BankSim baseline) ===")
        print(comparison_df.to_string(index=False))

        print("\n=== Metrikánkénti győztes ===")
        for metric in ["auc_roc", "auc_pr", "precision_fraud", "recall_fraud", "f1_fraud"]:
            best_row = comparison_df.loc[comparison_df[metric].idxmax()]
            print(f"{metric}: {best_row['model']} ({best_row[metric]:.4f})")

    def save_results(self, comparison_df):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)

        # CSV: append to existing file if present
        if os.path.exists(self.csv_path):
            comparison_df.to_csv(self.csv_path, mode="a", header=False, index=False)
        else:
            comparison_df.to_csv(self.csv_path, mode="w", header=True, index=False)

        # JSON: append to existing file if present
        # Use pandas to_json -> json.loads to ensure native Python types (avoids numpy types issues)
        records = json.loads(comparison_df.to_json(orient='records', force_ascii=False))
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as jf:
                    existing = json.load(jf)
                if not isinstance(existing, list):
                    existing = [existing]
            except Exception:
                existing = []
            combined = existing + records
        else:
            combined = records

        with open(self.json_path, 'w', encoding='utf-8') as jf:
            json.dump(combined, jf, ensure_ascii=False, indent=2)

        print(f"\nEredmények elmentve ide: {self.csv_path} (append módban) és {self.json_path}")

    def save_models(self, models):
        os.makedirs(self.model_dir, exist_ok=True)

        for name, model in models.items():
            model_path = os.path.join(self.model_dir, f"banksim_clustering_baseline_{name}.joblib")
            joblib.dump(model, model_path)
            print(f"{name} elmentve: {model_path}")


if __name__ == "__main__":
    preprocessor = Preprocessor(input_path="../data/raw/BankSim.csv")
    cleaned_df = preprocessor.run()

    for method in ("time", "customer"):
        experiment = BaselineFraudExperiment(data_path=preprocessor.output_path, split_method=method)
        experiment.run()