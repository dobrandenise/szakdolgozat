import json
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from clustering import Clusterer
from data_split import DataSplit
from feature_engineering import FeatureEngineering
from models import RandomForestModel
from preprocessor import Preprocessor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "BankSim.csv"
RESULTS_PATH = PROJECT_ROOT / "metrics" / "rf_comparison_time.json"
RF_PARAMS = {
	"n_estimators": 727,
	"max_depth": 16,
	"max_features": 0.45115101356959497,
	"min_samples_leaf": 7,
	"random_state": 42,
}
KMEANS_FEATURES = ["cluster_id", "dist_to_centroid"]
HDBSCAN_FEATURES = ["hdbscan_cluster_id", "is_hdbscan_noise"]


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
	categorical_cols = [
		col for col in ["age", "gender", "category", "merchant", "customer"]
		if col in df.columns
	]
	if not categorical_cols:
		return df.copy()

	encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
	encoded_df = df.copy()
	encoded_df[categorical_cols] = encoder.fit_transform(encoded_df[categorical_cols])
	return encoded_df


def encode_split(
	train_df: pd.DataFrame,
	val_df: pd.DataFrame,
	test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	split_sizes = [len(train_df), len(val_df), len(test_df)]
	encoded_df = encode_categoricals(
		pd.concat([train_df, val_df, test_df], ignore_index=True)
	)
	train_end = split_sizes[0]
	val_end = train_end + split_sizes[1]
	return (
		encoded_df.iloc[:train_end].reset_index(drop=True),
		encoded_df.iloc[train_end:val_end].reset_index(drop=True),
		encoded_df.iloc[val_end:].reset_index(drop=True),
	)


def run_random_forest(
	name: str,
	splits: dict[str, pd.DataFrame],
	cluster_features: list[str],
) -> dict:
	train_df = splits["train"]
	test_df = splits["test"]
	model = RandomForestModel(random_state=42)
	model.fit(
		X_train=train_df.drop(columns=["fraud"]),
		y_train=train_df["fraud"],
		params=RF_PARAMS,
	)
	result = model.evaluate(
		X_test=test_df.drop(columns=["fraud"]),
		y_test=test_df["fraud"],
		model_name=name,
	)
	result["cluster_features"] = cluster_features
	return result


def run_comparison(input_path: str | Path = RAW_DATA_PATH) -> list[dict]:
	cleaned_df = Preprocessor(input_path=input_path).run()
	dataset_df = FeatureEngineering(cleaned_df).add_dataset_wide_features()

	splitter = DataSplit(dataset_df)
	train_df, val_df, test_df = splitter.time_split()

	feature_engineering = FeatureEngineering(dataset_df)
	train_df, val_df, test_df = (
		feature_engineering.fit_and_add_train_dependent_features(
			train_df, splitter, split_name="time"
		)
	)
	train_df, val_df, test_df = encode_split(train_df, val_df, test_df)
	base_splits = {"train": train_df, "val": val_df, "test": test_df}

	clusterer = Clusterer(
		{"n_clusters": 3, "random_state": 42, "n_init": 10},
		{"min_cluster_size": 400, "min_samples": 30},
	)
	clusterer.fit(train_df)

	kmeans_splits = {
		split: clusterer.predict_kmeans(df)
		for split, df in base_splits.items()
	}
	hdbscan_splits = {
		split: clusterer.predict_hdbscan(df)
		for split, df in base_splits.items()
	}
	both_splits = {
		split: clusterer.transform(df)
		for split, df in base_splits.items()
	}

	experiments = [
		("without_cluster_features", base_splits, []),
		("kmeans_features", kmeans_splits, KMEANS_FEATURES),
		("hdbscan_features", hdbscan_splits, HDBSCAN_FEATURES),
		("kmeans_and_hdbscan_features", both_splits, KMEANS_FEATURES + HDBSCAN_FEATURES),
	]
	results = [
		run_random_forest(name, splits, cluster_features)
		for name, splits, cluster_features in experiments
	]

	RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
	with RESULTS_PATH.open("w", encoding="utf-8") as results_file:
		json.dump(results, results_file, indent=2, ensure_ascii=False)

	print(f"A négy RF összehasonlító eredmény elmentve ide: {RESULTS_PATH}")
	return results


if __name__ == "__main__":
	run_comparison()
