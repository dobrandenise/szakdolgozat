from pathlib import Path

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from clustering import Clusterer
from feature_engineering import FeatureEngineering
from models import RandomForestModel
from preprocessor import Preprocessor
from data_split import DataSplit


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "BankSim.csv"
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "BankSim_cleaned.csv"
FEATURED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "BankSim_featured.csv"
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

def run_pipeline(input_path=RAW_DATA_PATH):
    """Run preprocessing, feature engineering, data splitting and clustering in order."""
    preprocessor = Preprocessor(input_path=str(input_path))
    cleaned_df = preprocessor.run()
    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)

    fe = FeatureEngineering(cleaned_df)
    dataset_wide_df = fe.add_dataset_wide_features()

    data_splitter = DataSplit(dataset_wide_df)
    customer_train, customer_val, customer_test = data_splitter.customer_split()
    time_train, time_val, time_test = data_splitter.time_split()

    fe_wide = FeatureEngineering(dataset_wide_df)
    customer_train, customer_val, customer_test = fe_wide.fit_and_add_train_dependent_features(customer_train, data_splitter, split_name="customer")
    time_train, time_val, time_test = fe_wide.fit_and_add_train_dependent_features(time_train, data_splitter, split_name="time")

    customer_df = pd.concat([customer_train, customer_val, customer_test], ignore_index=True)
    time_df = pd.concat([time_train, time_val, time_test], ignore_index=True)

    clusterer_high_signal = Clusterer({"n_clusters": 3, "random_state": 42, "n_init": 10}, {"min_cluster_size": 400, "min_samples": 30})
    customer_df = clusterer_high_signal.fit_transform(customer_train, customer_df)
    time_df = clusterer_high_signal.fit_transform(time_train, time_df)

    customer_df = encode_categoricals(customer_df)
    time_df = encode_categoricals(time_df)

    data_customer_splitter = DataSplit(customer_df)
    customer_train, customer_val, customer_test = data_customer_splitter.customer_split()
    data_time_splitter = DataSplit(time_df)
    time_train, time_val, time_test = data_time_splitter.time_split()

#---RandomForest modell---
    RF = RandomForestModel()
    RF.tune(
        X_train=customer_train.drop(columns=["fraud"]),
        y_train=customer_train["fraud"],
        X_val=customer_val.drop(columns=["fraud"]),
        y_val=customer_val["fraud"],
        n_trials=100,
    )
    RF.fit(
        X_train=customer_train.drop(columns=["fraud"]),
        y_train=customer_train["fraud"],
    )
    RF.evaluate(
        X_test=customer_test.drop(columns=["fraud"]),
        y_test=customer_test["fraud"],
        save_path=PROJECT_ROOT / "metrics" / "rf_evaluation.json",
    )

    """
    customer_train_path = PROJECT_ROOT / "data" / "processed" / "dataset_customer_split_train.csv"
    time_train_path = PROJECT_ROOT / "data" / "processed" / "dataset_time_split_train.csv"
    customer_train.to_csv(customer_train_path, index=False)
    time_train.to_csv(time_train_path, index=False)
    customer_test_path = PROJECT_ROOT / "data" / "processed" / "dataset_customer_split_test.csv"
    time_test_path = PROJECT_ROOT / "data" / "processed" / "dataset_time_split_test.csv"
    customer_test.to_csv(customer_test_path, index=False)
    time_test.to_csv(time_test_path, index=False)
    """
    print(f"Pipeline futtatva. Tisztított adat elmentve.")


if __name__ == "__main__":
    run_pipeline()
