from pathlib import Path

from clustering import Clusterer
from feature_engineering import FeatureEngineering
from preprocessor import Preprocessor
from data_split import DataSplit


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "BankSim.csv"
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "BankSim_cleaned.csv"

def run_pipeline(
    input_path=RAW_DATA_PATH,
    cleaned_output_path=CLEANED_DATA_PATH,
):
    """Run preprocessing, clustering, and feature engineering in order."""
    preprocessor = Preprocessor(
        input_path=str(input_path),
        output_path=str(cleaned_output_path),
    )
    cleaned_df = preprocessor.run()

    clustering_df = preprocessor.encode_categorical_columns(cleaned_df.copy())
    clusterer = Clusterer()
    clustered_df = clusterer.run(clustering_df)
    cluster_columns = ["cluster_id", "dist_to_centroid", "is_hdbscan_noise"]
    cleaned_df[cluster_columns] = clustered_df[cluster_columns]

    feature_engineering = FeatureEngineering(cleaned_df)
    engineered_df = feature_engineering.run_all()
    data_splitter = DataSplit(engineered_df)
    customer_train_df, customer_val_df, customer_test_df = data_splitter.customer_split()
    time_train_df, time_val_df, time_test_df = data_splitter.time_split()

    customer_test_path = PROJECT_ROOT / "data" / "processed" / "dataset_customer_split_test.csv"
    time_test_path = PROJECT_ROOT / "data" / "processed" / "dataset_time_split_test.csv"
    customer_test_df.to_csv(customer_test_path, index=False)
    time_test_df.to_csv(time_test_path, index=False)

    print(f"Pipeline futtatva. Tisztított adat elmentve: {cleaned_output_path}")
    


if __name__ == "__main__":
    run_pipeline()
