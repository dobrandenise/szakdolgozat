from pathlib import Path

#from clustering import Clusterer
from feature_engineering import FeatureEngineering
from preprocessor import Preprocessor
from data_split import DataSplit


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "BankSim.csv"
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "BankSim_cleaned.csv"
FEATURED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "BankSim_featured.csv"

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

    customer_train_path = PROJECT_ROOT / "data" / "optimalization" / "customer_train.csv"
    time_train_path = PROJECT_ROOT / "data" / "optimalization" / "time_train.csv"
    customer_train.to_csv(customer_train_path, index=False)
    time_train.to_csv(time_train_path, index=False)

"""
    clusterer = Clusterer()
    customer_train = clusterer.fit_transform(customer_train)
    time_train = clusterer.fit_transform(time_train)

    customer_train_path = PROJECT_ROOT / "data" / "processed" / "dataset_customer_split_train.csv"
    time_train_path = PROJECT_ROOT / "data" / "processed" / "dataset_time_split_train.csv"
    customer_train.to_csv(customer_train_path, index=False)
    time_train.to_csv(time_train_path, index=False)
    customer_test_path = PROJECT_ROOT / "data" / "processed" / "dataset_customer_split_test.csv"
    time_test_path = PROJECT_ROOT / "data" / "processed" / "dataset_time_split_test.csv"
    customer_test.to_csv(customer_test_path, index=False)
    time_test.to_csv(time_test_path, index=False)

    print(f"Pipeline futtatva. Tisztított adat elmentve.")
"""

if __name__ == "__main__":
    run_pipeline()
