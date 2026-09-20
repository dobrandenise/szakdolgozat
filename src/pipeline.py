from pathlib import Path

from clustering import Clusterer
from feature_engineering import FeatureEngineering
from preprocessor import Preprocessor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "BankSim.csv"
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "BankSim_cleaned.csv"
ENGINEERED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_engineered.csv"


def run_pipeline(
    input_path=RAW_DATA_PATH,
    cleaned_output_path=CLEANED_DATA_PATH,
    engineered_output_path=ENGINEERED_DATA_PATH,
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

    engineered_output_path = Path(engineered_output_path)
    engineered_output_path.parent.mkdir(parents=True, exist_ok=True)
    engineered_df.to_csv(engineered_output_path, index=False)
    print(f"Feature-engineered adat elmentve ide: {engineered_output_path}")

    return engineered_df


if __name__ == "__main__":
    run_pipeline()
