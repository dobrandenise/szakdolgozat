from pathlib import Path
import os

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from clustering import Clusterer
from feature_engineering import FeatureEngineering
from models import RandomForestModel, XGBoostModel, GraphSAGEModel
from preprocessor import Preprocessor
from data_split import DataSplit


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "BankSim.csv"
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "BankSim_cleaned.csv"
FEATURED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "BankSim_featured.csv"
XGB_PARAMS = {
	"n_estimators": 577,
	"max_depth": 12,
	"learning_rate": 0.022070435848525118,
	"subsample": 0.9053079686928301,
	"colsample_bytree": 0.6209063298688097,
	"random_state": 42,
	"tree_method": "hist",
	"device": "cuda",
}
RF_PARAMS = {
	"n_estimators": 727,
	"max_depth": 16,
	"max_features": 0.45115101356959497,
	"min_samples_leaf": 7,
	"random_state": 42,
}
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

def run_pipeline(input_path=RAW_DATA_PATH, model="XGB"):
    """Run preprocessing, feature engineering, data splitting and clustering in order."""
    
    preprocessor = Preprocessor(input_path=RAW_DATA_PATH)
    cleaned_df = preprocessor.run()

    fe = FeatureEngineering(cleaned_df)
    dataset_wide_df = fe.add_dataset_wide_features()

    data_splitter = DataSplit(dataset_wide_df)
    customer_train, customer_val, customer_test = data_splitter.customer_split()
    #time_train, time_val, time_test = data_splitter.time_split()

    fe_wide = FeatureEngineering(dataset_wide_df)
    customer_train, customer_val, customer_test = fe_wide.fit_and_add_train_dependent_features(customer_train, data_splitter, split_name="customer")
    #time_train, time_val, time_test = fe_wide.fit_and_add_train_dependent_features(time_train, data_splitter, split_name="time")

    graph_train, graph_val, graph_test = (
        customer_train.copy(), customer_val.copy(), customer_test.copy()
    )

    customer_df = pd.concat([customer_train, customer_val, customer_test], ignore_index=True)
    #time_df = pd.concat([time_train, time_val, time_test], ignore_index=True)

    customer_df = encode_categoricals(customer_df)
    #time_df = encode_categoricals(time_df)

    data_customer_splitter = DataSplit(customer_df)
    customer_train, customer_val, customer_test = data_customer_splitter.customer_split()
    #data_time_splitter = DataSplit(time_df)
    #time_train, time_val, time_test = data_time_splitter.time_split()

    if(model == "RF"):
        print("---RandomForest modell---")
        RF = RandomForestModel()
        RF.fit(
            X_train=customer_train.drop(columns=["fraud"]),
            y_train=customer_train["fraud"],
            params=RF_PARAMS,
        )
        threshold = RF.tune_threshold(
            X_val=customer_val.drop(columns=["fraud"]),
            y_val=customer_val["fraud"],
            amounts_val=customer_val["amount"],
        )
        RF.evaluate(
            X_test=customer_test.drop(columns=["fraud"]),
            y_test=customer_test["fraud"],
            amounts_test=customer_test["amount"],
            threshold=threshold,
            save_path=PROJECT_ROOT / "metrics" / "rf_evaluation.json",
        )
    elif(model == "XGB"):
        print("---XGBoost modell---")
        XGB= XGBoostModel()
        XGB.fit(
            X_train=customer_train.drop(columns=["fraud"]),
            y_train=customer_train["fraud"],
            params=XGB_PARAMS,
        )
        threshold = XGB.tune_threshold(
            X_val=customer_val.drop(columns=["fraud"]),
            y_val=customer_val["fraud"],
            amounts_val=customer_val["amount"],
        )
        XGB.evaluate(
            X_test=customer_test.drop(columns=["fraud"]),
            y_test=customer_test["fraud"],
            amounts_test=customer_test["amount"],
            save_path=PROJECT_ROOT / "metrics" / "xgb_evaluation.json",
            threshold=threshold,
        )
    elif(model == "GNN"):
        print("---GraphSAGE modell---")
        GNN = GraphSAGEModel(eval_chunk_steps=1)
        GNN.fit(train_df=graph_train, val_df=graph_val)
        threshold = GNN.tune_threshold(
            X_val=graph_val.drop(columns=["fraud"]),
            y_val=graph_val["fraud"],
            amounts_val=graph_val["amount"],
        )
        test_context = pd.concat(
            [graph_train.drop(columns=["fraud"]), graph_val.drop(columns=["fraud"])],
            ignore_index=True,
        )
        GNN.evaluate(
            X_test=graph_test.drop(columns=["fraud"]),
            y_test=graph_test["fraud"],
            amounts_test=graph_test["amount"],
            threshold=threshold,
            context_df=test_context,
            save_path=PROJECT_ROOT / "metrics" / "gnn_evaluation.json",
        )
    else:
        raise ValueError("A MODEL környezeti változó értéke csak RF, XGB vagy GNN lehet.")
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
    model = os.getenv("MODEL", "XGB").strip().upper()
    if model not in {"RF", "XGB","GNN"}:
        raise SystemExit("A MODEL környezeti változó értéke csak RF, XGB vagy GNN lehet.")
    run_pipeline(model=model)
