from pathlib import Path
import os
import json
from datetime import datetime, timezone

import cupy as cp
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from feature_engineering import FeatureEngineering
from models import (
    GraphSAGEModel,
    RandomForestModel,
    XGBoostModel,
    classification_metrics,
    compute_cost,
)
from preprocessor import Preprocessor
from data_split import DataSplit


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "BankSim.csv"
RF_XGB_EXCLUDED_COLUMNS = [
    "fraud", "_pipeline_row_id", "customer",
    "customer_fraud_prior", "risk_score_composite",
]
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


def _align_probabilities(reference_df, scored_df, probabilities, id_column):
    scored_ids = pd.Index(scored_df[id_column])
    if not scored_ids.is_unique or not reference_df[id_column].is_unique:
        raise ValueError("A hibrid predikciók tranzakcióazonosítói nem egyediek.")

    positions = scored_ids.get_indexer(reference_df[id_column])
    if (
        len(reference_df) != len(scored_df)
        or len(positions) != len(reference_df)
        or (positions < 0).any()
    ):
        raise ValueError("Az XGBoost és a GNN adathalmazai nem egyeznek meg.")

    probabilities = np.asarray(probabilities).reshape(-1)
    if len(probabilities) != len(scored_df):
        raise ValueError("Az XGBoost predikciók száma nem egyezik a sorok számával.")
    return probabilities[positions]


def _align_rows(reference_df, rows_df, id_column):
    row_ids = pd.Index(rows_df[id_column])
    if not row_ids.is_unique or not reference_df[id_column].is_unique:
        raise ValueError("A hibrid adatok tranzakcióazonosítói nem egyediek.")

    positions = row_ids.get_indexer(reference_df[id_column])
    if len(reference_df) != len(rows_df) or (positions < 0).any():
        raise ValueError("A hibrid modell adathalmazai nem egyeznek meg.")
    return rows_df.iloc[positions].reset_index(drop=True)


def _tune_probability_ensemble(
    y_val,
    amounts_val,
    xgb_probabilities,
    gnn_probabilities,
    thresholds=None,
):
    y_val = np.asarray(y_val, dtype=np.int32)
    amounts_val = np.asarray(amounts_val, dtype=float)
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.8, 71)

    best_cost = float("inf")
    best_weight = None
    best_threshold = None
    fraud_mask = y_val == 1
    non_fraud_mask = ~fraud_mask

    for gnn_weight in np.linspace(0.0, 1.0, 21):
        probabilities = (
            (1.0 - gnn_weight) * xgb_probabilities
            + gnn_weight * gnn_probabilities
        )
        predictions = probabilities[:, None] >= thresholds[None, :]
        costs = (
            (amounts_val[:, None] * (fraud_mask[:, None] & ~predictions)).sum(axis=0)
            + 10.0 * (non_fraud_mask[:, None] & predictions).sum(axis=0)
            + 10.0 * (fraud_mask[:, None] & predictions).sum(axis=0)
        )
        threshold_idx = int(np.argmin(costs))
        if costs[threshold_idx] < best_cost:
            best_cost = float(costs[threshold_idx])
            best_weight = float(gnn_weight)
            best_threshold = float(thresholds[threshold_idx])

    return best_weight, best_threshold


def encode_categoricals(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    categorical_cols = [
        col for col in ["age", "gender", "category", "merchant"]
        if col in train.columns
    ]
    encoded_frames = [train.copy(), val.copy(), test.copy()]
    if not categorical_cols:
        return tuple(encoded_frames)

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(train[categorical_cols])
    for source, encoded in zip((train, val, test), encoded_frames):
        encoded[categorical_cols] = encoder.transform(source[categorical_cols])
    return tuple(encoded_frames)

def run_pipeline(input_path=RAW_DATA_PATH, model="XGB"):
    """Előfeldolgozza, feature-özi, felosztja és modellezi az adatokat."""
    
    preprocessor = Preprocessor(input_path=input_path)
    cleaned_df = preprocessor.run()

    fe = FeatureEngineering(cleaned_df)
    dataset_wide_df = fe.add_dataset_wide_features()
    ensemble_id_column = "_pipeline_row_id"
    dataset_wide_df[ensemble_id_column] = np.arange(len(dataset_wide_df))

    data_splitter = DataSplit(dataset_wide_df)
    customer_train, customer_val, customer_test = data_splitter.customer_split()
    #time_train, time_val, time_test = data_splitter.time_split()

    fe_wide = FeatureEngineering(dataset_wide_df)
    customer_train, customer_val, customer_test = fe_wide.fit_and_add_train_dependent_features(customer_train, data_splitter, split_name="customer")
    #time_train, time_val, time_test = fe_wide.fit_and_add_train_dependent_features(time_train, data_splitter, split_name="time")

    graph_train, graph_val, graph_test = (
        customer_train.copy(), customer_val.copy(), customer_test.copy()
    )
    raw_split = {
        "train": customer_train.copy(),
        "val": customer_val.copy(),
        "test": customer_test.copy(),
    }

    customer_df = pd.concat([customer_train, customer_val, customer_test], ignore_index=True)

    def _subset(df, ids_df):
        return (
            df[df[ensemble_id_column].isin(ids_df[ensemble_id_column])]
            .sort_values(ensemble_id_column)
            .reset_index(drop=True)
        )

    customer_train = _subset(customer_df, raw_split["train"])
    customer_val = _subset(customer_df, raw_split["val"])
    customer_test = _subset(customer_df, raw_split["test"])
    assert len(customer_train) == len(raw_split["train"])
    assert len(customer_val) == len(raw_split["val"])
    assert len(customer_test) == len(raw_split["test"])
    customer_train, customer_val, customer_test = encode_categoricals(
        customer_train, customer_val, customer_test
    )

    if(model == "RF"):
        print("---RandomForest modell---")
        RF = RandomForestModel()
        RF.fit(
            X_train=customer_train.drop(columns=RF_XGB_EXCLUDED_COLUMNS, errors="ignore"),
            y_train=customer_train["fraud"],
            params=RF_PARAMS,
        )
        threshold = RF.tune_threshold(
            X_val=customer_val.drop(columns=RF_XGB_EXCLUDED_COLUMNS, errors="ignore"),
            y_val=customer_val["fraud"],
            amounts_val=customer_val["amount"],
        )
        RF.evaluate(
            X_test=customer_test.drop(columns=RF_XGB_EXCLUDED_COLUMNS, errors="ignore"),
            y_test=customer_test["fraud"],
            amounts_test=customer_test["amount"],
            threshold=threshold,
            save_path=PROJECT_ROOT / "metrics" / "rf_evaluation.json",
        )
    elif(model == "XGB"):
        print("---XGBoost modell---")
        XGB= XGBoostModel()
        XGB.fit(
            X_train=customer_train.drop(columns=RF_XGB_EXCLUDED_COLUMNS, errors="ignore"),
            y_train=customer_train["fraud"],
            params=XGB_PARAMS,
        )
        threshold = XGB.tune_threshold(
            X_val=customer_val.drop(columns=RF_XGB_EXCLUDED_COLUMNS, errors="ignore"),
            y_val=customer_val["fraud"],
            amounts_val=customer_val["amount"],
        )
        XGB.evaluate(
            X_test=customer_test.drop(columns=RF_XGB_EXCLUDED_COLUMNS, errors="ignore"),
            y_test=customer_test["fraud"],
            amounts_test=customer_test["amount"],
            save_path=PROJECT_ROOT / "metrics" / "xgb_evaluation.json",
            threshold=threshold,
        )
    elif(model == "GNN"):
        print("---GraphSAGE modell---")
        GNN = GraphSAGEModel(eval_chunk_steps=1,pos_weight=40)
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
    elif(model == "HYBRID"):
        print("---XGBoost + GraphSAGE hibrid modell---")
        XGB = XGBoostModel()
        XGB.fit(
            X_train=customer_train.drop(columns=RF_XGB_EXCLUDED_COLUMNS, errors="ignore"),
            y_train=customer_train["fraud"],
            params=XGB_PARAMS,
        )

        GNN = GraphSAGEModel(eval_chunk_steps=1, pos_weight=40)
        GNN.fit(train_df=graph_train, val_df=graph_val)

        xgb_val_probabilities = cp.asnumpy(
            XGB.predict_proba(
                customer_val.drop(columns=RF_XGB_EXCLUDED_COLUMNS, errors="ignore")
            )
        )
        gnn_val_probabilities = GNN.predict_proba(
            graph_val.drop(columns=["fraud"]),
            context_df="train",
        )
        gnn_val_probabilities = np.asarray(gnn_val_probabilities).reshape(-1)
        if not np.array_equal(
            _align_rows(graph_val, customer_val, ensemble_id_column)["fraud"].to_numpy(),
            graph_val["fraud"].to_numpy(),
        ):
            raise ValueError("A hibrid validációs címkék sorrendje nem egyezik.")
        xgb_val_probabilities = _align_probabilities(
            graph_val, customer_val, xgb_val_probabilities, ensemble_id_column
        )
        gnn_weight, threshold = _tune_probability_ensemble(
            y_val=graph_val["fraud"],
            amounts_val=graph_val["amount"],
            xgb_probabilities=xgb_val_probabilities,
            gnn_probabilities=gnn_val_probabilities,
        )

        xgb_test_probabilities = cp.asnumpy(
            XGB.predict_proba(
                customer_test.drop(columns=RF_XGB_EXCLUDED_COLUMNS, errors="ignore")
            )
        )
        test_context = pd.concat(
            [graph_train.drop(columns=["fraud"]), graph_val.drop(columns=["fraud"])],
            ignore_index=True,
        )
        gnn_test_probabilities = GNN.predict_proba(
            graph_test.drop(columns=["fraud"]),
            context_df=test_context,
        )
        gnn_test_probabilities = np.asarray(gnn_test_probabilities).reshape(-1)
        if not np.array_equal(
            _align_rows(graph_test, customer_test, ensemble_id_column)["fraud"].to_numpy(),
            graph_test["fraud"].to_numpy(),
        ):
            raise ValueError("A hibrid tesztcímkék sorrendje nem egyezik.")
        xgb_test_probabilities = _align_probabilities(
            graph_test, customer_test, xgb_test_probabilities, ensemble_id_column
        )
        test_probabilities = (
            (1.0 - gnn_weight) * xgb_test_probabilities
            + gnn_weight * gnn_test_probabilities
        )
        predictions = (test_probabilities >= threshold).astype(int)
        metric_values = classification_metrics(
            graph_test["fraud"], test_probabilities, threshold=threshold
        )
        tn, fp, fn, tp = cp.asnumpy(
            metric_values["confusion_matrix"]
        ).ravel()
        y_test = graph_test["fraud"].to_numpy()
        amounts_test = graph_test["amount"].to_numpy(dtype=float)
        total_cost = compute_cost(y_test, predictions, amounts_test)
        cost_at_05 = compute_cost(
            y_test, (test_probabilities >= 0.5).astype(int), amounts_test
        )
        baseline_cost = compute_cost(
            y_test, np.zeros_like(y_test), amounts_test
        )
        metrics = {
            "model_name": "XGBoostGraphSAGEEnsemble",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_params": {
                "xgboost": XGB_PARAMS,
                "graphsage": {
                    "hidden_dim": GNN.hidden_dim,
                    "num_layers": GNN.num_layers,
                    "dropout": GNN.dropout,
                    "pos_weight": GNN.pos_weight,
                    "eval_chunk_steps": GNN.eval_chunk_steps,
                },
            },
            "combination": "weighted_probability_average",
            "gnn_weight": gnn_weight,
            "threshold": threshold,
            "n_test_samples": int(len(y_test)),
            "fraud_rate": float(y_test.mean()) if len(y_test) else 0.0,
            "AUC-ROC": metric_values["AUC-ROC"],
            "AUC-PR": metric_values["AUC-PR"],
            "precision_fraud": metric_values["precision_fraud"],
            "accuracy": metric_values["accuracy"],
            "recall_fraud": metric_values["recall_fraud"],
            "f1_fraud": metric_values["f1_fraud"],
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
            "cost": {
                "fp_cost": 10.0,
                "tp_cost": 10.0,
                "fn_multiplier": 1.0,
                "threshold": threshold,
                "total_cost": total_cost,
                "total_cost_at_0.5": cost_at_05,
                "baseline_cost": baseline_cost,
                "cost_savings": (
                    1 - total_cost / baseline_cost if baseline_cost else 0.0
                ),
            },
        }
        results_path = PROJECT_ROOT / "metrics" / "hybrid_evaluation.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        existing_results = []
        if results_path.exists():
            with results_path.open("r", encoding="utf-8") as results_file:
                existing_results = json.load(results_file)
            if not isinstance(existing_results, list):
                existing_results = [existing_results]
        existing_results.append(metrics)
        with results_path.open("w", encoding="utf-8") as results_file:
            json.dump(existing_results, results_file, indent=2, ensure_ascii=False)
        print(f"A hibrid modell eredménye elmentve ide: {results_path}")
    else:
        raise ValueError("A MODEL értéke csak RF, XGB, GNN vagy HYBRID lehet.")
    print("Pipeline futtatva.")


if __name__ == "__main__":
    model = os.getenv("MODEL", "XGB").strip().upper()
    if model not in {"RF", "XGB", "GNN", "HYBRID"}:
        raise SystemExit("A MODEL értéke csak RF, XGB, GNN vagy HYBRID lehet.")
    run_pipeline(model=model)
