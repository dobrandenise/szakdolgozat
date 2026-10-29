from pathlib import Path
import os
import json
import statistics
import time
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
    _synchronize_accelerators,
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
	"n_estimators": 747,
	"max_depth": 10,
	"learning_rate": 0.015613690158405065,
	"subsample": 0.6387713040323718,
	"colsample_bytree": 0.6680837435903048,
    "reg_lambda": 5.142633483677166,
	"random_state": 42,
	"tree_method": "hist",
	"device": "cuda",
}
RF_PARAMS = {
    'n_estimators': 593,
    'max_depth': 16, 
    'max_features': 0.48508241086548237, 
    'min_samples_leaf': 5,
	"random_state": 42,
}
GNN_PARAMS = {
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.2890856026631795,
    "lr": 0.0003017731896320215,
    "sup_batch_size": 8192,
    "pos_weight": 20,
    "eval_chunk_steps": 1
}
RF_THRESHOLD = 0.1
XGB_THRESHOLD = 0.36
GNN_THRESHOLD = 0.85
HYBRID_GNN_WEIGHT = 0.415
HYBRID_THRESHOLD = 0.53
MODEL_REPEAT_COUNT = 10


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

def run_pipeline(input_path=RAW_DATA_PATH, model="XGB", run_number=None, run_group=None):
    """Előfeldolgozza, feature-özi, felosztja és modellezi az adatokat."""

    pipeline_started = time.perf_counter()
    preprocessor = Preprocessor(input_path=input_path)
    cleaned_df = preprocessor.run()

    fe = FeatureEngineering(cleaned_df)
    dataset_wide_df = fe.add_dataset_wide_features()
    ensemble_id_column = "_pipeline_row_id"
    dataset_wide_df[ensemble_id_column] = np.arange(len(dataset_wide_df))

    data_splitter = DataSplit(dataset_wide_df)
    customer_train, customer_val, customer_test = data_splitter.customer_split()

    fe_wide = FeatureEngineering(dataset_wide_df)
    customer_train, customer_val, customer_test = fe_wide.fit_and_add_train_dependent_features(customer_train, data_splitter, split_name="customer")

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
    preprocessing_seconds = time.perf_counter() - pipeline_started
    hyperparameter_tuning_seconds = 0.0
    training_seconds = 0.0
    threshold_tuning_seconds = 0.0
    inference_by_model_seconds = None
    training_by_model_seconds = None

    if(model == "RF"):
        print("---RandomForest modell---")
        RF = RandomForestModel()
        _synchronize_accelerators()
        training_started = time.perf_counter()
        RF.fit(
            X_train=customer_train.drop(columns=RF_XGB_EXCLUDED_COLUMNS, errors="ignore"),
            y_train=customer_train["fraud"],
            params=RF_PARAMS,
        )
        _synchronize_accelerators()
        training_seconds = time.perf_counter() - training_started
        result = RF.evaluate(
            X_test=customer_test.drop(columns=RF_XGB_EXCLUDED_COLUMNS, errors="ignore"),
            y_test=customer_test["fraud"],
            amounts_test=customer_test["amount"],
            threshold=RF_THRESHOLD,
        )
        results_path = PROJECT_ROOT / "metrics" / "rf_evaluation.json"
    elif(model == "XGB"):
        print("---XGBoost modell---")
        XGB = XGBoostModel()
        _synchronize_accelerators()
        training_started = time.perf_counter()
        XGB.fit(
            X_train=customer_train.drop(columns=RF_XGB_EXCLUDED_COLUMNS, errors="ignore"),
            y_train=customer_train["fraud"],
            params=XGB_PARAMS
        )
        _synchronize_accelerators()
        training_seconds = time.perf_counter() - training_started
        result = XGB.evaluate(
            X_test=customer_test.drop(columns=RF_XGB_EXCLUDED_COLUMNS, errors="ignore"),
            y_test=customer_test["fraud"],
            amounts_test=customer_test["amount"],
            threshold=XGB_THRESHOLD,
        )
        results_path = PROJECT_ROOT / "metrics" / "xgb_evaluation.json"
    elif(model == "GNN"):
        print("---GraphSAGE modell---")

        GNN = GraphSAGEModel(
            hidden_dim=GNN_PARAMS["hidden_dim"],
            num_layers=GNN_PARAMS["num_layers"],
            dropout=GNN_PARAMS["dropout"],
            lr=GNN_PARAMS["lr"],
            sup_batch_size=GNN_PARAMS["sup_batch_size"],
            eval_chunk_steps=GNN_PARAMS["eval_chunk_steps"],
            pos_weight=GNN_PARAMS["pos_weight"],
        )
        _synchronize_accelerators()
        training_started = time.perf_counter()
        GNN.fit(train_df=graph_train, val_df=graph_val)
        _synchronize_accelerators()
        gnn_fit_seconds = time.perf_counter() - training_started
        preprocessing_seconds += GNN.preprocessing_seconds
        training_seconds = gnn_fit_seconds - GNN.preprocessing_seconds
        test_context = pd.concat(
            [graph_train.drop(columns=["fraud"]), graph_val.drop(columns=["fraud"])],
            ignore_index=True,
        )
        result = GNN.evaluate(
            X_test=graph_test.drop(columns=["fraud"]),
            y_test=graph_test["fraud"],
            amounts_test=graph_test["amount"],
            threshold=GNN_THRESHOLD,
            context_df=test_context,
        )
        results_path = PROJECT_ROOT / "metrics" / "gnn_evaluation.json"
    elif(model == "HYBRID"):
        print("---XGBoost + GraphSAGE hibrid modell---")
        XGB = XGBoostModel()
        _synchronize_accelerators()
        xgb_training_started = time.perf_counter()
        XGB.fit(
            X_train=customer_train.drop(columns=RF_XGB_EXCLUDED_COLUMNS, errors="ignore"),
            y_train=customer_train["fraud"],
            params=XGB_PARAMS,
        )
        _synchronize_accelerators()
        xgb_training_seconds = time.perf_counter() - xgb_training_started

        GNN = GraphSAGEModel(
            hidden_dim=GNN_PARAMS["hidden_dim"],
            num_layers=GNN_PARAMS["num_layers"],
            dropout=GNN_PARAMS["dropout"],
            lr=GNN_PARAMS["lr"],
            sup_batch_size=GNN_PARAMS["sup_batch_size"],
            pos_weight=GNN_PARAMS["pos_weight"],
            eval_chunk_steps=GNN_PARAMS["eval_chunk_steps"]
        )
        _synchronize_accelerators()
        gnn_training_started = time.perf_counter()
        GNN.fit(train_df=graph_train, val_df=graph_val)
        _synchronize_accelerators()
        gnn_fit_seconds = time.perf_counter() - gnn_training_started
        preprocessing_seconds += GNN.preprocessing_seconds
        gnn_training_seconds = gnn_fit_seconds - GNN.preprocessing_seconds
        training_seconds = xgb_training_seconds + gnn_training_seconds
        training_by_model_seconds = {
            "xgboost": xgb_training_seconds,
            "graphsage": gnn_training_seconds,
        }

        test_context = pd.concat(
            [graph_train.drop(columns=["fraud"]), graph_val.drop(columns=["fraud"])],
            ignore_index=True,
        )
        _synchronize_accelerators()
        inference_started = time.perf_counter()
        xgb_test_probabilities = cp.asnumpy(
            XGB.predict_proba(
                customer_test.drop(columns=RF_XGB_EXCLUDED_COLUMNS, errors="ignore")
            )
        )
        _synchronize_accelerators()
        xgb_inference_seconds = time.perf_counter() - inference_started
        inference_started = time.perf_counter()
        gnn_test_probabilities = GNN.predict_proba(
            graph_test.drop(columns=["fraud"]),
            context_df=test_context,
        )
        _synchronize_accelerators()
        gnn_inference_seconds = time.perf_counter() - inference_started
        inference_by_model_seconds = {
            "xgboost": xgb_inference_seconds,
            "graphsage": gnn_inference_seconds,
        }
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
            (1.0 - HYBRID_GNN_WEIGHT) * xgb_test_probabilities
            + HYBRID_GNN_WEIGHT * gnn_test_probabilities
        )
        predictions = (test_probabilities >= HYBRID_THRESHOLD).astype(int)
        metric_values = classification_metrics(
            graph_test["fraud"], test_probabilities, threshold=HYBRID_THRESHOLD
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
        result = {
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
            "gnn_weight": HYBRID_GNN_WEIGHT,
            "threshold": HYBRID_THRESHOLD,
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
                "threshold": HYBRID_THRESHOLD,
                "total_cost": total_cost,
                "total_cost_at_0.5": cost_at_05,
                "baseline_cost": baseline_cost,
                "cost_savings": (
                    1 - total_cost / baseline_cost if baseline_cost else 0.0
                ),
            },
        }
        results_path = PROJECT_ROOT / "metrics" / "hybrid_evaluation.json"
    else:
        raise ValueError("A MODEL értéke csak RF, XGB, GNN vagy HYBRID lehet.")

    timing_seconds = result.get("timing_seconds", {})
    timing_seconds.update({
        "preprocessing": preprocessing_seconds,
        "hyperparameter_tuning": hyperparameter_tuning_seconds,
        "training": training_seconds,
        "threshold_tuning": threshold_tuning_seconds,
        "total_pipeline": time.perf_counter() - pipeline_started,
    })
    if inference_by_model_seconds is not None:
        timing_seconds["inference"] = sum(inference_by_model_seconds.values())
        timing_seconds["inference_by_model"] = inference_by_model_seconds
    if training_by_model_seconds is not None:
        timing_seconds["training_by_model"] = training_by_model_seconds
    inference_seconds = timing_seconds["inference"]
    if result["n_test_samples"] > 0:
        timing_seconds["inference_ms_per_sample"] = (
            inference_seconds * 1000 / result["n_test_samples"]
        )
        timing_seconds["inference_samples_per_second"] = (
            result["n_test_samples"] / inference_seconds
            if inference_seconds > 0
            else None
        )
    result["timing_seconds"] = timing_seconds
    if run_number is not None:
        result["run_number"] = run_number
    if run_group is not None:
        result["run_group"] = run_group
    results_path.parent.mkdir(parents=True, exist_ok=True)
    existing_results = []
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as results_file:
            existing_results = json.load(results_file)
        if not isinstance(existing_results, list):
            existing_results = [existing_results]
    existing_results.append(result)
    with results_path.open("w", encoding="utf-8") as results_file:
        json.dump(existing_results, results_file, indent=2, ensure_ascii=False)
    print(f"Az eredmény elmentve ide: {results_path}")
    print("Pipeline futtatva.")
    return result


if __name__ == "__main__":
    model = os.getenv("MODEL", "XGB").strip().upper()
    if model not in {"RF", "XGB", "GNN", "HYBRID"}:
        raise SystemExit("A MODEL értéke csak RF, XGB, GNN vagy HYBRID lehet.")
    if model in {"RF", "XGB", "GNN", "HYBRID"}:
        run_group = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        results = [
            run_pipeline(model=model, run_number=run_number, run_group=run_group)
            for run_number in range(1, MODEL_REPEAT_COUNT + 1)
        ]
        timing_keys = ("preprocessing", "training", "inference", "total_pipeline")
        timing_summary = {
            "run_count": len(results),
            **{
                f"average_{key}_seconds": statistics.fmean(
                    result["timing_seconds"][key] for result in results
                )
                for key in timing_keys
            },
        }
        results_path = PROJECT_ROOT / "metrics" / f"{model.lower()}_evaluation.json"
        with results_path.open("r", encoding="utf-8") as results_file:
            all_results = json.load(results_file)
        for result in all_results:
            if result.get("run_group") == run_group:
                result["timing_summary"] = timing_summary
        with results_path.open("w", encoding="utf-8") as results_file:
            json.dump(all_results, results_file, indent=2, ensure_ascii=False)
        print(f"{model} átlagos időeredmények: {timing_summary}")
    else:
        run_pipeline(model=model)
