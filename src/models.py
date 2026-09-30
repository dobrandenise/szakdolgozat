"""
Három modellcsalád implementációja a fraud detection szakdolgozathoz:
 
    1. RandomForestModel  - cuML CUDA Random Forest
    2. XGBoostModel        - CUDA histogram gradient boosting
    3. GraphSAGEModel      - customer-merchant bipartit gráf, induktív node-kezeléssel
 
Feltételezett bemenet (illeszkedik a meglévő BaselineFraudExperiment / DataSplit
kimenetéhez):
 
    - RandomForestModel / XGBoostModel:
        már kódolt (OrdinalEncoder-rel fit-elt) X_train / X_val / X_test
        DataFrame-ek és y_train / y_val / y_test Series-ek, ugyanúgy, ahogy az
        encode_features(train_features, test_features) előállítja őket.
 
  - GraphSAGEModel:
        a nyers (de tisztított) tranzakciós train/val/test DataFrame-ek, az
        alábbi oszlopokkal: customer, merchant, category, amount, step, fraud
        (+ opcionálisan a 4. fejezetben épített history/ratio/prior/cluster
        feature-ök numerikus oszlopokként).
 
Mind a három osztály egységes felületet ad:
 
    model.fit(...)
    model.predict_proba(...)  -> 1D numpy array a fraud-valószínűségekkel
 
így közvetlenül behelyettesíthető a meglévő compare_models() kiértékelő
logikába (PR-AUC, threshold-optimalizáció stb.), a 7-8. fejezetben rögzített
protokoll szerint (time-based split, fit kizárólag a train szakaszon).
 
FÜGGŐSÉGEK:
    RAPIDS cuML, CuPy, cuDF, XGBoost, Optuna, PyTorch, TorchMetrics,
    PyTorch Geometric.
"""
 
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import cupy as cp
import cudf
from cuml.ensemble import RandomForestClassifier as CuMLRandomForestClassifier
from cuml.metrics import roc_auc_score
from xgboost import XGBClassifier
 
 
# ---------------------------------------------------------------------------
# GPU helpers and metrics
# ---------------------------------------------------------------------------

def _to_gpu_features(X) -> cudf.DataFrame | cp.ndarray:
    if isinstance(X, cudf.DataFrame):
        return X
    if isinstance(X, pd.DataFrame):
        return cudf.from_pandas(X)
    return cp.asarray(X)


def _to_gpu_labels(y) -> cudf.Series | cp.ndarray:
    if isinstance(y, cudf.Series):
        return y.astype("int32")
    if isinstance(y, pd.Series):
        return cudf.from_pandas(y.astype("int32"))
    return cp.asarray(y, dtype=cp.int32)


def _to_cupy(values) -> cp.ndarray:
    if isinstance(values, cudf.DataFrame):
        return values.to_cupy()
    if isinstance(values, cudf.Series):
        return values.to_cupy()
    if isinstance(values, (pd.Series, pd.DataFrame)):
        return cp.asarray(values.to_numpy())
    return cp.asarray(values)


def _average_precision_score(y_true, y_score) -> float:
    labels = _to_cupy(y_true).ravel().astype(cp.int32)
    scores = _to_cupy(y_score).ravel()
    order = cp.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    # Evaluate only at the end of each tied-score group, matching average precision.
    group_ends = cp.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    true_positives = cp.cumsum(sorted_labels)[group_ends]
    total_positives = cp.sum(labels)
    positive_increments = cp.diff(cp.r_[cp.array([0], dtype=true_positives.dtype), true_positives])
    ranks = cp.arange(1, len(sorted_labels) + 1)[group_ends]
    precision = true_positives / ranks
    return float((cp.sum(precision * positive_increments) / total_positives).item())


def classification_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    labels = _to_cupy(y_true).ravel().astype(cp.int32)
    scores = _to_cupy(y_score).ravel()
    predictions = (scores >= threshold).astype(cp.int32)

    tn = cp.sum((labels == 0) & (predictions == 0))
    fp = cp.sum((labels == 0) & (predictions == 1))
    fn = cp.sum((labels == 1) & (predictions == 0))
    tp = cp.sum((labels == 1) & (predictions == 1))
    accuracy = (tn + tp) / cp.maximum(tn + fp + fn + tp, 1)
    precision = tp / cp.maximum(tp + fp, 1)
    recall = tp / cp.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / cp.maximum(precision + recall, 1e-12)
    matrix = cp.stack((cp.stack((tn, fp)), cp.stack((fn, tp))))

    return {
        "AUC-ROC": float(roc_auc_score(labels, scores)),
        "AUC-PR": _average_precision_score(labels, scores),
        "accuracy": float(accuracy.item()),
        "precision_fraud": float(precision.item()),
        "recall_fraud": float(recall.item()),
        "f1_fraud": float(f1.item()),
        "confusion_matrix": matrix,
    }

def compute_cost(y_true, y_pred, amounts, fp_cost=10.0, tp_cost=10.0,
                 fn_multiplier=1.0):
    """FN = összeg * fn_multiplier; FP és TP = fix költség."""
    y_true = cp.asnumpy(_to_cupy(y_true))
    y_pred = cp.asnumpy(_to_cupy(y_pred))
    amounts = cp.asnumpy(_to_cupy(amounts)).astype(float)

    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    tp_mask = (y_true == 1) & (y_pred == 1)

    fn_cost_total = np.sum(amounts[fn_mask] * fn_multiplier)
    return float(fn_cost_total + fp_cost * fp_mask.sum() + tp_cost * tp_mask.sum())


def find_optimal_threshold(y_true, y_proba, amounts, fp_cost=10.0, tp_cost=10.0,
                           fn_multiplier=1.0, thresholds=None):
    """Azt a küszöböt keresi, amelyiknél a teljes költség minimális."""
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.8, 69)
    y_proba = cp.asnumpy(_to_cupy(y_proba))

    costs = np.array([
        compute_cost(y_true, (y_proba >= t).astype(int), amounts,
                     fp_cost, tp_cost, fn_multiplier)
        for t in thresholds
    ])
    best_idx = int(np.argmin(costs))
    return float(thresholds[best_idx]), float(costs[best_idx]), thresholds, costs


# ---------------------------------------------------------------------------
# 1. CUDA Random Forest
# ---------------------------------------------------------------------------
 
class RandomForestModel:
    """
    cuML Random Forest a kézzel épített feature-táblán.
    """
 
    def __init__(self, random_state: int = 42, fp_cost: float = 10.0,
                 tp_cost: float = 10.0, fn_multiplier: float = 1.0):
        self.random_state = random_state
        self.model: CuMLRandomForestClassifier | None = None
        self.best_params: dict | None = None
        self.threshold = 0.5
        self.fp_cost = fp_cost
        self.tp_cost = tp_cost
        self.fn_multiplier = fn_multiplier
 
    def tune(self, X_train, y_train, X_val, y_val, n_trials: int = 100) -> dict:
        import optuna
 
        def objective(trial: "optuna.Trial") -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_int("max_depth", 4, 24),
                "max_features": trial.suggest_float("max_features", 0.2, 1.0),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
                "random_state": self.random_state,
            }
            clf = CuMLRandomForestClassifier(**params)
            clf.fit(_to_gpu_features(X_train), _to_gpu_labels(y_train))
            val_proba = _to_cupy(clf.predict_proba(_to_gpu_features(X_val)))[:, 1]
            return _average_precision_score(_to_gpu_labels(y_val), val_proba)
 
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        self.best_params = study.best_params
        return self.best_params
 
    def fit(self, X_train, y_train, params: dict | None = None) -> "RandomForestModel":
        params = params or self.best_params or {
            "n_estimators": 400,
            "max_depth": 12,
            "class_weight": "balanced",
            "random_state": self.random_state,
        }
        self.model = CuMLRandomForestClassifier(**params)
        self.model.fit(_to_gpu_features(X_train), _to_gpu_labels(y_train))
        return self
 
    def predict_proba(self, X) -> cp.ndarray:
        if self.model is None:
            raise RuntimeError("A modellt előbb fit()-elni kell.")
        return _to_cupy(self.model.predict_proba(_to_gpu_features(X)))[:, 1]

    def tune_threshold(self, X_val, y_val, amounts_val, thresholds=None) -> float:
        """Költségminimalizáló küszöb keresése a VALIDÁCIÓS halmazon."""
        if self.model is None:
            raise RuntimeError("A modellt előbb fit()-elni kell.")
        val_proba = self.predict_proba(X_val)
        self.threshold, _, _, _ = find_optimal_threshold(
            y_val, val_proba, amounts_val,
            fp_cost=self.fp_cost, tp_cost=self.tp_cost,
            fn_multiplier=self.fn_multiplier, thresholds=thresholds,
        )
        return self.threshold

    def evaluate(self, X_test, y_test, save_path: str | Path | None = None,
                 model_name: str = "RandomForestModel", amounts_test=None,
                 threshold: float | None = None) -> dict:
        """Futási metrikák és a modell paraméterei mentése JSON fájlba."""
        if self.model is None:
            raise RuntimeError("A modellt előbb fit()-elni kell.")

        threshold = self.threshold if threshold is None else threshold
        y_proba = self.predict_proba(X_test)
        y_pred = (y_proba >= threshold).astype(int)
        
        metric_values = classification_metrics(y_test, y_proba, threshold=threshold)
        tn, fp, fn, tp = cp.asnumpy(metric_values["confusion_matrix"]).ravel()
        
        metrics = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(timespec="seconds") + "Z",
            "model_params": self.model.get_params(),
            "best_params": self.best_params,
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
        }

        if amounts_test is not None:
            amounts_arr = np.asarray(amounts_test, dtype=float)
            y_arr = np.asarray(y_test)
            cost_opt = compute_cost(y_arr, y_pred, amounts_arr, self.fp_cost,
                                    self.tp_cost, self.fn_multiplier)
            cost_default = compute_cost(y_arr, (y_proba >= 0.5).astype(int),
                                        amounts_arr, self.fp_cost, self.tp_cost,
                                        self.fn_multiplier)
            cost_baseline = compute_cost(
                y_arr, np.zeros_like(y_arr), amounts_arr,
                self.fp_cost, self.tp_cost, self.fn_multiplier,
            )
            metrics["cost"] = {
                "fp_cost": self.fp_cost,
                "tp_cost": self.tp_cost,
                "fn_multiplier": self.fn_multiplier,
                "threshold": threshold,
                "total_cost": cost_opt,
                "total_cost_at_0.5": cost_default,
                "baseline_cost": cost_baseline,
                "cost_savings": 1 - cost_opt / cost_baseline if cost_baseline else 0.0,
            }

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            existing = []
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]

            existing.append(metrics)

            with path.open("w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)

        return metrics

# ---------------------------------------------------------------------------
# 2. XGBoost CUDA histogram classifier
# ---------------------------------------------------------------------------
 
class XGBoostModel:
    """
    XGBoost binary classifier CUDA-s histogram tree builderrel.
    """
 
    def __init__(self, random_state: int = 42, fp_cost: float = 10.0,
                 tp_cost: float = 10.0, fn_multiplier: float = 1.0):
        self.random_state = random_state
        self.model: XGBClassifier | None = None
        self.best_params: dict | None = None
        self.threshold = 0.5
        self.fp_cost = fp_cost
        self.tp_cost = tp_cost
        self.fn_multiplier = fn_multiplier
 
    def tune(self, X_train, y_train, X_val, y_val, n_trials: int = 50) -> dict:
        import optuna
 
        def objective(trial: "optuna.Trial") -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "random_state": self.random_state,
                "tree_method": "hist",
                "device": "cuda",
                "scale_pos_weight": float(
                    (y_train == 0).sum() / max((y_train == 1).sum(), 1)
                ),
            }
            clf = XGBClassifier(**params)
            clf.fit(_to_gpu_features(X_train), _to_gpu_labels(y_train))
            val_proba = cp.asarray(clf.predict_proba(_to_gpu_features(X_val)))[:, 1]
            return _average_precision_score(_to_gpu_labels(y_val), val_proba)
 
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        self.best_params = study.best_params
        return self.best_params
 
    def fit(self, X_train, y_train, params: dict | None = None) -> "XGBoostModel":
        params = params or self.best_params or {
            "n_estimators": 400,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.random_state,
            "tree_method": "hist",
            "device": "cuda",
            "scale_pos_weight": float(
                (y_train == 0).sum() / max((y_train == 1).sum(), 1)
            ),
        }
        params = dict(params)
        params.setdefault("tree_method", "hist")
        params.setdefault("device", "cuda")
        params.setdefault(
            "scale_pos_weight",
            float((y_train == 0).sum() / max((y_train == 1).sum(), 1)),
        )
        self.model = XGBClassifier(**params)
        self.model.fit(_to_gpu_features(X_train), _to_gpu_labels(y_train))
        return self
 
    def predict_proba(self, X) -> cp.ndarray:
        if self.model is None:
            raise RuntimeError("A modellt előbb fit()-elni kell.")
        return cp.asarray(self.model.predict_proba(_to_gpu_features(X)))[:, 1]

    def tune_threshold(self, X_val, y_val, amounts_val, thresholds=None) -> float:
        """Költségminimalizáló küszöb keresése a validációs halmazon."""
        if self.model is None:
            raise RuntimeError("A modellt előbb fit()-elni kell.")
        val_proba = self.predict_proba(X_val)
        self.threshold, _, _, _ = find_optimal_threshold(
            y_val, val_proba, amounts_val,
            fp_cost=self.fp_cost, tp_cost=self.tp_cost,
            fn_multiplier=self.fn_multiplier, thresholds=thresholds,
        )
        return self.threshold

    def evaluate(self, X_test, y_test, save_path: str | Path | None = None,
                 model_name: str = "XGBoostModel", amounts_test=None,
                 threshold: float | None = None) -> dict:
        """Futási metrikák és a modell paramétereinek mentése JSON fájlba."""
        if self.model is None:
            raise RuntimeError("A modellt előbb fit()-elni kell.")

        threshold = self.threshold if threshold is None else threshold
        y_proba = self.predict_proba(X_test)
        y_pred = (y_proba >= threshold).astype(int)
        metric_values = classification_metrics(y_test, y_proba, threshold=threshold)
        tn, fp, fn, tp = cp.asnumpy(metric_values["confusion_matrix"]).ravel()

        metrics = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(timespec="seconds") + "Z",
            "model_params": self.model.get_params(),
            "best_params": self.best_params,
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
        }

        if amounts_test is not None:
            amounts_arr = np.asarray(amounts_test, dtype=float)
            y_arr = np.asarray(y_test)
            cost_opt = compute_cost(
                y_arr, y_pred, amounts_arr, self.fp_cost, self.tp_cost,
                self.fn_multiplier,
            )
            cost_default = compute_cost(
                y_arr, (y_proba >= 0.5).astype(int), amounts_arr,
                self.fp_cost, self.tp_cost, self.fn_multiplier,
            )
            cost_baseline = compute_cost(
                y_arr, np.zeros_like(y_arr), amounts_arr,
                self.fp_cost, self.tp_cost, self.fn_multiplier,
            )
            metrics["cost"] = {
                "fp_cost": self.fp_cost,
                "tp_cost": self.tp_cost,
                "fn_multiplier": self.fn_multiplier,
                "threshold": threshold,
                "total_cost": cost_opt,
                "total_cost_at_0.5": cost_default,
                "baseline_cost": cost_baseline,
                "cost_savings": 1 - cost_opt / cost_baseline if cost_baseline else 0.0,
            }

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            existing = []
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]

            existing.append(metrics)

            with path.open("w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)

        return metrics

# ---------------------------------------------------------------------------
# 3. GraphSAGE (customer-merchant bipartit gráf)
# ---------------------------------------------------------------------------
 

# Fraud-címkéből származtatott oszlopok: a GraphSAGE-ben NEM használhatók.
#   - *_fraud_prior (customer / merchant / category): a `fraud` címke kumulált,
#     simított átlaga. Még ha csak múltból is számolt, a val/test szakaszon
#     csak címke-visszacsatolással lenne újraszámolható, és a GNN feladata
#     éppen az, hogy ezt a szerkezetből tanulja meg.
#   - risk_score_composite: a három prior + is_amount_outlier determinisztikus
#     súlyozott összege (0.35 / 0.35 / 0.15 / 0.15) -> ugyanazt a címke-információt
#     hordozza.
#   - category_is_high_risk: a kategória teljes train-beli fraud-rátájából
#     küszöbölt, statikus jelző (a saját és a jövőbeli címkéket is látta).
GRAPH_LABEL_DERIVED_COLS = (
    "fraud",
    "customer_fraud_prior",
    "merchant_fraud_prior",
    "category_fraud_prior",
    "risk_score_composite",
    "category_is_high_risk",
)

# Statikus node-attribútumok (idővel nem változnak, a tranzakció pillanatában
# ismertek -> induktívan is biztonságosak). One-hot kódolást kapnak.
GRAPH_CUSTOMER_FEATURES = ("age", "gender")
GRAPH_MERCHANT_FEATURES = ("category",)

# Tranzakció-szintű él-feature-ök: mind csak a tranzakció saját értékeiből vagy
# a múltbeli (history) adatból számolt.
GRAPH_EDGE_FEATURES = (
    "amount_log",
    "amount_zscore_customer",
    "amount_ratio_customer",
    "amount_zscore_category",
    "amount_ratio_merchant_avg",
    "customer_tx_count_hist",
    "customer_days_since_first_tx",
    "customer_category_diversity_hist",
    "is_new_category_for_customer",
    "merchant_tx_count_hist",
    "category_tx_count_hist",
    "tx_count_last_7d",
    "tx_count_last_30d",
    "days_since_last_tx_customer",
    "cumulative_spend_30d",
    "is_amount_outlier",
)


class GraphSAGEModel:
    """
    Customer-merchant bipartit gráf, heterogén GraphSAGE réteggel (SAGEConv +
    to_hetero). Minden tranzakció egy (customer, merchant) él, a fraud a
    tranzakció (él) szintű célváltozó -- a predikció a két végpont node
    embeddingjéből + él-feature-ökből történik egy MLP-vel.

    Induktivitás: a node-index és az összes előfeldolgozás (one-hot szótárak,
    skálázó) kizárólag a train szakaszból tanulódik. A val/test szakaszban
    megjelenő ÚJ customer/merchant új node-ként kerül a gráfba, a train-ben
    tanult statikus attribútumokkal (age, gender / category; ismeretlen érték ->
    csupa nulla one-hot). Ha customer_feature_cols=[], a customer node kezdő
    feature-je csupa nulla, és az embeddingjét kizárólag a szomszédos merchant
    node-ok adják.

    Adatszivárgás elleni szabályok (az előző review alapján javítva):
      1. Nincs fraud-címkéből származtatott feature (GRAPH_LABEL_DERIVED_COLS);
         konstruktor hibát dob, ha ilyet adsz meg (allow_label_derived=True-val
         ablation céljából felülírható).
      2. A node-feature-ök statikus attribútumok, NEM ablakonként újraszámolt
         tranzakció-átlagok, így train/val/test között konzisztensek.
      3. Skálázó és szótárak fit-je csak a train-en történik.
      4. A val/test gráf = context (korábbi szakaszok élei) + a kiértékelt
         szakasz élei; a loss/metrika csak a kiértékelt szakasz éleire fut.
         A SAGEConv nem használ edge_attr-t és a node-feature-ök statikusak,
         ezért sem címke, sem él-feature nem áramlik az üzenetküldésen át;
         csak a gráf szerkezete (ki kivel tranzaktált) látszik.
      5. eval_chunk_steps megadásakor a predict_proba a kiértékelt szakaszt
         `step` szerinti darabokra bontja, és egy darab gráfja a context-ből és
         a df-ből is csak a KORÁBBI step-ek éleit (+ a darab saját éleit)
         látja -> a szerkezeti "előre-nézés" is megszűnik. Ez customer-split
         adatnál (val/test customerek ugyanabban az időtartományban mozognak,
         mint a train) különösen számít. eval_chunk_steps=None esetén a
         context minden éle látszik (csak szerkezet, címke nem).

    Használat:
        model = GraphSAGEModel()
        model.fit(train_df, val_df)
        p_val  = model.predict_proba(val_df)                      # context = train
        p_test = model.predict_proba(test_df,
                     context_df=pd.concat([train_df, val_df]))    # context = train+val
        p_trn  = model.predict_proba(train_df, context_df=None)   # csak a train gráf
    """

    def __init__(self, customer_feature_cols=None, merchant_feature_cols=None,
                 edge_feature_cols=None, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.3, lr: float = 1e-3, device: str | None = None,
                 sup_batch_size: int = 16384, pos_weight="auto",
                 eval_chunk_steps: int | None = None,
                 allow_label_derived: bool = False, random_state: int = 42,
                 fp_cost: float = 10.0, tp_cost: float = 10.0,
                 fn_multiplier: float = 1.0):
        self.customer_feature_cols = list(
            GRAPH_CUSTOMER_FEATURES if customer_feature_cols is None else customer_feature_cols)
        self.merchant_feature_cols = list(
            GRAPH_MERCHANT_FEATURES if merchant_feature_cols is None else merchant_feature_cols)
        self.edge_feature_cols = list(
            GRAPH_EDGE_FEATURES if edge_feature_cols is None else edge_feature_cols)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.sup_batch_size = sup_batch_size
        self.pos_weight = pos_weight
        self.eval_chunk_steps = eval_chunk_steps
        self.allow_label_derived = allow_label_derived
        self.random_state = random_state
        self.fp_cost = fp_cost
        self.tp_cost = tp_cost
        self.fn_multiplier = fn_multiplier
        self.threshold = 0.5

        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.customer_index: dict | None = None     # csak a train-ben látott node-ok
        self.merchant_index: dict | None = None
        self._train_df: pd.DataFrame | None = None  # alapértelmezett context a predikcióhoz
        self._vocab: dict = {}
        self._edge_num_cols: list = []
        self._edge_bin_cols: list = []
        self._edge_mean = None
        self._edge_std = None
        self._edge_dim = 0
        self._fitted = False

        self._check_feature_columns()

    # ------------------------------------------------------------------
    # Előfeldolgozás (minden fit csak a train-en)
    # ------------------------------------------------------------------

    def _check_feature_columns(self) -> None:
        if self.allow_label_derived:
            return
        used = set(self.customer_feature_cols) | set(self.merchant_feature_cols) \
            | set(self.edge_feature_cols)
        bad = sorted(used & set(GRAPH_LABEL_DERIVED_COLS))
        if bad:
            raise ValueError(
                f"Fraud-címkéből származtatott oszlop(ok) a feature-ök között: {bad}. "
                "Ez target leakage; vedd ki őket (vagy allow_label_derived=True ablationhöz)."
            )

    def _require_columns(self, df: pd.DataFrame, need_label: bool = False) -> None:
        need = {"customer", "merchant", "step"} | set(self.customer_feature_cols) \
            | set(self.merchant_feature_cols) | set(self.edge_feature_cols)
        if need_label:
            need.add("fraud")
        missing = sorted(need - set(df.columns))
        if missing:
            raise KeyError(f"Hiányzó oszlop(ok): {missing}")

    @staticmethod
    def _slog(x: np.ndarray) -> np.ndarray:
        """Előjeles log1p: a ferde (count, ratio, spend) oszlopok tömörítése."""
        return np.sign(x) * np.log1p(np.abs(x))

    def _fit_preprocessing(self, train_df: pd.DataFrame) -> None:
        self._vocab = {
            col: sorted(train_df[col].unique())
            for col in self.customer_feature_cols + self.merchant_feature_cols
        }
        self._edge_num_cols, self._edge_bin_cols = [], []
        for col in self.edge_feature_cols:
            (self._edge_bin_cols if train_df[col].nunique() <= 2
             else self._edge_num_cols).append(col)

        if self._edge_num_cols:
            z = self._slog(train_df[self._edge_num_cols].to_numpy(dtype=np.float64))
            self._edge_mean = z.mean(axis=0)
            std = z.std(axis=0)
            std[std < 1e-8] = 1.0
            self._edge_std = std
        else:
            self._edge_mean = np.zeros(0)
            self._edge_std = np.ones(0)
        self._edge_dim = len(self._edge_num_cols) + len(self._edge_bin_cols)

    def _encode_edges(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        if self._edge_num_cols:
            z = self._slog(df[self._edge_num_cols].to_numpy(dtype=np.float64))
            z = np.clip((z - self._edge_mean) / self._edge_std, -10.0, 10.0)
            parts.append(z)
        if self._edge_bin_cols:
            parts.append(df[self._edge_bin_cols].to_numpy(dtype=np.float64))
        return np.concatenate(parts, axis=1).astype(np.float32)

    @staticmethod
    def _extend_index(index: dict, values) -> None:
        """Ismeretlen node új, folytatólagos indexet kap (induktív eset)."""
        for v in pd.unique(values):
            if v not in index:
                index[v] = len(index)

    def _node_features(self, graph_df: pd.DataFrame, key: str, index: dict,
                       cols: list) -> np.ndarray:
        """Statikus attribútumok one-hot kódolása a train-ben tanult szótárral."""
        n = len(index)
        if not cols:
            return np.zeros((n, 1), dtype=np.float32)
        first = graph_df.drop_duplicates(key)[[key] + cols].set_index(key)
        table = first.reindex(list(index))        # a dict sorrendje = node-index sorrend
        parts = []
        for col in cols:
            vocab = self._vocab[col]
            pos = pd.Index(vocab).get_indexer(table[col])
            pos = np.where(pos < 0, len(vocab), pos)      # ismeretlen -> utolsó slot
            onehot = np.zeros((n, len(vocab) + 1), dtype=np.float32)
            onehot[np.arange(n), pos] = 1.0
            parts.append(onehot)
        return np.concatenate(parts, axis=1)

    # ------------------------------------------------------------------
    # Gráfépítés
    # ------------------------------------------------------------------

    def _build_graph(self, context_df: pd.DataFrame | None, target_df: pd.DataFrame) -> dict:
        """
        Gráf = context élek + target élek (üzenetküldéshez). A supervision
        (loss / predikció) kizárólag a target élekre vonatkozik.
        """
        import torch
        from torch_geometric.data import HeteroData

        if context_df is None or len(context_df) == 0:
            graph_df = target_df.reset_index(drop=True)
        else:
            graph_df = pd.concat([context_df, target_df], ignore_index=True)
        n_ctx = len(graph_df) - len(target_df)

        cust_index = dict(self.customer_index)       # másolat: predikció nem módosítja az állapotot
        merch_index = dict(self.merchant_index)
        self._extend_index(cust_index, graph_df["customer"].to_numpy())
        self._extend_index(merch_index, graph_df["merchant"].to_numpy())

        cust_x = self._node_features(graph_df, "customer", cust_index, self.customer_feature_cols)
        merch_x = self._node_features(graph_df, "merchant", merch_index, self.merchant_feature_cols)
        src = graph_df["customer"].map(cust_index).to_numpy(dtype=np.int64)
        dst = graph_df["merchant"].map(merch_index).to_numpy(dtype=np.int64)

        data = HeteroData()
        data["customer"].x = torch.from_numpy(cust_x)
        data["merchant"].x = torch.from_numpy(merch_x)
        edge_index = torch.from_numpy(np.stack([src, dst]))
        data["customer", "transacts", "merchant"].edge_index = edge_index
        # fordított él: merchant -> customer irányban is folyjon információ
        data["merchant", "rev_transacts", "customer"].edge_index = edge_index.flip(0)
        data = data.to(self.device)

        if "fraud" in target_df.columns:
            y = target_df["fraud"].to_numpy(dtype=np.float32)
        else:
            y = np.zeros(len(target_df), dtype=np.float32)

        def _t(a):
            return torch.from_numpy(np.ascontiguousarray(a)).to(self.device)

        return {
            "data": data,
            "src": _t(src[n_ctx:]),
            "dst": _t(dst[n_ctx:]),
            "attr": _t(self._encode_edges(target_df)),
            "y": _t(y),
        }

    def _build_model(self):
        import torch.nn as nn
        from torch_geometric.nn import SAGEConv

        class GNNEncoder(nn.Module):
            def __init__(self, hidden_dim, num_layers, dropout):
                super().__init__()
                self.convs = nn.ModuleList(
                    [SAGEConv((-1, -1), hidden_dim) for _ in range(num_layers)]
                )
                # nn.Dropout modul (nem torch.dropout train=self.training):
                # a to_hetero fx-trace-e után is követi a train()/eval() állapotot.
                self.drop = nn.Dropout(dropout)

            def forward(self, x, edge_index):
                for i, conv in enumerate(self.convs):
                    x = conv(x, edge_index)
                    if i < len(self.convs) - 1:
                        x = self.drop(x.relu())
                return x

        return GNNEncoder(self.hidden_dim, self.num_layers, self.dropout)

    def _edge_logits(self, graph: dict, idx=None):
        """A supervision élek logitjai (idx: opcionális mini-batch index)."""
        import torch

        encoder, classifier = self.model
        out = encoder(graph["data"].x_dict, graph["data"].edge_index_dict)
        src, dst, attr = graph["src"], graph["dst"], graph["attr"]
        if idx is not None:
            src, dst, attr = src[idx], dst[idx], attr[idx]
        z = torch.cat([out["customer"][src], out["merchant"][dst], attr], dim=-1)
        return classifier(z).squeeze(-1)

    def _set_seed(self) -> None:
        import torch
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    # ------------------------------------------------------------------
    # Tanítás
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
            epochs: int = 100, patience: int = 8) -> "GraphSAGEModel":
        import torch
        import torch.nn as nn
        from torch_geometric.nn import to_hetero
        from torchmetrics.classification import BinaryAveragePrecision

        self._require_columns(train_df, need_label=True)
        self._require_columns(val_df, need_label=True)
        self._set_seed()

        # 1) minden "fit" kizárólag a train-en
        self._fit_preprocessing(train_df)
        self.customer_index, self.merchant_index = {}, {}
        self._extend_index(self.customer_index, np.sort(train_df["customer"].unique()))
        self._extend_index(self.merchant_index, np.sort(train_df["merchant"].unique()))
        self._train_df = train_df

        # 2) gráfok: train = csak train élek; val = train (context) + val élek
        train_graph = self._build_graph(None, train_df)
        val_graph = self._build_graph(train_df, val_df)

        encoder = to_hetero(
            self._build_model(), train_graph["data"].metadata(), aggr="mean"
        ).to(self.device)
        classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + self._edge_dim, self.hidden_dim),
            nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        ).to(self.device)

        # lazy (-1) paraméterek inicializálása az optimizer létrehozása ELŐTT
        with torch.no_grad():
            encoder(train_graph["data"].x_dict, train_graph["data"].edge_index_dict)
        self.model = (encoder, classifier)

        params = list(encoder.parameters()) + list(classifier.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=4)   # PR-AUC-ot maximalizálunk

        y_train = train_graph["y"]
        if self.pos_weight == "auto":
            pw = float((y_train == 0).sum() / max(float((y_train == 1).sum()), 1.0))
        elif self.pos_weight is None:
            pw = 1.0
        else:
            pw = float(self.pos_weight)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=self.device))

        val_ap = BinaryAveragePrecision(thresholds=None).to(self.device)
        val_labels = val_graph["y"].long()
        n_sup = len(y_train)
        best_val, best_state, no_improve = -1.0, None, 0

        for epoch in range(epochs):
            encoder.train()
            classifier.train()
            perm = torch.randperm(n_sup, device=self.device)
            for idx in perm.split(self.sup_batch_size):      # több gradiens-lépés / epoch
                optimizer.zero_grad()
                loss = criterion(self._edge_logits(train_graph, idx), y_train[idx])
                loss.backward()
                optimizer.step()

            encoder.eval()
            classifier.eval()
            with torch.no_grad():
                val_probs = torch.sigmoid(self._edge_logits(val_graph))
            val_pr_auc = float(val_ap(val_probs, val_labels).item())
            val_ap.reset()
            scheduler.step(val_pr_auc)

            if val_pr_auc > best_val:
                best_val = val_pr_auc
                best_state = (
                    {k: v.clone() for k, v in encoder.state_dict().items()},
                    {k: v.clone() for k, v in classifier.state_dict().items()},
                )
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            encoder.load_state_dict(best_state[0])
            classifier.load_state_dict(best_state[1])
        self.best_val_pr_auc = best_val
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Predikció
    # ------------------------------------------------------------------

    def predict_proba(self, df: pd.DataFrame, context_df="train") -> np.ndarray:
        """
        Fraud-valószínűségek a df soraira (eredeti sorrendben).

        context_df: a gráf üzenetküldő éleit adó, már ismert tranzakciók.
              "train"      -> a fit()-nél használt train (val predikcióhoz)
              DataFrame    -> pl. concat([train, val]) a teszthez
              None         -> nincs context (pl. a train halmaz pontozásához)
        """
        import torch

        if not self._fitted:
            raise RuntimeError("A modellt előbb fit()-elni kell.")
        self._require_columns(df)

        if isinstance(context_df, str):
            if context_df != "train":
                raise ValueError("context_df: 'train', DataFrame vagy None lehet.")
            context_df = self._train_df
        encoder, classifier = self.model
        encoder.eval()
        classifier.eval()

        if not self.eval_chunk_steps:
            graph = self._build_graph(context_df, df)
            with torch.no_grad():
                return torch.sigmoid(self._edge_logits(graph)).cpu().numpy()

        # szigorú időrendi kiértékelés: egy darab csak a korábbi éleket látja
        steps = np.sort(df["step"].unique())
        step_arr = df["step"].to_numpy()
        probs = np.zeros(len(df), dtype=np.float32)
        k = int(self.eval_chunk_steps)
        for start in range(0, len(steps), k):
            chunk = steps[start:start + k]
            mask = np.isin(step_arr, chunk)
            ctx_past = None
            if context_df is not None and len(context_df):
                ctx_past = context_df[context_df["step"].to_numpy() < chunk[0]]
            parts = [p for p in (ctx_past, df[step_arr < chunk[0]])
                     if p is not None and len(p)]
            ctx = pd.concat(parts, ignore_index=True) if parts else None
            graph = self._build_graph(ctx, df[mask])
            with torch.no_grad():
                probs[mask] = torch.sigmoid(self._edge_logits(graph)).cpu().numpy()
        return probs

    def tune_threshold(self, X_val, y_val, amounts_val, thresholds=None) -> float:
        """Költségminimalizáló küszöb keresése a validációs halmazon."""
        if not self._fitted:
            raise RuntimeError("A modellt előbb fit()-elni kell.")
        val_proba = self.predict_proba(X_val, context_df="train")
        self.threshold, _, _, _ = find_optimal_threshold(
            y_val, val_proba, amounts_val,
            fp_cost=self.fp_cost, tp_cost=self.tp_cost,
            fn_multiplier=self.fn_multiplier, thresholds=thresholds,
        )
        return self.threshold

    def evaluate(self, X_test, y_test, save_path: str | Path | None = None,
                 model_name: str = "GraphSAGEModel", amounts_test=None,
                 threshold: float | None = None, context_df="train") -> dict:
        """Értékeli a gráfmodellt; a küszöböt és költséget is menti."""
        if not self._fitted:
            raise RuntimeError("A modellt előbb fit()-elni kell.")

        threshold = self.threshold if threshold is None else threshold
        y_proba = self.predict_proba(X_test, context_df=context_df)
        y_pred = (y_proba >= threshold).astype(int)
        metric_values = classification_metrics(y_test, y_proba, threshold=threshold)
        tn, fp, fn, tp = cp.asnumpy(metric_values["confusion_matrix"]).ravel()
        model_params = {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "sup_batch_size": self.sup_batch_size,
            "pos_weight": self.pos_weight,
            "eval_chunk_steps": self.eval_chunk_steps,
        }
        metrics = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(timespec="seconds") + "Z",
            "model_params": model_params,
            "best_val_pr_auc": self.best_val_pr_auc,
            "threshold": threshold,
            "n_test_samples": int(len(y_test)),
            "fraud_rate": float(np.asarray(y_test).mean()) if len(y_test) else 0.0,
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
        }

        if amounts_test is not None:
            amounts_arr = np.asarray(amounts_test, dtype=float)
            y_arr = np.asarray(y_test)
            cost_opt = compute_cost(
                y_arr, y_pred, amounts_arr, self.fp_cost, self.tp_cost,
                self.fn_multiplier,
            )
            cost_default = compute_cost(
                y_arr, (y_proba >= 0.5).astype(int), amounts_arr,
                self.fp_cost, self.tp_cost, self.fn_multiplier,
            )
            cost_baseline = compute_cost(
                y_arr, np.zeros_like(y_arr), amounts_arr,
                self.fp_cost, self.tp_cost, self.fn_multiplier,
            )
            metrics["cost"] = {
                "fp_cost": self.fp_cost,
                "tp_cost": self.tp_cost,
                "fn_multiplier": self.fn_multiplier,
                "threshold": threshold,
                "total_cost": cost_opt,
                "total_cost_at_0.5": cost_default,
                "baseline_cost": cost_baseline,
                "cost_savings": 1 - cost_opt / cost_baseline if cost_baseline else 0.0,
            }

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            existing = []
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]
            existing.append(metrics)
            with path.open("w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)

        return metrics