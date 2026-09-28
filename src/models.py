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


# ---------------------------------------------------------------------------
# 1. CUDA Random Forest
# ---------------------------------------------------------------------------
 
class RandomForestModel:
    """
    cuML Random Forest a kézzel épített feature-táblán.
    """
 
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model: CuMLRandomForestClassifier | None = None
        self.best_params: dict | None = None
 
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

    def evaluate(self, X_test, y_test, save_path: str | Path | None = None, model_name: str = "RandomForestModel") -> dict:
        """Futási metrikák és a modell paraméterei mentése JSON fájlba."""
        if self.model is None:
            raise RuntimeError("A modellt előbb fit()-elni kell.")

        y_proba = self.predict_proba(X_test)
        metric_values = classification_metrics(y_test, y_proba)
        tn, fp, fn, tp = cp.asnumpy(metric_values["confusion_matrix"]).ravel()
        metrics = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(timespec="seconds") + "Z",
            "model_params": self.model.get_params(),
            "best_params": self.best_params,
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

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]
                else:
                    existing = []

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
 
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model: XGBClassifier | None = None
        self.best_params: dict | None = None
 
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

# ---------------------------------------------------------------------------
# 3. GraphSAGE (customer-merchant bipartit gráf)
# ---------------------------------------------------------------------------
 
class GraphSAGEModel:
    """
    Customer-merchant bipartit gráf, heterogén GraphSAGE réteggel (SAGEConv +
    to_hetero). Minden tranzakció egy (customer, merchant) él, a fraud a
    tranzakció (él) szintű célváltozó — a predikció a két végpont node
    embeddingjéből + él-feature-ökből (amount, category, history feature-ök)
    történik egy MLP-vel.
 
    Induktivitás: a gráf a train szakasz customereiből/merchantjeiből épül.
    A val/test szakaszban megjelenő ÚJ customer node üres (nulla-vektor)
    kezdő feature-rel kerül be, a GraphSAGE aggregátorai a szomszédos
    (train-ben már látott) merchant node-okból építik fel az embeddingjét —
    ez pont az induktivitási tulajdonság, amiért ezt a modellt választottuk.
    """
 
    def __init__(self, customer_feature_cols, merchant_feature_cols,
                 edge_feature_cols, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.3, lr: float = 1e-3, device: str | None = None):
        self.customer_feature_cols = customer_feature_cols
        self.merchant_feature_cols = merchant_feature_cols
        self.edge_feature_cols = edge_feature_cols
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
 
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.customer_index = None
        self.merchant_index = None
 
    def _build_hetero_graph(self, df: pd.DataFrame, fit_index: bool):
        import torch
        from torch_geometric.data import HeteroData
 
        if fit_index:
            self.customer_index = {c: i for i, c in enumerate(sorted(df["customer"].unique()))}
            self.merchant_index = {m: i for i, m in enumerate(sorted(df["merchant"].unique()))}
        else:
            # induktív eset: ismeretlen customer/merchant kap egy új, üres slotot
            for c in df["customer"].unique():
                if c not in self.customer_index:
                    self.customer_index[c] = len(self.customer_index)
            for m in df["merchant"].unique():
                if m not in self.merchant_index:
                    self.merchant_index[m] = len(self.merchant_index)
 
        n_customers = len(self.customer_index)
        n_merchants = len(self.merchant_index)
 
        customer_feat = np.zeros((n_customers, len(self.customer_feature_cols)), dtype=np.float32)
        merchant_feat = np.zeros((n_merchants, len(self.merchant_feature_cols)), dtype=np.float32)
 
        cust_agg = df.groupby("customer")[self.customer_feature_cols].mean()
        for cust, row in cust_agg.iterrows():
            customer_feat[self.customer_index[cust]] = row.to_numpy(dtype=np.float32)
 
        merch_agg = df.groupby("merchant")[self.merchant_feature_cols].mean()
        for merch, row in merch_agg.iterrows():
            merchant_feat[self.merchant_index[merch]] = row.to_numpy(dtype=np.float32)
 
        src = df["customer"].map(self.customer_index).to_numpy(dtype=np.int64)
        dst = df["merchant"].map(self.merchant_index).to_numpy(dtype=np.int64)
        edge_feat = df[self.edge_feature_cols].to_numpy(dtype=np.float32)
        labels = df["fraud"].to_numpy(dtype=np.float32)
 
        data = HeteroData()
        data["customer"].x = torch.from_numpy(customer_feat)
        data["merchant"].x = torch.from_numpy(merchant_feat)
        data["customer", "transacts", "merchant"].edge_index = torch.from_numpy(np.stack([src, dst]))
        data["customer", "transacts", "merchant"].edge_attr = torch.from_numpy(edge_feat)
        data["customer", "transacts", "merchant"].y = torch.from_numpy(labels)
        # reverse él az üzenetváltáshoz (merchant -> customer irányban is folyjon információ)
        data["merchant", "rev_transacts", "customer"].edge_index = torch.from_numpy(np.stack([dst, src]))
 
        return data.to(self.device)
 
    def _build_model(self):
        import torch
        import torch.nn as nn
        from torch_geometric.nn import SAGEConv, to_hetero
 
        class GNNEncoder(nn.Module):
            def __init__(self, hidden_dim, num_layers, dropout):
                super().__init__()
                self.convs = nn.ModuleList()
                for _ in range(num_layers):
                    self.convs.append(SAGEConv((-1, -1), hidden_dim))
                self.dropout = dropout
 
            def forward(self, x, edge_index):
                for i, conv in enumerate(self.convs):
                    x = conv(x, edge_index)
                    if i < len(self.convs) - 1:
                        x = torch.relu(x)
                        x = torch.dropout(x, p=self.dropout, train=self.training)
                return x
 
        encoder = GNNEncoder(self.hidden_dim, self.num_layers, self.dropout)
        return encoder
 
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
            epochs: int = 100, patience: int = 8) -> "GraphSAGEModel":
        import torch
        import torch.nn as nn
        from torch_geometric.nn import to_hetero
 
        train_graph = self._build_hetero_graph(train_df, fit_index=True)
        val_graph = self._build_hetero_graph(val_df, fit_index=False)
 
        encoder = self._build_model()
        encoder = to_hetero(encoder, train_graph.metadata(), aggr="mean").to(self.device)
 
        edge_dim = len(self.edge_feature_cols)
        classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + edge_dim, self.hidden_dim),
            nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        ).to(self.device)
 
        params = list(encoder.parameters()) + list(classifier.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
        criterion = nn.BCEWithLogitsLoss()
 
        def forward_edges(graph):
            out = encoder(graph.x_dict, graph.edge_index_dict)
            edge_index = graph["customer", "transacts", "merchant"].edge_index
            edge_attr = graph["customer", "transacts", "merchant"].edge_attr
            cust_emb = out["customer"][edge_index[0]]
            merch_emb = out["merchant"][edge_index[1]]
            combined = torch.cat([cust_emb, merch_emb, edge_attr], dim=-1)
            return classifier(combined).squeeze(-1)
 
        from torchmetrics.classification import BinaryAveragePrecision

        val_average_precision = BinaryAveragePrecision(thresholds=None).to(self.device)
        best_val_pr_auc = -1.0
        best_state = None
        epochs_no_improve = 0
 
        for epoch in range(epochs):
            encoder.train()
            classifier.train()
            optimizer.zero_grad()
            logits = forward_edges(train_graph)
            loss = criterion(logits, train_graph["customer", "transacts", "merchant"].y)
            loss.backward()
            optimizer.step()
 
            encoder.eval()
            classifier.eval()
            with torch.no_grad():
                val_logits = forward_edges(val_graph)
                val_probs = torch.sigmoid(val_logits)
                val_labels = val_graph["customer", "transacts", "merchant"].y.long()
            val_pr_auc = float(val_average_precision(val_probs, val_labels).item())
            val_average_precision.reset()
            scheduler.step(val_pr_auc)
 
            if val_pr_auc > best_val_pr_auc:
                best_val_pr_auc = val_pr_auc
                best_state = (
                    {k: v.clone() for k, v in encoder.state_dict().items()},
                    {k: v.clone() for k, v in classifier.state_dict().items()},
                )
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break
 
        if best_state is not None:
            encoder.load_state_dict(best_state[0])
            classifier.load_state_dict(best_state[1])
 
        self.model = (encoder, classifier, forward_edges)
        return self
 
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        import torch
 
        if self.model is None:
            raise RuntimeError("A modellt előbb fit()-elni kell.")
        encoder, classifier, forward_edges = self.model
 
        graph = self._build_hetero_graph(df, fit_index=False)
        encoder.eval()
        classifier.eval()
        with torch.no_grad():
            logits = forward_edges(graph)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs