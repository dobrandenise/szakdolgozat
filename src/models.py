"""
Három modellcsalád implementációja a fraud detection szakdolgozathoz:
 
  1. RandomForestModel  - tabuláris, kézzel épített feature-ökön
  2. HistGBModel         - tabuláris, kézzel épített feature-ökön
  3. GraphSAGEModel      - customer-merchant bipartit gráf, induktív node-kezeléssel
 
Feltételezett bemenet (illeszkedik a meglévő BaselineFraudExperiment / DataSplit
kimenetéhez):
 
  - RandomForestModel / HistGBModel:
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
    pip install scikit-learn optuna torch torch-geometric
    (az Optuna és a torch-geometric csak akkor kell, ha a tune()/GraphSAGE
    metódusokat ténylegesen használod)
"""
 
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
 
 
# ---------------------------------------------------------------------------
# 1. Random Forest
# ---------------------------------------------------------------------------
 
class RandomForestModel:
    """
    RF a kézzel épített feature-táblán. Optuna-alapú hangolás opcionális
    (tune=True), validation PR-AUC-ra optimalizálva, ahogy a 8. fejezetben
    rögzítettük a DNN-hez is.
    """
 
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model: RandomForestClassifier | None = None
        self.best_params: dict | None = None
 
    def tune(self, X_train, y_train, X_val, y_val, n_trials: int = 100) -> dict:
        import optuna
 
        def objective(trial: "optuna.Trial") -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_int("max_depth", 4, 24),
                "max_features": trial.suggest_float("max_features", 0.2, 1.0),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "class_weight": trial.suggest_categorical(
                    "class_weight", ["balanced", "balanced_subsample", None]
                ),
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            clf = RandomForestClassifier(**params)
            clf.fit(X_train, y_train)
            val_proba = clf.predict_proba(X_val)[:, 1]
            return average_precision_score(y_val, val_proba)
 
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
            "n_jobs": -1,
        }
        self.model = RandomForestClassifier(**params)
        self.model.fit(X_train, y_train)
        return self
 
    def predict_proba(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("A modellt előbb fit()-elni kell.")
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test, y_test, save_path: str | Path | None = None, model_name: str = "RandomForestModel") -> dict:
        """Futási metrikák és a modell paraméterei mentése JSON fájlba."""
        if self.model is None:
            raise RuntimeError("A modellt előbb fit()-elni kell.")

        y_proba = self.predict_proba(X_test)
        y_pred = self.model.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics = {
            "model_name": model_name,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model_params": self.model.get_params(),
            "best_params": self.best_params,
            "n_test_samples": int(len(y_test)),
            "fraud_rate": float(y_test.mean()) if len(y_test) else 0.0,
            "AUC-ROC": float(roc_auc_score(y_test, y_proba)),
            "AUC-PR": float(average_precision_score(y_test, y_proba)),
            "precision_fraud": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall_fraud": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_fraud": float(f1_score(y_test, y_pred, zero_division=0)),
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
            with path.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

        return metrics

# ---------------------------------------------------------------------------
# 2. HistGradientBoosting
# ---------------------------------------------------------------------------
 
class HistGBModel:
    """
    HGB a kézzel épített feature-táblán, ugyanazon Optuna-protokollal, mint az
    RF-nél. A kulcs paraméterek (max_iter, learning_rate, max_leaf_nodes,
    l2_regularization, min_samples_leaf) a korábban rögzített javaslat szerint.
    """
 
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model: HistGradientBoostingClassifier | None = None
        self.best_params: dict | None = None
 
    def tune(self, X_train, y_train, X_val, y_val, n_trials: int = 50) -> dict:
        import optuna
 
        def objective(trial: "optuna.Trial") -> float:
            params = {
                "max_iter": trial.suggest_int("max_iter", 100, 600),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 1.0, log=True),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100),
                "random_state": self.random_state,
            }
            clf = HistGradientBoostingClassifier(**params)
            clf.fit(X_train, y_train)
            val_proba = clf.predict_proba(X_val)[:, 1]
            return average_precision_score(y_val, val_proba)
 
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        self.best_params = study.best_params
        return self.best_params
 
    def fit(self, X_train, y_train, params: dict | None = None) -> "HistGBModel":
        params = params or self.best_params or {
            "max_iter": 300,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "random_state": self.random_state,
        }
        self.model = HistGradientBoostingClassifier(**params)
        self.model.fit(X_train, y_train)
        return self
 
    def predict_proba(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("A modellt előbb fit()-elni kell.")
        return self.model.predict_proba(X)[:, 1]

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
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                val_labels = val_graph["customer", "transacts", "merchant"].y.cpu().numpy()
            val_pr_auc = average_precision_score(val_labels, val_probs)
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