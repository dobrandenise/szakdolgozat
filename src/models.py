"""
Négy modellcsalád implementációja a fraud detection szakdolgozathoz:
 
  1. RandomForestModel  - tabuláris, kézzel épített feature-ökön
  2. HistGBModel         - tabuláris, kézzel épített feature-ökön
  3. TransformerModel    - per-customer tranzakció-szekvencia, causal attention
  4. GraphSAGEModel      - customer-merchant bipartit gráf, induktív node-kezeléssel
 
Feltételezett bemenet (illeszkedik a meglévő BaselineFraudExperiment / DataSplit
kimenetéhez):
 
  - RandomForestModel / HistGBModel:
        már kódolt (OrdinalEncoder-rel fit-elt) X_train / X_val / X_test
        DataFrame-ek és y_train / y_val / y_test Series-ek, ugyanúgy, ahogy az
        encode_features(train_features, test_features) előállítja őket.
 
  - TransformerModel / GraphSAGEModel:
        a nyers (de tisztított) tranzakciós train/val/test DataFrame-ek, az
        alábbi oszlopokkal: customer, merchant, category, amount, step, fraud
        (+ opcionálisan a 4. fejezetben épített history/ratio/prior/cluster
        feature-ök numerikus oszlopokként).
 
Mind a négy osztály egységes felületet ad:
 
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
 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
 
 
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
 
    def tune(self, X_train, y_train, X_val, y_val, n_trials: int = 50) -> dict:
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
# 3. Transformer (per-customer szekvencia, causal attention)
# ---------------------------------------------------------------------------
 
class _TransactionSequenceDataset:
    """
    Customerenként, step szerint rendezett tranzakció-szekvenciákat épít.
    Numerikus feature-ök összefűzve, a category/merchant embeddingként kerül
    be a modellbe. Fix max_len-re paddol/vág (padding maszkkal).
    """
 
    def __init__(self, df: pd.DataFrame, numeric_cols, category_col, merchant_col,
                 target_col, max_len: int, category_vocab=None, merchant_vocab=None):
        import torch
 
        self.max_len = max_len
        self.numeric_cols = list(numeric_cols)
 
        self.category_vocab = category_vocab or {
            v: i + 1 for i, v in enumerate(sorted(df[category_col].unique()))
        }
        self.merchant_vocab = merchant_vocab or {
            v: i + 1 for i, v in enumerate(sorted(df[merchant_col].unique()))
        }
 
        sequences = []
        for _, group in df.sort_values("step").groupby(df["customer"] if "customer" in df else df.index):
            group = group.sort_values("step")
            numeric = group[self.numeric_cols].to_numpy(dtype=np.float32)
            cats = group[category_col].map(self.category_vocab).fillna(0).to_numpy(dtype=np.int64)
            merch = group[merchant_col].map(self.merchant_vocab).fillna(0).to_numpy(dtype=np.int64)
            labels = group[target_col].to_numpy(dtype=np.float32)
            sequences.append((numeric, cats, merch, labels))
 
        self.sequences = sequences
 
    def __len__(self):
        return len(self.sequences)
 
    def __getitem__(self, idx):
        import torch
 
        numeric, cats, merch, labels = self.sequences[idx]
        seq_len = min(len(labels), self.max_len)
 
        num_pad = np.zeros((self.max_len, len(self.numeric_cols)), dtype=np.float32)
        cat_pad = np.zeros(self.max_len, dtype=np.int64)
        merch_pad = np.zeros(self.max_len, dtype=np.int64)
        label_pad = np.zeros(self.max_len, dtype=np.float32)
        attn_mask = np.zeros(self.max_len, dtype=np.bool_)
 
        # csak az utolsó max_len tranzakciót tartjuk meg (legfrissebb kontextus)
        num_pad[:seq_len] = numeric[-seq_len:]
        cat_pad[:seq_len] = cats[-seq_len:]
        merch_pad[:seq_len] = merch[-seq_len:]
        label_pad[:seq_len] = labels[-seq_len:]
        attn_mask[:seq_len] = True
 
        return (
            torch.from_numpy(num_pad),
            torch.from_numpy(cat_pad),
            torch.from_numpy(merch_pad),
            torch.from_numpy(label_pad),
            torch.from_numpy(attn_mask),
        )
 
 
class _TransformerNet:
    """nn.Module wrapper, csak import-on belül definiálva, hogy torch nélkül is betölthető legyen a fájl."""
 
    @staticmethod
    def build(num_numeric: int, num_categories: int, num_merchants: int,
              d_model: int = 64, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.2):
        import torch
        import torch.nn as nn
 
        class TransformerNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.category_emb = nn.Embedding(num_categories + 1, 8, padding_idx=0)
                self.merchant_emb = nn.Embedding(num_merchants + 1, 16, padding_idx=0)
                self.input_proj = nn.Linear(num_numeric + 8 + 16, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                    dropout=dropout, batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                self.head = nn.Sequential(
                    nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(d_model // 2, 1),
                )
 
            def forward(self, numeric, cats, merch, attn_mask):
                cat_e = self.category_emb(cats)
                merch_e = self.merchant_emb(merch)
                x = torch.cat([numeric, cat_e, merch_e], dim=-1)
                x = self.input_proj(x)
 
                seq_len = x.size(1)
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
                )
                key_padding_mask = ~attn_mask
 
                out = self.encoder(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
                logits = self.head(out).squeeze(-1)
                return logits
 
        return TransformerNet()
 
 
class TransformerModel:
    """
    Per-customer tranzakció-szekvencia Transformer, causal maszkkal (csak a
    korábbi tranzakciókra figyelhet minden lépés — nincs jövőbeli leakage).
    Focal loss-t használ az extrém imbalance miatt (γ=2.0, α≈0.90-0.95,
    ahogy a 6. fejezetben rögzítettük).
    """
 
    def __init__(self, numeric_cols, category_col="category", merchant_col="merchant",
                 target_col="fraud", max_len: int = 64, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.2,
                 lr: float = 1e-3, focal_gamma: float = 2.0, focal_alpha: float = 0.9,
                 device: str | None = None):
        self.numeric_cols = numeric_cols
        self.category_col = category_col
        self.merchant_col = merchant_col
        self.target_col = target_col
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
 
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        self.category_vocab = None
        self.merchant_vocab = None
 
    @staticmethod
    def _focal_loss(logits, targets, mask, gamma, alpha):
        import torch
        import torch.nn.functional as F
 
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, alpha, 1 - alpha)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = alpha_t * (1 - pt) ** gamma * bce
        loss = loss * mask
        return loss.sum() / mask.sum().clamp(min=1)
 
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
            epochs: int = 100, batch_size: int = 64, patience: int = 8) -> "TransformerModel":
        import torch
        from torch.utils.data import DataLoader
 
        train_ds = _TransactionSequenceDataset(
            train_df, self.numeric_cols, self.category_col, self.merchant_col,
            self.target_col, self.max_len,
        )
        self.category_vocab = train_ds.category_vocab
        self.merchant_vocab = train_ds.merchant_vocab
 
        val_ds = _TransactionSequenceDataset(
            val_df, self.numeric_cols, self.category_col, self.merchant_col,
            self.target_col, self.max_len,
            category_vocab=self.category_vocab, merchant_vocab=self.merchant_vocab,
        )
 
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
 
        self.net = _TransformerNet.build(
            num_numeric=len(self.numeric_cols),
            num_categories=len(self.category_vocab),
            num_merchants=len(self.merchant_vocab),
            d_model=self.d_model, n_heads=self.n_heads,
            n_layers=self.n_layers, dropout=self.dropout,
        ).to(self.device)
 
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
 
        best_val_pr_auc = -1.0
        best_state = None
        epochs_no_improve = 0
 
        for epoch in range(epochs):
            self.net.train()
            for numeric, cats, merch, labels, mask in train_loader:
                numeric, cats, merch = numeric.to(self.device), cats.to(self.device), merch.to(self.device)
                labels, mask = labels.to(self.device), mask.to(self.device).float()
 
                optimizer.zero_grad()
                logits = self.net(numeric, cats, merch, mask.bool())
                loss = self._focal_loss(logits, labels, mask, self.focal_gamma, self.focal_alpha)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
 
            val_probs, val_labels = self._predict_masked(val_loader)
            val_pr_auc = average_precision_score(val_labels, val_probs)
            scheduler.step(val_pr_auc)
 
            if val_pr_auc > best_val_pr_auc:
                best_val_pr_auc = val_pr_auc
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break
 
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self
 
    def _predict_masked(self, loader):
        import torch
 
        self.net.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for numeric, cats, merch, labels, mask in loader:
                numeric, cats, merch = numeric.to(self.device), cats.to(self.device), merch.to(self.device)
                mask_bool = mask.to(self.device).bool()
                logits = self.net(numeric, cats, merch, mask_bool)
                probs = torch.sigmoid(logits)
                all_probs.append(probs[mask_bool].cpu().numpy())
                all_labels.append(labels[mask].numpy())
        return np.concatenate(all_probs), np.concatenate(all_labels)
 
    def predict_proba(self, df: pd.DataFrame, batch_size: int = 64) -> np.ndarray:
        from torch.utils.data import DataLoader
 
        if self.net is None:
            raise RuntimeError("A modellt előbb fit()-elni kell.")
 
        ds = _TransactionSequenceDataset(
            df, self.numeric_cols, self.category_col, self.merchant_col,
            self.target_col, self.max_len,
            category_vocab=self.category_vocab, merchant_vocab=self.merchant_vocab,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        probs, _ = self._predict_masked(loader)
        return probs

# ---------------------------------------------------------------------------
# 4. GraphSAGE (customer-merchant bipartit gráf)
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