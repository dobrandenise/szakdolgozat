<h1 style="color:#B03060;">A 4 modell</h1>

## Tanítási folyamat

| Lépés | Mi történik? |
|--------|-------------|
| `tune()` | Az Optuna különböző hiperparaméter-kombinációkat próbál ki. |
| `objective()` | Egy adott paraméterkészlettel modellt tanít és kiértékel. |
| `fit()` | A legjobb paraméterekkel végleges modellt tanít. |
| `predict_proba()` | Fraud valószínűséget ad minden rekordra. |
| `average_precision_score()` | Az Optuna ezt maximalizálja (AUC-PR). |
| `study.best_params` | Az Optuna által talált legjobb hiperparaméterek. |

---

### Egyszerű folyamatábra

```text
Optuna
   │
   ▼
tune()
   │
   ▼
objective()
   │
   ├── Modell tanítása
   ├── predict_proba()
   └── average_precision_score()
   │
   ▼
study.best_params
   │
   ▼
fit()
   │
   ▼
Végleges modell
   │
   ▼
Fraud valószínűség előrejelzés
```
---

## RandomForest (RF) 🔵

### Hiperparaméterek

|   | Hiperparaméter | Mit szabályoz? | Kis érték hatása | Nagy érték hatása | A kódban |
|---|---|---|---|---|---|
| 1 | `n_estimators` | A döntési fák száma | Gyorsabb tanítás, nagyobb szórás | Stabilabb modell, lassabb tanítás | 200-800 |
| 2 | `max_depth` | Egy fa maximális mélysége | Egyszerűbb modell, underfitting veszély | Komplexebb modell, overfitting veszély | 4-24 |
| 3 | `max_features` | Egy splitnél vizsgált feature-ök aránya | Változatosabb fák, nagyobb randomitás | Hasonlóbb fák, kisebb randomitás | 0.2-1.0 |
| 4 | `min_samples_leaf` | Egy levélben minimálisan szükséges minták száma | Részletesebb szabályok, overfitting veszély | Simább modell, jobb generalizáció | 1-20 |
| 5 | `class_weight` | Az osztályok súlyozása tanításkor | Nincs súlyozás esetén a többségi osztály dominál | Ritka osztály (fraud) nagyobb figyelmet kap | `None`, `balanced`, `balanced_subsample` |
| 6 | `random_state` | A véletlenszerűség magja | Csak reprodukálhatóságot befolyásolja | Csak reprodukálhatóságot befolyásolja | 42 |
| 7 | `n_jobs` | Használt CPU magok száma | Lassabb futás | Gyorsabb futás | -1 |

---

### A három legfontosabb fraud detektálásnál

|   | Hiperparaméter | Miért fontos? |
|---|---|---|
| 1 | `class_weight` | Meghatározza, mennyire figyeljen a modell a ritka fraud tranzakciókra. |
| 2 | `min_samples_leaf` | Megakadályozhatja, hogy a modell néhány egyedi fraud esetet bemagoljon. |
| 3 | `max_depth` | Közvetlenül szabályozza a modell komplexitását és az overfitting mértékét. |

---

### `class_weight` opciók

|   | Érték | Jelentés |
|---|---|---|
| 1 | `None` | Minden tranzakció azonos súlyú. |
| 2 | `balanced` | A ritkább osztály (fraud) automatikusan nagyobb súlyt kap. |
| 3 | `balanced_subsample` | Ugyanaz, mint a `balanced`, de minden fa bootstrap mintáján külön számolja a súlyokat. |

---

### Gyors összefoglaló

- `n_estimators` → Hány fa legyen?
- `max_depth` → Milyen mélyek legyenek a fák?
- `max_features` → Hány feature-t lásson egy splitnél?
- `min_samples_leaf` → Milyen kicsi levelek lehetnek?
- `class_weight` → Mennyire figyeljen a fraudokra?
- `random_state` → Reprodukálhatóság.
- `n_jobs` → Hány CPU magot használjon?

---
# Random Forest architektúra

```text
                    ┌─────────────────┐
                    │   X_train       │
                    │   y_train       │
                    └────────┬────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  Optuna tuning      │
                  │  (50 trial)         │
                  └────────┬────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │ Hiperparaméter keresés              │
         │                                     │
         │ n_estimators                        │
         │ max_depth                           │
         │ max_features                        │
         │ min_samples_leaf                    │
         │ class_weight                        │
         └─────────────────┬───────────────────┘
                           │
                           ▼
                ┌──────────────────┐
                │ Legjobb param.   │
                │ best_params      │
                └────────┬─────────┘
                         │
                         ▼
            ┌──────────────────────────┐
            │ RandomForest.fit()       │
            │ teljes train adaton      │
            └──────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ Validation predikciók       │
         │ predict_proba(X_val)        │
         └───────────┬─────────────────┘
                     │
                     ▼
         ┌─────────────────────────────┐
         │ PR-AUC számítás             │
         │ Average Precision Score     │
         └───────────┬─────────────────┘
                     │
                     ▼
         ┌─────────────────────────────┐
         │ Threshold optimalizáció     │
         │                             │
         │ Cost =                      │
         │ C_FP × FP + C_FN × FN       │
         └───────────┬─────────────────┘
                     │
                     ▼
          ┌──────────────────────────┐
          │ Optimális threshold      │
          │ pl. 0.73                 │
          └──────────┬───────────────┘
                     │
                     ▼
           ┌─────────────────────────┐
           │ X_test                  │
           │ predict_proba()         │
           └─────────┬───────────────┘
                     │
                     ▼
      ┌──────────────────────────────────┐
      │ y_pred = (proba >= threshold)    │
      └────────────────┬─────────────────┘
                       │
                       ▼
           ┌────────────────────────┐
           │ Végső teszt metrikák   │
           │ PR-AUC                 │
           │ ROC-AUC                │
           │ Precision              │
           │ Recall                 │
           │ F1                     │
           │ Cost                   │
           └────────────────────────┘
```
---

## HistGradientBoosting (HGB) 🟢

### Hiperparaméterek

|   | Hiperparaméter | Mit szabályoz? | Kis érték hatása | Nagy érték hatása | Fraud detektálásnál |
|---|---|---|---|---|---|
| 1 | `learning_rate` | Mekkora lépésekben javítja a modell a hibáit az egyes boosting iterációk során | Lassabb tanulás, stabilabb modell, jobb generalizáció | Gyorsabb tanulás, de nagyobb overfit kockázat | Általában 0.03-0.1 közötti értékek szoktak jól működni |
| 2 | `max_iter` | Hány fát építsen egymás után a boosting folyamat során | Gyorsabb tanítás, kisebb modell, de alultanulhat | Pontosabb lehet, de lassabb és overfitelhet | Erősen összefügg a `learning_rate` paraméterrel |
| 3 | `max_leaf_nodes` | Egy fa maximális levélszáma. A fa komplexitását határozza meg | Egyszerűbb döntési szabályok | Összetettebb mintázatok felismerése | Csalási mintázatok gyakran komplexek, ezért általában hasznos lehet a magasabb érték |
| 4 | `min_samples_leaf` | Egy levélben minimálisan hány rekordnak kell lennie | Nagyon specifikus szabályok tanulása | Általánosabb szabályok tanulása | Az egyik legfontosabb overfit elleni paraméter imbalanced adatoknál |
| 5 | `l2_regularization` | Bünteti a túl komplex modelleket | Kevesebb regularizáció, nagyobb overfit veszély | Erősebb regularizáció, stabilabb modell | Segíthet csökkenteni a zaj megtanulását |
| 6 | `random_state` | Reprodukálhatóságot biztosít | - | - | Ugyanazzal a seed-del ugyanazokat az eredményeket kapod |

---
## HistGradientBoosting architektúra

```text
                    ┌─────────────────┐
                    │   Raw Dataset   │
                    └────────┬────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ Feature Engineering     │
                │ (kézzel készített feat.)│
                └────────┬────────────────┘
                         │
                         ▼
                ┌─────────────────────────┐
                │ Train / Val Split       │
                └───────┬─────────┬───────┘
                        │         │
                        │         ▼
                        │   Validation Set
                        │
                        ▼
                 Training Set
                        │
                        ▼
             ┌──────────────────────┐
             │      Optuna          │
             └──────────┬───────────┘
                        │
       ┌────────────────┼────────────────┐
       │                │                │
       ▼                ▼                ▼

 max_iter      learning_rate    max_leaf_nodes
       │                │                │
       └────────────────┼────────────────┘
                        │
                        ▼

           l2_regularization
                        │
                        ▼

           min_samples_leaf
                        │
                        ▼

 ┌────────────────────────────────────────┐
 │ HistGradientBoostingClassifier         │
 └────────────────┬───────────────────────┘
                  │
                  ▼
          fit(X_train, y_train)
                  │
                  ▼
      predict_proba(X_val)[:,1]
                  │
                  ▼
      average_precision_score
                  │
                  ▼
      Best Hyperparameters
                  │
                  ▼
     Final HistGradientBoosting
                  │
                  ▼
         predict_proba(X_test)
                  │
                  ▼
         Fraud Probability
```
---

## TransformerModel hiperparaméterek 🟡

| Hiperparaméter | Jelentés |
|---------------|----------|
| `max_len` | Egy customerből legfeljebb ennyi utolsó tranzakció kerül a szekvenciába. |
| `d_model` | A Transformer belső reprezentációjának dimenziója. |
| `n_heads` | Az attention fejek száma a Multi-Head Attention rétegben. |
| `n_layers` | A Transformer Encoder blokkok száma. |
| `dropout` | Regularizáció overfitting csökkentésére. |
| `lr` | AdamW optimizer learning rate-je. |
| `focal_gamma` | A Focal Loss fókuszáló paramétere. Nagyobb értéknél jobban a nehéz esetekre koncentrál. |
| `focal_alpha` | A pozitív (fraud) osztály súlya a Focal Loss-ban. |
| `batch_size` | Egy tanítási lépésben feldolgozott customer-szekvenciák száma. |
| `epochs` | Maximális tanítási epochok száma. |
| `patience` | Hány javulás nélküli epoch után álljon le az Early Stopping. |

---
### Transformer architektúra

1. **Numerikus feature-ök**
   - Standardizált numerikus változók (pl. amount, age, distance).

2. **Category Embedding**
   - A tranzakció kategóriájából 8 dimenziós embedding készül.

3. **Merchant Embedding**
   - A merchant azonosítóból 16 dimenziós embedding készül.

4. **Feature összefűzés**
   - Numerikus feature-ök + category embedding + merchant embedding.

5. **Input Projection**
   - A kombinált feature-vektor `d_model` dimenzióra vetül.

6. **Transformer Encoder**
   - `n_layers` darab Encoder blokk.
   - `n_heads` fejű Multi-Head Self-Attention.
   - Causal mask: csak a múltbeli tranzakciókra figyelhet.

7. **MLP Head**
   - `Linear(d_model → d_model/2)`
   - `ReLU`
   - `Dropout`
   - `Linear(d_model/2 → 1)`

8. **Kimenet**
   - Tranzakciónként egy logit.
   - Sigmoid után fraud valószínűség.

---
### Tanítási stratégia

| Elem | Szerepe |
|--------|----------|
| Focal Loss | Az extrém imbalance kezelése. |
| AdamW | Paraméteroptimalizálás. |
| Weight Decay | L2 regularizáció. |
| Gradient Clipping | Gradient explosion megelőzése. |
| ReduceLROnPlateau | Stagnálás esetén csökkenti a learning rate-et. |
| Early Stopping | Megakadályozza a túl sokáig tartó tanítást. |
| PR-AUC | A validációs modellkiválasztás metrikája. |

---
### Transformer architektúra

```text
Customer tranzakciós szekvencia
(max_len = 64)
        │
        ▼
┌────────────────────┐
│ Numerikus feature  │
└────────────────────┘
        │
        ├───────────────┐
        │               │
        ▼               ▼
┌─────────────┐  ┌─────────────┐
│ Category ID │  │ Merchant ID │
└─────────────┘  └─────────────┘
        │               │
        ▼               ▼
┌─────────────┐  ┌─────────────┐
│ Embedding 8 │  │Embedding 16 │
└─────────────┘  └─────────────┘
        │               │
        └───────┬───────┘
                │
                ▼
      Feature Concatenation
     (Numeric + Cat + Merch)
                │
                ▼
┌──────────────────────────┐
│      Input Projection    │
│     (input → d_model)    │
└──────────────────────────┘
                │
                ▼
┌──────────────────────────┐
│   Transformer Encoder    │
│      n_layers = 2        │
│      n_heads = 4         │
│      Causal Mask         │
└──────────────────────────┘
                │
                ▼
┌──────────────────────────┐
│        MLP Head          │
│ Linear(d_model,d_model/2)│
│ ReLU                     │
│ Dropout                  │
│ Linear(d_model/2,1)      │
└──────────────────────────┘
                │
                ▼
            Logit
                │
                ▼
           Sigmoid
                │
                ▼
      Fraud valószínűség
```
---

## GraphSAGE hiperparaméterek 🔴

| Hiperparaméter | Mit szabályoz? | Tipikus érték | Fraud projektnél |
|---------------|----------------|---------------|------------------|
| `hidden_dim` | Embedding mérete (node-reprezentáció dimenziója) | 32-256 | 64 vagy 128 jó kiindulás |
| `num_layers` | Hány GraphSAGE réteg fut egymás után | 2-4 | 2-3 ajánlott |
| `dropout` | Kikapcsolt neuronok aránya tanításkor | 0.1-0.5 | 0.2-0.4 |
| `lr` | Learning rate | 1e-4 - 1e-2 | 1e-3 |
| `epochs` | Maximális epoch szám | 50-500 | 100 |
| `patience` | Early stopping türelmi küszöb | 5-20 | 8 |
| `weight_decay` | L2 regularizáció | 0 - 1e-3 | 1e-4 |
| `scheduler_patience` | LR csökkentés javulás hiányában | 2-10 | 4 |
| `scheduler_factor` | LR szorzó csökkentéskor | 0.1-0.8 | 0.5 |

---
### GraphSAGE architektúra

| Komponens | Szerepe |
|-----------|----------|
| Customer node | Ügyfél reprezentáció |
| Merchant node | Kereskedő reprezentáció |
| Edge (tranzakció) | Customer-Merchant kapcsolat |
| Customer node feature-k | Ügyfélszintű aggregált jellemzők |
| Merchant node feature-k | Merchant szintű aggregált jellemzők |
| Edge feature-k | Tranzakciós feature-k (összeg, időpont stb.) |
| GraphSAGE rétegek | Szomszédok információinak aggregálása |
| Node embedding | Tanult rejtett reprezentáció |
| MLP classifier | Fraud valószínűség számítása |
| Sigmoid | 0-1 közötti fraud valószínűség |

---
### GraphSAGE rétegek

| Réteg | Input | Output |
|---------|---------|---------|
| Customer feature-k | Customer aggregált feature-k | Customer node feature |
| Merchant feature-k | Merchant aggregált feature-k | Merchant node feature |
| GraphSAGE Layer 1 | Node feature-k + szomszédok | Embedding |
| ReLU | Első GraphSAGE kimenete | Nemlinearitás |
| Dropout | ReLU kimenet | Regularizáció |
| GraphSAGE Layer 2 | Frissített embeddingek | Végső embedding |
| Concat | Customer emb + Merchant emb + Edge feature | Kombinált feature vektor |
| MLP | Kombinált feature vektor | Logit |
| Sigmoid | Logit | Fraud valószínűség |

---
### Message Passing

| Lépés | Mi történik? |
|---------|-------------|
| 1 | Customer node összegyűjti a szomszédos merchantok információit |
| 2 | Merchant node összegyűjti a szomszédos customerek információit |
| 3 | Saját feature + szomszéd feature összevonása |
| 4 | Lineáris transzformáció |
| 5 | ReLU aktiváció |
| 6 | Új embedding létrehozása |
| 7 | Következő réteg ugyanezt ismétli |

---

### Fraud predikció

| Lépés | Mi történik? |
|---------|-------------|
| 1 | Customer node embedding számítás |
| 2 | Merchant node embedding számítás |
| 3 | Tranzakció edge feature-k betöltése |
| 4 | Customer embedding + Merchant embedding + Edge feature összefűzése |
| 5 | MLP osztályozó futtatása |
| 6 | Logit generálása |
| 7 | Sigmoid alkalmazása |
| 8 | Fraud valószínűség előállítása |

---

### GraphSAGE architektúra

```text
Customer Feature-k
        │
        ▼
 ┌─────────────┐
 │ Customer    │
 │    Node     │
 └─────────────┘
        │
        │
        ▼
=========================
 Customer ↔ Merchant
   Tranzakciós élek
=========================
        ▲
        │
        │
 ┌─────────────┐
 │ Merchant    │
 │    Node     │
 └─────────────┘
        ▲
        │
Merchant Feature-k


        │
        ▼

┌──────────────────────┐
│ GraphSAGE Layer 1    │
└──────────────────────┘
          │
          ▼
       ReLU
          │
          ▼
      Dropout
          │
          ▼
┌──────────────────────┐
│ GraphSAGE Layer 2    │
└──────────────────────┘
          │
          ▼

Customer Embedding (64)

Merchant Embedding (64)

          │
          │
          └──────────┐
                     │
                     ▼

       Edge Feature-k
(amount, category, history...)

                     │
                     ▼

      Concatenation
      [Cust | Merch | Edge]

                     │
                     ▼

         MLP Classifier

                     │
                     ▼

            Logit

                     │
                     ▼

           Sigmoid

                     │
                     ▼

      Fraud Probability
```
