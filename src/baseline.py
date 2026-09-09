import joblib
import pandas as pd
import time
import json
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix, precision_recall_fscore_support


# 1. Adatbetöltés
df = pd.read_csv("../data/raw/BankSim.csv")

# 2. Alapinfó: oszlopok, dtype-ok, alak
print("=== Alak ===")
print(df.shape)

print("\n=== Oszlopok és dtype-ok ===")
print(df.dtypes)

print("\n=== Első néhány sor ===")
print(df.head())

# 3. Hiányzó értékek
print("\n=== Hiányzó értékek oszloponként ===")
print(df.isnull().sum())

# 4. Fraud osztályeloszlás
print("\n=== Fraud osztályeloszlás (darabszám) ===")
print(df["fraud"].value_counts())

print("\n=== Fraud osztályeloszlás (arány, %) ===")
print(df["fraud"].value_counts(normalize=True) * 100)

# 1. Releváns oszlopok kiválasztása
feature_cols = ["step", "age", "gender", "merchant", "category", "amount"] #a customer-t kivettem belőle, mert a gépem nem bírta el a OneHot-ingot a customer-szinten
df_features = df[feature_cols].copy()

# 2. Célváltozó leválasztása
y = df["fraud"].copy()

# 3. Csoportkulcs a customer-szintű splithez (3. lépésben GroupShuffleSplit-hez kell)
groups = df["customer"].copy()

# 4. Kategorikus / numerikus oszlopok szétválasztása
categorical_cols = ["age", "gender", "merchant", "category"]
numeric_cols = ["step", "amount"]

# 5. OneHotEncoder a kategorikus oszlopokra
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_cats = encoder.fit_transform(df_features[categorical_cols])
encoded_cat_cols = encoder.get_feature_names_out(categorical_cols)

encoded_df = pd.DataFrame(encoded_cats, columns=encoded_cat_cols, index=df_features.index)

# 6. Numerikus oszlopok hozzáfűzése
X = pd.concat([encoded_df, df_features[numeric_cols]], axis=1)

# 7. Ellenőrzés
print("X alakja:", X.shape)
print("X oszlopai:\n", list(X.columns))
#print("Egyedi customer-ek száma:", df["customer"].nunique())

# 7. Train/test split - 75/25, stratifikált, rögzített random_state
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

# Ellenőrzés: méretek és fraud-arány train/test között
print("X_train alakja:", X_train.shape)
print("X_test alakja:", X_test.shape)

print("\nFraud arány train-ben:")
print(y_train.value_counts(normalize=True) * 100)

print("\nFraud arány test-ben:")
print(y_test.value_counts(normalize=True) * 100)

# Modellek definiálása

rf_model = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

hgb_model = HistGradientBoostingClassifier(
    random_state=42,
    class_weight="balanced"  # sklearn >= 1.2 szükséges
)

models = {
    "RandomForest": rf_model,
    "HistGradientBoosting": hgb_model
}

results = {}

for name, model in models.items():
    print(f"\n=== {name} ===")

    # Illesztés
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Betanítási idő: {elapsed:.1f} mp")

    # Predikció (valószínűség a pozitív/fraud osztályra)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Metrikák
    auc_roc = roc_auc_score(y_test, y_proba)
    auc_pr = average_precision_score(y_test, y_proba)

    results[name] = {"AUC-ROC": auc_roc, "AUC-PR": auc_pr, "fit_time_sec": elapsed}

    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR:  {auc_pr:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["legit", "fraud"]))

print("\n=== Összefoglaló ===")
for name, metrics in results.items():
    print(f"{name}: AUC-ROC={metrics['AUC-ROC']:.4f}, AUC-PR={metrics['AUC-PR']:.4f}")

for name in models.keys():
    print(f"\n=== {name} — Confusion Matrix ===")
    cm = confusion_matrix(y_test, models[name].predict(X_test))
    print(cm)
    print(f"(sorok: valós [legit, fraud], oszlopok: predikált [legit, fraud])")

    tn, fp, fn, tp = cm.ravel()
    print(f"True Negative:  {tn}")
    print(f"False Positive: {fp}")
    print(f"False Negative: {fn}")
    print(f"True Positive:  {tp}")

# --- Modellek összevetése + részletes metrikák összegyűjtése ---

comparison_rows = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1
    )
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    comparison_rows.append({
        "dataset": "BankSim",
        "model": name,
        "auc_roc": results[name]["AUC-ROC"],
        "auc_pr": results[name]["AUC-PR"],
        "precision_fraud": precision,
        "recall_fraud": recall,
        "f1_fraud": f1,
        "fit_time_sec": results[name]["fit_time_sec"],
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "n_test": len(y_test)
    })

comparison_df = pd.DataFrame(comparison_rows)

print("=== Modellek összevetése (BankSim baseline) ===")
print(comparison_df.to_string(index=False))

# Melyik modell jobb melyik metrikában?
print("\n=== Metrikánkénti győztes ===")
for metric in ["auc_roc", "auc_pr", "precision_fraud", "recall_fraud", "f1_fraud"]:
    best_row = comparison_df.loc[comparison_df[metric].idxmax()]
    print(f"{metric}: {best_row['model']} ({best_row[metric]:.4f})")

# --- 8. Eredmények mentése ---
csv_path = "../metrics/baseline_met_results.csv"
if os.path.exists(csv_path):
    comparison_df.to_csv(csv_path, mode="a", header=False, index=False)
else:
    comparison_df.to_csv(csv_path, mode="w", header=True, index=False)

# JSON mentés (readable, egy külön fájl per futtatás)
json_path = "../metrics/baseline_metrics_results.json"
comparison_df.to_json(json_path, orient="records", indent=2, force_ascii=False)

print(f"\nEredmények elmentve ide: {csv_path} (append módban) és {json_path}")

for name, model in models.items():
    model_path = f"../models/banksim_baseline_{name}.joblib"
    joblib.dump(model, model_path)
    print(f"{name} elmentve: {model_path}")