import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time
import matplotlib.pyplot as plt

# --- 1. Adat betöltése ---
input_path = "../data/raw/BankSim.csv"
df = pd.read_csv(input_path)

# --- 2. String oszlopok tisztítása (extra idézőjelek eltávolítása) ---
# Tipikus jelenség, amikor az eredeti CSV-ben az értékek "'19'" formában szerepelnek
string_cols = ["age", "gender", "category"]

for col in string_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.strip("'")   # egyszeres idézőjel levágása
            .str.strip('"')   # dupla idézőjel levágása, ha lenne
        )

# --- 3. Fraud oszlop leválasztása ---
# Csak kiértékeléshez kell, a klaszterezés bemenetébe NEM megy bele
if "fraud" not in df.columns:
    raise ValueError("Nem található 'fraud' oszlop a datasetben.")

y_fraud = df["fraud"].copy()
df_features = df.drop(columns=["step", "customer", "zipcodeOri", "merchant", "zipMerchant", "fraud"])

print(f"Betöltött sorok száma: {len(df_features)}")
print(f"Fraud arány: {y_fraud.mean():.4f}")
print(df_features.dtypes)

if "amount" not in df_features.columns:
    raise ValueError("Nem található 'amount' oszlop a datasetben.")

df_features["amount_log"] = np.log1p(df_features["amount"])

numeric_cols = ["amount_log"]
categorical_cols = ["category", "age", "gender"]

missing = [c for c in categorical_cols if c not in df_features.columns]
if missing:
    raise ValueError(f"Hiányzó kategorikus oszlop(ok): {missing}")

# --- 3. ColumnTransformer: StandardScaler + OneHotEncoder ---
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_cols),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ]
)

X = preprocessor.fit_transform(df_features[numeric_cols + categorical_cols])

cat_feature_names = preprocessor.named_transformers_["categorical"].get_feature_names_out(categorical_cols)
all_feature_names = numeric_cols + list(cat_feature_names)

X_df = pd.DataFrame(X, columns=all_feature_names, index=df_features.index)

print(f"Feature-mátrix alakja: {X_df.shape}")
print(f"Feature-ek: {list(X_df.columns)}")

k_range = range(5, 11)  # 5,6,7,8,9,10
results = []

for k in k_range:
    start = time.time()

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    inertia = kmeans.inertia_
    sil_score = silhouette_score(X, labels)

    elapsed = time.time() - start

    results.append({
        "k": k,
        "inertia": inertia,
        "silhouette": sil_score,
        "time_sec": elapsed
    })

    print(f"k={k}: inertia={inertia:.2f}, silhouette={sil_score:.4f}, idő={elapsed:.1f}s")

results_df = pd.DataFrame(results)
print(results_df)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Elbow-görbe (inertia) ---
axes[0].plot(results_df["k"], results_df["inertia"], marker="o")
axes[0].set_xlabel("k (klaszterszám)")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow-módszer")
axes[0].set_xticks(results_df["k"])
axes[0].grid(alpha=0.3)

# --- Silhouette score ---
axes[1].plot(results_df["k"], results_df["silhouette"], marker="o", color="darkorange")
axes[1].set_xlabel("k (klaszterszám)")
axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Silhouette score k függvényében")
axes[1].set_xticks(results_df["k"])
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("../plots/kmeans_k_selection.png", dpi=150)
plt.show()

# --- Legjobb k silhouette alapján ---
best_k_silhouette = results_df.loc[results_df["silhouette"].idxmax(), "k"]
print(f"Silhouette alapján legjobb k: {best_k_silhouette}")

# --- Inertia relatív csökkenése k-ról k+1-re (elbow-töréspont numerikus jelzése) ---
results_df["inertia_drop_pct"] = results_df["inertia"].pct_change() * -100
print(results_df[["k", "inertia", "inertia_drop_pct", "silhouette"]])
