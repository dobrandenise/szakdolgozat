import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


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

# --- Végső k explicit megadása ---
final_k = 9  # <-- ide írd be a döntésed (silhouette + elbow együttes mérlegelése alapján)

# --- Végleges K-Means illesztése ---
final_kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
cluster_labels = final_kmeans.fit_predict(X)

# --- Cluster-címke hozzáadása az eredeti (tisztított, de nem transzformált) datasethez ---
df_features["cluster"] = cluster_labels

# --- Fraud oszlop visszacsatolása csak ellenőrzés/profilozás céljából ---
df_features["fraud"] = y_fraud

print(f"Végleges klaszterszám: {final_k}")
print(df_features["cluster"].value_counts().sort_index())

# --- Gyors ellenőrzés: fraud arány klaszterenként ---
fraud_rate_by_cluster = df_features.groupby("cluster")["fraud"].mean().sort_values(ascending=False)
print("\nFraud arány klaszterenként:")
print(fraud_rate_by_cluster)

# --- Mentés ---
output_path = "../data/processed/dataset_with_clusters.csv"
df_features.to_csv(output_path, index=False)
print(f"\nMentve: {output_path}")

# --- Klaszterenkénti elemszám és fraud arány ---
cluster_summary = df_features.groupby("cluster").agg(
    count=("fraud", "size"),
    fraud_count=("fraud", "sum"),
    fraud_rate=("fraud", "mean")
).sort_values("fraud_rate", ascending=False)

# --- Globális fraud arány (viszonyítási alap) ---
global_fraud_rate = df_features["fraud"].mean()
cluster_summary["lift"] = cluster_summary["fraud_rate"] / global_fraud_rate

print(f"Globális fraud arány: {global_fraud_rate:.4f}\n")
print(cluster_summary)

# --- Mentés ---
cluster_summary.to_csv("../data/processed/cluster_fraud_summary.csv")

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_ylim(0, 0.21)
ax.set_yticks(np.arange(0, 0.201, 0.02))

# Klaszterek sorrendje fraud_rate szerint csökkenő (már így van a cluster_summary-ban)
clusters = cluster_summary.index.astype(str)
fraud_rates = cluster_summary["fraud_rate"]
counts = cluster_summary["count"]

bars = ax.bar(clusters, fraud_rates, color="firebrick", alpha=0.8)

# Globális átlag vonal
ax.axhline(global_fraud_rate, color="gray", linestyle="--", linewidth=1.5,
           label=f"Globális átlag ({global_fraud_rate:.3f})")

# Elemszám feltüntetése minden oszlop tetején
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height,
             f"n={count:,}", ha="center", va="bottom", fontsize=8, rotation=0)

ax.set_xlabel("Klaszter")
ax.set_ylabel("Fraud arány")
ax.set_title(f"Csalási arány klaszterenként (k={final_k})")
ax.legend()
plt.tight_layout()
plt.savefig("../plots/cluster_fraud_rate_bar.png", dpi=150)
plt.show()

# --- Végleges, tiszta kimeneti oszlopok kiválasztása ---
df["cluster"] = cluster_labels

# --- Mentés ---
final_output_path = "../data/processed/dataset_with_clusters_final.csv"
df.to_csv(final_output_path, index=False)

print(f"Végleges adathalmaz mentve: {final_output_path}")
print(f"Alak: {df.shape}")
print(df.head())