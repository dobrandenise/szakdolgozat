import sys
from pathlib import Path

import cupy as cp
import numpy as np
import optuna
import pandas as pd
from cuml.metrics.cluster import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from clustering import Clusterer
except ModuleNotFoundError:  # pragma: no cover
    from src.clustering import Clusterer

RAW_DATA_PATH = PROJECT_ROOT / "data" / "optimalization" / "time_train.csv"
RESULTS_PATH = PROJECT_ROOT / "metrics" / "optuna_cluster_search_results.csv"
KMEANS_ANALYSIS_PATH = PROJECT_ROOT / "metrics" / "kmeans_cluster_analysis.csv"
HDBSCAN_ANALYSIS_PATH = PROJECT_ROOT / "metrics" / "hdbscan_cluster_analysis.csv"
DEFAULT_N_TRIALS = 50
DEFAULT_SAMPLE_SIZE = 50000

RISK_FOCUSED_FEATURES = [
    "step",
    "amount_log",
    "amount_zscore_customer",
    "amount_ratio_customer",
    "amount_zscore_category",
    "category_is_high_risk",
    "customer_fraud_prior",
    "merchant_fraud_prior",
    "category_fraud_prior",
    "risk_score_composite",
    "customer_tx_count_hist",
    "merchant_tx_count_hist",
    "category_tx_count_hist",
    "tx_count_last_30d",
    "days_since_last_tx_customer",
    "cumulative_spend_30d",
]

HIGH_SIGNAL_NUMERICAL_FEATURES = [
    "step",
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
    "category_is_high_risk",
    "tx_count_last_7d",
    "tx_count_last_30d",
    "days_since_last_tx_customer",
    "cumulative_spend_30d",
    "is_amount_outlier",
]

BEHAVIORAL_ONLY_FEATURES = [
    "customer_tx_count_hist",
    "customer_days_since_first_tx",
    "customer_category_diversity_hist",
    "is_new_category_for_customer",
    "merchant_tx_count_hist",
    "category_tx_count_hist",
    "category_is_high_risk",
    "tx_count_last_7d",
    "tx_count_last_30d",
    "days_since_last_tx_customer",
    "cumulative_spend_30d",
    "amount_log",
    "amount_ratio_customer",
    "amount_zscore_customer",
]

FEATURE_SETS = {
    "risk_focused": RISK_FOCUSED_FEATURES,
    "high_signal_numerical": HIGH_SIGNAL_NUMERICAL_FEATURES,
    "behavioral_only": BEHAVIORAL_ONLY_FEATURES,
}


def load_feature_dataset(
    path: Path,
    feature_columns: list[str] | None = None,
    include_target: bool = False,
) -> pd.DataFrame:
    """Betölti a feature-only datasetet, és eltávolítja a targetet, ha az még benne van."""
    df = pd.read_csv(path)

    fraud = df["fraud"].copy() if include_target and "fraud" in df.columns else None
    if "fraud" in df.columns:
        df = df.drop(columns=["fraud"])

    selected_columns = feature_columns or Clusterer.FEATURE_COLUMNS
    missing = [col for col in selected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Hiányzó klaszterezési oszlop(ok): {missing}")

    feature_df = df.loc[:, selected_columns].copy()
    if fraud is not None:
        feature_df["fraud"] = fraud.to_numpy()
    return feature_df


def summarize_hdbscan_clusters(transformed_df: pd.DataFrame) -> pd.DataFrame:
    """Klaszterméretek és fraud-arányok összesítése HDBSCAN-címkék szerint."""
    if "hdbscan_cluster_id" not in transformed_df.columns:
        raise ValueError("A DataFrame nem tartalmaz hdbscan_cluster_id oszlopot.")
    if "fraud" not in transformed_df.columns:
        raise ValueError("A klaszterenkénti fraud-arányhoz szükséges a fraud oszlop.")

    summary = (
        transformed_df.groupby("hdbscan_cluster_id", dropna=False)
        .agg(
            cluster_size=("hdbscan_cluster_id", "size"),
            fraud_count=("fraud", "sum"),
            fraud_rate=("fraud", "mean"),
        )
        .reset_index()
        .sort_values("hdbscan_cluster_id")
    )
    summary["is_noise"] = summary["hdbscan_cluster_id"].eq(-1)
    return summary


def summarize_kmeans_clusters(transformed_df: pd.DataFrame) -> pd.DataFrame:
    """KMeans-klaszterek méretének, fraud-arányának és távolságának összesítése."""
    required_columns = {"cluster_id", "dist_to_centroid", "fraud"}
    missing = required_columns - set(transformed_df.columns)
    if missing:
        raise ValueError(f"Hiányzó KMeans-statisztikai oszlop(ok): {sorted(missing)}")

    return (
        transformed_df.groupby("cluster_id")
        .agg(
            cluster_size=("cluster_id", "size"),
            fraud_count=("fraud", "sum"),
            fraud_rate=("fraud", "mean"),
            mean_dist_to_centroid=("dist_to_centroid", "mean"),
            max_dist_to_centroid=("dist_to_centroid", "max"),
        )
        .reset_index()
        .sort_values("cluster_id")
    )


def inspect_hdbscan_configuration(
    data_path: Path = RAW_DATA_PATH,
    feature_set_name: str = "risk_focused",
    kmeans_n_clusters: int = 5,
    min_cluster_size: int = 600,
    min_samples: int = 20,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = 42,
) -> pd.DataFrame:
    """Egy HDBSCAN-beállítás részletes ellenőrzése fraud-címkével."""
    if feature_set_name not in FEATURE_SETS:
        raise ValueError(f"Ismeretlen feature_set: {feature_set_name}")

    data = load_feature_dataset(
        data_path,
        feature_columns=FEATURE_SETS[feature_set_name],
        include_target=True,
    )
    sample_df = data.sample(n=min(len(data), sample_size), random_state=seed).copy()
    feature_df = sample_df.drop(columns=["fraud"])

    clusterer = Clusterer(
        kmeans_params={"n_clusters": kmeans_n_clusters, "random_state": seed, "n_init": 10},
        hdbscan_params={"min_cluster_size": min_cluster_size, "min_samples": min_samples},
        feature_columns=list(feature_df.columns),
    )
    transformed = clusterer.fit_transform(sample_df)
    fitted_labels = cp.asnumpy(clusterer.hdbscan_model.labels_).astype(np.int32)
    transformed["hdbscan_cluster_id"] = fitted_labels
    transformed["is_hdbscan_noise"] = (fitted_labels == -1).astype(np.int8)
    summary = summarize_hdbscan_clusters(transformed)

    non_noise = transformed[transformed["hdbscan_cluster_id"] != -1]
    if non_noise["hdbscan_cluster_id"].nunique() >= 2:
        matrix = clusterer.scaler.transform(clusterer._build_raw_matrix(non_noise))
        labels = cp.asarray(non_noise["hdbscan_cluster_id"].to_numpy(dtype=np.int32))
        hdbscan_silhouette = float(silhouette_score(matrix, labels, metric="euclidean"))
    else:
        hdbscan_silhouette = float("nan")

    summary["feature_set"] = feature_set_name
    summary["kmeans_n_clusters"] = kmeans_n_clusters
    summary["min_cluster_size"] = min_cluster_size
    summary["min_samples"] = min_samples
    summary["hdbscan_silhouette"] = hdbscan_silhouette
    return summary


def inspect_hdbscan_parameter_grid(
    data_path: Path = RAW_DATA_PATH,
    feature_set_name: str = "risk_focused",
    kmeans_n_clusters: int = 5,
    min_cluster_sizes: tuple[int, ...] = (300, 400, 500, 600, 800),
    min_samples_values: tuple[int, ...] = (20, 40, 60, 80),
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = 42,
    output_path: Path = HDBSCAN_ANALYSIS_PATH,
) -> pd.DataFrame:
    """Több HDBSCAN-paramétert hasonlít össze, az Optuna futtatásától függetlenül."""
    reports = []
    for min_cluster_size in min_cluster_sizes:
        for min_samples in min_samples_values:
            report = inspect_hdbscan_configuration(
                data_path=data_path,
                feature_set_name=feature_set_name,
                kmeans_n_clusters=kmeans_n_clusters,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                sample_size=sample_size,
                seed=seed,
            )
            reports.append(report.assign(parameter_trial=f"{min_cluster_size}_{min_samples}"))

    result = pd.concat(reports, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def _cluster_objective(
    trial: optuna.Trial,
    feature_df: pd.DataFrame,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> float:
    """Optuna objective. A clustering nem kap címkét, csak a feature vektorokat látja."""
    sample_df = feature_df.sample(n=min(len(feature_df), sample_size), random_state=42).copy()
    clustering_df = sample_df.drop(columns=["fraud"], errors="ignore")

    k = trial.suggest_int("kmeans_n_clusters", 3, 5)
    min_cluster_size = trial.suggest_categorical(
        "hdbscan_min_cluster_size",
        [350, 400, 450, 500, 550, 600],
    )
    min_samples = trial.suggest_categorical(
        "hdbscan_min_samples",
        [30, 40, 50],
    )

    clusterer = Clusterer(
        kmeans_params={"n_clusters": k, "random_state": 42, "n_init": 10},
        hdbscan_params={"min_cluster_size": min_cluster_size, "min_samples": min_samples},
        feature_columns=list(clustering_df.columns),
    )

    clusterer.fit(clustering_df)
    transformed = clusterer.transform(clustering_df)
    if "fraud" in sample_df.columns:
        transformed["fraud"] = sample_df["fraud"].to_numpy()

    labels = transformed["cluster_id"].to_numpy()
    unique_clusters = np.unique(labels)
    if len(unique_clusters) < 2:
        return -1.0

    X_scaled = clusterer.scaler.transform(clusterer._build_raw_matrix(clustering_df))
    labels_cp = cp.asarray(labels.astype(np.int32))
    sil_score = float(silhouette_score(X_scaled, labels_cp, metric="euclidean"))

    kmeans_summary = summarize_kmeans_clusters(transformed)
    noise_ratio = float(transformed["is_hdbscan_noise"].mean())
    score = sil_score - 0.35 * noise_ratio

    trial.set_user_attr("n_clusters_found", int(len(unique_clusters)))
    trial.set_user_attr("noise_ratio", noise_ratio)
    trial.set_user_attr("silhouette", float(sil_score))
    trial.set_user_attr("kmeans_n_clusters", int(k))
    trial.set_user_attr("hdbscan_min_cluster_size", int(min_cluster_size))
    trial.set_user_attr("hdbscan_min_samples", int(min_samples))
    trial.set_user_attr("max_kmeans_fraud_rate", float(kmeans_summary["fraud_rate"].max()))
    trial.set_user_attr("max_kmeans_cluster_size", int(kmeans_summary["cluster_size"].max()))

    return score


def run_optuna_tuning(
    data_path: Path = RAW_DATA_PATH,
    n_trials: int = DEFAULT_N_TRIALS,
    seed: int = 42,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    feature_columns: list[str] | None = None,
    feature_set_name: str | None = None,
) -> dict:
    selected_columns = feature_columns
    if feature_set_name is not None:
        if feature_set_name not in FEATURE_SETS:
            raise ValueError(f"Ismeretlen feature_set: {feature_set_name}. Elérhető: {list(FEATURE_SETS.keys())}")
        selected_columns = FEATURE_SETS[feature_set_name]

    feature_df = load_feature_dataset(
        data_path,
        feature_columns=selected_columns,
        include_target=True,
    )
    print(f"Dataset mérete: {feature_df.shape}")
    if feature_set_name is not None:
        print(f"Feature set: {feature_set_name}")
        print(f"Használt feature-ek: {list(feature_df.columns)}")
    print(f"Optuna teljes futás: {n_trials} trial, sample_size={min(len(feature_df), sample_size)}")
    print("A clustering bemenetből kivettük a fraud címkét, mert ez címke/target oszlop, nem feature.")

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=3,
        interval_steps=1,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(
        lambda trial: _cluster_objective(trial, feature_df, sample_size=sample_size),
        n_trials=n_trials,
    )

    best = study.best_trial
    best_result = {
        "best_score": best.value,
        "kmeans_n_clusters": best.params["kmeans_n_clusters"],
        "hdbscan_min_cluster_size": best.params["hdbscan_min_cluster_size"],
        "hdbscan_min_samples": best.params["hdbscan_min_samples"],
        "n_clusters_found": best.user_attrs["n_clusters_found"],
        "noise_ratio": best.user_attrs["noise_ratio"],
        "silhouette": best.user_attrs["silhouette"],
        "max_kmeans_fraud_rate": best.user_attrs["max_kmeans_fraud_rate"],
        "max_kmeans_cluster_size": best.user_attrs["max_kmeans_cluster_size"],
    }

    print("\n=== Legjobb paraméterek ===")
    for key, value in best_result.items():
        print(f"{key}: {value}")

    results_df = study.trials_dataframe().sort_values("value", ascending=False)
    if not RESULTS_PATH.parent.exists():
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nTeljes Optuna eredmény CSV mentve: {RESULTS_PATH}")
    print("\nTop 5 trial:")
    print(results_df.head(5)[["number", "value", "params_kmeans_n_clusters", "params_hdbscan_min_cluster_size", "params_hdbscan_min_samples"]])

    best_sample = feature_df.sample(
        n=min(len(feature_df), sample_size),
        random_state=42,
    ).copy()
    best_clustering_df = best_sample.drop(columns=["fraud"], errors="ignore")
    best_clusterer = Clusterer(
        kmeans_params={
            "n_clusters": best.params["kmeans_n_clusters"],
            "random_state": seed,
            "n_init": 10,
        },
        hdbscan_params={
            "min_cluster_size": best.params["hdbscan_min_cluster_size"],
            "min_samples": best.params["hdbscan_min_samples"],
        },
        feature_columns=list(best_clustering_df.columns),
    )
    best_clusterer.fit(best_clustering_df)
    best_transformed = best_clusterer.transform(best_clustering_df)
    best_transformed["fraud"] = best_sample["fraud"].to_numpy()
    kmeans_summary = summarize_kmeans_clusters(best_transformed)
    kmeans_summary["feature_set"] = feature_set_name or "default"
    kmeans_summary["kmeans_n_clusters"] = best.params["kmeans_n_clusters"]
    kmeans_summary["hdbscan_min_cluster_size"] = best.params["hdbscan_min_cluster_size"]
    kmeans_summary["hdbscan_min_samples"] = best.params["hdbscan_min_samples"]
    KMEANS_ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    kmeans_summary.to_csv(KMEANS_ANALYSIS_PATH, index=False)
    print(f"KMeans klaszterstatisztika mentve: {KMEANS_ANALYSIS_PATH}")

    return best_result


if __name__ == "__main__":
    selected_feature_set = "high_signal_numerical"

    print(f"\n===== OPTUNA: {selected_feature_set.upper()} =====")
    best_params = run_optuna_tuning(
        n_trials=DEFAULT_N_TRIALS,
        feature_set_name=selected_feature_set,
    )

    print(f"\n===== HDBSCAN GRID: {selected_feature_set.upper()} =====")
    hdbscan_results = inspect_hdbscan_parameter_grid(
        feature_set_name=selected_feature_set,
        kmeans_n_clusters=best_params["kmeans_n_clusters"],
        min_cluster_sizes=(350, 400, 450, 500, 550, 600),
        min_samples_values=(30, 40, 50),
    )
    print(
        hdbscan_results[
            [
                "hdbscan_cluster_id",
                "cluster_size",
                "fraud_count",
                "fraud_rate",
                "is_noise",
                "hdbscan_silhouette",
                "min_cluster_size",
                "min_samples",
            ]
        ].head(20).to_string(index=False)
    )

