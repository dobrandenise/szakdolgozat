import numpy as np
import pandas as pd
from pathlib import Path

from data_split import DataSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class FeatureEngineering:
    """
    A projektterv 4. fejezetében ("Új feature javaslatok") felsorolt oszlopok
    előállítása. Minden oszlophoz külön metódus tartozik.

    FONTOS — kauzalitás:
    Minden history-alapú feature-t (fraud_prior-ok, tx_count_hist-ek,
    amount z-score/ratio, tenure, stb.) *expanding window*, entitásonként
    (customer/merchant/category) számolunk: egy adott sor csak az adott
    entitás **korábbi step-jeiből** számolt statisztikát kaphatja meg — a
    saját napi (step) tranzakcióit sosem, hogy egy napon belüli tranzakciók
    ne "lássák" egymást.

    Emiatt ezt az osztályt a **teljes, step szerint rendezett, megtisztított
    datasetet** kapva kell futtatni, *a train/val/test split ELŐTT* — mivel
    minden feature csakis az adott sor előtti napokra támaszkodik, ez nem
    okoz leakage-t, bármelyik split-résznek adod is majd oda az eredményt.

    Kivétel: az `is_amount_outlier` (IQR-határ) és a `global_fraud_prior`
    (Bayes-simításhoz) olyan statisztikák, amik a teljes populáció "globális"
    jellemzői — ha szigorúan train-only fit-et szeretnél, add át őket az
    `amount_outlier_bounds` / `global_fraud_prior` konstruktor-paraméterekkel
    a train-en számolt értékekkel, ahelyett hogy a metódusok magukat a
    kapott df-et használnák a becsléshez.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        step_col="step",
        customer_col="customer",
        merchant_col="merchant",
        category_col="category",
        amount_col="amount",
        fraud_col="fraud",
        bayes_m=20,
        global_fraud_prior=None,
        amount_outlier_bounds=None,
    ):
        self.df = df.sort_values([customer_col, step_col], kind="mergesort").reset_index(drop=True)

        self.step_col = step_col
        self.customer_col = customer_col
        self.merchant_col = merchant_col
        self.category_col = category_col
        self.amount_col = amount_col
        self.fraud_col = fraud_col
        self.bayes_m = bayes_m
        self.global_fraud_prior = global_fraud_prior
        self.amount_outlier_bounds = amount_outlier_bounds  
    # ------------------------------------------------------------------
    # Belső segédfüggvények (kauzális, napi granularitású history-számítás)
    # ------------------------------------------------------------------

    def _causal_daily_history(self, entity_col, value_cols):
        """
        entity_col szerint napi (step) bontásban összegzi a value_cols
        oszlopokat, majd entitásonként kumulálja és egy nappal eltolja
        (shift), így minden (entity, step) párhoz a *megelőző* napok
        összesített értékét adja — a mai nap sosem szerepel benne.
        Visszaadja ugyanolyan hosszú DataFrame-et, mint self.df, "hist_"
        előtaggal ellátott oszlopokkal.
        """
        daily = (
            self.df.groupby([entity_col, self.step_col])[value_cols]
            .sum()
            .reset_index()
            .sort_values([entity_col, self.step_col], kind="mergesort")
        )

        cum = daily.groupby(entity_col)[value_cols].cumsum()
        hist = cum.groupby(daily[entity_col]).shift(1)
        hist.columns = [f"hist_{c}" for c in value_cols]

        daily_hist = pd.concat([daily[[entity_col, self.step_col]], hist], axis=1)

        merged = self.df[[entity_col, self.step_col]].merge(
            daily_hist, on=[entity_col, self.step_col], how="left"
        )
        hist_cols = [f"hist_{c}" for c in value_cols]
        return merged[hist_cols].fillna(0.0).set_index(self.df.index)

    def _bayes_smooth(self, fraud_sum, n):
        return (fraud_sum + self.bayes_m * self.global_fraud_prior) / (n + self.bayes_m)

    # ------------------------------------------------------------------
    # #1 amount_log
    # ------------------------------------------------------------------

    def add_amount_log(self):
        self.df["amount_log"] = np.log1p(self.df[self.amount_col])
        return self

    # ------------------------------------------------------------------
    # #2 is_amount_outlier
    # ------------------------------------------------------------------

    def add_is_amount_outlier(self):
        if self.amount_outlier_bounds is not None:
            lower, upper = self.amount_outlier_bounds
        else:
            q1, q3 = self.df[self.amount_col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            self.amount_outlier_bounds = (lower, upper)

        self.df["is_amount_outlier"] = (
            (self.df[self.amount_col] < lower) | (self.df[self.amount_col] > upper)
        ).astype(int)
        return self

    # ------------------------------------------------------------------
    # #3 amount_zscore_customer
    # ------------------------------------------------------------------

    def _customer_amount_history(self):
        self.df["_amount_sq"] = self.df[self.amount_col] ** 2
        self.df["_ones"] = 1

        hist = self._causal_daily_history(
            self.customer_col, ["_ones", self.amount_col, "_amount_sq"]
        )
        n = hist["hist__ones"]
        amount_sum = hist[f"hist_{self.amount_col}"]
        amount_sumsq = hist["hist__amount_sq"]

        hist_mean = np.where(n > 0, amount_sum / n.replace(0, np.nan), np.nan)
        hist_var = np.where(n > 1, amount_sumsq / n.replace(0, np.nan) - hist_mean**2, np.nan)
        hist_std = np.sqrt(np.clip(hist_var, a_min=0, a_max=None))

        self.df = self.df.drop(columns=["_amount_sq", "_ones"])
        return n, hist_mean, hist_std

    def add_amount_zscore_customer(self):
        n, hist_mean, hist_std = self._customer_amount_history()
        self.df["amount_zscore_customer"] = np.where(
            (n > 1) & (hist_std > 0), (self.df[self.amount_col] - hist_mean) / hist_std, 0.0
        )
        return self

    # #4 amount_ratio_customer

    def add_amount_ratio_customer(self):
        n, hist_mean, _ = self._customer_amount_history()
        self.df["amount_ratio_customer"] = np.where(
            (n > 0) & (hist_mean > 0), self.df[self.amount_col] / hist_mean, 1.0
        )
        return self

    # ------------------------------------------------------------------
    # #5 amount_zscore_category
    # ------------------------------------------------------------------

    def add_amount_zscore_category(self):
        self.df["_amount_sq"] = self.df[self.amount_col] ** 2
        self.df["_ones"] = 1

        hist = self._causal_daily_history(
            self.category_col, ["_ones", self.amount_col, "_amount_sq"]
        )
        n = hist["hist__ones"]
        amount_sum = hist[f"hist_{self.amount_col}"]
        amount_sumsq = hist["hist__amount_sq"]

        hist_mean = np.where(n > 0, amount_sum / n.replace(0, np.nan), np.nan)
        hist_var = np.where(n > 1, amount_sumsq / n.replace(0, np.nan) - hist_mean**2, np.nan)
        hist_std = np.sqrt(np.clip(hist_var, a_min=0, a_max=None))

        self.df["amount_zscore_category"] = np.where(
            (n > 1) & (hist_std > 0), (self.df[self.amount_col] - hist_mean) / hist_std, 0.0
        )

        self.df = self.df.drop(columns=["_amount_sq", "_ones"])
        return self

    # ------------------------------------------------------------------
    # #6 amount_ratio_merchant_avg
    # ------------------------------------------------------------------

    def add_amount_ratio_merchant_avg(self):
        self.df["_ones"] = 1
        hist = self._causal_daily_history(self.merchant_col, ["_ones", self.amount_col])
        n = hist["hist__ones"]
        amount_sum = hist[f"hist_{self.amount_col}"]

        hist_mean = np.where(n > 0, amount_sum / n.replace(0, np.nan), np.nan)
        self.df["amount_ratio_merchant_avg"] = np.where(
            (n > 0) & (hist_mean > 0), self.df[self.amount_col] / hist_mean, 1.0
        )

        self.df = self.df.drop(columns=["_ones"])
        return self

    # ------------------------------------------------------------------
    # #7 customer_fraud_prior
    # ------------------------------------------------------------------

    def _customer_fraud_history(self):
        self.df["_ones"] = 1
        hist = self._causal_daily_history(self.customer_col, ["_ones", self.fraud_col])
        n = hist["hist__ones"]
        fraud_sum = hist[f"hist_{self.fraud_col}"]
        self.df = self.df.drop(columns=["_ones"])
        return n, fraud_sum

    def add_customer_fraud_prior(self):
        n, fraud_sum = self._customer_fraud_history()
        self.df["customer_fraud_prior"] = self._bayes_smooth(fraud_sum, n)
        return self

    # #8 customer_tx_count_hist

    def add_customer_tx_count_hist(self):
        n, _ = self._customer_fraud_history()
        self.df["customer_tx_count_hist"] = n.astype(int)
        return self

    # ------------------------------------------------------------------
    # #9 customer_days_since_first_tx
    # ------------------------------------------------------------------

    def add_customer_days_since_first_tx(self):
        first_step = self.df.groupby(self.customer_col)[self.step_col].transform("min")
        self.df["customer_days_since_first_tx"] = self.df[self.step_col] - first_step
        return self

    # ------------------------------------------------------------------
    # #10 customer_category_diversity_hist
    # ------------------------------------------------------------------

    def add_customer_category_diversity_hist(self):
        seen_categories = set()
        diversity = np.zeros(len(self.df), dtype=int)

        # entitásonkénti (customer) futó egyedi-kategória-számlálás, step sorrendben
        # (self.df már customer+step szerint rendezett a __init__-ben)
        last_customer = None
        for i, (cust, cat) in enumerate(zip(self.df[self.customer_col], self.df[self.category_col])):
            if cust != last_customer:
                seen_categories = set()
                last_customer = cust
            diversity[i] = len(seen_categories)
            seen_categories.add(cat)

        self.df["customer_category_diversity_hist"] = diversity
        return self

    # ------------------------------------------------------------------
    # #11 is_new_category_for_customer
    # ------------------------------------------------------------------

    def add_is_new_category_for_customer(self):
        if "customer_category_diversity_hist" not in self.df.columns:
            self.add_customer_category_diversity_hist()

        seen = set()
        is_new = np.zeros(len(self.df), dtype=int)
        last_customer = None
        for i, (cust, cat) in enumerate(zip(self.df[self.customer_col], self.df[self.category_col])):
            if cust != last_customer:
                seen = set()
                last_customer = cust
            is_new[i] = int(cat not in seen)
            seen.add(cat)

        self.df["is_new_category_for_customer"] = is_new
        return self

    # ------------------------------------------------------------------
    # #12 merchant_fraud_prior
    # ------------------------------------------------------------------

    def _merchant_fraud_history(self):
        self.df["_ones"] = 1
        hist = self._causal_daily_history(self.merchant_col, ["_ones", self.fraud_col])
        n = hist["hist__ones"]
        fraud_sum = hist[f"hist_{self.fraud_col}"]
        self.df = self.df.drop(columns=["_ones"])
        return n, fraud_sum

    def add_merchant_fraud_prior(self):
        n, fraud_sum = self._merchant_fraud_history()
        self.df["merchant_fraud_prior"] = self._bayes_smooth(fraud_sum, n)
        return self

    # #12b merchant_tx_count_hist

    def add_merchant_tx_count_hist(self):
        n, _ = self._merchant_fraud_history()
        self.df["merchant_tx_count_hist"] = n.astype(int)
        return self

    # ------------------------------------------------------------------
    # #13 category_fraud_prior
    # ------------------------------------------------------------------

    def _category_fraud_history(self):
        self.df["_ones"] = 1
        hist = self._causal_daily_history(self.category_col, ["_ones", self.fraud_col])
        n = hist["hist__ones"]
        fraud_sum = hist[f"hist_{self.fraud_col}"]
        self.df = self.df.drop(columns=["_ones"])
        return n, fraud_sum

    def add_category_fraud_prior(self):
        n, fraud_sum = self._category_fraud_history()
        self.df["category_fraud_prior"] = self._bayes_smooth(fraud_sum, n)
        return self

    # #13b category_tx_count_hist

    def add_category_tx_count_hist(self):
        n, _ = self._category_fraud_history()
        self.df["category_tx_count_hist"] = n.astype(int)
        return self

    # ------------------------------------------------------------------
    # #14 category_is_high_risk
    # ------------------------------------------------------------------

    def add_category_is_high_risk(self, high_risk_categories=None):
        high_risk_categories = high_risk_categories or [
            "es_leisure",
            "es_travel",
            "es_sportsandtoys",
            "es_hotelservices",
        ]
        self.df["category_is_high_risk"] = self.df[self.category_col].isin(high_risk_categories).astype(int)
        return self

    # ------------------------------------------------------------------
    # #15 tx_count_last_n_days
    # ------------------------------------------------------------------

    def _tx_count_last_n_days(self, window):
        self.df["_ones"] = 1
        daily = (
            self.df.groupby([self.customer_col, self.step_col])["_ones"]
            .sum()
            .reset_index()
            .sort_values([self.customer_col, self.step_col], kind="mergesort")
        )

        col_name = f"tx_count_last_{window}d"
        rolling_vals = (
            daily.groupby(self.customer_col)["_ones"]
            .apply(lambda s: s.rolling(window=window, min_periods=1).sum().shift(1).fillna(0.0))
            .reset_index(level=0, drop=True)
        )
        daily[col_name] = rolling_vals.values

        merged = self.df[[self.customer_col, self.step_col]].merge(
            daily[[self.customer_col, self.step_col, col_name]],
            on=[self.customer_col, self.step_col],
            how="left",
        )
        self.df[col_name] = merged[col_name].fillna(0.0).astype(int).values

        self.df = self.df.drop(columns=["_ones"])
        return self

    def add_tx_count_last_7d(self):
        return self._tx_count_last_n_days(7)

    def add_tx_count_last_30d(self):
        return self._tx_count_last_n_days(30)

    # ------------------------------------------------------------------
    # #16 days_since_last_tx_customer
    # ------------------------------------------------------------------

    def add_days_since_last_tx_customer(self):
        unique_steps = (
            self.df[[self.customer_col, self.step_col]]
            .drop_duplicates()
            .sort_values([self.customer_col, self.step_col], kind="mergesort")
        )
        unique_steps["prev_step"] = unique_steps.groupby(self.customer_col)[self.step_col].shift(1)

        merged = self.df.merge(unique_steps, on=[self.customer_col, self.step_col], how="left")
        gap = merged[self.step_col] - merged["prev_step"]
        self.df["days_since_last_tx_customer"] = gap.fillna(-1).astype(int).values
        return self

    # ------------------------------------------------------------------
    # #17 cumulative_spend_30d
    # ------------------------------------------------------------------

    def add_cumulative_spend_30d(self, window=30):
        daily = (
            self.df.groupby([self.customer_col, self.step_col])[self.amount_col]
            .sum()
            .reset_index()
            .sort_values([self.customer_col, self.step_col], kind="mergesort")
        )

        rolling_vals = (
            daily.groupby(self.customer_col)[self.amount_col]
            .apply(lambda s: s.rolling(window=window, min_periods=1).sum().shift(1).fillna(0.0))
            .reset_index(level=0, drop=True)
        )
        daily["cumulative_spend_30d"] = rolling_vals.values

        merged = self.df[[self.customer_col, self.step_col]].merge(
            daily[[self.customer_col, self.step_col, "cumulative_spend_30d"]],
            on=[self.customer_col, self.step_col],
            how="left",
        )
        self.df["cumulative_spend_30d"] = merged["cumulative_spend_30d"].fillna(0.0).values
        return self

    # ------------------------------------------------------------------
    # #20 risk_score_composite
    # Egyszerű, súlyozott kombináció a fentebb már előállított jelekből --
    # csak az összetevők megléte után hívható.
    # ------------------------------------------------------------------

    def add_risk_score_composite(self, weights=None):
        weights = weights or {
            "customer_fraud_prior": 0.35,
            "merchant_fraud_prior": 0.35,
            "category_fraud_prior": 0.15,
            "is_amount_outlier": 0.15,
        }

        missing = [c for c in weights if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"A risk_score_composite-hez hiányzó oszlopok: {missing} "
                "-- előbb hívd meg a megfelelő add_* metódusokat."
            )

        score = sum(self.df[col] * w for col, w in weights.items())
        self.df["risk_score_composite"] = score
        return self

    # ------------------------------------------------------------------
    # Az összes (klaszterezést/HDBSCAN-t nem igénylő) feature előállítása
    # egyben, a projektterv 4. fejezetének sorrendjében.
    # ------------------------------------------------------------------

    def add_dataset_wide_features(self) -> pd.DataFrame:
        """
        A teljes (train+val+test) adaton, split ELŐTT futtatandó feature-ök.
        Ezek mind kauzálisak (expanding window, entitásonként), és nem
        használnak semmilyen populáció-szintű, train-fit statisztikát —
        ezért időben biztonságosan futtathatók a teljes datasetre.
        """
        feature_methods = (
            self.add_amount_log,
            self.add_amount_zscore_customer,
            self.add_amount_ratio_customer,
            self.add_amount_zscore_category,
            self.add_amount_ratio_merchant_avg,
            self.add_customer_tx_count_hist,
            self.add_customer_days_since_first_tx,
            self.add_customer_category_diversity_hist,
            self.add_is_new_category_for_customer,
            self.add_merchant_tx_count_hist,
            self.add_category_tx_count_hist,
            self.add_category_is_high_risk,
            self.add_tx_count_last_7d,
            self.add_tx_count_last_30d,
            self.add_days_since_last_tx_customer,
            self.add_cumulative_spend_30d,
        )
        print("=== Dataset-szintű (split-független) feature engineering ===")
        for add_feature in feature_methods:
            add_feature()
        return self.df

    def fit_and_add_train_dependent_features(self, train_df: pd.DataFrame, splitter: DataSplit, split_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        A populáció-szintű (nem entitás-history, hanem globális) statisztikákat
        -- amount IQR-határok, globális fraud-arány a Bayes-simításhoz --
        KIZÁRÓLAG a train_df-ből számolja, hogy a val/test ne szivároghasson be.

        A fit-elt statisztikákkal utána a rájuk épülő oszlopokat a teljes
        (self.df, azaz train+val+test) adatra adja hozzá -- a fraud_prior
        oszlopok maga a history-számítás továbbra is a teljes idősoron,
        entitásonként kauzális, csak a simítási konstans jön train-ről.
        """
        q1, q3 = train_df[self.amount_col].quantile([0.25, 0.75])
        iqr = q3 - q1
        self.amount_outlier_bounds = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        self.global_fraud_prior = train_df[self.fraud_col].mean()

        print("=== Train-fit statisztikákra épülő feature engineering ===")
        feature_methods = (
            self.add_is_amount_outlier,
            self.add_customer_fraud_prior,
            self.add_merchant_fraud_prior,
            self.add_category_fraud_prior,
            self.add_risk_score_composite,
        )
        for add_feature in feature_methods:
            add_feature()
        if split_name == "customer":
            return splitter.customer_split(self.df)
        elif split_name == "time":
            return splitter.time_split(self.df)
        else:
            raise ValueError(f"Ismeretlen split_name: {split_name}. Használj 'customer' vagy 'time'.")

if __name__ == "__main__":
    fe = FeatureEngineering(pd.read_csv(PROJECT_ROOT / "data" / "processed" / "dataset_cleaned.csv"))
    result_df = fe.add_dataset_wide_features()
    result2_df = fe.fit_and_add_train_dependent_features(result_df[result_df["step"] < 100])
    print(result_df.shape)
    print(result_df.columns.tolist())