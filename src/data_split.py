import pandas as pd

class DataSplit:
    """
    Train/validation/test particionálás egy meglévő DataFrame-en.

    Bemenet: a feldolgozott DataFrame (df).

    Két particionálási stratégiát kínál:

      - customer_split(): customerenkénti (group-szintű) 60/20/20 felosztás —
        a customer-ek (nem az egyes sorok!) kerülnek train/val/test-be, majd
        mindhárom rész belsőleg step szerint növekvő sorrendbe rendeződik.

      - time_split(): tisztán step szerinti (idősoros) 60/20/20 felosztás —
        a teljes adatot step szerint növekvő sorrendbe rendezi, majd
        sorrendben vágja train/val/test részekre, customer-csoportosítás
        nélkül.
    """

    def __init__(self, df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42):
        self.df = df.copy()
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"A train/val/test arányoknak 1.0-ra kell összegződniük, jelenleg: {total}")

    def customer_split(self, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Customerenkénti 60/20/20 split: a customer-ek véletlenszerűen
        (self.random_state alapján, reprodukálhatóan) kerülnek train/val/test
        csoportba, majd mindhárom rész step szerint növekvő sorrendbe
        rendeződik.
        """

        print("=== Customer split futtatása ===")

        df = self.df if df is None else df

        customers = (df["customer"].drop_duplicates().sample(frac=1.0, random_state=self.random_state).reset_index(drop=True))

        n = len(customers)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_customers = set(customers[:n_train])
        val_customers = set(customers[n_train:n_train + n_val])
        test_customers = set(customers[n_train + n_val:])

        train_df = df[df["customer"].isin(train_customers)].sort_values("step").reset_index(drop=True)
        val_df = df[df["customer"].isin(val_customers)].sort_values("step").reset_index(drop=True)
        test_df = df[df["customer"].isin(test_customers)].sort_values("step").reset_index(drop=True)

        print(
            f"customer_split -> customerek: train={len(train_customers)}, "
            f"val={len(val_customers)}, test={len(test_customers)}"
        )
        print(
            f"customer_split -> sorok: train={len(train_df)}, "
            f"val={len(val_df)}, test={len(test_df)}"
        )

        return train_df, val_df, test_df

    def time_split(self, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Tisztán step szerinti (idősoros) 60/20/20 split: a teljes adatot
        step szerint növekvő sorrendbe rendezi, majd sorrendben vágja
        train/val/test részekre (customer-csoportosítás nélkül).
        """
        print("=== Time split futtatása ===")

        df = (self.df if df is None else df).sort_values("step").reset_index(drop=True)

        n = len(df)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_df = df.iloc[:n_train].reset_index(drop=True)
        val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
        test_df = df.iloc[n_train + n_val:].reset_index(drop=True)

        print(f"time_split -> sorok: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        print(f"time_split -> train step tartomány: {train_df['step'].min()}-{train_df['step'].max()}")
        print(f"time_split -> val step tartomány:   {val_df['step'].min()}-{val_df['step'].max()}")
        print(f"time_split -> test step tartomány:  {test_df['step'].min()}-{test_df['step'].max()}")

        return train_df, val_df, test_df


if __name__ == "__main__":

    dataframe = pd.read_csv("../data/processed/dataset_cleaned.csv")
    splitter = DataSplit(dataframe)

    train_c, val_c, test_c = splitter.customer_split()
    train_t, val_t, test_t = splitter.time_split()