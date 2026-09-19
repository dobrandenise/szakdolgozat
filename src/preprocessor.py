import os
import pandas as pd

class Preprocessor:
    """
    Nyers CSV beolvasása és a szükséges tisztítási lépések elvégzése.

    Bemenet:  a nyers CSV elérési útja (input_path).
    Kimenet:  megtisztított CSV fájl (output_path) + a tisztított DataFrame.

    Megjegyzés: a jelenlegi preprocessor.py egy már megtisztított,
    klaszter-oszlopot is tartalmazó fájlt (`dataset_with_clusters_final.csv`)
    olvas be — ez arra utal, hogy a klaszterezés egy külön (nem mellékelt)
    lépésben történik. Ez az osztály ezért csak az általános adattisztítási
    lépéseket végzi (idézőjel-eltávolítás, konstans oszlopok eldobása,
    duplikátumok és hiányzó értékek kezelése); a `cluster` oszlopnak a
    tisztított CSV-ben már jelen kell lennie, vagy a Baseline hívása előtt
    hozzá kell fűzni.

    """

    def __init__(
            self,
            input_path,
            output_path=None,
            strip_quote_cols=None,
            constant_cols_to_drop=None,
    ):
        self.input_path = input_path
        self.output_path = output_path or self._default_output_path(input_path)

        self.strip_quote_cols = strip_quote_cols or [
            "customer",
            "age",
            "gender",
            "merchant",
            "zipcodeOri",
            "zipMerchant",
            "category",
        ]
        self.constant_cols_to_drop = constant_cols_to_drop or [
            "zipcodeOri",
            "zipMerchant",
        ]

    @staticmethod
    def _default_output_path(input_path):
        base_dir = os.path.dirname(input_path)
        return os.path.join(base_dir, "dataset_cleaned.csv")

    def load_data(self):
        return pd.read_csv(self.input_path)

    def strip_quotes(self, df):
        for col in self.strip_quote_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip("'")
        return df

    def drop_constant_columns(self, df):
        cols_present = [c for c in self.constant_cols_to_drop if c in df.columns]
        if cols_present:
            print(f"Konstans oszlopok eldobása: {cols_present}")
            df = df.drop(columns=cols_present)
        return df

    def drop_duplicates(self, df):
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        if before != after:
            print(f"Duplikált sorok eldobva: {before - after}")
        return df

    def drop_missing_values(self, df):
        before = len(df)
        df = df.dropna()
        after = len(df)
        if before != after:
            print(f"Hiányzó értékes sorok eldobva: {before - after}")
        return df

    def clean(self, df):
        df = self.strip_quotes(df)
        df = self.drop_constant_columns(df)
        df = self.drop_duplicates(df)
        df = self.drop_missing_values(df)
        return df

    def save_cleaned(self, df):
        out_dir = os.path.dirname(self.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"Tisztított adat elmentve ide: {self.output_path}")

    def run(self):
        df = self.load_data()
        print("=== Nyers adat alakja ===")
        print(df.shape)

        df = self.clean(df)

        print("\n=== Tisztított adat alakja ===")
        print(df.shape)

        self.save_cleaned(df)
        return df

if __name__ == "__main__":
    preprocessor = Preprocessor(input_path="../data/raw/BankSim.csv", output_path="../data/processed/BankSim_cleaned.csv")
    cleaned_df = preprocessor.run()