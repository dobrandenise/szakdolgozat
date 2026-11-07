import pandas as pd
from preprocessor import Preprocessor
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "BankSim.csv"

preprocessor = Preprocessor(input_path=RAW_DATA_PATH)
df = preprocessor.run()

# 1. Osztályeloszlás
print(df['fraud'].value_counts())
print()
print("FRAUD kategóriák aránya:")
fraud_tx = df[df["fraud"] == 1]
fraud_share = (fraud_tx["category"].value_counts(normalize=True) * 100)
print(fraud_share)
print()
print("Az összes tranzakció hány százaléka tartozik az adott kategóriához?")
print(df["category"].value_counts(normalize=True) * 100)
print()
# 2. Összegek
print(df.groupby('fraud')['amount'].describe())
print()
# 3. Kategóriák
cat_stats = df.groupby('category')['fraud'].agg(['count','mean']).sort_values('mean', ascending=False)
print(cat_stats)
print()
# 4. Step szerinti tranzakciószám
print(df.groupby('step').size().describe())
print()
# 5. Demográfia
print(df.groupby('age')['fraud'].agg(['count','mean']))
print(df.groupby('gender')['fraud'].agg(['count','mean']))

