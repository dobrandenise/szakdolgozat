import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/denise/szakdolgozat/data/raw/BankSim.csv")

counts = df["fraud"].value_counts().sort_index()

plt.figure(figsize=(6, 4))
plt.bar(
    ["Jogszerű", "Csalárd"],
    [counts[0], counts[1]]
)

plt.ylabel("Tranzakciók száma")
plt.title("Az osztályeloszlás a BankSim adathalmazban")

plt.tight_layout()
plt.savefig("/home/denise/szakdolgozat/plots/osztalyeloszlas.png", dpi=300)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/denise/szakdolgozat/data/raw/BankSim.csv")

fraud_rate = (
    df.groupby("category")["fraud"]
      .mean()
      .sort_values(ascending=False)
      * 100
)

plt.figure(figsize=(10, 5))
fraud_rate.plot(kind="bar")

plt.ylabel("Csalásarány (%)")
plt.xlabel("Kategória")
plt.title("Csalásarány vásárlási kategóriánként")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.savefig("/home/denise/szakdolgozat/plots/kategoriankenti_csalasarany.png", dpi=300)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/denise/szakdolgozat/data/raw/BankSim.csv")

legal = df[df["fraud"] == 0]["amount"]
fraud = df[df["fraud"] == 1]["amount"]

plt.figure(figsize=(6, 5))

plt.boxplot(
    [legal, fraud],
    labels=["Jogszerű", "Csalárd"],

    showfliers=False
)

plt.ylabel("Összeg (euró)")
plt.title("A tranzakcióösszegek eloszlása")

plt.tight_layout()
plt.savefig("/home/denise/szakdolgozat/plots/osszeg_eloszlas_boxplot.png", dpi=300)
plt.show()
