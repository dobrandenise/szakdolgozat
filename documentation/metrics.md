# Metrikák

## Modell metrikák (Model metrics)

| Mutató | Képlet | Jelentés |
|----------|----------|----------|
| `fraud_rate` | `fraud_rate = N_fraud / N_total` | A fraud (pozitív osztály) aránya a vizsgált adathalmazban. Megmutatja, hogy a minták hány százaléka csalás. |
| `AUC-ROC` | ROC-görbe alatti terület | A modell általános szeparációs képességét méri a fraud és nem fraud tranzakciók között. Az 1,0 tökéletes elkülönítést, a 0,5 véletlen találgatást jelent. |
| `AUC-PR` | Precision-Recall görbe alatti terület | Erősen kiegyensúlyozatlan adatállományoknál különösen fontos mutató, mert a fraud osztály felismerésére fókuszál, és kevésbé érzékeny a negatív osztály dominanciájára. |
| `precision_fraud` | `TP / (TP + FP)` | A fraudnak jelölt tranzakciók közül mennyi volt valóban fraud. Magas értéke alacsony téves riasztási arányt jelent. |
| `accuracy` | `(TP + TN) / (TP + TN + FP + FN)` | Az összes helyesen osztályozott minta aránya. Erősen kiegyensúlyozatlan adatok esetén önmagában félrevezető lehet. |
| `recall_fraud` | `TP / (TP + FN)` | A valós fraud esetek közül mennyit talált meg a modell. A fraud felderítési arányát mutatja. |
| `f1_fraud` | `2 × (Precision × Recall) / (Precision + Recall)` | A precision és recall harmonikus átlaga. Akkor hasznos, ha mind a téves riasztások, mind a kihagyott fraud esetek minimalizálása fontos. |
| `tn` | - | True Negative: valóban nem fraud tranzakció, amelyet a modell is nem fraudnak jelölt. |
| `fp` | - | False Positive: nem fraud tranzakció, amelyet a modell tévesen fraudnak jelölt. |
| `fn` | - | False Negative: Fraud tranzakció, amelyet a modell tévesen nem fraudnak jelölt, vagyis a csalást nem sikerült felismerni. |
| `tp` | - | True Positive: Fraud tranzakció, amelyet a modell helyesen fraudnak jelölt. |
| `confusion_matrix` | `[[TN, FP], [FN, TP]]` | A modell osztályozási eredményeinek összefoglalása TN, FP, FN és TP értékek segítségével. |

### Konfúziós mátrix

|  | Predicted Legit | Predicted Fraud |
|---|---|---|
| Actual Legit | TN | FP |
| Actual Fraud | FN | TP |


## Költségfüggvény (Cost Metrics)

| Mutató | Képlet | Jelentés |
|----------|----------|----------|
| `fp_cost` | `Cost_FP = FP × fp_cost` | Egy False Positive esethez rendelt költség. A téves riasztások üzleti terhét reprezentálja (pl. manuális ellenőrzés, ügyfélértesítés vagy felesleges vizsgálat költsége). |
| `tp_cost` | `Cost_TP = TP × tp_cost` | Egy True Positive eset kezelési költsége. Akkor is keletkezhet költség, ha a fraudot sikeresen azonosította a rendszer, például manuális felülvizsgálat vagy további ellenőrzési lépések miatt. |
| `fn_power` | `Cost_FN = amount^fn_power` | A False Negative költségfüggvényének kitevője. A fraud összegének növekedésével nemlineárisan emeli a kihagyott fraud költségét, így a nagy értékű csalások aránytalanul nagyobb büntetést kapnak. |
| `threshold` | `ŷ = fraud, ha P(fraud) ≥ threshold` | Az a valószínűségi küszöb, amely felett a tranzakció fraudnak minősül. A threshold módosításával a precision és recall közötti kompromisszum szabályozható. |
| `total_cost` | `Cost_FP + Cost_TP + Cost_FN` | A kiválasztott threshold mellett keletkező teljes üzleti költség. Tartalmazza a TP, FP és FN eseményekhez kapcsolódó összes költséget. |
| `total_cost_at_0.5` | `Total Cost(threshold=0.5)` | A standard 0,5-ös threshold alkalmazásával számított teljes költség. Referenciaként használható az optimalizált threshold eredményének összehasonlítására. |
| `baseline_cost` | `Σ Cost_FN` | Az a költség, amely akkor keletkezne, ha a modell nem lenne használva, és minden fraud veszteség realizálódna. Ez szolgál alapállapotként a megtakarítás számításához. |
| `cost_savings` | `(baseline_cost - total_cost) / baseline_cost` | A modell által elért költségmegtakarítás aránya a baseline költséghez viszonyítva. Megmutatja, hogy a modell a potenciális veszteség mekkora részét előzte meg. |
| `cost` | - | A költségkiértékelés összesített eredményeit tartalmazó objektum, amely a fenti mutatókat foglalja össze. |
---
### Teljes költségfüggvény

```text
Total Cost = FP × fp_cost + TP × tp_cost + Σ(Fraud AmountFN ^ fn_power)
```

ahol:

- `FP`: False Positive esetek száma
- `TP`: True Positive esetek száma
- `Fraud AmountFN`: a kihagyott fraud tranzakció összege
- `fn_power`: a nemlineáris büntetés kitevője

---
### Fontos megjegyzés

A fraud detektálási feladat üzleti célja nem az accuracy maximalizálása, hanem a teljes költség (`total_cost`) minimalizálása. Emiatt a modell threshold-ja a költségfüggvény alapján kerül optimalizálásra, nem pedig a hagyományos 0,5-ös döntési küszöb szerint.

## Időmérési metrikák (Timing metrics)

Az időadatok a JSON rekord `timing_seconds` objektumában szerepelnek. Az időtartamok másodpercben értendők; a GPU-s műveleteknél az időmérés előtt és után szinkronizálás történik.

| Mező | Mértékegység | Jelentés |
|----------|----------|----------|
| `preprocessing` | másodperc | Adatbetöltés, tisztítás, feature engineering, adatok felosztása és kódolása. GraphSAGE esetén a tanításhoz szükséges gráfok előkészítési idejét is tartalmazza. |
| `hyperparameter_tuning` | másodperc | A hyperparaméter-keresés teljes ideje, beleértve a próbafuttatásokat. Rögzített paraméterekkel futó modellnél 0. |
| `training` | másodperc | A kiválasztott paraméterekkel végzett végső modellillesztés ideje. GraphSAGE-nél nem tartalmazza a gráf előkészítését; hibrid modellnél az XGBoost és GraphSAGE tanítási idejének összege. |
| `training_by_model` | másodperc, modellenként | Hibrid modellnél az egyes komponensek (`xgboost`, `graphsage`) végső tanítási ideje. |
| `threshold_tuning` | másodperc | Az optimális döntési küszöb meghatározásának ideje a validációs adatokon. A hibrid modellnél a súly és a küszöb keresését, valamint a szükséges validációs predikciókat is tartalmazza. Rögzített paraméterrel futó modellnél 0. |
| `inference` | másodperc | A teszthalmaz fraud-valószínűségeinek kiszámítási ideje. A metrikák és költségek kiszámítása nem része ennek az időnek. |
| `inference_by_model` | másodperc, modellenként | Hibrid modellnél az `xgboost` és a `graphsage` komponens teszthalmazon mért predikciós ideje. |
| `inference_ms_per_sample` | milliszekundum/minta | Egy tesztminta átlagos predikciós ideje: `inference × 1000 / n_test_samples`. |
| `inference_samples_per_second` | minta/másodperc | Tesztinferencia sebessége: `n_test_samples / inference`. |
| `total_pipeline` | másodperc | Az adat-előkészítés kezdetétől a metrikák kiszámításáig mért teljes futásidő. Tartalmazza az előfeldolgozást, a hangolást, a tanítást és az értékelést; a JSON-fájlba írást nem. |

Az `inference` a teszthalmaz modell általi pontozását méri, nem az adatbetöltést vagy az előfeldolgozást. A `total_pipeline` ezzel szemben a teljes mérési folyamatot foglalja össze, ezért a modell futásidejének gyakorlati, végponttól végpontig tartó becslésére használható. A hangolási időt a végső `training` időtől külön kell értelmezni: a hangolás alatt több modellpróba is lefuthat.

RF, XGBoost, GraphSAGE és a hibrid modell ismételt futtatásakor a `__main__` tízszer meghívja a `run_pipeline` függvényt; így minden ismétlés önállóan elvégzi az előfeldolgozást, tanítást és tesztelést, rögzített paraméterekkel és küszöbbel, hangolás nélkül. A hibrid modell az előre rögzített XGBoost–GraphSAGE súlyozást is használja. Minden eredményrekord tartalmazza a futássorszámot (`run_number`), az ismétléseket összekapcsoló azonosítót (`run_group`) és a saját időadatait. A `timing_summary` a csoport ismétlésszámát és az előfeldolgozás, tanítás, inference, illetve teljes pipeline átlagidejét tartalmazza; az összes futás rekordjába bekerül.
