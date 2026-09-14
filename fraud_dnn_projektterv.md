# Technikai Projektterv — Neurális Hálózat Alapú Fraud Detektáló Rendszer
### BankSim tranzakciós adathalmaz (bs140513_032310.csv) alapján

**Cél:** a jelenlegi KMeans + RandomForest/HistGradientBoosting baseline kiváltása/kiegészítése egy mély neurális hálózat alapú rendszerrel, elsődlegesen a **False Positive (FP) ráta érdemi csökkentése** érdekében, a Fraud Recall számottevő romlása nélkül.

---

## 0. Vezetői összefoglaló

A csatolt CSV (594 643 tranzakció, 10 oszlop) elemzése és a baseline szkriptek (`clustering.py`, `best_k.py`, `baseline.py`) tényleges lefuttatása alapján a következő adja a terv gerincét:

- A jelenlegi **HistGradientBoosting baseline** a stratifikált teszthalmazon **7 523 False Positive**-t termel mindössze **1 725 True Positive** mellett (FPR ≈ 5,12%) — ez konkrétan és számszerűen igazolja a leírt problémát: minden 4,4 riasztásból csak 1 a valódi csalás.
- A **RandomForest** ezzel szemben sokkal konzervatívabb (FP = 218, FPR ≈ 0,15%), de cserébe a Recall csak 64% — a csalások több mint egyharmadát elengedi.
- Egy gyors feature-fontosság becslés szerint a predikciós erő **~99,5%-a** három csoportból jön: **category (43,4%) + merchant (28,8%) + amount (27,3%)**. Az `age` és `gender` gyakorlatilag elhanyagolható (együtt < 0,3%).
- **Időbeli eltolódás (temporal drift)** egyértelműen kimutatható: a step (nap) és a napi fraud-arány korrelációja **-0,94**, és time-based validáció mellett a Recall drasztikusan visszaesik (RF: 64%→47%, HGB: 96%→86%). Ez azt jelenti, hogy a végleges validációs stratégiának **időalapúnak** kell lennie, nem pusztán random stratifikáltnak.
- A tervezett DNN-rendszer legfontosabb pillérei: **causal (jövőmentes) customer/merchant profilozás**, **out-of-fold target encoding** a leakage elkerülésére, **Focal Loss / cost-sensitive tanítás**, **valószínűség-kalibráció**, **threshold-optimalizáció** és egy **két-lépcsős (high-recall szűrő → high-precision döntő) architektúra** emberi felülvizsgálati sávval.
- Reális, adatalapú célkitűzés: a jelenlegi HGB-szintű FP (≈7 500) **30–45%-os csökkentése** (kb. 4 100–5 300 FP-re) Recall ≥ 90% megtartása mellett, konfigurálható üzleti threshold-dal.

---

## 1. Dataset elemzés

### 1.1 Alapadatok

| Jellemző | Érték |
|---|---|
| Sorok száma | 594 643 |
| Oszlopok száma | 10 |
| Hiányzó érték | 0 (egyetlen oszlopban sem) |
| Duplikált sor | 0 |
| Időtartam | `step` = 0–179 (180 nap, kb. 6 hónap) |
| Fraud arány | 7 200 / 594 643 = **1,21%** (legit:fraud ≈ **81,6 : 1**) |

### 1.2 Oszloponkénti elemzés

| Oszlop | Típus | Egyedi érték | Megjegyzés |
|---|---|---|---|
| `step` | int | 180 | Nap-index 0–179; erős időbeli mintázat (ld. 1.5) |
| `customer` | kategorikus ID | 4 112 | Magas kardinalitás — nem OHE-elhető direktben; ~144,6 tranzakció/ügyfél átlagosan (SD 42,8; min 5, max 265) |
| `age` | kategorikus | 8 (0–6 + `U`) | Gyakorlatilag lapos fraud-eloszlás, **elhanyagolható önálló szignál** |
| `gender` | kategorikus | 4 (F, M, E, U) | Gyenge-közepes szignál; `E`/`U` extrém ritka (0,2% / 0,09%) |
| `zipcodeOri` | kategorikus | **1** | **Konstans** (`28007`) — nulla infótartalom, törlendő |
| `merchant` | kategorikus ID | 50 | **Erősen prediktív** — egyes merchantok fraud-rátája 0–96% között szór |
| `zipMerchant` | kategorikus | **1** | **Konstans** (`28007`) — nulla infótartalom, törlendő |
| `category` | kategorikus | 15 | **A legerősebb egyedi feature**; `es_transportation` a sorok 84,9%-a, és fraud-rátája **0%** |
| `amount` | float | 23 767 | Erősen jobbra ferde (skewness = 32,37); fraud tranzakciók átlaga **16,7×** magasabb, mint a legiteké |
| `fraud` | int (0/1) | 2 | Célváltozó, extrém imbalance |

**Azonnali adattisztítási teendők:**
1. `zipcodeOri` és `zipMerchant` eldobása (konstans, egyetlen bitnyi infót sem hordoz ezen a mintán).
2. String-oszlopok idézőjeleinek eltávolítása (`'C123...'` → `C123...`), ahogy a `clustering.py`/`best_k.py` már teszi.
3. `age`/`gender` ritka kategóriáinak (`U`, `E`) kezelése — összevonás egy "other/unknown" szintbe javasolt a modellstabilitás érdekében, mivel néhány száz/ezer mintás kategóriákról van szó.
4. `customer` és `merchant` **nem** kerülhet be nyers One-Hot Encoding-ként a `customer` esetén (4 112 dimenzió → dimenzióátok, túltanulási kockázat) — helyette embedding réteg vagy aggregált (viselkedési) feature-ök.

### 1.3 Class imbalance

Fraud = **1,21%**, ami erős, de nem szélsőséges (nem 1:1000 nagyságrend) imbalance — kezelhető class-weighting/Focal Loss eszközökkel SMOTE nélkül is (ld. 7. fejezet).

### 1.4 `amount` eloszlás és outlierek

| Metrika | Legit | Fraud |
|---|---|---|
| Átlag | 31,85 | **530,93** |
| Szórás | 31,47 | 835,59 |
| Medián (fraud) | — | 319,18 |
| Teljes minta mediánja | 26,90 | |
| Max | — | 8 329,96 |

- `amount` ↔ `fraud` korreláció: **0,490** (nyers), **0,285** (log-transzformált) — a nyers `amount` meglepően erős lineáris szignál, de a log-transzformáció is szükséges a neurális háló numerikus stabilitásához (a skewness 32,37 enélkül instabil gradienst okozna).
- **IQR-alapú outlier-analízis:** felső határ = 85,74 (Q3 + 1,5×IQR). A tranzakciók **4,34%-a** esik e fölé, és ezek fraud-rátája **24,05%**, szemben a nem-outlierek **0,18%**-ával — **133-szoros lift**. Az `amount`-outlier jelző önmagában az egyik legerősebb egyszerű feature.

### 1.5 Kategóriánkénti fraud-arány — a legfontosabb egyedi jel

| Category | N | Fraud arány | Lift (globálishoz) |
|---|---:|---:|---:|
| es_leisure | 499 | 94,99% | 78,5× |
| es_travel | 728 | 79,40% | 65,6× |
| es_sportsandtoys | 4 002 | 49,53% | 40,9× |
| es_hotelservices | 1 744 | 31,42% | 26,0× |
| es_otherservices | 912 | 25,00% | 20,6× |
| es_home | 1 986 | 15,21% | 12,6× |
| es_health | 16 133 | 10,51% | 8,7× |
| es_tech | 2 370 | 6,67% | 5,5× |
| es_wellnessandbeauty | 15 086 | 4,76% | 3,9× |
| es_hyper | 6 098 | 4,59% | 3,8× |
| es_barsandrestaurants | 6 373 | 1,88% | 1,6× |
| es_fashion | 6 454 | 1,80% | 1,5× |
| es_contents | 885 | 0,00% | 0× |
| es_food | 26 254 | 0,00% | 0× |
| **es_transportation** | **505 119 (84,9%!)** | **0,00%** | 0× |

**Kritikus megfigyelés:** a három legnagyobb volumenű kategória (`es_transportation`, `es_food`, `es_contents` — együtt a minta 89,6%-a) fraud-rátája **pontosan 0%** ezen a datasetten. Ez egyszerre (a) rendkívül erős, szinte determinisztikus jel a modellnek, és (b) módszertani figyelmeztetés: ha ez szimulációs artifact (a BankSim generátor nem termel fraud-ot ezekben a kategóriákban), akkor **éles, valós adaton ez a szabály várhatóan nem fog ilyen élesen teljesülni** — a modellnek nem szabad "megtanulnia", hogy ezek a kategóriák *soha* nem lehetnek csalások, csak hogy *ritkán*. Ezt a 9. fejezet (Kockázatok) is részletezi.

### 1.6 `age` / `gender` — gyenge jelek

| age | N | Fraud% | | gender | N | Fraud% |
|---|---:|---:|---|---|---:|---:|
| 0 | 2 452 | 1,96% | | F | 324 565 | 1,47% |
| 1 | 58 131 | 1,19% | | M | 268 385 | 0,91% |
| 2 | 187 310 | 1,25% | | E | 1 178 | 0,59% |
| 3 | 147 131 | 1,19% | | U | 515 | 0,00% |
| 4 | 109 025 | 1,29% | | | | |
| 5 | 62 642 | 1,10% | | | | |
| 6 | 26 774 | 0,97% | | | | |
| U | 1 178 | 0,59% | | | | |

Az `age` fraud-eloszlása gyakorlatilag lapos (0,97–2,0% sáv) — önmagában alig van diszkriminatív ereje. A `gender` esetén F valamivel magasabb rátát mutat, de a hatás közepes-gyenge.

### 1.7 `merchant` — erős, de finomszemcsés jel

50 merchant, tranzakciószám erős szórással (medián 604, max ~299 693 — utóbbi nyilvánvalóan az `es_transportation` domináns merchantja). A fraud-ráta merchantonként **0%-tól 96,3%-ig** terjed (pl. `M1294758098`: 96,3% [n=191], `M980657600`: 83,2% [n=1 769]). Mivel a `merchant` és `category` erősen összefügg (egy merchant jellemzően egy kategóriában működik), a kettő részben redundáns, de a `merchant` finomabb felbontást ad, mint a `category` önmagában (ld. a feature-fontosság becslést, 1.9).

### 1.8 Időbeli mintázat (`step`) — temporal drift

A napi (step-szintű) fraud-arány és maga a `step` között a korreláció **-0,9355** — azaz a szimulált időszak előrehaladtával a fraud-arány **csökken**. Ez később, a 3. fejezet baseline-reprodukciójában konkrét bizonyítékkal is alátámasztásra kerül: time-based train/test split mellett a modellek Recall-ja számottevően visszaesik a random splithez képest. **Ez az egyik legfontosabb módszertani következtetés:** a validációs stratégiának időalapúnak kell lennie.

### 1.9 Feature-fontosság gyors becslése

Egy gyors RandomForest (150 fa, max_depth=12, class_weight="balanced") illesztésével `step`, `age`, `gender`, `category`, `merchant`, `amount` bemeneteken (OHE-elt kategorikusokkal), a fontosságokat feature-csoportonként összegezve:

| Feature-csoport | Aggregált fontosság |
|---|---:|
| category | 43,40% |
| merchant | 28,79% |
| amount | 27,26% |
| step | 0,27% |
| gender | 0,20% |
| age | 0,08% |

**Következmény a tervre nézve:** a `category` + `merchant` + `amount` hármas adja a magyarázott fontosság **~99,5%-át**. A feature engineering erőforrásait elsősorban ezek finomítására (interakciók, arányok, history) érdemes fordítani, nem az `age`/`gender` köré épített bonyolult konstrukciókra.

### 1.10 Velocity-jellegű jelek — korlátozott érvényesség ezen a datasetten

Ügyfelenkénti napi tranzakciószám: átlag **1,039** (SD 0,21), **max 5**. Ez azt jelenti, hogy a klasszikus "sebesség" (velocity) alapú csalásjelzők (pl. "X tranzakció Y percen belül") ezen a *szimulált* datasetten **eleve gyenge szignált** adnak, mert a generátor szinte mindig legfeljebb napi 1 tranzakciót rendel egy ügyfélhez. Ez fontos korlátozás: a 4. fejezetben javasolt velocity-feature-öket inkább **termelési robusztusság / jövőbeli valós adat** céljából érdemes implementálni, nem azért, mert *ezen* a datasetten nagy hozamot várnánk tőlük.

### 1.11 Ügyfél-szintű profil és a leakage kockázata

- Ügyfelenkénti amount-tól való eltérés (z-score az ügyfél saját átlagához/szórásához képest) ↔ fraud korreláció: **0,263**
- Ügyfél átlagához viszonyított arány (`amount / customer_mean_amount`) ↔ fraud korreláció: **0,385**
- Kategória-szintű z-score ↔ fraud korreláció: 0,143 (gyengébb, mert a `category` már önmagában erősen szeparál)
- Ügyfelenkénti fraud-arány eloszlása: átlag 2,35%, medián 0%, max 94,6%. **2 629 ügyfélnek (63,9%) soha nincs fraud tranzakciója**, **1 483-nak (36,1%) van legalább egy**, és ezek **71,3%-ának (1 057 fő) több is** — azaz **visszatérő elkövetői minta** figyelhető meg.

Ez erős érv amellett, hogy a customer-history alapú feature-ök (pl. "ennek az ügyfélnek volt-e már korábban fraud tranzakciója") komoly prediktív erővel bírnának — **DE** ezeket kizárólag **kauzálisan** (csak a vizsgált tranzakció *előtti* tranzakciók alapján, "expanding window" logikával) szabad számolni, különben súlyos data leakage történik (a modell gyakorlatilag "megkapja a választ").

Naiv (nem out-of-fold, in-sample) target encoding tesztje jól mutatja a veszélyt: a `merchant`-enkénti fraud-arány in-sample korrelációja a valódi `fraud`-dal **0,732**, a `category`-é **0,571** — ezek gyanúsan magas számok, amik **kizárólag out-of-fold / időben visszamenőleges (expanding) számítással** használhatók biztonságosan modellbemenetként (ld. 4. és 7. fejezet).

---

## 2. Baseline értékelés

### 2.1 A jelenlegi pipeline (a csatolt szkriptek alapján)

1. `best_k.py` / `clustering.py`: `amount_log` + OHE(`category`, `age`, `gender`) → StandardScaler + KMeans, k=5..10 közül a silhouette-score és elbow-görbe alapján **k=9** kerül kiválasztásra.
2. `baseline.py`: bemeneti feature-ök = `step, age, gender, category, amount, cluster` (OHE + numerikus), 75/25 train/test split, **stratifikált, de nem customer-csoportosított** (`train_test_split(..., stratify=y, random_state=42)` — a kód beemeli a `groups` változót, de ténylegesen nem használ `GroupShuffleSplit`-et).
3. Modellek: `RandomForestClassifier(class_weight="balanced")` és `HistGradientBoostingClassifier(class_weight="balanced")`, alapértelmezett 0,5 döntési küszöbbel.

### 2.2 Reprodukált eredmények (a csatolt szkriptek logikáját lefuttatva a nyers CSV-n)

Mivel a `dataset_with_clusters_final.csv` (amit a `baseline.py` ténylegesen beolvas) nem volt csatolva, a `clustering.py` logikáját (KMeans, k=9) reprodukáltam a nyers CSV-n, majd lefuttattam a `baseline.py` modell-részét — így a lenti számok **közvetlenül a te pipeline-od valós viselkedését** mutatják.

**A) Random stratifikált split (a `baseline.py` jelenlegi módszere):**

| Modell | AUC-ROC | AUC-PR | Precision | Recall | F1 | FPR | TN | **FP** | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RandomForest | 0,951 | 0,762 | 0,841 | 0,641 | 0,727 | 0,148% | 146 643 | **218** | 647 | 1 153 |
| HistGradientBoosting | 0,993 | 0,801 | 0,187 | 0,958 | 0,312 | 5,123% | 139 338 | **7 523** | 75 | 1 725 |

**Ez pontosan igazolja a leírt problémát:** a HGB modell magasabb AUC-t és Recall-t ér el, de **minden 4,4 riasztásból csak 1 valódi csalás** (Precision = 18,7%) — 7 523 legitim tranzakciót blokkolna/jelölne feleslegesen. Az RF ezzel szemben nagyon konzervatív (218 FP), de a csalások **35,9%-át elengedi**.

**B) Customer-csoportosított split (`GroupShuffleSplit` ügyfél szerint) — robusztussági ellenőrzés:**

| Modell | AUC-ROC | Precision | Recall | F1 | FPR | FP |
|---|---:|---:|---:|---:|---:|---:|
| RandomForest | 0,954 | 0,852 | 0,653 | 0,739 | 0,135% | 198 |
| HistGradientBoosting | 0,993 | 0,180 | 0,964 | 0,304 | 5,205% | 7 617 |

A számok gyakorlatilag megegyeznek a random splittel — ezen a konkrét feature-készleten (customer ID közvetlenül nincs bevonva) a random split **nem** okoz durva csoport-leakage-t. Ugyanakkor amint customer-history feature-ök épülnek be (4. fejezet), a grouped/time-based split **kötelezővé válik**.

**C) Time-based split (step ≤ 139 → train, step > 139 → test, kb. az utolsó ~25% "jövő" adata) — a legfontosabb teszt:**

| Modell | AUC-ROC | Precision | Recall | F1 | FPR | FP |
|---|---:|---:|---:|---:|---:|---:|
| RandomForest | 0,946 | 0,898 | **0,469** | 0,617 | 0,058% | 85 |
| HistGradientBoosting | 0,992 | 0,257 | **0,863** | 0,396 | 2,730% | 3 993 |

**Ez a legfontosabb eredmény a tervezéshez:** valódi, jövőbeli adatra validálva mindkét modell Recall-ja drasztikusan visszaesik (RF: 64,1%→46,9%, HGB: 95,8%→86,3%), miközben a FP-szám is csökken (ami a fraud-arány időbeli csökkenésének is betudható). Ez konkrét, számszerű bizonyíték a temporal drift jelenlétére, és **erősen alátámasztja, hogy a végső validációs/tesztelési stratégia time-based legyen** (ld. 7. fejezet), különben a fejlesztés alatt mért metrikák túl optimisták lesznek.

### 2.3 Módszertani kritikai pontok a jelenlegi pipeline-on

1. **Nincs valódi time-based validáció** — a random stratifikált split szisztematikusan túlbecsüli a Recall-t (ld. 2.2/C).
2. **Fix 0,5 döntési küszöb** — nincs threshold-optimalizáció, holott ez önmagában, modellváltás nélkül is jelentős FP-csökkentést hozhat (ld. 7. fejezet).
3. **A `cluster` feature valószínűleg redundáns** — a KMeans ugyanazokból a bemenetekből (`amount_log`, `category`, `age`, `gender` OHE) készül, amiket a klasszifikátor egyébként is közvetlenül lát. Fatörzs-alapú modellek (RF, HGB) enélkül is képesek ugyanezt a particionálást megtanulni — a cluster-ID hozzáadott értéke emiatt valószínűleg marginális (ld. 3. fejezet).
4. **Nincs valószínűség-kalibráció** — `class_weight="balanced"` torzítja a predikált valószínűségeket, ami megnehezíti egy üzletileg értelmezhető, stabil threshold beállítását.
5. **`customer` egyáltalán nincs felhasználva** (a `baseline.py` kommentje szerint a OneHotEncoding "nem bírta el" a gépi erőforrás) — ez pontosan az a hiányosság, amit az aggregált/causal customer-profil feature-ök (4. fejezet) orvosolnak anélkül, hogy 4 112 dimenziós OHE-t kellene létrehozni.

---

## 3. KMeans vs HDBSCAN összehasonlítás

### 3.1 Mire használjuk a klaszterezést?

Három lehetséges szerep:
1. **Feature-generálás** — `cluster_id` / `distance_to_centroid` mint bemenet a klasszifikátornak.
2. **Szegmentálás** — klaszterenként külön modell (ld. 5. fejezet "Clusterenként külön modell" opció).
3. **Anomália-előszűrés** — a klaszterezés zaj/outlier jelzése (elsősorban HDBSCAN erőssége) önálló riasztási jelként vagy második rétegként.

### 3.2 KMeans a jelen adaton

A `clustering.py` k=9-cel fut, `amount_log` + OHE(`category`, `age`, `gender`) bemeneten. **Probléma:** mivel a klaszterezés bemenete pontosan ugyanaz a néhány feature, amit a downstream klasszifikátor is közvetlenül megkap, a `cluster_id` információtartalma nagy átfedésben van a nyers bemenetekkel — egy elég mély fatörzs-modell (vagy egy elég széles MLP) enélkül is meg tudja tanulni ugyanazt a felosztást. Ez nem jelenti azt, hogy a `cluster_id` haszontalan (egy nemlineáris, tömörített reprezentáció még segíthet a konvergencián), de **nem várható tőle nagy önálló hozzáadott érték** a `category`+`merchant`+`amount` feature-ök mellett (1.9 szerint ezek adják az importancia 99,5%-át).

**KMeans előnyei itt:** gyors, O(n·k) skálázódik jól 594k sorra, determinisztikus, egyszerűen reprodukálható, jól illeszkedik a jelenlegi pipeline-ba.
**KMeans hátrányai:** feltételezi a gömbszerű, hasonló méretű klasztereket; a kategorikus OHE-tér nem euklideszi jellegű, ami torzíthatja a távolságszámítást; nincs natív "zaj/outlier" fogalma — minden pont kap egy klasztert, még a nyilvánvaló kiugró tranzakciók is.

### 3.3 HDBSCAN mint alternatíva

**Előnyök:**
- Nem kell előre megadni a klaszterszámot.
- Sűrűségalapú — jobban kezeli a nem-gömb alakú, eltérő sűrűségű struktúrákat.
- **Explicit zaj-címke (-1)** — a "sehova nem illeszkedő" tranzakciók automatikusan megjelölődnek, ami *önmagában* egy hasznos anomália-jelző lehet (a 24,05% vs 0,18% outlier-fraud lift, 1.4, arra utal, hogy a "nem tipikus" tranzakciók erősen fraud-gyanúsak).

**Hátrányok:**
- Számításigényesebb nagy n-en (594 643 sor); az alap `hdbscan` csomag `O(n log n)` – `O(n²)` között mozoghat a paraméterezéstől függően — **GPU-gyorsított változat (RAPIDS `cuml.cluster.HDBSCAN`, CUDA 12.6-kompatibilis) erősen javasolt**, CPU-n célszerű mintavételezéssel (pl. 100–150k sor) validálni előbb.
- Érzékeny a `min_cluster_size` / `min_samples` paraméterekre; kategorikus OHE-tér itt is torzíthat, ezért a metrikus tér megválasztása (pl. Gower-távolság vegyes típusú adatra, vagy embedding-alapú előfeldolgozás) kritikus.
- A zaj-pontok aránya könnyen elszalad, ha a paraméterezés nem megfelelő (túl sok/túl kevés "-1").

### 3.4 Javaslat

- **Futtasd le mindkettőt**, de ne várj nagy önálló hozamot egyiktől sem a category/merchant/amount melletti *közvetlen* klasszifikációs bemenetként.
- A **HDBSCAN zaj-jelzőjét (`is_hdbscan_noise`)** érdemesebb **kiegészítő anomália-feature-ként** vagy a két-lépcsős architektúra (7. fejezet) előszűrőjeként hasznosítani, nem elsődleges szegmentálóként.
- A **KMeans-t** inkább a "Clusterenként külön modell" (5. fejezet) kísérlethez érdemes megtartani, ahol a klaszterek nem feature-ként, hanem *routing*-eszközként funkcionálnak.
- Mindkét klaszterezést **kizárólag train-adaton illeszd**, a val/test adatra csak `predict`/`transform` — különben ez is leakage-forrás.

---

## 4. Új feature javaslatok

Minden javaslat a **kauzalitás elvét** követi: history-alapú feature-ök kizárólag a vizsgált tranzakció *előtti* eseményekből számolhatók (expanding window, `step` szerint rendezve), különben leakage keletkezik. A target-encoding jellegű feature-ök (`merchant_fraud_prior`, `category_fraud_prior`) **kizárólag out-of-fold vagy időben visszamenőleges (csak múltbeli fraud-címkék) számítással** kerülhetnek be.

| # | Feature | Cél / leírás | Várható hatás | Implementációs nehézség | FP-csökkentő potenciál |
|---|---|---|---|---|---|
| 1 | `amount_log` | `log1p(amount)` — a skewness=32,37 miatt szükséges numerikus stabilizálás | Stabilabb gradiens, jobb NN-konvergencia | Alacsony | Közepes |
| 2 | `is_amount_outlier` | IQR-alapú (>85,74) bináris jelző | 133× lift a fraud-rátában (1.4) — erős önálló jel | Alacsony | **Magas** |
| 3 | `amount_zscore_customer` | (amount − ügyfél múltbeli átlaga) / ügyfél múltbeli szórása | Korrelált a fraud-dal (0,263 exploratory) — "szokatlan összeg ennek az ügyfélnek" | Közepes (expanding window kell) | Magas |
| 4 | `amount_ratio_customer` | amount / ügyfél múltbeli átlagos költése | Erősebb, mint a z-score (0,385 exploratory) | Közepes | Magas |
| 5 | `amount_zscore_category` | (amount − kategória múltbeli átlaga) / kategória szórása | "Szokatlan összeg ehhez a kategóriához képest" | Közepes | Közepes |
| 6 | `amount_ratio_merchant_avg` | amount / az adott merchant múltbeli átlagos tranzakciója | Merchant-szintű anomália | Közepes | Közepes-Magas |
| 7 | `customer_fraud_prior` | Ügyfél múltbeli fraud-rátája (csak korábbi tranzakciókból, expanding) | 36,1% ügyfélnek volt már fraud-ja, ezek 71,3%-a visszatérő — erős visszaeső-mintázat | Közepes-Magas (szigorú időbeli particionálás kell) | **Magas**, de **leakage-kritikus** |
| 8 | `customer_tx_count_hist` | Ügyfél eddigi tranzakcióinak száma | "Új" vs "törzs" ügyfél megkülönböztetése | Alacsony | Közepes |
| 9 | `customer_days_since_first_tx` | Az ügyfél "kora" a rendszerben (napokban) | Friss fiókok kockázatelemzése | Alacsony | Közepes |
| 10 | `customer_category_diversity_hist` | Ügyfél eddig hány különböző kategóriában vásárolt (nunique) | Átlag 7,7 kategória/ügyfél; szokatlanul szűk/széles reptoár kockázati jel lehet | Közepes | Közepes |
| 11 | `is_new_category_for_customer` | Az adott kategóriában vásárolt-e már korábban ez az ügyfél | "Először vásárol ilyesmit" — klasszikus fraud-mintázat | Közepes | Magas |
| 12 | `merchant_fraud_prior` | Merchant múltbeli fraud-rátája (expanding, out-of-fold) | A merchant-fraud korreláció (naiv, in-sample) 0,73 — nagyon erős jel, óvatos implementációval | Magas (leakage-veszély!) | **Magas** |
| 13 | `category_fraud_prior` | Kategória múltbeli fraud-rátája (expanding, out-of-fold) | Hasonló, mint #12, de kategória-szinten (naiv korreláció 0,57) | Közepes-Magas | Magas |
| 14 | `category_is_high_risk` | Bináris flag a historikusan magas fraud-rátájú kategóriákra (leisure/travel/sportsandtoys/hotelservices) | Egyszerű, robusztus, kevésbé leakage-érzékeny, mint a folytonos prior | Alacsony | Közepes-Magas |
| 15 | `tx_count_last_7d` / `tx_count_last_30d` | Rolling tranzakciószám az ügyfélnél | Klasszikus velocity-jel — **ezen a datasetten gyenge** (napi tx/ügyfél átlag 1,04, max 5), de production-robusztusság miatt ajánlott | Közepes-Magas (rolling window per customer) | Alacsony (ezen a datasetten), Magas (valós adaton) |
| 16 | `days_since_last_tx_customer` | Napok száma az ügyfél előző tranzakciója óta | "Hirtelen aktivitás hosszú csend után" mintázat | Közepes | Közepes |
| 17 | `cumulative_spend_30d` | Ügyfél 30 napos gördülő összköltése | Költési minta hirtelen megugrása | Közepes-Magas | Közepes |
| 18 | `cluster_id` (KMeans/HDBSCAN) + `dist_to_centroid` | Klaszter-hovatartozás és centroid-távolság | Lásd 3. fejezet — várhatóan korlátozott önálló hozam | Alacsony-Közepes | Alacsony-Közepes |
| 19 | `is_hdbscan_noise` | HDBSCAN zaj-címke (bináris) | Kiegészítő anomália-jelző, a 24× outlier-liftre alapozva | Közepes-Magas (HDBSCAN skálázás) | Közepes |
| 20 | `risk_score_composite` | Több fenti jel súlyozott (pl. logreg-fit vagy szabályalapú) kombinációja egyetlen meta-feature-ré | Egyszerűsíti a downstream modell dolgát, jó interpretálhatóság a human-review réteghez | Közepes | Közepes |

**Megjegyzés az `age`/`gender` köré épített feature-ökről:** a 1.9 fejezet szerint ezek együttes fontossága < 0,3%, ezért **nem javasolt** rájuk komplex feature-eket építeni (pl. age×category interakció) — ez erőforrás-pazarlás lenne ehhez az adathalmazhoz képest.

---

## 5. Neurális háló architektúrák

### A) Egyszerű MLP (gyors baseline-DNN, első kísérlet)

| Paraméter | Javaslat |
|---|---|
| Rejtett rétegek | 3 réteg: 256 → 128 → 64 neuron |
| Aktiváció | ReLU (vagy GELU a jobb gradiensáramlásért) |
| Batch Normalization | minden Linear réteg után, aktiváció előtt |
| Dropout | 0,3 az első két rétegen, 0,2 az utolsón |
| Residual kapcsolat | opcionális, ezen a mérethez (3 réteg) nem kritikus |
| Regularizáció | L2 (weight decay 1e-5–1e-4) + Dropout + BatchNorm együtt |
| Kimenet | 1 neuron, sigmoid (vagy logit + BCEWithLogitsLoss numerikus stabilitásért) |
| Kategorikus kezelés | egyszerű OHE vagy alacsony-dimenziós embedding (`category`: 15→8, `merchant`: 50→16, `age`: 8→4, `gender`: 4→2) |

Ez a modell **gyors iterációs alap** — célja megmutatni, hogy egy alapszintű NN már a causal feature-ökkel (4. fejezet) és Focal Loss-szal (7. fejezet) felülmúlja-e a jelenlegi HGB Precision/Recall trade-off-ját.

### B) Fejlettebb Deep Neural Network

- **Embedding rétegek** minden kategorikus változóra (a fenti dimenziókkal), amik konkatenálva kerülnek a numerikus feature-ökkel (amount_log, engineered ratio/z-score feature-ök) egy közös bemeneti vektorba.
- **Residual blokkok:** 2–3 db `[Linear → BatchNorm → GELU → Dropout → Linear → BatchNorm] + skip-connection` blokk, 128–256 rejtett dimenzióval — segíti a mélyebb háló (5–8 réteg) taníthatóságát instabilitás nélkül.
- **Self-attention komponens (opcionális, könnyű verzió):** egyetlen multi-head attention réteg (2–4 fej) a bemeneti feature-embeddingek felett, hogy a modell megtanulja, mely feature-párok (pl. `category` × `amount_zscore`) együttesen relevánsak — ez lényegében egy "mini FT-Transformer" blokk (ld. C pont).
- **Normalizáció:** LayerNorm az attention blokk körül, BatchNorm a feed-forward részekben.
- **Kimeneti fej:** 2 rétegű MLP (64→1) a végső reprezentációból, sigmoid/logit kimenettel.

### C) Tabular Deep Learning modellek összehasonlítása

| Modell | Erősség | Gyengeség ezen a feladaton | Illeszkedés |
|---|---|---|---|
| **TabNet** | Szekvenciális feature-szelekció (attention-alapú "melyik feature-t nézzem most"), beépített interpretálhatóság (feature-fontosság maszkokból) | Overkill ~10–40 feature-re; a szekvenciális döntéshozatal előnye sok-száz feature-nél mutatkozik meg igazán; nehezebben hangolható (sok hiperparaméter: `n_steps`, `n_d`, `n_a`, `gamma`) | Közepes |
| **FT-Transformer** | Minden feature (numerikus is) tokenizálva, majd Transformer-enkóder — erős a feature-interakciók megtanulásában, jó publikált benchmark-eredmények tabular fraud/credit adatokon | Számításigényesebb, mint egy MLP; kisebb datasetnél (itt 594k sor még belefér) hajlamos túltanulni erős regularizáció nélkül | **Magas — elsődleges DL-jelölt** |
| **TabTransformer** | Csak a kategorikus feature-öket transzformálja attention-nal, a numerikusokat külön ágon kezeli, majd konkatenál | Egyszerűbb, mint FT-Transformer, de a numerikus-kategorikus interakciót kevésbé ragadja meg (itt pedig az `amount`×`category` interakció kulcsfontosságú) | Közepes-Magas |
| **NODE** (Neural Oblique Decision Ensembles) | Differenciálható "fa-szerű" struktúra, ötvözi a gradient boosting induktív torzítását a végponttól-végpontig tanítással | Kevésbé elterjedt/karbantartott implementációk, nehezebb éles üzemeltetés | Közepes |
| **DeepFM** | Faktorizációs gépek + DNN, kifejezetten sok ritka kategorikus feature-re (CTR-predikcióra) tervezve | A mi feature-terünk nem "sok ezer sparse ID" jellegű (csak 50 merchant, 15 category) — a FM-komponens előnye itt korlátozott | Alacsony-Közepes |

**Ajánlás:** az adatszerkezet (mérsékelt sorszám: 594k; kevés, de heterogén — kategorikus + numerikus — feature: kb. 6 nyers + 15–20 engineered; erős, nemlineáris feature-interakciók a category/merchant/amount háromszögben) alapján az **FT-Transformer** a legjobb DL-jelölt fő architektúraként, egy **könnyű, embedding-alapú MLP+residual (B pont)** pedig production-barát, gyorsabban szervizelhető alternatívaként/fallback-ként. A TabNet és NODE inkább másodkörös kísérletnek javasolt, a DeepFM ezen a feature-téren valószínűleg nem ad hozzáadott értéket.

---

## 6. False Positive csökkentési stratégia

Ez a projekt legfontosabb fejezete, mivel a jelenlegi HGB baseline **7 523 FP-t** termel (2.2/A) — ennek érdemi (30–45%-os) csökkentése a fő üzleti cél.

### 6.1 Precision vs Recall tradeoff — a saját adatokon illusztrálva

A 2.2 fejezet baseline-számai önmagukban megmutatják a tradeoff két szélsőségét:
- **HGB (recall-orientált, 0,5 küszöb):** Recall 95,8%, de Precision csak 18,7% → 7 523 FP.
- **RF (precision-orientált):** Precision 84,1%, de Recall csak 64,1% → 647 FN (elszalasztott csalás).

A cél egy olyan modell/pipeline, amely **az RF alacsony FP-jét közelíti, miközben a HGB Recall-jához közelít** — ez klasszikusan nem érhető el egyetlen küszöbérték hangolásával egyetlen modellen, hanem a lenti technikák kombinációjával.

### 6.2 Threshold-optimalizáció

- A `predict_proba` kimenetét **ne** a triviális 0,5-ös küszöbbel vágd — építs PR-görbét (Precision-Recall curve) a validációs halmazon, és válaszd ki azt a küszöböt, ami a kívánt Recall (pl. ≥90%) mellett minimalizálja az FP-t, vagy explicit költségfüggvényt optimalizálj.
- **Költségalapú küszöb:** definiálj egy üzleti költségfüggvényt, pl. `Cost = C_FP × FP + C_FN × FN`, ahol `C_FN` (elszalasztott csalás átlagos vesztesége — emlékezz, hogy a fraud tranzakciók átlagos összege 530,93, szemben a legit 31,85-tel) jellemzően jóval nagyobb, mint `C_FP` (egy hamis riasztás kivizsgálási/ügyfél-kellemetlenségi költsége). Ez a küszöböt automatikusan a magasabb Recall felé tolja el, de kontrollált módon.
- A küszöböt **kizárólag validációs (nem teszt) adaton** válaszd, hogy a teszt-metrikák torzítatlanok maradjanak.

### 6.3 Cost-sensitive learning

- `class_weight` helyett/mellett **minta-szintű súlyozás**, amely az `amount`-tal arányos (egy 2000-es összegű tévesen elszalasztott csalás rosszabb, mint egy 5-ös). Óvatosan: ez torzíthatja a modellt a nagy összegek felé, validáld külön a kis összegű csalások Recall-ját is.
- Kezdő pozitív/negatív súlyarány: a globális imbalance alapján **~81,6 : 1**, de ez tapasztalatilag gyakran túl agresszív (ld. HGB `class_weight="balanced"` eredménye, 7 523 FP) — finomhangolandó, tesztelj 10:1–40:1 tartományt is.

### 6.4 Focal Loss

`FL(p_t) = -α(1-p_t)^γ log(p_t)`

- Kezdő értékek: **γ = 2,0**, **α = 0,90–0,95** a pozitív (fraud) osztályra (a szokásos α=0,25 irodalmi alapérték itt túl alacsony lenne az 1,21%-os fraud-arányhoz képest).
- A Focal Loss előnye a puszta class-weighting felett: dinamikusan lefókuszál a már jól klasszifikált (könnyű) mintákról, és a nehéz, határeseti mintákra koncentrál — pontosan ezekben van a jelenlegi FP-probléma gyökere.

### 6.5 Weighted BCE Loss

Egyszerűbb alternatíva/baseline a Focal Loss mellé: `pos_weight ≈ 81,6` kiindulásként (`BCEWithLogitsLoss(pos_weight=...)` PyTorch-ban), majd grid-search 20–100 tartományban validációs PR-AUC alapján.

### 6.6 Hard Negative Mining

A reprodukált HGB baseline **7 523 konkrét FP-esete** (2.2/A) kiváló kiindulópont: ezeket a "nehéz negatívokat" (magas score-t kapott, de valójában legit tranzakciók) érdemes külön gyűjteni, és a második tanítási körben felülsúlyozni / kifejezetten ezekre validálni, hogy a végső modell megtanulja megkülönböztetni őket a valódi csalásoktól.

### 6.7 Probability calibration

- **Platt scaling** (logisztikus) vagy **isotonic regresszió** a validációs halmazon, tanítás után.
- Kritikus, mert a `class_weight`/Focal Loss/pos_weight torzítja a nyers kimeneti valószínűségeket — kalibráció nélkül a threshold-optimalizáció (6.2) és a human-review sávozás (6.10) megbízhatatlan lesz.
- Ellenőrzés: reliability diagram (kalibrációs görbe) + Brier score.

### 6.8 Ensemble filtering

Kombináld a meglévő RF-et (alacsony FP, 218) az új DNN-nel (várhatóan magasabb Recall):
- **"AND" logika:** csak akkor riasszon, ha mindkét modell egyetért → FP tovább csökken, de Recall is csökkenhet.
- **Súlyozott átlag / stacking:** egy meta-modell (pl. logisztikus regresszió) tanulja meg optimálisan kombinálni az RF, HGB és DNN score-jait — ez jellemzően jobb Pareto-fronton mozog, mint bármelyik önmagában.

### 6.9 Two-stage fraud detection (kiemelt javaslat)

1. **1. lépés — magas Recall, alacsony küszöb:** a DNN (vagy HGB) alacsony threshold-dal fut, hogy szinte minden valódi csalást elkapjon (Recall ~95-98%), a Precision itt nem prioritás.
2. **2. lépés — precision-fókuszú döntő:** egy második, kisebb, erősen regularizált modell (vagy az RF) *csak* az 1. lépés által gyanúsnak jelölt tranzakciókon fut, és ezekre már sokkal szűkebb, megbízhatóbb halmazon tud precíz döntést hozni — mivel a bemeneti halmaz már erősen dúsított fraud-arányú, a második modell tanítása/kalibrálása is stabilabb.

Ez a felépítés direkt választ ad a probléma-megfogalmazásra: **az 1. lépés tartja a Recall-t, a 2. lépés vágja vissza az FP-t.**

### 6.10 Human-review layer

- Definiálj egy "szürke zónát" a kalibrált valószínűségben (pl. 0,3–0,7): ezek automatikus emberi felülvizsgálatba kerülnek.
- 0,7 fölött: automatikus blokkolás/hold.
- 0,3 alatt: automatikus jóváhagyás.
- Ez nem csökkenti az algoritmikus FP-t, de **üzletileg csökkenti a téves automatikus elutasítások arányát**, mert a legbizonytalanabb eseteket ember dönti el.

### 6.11 Konkrét, számszerű cél

Jelenlegi állapot (HGB, 6.2 nélkül): **FP = 7 523**, Recall = 95,8%.
Jelenlegi állapot (RF): **FP = 218**, Recall = 64,1%.

**Javasolt cél:** a fenti technikák (threshold-optimalizáció + Focal Loss + kalibráció + two-stage) kombinációjával **FP csökkentése 30–45%-kal a HGB-szinthez képest (≈4 100–5 300 FP-re), Recall ≥ 90% megtartása mellett.** Ez egyszerre jelentősen jobb, mint a HGB (kevesebb hamis riasztás), és jelentősen jobb, mint az RF (magasabb Recall) — reális, mindkét baseline-nál Pareto-dominánsabb célsáv.

---

## 7. Tanítási és validációs terv

### 7.1 Train/validation/test stratégia — **time-based split kötelező**

A 2.2/C eredmény (Recall RF: 64%→47%, HGB: 96%→86% time-based split alatt) egyértelműen bizonyítja, hogy a random split túlbecsüli a valós teljesítményt. Javasolt felosztás a `step` (0–179) mentén:

| Szakasz | Step-tartomány | Arány | Szerep |
|---|---|---|---|
| Train | 0–134 | ~75% | Modell tanítás, feature-history számítás bázisa |
| Validation | 135–154 | ~11% | Threshold-optimalizáció, hiperparaméter-hangolás, korai leállás |
| Test | 155–179 | ~14% | Végső, torzítatlan kiértékelés — **csak egyszer** futtatva |

### 7.2 Customer-level leakage elkerülése

Minden history-alapú feature (4. fejezet, #3, #4, #7–11, #15–17) kizárólag **a saját szakaszának korábbi** tranzakcióira támaszkodhat (expanding window `step` szerint). A train-szakasz history-ja **nem** nyúlhat át a validation/test szakaszba (pl. a `customer_fraud_prior` a test-szakaszban is csak a train+val korábbi adataiból, ill. a test-en belül is csak a *korábbi* saját tranzakciókból számolható, sosem a jövőből).

### 7.3 Cross-validation stratégia

- **`TimeSeriesSplit`** (sklearn) vagy **purged/embargo k-fold** a `step` mentén, kiegészítve egy rövid "embargo" ablakkal a fold-határok környékén, hogy a history-feature-ök ne szivárogjanak át a fold-határon.
- Ha customer-szintű aggregációkat is validálni kell, kombináld a time-alapú foldolást egy **GroupKFold**-jellegű ellenőrzéssel is (annak igazolására, hogy egy adott ügyfél train/val közötti szétosztása nem torzít).

### 7.4 Osztály-egyensúly kezelése

- **Elsődleges eszköz:** class-weighting / Focal Loss (6.4–6.5) — nem szintetikus mintagenerálás.
- **SMOTE előnyei:** egyszerű, gyorsan tesztelhető, jól ismert baseline-technika.
- **SMOTE hátrányai *ezen* az adaton:** a feature-tér erősen kategorikus (OHE `category`, `merchant`, `age`, `gender`), a klasszikus SMOTE lineáris interpolációja **nem létező one-hot kombinációkat** hozhat létre (pl. töredékes `category`-vektorokat), ami zajt visz be. Mivel emellett a `category`-nak önmagában 3 szintje 0% fraud-rátájú (1.5), a szintetikus minták könnyen "irreális" tranzakciókat generálhatnak. **Ha mégis használod:** kizárólag **SMOTE-NC** (kategorikus-tudatos változat) train-only, és csakis a train-szakaszon belül, a CV fold-ok belsejében (nem a teljes train-en előre, hogy ne szivárogjon a validációba).
- **Javasolt sorrend:** először tesztelj tisztán class-weighting/Focal Loss-t (SMOTE nélkül), és csak ha ez plafont ér, vizsgáld meg a SMOTE-NC-t kiegészítésként.

### 7.5 Augmentációs lehetőségek

- Kis Gauss-zaj hozzáadása az `amount`-hoz (jitter) — egyszerű, olcsó regularizáció.
- Mixup/CutMix-szerű technikák tabular adatra (numerikus feature-ök interpolációja **azonos osztályon belül**, hogy elkerüljük a SMOTE-nál látott kategorikus problémát).
- VAE/GAN-alapú szintetikus fraud-minta generálás — **csak másodkörös kísérletként**, alapos validációval (a szintetikus minták valós eloszlás-illeszkedésének ellenőrzésével), mert könnyen torzíthat.

---

## 8. Hiperparaméterek — javasolt induló értékek

| Hiperparaméter | Javasolt kezdőérték | Megjegyzés |
|---|---|---|
| Learning rate | 1e-3 (AdamW-hez) | Cosine annealing vagy ReduceLROnPlateau ütemezővel |
| Batch size | 1024–2048 | 594k sor mellett stabil gradiensbecslés; GPU-memória szerint hangolandó |
| Optimizer | AdamW | weight_decay = 1e-4–1e-5 |
| Dropout | 0,2–0,3 | rétegenként csökkenő (0,3 → 0,2 → 0,1 a mélyebb rétegek felé) |
| Rejtett neuronszám (MLP) | 256 → 128 → 64 | A) pontban részletezve |
| Embedding dimenziók | category: 8, merchant: 16, age: 4, gender: 2 | ökölszabály: min(50, nunique/2) körüli tartomány, itt manuálisan finomítva |
| Epochok | max 100 | early stopping mellett ritkán fut ennyit |
| Early stopping | patience = 8–10 epoch, val PR-AUC alapján | ne accuracy vagy loss alapján — az imbalance miatt félrevezető |
| LR scheduler | ReduceLROnPlateau (factor=0,5, patience=4) vagy CosineAnnealingWarmRestarts | |
| Focal Loss γ / α | γ=2,0 / α=0,90–0,95 | ld. 6.4 |
| Gradiens vágás (grad clipping) | max_norm=1,0 | attention/residual blokkoknál stabilizál |
| Batch normalization momentum | 0,1 (default) | nagy batch-nél ritkán kell hangolni |

### Kiértékelési metrikák és célszámok

Az Accuracy **félrevezető** ezen az 1,21%-os fraud-arányú adathalmazon (egy "mindig legit" modell is 98,79%-os accuracy-t érne el) — a kiértékelés kizárólag az alábbi metrikákra épüljön:

| Metrika | Baseline (HGB, random split) | Baseline (RF, random split) | **Cél (új rendszer)** |
|---|---:|---:|---:|
| Precision (fraud) | 18,7% | 84,1% | **≥ 40–50%** |
| Recall (Fraud Recall) | 95,8% | 64,1% | **≥ 90%** |
| F1 | 0,312 | 0,727 | **≥ 0,55–0,65** |
| PR-AUC | 0,801 | 0,762 | **≥ 0,82** |
| ROC-AUC | 0,993 | 0,951 | **≥ 0,99** (kevésbé informatív imbalance mellett, de kontroll) |
| False Positive Rate | 5,12% | 0,15% | **≤ 3,0–3,5%** |
| False Discovery Rate (1−Precision) | 81,3% | 15,9% | **≤ 50–60%** |
| False Positive szám (abszolút, teszthalmazon) | 7 523 | 218 | **≈ 4 100–5 300** |

A célszámokat **time-based teszthalmazon** kell mérni (7.1), nem random stratifikáltan — a 2.2/C eredmények alapján a random split szisztematikusan optimistább képet ad.

---

## 9. Kockázatok

| Kockázat | Leírás | Mitigáció |
|---|---|---|
| **Data leakage a history-feature-ökből** | A customer/merchant fraud-prior és aggregált statisztikák könnyen "jövőt szivárogtatnak", ha nem szigorúan kauzálisan számolják | Expanding-window implementáció, purged/embargo CV (7.3), kódreview checklist |
| **`es_transportation`/`es_food`/`es_contents` "soha nem fraud" szabály túlfittelése** | Ezek a kategóriák a minta 89,6%-át adják és pontosan 0% fraud-rátájúak — ha ez BankSim-szimulációs artifact, éles adaton veszélyes lehet, ha a modell 100%-ban kizárja ezekben a fraud lehetőségét | Ne engedj a modellnek zéró valószínűséget adni semmilyen kategóriára (label smoothing / minimális prior-simítás); külön monitorozd ezekben a kategóriákban is a (ritka) éles fraud-eseteket production után |
| **Temporal drift folytatódása éles üzemben** | A -0,94-es step-fraud korreláció arra utal, hogy a fraud-minták időben változnak; ez éles rendszerben is várható (támadói taktikaváltás) | Rendszeres újratanítás / online tanulás, drift-monitoring (pl. PSI — Population Stability Index — a score-eloszláson), időszakos re-kalibráció |
| **Merchant/category target encoding túlillesztése** | A naiv (in-sample) target encoding korrelációi (0,73 / 0,57) gyanúsan magasak — modell könnyen "memorizálhatja" ahelyett, hogy általánosítana, főleg kis mintaszámú merchanteknél | Out-of-fold encoding, Bayes-simítás (shrinkage a globális átlag felé kis n esetén), monitorozd az új/ritka merchantok teljesítményét külön |
| **Class imbalance instabilitás kis batch-eken** | 81,6:1 arány mellett kis batch-mérettel előfordulhat, hogy egy batch-ben egyetlen pozitív minta sincs | Stratifikált batch-mintavételezés (weighted sampler), vagy elég nagy batch méret |
| **Kalibráció torzulása class-weighting/Focal Loss mellett** | A súlyozott loss torzítja a nyers valószínűségeket, ami megbízhatatlanná teszi a fix threshold-politikát | Post-hoc kalibráció (Platt/isotonic), rendszeres reliability-diagram ellenőrzés |
| **`customer`/`merchant` ID overfitting** | 4 112 ügyfél, 50 merchant — kis n-ű csoportoknál a history-alapú feature-ök zajosak lehetnek | Minimum mintaszám-küszöb a history-feature-ök megbízhatóságához, Bayes-simítás, "cold start" külön kezelése (új ügyfél/merchant esetén globális prior) |
| **Modellkomplexitás vs. üzemeltethetőség** | FT-Transformer/attention-alapú modellek nehezebben magyarázhatók és auditálhatók, mint az RF — bankfelügyeleti/compliance elvárás lehet a magyarázhatóság | SHAP/attention-súly alapú magyarázat réteg beépítése; tartsd életben az egyszerűbb MLP-t interpretálható fallbackként |
| **Szintetikus (BankSim) adat vs. valós termelési adat eltérése** | A teljes elemzés egy szimulált datasetre épül; a valós tranzakciós minták (pl. valódi zipcode-diverzitás, valós velocity-minták) eltérhetnek | Az architektúra és a feature-engineering elveket (kauzalitás, calibration, two-stage) valós adaton is validálni kell újra, a konkrét számokat (FP-célok) nem szabad változtatás nélkül átvinni |

---

## 10. Végső ajánlott architektúra

**Klaszterezés:** KMeans (k=9, a meglévő `clustering.py` szerint) **kiegészítve** egy HDBSCAN zaj-jelzővel (`is_hdbscan_noise`) — mindkettő csak kiegészítő feature-ként, nem elsődleges szegmentálóként (3.4).

**Feature-készlet:** a nyers `step` (ciklikus kódolással), `amount_log`, `is_amount_outlier`, embedding-alapú `category`/`merchant`/`age`/`gender`, plusz a 4. fejezet kauzális history-feature-jei közül elsőként: `amount_zscore_customer`, `amount_ratio_customer`, `customer_fraud_prior` (Bayes-simítással), `merchant_fraud_prior` (out-of-fold, Bayes-simítással), `category_is_high_risk`, `is_new_category_for_customer`, `cluster_id`, `is_hdbscan_noise`.

**Architektúra:** **B) Fejlettebb DNN** (embedding rétegek + 2–3 residual blokk + 1 könnyű self-attention réteg a feature-interakciókhoz), Focal Loss-szal (γ=2, α≈0,92) tanítva, AdamW optimalizálóval, time-based train/val/test split mellett. Ezzel párhuzamosan egy **FT-Transformer** kísérlet fut második jelöltként — amelyik a validációs PR-AUC-n és a kalibrált threshold melletti FP-számban jobbnak bizonyul, az megy tovább.

**Döntési réteg:** **két-lépcsős** felépítés (6.9) — a DNN magas Recall-lal szűr, a meglévő, jól bevált **RandomForest** (alacsony FP-je miatt) szolgál második, precision-fókuszú döntőként a gyanús tranzakciókon; kalibrált valószínűségekkel és human-review sávval (0,3–0,7) a szürke zónában.

**Loss:** Focal Loss (elsődleges), Weighted BCE (összehasonlító baseline).

**Becsült teljesítményjavulás a baseline-hoz képest** (time-based teszthalmazon, a 2.2/C és 6.11 alapján extrapolálva):

| Metrika | HGB baseline (time-split) | Cél (új rendszer) |
|---|---:|---:|
| FP (abszolút) | 3 993 | **≈ 2 400–2 800** (35–40%-os csökkenés) |
| Recall | 86,3% | **≥ 88–90%** (megtartva/enyhén javítva) |
| Precision | 25,7% | **≥ 40–45%** |

*(A random-split baseline (7 523 FP) helyett szándékosan a reálisabb time-split baseline-hoz (3 993 FP) viszonyítunk — ez a valós, éles-üzemi elvárásokhoz közelebb álló referencia.)*

---

## 11. Teljes fejlesztési roadmap

| # | Fázis | Fő tevékenységek | Kimenet |
|---|---|---|---|
| 1 | **Adattisztítás** | `zipcodeOri`/`zipMerchant` eldobása, idézőjel-tisztítás, ritka `age`/`gender` kategóriák összevonása, time-based train/val/test particionálás rögzítése | Tiszta, verziózott dataset + split-definíció |
| 2 | **Feature Engineering** | 4. fejezet feature-jeinek implementálása expanding-window logikával, out-of-fold target encoding pipeline, egységteszt a leakage ellen (pl. "train fold csak múltbeli adatot lát" ellenőrzés) | Feature store / reprodukálható feature-pipeline |
| 3 | **Clustering kísérletek** | KMeans (k=9 megerősítése/újrafuttatása az új split-en), HDBSCAN kísérlet (mintavételezve + GPU-n teljes méretben), zaj-jelző validálása | `cluster_id`, `is_hdbscan_noise` feature-ök |
| 4 | **Baseline reprodukció** | A meglévő RF/HGB pipeline újrafuttatása az új, time-based split-en és a bővített feature-készleten (fair összehasonlítási alap) | Frissített baseline-metrikák |
| 5 | **MLP fejlesztés** | A) egyszerű MLP tanítása Focal Loss-szal, gyors iteráció a feature-hasznosság ellenőrzésére | Első DNN-kísérlet eredményei |
| 6 | **Deep Network fejlesztés** | B) fejlettebb DNN (embedding+residual+attention) és C) FT-Transformer párhuzamos fejlesztése | Két versengő DL-jelölt |
| 7 | **Hyperparameter tuning** | Optuna/Ray Tune alapú keresés (learning rate, dropout, embedding-dimenziók, Focal Loss γ/α) validációs PR-AUC-ra optimalizálva | Hangolt modell-konfiguráció |
| 8 | **Threshold optimalizáció** | PR-görbe és költségfüggvény-alapú küszöbválasztás validáción, kalibráció (Platt/isotonic) | Kalibrált modell + üzleti threshold-politika |
| 9 | **Ensemble kialakítás** | Two-stage (DNN→RF) és/vagy stacking kísérletek, human-review sáv definiálása | Végleges döntési pipeline |
| 10 | **Élesítési terv** | Batch/real-time inferencia terv, monitoring (drift, kalibráció-eltolódás, FP/Recall trend), újratanítási ütemterv, A/B teszt terve a régi pipeline ellen | Termelésre kész rendszer + monitoring dashboard terv |

**Technikai stack:** Python 3.12, CUDA 12.6-kompatibilis PyTorch (elsődleges javaslat a rugalmas custom loss/embedding/attention miatt; TensorFlow/Keras alternatívaként, ha a meglévő csapat abban jártasabb), `scikit-learn` (baseline, kalibráció, metrikák), opcionálisan `pytorch-tabular` vagy `rtdl` könyvtár az FT-Transformer/TabNet gyors prototípusozásához, RAPIDS `cuml` GPU-gyorsított HDBSCAN-hoz nagy mintán.

---

## Záró konkrét ajánlás

A rendelkezésre álló adatok alapján az ajánlott végső architektúra:

- **Klaszterezés:** KMeans (k=9, megtartva a meglévő `clustering.py` logikáját) + HDBSCAN zaj-jelző kiegészítő feature-ként — **nem** elsődleges szegmentálóként, mert a feature-fontosság elemzés (1.9) szerint a category/merchant/amount hármas már önmagában lefedi az információ 99,5%-át.
- **Modell:** **B) Fejlettebb, embedding-alapú Deep Neural Network** reziduális blokkokkal és egy könnyű self-attention réteggel, mint elsődleges production-jelölt; **FT-Transformer** párhuzamos kísérletként, ha az erőforrások engedik.
- **Feature-ök:** a 4. fejezetben felsorolt 20 javaslatból elsődlegesen a **kauzális customer/merchant profil- és arány-feature-ök** (`amount_ratio_customer`, `customer_fraud_prior`, `merchant_fraud_prior`, `is_amount_outlier`, `is_new_category_for_customer`, `category_is_high_risk`) — ezek adják várhatóan a legnagyobb FP-csökkentő hozamot, mert közvetlenül a category/merchant/amount háromszög (1.9) finomítására épülnek.
- **Loss függvény:** **Focal Loss** (γ=2,0, α≈0,90–0,95), Weighted BCE összehasonlító alternatívaként.
- **Döntési réteg:** **két-lépcsős** architektúra (DNN magas Recall-lal → RandomForest mint precision-döntő) + kalibrált threshold + human-review sáv.
- **Becsült teljesítményjavulás:** a reálisabb, time-based teszthalmazon mért HGB-baseline-hoz (FP=3 993, Recall=86,3%) képest **~35–40%-os FP-csökkenés** (≈2 400–2 800 FP-re) várható **Recall ≥ 88–90%** megtartása/enyhe javítása mellett — ez a fő üzleti cél (FP jelentős csökkentése, Recall romlása nélkül) számszerűsített, reális teljesítése.
