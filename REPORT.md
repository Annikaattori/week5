# Week 5 raportti: Student Habits vs Academic Performance

## 1) Datasetin valinta
Valittu datasetti on Kagglesta: `jayaantanaath/student-habits-vs-academic-performance`.

Perustelut minimivaatimuksiin:
- Datasetti sisältää yli 200 riviä.
- Mukana on useita selittäviä muuttujia (opiskelutottumuksia, taustatekijöitä, ym.).
- Mukana on numeerinen target regressiolle (`exam_score`).
- Luokittelulle käytetään `performance_level`-saraketta, tai jos sitä ei löydy, luodaan luokat `Low/Medium/High` binaamalla `exam_score` kvantiilien mukaan.

## 2) Kaksi ennustetehtävää

### A) Regressio – ennustetaan numero
- **Target:** `exam_score`
- **Features:** kaikki muut sarakkeet (pipeline käsittelee numeeriset ja kategoriset automaattisesti).
- **Miksi hyödyllinen:** ennuste auttaa tunnistamaan opiskelijat, jotka hyötyvät varhaisesta tuesta.
- **Baseline-odotus:** parempi kuin "aina keskiarvo"; käytännössä hyvä taso on noin 10–20 % RMSE-parannus.

### B) Luokittelu – ennustetaan luokka
- **Target:** `performance_level` (tai binaamalla luotu luokka)
- **Features:** kaikki muut sarakkeet
- **Miksi hyödyllinen:** auttaa kohdistamaan toimenpiteitä riskiryhmiin.
- **Baseline-odotus:** parempi kuin "aina enemmistöluokka".
- Jos data on epätasapainoinen, painotetaan F1/recall-mittareita accuracy-mittarin lisäksi.

## 3) Datan valmistelu
- Puuttuvat arvot:
  - numeeriset: mediaani
  - kategoriset: moodi
- Kategoriset muuttujat muunnetaan numeerisiksi one-hot-enkoodauksella.
- Skaalaus:
  - käytössä logistisessa regressiossa (`StandardScaler`)
  - ei pakollinen puupohjaisille malleille
- Train-test split: 80/20
- Cross-validation: 5-fold regressiolle (Linear Regression)

## 4) Regressio: baseline + 2 mallia
- **Baseline:** `DummyRegressor(strategy="mean")`
- **Mallit:**
  1. Linear Regression
  2. Random Forest Regressor (`n_estimators=300`, `max_depth=12`)
- **Metriikat:** MAE, RMSE, R²
- Sovellus näyttää myös `Predicted vs Actual` -kuvan.

## 5) Luokittelu: baseline + 2 mallia
- **Baseline:** `DummyClassifier(strategy="most_frequent")`
- **Mallit:**
  1. Logistic Regression (`max_iter=2000`)
  2. Random Forest Classifier (`n_estimators=300`, `max_depth=12`)
- **Metriikat:** Accuracy, Precision, Recall, F1
- Lisäksi näytetään confusion matrix.

## 6) Visualisoinnit
App sisältää vähintään seuraavat kuvat:
- histogrammi (EDA)
- regressio: Predicted vs Actual
- luokittelu: confusion matrix heatmap

## 7) Tulkinta & keskustelu
Appissa raportoidaan:
- paras regressiomalli vs baseline
- paras luokittelumalli vs baseline
- luokkajakauma (epätasapainon arvio)
- cross-validation-yhteenveto regressiolle

Jatkokehitys:
- hyperparametrien viritys (GridSearch / RandomizedSearch)
- feature engineering
- mahdollinen mallikalibrointi luokittelussa

## 8) Streamlit-app VPS:lle
Appin rakenne seuraa tehtävänannon osiota:
- A) Data & EDA
- B) Modeling (regressio + luokittelu)
- C) Visualizations
- D) Documentation

Käynnistys:
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

VPS-polku voidaan asettaa reverse proxyn kautta URL-osoitteeseen `http://IP/week5`.
