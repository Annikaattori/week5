# week5

Streamlit-sovellus, joka toteuttaa kaksi supervised learning -tehtävää samasta datasetistä:
- regressio (`exam_score`)
- luokittelu (`performance_level` tai johdettu luokka)

## Käyttö

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## Sisältö
- `app.py` – varsinainen Streamlit-app
- `REPORT.md` – tehtävänannon mukainen raporttirunko
- `requirements.txt` – riippuvuudet
