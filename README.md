# Toroidal transformer EOL failure prediction (RuleFit)

IT5514 Applied Machine Learning assignment. A sparse rule ensemble (RuleFit, 39 IF-THEN rules) predicts
whether a toroidal transformer will fail its first end-of-line (EOL) electrical test, using only data
recorded before the test. A Streamlit app shows each prediction with the rules that fired and an exact
contribution waterfall.

**Live app:** https://toroid-eol-rulefit.streamlit.app/ (Streamlit Community Cloud; if it shows "This app has gone to sleep", click the wake-up button and wait about a minute)

## Repository contents

| File | Purpose |
|---|---|
| `streamlit_app.py`, `app_pages/`, `app_common.py`, `.streamlit/` | Streamlit front-end |
| `rulefit.py` | RuleFit classifier (Friedman & Popescu 2008), own implementation |
| `toroid_preprocess.py` | cleaning, leakage control, feature engineering (`FeatureBuilder`) |
| `toroid_eol_rulefit.ipynb` | EDA, model selection, evaluation vs LR / RF / CatBoost, explainability |
| `models/` | saved RuleFit model, feature-builder reference values, metadata, rule list |
| `figures/` | figures exported by the notebook |
| `toroid_test_data.csv` | raw MES / test-station export, one row per EOL test attempt (13,007 rows × 74 columns) |
| `demo_export_week34.csv` | small raw export (17–23 Aug 2026) for the batch-upload demo |
| `requirements.txt` | app dependencies (pinned) |
| `requirements-notebook.txt` | extra dependencies for re-running the notebook |

## Run the app locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app has three pages:

- **Unit risk check:** pick a recorded unit from the unseen test period, or start from a product template, then edit the process values. You see:
  - the predicted failure risk and the recommended action (standard flow / in-process check / hold for review);
  - an exact contribution waterfall;
  - the rules that fire, the safe windows met and the safe windows missed.
- **Batch scoring:** upload a raw MES / test-station export (e.g. `demo_export_week34.csv`) or use the Jul–Aug 2026 test period. You get:
  - risk KPIs, the risk distribution, and predicted vs observed risk per machine;
  - a ranked table you can download.
- **Model & rules:** test metrics compared with the course models, global importance, and the complete rule list.

## Re-run the notebook

```bash
pip install -r requirements-notebook.txt
python -m nbconvert --to notebook --execute --inplace toroid_eol_rulefit.ipynb
```

The notebook retrains the model and rewrites `models/` and `figures/` (about 8 minutes).

## Results (test period Jul–Aug 2026, base failure rate 6.9 %)

| Model | ROC-AUC | PR-AUC | Brier | Size |
|---|---|---|---|---|
| **RuleFit** | **0.787** | 0.291 | **0.058** | 39 rules |
| CatBoost | 0.781 | **0.304** | 0.084 | 215 trees |
| Random Forest | 0.769 | 0.263 | 0.075 | 600 trees |
| Logistic Regression | 0.753 | 0.265 | 0.253 | 206 coefficients |
