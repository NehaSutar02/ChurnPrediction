# 🛡️ ChurnGuard

Multi-industry customer churn prediction with a Random Forest, a reusable preprocessing pipeline, and a Streamlit app that scores customers and suggests retention actions.
Built for the Cognizant Technoverse 2026 hackathon (CMT theme).

**Live demo:** <streamlit app link>

<screenshot of the app here>

## Results (Telecom, real data)
Dataset: IBM Telco Customer Churn (Kaggle), 26.5% churn rate.

| Metric (held-out test set) | Value |
|---|---|
| ROC-AUC | 0.82 |
| Accuracy | 75.7% |
| Churn recall | 69.3% (49.5% with default 0.5 threshold) |

Class weighting alone did not improve recall; tuning the decision threshold on a validation split did, at the cost of some accuracy.

## Industries
- **Telecom:** real data (above).
- **SaaS and OTT:** *synthetic* data from `src/make_synthetic.py`, used to validate that one codebase supports multiple schemas. Their metrics are not real-world performance.

## How it works
- `src/config.py`: one entry per industry (file path, target, id column)
- `src/preprocess.py`: cleaning, industry-specific features, scaling and encoding
- `src/train.py`: trains baseline vs class-weighted models, tunes the threshold, saves the model
- `src/predict.py`: scores new customers and maps top drivers to retention actions
- `app.py`: Streamlit UI

## Run locally
```bash
pip install -r requirements.txt
python -m src.make_synthetic      # SaaS/OTT demo data
python -m src.train telecom       # needs data/raw/telecom.csv from Kaggle
python -m streamlit run app.py
```

## Limitations and next steps
- Single train/test split (no cross-validation yet)
- Recommendations follow global drivers, not per-customer explanations
- SaaS/OTT need real datasets
- Planned: SHAP explanations, cross-validation, LLM-generated retention messages