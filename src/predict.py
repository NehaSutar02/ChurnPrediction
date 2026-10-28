import joblib
import pandas as pd

from src.config import INDUSTRIES
from src.preprocess import clean

TIPS = {
    "contract": "Offer an incentive to move to a longer-term plan.",
    "tenure": "Run an early-life onboarding / loyalty campaign for newer customers.",
    "charges": "Review pricing, offer a better-fit plan or a targeted discount.",
    "spend": "Review pricing, offer a better-fit plan or a targeted discount.",
    "support": "Proactively reach out to customers with support issues.",
    "usage": "Trigger a re-engagement campaign for customers whose usage is dropping.",
    "adoption": "Run product-adoption nudges and training for accounts using few features.",
    "login": "Trigger a re-engagement campaign for accounts that stopped logging in.",
    "watch": "Send personalised content recommendations to viewers whose watch time is falling.",
    "hours": "Send personalised content recommendations to viewers whose watch time is falling.",
    "inactive": "Trigger a win-back message for customers who have gone quiet.",
    "payment": "Fix failed payments early: retry reminders and alternative payment methods.",
    "nps": "Follow up personally with low-NPS accounts before renewal.",

}


def score(df: pd.DataFrame, industry: str):
    art = joblib.load(f"models/{industry}.joblib")
    model = art["model"]

    X = clean(df, industry)
    missing = [c for c in model.feature_names_in_ if c not in X.columns]
    if missing:
        raise ValueError(f"Uploaded file is missing columns: {missing}")
    X = X[model.feature_names_in_]

    out = pd.DataFrame(index=df.index)
    id_col = INDUSTRIES[industry].get("id")
    if id_col in df.columns:
        out[id_col] = df[id_col]
    out["churn_probability"] = model.predict_proba(X)[:, 1].round(3)
    out["risk_level"] = pd.cut(out["churn_probability"], [-0.01, 0.4, 0.7, 1.0],
                               labels=["Low", "Medium", "High"])
    return out, art["top_features"]


def recommendations(top_features):
    tips = []
    for name, _ in top_features:
        for key, tip in TIPS.items():
            if key in name.lower() and tip not in tips:
                tips.append(tip)
    return tips