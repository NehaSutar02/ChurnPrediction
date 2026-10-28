import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import INDUSTRIES

POSITIVE_LABELS = {"yes", "1", "true", "churn", "churned"}


def clean(df: pd.DataFrame, industry: str) -> pd.DataFrame:
    cfg = INDUSTRIES[industry]
    df = df.copy()
    df = df.drop(columns=[c for c in (cfg.get("id"), cfg["target"]) if c in df.columns])

    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() > 0.9:
            df[col] = converted
    return add_features(df, industry)


def add_features(df: pd.DataFrame, industry: str) -> pd.DataFrame:
    if industry == "telecom" and {"TotalCharges", "tenure"} <= set(df.columns):
        df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
    if industry == "saas" and {"MonthlyCharges", "seats"} <= set(df.columns):
        df["price_per_seat"] = df["MonthlyCharges"] / df["seats"].clip(lower=1)
    if industry == "ott" and {"watch_hours_30d", "devices_count"} <= set(df.columns):
        df["hours_per_device"] = df["watch_hours_30d"] / df["devices_count"].clip(lower=1)
    return df

def load_data(industry: str):
    cfg = INDUSTRIES[industry]
    raw = pd.read_csv(cfg["path"])
    y = raw[cfg["target"]].astype(str).str.strip().str.lower().isin(POSITIVE_LABELS).astype(int)
    X = clean(raw, industry)
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])