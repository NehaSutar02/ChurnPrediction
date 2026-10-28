import json
import sys
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_curve,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import build_preprocessor, load_data


def make_model(X, class_weight=None):
    return Pipeline([
        ("prep", build_preprocessor(X)),
        ("clf", RandomForestClassifier(n_estimators=300, class_weight=class_weight,
                                       random_state=42, n_jobs=-1)),
    ])


def evaluate(model, X, y, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {"accuracy": round(accuracy_score(y, pred), 4),
            "roc_auc": round(roc_auc_score(y, proba), 4),
            "churn_recall": round(recall_score(y, pred), 4)}


def main(industry: str):
    X, y = load_data(industry)
    print(f"Churn rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    baseline = make_model(X).fit(X_fit, y_fit)
    weighted = make_model(X, "balanced").fit(X_fit, y_fit)

    prec, rec, thr = precision_recall_curve(y_val, weighted.predict_proba(X_val)[:, 1])
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    threshold = float(thr[f1[:-1].argmax()])

    print("Baseline          :", evaluate(baseline, X_test, y_test))
    print("Class weighting   :", evaluate(weighted, X_test, y_test))
    print(f"Weight + thr={threshold:.2f}:", evaluate(weighted, X_test, y_test, threshold))

    final = make_model(X, "balanced").fit(X_train, y_train)
    names = final.named_steps["prep"].get_feature_names_out()
    importances = final.named_steps["clf"].feature_importances_
    top = sorted(zip(names, importances), key=lambda t: -t[1])[:10]
    print("\nTop churn drivers:")
    for name, score in top:
        print(f"  {name:45s} {score:.3f}")

    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": final, "threshold": threshold,
                 "top_features": [(n, float(s)) for n, s in top]},
                f"models/{industry}.joblib")
    metrics = evaluate(final, X_test, y_test, threshold)
    Path(f"models/{industry}_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
