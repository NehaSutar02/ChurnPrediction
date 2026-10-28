import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.config import INDUSTRIES
from src.predict import recommendations, score

st.set_page_config(page_title="ChurnGuard", page_icon="🛡️", layout="wide")

LABELS = {"telecom": "Telecom", "saas": "SaaS", "ott": "OTT"}
RISK_ORDER = ["Low", "Medium", "High"]
RISK_COLORS = alt.Scale(domain=RISK_ORDER, range=["#2ecc71", "#f1c40f", "#e74c3c"])

st.markdown(
    """
    <style>
    .hero {
        padding: 1.6rem 2rem; border-radius: 16px; margin-bottom: 1.2rem;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
    }
    .hero h1 { margin: 0; font-size: 2.2rem; color: #fff; }
    .hero p { margin: .3rem 0 0; font-size: 1.05rem; color: #fff; opacity: .92; }
    [data-testid="stMetric"] {
        background: rgba(128,128,128,.10); border: 1px solid rgba(128,128,128,.25);
        padding: 14px 18px; border-radius: 14px;
    }
    </style>
    <div class="hero">
        <h1>🛡️ ChurnGuard</h1>
        <p>Spot the customers about to leave, and know what to do about it.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def pretty(name: str) -> str:
    return name.split("__", 1)[-1].replace("_", " ")


# ---------------- Sidebar ----------------
# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    industry = st.selectbox("Industry", list(INDUSTRIES), format_func=lambda k: LABELS.get(k, k))
    cfg = INDUSTRIES[industry]
    if cfg.get("synthetic"):
        st.warning("Demo data: this model is trained on synthetic data, not real customers.")

    st.divider()
    st.subheader("Model performance")
    metrics_path = Path(f"models/{industry}_metrics.json")
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        st.metric("ROC-AUC", f"{m['roc_auc']:.2f}")
        st.metric("Churn recall", f"{m['churn_recall']:.1%}")
        st.metric("Accuracy", f"{m['accuracy']:.1%}")
        st.caption("Measured on a held-out test set.")
    else:
        st.warning("No trained model for this industry yet.")

    st.divider()
    st.subheader("Model performance")
    metrics_path = Path(f"models/{industry}_metrics.json")
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        st.metric("ROC-AUC", f"{m['roc_auc']:.2f}")
        st.metric("Churn recall", f"{m['churn_recall']:.1%}")
        st.metric("Accuracy", f"{m['accuracy']:.1%}")
        st.caption("Measured on a held-out test set.")
    else:
        st.warning("No trained model for this industry yet.")

# ---------------- Data input ----------------
file = st.file_uploader("Upload customer data (CSV)", type="csv")
use_sample = st.checkbox(f"Or use the sample {LABELS.get(industry, industry)} dataset (demo)")

df = None
if file:
    df = pd.read_csv(file)
elif use_sample:
    sample_path = Path(f"data/sample/{industry}_sample.csv")
    if sample_path.exists():
        df = pd.read_csv(sample_path)
    else:
        st.error(f"Sample file not found: {sample_path}")

if df is None:
    st.info("Upload a CSV file or tick the sample dataset to see churn risk scores.")
    st.stop()

# ---------------- Scoring ----------------
try:
    with st.spinner("Scoring customers..."):
        result, top = score(df, industry)
except (FileNotFoundError, ValueError) as e:
    st.error(str(e))
    st.stop()

details = pd.concat(
    [result, df.drop(columns=[c for c in result.columns if c in df.columns] + [cfg["target"]],
                     errors="ignore")],
    axis=1,
)

n = len(result)
high = int((result["risk_level"] == "High").sum())
medium = int((result["risk_level"] == "Medium").sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers", f"{n:,}")
k2.metric("High risk", f"{high:,}", f"{high / n:.1%} of customers", delta_color="off")
k3.metric("Avg churn probability", f"{result['churn_probability'].mean():.1%}")
if "MonthlyCharges" in df.columns:
    charges = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    k4.metric("Monthly revenue at risk", f"${charges[result['risk_level'] == 'High'].sum():,.0f}")
else:
    k4.metric("Medium risk", f"{medium:,}")

st.caption("Risk levels: Low < 40%, Medium 40–70%, High ≥ 70% churn probability.")

tab1, tab2, tab3 = st.tabs(["📊 Overview", "👥 Customers", "🎯 Drivers & Actions"])

# ---------------- Tab 1: Overview ----------------
with tab1:
    left, right = st.columns(2)

    counts = result["risk_level"].value_counts().reindex(RISK_ORDER).fillna(0).reset_index()
    counts.columns = ["risk", "count"]
    donut = (
        alt.Chart(counts)
        .mark_arc(innerRadius=70)
        .encode(
            theta="count:Q",
            color=alt.Color("risk:N", scale=RISK_COLORS, legend=alt.Legend(title="Risk")),
            tooltip=["risk", "count"],
        )
        .properties(title="Customers by risk level", height=320)
    )
    left.altair_chart(donut, use_container_width=True)

    hist = (
        alt.Chart(result[["churn_probability"]])
        .mark_bar(color="#7c3aed")
        .encode(
            x=alt.X("churn_probability:Q", bin=alt.Bin(maxbins=20), title="Churn probability"),
            y=alt.Y("count()", title="Customers"),
        )
        .properties(title="Distribution of churn probability", height=320)
    )
    right.altair_chart(hist, use_container_width=True)

# ---------------- Tab 2: Customers ----------------
with tab2:
    levels = st.multiselect("Show risk levels", RISK_ORDER, default=RISK_ORDER)
    view = details[details["risk_level"].isin(levels)].sort_values(
        "churn_probability", ascending=False)

    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "churn_probability": st.column_config.ProgressColumn(
                "Churn probability", min_value=0.0, max_value=1.0, format="%.2f"),
        },
    )
    st.download_button(
        "⬇️ Download scored customers (CSV)",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name=f"churnguard_{industry}_scores.csv",
        mime="text/csv",
    )

# ---------------- Tab 3: Drivers & Actions ----------------
with tab3:
    drivers = pd.DataFrame(top, columns=["feature", "importance"])
    drivers["feature"] = drivers["feature"].map(pretty)
    chart = (
        alt.Chart(drivers)
        .mark_bar(color="#4f46e5")
        .encode(
            x=alt.X("importance:Q", title="Importance"),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")],
        )
        .properties(title="What drives churn the most", height=380)
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Retention recommendations")
    tips = recommendations(top)
    if tips:
        for tip in tips:
            st.info(f"💡 {tip}")
    else:
        st.write("No recommendations matched the top drivers.")
    st.caption("Recommendations follow the model's overall top drivers, not each individual customer.")

st.divider()
st.caption("ChurnGuard · Random Forest churn model · built for the Cognizant Technoverse 2026 hackathon")