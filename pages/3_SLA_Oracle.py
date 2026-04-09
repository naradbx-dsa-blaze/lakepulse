"""
LakePulse — SLA Oracle
Trains a RandomForest classifier on historical job run data to predict
which upcoming jobs are at risk of breaching their SLA.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from data import get_job_history, DEMO_MODE
from utils import sla_risk_badge, BRAND_ORANGE, BRAND_RED, BRAND_GREEN, PALETTE

st.set_page_config(page_title="SLA Oracle · LakePulse", page_icon="🔮", layout="wide")
st.title("🔮 SLA Oracle")
st.caption("ML-powered job delay prediction · Trained on historical run durations")

@st.cache_data(ttl=300)
def load_and_train():
    df = get_job_history()
    df["duration_min"]  = pd.to_numeric(df["duration_min"], errors="coerce")
    df["queue_min"]     = pd.to_numeric(df["queue_min"],    errors="coerce")
    df["trigger_time"]  = pd.to_datetime(df["trigger_time"], errors="coerce")
    df = df.dropna(subset=["duration_min", "trigger_time"])

    # Per-job p75 threshold → "is_late"
    p75 = df.groupby("job_name")["duration_min"].quantile(0.75).rename("p75")
    df  = df.join(p75, on="job_name")
    df["is_late"] = (df["duration_min"] > df["p75"]).astype(int)

    # Features
    df["hour_of_day"] = df["trigger_time"].dt.hour
    df["day_of_week"] = df["trigger_time"].dt.dayofweek
    df["rolling_avg"] = df.groupby("job_name")["duration_min"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    le = LabelEncoder()
    df["job_enc"] = le.fit_transform(df["job_name"].fillna("unknown"))

    feats = ["job_enc", "hour_of_day", "day_of_week", "queue_min", "rolling_avg"]
    X = df[feats].fillna(0)
    y = df["is_late"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    clf.fit(X_tr, y_tr)

    report = classification_report(y_te, clf.predict(X_te), output_dict=True)
    accuracy = report["accuracy"]

    # Predict on all rows
    df["delay_prob"] = clf.predict_proba(X)[:, 1]
    df["risk"]       = df["delay_prob"].apply(sla_risk_badge)
    df["p75_label"]  = df["p75"].apply(lambda v: f"{v:.0f} min SLA")

    importances = pd.Series(clf.feature_importances_, index=feats).sort_values(ascending=False)
    return df, accuracy, importances, le

with st.spinner("Training SLA Oracle model..."):
    df, accuracy, importances, le = load_and_train()

# ── KPIs ──────────────────────────────────────────────────────────────────────
high_risk   = (df["delay_prob"] >= 0.75).sum()
medium_risk = ((df["delay_prob"] >= 0.45) & (df["delay_prob"] < 0.75)).sum()
on_track    = (df["delay_prob"] < 0.45).sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("🎯 Model Accuracy",  f"{accuracy:.1%}")
c2.metric("🔴 High Risk Jobs",  str(int(high_risk)),   delta_color="inverse")
c3.metric("🟠 Medium Risk",     str(int(medium_risk)), delta_color="inverse")
c4.metric("🟢 On Track",        str(int(on_track)))

st.markdown("---")

# ── Charts ────────────────────────────────────────────────────────────────────
cl, cr = st.columns(2)

with cl:
    st.subheader("Delay Probability Distribution")
    fig = px.histogram(
        df, x="delay_prob", nbins=20, color_discrete_sequence=[BRAND_ORANGE],
        labels={"delay_prob": "Delay Probability", "count": "Job Runs"},
    )
    fig.add_vline(x=0.45, line_dash="dot", line_color=BRAND_ORANGE, annotation_text="Medium risk threshold")
    fig.add_vline(x=0.75, line_dash="dot", line_color=BRAND_RED,    annotation_text="High risk threshold")
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with cr:
    st.subheader("Feature Importance")
    feat_labels = {
        "job_enc":     "Job Identity",
        "rolling_avg": "Recent Avg Duration",
        "queue_min":   "Queue Time",
        "hour_of_day": "Hour of Day",
        "day_of_week": "Day of Week",
    }
    imp_df = importances.reset_index()
    imp_df.columns = ["feature", "importance"]
    imp_df["feature"] = imp_df["feature"].map(feat_labels)
    fig = px.bar(
        imp_df, x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale=[BRAND_GREEN, BRAND_ORANGE],
        labels={"importance": "Importance", "feature": ""},
    )
    fig.update_layout(height=320, coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

# ── Risk by job ────────────────────────────────────────────────────────────────
st.subheader("Risk Profile by Job")
by_job = (
    df.groupby("job_name")
    .agg(
        avg_delay_prob=("delay_prob", "mean"),
        avg_duration=("duration_min", "mean"),
        run_count=("run_id", "count"),
        fail_count=("result_state", lambda x: (x != "SUCCEEDED").sum()),
    )
    .reset_index()
    .sort_values("avg_delay_prob", ascending=False)
)
by_job["fail_rate"] = (by_job["fail_count"] / by_job["run_count"] * 100).round(1)

fig = px.scatter(
    by_job,
    x="avg_duration", y="avg_delay_prob",
    size="run_count", color="fail_rate",
    hover_data=["job_name", "run_count"],
    text="job_name",
    color_continuous_scale=[BRAND_GREEN, BRAND_ORANGE, BRAND_RED],
    labels={
        "avg_duration":    "Avg Duration (min)",
        "avg_delay_prob":  "Avg Delay Probability",
        "fail_rate":       "Fail Rate (%)",
    },
)
fig.add_hline(y=0.75, line_dash="dot", line_color=BRAND_RED)
fig.add_hline(y=0.45, line_dash="dot", line_color=BRAND_ORANGE)
fig.update_traces(textposition="top center")
fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)

# ── At-risk job table ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("⚠️ Runs at Risk of SLA Breach")

at_risk = df[df["delay_prob"] >= 0.45].sort_values("delay_prob", ascending=False).head(20)
st.dataframe(
    at_risk[[
        "job_name", "creator_user_name", "trigger_time",
        "duration_min", "p75_label", "delay_prob", "risk",
    ]].rename(columns={
        "job_name":           "Job",
        "creator_user_name":  "Owner",
        "trigger_time":       "Last Triggered",
        "duration_min":       "Duration (min)",
        "p75_label":          "SLA (p75)",
        "delay_prob":         "Delay Probability",
        "risk":               "Risk",
    }).style.format({"Duration (min)": "{:.1f}", "Delay Probability": "{:.1%}"}),
    use_container_width=True,
    hide_index=True,
    height=380,
)
