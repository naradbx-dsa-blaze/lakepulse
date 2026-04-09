"""
LakePulse — Home / Overview
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import plotly.express as px
import streamlit as st

from data import get_billing_trend, get_dbu_waste, get_job_history, DEMO_MODE
from utils import fmt_currency, fmt_dbu, BRAND_ORANGE, BRAND_RED, BRAND_GREEN, PALETTE

st.set_page_config(
    page_title="LakePulse — Databricks FinOps",
    page_icon="⚡",
    layout="wide",
)

# ── Branding ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("# ⚡")
with col_title:
    st.markdown("## LakePulse")
    st.caption("Native Databricks FinOps · Kill DBU waste · Predict SLA breaches · Track ESG")

if DEMO_MODE:
    st.info(
        "📊 **Demo mode** — set `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, and "
        "`DATABRICKS_WAREHOUSE_ID` to connect to your workspace.",
        icon="ℹ️",
    )

st.markdown("---")

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load():
    billing = get_billing_trend()
    waste   = get_dbu_waste()
    jobs    = get_job_history()
    billing["date"]               = pd.to_datetime(billing["date"])
    billing["estimated_cost_usd"] = pd.to_numeric(billing["estimated_cost_usd"], errors="coerce")
    billing["total_dbu"]          = pd.to_numeric(billing["total_dbu"], errors="coerce")
    waste["total_dbu"]            = pd.to_numeric(waste["total_dbu"], errors="coerce")
    waste["estimated_cost_usd"]   = pd.to_numeric(waste["estimated_cost_usd"], errors="coerce")
    jobs["duration_min"]          = pd.to_numeric(jobs["duration_min"], errors="coerce")
    return billing, waste, jobs

with st.spinner("Loading telemetry..."):
    billing, waste, jobs = load()

# ── KPI metrics ────────────────────────────────────────────────────────────────
now = pd.Timestamp.now()
cur  = billing[billing["date"].dt.month == now.month]["estimated_cost_usd"].sum()
prev = billing[billing["date"].dt.month == (now.month - 1 if now.month > 1 else 12)]["estimated_cost_usd"].sum()
total_waste  = waste["estimated_cost_usd"].sum()
total_dbu_90 = billing["total_dbu"].sum()
failed_jobs  = (jobs["result_state"] != "SUCCEEDED").sum()
fail_rate    = failed_jobs / max(len(jobs), 1) * 100
carbon_t     = total_dbu_90 * 0.14 * 0.386 / 1000  # tCO₂

c1, c2, c3, c4 = st.columns(4)
c1.metric("💰 MTD Spend",         fmt_currency(cur),         fmt_currency(cur - prev))
c2.metric("🔥 Recoverable Waste", fmt_currency(total_waste), f"{len(waste)} clusters", delta_color="inverse")
c3.metric("⚠️ Job Failure Rate",  f"{fail_rate:.1f}%",       f"{int(failed_jobs)} failed runs", delta_color="inverse")
c4.metric("🌱 Est. Carbon (90d)", f"{carbon_t:.2f} tCO₂",   "Last 90 days")

st.markdown("---")

# ── Charts row 1 ───────────────────────────────────────────────────────────────
cl, cr = st.columns(2)

with cl:
    st.subheader("Daily Spend by Product (90 days)")
    daily = billing.groupby(["date", "product"])["estimated_cost_usd"].sum().reset_index()
    fig = px.area(
        daily, x="date", y="estimated_cost_usd", color="product",
        labels={"estimated_cost_usd": "Cost (USD)", "date": "", "product": "Product"},
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=300, legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

with cr:
    st.subheader("Spend Mix by Product")
    by_prod = billing.groupby("product")["estimated_cost_usd"].sum().reset_index()
    fig = px.pie(
        by_prod, values="estimated_cost_usd", names="product",
        hole=0.45, color_discrete_sequence=PALETTE,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ── Charts row 2 ───────────────────────────────────────────────────────────────
cl2, cr2 = st.columns(2)

with cl2:
    st.subheader("🔥 Top Wasteful Clusters (30 days)")
    top = (
        waste.nlargest(10, "estimated_cost_usd")
        .assign(label=lambda d: d["cluster_name"].str[:30])
    )
    fig = px.bar(
        top, x="estimated_cost_usd", y="label", orientation="h",
        labels={"estimated_cost_usd": "Estimated Cost (USD)", "label": ""},
        color="estimated_cost_usd",
        color_continuous_scale=["#FFD700", BRAND_ORANGE, BRAND_RED],
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320, coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

with cr2:
    st.subheader("Job Outcomes (30 days)")
    outcome = jobs["result_state"].value_counts().reset_index()
    outcome.columns = ["state", "count"]
    color_map = {"SUCCEEDED": BRAND_GREEN, "FAILED": BRAND_RED, "TIMED_OUT": BRAND_ORANGE}
    fig = px.bar(
        outcome, x="state", y="count", color="state",
        color_discrete_map=color_map,
        labels={"count": "Run Count", "state": ""},
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ── Navigation cards ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Navigate to a module")
n1, n2, n3, n4, n5, n6 = st.columns(6)
cards = [
    ("💰", "DBU Waste Killer",       "Kill idle clusters & runaway jobs"),
    ("🔥", "Bottleneck Detector",    "Shuffle spill & data skew analysis"),
    ("🔮", "SLA Oracle",             "Predict job delays before they happen"),
    ("🗺️", "Data Heatmap",           "Find unused tables & prune assets"),
    ("📊", "What-If Projections",    "Simulate cost under different configs"),
    ("🌱", "ESG Tracking",           "Carbon footprint & sustainability score"),
]
for col, (icon, name, desc) in zip([n1, n2, n3, n4, n5, n6], cards):
    col.markdown(f"**{icon} {name}**")
    col.caption(desc)
