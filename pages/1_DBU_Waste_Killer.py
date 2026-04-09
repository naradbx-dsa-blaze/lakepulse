"""
LakePulse — DBU Waste Killer
Surfaces idle ALL_PURPOSE clusters, estimates recoverable savings, and recommends
auto-termination policies.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.express as px
import streamlit as st

from data import get_dbu_waste, DEMO_MODE
from utils import fmt_currency, fmt_dbu, waste_severity, BRAND_ORANGE, BRAND_RED, BRAND_GREEN, PALETTE

st.set_page_config(page_title="DBU Waste Killer · LakePulse", page_icon="💰", layout="wide")
st.title("💰 DBU Waste Killer")
st.caption("Identifies ALL_PURPOSE clusters burning DBUs with low utilisation · Last 30 days")

# ── Filters ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    min_cost = st.slider("Min Estimated Cost (USD)", 0, 500, 20, 10)
    top_n    = st.slider("Show Top N Clusters", 5, 25, 15)

@st.cache_data(ttl=300)
def load():
    df = get_dbu_waste()
    df["total_dbu"]          = pd.to_numeric(df["total_dbu"],          errors="coerce")
    df["estimated_cost_usd"] = pd.to_numeric(df["estimated_cost_usd"], errors="coerce")
    df["lifetime_hours"]     = pd.to_numeric(df["lifetime_hours"],     errors="coerce")
    return df

with st.spinner("Scanning system tables..."):
    df = load()

df_filtered = df[df["estimated_cost_usd"] >= min_cost].nlargest(top_n, "estimated_cost_usd")

# ── KPIs ──────────────────────────────────────────────────────────────────────
total_waste   = df["estimated_cost_usd"].sum()
critical      = (df["estimated_cost_usd"] >= 500).sum()
high          = ((df["estimated_cost_usd"] >= 100) & (df["estimated_cost_usd"] < 500)).sum()
total_dbu     = df["total_dbu"].sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("💸 Total Recoverable",   fmt_currency(total_waste))
c2.metric("🔴 Critical Clusters",   str(int(critical)),  delta_color="inverse")
c3.metric("🟠 High-Cost Clusters",  str(int(high)),      delta_color="inverse")
c4.metric("📦 Total DBU Wasted",    fmt_dbu(total_dbu))

st.markdown("---")

# ── Charts ────────────────────────────────────────────────────────────────────
cl, cr = st.columns(2)

with cl:
    st.subheader("Cost by Cluster")
    fig = px.bar(
        df_filtered.sort_values("estimated_cost_usd"),
        x="estimated_cost_usd",
        y=df_filtered.sort_values("estimated_cost_usd")["cluster_name"].str[:35],
        orientation="h",
        color="estimated_cost_usd",
        color_continuous_scale=["#FFD700", BRAND_ORANGE, BRAND_RED],
        labels={"estimated_cost_usd": "USD", "y": ""},
    )
    fig.update_layout(height=380, coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with cr:
    st.subheader("DBU vs Lifetime Hours")
    fig = px.scatter(
        df_filtered,
        x="lifetime_hours",
        y="total_dbu",
        size="estimated_cost_usd",
        color="owner",
        hover_data=["cluster_name", "sku_name"],
        labels={"lifetime_hours": "Cluster Lifetime (hrs)", "total_dbu": "DBU Consumed"},
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ── Kill list table ────────────────────────────────────────────────────────────
st.subheader("🗡️ Recommended Kill List")

display = df_filtered.copy()
display["severity"]     = display["estimated_cost_usd"].apply(waste_severity)
display["recommendation"] = display.apply(
    lambda r: "Terminate immediately"
    if r["estimated_cost_usd"] >= 500
    else ("Set auto-termination ≤ 30 min" if r["estimated_cost_usd"] >= 100 else "Review utilisation"),
    axis=1,
)
display["savings_30d"] = display["estimated_cost_usd"].apply(fmt_currency)

st.dataframe(
    display[[
        "cluster_name", "owner", "sku_name",
        "total_dbu", "lifetime_hours", "estimated_cost_usd",
        "severity", "recommendation",
    ]].rename(columns={
        "cluster_name":       "Cluster",
        "owner":              "Owner",
        "sku_name":           "SKU",
        "total_dbu":          "DBU",
        "lifetime_hours":     "Lifetime (hrs)",
        "estimated_cost_usd": "Cost (USD)",
        "severity":           "Severity",
        "recommendation":     "Action",
    }).style.format({"DBU": "{:.1f}", "Cost (USD)": "${:.2f}", "Lifetime (hrs)": "{:.0f}"}),
    use_container_width=True,
    hide_index=True,
    height=420,
)

total_savings = df_filtered["estimated_cost_usd"].sum()
st.success(f"✅ Applying all recommendations could save **{fmt_currency(total_savings)}** in the next 30 days.")

# ── Owner breakdown ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Waste by Owner")
by_owner = df.groupby("owner")["estimated_cost_usd"].sum().reset_index().sort_values("estimated_cost_usd", ascending=False)
fig = px.bar(by_owner, x="owner", y="estimated_cost_usd",
             color="estimated_cost_usd",
             color_continuous_scale=[BRAND_GREEN, BRAND_ORANGE, BRAND_RED],
             labels={"estimated_cost_usd": "Cost (USD)", "owner": "Owner"})
fig.update_layout(height=300, coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)
