"""
LakePulse — ESG Tracking
Estimates carbon footprint from Databricks DBU consumption and tracks
sustainability trends over time.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import get_esg_metrics, KWH_PER_DBU, KG_CO2_PER_KWH, TREES_ABSORB_KG_CO2_PER_YEAR, DEMO_MODE
from utils import BRAND_ORANGE, BRAND_RED, BRAND_GREEN, BRAND_BLUE, PALETTE, gauge

st.set_page_config(page_title="ESG Tracking · LakePulse", page_icon="🌱", layout="wide")
st.title("🌱 ESG & Sustainability Tracking")
st.caption(
    f"Estimates: 1 DBU ≈ {KWH_PER_DBU} kWh · US grid avg {KG_CO2_PER_KWH} kg CO₂/kWh · "
    "Figures are indicative — consult your cloud provider for certified emissions data."
)

with st.sidebar:
    st.header("Assumptions")
    kwh_per_dbu   = st.slider("kWh per DBU",        0.05, 0.50, KWH_PER_DBU,  0.01)
    kg_co2_per_kwh = st.slider("kg CO₂ per kWh",   0.1,  0.8,  KG_CO2_PER_KWH, 0.01)
    region_label  = st.selectbox("Region Grid",
                                 ["US Average (0.386)", "EU Average (0.276)", "APAC Average (0.512)"])

@st.cache_data(ttl=300)
def load():
    return get_esg_metrics()

with st.spinner("Calculating carbon footprint..."):
    df = load()

# Override with sidebar values
df["kwh"]        = df["total_dbu"] * kwh_per_dbu
df["kg_co2"]     = df["kwh"] * kg_co2_per_kwh
df["trees_equiv"]= df["kg_co2"] / (TREES_ABSORB_KG_CO2_PER_YEAR / 365)

# ── Summary stats ──────────────────────────────────────────────────────────────
total_kwh    = df["kwh"].sum()
total_co2    = df["kg_co2"].sum()
total_trees  = df["trees_equiv"].sum()
total_dbu    = df["total_dbu"].sum()

# Sustainability score: lower carbon intensity = higher score
# Benchmark: 10 DBU/kg CO₂ = 50 score; 5 = 25; 20 = 100
dbu_per_kg    = total_dbu / max(total_co2, 1)
esg_score     = min(100, max(0, dbu_per_kg * 5))

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("⚡ Total Energy",           f"{total_kwh/1000:.1f} MWh",  "90 days")
c2.metric("💨 Total CO₂ Emitted",      f"{total_co2/1000:.2f} tCO₂", "90 days")
c3.metric("🌳 Trees to Offset",        f"{int(total_trees):,}",       "trees × 1 year")
c4.metric("🌍 Carbon Intensity",       f"{total_co2/max(total_dbu,1)*1000:.1f} g CO₂/DBU")

st.markdown("---")

# ── Gauge + trend ──────────────────────────────────────────────────────────────
cl, cc, cr = st.columns([2, 2, 3])

with cl:
    st.subheader("Sustainability Score")
    st.caption("Higher = more DBU per kg CO₂ (better efficiency)")
    fig = gauge(esg_score, "ESG Score", max_val=100, threshold_warn=40, threshold_crit=60)
    st.plotly_chart(fig, use_container_width=True)

with cc:
    st.subheader("Carbon Budget")
    # Simple budget: assume 5 tCO₂/month target
    monthly_budget_kg = 5000
    days_so_far = 90
    daily_avg   = total_co2 / days_so_far
    month_proj  = daily_avg * 30
    pct_of_budget = month_proj / monthly_budget_kg * 100

    st.metric("Monthly CO₂ Target", f"{monthly_budget_kg/1000:.0f} tCO₂")
    st.metric("Projected This Month", f"{month_proj/1000:.2f} tCO₂",
              f"{pct_of_budget - 100:.1f}% vs target",
              delta_color="inverse" if pct_of_budget > 100 else "normal")
    if pct_of_budget > 100:
        st.error(f"⚠️ On track to exceed budget by {pct_of_budget - 100:.0f}%")
    else:
        st.success(f"✅ On track — {100 - pct_of_budget:.0f}% below target")

with cr:
    st.subheader("CO₂ by Product (90 days)")
    by_prod = df.groupby("product")["kg_co2"].sum().reset_index()
    fig = px.pie(
        by_prod, values="kg_co2", names="product",
        hole=0.4,
        color_discrete_sequence=["#2DC653", "#457B9D", "#E9C46A", "#FF6B35"],
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Time series ────────────────────────────────────────────────────────────────
cl2, cr2 = st.columns(2)

with cl2:
    st.subheader("Daily CO₂ Emissions Trend")
    daily = df.groupby("date").agg(kg_co2=("kg_co2", "sum"), total_dbu=("total_dbu", "sum")).reset_index()
    daily["7d_avg"] = daily["kg_co2"].rolling(7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily["date"], y=daily["kg_co2"],
                         name="Daily CO₂", marker_color=BRAND_BLUE, opacity=0.6))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["7d_avg"],
                             name="7-day Avg", line=dict(color=BRAND_ORANGE, width=2)))
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                      yaxis_title="kg CO₂", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

with cr2:
    st.subheader("Energy Efficiency Over Time")
    daily["g_co2_per_dbu"] = (daily["kg_co2"] / daily["total_dbu"].replace(0, 1)) * 1000
    daily["30d_avg"] = daily["g_co2_per_dbu"].rolling(30, min_periods=1).mean()
    fig = px.line(
        daily, x="date", y="g_co2_per_dbu",
        labels={"g_co2_per_dbu": "g CO₂ / DBU", "date": ""},
        color_discrete_sequence=[BRAND_GREEN],
    )
    fig.add_scatter(x=daily["date"], y=daily["30d_avg"],
                    mode="lines", name="30-day avg",
                    line=dict(color=BRAND_ORANGE, width=2, dash="dot"))
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ── Recommendations ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("♻️ Green Computing Recommendations")

recs = [
    ("Use Spot/Preemptible instances",
     "Reduces energy consumption by optimising instance packing in the data center.",
     "~15-25% lower energy per workload"),
    ("Enable Photon acceleration",
     "Photon completes the same work faster, reducing cluster-hours and energy.",
     "Up to 2× faster = up to 50% less energy"),
    ("Schedule jobs in low-carbon regions",
     "Route workloads to regions with higher renewable energy mix (e.g., us-west-2 on AWS).",
     "Grid intensity: us-west-2 ≈ 0.18 vs us-east-1 ≈ 0.39 kg CO₂/kWh"),
    ("Right-size clusters with autoscale",
     "Idle capacity wastes energy. Tight autoscale bounds cut unnecessary node-hours.",
     "Typical saving: 20-40% energy"),
    ("Use Delta OPTIMIZE and ZORDER",
     "Fewer data scans = shorter queries = less compute = lower carbon.",
     "Reduce scan time by 30-70% on large tables"),
]

for i, (title, desc, saving) in enumerate(recs):
    with st.container(border=True):
        rc1, rc2 = st.columns([5, 2])
        rc1.markdown(f"**🌿 {title}**")
        rc1.caption(desc)
        rc2.info(saving)
