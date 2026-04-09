"""
LakePulse — What-If Projections
Interactive cost simulation: adjust cluster policies, autoscale configs,
and DBU pricing to project savings.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import get_billing_trend, get_dbu_waste, DEMO_MODE
from utils import fmt_currency, BRAND_ORANGE, BRAND_RED, BRAND_GREEN, BRAND_BLUE, PALETTE

st.set_page_config(page_title="What-If Projections · LakePulse", page_icon="📊", layout="wide")
st.title("📊 What-If Projections")
st.caption("Simulate cost impact of configuration changes · Actionable DBU savings scenarios")

@st.cache_data(ttl=300)
def load():
    trend = get_billing_trend()
    waste = get_dbu_waste()
    trend["date"]               = pd.to_datetime(trend["date"])
    trend["estimated_cost_usd"] = pd.to_numeric(trend["estimated_cost_usd"], errors="coerce")
    trend["total_dbu"]          = pd.to_numeric(trend["total_dbu"], errors="coerce")
    waste["estimated_cost_usd"] = pd.to_numeric(waste["estimated_cost_usd"], errors="coerce")
    waste["total_dbu"]          = pd.to_numeric(waste["total_dbu"], errors="coerce")
    return trend, waste

with st.spinner("Loading billing data..."):
    trend, waste = load()

baseline_monthly = trend.groupby(trend["date"].dt.to_period("M"))["estimated_cost_usd"].sum().iloc[-2]  # last full month
baseline_dbu_day = trend["total_dbu"].sum() / 90

# ── Scenario Controls ─────────────────────────────────────────────────────────
st.subheader("⚙️ Configure Scenarios")
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("**Cluster Auto-Termination**")
    autotermination_pct = st.slider(
        "% of ALL_PURPOSE clusters with auto-term enabled", 0, 100, 40, 5,
        help="Industry benchmark: 80%+ clusters should have auto-termination ≤ 30 min"
    )
    idle_reduction = st.slider("Expected idle DBU reduction (%)", 5, 50, 20, 5)

with col_b:
    st.markdown("**Spot Instance Adoption**")
    spot_pct = st.slider(
        "% of workloads migrated to Spot/Preemptible", 0, 80, 30, 5,
        help="Spot instances can reduce compute cost by 60-80%"
    )
    spot_discount = st.slider("Spot discount vs On-Demand (%)", 40, 80, 65, 5)

with col_c:
    st.markdown("**Job Cluster vs All-Purpose**")
    jobs_migration_pct = st.slider(
        "% of ALL_PURPOSE workloads moved to Job Clusters", 0, 80, 25, 5,
        help="Job clusters (per-run) are ~30% cheaper than always-on ALL_PURPOSE"
    )
    job_cluster_saving = st.slider("Job cluster savings vs ALL_PURPOSE (%)", 10, 40, 30, 5)

st.markdown("---")

# ── Projection calculation ────────────────────────────────────────────────────
waste_total   = waste["estimated_cost_usd"].sum()

saving_autotermination = waste_total * (autotermination_pct / 100) * (idle_reduction / 100)
saving_spot           = baseline_monthly * (spot_pct / 100) * (spot_discount / 100)
saving_job_cluster    = baseline_monthly * (jobs_migration_pct / 100) * (job_cluster_saving / 100)
total_monthly_saving  = saving_autotermination + saving_spot + saving_job_cluster
annual_saving         = total_monthly_saving * 12
projected_monthly     = max(0, baseline_monthly - total_monthly_saving)
saving_pct            = total_monthly_saving / max(baseline_monthly, 1) * 100

# ── Summary metrics ────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("📌 Baseline Monthly Spend",    fmt_currency(baseline_monthly))
c2.metric("💡 Projected Monthly Spend",   fmt_currency(projected_monthly),
          f"-{fmt_currency(total_monthly_saving)}", delta_color="inverse")
c3.metric("📉 Monthly Savings",           fmt_currency(total_monthly_saving), f"{saving_pct:.1f}%")
c4.metric("📅 Projected Annual Savings",  fmt_currency(annual_saving))

st.markdown("---")

# ── Waterfall chart ────────────────────────────────────────────────────────────
cl, cr = st.columns([3, 2])

with cl:
    st.subheader("Savings Waterfall")
    measures = ["absolute", "relative", "relative", "relative", "total"]
    x_labels = [
        "Baseline",
        "Auto-Termination",
        "Spot Instances",
        "Job Clusters",
        "Projected",
    ]
    y_values = [
        baseline_monthly,
        -saving_autotermination,
        -saving_spot,
        -saving_job_cluster,
        projected_monthly,
    ]
    colors = [BRAND_BLUE, BRAND_GREEN, BRAND_GREEN, BRAND_GREEN, BRAND_ORANGE]

    fig = go.Figure(go.Waterfall(
        name="",
        orientation="v",
        measure=measures,
        x=x_labels,
        y=y_values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": BRAND_GREEN}},
        increasing={"marker": {"color": BRAND_RED}},
        totals={"marker": {"color": BRAND_ORANGE}},
        text=[fmt_currency(abs(v)) for v in y_values],
        textposition="outside",
    ))
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0),
                      yaxis_title="Monthly Cost (USD)", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with cr:
    st.subheader("Savings Breakdown")
    breakdown = pd.DataFrame({
        "Lever":   ["Auto-Termination", "Spot Instances", "Job Clusters"],
        "Saving":  [saving_autotermination, saving_spot, saving_job_cluster],
    })
    fig = px.pie(
        breakdown, values="Saving", names="Lever",
        hole=0.4,
        color_discrete_sequence=[BRAND_GREEN, BRAND_BLUE, BRAND_ORANGE],
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ── 12-month projection ────────────────────────────────────────────────────────
st.subheader("12-Month Cost Projection")

months      = pd.date_range(start=pd.Timestamp.now().replace(day=1), periods=12, freq="MS")
growth_rate = st.slider("Assumed monthly workload growth (%)", 0.0, 5.0, 1.5, 0.5) / 100

baseline_proj  = [baseline_monthly  * ((1 + growth_rate) ** i) for i in range(12)]
optimised_proj = [projected_monthly * ((1 + growth_rate) ** i) for i in range(12)]

proj_df = pd.DataFrame({
    "Month":     months,
    "Baseline":  baseline_proj,
    "Optimised": optimised_proj,
})
proj_melt = proj_df.melt("Month", var_name="Scenario", value_name="Cost")

fig = px.line(
    proj_melt, x="Month", y="Cost", color="Scenario",
    color_discrete_map={"Baseline": BRAND_RED, "Optimised": BRAND_GREEN},
    labels={"Cost": "Monthly Cost (USD)", "Month": ""},
    markers=True,
)
fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
cumulative_saving = sum(b - o for b, o in zip(baseline_proj, optimised_proj))
st.plotly_chart(fig, use_container_width=True)
st.success(f"🎯 Cumulative 12-month savings with current settings: **{fmt_currency(cumulative_saving)}**")

# ── Recommendation cards ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Actionable Recommendations")

recs = [
    ("🔴" if autotermination_pct < 60 else "🟢",
     "Auto-Termination",
     f"{autotermination_pct}% of clusters covered",
     "Set `autotermination_minutes: 30` on all interactive clusters via cluster policy.",
     fmt_currency(saving_autotermination)),
    ("🔴" if spot_pct < 30 else "🟢",
     "Spot Instances",
     f"{spot_pct}% of workloads on Spot",
     "Use `SPOT_WITH_FALLBACK` availability on non-critical JOBS clusters.",
     fmt_currency(saving_spot)),
    ("🔴" if jobs_migration_pct < 40 else "🟢",
     "Job Clusters",
     f"{jobs_migration_pct}% migrated from ALL_PURPOSE",
     "Convert scheduled jobs from interactive to job clusters via `new_cluster` in Workflows.",
     fmt_currency(saving_job_cluster)),
]

for icon, name, status, action, saving in recs:
    with st.container(border=True):
        rc1, rc2, rc3 = st.columns([1, 5, 2])
        rc1.markdown(f"## {icon}")
        rc2.markdown(f"**{name}** · {status}")
        rc2.caption(action)
        rc3.metric("Monthly Saving", saving)
