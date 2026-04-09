"""
LakePulse — Bottleneck Detector
Surfaces shuffle spill and data skew from system.query.history.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import get_bottlenecks, DEMO_MODE
from utils import fmt_gb, BRAND_ORANGE, BRAND_RED, BRAND_GREEN, BRAND_BLUE, PALETTE

st.set_page_config(page_title="Bottleneck Detector · LakePulse", page_icon="🔥", layout="wide")
st.title("🔥 Bottleneck Detector")
st.caption("Identifies shuffle spill, data skew, and memory pressure from query history · Last 7 days")

with st.sidebar:
    st.header("Filters")
    min_dur  = st.slider("Min Duration (min)", 0, 60, 5)
    min_spill = st.slider("Min Shuffle Read (GB)", 0.0, 20.0, 0.0, 0.5)

@st.cache_data(ttl=300)
def load():
    df = get_bottlenecks()
    for col in ["duration_min", "shuffle_read_gb", "shuffle_write_gb", "peak_memory_gb", "rows_read", "rows_written"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

with st.spinner("Analysing query telemetry..."):
    df = load()

df_f = df[(df["duration_min"] >= min_dur) & (df["shuffle_read_gb"] >= min_spill)]

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("🔍 Queries Analysed",     str(len(df)))
c2.metric("⚠️ Shuffle Spill Total",  fmt_gb(df["shuffle_read_gb"].sum()))
c3.metric("💾 Peak Memory (max)",    fmt_gb(df["peak_memory_gb"].max()))
c4.metric("⏱️ Slowest Query",        f"{df['duration_min'].max():.0f} min")

st.markdown("---")

# ── Scatter: duration vs shuffle ──────────────────────────────────────────────
cl, cr = st.columns(2)

with cl:
    st.subheader("Shuffle Spill vs Query Duration")
    fig = px.scatter(
        df_f,
        x="duration_min", y="shuffle_read_gb",
        size="peak_memory_gb", color="executed_by",
        hover_data=["query_snippet", "shuffle_write_gb"],
        labels={"duration_min": "Duration (min)", "shuffle_read_gb": "Shuffle Read (GB)"},
        color_discrete_sequence=PALETTE,
    )
    fig.add_hline(y=10, line_dash="dot", line_color=BRAND_RED,
                  annotation_text="Spill threshold (10 GB)", annotation_position="top right")
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with cr:
    st.subheader("Memory Pressure by User")
    by_user = df.groupby("executed_by").agg(
        avg_memory=("peak_memory_gb", "mean"),
        total_shuffle=("shuffle_read_gb", "sum"),
        query_count=("query_id", "count"),
    ).reset_index().sort_values("total_shuffle", ascending=False)

    fig = px.bar(
        by_user, x="executed_by", y="total_shuffle",
        color="avg_memory",
        color_continuous_scale=[BRAND_GREEN, BRAND_ORANGE, BRAND_RED],
        labels={"total_shuffle": "Total Shuffle Read (GB)", "executed_by": "User", "avg_memory": "Avg Memory (GB)"},
    )
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ── Top offenders table ────────────────────────────────────────────────────────
st.subheader("🏆 Top Shuffle Offenders")

display = df_f.nlargest(15, "shuffle_read_gb").copy()
display["skew_ratio"] = (display["shuffle_write_gb"] / display["shuffle_read_gb"].replace(0, 1)).round(2)
display["bottleneck"] = display.apply(
    lambda r: "🔴 Severe Spill"    if r["shuffle_read_gb"] > 20
    else "🟠 High Shuffle"         if r["shuffle_read_gb"] > 5
    else "🟡 Memory Pressure"      if r["peak_memory_gb"] > 16
    else "🟢 Monitor",
    axis=1,
)
display["fix"] = display.apply(
    lambda r: "Repartition or use AQE; check for cartesian join"
    if r["shuffle_read_gb"] > 20
    else ("Enable AQE: spark.sql.adaptive.enabled=true" if r["shuffle_read_gb"] > 5
          else ("Increase executor memory or use broadcast join" if r["peak_memory_gb"] > 16
                else "No immediate action required")),
    axis=1,
)

st.dataframe(
    display[[
        "query_snippet", "executed_by", "duration_min",
        "shuffle_read_gb", "shuffle_write_gb", "peak_memory_gb",
        "skew_ratio", "bottleneck", "fix",
    ]].rename(columns={
        "query_snippet":   "Query",
        "executed_by":     "User",
        "duration_min":    "Duration (min)",
        "shuffle_read_gb": "Shuffle Read (GB)",
        "shuffle_write_gb":"Shuffle Write (GB)",
        "peak_memory_gb":  "Peak Mem (GB)",
        "skew_ratio":      "Skew Ratio",
        "bottleneck":      "Issue",
        "fix":             "Recommended Fix",
    }).style.format({
        "Duration (min)":     "{:.1f}",
        "Shuffle Read (GB)":  "{:.2f}",
        "Shuffle Write (GB)": "{:.2f}",
        "Peak Mem (GB)":      "{:.2f}",
        "Skew Ratio":         "{:.2f}",
    }),
    use_container_width=True,
    hide_index=True,
    height=420,
)

# ── Distribution ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Shuffle Read Distribution")
fig = px.histogram(df, x="shuffle_read_gb", nbins=20,
                   color_discrete_sequence=[BRAND_BLUE],
                   labels={"shuffle_read_gb": "Shuffle Read (GB)", "count": "Queries"})
fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)
