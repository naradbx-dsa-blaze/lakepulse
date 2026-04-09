"""
LakePulse — Data Popularity Heatmap
Surfaces frequently accessed and completely unused tables using audit logs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import get_data_popularity, DEMO_MODE
from utils import BRAND_ORANGE, BRAND_RED, BRAND_GREEN, BRAND_BLUE, PALETTE

st.set_page_config(page_title="Data Heatmap · LakePulse", page_icon="🗺️", layout="wide")
st.title("🗺️ Data Popularity Heatmap")
st.caption("Table access frequency from audit logs · Identifies unused assets for pruning · Last 90 days")

with st.sidebar:
    st.header("Filters")
    unused_days = st.slider("Mark Unused After (days)", 7, 90, 30)
    min_access  = st.number_input("Min Access Count", 0, 100, 0)
    catalog_filter = st.text_input("Filter by Catalog (optional)", "")

@st.cache_data(ttl=300)
def load():
    df = get_data_popularity()
    df["access_count"]     = pd.to_numeric(df["access_count"],     errors="coerce").fillna(0)
    df["days_since_access"]= pd.to_numeric(df["days_since_access"],errors="coerce").fillna(0)
    df["unique_users"]     = pd.to_numeric(df["unique_users"],     errors="coerce").fillna(0)
    df["catalog"]  = df["table_name"].str.split(".").str[0]
    df["schema"]   = df["table_name"].str.split(".").str[1]
    df["table"]    = df["table_name"].str.split(".").str[-1]
    return df

with st.spinner("Scanning audit logs..."):
    df = load()

if catalog_filter:
    df = df[df["catalog"].str.contains(catalog_filter, case=False, na=False)]
df = df[df["access_count"] >= min_access]

unused  = df[df["days_since_access"] >= unused_days]
popular = df.nlargest(10, "access_count")

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("📊 Tables Tracked",      str(len(df)))
c2.metric("🗑️ Unused Tables",       str(len(unused)),  delta_color="inverse")
c3.metric("🔥 Hot Tables (>100)",   str((df["access_count"] > 100).sum()))
c4.metric("👤 Unique Users",         str(int(df["unique_users"].sum())))

st.markdown("---")

# ── Heatmap ────────────────────────────────────────────────────────────────────
cl, cr = st.columns([3, 2])

with cl:
    st.subheader("Access Count by Schema")
    pivot = df.groupby(["catalog", "schema"])["access_count"].sum().reset_index()
    pivot_wide = pivot.pivot(index="schema", columns="catalog", values="access_count").fillna(0)
    fig = go.Figure(go.Heatmap(
        z=pivot_wide.values,
        x=pivot_wide.columns.tolist(),
        y=pivot_wide.index.tolist(),
        colorscale=[[0, "#f0f4ff"], [0.5, BRAND_ORANGE], [1, BRAND_RED]],
        hoverongaps=False,
        text=pivot_wide.values.astype(int),
        texttemplate="%{text}",
    ))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="Catalog", yaxis_title="Schema")
    st.plotly_chart(fig, use_container_width=True)

with cr:
    st.subheader("🔥 Top 10 Hottest Tables")
    fig = px.bar(
        popular.sort_values("access_count"),
        x="access_count",
        y=popular.sort_values("access_count")["table"].str[:30],
        orientation="h",
        color="unique_users",
        color_continuous_scale=[BRAND_BLUE, BRAND_ORANGE],
        labels={"access_count": "Accesses", "y": "", "unique_users": "Unique Users"},
    )
    fig.update_layout(height=380, coloraxis_showscale=True, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ── Access timeline scatter ────────────────────────────────────────────────────
st.subheader("Access Frequency vs Days Since Last Access")
df["status"] = df.apply(
    lambda r: "🔴 Zombie (prune candidate)"  if r["days_since_access"] >= unused_days and r["access_count"] < 5
    else ("🟠 Cooling down"                   if r["days_since_access"] >= unused_days
    else ("🔥 Hot"                            if r["access_count"] > 100
    else "🟢 Active")),
    axis=1,
)
color_map = {
    "🔴 Zombie (prune candidate)": BRAND_RED,
    "🟠 Cooling down":             BRAND_ORANGE,
    "🔥 Hot":                      "#E63946",
    "🟢 Active":                   BRAND_GREEN,
}
fig = px.scatter(
    df,
    x="days_since_access", y="access_count",
    color="status", size="unique_users",
    hover_data=["table_name", "unique_users"],
    color_discrete_map=color_map,
    labels={"days_since_access": "Days Since Last Access", "access_count": "Total Accesses (90 days)"},
)
fig.add_vline(x=unused_days, line_dash="dot", line_color=BRAND_RED,
              annotation_text=f"Unused threshold ({unused_days} days)")
fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)

# ── Prune candidates ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"🗑️ Prune Candidates (not accessed in {unused_days}+ days)")

prune = (
    unused.sort_values("days_since_access", ascending=False)
    [["table_name", "catalog", "schema", "access_count", "unique_users", "days_since_access", "last_accessed"]]
    .rename(columns={
        "table_name":       "Full Table Name",
        "catalog":          "Catalog",
        "schema":           "Schema",
        "access_count":     "Total Accesses",
        "unique_users":     "Unique Users",
        "days_since_access":"Days Unused",
        "last_accessed":    "Last Accessed",
    })
)

if len(prune):
    st.dataframe(prune.style.format({"Total Accesses": "{:.0f}", "Days Unused": "{:.0f}"}),
                 use_container_width=True, hide_index=True, height=360)
    st.warning(f"⚠️ {len(prune)} tables unused for {unused_days}+ days. Consider dropping or archiving to save storage costs.")
else:
    st.success("✅ No unused tables found with current filter settings.")
