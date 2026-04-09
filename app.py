"""
LakePulse — Native Databricks FinOps Suite
"""
import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import (
    get_billing_trend, get_dbu_waste, get_job_history,
    get_bottlenecks, get_data_popularity,
    CLOUD_DBU_DEFAULTS, DBU_PRICE_DEFAULT,
    get_workspace_info, _is_live,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LakePulse", page_icon="⚡", layout="wide")
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.5rem !important; }
.block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

ORANGE = "#FF6B35"; RED = "#E63946"; GREEN = "#2DC653"; BLUE = "#457B9D"
PALETTE = [ORANGE, BLUE, GREEN, "#A8DADC", "#E9C46A", "#F1FAEE"]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ LakePulse")
    st.markdown("---")

    # Connection
    st.markdown("### Connection")
    warehouse_id = st.text_input(
        "SQL Warehouse ID",
        value=os.getenv("DATABRICKS_WAREHOUSE_ID", ""),
        help="Databricks → SQL Warehouses → your warehouse → Connection Details",
    )

    @st.cache_data(ttl=3600)
    def _ws_info(wh: str) -> dict:
        return get_workspace_info(wh)

    ws_info = _ws_info(warehouse_id)

    if _is_live(warehouse_id):
        st.success(f"Live · {ws_info['cloud']}")
        if ws_info["host"]:
            st.caption(ws_info["host"].replace("https://", ""))
    else:
        st.warning("Demo data — enter a Warehouse ID to connect")

    st.markdown("---")

    # Pricing
    st.markdown("### Pricing")
    cloud_options = ["AWS", "Azure", "GCP"]
    detected_cloud = ws_info["cloud"]
    cloud = st.selectbox(
        "Cloud",
        cloud_options,
        index=cloud_options.index(detected_cloud),
        help="Auto-detected from workspace URL when connected. Change if needed.",
    )

    defaults = CLOUD_DBU_DEFAULTS[cloud]

    with st.expander("DBU prices — edit to match your contract", expanded=False):
        st.caption("Default: Databricks public list prices (databricks.com/product/pricing, Enterprise, 2024)")
        prices = {
            "ALL_PURPOSE": st.number_input("ALL_PURPOSE  $/DBU", min_value=0.0,
                                           value=float(defaults["ALL_PURPOSE"]),
                                           step=0.01, format="%.3f"),
            "JOBS":        st.number_input("JOBS  $/DBU",        min_value=0.0,
                                           value=float(defaults["JOBS"]),
                                           step=0.01, format="%.3f"),
            "DLT":         st.number_input("DLT  $/DBU",         min_value=0.0,
                                           value=float(defaults["DLT"]),
                                           step=0.01, format="%.3f"),
            "SQL":         st.number_input("SQL  $/DBU",          min_value=0.0,
                                           value=float(defaults["SQL"]),
                                           step=0.01, format="%.3f"),
        }

    discount = st.slider(
        "Negotiated discount off list (%)", 0, 70, 0, 1,
        help="Your committed-use, EA, or partner discount. Applied on top of the prices above.",
    )
    if discount:
        prices = {k: round(v * (1 - discount / 100), 4) for k, v in prices.items()}
        st.caption(
            f"After {discount}% discount — "
            f"AP ${prices['ALL_PURPOSE']:.3f} · "
            f"Jobs ${prices['JOBS']:.3f} · "
            f"DLT ${prices['DLT']:.3f} · "
            f"SQL ${prices['SQL']:.3f} /DBU"
        )

    price_default = round(sum(prices.values()) / len(prices), 4)

    spot_label = {"AWS": "Spot", "Azure": "Spot VMs", "GCP": "Preemptible"}[cloud]


# ── Header ─────────────────────────────────────────────────────────────────────
h_left, h_right = st.columns([4, 1])
with h_left:
    st.markdown("## ⚡ LakePulse")
    st.caption(
        "Databricks FinOps · "
        "system.billing.usage · system.compute.clusters · "
        "system.query.history · system.lakeflow.job_run_timeline · system.access.audit"
    )
with h_right:
    if _is_live(warehouse_id) and ws_info["host"]:
        st.markdown(f"**{ws_info['cloud']}**")
        st.caption(ws_info["host"].replace("https://", ""))


# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_all(wh: str):
    billing    = get_billing_trend(wh)
    waste      = get_dbu_waste(wh)
    jobs       = get_job_history(wh)
    bottleneck = get_bottlenecks(wh)
    popularity = get_data_popularity(wh)
    for df, cols in [
        (billing,    ["total_dbu", "estimated_cost_usd"]),
        (waste,      ["total_dbu", "estimated_cost_usd", "lifetime_hours"]),
        (jobs,       ["duration_min", "queue_min"]),
        (bottleneck, ["duration_min", "shuffle_read_gb", "shuffle_write_gb", "peak_memory_gb"]),
        (popularity, ["access_count", "days_since_access", "unique_users"]),
    ]:
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    billing["date"]      = pd.to_datetime(billing["date"], errors="coerce")
    jobs["trigger_time"] = pd.to_datetime(jobs["trigger_time"], errors="coerce")
    return billing, waste, jobs, bottleneck, popularity

with st.spinner("Loading telemetry from system tables..."):
    try:
        billing, waste, jobs, bottleneck, popularity = load_all(warehouse_id)
    except Exception as e:
        st.error(f"Failed to query system tables: {e}")
        st.stop()

# ── Apply user pricing on top of cached raw DBU counts ──────────────────────
# Data layer returns raw total_dbu; we recalculate cost here so sidebar changes
# take effect immediately without invalidating the 5-minute data cache.
billing = billing.copy()
billing["estimated_cost_usd"] = (
    billing["total_dbu"] * billing["product"].map(prices).fillna(price_default)
).round(2)

waste = waste.copy()
waste["estimated_cost_usd"] = (waste["total_dbu"] * prices["ALL_PURPOSE"]).round(2)


# ── Tabs ───────────────────────────────────────────────────────────────────────
t0, t1, t2, t3, t4, t5, t6 = st.tabs([
    "Overview", "DBU Waste", "Bottlenecks",
    "SLA Oracle", "Data Heatmap", "What-If", "Budget",
])


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with t0:
    today     = pd.Timestamp.now()
    cur_m     = today.month
    prev_m    = cur_m - 1 if cur_m > 1 else 12
    prev_year = today.year if cur_m > 1 else today.year - 1

    cur_month_spend  = billing[billing["date"].dt.month == cur_m]["estimated_cost_usd"].sum()
    prev_month_spend = billing[
        (billing["date"].dt.month == prev_m) & (billing["date"].dt.year == prev_year)
    ]["estimated_cost_usd"].sum()

    waste_cost = waste["estimated_cost_usd"].sum()
    failed     = (jobs["result_state"] != "SUCCEEDED").sum()
    fail_rate  = failed / max(len(jobs), 1) * 100
    sql_spend  = billing[billing["product"] == "SQL"]["estimated_cost_usd"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "MTD Spend",
        f"${cur_month_spend:,.0f}",
        f"${cur_month_spend - prev_month_spend:+,.0f} vs last month",
        help="system.billing.usage · Cost = DBUs × your configured price per product",
    )
    c2.metric(
        "Recoverable Waste",
        f"${waste_cost:,.0f}",
        f"{len(waste)} ALL_PURPOSE clusters",
        delta_color="inverse",
        help="ALL_PURPOSE DBU spend last 30 days at your configured price. "
             "Waste = clusters that can be terminated, right-sized, or moved to job clusters.",
    )
    c3.metric(
        "Job Failure Rate",
        f"{fail_rate:.1f}%",
        f"{int(failed)} failed runs",
        delta_color="inverse",
        help="system.lakeflow.job_run_timeline · Includes FAILED + TIMED_OUT states · Last 30 days",
    )
    c4.metric(
        "SQL Warehouse Spend (90d)",
        f"${sql_spend:,.0f}",
        help="system.billing.usage · SQL product type only · Reflects your configured SQL price",
    )

    st.markdown("---")
    cl, cr = st.columns(2)
    with cl:
        daily = billing.groupby(["date", "product"])["estimated_cost_usd"].sum().reset_index()
        fig = px.area(
            daily, x="date", y="estimated_cost_usd", color="product",
            color_discrete_sequence=PALETTE,
            labels={"estimated_cost_usd": "Cost (USD)", "date": "", "product": "Product"},
            title="Daily Spend by Product — Last 90 Days",
        )
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=40,b=0),
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        by_prod = billing.groupby("product")["estimated_cost_usd"].sum().reset_index()
        fig = px.pie(by_prod, values="estimated_cost_usd", names="product",
                     hole=0.45, color_discrete_sequence=PALETTE, title="Spend Mix by Product")
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=40,b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    top10 = waste.nlargest(10, "estimated_cost_usd").sort_values("estimated_cost_usd")
    fig = px.bar(
        top10, x="estimated_cost_usd", y=top10["cluster_name"].str[:35],
        orientation="h", color="estimated_cost_usd",
        color_continuous_scale=[GREEN, ORANGE, RED],
        labels={"estimated_cost_usd": "USD", "y": ""},
        title="Top 10 Clusters by Cost — Last 30 Days",
    )
    fig.update_layout(height=320, coloraxis_showscale=False, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# DBU WASTE KILLER
# ══════════════════════════════════════════════════════════════════════════════
with t1:
    st.subheader("DBU Waste Killer")
    st.caption(
        "system.billing.usage × system.compute.clusters · "
        "ALL_PURPOSE clusters · Last 30 days · "
        f"Priced at ${prices['ALL_PURPOSE']:.3f}/DBU (ALL_PURPOSE)"
    )

    min_cost = st.slider("Min cost threshold (USD)", 0, 500, 20, 10, key="w_slider")
    df_w = waste[waste["estimated_cost_usd"] >= min_cost].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Recoverable Spend",  f"${df_w['estimated_cost_usd'].sum():,.0f}",
              help=f"ALL_PURPOSE DBU spend × ${prices['ALL_PURPOSE']:.3f}/DBU (your configured price)")
    c2.metric("Critical (≥ $500)", str(int((df_w["estimated_cost_usd"] >= 500).sum())),
              delta_color="inverse")
    c3.metric("Total DBU Consumed", f"{df_w['total_dbu'].sum():,.0f}")

    cl, cr = st.columns(2)
    with cl:
        top = df_w.nlargest(15, "estimated_cost_usd").sort_values("estimated_cost_usd")
        fig = px.bar(top, x="estimated_cost_usd", y=top["cluster_name"].str[:30],
                     orientation="h", color="estimated_cost_usd",
                     color_continuous_scale=[GREEN, ORANGE, RED],
                     labels={"estimated_cost_usd": "USD", "y": ""}, title="Cost by Cluster")
        fig.update_layout(height=380, coloraxis_showscale=False, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        by_owner = (df_w.groupby("owner")["estimated_cost_usd"].sum()
                    .reset_index().sort_values("estimated_cost_usd", ascending=False))
        fig = px.bar(by_owner, x="owner", y="estimated_cost_usd",
                     color="estimated_cost_usd", color_continuous_scale=[GREEN, ORANGE, RED],
                     labels={"estimated_cost_usd": "USD", "owner": ""}, title="Waste by Owner")
        fig.update_layout(height=380, coloraxis_showscale=False, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    def _sev(v):
        return "Critical" if v >= 500 else ("High" if v >= 100 else ("Medium" if v >= 20 else "Low"))

    def _act(r):
        if r["estimated_cost_usd"] >= 500: return "Terminate immediately"
        if r["estimated_cost_usd"] >= 100: return "Set autotermination ≤ 30 min via cluster policy"
        return "Review utilisation pattern"

    df_w["severity"] = df_w["estimated_cost_usd"].apply(_sev)
    df_w["action"]   = df_w.apply(_act, axis=1)

    st.subheader("Kill List")
    st.dataframe(
        df_w[["cluster_name","owner","sku_name","total_dbu","lifetime_hours","estimated_cost_usd","severity","action"]]
          .rename(columns={
              "cluster_name":"Cluster","owner":"Owner","sku_name":"Instance",
              "total_dbu":"DBU","lifetime_hours":"Lifetime (hrs)",
              "estimated_cost_usd":"Cost (USD)","severity":"Severity","action":"Action",
          })
          .style.format({"DBU":"{:.1f}","Cost (USD)":"${:.2f}","Lifetime (hrs)":"{:.0f}"}),
        use_container_width=True, hide_index=True, height=400,
    )
    st.info(
        f"Acting on this list recovers **${df_w['estimated_cost_usd'].sum():,.0f}** "
        "over the next 30 days at current run rate."
    )


# ══════════════════════════════════════════════════════════════════════════════
# BOTTLENECK DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
with t2:
    st.subheader("Bottleneck Detector")
    st.caption(
        "system.query.history · Queries > 60 s · Last 7 days · "
        "shuffle_read_bytes & peak_memory_bytes from the metrics struct"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Queries Analysed", str(len(bottleneck)))
    c2.metric(
        "Total Shuffle Spill", f"{bottleneck['shuffle_read_gb'].sum():.1f} GB",
        help="shuffle_read_bytes from system.query.history.metrics. "
             "Spill to disk always hurts performance; >10 GB is severe.",
    )
    c3.metric(
        "Slowest Query", f"{bottleneck['duration_min'].max():.0f} min",
        help="total_duration_ms / 60000 from system.query.history",
    )

    cl, cr = st.columns(2)
    with cl:
        fig = px.scatter(
            bottleneck, x="duration_min", y="shuffle_read_gb",
            size="peak_memory_gb", color="executed_by",
            hover_data=["query_snippet"],
            color_discrete_sequence=PALETTE,
            labels={"duration_min":"Duration (min)","shuffle_read_gb":"Shuffle Read (GB)"},
            title="Shuffle Spill vs Duration",
        )
        fig.add_hline(y=10, line_dash="dot", line_color=RED,
                      annotation_text="10 GB — severe spill")
        fig.update_layout(height=360, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        fig = px.histogram(
            bottleneck, x="shuffle_read_gb", nbins=15,
            color_discrete_sequence=[BLUE],
            labels={"shuffle_read_gb":"Shuffle Read (GB)"},
            title="Shuffle Read Distribution",
        )
        fig.update_layout(height=360, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    def _fix(r):
        if r["shuffle_read_gb"] > 20:
            return "Severe Spill — check for cartesian join or skewed keys; enable AQE (spark.sql.adaptive.enabled=true)"
        if r["shuffle_read_gb"] > 5:
            return "High Shuffle — enable AQE; consider REPARTITION hint or salting skewed keys"
        if r["peak_memory_gb"] > 16:
            return "Memory Pressure — increase executor memory or use broadcast join for the smaller table"
        return "Monitor"

    bottleneck["diagnosis"] = bottleneck.apply(_fix, axis=1)
    st.subheader("Top Offenders")
    st.dataframe(
        bottleneck.nlargest(15, "shuffle_read_gb")
          [["query_snippet","executed_by","duration_min","shuffle_read_gb","shuffle_write_gb","peak_memory_gb","diagnosis"]]
          .rename(columns={
              "query_snippet":"Query","executed_by":"User","duration_min":"Duration (min)",
              "shuffle_read_gb":"Shuffle Read (GB)","shuffle_write_gb":"Shuffle Write (GB)",
              "peak_memory_gb":"Peak Mem (GB)","diagnosis":"Fix",
          })
          .style.format({"Duration (min)":"{:.1f}","Shuffle Read (GB)":"{:.2f}",
                         "Shuffle Write (GB)":"{:.2f}","Peak Mem (GB)":"{:.2f}"}),
        use_container_width=True, hide_index=True, height=400,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SLA ORACLE
# ══════════════════════════════════════════════════════════════════════════════
with t3:
    st.subheader("SLA Oracle")
    st.caption(
        "system.lakeflow.job_run_timeline · RandomForest classifier · "
        "SLA = p75 of historical run duration per job · Last 30 days"
    )

    @st.cache_resource
    def train_model(wh: str):
        df = jobs.dropna(subset=["duration_min", "trigger_time"]).copy()
        p75       = df.groupby("job_name")["duration_min"].quantile(0.75).rename("p75")
        df        = df.join(p75, on="job_name")
        df["is_late"]     = (df["duration_min"] > df["p75"]).astype(int)
        df["hour"]        = df["trigger_time"].dt.hour
        df["dow"]         = df["trigger_time"].dt.dayofweek
        df["rolling_avg"] = df.groupby("job_name")["duration_min"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
        le             = LabelEncoder()
        df["job_enc"]  = le.fit_transform(df["job_name"].fillna("unknown"))
        feats          = ["job_enc","hour","dow","queue_min","rolling_avg"]
        X              = df[feats].fillna(0)
        y              = df["is_late"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        clf.fit(X_tr, y_tr)
        df["delay_prob"] = clf.predict_proba(X)[:, 1]
        imps = pd.Series(clf.feature_importances_, index=feats).sort_values(ascending=False)
        return df, clf.score(X_te, y_te), imps

    with st.spinner("Training SLA Oracle on job run history..."):
        sla_df, accuracy, imps = train_model(warehouse_id)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Model Accuracy", f"{accuracy:.1%}",
        help="Hold-out test set accuracy (80/20 split). Predicts whether a run will exceed its job's p75 historical duration.",
    )
    c2.metric("High Risk (≥75%)",   str(int((sla_df["delay_prob"] >= 0.75).sum())), delta_color="inverse")
    c3.metric("Medium Risk (≥45%)", str(int(((sla_df["delay_prob"] >= 0.45) & (sla_df["delay_prob"] < 0.75)).sum())), delta_color="inverse")
    c4.metric("On Track",           str(int((sla_df["delay_prob"] < 0.45).sum())))

    cl, cr = st.columns(2)
    with cl:
        fig = px.histogram(sla_df, x="delay_prob", nbins=20, color_discrete_sequence=[ORANGE],
                           labels={"delay_prob":"Delay Probability"},
                           title="Delay Probability Distribution")
        fig.add_vline(x=0.45, line_dash="dot", line_color=ORANGE, annotation_text="Medium (45%)")
        fig.add_vline(x=0.75, line_dash="dot", line_color=RED,    annotation_text="High (75%)")
        fig.update_layout(height=320, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        labels = {
            "job_enc":"Job Identity","rolling_avg":"Recent Avg Duration",
            "queue_min":"Queue Time","hour":"Hour of Day","dow":"Day of Week",
        }
        imp_df = imps.reset_index(); imp_df.columns = ["feature","importance"]
        imp_df["feature"] = imp_df["feature"].map(labels)
        fig = px.bar(imp_df, x="importance", y="feature", orientation="h",
                     color="importance", color_continuous_scale=[GREEN, ORANGE],
                     labels={"importance":"Importance","feature":""}, title="Feature Importance")
        fig.update_layout(height=320, coloraxis_showscale=False, margin=dict(l=0,r=0,t=40,b=0),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    def _risk(p):
        return "High Risk" if p >= 0.75 else ("Medium Risk" if p >= 0.45 else "On Track")

    at_risk = sla_df[sla_df["delay_prob"] >= 0.45].sort_values("delay_prob", ascending=False).head(20).copy()
    at_risk["risk"] = at_risk["delay_prob"].apply(_risk)
    at_risk["p75"]  = at_risk["p75"].round(1)

    st.subheader("At-Risk Runs — predicted to exceed job's p75 SLA")
    st.dataframe(
        at_risk[["job_name","trigger_type","trigger_time","duration_min","p75","delay_prob","risk"]]
          .rename(columns={
              "job_name":"Job","trigger_type":"Trigger","trigger_time":"Started",
              "duration_min":"Duration (min)","p75":"SLA p75 (min)",
              "delay_prob":"Delay Prob","risk":"Risk",
          })
          .style.format({"Duration (min)":"{:.1f}","SLA p75 (min)":"{:.1f}","Delay Prob":"{:.1%}"}),
        use_container_width=True, hide_index=True, height=380,
    )


# ══════════════════════════════════════════════════════════════════════════════
# DATA HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
with t4:
    st.subheader("Data Popularity Heatmap")
    st.caption(
        "system.access.audit · "
        "getTable + selectFromTable + createTableAsSelect events · Last 90 days"
    )

    unused_days = st.slider("Mark unused after (days)", 7, 90, 30, key="h_slider")
    pop = popularity.copy()
    pop["catalog"] = pop["table_name"].str.split(".").str[0]
    pop["schema"]  = pop["table_name"].str.split(".").str[1]
    pop["table"]   = pop["table_name"].str.split(".").str[-1]
    pop["status"]  = pop.apply(
        lambda r: "Zombie"  if r["days_since_access"] >= unused_days and r["access_count"] < 5
        else ("Cooling"     if r["days_since_access"] >= unused_days
        else ("Hot"         if r["access_count"] > 100
        else "Active")), axis=1,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Tables Tracked",    str(len(pop)))
    c2.metric("Unused Tables",     str(int((pop["days_since_access"] >= unused_days).sum())),
              delta_color="inverse",
              help=f"No access events in system.access.audit for {unused_days}+ days")
    c3.metric("Hot Tables (>100)", str(int((pop["access_count"] > 100).sum())))

    cl, cr = st.columns([3, 2])
    with cl:
        pivot      = pop.groupby(["catalog","schema"])["access_count"].sum().reset_index()
        pivot_wide = pivot.pivot(index="schema", columns="catalog", values="access_count").fillna(0)
        fig = go.Figure(go.Heatmap(
            z=pivot_wide.values,
            x=pivot_wide.columns.tolist(),
            y=pivot_wide.index.tolist(),
            colorscale=[[0,"#f0f4ff"],[0.5,ORANGE],[1,RED]],
            text=pivot_wide.values.astype(int), texttemplate="%{text}",
        ))
        fig.update_layout(height=380, margin=dict(l=0,r=0,t=10,b=0),
                          xaxis_title="Catalog", yaxis_title="Schema",
                          title="Access Count by Catalog × Schema")
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        top10 = pop.nlargest(10, "access_count").sort_values("access_count")
        fig = px.bar(top10, x="access_count", y=top10["table"].str[:25], orientation="h",
                     color="unique_users", color_continuous_scale=[BLUE, ORANGE],
                     labels={"access_count":"Access Events","y":"","unique_users":"Unique Users"},
                     title="Top 10 Most Accessed Tables")
        fig.update_layout(height=380, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    zombies = pop[pop["status"] == "Zombie"].sort_values("days_since_access", ascending=False)
    st.subheader(f"Prune Candidates — {len(zombies)} tables with < 5 accesses in {unused_days}+ days")
    if len(zombies):
        st.dataframe(
            zombies[["table_name","catalog","schema","access_count","unique_users","days_since_access"]]
              .rename(columns={
                  "table_name":"Full Name","catalog":"Catalog","schema":"Schema",
                  "access_count":"Access Events","unique_users":"Unique Users",
                  "days_since_access":"Days Since Last Access",
              })
              .style.format({"Access Events":"{:.0f}","Days Since Last Access":"{:.0f}"}),
            use_container_width=True, hide_index=True, height=320,
        )
        st.warning(f"{len(zombies)} unused tables found. DROP or ARCHIVE to reduce Delta storage costs.")
    else:
        st.success(f"No unused tables found with a {unused_days}-day threshold.")


# ══════════════════════════════════════════════════════════════════════════════
# WHAT-IF PROJECTIONS
# ══════════════════════════════════════════════════════════════════════════════
with t5:
    st.subheader("What-If Projections")
    st.caption(
        "system.billing.usage · Cost uses your sidebar pricing · "
        "Savings modelled from Databricks best practices docs"
    )

    today  = pd.Timestamp.now()
    prev_m = today.month - 1 if today.month > 1 else 12
    prev_y = today.year if today.month > 1 else today.year - 1
    baseline = billing[
        (billing["date"].dt.month == prev_m) & (billing["date"].dt.year == prev_y)
    ]["estimated_cost_usd"].sum()

    ca, cb, cc = st.columns(3)
    with ca:
        st.markdown("**Auto-Termination**")
        at_pct = st.slider("% clusters with autotermination ≤ 30 min", 0, 100, 40, 5, key="at",
                           help="Industry target: ≥80%. Set via cluster policies in Databricks.")
        at_red = st.slider("Expected idle time reduction (%)", 5, 50, 20, 5, key="atr",
                           help="Idle ALL_PURPOSE clusters typically waste 20-40% of their DBU budget.")
    with cb:
        st.markdown(f"**{spot_label} Instances**")
        sp_pct  = st.slider(f"% workloads migrated to {spot_label}", 0, 80, 30, 5, key="sp",
                            help=f"Best for JOBS and DLT workloads. Not recommended for interactive clusters.")
        sp_disc = st.slider(f"{spot_label} discount vs On-Demand (%)", 50, 90, 70, 5, key="spd",
                            help=f"AWS Spot Advisor / Azure pricing docs show 70-90% savings on common instance types.")
    with cc:
        st.markdown("**Job Clusters vs All-Purpose**")
        jb_pct  = st.slider("% AP workloads moved to Job clusters", 0, 80, 25, 5, key="jb",
                            help="Job clusters terminate after each run — no idle billing.")
        jb_save = st.slider("Job cluster savings vs AP (%)", 10, 60, 35, 5, key="jbs",
                            help="Databricks docs: job clusters typically 30-50% cheaper for scheduled workloads.")

    waste_pool = waste["estimated_cost_usd"].sum()
    s_at  = waste_pool * (at_pct / 100) * (at_red / 100)
    s_sp  = baseline   * (sp_pct / 100) * (sp_disc / 100)
    s_jb  = baseline   * (jb_pct / 100) * (jb_save / 100)
    total = s_at + s_sp + s_jb
    proj  = max(0, baseline - total)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline (last month)",   f"${baseline:,.0f}")
    c2.metric("Projected Monthly Spend",  f"${proj:,.0f}",   f"-${total:,.0f}")
    c3.metric("Monthly Savings",          f"${total:,.0f}",   f"{total/max(baseline,1)*100:.1f}%")
    c4.metric("Projected Annual Savings", f"${total*12:,.0f}")

    st.markdown("---")
    cl, cr = st.columns([3, 2])
    with cl:
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute","relative","relative","relative","total"],
            x=["Baseline","Auto-Termination",f"{spot_label}","Job Clusters","Projected"],
            y=[baseline, -s_at, -s_sp, -s_jb, proj],
            connector={"line":{"color":"rgb(63,63,63)"}},
            decreasing={"marker":{"color":GREEN}},
            totals={"marker":{"color":ORANGE}},
            text=[f"${abs(v):,.0f}" for v in [baseline,-s_at,-s_sp,-s_jb,proj]],
            textposition="outside",
        ))
        fig.update_layout(height=380, margin=dict(l=0,r=0,t=10,b=0),
                          yaxis_title="Monthly Cost (USD)", showlegend=False,
                          title="Savings Waterfall")
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        bd  = pd.DataFrame({
            "Lever":  ["Auto-Termination", f"{spot_label}", "Job Clusters"],
            "Saving": [s_at, s_sp, s_jb],
        })
        fig = px.pie(bd, values="Saving", names="Lever", hole=0.4,
                     color_discrete_sequence=[GREEN, BLUE, ORANGE], title="Savings Mix")
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(height=380, margin=dict(l=0,r=0,t=40,b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    growth  = st.slider("Assumed monthly workload growth (%)", 0.0, 5.0, 1.5, 0.5, key="gr") / 100
    months  = pd.date_range(start=pd.Timestamp.now().replace(day=1), periods=12, freq="MS")
    proj_df = pd.DataFrame({
        "Month":     months,
        "Baseline":  [baseline * (1+growth)**i for i in range(12)],
        "Optimised": [proj     * (1+growth)**i for i in range(12)],
    })
    fig = px.line(
        proj_df.melt("Month", var_name="Scenario", value_name="Cost"),
        x="Month", y="Cost", color="Scenario",
        color_discrete_map={"Baseline":RED,"Optimised":GREEN},
        markers=True,
        labels={"Cost":"Monthly Cost (USD)","Month":""},
        title="12-Month Cost Projection",
    )
    fig.update_layout(height=300, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig, use_container_width=True)

    cum = sum((baseline - proj) * (1+growth)**i for i in range(12))
    st.success(f"Cumulative 12-month savings with current levers: **${cum:,.0f}**")


# ══════════════════════════════════════════════════════════════════════════════
# BUDGET TRACKER
# ══════════════════════════════════════════════════════════════════════════════
with t6:
    st.subheader("Budget Tracker")
    st.caption(
        "system.billing.usage · Cost calculated using your sidebar pricing · "
        "Projections based on current month's daily run rate"
    )

    today  = pd.Timestamp.now()
    prev_m = today.month - 1 if today.month > 1 else 12
    prev_y = today.year if today.month > 1 else today.year - 1
    prev_spend = billing[
        (billing["date"].dt.month == prev_m) & (billing["date"].dt.year == prev_y)
    ]["estimated_cost_usd"].sum()

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        monthly_budget = st.number_input(
            "Monthly budget (USD)",
            min_value=0,
            value=max(1000, int(prev_spend * 1.1)),
            step=500,
            help="Set your monthly spend target. Pre-filled with 10% above last month's actual spend.",
        )
    with col_b2:
        alert_at = st.slider(
            "Alert when projected spend exceeds (% of budget)", 50, 100, 85, 5,
            help="Shows a warning when projected month-end spend crosses this threshold.",
        )

    days_elapsed   = today.day
    days_in_month  = 30

    cur_month_data = billing[
        (billing["date"].dt.month == today.month) & (billing["date"].dt.year == today.year)
    ]
    mtd        = cur_month_data["estimated_cost_usd"].sum()
    daily_rate = mtd / max(days_elapsed, 1)
    proj_eom   = daily_rate * days_in_month
    remaining  = max(0, monthly_budget - mtd)
    vs_bgt_pct = (proj_eom - monthly_budget) / max(monthly_budget, 1) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MTD Spend",         f"${mtd:,.0f}",         f"Day {days_elapsed}")
    c2.metric("Daily Run Rate",    f"${daily_rate:,.0f}/d")
    c3.metric(
        "Projected Month-End",     f"${proj_eom:,.0f}",
        f"{vs_bgt_pct:+.1f}% vs budget",
        delta_color="inverse" if vs_bgt_pct > 0 else "normal",
    )
    c4.metric("Budget Remaining",  f"${remaining:,.0f}")

    pct_proj = proj_eom / max(monthly_budget, 1) * 100
    if proj_eom > monthly_budget:
        st.error(
            f"Over budget — on track to spend **${proj_eom:,.0f}** against **${monthly_budget:,.0f}** budget "
            f"(+${proj_eom - monthly_budget:,.0f})"
        )
    elif pct_proj >= alert_at:
        st.warning(f"Approaching limit — projected to use {pct_proj:.0f}% of ${monthly_budget:,} budget this month")
    else:
        st.success(f"On track — projected to use {pct_proj:.0f}% of ${monthly_budget:,} budget")

    st.markdown("---")
    cl, cr = st.columns(2)
    with cl:
        # Daily spend + cumulative vs budget pace for current month
        daily = (
            billing[billing["date"].dt.month == today.month]
            .groupby("date")["estimated_cost_usd"].sum()
            .reset_index().sort_values("date")
        )
        daily["cumulative"] = daily["estimated_cost_usd"].cumsum()

        if len(daily):
            budget_pace = [monthly_budget / days_in_month * (i+1) for i in range(len(daily))]
        else:
            budget_pace = []

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily["date"], y=daily["estimated_cost_usd"],
            name="Daily spend", marker_color=BLUE, opacity=0.7,
        ))
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["cumulative"],
            name="Cumulative spend", line=dict(color=ORANGE, width=2), yaxis="y2",
        ))
        if budget_pace:
            fig.add_trace(go.Scatter(
                x=daily["date"], y=budget_pace,
                name="Budget pace", line=dict(color=RED, dash="dot", width=1.5), yaxis="y2",
            ))
        fig.update_layout(
            height=320, margin=dict(l=0,r=0,t=10,b=0),
            title="Current Month — Daily Spend vs Budget Pace",
            yaxis=dict(title="Daily (USD)"),
            yaxis2=dict(title="Cumulative (USD)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=-0.3),
        )
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        prod_mtd = (
            cur_month_data.groupby("product")["estimated_cost_usd"].sum()
            .reset_index().sort_values("estimated_cost_usd")
        )
        prod_mtd["% of budget"] = (prod_mtd["estimated_cost_usd"] / max(monthly_budget, 1) * 100).round(1)
        fig = px.bar(
            prod_mtd, x="estimated_cost_usd", y="product", orientation="h",
            color="estimated_cost_usd", color_continuous_scale=[GREEN, ORANGE, RED],
            text=prod_mtd["% of budget"].astype(str) + "%",
            labels={"estimated_cost_usd":"MTD Spend (USD)","product":""},
            title="MTD Spend by Product",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=320, coloraxis_showscale=False, margin=dict(l=0,r=0,t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Monthly totals vs budget line
    monthly_totals = (
        billing.assign(month=billing["date"].dt.to_period("M").astype(str))
        .groupby("month")["estimated_cost_usd"].sum()
        .reset_index()
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly_totals["month"], y=monthly_totals["estimated_cost_usd"],
        name="Monthly spend", marker_color=BLUE,
    ))
    fig.add_hline(
        y=monthly_budget, line_dash="dot", line_color=RED,
        annotation_text=f"Budget ${monthly_budget:,}",
    )
    fig.update_layout(
        height=280, margin=dict(l=0,r=0,t=10,b=0),
        yaxis_title="USD",
        title="Monthly Spend vs Budget — Last 90 Days",
    )
    st.plotly_chart(fig, use_container_width=True)
