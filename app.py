"""
LakePulse — Native Databricks FinOps Suite
Single-file Streamlit app with tabs (no subdirectories for Databricks Apps compatibility).
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
    KWH_PER_DBU, KG_CO2_PER_KWH, TREES_KG_CO2_PER_YEAR, DEMO_MODE,
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

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## ⚡ LakePulse")
st.caption("Native Databricks FinOps · Kill DBU waste · Predict SLA breaches · Track ESG")
if DEMO_MODE:
    st.info("📊 **Demo mode** — set `DATABRICKS_WAREHOUSE_ID` as an App environment variable to connect to real system tables.", icon="ℹ️")

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_all():
    billing    = get_billing_trend()
    waste      = get_dbu_waste()
    jobs       = get_job_history()
    bottleneck = get_bottlenecks()
    popularity = get_data_popularity()
    for df, cols in [
        (billing,    ["total_dbu","estimated_cost_usd"]),
        (waste,      ["total_dbu","estimated_cost_usd","lifetime_hours"]),
        (jobs,       ["duration_min","queue_min"]),
        (bottleneck, ["duration_min","shuffle_read_gb","shuffle_write_gb","peak_memory_gb"]),
        (popularity, ["access_count","days_since_access","unique_users"]),
    ]:
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    billing["date"]       = pd.to_datetime(billing["date"], errors="coerce")
    jobs["trigger_time"]  = pd.to_datetime(jobs["trigger_time"], errors="coerce")
    return billing, waste, jobs, bottleneck, popularity

with st.spinner("Loading telemetry..."):
    billing, waste, jobs, bottleneck, popularity = load_all()

# ── Tabs ───────────────────────────────────────────────────────────────────────
t0, t1, t2, t3, t4, t5, t6 = st.tabs([
    "🏠 Overview", "💰 DBU Waste Killer", "🔥 Bottleneck Detector",
    "🔮 SLA Oracle", "🗺️ Data Heatmap", "📊 What-If", "🌱 ESG",
])

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with t0:
    now        = pd.Timestamp.now()
    cur_month  = billing[billing["date"].dt.month == now.month]["estimated_cost_usd"].sum()
    prev_month = billing[billing["date"].dt.month == (now.month - 1 if now.month > 1 else 12)]["estimated_cost_usd"].sum()
    waste_cost = waste["estimated_cost_usd"].sum()
    failed     = (jobs["result_state"] != "SUCCEEDED").sum()
    fail_rate  = failed / max(len(jobs), 1) * 100
    carbon_t   = billing["total_dbu"].sum() * KWH_PER_DBU * KG_CO2_PER_KWH / 1000

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 MTD Spend",         f"${cur_month:,.0f}",    f"${cur_month - prev_month:+,.0f} vs last month")
    c2.metric("🔥 Recoverable Waste", f"${waste_cost:,.0f}",   f"{len(waste)} clusters", delta_color="inverse")
    c3.metric("⚠️ Job Failure Rate",  f"{fail_rate:.1f}%",     f"{int(failed)} failed", delta_color="inverse")
    c4.metric("🌱 Carbon (90d)",       f"{carbon_t:.2f} tCO₂", "indicative")

    st.markdown("---")
    cl, cr = st.columns(2)
    with cl:
        daily = billing.groupby(["date","product"])["estimated_cost_usd"].sum().reset_index()
        fig = px.area(daily, x="date", y="estimated_cost_usd", color="product",
                      color_discrete_sequence=PALETTE,
                      labels={"estimated_cost_usd":"Cost (USD)","date":"","product":"Product"},
                      title="Daily Spend by Product (90 days)")
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=40,b=0), legend=dict(orientation="h",y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        by_prod = billing.groupby("product")["estimated_cost_usd"].sum().reset_index()
        fig = px.pie(by_prod, values="estimated_cost_usd", names="product",
                     hole=0.45, color_discrete_sequence=PALETTE, title="Spend Mix")
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=40,b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    top10 = waste.nlargest(10,"estimated_cost_usd").sort_values("estimated_cost_usd")
    fig = px.bar(top10, x="estimated_cost_usd", y=top10["cluster_name"].str[:35],
                 orientation="h", color="estimated_cost_usd",
                 color_continuous_scale=[GREEN,ORANGE,RED],
                 labels={"estimated_cost_usd":"USD","y":""},
                 title="Top 10 Wasteful Clusters")
    fig.update_layout(height=320, coloraxis_showscale=False, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# DBU WASTE KILLER
# ══════════════════════════════════════════════════════════════════════════════
with t1:
    st.subheader("💰 DBU Waste Killer")
    st.caption("Surfaces ALL_PURPOSE clusters burning DBUs with low utilisation · system.billing.usage + system.compute.clusters · Last 30 days")

    min_cost = st.slider("Min Cost (USD)", 0, 500, 20, 10, key="w_slider")
    df_w = waste[waste["estimated_cost_usd"] >= min_cost].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("💸 Recoverable",      f"${df_w['estimated_cost_usd'].sum():,.0f}")
    c2.metric("🔴 Critical (≥$500)", str(int((df_w["estimated_cost_usd"] >= 500).sum())), delta_color="inverse")
    c3.metric("📦 Total DBU",         f"{df_w['total_dbu'].sum():,.0f}")

    cl, cr = st.columns(2)
    with cl:
        top = df_w.nlargest(15,"estimated_cost_usd").sort_values("estimated_cost_usd")
        fig = px.bar(top, x="estimated_cost_usd", y=top["cluster_name"].str[:30],
                     orientation="h", color="estimated_cost_usd",
                     color_continuous_scale=[GREEN,ORANGE,RED],
                     labels={"estimated_cost_usd":"USD","y":""}, title="Cost by Cluster")
        fig.update_layout(height=380, coloraxis_showscale=False, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        by_owner = df_w.groupby("owner")["estimated_cost_usd"].sum().reset_index().sort_values("estimated_cost_usd",ascending=False)
        fig = px.bar(by_owner, x="owner", y="estimated_cost_usd",
                     color="estimated_cost_usd", color_continuous_scale=[GREEN,ORANGE,RED],
                     labels={"estimated_cost_usd":"USD","owner":""}, title="Waste by Owner")
        fig.update_layout(height=380, coloraxis_showscale=False, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    def _sev(v):
        return "🔴 Critical" if v>=500 else ("🟠 High" if v>=100 else ("🟡 Medium" if v>=20 else "🟢 Low"))
    def _act(r):
        if r["estimated_cost_usd"]>=500: return "Terminate immediately"
        if r["estimated_cost_usd"]>=100: return "Set auto-termination ≤ 30 min"
        return "Review utilisation"

    df_w["severity"] = df_w["estimated_cost_usd"].apply(_sev)
    df_w["action"]   = df_w.apply(_act, axis=1)
    st.subheader("🗡️ Kill List")
    st.dataframe(
        df_w[["cluster_name","owner","sku_name","total_dbu","lifetime_hours","estimated_cost_usd","severity","action"]]
          .rename(columns={"cluster_name":"Cluster","owner":"Owner","sku_name":"SKU","total_dbu":"DBU",
                           "lifetime_hours":"Lifetime (hrs)","estimated_cost_usd":"Cost (USD)",
                           "severity":"Severity","action":"Action"})
          .style.format({"DBU":"{:.1f}","Cost (USD)":"${:.2f}","Lifetime (hrs)":"{:.0f}"}),
        use_container_width=True, hide_index=True, height=400,
    )
    st.success(f"✅ Applying all actions saves **${df_w['estimated_cost_usd'].sum():,.0f}** in the next 30 days.")


# ══════════════════════════════════════════════════════════════════════════════
# BOTTLENECK DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
with t2:
    st.subheader("🔥 Bottleneck Detector")
    st.caption("Shuffle spill & data skew from system.query.history · Last 7 days")

    c1, c2, c3 = st.columns(3)
    c1.metric("🔍 Queries Analysed", str(len(bottleneck)))
    c2.metric("⚠️ Total Spill",      f"{bottleneck['shuffle_read_gb'].sum():.1f} GB")
    c3.metric("⏱️ Slowest Query",    f"{bottleneck['duration_min'].max():.0f} min")

    cl, cr = st.columns(2)
    with cl:
        fig = px.scatter(bottleneck, x="duration_min", y="shuffle_read_gb",
                         size="peak_memory_gb", color="executed_by",
                         hover_data=["query_snippet"],
                         color_discrete_sequence=PALETTE,
                         labels={"duration_min":"Duration (min)","shuffle_read_gb":"Shuffle Read (GB)"},
                         title="Shuffle Spill vs Duration")
        fig.add_hline(y=10, line_dash="dot", line_color=RED, annotation_text="10 GB threshold")
        fig.update_layout(height=360, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        fig = px.histogram(bottleneck, x="shuffle_read_gb", nbins=15,
                           color_discrete_sequence=[BLUE],
                           labels={"shuffle_read_gb":"Shuffle Read (GB)"},
                           title="Shuffle Read Distribution")
        fig.update_layout(height=360, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    def _fix(r):
        if r["shuffle_read_gb"] > 20: return "🔴 Severe Spill — repartition or check cartesian join; enable AQE"
        if r["shuffle_read_gb"] > 5:  return "🟠 High Shuffle — set spark.sql.adaptive.enabled=true"
        if r["peak_memory_gb"] > 16:  return "🟡 Memory Pressure — increase executor memory or broadcast join"
        return "🟢 Monitor"

    bottleneck["diagnosis"] = bottleneck.apply(_fix, axis=1)
    st.subheader("Top Offenders")
    st.dataframe(
        bottleneck.nlargest(15,"shuffle_read_gb")
          [["query_snippet","executed_by","duration_min","shuffle_read_gb","shuffle_write_gb","peak_memory_gb","diagnosis"]]
          .rename(columns={"query_snippet":"Query","executed_by":"User","duration_min":"Duration (min)",
                           "shuffle_read_gb":"Shuffle Read (GB)","shuffle_write_gb":"Shuffle Write (GB)",
                           "peak_memory_gb":"Peak Mem (GB)","diagnosis":"Diagnosis"})
          .style.format({"Duration (min)":"{:.1f}","Shuffle Read (GB)":"{:.2f}",
                         "Shuffle Write (GB)":"{:.2f}","Peak Mem (GB)":"{:.2f}"}),
        use_container_width=True, hide_index=True, height=400,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SLA ORACLE
# ══════════════════════════════════════════════════════════════════════════════
with t3:
    st.subheader("🔮 SLA Oracle")
    st.caption("RandomForest classifier on system.lakeflow.job_run_timeline · Predicts delay probability per run")

    @st.cache_resource
    def train_model():
        df = jobs.dropna(subset=["duration_min","trigger_time"]).copy()
        p75 = df.groupby("job_name")["duration_min"].quantile(0.75).rename("p75")
        df  = df.join(p75, on="job_name")
        df["is_late"]     = (df["duration_min"] > df["p75"]).astype(int)
        df["hour"]        = df["trigger_time"].dt.hour
        df["dow"]         = df["trigger_time"].dt.dayofweek
        df["rolling_avg"] = df.groupby("job_name")["duration_min"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
        le            = LabelEncoder()
        df["job_enc"] = le.fit_transform(df["job_name"].fillna("unknown"))
        feats         = ["job_enc","hour","dow","queue_min","rolling_avg"]
        X             = df[feats].fillna(0)
        y             = df["is_late"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        clf.fit(X_tr, y_tr)
        df["delay_prob"] = clf.predict_proba(X)[:, 1]
        imps = pd.Series(clf.feature_importances_, index=feats).sort_values(ascending=False)
        return df, clf.score(X_te, y_te), imps

    with st.spinner("Training SLA Oracle..."):
        sla_df, accuracy, imps = train_model()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Model Accuracy", f"{accuracy:.1%}")
    c2.metric("🔴 High Risk",      str(int((sla_df["delay_prob"] >= 0.75).sum())),  delta_color="inverse")
    c3.metric("🟠 Medium Risk",    str(int(((sla_df["delay_prob"] >= 0.45) & (sla_df["delay_prob"] < 0.75)).sum())), delta_color="inverse")
    c4.metric("🟢 On Track",       str(int((sla_df["delay_prob"] < 0.45).sum())))

    cl, cr = st.columns(2)
    with cl:
        fig = px.histogram(sla_df, x="delay_prob", nbins=20, color_discrete_sequence=[ORANGE],
                           labels={"delay_prob":"Delay Probability"}, title="Delay Probability Distribution")
        fig.add_vline(x=0.45, line_dash="dot", line_color=ORANGE, annotation_text="Medium risk")
        fig.add_vline(x=0.75, line_dash="dot", line_color=RED,    annotation_text="High risk")
        fig.update_layout(height=320, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        labels = {"job_enc":"Job Identity","rolling_avg":"Recent Avg Duration",
                  "queue_min":"Queue Time","hour":"Hour of Day","dow":"Day of Week"}
        imp_df = imps.reset_index(); imp_df.columns = ["feature","importance"]
        imp_df["feature"] = imp_df["feature"].map(labels)
        fig = px.bar(imp_df, x="importance", y="feature", orientation="h",
                     color="importance", color_continuous_scale=[GREEN,ORANGE],
                     labels={"importance":"Importance","feature":""}, title="Feature Importance")
        fig.update_layout(height=320, coloraxis_showscale=False, margin=dict(l=0,r=0,t=40,b=0),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    def _risk(p):
        return "🔴 High Risk" if p>=0.75 else ("🟠 Medium Risk" if p>=0.45 else "🟢 On Track")

    at_risk = sla_df[sla_df["delay_prob"] >= 0.45].sort_values("delay_prob",ascending=False).head(20).copy()
    at_risk["risk"] = at_risk["delay_prob"].apply(_risk)
    st.subheader("⚠️ At-Risk Runs")
    st.dataframe(
        at_risk[["job_name","creator_user_name","trigger_time","duration_min","delay_prob","risk"]]
          .rename(columns={"job_name":"Job","creator_user_name":"Owner","trigger_time":"Triggered",
                           "duration_min":"Duration (min)","delay_prob":"Delay Prob","risk":"Risk"})
          .style.format({"Duration (min)":"{:.1f}","Delay Prob":"{:.1%}"}),
        use_container_width=True, hide_index=True, height=380,
    )


# ══════════════════════════════════════════════════════════════════════════════
# DATA HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
with t4:
    st.subheader("🗺️ Data Popularity Heatmap")
    st.caption("Table access frequency from system.access.audit · Last 90 days")

    unused_days = st.slider("Mark unused after (days)", 7, 90, 30, key="h_slider")
    pop = popularity.copy()
    pop["catalog"] = pop["table_name"].str.split(".").str[0]
    pop["schema"]  = pop["table_name"].str.split(".").str[1]
    pop["table"]   = pop["table_name"].str.split(".").str[-1]
    pop["status"]  = pop.apply(
        lambda r: "🔴 Zombie"  if r["days_since_access"] >= unused_days and r["access_count"] < 5
        else ("🟠 Cooling"     if r["days_since_access"] >= unused_days
        else ("🔥 Hot"         if r["access_count"] > 100
        else "🟢 Active")), axis=1,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("📊 Tables Tracked", str(len(pop)))
    c2.metric("🗑️ Unused",        str(int((pop["days_since_access"] >= unused_days).sum())), delta_color="inverse")
    c3.metric("🔥 Hot (>100)",    str(int((pop["access_count"] > 100).sum())))

    cl, cr = st.columns([3,2])
    with cl:
        pivot      = pop.groupby(["catalog","schema"])["access_count"].sum().reset_index()
        pivot_wide = pivot.pivot(index="schema", columns="catalog", values="access_count").fillna(0)
        fig = go.Figure(go.Heatmap(
            z=pivot_wide.values, x=pivot_wide.columns.tolist(), y=pivot_wide.index.tolist(),
            colorscale=[[0,"#f0f4ff"],[0.5,ORANGE],[1,RED]],
            text=pivot_wide.values.astype(int), texttemplate="%{text}",
        ))
        fig.update_layout(height=380, margin=dict(l=0,r=0,t=10,b=0),
                          xaxis_title="Catalog", yaxis_title="Schema", title="Access Heatmap by Schema")
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        top10 = pop.nlargest(10,"access_count").sort_values("access_count")
        fig = px.bar(top10, x="access_count", y=top10["table"].str[:25], orientation="h",
                     color="unique_users", color_continuous_scale=[BLUE,ORANGE],
                     labels={"access_count":"Accesses","y":"","unique_users":"Users"},
                     title="Top 10 Hottest Tables")
        fig.update_layout(height=380, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    zombies = pop[pop["status"] == "🔴 Zombie"].sort_values("days_since_access", ascending=False)
    st.subheader(f"🗑️ Prune Candidates — {len(zombies)} tables unused for {unused_days}+ days")
    if len(zombies):
        st.dataframe(
            zombies[["table_name","catalog","schema","access_count","unique_users","days_since_access"]]
              .rename(columns={"table_name":"Full Name","catalog":"Catalog","schema":"Schema",
                               "access_count":"Accesses","unique_users":"Users","days_since_access":"Days Unused"})
              .style.format({"Accesses":"{:.0f}","Days Unused":"{:.0f}"}),
            use_container_width=True, hide_index=True, height=320,
        )
        st.warning(f"⚠️ Drop or archive {len(zombies)} zombie tables to reduce storage costs.")
    else:
        st.success("✅ No zombie tables with current threshold.")


# ══════════════════════════════════════════════════════════════════════════════
# WHAT-IF PROJECTIONS
# ══════════════════════════════════════════════════════════════════════════════
with t5:
    st.subheader("📊 What-If Projections")
    st.caption("Simulate cost impact of FinOps changes against billing telemetry")

    prev_m   = pd.Timestamp.now().month - 1 if pd.Timestamp.now().month > 1 else 12
    baseline = billing[billing["date"].dt.month == prev_m]["estimated_cost_usd"].sum()

    ca, cb, cc = st.columns(3)
    with ca:
        st.markdown("**Auto-Termination**")
        at_pct   = st.slider("% clusters covered", 0, 100, 40, 5, key="at")
        at_red   = st.slider("Idle reduction (%)",  5,  50, 20, 5, key="atr")
    with cb:
        st.markdown("**Spot Instances**")
        sp_pct   = st.slider("% workloads on Spot", 0, 80, 30, 5, key="sp")
        sp_disc  = st.slider("Spot discount (%)",   40, 80, 65, 5, key="spd")
    with cc:
        st.markdown("**Job vs All-Purpose**")
        jb_pct   = st.slider("% moved to Job clusters", 0, 80, 25, 5, key="jb")
        jb_save  = st.slider("Job cluster savings (%)",  10, 40, 30, 5, key="jbs")

    waste_pool = waste["estimated_cost_usd"].sum()
    s_at   = waste_pool * (at_pct/100) * (at_red/100)
    s_sp   = baseline   * (sp_pct/100) * (sp_disc/100)
    s_jb   = baseline   * (jb_pct/100) * (jb_save/100)
    total  = s_at + s_sp + s_jb
    proj   = max(0, baseline - total)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📌 Baseline",        f"${baseline:,.0f}/mo")
    c2.metric("💡 Projected",        f"${proj:,.0f}/mo",    f"-${total:,.0f}")
    c3.metric("📉 Monthly Savings",  f"${total:,.0f}",      f"{total/max(baseline,1)*100:.1f}%")
    c4.metric("📅 Annual Savings",   f"${total*12:,.0f}")

    st.markdown("---")
    cl, cr = st.columns([3,2])
    with cl:
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute","relative","relative","relative","total"],
            x=["Baseline","Auto-Termination","Spot Instances","Job Clusters","Projected"],
            y=[baseline, -s_at, -s_sp, -s_jb, proj],
            connector={"line":{"color":"rgb(63,63,63)"}},
            decreasing={"marker":{"color":GREEN}},
            totals={"marker":{"color":ORANGE}},
            text=[f"${abs(v):,.0f}" for v in [baseline,-s_at,-s_sp,-s_jb,proj]],
            textposition="outside",
        ))
        fig.update_layout(height=380, margin=dict(l=0,r=0,t=10,b=0),
                          yaxis_title="Monthly Cost (USD)", showlegend=False, title="Savings Waterfall")
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        bd = pd.DataFrame({"Lever":["Auto-Termination","Spot","Job Clusters"],"Saving":[s_at,s_sp,s_jb]})
        fig = px.pie(bd, values="Saving", names="Lever", hole=0.4,
                     color_discrete_sequence=[GREEN,BLUE,ORANGE], title="Savings Mix")
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(height=380, margin=dict(l=0,r=0,t=40,b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    growth  = st.slider("Monthly workload growth (%)", 0.0, 5.0, 1.5, 0.5, key="gr") / 100
    months  = pd.date_range(start=pd.Timestamp.now().replace(day=1), periods=12, freq="MS")
    proj_df = pd.DataFrame({
        "Month":     months,
        "Baseline":  [baseline * (1+growth)**i for i in range(12)],
        "Optimised": [proj     * (1+growth)**i for i in range(12)],
    })
    fig = px.line(proj_df.melt("Month",var_name="Scenario",value_name="Cost"),
                  x="Month", y="Cost", color="Scenario",
                  color_discrete_map={"Baseline":RED,"Optimised":GREEN},
                  markers=True, labels={"Cost":"Monthly Cost (USD)","Month":""},
                  title="12-Month Projection")
    fig.update_layout(height=300, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig, use_container_width=True)
    cum = sum((baseline - proj) * (1+growth)**i for i in range(12))
    st.success(f"🎯 Cumulative 12-month savings: **${cum:,.0f}**")


# ══════════════════════════════════════════════════════════════════════════════
# ESG TRACKING
# ══════════════════════════════════════════════════════════════════════════════
with t6:
    st.subheader("🌱 ESG & Sustainability Tracking")
    st.caption(f"1 DBU ≈ {KWH_PER_DBU} kWh · US grid avg {KG_CO2_PER_KWH} kg CO₂/kWh · Indicative figures only")

    esg               = billing.copy()
    esg["kwh"]        = esg["total_dbu"] * KWH_PER_DBU
    esg["kg_co2"]     = esg["kwh"] * KG_CO2_PER_KWH
    esg["trees_equiv"]= esg["kg_co2"] / (TREES_KG_CO2_PER_YEAR / 365)

    total_kwh   = esg["kwh"].sum()
    total_co2   = esg["kg_co2"].sum()
    total_trees = esg["trees_equiv"].sum()
    month_proj  = (total_co2 / 90) * 30
    budget_kg   = 5000
    vs_budget   = (month_proj - budget_kg) / budget_kg * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⚡ Energy (90d)",       f"{total_kwh/1000:.1f} MWh")
    c2.metric("💨 CO₂ (90d)",          f"{total_co2/1000:.2f} tCO₂")
    c3.metric("🌳 Trees to Offset",    f"{int(total_trees):,}")
    c4.metric("📅 Projected Monthly",  f"{month_proj/1000:.2f} tCO₂",
              f"{vs_budget:+.1f}% vs {budget_kg/1000:.0f}t budget",
              delta_color="inverse" if vs_budget > 0 else "normal")

    st.markdown("---")
    cl, cr = st.columns(2)
    with cl:
        daily = esg.groupby("date").agg(kg_co2=("kg_co2","sum")).reset_index()
        daily["7d_avg"] = daily["kg_co2"].rolling(7, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily["date"], y=daily["kg_co2"],  name="Daily CO₂",
                             marker_color=BLUE, opacity=0.6))
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["7d_avg"], name="7-day Avg",
                                 line=dict(color=ORANGE, width=2)))
        fig.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0),
                          yaxis_title="kg CO₂", legend=dict(orientation="h",y=-0.2),
                          title="Daily CO₂ Emissions (90 days)")
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        by_prod = esg.groupby("product")["kg_co2"].sum().reset_index()
        fig = px.pie(by_prod, values="kg_co2", names="product", hole=0.4,
                     color_discrete_sequence=[GREEN,BLUE,ORANGE,"#E9C46A"],
                     title="CO₂ by Product")
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(height=320, margin=dict(l=0,r=0,t=40,b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("♻️ Green Recommendations")
    for title, desc in [
        ("Spot/Preemptible instances",     "Optimises instance packing, reduces energy per workload by 15-25%."),
        ("Enable Photon acceleration",     "Same work faster = fewer cluster-hours = less energy. Up to 50% reduction."),
        ("Tight autoscale bounds",         "Idle nodes waste energy. Match min/max workers to actual workload patterns."),
        ("Delta OPTIMIZE + ZORDER",        "Fewer scans = shorter queries = lower carbon. Reduce scan time 30-70%."),
        ("Schedule in low-carbon regions", "us-west-2: 0.18 kg CO₂/kWh vs us-east-1: 0.39 kg CO₂/kWh."),
    ]:
        with st.container(border=True):
            st.markdown(f"**🌿 {title}**")
            st.caption(desc)
