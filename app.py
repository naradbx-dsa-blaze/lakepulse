"""
LakePulse — Native Databricks FinOps Suite
Framework : Dash + Gunicorn (4 workers × 4 gthreads = 16 concurrent requests)
Design    : Dark navy FinOps dashboard — no Streamlit
Data      : Real Databricks system tables only — connect screen required
"""
import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, no_update, ctx

from data import (
    get_billing_trend, get_dbu_waste, get_job_history,
    get_bottlenecks, get_data_popularity,
    CLOUD_DBU_DEFAULTS, DBU_PRICE_DEFAULT,
    validate_connection,
)
from theme import C, CHART, PAL_MAIN, PAL_HEAT, PAL_SAVINGS, FONT

# ─── App init ─────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="LakePulse",
    update_title=None,
)
server = app.server  # Gunicorn entry point

# ─── Pages ────────────────────────────────────────────────────────────────────
PAGES = [
    ("overview",    "Overview",            "bi-grid-1x2"),
    ("waste",       "DBU Waste Killer",    "bi-lightning-charge"),
    ("bottleneck",  "Bottleneck Detector", "bi-bug"),
    ("sla",         "SLA Oracle",          "bi-clock-history"),
    ("heatmap",     "Data Heatmap",        "bi-fire"),
    ("whatif",      "What-If Projections", "bi-sliders"),
    ("budget",      "Budget Tracker",      "bi-wallet2"),
]

# ─── Style helpers ─────────────────────────────────────────────────────────────
def _style(**kw):
    return kw

def card(children, style=None):
    s = {
        "backgroundColor": C["card"],
        "border": f"1px solid {C['border']}",
        "borderRadius": "8px",
        "padding": "16px",
        "marginBottom": "12px",
    }
    if style:
        s.update(style)
    return html.Div(children, style=s)

def kpi(label, value, delta=None, good_direction="up", icon=None, color=None):
    delta_color = C["text_dim"]
    if delta is not None:
        try:
            num = float(str(delta).replace("$","").replace(",","").replace("%","").replace("+",""))
            if good_direction == "up":
                delta_color = C["green"] if num >= 0 else C["red"]
            else:
                delta_color = C["green"] if num <= 0 else C["red"]
        except Exception:
            pass
    accent = color or C["teal"]
    return html.Div(
        style={
            "backgroundColor": C["card"],
            "border": f"1px solid {C['border']}",
            "borderLeft": f"3px solid {accent}",
            "borderRadius": "6px",
            "padding": "14px 16px",
        },
        children=[
            html.Div(label, style={"fontSize": "11px", "color": C["text_dim"],
                                   "textTransform": "uppercase", "letterSpacing": "0.6px",
                                   "marginBottom": "6px"}),
            html.Div(str(value), style={"fontSize": "22px", "fontWeight": "700",
                                        "color": C["text"], "lineHeight": "1"}),
            html.Div(str(delta) if delta is not None else "",
                     style={"fontSize": "11px", "color": delta_color, "marginTop": "4px"}),
        ],
    )

def section(title, subtitle=None):
    return html.Div([
        html.Div(title, style={"fontSize": "15px", "fontWeight": "700",
                               "color": C["text"], "marginBottom": "2px"}),
        html.Div(subtitle or "", style={"fontSize": "11px", "color": C["text_dim"],
                                        "marginBottom": "12px"}),
    ])

def chart_card(fig, height=300):
    fig.update_layout(**{**CHART, "height": height})
    return card(dcc.Graph(figure=fig, config={"displayModeBar": False},
                          style={"height": f"{height}px"}))

def _input(fid, placeholder="", value=""):
    return dcc.Input(
        id=fid, value=value, placeholder=placeholder, debounce=False,
        style={
            "width": "100%", "backgroundColor": C["input_bg"],
            "border": f"1px solid {C['border']}", "borderRadius": "5px",
            "color": C["text"], "padding": "8px 12px", "fontSize": "13px",
            "outline": "none", "boxSizing": "border-box",
        },
    )

def _num_input(fid, value, step=0.01, min_val=0.0):
    return dcc.Input(
        id=fid, type="number", value=value, min=min_val, step=step, debounce=True,
        style={
            "width": "100%", "backgroundColor": C["input_bg"],
            "border": f"1px solid {C['border']}", "borderRadius": "5px",
            "color": C["text"], "padding": "6px 10px", "fontSize": "12px",
            "outline": "none",
        },
    )

def _slider(fid, min_v, max_v, value, step=5, marks=None):
    return dcc.Slider(
        id=fid, min=min_v, max=max_v, value=value, step=step,
        marks=marks or {min_v: str(min_v), max_v: str(max_v)},
        tooltip={"placement": "bottom", "always_visible": False},
    )

def _table(df, height=350):
    return dash.dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        page_size=15,
        sort_action="native",
        style_table={"overflowX": "auto", "minWidth": "100%"},
        style_header={
            "backgroundColor": C["card_alt"], "color": C["teal"],
            "fontWeight": "700", "fontSize": "11px", "padding": "9px 12px",
            "border": f"1px solid {C['border']}",
            "textTransform": "uppercase", "letterSpacing": "0.4px",
        },
        style_data={
            "backgroundColor": C["card"], "color": C["text"],
            "fontSize": "12px", "padding": "8px 12px",
            "border": f"1px solid {C['border']}",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": C["card_alt"]},
        ],
        style_cell={"fontFamily": FONT, "textAlign": "left",
                    "whiteSpace": "nowrap", "overflow": "hidden",
                    "textOverflow": "ellipsis", "maxWidth": "220px"},
    )

# ─── Connect screen ───────────────────────────────────────────────────────────
connect_screen = html.Div(
    id="connect-screen",
    style={
        "position": "fixed", "inset": "0",
        "backgroundColor": C["bg"],
        "display": "flex", "alignItems": "center", "justifyContent": "center",
        "zIndex": "9999", "fontFamily": FONT,
    },
    children=[
        html.Div(
            style={
                "width": "420px",
                "backgroundColor": C["card"],
                "border": f"1px solid {C['border']}",
                "borderRadius": "12px",
                "padding": "40px 36px",
                "boxShadow": "0 24px 64px rgba(0,0,0,0.6)",
            },
            children=[
                # Logo
                html.Div([
                    html.Span("⚡", style={"fontSize": "32px"}),
                    html.Span(" LakePulse", style={
                        "fontSize": "26px", "fontWeight": "800",
                        "color": C["teal"], "letterSpacing": "-0.5px",
                    }),
                ], style={"marginBottom": "6px"}),
                html.Div("Native Databricks FinOps Suite",
                         style={"color": C["text_dim"], "fontSize": "13px",
                                "marginBottom": "28px"}),

                # Warehouse ID input
                html.Label("SQL Warehouse ID",
                            style={"fontSize": "12px", "fontWeight": "600",
                                   "color": C["text_dim"], "marginBottom": "6px",
                                   "display": "block", "textTransform": "uppercase",
                                   "letterSpacing": "0.5px"}),
                _input("wh-input", placeholder="e.g. d4ef05c5632d476b"),
                html.Div(
                    "Find it in Databricks → SQL Warehouses → Connection Details",
                    style={"fontSize": "11px", "color": C["text_muted"],
                           "marginTop": "5px", "marginBottom": "20px"},
                ),

                # Connect button
                html.Button(
                    "Connect to Workspace",
                    id="btn-connect",
                    n_clicks=0,
                    style={
                        "width": "100%", "padding": "11px",
                        "backgroundColor": C["teal"], "color": "#000",
                        "border": "none", "borderRadius": "6px",
                        "fontSize": "14px", "fontWeight": "700",
                        "cursor": "pointer", "letterSpacing": "0.3px",
                    },
                ),
                html.Div(id="connect-error",
                         style={"color": C["red"], "fontSize": "12px",
                                "marginTop": "10px", "minHeight": "18px"}),

                # System tables note
                html.Div(
                    style={"marginTop": "28px", "paddingTop": "20px",
                           "borderTop": f"1px solid {C['divider']}"},
                    children=[
                        html.Div("Reads from system tables:",
                                 style={"fontSize": "11px", "color": C["text_muted"],
                                        "marginBottom": "6px"}),
                        *[html.Div(t, style={"fontSize": "11px", "color": C["text_dim"],
                                             "fontFamily": "monospace", "marginBottom": "2px"})
                          for t in [
                              "system.billing.usage",
                              "system.compute.clusters",
                              "system.query.history",
                              "system.lakeflow.job_run_timeline",
                              "system.access.audit",
                          ]],
                    ],
                ),
            ],
        ),
    ],
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
def _nav_btn(page_id, label, icon, active=False):
    return html.Button(
        [html.I(className=f"bi {icon}", style={"marginRight": "9px", "fontSize": "14px"}),
         label],
        id=f"nav-{page_id}",
        n_clicks=0,
        style={
            "width": "100%", "textAlign": "left",
            "padding": "10px 16px",
            "backgroundColor": C["teal"] + "22" if active else "transparent",
            "color": C["teal"] if active else C["nav_text"],
            "border": "none",
            "borderLeft": f"3px solid {C['teal']}" if active else "3px solid transparent",
            "borderRadius": "0",
            "fontSize": "13px", "fontWeight": "600" if active else "400",
            "cursor": "pointer", "display": "flex", "alignItems": "center",
            "fontFamily": FONT,
        },
    )

sidebar = html.Div(
    id="sidebar",
    style={
        "width": "210px", "minWidth": "210px",
        "backgroundColor": C["sidebar"],
        "borderRight": f"1px solid {C['border']}",
        "height": "100vh", "display": "flex",
        "flexDirection": "column", "flexShrink": "0",
        "overflowY": "auto",
    },
    children=[
        # Logo
        html.Div(
            style={"padding": "20px 16px 16px",
                   "borderBottom": f"1px solid {C['border']}"},
            children=[
                html.Div([
                    html.Span("⚡", style={"fontSize": "18px"}),
                    html.Span(" LakePulse",
                              style={"fontSize": "16px", "fontWeight": "800",
                                     "color": C["teal"], "letterSpacing": "-0.3px"}),
                ]),
                html.Div("FinOps Suite",
                         style={"fontSize": "10px", "color": C["text_muted"],
                                "marginTop": "2px"}),
            ],
        ),
        # Workspace badge
        html.Div(id="sidebar-workspace",
                 style={"padding": "10px 16px",
                        "borderBottom": f"1px solid {C['border']}",
                        "fontSize": "11px", "color": C["text_muted"]}),
        # Nav items
        html.Div(
            [_nav_btn(pid, lbl, ico, active=(pid == "overview"))
             for pid, lbl, ico in PAGES],
            style={"marginTop": "8px", "flex": "1"},
        ),
        # Pricing panel
        html.Div(
            style={"padding": "14px 16px",
                   "borderTop": f"1px solid {C['border']}"},
            children=[
                html.Div("PRICING", style={"fontSize": "10px", "color": C["text_muted"],
                                           "letterSpacing": "0.8px", "marginBottom": "8px"}),
                html.Div("Cloud", style={"fontSize": "11px", "color": C["text_dim"],
                                         "marginBottom": "4px"}),
                dcc.Dropdown(
                    id="cloud-select",
                    options=[{"label": c, "value": c} for c in ["AWS", "Azure", "GCP"]],
                    value="AWS", clearable=False,
                    style={"fontSize": "12px", "marginBottom": "8px"},
                ),
                *[html.Div([
                    html.Div(f"{k}  $/DBU",
                             style={"fontSize": "10px", "color": C["text_dim"],
                                    "marginBottom": "2px"}),
                    _num_input(f"price-{k.lower().replace('_','-')}", v, 0.01),
                  ], style={"marginBottom": "6px"})
                  for k, v in CLOUD_DBU_DEFAULTS["AWS"].items()],
                html.Div("Discount %", style={"fontSize": "11px", "color": C["text_dim"],
                                               "marginTop": "6px", "marginBottom": "4px"}),
                _num_input("price-discount", 0, 1, 0),
                # Disconnect
                html.Button(
                    "Disconnect",
                    id="btn-disconnect",
                    n_clicks=0,
                    style={
                        "width": "100%", "marginTop": "12px",
                        "padding": "7px", "fontSize": "12px",
                        "backgroundColor": "transparent",
                        "color": C["text_dim"],
                        "border": f"1px solid {C['border']}",
                        "borderRadius": "5px", "cursor": "pointer",
                        "fontFamily": FONT,
                    },
                ),
            ],
        ),
    ],
)

# ─── Page skeletons (content filled by callbacks) ─────────────────────────────
def _page(page_id, children=None):
    return html.Div(
        id=f"page-{page_id}",
        style={"display": "block" if page_id == "overview" else "none",
               "padding": "20px 24px", "overflowY": "auto",
               "height": "100vh", "boxSizing": "border-box"},
        children=children or [html.Div(id=f"content-{page_id}")],
    )

# ─── Dashboard shell ─────────────────────────────────────────────────────────
dashboard = html.Div(
    id="dashboard",
    style={"display": "none", "height": "100vh", "overflow": "hidden", "fontFamily": FONT},
    children=[
        html.Link(rel="stylesheet",
                  href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"),
        html.Div(
            style={"display": "flex", "height": "100vh", "backgroundColor": C["bg"]},
            children=[
                sidebar,
                # Main scroll area
                html.Div(
                    style={"flex": "1", "overflowY": "auto", "minWidth": "0",
                           "backgroundColor": C["bg"]},
                    children=[
                        _page("overview"),
                        _page("waste",
                              children=[
                                  html.Div(id="waste-controls",
                                           style={"marginBottom": "14px"}),
                                  html.Div(id="content-waste"),
                              ]),
                        _page("bottleneck"),
                        _page("sla"),
                        _page("heatmap",
                              children=[
                                  html.Div(id="heatmap-controls",
                                           style={"marginBottom": "14px"}),
                                  html.Div(id="content-heatmap"),
                              ]),
                        _page("whatif",
                              children=[
                                  html.Div(id="whatif-controls",
                                           style={"marginBottom": "14px"}),
                                  html.Div(id="content-whatif"),
                              ]),
                        _page("budget",
                              children=[
                                  html.Div(id="budget-controls",
                                           style={"marginBottom": "14px"}),
                                  html.Div(id="content-budget"),
                              ]),
                    ],
                ),
            ],
        ),
        # Loading overlay while data fetches
        dcc.Loading(id="global-loader", type="circle", color=C["teal"],
                    children=html.Div(id="loader-target")),
    ],
)

# ─── Full layout ──────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={"fontFamily": FONT, "backgroundColor": C["bg"]},
    children=[
        dcc.Store(id="wh-store",     data=None),
        dcc.Store(id="data-store",   data=None),
        dcc.Store(id="prices-store", data=CLOUD_DBU_DEFAULTS["AWS"]),
        dcc.Store(id="active-page",  data="overview"),
        connect_screen,
        dashboard,
    ],
)

# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. Connect ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("wh-store",        "data"),
    Output("connect-screen",  "style"),
    Output("dashboard",       "style"),
    Output("connect-error",   "children"),
    Output("sidebar-workspace","children"),
    Input("btn-connect",      "n_clicks"),
    State("wh-input",         "value"),
    prevent_initial_call=True,
)
def connect(_, wh_id):
    if not wh_id or not wh_id.strip():
        return no_update, no_update, no_update, "Please enter a Warehouse ID.", no_update
    result = validate_connection(wh_id.strip())
    if not result["ok"]:
        return no_update, no_update, no_update, f"❌ {result['error']}", no_update
    host  = result.get("host", "").replace("https://", "")
    cloud = result.get("cloud", "AWS")
    badge = [
        html.Div("✓ Connected", style={"color": C["green"], "fontWeight": "600",
                                        "fontSize": "11px"}),
        html.Div(f"{cloud} · {host[:28]}{'…' if len(host) > 28 else ''}",
                 style={"color": C["text_muted"], "fontSize": "10px",
                        "marginTop": "2px", "wordBreak": "break-all"}),
    ]
    hide  = {"display": "none"}
    show  = {"display": "flex", "height": "100vh", "overflow": "hidden", "fontFamily": FONT}
    return wh_id.strip(), hide, show, "", badge

# ── 2. Disconnect ──────────────────────────────────────────────────────────────
@app.callback(
    Output("wh-store",       "data",  allow_duplicate=True),
    Output("data-store",     "data",  allow_duplicate=True),
    Output("connect-screen", "style", allow_duplicate=True),
    Output("dashboard",      "style", allow_duplicate=True),
    Input("btn-disconnect",  "n_clicks"),
    prevent_initial_call=True,
)
def disconnect(_):
    show_connect = {
        "position": "fixed", "inset": "0", "backgroundColor": C["bg"],
        "display": "flex", "alignItems": "center", "justifyContent": "center",
        "zIndex": "9999", "fontFamily": FONT,
    }
    return None, None, show_connect, {"display": "none"}

# ── 3. Load all data after connect ────────────────────────────────────────────
@app.callback(
    Output("data-store",     "data",     allow_duplicate=True),
    Output("loader-target",  "children"),
    Input("wh-store",        "data"),
    prevent_initial_call=True,
)
def load_all(wh_id):
    if not wh_id:
        return no_update, no_update
    billing    = get_billing_trend(wh_id)
    waste      = get_dbu_waste(wh_id)
    jobs       = get_job_history(wh_id)
    bottleneck = get_bottlenecks(wh_id)
    popularity = get_data_popularity(wh_id)
    # Coerce numeric columns
    for df, cols in [
        (billing,    ["total_dbu", "estimated_cost_usd"]),
        (waste,      ["total_dbu", "estimated_cost_usd", "lifetime_hours"]),
        (jobs,       ["duration_min", "queue_min"]),
        (bottleneck, ["duration_min", "shuffle_read_gb", "shuffle_write_gb", "peak_memory_gb"]),
        (popularity, ["access_count", "days_since_access", "unique_users"]),
    ]:
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return {
        "billing":    billing.assign(date=billing["date"].astype(str)).to_dict("records"),
        "waste":      waste.to_dict("records"),
        "jobs":       jobs.assign(trigger_time=jobs["trigger_time"].astype(str)).to_dict("records"),
        "bottleneck": bottleneck.to_dict("records"),
        "popularity": popularity.to_dict("records"),
    }, ""

# ── 4. Pricing store ───────────────────────────────────────────────────────────
@app.callback(
    Output("prices-store",         "data"),
    Input("cloud-select",           "value"),
    Input("price-all-purpose",      "value"),
    Input("price-jobs",             "value"),
    Input("price-dlt",              "value"),
    Input("price-sql",              "value"),
    Input("price-discount",         "value"),
    prevent_initial_call=False,
)
def update_prices(cloud, ap, jobs, dlt, sql, discount):
    defaults = CLOUD_DBU_DEFAULTS.get(cloud or "AWS", CLOUD_DBU_DEFAULTS["AWS"])
    disc = (discount or 0) / 100
    def _p(v, default):
        raw = float(v) if v is not None else default
        return round(raw * (1 - disc), 4)
    return {
        "ALL_PURPOSE": _p(ap,  defaults["ALL_PURPOSE"]),
        "JOBS":        _p(jobs, defaults["JOBS"]),
        "DLT":         _p(dlt,  defaults["DLT"]),
        "SQL":         _p(sql,  defaults["SQL"]),
        "DEFAULT":     round(DBU_PRICE_DEFAULT * (1 - disc), 4),
    }

# ── 5. Navigation ──────────────────────────────────────────────────────────────
_NAV_IDS     = [f"nav-{pid}" for pid, *_ in PAGES]
_PAGE_IDS    = [pid for pid, *_ in PAGES]
_NAV_OUTPUTS = [Output(f"nav-{pid}", "style") for pid, *_ in PAGES]
_PAGE_OUTPUTS= [Output(f"page-{pid}", "style") for pid, *_ in PAGES]

@app.callback(
    Output("active-page", "data"),
    *_NAV_OUTPUTS,
    *_PAGE_OUTPUTS,
    *[Input(f"nav-{pid}", "n_clicks") for pid, *_ in PAGES],
    prevent_initial_call=True,
)
def nav(*_clicks):
    triggered = ctx.triggered_id or "nav-overview"
    active_pid = triggered.replace("nav-", "")

    def _nav_style(pid):
        active = pid == active_pid
        return {
            "width": "100%", "textAlign": "left",
            "padding": "10px 16px",
            "backgroundColor": C["teal"] + "22" if active else "transparent",
            "color": C["teal"] if active else C["nav_text"],
            "border": "none",
            "borderLeft": f"3px solid {C['teal']}" if active else "3px solid transparent",
            "borderRadius": "0",
            "fontSize": "13px", "fontWeight": "600" if active else "400",
            "cursor": "pointer", "display": "flex", "alignItems": "center",
            "fontFamily": FONT,
        }

    def _page_style(pid):
        return {
            "display": "block" if pid == active_pid else "none",
            "padding": "20px 24px",
            "overflowY": "auto", "height": "100vh", "boxSizing": "border-box",
        }

    return (
        active_pid,
        *[_nav_style(pid) for pid, *_ in PAGES],
        *[_page_style(pid) for pid, *_ in PAGES],
    )

# ── Helper: recompute billing costs ────────────────────────────────────────────
def _reprice_billing(billing_records, prices):
    df = pd.DataFrame(billing_records)
    df["date"]              = pd.to_datetime(df["date"], errors="coerce")
    df["total_dbu"]         = pd.to_numeric(df["total_dbu"], errors="coerce").fillna(0)
    df["estimated_cost_usd"]= (
        df["total_dbu"] * df["product"].map({
            "ALL_PURPOSE": prices.get("ALL_PURPOSE", 0.55),
            "JOBS":        prices.get("JOBS", 0.20),
            "DLT":         prices.get("DLT", 0.36),
            "SQL":         prices.get("SQL", 0.22),
        }).fillna(prices.get("DEFAULT", 0.40))
    ).round(2)
    return df

# ══════════════════════════════════════════════════════════════════════════════
# PAGE RENDERS
# ══════════════════════════════════════════════════════════════════════════════

# ── Overview ──────────────────────────────────────────────────────────────────
@app.callback(
    Output("content-overview", "children"),
    Input("data-store",        "data"),
    Input("prices-store",      "data"),
    prevent_initial_call=True,
)
def render_overview(data, prices):
    if not data:
        return html.Div("Waiting for data…", style={"color": C["text_dim"], "padding": "20px"})

    billing    = _reprice_billing(data["billing"], prices)
    waste      = pd.DataFrame(data["waste"])
    jobs       = pd.DataFrame(data["jobs"])
    today      = pd.Timestamp.now()
    cur_m, cur_y = today.month, today.year
    prev_m = cur_m - 1 if cur_m > 1 else 12
    prev_y = cur_y if cur_m > 1 else cur_y - 1

    mtd   = billing[(billing["date"].dt.month == cur_m) & (billing["date"].dt.year == cur_y)]["estimated_cost_usd"].sum()
    prev  = billing[(billing["date"].dt.month == prev_m) & (billing["date"].dt.year == prev_y)]["estimated_cost_usd"].sum()
    waste["estimated_cost_usd"] = waste["total_dbu"].astype(float) * prices.get("ALL_PURPOSE", 0.55)
    waste_cost  = waste["estimated_cost_usd"].sum()
    fail_rate   = (jobs["result_state"].ne("SUCCEEDED")).mean() * 100 if len(jobs) else 0
    sql_spend   = billing[billing["product"] == "SQL"]["estimated_cost_usd"].sum()

    kpis = dbc.Row([
        dbc.Col(kpi("MTD Spend",            f"${mtd:,.0f}",
                    f"${mtd-prev:+,.0f} vs last month", "up",  color=C["teal"]), md=3),
        dbc.Col(kpi("Recoverable Waste",    f"${waste_cost:,.0f}",
                    f"{len(waste)} ALL_PURPOSE clusters", "down", color=C["orange"]), md=3),
        dbc.Col(kpi("Job Failure Rate",     f"{fail_rate:.1f}%",
                    f"{int(fail_rate/100*len(jobs))} failed runs", "down", color=C["red"]), md=3),
        dbc.Col(kpi("SQL Warehouse (90d)", f"${sql_spend:,.0f}",
                    "system.billing.usage", color=C["blue"]), md=3),
    ], className="mb-3 g-2")

    # Daily spend area chart
    daily = billing.groupby(["date","product"])["estimated_cost_usd"].sum().reset_index()
    fig1 = px.area(daily, x="date", y="estimated_cost_usd", color="product",
                   color_discrete_sequence=PAL_MAIN,
                   labels={"estimated_cost_usd":"Cost (USD)","date":"","product":"Product"},
                   title="Daily Spend by Product — Last 90 Days")
    fig1.update_layout(**{**CHART, "height": 280})

    # Spend mix pie
    by_prod = billing.groupby("product")["estimated_cost_usd"].sum().reset_index()
    fig2 = px.pie(by_prod, values="estimated_cost_usd", names="product", hole=0.45,
                  color_discrete_sequence=PAL_MAIN, title="Spend Mix")
    fig2.update_traces(textinfo="percent+label", textfont_size=11)
    fig2.update_layout(**{**CHART, "height": 280, "showlegend": False})

    # Top clusters bar
    top10 = waste.nlargest(10, "estimated_cost_usd").sort_values("estimated_cost_usd")
    fig3 = px.bar(top10, x="estimated_cost_usd", y=top10["cluster_name"].str[:30],
                  orientation="h", color="estimated_cost_usd",
                  color_continuous_scale=PAL_HEAT,
                  labels={"estimated_cost_usd":"USD","y":""},
                  title="Top 10 Clusters by Cost — Last 30 Days")
    fig3.update_layout(**{**CHART, "height": 300, "coloraxis_showscale": False})

    return html.Div([
        section("Overview",
                "system.billing.usage · system.compute.clusters · system.lakeflow.job_run_timeline"),
        kpis,
        dbc.Row([
            dbc.Col(card(dcc.Graph(figure=fig1, config={"displayModeBar":False})), md=8),
            dbc.Col(card(dcc.Graph(figure=fig2, config={"displayModeBar":False})), md=4),
        ], className="g-2 mb-2"),
        card(dcc.Graph(figure=fig3, config={"displayModeBar":False})),
    ])

# ── DBU Waste Killer ──────────────────────────────────────────────────────────
@app.callback(
    Output("waste-controls", "children"),
    Input("data-store", "data"),
    prevent_initial_call=True,
)
def render_waste_controls(data):
    if not data: return no_update
    return card([
        html.Div("Min cost threshold (USD)", style={"fontSize":"12px","color":C["text_dim"],"marginBottom":"6px"}),
        dcc.Slider(id="waste-slider", min=0, max=500, value=20, step=10,
                   marks={0:"$0",100:"$100",250:"$250",500:"$500"},
                   tooltip={"placement":"bottom","always_visible":False}),
    ])

@app.callback(
    Output("content-waste", "children"),
    Input("data-store",     "data"),
    Input("prices-store",   "data"),
    Input("waste-slider",   "value"),
    prevent_initial_call=True,
)
def render_waste(data, prices, min_cost):
    if not data: return html.Div("Waiting…", style={"color": C["text_dim"]})
    min_cost = min_cost or 0
    waste = pd.DataFrame(data["waste"])
    waste["estimated_cost_usd"] = (waste["total_dbu"].astype(float) * prices.get("ALL_PURPOSE", 0.55)).round(2)
    df = waste[waste["estimated_cost_usd"] >= min_cost].copy()

    def _sev(v):
        return "Critical" if v >= 500 else ("High" if v >= 100 else ("Medium" if v >= 20 else "Low"))
    def _act(r):
        if r["estimated_cost_usd"] >= 500: return "Terminate immediately"
        if r["estimated_cost_usd"] >= 100: return "Set autotermination ≤ 30 min"
        return "Review utilisation pattern"
    df["Severity"] = df["estimated_cost_usd"].apply(_sev)
    df["Action"]   = df.apply(_act, axis=1)

    kpis = dbc.Row([
        dbc.Col(kpi("Recoverable Spend",  f"${df['estimated_cost_usd'].sum():,.0f}", color=C["orange"]), md=4),
        dbc.Col(kpi("Critical (≥ $500)",  str(int((df["estimated_cost_usd"] >= 500).sum())), color=C["red"]), md=4),
        dbc.Col(kpi("Total DBU Consumed", f"{df['total_dbu'].astype(float).sum():,.0f}", color=C["blue"]), md=4),
    ], className="mb-3 g-2")

    top = df.nlargest(15, "estimated_cost_usd").sort_values("estimated_cost_usd")
    fig1 = px.bar(top, x="estimated_cost_usd", y=top["cluster_name"].str[:28], orientation="h",
                  color="estimated_cost_usd", color_continuous_scale=PAL_HEAT,
                  labels={"estimated_cost_usd":"USD","y":""}, title="Cost by Cluster")
    fig1.update_layout(**{**CHART, "coloraxis_showscale": False, "height": 340})

    by_owner = df.groupby("owner")["estimated_cost_usd"].sum().reset_index().sort_values("estimated_cost_usd", ascending=False)
    fig2 = px.bar(by_owner, x="owner", y="estimated_cost_usd", color="estimated_cost_usd",
                  color_continuous_scale=PAL_HEAT,
                  labels={"estimated_cost_usd":"USD","owner":""}, title="Waste by Owner")
    fig2.update_layout(**{**CHART, "coloraxis_showscale": False, "height": 340})

    kill_df = df[["cluster_name","owner","sku_name","total_dbu","lifetime_hours","estimated_cost_usd","Severity","Action"]].copy()
    kill_df = kill_df.rename(columns={"cluster_name":"Cluster","owner":"Owner","sku_name":"Instance",
                                       "total_dbu":"DBU","lifetime_hours":"Lifetime (h)",
                                       "estimated_cost_usd":"Cost (USD)"})
    kill_df["DBU"]        = kill_df["DBU"].apply(lambda x: f"{float(x):.1f}")
    kill_df["Cost (USD)"] = kill_df["Cost (USD)"].apply(lambda x: f"${float(x):,.2f}")

    return html.Div([
        section("DBU Waste Killer",
                "system.billing.usage × system.compute.clusters · ALL_PURPOSE clusters · Last 30 days"),
        kpis,
        dbc.Row([
            dbc.Col(card(dcc.Graph(figure=fig1, config={"displayModeBar":False})), md=7),
            dbc.Col(card(dcc.Graph(figure=fig2, config={"displayModeBar":False})), md=5),
        ], className="g-2 mb-2"),
        card([
            html.Div("Kill List", style={"fontSize":"14px","fontWeight":"700","color":C["text"],"marginBottom":"10px"}),
            _table(kill_df),
            html.Div(f"Acting on this list recovers ${df['estimated_cost_usd'].sum():,.0f} over the next 30 days.",
                     style={"fontSize":"12px","color":C["orange"],"marginTop":"10px"}),
        ]),
    ])

# ── Bottleneck Detector ───────────────────────────────────────────────────────
@app.callback(
    Output("content-bottleneck", "children"),
    Input("data-store",          "data"),
    prevent_initial_call=True,
)
def render_bottleneck(data):
    if not data: return html.Div("Waiting…", style={"color": C["text_dim"]})
    bot = pd.DataFrame(data["bottleneck"])

    kpis = dbc.Row([
        dbc.Col(kpi("Queries Analysed",   str(len(bot)), color=C["blue"]), md=4),
        dbc.Col(kpi("Total Shuffle Spill",f"{bot['shuffle_read_gb'].sum():.1f} GB", color=C["orange"]), md=4),
        dbc.Col(kpi("Slowest Query",      f"{bot['duration_min'].max():.0f} min", color=C["red"]), md=4),
    ], className="mb-3 g-2")

    def _fix(r):
        if r["shuffle_read_gb"] > 20:
            return "Severe Spill — check cartesian join / skew; enable AQE"
        if r["shuffle_read_gb"] > 5:
            return "High Shuffle — enable AQE; REPARTITION hint or salt skewed keys"
        if r["peak_memory_gb"] > 16:
            return "Memory Pressure — increase executor memory or broadcast small table"
        return "Monitor"
    bot["Fix"] = bot.apply(_fix, axis=1)

    fig1 = px.scatter(bot, x="duration_min", y="shuffle_read_gb", size="peak_memory_gb",
                      color="executed_by", hover_data=["query_snippet"],
                      color_discrete_sequence=PAL_MAIN,
                      labels={"duration_min":"Duration (min)","shuffle_read_gb":"Shuffle Read (GB)"},
                      title="Shuffle Spill vs Duration")
    fig1.add_hline(y=10, line_dash="dot", line_color=C["red"], annotation_text="10 GB — severe spill",
                   annotation_font_color=C["red"])
    fig1.update_layout(**{**CHART, "height": 320})

    fig2 = px.histogram(bot, x="shuffle_read_gb", nbins=15,
                        color_discrete_sequence=[C["blue"]],
                        labels={"shuffle_read_gb":"Shuffle Read (GB)"},
                        title="Shuffle Read Distribution")
    fig2.update_layout(**{**CHART, "height": 320})

    tbl = bot.nlargest(15, "shuffle_read_gb")[
        ["query_snippet","executed_by","duration_min","shuffle_read_gb","shuffle_write_gb","peak_memory_gb","Fix"]
    ].rename(columns={"query_snippet":"Query","executed_by":"User","duration_min":"Duration (min)",
                       "shuffle_read_gb":"Shuffle Read (GB)","shuffle_write_gb":"Written (GB)",
                       "peak_memory_gb":"Peak Mem (GB)"})
    tbl["Duration (min)"]    = tbl["Duration (min)"].apply(lambda x: f"{float(x):.1f}")
    tbl["Shuffle Read (GB)"] = tbl["Shuffle Read (GB)"].apply(lambda x: f"{float(x):.2f}")

    return html.Div([
        section("Bottleneck Detector",
                "system.query.history · Queries > 60s · shuffle_read_bytes, spilled_local_bytes · Last 7 days"),
        kpis,
        dbc.Row([
            dbc.Col(card(dcc.Graph(figure=fig1, config={"displayModeBar":False})), md=7),
            dbc.Col(card(dcc.Graph(figure=fig2, config={"displayModeBar":False})), md=5),
        ], className="g-2 mb-2"),
        card([html.Div("Top Offenders", style={"fontSize":"14px","fontWeight":"700","color":C["text"],"marginBottom":"10px"}),
              _table(tbl)]),
    ])

# ── SLA Oracle ────────────────────────────────────────────────────────────────
@app.callback(
    Output("content-sla", "children"),
    Input("data-store",   "data"),
    prevent_initial_call=True,
)
def render_sla(data):
    if not data: return html.Div("Waiting…", style={"color": C["text_dim"]})
    jobs = pd.DataFrame(data["jobs"])
    jobs["trigger_time"] = pd.to_datetime(jobs["trigger_time"], errors="coerce")
    jobs["duration_min"] = pd.to_numeric(jobs["duration_min"], errors="coerce").fillna(0)
    jobs["queue_min"]    = pd.to_numeric(jobs["queue_min"],    errors="coerce").fillna(0)
    df = jobs.dropna(subset=["duration_min","trigger_time"]).copy()

    p75 = df.groupby("job_name")["duration_min"].quantile(0.75).rename("p75")
    df  = df.join(p75, on="job_name")
    df["is_late"]     = (df["duration_min"] > df["p75"]).astype(int)
    df["hour"]        = df["trigger_time"].dt.hour
    df["dow"]         = df["trigger_time"].dt.dayofweek
    df["rolling_avg"] = df.groupby("job_name")["duration_min"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    le = LabelEncoder()
    df["job_enc"] = le.fit_transform(df["job_name"].fillna("unknown"))
    feats = ["job_enc","hour","dow","queue_min","rolling_avg"]
    X = df[feats].fillna(0); y = df["is_late"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    df["delay_prob"] = clf.predict_proba(X)[:, 1]
    accuracy = clf.score(X_te, y_te)
    imps = pd.Series(clf.feature_importances_, index=feats).sort_values(ascending=False).reset_index()
    imps.columns = ["Feature","Importance"]
    imps["Feature"] = imps["Feature"].map({
        "job_enc":"Job Identity","rolling_avg":"Recent Avg Duration",
        "queue_min":"Queue Time","hour":"Hour of Day","dow":"Day of Week",
    })

    high   = int((df["delay_prob"] >= 0.75).sum())
    medium = int(((df["delay_prob"] >= 0.45) & (df["delay_prob"] < 0.75)).sum())
    ontime = int((df["delay_prob"] < 0.45).sum())

    kpis = dbc.Row([
        dbc.Col(kpi("Model Accuracy",    f"{accuracy:.1%}", color=C["teal"]), md=3),
        dbc.Col(kpi("High Risk (≥75%)",  str(high),         color=C["red"]),    md=3),
        dbc.Col(kpi("Medium Risk (≥45%)",str(medium),       color=C["yellow"]), md=3),
        dbc.Col(kpi("On Track",          str(ontime),       color=C["green"]),  md=3),
    ], className="mb-3 g-2")

    fig1 = px.histogram(df, x="delay_prob", nbins=20, color_discrete_sequence=[C["orange"]],
                        labels={"delay_prob":"Delay Probability"}, title="Delay Probability Distribution")
    fig1.add_vline(x=0.45, line_dash="dot", line_color=C["yellow"], annotation_text="Medium",
                   annotation_font_color=C["yellow"])
    fig1.add_vline(x=0.75, line_dash="dot", line_color=C["red"],    annotation_text="High",
                   annotation_font_color=C["red"])
    fig1.update_layout(**{**CHART, "height": 300})

    fig2 = px.bar(imps, x="Importance", y="Feature", orientation="h",
                  color="Importance", color_continuous_scale=PAL_SAVINGS,
                  title="Feature Importance")
    fig2.update_layout(**{**CHART, "height": 300, "coloraxis_showscale": False,
                          "yaxis": {"autorange":"reversed"}})

    at_risk = df[df["delay_prob"] >= 0.45].sort_values("delay_prob", ascending=False).head(20).copy()
    at_risk["Risk"] = at_risk["delay_prob"].apply(
        lambda p: "High Risk" if p >= 0.75 else "Medium Risk"
    )
    tbl = at_risk[["job_name","trigger_type","trigger_time","duration_min","p75","delay_prob","Risk"]].copy()
    tbl["trigger_time"] = tbl["trigger_time"].dt.strftime("%Y-%m-%d %H:%M")
    tbl["duration_min"] = tbl["duration_min"].apply(lambda x: f"{x:.1f}")
    tbl["delay_prob"]   = tbl["delay_prob"].apply(lambda x: f"{x:.1%}")
    tbl = tbl.rename(columns={"job_name":"Job","trigger_type":"Trigger","trigger_time":"Started",
                               "duration_min":"Duration (min)","p75":"SLA p75 (min)","delay_prob":"Delay Prob"})

    return html.Div([
        section("SLA Oracle",
                "system.lakeflow.job_run_timeline · RandomForest · SLA = p75 of historical duration per job"),
        kpis,
        dbc.Row([
            dbc.Col(card(dcc.Graph(figure=fig1, config={"displayModeBar":False})), md=6),
            dbc.Col(card(dcc.Graph(figure=fig2, config={"displayModeBar":False})), md=6),
        ], className="g-2 mb-2"),
        card([html.Div("At-Risk Runs", style={"fontSize":"14px","fontWeight":"700","color":C["text"],"marginBottom":"10px"}),
              _table(tbl)]),
    ])

# ── Data Heatmap ──────────────────────────────────────────────────────────────
@app.callback(
    Output("heatmap-controls", "children"),
    Input("data-store", "data"),
    prevent_initial_call=True,
)
def render_heatmap_controls(data):
    if not data: return no_update
    return card([
        html.Div("Mark unused after (days without access)", style={"fontSize":"12px","color":C["text_dim"],"marginBottom":"6px"}),
        dcc.Slider(id="heatmap-slider", min=7, max=90, value=30, step=7,
                   marks={7:"7d",30:"30d",60:"60d",90:"90d"},
                   tooltip={"placement":"bottom","always_visible":False}),
    ])

@app.callback(
    Output("content-heatmap", "children"),
    Input("data-store",       "data"),
    Input("heatmap-slider",   "value"),
    prevent_initial_call=True,
)
def render_heatmap(data, unused_days):
    if not data: return html.Div("Waiting…", style={"color": C["text_dim"]})
    unused_days = unused_days or 30
    pop = pd.DataFrame(data["popularity"])
    pop["catalog"] = pop["table_name"].str.split(".").str[0].fillna("unknown")
    pop["schema"]  = pop["table_name"].str.split(".").str[1].fillna("unknown")
    pop["table"]   = pop["table_name"].str.split(".").str[-1].fillna("unknown")
    pop["status"]  = pop.apply(
        lambda r: "Zombie"  if r["days_since_access"] >= unused_days and r["access_count"] < 5
        else ("Cooling"     if r["days_since_access"] >= unused_days
        else ("Hot"         if r["access_count"] > 100 else "Active")), axis=1,
    )

    kpis = dbc.Row([
        dbc.Col(kpi("Tables Tracked",    str(len(pop)), color=C["blue"]), md=4),
        dbc.Col(kpi("Unused Tables",     str(int((pop["days_since_access"] >= unused_days).sum())), color=C["orange"]), md=4),
        dbc.Col(kpi("Hot Tables (>100)", str(int((pop["access_count"] > 100).sum())), color=C["teal"]), md=4),
    ], className="mb-3 g-2")

    pivot      = pop.groupby(["catalog","schema"])["access_count"].sum().reset_index()
    pivot_wide = pivot.pivot(index="schema", columns="catalog", values="access_count").fillna(0)
    fig1 = go.Figure(go.Heatmap(
        z=pivot_wide.values, x=pivot_wide.columns.tolist(), y=pivot_wide.index.tolist(),
        colorscale=PAL_HEAT, text=pivot_wide.values.astype(int), texttemplate="%{text}",
    ))
    fig1.update_layout(**{**CHART, "height": 350,
                          "xaxis_title":"Catalog","yaxis_title":"Schema",
                          "title":"Access Count by Catalog × Schema"})

    top10 = pop.nlargest(10, "access_count").sort_values("access_count")
    fig2 = px.bar(top10, x="access_count", y=top10["table"].str[:22], orientation="h",
                  color="unique_users", color_continuous_scale=["#3B82F6","#F97316"],
                  labels={"access_count":"Access Events","y":"","unique_users":"Users"},
                  title="Top 10 Most Accessed Tables")
    fig2.update_layout(**{**CHART, "height": 350})

    zombies = pop[pop["status"] == "Zombie"].sort_values("days_since_access", ascending=False)
    tbl_df  = zombies[["table_name","catalog","schema","access_count","unique_users","days_since_access"]].rename(
        columns={"table_name":"Full Name","catalog":"Catalog","schema":"Schema",
                 "access_count":"Access Events","unique_users":"Unique Users",
                 "days_since_access":"Days Since Access"})

    return html.Div([
        section("Data Popularity Heatmap",
                "system.access.audit · getTable + selectFromTable + createTableAsSelect · Last 90 days"),
        kpis,
        dbc.Row([
            dbc.Col(card(dcc.Graph(figure=fig1, config={"displayModeBar":False})), md=7),
            dbc.Col(card(dcc.Graph(figure=fig2, config={"displayModeBar":False})), md=5),
        ], className="g-2 mb-2"),
        card([
            html.Div(f"Prune Candidates — {len(zombies)} tables with < 5 accesses in {unused_days}+ days",
                     style={"fontSize":"14px","fontWeight":"700","color":C["orange"],"marginBottom":"10px"}),
            _table(tbl_df) if len(zombies) else
            html.Div(f"No unused tables found with {unused_days}-day threshold.",
                     style={"color":C["green"],"padding":"16px","fontSize":"13px"}),
        ]),
    ])

# ── What-If ───────────────────────────────────────────────────────────────────
@app.callback(
    Output("whatif-controls", "children"),
    Input("data-store", "data"),
    prevent_initial_call=True,
)
def render_whatif_controls(data):
    if not data: return no_update
    def _lbl(txt): return html.Div(txt, style={"fontSize":"11px","color":C["text_dim"],"marginBottom":"4px"})
    return card([
        dbc.Row([
            dbc.Col([
                html.Div("Auto-Termination", style={"fontSize":"12px","fontWeight":"600","color":C["teal"],"marginBottom":"8px"}),
                _lbl("% clusters with ≤ 30 min auto-stop"),
                dcc.Slider(id="wi-at-pct",  min=0, max=100, value=40, step=5,
                           marks={0:"0%",50:"50%",100:"100%"},
                           tooltip={"placement":"bottom","always_visible":False}),
                _lbl("Expected idle time reduction (%)"),
                dcc.Slider(id="wi-at-red",  min=5,  max=50, value=20, step=5,
                           marks={5:"5%",25:"25%",50:"50%"},
                           tooltip={"placement":"bottom","always_visible":False}),
            ], md=4),
            dbc.Col([
                html.Div("Spot Instances", style={"fontSize":"12px","fontWeight":"600","color":C["teal"],"marginBottom":"8px"}),
                _lbl("% workloads on Spot/Preemptible"),
                dcc.Slider(id="wi-sp-pct",  min=0, max=80, value=30, step=5,
                           marks={0:"0%",40:"40%",80:"80%"},
                           tooltip={"placement":"bottom","always_visible":False}),
                _lbl("Spot discount vs on-demand (%)"),
                dcc.Slider(id="wi-sp-disc", min=50, max=90, value=70, step=5,
                           marks={50:"50%",70:"70%",90:"90%"},
                           tooltip={"placement":"bottom","always_visible":False}),
            ], md=4),
            dbc.Col([
                html.Div("Job Clusters", style={"fontSize":"12px","fontWeight":"600","color":C["teal"],"marginBottom":"8px"}),
                _lbl("% AP workloads → Job clusters"),
                dcc.Slider(id="wi-jb-pct",  min=0, max=80, value=25, step=5,
                           marks={0:"0%",40:"40%",80:"80%"},
                           tooltip={"placement":"bottom","always_visible":False}),
                _lbl("Job cluster savings vs AP (%)"),
                dcc.Slider(id="wi-jb-save", min=10, max=60, value=35, step=5,
                           marks={10:"10%",35:"35%",60:"60%"},
                           tooltip={"placement":"bottom","always_visible":False}),
            ], md=4),
        ], className="g-2"),
    ])

@app.callback(
    Output("content-whatif", "children"),
    Input("data-store",   "data"),
    Input("prices-store", "data"),
    Input("wi-at-pct",    "value"),
    Input("wi-at-red",    "value"),
    Input("wi-sp-pct",    "value"),
    Input("wi-sp-disc",   "value"),
    Input("wi-jb-pct",    "value"),
    Input("wi-jb-save",   "value"),
    prevent_initial_call=True,
)
def render_whatif(data, prices, at_pct, at_red, sp_pct, sp_disc, jb_pct, jb_save):
    if not data: return html.Div("Waiting…", style={"color": C["text_dim"]})
    billing = _reprice_billing(data["billing"], prices)
    waste   = pd.DataFrame(data["waste"])
    waste["estimated_cost_usd"] = (waste["total_dbu"].astype(float) * prices.get("ALL_PURPOSE", 0.55)).round(2)

    today  = pd.Timestamp.now()
    prev_m = today.month - 1 if today.month > 1 else 12
    prev_y = today.year if today.month > 1 else today.year - 1
    baseline = billing[
        (billing["date"].dt.month == prev_m) & (billing["date"].dt.year == prev_y)
    ]["estimated_cost_usd"].sum() or billing["estimated_cost_usd"].sum() / 3

    waste_pool = waste["estimated_cost_usd"].sum()
    s_at   = waste_pool * ((at_pct or 0) / 100) * ((at_red  or 0) / 100)
    s_sp   = baseline   * ((sp_pct or 0) / 100) * ((sp_disc or 0) / 100)
    s_jb   = baseline   * ((jb_pct or 0) / 100) * ((jb_save or 0) / 100)
    total  = s_at + s_sp + s_jb
    proj   = max(0, baseline - total)

    kpis = dbc.Row([
        dbc.Col(kpi("Baseline (last month)",   f"${baseline:,.0f}", color=C["blue"]),   md=3),
        dbc.Col(kpi("Projected Monthly Spend", f"${proj:,.0f}",     f"-${total:,.0f}", "down", color=C["teal"]), md=3),
        dbc.Col(kpi("Monthly Savings",         f"${total:,.0f}",    f"{total/max(baseline,1)*100:.1f}%", "up", color=C["green"]), md=3),
        dbc.Col(kpi("Annual Savings",          f"${total*12:,.0f}", color=C["green"]), md=3),
    ], className="mb-3 g-2")

    fig1 = go.Figure(go.Waterfall(
        orientation="v", measure=["absolute","relative","relative","relative","total"],
        x=["Baseline","Auto-Termination","Spot","Job Clusters","Projected"],
        y=[baseline, -s_at, -s_sp, -s_jb, proj],
        connector={"line":{"color":C["border"]}},
        decreasing={"marker":{"color":C["green"]}},
        totals={"marker":{"color":C["orange"]}},
        text=[f"${abs(v):,.0f}" for v in [baseline,-s_at,-s_sp,-s_jb,proj]],
        textposition="outside",
    ))
    fig1.update_layout(**{**CHART,"height":360,"yaxis_title":"Monthly Cost (USD)","showlegend":False,
                          "title":"Savings Waterfall"})

    months = pd.date_range(start=pd.Timestamp.now().replace(day=1), periods=12, freq="MS")
    proj_df = pd.DataFrame({
        "Month":     [str(m.date()) for m in months],
        "Baseline":  [baseline * 1.015**i for i in range(12)],
        "Optimised": [proj    * 1.015**i for i in range(12)],
    }).melt("Month", var_name="Scenario", value_name="Cost")
    fig2 = px.line(proj_df, x="Month", y="Cost", color="Scenario", markers=True,
                   color_discrete_map={"Baseline":C["red"],"Optimised":C["green"]},
                   title="12-Month Cost Projection")
    fig2.update_layout(**{**CHART,"height":280})

    return html.Div([
        section("What-If Projections",
                "system.billing.usage · Savings modelled from Databricks best practices"),
        kpis,
        dbc.Row([
            dbc.Col(card(dcc.Graph(figure=fig1, config={"displayModeBar":False})), md=6),
            dbc.Col(card(dcc.Graph(figure=fig2, config={"displayModeBar":False})), md=6),
        ], className="g-2"),
    ])

# ── Budget Tracker ─────────────────────────────────────────────────────────────
@app.callback(
    Output("budget-controls", "children"),
    Input("data-store", "data"),
    prevent_initial_call=True,
)
def render_budget_controls(data):
    if not data: return no_update
    billing  = _reprice_billing(data["billing"], CLOUD_DBU_DEFAULTS["AWS"])
    today    = pd.Timestamp.now()
    prev_m   = today.month - 1 if today.month > 1 else 12
    prev_y   = today.year if today.month > 1 else today.year - 1
    prev_spend = billing[
        (billing["date"].dt.month == prev_m) & (billing["date"].dt.year == prev_y)
    ]["estimated_cost_usd"].sum()
    default_budget = max(1000, int(prev_spend * 1.1))

    return card(dbc.Row([
        dbc.Col([
            html.Div("Monthly Budget (USD)", style={"fontSize":"11px","color":C["text_dim"],"marginBottom":"4px"}),
            _num_input("budget-amount", default_budget, 500, 0),
        ], md=4),
        dbc.Col([
            html.Div("Alert threshold (% of budget)", style={"fontSize":"11px","color":C["text_dim"],"marginBottom":"4px"}),
            dcc.Slider(id="budget-alert", min=50, max=100, value=85, step=5,
                       marks={50:"50%",85:"85%",100:"100%"},
                       tooltip={"placement":"bottom","always_visible":False}),
        ], md=8),
    ]))

@app.callback(
    Output("content-budget", "children"),
    Input("data-store",    "data"),
    Input("prices-store",  "data"),
    Input("budget-amount", "value"),
    Input("budget-alert",  "value"),
    prevent_initial_call=True,
)
def render_budget(data, prices, monthly_budget, alert_at):
    if not data: return html.Div("Waiting…", style={"color": C["text_dim"]})
    billing = _reprice_billing(data["billing"], prices)
    today   = pd.Timestamp.now()
    cur_m, cur_y = today.month, today.year
    monthly_budget = monthly_budget or 10000
    alert_at       = alert_at or 85

    cur = billing[(billing["date"].dt.month == cur_m) & (billing["date"].dt.year == cur_y)]
    mtd        = cur["estimated_cost_usd"].sum()
    daily_rate = mtd / max(today.day, 1)
    proj_eom   = daily_rate * 30
    remaining  = max(0, monthly_budget - mtd)
    vs_bgt_pct = (proj_eom - monthly_budget) / max(monthly_budget, 1) * 100

    if proj_eom > monthly_budget:
        alert = html.Div(
            f"⚠ Over budget — on track to spend ${proj_eom:,.0f} vs ${monthly_budget:,.0f} budget (+${proj_eom-monthly_budget:,.0f})",
            style={"backgroundColor":"#EF444422","border":f"1px solid {C['red']}","borderRadius":"6px",
                   "padding":"10px 14px","color":C["red"],"fontSize":"13px","marginBottom":"12px"},
        )
    elif proj_eom / max(monthly_budget,1) * 100 >= alert_at:
        alert = html.Div(
            f"Approaching limit — projected to use {proj_eom/max(monthly_budget,1)*100:.0f}% of ${monthly_budget:,.0f}",
            style={"backgroundColor":"#F9731622","border":f"1px solid {C['orange']}","borderRadius":"6px",
                   "padding":"10px 14px","color":C["orange"],"fontSize":"13px","marginBottom":"12px"},
        )
    else:
        alert = html.Div(
            f"✓ On track — projected to use {proj_eom/max(monthly_budget,1)*100:.0f}% of ${monthly_budget:,.0f}",
            style={"backgroundColor":"#22C55E22","border":f"1px solid {C['green']}","borderRadius":"6px",
                   "padding":"10px 14px","color":C["green"],"fontSize":"13px","marginBottom":"12px"},
        )

    kpis = dbc.Row([
        dbc.Col(kpi("MTD Spend",          f"${mtd:,.0f}",          f"Day {today.day}", color=C["blue"]),    md=3),
        dbc.Col(kpi("Daily Run Rate",     f"${daily_rate:,.0f}/d",                     color=C["teal"]),    md=3),
        dbc.Col(kpi("Projected Month-End",f"${proj_eom:,.0f}",      f"{vs_bgt_pct:+.1f}% vs budget",
                    "down",  color=C["orange"] if vs_bgt_pct > 0 else C["green"]), md=3),
        dbc.Col(kpi("Budget Remaining",   f"${remaining:,.0f}",                        color=C["green"]),   md=3),
    ], className="mb-3 g-2")

    daily_agg = cur.groupby("date")["estimated_cost_usd"].sum().reset_index().sort_values("date")
    daily_agg["Cumulative"] = daily_agg["estimated_cost_usd"].cumsum()
    n = len(daily_agg)
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=daily_agg["date"], y=daily_agg["estimated_cost_usd"],
                          name="Daily", marker_color=C["blue"], opacity=0.7))
    fig1.add_trace(go.Scatter(x=daily_agg["date"], y=daily_agg["Cumulative"],
                              name="Cumulative", line=dict(color=C["orange"],width=2), yaxis="y2"))
    if n:
        pace_x = daily_agg["date"].tolist()
        pace_y = [monthly_budget / 30 * (i+1) for i in range(n)]
        fig1.add_trace(go.Scatter(x=pace_x, y=pace_y, name="Budget Pace",
                                  line=dict(color=C["red"],dash="dot",width=1.5), yaxis="y2"))
    fig1.update_layout(**{**CHART,"height":300,"title":"Current Month — Daily Spend vs Budget Pace",
                          "yaxis":{"title":"Daily (USD)"},
                          "yaxis2":{"title":"Cumulative (USD)","overlaying":"y","side":"right"}})

    prod_mtd = cur.groupby("product")["estimated_cost_usd"].sum().reset_index().sort_values("estimated_cost_usd")
    prod_mtd["% Budget"] = (prod_mtd["estimated_cost_usd"] / max(monthly_budget,1) * 100).round(1)
    fig2 = px.bar(prod_mtd, x="estimated_cost_usd", y="product", orientation="h",
                  color="estimated_cost_usd", color_continuous_scale=PAL_HEAT,
                  text=prod_mtd["% Budget"].astype(str) + "%",
                  labels={"estimated_cost_usd":"MTD Spend (USD)","product":""},
                  title="MTD Spend by Product")
    fig2.update_traces(textposition="outside")
    fig2.update_layout(**{**CHART,"height":300,"coloraxis_showscale":False})

    monthly = (
        billing.assign(month=billing["date"].dt.to_period("M").astype(str))
        .groupby("month")["estimated_cost_usd"].sum().reset_index()
    )
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=monthly["month"], y=monthly["estimated_cost_usd"],
                          name="Monthly Spend", marker_color=C["blue"]))
    fig3.add_hline(y=monthly_budget, line_dash="dot", line_color=C["red"],
                   annotation_text=f"Budget ${monthly_budget:,}", annotation_font_color=C["red"])
    fig3.update_layout(**{**CHART,"height":260,"yaxis_title":"USD",
                          "title":"Monthly Spend vs Budget — Last 90 Days"})

    return html.Div([
        section("Budget Tracker",
                "system.billing.usage · Projections based on current month daily run rate"),
        kpis, alert,
        dbc.Row([
            dbc.Col(card(dcc.Graph(figure=fig1, config={"displayModeBar":False})), md=7),
            dbc.Col(card(dcc.Graph(figure=fig2, config={"displayModeBar":False})), md=5),
        ], className="g-2 mb-2"),
        card(dcc.Graph(figure=fig3, config={"displayModeBar":False})),
    ])

# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("DATABRICKS_APP_PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
