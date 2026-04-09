"""Shared formatting and UI helpers for LakePulse."""
import pandas as pd
import plotly.graph_objects as go

BRAND_ORANGE = "#FF6B35"
BRAND_RED    = "#E63946"
BRAND_GREEN  = "#2DC653"
BRAND_BLUE   = "#457B9D"
BRAND_DARK   = "#1D3557"

PALETTE = [BRAND_ORANGE, BRAND_BLUE, BRAND_GREEN, "#A8DADC", "#F1FAEE", "#E9C46A"]


def fmt_currency(v: float) -> str:
    if v >= 1_000_000:
        return f"${v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"${v/1_000:.1f}K"
    return f"${v:.2f}"


def fmt_dbu(v: float) -> str:
    if v >= 1_000:
        return f"{v/1_000:.1f}K DBU"
    return f"{v:.1f} DBU"


def fmt_gb(v: float) -> str:
    if v >= 1_000:
        return f"{v/1_000:.1f} TB"
    return f"{v:.2f} GB"


def waste_severity(cost_usd: float) -> str:
    if cost_usd >= 500:
        return "🔴 Critical"
    if cost_usd >= 100:
        return "🟠 High"
    if cost_usd >= 20:
        return "🟡 Medium"
    return "🟢 Low"


def sla_risk_badge(prob: float) -> str:
    if prob >= 0.75:
        return "🔴 High Risk"
    if prob >= 0.45:
        return "🟠 Medium Risk"
    return "🟢 On Track"


def gauge(value: float, title: str, max_val: float = 100,
          threshold_warn: float = 60, threshold_crit: float = 80) -> go.Figure:
    color = BRAND_GREEN
    if value >= threshold_crit:
        color = BRAND_RED
    elif value >= threshold_warn:
        color = BRAND_ORANGE

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, max_val]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, threshold_warn],  "color": "#eaffea"},
                {"range": [threshold_warn, threshold_crit], "color": "#fff3cd"},
                {"range": [threshold_crit, max_val],        "color": "#ffe0e0"},
            ],
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def sparkline(series: pd.Series, color: str = BRAND_ORANGE) -> go.Figure:
    fig = go.Figure(go.Scatter(
        y=series, mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=f"rgba(255,107,53,0.15)",
    ))
    fig.update_layout(
        height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
