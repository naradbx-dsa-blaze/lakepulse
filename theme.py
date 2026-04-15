# ─── LakePulse Dark Theme ─────────────────────────────────────────────────────
# Edit values here — every color in the app is derived from this file.

C = {
    # ── Backgrounds ────────────────────────────────────────────────────────────
    "bg":          "#0B1220",   # page background
    "sidebar":     "#0E1828",   # left sidebar
    "card":        "#152234",   # card / panel background
    "card_alt":    "#1C2D42",   # slightly elevated card
    "input_bg":    "#1A2B3F",   # input / select background
    "border":      "#1E3A5F",   # card and input borders
    "divider":     "#1A2B3F",   # horizontal rules

    # ── Accents ────────────────────────────────────────────────────────────────
    "teal":        "#10D9C0",   # primary accent — active nav, connect btn
    "teal_dim":    "#0BA898",   # teal hover
    "blue":        "#3B82F6",   # secondary accent — info, SQL
    "orange":      "#F97316",   # waste / warning
    "red":         "#EF4444",   # critical / over-budget
    "green":       "#22C55E",   # savings / success / on-track
    "yellow":      "#EAB308",   # medium risk

    # ── Text ───────────────────────────────────────────────────────────────────
    "text":        "#E2E8F0",   # primary text
    "text_dim":    "#94A3B8",   # secondary / captions
    "text_muted":  "#4B6784",   # placeholder / very dim

    # ── Nav ────────────────────────────────────────────────────────────────────
    "nav_active":  "#10D9C0",   # active nav item text
    "nav_hover":   "#1C2D42",   # nav item hover background
    "nav_text":    "#7FA3BF",   # inactive nav text
}

# ── Plotly chart base config (apply with fig.update_layout(**CHART)) ──────────
CHART = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94A3B8", size=11, family="Inter, -apple-system, BlinkMacSystemFont, sans-serif"),
    margin=dict(l=8, r=8, t=36, b=8),
    height=300,
    legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
    coloraxis_colorbar=dict(tickfont=dict(color="#94A3B8")),
)

# ── Plotly color sequences ────────────────────────────────────────────────────
PAL_MAIN    = ["#10D9C0", "#3B82F6", "#F97316", "#EAB308", "#22C55E", "#EF4444"]
PAL_HEAT    = [[0, "#152234"], [0.5, "#F97316"], [1.0, "#EF4444"]]
PAL_SAVINGS = [[0, "#22C55E"], [1.0, "#10D9C0"]]

# ── Font ─────────────────────────────────────────────────────────────────────
FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
