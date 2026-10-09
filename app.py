"""
Keepp Capital — Target Allocation Monte Carlo
Streamlit front end, client edition.

Written for clients without a finance or maths background: results are told
in money and plain words first, with the technical figures one click away.
Every number still comes from the same engine / optimizer modules — this file
contains layout only.

Deliberately makes ZERO external network requests at runtime: no Google Fonts,
no CDN, no remote data. Everything is served by the app itself so it loads
normally from mainland China without a VPN. Do not add a webfont <link>, a CDN
<script>, or a remote image to this file — any one of them will hang the page
load behind the Great Firewall.
"""

import hmac

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import jpm_ltcma as jpm
import optimizer as opt
import run_store as rs
import universe as uni
from engine import simulate

# The line the cash floor applies to. Named once so the optimizer panel and the
# risk-free default cannot drift apart from each other.
CASH_LINE = "U.S. Cash"

# --------------------------------------------------------------------------
# Page config + palette
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Keepp · Portfolio Outlook",
    page_icon="◧",
    layout="wide",
    initial_sidebar_state="expanded",
)

INK = "#0F1719"
INK_2 = "#3F4D4F"
INK_3 = "#6C7C7D"
ACCENT = "#00877A"
ACCENT_DK = "#00655B"
BAND_OUTER = "#D3E9E5"
BAND_INNER = "#8FC7BE"
BRICK = "#BC3D18"
DEEP_BLUE = "#1C5CAB"
GRID = "#E3EAE9"
SURFACE = "#FCFDFD"

# Categorical order for holdings (validated: CVD-safe on the light surface).
# Fixed order, never cycled — a ninth holding folds into "Other".
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER_COLOR = "#A9B4B5"
UP_COLOR = "#2a78d6"      # diverging pair for "more / less of this"
DOWN_COLOR = "#eb6834"

# System font stacks only. "Microsoft YaHei" / "PingFang SC" keep the UI sane on
# a Chinese-locale OS without fetching anything.
SANS = ('-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, '
        '"Helvetica Neue", "PingFang SC", "Microsoft YaHei", Arial, sans-serif')

st.markdown(f"""
<style>
  html, body, [class*="css"], .stApp {{ font-family: {SANS}; }}
  .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1240px; }}
  div[data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}

  .eyebrow {{ font-size: .72rem; letter-spacing: .14em; text-transform: uppercase;
             color: {ACCENT}; font-weight: 600; margin-bottom: .3rem; }}
  .lede {{ color: {INK_3}; font-size: 1rem; max-width: 70ch; line-height: 1.55;
          margin-top: -.4rem; }}
  .muted {{ color: {INK_3}; font-size: .85rem; line-height: 1.55; }}
  .foot {{ color: {INK_3}; font-size: .78rem; line-height: 1.6; }}
  .sect {{ font-size: 1.12rem; font-weight: 650; margin: 1.8rem 0 .2rem; color: {INK}; }}
  .sectsub {{ color: {INK_3}; font-size: .86rem; margin-bottom: .8rem; }}

  .stMarkdown h1 {{ font-size: 2rem !important; font-weight: 700 !important; }}

  /* Tabs: larger, calmer */
  button[data-baseweb="tab"] p, [data-testid="stTab"] p,
  [role="tab"] p {{ font-size: 1rem !important; font-weight: 600; }}
  [role="tablist"] {{ gap: 1.4rem; }}

  /* Hero */
  .hero {{ background: linear-gradient(135deg, #E9F4F2 0%, #F6FAF9 70%);
          border: 1px solid #D6E8E4; border-radius: 16px; padding: 26px 30px; }}
  .hero .q {{ color: {INK_2}; font-size: 1.02rem; }}
  .hero .big {{ font-size: 3rem; font-weight: 700; color: {ACCENT_DK};
               letter-spacing: -.02em; line-height: 1.1; margin: .25rem 0 .35rem;
               font-variant-numeric: tabular-nums; }}
  .hero .big small {{ font-size: 1.1rem; font-weight: 600; color: {ACCENT};
                     margin-left: .6rem; letter-spacing: 0; }}
  .hero .s {{ color: {INK_2}; font-size: .95rem; line-height: 1.55; max-width: 72ch; }}

  /* Cards */
  .card {{ background: #fff; border: 1px solid {GRID}; border-radius: 14px;
          padding: 16px 18px; height: 100%; }}
  .card .lbl {{ color: {INK_3}; font-size: .82rem; font-weight: 550;
               display: flex; align-items: center; gap: 7px; }}
  .card .val {{ font-size: 1.85rem; font-weight: 700; color: {INK}; line-height: 1.15;
               margin: 6px 0 4px; font-variant-numeric: tabular-nums; letter-spacing: -.01em; }}
  .card .val.hot {{ color: {BRICK}; }}
  .card .sub {{ color: {INK_3}; font-size: .8rem; line-height: 1.45; }}
  .card .arrow {{ color: {INK_3}; font-weight: 400; font-size: 1.2rem; margin: 0 .25rem; }}
  .card .was {{ color: {INK_3}; font-weight: 500; font-size: 1.15rem; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  .scen {{ border-top: 4px solid var(--c); }}

  /* Risk meter */
  .meter {{ display: flex; gap: 4px; margin: 10px 0 6px; }}
  .meter span {{ flex: 1; height: 8px; border-radius: 4px; background: #E6ECEB; }}
  .meter span.on {{ background: var(--c); }}
  .meterlbl {{ display: flex; justify-content: space-between; color: {INK_3};
              font-size: .7rem; }}

  /* Holdings legend */
  .stMarkdown table.leg, table.leg {{ width: 100%; border-collapse: collapse;
             font-size: .88rem; border: none !important; background: transparent; }}
  table.leg tr, table.leg tr:nth-child(even) {{ background: transparent !important; }}
  table.leg td {{ padding: 7px 4px !important; border: none !important;
                 border-bottom: 1px solid #EEF2F1 !important; vertical-align: top; }}
  .leg td.w {{ text-align: right; font-weight: 650; white-space: nowrap;
              font-variant-numeric: tabular-nums; }}
  .leg .d {{ color: {INK_3}; font-size: .76rem; }}
  .sw {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block;
        margin-top: 3px; }}

  /* Chart legend strip */
  .keys {{ display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: .8rem;
          color: {INK_2}; margin-top: -.2rem; }}
  .keys span {{ display: inline-flex; align-items: center; gap: 7px; }}
  .keys i {{ width: 16px; height: 10px; border-radius: 2px; display: inline-block; }}

  /* Total bar on the mix tab */
  .total {{ border-radius: 12px; padding: 12px 16px; font-size: .92rem;
           display: flex; justify-content: space-between; align-items: center; }}
  .total.ok {{ background: #E9F4F2; color: {ACCENT_DK}; }}
  .total {{ background: #F1F4F4; color: {INK_2}; }}
  .total.bad {{ background: #FBEFEA; color: {BRICK}; }}
  .bar {{ height: 6px; background: #E6ECEB; border-radius: 3px; margin-top: 8px;
         overflow: hidden; }}
  .bar > div {{ height: 100%; border-radius: 3px; }}

  /* Asset rows on Your mix: the tick box's own label is the asset name */
  .stCheckbox label p {{ font-weight: 650; font-size: .95rem; }}
  /* A held asset's tick is locked ON, and Streamlit dims a disabled label —
     which made the holdings look switched off. Keep the ink, keep the mark
     filled; the cursor and the tooltip still say it cannot be changed. */
  [data-testid="stCheckbox"] label[data-disabled="true"] p {{
      color: {INK}; opacity: 1; }}
  [data-testid="stCheckbox"] label[data-disabled="true"] {{ opacity: 1;
      cursor: default; }}
  [data-testid="stCheckbox"] label[data-disabled="true"][data-selected="true"] svg {{
      opacity: 1; }}
  .d2 {{ color: {INK_3}; font-size: .78rem; line-height: 1.45;
        margin: 2px 0 2px 28px; }}
  .d2 .sw {{ vertical-align: -1px; }}

  .sleeve {{ padding-top: 6px; }}
  .sleeve .n {{ font-weight: 650; font-size: .95rem; display: flex; gap: 8px;
               align-items: center; }}
  .sleeve .d {{ color: {INK_3}; font-size: .78rem; line-height: 1.4; margin-left: 20px; }}

  .note {{ border-left: 3px solid {INK_3}; background: #F4F7F7; padding: 10px 14px;
          border-radius: 6px; font-size: .84rem; line-height: 1.6; color: {INK_2}; }}

  .modechip {{ display: inline-block; font-size: .66rem; font-weight: 600;
              letter-spacing: .08em; text-transform: uppercase; padding: 3px 9px;
              border-radius: 20px; vertical-align: middle; margin-left: 10px;
              background: #F3E7DF; color: {BRICK}; }}

  .srcname {{ font-weight: 600; font-size: .85rem; }}
  .srcmini {{ font-size: .72rem; color: {INK_3}; }}
  .srctext {{ font-size: .82rem; line-height: 1.65; opacity: .85; }}

  section[data-testid="stSidebar"] {{ width: 330px !important; }}
  section[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}

  @media (max-width: 640px) {{
    .hero {{ padding: 18px; }}
    .hero .big {{ font-size: 2.1rem; }}
    .card .val {{ font-size: 1.45rem; }}
  }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Optional password gate (set app_password in .streamlit/secrets.toml, or in
# Streamlit Community Cloud under Manage app -> Settings -> Secrets)
# --------------------------------------------------------------------------
def password_ok() -> bool:
    # st.secrets raises if no secrets.toml exists at all, so treat "no secrets
    # file" the same as "no password configured" rather than crashing the app.
    try:
        secret = st.secrets.get("app_password", "")
    except Exception:
        secret = ""
    if not secret:
        return True                      # no password configured -> open
    if st.session_state.get("_authed"):
        return True

    st.markdown("### Keepp Portfolio Outlook")
    st.caption("Enter the access password to continue.")
    pw = st.text_input("Password", type="password", label_visibility="collapsed")
    if pw:
        if hmac.compare_digest(pw, str(secret)):
            st.session_state["_authed"] = True
            st.rerun()
        else:
            st.error("That password is not correct.")
    return False


if not password_ok():
    st.stop()


# --------------------------------------------------------------------------
# Base case — J.P. Morgan 2026 LTCMA. Returns, volatilities and the sleeve list
# are hardcoded and cannot be edited in Base; the WEIGHTS can.
# See git history for the full provenance notes on each sleeve.
# --------------------------------------------------------------------------
CATEGORY_KEYS = jpm.CATEGORY_KEYS                   # ["FI", "EQ", "ALT"]
CATEGORIES = [jpm.CATEGORY_NAME[k] for k in CATEGORY_KEYS]
KEY_OF = dict(zip(CATEGORIES, CATEGORY_KEYS))

# Returns are COMPOUND (geometric): engine.py converts with ln(1+g)/12.
BASE_SLEEVES = [
    # category, LTCMA line,                 weight, return,   vol,   basis
    ("FI",  "U.S. Inv Grade Corporate Bonds", 45.0,   5.00,   7.02,
     "Keepp · global IG blend"),
    ("FI",  "U.S. Cash",                      17.0,   3.80,   0.67,
     "Keepp · 3M T-bill"),
    ("EQ",  "AC World Equity",                20.0,      7,   16.78,
     "Keepp · MSCI World 10y"),
    ("ALT", "Private Equity",                 10.0,  10.20,  19.78,  ""),
    ("ALT", "Diversified Hedge Funds",         0.0,   5.30,   5.75,  ""),
    ("ALT", "Commodities",                     8.0,   4.60,  18.32,  ""),
]

MODE_BASE = "Base · JPM 2026 LTCMA"
MODE_CUSTOM = "Custom"

# Plain-language names for the holdings a client is most likely to see.
# Anything not listed falls back to J.P. Morgan's own line name.
FRIENDLY = {
    "U.S. Inv Grade Corporate Bonds": (
        "High-quality bonds",
        "Loans to large, financially strong companies. Steady income."),
    "U.S. Cash": (
        "Cash",
        "Short-term government bills. The safest, lowest-return holding."),
    "AC World Equity": (
        "Global stocks",
        "Shares in companies around the world. Main source of growth."),
    "Private Equity": (
        "Private companies",
        "Stakes in companies not listed on a stock exchange. Long-term money."),
    "Diversified Hedge Funds": (
        "Hedge funds",
        "Actively managed funds that aim for steadier returns."),
    "Commodities": (
        "Commodities",
        "Raw materials such as energy, metals and crops. Helps against inflation."),
}
CATEGORY_PLAIN = {
    "Fixed Income": "Bonds · steadier income",
    "Equities": "Stocks · growth with bigger swings",
    "Alternatives": "Alternatives · diversifiers",
}

SOURCES = [
    ("Fixed Income", "5.00% · 7.02%",
     "Return: index YTM of global IG corporates — iShares Global Corp Bond UCITS trailing "
     "distribution yield 4.26% plus pull-to-par, cross-checked against JPM 2026 LTCMA US IG "
     "at 5.2%. Volatility: regime anchor √(25%·3.3%² + 50%·6.1%² + 25%·10.6%²) = 7.02%."),
    ("Public Equities &amp; Mutual Funds", "12.88% · 14.47%",
     "Return: J.P. Morgan 2026 LTCMA, AC World Equity, "
     "Volatility:  "
     "J.P. Morgan 2026 LTCMA, AC World Equity"),
    ("Private Equity &amp; Venture Capital", "10.20% · 19.78%",
     "J.P. Morgan 2026 LTCMA, Private Equity (USD), <b>compound</b> return. The "
     "original workbook used 11.90%, which is JPM's <i>arithmetic</i> figure for the "
     "same line — feeding that into a field the model treats as geometric "
     "double-counts the variance drag. Corrected here."),
    ("Diversified Hedge Funds", "5.30% · 5.75%",
     "J.P. Morgan 2026 LTCMA, diversified hedge funds (USD) — leveraged and net of "
     "fees. Base weight is 0% because the deck's target allocation has no hedge fund "
     "sleeve — allocate to it in Custom mode."),
    ("Commodities &amp; Real Assets", "4.60% · 18.32%",
     "J.P. Morgan 2026 LTCMA, broad commodities basket (USD), <b>compound</b> return. "
     "As with Private Equity, the original workbook used the arithmetic figure "
     "(6.15%). Corrected here."),
    ("Cash &amp; Equivalents", "3.80% · 0.67%",
     "Return: current USD risk-free rate, 3-month T-bill at 3.80% (15-Jul-2026). "
     "Volatility: JPM 2026 LTCMA US Cash. Includes the 2% digital-assets sleeve, "
     "folded into cash per instruction."),
    ("Sub-categories", "3 categories · 58 lines",
     "Sleeves are grouped into J.P. Morgan's own three categories — Fixed Income "
     "(21 lines), Equities (18) and Alternatives (19). Picking a line fills Return "
     "and Vol from the <b>compound</b> column; the line also supplies the sleeve's "
     "row of the correlation matrix."),
    ("Correlations", "JPM 2026 LTCMA, USD",
     "<b>The full published 59 × 59 matrix</b>, shipped unmodified as "
     "<code>jpm_correlations.csv</code>. The matrix used in any run is a sub-matrix "
     "of it. As published it is not positive semi-definite (smallest eigenvalue "
     "−0.0161); it is repaired once at load with Higham's nearest-correlation "
     "algorithm, and no pair moves as much as 0.01."),
    ("Keepp-sourced figures", "Custom mode only",
     "Three lines carried Keepp's own return and volatility rather than "
     "J.P. Morgan's: a global IG blend (5.00% · 7.02%), MSCI World realised "
     "history (7.00% · 16.78%) and the live 3-month T-bill (3.80% · 0.67%). "
     "<b>Standard mode no longer uses them</b> — every one of its 58 asset "
     "classes now takes J.P. Morgan's published compound figure, so the whole "
     "table comes from one source. They remain as the starting numbers in "
     "<b>Custom</b>, where any figure can be typed over. The two that differ "
     "from J.P. Morgan are U.S. Inv Grade Corporate Bonds (5.00% vs 5.20%) "
     "and U.S. Cash (3.80% vs 3.10%); AC World Equity is the same either "
     "way."),
    ("Method", "Regime-based vol",
     "Returns are entered as geometric per annum and simulated as correlated lognormal "
     "monthly draws, with the portfolio rebalanced to target weights every month. Vol "
     "anchors come from 120 months of returns split into calm / normal / stress regimes by "
     "the bottom quartile, middle 50% and top quartile of 6-month rolling MSCI World "
     "volatility, weighted 25 / 50 / 25."),
]


# --------------------------------------------------------------------------
# Formatting helpers — every client-facing number goes through these
# --------------------------------------------------------------------------
def money(x: float) -> str:
    """$1.97M, $845K, $1.20B — three significant figures, no clutter."""
    a = abs(x)
    sign = "−" if x < 0 else ""
    if a >= 1e9:
        return f"{sign}${a/1e9:.2f}B"
    if a >= 1e6:
        return f"{sign}${a/1e6:.2f}M"
    if a >= 1e3:
        return f"{sign}${a/1e3:,.0f}K"
    return f"{sign}${a:,.0f}"


def pct(x: float, dp: int = 1, sign: bool = False) -> str:
    s = f"{x*100:+.{dp}f}%" if sign else f"{x*100:.{dp}f}%"
    return s.replace("-", "−")


def one_in(p: float) -> str:
    if p <= 0:
        return "no simulated case"
    if p < 0.001:
        return "less than 1 in 1,000"
    if p >= 0.5:
        return f"{p*100:.0f} in 100"
    return f"about 1 in {round(1/p):,}"


RISK_LEVELS = [(0.04, "Very low"), (0.08, "Low"), (0.12, "Moderate"),
               (0.16, "High"), (9.99, "Very high")]
RISK_COLORS = ["#1baf7a", "#1baf7a", "#eda100", "#eb6834", BRICK]


def risk_level(vol: float):
    for i, (cut, name) in enumerate(RISK_LEVELS):
        if vol < cut:
            return i, name
    return len(RISK_LEVELS) - 1, RISK_LEVELS[-1][1]


def plain_name(line_label: str):
    """(display name, one-line description) for an LTCMA line label.

    The NAME is always J.P. Morgan's own, so the row, the dropdown that added
    it, the donut and Custom mode all say the same thing — with 58 lines on
    offer, a row reading "Global stocks" while the picker says "AC World
    Equity" is a worse trade than plainer wording. The plain-English
    description below the name survives for the lines that have one.
    """
    name = jpm.resolve(line_label)
    cat = jpm.category_of(line_label)
    cat_name = jpm.CATEGORY_NAME.get(cat, "") if cat else ""
    desc = (FRIENDLY[name][1] if name in FRIENDLY
            else CATEGORY_PLAIN.get(cat_name, ""))
    return (jpm.LABEL.get(name, name) if name else str(line_label), desc)


def card(label, value, sub="", hot=False, dot=None, extra_cls="", style=""):
    d = f'<span class="dot" style="background:{dot}"></span>' if dot else ""
    return (f'<div class="card {extra_cls}" style="{style}">'
            f'<div class="lbl">{d}{label}</div>'
            f'<div class="val{" hot" if hot else ""}">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


def section(title, sub=None):
    st.markdown(f'<div class="sect">{title}</div>'
                + (f'<div class="sectsub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


def color_map(names):
    """Fixed-order colour per holding. Up to 8 distinct; beyond that the tail
    folds into 'Other' so no hue is ever generated or cycled."""
    names = list(dict.fromkeys(names))
    keep = names if len(names) <= len(SERIES) else names[:len(SERIES) - 1]
    return {n: SERIES[i] for i, n in enumerate(keep)}


def holdings_frame(names, weights, cmap):
    """Donut-ready rows, tail folded into 'Other', zero weights dropped."""
    rows, other = [], 0.0
    for n, w in zip(names, weights):
        if w <= 1e-6:
            continue
        if n in cmap:
            rows.append({"name": n, "w": float(w), "color": cmap[n]})
        else:
            other += float(w)
    if other > 1e-6:
        rows.append({"name": "Other", "w": other, "color": OTHER_COLOR})
    df = pd.DataFrame(rows, columns=["name", "w", "color"])
    df["order"] = range(len(df))
    df["lbl"] = df["w"].map(lambda v: f"{v:.1f}%")
    return df


def donut(df, center_top, center_bottom, size=240):
    if df.empty:
        return None
    r_out = size / 2 - 6
    base = alt.Chart(df).encode(
        theta=alt.Theta("w:Q", stack=True),
        order=alt.Order("order:Q"),
        color=alt.Color("name:N", legend=None,
                        scale=alt.Scale(domain=list(df["name"]),
                                        range=list(df["color"]))),
        tooltip=[alt.Tooltip("name:N", title="Holding"),
                 alt.Tooltip("lbl:N", title="Share")])
    ring = base.mark_arc(innerRadius=r_out * 0.62, outerRadius=r_out,
                         stroke="#ffffff", strokeWidth=2, cornerRadius=3)
    c1 = alt.Chart(pd.DataFrame([{"t": center_top}])).mark_text(
        fontSize=22, fontWeight=700, color=INK, dy=-6).encode(text="t:N")
    c2 = alt.Chart(pd.DataFrame([{"t": center_bottom}])).mark_text(
        fontSize=11, color=INK_3, dy=14).encode(text="t:N")
    return (ring + c1 + c2).properties(width=size, height=size).configure_view(
        strokeWidth=0)


def legend_html(df, descs=None):
    rows = []
    for _, r in df.iterrows():
        d = (descs or {}).get(r["name"], "")
        rows.append(
            f'<tr><td style="width:18px"><span class="sw" style="background:{r["color"]}">'
            f'</span></td><td>{r["name"]}'
            + (f'<div class="d">{d}</div>' if d else "")
            + f'</td><td class="w">{r["w"]:.1f}%</td></tr>')
    return f'<table class="leg">{"".join(rows)}</table>'


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
COLS = ["LTCMA line", "Weight %", "Return %", "Vol %"]
# BASE_SLEEVES is now only the seed for CUSTOM mode (custom_alloc_seed) and
# the provenance notes below. Standard no longer presets anything.
BASE_WEIGHTS = [w for _, _, w, _, _, _ in BASE_SLEEVES]

def std_numbers(line_name):
    """(compound return %, vol %) Standard mode uses for one LTCMA line.

    J.P. Morgan's published COMPOUND column for all 58 lines, with no
    exceptions — the three Keepp overrides that used to sit on top of it went
    with the preset sleeves, so every line in Standard now comes from one
    source and nothing is editable. Never the arithmetic column: engine.py
    treats this field as geometric and derives mu = g + sigma^2/2 itself.
    (Custom mode still seeds the deck's own figures; see custom_alloc_seed.)
    """
    row = jpm.BY_NAME[line_name]
    return float(row[0]), float(row[2])


def std_alloc(lines, weights) -> pd.DataFrame:
    """Standard-mode allocation frame, ordered by J.P. Morgan category."""
    rows = []
    for key in CATEGORY_KEYS:
        for ln in lines:
            if jpm.category_of(ln) != key:
                continue
            r, v = std_numbers(ln)
            rows.append({"Category": jpm.CATEGORY_NAME[key],
                         "LTCMA line": jpm.LABEL[ln],
                         "Weight %": float(weights.get(ln, 0.0)),
                         "Return %": r, "Vol %": v})
    return pd.DataFrame(rows, columns=["Category", "LTCMA line", "Weight %",
                                       "Return %", "Vol %"])


def seed_standard():
    """Standard starts EMPTY — no preset sleeves.

    Every portfolio is built by picking lines out of the three category
    dropdowns, so nothing is assumed about what the client holds.
    """
    return ([], {}, {})


def custom_alloc_seed() -> pd.DataFrame:
    return pd.DataFrame(
        [{"Category": jpm.CATEGORY_NAME[k], "LTCMA line": jpm.LABEL[line],
          "Weight %": w, "Return %": r, "Vol %": v}
         for k, line, w, r, v, _ in BASE_SLEEVES])


def sleeve_inputs(alloc: pd.DataFrame):
    corr = corr_of(alloc)
    return (tuple(alloc["Weight %"]), tuple(alloc["Return %"]),
            tuple(alloc["Vol %"]), corr)


def corr_of(alloc: pd.DataFrame):
    """Published correlation sub-matrix — shared by simulation and optimizer."""
    return jpm.corr_for(list(alloc["LTCMA line"]))


def bump_editors():
    """Force every keyed input widget to remount after a programmatic write.

    st.data_editor keeps edits as a delta and re-applies them over whatever
    frame it is handed (the old 94.71% Normalize bug); sliders keep their own
    value. A nonce in every key gives each a clean slate.
    """
    st.session_state.editor_nonce = st.session_state.get("editor_nonce", 0) + 1


if "custom_alloc" not in st.session_state:
    st.session_state.editor_nonce = 0
    st.session_state.custom_alloc = custom_alloc_seed()
    # Standard mode: an ordered list of LTCMA lines, a weight for each, and a
    # tick saying whether the optimizer may use it. Anything above 0% is
    # always ticked; a 0% line can be ticked to offer it to the optimizer
    # without holding it today.
    (st.session_state.std_lines, st.session_state.std_w,
     st.session_state.std_tick) = seed_standard()
    st.session_state.mode = MODE_BASE

if "saved_runs" not in st.session_state:
    st.session_state.saved_runs = []

NONCE = st.session_state.get("editor_nonce", 0)

# A mode switch requested by "Apply as Custom sleeves" on the previous run. It
# must land before the radio that owns the `mode` key is instantiated.
if "pending_mode" in st.session_state:
    st.session_state.mode = st.session_state.pop("pending_mode")


@st.cache_data(show_spinner=False, max_entries=32)
def run_sim(w, r, v, corr_flat, paths, years, seed):
    corr = np.array(corr_flat, dtype=float).reshape(len(w), len(w))
    return simulate(list(w), list(r), list(v), corr, paths, years, seed)


STEP_PP = 0.01


@st.cache_data(show_spinner=False, max_entries=16)
def feasible_span(metric, cache_key, _mu, _cov, _limits):
    """Achievable range of a target return / volatility budget. Cached.
    `cache_key` MUST change whenever the feasible set does."""
    fn = opt.return_span if metric == "return" else opt.vol_span
    return fn(_mu, _cov, _limits)


# --------------------------------------------------------------------------
# Sidebar — the only three questions a client needs to answer
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="eyebrow">Keepp Capital</div>', unsafe_allow_html=True)
    st.markdown("### Your plan")

    amount = st.number_input(
        "How much are you investing? (USD)", min_value=1_000, max_value=10_000_000_000,
        value=1_000_000, step=50_000, format="%d",
        help="Used only to show results in dollars. It does not change the "
             "percentages.")
    st.caption(f"= **${amount:,.0f}**")
    years = st.slider("For how many years?", 1, 30, 10,
                      help="How long the money stays invested.")

    st.write("")
    # Always visible — the manager's most-used switch, so it is not hidden in
    # the expander. The widget key stays "mode" so pending_mode still works.
    mode = st.radio(
        "Market assumptions", [MODE_BASE, MODE_CUSTOM], key="mode",
        format_func=lambda m: ("Standard" if m == MODE_BASE else "Custom"),
        captions=["J.P. Morgan 2026 house view — only the mix can change",
                  "Edit holdings, returns and risk figures"],
        help="Standard keeps Keepp's signed-off assumptions. Custom lets you "
             "add any of J.P. Morgan's 58 asset classes and type your own "
             "return and risk numbers.")

    st.write("")
    with st.expander("Advanced settings"):
        paths = st.select_slider(
            "Number of simulated futures",
            options=[1_000, 2_500, 5_000, 10_000, 25_000, 50_000], value=10_000,
            help="More futures = smoother, more precise results, but slower.")
        seed = st.number_input(
            "Random seed", value=20260722, step=1,
            help="Fix the seed so everyone running the same inputs sees the "
                 "same numbers.")
    is_base = mode == MODE_BASE

    st.markdown(
        '<div class="foot" style="margin-top:1.2rem">Projections are simulations '
        'based on long-term market assumptions, not guarantees. Figures are in US '
        'dollars before fees, taxes and inflation.</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Header + navigation
# --------------------------------------------------------------------------
chip = "" if is_base else '<span class="modechip">Custom assumptions</span>'
st.markdown('<div class="eyebrow">Keepp Capital · Portfolio outlook</div>',
            unsafe_allow_html=True)
st.markdown(f"# How could your portfolio grow?{chip}", unsafe_allow_html=True)
st.markdown(
    f'<p class="lede">We ran {paths:,} possible futures for the markets and '
    f'tracked your portfolio through each one. Here is the range of what could '
    f'happen over the next {years} year{"s" if years != 1 else ""}.</p>',
    unsafe_allow_html=True)

tab_over, tab_mix, tab_opt, tab_det = st.tabs(
    ["Overview", "Your mix", "Improve the mix", "Details & saved runs"],
    key="nav", on_change="rerun")


# ==========================================================================
# TAB — Your mix (runs first: everything else depends on the allocation)
# ==========================================================================
with tab_mix:
    mix_left, mix_right = st.columns([1.95, 1], gap="large")

    with mix_left:
        if is_base:
            section("What's in your portfolio",
                    f"Open an asset class, pick an investment from its "
                    "dropdown, then set its share with the slider. All "
                    f"{len(uni.INVESTABLE)} J.P. Morgan asset classes are "
                    "available; "
                    "their expected return and risk are fixed to the 2026 "
                    "LTCMA and cannot be edited here. The <b>tick</b> says "
                    "whether the optimizer on <i>Improve the mix</i> may use "
                    "that asset — anything above 0% is always in.")

            lines = list(st.session_state.std_lines)
            w_state = dict(st.session_state.std_w)
            t_state = dict(st.session_state.std_tick)

            # The weight each slider is ALREADY showing. Streamlit refreshes a
            # keyed widget's session_state entry before the script runs, so
            # this is the value after the user's drag — reading std_w instead
            # would leave the category totals and the colours one interaction
            # behind. Falls back to stored state on the first render.
            live = {ln: float(st.session_state.get(f"sw_{ln}_{NONCE}",
                                                   w_state.get(ln, 0.0)))
                    for ln in lines}
            # Colours are the donut's own: same order, so a holding is the
            # same colour in the list, in the donut and on the Overview.
            held_order = [ln for key in CATEGORY_KEYS for ln in lines
                          if jpm.category_of(ln) == key and live[ln] > 0]
            ROW_COLOR = color_map([plain_name(jpm.LABEL[ln])[0]
                                   for ln in held_order])

            new_w, new_t, remove, add_line = {}, {}, None, None
            for key in CATEGORY_KEYS:
                cat = jpm.CATEGORY_NAME[key]
                mine = [ln for ln in lines if jpm.category_of(ln) == key]
                held = [ln for ln in mine if live[ln] > 0]
                c_sum = sum(live[ln] for ln in mine)
                n_tick = sum(1 for ln in mine
                             if t_state.get(ln, False) or live[ln] > 0)
                pool = len(jpm.CHOICES[key])
                label = (
                    f"{CATEGORY_PLAIN.get(cat, cat)}  ·  nothing added  ·  "
                    f"{pool} to choose from" if not mine else
                    f"{CATEGORY_PLAIN.get(cat, cat)}  ·  {c_sum:.1f}%  ·  "
                    f"{len(held)} held, {n_tick} in the optimizer  ·  "
                    f"{pool - len(mine)} more to choose from")
                # Open when the class has anything listed — a freshly added
                # asset sits at 0%, and keying this off `held` collapsed the
                # class the moment it was picked. Everything is open on a
                # blank portfolio, where the pickers are the only thing to do.
                with st.expander(label, expanded=bool(mine) or not lines):
                    for ln in mine:
                        nm, desc = plain_name(jpm.LABEL[ln])
                        r, v = std_numbers(ln)
                        w_now = live[ln]
                        colr = ROW_COLOR.get(nm, OTHER_COLOR)
                        t1, t2, t3 = st.columns(
                            [1.85, 1.25, 0.26], vertical_alignment="center")
                        with t1:
                            # The asset's name IS the checkbox label: on a
                            # phone the columns stack, and a bare tick box
                            # above a separate name reads as unrelated.
                            tick = st.checkbox(
                                nm,
                                value=bool(t_state.get(ln, False) or w_now > 0),
                                disabled=w_now > 0, key=f"tk_{ln}_{NONCE}",
                                help=("Held, so it is always offered to the "
                                      "optimizer." if w_now > 0 else
                                      "Tick to let the optimizer put money "
                                      "into this, even though you hold none "
                                      "today."))
                            st.markdown(
                                f'<div class="d2"><span class="sw" '
                                f'style="background:{colr};margin:0 6px 0 0;'
                                f'opacity:{1 if w_now > 0 else .35}"></span>'
                                f'{desc + " · " if desc else ""}'
                                f'Expected ~{r:.1f}% a year · swings '
                                f'±{v:.1f}%</div>', unsafe_allow_html=True)
                        val = t2.slider(
                            nm, 0.0, 100.0, value=w_now, step=0.5,
                            format="%.1f%%", key=f"sw_{ln}_{NONCE}",
                            label_visibility="collapsed")
                        if w_now <= 0 and t3.button(
                                "✕", key=f"rm_{ln}_{NONCE}",
                                help="Take this asset off the list"):
                            remove = ln
                        new_w[ln] = float(val)
                        new_t[ln] = bool(tick) or float(val) > 0

                    rest = [jpm.LABEL[n] for n, *_ , c in jpm.LINES
                            if c == key and n not in mine]
                    pick = st.selectbox(
                        f"Add from {cat}", ["— add an asset —"] + rest,
                        key=f"add_{key}_{NONCE}", label_visibility="collapsed",
                        help=f"Any of the {pool} J.P. Morgan {cat} lines. It "
                             f"arrives at 0%, already ticked for the optimizer.")
                    if pick != "— add an asset —":
                        add_line = jpm.resolve(pick)

            st.session_state.std_w = new_w
            st.session_state.std_tick = new_t

            if remove is not None:
                st.session_state.std_lines = [x for x in lines if x != remove]
                st.session_state.std_w.pop(remove, None)
                st.session_state.std_tick.pop(remove, None)
                bump_editors()
                st.rerun()
            if add_line is not None and add_line not in lines:
                st.session_state.std_lines = lines + [add_line]
                st.session_state.std_w[add_line] = 0.0
                st.session_state.std_tick[add_line] = True
                bump_editors()
                st.rerun()

            if not lines:
                st.info("**No assets yet.** Open an asset class above and pick "
                        "one from its dropdown to start building the "
                        "portfolio.", icon=":material/add_circle:")
            else:
                st.caption(
                    f"Ticked assets are the pool the optimizer re-weights on "
                    f"**Improve the mix**. Untick or press **✕** on a 0% asset "
                    f"to drop it. All {len(uni.INVESTABLE)} J.P. Morgan lines "
                    f"are available from the dropdowns, each on its published "
                    f"2026 LTCMA return and volatility.")

            # Everything the portfolio could hold — the optimizer's pool.
            POOL_LINES = [ln for ln in st.session_state.std_lines
                          if st.session_state.std_tick.get(ln, False)
                          or st.session_state.std_w.get(ln, 0.0) > 0]
            pool_alloc = std_alloc(POOL_LINES, st.session_state.std_w)
            # What the portfolio actually holds — what gets simulated.
            alloc = std_alloc(
                [ln for ln in st.session_state.std_lines
                 if st.session_state.std_w.get(ln, 0.0) > 0],
                st.session_state.std_w)

            st.write("")
            if lines and st.button("↺ Clear all assets"):
                (st.session_state.std_lines, st.session_state.std_w,
                 st.session_state.std_tick) = seed_standard()
                bump_editors()
                st.rerun()
        else:
            section("What's in your portfolio",
                    "One table per asset class. Pick an investment from the list "
                    "and it fills in J.P. Morgan's figures — type over them to use "
                    "your own. **+** adds a row; select a row and press Delete to "
                    "remove it. ★ marks an expansion-proposal line.")
            store = st.session_state.custom_alloc
            parts, dirty_any = [], False

            for key in CATEGORY_KEYS:
                cat = jpm.CATEGORY_NAME[key]
                g_prev = (store[store["Category"] == cat]
                          .reset_index(drop=True)[COLS])
                g_sum = pd.to_numeric(g_prev["Weight %"],
                                      errors="coerce").fillna(0).sum()
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:baseline;margin:1rem 0 .15rem">'
                    f'<span style="font-weight:650">{CATEGORY_PLAIN.get(cat, cat)}</span>'
                    f'<span class="muted">{g_sum:.1f}% of portfolio</span></div>',
                    unsafe_allow_html=True)
                edited = st.data_editor(
                    g_prev, key=f"grp_{key}_{NONCE}", hide_index=True,
                    width='stretch', num_rows="dynamic",
                    column_config={
                        "LTCMA line": st.column_config.SelectboxColumn(
                            "Investment", options=jpm.CHOICES[key],
                            width="medium", required=True,
                            help=f"J.P. Morgan 2026 LTCMA lines in {cat}. "
                                 f"Type to filter."),
                        # step MUST stay 0.01 — a coarser step snaps the
                        # displayed value.
                        "Weight %": st.column_config.NumberColumn(
                            "Share %", min_value=0.0, max_value=100.0,
                            step=0.01, format="%.2f"),
                        "Return %": st.column_config.NumberColumn(
                            "Expected return % / yr", min_value=-20.0,
                            max_value=60.0, step=0.01, format="%.2f",
                            help="Compound (geometric) return per year."),
                        "Vol %": st.column_config.NumberColumn(
                            "Yearly swings %", min_value=0.0, max_value=100.0,
                            step=0.01, format="%.2f",
                            help="Volatility: how much the value typically "
                                 "moves up or down in a year."),
                    },
                )
                synced, d = jpm.sync_group(g_prev, edited, key)
                dirty_any = dirty_any or d
                synced.insert(0, "Category", cat)
                parts.append(synced)

            alloc = (pd.concat(parts, ignore_index=True) if parts
                     else custom_alloc_seed().iloc[0:0])
            st.session_state.custom_alloc = alloc
            pool_alloc = alloc            # Custom has no separate tick list
            if dirty_any:
                bump_editors()
                st.rerun()

            st.write("")
            if st.button("↺ Reset to J.P. Morgan defaults"):
                st.session_state.custom_alloc = custom_alloc_seed()
                bump_editors()
                st.rerun()

    alloc = alloc.reset_index(drop=True)
    for _c in ("Weight %", "Return %", "Vol %"):
        alloc[_c] = pd.to_numeric(alloc[_c], errors="coerce").fillna(0.0)

    wsum = float(alloc["Weight %"].sum())
    off = abs(wsum - 100.0) > 0.005
    # Standard now opens with nothing in it, so "0%" is the normal first
    # state, not a mistake. It gets an invitation, not a red warning.
    empty = len(alloc) == 0

    # Friendly names + colours, fixed by position in the allocation
    ALLOC_NAMES = [plain_name(x)[0] for x in alloc["LTCMA line"]]
    ALLOC_DESCS = {plain_name(x)[0]: plain_name(x)[1] for x in alloc["LTCMA line"]}
    CMAP = color_map(ALLOC_NAMES)

    with mix_right:
        section("At a glance")
        if empty:
            st.markdown(
                '<div class="total"><div><b>Nothing added yet</b></div>'
                '<div>0 holdings</div></div>', unsafe_allow_html=True)
            st.caption("Pick assets on the left and they appear here, with "
                       "their share of the portfolio.")
        elif off:
            gap = 100.0 - wsum
            msg = (f"{abs(gap):.1f}% still to place" if gap > 0
                   else f"{abs(gap):.1f}% too much")
            fill = min(wsum, 100.0)
            st.markdown(
                f'<div class="total bad"><div><b>Total {wsum:.1f}%</b></div>'
                f'<div>{msg}</div></div>'
                f'<div class="bar"><div style="width:{fill}%;background:{BRICK}"></div></div>',
                unsafe_allow_html=True)
            st.write("")
            if st.button("Scale everything to 100%", type="primary", width='stretch'):
                if wsum > 0:
                    if is_base:
                        st.session_state.std_w = {
                            ln: round(w / wsum * 100, 4)
                            for ln, w in st.session_state.std_w.items()}
                    else:
                        st.session_state.custom_alloc["Weight %"] = (
                            pd.to_numeric(st.session_state.custom_alloc["Weight %"],
                                          errors="coerce").fillna(0) / wsum * 100
                        ).round(4)
                    bump_editors()
                    st.rerun()
                else:
                    st.warning("Give at least one holding a share above zero.")
        else:
            st.markdown(
                '<div class="total ok"><div><b>Total 100%</b></div>'
                '<div>✓ Ready</div></div>', unsafe_allow_html=True)

        hdf = holdings_frame(ALLOC_NAMES, list(alloc["Weight %"]), CMAP)
        ch = donut(hdf, f"{len(hdf)}", "holdings", size=230)
        if ch is not None:
            st.altair_chart(ch, width='content')
            st.markdown(legend_html(hdf), unsafe_allow_html=True)

if off:
    for t in (tab_over, tab_opt, tab_det):
        with t:
            if empty:
                st.info("**Start on the Your mix tab.** Pick the investments "
                        "this portfolio holds from the three asset-class "
                        "dropdowns and give each one a share. Results appear "
                        "here as soon as the shares add up to 100%.",
                        icon=":material/arrow_back:")
            else:
                st.warning(
                    f"Your mix adds up to **{wsum:.1f}%**, not 100%. Open "
                    f"**Your mix** to adjust it, or press *Scale everything "
                    f"to 100%* there.", icon=":material/warning:")
    st.stop()


# ==========================================================================
# Simulation (cached)
# ==========================================================================
w_t, r_t, v_t, corr_full = sleeve_inputs(alloc)
with st.spinner("Simulating markets…"):
    res, repaired = run_sim(w_t, r_t, v_t, tuple(corr_full.flatten()),
                            int(paths), int(years), int(seed))
term = res.rows[-1]
T = term["_years"]
P_LOSS = term["P(loss)"]
MED_X, LO_X, HI_X = term["Median x"], term["5th x"], term["95th x"]
BAD_YEAR = res.exp_arith - 1.645 * res.analytic_vol     # approx. 1-in-20 year
R_IDX, R_NAME = risk_level(res.analytic_vol)


def fan_chart(res, amount, height=380):
    fan = pd.DataFrame(res.fan, columns=["p05", "p25", "p50", "p75", "p95"]) * amount
    fan["Year"] = res.fan_years
    for c in ("p05", "p25", "p50", "p75", "p95"):
        fan[c + "_l"] = fan[c].map(money)
    fan["yr_l"] = fan["Year"].map(lambda y: f"Year {y:g}")

    ax_y = alt.Axis(title=None, format="$.3~s", labelColor=INK_3, gridColor=GRID,
                    domain=False, ticks=False, tickCount=6, labelFontSize=12,
                    labelPadding=8)
    ax_x = alt.Axis(title="Years from today", titleColor=INK_3, labelColor=INK_3,
                    grid=False, tickMinStep=1, format="d", labelFontSize=12,
                    domainColor=GRID, tickColor=GRID, titleFontWeight=400)
    xs = alt.Scale(domain=[0, float(res.years)], nice=False)
    ys = alt.Scale(zero=False, nice=True)
    b = alt.Chart(fan)
    outer = b.mark_area(color=BAND_OUTER, opacity=.9).encode(
        x=alt.X("Year:Q", axis=ax_x, scale=xs),
        y=alt.Y("p05:Q", axis=ax_y, scale=ys), y2="p95:Q")
    inner = b.mark_area(color=BAND_INNER, opacity=.9).encode(
        x="Year:Q", y=alt.Y("p25:Q", scale=ys), y2="p75:Q")
    start = alt.Chart(pd.DataFrame({"y": [amount]})).mark_rule(
        color=INK_3, strokeDash=[4, 4], opacity=.8).encode(y="y:Q")
    median = b.mark_line(color=ACCENT_DK, strokeWidth=2.5).encode(
        x="Year:Q", y=alt.Y("p50:Q", scale=ys))

    hover = alt.selection_point(fields=["Year"], nearest=True, on="pointerover",
                                empty=False, clear="pointerout")
    rule = b.mark_rule(color=INK_3, strokeWidth=1).encode(
        x="Year:Q",
        opacity=alt.condition(hover, alt.value(.6), alt.value(0)),
        tooltip=[alt.Tooltip("yr_l:N", title="When"),
                 alt.Tooltip("p95_l:N", title="Strong markets"),
                 alt.Tooltip("p50_l:N", title="Typical"),
                 alt.Tooltip("p05_l:N", title="Weak markets")],
    ).add_params(hover)
    dots = b.transform_fold(["p05", "p50", "p95"], as_=["k", "v"]).mark_circle(
        size=70, color=ACCENT_DK, stroke="white", strokeWidth=2).encode(
        x="Year:Q", y=alt.Y("v:Q", scale=ys),
        opacity=alt.condition(hover, alt.value(1), alt.value(0)))

    end = fan.iloc[-1]
    ends = pd.DataFrame([
        {"Year": end["Year"], "v": end["p95"], "t": end["p95_l"], "c": DEEP_BLUE},
        {"Year": end["Year"], "v": end["p50"], "t": end["p50_l"], "c": ACCENT_DK},
        {"Year": end["Year"], "v": end["p05"], "t": end["p05_l"], "c": BRICK},
    ])
    tags = alt.Chart(ends).mark_text(align="left", dx=8, fontSize=13,
                                     fontWeight=700).encode(
        x="Year:Q", y=alt.Y("v:Q", scale=ys), text="t:N",
        color=alt.Color("c:N", scale=None))
    return (outer + inner + start + median + rule + dots + tags).properties(
        height=height, padding={"left": 4, "right": 70, "top": 10, "bottom": 4}
    ).configure_view(strokeWidth=0)


# ==========================================================================
# TAB — Overview
# ==========================================================================
with tab_over:
    if repaired:
        st.warning("The correlation sub-matrix needed a numerical repair before "
                   "simulating. This should not happen with the published J.P. "
                   "Morgan matrix — worth reporting if you see it.")

    gain = MED_X * amount - amount
    st.markdown(
        f'<div class="hero">'
        f'<div class="q">If you invest <b>{money(amount)}</b> today, in '
        f'<b>{T} years</b> it could be worth about</div>'
        f'<div class="big">{money(MED_X * amount)}'
        f'<small>{"+" if gain >= 0 else "−"}{money(abs(gain))} · '
        f'{pct(term["Median"])} a year</small></div>'
        f'<div class="s">That is the <b>typical</b> outcome — half of the simulated '
        f'futures end higher, half lower. In 9 out of 10 futures the portfolio ends '
        f'between <b>{money(LO_X * amount)}</b> and <b>{money(HI_X * amount)}</b>.'
        f'</div></div>', unsafe_allow_html=True)

    section(f"Three ways the next {T} years could go")
    s1, s2, s3 = st.columns(3)
    for col, colr, title, x, ann, sub in (
        (s1, BRICK, "If markets are weak", LO_X, term["5th"],
         "Only 1 in 20 futures ends lower than this."),
        (s2, ACCENT, "Typical outcome", MED_X, term["Median"],
         "The middle of all simulated futures."),
        (s3, DEEP_BLUE, "If markets are strong", HI_X, term["95th"],
         "Only 1 in 20 futures ends higher than this."),
    ):
        col.markdown(card(
            title, money(x * amount),
            f'{pct(ann, sign=True)} a year · {pct(x - 1, 0, sign=True)} in total<br>{sub}',
            dot=colr, extra_cls="scen", style=f"--c:{colr}"), unsafe_allow_html=True)

    section("How your money could grow, year by year",
            "Each shaded band shows where the portfolio is likely to be. "
            "Hover over the chart to read any year.")
    st.altair_chart(fan_chart(res, amount), width='stretch')
    st.markdown(
        f'<div class="keys">'
        f'<span><i style="background:{ACCENT_DK};height:3px"></i>Typical path</span>'
        f'<span><i style="background:{BAND_INNER}"></i>Likely range (half of futures)</span>'
        f'<span><i style="background:{BAND_OUTER}"></i>Wide range (9 in 10 futures)</span>'
        f'<span><i style="border-top:2px dashed {INK_3};height:0"></i>'
        f'What you put in</span></div>', unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        section("How much risk are you taking?")
        rc = RISK_COLORS[R_IDX]
        segs = "".join(
            f'<span class="{"on" if i <= R_IDX else ""}"></span>'
            for i in range(len(RISK_LEVELS)))
        st.markdown(
            f'<div class="card"><div class="lbl">Risk level</div>'
            f'<div class="val" style="color:{rc}">{R_NAME}</div>'
            f'<div class="meter" style="--c:{rc}">{segs}</div>'
            f'<div class="meterlbl"><span>Very low</span><span>Very high</span></div>'
            f'<div class="sub" style="margin-top:10px">In a normal year the value '
            f'moves up or down by about <b>±{pct(res.analytic_vol)}</b>.</div></div>',
            unsafe_allow_html=True)
        st.write("")
        k1, k2 = st.columns(2)
        k1.markdown(card(
            "A bad year (1 in 20)",
            pct(BAD_YEAR, sign=True),
            f"≈ {money(BAD_YEAR * amount)} on {money(amount)}. "
            f"Rough estimate for a single year.",
            hot=BAD_YEAR < 0), unsafe_allow_html=True)
        k2.markdown(card(
            "Chance of a loss",
            pct(P_LOSS),
            f"{one_in(P_LOSS).capitalize()} futures end below "
            f"{money(amount)} after {T} years.",
            hot=P_LOSS > 0.05), unsafe_allow_html=True)

    with right:
        section("What you're invested in")
        hdf = holdings_frame(ALLOC_NAMES, list(alloc["Weight %"]), CMAP)
        d1, d2 = st.columns([1, 1.25], vertical_alignment="center")
        with d1:
            ch = donut(hdf, f"{len(hdf)}", "holdings", size=220)
            if ch is not None:
                st.altair_chart(ch, width='content')
        with d2:
            st.markdown(legend_html(hdf, ALLOC_DESCS), unsafe_allow_html=True)

    st.write("")
    st.markdown(
        '<div class="note">These results come from computer simulations based on '
        'long-term market assumptions. They are not a promise or forecast of any '
        'specific outcome. Real markets can behave worse than a model expects, '
        'especially in a crisis. Figures are before fees, taxes and inflation.</div>',
        unsafe_allow_html=True)


# ==========================================================================
# TAB — Improve the mix (optimizer)
# ==========================================================================
OBJ_SHARPE = "Maximum Sharpe ratio"
OBJ_MINVOL = "Minimum volatility for a target return"
OBJ_MAXRET = "Maximum return for a volatility budget"
OBJ_GROWTH = "Maximum long-run growth"
OBJ_PLAIN = {
    OBJ_SHARPE: ("Best balance", "Most return for each unit of risk"),
    OBJ_GROWTH: ("Most growth", "Grow the typical outcome as much as possible"),
    OBJ_MINVOL: ("Least risk for a return", "Pick a return, keep swings smallest"),
    OBJ_MAXRET: ("Most return for a risk level", "Pick a risk limit, earn the most"),
}

SCOPE_SLEEVES = "The sleeves above"
SCOPE_UNIVERSE = f"All {len(uni.INVESTABLE)} LTCMA lines"


def range_strip(span, chosen, metric, current_pp):
    """Bar showing what is reachable, and where you are today."""
    lo_s, hi_s = span.lo * 100, span.hi * 100
    pad = max((hi_s - lo_s) * 0.12, 0.05)
    x0, x1 = lo_s - pad, hi_s + pad
    unit = "return" if metric == "return" else "risk"
    axis = alt.Axis(title=None, labels=False, ticks=False, grid=False, domain=False)
    xs = alt.Scale(domain=[x0, x1], nice=False)
    TOP, BOT = 6, 22
    reachable = alt.Chart(pd.DataFrame([{
        "a": lo_s, "b": hi_s, "lbl": f"{lo_s:.2f}% – {hi_s:.2f}%"}])).mark_rect(
        color=BAND_OUTER, y=TOP, y2=BOT, cornerRadius=8).encode(
        x=alt.X("a:Q", axis=axis, scale=xs), x2="b:Q",
        tooltip=[alt.Tooltip("lbl:N", title=f"Possible {unit}")])
    outside = current_pp < lo_s or current_pp > hi_s
    cur = alt.Chart(pd.DataFrame([{
        "x": float(np.clip(current_pp, lo_s, hi_s)),
        "lbl": (f"{current_pp:.2f}% (outside the possible range)" if outside
                else f"{current_pp:.2f}%")}])).mark_point(
        size=90, shape="triangle-up", filled=True, color=INK_3, opacity=1,
        y=BOT + 7).encode(x=alt.X("x:Q", scale=xs),
                          tooltip=[alt.Tooltip("lbl:N", title="Your mix today")])
    pick = alt.Chart(pd.DataFrame([{"x": float(chosen)}])).mark_circle(
        size=260, color=ACCENT, stroke="white", strokeWidth=3, opacity=1,
        y=(TOP + BOT) / 2).encode(x=alt.X("x:Q", scale=xs))

    def _end(x, text, align):
        return alt.Chart(pd.DataFrame([{"x": x, "t": text}])).mark_text(
            fontSize=11, color=INK_3, y=BOT + 20, align=align).encode(
            x=alt.X("x:Q", scale=xs), text="t:N")

    ends = _end(lo_s, f"{lo_s:.1f}%", "left") + _end(hi_s, f"{hi_s:.1f}%", "right")
    return (reachable + ends + cur + pick).properties(height=50).configure_view(
        strokeWidth=0)


def change_chart(names, cur, new):
    rows = [{"name": n, "chg": (b - a) * 100, "now": a * 100, "new": b * 100}
            for n, a, b in zip(names, cur, new) if abs(b - a) * 100 >= 0.05]
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("chg", ascending=False)
    df["dir"] = np.where(df["chg"] >= 0, "More", "Less")
    df["t"] = df["chg"].map(lambda v: f"{v:+.1f} pts".replace("-", "−"))
    df["tip"] = [f"{a:.1f}% → {b:.1f}%" for a, b in zip(df["now"], df["new"])]
    lim = float(df["chg"].abs().max()) * 1.35 + 1
    y = alt.Y("name:N", sort=list(df["name"]), title=None,
              axis=alt.Axis(labelFontSize=12, labelColor=INK, domain=False,
                            ticks=False, labelPadding=10, labelLimit=220))
    x = alt.X("chg:Q", title=None, scale=alt.Scale(domain=[-lim, lim]),
              axis=alt.Axis(labels=False, ticks=False, grid=False, domain=False))
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=16).encode(
        x=x, y=y,
        color=alt.Color("dir:N", legend=None, scale=alt.Scale(
            domain=["More", "Less"], range=[UP_COLOR, DOWN_COLOR])),
        tooltip=[alt.Tooltip("name:N", title="Holding"),
                 alt.Tooltip("tip:N", title="Share"),
                 alt.Tooltip("t:N", title="Change")])
    pos = alt.Chart(df[df["chg"] >= 0]).mark_text(
        align="left", dx=6, fontSize=12, fontWeight=600, color=INK_2).encode(
        x="chg:Q", y=alt.Y("name:N", sort=list(df["name"])), text="t:N")
    neg = alt.Chart(df[df["chg"] < 0]).mark_text(
        align="right", dx=-6, fontSize=12, fontWeight=600, color=INK_2).encode(
        x="chg:Q", y=alt.Y("name:N", sort=list(df["name"])), text="t:N")
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=INK_3).encode(x="x:Q")
    return (bars + zero + pos + neg).properties(
        height=max(34 * len(df), 80)).configure_view(strokeWidth=0)


with tab_opt:
    section("Let the model suggest a better mix",
            "Pick which investments it may use and a goal. It searches for the "
            "best combination — with the same return and risk assumptions as the "
            "Overview — and shows what would change.")

    # Step 1 — the pool. Visible, not tucked in Advanced: "only what I hold"
    # and "anything in J.P. Morgan's catalogue" are two different questions.
    scope = st.radio(
        "1 · Which investments can it use?", [SCOPE_SLEEVES, SCOPE_UNIVERSE],
        key="opt_scope", horizontal=True,
        format_func=lambda s_: (
            ("My ticked assets" if is_base else "Only my current holdings")
            if s_ == SCOPE_SLEEVES
            else f"All {len(uni.INVESTABLE)} asset classes"),
        captions=[(f"The {len(pool_alloc)} assets ticked on Your mix"
                   if is_base else "Re-weight what the portfolio already holds"),
                  "Build the best mix from J.P. Morgan's full catalogue"],
        help=f"All {len(uni.INVESTABLE)} asset classes: every J.P. Morgan 2026 "
             f"LTCMA line is a candidate. Lines you already hold keep your own "
             f"return and risk figures; the rest use J.P. Morgan's. Takes about "
             f"1–2 seconds.")
    is_universe = scope == SCOPE_UNIVERSE

    # The ticked pool, not just what is held: a 0% ticked asset is exactly a
    # candidate the optimizer may move into.
    opt_alloc = uni.universe_alloc(alloc) if is_universe else pool_alloc
    opt_lines = [jpm.resolve(x) for x in opt_alloc["LTCMA line"]]
    opt_cats = list(opt_alloc["Category"])
    n_lines = len(opt_lines)
    cash_pos = [i for i, nm in enumerate(opt_lines) if nm == CASH_LINE]

    opt_mu, opt_vols = opt.inputs_from_percent(opt_alloc["Return %"],
                                               opt_alloc["Vol %"])
    opt_cov = opt.covariance(opt_vols, jpm.corr_for(opt_alloc["LTCMA line"]))
    cur_w = np.asarray(opt_alloc["Weight %"], dtype=float) / 100.0

    groups = {}
    for i, cat in enumerate(opt_cats):
        groups.setdefault(cat, []).append(i)
    groups = {k: np.array(v) for k, v in groups.items()}
    cur_cat_pct = {k: float(cur_w[i].sum()) * 100.0 for k, i in groups.items()}

    if n_lines < 2:
        st.info("Add a second holding to use this — with one there is nothing to "
                "rebalance between.")
    else:
        objective = st.radio(
            "2 · What's the goal?", list(OBJ_PLAIN), key="opt_objective",
            horizontal=True,
            format_func=lambda o: OBJ_PLAIN[o][0],
            captions=[OBJ_PLAIN[o][1] for o in OBJ_PLAIN])
        cur_r_pct, cur_v_pct = [x * 100 for x in
                                opt.portfolio(cur_w, opt_mu, opt_cov)]
        # Filled further down, once the advanced limits are known.
        target_slot = st.container()

        with st.expander("Advanced limits"):
            oc1, oc2, oc3 = st.columns(3)
            rf_default = (float(opt_alloc["Return %"].iloc[cash_pos[0]])
                          if cash_pos else 3.80)
            rf_pct = oc1.number_input(
                "Risk-free rate %", value=round(rf_default, 2),
                min_value=0.0, max_value=20.0, step=0.05, format="%.2f",
                help="Defaults to this allocation's own U.S. Cash return.")
            floor_pct = oc2.number_input(
                "Keep at least this much cash %", value=5.00, min_value=0.0,
                max_value=100.0, step=0.5, format="%.2f", disabled=not cash_pos,
                help=("Minimum weight on the cash line." if cash_pos else
                      "No cash line in this allocation, so nothing to floor."))
            cap_pct = oc3.number_input(
                "Max in any one holding %", value=100.00, min_value=0.01,
                max_value=100.0, step=1.0, format="%.2f",
                help="Stops the model concentrating everything in one asset. "
                     "Leave at 100% for no cap.")

            st.markdown('<div class="muted" style="margin:.6rem 0 .3rem">'
                        '<b>Limits per asset class</b> — by default, each can move '
                        '10 points either way from today.</div>',
                        unsafe_allow_html=True)
            band_seed = pd.DataFrame([
                {"Category": k,
                 "Min %": round(max(0.0, cur_cat_pct[k] - 10.0), 2),
                 "Max %": round(min(100.0, cur_cat_pct[k] + 10.0), 2)}
                for k in groups])
            bands_df = st.data_editor(
                band_seed, key=f"opt_bands_{NONCE}", hide_index=True,
                width='stretch', num_rows="fixed", disabled=["Category"],
                column_config={
                    "Category": st.column_config.TextColumn("Asset class",
                                                            width="medium"),
                    "Min %": st.column_config.NumberColumn(
                        "At least %", min_value=0.0, max_value=100.0, step=0.5,
                        format="%.2f"),
                    "Max %": st.column_config.NumberColumn(
                        "At most %", min_value=0.0, max_value=100.0, step=0.5,
                        format="%.2f"),
                })
            show_stability = st.checkbox(
                "Also run a stability check (slower, ~10s)", value=False,
                disabled=is_universe,
                help=("Re-solves 40 times with every expected return nudged by "
                      "random noise and reports how far each weight moves."
                      if not is_universe else
                      "Not available across all 58 lines — it would take minutes."))
            show_stability = show_stability and not is_universe

        lower = np.zeros(n_lines)
        upper = np.full(n_lines, cap_pct / 100.0)
        if cash_pos:
            lower[cash_pos[0]] = floor_pct / 100.0
            upper[cash_pos[0]] = max(upper[cash_pos[0]], floor_pct / 100.0)
        band_map = {r["Category"]: (float(r["Min %"]) / 100.0,
                                    float(r["Max %"]) / 100.0)
                    for _, r in bands_df.iterrows()}
        limits = opt.Limits(lower, upper, groups,
                            {k: band_map.get(k, (0.0, 1.0)) for k in groups})

        target_pct = budget_pct = None
        if objective in (OBJ_MINVOL, OBJ_MAXRET):
            metric = "return" if objective == OBJ_MINVOL else "vol"
            label = ("Return you want (average per year)" if metric == "return"
                     else "Most yearly swings you'll accept")
            cache_key = (
                tuple(opt_lines),
                tuple(np.round(opt_mu, 10)), tuple(np.round(opt_vols, 10)),
                float(np.round(opt_cov.sum(), 10)),
                float(np.round(np.trace(opt_cov), 10)),
                tuple(sorted((k, round(v[0], 8), round(v[1], 8))
                             for k, v in limits.sleeve_band.items())),
                round(float(floor_pct), 6), round(float(cap_pct), 6),
                bool(is_universe), metric,
            )
            with target_slot:
                try:
                    span = feasible_span(metric, cache_key, opt_mu, opt_cov, limits)
                except opt.Infeasible as e:
                    span = None
                    st.error(f"**These limits cannot all be met.** {e}")
                except Exception as e:
                    span = None
                    st.error(f"**Could not work out the possible range.** "
                             f"{type(e).__name__}: {e}")

                if span is not None:
                    lo_pp, hi_pp = [x * 100 for x in
                                    opt.slider_bounds(span, STEP_PP / 100.0)]
                    if span.degenerate or hi_pp - lo_pp < STEP_PP / 2:
                        st.info(f"Your limits leave only one possible "
                                f"{'return' if metric == 'return' else 'risk level'}"
                                f": **{lo_pp:.2f}%**. Loosen a limit under "
                                f"*Advanced limits* to open a range.")
                        chosen = lo_pp
                    else:
                        key = f"opt_target_{metric}"
                        if key in st.session_state:
                            st.session_state[key] = float(
                                np.clip(st.session_state[key], lo_pp, hi_pp))
                        else:
                            seed_v = cur_r_pct if metric == "return" else cur_v_pct
                            st.session_state[key] = float(
                                np.clip(round(seed_v, 2), lo_pp, hi_pp))
                        tc1, tc2 = st.columns([1, 1.2], gap="large")
                        chosen = tc1.slider(
                            label, min_value=float(lo_pp), max_value=float(hi_pp),
                            step=STEP_PP, format="%.2f%%", key=key,
                            help="Only what your limits make possible is offered. "
                                 "Returns here are the average-year (arithmetic) "
                                 "figure the optimizer works with.")
                        with tc2:
                            st.markdown(
                                f'<div class="muted">Possible range with your limits. '
                                f'▲ marks your mix today '
                                f'({cur_r_pct if metric == "return" else cur_v_pct:.2f}%).'
                                f'</div>', unsafe_allow_html=True)
                            st.altair_chart(
                                range_strip(span, chosen, metric,
                                            cur_r_pct if metric == "return"
                                            else cur_v_pct),
                                width='stretch')
                    if metric == "return":
                        target_pct = chosen
                    else:
                        budget_pct = chosen

        run_opt = st.button("Suggest a mix", type="primary", icon=":material/auto_awesome:")

        if run_opt:
            # Across the catalogue the frontier is NOT solved up front: at
            # n=58 it is 24 QPs at ~0.6s each, and only the optional map in an
            # expander needs it. Best balance uses the one-QP tangency
            # (universe.max_sharpe_qp) instead — measured 1.4s vs 19s, and at
            # least as high a Sharpe (test_universe.py). Over the sleeves every
            # QP is ~1ms, so that path is unchanged.
            import time as _time
            _t0 = _time.perf_counter()
            spinner_msg = (f"Searching all {n_lines} asset classes…"
                           if is_universe else "Searching for the best mix…")
            try:
                with st.spinner(spinner_msg):
                    w0 = opt.feasible_start(limits)          # raises Infeasible
                    frontier = (None if is_universe else
                                opt.efficient_frontier(opt_mu, opt_cov, limits,
                                                       w0, 60))
                    rf = rf_pct / 100.0
                    if objective == OBJ_SHARPE:
                        if is_universe:
                            w_opt, _ = uni.max_sharpe_qp(opt_mu, opt_cov,
                                                         limits, rf)
                        else:
                            w_opt, _, _, _ = opt.max_sharpe(
                                opt_mu, opt_cov, limits, w0, rf)
                        note = None
                    elif objective == OBJ_GROWTH:
                        r_ = opt.max_growth(opt_mu, opt_cov, limits, w0)
                        w_opt, note = r_.x, (None if r_.success else str(r_.message))
                    elif objective == OBJ_MINVOL:
                        r_ = opt.min_variance(opt_mu, opt_cov, limits, w0,
                                              target_return=target_pct / 100.0)
                        w_opt, note = r_.x, (None if r_.success else str(r_.message))
                    else:
                        r_ = opt.max_return(opt_mu, opt_cov, limits, w0,
                                            vol_budget=budget_pct / 100.0)
                        w_opt, note = r_.x, (None if r_.success else str(r_.message))
                    w_opt = np.clip(np.asarray(w_opt, dtype=float),
                                    limits.lower, limits.upper)
                    w_opt = w_opt / w_opt.sum()
                    band = opt.stability(opt_mu, opt_cov, limits, rf) \
                        if show_stability else (None, None)
                st.session_state.opt_result = {
                    "lines": opt_lines, "cats": opt_cats, "w": w_opt,
                    "frontier": frontier, "rf": rf, "objective": objective,
                    "note": note, "stability": band,
                    "cur_w": cur_w.copy(), "mode": mode,
                    "universe": is_universe, "alloc": opt_alloc.copy(),
                    "target_pct": target_pct, "budget_pct": budget_pct,
                    "limits": limits, "seconds": _time.perf_counter() - _t0,
                }
            except opt.Infeasible as e:
                st.session_state.pop("opt_result", None)
                st.error(f"**These limits cannot all be met.** {e}")
            except Exception as e:
                st.session_state.pop("opt_result", None)
                st.error(f"**The model could not find a solution.** "
                         f"{type(e).__name__}: {e}")

        # ------------------------------------------------------------------
        # Results
        # ------------------------------------------------------------------
        R = st.session_state.get("opt_result")
        if R and R["lines"] == opt_lines:
            w_opt = R["w"]
            rf = R["rf"]
            if R.get("note"):
                st.warning(f"The search did not fully settle — this is the best "
                           f"mix it reached, not a proven optimum. ({R['note']})")

            r_opt, v_opt = opt.portfolio(w_opt, opt_mu, opt_cov)
            r_cur, v_cur = opt.portfolio(cur_w, opt_mu, opt_cov)
            sh_opt = opt.sharpe(w_opt, opt_mu, opt_cov, rf)
            sh_cur = opt.sharpe(cur_w, opt_mu, opt_cov, rf)

            applied = (uni.applied_sleeves(R["alloc"], w_opt)
                       if R.get("universe") else None)   # catalogue display
            # The "after" portfolio, as an allocation frame. Built from the
            # frame the optimizer actually worked on (R["alloc"]) — with a
            # ticked 0% asset in the pool that frame is LONGER than `alloc`,
            # so reusing `alloc` here mismatched the weights and the
            # correlation sub-matrix.
            after_frame = uni.applied_sleeves(R["alloc"], w_opt)
            a_w, a_r, a_v = (after_frame["Weight %"], after_frame["Return %"],
                             after_frame["Vol %"])
            a_corr = corr_of(after_frame)

            before = res
            after = run_sim(tuple(a_w), tuple(a_r), tuple(a_v),
                            tuple(a_corr.flatten()),
                            int(paths), int(years), int(seed))[0]
            b_t, a_t = before.rows[-1], after.rows[-1]

            _pool = (f"all {n_lines} asset classes" if R.get("universe")
                     else "your current holdings")
            _secs = R.get("seconds")
            section("Your mix today vs. the suggestion",
                    f"Goal: <b>{OBJ_PLAIN[R['objective']][0]}</b> · chosen from "
                    f"<b>{_pool}</b>"
                    + (f" · found in {_secs:.1f} s" if _secs is not None else "")
                    + f". Outcomes are for {money(amount)} over {T} years.")

            def cmp_card(label, was, now, better, sub):
                colr = ACCENT if better else (BRICK if better is False else INK)
                return (f'<div class="card"><div class="lbl">{label}</div>'
                        f'<div class="val"><span class="was">{was}</span>'
                        f'<span class="arrow">→</span>'
                        f'<span style="color:{colr}">{now}</span></div>'
                        f'<div class="sub">{sub}</div></div>')

            def _b(new, old, higher_is_better=True, tol=1e-9):
                if abs(new - old) < tol:
                    return None
                return (new > old) == higher_is_better

            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(cmp_card(
                "Typical value", money(b_t["Median x"] * amount),
                money(a_t["Median x"] * amount),
                _b(a_t["Median x"], b_t["Median x"]),
                f"{money((a_t['Median x'] - b_t['Median x']) * amount)} "
                f"difference after {T} years"), unsafe_allow_html=True)
            k2.markdown(cmp_card(
                "Growth per year", pct(b_t["Median"], 2), pct(a_t["Median"], 2),
                _b(a_t["Median"], b_t["Median"]),
                "Typical yearly growth"), unsafe_allow_html=True)
            k3.markdown(cmp_card(
                "Yearly swings", f"±{pct(v_cur)}", f"±{pct(v_opt)}",
                _b(v_opt, v_cur, higher_is_better=False),
                f"Risk level: {risk_level(v_cur)[1]} → {risk_level(v_opt)[1]}"),
                unsafe_allow_html=True)
            k4.markdown(cmp_card(
                "Chance of a loss", pct(b_t["P(loss)"]), pct(a_t["P(loss)"]),
                _b(a_t["P(loss)"], b_t["P(loss)"], higher_is_better=False),
                f"Ending below {money(amount)} after {T} years"),
                unsafe_allow_html=True)

            # ---- donuts: now vs suggested ---------------------------------
            POOL_NAMES = [plain_name(x)[0] for x in R["alloc"]["LTCMA line"]]
            if R.get("universe"):
                new_names = [plain_name(x)[0] for x in applied["LTCMA line"]]
                new_weights = list(applied["Weight %"])
            else:
                new_names = POOL_NAMES
                new_weights = [float(x) * 100 for x in w_opt]
            # Colour follows the holding. Holdings shown on the Your mix tab
            # keep that colour; slots belonging to holdings that are empty in
            # BOTH donuts are freed and handed to incoming lines, largest
            # first. Only what is still left over folds into "Other" — across
            # the catalogue the suggestion usually sells most of today's mix,
            # so reserving every old slot would push most of it into grey.
            now_w = dict(zip(ALLOC_NAMES, alloc["Weight %"]))
            new_w = dict(zip(new_names, new_weights))
            present = ({n for n, w in now_w.items() if w >= 0.05}
                       | {n for n, w in new_w.items() if w >= 0.05})
            cmp_map = {n: c for n, c in CMAP.items() if n in present}
            free = [c for c in SERIES if c not in cmp_map.values()]
            incoming = [n for n, w in sorted(new_w.items(), key=lambda t: -t[1])
                        if w >= 0.05 and n not in cmp_map]
            for n, c in zip(incoming, free):
                cmp_map[n] = c
            folded = [n for n in incoming if n not in cmp_map]
            descs = {**ALLOC_DESCS,
                     **({plain_name(x)[0]: plain_name(x)[1]
                         for x in applied["LTCMA line"]} if applied is not None else {})}
            if folded:
                descs["Other"] = ", ".join(
                    f"{n} {new_w[n]:.1f}%" for n in folded)

            st.write("")
            dl, dr, dc = st.columns([1, 1, 1.35], gap="large")
            now_df = holdings_frame(ALLOC_NAMES, list(alloc["Weight %"]), cmp_map)
            new_df = holdings_frame(new_names, new_weights, cmp_map)
            with dl:
                st.markdown('<div class="sect" style="margin-top:.4rem">Today</div>',
                            unsafe_allow_html=True)
                ch = donut(now_df, pct(v_cur), "yearly swings", size=210)
                if ch is not None:
                    st.altair_chart(ch, width='content')
            with dr:
                st.markdown('<div class="sect" style="margin-top:.4rem;color:'
                            f'{ACCENT_DK}">Suggested</div>', unsafe_allow_html=True)
                ch = donut(new_df, pct(v_opt), "yearly swings", size=210)
                if ch is not None:
                    st.altair_chart(ch, width='content')
            with dc:
                leg = pd.DataFrame([{"name": n, "color": c}
                                    for n, c in cmp_map.items()],
                                   columns=["name", "color"])
                nw = dict(zip(new_df["name"], new_df["w"]))
                leg["w"] = [nw.get(n, 0.0) for n in leg["name"]]
                # largest suggested share first; holdings being sold (0%)
                # stay listed at the bottom — that IS part of the answer
                leg = leg.sort_values("w", ascending=False, kind="stable")
                if folded:
                    leg = pd.concat([leg, pd.DataFrame([{
                        "name": "Other", "color": OTHER_COLOR,
                        "w": nw.get("Other", 0.0)}])], ignore_index=True)
                st.markdown('<div class="muted" style="margin:.5rem 0 .2rem">'
                            'Suggested shares</div>', unsafe_allow_html=True)
                st.markdown(legend_html(leg, descs), unsafe_allow_html=True)

            # ---- by asset class (full catalogue only) ---------------------
            if R.get("universe"):
                used = int((w_opt >= uni.DUST_PCT / 100).sum())
                section("By asset class",
                        f"The suggestion uses <b>{used} of {n_lines}</b> asset "
                        f"classes (holdings under {uni.DUST_PCT:.2f}% are left "
                        f"out).")
                cat_df = uni.category_table(opt_cats, cur_w, w_opt)
                ccols = st.columns(len(cat_df))
                for col, (_, row) in zip(ccols, cat_df.iterrows()):
                    d = row["Change (pp)"]
                    col.markdown(card(
                        CATEGORY_PLAIN.get(row["Category"], row["Category"]),
                        f'<span class="was">{row["Current %"]:.0f}%</span>'
                        f'<span class="arrow">→</span>{row["Optimal %"]:.0f}%',
                        f'{d:+.1f} pts'.replace("-", "−")),
                        unsafe_allow_html=True)

            # ---- what changes ---------------------------------------------
            section("What would change",
                    "Percentage points of the portfolio moved into (blue) or out of "
                    "(orange) each holding.")
            if R.get("universe"):
                chg_names = [plain_name(jpm.LABEL.get(x, x))[0] for x in opt_lines]
                chg = change_chart(chg_names, cur_w, w_opt)
            else:
                chg = change_chart(POOL_NAMES, cur_w, w_opt)
            if chg is None:
                st.info("Your current mix already meets this goal — no changes "
                        "suggested.")
            else:
                st.altair_chart(chg, width='stretch')

            # ---- apply ----------------------------------------------------
            ac1, ac2 = st.columns([1, 2.5], vertical_alignment="center")
            btn = ("Use this mix (switches to Custom)"
                   if R.get("universe") and not is_base else "Use this mix")
            if ac1.button(btn, type="primary", width='stretch', icon=":material/check:"):
                if is_base:
                    # Standard can hold any LTCMA line now, so a catalogue
                    # optimum no longer has to be pushed into Custom: it is
                    # written straight into the Standard list, which keeps the
                    # locked house-view returns the mode exists to protect.
                    keep = [ln for ln in R["lines"]
                            if float(w_opt[R["lines"].index(ln)]) * 100
                            >= uni.DUST_PCT] if R.get("universe") else list(R["lines"])
                    tot = sum(float(w_opt[R["lines"].index(ln)]) for ln in keep)
                    st.session_state.std_lines = [
                        ln for ln in st.session_state.std_lines if ln in keep
                    ] + [ln for ln in keep if ln not in st.session_state.std_lines]
                    st.session_state.std_w = {
                        ln: round(float(w_opt[R["lines"].index(ln)]) / tot * 100, 4)
                        for ln in keep}
                    st.session_state.std_tick = {ln: True for ln in keep}
                elif R.get("universe"):
                    st.session_state.custom_alloc = applied.copy()
                    st.session_state.pending_mode = MODE_CUSTOM
                else:
                    ca = st.session_state.custom_alloc.copy()
                    ca["Weight %"] = [round(float(x) * 100, 4) for x in w_opt]
                    st.session_state.custom_alloc = ca
                # MUST bump — see bump_editors().
                bump_editors()
                st.rerun()
            ac2.markdown(
                '<div class="muted">Replaces the mix on the <b>Your mix</b> tab. '
                'You can always go back with <i>Reset</i> there.'
                + (f" Holdings under {uni.DUST_PCT:.2f}% are dropped and the rest "
                   f"scaled to 100%." if R.get("universe") else "")
                + (" The suggested assets stay in Standard, on Keepp's own "
                   "return and risk figures." if is_base and R.get("universe")
                   else "")
                + '</div>', unsafe_allow_html=True)

            st.write("")
            st.markdown(
                '<div class="note"><b>Treat this as a starting point for a '
                'conversation, not a final answer.</b> The suggestion is very '
                'sensitive to the return assumptions — a small change in one '
                'forecast can shift the mix a lot — and the model assumes '
                'investments keep moving together the way they normally do, '
                'which often fails in a crisis.</div>', unsafe_allow_html=True)

            # ---- for the analyst ------------------------------------------
            with st.expander("Risk/return map (efficient frontier)"):
                if R.get("frontier") is None:
                    st.markdown(
                        '<div class="muted">The map traces the best possible '
                        'return at every risk level. Across all asset classes it '
                        'takes about 15 seconds, so it is drawn only on request.'
                        '</div>', unsafe_allow_html=True)
                    if st.button("Draw the map", icon=":material/show_chart:"):
                        with st.spinner("Tracing the frontier across all asset "
                                        "classes…"):
                            try:
                                _lim = R["limits"]
                                st.session_state.opt_result["frontier"] = (
                                    opt.efficient_frontier(
                                        opt_mu, opt_cov, _lim,
                                        opt.feasible_start(_lim), 24))
                            except Exception as e:
                                st.error(f"Could not draw the map: {e}")
                            else:
                                st.rerun()
                else:
                    f_r, f_v, _ = R["frontier"]
                    fdf = pd.DataFrame({"vol": f_v * 100, "ret": f_r * 100})
                    pts = pd.DataFrame([
                        {"vol": v_cur * 100, "ret": r_cur * 100, "what": "Today",
                         "sharpe": sh_cur},
                        {"vol": v_opt * 100, "ret": r_opt * 100, "what": "Suggested",
                         "sharpe": sh_opt}])
                    assets = pd.DataFrame({
                        "vol": opt_vols * 100, "ret": opt_mu * 100,
                        "what": [plain_name(jpm.LABEL.get(x, x))[0] for x in opt_lines]})

                    def _axis(title):
                        return alt.Axis(title=title, labelColor=INK_3, titleColor=INK_3,
                                        gridColor=GRID, labelFontSize=11, format=".0f",
                                        titleFontWeight=400, domain=False)

                    x_enc = alt.X("vol:Q", axis=_axis("Yearly swings (volatility) %"),
                                  scale=alt.Scale(zero=True, nice=True))
                    y_enc = alt.Y("ret:Q", axis=_axis("Average return per year %"),
                                  scale=alt.Scale(zero=False, nice=True))
                    layers = [
                        alt.Chart(assets).mark_circle(size=50, color=INK_3,
                                                      opacity=.45).encode(
                            x=x_enc, y=y_enc,
                            tooltip=[alt.Tooltip("what:N", title="Holding"),
                                     alt.Tooltip("ret:Q", format=".2f", title="Return %"),
                                     alt.Tooltip("vol:Q", format=".2f", title="Swings %")]),
                        alt.Chart(fdf).mark_line(color=ACCENT, strokeWidth=2).encode(
                            x=x_enc, y=y_enc),
                    ]
                    if R["objective"] == OBJ_SHARPE and v_opt > 0:
                        v_max = float(max(f_v.max(), v_cur, opt_vols.max())) * 100
                        cml = pd.DataFrame({"vol": [0.0, v_max],
                                            "ret": [rf * 100, rf * 100 + sh_opt * v_max]})
                        layers.append(alt.Chart(cml).mark_line(
                            color=ACCENT, strokeDash=[4, 4], strokeWidth=1,
                            opacity=.6).encode(x=x_enc, y=y_enc))
                    if R.get("target_pct") is not None:
                        layers.append(alt.Chart(pd.DataFrame(
                            [{"ret": float(R["target_pct"])}])).mark_rule(
                            color=INK_3, strokeDash=[3, 3]).encode(y="ret:Q"))
                    if R.get("budget_pct") is not None:
                        layers.append(alt.Chart(pd.DataFrame(
                            [{"vol": float(R["budget_pct"])}])).mark_rule(
                            color=INK_3, strokeDash=[3, 3]).encode(x="vol:Q"))
                    layers.append(
                        alt.Chart(pts).mark_point(size=200, filled=True, opacity=1,
                                                  stroke="white", strokeWidth=2).encode(
                            x=x_enc, y=y_enc,
                            color=alt.Color("what:N", scale=alt.Scale(
                                domain=["Today", "Suggested"], range=[INK_3, ACCENT]),
                                legend=alt.Legend(title=None, orient="top-left")),
                            shape=alt.Shape("what:N", scale=alt.Scale(
                                domain=["Today", "Suggested"],
                                range=["circle", "diamond"]), legend=None),
                            tooltip=[alt.Tooltip("what:N", title=""),
                                     alt.Tooltip("ret:Q", format=".2f", title="Return %"),
                                     alt.Tooltip("vol:Q", format=".2f", title="Swings %"),
                                     alt.Tooltip("sharpe:Q", format=".3f",
                                                 title="Sharpe")]))
                    st.altair_chart(alt.layer(*layers).properties(height=340)
                                    .configure_view(strokeWidth=0), width='stretch')
                    st.caption(
                        "Grey dots are the individual asset classes. The line shows the "
                        "best possible return for each level of risk under your limits "
                        "(the efficient frontier); the dashed line is the capital market "
                        f"line. Sharpe ratio: {sh_cur:.2f} today → {sh_opt:.2f} "
                        f"suggested. Returns here are arithmetic: "
                        f"{r_cur*100:.2f}% → {r_opt*100:.2f}%.")

            with st.expander("Exact numbers"):
                p25, p75 = R.get("stability", (None, None))
                pct_cfg = {
                    "Current %": st.column_config.NumberColumn("Today %", format="%.2f"),
                    "Optimal %": st.column_config.NumberColumn("Suggested %",
                                                               format="%.2f"),
                    "Change (pp)": st.column_config.NumberColumn("Change (pts)",
                                                                 format="%+.2f"),
                }
                if R.get("universe"):
                    cat_df = uni.category_table(opt_cats, cur_w, w_opt)
                    st.markdown("**By asset class**")
                    st.dataframe(cat_df, hide_index=True, width='stretch',
                                 column_config={"Category": st.column_config.TextColumn(
                                     "Asset class", width="medium"), **pct_cfg})
                    tabs_ = uni.line_tables(opt_lines, opt_cats, cur_w, w_opt,
                                            band=None if p25 is None else (p25, p75))
                    for cat in cat_df["Category"]:
                        t = tabs_.get(cat)
                        if t is None or t.empty:
                            continue
                        st.markdown(f"**{cat}**")
                        st.dataframe(t, hide_index=True, width='stretch',
                                     column_config={"LTCMA line": st.column_config
                                                    .TextColumn("Investment",
                                                                width="large"),
                                                    **pct_cfg})
                    st.caption(f"Lines under {uni.DUST_PCT:.2f}% are summed into a "
                               f"single residual row so each table still adds up.")
                else:
                    rows = []
                    for i, nm in enumerate(opt_lines):
                        row = {"Holding": plain_name(jpm.LABEL.get(nm, nm))[0],
                               "LTCMA line": jpm.LABEL.get(nm, nm),
                               "Current %": cur_w[i] * 100,
                               "Optimal %": w_opt[i] * 100,
                               "Change (pp)": (w_opt[i] - cur_w[i]) * 100}
                        if p25 is not None:
                            row["Stable range %"] = (f"{p25[i]*100:.0f} – "
                                                     f"{p75[i]*100:.0f}")
                        rows.append(row)
                    st.dataframe(pd.DataFrame(rows), hide_index=True,
                                 width='stretch',
                                 column_config={"LTCMA line": st.column_config
                                                .TextColumn("J.P. Morgan line"),
                                                **pct_cfg})
                if p25 is not None:
                    widest = int(np.argmax(p75 - p25))
                    st.caption(
                        f"**Stability:** across 40 re-solves with each expected "
                        f"return nudged by 0.5%, the least stable holding is "
                        f"**{plain_name(jpm.LABEL.get(opt_lines[widest], opt_lines[widest]))[0]}**"
                        f", whose suggested share ranges "
                        f"{p25[widest]*100:.0f}–{p75[widest]*100:.0f}%. A wide range "
                        f"means the suggestion for that holding is not reliable.")
        elif R:
            st.info("Your mix or the search options changed since the last "
                    "suggestion. Press **Suggest a mix** again.")


# ==========================================================================
# TAB — Details & saved runs
# ==========================================================================
with tab_det:
    section("How the range of outcomes changes with time",
            f"Value of {money(amount)} after each period. The thick bar holds half "
            f"of all futures; the thin bar holds 9 in 10; the dot is the typical "
            f"outcome.")
    rr = pd.DataFrame(res.rows)
    rng_df = pd.DataFrame({
        "h": rr["Horizon"],
        "lo": rr["5th x"] * amount, "hi": rr["95th x"] * amount,
        "med": rr["Median x"] * amount,
        "q1": (1 + rr["25th"]) ** rr["_years"] * amount,
        "q3": (1 + rr["75th"]) ** rr["_years"] * amount,
    })
    for c in ("lo", "hi", "med", "q1", "q3"):
        rng_df[c + "_l"] = rng_df[c].map(money)
    order = list(rng_df["h"])
    yh = alt.Y("h:N", sort=order, title=None,
               axis=alt.Axis(labelFontSize=12, labelColor=INK, domain=False,
                             ticks=False, labelPadding=10))
    xh = alt.X("lo:Q", title=None, scale=alt.Scale(zero=False, nice=True),
               axis=alt.Axis(format="$.3~s", labelColor=INK_3, gridColor=GRID,
                             domain=False, ticks=False))
    tip = [alt.Tooltip("h:N", title="After"),
           alt.Tooltip("hi_l:N", title="Strong markets"),
           alt.Tooltip("q3_l:N", title="Better half starts"),
           alt.Tooltip("med_l:N", title="Typical"),
           alt.Tooltip("q1_l:N", title="Weaker half ends"),
           alt.Tooltip("lo_l:N", title="Weak markets")]
    rb = alt.Chart(rng_df)
    wide = rb.mark_bar(height=8, color=BAND_OUTER, cornerRadius=4).encode(
        y=yh, x=xh, x2="hi:Q", tooltip=tip)
    mid = rb.mark_bar(height=18, color=BAND_INNER, cornerRadius=4).encode(
        y=alt.Y("h:N", sort=order), x="q1:Q", x2="q3:Q", tooltip=tip)
    med = rb.mark_circle(size=110, color=ACCENT_DK, stroke="white",
                         strokeWidth=2, opacity=1).encode(
        y=alt.Y("h:N", sort=order), x="med:Q", tooltip=tip)
    medt = rb.mark_text(dy=-18, fontSize=12, fontWeight=700, color=ACCENT_DK).encode(
        y=alt.Y("h:N", sort=order), x="med:Q", text="med_l:N")
    start_r = alt.Chart(pd.DataFrame({"x": [amount]})).mark_rule(
        color=INK_3, strokeDash=[4, 4]).encode(x="x:Q")
    st.altair_chart((wide + mid + start_r + med + medt).properties(
        height=max(64 * len(rng_df), 140)).configure_view(strokeWidth=0),
        width='stretch')

    section("The full numbers", "Growth per year at different points in the range "
            "of outcomes (5th percentile = only 1 in 20 futures does worse).")
    df = pd.DataFrame(res.rows).drop(columns=["_years"])
    pct_cols = ["5th", "25th", "Median", "75th", "95th", "Mean", "P(loss)"]
    x_cols = ["Median x", "5th x", "95th x"]
    disp = df.copy()
    for c in pct_cols:
        disp[c] = disp[c] * 100
    NICE = {"5th": "Weak (5th)", "25th": "25th", "Median": "Typical (median)",
            "75th": "75th", "95th": "Strong (95th)", "Mean": "Average",
            "P(loss)": "Chance of loss", "Median x": "Typical ×",
            "5th x": "Weak ×", "95th x": "Strong ×"}
    st.dataframe(
        disp, hide_index=True, width='stretch',
        column_config={
            "Horizon": st.column_config.TextColumn("After", width="small"),
            **{c: st.column_config.NumberColumn(NICE[c], format="%.2f%%")
               for c in pct_cols},
            **{c: st.column_config.NumberColumn(
                NICE[c], format="%.3f",
                help="Multiple of the starting amount (1.00 = break-even).")
               for c in x_cols},
        })

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(card("Average return / yr", pct(res.exp_arith, 2),
                     "Arithmetic, per annum"), unsafe_allow_html=True)
    m2.markdown(card("Volatility", pct(res.analytic_vol, 2),
                     f"Simulated {pct(res.sim_vol, 2)}"), unsafe_allow_html=True)
    m3.markdown(card("Simulated futures", f"{res.paths:,}",
                     f"{res.elapsed_ms:.0f} ms · seed {seed}"), unsafe_allow_html=True)
    m4.markdown(card("Holdings", f"{int((alloc['Weight %'] > 0).sum())}",
                     "Rebalanced to target every month"), unsafe_allow_html=True)
    st.write("")

    csv = disp.round(4).to_csv(index=False)
    st.download_button("Download table (CSV)", csv,
                       f"keepp_monte_carlo_{'base' if is_base else 'custom'}.csv",
                       "text/csv", icon=":material/download:")

    # ---- saved runs ------------------------------------------------------
    section("Saved scenarios",
            "Save the current mix and results to compare several options side by "
            "side. Saved scenarios last until you close this browser tab — "
            "download the Excel file to keep them.")
    saved = st.session_state.saved_runs
    next_seq = len(saved) + 1
    default_label = rs.auto_label(next_seq, "Base" if is_base else "Custom", years)

    sc1, sc2, sc3 = st.columns([2.4, 1, 1], vertical_alignment="bottom")
    label_in = sc1.text_input("Name this scenario", value=default_label,
                              key=f"run_label_{NONCE}",
                              placeholder="e.g. More stocks")
    if sc2.button("Save scenario", width='stretch', type="primary"):
        st.session_state.saved_runs.append(rs.make_record(
            next_seq, label_in, "Base" if is_base else "Custom",
            alloc, paths, years, seed, res))
        bump_editors()
        st.rerun()
    if sc3.button("Clear all", width='stretch', disabled=not saved):
        st.session_state.saved_runs = []
        st.rerun()

    if saved:
        if len(saved) >= 2:
            cmp = pd.DataFrame([{
                "run": f"{r.seq}. {r.label}",
                "val": r.median_wealth_x * amount,
                "val_l": money(r.median_wealth_x * amount),
                "loss": f"{r.p_loss_pct:.1f}%",
                "yrs": f"{r.years} yrs",
                "vol": f"±{r.analytic_vol_pct:.1f}%"} for r in saved])
            bars = alt.Chart(cmp).mark_bar(color=ACCENT, cornerRadiusEnd=4,
                                           height=18).encode(
                x=alt.X("val:Q", title=None, axis=alt.Axis(
                    format="$.3~s", labelColor=INK_3, gridColor=GRID,
                    domain=False, ticks=False)),
                y=alt.Y("run:N", sort=list(cmp["run"]), title=None,
                        axis=alt.Axis(labelFontSize=12, labelColor=INK,
                                      domain=False, ticks=False, labelLimit=260)),
                tooltip=[alt.Tooltip("run:N", title="Scenario"),
                         alt.Tooltip("val_l:N", title="Typical value"),
                         alt.Tooltip("yrs:N", title="Horizon"),
                         alt.Tooltip("vol:N", title="Yearly swings"),
                         alt.Tooltip("loss:N", title="Chance of loss")])
            txt = alt.Chart(cmp).mark_text(align="left", dx=6, fontSize=12,
                                           fontWeight=600, color=INK_2).encode(
                x="val:Q", y=alt.Y("run:N", sort=list(cmp["run"])), text="val_l:N")
            st.markdown(f'<div class="muted">Typical value of {money(amount)} at '
                        f'the end of each scenario</div>', unsafe_allow_html=True)
            st.altair_chart((bars + txt).properties(
                height=max(36 * len(cmp), 90)).configure_view(strokeWidth=0),
                width='stretch')

        hist_df = rs.history_table(saved)
        st.dataframe(
            hist_df, hide_index=True, width='stretch',
            column_config={
                "#": st.column_config.NumberColumn(width="small"),
                "Label": st.column_config.TextColumn("Scenario", width="medium"),
                "Saved": st.column_config.TextColumn(width="medium"),
                "Paths": st.column_config.NumberColumn("Futures"),
                "Sleeves": st.column_config.NumberColumn("Holdings"),
                "Return %": st.column_config.NumberColumn("Avg return %",
                                                          format="%.2f"),
                "Volatility %": st.column_config.NumberColumn("Swings %",
                                                              format="%.2f"),
                "Median wealth": st.column_config.NumberColumn("Typical ×",
                                                               format="%.3f×"),
                "P(loss) %": st.column_config.NumberColumn("Chance of loss %",
                                                           format="%.2f"),
            })
        st.download_button(
            f"Download Excel ({len(saved)} scenario{'s' if len(saved) != 1 else ''})",
            rs.to_excel_bytes(saved), "keepp_saved_runs.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            icon=":material/download:")
    else:
        st.caption("No scenarios saved yet.")

    # ---- sources ---------------------------------------------------------
    section("Where the assumptions come from")
    with st.expander("J.P. Morgan 2026 LTCMA · sources and method"):
        for name, mini, text in SOURCES:
            a, b = st.columns([1, 3])
            a.markdown(f'<div class="srcname">{name}</div>'
                       f'<div class="srcmini">{mini}</div>', unsafe_allow_html=True)
            b.markdown(f'<div class="srctext">{text}</div>', unsafe_allow_html=True)
            st.write("")

    st.markdown(
        '<div class="foot" style="margin-top:1.2rem">'
        'Modelling limits: correlations are static, with no regime-switching or '
        'stress-period correlation lift inside the simulation even though the '
        'volatility anchors are regime-weighted; lognormal returns are thin-tailed '
        'relative to realised markets; figures are nominal USD with no fees, taxes or '
        'cashflows. The "bad year" figure on the Overview is a normal approximation '
        '(average return − 1.645 × volatility), not a simulated number.'
        '</div>', unsafe_allow_html=True)
