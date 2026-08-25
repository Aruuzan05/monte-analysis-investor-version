"""
Keepp Capital — Target Allocation Monte Carlo
Streamlit front end.

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

from engine import simulate

# --------------------------------------------------------------------------
# Page config + palette
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Keepp Allocation Simulator",
    page_icon="◧",
    layout="wide",
    initial_sidebar_state="expanded",
)

INK = "#0F1719"
INK_3 = "#6C7C7D"
ACCENT = "#00877A"
BAND_OUTER = "#D3E9E5"
BAND_INNER = "#8FC7BE"
BRICK = "#BC3D18"
GRID = "#DFE7E6"

# System font stacks only. "Microsoft YaHei" / "PingFang SC" keep the UI sane on
# a Chinese-locale OS without fetching anything.
SANS = ('-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, '
        '"Helvetica Neue", "PingFang SC", "Microsoft YaHei", Arial, sans-serif')
MONO = ('ui-monospace, SFMono-Regular, "SF Mono", Consolas, '
        '"Liberation Mono", Menlo, monospace')

st.markdown(f"""
<style>
  html, body, [class*="css"], .stApp {{ font-family: {SANS}; }}
  .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }}

  /* numbers read as data, not prose */
  [data-testid="stMetricValue"] {{
      font-family: {MONO}; font-size: 1.85rem; font-weight: 500;
      letter-spacing: -.02em; font-variant-numeric: tabular-nums;
  }}
  [data-testid="stMetricLabel"] p {{
      font-family: {MONO}; font-size: .68rem; letter-spacing: .1em;
      text-transform: uppercase; color: {INK_3};
  }}
  [data-testid="stMetricDelta"] {{ font-size: .75rem; }}
  [data-testid="stMetric"] {{
      background: var(--secondary-background-color);
      border: 1px solid {GRID}; border-radius: 6px; padding: 14px 16px;
  }}
  div[data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}

  h1 {{ font-size: 1.85rem; font-weight: 600; letter-spacing: -.015em; margin-bottom: .1rem; }}
  h2 {{ font-size: 1.12rem; font-weight: 600; letter-spacing: -.005em;
        margin-top: 1.9rem; margin-bottom: .5rem; }}
  h3 {{ font-size: .95rem; font-weight: 600; }}

  .eyebrow {{ font-family: {MONO}; font-size: .68rem; letter-spacing: .14em;
             text-transform: uppercase; color: {ACCENT}; margin-bottom: .35rem; }}
  .sub {{ color: {INK_3}; font-size: .88rem; max-width: 78ch; line-height: 1.55; }}
  .foot {{ color: {INK_3}; font-size: .78rem; line-height: 1.6; }}
  .srcname {{ font-weight: 600; font-size: .85rem; }}
  .srcmini {{ font-family: {MONO}; font-size: .7rem; color: {INK_3}; }}
  .srctext {{ font-size: .82rem; line-height: 1.65; color: var(--text-color); opacity: .82; }}

  /* Stat cards. These live in CSS, NOT in inline style attributes: the font
     stacks contain double-quoted family names, which would terminate a
     style="..." attribute and silently drop the whole rule. */
  .statcard {{ background: var(--secondary-background-color); border: 1px solid {GRID};
              border-radius: 6px; padding: 14px 16px; height: 100%; }}
  .statk {{ font-family: {MONO}; font-size: .68rem; letter-spacing: .1em;
           text-transform: uppercase; color: {INK_3}; margin-bottom: 6px; }}
  .statv {{ font-family: {MONO}; font-size: 1.8rem; font-weight: 500;
           letter-spacing: -.02em; line-height: 1; font-variant-numeric: tabular-nums; }}
  .statv.hot {{ color: {BRICK}; }}
  .statn {{ font-size: .72rem; color: {INK_3}; margin-top: 7px; }}

  /* Streamlit's own heading classes outrank a bare h1/h2 selector. */
  .stMarkdown h1 {{ font-size: 1.9rem !important; }}
  .stMarkdown h2 {{ font-size: 1.15rem !important; margin-top: 2rem !important; }}

  section[data-testid="stSidebar"] {{ width: 452px !important; }}
  section[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}
  hr {{ margin: .9rem 0; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Optional password gate (set app_password in .streamlit/secrets.toml)
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

    st.markdown("### Keepp Allocation Simulator")
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
# Hardcoded J.P. Morgan 2026 LTCMA defaults
# --------------------------------------------------------------------------
ASSETS = ["Fixed Income", "Public Equities & Mutual Funds",
          "Private Equity & Venture Capital", "Commodities & Real Assets",
          "Cash & Equivalents"]
TAGS = ["FI", "EQ", "PE", "CMD", "CASH"]
# The sidebar is 400px wide; the full names below clip the numeric columns, so
# the editor shows these and the sources section carries the full names.
SHORT = ["Fixed Income", "Public Equities", "Private Equity", "Commodities", "Cash"]
DEF_W = [45.0, 20.0, 10.0, 8.0, 17.0]
DEF_R = [5.00, 12.88, 11.90, 6.10, 3.80]
DEF_V = [7.02, 14.47, 19.80, 18.30, 0.67]
DEF_CORR = [[1.00, 0.68, 0.30, 0.10, 0.00],
            [0.68, 1.00, 0.80, 0.30, 0.00],
            [0.30, 0.80, 1.00, 0.25, 0.00],
            [0.10, 0.30, 0.25, 1.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 1.00]]
EQ_BASE, EQ_CONS = 12.88, 7.00
EQ_IDX = 1

SOURCES = [
    ("Fixed Income", "5.00% · 7.02%",
     "Return: index YTM of global IG corporates — iShares Global Corp Bond UCITS trailing "
     "distribution yield 4.26% plus pull-to-par, cross-checked against JPM 2026 LTCMA US IG "
     "at 5.2%. Volatility: regime anchor √(25%·3.3%² + 50%·6.1%² + 25%·10.6%²) = 7.02%."),
    ("Public Equities & Mutual Funds", "12.88% · 14.47%",
     "Return: MSCI World 10-year annualized total return (iShares MSCI World ETF URTH, "
     "adjusted close, Jul-2016 to Jul-2026). Volatility: regime anchor "
     "√(25%·5.1%² + 50%·13.9%² + 25%·20.6%²) = 14.47%. The Conservative scenario "
     "substitutes the JPM 7.0% forward return."),
    ("Private Equity & Venture Capital", "11.90% · 19.80%",
     "J.P. Morgan 2026 Long-Term Capital Market Assumptions, Private Equity (USD)."),
    ("Commodities & Real Assets", "6.10% · 18.30%",
     "J.P. Morgan 2026 LTCMA, broad commodities basket (USD)."),
    ("Cash & Equivalents", "3.80% · 0.67%",
     "Return: current USD risk-free rate, 3-month T-bill at 3.80% (15-Jul-2026). "
     "Volatility: JPM 2026 LTCMA US Cash. Includes the 2% digital-assets sleeve, "
     "folded into cash per instruction."),
    ("Method", "Regime-based vol",
     "Returns are entered as geometric per annum and simulated as correlated lognormal "
     "monthly draws, with the portfolio rebalanced to target weights every month. Vol "
     "anchors come from 120 months of returns split into calm / normal / stress regimes by "
     "the bottom quartile, middle 50% and top quartile of 6-month rolling MSCI World "
     "volatility, weighted 25 / 50 / 25. Regimes are classified on equity volatility and "
     "applied to both assets, so the bond stress figure reflects equity-stress periods "
     "such as 2020 and 2022."),
]


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def bump_editors():
    """Force the data_editors to remount.

    st.data_editor keyed with a constant string keeps the user's edits as a
    delta and re-applies them on top of whatever dataframe it is handed. So
    writing a new dataframe into session_state is not enough — after a
    Normalize the edited cell would be restored to its pre-normalized value
    (weights summing to 94.71% instead of 100%). Changing the widget key
    remounts the editor with a clean slate.
    """
    st.session_state.editor_nonce = st.session_state.get("editor_nonce", 0) + 1


def reset_state():
    bump_editors()
    st.session_state.alloc = pd.DataFrame(
        {"Asset class": SHORT, "Weight %": DEF_W,
         "Return %": DEF_R, "Vol %": DEF_V})
    st.session_state.corr = pd.DataFrame(DEF_CORR, index=TAGS, columns=TAGS)


if "alloc" not in st.session_state:
    st.session_state.editor_nonce = 0
    reset_state()

NONCE = st.session_state.get("editor_nonce", 0)


@st.cache_data(show_spinner=False, max_entries=32)
def run_sim(w, r, v, corr_flat, paths, years, seed):
    corr = np.array(corr_flat, dtype=float).reshape(len(w), len(w))
    return simulate(list(w), list(r), list(v), corr, paths, years, seed)


# --------------------------------------------------------------------------
# Sidebar — inputs
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="eyebrow">Keepp Capital</div>', unsafe_allow_html=True)
    st.markdown("### Assumptions")

    c1, c2 = st.columns(2)
    if c1.button("Base · 12.88%", width='stretch',
                 help="MSCI World 10-year historical equity return"):
        st.session_state.alloc.loc[EQ_IDX, "Return %"] = EQ_BASE
        bump_editors()
        st.rerun()
    if c2.button("Conservative · 7.0%", width='stretch',
                 help="J.P. Morgan forward-looking equity return"):
        st.session_state.alloc.loc[EQ_IDX, "Return %"] = EQ_CONS
        bump_editors()
        st.rerun()

    alloc = st.data_editor(
        st.session_state.alloc,
        key=f"alloc_editor_{NONCE}",
        hide_index=True,
        width='stretch',
        num_rows="fixed",
        column_config={
            # step MUST stay 0.01: a coarser step makes the editor snap the
            # *displayed* value to it, so 12.88 renders as 12.80 and a single
            # click on the cell would commit the wrong number.
            "Asset class": st.column_config.TextColumn(disabled=True, width="small"),
            "Weight %": st.column_config.NumberColumn(
                min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
            "Return %": st.column_config.NumberColumn(
                min_value=-20.0, max_value=60.0, step=0.01, format="%.2f"),
            "Vol %": st.column_config.NumberColumn(
                min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
        },
    )
    st.session_state.alloc = alloc

    wsum = float(alloc["Weight %"].sum())
    off = abs(wsum - 100.0) > 0.005
    lc, rc = st.columns([2, 1])
    lc.markdown(
        f'<div style="font-family:{MONO};font-size:.8rem;padding-top:.45rem">'
        f'TOTAL &nbsp;<b style="color:{BRICK if off else ACCENT}">{wsum:.2f}%</b></div>',
        unsafe_allow_html=True)
    if off and rc.button("Normalize", width='stretch'):
        if wsum > 0:
            st.session_state.alloc["Weight %"] = (
                st.session_state.alloc["Weight %"] / wsum * 100).round(4)
            bump_editors()
            st.rerun()
        else:
            st.warning("Enter at least one positive weight.")

    with st.expander("Advanced"):
        paths = st.select_slider(
            "Simulation paths", options=[1_000, 2_500, 5_000, 10_000, 25_000, 50_000],
            value=10_000)
        years = st.slider("Horizon (years)", 1, 30, 10)
        seed = st.number_input("Random seed", value=20260722, step=1,
                               help="Fix the seed so everyone running the same "
                                    "inputs sees the same numbers.")
        st.caption("Correlation matrix")
        corr_edit = st.data_editor(
            st.session_state.corr, key=f"corr_editor_{NONCE}", width='stretch',
            column_config={t: st.column_config.NumberColumn(
                min_value=-1.0, max_value=1.0, step=0.01, format="%.2f") for t in TAGS},
        )
        st.session_state.corr = corr_edit
        st.caption("Edits are mirrored across the diagonal before the run. "
                   "EQ–FI 0.68 is the 10-year monthly correlation; PE–EQ 0.80 "
                   "follows the J.P. Morgan de-smoothed convention.")

    st.divider()
    if st.button("Reset to JPM LTCMA defaults", width='stretch'):
        reset_state()
        st.rerun()


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown('<div class="eyebrow">Keepp Capital · Target Allocation</div>',
            unsafe_allow_html=True)
st.markdown("# Monte Carlo Allocation Simulator")
st.markdown(
    '<p class="sub">Correlated lognormal paths, monthly steps, rebalanced to target '
    'weights each month. Defaults are the J.P. Morgan 2026 Long-Term Capital Market '
    'Assumptions used in the July 2027 deck.</p>', unsafe_allow_html=True)
st.write("")

if off:
    st.error(f"**Weights total {wsum:.2f}%.** They must sum to 100% before the "
             f"simulation can run. Adjust a weight in the sidebar, or use Normalize "
             f"to scale them all proportionally.")
    st.stop()

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
corr_sym = st.session_state.corr.to_numpy(dtype=float)
corr_sym = (corr_sym + corr_sym.T) / 2.0
np.fill_diagonal(corr_sym, 1.0)

with st.spinner("Running simulation…"):
    res, repaired = run_sim(
        tuple(alloc["Weight %"]), tuple(alloc["Return %"]), tuple(alloc["Vol %"]),
        tuple(corr_sym.flatten()), int(paths), int(years), int(seed))

if repaired:
    st.warning("The correlation matrix you entered was not internally consistent "
               "(not positive semi-definite), so it was adjusted to the nearest "
               "valid matrix before simulating. Results are usable, but check the "
               "correlations you edited.")

term = res.rows[-1]

# st.metric's delta slot always draws an up/down arrow, which would imply a
# change against a prior period that does not exist here. Captions instead.
stat_cols = st.columns(4)
STATS = [
    ("Expected return", f"{res.exp_arith*100:.2f}%", "Arithmetic, per annum"),
    ("Portfolio volatility", f"{res.analytic_vol*100:.2f}%",
     f"Simulated {res.sim_vol*100:.2f}%"),
    (f"Median wealth · {term['_years']}y", f"{term['Median x']:.2f}×",
     f"{term['Median']*100:.2f}% annualized"),
    (f"P(loss) · {term['_years']}y", f"{term['P(loss)']*100:.2f}%",
     "Below 1.00× at horizon end"),
]
for col, (label, value, note) in zip(stat_cols, STATS):
    hot = " hot" if (label.startswith("P(loss)") and term["P(loss)"] > 0.05) else ""
    col.markdown(
        f'<div class="statcard"><div class="statk">{label}</div>'
        f'<div class="statv{hot}">{value}</div>'
        f'<div class="statn">{note}</div></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Fan chart
# --------------------------------------------------------------------------
st.markdown("## Wealth fan chart")

fan = pd.DataFrame(res.fan, columns=["p05", "p25", "p50", "p75", "p95"])
fan["Year"] = res.fan_years
for c in ("p05", "p25", "p50", "p75", "p95"):
    fan[c + "_lbl"] = fan[c].map(lambda v: f"{v:.3f}×")

log_scale = st.checkbox("Log scale", value=False)
yscale = alt.Scale(type="log", nice=False) if log_scale else alt.Scale(zero=False, nice=True)

axis_y = alt.Axis(title="Portfolio value (multiple of start)", format=".2f", labelFont="monospace",
                  labelColor=INK_3, titleColor=INK_3, gridColor=GRID, domain=False,
                  tickCount=6, labelFontSize=11)
axis_x = alt.Axis(title="Years from today", titleColor=INK_3,
                  labelFont="monospace", labelColor=INK_3,
                  grid=False, tickCount=min(res.years, 10), labelFontSize=11)

base = alt.Chart(fan)
outer = base.mark_area(color=BAND_OUTER).encode(
    x=alt.X("Year:Q", axis=axis_x, scale=alt.Scale(nice=False)),
    y=alt.Y("p05:Q", axis=axis_y, scale=yscale), y2="p95:Q")
inner = base.mark_area(color=BAND_INNER).encode(
    x="Year:Q", y=alt.Y("p25:Q", scale=yscale), y2="p75:Q")
breakeven = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(
    color=INK_3, strokeDash=[3, 3], opacity=.7, size=1).encode(y=alt.Y("y:Q", scale=yscale))
median = base.mark_line(color=ACCENT, strokeWidth=2).encode(
    x="Year:Q", y=alt.Y("p50:Q", scale=yscale))

hover = alt.selection_point(fields=["Year"], nearest=True, on="pointerover",
                            empty=False, clear="pointerout")
rule = base.mark_rule(color=INK_3, strokeDash=[3, 3]).encode(
    x="Year:Q",
    opacity=alt.condition(hover, alt.value(0.8), alt.value(0)),
    tooltip=[alt.Tooltip("Year:Q", format=".2f"),
             alt.Tooltip("p95_lbl:N", title="95th"),
             alt.Tooltip("p75_lbl:N", title="75th"),
             alt.Tooltip("p50_lbl:N", title="Median"),
             alt.Tooltip("p25_lbl:N", title="25th"),
             alt.Tooltip("p05_lbl:N", title="5th")],
).add_params(hover)
dot = base.mark_point(color=ACCENT, filled=True, size=60).encode(
    x="Year:Q", y=alt.Y("p50:Q", scale=yscale),
    opacity=alt.condition(hover, alt.value(1), alt.value(0)))

chart = (outer + inner + breakeven + median + rule + dot).properties(height=400)
chart = chart.configure_view(strokeWidth=0).configure_axis(
    labelFont="monospace", titleFontSize=11)
st.altair_chart(chart, width='stretch')

lg1, lg2, lg3, lg4 = st.columns([1.3, 1.3, 1, 3])
for col, colr, lbl in ((lg1, BAND_OUTER, "5th – 95th percentile"),
                       (lg2, BAND_INNER, "25th – 75th percentile"),
                       (lg3, ACCENT, "Median path")):
    col.markdown(
        f'<div style="font-size:.78rem;display:flex;align-items:center;gap:7px">'
        f'<span style="width:14px;height:10px;border-radius:2px;background:{colr};'
        f'display:inline-block"></span>{lbl}</div>', unsafe_allow_html=True)
lg4.markdown(f'<div class="foot" style="text-align:right">'
             f'{res.paths:,} paths · {res.elapsed_ms:.0f} ms</div>',
             unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Results table
# --------------------------------------------------------------------------
st.markdown("## Annualized return percentiles")

df = pd.DataFrame(res.rows).drop(columns=["_years"])
pct_cols = ["5th", "25th", "Median", "75th", "95th", "Mean", "P(loss)"]
x_cols = ["Median x", "5th x", "95th x"]

# format="percent" trims trailing zeros (10.00% renders as "10%"), which
# breaks the decimal alignment across the row. Scale to percent points and
# format explicitly instead.
disp = df.copy()
for c in pct_cols:
    disp[c] = disp[c] * 100

st.dataframe(
    disp, hide_index=True, width='stretch',
    column_config={
        "Horizon": st.column_config.TextColumn(width="small"),
        **{c: st.column_config.NumberColumn(c, format="%.2f%%") for c in pct_cols},
        **{c: st.column_config.NumberColumn(c.replace(" x", " ×"), format="%.3f")
           for c in x_cols},
    },
)

csv = disp.round(4).to_csv(index=False)
d1, d2 = st.columns([1, 4])
d1.download_button("Download CSV", csv, "keepp_monte_carlo.csv", "text/csv",
                   width='stretch')
d2.markdown(
    f'<div class="foot" style="padding-top:.5rem">'
    f'{res.paths:,} paths · {res.years}-year horizon · seed {seed} · monthly rebalancing. '
    f'P(loss) is the probability the portfolio sits below its starting value at the end '
    f'of the horizon; the × columns are wealth multiples on a 1.00× start.</div>',
    unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------
st.markdown("## Where the defaults come from")
with st.expander("J.P. Morgan 2026 LTCMA · sources and method", expanded=False):
    for name, mini, text in SOURCES:
        a, b = st.columns([1, 3])
        a.markdown(f'<div class="srcname">{name}</div>'
                   f'<div class="srcmini">{mini}</div>', unsafe_allow_html=True)
        b.markdown(f'<div class="srctext">{text}</div>', unsafe_allow_html=True)
        st.write("")

st.markdown(
    '<div class="foot" style="margin-top:1.4rem">'
    'Modelling limits worth stating to any reader: correlations are static, with no '
    'regime-switching or stress-period correlation lift inside the simulation even though '
    'the volatility anchors are regime-weighted; lognormal returns are thin-tailed relative '
    'to realised markets; figures are nominal USD with no fees, taxes or cashflows.'
    '</div>', unsafe_allow_html=True)
