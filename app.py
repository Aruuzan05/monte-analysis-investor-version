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

import jpm_ltcma as jpm
from engine import expand_correlation, simulate

# --------------------------------------------------------------------------
# Page config + palette
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Keepp Allocation Simulator",
    page_icon="◧",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

  div[data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}

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

  /* Mode chip shown next to the page title. */
  .modechip {{ display: inline-block; font-family: {MONO}; font-size: .66rem;
              letter-spacing: .1em; text-transform: uppercase; padding: 3px 8px;
              border-radius: 4px; vertical-align: middle; margin-left: 10px; }}
  .modechip.base {{ background: {BAND_OUTER}; color: #00655B; }}
  .modechip.custom {{ background: #F3E7DF; color: {BRICK}; }}

  /* Streamlit's own heading classes outrank a bare h1/h2 selector. */
  .stMarkdown h1 {{ font-size: 1.9rem !important; }}
  .stMarkdown h2 {{ font-size: 1.15rem !important; margin-top: 2rem !important; }}

  section[data-testid="stSidebar"] {{ width: 372px !important; }}
  section[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}
  hr {{ margin: .9rem 0; }}

  @media (max-width: 430px) {{
    .statv {{ font-size: 1.45rem; }}
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
# Base case — J.P. Morgan 2026 LTCMA (hardcoded, never edited in Base mode)
#
# Diversified Hedge Funds carries a 0% base weight on purpose: the July 2027
# deck's target allocation has no hedge fund sleeve, so a zero weight keeps the
# Base case numerically identical to the signed-off Excel workbook. Allocate to
# it in Custom mode.
# --------------------------------------------------------------------------
ASSETS = ["Fixed Income", "Public Equities & Mutual Funds",
          "Private Equity & Venture Capital", "Diversified Hedge Funds",
          "Commodities & Real Assets", "Cash & Equivalents"]
TAGS = ["FI", "EQ", "PE", "HF", "CMD", "CASH"]
# The sidebar is 452px wide; the full names above clip the numeric columns, so
# the editor shows these and the sources section carries the full names.
SHORT = ["Fixed Income", "Public Equities", "Private Equity",
         "Hedge Funds", "Commodities", "Cash"]

DEF_W = [45.0, 20.0, 10.0, 0.0, 8.0, 17.0]
# Returns are COMPOUND (geometric). engine.py converts with ln(1+g)/12 and
# derives the arithmetic return as g + sigma^2/2, so feeding JPM's arithmetic
# column here would double-count the variance drag. The original workbook did
# that for Private Equity (11.90 instead of 10.20) and Commodities (6.15
# instead of 4.60); both are corrected below, as is the hedge fund volatility
# (5.75 is JPM's published figure, replacing an earlier 6.50 estimate).
DEF_R = [5.00, 12.88, 10.20, 5.30, 4.60, 3.80]
DEF_V = [7.02, 14.47, 19.78, 5.75, 18.32, 0.67]

# Which LTCMA line each bucket resolves to. Three of the six are deliberately
# NOT a JPM line and stay hand-set — that is visible in the UI rather than
# buried in a footnote.
BASE_SUB = ["Manual — global IG corporate blend",
            "Manual — MSCI World 10y history",
            jpm.LABEL["Private Equity"],
            jpm.LABEL["Diversified Hedge Funds"],
            jpm.LABEL["Commodities"],
            "Manual — 3M U.S. T-bill"]
# The same six as dropdown values (Custom mode uses the catalogue's own labels).
CUSTOM_SUB = [jpm.MANUAL, jpm.MANUAL, jpm.LABEL["Private Equity"],
              jpm.LABEL["Diversified Hedge Funds"], jpm.LABEL["Commodities"],
              jpm.MANUAL]
CHOICE_KEYS = ["FI", "EQ", "PE", "HF", "CMD", "CASH"]

# Correlation BETWEEN two sleeves that share a general asset type. J.P. Morgan
# publishes no line-level matrix, so one number per bucket stands in — editable
# in Advanced, and only used once a bucket holds more than one sleeve.
# Real assets get a much lower figure than the rest: commodities, real estate
# and infrastructure genuinely do not move together.
WITHIN_BUCKET = [0.85, 0.85, 0.75, 0.80, 0.45, 0.90]

# Order: FI, EQ, PE, HF, CMD, CASH. Verified positive semi-definite
# (min eigenvalue 0.067), so it factorises without repair.
DEF_CORR = [
    [1.00, 0.68, 0.30, 0.35, 0.10, 0.00],
    [0.68, 1.00, 0.80, 0.75, 0.30, 0.00],
    [0.30, 0.80, 1.00, 0.65, 0.25, 0.00],
    [0.35, 0.75, 0.65, 1.00, 0.35, 0.00],
    [0.10, 0.30, 0.25, 0.35, 1.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
]
N = len(ASSETS)

MODE_BASE = "Base · JPM 2026 LTCMA"
MODE_CUSTOM = "Custom"

SOURCES = [
    ("Fixed Income", "5.00% · 7.02%",
     "Return: index YTM of global IG corporates — iShares Global Corp Bond UCITS trailing "
     "distribution yield 4.26% plus pull-to-par, cross-checked against JPM 2026 LTCMA US IG "
     "at 5.2%. Volatility: regime anchor √(25%·3.3%² + 50%·6.1%² + 25%·10.6%²) = 7.02%."),
    ("Public Equities &amp; Mutual Funds", "12.88% · 14.47%",
     "Return: MSCI World 10-year annualized total return (iShares MSCI World ETF URTH, "
     "adjusted close, Jul-2016 to Jul-2026). Volatility: regime anchor "
     "√(25%·5.1%² + 50%·13.9%² + 25%·20.6%²) = 14.47%."),
    ("Private Equity &amp; Venture Capital", "10.20% · 19.78%",
     "J.P. Morgan 2026 LTCMA, Private Equity (USD), <b>compound</b> return. The "
     "original workbook used 11.90%, which is JPM's <i>arithmetic</i> figure for the "
     "same line — feeding that into a field the model treats as geometric "
     "double-counts the variance drag. Corrected here."),
    ("Diversified Hedge Funds", "5.30% · 5.75%",
     "J.P. Morgan 2026 LTCMA, diversified hedge funds (USD) — leveraged and net of "
     "fees. Both figures are now JPM's published numbers; an earlier build carried a "
     "6.5% volatility estimate because the vol column was not public at the time. "
     "Base weight is 0% because the deck's target allocation has no hedge fund "
     "sleeve — allocate to it in Custom mode."),
    ("Commodities &amp; Real Assets", "4.60% · 18.32%",
     "J.P. Morgan 2026 LTCMA, broad commodities basket (USD), <b>compound</b> return. "
     "As with Private Equity, the original workbook used the arithmetic figure "
     "(6.15%). Corrected here."),
    ("Cash &amp; Equivalents", "3.80% · 0.67%",
     "Return: current USD risk-free rate, 3-month T-bill at 3.80% (15-Jul-2026). "
     "Volatility: JPM 2026 LTCMA US Cash. Includes the 2% digital-assets sleeve, "
     "folded into cash per instruction."),
    ("Sub-categories", "58 LTCMA lines",
     "In Custom mode each bucket can be pointed at any J.P. Morgan 2026 LTCMA line "
     "from its own family — 21 fixed income, 18 equity and the alternatives split "
     "into private markets, hedge funds and real assets. Picking one fills Return and "
     "Vol from the <b>compound</b> column. Editing either number by hand flips the "
     "label back to Manual entry, so a row never claims a JPM provenance for figures "
     "that are no longer JPM's. Fixed Income, Public Equities and Cash ship as Manual "
     "because their base figures are deliberately not LTCMA lines — a global IG "
     "blend, MSCI World realised history, and the live 3-month T-bill. "
     "<b>Note:</b> every forward-looking JPM equity line sits 5-6pp below the 12.88% "
     "realised figure, with higher volatility."),
    ("Correlations", "Convention",
     "EQ–FI 0.68 is the 10-year monthly correlation. PE–EQ 0.80 follows the J.P. Morgan "
     "de-smoothed convention. The hedge fund row (EQ 0.75, PE 0.65, FI 0.35, CMD 0.35) is "
     "LTCMA-style convention reflecting the equity beta embedded in a diversified hedge "
     "fund composite — <b>not</b> a published J.P. Morgan matrix. Editable in Custom mode. "
     "They stay at the six-bucket level: swapping a sub-category does not change the "
     "correlation row, because J.P. Morgan does not publish a line-level matrix."),
    ("Method", "Regime-based vol",
     "Returns are entered as geometric per annum and simulated as correlated lognormal "
     "monthly draws, with the portfolio rebalanced to target weights every month. Vol "
     "anchors come from 120 months of returns split into calm / normal / stress regimes by "
     "the bottom quartile, middle 50% and top quartile of 6-month rolling MSCI World "
     "volatility, weighted 25 / 50 / 25."),
]


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def base_alloc() -> pd.DataFrame:
    return pd.DataFrame({"Asset type": SHORT, "LTCMA line": BASE_SUB,
                         "Weight %": DEF_W, "Return %": DEF_R, "Vol %": DEF_V})


def custom_alloc_seed() -> pd.DataFrame:
    """Custom starts as one sleeve per general asset type. Rows are dynamic:
    a bucket can hold as many sleeves as you like, each with its own LTCMA
    line and weight."""
    return pd.DataFrame({"Asset type": SHORT, "LTCMA line": CUSTOM_SUB,
                         "Weight %": DEF_W, "Return %": DEF_R, "Vol %": DEF_V})


def base_corr() -> pd.DataFrame:
    return pd.DataFrame(DEF_CORR, index=TAGS, columns=TAGS)


def base_within() -> pd.DataFrame:
    return pd.DataFrame({"Within-type correlation": WITHIN_BUCKET}, index=SHORT)


def sleeve_inputs(alloc: pd.DataFrame, bucket_corr, within):
    """Turn the sleeve table into the arrays simulate() needs.

    Each row is one sleeve. Its general asset type decides which correlation
    row it uses, so two equity sleeves correlate with each other at the
    within-type figure and with everything else at the published bucket
    figure.
    """
    bucket_of = [SHORT.index(b) if b in SHORT else 0
                 for b in alloc["Asset type"]]
    corr = expand_correlation(bucket_of, bucket_corr, within)
    return (tuple(alloc["Weight %"]), tuple(alloc["Return %"]),
            tuple(alloc["Vol %"]), corr)


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


if "custom_alloc" not in st.session_state:
    st.session_state.editor_nonce = 0
    # Custom starts as a copy of the base case, so switching modes is a
    # starting point rather than a blank slate.
    st.session_state.custom_alloc = custom_alloc_seed()
    st.session_state.custom_corr = base_corr()
    st.session_state.custom_within = base_within()
    st.session_state.mode = MODE_BASE

NONCE = st.session_state.get("editor_nonce", 0)


@st.cache_data(show_spinner=False, max_entries=32)
def run_sim(w, r, v, corr_flat, paths, years, seed):
    """Cached on the flattened sleeve-level inputs, so N is whatever the
    allocation table currently holds — not fixed at six."""
    corr = np.array(corr_flat, dtype=float).reshape(len(w), len(w))
    return simulate(list(w), list(r), list(v), corr, paths, years, seed)


# --------------------------------------------------------------------------
# Sidebar — inputs
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="eyebrow">Keepp Capital</div>', unsafe_allow_html=True)
    st.markdown("### Assumptions")

    mode = st.radio(
        "Assumption set", [MODE_BASE, MODE_CUSTOM],
        key="mode", label_visibility="collapsed",
        captions=["Locked to the deck", "Edit everything"],
    )
    is_base = mode == MODE_BASE

    if is_base:
        corr_df = base_corr()
        within_df = base_within()
    else:
        corr_df = st.session_state.custom_corr
        within_df = st.session_state.custom_within

    st.divider()
    paths = st.select_slider(
        "Simulation paths", options=[1_000, 2_500, 5_000, 10_000, 25_000, 50_000],
        value=10_000)
    years = st.slider("Horizon (years)", 1, 30, 10)
    seed = st.number_input("Random seed", value=20260722, step=1,
                           help="Fix the seed so everyone running the same "
                                "inputs sees the same numbers.")

    with st.expander("Correlations"):
        st.caption("Between asset types")
        if is_base:
            st.dataframe(corr_df, width='stretch',
                         column_config={t: st.column_config.NumberColumn(format="%.2f")
                                        for t in TAGS})
            st.caption("Between two sleeves of the same type")
            st.dataframe(within_df, width='stretch',
                         column_config={"Within-type correlation":
                                        st.column_config.NumberColumn(format="%.2f")})
            st.caption("Switch to **Custom** to edit either.")
        else:
            corr_df = st.data_editor(
                st.session_state.custom_corr, key=f"corr_editor_{NONCE}",
                width='stretch',
                column_config={t: st.column_config.NumberColumn(
                    min_value=-1.0, max_value=1.0, step=0.01, format="%.2f")
                    for t in TAGS})
            st.session_state.custom_corr = corr_df
            st.caption("Between two sleeves of the **same** type — only used "
                       "once a type holds more than one sleeve. J.P. Morgan "
                       "publishes no line-level matrix, so this one number per "
                       "type stands in for it.")
            within_df = st.data_editor(
                st.session_state.custom_within, key=f"within_editor_{NONCE}",
                width='stretch',
                column_config={"Within-type correlation":
                               st.column_config.NumberColumn(
                                   min_value=-1.0, max_value=1.0, step=0.01,
                                   format="%.2f")})
            st.session_state.custom_within = within_df
            st.caption("Type-matrix edits are mirrored across the diagonal. An "
                       "inconsistent matrix is repaired to the nearest valid "
                       "one, with a warning.")

    if not is_base:
        st.divider()
        if st.button("Reset Custom to JPM defaults", width='stretch'):
            st.session_state.custom_alloc = custom_alloc_seed()
            st.session_state.custom_corr = base_corr()
            st.session_state.custom_within = base_within()
            bump_editors()
            st.rerun()


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
chip = ('<span class="modechip base">Base · JPM LTCMA</span>' if is_base
        else '<span class="modechip custom">Custom assumptions</span>')
st.markdown('<div class="eyebrow">Keepp Capital · Target Allocation</div>',
            unsafe_allow_html=True)
st.markdown(f"# Monte Carlo Allocation Simulator{chip}", unsafe_allow_html=True)
st.markdown(
    '<p class="sub">Correlated lognormal paths, monthly steps, rebalanced to target '
    'weights each month. The base case is the J.P. Morgan 2026 Long-Term Capital Market '
    'Assumptions used in the July 2027 deck; Custom unlocks every input.</p>',
    unsafe_allow_html=True)
st.write("")

# --------------------------------------------------------------------------
# Allocation — one row per sleeve. Lives in the main column rather than the
# sidebar: with five columns and dynamic rows it needs the width.
# --------------------------------------------------------------------------
st.markdown("## Allocation")

if is_base:
    alloc = base_alloc()
    st.dataframe(
        alloc, hide_index=True, width='stretch',
        column_config={
            "Asset type": st.column_config.TextColumn(width="medium"),
            "LTCMA line": st.column_config.TextColumn(width="large"),
            "Weight %": st.column_config.NumberColumn(format="%.2f"),
            "Return %": st.column_config.NumberColumn(format="%.2f"),
            "Vol %": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption("Locked to the J.P. Morgan 2026 LTCMA figures behind the July "
               "2027 deck — one sleeve per asset type. Switch to **Custom** in "
               "the sidebar to split a type across several LTCMA lines.")
else:
    st.caption("Each asset type has its own table and its own dropdown, so the "
               "list only ever offers that type's LTCMA lines. **+** adds a "
               "sleeve to that type; select a row and press delete to remove "
               "it. Type in a cell to filter. Picking a line fills Return and "
               "Vol; typing over either number flips the row to Manual entry. "
               "★ marks one of the 14 expansion-proposal lines.")

    store = st.session_state.custom_alloc
    parts, dirty_any = [], False
    SLEEVE_COLS = ["LTCMA line", "Weight %", "Return %", "Vol %"]

    # Two columns of three groups: six full-width tables stacked would push the
    # results a screen and a half down the page.
    col_l, col_r = st.columns(2, gap="medium")
    for gi, (short, key) in enumerate(zip(SHORT, CHOICE_KEYS)):
        host = col_l if gi % 2 == 0 else col_r
        with host:
            g_prev = (store[store["Asset type"] == short]
                      .reset_index(drop=True)[SLEEVE_COLS])
            g_sum = pd.to_numeric(g_prev["Weight %"], errors="coerce").fillna(0).sum()
            host.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:baseline;margin:.5rem 0 .15rem">'
                f'<span style="font-weight:600;font-size:.88rem">{short}</span>'
                f'<span style="font-family:{MONO};font-size:.75rem;color:{INK_3}">'
                f'{g_sum:.2f}%{"  ·  " + str(len(g_prev)) + " sleeves" if len(g_prev) != 1 else ""}'
                f'</span></div>', unsafe_allow_html=True)
            edited = st.data_editor(
                g_prev,
                key=f"grp_{key}_{NONCE}",
                hide_index=True,
                width='stretch',
                num_rows="dynamic",
                column_config={
                    # Options are scoped to THIS asset type — that is the whole
                    # point of one editor per group. Streamlit's SelectboxColumn
                    # carries a single option list per column, so a shared table
                    # could not do this.
                    "LTCMA line": st.column_config.SelectboxColumn(
                        options=jpm.CHOICES[key], width="medium", required=True,
                        help=f"J.P. Morgan 2026 LTCMA lines available for "
                             f"{short}. Type to filter. ★ marks an "
                             f"expansion-proposal line."),
                    # step MUST stay 0.01: a coarser step makes the editor snap
                    # the *displayed* value to it, so 12.88 renders as 12.80 and
                    # a single click would commit the wrong number.
                    "Weight %": st.column_config.NumberColumn(
                        min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
                    "Return %": st.column_config.NumberColumn(
                        min_value=-20.0, max_value=60.0, step=0.01, format="%.2f"),
                    "Vol %": st.column_config.NumberColumn(
                        min_value=0.0, max_value=100.0, step=0.01, format="%.2f"),
                },
            )
            synced, d = jpm.sync_group(g_prev, edited, key)
            dirty_any = dirty_any or d
            synced.insert(0, "Asset type", short)
            parts.append(synced)

    alloc = (pd.concat(parts, ignore_index=True) if parts
             else custom_alloc_seed().iloc[0:0])
    st.session_state.custom_alloc = alloc
    if dirty_any:
        bump_editors()
        st.rerun()

alloc = alloc.reset_index(drop=True)
for _c in ("Weight %", "Return %", "Vol %"):
    alloc[_c] = pd.to_numeric(alloc[_c], errors="coerce").fillna(0.0)

wsum = float(alloc["Weight %"].sum())
off = abs(wsum - 100.0) > 0.005

tc1, tc2 = st.columns([1, 3])
tc1.markdown(
    f'<div style="font-family:{MONO};font-size:.85rem;padding-top:.4rem">'
    f'TOTAL &nbsp;<b style="color:{BRICK if off else ACCENT}">{wsum:.2f}%</b>'
    f'&nbsp;&nbsp;<span style="color:{INK_3};font-size:.78rem">'
    f'{len(alloc)} sleeves</span></div>', unsafe_allow_html=True)
if not is_base and off:
    if tc1.button("Normalize to 100%", width='stretch'):
        if wsum > 0:
            st.session_state.custom_alloc["Weight %"] = (
                pd.to_numeric(st.session_state.custom_alloc["Weight %"],
                              errors="coerce").fillna(0) / wsum * 100).round(4)
            bump_editors()
            st.rerun()
        else:
            st.warning("Enter at least one positive weight.")

if off:
    st.error(f"**Weights total {wsum:.2f}%.** They must sum to 100% before the "
             f"simulation can run. Adjust a weight in the sidebar, or use Normalize "
             f"to scale them all proportionally.")
    st.stop()

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
# Symmetrise the between-type matrix, then expand it to one row per sleeve.
corr_sym = corr_df.to_numpy(dtype=float)
corr_sym = (corr_sym + corr_sym.T) / 2.0
np.fill_diagonal(corr_sym, 1.0)
within = np.clip(within_df.to_numpy(dtype=float).ravel(), -1.0, 1.0)

w_t, r_t, v_t, corr_full = sleeve_inputs(alloc, corr_sym, within)

with st.spinner("Running simulation…"):
    res, repaired = run_sim(
        w_t, r_t, v_t, tuple(corr_full.flatten()),
        int(paths), int(years), int(seed))

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

axis_y = alt.Axis(title="Portfolio value (multiple of start)", format=".2f",
                  labelFont="monospace", labelColor=INK_3, titleColor=INK_3,
                  gridColor=GRID, domain=False, tickCount=6, labelFontSize=11)
axis_x = alt.Axis(title="Years from today", titleColor=INK_3,
                  labelFont="monospace", labelColor=INK_3,
                  grid=False, tickCount=min(res.years, 10), labelFontSize=11)

base_c = alt.Chart(fan)
outer = base_c.mark_area(color=BAND_OUTER).encode(
    x=alt.X("Year:Q", axis=axis_x, scale=alt.Scale(nice=False)),
    y=alt.Y("p05:Q", axis=axis_y, scale=yscale), y2="p95:Q")
inner = base_c.mark_area(color=BAND_INNER).encode(
    x="Year:Q", y=alt.Y("p25:Q", scale=yscale), y2="p75:Q")
breakeven = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(
    color=INK_3, strokeDash=[3, 3], opacity=.7, size=1).encode(y=alt.Y("y:Q", scale=yscale))
median = base_c.mark_line(color=ACCENT, strokeWidth=2).encode(
    x="Year:Q", y=alt.Y("p50:Q", scale=yscale))

hover = alt.selection_point(fields=["Year"], nearest=True, on="pointerover",
                            empty=False, clear="pointerout")
rule = base_c.mark_rule(color=INK_3, strokeDash=[3, 3]).encode(
    x="Year:Q",
    opacity=alt.condition(hover, alt.value(0.8), alt.value(0)),
    tooltip=[alt.Tooltip("Year:Q", format=".2f"),
             alt.Tooltip("p95_lbl:N", title="95th"),
             alt.Tooltip("p75_lbl:N", title="75th"),
             alt.Tooltip("p50_lbl:N", title="Median"),
             alt.Tooltip("p25_lbl:N", title="25th"),
             alt.Tooltip("p05_lbl:N", title="5th")],
).add_params(hover)
dot = base_c.mark_point(color=ACCENT, filled=True, size=60).encode(
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
d1.download_button("Download CSV", csv,
                   f"keepp_monte_carlo_{'base' if is_base else 'custom'}.csv",
                   "text/csv", width='stretch')
d2.markdown(
    f'<div class="foot" style="padding-top:.5rem">'
    f'<b>{"Base · JPM 2026 LTCMA" if is_base else "Custom assumptions"}</b> · '
    f'{res.paths:,} paths · {len(alloc)} sleeves · {res.years}-year horizon · '
    f'seed {seed} · monthly rebalancing. P(loss) is the probability the portfolio sits below its starting '
    f'value at the end of the horizon; the × columns are wealth multiples on a '
    f'1.00× start.</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------
st.markdown("## Where the base assumptions come from")
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
