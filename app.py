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
import optimizer as opt
from engine import simulate

# The line the cash floor applies to. Named once so the optimizer panel and the
# risk-free default cannot drift apart from each other.
CASH_LINE = "U.S. Cash"

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
# Base case — J.P. Morgan 2026 LTCMA. Returns, volatilities and the sleeve list
# are hardcoded and cannot be edited in Base; the WEIGHTS can, so a manager can
# re-weight the deck's own assumptions without leaving them. The shipped weights
# below are the deck's, and "Reset weights to the deck" restores them.
#
# Diversified Hedge Funds carries a 0% base weight on purpose: the July 2027
# deck's target allocation has no hedge fund sleeve, so a zero weight keeps the
# shipped Base case numerically identical to the signed-off Excel workbook.
# --------------------------------------------------------------------------
# The three J.P. Morgan categories. Every sleeve belongs to one of them, and
# each gets its own table whose dropdown offers exactly that category's lines.
CATEGORY_KEYS = jpm.CATEGORY_KEYS                   # ["FI", "EQ", "ALT"]
CATEGORIES = [jpm.CATEGORY_NAME[k] for k in CATEGORY_KEYS]
KEY_OF = dict(zip(CATEGORIES, CATEGORY_KEYS))

# Base case — the July 2027 deck's target allocation, expressed as sleeves.
# Returns are COMPOUND (geometric): engine.py converts with ln(1+g)/12 and
# derives the arithmetic return as g + sigma^2/2, so JPM's arithmetic column
# would double-count the variance drag. The original workbook did that for
# Private Equity (11.90 not 10.20) and Commodities (6.15 not 4.60).
#
# Every sleeve sits on an LTCMA line, and that line is what selects the row of
# J.P. Morgan's correlation matrix the sleeve uses. Three of the six carry
# Keepp's own return and volatility rather than J.P. Morgan's — a global IG
# blend, MSCI World realised history, and the live 3-month T-bill — so they are
# expressed as EDITED versions of the closest line. The `basis` field is the
# one-line provenance named in the caption under the Base table; an empty
# string means the sleeve is J.P. Morgan's numbers untouched.
#
# Filing them this way rather than as anonymous "manual" rows is what removed
# the old hand-picked correlation proxy column. The correlation rows are
# unchanged: the three proxies used to be exactly these three lines, so Base's
# return, volatility and percentiles are identical to the previous build.
#
# Hedge funds sit at 0%: the deck has no hedge fund sleeve, and a zero weight
# keeps Base's return and volatility identical to the signed-off workbook.
BASE_SLEEVES = [
    # category, LTCMA line,                 weight, return,   vol,   basis
    ("FI",  "U.S. Inv Grade Corporate Bonds", 45.0,   5.00,   7.02,
     "Keepp · global IG blend"),
    ("FI",  "U.S. Cash",                      17.0,   3.80,   0.67,
     "Keepp · 3M T-bill"),
    ("EQ",  "AC World Equity",                20.0,  12.88,  14.47,
     "Keepp · MSCI World 10y"),
    ("ALT", "Private Equity",                 10.0,  10.20,  19.78,  ""),
    ("ALT", "Diversified Hedge Funds",         0.0,   5.30,   5.75,  ""),
    ("ALT", "Commodities",                     8.0,   4.60,  18.32,  ""),
]

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
    ("Sub-categories", "3 categories · 58 lines",
     "Sleeves are grouped into J.P. Morgan's own three categories — Fixed Income "
     "(21 lines), Equities (18) and Alternatives (19). Each category has its own "
     "table and dropdown, so the list only ever offers that category's lines. "
     "Picking one fills Return and Vol from the <b>compound</b> column. Editing "
     "either number by hand keeps the sleeve on its line — the line is what "
     "supplies the correlation row, so an override changes the figures without "
     "detaching the sleeve from the matrix."),
    ("Correlations", "JPM 2026 LTCMA, USD",
     "<b>The full published 59 × 59 matrix</b>, shipped unmodified as "
     "<code>jpm_correlations.csv</code> so it can be diffed against the report. "
     "Nothing is assumed: the matrix used in any run is a sub-matrix of it, "
     "selected by each sleeve's correlation row. Earlier builds carried a "
     "fabricated 6 × 6 plus a guessed within-type number; both are gone, as is "
     "the hand-picked correlation proxy that replaced them. "
     "<b>One numerical note:</b> as published the matrix is not positive "
     "semi-definite (smallest eigenvalue −0.0161), which is normal for figures "
     "rounded to two decimals and is compounded by U.S. Core and U.S. "
     "Value-Added Real Estate sharing one underlying series. It is repaired once "
     "at load with Higham's nearest-correlation algorithm; the largest single "
     "cell moves 0.0066 and no pair moves as much as 0.01."),
    ("Keepp-sourced sleeves", "3 of 6 in Base",
     "Three base sleeves carry Keepp's own return and volatility rather than "
     "J.P. Morgan's: a global IG blend (5.00% · 7.02%), MSCI World realised "
     "10-year history (12.88% · 14.47%) and the live 3-month T-bill (3.80% · "
     "0.67%). Each is filed under the line it most closely is — U.S. Inv Grade "
     "Corporate Bonds, AC World Equity and U.S. Cash respectively — which is "
     "what supplies its correlation row. <b>Nothing is chosen by hand:</b> "
     "earlier builds asked the user to pick a separate correlation proxy for "
     "every sleeve. Filing a sleeve under a line answers that question once, "
     "at the point where the user is already naming the asset."),
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
COLS = ["LTCMA line", "Weight %", "Return %", "Vol %"]


# The deck's own weights, kept separately so "reset to the deck" always has
# something to reset to no matter what the user has typed.
BASE_WEIGHTS = [w for _, _, w, _, _, _ in BASE_SLEEVES]


def base_alloc(weights=None) -> pd.DataFrame:
    """The deck's six sleeves. Weights are the caller's; everything else is the
    deck's and is not editable in Base — that is the whole point of Base."""
    ws = BASE_WEIGHTS if weights is None else list(weights)
    return pd.DataFrame(
        [{"Category": jpm.CATEGORY_NAME[k], "LTCMA line": jpm.LABEL[line],
          "Weight %": float(w), "Return %": r, "Vol %": v}
         for (k, line, _, r, v, _), w in zip(BASE_SLEEVES, ws)])


def custom_alloc_seed() -> pd.DataFrame:
    """Custom starts from Base — same six sleeves, same lines, same numbers."""
    return pd.DataFrame(
        [{"Category": jpm.CATEGORY_NAME[k], "LTCMA line": jpm.LABEL[line],
          "Weight %": w, "Return %": r, "Vol %": v}
         for k, line, w, r, v, _ in BASE_SLEEVES])


# The three sleeves whose Return and Vol are Keepp's rather than J.P. Morgan's.
# Named in a caption under the Base table instead of a column: the provenance
# still needs saying, but not once per row on every render.
KEEPP_SOURCED = [(line, basis) for _, line, _, _, _, basis in BASE_SLEEVES if basis]


def sleeve_inputs(alloc: pd.DataFrame):
    """Turn the sleeve table into the arrays simulate() needs.

    The correlation matrix is a sub-matrix of J.P. Morgan's published 59x59,
    selected by each sleeve's LTCMA line. Nothing is assumed and nothing is
    borrowed — the numbers come straight from the CSV (repaired once at import).
    """
    corr = corr_of(alloc)
    return (tuple(alloc["Weight %"]), tuple(alloc["Return %"]),
            tuple(alloc["Vol %"]), corr)


def corr_of(alloc: pd.DataFrame):
    """The published correlation sub-matrix for this allocation's lines.

    Shared by the simulation and the optimizer, so the two cannot end up
    disagreeing about how the sleeves move together.
    """
    return jpm.corr_for(list(alloc["LTCMA line"]))


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
    st.session_state.base_weights = list(BASE_WEIGHTS)
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
        captions=["Deck assumptions, your weights", "Edit everything"],
    )
    is_base = mode == MODE_BASE

    st.divider()
    paths = st.select_slider(
        "Simulation paths", options=[1_000, 2_500, 5_000, 10_000, 25_000, 50_000],
        value=10_000)
    years = st.slider("Horizon (years)", 1, 30, 10)
    seed = st.number_input("Random seed", value=20260722, step=1,
                           help="Fix the seed so everyone running the same "
                                "inputs sees the same numbers.")

    st.divider()
    if is_base:
        if st.button("Reset weights to the deck", width='stretch'):
            st.session_state.base_weights = list(BASE_WEIGHTS)
            bump_editors()
            st.rerun()
    else:
        if st.button("Reset Custom to JPM defaults", width='stretch'):
            st.session_state.custom_alloc = custom_alloc_seed()
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
    'weights each month. The base case holds the J.P. Morgan 2026 Long-Term Capital Market '
    'Assumptions used in the July 2027 deck and lets you re-weight them; Custom unlocks '
    'every input.</p>',
    unsafe_allow_html=True)
st.write("")

# --------------------------------------------------------------------------
# Allocation — one row per sleeve. Lives in the main column rather than the
# sidebar: with five columns and dynamic rows it needs the width.
# --------------------------------------------------------------------------
st.markdown("## Allocation")

if is_base:
    # Weights are editable; the sleeve list, returns and volatilities are not.
    # Re-weighting the deck's own assumptions is the question a PM actually
    # asks of a base case ("what if we took equities to 25?"), and answering it
    # here rather than in Custom keeps the answer on figures nobody has
    # touched. Every other column is disabled, so a Base run is always the
    # deck's assumptions — only the mix moves.
    edited_base = st.data_editor(
        base_alloc(st.session_state.base_weights),
        key=f"base_{NONCE}",
        hide_index=True,
        width='stretch',
        num_rows="fixed",
        disabled=["Category", "LTCMA line", "Return %", "Vol %"],
        column_config={
            "Category": st.column_config.TextColumn(width="small"),
            "LTCMA line": st.column_config.TextColumn(
                width="medium",
                help="The J.P. Morgan line this sleeve is filed under. It "
                     "supplies the sleeve's row of the published correlation "
                     "matrix. Fixed in Base — switch to Custom to change it."),
            # step MUST stay 0.01, as in the Custom tables: a coarser step makes
            # the editor snap the *displayed* value to it.
            "Weight %": st.column_config.NumberColumn(
                min_value=0.0, max_value=100.0, step=0.01, format="%.2f",
                help="The only editable field in Base. Returns, volatilities "
                     "and the sleeve list stay as the deck has them."),
            "Return %": st.column_config.NumberColumn(format="%.2f"),
            "Vol %": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    # Read the weights straight back out, so the stats below reflect the edit on
    # this run rather than one interaction late. Session state only has to carry
    # them across the next rerun.
    new_w = [float(x) if x == x else 0.0
             for x in pd.to_numeric(edited_base["Weight %"], errors="coerce")]
    st.session_state.base_weights = new_w
    alloc = base_alloc(new_w)

    st.caption(
        "Returns and volatilities are locked to the J.P. Morgan 2026 LTCMA "
        "figures behind the July 2027 deck — **only the weights are editable "
        "here.** Three sleeves carry Keepp's own Return and Vol rather than "
        "J.P. Morgan's — "
        + ", ".join(f"**{ln}** ({b.split('·', 1)[1].strip()})"
                    for ln, b in KEEPP_SOURCED)
        + " — and are filed under the line that supplies their correlation "
          "row. Switch to **Custom** in the sidebar to add sleeves or change "
          "any other input.")
else:
    st.caption("One table per J.P. Morgan category — each dropdown offers only "
               "that category's LTCMA lines. **+** adds a sleeve; select a row "
               "and press delete to remove it. Type in a cell to filter. "
               "Picking a line fills Return and Vol with J.P. Morgan's figures; "
               "type over either one and the sleeve keeps your number and stays "
               "on its line. ★ marks one of the 14 expansion-proposal lines.")

    store = st.session_state.custom_alloc
    parts, dirty_any = [], False

    for key in CATEGORY_KEYS:
        cat = jpm.CATEGORY_NAME[key]
        g_prev = (store[store["Category"] == cat]
                  .reset_index(drop=True)[COLS])
        g_sum = pd.to_numeric(g_prev["Weight %"], errors="coerce").fillna(0).sum()
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:baseline;margin:1rem 0 .15rem">'
            f'<span style="font-weight:600;font-size:.92rem">{cat}</span>'
            f'<span style="font-family:{MONO};font-size:.75rem;color:{INK_3}">'
            f'{g_sum:.2f}%  ·  {len(g_prev)} sleeve{"s" if len(g_prev) != 1 else ""}'
            f'  ·  {len(jpm.CHOICES[key])} LTCMA lines available</span></div>',
            unsafe_allow_html=True)
        edited = st.data_editor(
            g_prev,
            key=f"grp_{key}_{NONCE}",
            hide_index=True,
            width='stretch',
            num_rows="dynamic",
            column_config={
                # Options are scoped to THIS category — the reason each category
                # gets its own editor. Streamlit's SelectboxColumn carries a
                # single option list per column, so one shared table could not.
                "LTCMA line": st.column_config.SelectboxColumn(
                    options=jpm.CHOICES[key], width="medium", required=True,
                    help=f"J.P. Morgan 2026 LTCMA lines in {cat}. Type to "
                         f"filter. ★ marks an expansion-proposal line. The "
                         f"line also supplies this sleeve's row of the "
                         f"published correlation matrix."),
                # step MUST stay 0.01: a coarser step makes the editor snap the
                # *displayed* value to it, so 12.88 renders as 12.80 and a
                # single click would commit the wrong number.
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
        synced.insert(0, "Category", cat)
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
if off:
    if tc1.button("Normalize to 100%", width='stretch'):
        if wsum > 0:
            if is_base:
                st.session_state.base_weights = [
                    round(x / wsum * 100, 4) for x in st.session_state.base_weights]
            else:
                st.session_state.custom_alloc["Weight %"] = (
                    pd.to_numeric(st.session_state.custom_alloc["Weight %"],
                                  errors="coerce").fillna(0) / wsum * 100).round(4)
            bump_editors()
            st.rerun()
        else:
            st.warning("Enter at least one positive weight.")

if off:
    st.error(f"**Weights total {wsum:.2f}%.** They must sum to 100% before the "
             f"simulation can run. Adjust a weight in the table above, or use "
             f"Normalize to scale them all proportionally.")
    st.stop()

# --------------------------------------------------------------------------
# Optimizer — the reverse question
#
# The rest of the app answers "what does this allocation produce?". This
# section answers "what is the best achievable return-and-volatility pair, and
# which weights get there?" — a mean-variance optimization over the SAME lines,
# returns, volatilities and published correlation sub-matrix the simulation
# uses, so the two can never disagree about the current portfolio.
#
# Shown in BOTH modes. Base allows weight edits and nothing else, and the
# optimizer's decision variables are exactly weights, so it fits Base without
# violating anything: the returns and volatilities it optimizes against are the
# locked J.P. Morgan figures. In Custom it optimizes against whatever the PM
# has typed. All numeric work lives in optimizer.py; this block is UI only.
# --------------------------------------------------------------------------
st.markdown("## Optimizer")
st.caption(
    "The reverse question: given the return and volatility assumptions above, "
    "what is the best achievable return-and-volatility pair, and which weights "
    "get there? Decision variables are the individual LTCMA line weights — "
    "long-only, summing to 100%, subject to a cash floor and per-category bands.")

OBJ_SHARPE = "Maximum Sharpe ratio"
OBJ_MINVOL = "Minimum volatility for a target return"
OBJ_MAXRET = "Maximum return for a volatility budget"
OBJ_GROWTH = "Maximum long-run growth"

opt_lines = [jpm.resolve(x) for x in alloc["LTCMA line"]]
opt_cats = list(alloc["Category"])
n_lines = len(opt_lines)
cash_pos = [i for i, nm in enumerate(opt_lines) if nm == CASH_LINE]

# arithmetic mu, per optimizer.py's single conversion point
opt_mu, opt_vols = opt.inputs_from_percent(alloc["Return %"], alloc["Vol %"])
opt_cov = opt.covariance(opt_vols, jpm.corr_for(alloc["LTCMA line"]))
cur_w = np.asarray(alloc["Weight %"], dtype=float) / 100.0

groups = {}
for i, cat in enumerate(opt_cats):
    groups.setdefault(cat, []).append(i)
groups = {k: np.array(v) for k, v in groups.items()}
cur_cat_pct = {k: float(cur_w[i].sum()) * 100.0 for k, i in groups.items()}

if n_lines < 2:
    st.info("Add a second sleeve to optimize — with one line there is nothing "
            "to allocate between.")
else:
    with st.expander("Objective and constraints", expanded=False):
        oc1, oc2 = st.columns([1.35, 1])
        with oc1:
            objective = st.radio(
                "Objective", [OBJ_SHARPE, OBJ_GROWTH, OBJ_MINVOL, OBJ_MAXRET],
                key="opt_objective",
                help="Sharpe maximises return per unit of risk above the "
                     "risk-free rate. Long-run growth maximises the MEDIAN "
                     "wealth this app's fan chart plots, which is a different "
                     "portfolio — it charges for the variance drag Sharpe "
                     "ignores.")
            cur_r_pct, cur_v_pct = [x * 100 for x in
                                    opt.portfolio(cur_w, opt_mu, opt_cov)]
            if objective == OBJ_MINVOL:
                target_pct = st.number_input(
                    "Target return %", value=round(cur_r_pct + 0.5, 2),
                    min_value=-20.0, max_value=60.0, step=0.05, format="%.2f")
            elif objective == OBJ_MAXRET:
                budget_pct = st.number_input(
                    "Volatility budget %", value=round(cur_v_pct, 2),
                    min_value=0.01, max_value=100.0, step=0.05, format="%.2f")
        with oc2:
            rf_default = (float(alloc["Return %"].iloc[cash_pos[0]])
                          if cash_pos else 3.80)
            rf_pct = st.number_input(
                "Risk-free rate %", value=round(rf_default, 2),
                min_value=0.0, max_value=20.0, step=0.05, format="%.2f",
                help="Defaults to this allocation's own U.S. Cash return, so "
                     "the tangency point stays consistent with your inputs.")
            floor_pct = st.number_input(
                "Cash floor %", value=5.00, min_value=0.0, max_value=100.0,
                step=0.5, format="%.2f",
                disabled=not cash_pos,
                help=("Minimum weight on the U.S. Cash line."
                      if cash_pos else
                      "No U.S. Cash line in this allocation, so there is "
                      "nothing to floor. Add one to enable this."))
            cap_pct = st.number_input(
                "Max weight per line %", value=100.00, min_value=0.01,
                max_value=100.0, step=1.0, format="%.2f",
                help="A cap on any single line. Leave at 100% for no cap. "
                     "Lowering it is the bluntest way to stop the optimizer "
                     "concentrating everything in one asset.")

        st.markdown(
            f'<div style="font-weight:600;font-size:.85rem;margin:.7rem 0 .1rem">'
            f'Category bands</div>'
            f'<div style="color:{INK_3};font-size:.78rem;margin-bottom:.35rem">'
            f'Minimum and maximum total weight per J.P. Morgan category. '
            f'Defaults are the current totals ±10pp.</div>',
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
                "Category": st.column_config.TextColumn(width="medium"),
                "Min %": st.column_config.NumberColumn(
                    min_value=0.0, max_value=100.0, step=0.5, format="%.2f"),
                "Max %": st.column_config.NumberColumn(
                    min_value=0.0, max_value=100.0, step=0.5, format="%.2f"),
            })
        show_stability = st.checkbox(
            "Also run a stability check (slower, ~10s)", value=False,
            help="Re-solves the optimum 40 times with every expected return "
                 "nudged by random noise, and reports how far each line's "
                 "optimal weight moves. The single most useful sanity check "
                 "on any mean-variance answer.")

    run_opt = st.button("Run optimization", type="primary")

    # ----------------------------------------------------------------------
    # Solve. Never on a keystroke — the frontier is 60 quadratic programs and
    # the PM is usually mid-typing. Only on an explicit click.
    # ----------------------------------------------------------------------
    if run_opt:
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
        try:
            with st.spinner("Solving…"):
                w0 = opt.feasible_start(limits)          # raises Infeasible
                frontier = opt.efficient_frontier(opt_mu, opt_cov, limits, w0, 60)
                rf = rf_pct / 100.0
                if objective == OBJ_SHARPE:
                    w_opt, _, _, _ = opt.max_sharpe(opt_mu, opt_cov, limits, w0, rf)
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
            }
        except opt.Infeasible as e:
            st.session_state.pop("opt_result", None)
            st.error(f"**These constraints cannot all be satisfied.** {e}")
        except Exception as e:                            # solver blew up
            st.session_state.pop("opt_result", None)
            st.error(f"**The optimizer could not solve this.** {type(e).__name__}: {e}")

    # ----------------------------------------------------------------------
    # Results
    # ----------------------------------------------------------------------
    R = st.session_state.get("opt_result")
    if R and R["lines"] == opt_lines:
        w_opt = R["w"]
        rf = R["rf"]
        if R.get("note"):
            st.warning(f"The solver did not fully converge — showing the best "
                       f"feasible point it reached, not a proven optimum. "
                       f"({R['note']})")

        r_opt, v_opt = opt.portfolio(w_opt, opt_mu, opt_cov)
        r_cur, v_cur = opt.portfolio(cur_w, opt_mu, opt_cov)
        sh_opt = opt.sharpe(w_opt, opt_mu, opt_cov, rf)
        sh_cur = opt.sharpe(cur_w, opt_mu, opt_cov, rf)

        c1, c2 = st.columns(2)
        c1.markdown(
            f'<div class="statcard"><div class="statk">Optimal expected return</div>'
            f'<div class="statv">{r_opt*100:.2f}%</div>'
            f'<div class="statn">Arithmetic, per annum &nbsp;·&nbsp; '
            f'{r_opt*100 - r_cur*100:+.2f}pp vs today</div></div>',
            unsafe_allow_html=True)
        c2.markdown(
            f'<div class="statcard"><div class="statk">Optimal volatility</div>'
            f'<div class="statv">{v_opt*100:.2f}%</div>'
            f'<div class="statn">Sharpe {sh_opt:.2f} vs {sh_cur:.2f} today</div>'
            f'</div>', unsafe_allow_html=True)
        st.write("")

        # ---- frontier chart ------------------------------------------------
        f_r, f_v, _ = R["frontier"]
        fdf = pd.DataFrame({"vol": f_v * 100, "ret": f_r * 100})
        pts = pd.DataFrame([
            {"vol": v_cur * 100, "ret": r_cur * 100, "what": "Current",
             "sharpe": sh_cur},
            {"vol": v_opt * 100, "ret": r_opt * 100, "what": "Optimal",
             "sharpe": sh_opt}])
        assets = pd.DataFrame({
            "vol": opt_vols * 100, "ret": opt_mu * 100,
            "what": [jpm.LABEL.get(x, x) for x in opt_lines]})

        def _axis(title):
            return alt.Axis(title=title, labelFont="monospace", labelColor=INK_3,
                            titleColor=INK_3, gridColor=GRID, labelFontSize=11,
                            format=".1f")

        x_enc = alt.X("vol:Q", axis=_axis("Volatility % p.a."),
                      scale=alt.Scale(zero=False, nice=True))
        y_enc = alt.Y("ret:Q", axis=_axis("Expected return % p.a."),
                      scale=alt.Scale(zero=False, nice=True))

        layers = [
            alt.Chart(assets).mark_point(size=42, color=INK_3, opacity=.5,
                                         filled=True).encode(
                x=x_enc, y=y_enc,
                tooltip=[alt.Tooltip("what:N", title="Line"),
                         alt.Tooltip("ret:Q", format=".2f", title="Return %"),
                         alt.Tooltip("vol:Q", format=".2f", title="Vol %")]),
            alt.Chart(fdf).mark_line(color=ACCENT, strokeWidth=2.2).encode(
                x=x_enc, y=y_enc),
        ]
        # capital market line, from (0, rf) through the tangency point
        if R["objective"] == OBJ_SHARPE and v_opt > 0:
            v_max = float(max(f_v.max(), v_cur, opt_vols.max())) * 100
            cml = pd.DataFrame({"vol": [0.0, v_max],
                                "ret": [rf * 100, rf * 100 + sh_opt * v_max]})
            layers.append(alt.Chart(cml).mark_line(
                color=ACCENT, strokeDash=[4, 4], strokeWidth=1,
                opacity=.65).encode(x=x_enc, y=y_enc))
        layers.append(
            alt.Chart(pts).mark_point(size=190, filled=True, opacity=1).encode(
                x=x_enc, y=y_enc,
                color=alt.Color("what:N", scale=alt.Scale(
                    domain=["Current", "Optimal"], range=[BRICK, ACCENT]),
                    legend=alt.Legend(title=None, orient="top-left")),
                shape=alt.Shape("what:N", scale=alt.Scale(
                    domain=["Current", "Optimal"], range=["circle", "diamond"]),
                    legend=None),
                tooltip=[alt.Tooltip("what:N", title=""),
                         alt.Tooltip("ret:Q", format=".2f", title="Return %"),
                         alt.Tooltip("vol:Q", format=".2f", title="Vol %"),
                         alt.Tooltip("sharpe:Q", format=".3f", title="Sharpe")]))

        chart = alt.layer(*layers).properties(height=330).configure_view(
            strokeWidth=0).configure_axis(labelFont="monospace", titleFontSize=11)
        st.altair_chart(chart, width='stretch')
        st.caption("Grey dots are the individual LTCMA lines. The line is the "
                   "efficient frontier under your constraints; the dashed line "
                   "is the capital market line through the optimum.")

        # ---- caveat, in the app and not only in the code -------------------
        st.markdown(
            f'<div style="border-left:3px solid {BRICK};background:#FBF3EF;'
            f'padding:11px 14px;border-radius:4px;font-size:.82rem;'
            f'line-height:1.6;margin:.5rem 0 1rem">'
            f'<b>Read the frontier as a diagnostic, not a target allocation.</b> '
            f'Mean-variance solutions are highly sensitive to the return '
            f'assumptions typed above: a 50bp change in one line\'s expected '
            f'return can move its optimal weight by tens of percentage points. '
            f'Correlations here are static — they do not rise in a crisis — so '
            f'the left tail of every portfolio on this chart, including the '
            f'optimal one, is likely flattered.</div>',
            unsafe_allow_html=True)

        # ---- weights table -------------------------------------------------
        p25, p75 = R.get("stability", (None, None))
        rows = []
        for i, nm in enumerate(opt_lines):
            row = {"Category": opt_cats[i],
                   "LTCMA line": jpm.LABEL.get(nm, nm),
                   "Current %": cur_w[i] * 100,
                   "Optimal %": w_opt[i] * 100,
                   "Change (pp)": (w_opt[i] - cur_w[i]) * 100}
            if p25 is not None:
                row["Stable range %"] = f"{p25[i]*100:.0f} – {p75[i]*100:.0f}"
            rows.append(row)
        wdf = pd.DataFrame(rows).reindex(
            pd.DataFrame(rows)["Change (pp)"].abs()
            .sort_values(ascending=False).index).reset_index(drop=True)

        sub = []
        for k, idx in groups.items():
            r_ = {"Category": k, "LTCMA line": "— category total —",
                  "Current %": cur_w[idx].sum() * 100,
                  "Optimal %": w_opt[idx].sum() * 100,
                  "Change (pp)": (w_opt[idx].sum() - cur_w[idx].sum()) * 100}
            if p25 is not None:
                r_["Stable range %"] = ""
            sub.append(r_)
        wdf = pd.concat([wdf, pd.DataFrame(sub)], ignore_index=True)

        st.dataframe(
            wdf, hide_index=True, width='stretch',
            column_config={
                "Category": st.column_config.TextColumn(width="small"),
                "LTCMA line": st.column_config.TextColumn(width="medium"),
                "Current %": st.column_config.NumberColumn(format="%.2f"),
                "Optimal %": st.column_config.NumberColumn(format="%.2f"),
                "Change (pp)": st.column_config.NumberColumn(format="%+.2f"),
                "Stable range %": st.column_config.TextColumn(
                    width="small",
                    help="Interquartile range of this line's optimal weight "
                         "across 40 re-solves with expected returns perturbed "
                         "by N(0, 0.5%). A wide range is not a recommendation."),
            })
        if p25 is not None:
            widest = int(np.argmax(p75 - p25))
            st.caption(
                f"**Stability:** across 40 re-solves with each expected return "
                f"nudged by random noise of 0.5%, the least stable line is "
                f"**{jpm.LABEL.get(opt_lines[widest], opt_lines[widest])}**, "
                f"whose optimal weight ranges "
                f"{p25[widest]*100:.0f}–{p75[widest]*100:.0f}%. Lines with a "
                f"wide range are an artefact of the return estimates, not a "
                f"recommendation.")

        # ---- apply + before/after ------------------------------------------
        ac1, ac2 = st.columns([1, 3])
        if ac1.button("Apply optimal weights", width='stretch'):
            new_pct = [round(float(x) * 100, 4) for x in w_opt]
            if is_base:
                st.session_state.base_weights = new_pct
            else:
                ca = st.session_state.custom_alloc.copy()
                ca["Weight %"] = new_pct
                st.session_state.custom_alloc = ca
            # MUST bump: st.data_editor keeps the user's edits as a delta and
            # re-applies them on top of whatever dataframe it is handed, so a
            # hand-edited weight cell would snap back over the optimal value.
            # This is the same failure that once made Normalize produce 94.71%.
            bump_editors()
            st.rerun()

        before = run_sim(tuple(alloc["Weight %"]), tuple(alloc["Return %"]),
                         tuple(alloc["Vol %"]), tuple(corr_of(alloc).flatten()),
                         int(paths), int(years), int(seed))[0]
        after = run_sim(tuple(float(x) * 100 for x in w_opt),
                        tuple(alloc["Return %"]), tuple(alloc["Vol %"]),
                        tuple(corr_of(alloc).flatten()),
                        int(paths), int(years), int(seed))[0]
        b_t, a_t = before.rows[-1], after.rows[-1]
        ac2.markdown(
            f'<div style="font-size:.82rem;line-height:1.75;padding-top:.35rem">'
            f'<b>If applied</b>, over {years} years: median wealth '
            f'<b>{b_t["Median x"]:.2f}× → {a_t["Median x"]:.2f}×</b>, '
            f'P(loss) <b>{b_t["P(loss)"]*100:.2f}% → {a_t["P(loss)"]*100:.2f}%</b>. '
            f'<span style="color:{INK_3}">Simulated, not analytic — the same '
            f'Monte Carlo as the rest of the page.</span></div>',
            unsafe_allow_html=True)
    elif R:
        st.info("The allocation changed since the last run. Press "
                "**Run optimization** again.")

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
w_t, r_t, v_t, corr_full = sleeve_inputs(alloc)

with st.spinner("Running simulation…"):
    res, repaired = run_sim(
        w_t, r_t, v_t, tuple(corr_full.flatten()),
        int(paths), int(years), int(seed))

if repaired:
    # Should be unreachable: the published matrix is repaired once at import,
    # and every principal sub-matrix of a positive semi-definite matrix is
    # itself positive semi-definite. Kept as a guard, not an expected path.
    st.warning("The correlation sub-matrix needed a numerical repair before "
               "simulating. This should not happen with the published J.P. "
               "Morgan matrix — worth reporting if you see it.")

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
