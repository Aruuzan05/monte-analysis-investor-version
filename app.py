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
import os
import time
from dataclasses import replace
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from keepp_mc import db as dbm
from keepp_mc import blackrock_cma as brc
from keepp_mc import jpm_ltcma as jpm
from keepp_mc import optimizer as opt
from keepp_mc import personal_lines as pl
from keepp_mc import cashflows as cfm
from keepp_mc import compare as cmpm
from keepp_mc import geography as geo
from keepp_mc import liquidity as liq
from keepp_mc import run_store as rs
from keepp_mc import universe as uni
from keepp_mc.engine import return_ladder, simulate

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
# Database + accounts. Saved runs go to the signed-in user's account
# (db.py), so the run history survives a refresh, a container restart and a
# change of device.
#
#   .streamlit/secrets.toml (or Community Cloud -> Settings -> Secrets):
#     [database]
#     url = "postgresql://user:pass@host/dbname"   # omit -> local SQLite file
#     [auth]
#     signup_code = "something-long"                # needed to create accounts
#
# With no signup_code, anyone may create the FIRST account; after that new
# accounts need the code, or `python manage_users.py add <name>`.
# --------------------------------------------------------------------------
SESSION_COOKIE = "keepp_session"
COOKIE_MAX_AGE = dbm.SESSION_TTL_DAYS * 86400


def _secret(*keys, default=""):
    """st.secrets lookup that treats "no secrets.toml at all" as unset."""
    try:
        node = st.secrets
        for k in keys:
            node = node[k]
        return node
    except Exception:
        return default


def database_url() -> str:
    url = _secret("database", "url") or os.environ.get("KEEPP_DATABASE_URL", "")
    if url:
        return str(url)
    data_dir = Path(os.environ.get("KEEPP_DATA_DIR", Path(__file__).parent / "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(data_dir / 'keepp.db').as_posix()}"


@st.cache_resource(show_spinner=False)
def get_db(url: str) -> dbm.RunDB:
    return dbm.RunDB(url)


try:
    DB = get_db(database_url())
    # cache_resource outlives a code update: after a redeploy (or a module
    # reload while developing) it can hand back a RunDB built by the OLD
    # db.py, which returns old-style RunRecords (e.g. without `source`).
    # The class object changes on every reload, so a mismatch means stale.
    if not isinstance(DB, dbm.RunDB):
        try:
            DB.engine.dispose()
        except Exception:
            pass
        get_db.clear()
        DB = get_db(database_url())
except Exception as e:                    # bad URL, database down, wrong password
    st.error("The run database could not be reached, so the app cannot start. "
             "Check `[database] url` in the app's secrets.")
    st.caption(f"{type(e).__name__}: {str(e).splitlines()[0][:300]}")
    st.stop()

SIGNUP_CODE = str(_secret("auth", "signup_code") or _secret("app_password") or "")


def _cookie_script(value: str, max_age: int):
    """Set (or, with max_age=0, clear) the login cookie in the browser.

    The cookie only carries the random session token; the server keeps its
    hash (db.py). It is what keeps the user signed in across a refresh:
    st.context.cookies hands it back on the next page load.
    """
    st.html(
        "<script>document.cookie='" + SESSION_COOKIE + "=" + value +
        "; Path=/; Max-Age=" + str(int(max_age)) + "; SameSite=Lax'"
        " + (location.protocol === 'https:' ? '; Secure' : '');</script>",
        unsafe_allow_javascript=True)


def _sign_in(uid: int, uname: str):
    token = DB.create_session(uid)
    st.session_state._uid, st.session_state._uname = uid, uname
    st.session_state._token = token
    st.session_state._set_cookie = token
    st.session_state.pop("_login_fails", None)


def sign_out():
    DB.delete_session(st.session_state.get("_token"))
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.session_state._clear_cookie = True


def current_user():
    """(user_id, username) for this browser session, or None."""
    if st.session_state.get("_uid") is not None:
        return st.session_state._uid, st.session_state._uname
    if st.session_state.get("_clear_cookie"):
        return None                       # just signed out; ignore stale cookie
    token = st.context.cookies.get(SESSION_COOKIE)
    hit = DB.user_for_token(token)
    if hit:
        st.session_state._uid, st.session_state._uname = hit
        st.session_state._token = token
    return hit


def login_screen():
    if st.session_state.pop("_clear_cookie", False):
        _cookie_script("", 0)
    _, mid, _ = st.columns([1, 1.25, 1])
    with mid:
        st.markdown('<div class="eyebrow">Keepp Capital</div>', unsafe_allow_html=True)
        st.markdown("### Keepp Portfolio Outlook")
        st.caption("Sign in to see your portfolio and your saved runs.")
        first_user = DB.user_count() == 0
        t_in, t_up = st.tabs(["Sign in", "Create account"])
        with t_in:
            with st.form("signin", border=False):
                u = st.text_input("Username", autocomplete="username")
                p = st.text_input("Password", type="password",
                                  autocomplete="current-password")
                go = st.form_submit_button("Sign in", type="primary",
                                           width="stretch")
            if go:
                fails = st.session_state.get("_login_fails", 0)
                if fails >= 3:            # slow down password guessing
                    time.sleep(min(2 ** (fails - 2), 10))
                uid = DB.verify_user(u, p)
                if uid is None:
                    st.session_state._login_fails = fails + 1
                    st.error("Wrong username or password.")
                else:
                    _sign_in(uid, DB.username(uid))
                    st.rerun()
        with t_up:
            if not (SIGNUP_CODE or first_user):
                st.info("New accounts are created by the administrator. "
                        "Ask them to add you.", icon=":material/lock:")
            else:
                with st.form("signup", border=False):
                    nu = st.text_input("Choose a username",
                                       help="3–64 characters: letters, "
                                            "numbers, . _ - @")
                    np1 = st.text_input("Choose a password", type="password",
                                        autocomplete="new-password",
                                        help=f"At least {dbm.MIN_PASSWORD_LEN} "
                                             f"characters.")
                    np2 = st.text_input("Repeat the password", type="password",
                                        autocomplete="new-password")
                    code = (st.text_input("Invitation code", type="password",
                                          help="Given to you by Keepp.")
                            if SIGNUP_CODE else "")
                    go2 = st.form_submit_button("Create account", type="primary",
                                                width="stretch")
                if first_user and not SIGNUP_CODE:
                    st.caption("No accounts exist yet — this one will be the "
                               "first. Set `[auth] signup_code` in the app's "
                               "secrets to let others register.")
                if go2:
                    if SIGNUP_CODE and not hmac.compare_digest(
                            code.encode(), SIGNUP_CODE.encode()):
                        st.error("That invitation code is not correct.")
                    elif np1 != np2:
                        st.error("The two passwords do not match.")
                    else:
                        try:
                            uid = DB.create_user(nu, np1)
                        except dbm.AuthError as e:
                            st.error(str(e))
                        else:
                            _sign_in(uid, DB.username(uid))
                            st.rerun()


USER = current_user()
if USER is None:
    login_screen()
    st.stop()
UID, UNAME = USER


# --------------------------------------------------------------------------
# Base case — J.P. Morgan 2027 LTCMA. Returns, volatilities and the sleeve list
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
    ("ALT", "Private Equity",                 10.0,   9.80,  20.31,  ""),
    ("ALT", "Diversified Hedge Funds",         0.0,   6.70,   5.94,  ""),
    ("ALT", "Commodities",                     8.0,   4.80,  17.97,  ""),
]

MODE_BASE = "Base · JPM 2027 LTCMA"
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
     "distribution yield 4.26% plus pull-to-par. J.P. Morgan 2027 LTCMA puts US IG at "
     "6.10% (2026: 5.20%), so this sleeve is now 1.1pp below J.P. Morgan rather than 0.2pp "
     "— the gap to review first. Volatility: regime anchor "
     "√(25%·3.3%² + 50%·6.1%² + 25%·10.6%²) = 7.02%."),
    ("Public Equities &amp; Mutual Funds", "7.00% · 16.78%",
     "Keepp's own figures: MSCI World realised history over 10 years. <b>These are what "
     "Base runs</b> — the panel previously showed 12.88% · 14.47%, the original "
     "workbook's numbers, while attributing them to J.P. Morgan; neither half was "
     "right. For comparison, J.P. Morgan 2027 LTCMA AC World Equity is 7.50% · 17.88% "
     "(2026: 7.00% · 16.78%)."),
    ("Private Equity &amp; Venture Capital", "9.80% · 20.31%",
     "J.P. Morgan 2027 LTCMA, Private Equity (USD), <b>compound</b> return "
     "(2026: 10.20% · 19.78%). The original workbook used the <i>arithmetic</i> figure "
     "for the same line — feeding that into a field the model treats as geometric "
     "double-counts the variance drag. The compound column is used throughout."),
    ("Diversified Hedge Funds", "6.70% · 5.94%",
     "J.P. Morgan 2027 LTCMA, diversified hedge funds (USD) — leveraged and net of "
     "fees (2026: 5.30% · 5.75%, the largest single upgrade in the edition at +1.4pp; "
     "these are industry medians, and J.P. Morgan is explicit that manager selection "
     "is vital). Base weight is 0% because the deck's target allocation has no hedge "
     "fund sleeve — allocate to it in Custom mode."),
    ("Commodities &amp; Real Assets", "4.80% · 17.97%",
     "J.P. Morgan 2027 LTCMA, broad commodities basket (USD), <b>compound</b> return "
     "(2026: 4.60% · 18.32%). As with Private Equity, the original workbook used the "
     "arithmetic figure. The compound column is used throughout."),
    ("Cash &amp; Equivalents", "3.80% · 0.67%",
     "Return: current USD risk-free rate, 3-month T-bill at 3.80% (15-Jul-2026). "
     "Volatility: JPM 2027 LTCMA US Cash. Includes the 2% digital-assets sleeve, "
     "folded into cash per instruction."),
    ("Sub-categories", "3 categories · 59 lines",
     "Sleeves are grouped into J.P. Morgan's own three categories — Fixed Income "
     "(21 lines), Equities (18) and Alternatives (20). Picking a line fills Return "
     "and Vol from the <b>compound</b> column; the line also supplies the sleeve's "
     "row of the correlation matrix. The 2027 edition dropped U.S. Value-Added Real "
     "Estate and added Global Core Real Estate, so the count is unchanged. It also "
     "added Americas Core Real Estate, which is U.S. Core Real Estate under a second "
     "name — same return, same volatility, correlation 1.00 — and is deliberately not "
     "offered, so no allocation can hold the same real estate twice."),
    ("Correlations", "JPM 2027 LTCMA, USD",
     "<b>The full published 60 × 60 matrix</b>, shipped unmodified as "
     "<code>jpm_correlations.csv</code>. The matrix used in any run is a sub-matrix "
     "of it. As published it is not positive semi-definite (smallest eigenvalue "
     "−0.0169); it is repaired once at load with Higham's nearest-correlation "
     "algorithm, and no pair moves as much as 0.01 — the largest single cell moves "
     "0.0047, slightly less than the 2026 matrix needed."),
    ("Keepp-sourced figures", "Custom mode only",
     "Three lines carried Keepp's own return and volatility rather than "
     "J.P. Morgan's: a global IG blend (5.00% · 7.02%), MSCI World realised "
     "history (7.00% · 16.78%) and the live 3-month T-bill (3.80% · 0.67%). "
     "<b>Standard mode no longer uses them</b> — all 58 of its published "
     "asset classes take J.P. Morgan's published compound figure, so the "
     "table comes from one source. The 59th, Digital Assets, is the one "
     "exception and is marked † wherever it appears; see its own row below. "
     "They remain as the starting numbers in "
     "<b>Custom</b>, where any figure can be typed over. On the 2027 edition "
     "all three differ from J.P. Morgan: U.S. Inv Grade Corporate Bonds "
     "(5.00% vs 6.10%), U.S. Cash (3.80% vs 3.20%) and AC World Equity "
     "(7.00% vs 7.50%) — AC World Equity matched on the 2026 edition and no "
     "longer does."),
    ("Digital Assets &dagger;", "10.00% · 46.70%",
     "<b>The one line here that J.P. Morgan does not publish as a row.</b> The "
     "2027 edition puts a figure on bitcoin for the first time — compound "
     "<b>10.0%</b>, arithmetic <b>20.9%</b>, volatility <b>55%</b> — and says "
     "<i>\"we do not yet include bitcoin in our main covariance matrices\"</i>. "
     "Those three figures do not reconcile under this app\'s convention: "
     "10.0 + 55²/2 = 25.12%, against their own 20.9%, off by 4.23pp, where the "
     "worst gap across the 58 published lines is 0.55pp. So the volatility used "
     "here is the one <b>implied by J.P. Morgan\'s own two return figures</b>, "
     "√(2 × (20.9 − 10.0)) = <b>46.70%</b>, with their 10.0% compound. That "
     "trio reproduces their published 20.9% arithmetic exactly. Their 55% is "
     "not used. <b>Correlations are Gold\'s row</b>, the closest published "
     "no-cashflow store of value; where both are held the pair is set to 0.99 "
     "rather than 1.00, which keeps the matrix factorisable. Figures are from "
     "press coverage of the report and are <b>not yet checked against the "
     "PDF</b>."),
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
    it, the donut and Custom mode all say the same thing — with 59 lines on
    offer, a row reading "Global stocks" while the picker says "AC World
    Equity" is a worse trade than plainer wording. The plain-English
    description below the name survives for the lines that have one.
    """
    name = SRC.resolve(line_label)
    cat = SRC.category_of(line_label)
    cat_name = SRC.CATEGORY_NAME.get(cat, "") if cat else ""
    desc = (FRIENDLY[name][1] if name in FRIENDLY
            else CATEGORY_PLAIN.get(cat_name, ""))
    return (SRC.LABEL.get(name, name) if name else str(line_label), desc)


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

    J.P. Morgan's published COMPOUND column for all 58 published lines (and
    the derived figures for Digital Assets, the one house line), with no
    exceptions — the three Keepp overrides that used to sit on top of it went
    with the preset sleeves, so every line in Standard now comes from one
    source and nothing is editable. Never the arithmetic column: engine.py
    treats this field as geometric and derives mu = g + sigma^2/2 itself.
    (Custom mode still seeds the deck's own figures; see custom_alloc_seed.)
    """
    row = SRC.BY_NAME[line_name]
    return float(row[0]), float(row[2])


def std_alloc(lines, weights) -> pd.DataFrame:
    """Standard-mode allocation frame, ordered by J.P. Morgan category."""
    rows = []
    for key in CATEGORY_KEYS:
        for ln in lines:
            if SRC.category_of(ln) != key:
                continue
            r, v = std_numbers(ln)
            rows.append({"Category": SRC.CATEGORY_NAME[key],
                         "LTCMA line": SRC.LABEL[ln],
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


# Keepp's deck sleeves on BlackRock: the nearest BlackRock line for each
# J.P. Morgan deck line. Commodities has no BlackRock line, so it uses the
# J.P. Morgan extra that BlackRock Custom mode already offers.
BR_SLEEVE_LINE = {
    "U.S. Inv Grade Corporate Bonds": "US credit (all maturities)",
    "U.S. Cash": "US cash",
    "AC World Equity": "All country large cap equities",
    "Private Equity": "US private equity (buyout)",
    "Diversified Hedge Funds": "Hedge funds (global)",
    "Commodities": "Commodities (J.P. Morgan)",
}


def custom_alloc_seed() -> pd.DataFrame:
    """Custom mode's starting table: Keepp's own deck sleeves — weights AND
    Keepp's return / vol figures — so the Keepp-vs-provider comparison shows
    straight away. On J.P. Morgan the sleeves sit on the deck's own lines; on
    BlackRock on the nearest BlackRock line (BR_SLEEVE_LINE). The provider
    side of the comparison is then that provider's published figure for the
    line, so the gap is Keepp's view against J.P. Morgan's or BlackRock's."""
    if getattr(SRC, "KEY", JPM_KEY) == BR_KEY:
        rows = []
        for k, line, w, r, v, _ in BASE_SLEEVES:
            br_line = BR_SLEEVE_LINE.get(line)
            if br_line is None or SRC.lookup(br_line) is None:
                continue                      # catalogue changed: skip
            rows.append({"Category": SRC.CATEGORY_NAME[k],
                         "LTCMA line": SRC.LABEL.get(br_line, br_line),
                         "Weight %": w, "Return %": r, "Vol %": v})
        return pd.DataFrame(rows, columns=["Category", "LTCMA line",
                                           "Weight %", "Return %", "Vol %"])
    return pd.DataFrame(
        [{"Category": SRC.CATEGORY_NAME[k], "LTCMA line": SRC.LABEL[line],
          "Weight %": w, "Return %": r, "Vol %": v}
         for k, line, w, r, v, _ in BASE_SLEEVES])


def sleeve_inputs(alloc: pd.DataFrame):
    corr = corr_of(alloc)
    return (tuple(alloc["Weight %"]), tuple(alloc["Return %"]),
            tuple(alloc["Vol %"]), corr)


def corr_of(alloc: pd.DataFrame):
    """Published correlation sub-matrix — shared by simulation and optimizer."""
    return pl.corr_for(SRC, list(alloc["LTCMA line"]))


def bump_editors():
    """Force every keyed input widget to remount after a programmatic write.

    st.data_editor keeps edits as a delta and re-applies them over whatever
    frame it is handed (the old 94.71% Normalize bug); sliders keep their own
    value. A nonce in every key gives each a clean slate.
    """
    st.session_state.editor_nonce = st.session_state.get("editor_nonce", 0) + 1


# --------------------------------------------------------------------------
# Market-assumptions source: J.P. Morgan 2027 LTCMA or BlackRock CMA.
#
# Each source keeps its OWN portfolio: the two catalogues have different
# lines, and correlations only exist inside one provider's matrix, so a
# portfolio is all-J.P. Morgan or all-BlackRock. Switching parks the current
# source's holdings and brings back the other's (or starts it empty).
# --------------------------------------------------------------------------
JPM_KEY, BR_KEY = "jpm", brc.KEY
PORTFOLIO_KEYS = ("std_lines", "std_w", "std_tick", "custom_alloc",
                  "custom_bases")
st.session_state.setdefault("cma_source", JPM_KEY)
st.session_state.setdefault("br_scenario", brc.DEFAULT_SCENARIO)
# A segmented control can be clicked off to "nothing selected"; keep the
# source that was active instead of leaving the run without assumptions.
if st.session_state.cma_source not in (JPM_KEY, BR_KEY):
    st.session_state.cma_source = st.session_state.get("_active_src", JPM_KEY)
if st.session_state.br_scenario not in brc.SCENARIOS:
    st.session_state.br_scenario = brc.DEFAULT_SCENARIO
st.session_state.setdefault("years", 10)
st.session_state.setdefault("paths", 10_000)
st.session_state.setdefault("sim_seed", 20260722)


def source_for(key, scenario, years):
    """The assumptions object for a source. J.P. Morgan is the jpm_ltcma
    module itself; BlackRock is built per scenario and horizon (its returns
    depend on the number of years)."""
    if key == BR_KEY:
        return brc.get(scenario, int(years))
    return jpm


def source_id_of(key, scenario):
    """What a saved run records about its assumptions."""
    return brc.source_id(scenario) if key == BR_KEY else JPM_KEY


# A restore must switch source before the switch logic below runs.
if "pending_restore" in st.session_state:
    _sid = getattr(st.session_state.pending_restore.record, "source", JPM_KEY)
    if str(_sid).startswith(BR_KEY):
        st.session_state.cma_source = BR_KEY
        st.session_state.br_scenario = brc.parse_source_id(_sid)
    else:
        st.session_state.cma_source = JPM_KEY

_prev_src = st.session_state.get("_active_src")
if _prev_src is not None and _prev_src != st.session_state.cma_source:
    _stash = st.session_state.setdefault("_src_stash", {})
    _stash[_prev_src] = {k: st.session_state[k] for k in PORTFOLIO_KEYS
                         if k in st.session_state}
    for _k in PORTFOLIO_KEYS:
        st.session_state.pop(_k, None)
    st.session_state.update(_stash.get(st.session_state.cma_source, {}))
    st.session_state.pop("opt_result", None)
    bump_editors()
st.session_state._active_src = st.session_state.cma_source

SRC = source_for(st.session_state.cma_source, st.session_state.br_scenario,
                 st.session_state.years)

if "custom_alloc" not in st.session_state:
    st.session_state.setdefault("editor_nonce", 0)
    st.session_state.custom_alloc = custom_alloc_seed()
    # Standard mode: an ordered list of LTCMA lines, a weight for each, and a
    # tick saying whether the optimizer may use it. Anything above 0% is
    # always ticked; a 0% line can be ticked to offer it to the optimizer
    # without holding it today.
    (st.session_state.std_lines, st.session_state.std_w,
     st.session_state.std_tick) = seed_standard()
    st.session_state.setdefault("mode", MODE_BASE)

# Simulation settings (years, paths, sim_seed) live in session state, set
# above, so a restored run can change them before the sidebar is drawn.

# A mode switch requested by "Apply as Custom sleeves" on the previous run. It
# must land before the radio that owns the `mode` key is instantiated.
if "pending_mode" in st.session_state:
    st.session_state.mode = st.session_state.pop("pending_mode")


def apply_restore(stored):
    """Put a saved run's inputs back into the app: mode, holdings, returns,
    risk figures, horizon, number of futures and seed. Runs at the top of the
    script, before any widget that owns these keys exists."""
    rec = stored.record
    is_std = rec.mode == "Base"
    st.session_state.mode = MODE_BASE if is_std else MODE_CUSTOM
    st.session_state.years = int(rec.years)
    st.session_state.paths = int(rec.paths)
    st.session_state.sim_seed = int(rec.seed)
    rsrc = source_for(st.session_state.cma_source, st.session_state.br_scenario,
                      rec.years)
    if is_std:
        held = [rsrc.resolve(s["LTCMA line"]) for s in rec.sleeves]
        saved_w = {rsrc.resolve(s["LTCMA line"]): float(s["Weight %"])
                   for s in rec.sleeves}
        extra = [ln for ln in stored.pool if ln not in held]
        # A saved run can name a line the active catalogue no longer has: an
        # LTCMA edition retires one (U.S. Value-Added Real Estate went in the
        # 2027 edition), or the run was saved under the other provider. Custom
        # mode already survives that — sync_group rule 1 falls the row back to
        # its category default — but Standard mode looks the line up directly
        # and would raise KeyError, losing the whole restore. Drop what the
        # catalogue does not know, keep the rest, and say what went.
        gone = list(dict.fromkeys(ln for ln in held + extra
                                  if ln not in rsrc.BY_NAME))
        if gone:
            held = [ln for ln in held if ln not in gone]
            extra = [ln for ln in extra if ln not in gone]
            lost = sum(saved_w.get(ln, 0.0) for ln in gone)
            st.toast(f"{', '.join(gone)} is no longer in the catalogue and was "
                     f"dropped from this run"
                     + (f" — {lost:g}% of the allocation. Re-weight before "
                        f"reading the result." if lost else "."),
                     icon=":material/warning:")
        st.session_state.std_lines = held + extra
        st.session_state.std_w = {ln: saved_w[ln] for ln in held}
        st.session_state.std_w.update({ln: 0.0 for ln in extra})
        st.session_state.std_tick = {ln: True for ln in held + extra}
    else:
        st.session_state.custom_alloc = pd.DataFrame(
            rec.sleeves, columns=rs.ALLOC_COLS)
        st.session_state.pop("custom_bases", None)
    # MUST bump — the editors would otherwise re-apply stale hand edits on top
    # of the restored values (see bump_editors()).
    bump_editors()


if "pending_restore" in st.session_state:
    apply_restore(st.session_state.pop("pending_restore"))
    SRC = source_for(st.session_state.cma_source, st.session_state.br_scenario,
                     st.session_state.years)

IS_BR = st.session_state.cma_source == BR_KEY
SRC_SHORT = brc.SHORT if IS_BR else "J.P. Morgan"
SRC_NAME = (f"{brc.NAME} · {brc.SCENARIOS[st.session_state.br_scenario]}"
            if IS_BR else "J.P. Morgan 2027 LTCMA")
SRC_ID = source_id_of(st.session_state.cma_source, st.session_state.br_scenario)
N_LINES = len(SRC.LINES)

# What the Details tab's "Where the assumptions come from" shows on BlackRock.
BR_SOURCES = [
    ("Publisher", f"BlackRock · {brc.EDITION}",
     f"BlackRock Capital Market Assumptions, {brc.EDITION} edition, data as of "
     f"{brc.AS_OF}. US-dollar figures, gross of fees, for passive index "
     f"exposure. {N_LINES} asset classes; the Global 60/40 portfolio row "
     f"is a portfolio, not an asset class, and is left out. BlackRock updates "
     f"the workbook every quarter."),
    ("Returns", "Geometric · by horizon",
     "BlackRock publishes a compound (geometric) annual return for 5, 7, 10, "
     "15, 20, 25 and 30 years. The app uses the column for the number of years "
     "you chose, interpolating between the two nearest horizons (e.g. 12 years "
     "sits 2/5 of the way from the 10- to the 15-year figure). Runs under 5 "
     "years use the 5-year figure."),
    ("Scenarios", "Base case + 2",
     "Base case is BlackRock's 'Starting point' sheet. 'AI productivity boom' "
     "and 'Global risk premia rise' are its two published scenarios; they "
     "change expected returns only."),
    ("Volatility", "As published",
     "BlackRock's annual volatility per asset class. US cash is published at "
     "0%; the optimizer floors it at a tiny positive value."),
    ("Real assets &amp; digital", "J.P. Morgan · Custom only",
     "BlackRock's CMA has <b>no commodities, gold or digital-asset line</b>, so "
     "a BlackRock portfolio could not hold a real-asset sleeve at all. Three "
     "J.P. Morgan lines are offered instead — Commodities, Gold and Digital "
     "Assets — in <b>Custom mode only</b>, so Standard stays a pure BlackRock "
     "catalogue. Their correlations are not a compromise: BlackRock's matrix is "
     "already built on J.P. Morgan's published correlations, so these three sit "
     "in the same construction as every other line. Their <b>returns and "
     "volatilities are J.P. Morgan's</b>, which has two consequences worth "
     "knowing: they <b>do not move when you change the scenario</b>, and they "
     "<b>do not move with the horizon</b> — BlackRock reprices its own assets "
     "across scenarios and publishes a figure per horizon; J.P. Morgan "
     "publishes one number."),
    ("Correlations", "BlackRock + J.P. Morgan",
     "BlackRock publishes each asset's correlation to government bonds and to "
     "equities only, not a full matrix. Those two numbers set the common part "
     "of every pair exactly. What two factors cannot capture — how closely "
     "related two members of one family are, such as levered and unlevered "
     "direct lending — is taken from J.P. Morgan's published matrix, via the "
     "closest J.P. Morgan line for each BlackRock asset. Method and mapping: "
     "blackrock_cma.py."),
    ("Categories", "3 groups",
     "BlackRock's Fixed income, Equities and Private markets are shown as the "
     "app's Fixed Income, Equities and Alternatives."),
]

NONCE = st.session_state.get("editor_nonce", 0)


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
    years = st.slider("For how many years?", 1, 30, key="years",
                      help="How long the money stays invested.")

    st.write("")
    # Which provider's assumptions. Read at the top of the script (key
    # "cma_source") so the whole run, and each source's own portfolio, follow it.
    st.segmented_control(
        "Assumptions from", [JPM_KEY, BR_KEY], key="cma_source",
        selection_mode="single", required=True, width="stretch",
        format_func=lambda k: ("J.P. Morgan" if k == JPM_KEY else "BlackRock"),
        help="J.P. Morgan 2027 Long-Term Capital Market Assumptions (59 asset "
             "classes, one long-term return each), or BlackRock's Capital "
             "Market Assumptions, August 2026 (34 asset classes, returns by "
             "horizon). Each keeps its own portfolio; switching back brings "
             "yours back.")
    with st.container():                   # BlackRock-only controls
        if IS_BR:
            st.selectbox(
                "BlackRock scenario", list(brc.SCENARIOS), key="br_scenario",
                format_func=lambda k: brc.SCENARIOS[k],
                help="BlackRock's base case, or one of its two published "
                     "scenarios. Scenarios change expected returns only; risk "
                     "and correlations are the same in all three.")
            st.caption(f"Using {SRC.HORIZON_NOTE}.")

    st.write("")
    # Always visible — the manager's most-used switch, so it is not hidden in
    # the expander. The widget key stays "mode" so pending_mode still works.
    mode = st.radio(
        "Market assumptions", [MODE_BASE, MODE_CUSTOM], key="mode",
        format_func=lambda m: ("Standard" if m == MODE_BASE else "Custom"),
        captions=[f"{SRC_SHORT} figures as published — only the mix can change",
                  "Edit holdings, returns and risk figures"],
        help=f"Standard keeps {SRC_SHORT}'s published assumptions. Custom lets "
             f"you add any of {SRC_SHORT}'s {N_LINES} asset classes and type "
             f"your own return and risk numbers.")

    st.write("")
    with st.expander("Advanced settings"):
        paths = st.select_slider(
            "Number of simulated futures",
            options=[1_000, 2_500, 5_000, 10_000, 25_000, 50_000], key="paths",
            help="More futures = smoother, more precise results, but slower.")
        seed = st.number_input(
            "Random seed", step=1, key="sim_seed",
            help="Fix the seed so everyone running the same inputs sees the "
                 "same numbers.")
    is_base = mode == MODE_BASE

    st.markdown(
        '<div class="foot" style="margin-top:1.2rem">Projections are simulations '
        'based on long-term market assumptions, not guarantees. Figures are in US '
        'dollars before fees, taxes and inflation.</div>', unsafe_allow_html=True)

    st.write("")
    ac_a, ac_b = st.columns([1.6, 1], vertical_alignment="center")
    ac_a.markdown(f'<div class="foot">Signed in as <b>{UNAME}</b></div>',
                  unsafe_allow_html=True)
    if ac_b.button("Sign out", key="sign_out", width="stretch"):
        sign_out()
        st.rerun()
    if "_set_cookie" in st.session_state:
        _cookie_script(st.session_state.pop("_set_cookie"), COOKIE_MAX_AGE)


# --------------------------------------------------------------------------
# Header + navigation
# --------------------------------------------------------------------------
chip = "" if is_base else '<span class="modechip">Custom assumptions</span>'
if IS_BR:
    chip += (f'<span class="modechip">BlackRock · '
             f'{brc.SCENARIOS[st.session_state.br_scenario]}</span>')
st.markdown('<div class="eyebrow">Keepp Capital · Portfolio outlook</div>',
            unsafe_allow_html=True)
st.markdown(f"# How could your portfolio grow?{chip}", unsafe_allow_html=True)
st.markdown(
    f'<p class="lede">We ran {paths:,} possible futures for the markets and '
    f'tracked your portfolio through each one. Here is the range of what could '
    f'happen over the next {years} year{"s" if years != 1 else ""}.</p>',
    unsafe_allow_html=True)

tab_over, tab_mix, tab_opt, tab_det, tab_cf = st.tabs(
    ["Overview", "Your mix", "Improve the mix", "Details & saved runs",
     "Contributions & withdrawals"],
    key="nav", on_change="rerun")


# --------------------------------------------------------------------------
# Geography — region breakdown and the optimizer's geography limits
# (keepp_mc/geography.py). A label only: no return, risk or correlation uses it.
# --------------------------------------------------------------------------
GEO_COLS = ["Region"] + [f"{sc} {mm} %" for sc in geo.SCOPES for mm in ("min", "max")]
REGION_COLOR = {"US": "#2a78d6", "Europe": "#1baf7a", "Asia": "#eb6834",
                "Japan": "#e87ba4", "Global": "#A9B4B5"}


def geo_limits_current() -> pd.DataFrame:
    """The geography-limits table as currently edited."""
    return _limits_current(
        "geo_base", "geo_limits",
        lambda: pd.DataFrame([{"Region": r, **{c: np.nan for c in GEO_COLS[1:]}}
                              for r in geo.REGIONS], columns=GEO_COLS))


def _limits_current(base_key, widget_key, seed) -> pd.DataFrame:
    """A fixed-row limits table as currently edited. The editors live on
    Improve the mix, which renders after Your mix, so their pending edits are
    applied here from the widget state to keep the warnings on Your mix current."""
    base = st.session_state.get(base_key)
    if base is None:
        base = seed()
        st.session_state[base_key] = base
    df = base.copy()
    edits = st.session_state.get(widget_key)
    if edits:
        try:
            rows = edits.get("edited_rows", {}) if hasattr(edits, "get") \
                else edits["edited_rows"]
        except Exception:
            rows = {}
        for i, changes in rows.items():
            for col, val in changes.items():
                if col in df.columns:
                    df.at[int(i), col] = val
    return df


def region_panel(alloc_df, key):
    """By-region bar + region x asset-type table for an allocation frame."""
    if alloc_df is None or len(alloc_df) == 0:
        return
    w = pd.to_numeric(alloc_df["Weight %"], errors="coerce").fillna(0.0)
    if float(w.sum()) <= 0:
        return
    lines, cats = list(alloc_df["LTCMA line"]), list(alloc_df["Category"])
    bd = geo.breakdown(SRC, lines, cats, list(w))
    section("By region")
    bar = pd.DataFrame({"Region": geo.REGIONS, "pct": bd["Total"].values,
                        "x": ["Portfolio"] * len(geo.REGIONS)})
    bar["lbl"] = bar["pct"].map(lambda v: f"{v:.1f}%")
    bar = bar[bar["pct"] > 0.005]
    st.altair_chart(
        alt.Chart(bar).mark_bar(height=26).encode(
            x=alt.X("sum(pct):Q", stack="zero", title=None,
                    axis=alt.Axis(format=".0f", labelExpr="datum.label + '%'",
                                  labelColor=INK_3, gridColor=GRID,
                                  domain=False, ticks=False)),
            y=alt.Y("x:N", title=None, axis=None),
            color=alt.Color("Region:N", sort=geo.REGIONS,
                            scale=alt.Scale(domain=geo.REGIONS,
                                            range=[REGION_COLOR[r] for r in geo.REGIONS]),
                            legend=alt.Legend(orient="bottom", title=None,
                                              labelColor=INK_2)),
            order=alt.Order("order:Q"),
            tooltip=[alt.Tooltip("Region:N"), alt.Tooltip("lbl:N", title="Share")])
        .transform_calculate(order=f"indexof({geo.REGIONS}, datum.Region)")
        .properties(height=78).configure_view(strokeWidth=0),
        width="stretch")
    tbl = (bd.reset_index().rename(columns={"index": "Region", "Alternatives": "Alts"})
           [["Region", "Total", "Stocks", "Bonds", "Alts"]])
    with st.expander("Region by asset type"):
        st.dataframe(tbl, hide_index=True, width="stretch", key=f"geo_tbl_{key}",
                     column_config={c: st.column_config.NumberColumn(format="%.1f%%")
                                    for c in ["Total", "Stocks", "Bonds", "Alts"]})
        st.caption("Broad indices such as AC World, EAFE and World government "
                   "bonds are split by fixed regional weights from their index "
                   "providers. Global = other countries, emerging markets, and "
                   "assets with no single region (commodities, gold, hedge "
                   "funds, global private markets).")
    bad = geo.check(geo.exposure_rows(SRC, lines, cats, geo_limits_current()),
                    list(w / 100.0))
    if bad:
        st.warning("**Outside your geography limits** (set on *Improve the mix*): "
                   + "; ".join(f"{n} {v*100:.1f}% (limit {lo*100:.0f}–{hi*100:.0f}%)"
                               for n, v, lo, hi in bad),
                   icon=":material/public:")


# --------------------------------------------------------------------------
# Liquidity — tier ladder and the optimizer's liquidity limits
# (keepp_mc/liquidity.py). Same pattern as geography: a label only.
# --------------------------------------------------------------------------
LIQ_COLS = ["Tiers", "Min %", "Max %"]
# Ordinal: one teal ramp, darkest = most liquid.
TIER_COLOR = dict(zip(liq.TIERS, ["#00544B", "#00877A", "#4FAE9F",
                                  "#93CEC4", "#C9E6E1"]))
TIER_NAME = {t: f"{t} · {liq.LABEL[t]}" for t in liq.TIERS}


def liq_limits_current() -> pd.DataFrame:
    """The liquidity-limits table as currently edited."""
    return _limits_current(
        "liq_base", "liq_limits",
        lambda: pd.DataFrame([{"Tiers": g, "Min %": np.nan, "Max %": np.nan}
                              for g in liq.GROUPS], columns=LIQ_COLS))


def liq_overrides() -> dict:
    """Personal-line tier / haircut overrides from Custom mode, in the shape
    liquidity.py takes (haircut in decimals)."""
    return {n: {"tier": o.get("tier"),
                "haircut": (None if o.get("haircut_pct") is None
                            or pd.isna(o.get("haircut_pct"))
                            else float(o["haircut_pct"]) / 100.0)}
            for n, o in st.session_state.get("pl_liq", {}).items()}


def fmt_days(d) -> str:
    if d < 14:
        return f"{d:.0f} day{'s' if round(d) != 1 else ''}"
    if d < 60:
        return f"{d / 7:.0f} weeks"
    if d < 730:
        return f"{d / 30.4:.0f} months"
    return f"{d / 365:.1f} years"


def liquidity_panel(alloc_df, key, table=False):
    """Liquidity ladder (stacked bar) + plain sentence for an allocation
    frame; with `table`, the tier x asset-type table and the limits warning."""
    if alloc_df is None or len(alloc_df) == 0:
        return
    w = pd.to_numeric(alloc_df["Weight %"], errors="coerce").fillna(0.0)
    if float(w.sum()) <= 0:
        return
    lines, cats = list(alloc_df["LTCMA line"]), list(alloc_df["Category"])
    ov = liq_overrides()
    bd = liq.breakdown(SRC, lines, cats, list(w), ov)
    days = liq.days_to_cash(SRC, lines, list(w), ov)
    section("How quickly it can become cash",
            liq.ladder_sentence(bd["Total"]) + f" Weighted-average time to "
            f"cash ≈ <b>{fmt_days(days)}</b>.")
    bar = pd.DataFrame({"Tier": [TIER_NAME[t] for t in liq.TIERS],
                        "pct": bd["Total"].values,
                        "when": [liq.TIME_TO_CASH[t] for t in liq.TIERS],
                        "x": ["Portfolio"] * len(liq.TIERS)})
    bar["lbl"] = bar["pct"].map(lambda v: f"{v:.1f}%")
    bar = bar[bar["pct"] > 0.005]
    names = [TIER_NAME[t] for t in liq.TIERS]
    st.altair_chart(
        alt.Chart(bar).mark_bar(height=26, stroke="white", strokeWidth=1).encode(
            x=alt.X("sum(pct):Q", stack="zero", title=None,
                    axis=alt.Axis(format=".0f", labelExpr="datum.label + '%'",
                                  labelColor=INK_3, gridColor=GRID,
                                  domain=False, ticks=False)),
            y=alt.Y("x:N", title=None, axis=None),
            color=alt.Color("Tier:N", sort=names,
                            scale=alt.Scale(domain=names,
                                            range=[TIER_COLOR[t] for t in liq.TIERS]),
                            legend=alt.Legend(orient="bottom", title=None,
                                              labelColor=INK_2, columns=3)),
            order=alt.Order("order:Q"),
            tooltip=[alt.Tooltip("Tier:N"), alt.Tooltip("lbl:N", title="Share"),
                     alt.Tooltip("when:N", title="Time to cash")])
        .transform_calculate(order=f"indexof({names}, datum.Tier)")
        .properties(height=92).configure_view(strokeWidth=0),
        width="stretch")
    if not table:
        return
    tbl = (bd.rename(columns={"Alternatives": "Alts"})
           .assign(Tier=[TIER_NAME[t] for t in liq.TIERS],
                   **{"Time to cash": [liq.TIME_TO_CASH[t] for t in liq.TIERS]})
           [["Tier", "Total", "Stocks", "Bonds", "Alts", "Time to cash"]])
    with st.expander("Liquidity tier by asset type"):
        st.dataframe(tbl, hide_index=True, width="stretch", key=f"liq_tbl_{key}",
                     column_config={c: st.column_config.NumberColumn(format="%.1f%%")
                                    for c in ["Total", "Stocks", "Bonds", "Alts"]})
        st.caption("Each cell is the share of the whole portfolio held in that "
                   "tier through that asset type; Total is the row sum. Each "
                   "asset class sits in one liquidity tier (a few are split, "
                   "e.g. macro hedge funds half L3, half L4). Time to cash "
                   "assumes an orderly sale; in a crisis, selling early costs a "
                   "haircut — up to 15% for L2, 30% for L3, 40% for L4 and 60% "
                   "for L5, from secondary-market pricing (Jefferies, Lazard, "
                   "2026).")
    bad = liq.check(liq.exposure_rows(SRC, lines, liq_limits_current(), ov),
                    list(w / 100.0))
    if bad:
        st.warning("**Outside your liquidity limits** (set on *Improve the mix*): "
                   + "; ".join(f"{n} {v*100:.1f}% (limit {lo*100:.0f}–{hi*100:.0f}%)"
                               for n, v, lo, hi in bad),
                   icon=":material/water_drop:")


# ==========================================================================
# TAB — Your mix (runs first: everything else depends on the allocation)
# ==========================================================================
# --------------------------------------------------------------------------
# Custom mode — Your mix is a two-view builder. Each holding gets a share and
# Keepp's own return and volatility (both start at the provider's figures:
# J.P. Morgan 2027, or BlackRock for the chosen scenario and horizon); the
# same portfolio is then simulated on Keepp's figures AND on the provider's,
# side by side, and Save stores both.
#
# Keepp's figures ARE Custom mode's Return % / Vol %: the builder reads and
# writes st.session_state.custom_alloc, so Overview, Improve the mix,
# restores and "Apply as Custom sleeves" all keep working unchanged and show
# the Keepp view. Logic lives in keepp_mc/compare.py; this is layout only.
# The provider is whatever the sidebar selects (SRC): a BlackRock portfolio
# is built from BlackRock lines and compared against BlackRock.
# --------------------------------------------------------------------------
KEEPP_C, JPM_C, BR_C = ACCENT, DEEP_BLUE, "#7B4F9E"
# Provider labels for the comparison: short name in tables, title on cards.
PROV = SRC_SHORT
PROV_TITLE = (f"BlackRock · {brc.SCENARIOS[st.session_state.br_scenario]}"
              if IS_BR else "J.P. Morgan 2027")


def cmp_diff(kind, d):
    """Signed difference, Keepp minus the provider, with a neutral more/less
    colour — a higher assumption is not 'better', only more optimistic."""
    if abs(d) < (0.5 if kind == "money" else 5e-5):
        return f'<span style="color:{INK_3}">— same</span>'
    colr, arrow = (UP_COLOR, "▲") if d > 0 else (DOWN_COLOR, "▼")
    txt = (("+" if d > 0 else "") + money(d) if kind == "money"
           else f"{d * 100:+.2f} pp".replace("-", "−"))
    return f'<span style="color:{colr};font-weight:650">{arrow} {txt}</span>'


def cmp_chart(fl, amount, height=400):
    """Both fans on one set of axes: 5–95 and 25–75 bands, typical path."""
    fl = fl.copy()
    for c in ("p05", "p50", "p95"):
        fl[c + "_l"] = fl[c].map(money)
    cs = alt.Scale(domain=["Keepp", PROV], range=[KEEPP_C, JPM_C])
    ax_y = alt.Axis(title=None, format="$.3~s", labelColor=INK_3, gridColor=GRID,
                    domain=False, ticks=False, tickCount=6, labelFontSize=12,
                    labelPadding=8)
    ax_x = alt.Axis(title="Years from today", titleColor=INK_3, labelColor=INK_3,
                    grid=False, tickMinStep=1, format="d", labelFontSize=12,
                    domainColor=GRID, tickColor=GRID, titleFontWeight=400)
    xs = alt.Scale(domain=[0, float(fl["Year"].max())], nice=False)
    ys = alt.Scale(zero=False, nice=True)
    b = alt.Chart(fl)
    col = alt.Color("View:N", scale=cs, legend=None)
    # One light 9-in-10 band per view; the two overlap, so no inner band —
    # four stacked translucent areas turned into an unreadable wash.
    outer = b.mark_area(opacity=.09).encode(
        x=alt.X("Year:Q", axis=ax_x, scale=xs),
        y=alt.Y("p05:Q", axis=ax_y, scale=ys), y2="p95:Q", color=col)
    edges = b.transform_fold(["p05", "p95"], as_=["k", "v"]).mark_line(
        strokeWidth=1.4, strokeDash=[5, 4], opacity=.9).encode(
        x="Year:Q", y=alt.Y("v:Q", scale=ys), color=col,
        detail="k:N")
    med = b.mark_line(strokeWidth=2.8).encode(
        x="Year:Q", y=alt.Y("p50:Q", scale=ys), color=col)
    start = alt.Chart(pd.DataFrame({"y": [amount]})).mark_rule(
        color=INK_3, strokeDash=[4, 4], opacity=.8).encode(y="y:Q")

    wide = fl.pivot_table(index="Year", columns="View",
                          values=["p05", "p50", "p95"]).reset_index()
    wide.columns = ["Year"] + [f"{v}|{p}" for p, v in wide.columns[1:]]
    for c in wide.columns[1:]:
        wide[c + "_l"] = wide[c].map(money)
    wide["yr_l"] = wide["Year"].map(lambda y: f"Year {y:g}")
    hover = alt.selection_point(fields=["Year"], nearest=True, on="pointerover",
                                empty=False, clear="pointerout")
    rule = alt.Chart(wide).mark_rule(color=INK_3, strokeWidth=1).encode(
        x="Year:Q",
        opacity=alt.condition(hover, alt.value(.6), alt.value(0)),
        tooltip=[alt.Tooltip("yr_l:N", title="When"),
                 alt.Tooltip("Keepp|p50_l:N", title="Keepp · typical"),
                 alt.Tooltip(f"{PROV}|p50_l:N", title=f"{PROV} · typical"),
                 alt.Tooltip("Keepp|p05_l:N", title="Keepp · weak markets"),
                 alt.Tooltip(f"{PROV}|p05_l:N", title=f"{PROV} · weak markets"),
                 alt.Tooltip("Keepp|p95_l:N", title="Keepp · strong markets"),
                 alt.Tooltip(f"{PROV}|p95_l:N",
                             title=f"{PROV} · strong markets")],
    ).add_params(hover)
    dots = b.mark_circle(size=70, stroke="white", strokeWidth=2).encode(
        x="Year:Q", y=alt.Y("p50:Q", scale=ys), color=col,
        opacity=alt.condition(hover, alt.value(1), alt.value(0)))

    end = fl[fl["Year"] == fl["Year"].max()]
    tags = alt.Chart(end).mark_text(align="left", dx=8, fontSize=13,
                                    fontWeight=700).encode(
        x="Year:Q", y=alt.Y("p50:Q", scale=ys), text="p50_l:N", color=col)
    return (outer + edges + start + med + rule + dots + tags).properties(
        height=height, padding={"left": 4, "right": 74, "top": 10, "bottom": 4}
    ).configure_view(strokeWidth=0)


def cmp_css():
    st.markdown(f"""<style>
      table.cmpt {{ width:100%; border-collapse:collapse; font-size:.92rem;
                   font-variant-numeric:tabular-nums; }}
      table.cmpt th {{ text-align:right; font-size:.78rem; font-weight:600;
                      color:{INK_3}; padding:8px 10px;
                      border-bottom:1px solid {GRID}; }}
      table.cmpt th:first-child, table.cmpt td:first-child {{ text-align:left; }}
      table.cmpt td {{ text-align:right; padding:9px 10px;
                      border-bottom:1px solid #EEF2F1; }}
      table.cmpt td.k {{ color:{ACCENT_DK}; font-weight:650; }}
      table.cmpt td.j {{ color:{DEEP_BLUE}; font-weight:650; }}
      table.cmpt td.b {{ color:{BR_C}; font-weight:650; }}
      .cmphead {{ border-radius:14px; padding:18px 22px; background:#fff;
                 border:1px solid {GRID}; border-top:5px solid var(--c); }}
      .cmphead .t {{ font-size:.8rem; font-weight:650; letter-spacing:.06em;
                    text-transform:uppercase; color:var(--c); }}
      .cmphead .v {{ font-size:2.3rem; font-weight:700; color:{INK};
                    line-height:1.15; margin:.25rem 0 .2rem;
                    font-variant-numeric:tabular-nums; }}
      .cmphead .s {{ color:{INK_2}; font-size:.88rem; line-height:1.5; }}
      .cmphdr {{ color:{INK_3}; font-size:.74rem; font-weight:600; }}
    </style>""", unsafe_allow_html=True)



def personal_liquidity_editor(alloc):
    # Liquidity of Keepp's personal lines: L5 by default, tier and
    # crisis haircut editable here (keepp_mc/liquidity.py). The base
    # frame is rebuilt only when the set of personal lines changes,
    # and the widget key changes with it, so edits never snap back.
    held_pl = [n for n in pl.NAMES
               if n in set(alloc["LTCMA line"].astype(str).str.strip())]
    if held_pl:
        prev = st.session_state.get("pl_liq", {})
        base = st.session_state.get("pl_liq_base")
        if base is None or list(base["Investment"]) != held_pl:
            base = pd.DataFrame([{
                "Investment": n,
                "Tier": TIER_NAME[prev.get(n, {}).get("tier")
                                  or liq.PERSONAL_DEFAULT],
                "Crisis haircut %": prev.get(n, {}).get("haircut_pct",
                                                        np.nan)}
                for n in held_pl])
            st.session_state.pl_liq_base = base
        st.markdown(
            '<div style="font-weight:650;margin:1rem 0 .15rem">'
            'Liquidity of your personal lines</div>'
            '<div class="muted" style="margin-bottom:.3rem">Locked (L5) '
            'unless you say otherwise. Leave the haircut blank to use the '
            'tier\'s default.</div>', unsafe_allow_html=True)
        pl_ed = st.data_editor(
            base, key=f"pl_liq_{'_'.join(held_pl)}", hide_index=True,
            width='stretch', num_rows="fixed", disabled=["Investment"],
            column_config={
                "Investment": st.column_config.TextColumn(width="medium"),
                "Tier": st.column_config.SelectboxColumn(
                    "Liquidity tier", options=list(TIER_NAME.values()),
                    required=True),
                "Crisis haircut %": st.column_config.NumberColumn(
                    min_value=0.0, max_value=100.0, step=0.5,
                    format="%.1f",
                    help="Discount to value if it had to be sold in a "
                         "crisis. Blank = the tier's midpoint ("
                         + ", ".join(f"{t} {liq.HAIRCUT[t]*100:g}%"
                                     for t in liq.TIERS) + ")."),
            })
        st.session_state.pl_liq = {
            r["Investment"]: {"tier": str(r["Tier"]).split(" · ")[0],
                              "haircut_pct": r["Crisis haircut %"]}
            for _, r in pl_ed.iterrows()}


def custom_builder():
    """Custom mode: pick the provider's lines, type share, Keepp return and
    Keepp vol. Returns the allocation table (Keepp figures)."""
    cmp_css()
    section("What's in your portfolio",
            "Open an asset class and pick an investment from its dropdown. Type "
            "its <b>share</b>, then <b>Keepp's expected return</b> — and "
            f"volatility, if Keepp's view differs. Both start at {SRC_NAME}'s "
            "figures, so anything left alone is identical in both views. "
            "Below, the same portfolio is simulated on Keepp's figures and on "
            f"{PROV}'s, side by side."
            + (f" BlackRock's returns are for {SRC.HORIZON_NOTE}." if IS_BR
               else ""))
    store = st.session_state.custom_alloc
    frame = cmpm.frame_from_alloc(store, SRC)
    # BlackRock reprices with the scenario and the horizon. A Keepp figure
    # nobody typed (still equal to the provider's figure last run) follows
    # the provider to its new value; a typed one stays. Without this an
    # untouched row would freeze at the old figure and show a "Keepp view"
    # that no one entered.
    # (provider family, scenario id, years): only a scenario or horizon
    # change WITHIN one provider re-bases; switching provider swaps portfolios.
    prov_key = (st.session_state.cma_source, SRC_ID,
                int(st.session_state.years))
    prev = st.session_state.get("_cmp_prov")
    if prev is not None and prev[0] != prov_key and prev[0][0] == prov_key[0] and len(frame):
        old_fig = prev[1]
        moved = False
        for i, row in frame.iterrows():
            o = old_fig.get(row["Line"])
            if o is None:
                continue
            if abs(row["Keepp return %"] - o[0]) < 1e-9:
                frame.at[i, "Keepp return %"] = row["Provider return %"]
                moved = moved or abs(o[0] - row["Provider return %"]) > 1e-9
            if abs(row["Keepp vol %"] - o[1]) < 1e-9:
                frame.at[i, "Keepp vol %"] = row["Provider vol %"]
                moved = moved or abs(o[1] - row["Provider vol %"]) > 1e-9
        if moved:
            st.session_state.custom_alloc = cmpm.to_alloc(frame, "keepp", SRC)
            st.session_state._cmp_prov = (prov_key, dict(zip(
                frame["Line"], zip(frame["Provider return %"],
                                   frame["Provider vol %"]))))
            bump_editors()
            st.rerun()
    st.session_state._cmp_prov = (prov_key, dict(zip(
        frame["Line"], zip(frame["Provider return %"], frame["Provider vol %"]))))
    dropped = len(store) - len(frame)
    if dropped > 0:
        st.caption(f"{dropped} row(s) were not {PROV} lines (or were "
                   f"listed twice) and were merged or left out.")
    lines = list(frame["Line"])
    W = dict(zip(frame["Line"], frame["Weight %"]))
    R = dict(zip(frame["Line"], frame["Keepp return %"]))
    V = dict(zip(frame["Line"], frame["Keepp vol %"]))
    live_w = {ln: float(st.session_state.get(f"kw_{ln}_{NONCE}", W[ln]) or 0)
              for ln in lines}

    new_w, new_r, new_v, remove, add_line = {}, {}, {}, None, None
    for key in cmpm.CATEGORY_KEYS:
        cat = cmpm.CATEGORY_NAME[key]
        mine = [ln for ln in lines if cmpm.category_of(ln, SRC) == key]
        opts = cmpm.choices(key, SRC)
        c_sum = sum(live_w[ln] for ln in mine)
        label = (f"{CATEGORY_PLAIN.get(cat, cat)}  ·  nothing added  ·  "
                 f"{len(opts)} to choose from" if not mine else
                 f"{CATEGORY_PLAIN.get(cat, cat)}  ·  {c_sum:.1f}%  ·  "
                 f"{len(mine)} listed")
        with st.expander(label, expanded=bool(mine) or not lines):
            with st.container():
                if mine:
                    h = st.columns([1.75, .8, .95, .95, .26])
                    for c, t in zip(h, ("Investment", "Share %",
                                        "Keepp return %", "Keepp vol %", "")):
                        c.markdown(f'<div class="cmphdr">{t}</div>',
                                   unsafe_allow_html=True)
                for ln in mine:
                    jr, jv = cmpm.provider_figures(ln, SRC)
                    via = cmpm.provider_reference(ln, SRC)
                    c1, c2, c3, c4, c5 = st.columns(
                        [1.75, .8, .95, .95, .26], vertical_alignment="center")
                    c1.markdown(
                        f'<div style="font-weight:650;font-size:.93rem">'
                        f'{cmpm.label_of(ln, SRC)}</div><div class="d2" '
                        f'style="margin-left:0">{PROV}: {jr:.2f}% · '
                        f'vol {jv:.2f}%'
                        + (f' <i>(via {via})</i>' if via != ln else "")
                        + '</div>', unsafe_allow_html=True)
                    wv = c2.number_input(
                        f"Share of {ln}", 0.0, 100.0, value=float(W[ln]),
                        step=0.5, format="%.2f", key=f"kw_{ln}_{NONCE}",
                        label_visibility="collapsed")
                    rv = c3.number_input(
                        f"Keepp return for {ln}", -20.0, 60.0,
                        value=float(R[ln]), step=0.1, format="%.2f",
                        key=f"kr_{ln}_{NONCE}", label_visibility="collapsed",
                        help=f"Compound return per year. {PROV}: {jr:.2f}%.")
                    vv = c4.number_input(
                        f"Keepp vol for {ln}", 0.0, 100.0, value=float(V[ln]),
                        step=0.1, format="%.2f", key=f"kv_{ln}_{NONCE}",
                        label_visibility="collapsed",
                        help=f"Yearly volatility. {PROV}: {jv:.2f}%.")
                    if c5.button("✕", key=f"krm_{ln}_{NONCE}",
                                 help="Take this asset off the list"):
                        remove = ln
                    new_w[ln], new_r[ln], new_v[ln] = (float(wv or 0),
                                                       float(rv), float(vv))
            rest = [o for o in opts if cmpm.resolve(o, SRC) not in mine]
            pick = st.selectbox(
                f"Add from {cat}", ["— add an asset —"] + rest,
                key=f"kadd_{key}_{NONCE}", label_visibility="collapsed",
                help=f"{SRC_NAME} {cat} lines"
                     + (", plus Keepp's personal private-markets lines"
                        if key == pl.CATEGORY else "")
                     + f". It arrives at 0% on {PROV}'s figures.")
            if pick != "— add an asset —":
                add_line = cmpm.resolve(pick, SRC)

    if remove is not None:
        lines = [x for x in lines if x != remove]
    if add_line is not None and add_line not in lines:
        lines = lines + [add_line]
        new_w[add_line] = 0.0
        new_r[add_line], new_v[add_line] = cmpm.provider_figures(add_line, SRC)
    alloc = cmpm.to_alloc(cmpm.build_frame(lines, new_w, new_r, new_v, SRC),
                          "keepp", SRC)
    st.session_state.custom_alloc = alloc
    if remove is not None or add_line is not None:
        bump_editors()
        st.rerun()
    if not lines:
        st.info("**No assets yet.** Open an asset class above and pick one "
                "from its dropdown to start building the portfolio.",
                icon=":material/add_circle:")

    personal_liquidity_editor(alloc)

    st.write("")
    b1, b2, b3 = st.columns([1.3, 1.3, 0.9])
    if lines and b1.button(f"↺ Keepp figures = {PROV}", key="k_eq_jpm",
                           help=f"Put every Keepp return and vol back to "
                                f"{PROV}'s, keeping the shares."):
        st.session_state.custom_alloc = cmpm.to_alloc(
            cmpm.build_frame(lines, new_w, {}, {}, SRC), "keepp", SRC)
        bump_editors()
        st.rerun()
    if b2.button("↺ Reset to Keepp defaults", key="k_reset",
                 help="Back to Keepp's deck sleeves, with Keepp's own return "
                      "and volatility figures."):
        st.session_state.custom_alloc = custom_alloc_seed()
        bump_editors()
        st.rerun()
    if lines and b3.button("Clear all assets", key="k_clear"):
        st.session_state.custom_alloc = custom_alloc_seed().iloc[0:0]
        bump_editors()
        st.rerun()
    return alloc


def render_comparison(frame):
    """Keepp vs provider results for a 100% portfolio. Returns the two
    engine Results (Keepp, provider) so Save can store both."""
    # ---- the two runs: same weights, correlations, seed and paths --------
    with st.spinner("Simulating both views…"):
        wk, rk, vk, ck = cmpm.sim_inputs(frame, "keepp", SRC)
        wj, rj, vj, cj = cmpm.sim_inputs(frame, "provider", SRC)
        res_k, rep_k = run_sim(wk, rk, vk, ck, int(paths), int(years), int(seed))
        res_j, rep_j = run_sim(wj, rj, vj, cj, int(paths), int(years), int(seed))
    if rep_k or rep_j:
        st.warning("The correlation sub-matrix needed a numerical repair "
                   "before simulating — worth reporting if you see it.")
    hk, hj = cmpm.headline(res_k, amount), cmpm.headline(res_j, amount)
    T_c = res_k.rows[-1]["_years"]

    st.divider()
    section(f"In {T_c} years, {money(amount)} could be worth about",
            "The typical outcome under each set of assumptions — half of the "
            "simulated futures end higher, half lower.")
    a1, a2 = st.columns(2, gap="medium")
    for col, colr, title, h in ((a1, KEEPP_C, "Keepp assumptions", hk),
                                (a2, JPM_C, PROV_TITLE, hj)):
        col.markdown(
            f'<div class="cmphead" style="--c:{colr}"><div class="t">{title}'
            f'</div><div class="v">{money(h["Typical value"])}</div>'
            f'<div class="s"><b>{pct(h["Typical return / yr"], 2)}</b> a year '
            f'typical · volatility <b>{pct(h["Volatility"], 2)}</b><br>9 in 10 '
            f'futures end between <b>{money(h["Weak markets (1 in 20)"])}</b> '
            f'and <b>{money(h["Strong markets (1 in 20)"])}</b>.</div></div>',
            unsafe_allow_html=True)
    d_val = hk["Typical value"] - hj["Typical value"]
    st.markdown(
        f'<div class="muted" style="margin-top:.6rem">Keepp\'s assumptions put '
        f'the typical outcome {cmp_diff("money", d_val)} '
        f'({cmp_diff("pct", hk["Typical return / yr"] - hj["Typical return / yr"])}'
        f' a year) against {PROV}\'s.</div>', unsafe_allow_html=True)

    section("Side by side")
    rows = []
    for k in hk:
        kind = "money" if k in cmpm.MONEY_ROWS else "pct"
        f = money if kind == "money" else (lambda x: pct(x, 2))
        rows.append(f'<tr><td>{k}</td><td class="k">{f(hk[k])}</td>'
                    f'<td class="j">{f(hj[k])}</td>'
                    f'<td>{cmp_diff(kind, hk[k] - hj[k])}</td></tr>')
    st.markdown(
        '<table class="cmpt"><tr><th>After '
        f'{T_c} years</th><th>Keepp</th><th>{PROV}</th><th>Difference'
        '</th></tr>' + "".join(rows) + '</table>', unsafe_allow_html=True)
    st.caption(f"Difference = Keepp minus {PROV}. ▲/▼ say which way, not "
               "which is better. pp = percentage points.")

    section("How the two paths compare, year by year",
            "Solid line: the typical path. Dashed lines and shading: the range "
            "9 in 10 futures fall within (weak to strong markets). Hover to read "
            "any year.")
    st.altair_chart(cmp_chart(cmpm.fan_long(res_k, res_j, amount), amount),
                    width="stretch")
    st.markdown(
        f'<div class="keys">'
        f'<span><i style="background:{KEEPP_C};height:3px"></i>Keepp</span>'
        f'<span><i style="background:{JPM_C};height:3px"></i>{PROV}</span>'
        f'<span><i style="border-top:2px dashed {INK_2};height:0"></i>'
        f'Weak / strong markets (1 in 20)</span>'
        f'<span><i style="border-top:2px dashed {INK_3};height:0"></i>'
        f'What you put in</span></div>', unsafe_allow_html=True)

    section("Where the views differ",
            "Each holding's figures under both sets of assumptions. "
            "<i>Return effect</i> is share × return difference: how much "
            "that holding moves the portfolio's average return.")
    att = cmpm.attribution(frame)
    show = pd.DataFrame({
        "Investment": att["Line"].map(lambda n: cmpm.label_of(n, SRC)),
        "Share %": att["Weight %"],
        f"{PROV} return %": att["Provider return %"],
        "Keepp return %": att["Keepp return %"],
        f"{PROV} vol %": att["Provider vol %"],
        "Keepp vol %": att["Keepp vol %"],
        "Return effect pp": att["Return effect pp"],
    })
    num = {c: st.column_config.NumberColumn(format="%.2f")
           for c in show.columns[1:]}
    num["Return effect pp"] = st.column_config.NumberColumn(format="%+.2f")
    num["Investment"] = st.column_config.TextColumn(width="medium")
    st.dataframe(show, hide_index=True, width="stretch", column_config=num)
    section("At every horizon")
    hz = cmpm.horizons(res_k, res_j, amount, PROV)
    st.dataframe(
        pd.DataFrame({
            "Horizon": hz["Horizon"],
            "Keepp": hz["Keepp typical value"].map(money),
            PROV: hz[f"{PROV} typical value"].map(money),
            "Keepp / yr": hz["Keepp typical return / yr"].map(
                lambda x: pct(x, 2)),
            f"{PROV} / yr": hz[f"{PROV} typical return / yr"].map(
                lambda x: pct(x, 2)),
            "Keepp chance of loss": hz["Keepp chance of loss"].map(
                lambda x: pct(x, 1)),
            f"{PROV} chance of loss": hz[f"{PROV} chance of loss"].map(
                lambda x: pct(x, 1)),
        }), hide_index=True, width="stretch")
    st.caption("Typical (median) value, the yearly return it implies, and the "
               "share of futures that end below what was put in.")

    summ = cmpm.summary(res_k, res_j, amount, PROV)
    meta = {"Amount (USD)": amount, "Years": int(years),
            "Simulated futures": int(paths), "Random seed": int(seed),
            "Benchmark assumptions": SRC_NAME,
            "Correlations": f"{PROV} (both views)",
            "Returns": "Compound (geometric), % per year"}
    st.download_button(
        "Download comparison (Excel)",
        cmpm.to_excel_bytes(frame, summ, hz, meta, PROV),
        f"keepp_vs_{'blackrock' if IS_BR else 'jpm'}_comparison.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        icon=":material/download:", key="cmp_dl")
    st.markdown(
        '<div class="note" style="margin-top:1rem">Both results use the same '
        f'weights, {PROV}\'s correlations and the <b>same '
        'simulated markets</b> (same random seed), so the gap between them comes '
        'from the return and volatility assumptions alone. Figures are '
        'simulations before fees, taxes and inflation, not guarantees.</div>',
        unsafe_allow_html=True)
    return res_k, res_j


with tab_mix:
    mix_left, mix_right = st.columns([1.95, 1], gap="large")

    with mix_left:
        if is_base:
            section("What's in your portfolio",
                    f"Open an asset class, pick an investment from its "
                    "dropdown, then set its share with the slider. All "
                    f"{N_LINES} {SRC_SHORT} asset classes are "
                    "available; "
                    f"their expected return and risk are fixed to {SRC_NAME} "
                    "and cannot be edited here. The <b>tick</b> says "
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
                          if SRC.category_of(ln) == key and live[ln] > 0]
            ROW_COLOR = color_map([plain_name(SRC.LABEL[ln])[0]
                                   for ln in held_order])

            new_w, new_t, remove, add_line = {}, {}, None, None
            for key in CATEGORY_KEYS:
                cat = SRC.CATEGORY_NAME[key]
                mine = [ln for ln in lines if SRC.category_of(ln) == key]
                held = [ln for ln in mine if live[ln] > 0]
                c_sum = sum(live[ln] for ln in mine)
                n_tick = sum(1 for ln in mine
                             if t_state.get(ln, False) or live[ln] > 0)
                pool = len(SRC.CHOICES[key])
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
                    with st.container():           # rows: varying count
                        for ln in mine:
                            nm, desc = plain_name(SRC.LABEL[ln])
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

                    rest = [SRC.LABEL[n] for n, *_ , c in SRC.LINES
                            if c == key and n not in mine]
                    pick = st.selectbox(
                        f"Add from {cat}", ["— add an asset —"] + rest,
                        key=f"add_{key}_{NONCE}", label_visibility="collapsed",
                        help=f"Any of the {pool} {SRC_SHORT} {cat} lines. It "
                             f"arrives at 0%, already ticked for the optimizer.")
                    if pick != "— add an asset —":
                        add_line = SRC.resolve(pick)

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
                    f"to drop it. All {N_LINES} {SRC_SHORT} lines "
                    f"are available from the dropdowns, each on its published "
                    f"return and volatility.")

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
            # Custom: the two-view builder, on whichever provider the
            # sidebar selects (J.P. Morgan or BlackRock).
            alloc = custom_builder()
            pool_alloc = alloc            # Custom has no separate tick list

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
        # One fixed slot for the status box. It is one element at 100% but
        # up to four when the total is off (box, spacer, Scale button,
        # warning); left loose, that moved the donut below to a new position
        # on every change, and the old donut lingered underneath the new one
        # until the run finished. Same rule wherever a block varies in size.
        with st.container():
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
        with st.container():
            if ch is not None:
                st.altair_chart(ch, width='content')
                st.markdown(legend_html(hdf), unsafe_allow_html=True)
        region_panel(alloc, "mix")
        liquidity_panel(alloc, "mix", table=True)

    # Custom mode: the two views side by side, before Save.
    CMP_ACTIVE = not is_base
    CMP_RES = None
    if CMP_ACTIVE and not empty and not off:
        CMP_FRAME = cmpm.frame_from_alloc(alloc, SRC)
        CMP_RES = render_comparison(CMP_FRAME)

    # Save: the only way a run reaches Saved scenarios. The click is acted on
    # after the simulation below, which is what produces the figures saved.
    # In the two-view builder it stores BOTH scenarios.
    st.divider()
    sv1, sv2 = st.columns([2.4, 1], vertical_alignment="bottom")
    sv1.text_input("Scenario name (optional)", key=f"save_name_{NONCE}",
                   max_chars=200, disabled=off,
                   placeholder="Leave blank for an automatic name")
    SAVE_CLICKED = sv2.button(
        "Save both scenarios" if CMP_ACTIVE else "Save scenario",
        type="primary", width='stretch', disabled=off,
        icon=":material/save:",
        help=("The mix must add up to 100% to be saved." if off else
              f"Saves two scenarios: this mix on Keepp's figures and on "
              f"{PROV}'s. Each can be restored from Saved scenarios."
              if CMP_ACTIVE else
              "Save this mix and its results to Saved scenarios."))

# --------------------------------------------------------------------------
# Saved runs panel — runs stored in the database with Save scenario.
# Defined here (before the 100% check) so it can also be shown when the
# current mix is incomplete, e.g. right after signing in to an empty
# Standard portfolio, which is exactly when someone wants an old run back.
# --------------------------------------------------------------------------
def _rename_run(run_id, key):
    if DB.rename_run(UID, run_id, st.session_state.get(key, "")):
        st.toast("Run renamed.", icon=":material/check:")


def numbered_runs():
    """The user's saved runs, oldest first, numbered 1..N (see rs.renumber)."""
    stored = DB.list_runs(UID)
    return [replace(x, record=r)
            for x, r in zip(stored, rs.renumber([x.record for x in stored]))]


VIEW_ALL, VIEW_KEEPP, VIEW_JPM, VIEW_BR = ("All", "Keepp (Custom)",
                                           "J.P. Morgan", "BlackRock")
SORT_NEW, SORT_OLD, SORT_HI, SORT_LO = ("Newest", "Oldest", "Return ↓",
                                        "Return ↑")
# Saved-scenarios table, left to right. Columns not named here (Assumptions,
# Years) follow at the end.
RUNS_COL_ORDER = ["On screen", "#", "View", "Label", "Return %",
                  "Volatility %", "P(loss) %", "Median wealth", "Saved",
                  "Sleeves", "Mode", "Paths", "Seed", "Assumptions", "Years"]
# The headers the table shows — and the Excel Summary sheet repeats.
RUNS_HEADERS = {"On screen": "Current", "Label": "Scenario",
                "Return %": "Avg return %", "P(loss) %": "Chance of loss %",
                "Median wealth": "Typical ×", "Sleeves": "Holdings",
                "Paths": "Futures"}


def runs_view_frame(hist_df: pd.DataFrame) -> pd.DataFrame:
    """The Saved scenarios table as the page shows it: same columns, order
    and headers. Used for the Excel Summary sheet."""
    cols = [c for c in RUNS_COL_ORDER if c in hist_df.columns]
    return hist_df[cols].rename(columns=RUNS_HEADERS).reset_index(drop=True)


def run_src(rec):
    """The assumptions object a saved run was made on: J.P. Morgan, or
    BlackRock for the run's own scenario and horizon."""
    sid = str(getattr(rec, "source", JPM_KEY) or JPM_KEY)
    if sid.startswith(BR_KEY):
        return source_for(BR_KEY, brc.parse_source_id(sid), rec.years)
    return jpm


def run_view(rec):
    """(view, family, label): view is 'keepp' or 'provider' (derived from the
    figures, see compare.classify), family 'jpm' or 'blackrock'."""
    fam = BR_KEY if str(rec.source).startswith(BR_KEY) else JPM_KEY
    try:
        view = cmpm.classify(rec, run_src(rec))
    except Exception:
        view = "keepp"
    label = ("Keepp (Custom)" if view == "keepp"
             else rs.source_label(rec.source))
    return view, fam, label


def run_pair_key(rec):
    try:
        return cmpm.pair_key(rec, run_src(rec))
    except Exception:
        return ("unpaired", rec.seq)


def view_matches(rec, flt):
    if flt in (None, VIEW_ALL):
        return True
    view, fam, _ = run_view(rec)
    if flt == VIEW_KEEPP:
        return view == "keepp"
    return view == "provider" and fam == (JPM_KEY if flt == VIEW_JPM
                                          else BR_KEY)


def provider_side_result(rec):
    """Simulate a Keepp run's portfolio on its provider's figures (same
    holdings, weights, horizon, futures and seed), for when that view was
    never saved. Cached by run_sim."""
    src = run_src(rec)
    frame = cmpm.frame_from_alloc(pd.DataFrame(rec.sleeves), src)
    return run_sim(*cmpm.sim_inputs(frame, "provider", src), int(rec.paths),
                   int(rec.years), int(rec.seed))[0]


def run_base_label(rec):
    """Strip saved-label suffixes to get the base name for grouping runs."""
    lbl = rec.label
    if lbl.endswith(" · Keepp assumptions"):
        return lbl[: -len(" · Keepp assumptions")]
    sl = rs.source_label(rec.source)
    suffix = f" · {sl}"
    if lbl.endswith(suffix):
        return lbl[: -len(suffix)]
    return lbl


def comparison_rows(picked, stored):
    """One row per portfolio group among `picked`.

    Groups runs by (base label, years): same name + same horizon = same
    portfolio regardless of assumptions. Within each group keepp / jpm / br
    slots are filled by run_view. Shared by the on-page table and the Excel
    Summary sheet."""
    views_cache = {x.id: run_view(x.record) for x in stored}

    # Group every stored run by (base_label, years)
    groups = {}   # key -> {"keepp": run, "jpm": run, "br": run, "seq": int}
    for x in stored:
        v, fam, _ = views_cache[x.id]
        key = (run_base_label(x.record), int(x.record.years))
        if key not in groups:
            groups[key] = {"keepp": None, "jpm": None, "br": None,
                           "seq": x.record.seq}
        slot = "keepp" if v == "keepp" else ("br" if fam == BR_KEY else "jpm")
        if groups[key][slot] is None:
            groups[key][slot] = x

    # Output order from picked runs, deduplicated by group key
    order, seen = [], set()
    for x in picked:
        key = (run_base_label(x.record), int(x.record.years))
        if key not in seen:
            seen.add(key)
            order.append(key)

    out = []
    for key in order:
        g = groups.get(key, {"keepp": None, "jpm": None, "br": None, "seq": 0})
        base_lbl, years = key
        mk = cmpm.run_metrics(g["keepp"].record) if g.get("keepp") else None
        mj = cmpm.run_metrics(g["jpm"].record) if g.get("jpm") else None
        mb = cmpm.run_metrics(g["br"].record) if g.get("br") else None
        out.append({
            "name": base_lbl,
            "seq": g.get("seq", 0),
            "years": years,
            "keepp": mk,
            "jpm": mj,
            "br": mb,
            "keepp_run": (f'#{g["keepp"].record.seq}' if g.get("keepp")
                          else "none"),
            "jpm_run": (f'#{g["jpm"].record.seq}' if g.get("jpm")
                        else "none"),
            "br_run": (f'#{g["br"].record.seq}' if g.get("br")
                       else "none"),
        })
    return out


CMP_METRICS = [("Average return / yr", "pct"), ("Typical return / yr", "pct"),
               ("Volatility", "pct")]


def comparison_excel_frame(rows):
    """Keepp / J.P. Morgan / BlackRock comparison table for Excel export.
    Two-level headers: (metric, view). Returns in %."""
    VIEWS = [("Keepp", "keepp"), ("J.P. Morgan", "jpm"), ("BlackRock", "br")]
    data = []
    for r in rows:
        d = {("Scenario", ""): r["name"], ("Years", ""): r["years"],
             ("Keepp run", ""): r["keepp_run"], ("JPM run", ""): r["jpm_run"],
             ("BR run", ""): r["br_run"]}
        for m, _ in CMP_METRICS:
            g = f"{m} (%)"
            for vlabel, vkey in VIEWS:
                md = r.get(vkey)
                d[(g, vlabel)] = md[m] * 100.0 if md else None
        data.append(d)
    df = pd.DataFrame(data)
    if len(df):
        df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def views_comparison(picked, stored):
    """Keepp vs provider returns for the selected scenarios (page)."""
    crow = comparison_rows(picked, stored)
    cmp_css()
    section("Keepp vs provider returns",
            "For each selected scenario: return figures on Keepp's assumptions, "
            "J.P. Morgan's, and BlackRock's, matched by scenario name.")
    head = ('<tr><th rowspan="2">Scenario</th>'
            '<th colspan="2" style="text-align:center">Keepp</th>'
            '<th colspan="2" style="text-align:center">J.P. Morgan</th>'
            '<th colspan="2" style="text-align:center">BlackRock</th>'
            '<th rowspan="2" style="text-align:center">Volatility<br>'
            '<small style="font-weight:400">(J.P. Morgan)</small></th>'
            '</tr><tr>'
            '<th>Avg / yr</th><th>Compounded</th>'
            '<th>Avg / yr</th><th>Compounded</th>'
            '<th>Avg / yr</th><th>Compounded</th>'
            '</tr>')
    body, chart_rows = [], []
    for r in crow:
        mk, mj, mb = r["keepp"], r["jpm"], r["br"]
        name = r["name"]

        def _c(d, m, css):
            v = ('<span class="muted">—</span>' if d is None
                 else pct(d[m], 2))
            return f'<td class="{css}">{v}</td>'

        vol = (pct(mj["Volatility"], 2) if mj
               else '<span class="muted">—</span>')
        body.append(
            f'<tr><td><b>{name}</b><div class="muted" style="font-size:.76rem">'
            f'{r["years"]}y · Keepp {r["keepp_run"]} · JPM {r["jpm_run"]}'
            f' · BR {r["br_run"]}'
            f'</div></td>'
            f'{_c(mk, "Average return / yr", "k")}'
            f'{_c(mk, "Typical return / yr", "k")}'
            f'{_c(mj, "Average return / yr", "j")}'
            f'{_c(mj, "Typical return / yr", "j")}'
            f'{_c(mb, "Average return / yr", "b")}'
            f'{_c(mb, "Typical return / yr", "b")}'
            f'<td>{vol}</td></tr>'
        )
        for who, d in (("Keepp", mk), ("J.P. Morgan", mj), ("BlackRock", mb)):
            if d:
                chart_rows.append({"Scenario": f"{r['seq']}. {name}",
                                   "Who": who,
                                   "r": d["Average return / yr"],
                                   "r_l": pct(d["Average return / yr"], 2)})
    st.markdown('<div style="overflow-x:auto"><table class="cmpt">' + head
                + "".join(body) + '</table></div>', unsafe_allow_html=True)
    st.markdown('<div class="foot" style="margin-top:.4rem">'
                'Volatility shown is J.P. Morgan\'s. Compounded = median '
                'simulated return / yr.</div>', unsafe_allow_html=True)

    if chart_rows:
        cdf = pd.DataFrame(chart_rows)
        sort = list(dict.fromkeys(cdf["Scenario"]))
        span = (cdf.groupby("Scenario")["r"].agg(["min", "max"])
                .reindex(sort).reset_index())
        cs = alt.Scale(domain=["Keepp", "J.P. Morgan", "BlackRock"],
                       range=[KEEPP_C, JPM_C, BR_C])
        lo, hi = float(cdf["r"].min()), float(cdf["r"].max())
        pad = max((hi - lo) * 0.15, 0.002)
        xs = alt.Scale(domain=[lo - pad, hi + pad], nice=False)
        ys = alt.Scale(domain=sort)
        y = alt.Y("Scenario:N", scale=ys, title=None,
                  axis=alt.Axis(labelLimit=280, labelColor=INK, domain=False,
                                ticks=False, labelFontSize=12))
        rule = alt.Chart(span).mark_rule(color=INK_3, strokeWidth=2,
                                         opacity=.5).encode(
            y=y, x=alt.X("min:Q", scale=xs), x2="max:Q")
        dots = alt.Chart(cdf).mark_circle(size=150, opacity=1).encode(
            y=y, x=alt.X("r:Q", title="Average return / yr", scale=xs,
                         axis=alt.Axis(format=".1%", labelColor=INK_3,
                                       gridColor=GRID, titleColor=INK_3,
                                       titleFontWeight=400)),
            color=alt.Color("Who:N", scale=cs, legend=alt.Legend(
                title=None, orient="top", labelColor=INK_2)),
            tooltip=[alt.Tooltip("Scenario:N"),
                     alt.Tooltip("Who:N", title="View"),
                     alt.Tooltip("r_l:N", title="Average return / yr")])
        h_px = max(56 * len(sort), 120)
        st.altair_chart((rule + dots).properties(height=h_px)
                        .configure_view(strokeWidth=0),
                        width="stretch", height=h_px + 70)


def saved_runs_panel(current_digest=None):
    # One slot: the panel's length varies (empty list, Undo row, compare
    # chart), and it sits above other content on the Details tab.
    with st.container():
        _saved_runs_body(current_digest)


def _saved_runs_body(current_digest):
    section("Saved scenarios",
            "A scenario is saved to your account when you press Save scenario "
            "at the bottom of Your mix, so it is still here after a refresh or "
            "on another device. Select rows in the table to restore, rename, "
            "compare or delete them.")
    try:
        stored = numbered_runs()[::-1]                  # newest first
    except Exception as e:                # database blip: keep the results page up
        st.warning("Saved scenarios could not be loaded just now: the database "
                   "did not respond. Nothing is lost; refresh the page in a "
                   "moment.", icon=":material/cloud_off:")
        st.caption(f"{type(e).__name__}: {str(e).splitlines()[0][:300]}")
        return
    all_stored = stored
    # Which runs to list: every view, only Keepp's own (Custom) figures, or
    # only one provider's published figures. The view is derived from each
    # run's figures (compare.classify), so older runs are sorted too.
    f_col, s_col = st.columns([1.35, 1], gap="medium")
    flt = f_col.segmented_control(
        "Show", [VIEW_ALL, VIEW_KEEPP, VIEW_JPM, VIEW_BR], key="runs_filter",
        default=VIEW_ALL, width="stretch",
        help="Keepp (Custom): runs on Keepp's own return / volatility "
             "figures. J.P. Morgan / BlackRock: runs on that provider's "
             "published figures, including Standard-mode runs.") or VIEW_ALL
    # Sort within whatever the filter shows. Return = the average (arithmetic)
    # return per year, the "Avg return %" column.
    srt = s_col.segmented_control(
        "Sort by", [SORT_NEW, SORT_OLD, SORT_HI, SORT_LO], key="runs_sort",
        default=SORT_NEW, width="stretch",
        help="Saved time, or average return per year. Applies to the list "
             "the Show filter selects.") or SORT_NEW
    stored = [x for x in all_stored if view_matches(x.record, flt)]
    if srt == SORT_OLD:
        stored = sorted(stored, key=lambda x: (x.record.created_at, x.id))
    elif srt in (SORT_HI, SORT_LO):
        stored = sorted(stored, key=lambda x: x.record.exp_return_pct,
                        reverse=srt == SORT_HI)
    else:                                       # newest first
        stored = sorted(stored, key=lambda x: (x.record.created_at, x.id),
                        reverse=True)
    saved = [x.record for x in stored]
    # The key changes whenever a run is added or removed, which clears the
    # selection: a new run lands at the top and would otherwise shift every
    # selected row onto a different run. The filter is part of it for the
    # same reason.
    tbl_key = (f"runs_tbl_{st.session_state.get('runs_tbl_v', 0)}_"
               f"{len(stored)}_{stored[0].id if stored else 0}_{flt}_{srt}")
    sel_state = st.session_state.get(tbl_key)
    sel_rows = []
    if sel_state is not None:
        try:
            sel_rows = [i for i in sel_state.selection.rows if i < len(stored)]
        except AttributeError:
            sel_rows = [i for i in sel_state["selection"]["rows"]
                        if i < len(stored)]
    picked = [stored[i] for i in sel_rows]
    one = picked[0] if len(picked) == 1 else None

    def _reset_selection():
        st.session_state.runs_tbl_v = st.session_state.get("runs_tbl_v", 0) + 1

    sc1, sc2, sc3 = st.columns([2.4, 1, 1], vertical_alignment="bottom")
    if one is not None:
        rkey = f"rename_{one.id}_{NONCE}"
        sc1.text_input("Name of the selected run (press Enter to rename)",
                       value=one.record.label, key=rkey, max_chars=200,
                       on_change=_rename_run, args=(one.id, rkey))
    else:
        sc1.text_input("Name of the selected run (press Enter to rename)",
                       value="", disabled=True, key=f"rename_none_{NONCE}",
                       placeholder=("Select one run in the table to rename it"
                                    if stored else "No runs saved yet"))
    if sc2.button("Restore run", width='stretch', type="primary",
                  disabled=one is None, icon=":material/history:",
                  help="Load the selected run's holdings, figures and "
                       "settings back into the app."):
        st.session_state.pending_restore = one
        _reset_selection()
        st.rerun()
    n_del = len(picked)
    if sc3.button(f"Delete ({n_del})" if n_del else "Delete", width='stretch',
                  disabled=not picked, icon=":material/delete:"):
        st.session_state._undo_ids = DB.delete_runs(UID, [x.id for x in picked])
        _reset_selection()
        st.rerun()

    with st.container():                   # Undo row: only after a delete
        undo = st.session_state.get("_undo_ids")
        if undo:
            u1, _, u2 = st.columns([2.4, 1, 1], vertical_alignment="center")
            u1.markdown(f'<div class="muted">Deleted {len(undo)} run'
                        f'{"s" if len(undo) != 1 else ""}.</div>',
                        unsafe_allow_html=True)
            if u2.button("Undo", width='stretch', icon=":material/undo:"):
                n = DB.undelete_runs(UID, undo)
                st.session_state._undo_ids = None
                if n < len(undo):
                    st.toast("Runs that were re-run after deleting are already "
                             "back in the list.")
                _reset_selection()
                st.rerun()

    if current_digest is not None:
        cur = next((x for x in all_stored if x.digest == current_digest), None)
        st.markdown(
            f'<div class="muted">The run on screen is saved as '
            f'<b>#{cur.record.seq} · {cur.record.label}</b>.</div>' if cur else
            '<div class="muted">The run on screen is not saved. Press '
            '<i>Save scenario</i> at the bottom of <b>Your mix</b> to add it '
            'to this list.</div>', unsafe_allow_html=True)

    if not saved:
        st.caption("No scenarios saved yet. Build a mix that adds up to 100% "
                   "and press Save scenario at the bottom of Your mix."
                   if not all_stored else
                   f"No saved scenarios in “{flt}”. Pick another filter "
                   "above.")
        return

    hist_df = rs.history_table(saved)
    hist_df.insert(0, "On screen", ["●" if x.digest == current_digest else ""
                                    for x in stored])
    hist_df.insert(2, "View", [run_view(r)[2] for r in saved])
    st.dataframe(
        hist_df, hide_index=True, width='stretch', key=tbl_key,
        on_select="rerun", selection_mode="multi-row",
        column_order=[c for c in RUNS_COL_ORDER if c in hist_df.columns],
        column_config={
            "On screen": st.column_config.TextColumn("", width="small",
                                                     help="The run shown above"),
            "#": st.column_config.NumberColumn(width="small"),
            "Label": st.column_config.TextColumn("Scenario", width="medium"),
            "View": st.column_config.TextColumn(
                "View", width="medium",
                help="Keepp (Custom) = Keepp's own figures; otherwise the "
                     "provider's published figures."),
            "Saved": st.column_config.TextColumn(width="medium"),
            "Paths": st.column_config.NumberColumn("Futures"),
            "Sleeves": st.column_config.NumberColumn("Holdings"),
            "Return %": st.column_config.NumberColumn("Avg return %",
                                                      format="%.2f"),
            "Volatility %": st.column_config.NumberColumn("Volatility %",
                                                          format="%.2f"),
            "Seed": st.column_config.NumberColumn(format="%d"),
            "Median wealth": st.column_config.NumberColumn("Typical ×",
                                                           format="%.3f×"),
            "P(loss) %": st.column_config.NumberColumn("Chance of loss %",
                                                       format="%.2f"),
        })
    if picked:
        views_comparison(picked, all_stored)
    else:
        st.caption("Select one or more scenarios in the table to see their "
                   "Keepp and provider returns side by side.")

    # Export = the table as shown (filter, sort, columns), narrowed to the
    # selected rows if any — so the Summary sheet matches the page row for
    # row. Run sheets follow the same order.
    rows_out = sorted(sel_rows) if picked else list(range(len(stored)))
    export = [stored[i].record for i in rows_out]
    list_df = runs_view_frame(hist_df.iloc[rows_out])
    # Summary sheet = the Keepp vs provider returns table, for the same runs.
    try:
        summary_df = comparison_excel_frame(
            comparison_rows([stored[i] for i in rows_out], all_stored))
    except Exception:                     # never block the download
        summary_df = None
    st.download_button(
        f"Download Excel ({len(export)} {'selected ' if picked else ''}"
        f"scenario{'s' if len(export) != 1 else ''})",
        rs.to_excel_bytes(export, summary=summary_df, scenarios=list_df),
        "keepp_saved_runs.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        icon=":material/download:")


if off:
    for t in (tab_over, tab_opt, tab_det, tab_cf):
        with t:
            if empty:
                st.info("**Start on the Your mix tab.** Pick the investments "
                        "this portfolio holds from the three asset-class "
                        "dropdowns and give each one a share. Results appear "
                        "here as soon as the shares add up to 100%. To pick up "
                        "an earlier run instead, restore it from **Details & "
                        "saved runs**.",
                        icon=":material/arrow_back:")
            else:
                st.warning(
                    f"Your mix adds up to **{wsum:.1f}%**, not 100%. Open "
                    f"**Your mix** to adjust it, or press *Scale everything "
                    f"to 100%* there.", icon=":material/warning:")
    with tab_det:
        saved_runs_panel()
    st.stop()


# ==========================================================================
# Simulation (cached)
# ==========================================================================
w_t, r_t, v_t, corr_full = sleeve_inputs(alloc)
with st.spinner("Simulating markets…"):
    res, repaired = run_sim(w_t, r_t, v_t, tuple(corr_full.flatten()),
                            int(paths), int(years), int(seed))
# Save only on the Save scenario button (Your mix tab). The digest covers
# everything that determines the numbers, so saving a scenario that is
# already in the list adds nothing.
RUN_MODE = "Base" if is_base else "Custom"
RUN_DIGEST = rs.run_digest(RUN_MODE, alloc[rs.ALLOC_COLS].to_dict("records"),
                           paths, years, seed, source=SRC_ID)
if SAVE_CLICKED and CMP_ACTIVE:
    # Two scenarios, both as Custom runs so personal lines restore intact:
    # the Keepp view (what this mix simulates) and the same mix on the
    # provider's figures (J.P. Morgan or BlackRock). Both carry SRC_ID, so a
    # restore brings back the right provider. Identical figures give identical digests, so the
    # second is then reported as already saved rather than stored twice.
    try:
        name = st.session_state.get(f"save_name_{NONCE}", "").strip()
        res_jpm = CMP_RES[1] if CMP_RES else run_sim(
            *cmpm.sim_inputs(CMP_FRAME, "provider", SRC), int(paths), int(years),
            int(seed))[0]
        out = []
        for side, a_df, r_obj, tag in (
                ("keepp", alloc, res, "Keepp"),
                ("provider", cmpm.to_alloc(CMP_FRAME, "provider", SRC),
                 res_jpm, PROV_TITLE)):
            lbl = name if name else f"{int(years)}y"
            dg = rs.run_digest("Custom", a_df[rs.ALLOC_COLS].to_dict("records"),
                               paths, years, seed, source=SRC_ID)
            new = DB.save_run(UID, rs.make_record(0, lbl, "Custom", a_df, paths,
                                                  years, seed, r_obj,
                                                  source=SRC_ID), dg)
            runs = numbered_runs()
            cur = next((x for x in reversed(runs) if x.digest == dg), None)
            t = f"#{cur.record.seq} · {lbl} ({tag})" if cur else tag
            out.append(f"{'Saved' if new else 'Already saved'} {t}")
        st.toast(" · ".join(out) + ".", icon=":material/check:")
    except Exception as e:                # never lose the results over a save
        st.toast(f"These runs could not be saved ({type(e).__name__}). Try "
                 f"again in a moment.", icon=":material/error:")
elif SAVE_CLICKED:
    try:
        name = st.session_state.get(f"save_name_{NONCE}", "")
        new = DB.save_run(UID, rs.make_record(0, name, RUN_MODE, alloc, paths,
                                              years, seed, res, source=SRC_ID),
                          RUN_DIGEST, pool=POOL_LINES if is_base else [])
        runs = numbered_runs()
        cur = next((x for x in reversed(runs) if x.digest == RUN_DIGEST), None)
        tag = f"#{cur.record.seq} · {cur.record.label}" if cur else ""
        st.toast(f"Saved as {tag}." if new else f"Already saved as {tag}.",
                 icon=":material/check:")
    except Exception as e:                # never lose the results over a save
        st.toast(f"This run could not be saved ({type(e).__name__}). Try "
                 f"again in a moment.", icon=":material/error:")

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
    with st.container():
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
            f'<div class="card"><div style="display:flex;'
            f'justify-content:space-between;gap:16px">'
            f'<div><div class="lbl">Risk level</div>'
            f'<div class="val" style="color:{rc}">{R_NAME}</div></div>'
            f'<div style="text-align:right"><div class="lbl">Volatility</div>'
            f'<div class="val">{pct(res.analytic_vol, 2)}</div></div></div>'
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
        region_panel(alloc, "over")
        liquidity_panel(alloc, "over")

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
# A radio option VALUE: kept constant across sources (the count is shown by
# the format_func), so switching source never invalidates the stored choice.
SCOPE_UNIVERSE = "All lines"


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
            else f"All {N_LINES} asset classes"),
        captions=[(f"The {len(pool_alloc)} assets ticked on Your mix"
                   if is_base else "Re-weight what the portfolio already holds"),
                  f"Build the best mix from {SRC_SHORT}'s full catalogue"],
        help=f"All {N_LINES} asset classes: every {SRC_NAME} "
             f"line is a candidate. Lines you already hold keep your own "
             f"return and risk figures; the rest use {SRC_SHORT}'s. Takes about "
             f"1–2 seconds.")
    is_universe = scope == SCOPE_UNIVERSE

    # The ticked pool, not just what is held: a 0% ticked asset is exactly a
    # candidate the optimizer may move into.
    opt_alloc = uni.universe_alloc(alloc, src=SRC) if is_universe else pool_alloc
    opt_lines = [SRC.resolve(x) for x in opt_alloc["LTCMA line"]]
    opt_cats = list(opt_alloc["Category"])
    n_lines = len(opt_lines)
    # Each source names its own cash line ("U.S. Cash" / "US cash"), so the cash
    # floor and the risk-free default follow the selected assumptions.
    cash_pos = [i for i, nm in enumerate(opt_lines) if nm == SRC.CASH_LINE]

    opt_mu, opt_vols = opt.inputs_from_percent(opt_alloc["Return %"],
                                               opt_alloc["Vol %"])
    opt_cov = opt.covariance(opt_vols, pl.corr_for(SRC, opt_alloc["LTCMA line"]))
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
                          if cash_pos else float(SRC.lookup(SRC.CASH_LINE)[0]))
            rf_pct = oc1.number_input(
                "Risk-free rate %", value=round(rf_default, 2),
                min_value=0.0, max_value=20.0, step=0.05, format="%.2f",
                help=f"Defaults to this allocation's own {SRC.CASH_LINE} return, "
                     f"or {SRC_SHORT}'s if you don't hold it.")
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
                      f"Not available across all {N_LINES} lines — it would take minutes."))
            show_stability = show_stability and not is_universe

            st.markdown('<div class="muted" style="margin:.8rem 0 .3rem">'
                        '<b>Geography limits</b> — share of the whole portfolio. '
                        'A cell showing <i>None</i> has no limit — type a number '
                        'to set one, delete it to remove it. Broad indices count '
                        'by their fixed regional split (e.g. AC World ≈ 64% US).'
                        '</div>', unsafe_allow_html=True)
            geo_cur = geo_limits_current()
            st.data_editor(
                st.session_state.geo_base, key="geo_limits", hide_index=True,
                width='stretch', num_rows="fixed", disabled=["Region"],
                column_config={
                    "Region": st.column_config.TextColumn(width="small"),
                    **{c: st.column_config.NumberColumn(
                        c.replace(" %", ""), min_value=0.0, max_value=100.0,
                        step=0.5, format="%.1f") for c in GEO_COLS[1:]}})

            st.markdown('<div class="muted" style="margin:.8rem 0 .3rem">'
                        '<b>Liquidity limits</b> — share of the whole portfolio, '
                        'e.g. at least 50% in L1–L2 (cash within a week), at most '
                        '25% in L4–L5 (gated or locked). Split lines count by '
                        'their share in each tier.</div>', unsafe_allow_html=True)
            liq_cur = liq_limits_current()
            st.data_editor(
                st.session_state.liq_base, key="liq_limits", hide_index=True,
                width='stretch', num_rows="fixed", disabled=["Tiers"],
                column_config={
                    "Tiers": st.column_config.TextColumn("Liquidity tiers",
                                                         width="medium"),
                    "Min %": st.column_config.NumberColumn(
                        "At least %", min_value=0.0, max_value=100.0, step=0.5,
                        format="%.1f"),
                    "Max %": st.column_config.NumberColumn(
                        "At most %", min_value=0.0, max_value=100.0, step=0.5,
                        format="%.1f")})

        lower = np.zeros(n_lines)
        upper = np.full(n_lines, cap_pct / 100.0)
        if cash_pos:
            lower[cash_pos[0]] = floor_pct / 100.0
            upper[cash_pos[0]] = max(upper[cash_pos[0]], floor_pct / 100.0)
        band_map = {r["Category"]: (float(r["Min %"]) / 100.0,
                                    float(r["Max %"]) / 100.0)
                    for _, r in bands_df.iterrows()}
        geo_rows = geo.exposure_rows(SRC, opt_lines, opt_cats, geo_cur)
        liq_rows = liq.exposure_rows(SRC, opt_lines, liq_cur, liq_overrides())
        limits = opt.Limits(lower, upper, groups,
                            {k: band_map.get(k, (0.0, 1.0)) for k in groups},
                            exposure=geo_rows + liq_rows)

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
                tuple((nm, round(lo, 8), round(hi, 8), tuple(np.round(a, 8)))
                      for nm, a, lo, hi in geo_rows + liq_rows),
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

        # The spinner and any error only exist on the run the button was
        # pressed; in their own slot they do not shift the results below.
        with st.container():
            if run_opt:
                # Across the catalogue the frontier is NOT solved up front: at
                # n=59 it is 24 QPs at ~0.6s each, and only the optional map in an
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
            with st.container():
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
            with st.container():                # full catalogue only
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

            # ---- by region ------------------------------------------------
            section("By region",
                    "Share of the whole portfolio in each region, today and "
                    "suggested. Geography limits you set are shown alongside.")
            g_now = geo.breakdown(SRC, opt_lines, opt_cats, list(cur_w * 100))
            g_new = geo.breakdown(SRC, opt_lines, opt_cats, list(w_opt * 100))
            g_tbl = pd.DataFrame({
                "Region": geo.REGIONS,
                "Today %": g_now["Total"].values,
                "Suggested %": g_new["Total"].values,
                "Change (pts)": (g_new["Total"] - g_now["Total"]).values,
                "Stocks today %": g_now["Stocks"].values,
                "Stocks suggested %": g_new["Stocks"].values})
            st.dataframe(g_tbl, hide_index=True, width='stretch',
                         column_config={
                             **{c: st.column_config.NumberColumn(format="%.1f%%")
                                for c in ["Today %", "Suggested %",
                                          "Stocks today %", "Stocks suggested %"]},
                             "Change (pts)": st.column_config.NumberColumn(
                                 format="%+.1f")})
            if geo_rows:
                st.caption("Geography limits applied: " + "; ".join(
                    f"{nm} {lo*100:.0f}–{hi*100:.0f}% → "
                    f"{float(np.asarray(a) @ w_opt) * 100:.1f}%"
                    for nm, a, lo, hi in geo_rows))

            # ---- by liquidity tier ----------------------------------------
            ov = liq_overrides()
            l_now = liq.exposure(SRC, opt_lines, cur_w * 100, ov)
            l_new = liq.exposure(SRC, opt_lines, w_opt * 100, ov)
            section("By liquidity",
                    "How quickly the portfolio could become cash, today and "
                    f"suggested. Suggested: {liq.ladder_sentence(l_new)}")
            l_tbl = pd.DataFrame({
                "Tier": [TIER_NAME[t] for t in liq.TIERS],
                "Time to cash": [liq.TIME_TO_CASH[t] for t in liq.TIERS],
                "Today %": l_now.values,
                "Suggested %": l_new.values,
                "Change (pts)": (l_new - l_now).values})
            st.dataframe(l_tbl, hide_index=True, width='stretch',
                         column_config={
                             **{c: st.column_config.NumberColumn(format="%.1f%%")
                                for c in ["Today %", "Suggested %"]},
                             "Change (pts)": st.column_config.NumberColumn(
                                 format="%+.1f")})
            d_now = liq.days_to_cash(SRC, opt_lines, cur_w, ov)
            d_new = liq.days_to_cash(SRC, opt_lines, w_opt, ov)
            st.caption(f"Weighted-average time to cash: {fmt_days(d_now)} today → "
                       f"{fmt_days(d_new)} suggested."
                       + ("  Liquidity limits applied: " + "; ".join(
                           f"{nm} {lo*100:.0f}–{hi*100:.0f}% → "
                           f"{float(np.asarray(a) @ w_opt) * 100:.1f}%"
                           for nm, a, lo, hi in liq_rows) if liq_rows else ""))

            # ---- what changes ---------------------------------------------
            section("What would change",
                    "Percentage points of the portfolio moved into (blue) or out of "
                    "(orange) each holding.")
            if R.get("universe"):
                chg_names = [plain_name(SRC.LABEL.get(x, x))[0] for x in opt_lines]
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
                        "what": [plain_name(SRC.LABEL.get(x, x))[0] for x in opt_lines]})

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
                        row = {"Holding": plain_name(SRC.LABEL.get(nm, nm))[0],
                               "LTCMA line": SRC.LABEL.get(nm, nm),
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
                                                .TextColumn(f"{SRC_SHORT} line"),
                                                **pct_cfg})
                if p25 is not None:
                    widest = int(np.argmax(p75 - p25))
                    st.caption(
                        f"**Stability:** across 40 re-solves with each expected "
                        f"return nudged by 0.5%, the least stable holding is "
                        f"**{plain_name(SRC.LABEL.get(opt_lines[widest], opt_lines[widest]))[0]}**"
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

    # ---- expected return, two ways --------------------------------------
    # Multiplying weights by the Return column is the obvious hand check and
    # gives a LOWER number than the simulation. Both answers live here, side
    # by side, with the three lines that get from one to the other. The maths,
    # and why the gap can only go this way, is in engine.return_ladder().
    # Collapsed by default: nothing in this tab moves unless it is opened.
    LAD = return_ladder(w_t, r_t, v_t, corr_full)
    with st.expander("Expected return, two ways"):
        c1, c2 = st.columns(2)
        c1.markdown(card("Weighted average", pct(LAD.weighted_simple, 2),
                         "Σ weight × Return %"), unsafe_allow_html=True)
        c2.markdown(card("Monte Carlo return", pct(LAD.compound, 2),
                         f"Simulated {pct(term['Median'], 2)} over {T}y"),
                    unsafe_allow_html=True)
        st.write("")
        c1, c2 = st.columns(2)
        c1.markdown(
            '<div class="srctext">Each holding\'s weight times its '
            'Return %, added up. It is the right answer only if the whole '
            'portfolio is as volatile as the average of its holdings.</div>',
            unsafe_allow_html=True)
        c2.markdown(
            '<div class="srctext">What the money actually compounds at — the '
            'Typical (median) row in the table above. The simulated figure '
            'differs from the arithmetic below only by sampling noise.</div>',
            unsafe_allow_html=True)
        st.write("")
        steps = pd.DataFrame([
            {"Line": "Weighted average",
             "Why": "Σ weight × Return %",
             "Per year": pct(LAD.weighted_simple, 2)},
            {"Line": "plus what each holding averages above its compound rate",
             "Why": "Σ weight × Vol² ÷ 2",
             "Per year": pct(LAD.variance_uplift, 2, sign=True)},
            {"Line": "minus what the portfolio loses to its own swings",
             "Why": f"Vol² ÷ 2, at Vol {pct(LAD.portfolio_vol, 2)}",
             "Per year": pct(0.0 - LAD.variance_drag, 2, sign=True)},
            {"Line": "Monte Carlo return",
             "Why": "what the portfolio compounds at",
             "Per year": pct(LAD.compound, 2)},
        ])
        st.dataframe(steps, hide_index=True, width='stretch',
                     column_config={
                         "Line": st.column_config.TextColumn(width="large"),
                         "Per year": st.column_config.TextColumn(
                             width="small")})
        st.markdown(
            f'<div class="srctext">The middle two lines do not cancel, which '
            f'is the whole difference — <b>{pct(LAD.compound - LAD.weighted_simple, 2, sign=True)}</b> '
            f'here, and it can only ever go this way. Your holdings average '
            f'<b>{pct(LAD.avg_sleeve_vol, 2)}</b> of volatility between them, '
            f'but held together they come to <b>{pct(LAD.portfolio_vol, 2)}</b>, '
            f'so the portfolio loses less to its swings than a weighted '
            f'average assumes. The two meet only when a portfolio holds one '
            f'thing, or holds things of equal volatility that move in '
            f'lockstep.</div>', unsafe_allow_html=True)
        if abs(LAD.weight_sum - 100.0) > 0.05:
            st.caption(f"Weights total {LAD.weight_sum:.2f}%, not 100%. Both "
                       f"figures are on the whole portfolio, so the missing "
                       f"{100 - LAD.weight_sum:+.2f}% earns nothing — which "
                       f"is also how the simulation treats it.")

    st.write("")

    csv = disp.round(4).to_csv(index=False)
    st.download_button("Download table (CSV)", csv,
                       f"keepp_monte_carlo_{'base' if is_base else 'custom'}.csv",
                       "text/csv", icon=":material/download:")

    # ---- saved runs ------------------------------------------------------
    saved_runs_panel(RUN_DIGEST)

    # ---- sources ---------------------------------------------------------
    section("Where the assumptions come from")
    with st.expander(f"{SRC_NAME} · sources and method"):
        for name, mini, text in (BR_SOURCES if IS_BR else SOURCES):
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



# ==========================================================================
# TAB — Contributions & withdrawals
#
# A separate simulation (keepp_mc/cashflows.py) on the same mix, assumptions,
# horizon, number of futures and seed as the tabs above. Nothing above reads
# from it, and its runs are not auto-saved: the main tabs keep the lump-sum
# model exactly as before.
# ==========================================================================
@st.cache_data(show_spinner=False, max_entries=16)
def run_cashflows(w, r, v, corr_flat, amount, flows, paths, years, seed, timing):
    corr = np.array(corr_flat, dtype=float).reshape(len(w), len(w))
    return cfm.simulate_cashflows(list(w), list(r), list(v), corr, amount,
                                  [cfm.Flow(*f) for f in flows], paths, years,
                                  seed, timing)


with tab_cf:
    section("Add money in, or take money out, along the way",
            f"The same {paths:,} futures as the other tabs — same mix, "
            f"assumptions and seed — but with money added or withdrawn each "
            f"month or year. Starts from {money(amount)} (set in the sidebar) "
            f"over {years} year{'s' if years != 1 else ''}. This tab does not "
            f"change any figure on the other tabs.")

    CF_COLS = ["Type", "Amount $", "How often", "From year", "To year",
               "Grows % a year"]
    if "cf_base" not in st.session_state:
        st.session_state.cf_base = pd.DataFrame(
            [{"Type": cfm.WITHDRAWAL, "Amount $": 50_000.0,
              "How often": cfm.YEARLY, "From year": 1, "To year": int(years),
              "Grows % a year": 0.0}], columns=CF_COLS)
    sched = st.data_editor(
        st.session_state.cf_base, key="cf_schedule", hide_index=True,
        num_rows="dynamic", width="stretch",
        column_config={
            "Type": st.column_config.SelectboxColumn(
                options=[cfm.CONTRIBUTION, cfm.WITHDRAWAL], required=True),
            "Amount $": st.column_config.NumberColumn(
                min_value=0.0, step=1_000.0, format="$%,.0f",
                help="Per payment, in today's dollars."),
            "How often": st.column_config.SelectboxColumn(
                options=[cfm.MONTHLY, cfm.YEARLY], required=True),
            "From year": st.column_config.NumberColumn(
                min_value=1, max_value=30, step=1, format="%d"),
            "To year": st.column_config.NumberColumn(
                min_value=1, max_value=30, step=1, format="%d",
                help="Last year with this payment, inclusive."),
            "Grows % a year": st.column_config.NumberColumn(
                min_value=-20.0, max_value=20.0, step=0.1, format="%.1f",
                help="E.g. 2.5 to keep pace with inflation. Applied from "
                     "today: year 2 pays the amount × 1.025."),
        })
    timing = st.radio(
        "When money moves", [cfm.END, cfm.START], horizontal=True,
        key="cf_timing",
        help="End of period: the month's (or year's) return is earned first, "
             "then the money moves. Start of period: the money moves first.")
    st.caption("Add a row with **+**. Rows past the horizon are ignored; select "
               "a row and press Delete to remove it.")

    flows = []
    for _, row in sched.iterrows():
        try:
            amt = float(row["Amount $"] or 0)
            fy, ly = int(row["From year"]), int(row["To year"])
            gr = float(row["Grows % a year"] or 0)
        except (TypeError, ValueError):
            continue
        if amt > 0 and row["Type"] in (cfm.CONTRIBUTION, cfm.WITHDRAWAL) \
                and row["How often"] in (cfm.MONTHLY, cfm.YEARLY):
            flows.append((row["Type"], amt, row["How often"], fy,
                          min(ly, int(years)), gr))

    cw, cr, cv, ccorr = sleeve_inputs(alloc)
    with st.spinner("Simulating with cash flows…"):
        cfr = run_cashflows(cw, cr, cv, tuple(ccorr.flatten()), float(amount),
                            tuple(flows), int(paths), int(years), int(seed),
                            timing)

    lasts = 1.0 - float(cfr.depleted.mean())
    p05, p50, p95 = cfr.value_pct[-1, 0], cfr.value_pct[-1, 2], cfr.value_pct[-1, 4]
    has_out = cfr.paid_out_planned > 0
    if has_out:
        lead = (f'The money lasts all <b>{years} years</b> in '
                f'<b>{lasts*100:.1f}%</b> of futures')
    else:
        lead = f'After <b>{years} years</b> the typical value is'
    st.markdown(
        f'<div class="hero"><div class="q">{lead}</div>'
        f'<div class="big">{money(p50)}<small>typical ending value</small></div>'
        f'<div class="s">In 9 out of 10 futures it ends between '
        f'<b>{money(p05)}</b> and <b>{money(p95)}</b>. With no money moving, '
        f'the same futures end at a typical <b>{money(cfr.no_flow_median)}</b>. '
        f'The investments themselves grow about <b>{pct(cfr.twr_median)}</b> a '
        f'year in the typical future — money moving in or out changes how much '
        f'is invested, not the return on it.</div></div>',
        unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    dead_m = cfr.depletion_month[cfr.depleted]
    k1.markdown(card("Chance the money lasts", pct(lasts),
                     ("No simulated future runs out." if lasts >= 1.0 else
                      f"{one_in(1 - lasts).capitalize()} futures run out "
                      f"within {years} years.") if has_out else
                     "Nothing is withdrawn, so it cannot run out.",
                     hot=has_out and lasts < 0.9), unsafe_allow_html=True)
    k2.markdown(card("If it runs out, typically in",
                     f"year {np.median(dead_m) / 12:.0f}" if len(dead_m) else "—",
                     f"Earliest: year {np.ceil(dead_m.min() / 12):.0f}."
                     if len(dead_m) else "No simulated future runs out."),
                unsafe_allow_html=True)
    k3.markdown(card("Money put in", money(cfr.paid_in),
                     "Planned contributions over the horizon."),
                unsafe_allow_html=True)
    k4.markdown(card("Money taken out", money(cfr.paid_out_planned),
                     f"Planned. Paid on average: {money(cfr.paid_out_mean)} "
                     f"(less when a future runs out)."),
                unsafe_allow_html=True)

    # ---- value over time, in dollars ----------------------------------------
    section("How the money could develop, year by year",
            "Shaded bands: where the value is in half (darker) and 9 in 10 "
            "(lighter) of futures. A future that runs out stays at $0.")
    vf = pd.DataFrame(cfr.value_pct, columns=["p05", "p25", "p50", "p75", "p95"])
    vf["Year"] = cfr.year_marks
    for c_ in ("p05", "p50", "p95"):
        vf[c_ + "_l"] = vf[c_].map(money)
    vax = alt.Axis(title=None, format="$.3~s", labelColor=INK_3, gridColor=GRID,
                   domain=False, ticks=False, tickCount=6, labelFontSize=12)
    xax = alt.Axis(title="Years from today", titleColor=INK_3, labelColor=INK_3,
                   grid=False, tickMinStep=1, format="d", labelFontSize=12,
                   domainColor=GRID, tickColor=GRID, titleFontWeight=400)
    vb = alt.Chart(vf)
    tip = [alt.Tooltip("Year:Q", title="Year"),
           alt.Tooltip("p95_l:N", title="Strong markets"),
           alt.Tooltip("p50_l:N", title="Typical"),
           alt.Tooltip("p05_l:N", title="Weak markets")]
    chart_v = (
        vb.mark_area(color=BAND_OUTER, opacity=.9).encode(
            x=alt.X("Year:Q", axis=xax), y=alt.Y("p05:Q", axis=vax), y2="p95:Q",
            tooltip=tip)
        + vb.mark_area(color=BAND_INNER, opacity=.9).encode(
            x="Year:Q", y="p25:Q", y2="p75:Q", tooltip=tip)
        + vb.mark_line(color=ACCENT_DK, strokeWidth=2.5).encode(
            x="Year:Q", y="p50:Q", tooltip=tip)
        + alt.Chart(pd.DataFrame({"y": [amount]})).mark_rule(
            color=INK_3, strokeDash=[4, 4], opacity=.8).encode(y="y:Q")
    ).properties(height=340).configure_view(strokeWidth=0)
    st.altair_chart(chart_v, width="stretch")

    if has_out:
        section("Chance the money has run out, by year")
        dp = pd.DataFrame({"Year": cfr.year_marks, "p": cfr.p_depleted * 100})
        dp["p_l"] = dp["p"].map(lambda x: f"{x:.1f}%")
        st.altair_chart(
            alt.Chart(dp).mark_area(color=BRICK, opacity=.18, line={"color": BRICK})
            .encode(x=alt.X("Year:Q", axis=xax),
                    y=alt.Y("p:Q", title=None, scale=alt.Scale(domain=[0, max(
                        5, float(dp["p"].max()) * 1.15)]),
                            axis=alt.Axis(format=".0f", labelExpr="datum.label + '%'",
                                          labelColor=INK_3, gridColor=GRID,
                                          domain=False, ticks=False)),
                    tooltip=[alt.Tooltip("Year:Q"),
                             alt.Tooltip("p_l:N", title="Run out by then")])
            .properties(height=200).configure_view(strokeWidth=0),
            width="stretch")

    with st.expander("Year-by-year numbers"):
        tbl = pd.DataFrame({
            "Year": cfr.year_marks,
            "Weak (5th)": cfr.value_pct[:, 0], "25th": cfr.value_pct[:, 1],
            "Typical (median)": cfr.value_pct[:, 2], "75th": cfr.value_pct[:, 3],
            "Strong (95th)": cfr.value_pct[:, 4],
            "Run out by then %": cfr.p_depleted * 100})
        st.dataframe(tbl, hide_index=True, width="stretch", column_config={
            **{c_: st.column_config.NumberColumn(format="$%,.0f")
               for c_ in ["Weak (5th)", "25th", "Typical (median)", "75th",
                          "Strong (95th)"]},
            "Run out by then %": st.column_config.NumberColumn(format="%.1f")})
        st.download_button("Download (CSV)", tbl.round(2).to_csv(index=False),
                           "keepp_cash_flows.csv", "text/csv",
                           icon=":material/download:")

    st.markdown(
        '<div class="foot" style="margin-top:1.2rem">How this tab works: each '
        'month the portfolio earns that month\'s simulated return and the '
        'scheduled money is added or taken out (before or after the return, as '
        'chosen). Withdrawals come out of every holding in proportion — '
        'including private assets, which in reality cannot always be sold. '
        'Amounts are in today\'s dollars and stay fixed unless a growth rate is '
        'set; nothing adjusts spending after a bad year. A future that cannot '
        'pay a withdrawal runs out and stays at $0. Thin-tailed returns make '
        'running out somewhat less likely here than in real markets. Results '
        'on this tab are not saved to Saved scenarios.'
        '</div>', unsafe_allow_html=True)
