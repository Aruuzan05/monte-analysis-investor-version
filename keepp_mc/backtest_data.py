"""
Historical proxy returns for the backtest tab — a port of data.py.

Nine asset lines, monthly USD total returns (decimals), Jan 1997 onwards:

  Line              Proxy (spliced where the live fund starts later)
  Corporate bonds   LQDSIM (simulated LQD, iShares IG corporate)
  Cash              3-month T-bill (FRED DTB3), last month-end yield / 12
  Global equity     VTSIM (simulated VT, Vanguard Total World)
  Private equity    VBRSIM to Nov 2006, then PSP (listed private equity)
  Hedge funds       EDHEC Funds of Funds to Aug 2009, then QAI
  Commodities       S&P GSCI to Jul 2006, then GSG
  Gold              XAU/USD to Nov 2004, then GLD
  Digital assets    BTC-USD from Oct 2014 (blank before)
  High yield        VWEHX, Vanguard High-Yield Corporate Fund

The app never downloads anything: it reads the snapshot in
keepp_mc/data/proxy_returns.csv, so every visitor sees the same numbers and a
page load costs no network calls. To refresh the snapshot run

    python scripts/update_proxy_returns.py

which calls build_returns() below (the data.py logic) and rewrites the file.

One change from data.py: High yield reads VWEHX from the same alm0421 price
repository as the other funds instead of yfinance. Same fund, adjusted close
(dividends reinvested), and it drops the Yahoo dependency, which rate-limits
cloud servers.

Pure pandas. No Streamlit import.
"""

from pathlib import Path

import pandas as pd

START = "1997-01-01"
SNAPSHOT = Path(__file__).parent / "data" / "proxy_returns.csv"

GH = "https://raw.githubusercontent.com"
ALM = f"{GH}/alm0421/alm0421/master/data"

LINES = ["Corporate bonds", "Cash", "Global equity", "Private equity",
         "Hedge funds", "Commodities", "Gold", "Digital assets", "High yield"]

SOURCES = {
    "Corporate bonds": "LQDSIM (simulated iShares IG corporate bond ETF)",
    "Cash": "3-month US T-bill (FRED DTB3)",
    "Global equity": "VTSIM (simulated Vanguard Total World Stock ETF)",
    "Private equity": "VBRSIM to Nov 2006, then PSP (listed private equity ETF)",
    "Hedge funds": "EDHEC Funds of Funds index to Aug 2009, then QAI",
    "Commodities": "S&P GSCI to Jul 2006, then GSG",
    "Gold": "XAU/USD spot to Nov 2004, then GLD",
    "Digital assets": "Bitcoin (BTC-USD), from Oct 2014",
    "High yield": "VWEHX (Vanguard High-Yield Corporate Fund)",
}


# --------------------------------------------------------------------------
# data.py, as functions
# --------------------------------------------------------------------------
def load_series(url, col, date_col="date"):
    """A CSV column as a date-indexed float Series."""
    df = pd.read_csv(url, na_values=[".", ""])
    s = df.set_index(pd.to_datetime(df[date_col]))[col].astype(float)
    return s.sort_index().dropna()


def monthly_returns(price):
    """Simple month-end to month-end returns."""
    return price.resample("ME").last().pct_change().dropna()


def splice(proxy_ret, live_ret, first_live_month):
    """Proxy returns before the first live month, live returns from it on."""
    cut = pd.Timestamp(first_live_month) + pd.offsets.MonthEnd(0)
    return pd.concat([proxy_ret[proxy_ret.index < cut],
                      live_ret[live_ret.index >= cut]])


def trim(s):
    return s[s.index >= pd.Timestamp(START)]


def build_returns() -> pd.DataFrame:
    """Download every source and return the monthly table (needs internet)."""
    price = lambda t: load_series(f"{ALM}/prices/{t}.csv", "adj_close")

    corp = trim(monthly_returns(price("LQDSIM")))

    dtb3 = load_series(f"{ALM}/macro/DTB3.csv", "value")
    cash = trim(dtb3.resample("ME").last().shift(1) / 100 / 12).dropna()

    equity = trim(monthly_returns(price("VTSIM")))

    pe = trim(splice(monthly_returns(price("VBRSIM")),
                     monthly_returns(price("PSP")), "2006-12"))

    edhec = pd.read_csv(
        f"{GH}/braverock/PerformanceAnalytics/master/data/edhec.csv", index_col=0)
    edhec.index = pd.to_datetime(edhec.index) + pd.offsets.MonthEnd(0)
    hf = trim(splice(edhec["Funds of Funds"].astype(float),
                     monthly_returns(price("QAI")), "2009-09"))

    gsci = load_series(f"{GH}/giacomomaggiore/margin-investing-etfs/master/"
                       f"data/%5ESPGSCI.csv", "adj close")
    comm = trim(splice(monthly_returns(gsci), monthly_returns(price("GSG")),
                       "2006-08"))

    xau = load_series(f"{GH}/tomas-charvat/Stock_market_analysis/master/"
                      f"Line_charts/xauusd_d.csv", "Close", "Date")
    gold = trim(splice(monthly_returns(xau), monthly_returns(price("GLD")),
                       "2004-12"))

    btc = trim(monthly_returns(price("BTC-USD")))
    hy = trim(monthly_returns(price("VWEHX")))

    out = pd.concat([corp, cash, equity, pe, hf, comm, gold, btc, hy], axis=1)
    out.columns = LINES
    # drop the month in progress: its "return" is only part of a month
    out = out[out.index < pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(0)]
    # a line that has started must not stop before the others end
    out = out.loc[: out.dropna(subset=[c for c in LINES if c != "Digital assets"]
                               ).index.max()]
    out.index.name = "date"
    return out


# --------------------------------------------------------------------------
# What the app reads
# --------------------------------------------------------------------------
def load_returns(path: Path = SNAPSHOT) -> pd.DataFrame:
    R = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return R[LINES].astype(float)


def coverage(R: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Proxy": [SOURCES[c] for c in R.columns],
        "From": [R[c].first_valid_index().strftime("%b %Y") for c in R.columns],
        "To": [R[c].last_valid_index().strftime("%b %Y") for c in R.columns],
        "Months": [int(R[c].notna().sum()) for c in R.columns],
    }, index=R.columns)
