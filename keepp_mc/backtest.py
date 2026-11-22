"""
Historical backtest and crisis stress test of a mix — the keepp_backtest.py
logic, trimmed to what the "Backtest & stress" tab shows.

What it answers: had this mix been held since 1997, rebalanced to its weights
every month, how would it have done, and how would it have come through the
five crises below? It uses only the WEIGHTS. Return, volatility and
correlation assumptions play no part: history supplies all of them, so a
Keepp run and the J.P. Morgan run of the same holdings backtest identically.

Step A  map every simulator line to one of nine historical proxy lines
        (PROXY_OF below), so the mix becomes nine weights w_i summing to 1
Step B  monthly portfolio return, rebalanced to target every month
            r_p(t) = sum_i w_eff(t,i) r(t,i)
        A line with no data in month t (Digital assets before Oct 2014) gets
        weight 0 and the others are scaled up pro rata, so the mix stays 100%
        invested:  w_eff(t,i) = w_i a(t,i) / sum_j w_j a(t,j),  a = 1 if data
Step C  value V_t = V_{t-1}(1 + r_p(t)), V_0 = 1;  drawdown DD_t = V_t / max V - 1
Step D  whole-period metrics (CAGR, volatility, Sharpe vs cash, max drawdown,
        worst 12 months, years breaking the -12% in-year drawdown limit)
Step E  crisis replay: buy at the first month-end of each window, measure
        through the second. Each line's contribution
            c_i = sum_t V_{t-1} w_eff(t,i) r(t,i)
        adds up exactly to the crisis return (checked).

Pure numpy / pandas. No Streamlit import.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import backtest_data as bd
from . import personal_lines as pl

LINES = bd.LINES
CASH_LINE = "Cash"
DRAWDOWN_LIMIT = -0.12      # KEEPP deck: maximum drawdown -12% a year

# (peak month-end, trough month-end): bought at the close of the first date,
# measured over the months in the half-open interval (start, end].
CRISES = {
    "Asian crisis 1997-98": ("1997-06-30", "1998-08-31"),
    "Dot-com bust 2000-02": ("2000-03-31", "2002-09-30"),
    "Global financial crisis 2007-09": ("2007-10-31", "2009-02-28"),
    "COVID-19 crash 2020": ("2019-12-31", "2020-03-31"),
    "2022 war and rate shock": ("2021-12-31", "2022-09-30"),
}

# --------------------------------------------------------------------------
# Step A — simulator line -> proxy line(s)
# --------------------------------------------------------------------------
# Keyed by J.P. Morgan line name. BlackRock lines go through their correlation
# proxy (blackrock_cma.CORR_PROXY) and Keepp's personal lines through theirs
# (personal_lines.PROXY), so every line in the app lands here.
#
# Nine proxies cannot represent 59 lines exactly. The rule:
#   all investment-grade, government and agency bonds -> Corporate bonds (LQD)
#   spread-heavy credit (HY, loans, EM debt, muni HY, direct lending) -> High yield
#   every listed equity, factor and REIT line -> Global equity
#   private equity and venture -> Private equity (listed PE, deliberately harsh)
#   core real assets and convertibles -> half Global equity, half Corporate bonds
# Treasuries are the weakest fit: in 2008 they rallied while LQD barely moved.
EQ, IG, HY = {"Global equity": 1.0}, {"Corporate bonds": 1.0}, {"High yield": 1.0}
HALF = {"Global equity": 0.5, "Corporate bonds": 0.5}

PROXY_OF = {
    "U.S. Cash": {"Cash": 1.0},
    **{n: IG for n in [
        "U.S. Short Duration Government/Credit", "U.S. Intermediate Treasuries",
        "U.S. Long Treasuries", "TIPS", "U.S. Aggregate Bonds", "U.S. Securitized",
        "U.S. Long Duration Government/Credit", "U.S. Inv Grade Corporate Bonds",
        "U.S. Long Corporate Bonds", "World Government Bonds hedged",
        "World Government Bonds", "World ex-U.S. Government Bonds hedged",
        "World ex-U.S. Government Bonds", "U.S. Muni 1-15 Yr Blend",
        "Commercial Mortgage Loans"]},
    **{n: HY for n in [
        "U.S. High Yield Bonds", "U.S. Leveraged Loans", "U.S. Muni High Yield",
        "Emerging Markets Sovereign Debt", "Emerging Markets Local Currency Debt",
        "Emerging Markets Corporate Bonds", "Direct Lending"]},
    **{n: EQ for n in [
        "U.S. Large Cap", "U.S. Mid Cap", "U.S. Small Cap", "Euro Area Large Cap",
        "UK Large Cap", "Japanese Equity", "Hong Kong Equity", "EAFE Equity",
        "Chinese Domestic Equity", "Emerging Markets Equity",
        "AC Asia ex-Japan Equity", "AC World Equity", "U.S. Equity Value Factor",
        "U.S. Equity Momentum Factor", "U.S. Equity Quality Factor",
        "U.S. Equity Minimum Volatility Factor",
        "U.S. Equity Dividend Yield Factor", "U.S. REITs"]},
    **{n: HALF for n in [
        "Global Convertible Bonds hedged", "U.S. Core Real Estate",
        "Global Core Real Estate", "European Core Real Estate",
        "Asia Pacific Core Real Estate", "Global Core Infrastructure",
        "Global Core Transport", "Global Timberland"]},
    "Commodities": {"Commodities": 1.0},
    "Gold": {"Gold": 1.0},
    "Digital Assets": {"Digital assets": 1.0},
    "Private Equity": {"Private equity": 1.0},
    "Venture Capital": {"Private equity": 1.0},
    **{n: {"Hedge funds": 1.0} for n in [
        "Diversified Hedge Funds", "Event Driven Hedge Funds",
        "Long Bias Hedge Funds", "Relative Value Hedge Funds",
        "Macro Hedge Funds"]},
}
# A line found nowhere above (a name the app learns later) falls back by category.
BY_CATEGORY = {"FI": IG, "EQ": EQ, "ALT": {"Hedge funds": 1.0}}


def jpm_name(src, label) -> str:
    """The J.P. Morgan line a simulator line backtests as."""
    name = pl.proxy_of(src, src.resolve(label) if hasattr(src, "resolve") else label)
    if getattr(src, "KEY", "jpm") == "blackrock":
        from . import blackrock_cma as brc
        name = brc.CORR_PROXY.get(name, name)
    return name


def proxy_split(src, label, category=None) -> dict:
    split = PROXY_OF.get(jpm_name(src, label))
    if split is None:
        key = src.category_of(label) if hasattr(src, "category_of") else None
        if key is None and category is not None:
            key = {v: k for k, v in getattr(src, "CATEGORY_NAME", {}).items()
                   }.get(category)
        split = BY_CATEGORY.get(key, {"Hedge funds": 1.0})
    return split


def split_text(split: dict) -> str:
    return " + ".join(f"{s:.0%} {k}" if s < 1 else k for k, s in split.items())


def map_alloc(alloc: pd.DataFrame, src):
    """(weights over LINES in decimals summing to 1 — or None for an empty mix,
    table of line -> proxy for display)."""
    w = pd.Series(0.0, index=LINES)
    rows = []
    for _, r in alloc.iterrows():
        wt = float(pd.to_numeric(r.get("Weight %"), errors="coerce") or 0.0)
        if wt <= 0:
            continue
        split = proxy_split(src, r["LTCMA line"], r.get("Category"))
        for k, s in split.items():
            w[k] += wt * s
        rows.append({"Holding": str(r["LTCMA line"]), "Weight %": wt,
                     "Backtested as": split_text(split)})
    total = w.sum()
    return (w / total if total > 0 else None), pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Steps B–E
# --------------------------------------------------------------------------
@dataclass
class Backtest:
    P: pd.DataFrame          # monthly portfolio returns, months x scenarios
    V: pd.DataFrame          # value paths from 1.0, incl. the month before the first return
    DD: pd.DataFrame         # drawdown from previous high
    metrics: pd.DataFrame    # metric x scenario
    crises: pd.DataFrame     # one row per (crisis, scenario)
    contrib: pd.DataFrame    # (crisis, scenario) x line, sums to the crisis return
    W: pd.DataFrame          # target weights, line x scenario


def effective_weights(R: pd.DataFrame, W: pd.DataFrame) -> np.ndarray:
    """Weights held each month, shape (months, lines, scenarios)."""
    A = R.notna().to_numpy(float)
    num = A[:, :, None] * W.to_numpy()[None, :, :]
    den = num.sum(axis=1, keepdims=True)
    if (den == 0).any():
        raise ValueError("A mix holds only lines with no data in some month.")
    return num / den


def yearly_drawdown(r: pd.Series) -> pd.Series:
    """Deepest fall inside each calendar year, from the previous year-end."""
    out = {}
    for y, g in r.groupby(r.index.year):
        v = np.concatenate([[1.0], (1 + g).cumprod().to_numpy()])
        out[y] = (v / np.maximum.accumulate(v) - 1).min()
    return pd.Series(out)


def _metrics(r: pd.Series, v: pd.Series, rf: pd.Series) -> dict:
    n = len(r)
    ex = r - rf
    dd = v / v.cummax() - 1
    trough = dd.idxmin()
    peak = v.loc[:trough].idxmax()
    after = v.loc[trough:]
    rec = after[after >= v.loc[peak]]
    rec_date = rec.index[0] if len(rec) else None
    roll12 = v / v.shift(12) - 1
    years = yearly_drawdown(r)
    bad = years[years < DRAWDOWN_LIMIT]
    cagr = v.iloc[-1] ** (12 / n) - 1
    return {
        "Growth multiple": v.iloc[-1],
        "CAGR": cagr,
        "Volatility": r.std() * np.sqrt(12),
        "Sharpe": ex.mean() / r.std() * np.sqrt(12),
        "Max drawdown": dd.min(),
        "Peak": peak, "Trough": trough, "Recovered": rec_date,
        "Months to recover": (len(v.loc[trough:rec_date]) - 1
                              if rec_date is not None else np.nan),
        "Worst 12 months": roll12.min(),
        "Worst 12 months ended": roll12.idxmin(),
        "Worst month": r.min(),
        "Worst month date": r.idxmin(),
        "Years below limit": int(len(bad)),
        "Which years": ", ".join(str(y) for y in bad.index) or "none",
    }


def run(R: pd.DataFrame, W: pd.DataFrame) -> Backtest:
    """R: monthly returns, months x LINES. W: weights, LINES x scenarios (decimals)."""
    R = R[list(W.index)]
    Weff = effective_weights(R, W)
    r = R.fillna(0.0).to_numpy()
    P = pd.DataFrame((Weff * r[:, :, None]).sum(axis=1), index=R.index,
                     columns=W.columns)
    V = (1 + P).cumprod()
    V.loc[P.index[0] - pd.offsets.MonthEnd(1)] = 1.0
    V = V.sort_index()
    DD = V / V.cummax() - 1

    rf = R[CASH_LINE] if CASH_LINE in R else pd.Series(0.0, index=R.index)
    metrics = pd.DataFrame({s: _metrics(P[s], V[s], rf) for s in P.columns})

    rows, contrib = [], {}
    for crisis, (a, b) in CRISES.items():
        months = R.loc[a:b].index[1:]
        if len(months) == 0:
            continue
        pos = [R.index.get_loc(t) for t in months]
        for k, s in enumerate(W.columns):
            val, path, c = 1.0, [1.0], np.zeros(len(W.index))
            for p in pos:
                w = Weff[p, :, k]
                c += val * w * r[p]
                val *= 1 + float(w @ r[p])
                path.append(val)
            path = np.array(path)
            ret = path[-1] - 1
            assert abs(c.sum() - ret) < 1e-10, "contributions must add up"
            if path[-1] >= 1:
                rec = 0
            else:
                rec, vt = np.nan, path[-1]
                for j, x in enumerate(P[s].loc[b:].iloc[1:], start=1):
                    vt *= 1 + x
                    if vt >= 1:
                        rec = j
                        break
            rows.append({
                "Crisis": crisis, "Scenario": s,
                "Window": f"{pd.Timestamp(a):%b %Y} – {pd.Timestamp(b):%b %Y}",
                "Return": ret,
                "Max drawdown": (path / np.maximum.accumulate(path) - 1).min(),
                "Months to recover": rec,
            })
            contrib[(crisis, s)] = pd.Series(c, index=W.index)
    C = pd.DataFrame(contrib).T
    C.index.names = ["Crisis", "Scenario"]
    return Backtest(P, V, DD, metrics, pd.DataFrame(rows), C, W)


def self_check(R: pd.DataFrame) -> None:
    """A 100% single-line mix must reproduce that line exactly."""
    for c in R.columns:
        W1 = pd.DataFrame({"one": (pd.Index(R.columns) == c).astype(float)},
                          index=R.columns)
        Rm = R.loc[R[c].notna()]
        got = run(Rm, W1).P["one"].to_numpy()
        assert np.allclose(got, Rm[c].to_numpy()), f"single-line check failed: {c}"
