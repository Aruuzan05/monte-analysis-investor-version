"""Checks for liquidity tiers (keepp_mc/liquidity.py) and the optimizer's
liquidity limits.

    python tests/test_liquidity.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from keepp_mc import blackrock_cma as brc
from keepp_mc import geography as geo
from keepp_mc import jpm_ltcma as jpm
from keepp_mc import liquidity as liq
from keepp_mc import optimizer as opt
from keepp_mc import personal_lines as pl
from keepp_mc import universe as uni

ok = True


def check(label, cond):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")


BR = brc.get(brc.DEFAULT_SCENARIO, 10)

print("=" * 78)
print("1. DATA")
print("=" * 78)
check("every J.P. Morgan line has a tier",
      all(n in liq.JPM for n, *_ in jpm.LINES) and len(liq.JPM) == len(jpm.LINES))
check("every BlackRock line has a tier",
      all(n in liq.BLACKROCK for n, *_ in BR.LINES + BR.EXTRA_LINES)
      and len(liq.BLACKROCK) == len(BR.LINES) + len(BR.EXTRA_LINES))
check("every line's shares sum to 1 (both providers)",
      all(abs(liq.tier_shares(jpm, n).sum() - 1) < 1e-12 for n, *_ in jpm.LINES)
      and all(abs(liq.tier_shares(BR, n).sum() - 1) < 1e-12 for n, *_ in BR.LINES))
check("tier numbers are ordered: slower and costlier down the ladder",
      all(liq.DAYS[a] < liq.DAYS[b] and liq.HAIRCUT[a] < liq.HAIRCUT[b]
          for a, b in zip(liq.TIERS, liq.TIERS[1:])))
check("haircuts are the range midpoints (L2 10%, L5 47.5%)",
      liq.HAIRCUT["L2"] == 0.10 and liq.HAIRCUT["L5"] == 0.475)
check("cash is L1 on both providers",
      liq.tier_label(jpm, "U.S. Cash") == "L1" and liq.tier_label(BR, "US cash") == "L1")
check("star labels resolve (U.S. Cash ★ is L1)",
      liq.tier_label(jpm, jpm.LABEL["U.S. Cash"]) == "L1")
check("PE / VC / direct lending are L5",
      all(liq.tier_label(jpm, n) == "L5"
          for n in ["Private Equity", "Venture Capital", "Direct Lending"]))
check("a split line shows both tiers (Macro hedge funds: L3/L4, 50/50)",
      liq.tier_label(jpm, "Macro Hedge Funds") == "L3/L4"
      and np.allclose(liq.tier_shares(jpm, "Macro Hedge Funds"), [0, 0, .5, .5, 0]))

print()
print("=" * 78)
print("2. PERSONAL LINES")
print("=" * 78)
check("personal lines default to L5 on both providers",
      all(liq.tier_label(s, n) == "L5" for s in (jpm, BR) for n in pl.NAMES))
ov = {pl.PERSONAL_PE: {"tier": "L4", "haircut": 0.25}}
check("tier override moves Personal PE to L4",
      liq.tier_label(jpm, pl.PERSONAL_PE, ov) == "L4")
check("haircut override is used as typed (25%)",
      liq.haircut(jpm, pl.PERSONAL_PE, ov) == 0.25)
check("blank haircut falls back to the tier's (L4 30%)",
      liq.haircut(jpm, pl.PERSONAL_PE, {pl.PERSONAL_PE: {"tier": "L4",
                                                          "haircut": None}}) == 0.30)
check("overrides never touch provider lines",
      liq.tier_label(jpm, "Private Equity", {"Private Equity": {"tier": "L1"}}) == "L5")

print()
print("=" * 78)
print("3. BREAKDOWN — hand-checkable")
print("=" * 78)
lines = ["U.S. Cash", "U.S. Large Cap", "Macro Hedge Funds", "Private Equity"]
cats = ["Fixed Income", "Equities", "Alternatives", "Alternatives"]
w = [10, 60, 10, 20]
bd = liq.breakdown(jpm, lines, cats, w)
check("L1 10, L2 60, L3 5, L4 5, L5 20",
      np.allclose(bd["Total"].values, [10, 60, 5, 5, 20]))
check("  L2 is all stocks; L5 is all alts",
      bd.loc["L2", "Stocks"] == 60 and bd.loc["L5", "Alternatives"] == 20)
check("  tiers add back to 100%", abs(bd["Total"].sum() - 100) < 1e-9)
s = liq.ladder_sentence(bd["Total"])
check(f"sentence: \"{s}\"",
      s == "70% can be cash within a week, 10% within a year, "
           "20% is locked for 3+ years.")
d = liq.days_to_cash(jpm, lines, w)
check(f"weighted days to cash = {d:.1f}",
      abs(d - (10 * 1 + 60 * 3.5 + 5 * 17.5 + 5 * liq.DAYS["L4"]
               + 20 * liq.DAYS["L5"]) / 100) < 1e-9)
h = liq.portfolio_haircut(jpm, lines, w)
check(f"weighted crisis haircut = {h:.4f}",
      abs(h - (60 * .10 + 5 * .225 + 5 * .30 + 20 * .475) / 100) < 1e-12)

print()
print("=" * 78)
print("4. OPTIMIZER LIMITS")
print("=" * 78)
ua = uni.universe_alloc(pd.DataFrame(columns=uni.ALLOC_COLS), src=jpm)
lines, cats = list(ua["LTCMA line"]), list(ua["Category"])
mu, sig = opt.inputs_from_percent(ua["Return %"], ua["Vol %"])
cov = opt.covariance(sig, jpm.corr_for(lines))
n = len(lines)
groups = {c: [i for i, x in enumerate(cats) if x == c] for c in dict.fromkeys(cats)}
bands = {c: (0.0, 1.0) for c in groups}
rf = 0.031


def limits_df(cells=None):
    df = pd.DataFrame([{"Tiers": g, "Min %": np.nan, "Max %": np.nan}
                       for g in liq.GROUPS])
    for (g, col), v in (cells or {}).items():
        df.loc[df["Tiers"].str.startswith(g), col] = v
    return df


check("blank table: no limits", liq.exposure_rows(jpm, lines, limits_df()) == ())
free = opt.Limits(np.zeros(n), np.ones(n), groups, bands)
w0, _ = uni.max_sharpe_qp(mu, cov, free, rf)
locked0 = liq.exposure(jpm, lines, w0 * 100)[["L4", "L5"]].sum()
cap = round(max(locked0 - 10, 1.0), 1)
rows = liq.exposure_rows(jpm, lines, limits_df({("L4–L5", "Max %"): cap}))
L = opt.Limits(np.zeros(n), np.ones(n), groups, bands, exposure=rows)
w1, _ = uni.max_sharpe_qp(mu, cov, L, rf)
locked1 = liq.exposure(jpm, lines, w1 * 100)[["L4", "L5"]].sum()
check(f"'L4–L5 at most {cap}%' holds (was {locked0:.2f}%, now {locked1:.2f}%)",
      locked1 <= cap + 0.01)
rows = liq.exposure_rows(jpm, lines, limits_df({("L1–L2", "Min %"): 80.0}))
L = opt.Limits(np.zeros(n), np.ones(n), groups, bands, exposure=rows)
w2 = opt.max_growth(mu, cov, L).x
liquid2 = liq.exposure(jpm, lines, w2 * 100)[["L1", "L2"]].sum()
check(f"max-growth with L1–L2 >= 80%: {liquid2:.2f}%", liquid2 >= 80 - 1e-4)
check("no breach reported on the constrained answer", liq.check(rows, w2) == [])
check("geography and liquidity limits combine in one Limits",
      len(opt.Limits(np.zeros(n), np.ones(n), groups, bands,
                     exposure=rows + geo.exposure_rows(
                         jpm, lines, cats, pd.DataFrame(
                             [{"Region": "US", "All assets max %": 60.0}]))).exposure) == 2)
sp = opt.return_span(mu, cov, opt.Limits(
    np.zeros(n), np.ones(n), groups, bands,
    exposure=liq.exposure_rows(jpm, lines, limits_df({("L1–L2", "Min %"): 90.0}))))
check("the return slider's top respects liquidity (L1–L2 at least 90% keeps "
      "the top-return line, Digital Assets in L3, out)",
      sp.hi < opt.return_span(mu, cov, free).hi)

print()
print("RESULT:", "all checks pass" if ok else "FAILURES ABOVE — investigate")
raise SystemExit(0 if ok else 1)
