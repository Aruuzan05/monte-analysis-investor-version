"""Checks for geography (keepp_mc/geography.py) and the optimizer's
geography limits.

    python tests/test_geography.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from keepp_mc import blackrock_cma as brc
from keepp_mc import geography as geo
from keepp_mc import jpm_ltcma as jpm
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
check("every J.P. Morgan line has a region",
      all(n in geo.JPM for n, *_ in jpm.LINES) and len(geo.JPM) == len(jpm.LINES))
check("every BlackRock line has a region",
      all(n in geo.BLACKROCK for n, *_ in BR.LINES) and len(geo.BLACKROCK) == len(BR.LINES))
for nm in ["ACWI", "WORLD", "EAFE", "WORLD_EX_US", "WGBI", "WGBI_EX_US"]:
    check(f"{nm} split sums to 100", sum(getattr(geo, nm).values()) == 100)
check("every line's shares sum to 1 (both providers)",
      all(abs(geo.shares(jpm, n).sum() - 1) < 1e-12 for n, *_ in jpm.LINES)
      and all(abs(geo.shares(BR, n).sum() - 1) < 1e-12 for n, *_ in BR.LINES))
check("regional lines are 100% in their region",
      geo.shares(jpm, "U.S. Large Cap")[0] == 1 and geo.shares(jpm, "Japanese Equity")[3] == 1
      and geo.shares(BR, "Europe large cap equities")[1] == 1)
check("AC World is 64% US", abs(geo.shares(jpm, "AC World Equity")[0] - 0.64) < 1e-12)
check("emerging markets count as Global", geo.shares(jpm, "Emerging Markets Equity")[4] == 1)
check("personal lines take their proxy's region (BlackRock: US PE -> US)",
      geo.shares(BR, pl.PERSONAL_PE)[0] == 1 and geo.shares(jpm, pl.PERSONAL_PE)[4] == 1)
check("star labels resolve (U.S. Cash ★ is US)", geo.shares(jpm, jpm.LABEL["U.S. Cash"])[0] == 1)

print()
print("=" * 78)
print("2. BREAKDOWN — hand-checkable")
print("=" * 78)
bd = geo.breakdown(jpm, ["AC World Equity", "U.S. Aggregate Bonds", "Gold"],
                   ["Equities", "Fixed Income", "Alternatives"], [50, 40, 10])
check("50% AC World + 40% US bonds + 10% gold: US = 0.64*50 + 40 = 72%",
      abs(bd.loc["US", "Total"] - 72) < 1e-9)
check("  US stocks 32%, US bonds 40%", abs(bd.loc["US", "Stocks"] - 32) < 1e-9
      and abs(bd.loc["US", "Bonds"] - 40) < 1e-9)
check("  Global = 0.06*50 + 10 = 13%", abs(bd.loc["Global", "Total"] - 13) < 1e-9)
check("  regions add back to 100%", abs(bd["Total"].sum() - 100) < 1e-9)

print()
print("=" * 78)
print("3. OPTIMIZER LIMITS")
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
    df = pd.DataFrame([{"Region": r, **{f"{s} {m} %": np.nan for s in geo.SCOPES
                                        for m in ("min", "max")}} for r in geo.REGIONS])
    for (region, col), v in (cells or {}).items():
        df.loc[df["Region"] == region, col] = v
    return df


free = opt.Limits(np.zeros(n), np.ones(n), groups, bands)
w0, _ = uni.max_sharpe_qp(mu, cov, free, rf)
check("no geography limits: the optimizer answer is unchanged",
      np.allclose(w0, uni.max_sharpe_qp(mu, cov, opt.Limits(np.zeros(n), np.ones(n),
                                                            groups, bands, exposure=()), rf)[0]))
us_st0 = geo.breakdown(jpm, lines, cats, w0 * 100).loc["US", "Stocks"]
target = round(us_st0 + 10, 1)
rows = geo.exposure_rows(jpm, lines, cats, limits_df({("US", "Stocks min %"): target}))
L = opt.Limits(np.zeros(n), np.ones(n), groups, bands, exposure=rows)
w1, _ = uni.max_sharpe_qp(mu, cov, L, rf)
us_st1 = geo.breakdown(jpm, lines, cats, w1 * 100).loc["US", "Stocks"]
check(f"'US stocks at least {target}%' binds exactly (was {us_st0:.2f}%, now {us_st1:.2f}%)",
      abs(us_st1 - target) < 0.01)
check("the constrained answer is fully invested", abs(w1.sum() - 1) < 1e-9)
rows = geo.exposure_rows(jpm, lines, cats, limits_df({("Asia", "All assets max %"): 5.0,
                                                        ("Europe", "All assets min %"): 15.0}))
L = opt.Limits(np.zeros(n), np.ones(n), groups, bands, exposure=rows)
w2 = opt.max_growth(mu, cov, L).x
b2 = geo.breakdown(jpm, lines, cats, w2 * 100)
check(f"max-growth with Asia <= 5% and Europe >= 15%: Asia {b2.loc['Asia','Total']:.2f}%, "
      f"Europe {b2.loc['Europe','Total']:.2f}%",
      b2.loc["Asia", "Total"] <= 5.0 + 1e-4 and b2.loc["Europe", "Total"] >= 15.0 - 1e-4)
rows = geo.exposure_rows(jpm, lines, cats, limits_df({("Japan", "Bonds min %"): 20.0}))
try:
    opt.feasible_start(opt.Limits(np.zeros(n), np.ones(n), groups, bands, exposure=rows))
    check("impossible 'Japan bonds at least 20%' is refused", False)
except opt.Infeasible as e:
    check(f"impossible 'Japan bonds at least 20%' is refused, named: \"{e}\"",
          "Japan bonds" in str(e) and "14.00%" in str(e))
sp = opt.return_span(mu, cov, opt.Limits(np.zeros(n), np.ones(n), groups, bands,
     exposure=geo.exposure_rows(jpm, lines, cats, limits_df({("Global", "All assets max %"): 10.0}))))
check("the return slider's top respects geography (Global, where the top-return "
      "line Private Equity sits, capped at 10%)",
      sp.hi < opt.return_span(mu, cov, free).hi)

print()
print("RESULT:", "all checks pass" if ok else "FAILURES ABOVE — investigate")
raise SystemExit(0 if ok else 1)
