"""Checks for the Backtest & stress tab (keepp_mc/backtest.py and the
bundled proxy returns, keepp_mc/backtest_data.py).

    python tests/test_backtest.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from keepp_mc import backtest as bt
from keepp_mc import backtest_data as bd
from keepp_mc import blackrock_cma as brc
from keepp_mc import jpm_ltcma as jpm
from keepp_mc import personal_lines as pl

ok = True


def check(label, cond):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")


R = bd.load_returns()

print("=" * 78)
print("1. DATA")
print("=" * 78)
check("snapshot has the nine lines in order, from Jan 1997",
      list(R.columns) == bd.LINES and R.index[0] == pd.Timestamp("1997-01-31"))
check("only Digital assets starts late",
      R.drop(columns="Digital assets").notna().all().all())
check("no line has a hole once it has started",
      all(not R.loc[R[c].first_valid_index():, c].isna().any() for c in R.columns))
y = (1 + R).groupby(R.index.year).prod() - 1
check(f"High yield (VWEHX) 2008 {y.loc[2008, 'High yield']:+.1%} / 2009 "
      f"{y.loc[2009, 'High yield']:+.1%} (fund published −21.3% / +39.0%)",
      abs(y.loc[2008, "High yield"] + .212) < .005
      and abs(y.loc[2009, "High yield"] - .392) < .005)

print()
print("=" * 78)
print("2. MAPPING")
print("=" * 78)
check("every J.P. Morgan line maps", all(n in bt.PROXY_OF for n, *_ in jpm.LINES))
BR = brc.get(brc.DEFAULT_SCENARIO, 10)
check("every BlackRock line (incl. the J.P. Morgan extras) maps",
      all(bt.jpm_name(BR, lb) in bt.PROXY_OF for lb in BR.LABEL.values()))
check("the personal lines map (PE -> Private equity, PC -> High yield)",
      bt.proxy_split(jpm, pl.PERSONAL_PE) == {"Private equity": 1.0}
      and bt.proxy_split(jpm, pl.PERSONAL_PC) == {"High yield": 1.0})
check("every split sums to 1 and uses known lines",
      all(abs(sum(s.values()) - 1) < 1e-12 and set(s) <= set(bt.LINES)
          for s in bt.PROXY_OF.values()))
alloc = pd.DataFrame({"Category": ["Fixed Income", "Alternatives", "Alternatives"],
                      "LTCMA line": ["U.S. Cash", "U.S. Core Real Estate", pl.PERSONAL_PC],
                      "Weight %": [20, 40, 40]})
w, table = bt.map_alloc(alloc, jpm)
check("map_alloc: 20 cash, core RE split 20/20 equity/bonds, 40 personal PC to HY",
      np.allclose(w[["Cash", "Global equity", "Corporate bonds", "High yield"]],
                  [.2, .2, .2, .4]) and abs(w.sum() - 1) < 1e-12 and len(table) == 3)
check("an empty mix maps to None",
      bt.map_alloc(alloc.assign(**{"Weight %": 0}), jpm)[0] is None)

print()
print("=" * 78)
print("3. ENGINE")
print("=" * 78)
try:
    bt.self_check(R)
    check("a 100% single-line mix reproduces that line exactly", True)
except AssertionError as e:
    check(str(e), False)

# Reference figures from the standalone keepp_backtest.py run (22 Nov 2026),
# history to Oct 2026.
W = pd.DataFrame({"C": [10, 7, 10, 10, 25, 0, 2, 1, 35],
                  "D": [10, 15, 10, 10, 30, 0, 3, 2, 20]}, index=bt.LINES) / 100
b = bt.run(R.loc[:"2026-10-31"], W)
cr = b.crises.set_index(["Crisis", "Scenario"])["Return"]
check(f"C through the GFC {cr[('Global financial crisis 2007-09', 'C')]:+.2%} "
      f"(standalone −29.69%)",
      abs(cr[("Global financial crisis 2007-09", "C")] + .2969) < 5e-5)
check(f"C through COVID {cr[('COVID-19 crash 2020', 'C')]:+.2%} (standalone −11.76%)",
      abs(cr[("COVID-19 crash 2020", "C")] + .1176) < 5e-5)
check(f"D CAGR {b.metrics.loc['CAGR', 'D']:.2%} (6.01%), max drawdown "
      f"{b.metrics.loc['Max drawdown', 'D']:.1%} (−28.0%)",
      abs(b.metrics.loc["CAGR", "D"] - .0601) < 5e-5
      and abs(b.metrics.loc["Max drawdown", "D"] + .280) < 5e-4)
check("crisis contributions add up to the crisis returns",
      np.allclose(b.contrib.sum(axis=1).loc[cr.index], cr))
check("no Bitcoin data in the GFC: its contribution is exactly 0",
      (b.contrib.xs("Global financial crisis 2007-09")["Digital assets"] == 0).all())

print()
print("RESULT:", "all checks pass" if ok else "FAILURES ABOVE — investigate")
raise SystemExit(0 if ok else 1)
