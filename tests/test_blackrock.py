"""Checks for the BlackRock assumptions (blackrock_cma.py).

Run after any change to blackrock_cma.py or a new blackrock_cma.csv:

    python tests/test_blackrock.py
"""
import numpy as np
import pandas as pd

import sys
from pathlib import Path

# Run as a script (python tests/test_x.py): put the repo root on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from keepp_mc import blackrock_cma as br
from keepp_mc import jpm_ltcma as jpm
from keepp_mc import universe as uni
from keepp_mc.engine import simulate

ok = True


def check(label, cond):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")


S = br.get(br.DEFAULT_SCENARIO, 10)
names = [n for n, *_ in S.LINES]
base = br._BASE.set_index("asset")

print("=" * 78)
print("1. DATA — what was taken from BlackRock's workbook")
print("=" * 78)
check("34 USD asset classes, 60/40 portfolio excluded",
      len(names) == 34 and "Global 60/40 portfolio" not in names)
check("three scenarios, each listing the same assets",
      sorted(br.DATA["scenario"].unique()) == sorted(br.SCENARIOS))
check("every asset sits in one of the app's three categories",
      all(S.category_of(n) in (br.FI, br.EQ, br.ALT) for n in names))
check("every category has lines (Fixed income 15, Equities 11, Private 8)",
      [len(S.CHOICES[k]) for k in br.CATEGORY_KEYS] == [15, 11, 8])
check("returns stored in percent (US large cap 5y = 9.44%)",
      abs(base.loc["US large cap equities", "ret_5y"] - 9.439035) < 1e-6)
check("vol stored in percent (US large cap = 19.35%)",
      abs(S.BY_NAME["US large cap equities"][2] - 19.3478) < 1e-3)
check("each default line exists in its category",
      all(br.DEFAULT_LINE[k] in S.CHOICES[k] for k in br.CATEGORY_KEYS))

print()
print("=" * 78)
print("2. HORIZON — returns follow the simulation's number of years")
print("=" * 78)
row = base.loc["US large cap equities"]
check("10-year run reads the 10-year column",
      abs(br.get(br.DEFAULT_SCENARIO, 10).BY_NAME["US large cap equities"][0]
          - round(row["ret_10y"], 4)) < 1e-9)
mid = row["ret_10y"] + (row["ret_15y"] - row["ret_10y"]) * 2 / 5
check("12-year run interpolates 2/5 of the way from 10y to 15y",
      abs(br.get(br.DEFAULT_SCENARIO, 12).BY_NAME["US large cap equities"][0]
          - round(mid, 4)) < 1e-9)
check("runs under 5 years use the 5-year figure",
      br.get(br.DEFAULT_SCENARIO, 3).BY_NAME["US large cap equities"][0]
      == round(row["ret_5y"], 4))
check("horizon note names the interpolation",
      "10- and 15-year" in br.horizon_note(12))
ai = br.get("AI productivity boom", 10).BY_NAME["US large cap equities"][0]
check("scenario changes returns (AI boom US large cap > base)",
      ai > br.get(br.DEFAULT_SCENARIO, 10).BY_NAME["US large cap equities"][0])
check("scenario does not change volatility",
      br.get("AI productivity boom", 10).BY_NAME["US large cap equities"][2]
      == S.BY_NAME["US large cap equities"][2])

print()
print("=" * 78)
print("3. CORRELATIONS — built from BlackRock's two published correlations")
print("=" * 78)
C = br.CORR
check("symmetric with unit diagonal",
      np.allclose(C, C.T) and np.allclose(np.diag(C), 1.0))
check("positive semi-definite (smallest eigenvalue >= -1e-10)",
      np.linalg.eigvalsh(C).min() > -1e-10)
check("Higham repair was cosmetic (< 1e-6)", br.CORR_MAX_REPAIR < 1e-6)
gov = br.CORR_INDEX["Global ex-US treasuries (hedged)"]
dev = max(abs(C[br.CORR_INDEX[n], gov] - base.loc[n, "corr_gov_bonds"])
          for n in names if not pd.isna(base.loc[n, "corr_gov_bonds"]))
check(f"BlackRock's published bond correlations reproduced (max dev {dev:.1e})",
      dev < 1e-6)
F = br._factor_inv(base.loc["Global ex-US treasuries (hedged)", "corr_equities"])
# implied correlation with the equity factor = c_i' F^-1 (rho_be, 1)
rbe = base.loc["Global ex-US treasuries (hedged)", "corr_equities"]
dev_e = max(abs(br.FACTOR_LOADINGS[br.CORR_INDEX[n]] @ F @ np.array([rbe, 1.0])
                - base.loc[n, "corr_equities"])
            for n in names if not pd.isna(base.loc[n, "corr_equities"]))
check(f"BlackRock's published equity correlations kept as loadings "
      f"(max dev {dev_e:.1e})", dev_e < 1e-6)


def pair(a, b):
    return C[br.CORR_INDEX[a], br.CORR_INDEX[b]]


check("family link kept: US credit vs long US credit > 0.9",
      pair("US credit (all maturities)", "US credit (10+ years)") > 0.9)
check("family link kept: Europe vs global ex-US equities > 0.9",
      pair("Europe large cap equities", "Global ex-US large cap equities") > 0.9)
check("cash is close to uncorrelated with bonds (|corr| < 0.2)",
      abs(pair("US cash", "US aggregate bond index")) < 0.2)
check("every BlackRock asset has a J.P. Morgan correlation proxy",
      all(br.CORR_PROXY[n] in jpm.BY_NAME for n in names))

print()
print("=" * 78)
print("4. INTERFACE — works where the app uses the J.P. Morgan module")
print("=" * 78)
check("lookup returns (compound return, vol)",
      S.lookup("US cash") == (S.BY_NAME["US cash"][0], S.BY_NAME["US cash"][2]))
check("resolve passes names through", S.resolve(" US cash ") == "US cash")
check("corr_for returns the sub-matrix in the asked order",
      np.allclose(S.corr_for(["US cash", "US high yield"]),
                  C[np.ix_([br.CORR_INDEX["US cash"], br.CORR_INDEX["US high yield"]],
                           [br.CORR_INDEX["US cash"], br.CORR_INDEX["US high yield"]])]))
ua = uni.universe_alloc(pd.DataFrame(
    [{"Category": "Equities", "LTCMA line": "US large cap equities",
      "Weight %": 100.0, "Return %": 9.0, "Vol %": 15.0}]), src=S)
check("universe_alloc lists all 34 BlackRock lines, keeping the user's numbers",
      len(ua) == 34 and float(ua.loc[ua["LTCMA line"] == "US large cap equities",
                                     "Return %"].iloc[0]) == 9.0)
prev = pd.DataFrame([{"LTCMA line": "US cash", "Weight %": 50.0,
                      "Return %": 3.0, "Vol %": 1.0}])
edited = prev.copy()
edited.loc[0, "LTCMA line"] = "US high yield"
out, dirty = S.sync_group(prev, edited, br.FI)
check("sync_group refills return and vol when the line changes",
      dirty and out.at[0, "Return %"] == S.BY_NAME["US high yield"][0])
check("source id round-trips the scenario",
      br.parse_source_id(br.source_id("AI productivity boom"))
      == "AI productivity boom")

sel = ["US aggregate bond index", "All country large cap equities",
       "US private equity (buyout)", "US cash"]
res, repaired = simulate([40, 40, 10, 10], [S.BY_NAME[n][0] for n in sel],
                         [S.BY_NAME[n][2] for n in sel], S.corr_for(sel),
                         2000, 10, 7)
check("engine simulates a BlackRock portfolio without repairing the matrix",
      not repaired and res.rows[-1]["Median x"] > 1.0)

# ==========================================================================
print()
print("=" * 78)
print("J.P. MORGAN EXTRAS — commodities, gold and digital inside BlackRock")
print("=" * 78)
# BlackRock's CMA has no commodities, gold or digital-asset line, so three
# J.P. Morgan lines are offered in Custom mode only.
import numpy as _np
from keepp_mc import personal_lines as _pl

E = br.get(br.DEFAULT_SCENARIO, 10)
extras = [n for n, *_ in E.EXTRA_LINES]
check(f"three extras are offered: {extras}",
      extras == list(br.JPM_EXTRAS))
check(f"BlackRock's own catalogue is untouched at {len(E.LINES)} lines",
      len(E.LINES) == 34 and not (set(extras) & {n for n, *_ in E.LINES}))
check("Standard mode cannot reach them — they are not in CHOICES",
      not any(n in E.CHOICES[k] for k in E.CATEGORY_KEYS for n in extras))
check("Custom mode can — personal_lines.choices() appends them under "
      "Alternatives",
      all(n in _pl.choices(E, br.ALT) for n in extras))
check("they carry J.P. Morgan's own return and vol",
      all(E.lookup(n) == jpm.lookup(br.JPM_EXTRAS[n]) for n in extras))
check("...including Digital Assets at 10.00 / 46.70",
      E.lookup("Digital Assets (J.P. Morgan)") == (10.0, 46.7))

# The disclosure the UI makes: these do not move with scenario or horizon,
# because J.P. Morgan publishes one number where BlackRock publishes many.
fixed = {sc: br.get(sc, 10).lookup("Gold (J.P. Morgan)") for sc in br.SCENARIOS}
check(f"they do not move across BlackRock's scenarios {sorted(set(fixed.values()))}",
      len(set(fixed.values())) == 1)
by_h = {y: br.get(br.DEFAULT_SCENARIO, y).lookup("Gold (J.P. Morgan)")
        for y in (5, 10, 30)}
check("they do not move across horizons either",
      len(set(by_h.values())) == 1)
moves = {y: br.get(br.DEFAULT_SCENARIO, y).lookup("US large cap equities")
         for y in (5, 10, 30)}
check("while BlackRock's own lines do move across horizons (the contrast the "
      "UI has to state)", len(set(moves.values())) > 1)

check(f"the matrix grew to {br.CORR.shape[0]}x{br.CORR.shape[1]} and its "
      f"repair is still cosmetic ({br.CORR_MAX_REPAIR:.5f} < 0.01)",
      br.CORR.shape == (37, 37) and br.CORR_MAX_REPAIR < 0.01)
check("the extras took their J.P. Morgan factor loadings — they have no "
      "published BlackRock ones",
      all(_np.isnan(br._BASE.set_index("asset").loc[n, "corr_gov_bonds"])
          for n in extras))
mixed = ["US large cap equities", "US cash", "US aggregate bond index"] + extras
CM = E.corr_for(mixed)
try:
    _np.linalg.cholesky(CM)
    mixed_ok = True
except _np.linalg.LinAlgError:
    mixed_ok = False
check(f"a BlackRock portfolio holding all three factorises without repair "
      f"(min eig {_np.linalg.eigvalsh(CM).min():+.5f})", mixed_ok)
gi, di = mixed.index("Gold (J.P. Morgan)"), mixed.index("Digital Assets (J.P. Morgan)")
check(f"gold x digital comes through as {CM[gi, di]:.4f}, inherited from the "
      f"J.P. Morgan construction rather than 1.00",
      0.9 < CM[gi, di] < 1.0)

print()
print("RESULT:", "all checks pass" if ok else "FAILURES ABOVE — investigate")
raise SystemExit(0 if ok else 1)
