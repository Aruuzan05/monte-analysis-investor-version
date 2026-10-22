"""Checks for Keepp's own lines (keepp_mc/personal_lines.py).

    python tests/test_personal_lines.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from keepp_mc import blackrock_cma as brc
from keepp_mc import jpm_ltcma as jpm
from keepp_mc import personal_lines as pl
from keepp_mc import universe as uni
from keepp_mc.engine import simulate

ok = True


def check(label, cond):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")


BR = brc.get(brc.DEFAULT_SCENARIO, 10)
PE, PC = pl.PERSONAL_PE, pl.PERSONAL_PC

print("=" * 78)
print("1. CORRELATIONS — each personal line copies its asset type's row")
print("=" * 78)
for src, name in ((jpm, "J.P. Morgan"), (BR, "BlackRock")):
    pe_proxy, pc_proxy = pl.PROXY[PE][pl._key(src)], pl.PROXY[PC][pl._key(src)]
    others = [n for n, *_ in src.LINES if n not in (pe_proxy, pc_proxy)][:12]
    C = pl.corr_for(src, [PE, PC] + others)
    R = src.corr_for([pe_proxy, pc_proxy] + others)
    check(f"{name}: identical to the proxies' rows against every other line",
          np.allclose(C, R))
    C2 = pl.corr_for(src, [PE, pe_proxy, PC, pc_proxy])
    check(f"{name}: a personal line vs its own proxy is {pl.SELF_CORR} "
          f"(not 1.0)", C2[0, 1] == pl.SELF_CORR and C2[2, 3] == pl.SELF_CORR)
    check(f"{name}: personal PE vs personal PC = PE proxy vs PC proxy",
          abs(C2[0, 2] - src.corr_for([pe_proxy, pc_proxy])[0, 1]) < 1e-12)
    check(f"{name}: matrix stays positive definite when both are held",
          np.linalg.eigvalsh(C2).min() > 0)

print()
print("=" * 78)
print("2. CUSTOM TABLE — offered under Alternatives only, numbers kept")
print("=" * 78)
check("in the Alternatives dropdown on both sources",
      PE in pl.choices(jpm, "ALT") and PC in pl.choices(BR, "ALT"))
check("not offered in Fixed Income or Equities",
      PE not in pl.choices(jpm, "FI") and PE not in pl.choices(jpm, "EQ"))
check("provider catalogues themselves unchanged (Standard mode never sees them)",
      PE not in jpm.CHOICES["ALT"] and PE not in BR.CHOICES["ALT"]
      and PE not in jpm.BY_NAME)
prev = pd.DataFrame([{"LTCMA line": jpm.LABEL["Commodities"], "Weight %": 10.0,
                      "Return %": 4.6, "Vol %": 18.32}])
picked = prev.copy(); picked.loc[0, "LTCMA line"] = PE
out, dirty = pl.sync_group(jpm, prev, picked, "ALT")
check("picking it keeps the line and starts from its proxy's figures",
      out.at[0, "LTCMA line"] == PE and dirty
      and (out.at[0, "Return %"], out.at[0, "Vol %"]) == jpm.lookup("Private Equity"))
typed = out.copy(); typed.loc[0, ["Return %", "Vol %"]] = [14.0, 12.0]
out2, _ = pl.sync_group(jpm, out, typed, "ALT")
check("typed return and volatility are kept",
      (out2.at[0, "Return %"], out2.at[0, "Vol %"]) == (14.0, 12.0))
fi_prev = pd.DataFrame([{"LTCMA line": jpm.LABEL["U.S. Cash"], "Weight %": 5.0,
                         "Return %": 3.1, "Vol %": 0.67}])
fi_bad = fi_prev.copy(); fi_bad.loc[0, "LTCMA line"] = PE
out3, _ = pl.sync_group(jpm, fi_prev, fi_bad, "FI")
check("refused in a Fixed Income table (falls back to the default line)",
      out3.at[0, "LTCMA line"] == jpm.LABEL[jpm.DEFAULT_LINE["FI"]])
ref = pd.DataFrame([{"LTCMA line": "U.S. Cash", "Weight %": 50.0,
                     "Return %": 3.1, "Vol %": 0.67}])
check("ordinary lines behave exactly as the provider's own sync_group",
      pl.sync_group(jpm, ref, ref, "FI")[0].equals(jpm.sync_group(ref, ref, "FI")[0]))

print()
print("=" * 78)
print("3. OPTIMIZER UNIVERSE + ENGINE")
print("=" * 78)
alloc = pd.DataFrame([
    {"Category": "Fixed Income", "LTCMA line": jpm.LABEL["U.S. Cash"],
     "Weight %": 40.0, "Return %": 3.1, "Vol %": 0.67},
    {"Category": "Alternatives", "LTCMA line": PE,
     "Weight %": 35.0, "Return %": 14.0, "Vol %": 22.0},
    {"Category": "Alternatives", "LTCMA line": PC,
     "Weight %": 25.0, "Return %": 10.0, "Vol %": 9.0},
])
ua = uni.universe_alloc(alloc, src=jpm)
row = ua[ua["LTCMA line"] == PE]
check("held personal lines join the 'all asset classes' universe with the "
      "user's numbers", len(ua) == len(jpm.LINES) + 2 and len(row) == 1
      and float(row["Return %"].iloc[0]) == 14.0)
check("a personal line not held is not added to the universe",
      len(uni.universe_alloc(alloc.iloc[:1], src=jpm)) == len(jpm.LINES))
res, repaired = simulate(list(alloc["Weight %"]), list(alloc["Return %"]),
                         list(alloc["Vol %"]),
                         pl.corr_for(jpm, alloc["LTCMA line"]), 2000, 10, 7)
check("engine runs a portfolio holding both, no matrix repair",
      not repaired and res.rows[-1]["Median x"] > 1)

print()
print("RESULT:", "all checks pass" if ok else "FAILURES ABOVE — investigate")
raise SystemExit(0 if ok else 1)
