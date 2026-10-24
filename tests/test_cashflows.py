"""Checks for contributions and withdrawals (keepp_mc/cashflows.py).

    python tests/test_cashflows.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from keepp_mc import cashflows as cf
from keepp_mc import jpm_ltcma as jpm
from keepp_mc.engine import simulate

ok = True


def check(label, cond):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")


LINES = [("U.S. Inv Grade Corporate Bonds", 45, 5.00, 7.02),
         ("U.S. Cash", 17, 3.80, 0.67), ("AC World Equity", 20, 7.00, 16.78),
         ("Private Equity", 10, 10.20, 19.78), ("Commodities", 8, 4.60, 18.32)]
W = [x[1] for x in LINES]; R = [x[2] for x in LINES]; V = [x[3] for x in LINES]
C = jpm.corr_for([x[0] for x in LINES])
ARGS = (W, R, V, C)


def run(flows, years=20, timing=cf.END, amount=1e6, paths=4000):
    return cf.simulate_cashflows(*ARGS, amount, flows, paths, years, 20260722, timing)


print("=" * 78)
print("1. NO CASH FLOWS = THE MAIN ENGINE, EXACTLY")
print("=" * 78)
main, _ = simulate(W, R, V, C, 4000, 20, 20260722)
none = run([])
for k, (col, q) in {"5th": (0, "5th x"), "Median": (2, "Median x"),
                    "95th": (4, "95th x")}.items():
    check(f"{k} ending value matches engine.simulate() to the cent",
          abs(none.value_pct[-1, col] - main.rows[-1][q] * 1e6) < 0.01)
check("nothing runs out, nothing paid", not none.depleted.any()
      and none.paid_in == 0 and none.paid_out_planned == 0)
check("investment growth rate equals the main tab's median rate",
      abs(none.twr_median - main.rows[-1]["Median"]) < 1e-12)

print()
print("=" * 78)
print("2. THE SCHEDULE")
print("=" * 78)
s = cf.schedule([cf.Flow(cf.WITHDRAWAL, 1000, cf.MONTHLY, 2, 3)], 5)
check("monthly withdrawal: months 13-36 only, negative",
      (s[12:36] == -1000).all() and (s[:12] == 0).all() and (s[36:] == 0).all())
s = cf.schedule([cf.Flow(cf.CONTRIBUTION, 50_000, cf.YEARLY, 1, 2)], 3)
check("yearly, end of period: paid in month 12 and 24",
      s[11] == 50_000 and s[23] == 50_000 and s.sum() == 100_000)
s = cf.schedule([cf.Flow(cf.CONTRIBUTION, 50_000, cf.YEARLY, 1, 2)], 3, cf.START)
check("yearly, start of period: paid in month 1 and 13", s[0] == s[12] == 50_000)
s = cf.schedule([cf.Flow(cf.WITHDRAWAL, 50_000, cf.YEARLY, 2, 3, 2.5)], 3)
check("growth applies from today: year 2 pays x1.025, year 3 x1.025^2",
      np.isclose(s[23], -51_250) and np.isclose(s[35], -50_000 * 1.025 ** 2))
check("rows past the horizon are ignored",
      cf.schedule([cf.Flow(cf.WITHDRAWAL, 1, cf.YEARLY, 1, 30)], 5).sum() == -5)

print()
print("=" * 78)
print("3. BEHAVIOUR")
print("=" * 78)
wd = run([cf.Flow(cf.WITHDRAWAL, 6000, cf.MONTHLY, 1, 20)])
check("7.2% a year withdrawn: some futures run out (between 5% and 40%)",
      0.05 < wd.depleted.mean() < 0.40)
check("a depleted future stays at $0", (wd.final[wd.depleted] == 0).all())
check("chance of running out never falls over time",
      (np.diff(wd.p_depleted) >= -1e-12).all())
check("paid out on average is less than planned when futures run out",
      wd.paid_out_mean < wd.paid_out_planned)
check("withdrawals do not change the investments' own growth rate",
      wd.twr_median == none.twr_median)
infl = run([cf.Flow(cf.WITHDRAWAL, 6000, cf.MONTHLY, 1, 20, 2.5)])
check("indexing withdrawals to inflation makes running out more likely",
      infl.depleted.mean() > wd.depleted.mean())
ct = run([cf.Flow(cf.CONTRIBUTION, 5000, cf.MONTHLY, 1, 20)])
check("contributions raise every percentile", (ct.value_pct[-1] > none.value_pct[-1]).all())
st_ = run([cf.Flow(cf.WITHDRAWAL, 6000, cf.MONTHLY, 1, 20)], timing=cf.START)
check("withdrawing at the start of each month is worse than at the end",
      st_.value_pct[-1, 2] < wd.value_pct[-1, 2])
big = run([cf.Flow(cf.WITHDRAWAL, 2e6, cf.YEARLY, 1, 1)], years=3)
check("a withdrawal larger than the portfolio empties every future at once",
      big.depleted.all() and np.allclose(big.paid_out_mean, big.paid_out_mean))

print()
print("RESULT:", "all checks pass" if ok else "FAILURES ABOVE — investigate")
raise SystemExit(0 if ok else 1)
