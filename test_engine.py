"""Reconcile the engine against Keepp_Capital_Target_Allocation_Monte_Carlo.xlsx.

Run after any change to engine.py or to the base assumptions:  python test_engine.py
"""
import numpy as np
from engine import simulate

# Order: FI, EQ, PE, HF, CMD, CASH  (must match app.py)
CORR = [[1.00, 0.68, 0.30, 0.35, 0.10, 0.00],
        [0.68, 1.00, 0.80, 0.75, 0.30, 0.00],
        [0.30, 0.80, 1.00, 0.65, 0.25, 0.00],
        [0.35, 0.75, 0.65, 1.00, 0.35, 0.00],
        [0.10, 0.30, 0.25, 0.35, 1.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 1.00]]
W = [45, 20, 10, 0, 8, 17]
R = [5.00, 12.88, 11.90, 5.30, 6.10, 3.80]
V = [7.02, 14.47, 19.80, 6.50, 18.30, 0.67]

# Workbook values, Base sheet
EXCEL = {3:  dict(p50=7.589, pl=4.79, wm=1.2454),
         5:  dict(p50=7.470, pl=1.32, wm=1.4336),
         10: dict(p50=7.517, pl=0.09, wm=2.0643)}

print("=" * 78)
print("BASE CASE — hedge funds at 0% weight, must reproduce the workbook")
print("=" * 78)
res, rep = simulate(W, R, V, CORR, 10000, 10, 20260722)
print(f"  {res.elapsed_ms:.0f} ms | corr repaired: {rep} (must be False)")
print(f"  analytic arith {res.exp_arith*100:.2f}%  (workbook 7.80%)")
print(f"  analytic vol   {res.analytic_vol*100:.2f}%  (workbook 7.40%)")
ok = True
for r in res.rows:
    e = EXCEL[r["_years"]]
    d_p50 = abs(r["Median"] * 100 - e["p50"])
    d_wm = abs(r["Median x"] - e["wm"])
    flag = "OK " if (d_p50 < 0.25 and d_wm < 0.02) else "OFF"
    ok &= flag == "OK "
    print(f"  {flag} {r['_years']:>2}y  median {r['Median']*100:6.2f}% (xl {e['p50']:5.2f})"
          f"  P(loss) {r['P(loss)']*100:5.2f}% (xl {e['pl']:4.2f})"
          f"  wealth {r['Median x']:.4f} (xl {e['wm']:.4f})")

print("\n" + "=" * 78)
print("HEDGE FUND SLEEVE — 10% funded from fixed income")
print("=" * 78)
W2 = [35, 20, 10, 10, 8, 17]
res2, _ = simulate(W2, R, V, CORR, 10000, 10, 20260722)
print(f"  analytic arith {res2.exp_arith*100:.2f}%  vol {res2.analytic_vol*100:.2f}%")
print(f"  10y median wealth {res2.rows[-1]['Median x']:.4f}"
      f"  P(loss) {res2.rows[-1]['P(loss)']*100:.2f}%")

print("\n" + "=" * 78)
print("EDGE CASES")
print("=" * 78)
a, _ = simulate(W, R, V, CORR, 5000, 10, 7)
b, _ = simulate(W, R, V, CORR, 5000, 10, 7)
print(f"  seed reproducible: {a.rows[-1]['Median x'] == b.rows[-1]['Median x']}")
bad = [row[:] for row in CORR]
bad[0][2] = bad[2][0] = -0.9          # inconsistent with FI-EQ 0.68 / EQ-PE 0.80
r3, rep3 = simulate(W, R, V, bad, 2000, 5, 1)
print(f"  non-PSD matrix repaired: {rep3} | finite result: "
      f"{np.isfinite(r3.rows[-1]['Median x'])}")
c, _ = simulate([0, 0, 0, 100, 0, 0], R, V, CORR, 5000, 10, 3)
print(f"  100% hedge funds: vol {c.analytic_vol*100:.2f}% (input 6.50%), "
      f"10y median {c.rows[-1]['Median']*100:.2f}% (input 5.30%)")

print("\nRESULT:", "base case reconciles to the workbook" if ok else "MISMATCH — investigate")
