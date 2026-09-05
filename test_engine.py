"""Reconcile the engine against Keepp_Capital_Target_Allocation_Monte_Carlo.xlsx.

Run after any change to engine.py or to the base assumptions:  python test_engine.py
"""
import numpy as np
from engine import expand_correlation, simulate

# Order: FI, EQ, PE, HF, CMD, CASH  (must match app.py)
CORR = [[1.00, 0.68, 0.30, 0.35, 0.10, 0.00],
        [0.68, 1.00, 0.80, 0.75, 0.30, 0.00],
        [0.30, 0.80, 1.00, 0.65, 0.25, 0.00],
        [0.35, 0.75, 0.65, 1.00, 0.35, 0.00],
        [0.10, 0.30, 0.25, 0.35, 1.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 1.00]]
W = [45, 20, 10, 0, 8, 17]
# Corrected base: JPM's COMPOUND column for PE and Commodities, and JPM's
# published hedge fund volatility.
R = [5.00, 12.88, 10.20, 5.30, 4.60, 3.80]
V = [7.02, 14.47, 19.78, 5.75, 18.32, 0.67]
# What the original workbook fed in: JPM's ARITHMETIC column for PE and CMD.
R_WORKBOOK = [5.00, 12.88, 11.90, 5.30, 6.10, 3.80]
V_WORKBOOK = [7.02, 14.47, 19.80, 6.50, 18.30, 0.67]

# Workbook values, Base sheet
EXCEL = {3:  dict(p50=7.589, pl=4.79, wm=1.2454),
         5:  dict(p50=7.470, pl=1.32, wm=1.4336),
         10: dict(p50=7.517, pl=0.09, wm=2.0643)}

print("=" * 78)
print("ENGINE CHECK — feed the workbook's own inputs, must reproduce the workbook")
print("=" * 78)
res, rep = simulate(W, R_WORKBOOK, V_WORKBOOK, CORR, 10000, 10, 20260722)
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
print("CORRECTED BASE — compound returns for PE and Commodities")
print("=" * 78)
resc, _ = simulate(W, R, V, CORR, 10000, 10, 20260722)
tc = resc.rows[-1]
print(f"  analytic arith {resc.exp_arith*100:.2f}%  vol {resc.analytic_vol*100:.2f}%")
print(f"  10y median wealth {tc['Median x']:.4f}  P(loss) {tc['P(loss)']*100:.2f}%")
print("  (workbook-input version above reads 7.80% / 2.06x — the gap is the "
      "arithmetic-fed-as-geometric error)")

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
print(f"  100% hedge funds: vol {c.analytic_vol*100:.2f}% (input {V[3]:.2f}%), "
      f"10y median {c.rows[-1]['Median']*100:.2f}% (input {R[3]:.2f}%)")

# The catalogue must stay consistent with what the app offers.
import jpm_ltcma as J
assert len(J.LINES) == 58, "LTCMA catalogue should hold 58 lines"
assert len({n for n, *_ in J.LINES}) == 58, "duplicate line name in the catalogue"
for key, opts in J.CHOICES.items():
    bad = [n for n in opts if n != J.MANUAL and J.lookup(n) is None]
    assert not bad, f"{key} references unresolvable lines: {bad}"
    assert opts[0] == J.MANUAL, f"{key} must offer Manual entry first"
# both the prefixed dropdown label and the bare name must resolve identically
# every spelling of the same line must resolve identically
assert (J.lookup("Private Equity")
        == J.lookup("ALT - Private Equity")          # pre-star label
        == J.lookup("ALT - Private Equity ★")        # current label
        == (10.20, 19.78))
assert J.lookup("Diversified Hedge Funds") == (5.30, 5.75)
assert J.lookup("Emerging Markets Equity") == (7.80, 20.93)      # EQ line
assert J.lookup("Emerging Markets Sovereign Debt") == (6.30, 8.83)  # FI line, same prefix
assert J.lookup(J.MANUAL) is None and J.lookup("not a line") is None
assert len(J.LABEL) == 58 and len(J.NAME_OF) == 58, "label map must be 1:1"
assert len(J.EXPANSION) == 14, "the brief lists 14 expansion-proposal lines"
assert J.EXPANSION <= set(J.BY_NAME), "expansion names must all exist"
assert J.LABEL["U.S. Large Cap"].endswith("\u2605"), "expansion lines carry a star"
assert not J.LABEL["U.S. Mid Cap"].endswith("\u2605"), "others must not"
# the star is a suffix, so lookup still resolves and filtering on the name works
assert J.lookup(J.LABEL["Private Equity"]) == (10.20, 19.78)
print(f"  LTCMA catalogue: {len(J.LINES)} lines, "
      f"{sum(len(v)-1 for v in J.CHOICES.values())} bucket options, all resolve")

# --- the per-group sleeve sync rules ---------------------------------------
import pandas as pd
COLS = ["LTCMA line", "Weight %", "Return %", "Vol %"]


def _grp(lines, w, r, v):
    return pd.DataFrame({"LTCMA line": list(lines), "Weight %": list(w),
                         "Return %": list(r), "Vol %": list(v)})


# every group offers Manual entry plus only its own lines
for k in J.GROUP_KEYS:
    opts = J.CHOICES[k]
    assert opts[0] == J.MANUAL, f"{k} must offer Manual entry first"
    assert len(set(opts)) == len(opts), f"{k} has duplicate options"
# the point of the change: the hedge fund list is hedge funds only
hf = [o for o in J.CHOICES["HF"] if o != J.MANUAL]
assert all("Hedge Fund" in o for o in hf) and len(hf) == 5, hf
assert not any("Cash" in o for o in J.CHOICES["HF"])
assert not any("Cash" in o for o in J.CHOICES["EQ"])
assert all(J.family_of(o) == "EQ" for o in J.CHOICES["EQ"] if o != J.MANUAL)
assert all(J.family_of(o) == "FI" for o in J.CHOICES["FI"] if o != J.MANUAL)
# and every one of the 58 lines is reachable from at least one group
reachable = {J.NAME_OF[o] for v in J.CHOICES.values() for o in v if o != J.MANUAL}
assert reachable == {n for n, *_ in J.LINES}, "some LTCMA line is unreachable"

# picking a line fills Return and Vol from the compound column
p = _grp([J.MANUAL], [20], [12.88], [14.47])
e = _grp([J.LABEL["AC World Equity"]], [20], [12.88], [14.47])
o, d = J.sync_group(p, e, "EQ")
assert (o.at[0, "Return %"], o.at[0, "Vol %"]) == (7.00, 16.78) and d

# typing over a number un-labels the JPM provenance
p = _grp([J.LABEL["Private Equity"]], [10], [10.20], [19.78])
e = p.copy(); e.at[0, "Return %"] = 14.0
o, d = J.sync_group(p, e, "PE")
assert o.at[0, "LTCMA line"] == J.MANUAL and o.at[0, "Return %"] == 14.0

# a weight change leaves the label alone and is a no-op rerun-wise
p = _grp([J.LABEL["Private Equity"]], [10], [10.20], [19.78])
e = p.copy(); e.at[0, "Weight %"] = 15.0
o, d = J.sync_group(p, e, "PE")
assert o.at[0, "LTCMA line"] == J.LABEL["Private Equity"] and not d
o, d = J.sync_group(p, p.copy(), "PE")
assert not d, "a no-op edit must not trigger a rerun loop"

# two sleeves inside one group is stable
two = _grp([J.LABEL["U.S. Large Cap"], J.LABEL["Emerging Markets Equity"]],
           [12, 8], [6.70, 7.80], [16.47, 20.93])
o, d = J.sync_group(two, two.copy(), "EQ")
assert len(o) == 2 and not d

# a row the editor just added arrives with NaNs
blank = two.copy()
blank.loc[len(blank)] = [None, float("nan"), float("nan"), float("nan")]
o, d = J.sync_group(two, blank, "EQ")
assert d and o.at[2, "LTCMA line"] == J.MANUAL and o.at[2, "Weight %"] == 0.0

# a line that is not in this group falls back to Manual rather than being kept
stale = _grp([J.LABEL["Diversified Hedge Funds"]], [10], [5.30], [5.75])
o, d = J.sync_group(_grp([J.MANUAL], [10], [0], [0]), stale, "EQ")
assert o.at[0, "LTCMA line"] == J.MANUAL and d, "out-of-group line must not stick"
print("  group scoping: HF list is hedge funds only, all 58 lines reachable, "
      "sync rules hold")

# --- multi-sleeve: several LTCMA lines inside one general asset type -------
SHORT_ = ["Fixed Income", "Public Equities", "Private Equity",
          "Hedge Funds", "Commodities", "Cash"]
WITHIN = [0.85, 0.85, 0.75, 0.80, 0.45, 0.90]

# one sleeve per type must reproduce the bucket matrix exactly
assert np.allclose(expand_correlation([0, 1, 2, 3, 4, 5], CORR, WITHIN),
                   np.array(CORR)), "single-sleeve case must be a no-op"

# INVARIANT: splitting a sleeve in two, same line, within-correlation 1.0,
# must be mathematically identical to the unsplit portfolio.
one, _ = simulate(W, R, V, expand_correlation([0, 1, 2, 3, 4, 5], CORR, [1.0] * 6),
                  4000, 10, 11)
split_w = [45, 12, 8, 10, 0, 8, 17]
split_r = [5.00, 12.88, 12.88, 10.20, 5.30, 4.60, 3.80]
split_v = [7.02, 14.47, 14.47, 19.78, 5.75, 18.32, 0.67]
buckets = [0, 1, 1, 2, 3, 4, 5]
two, _ = simulate(split_w, split_r, split_v,
                  expand_correlation(buckets, CORR, [1.0] * 6), 4000, 10, 11)
assert abs(one.analytic_vol - two.analytic_vol) < 1e-12, "split changed the vol"
assert abs(one.exp_arith - two.exp_arith) < 1e-12, "split changed the return"

# a realistic split: 12% U.S. Large Cap + 8% EM Equity inside Public Equities
real_r = [5.00, 6.70, 7.80, 10.20, 5.30, 4.60, 3.80]
real_v = [7.02, 16.47, 20.93, 19.78, 5.75, 18.32, 0.67]
C = expand_correlation(buckets, CORR, WITHIN)
assert C[1, 2] == 0.85, "two equity sleeves take the within-type figure"
assert C[0, 1] == 0.68, "FI-to-equity keeps the published between-type figure"
assert np.allclose(C, C.T) and np.allclose(np.diag(C), 1.0)
assert np.linalg.eigvalsh(C).min() > 0, "realistic split must stay PSD"
mix, rep_mix = simulate(split_w, real_r, real_v, C, 4000, 10, 11)
assert not rep_mix, "realistic split should not need repair"
print(f"  multi-sleeve: split invariant holds; 12% US Large Cap + 8% EM Equity "
      f"-> {mix.exp_arith*100:.2f}% / {mix.analytic_vol*100:.2f}% vol")

print("\nRESULT:", "engine reconciles to the workbook" if ok else "MISMATCH — investigate")
