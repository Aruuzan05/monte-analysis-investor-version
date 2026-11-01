"""Checks for the Keepp Capital Monte Carlo.

Run after any change to engine.py, jpm_ltcma.py or the LTCMA data:

    python tests/test_engine.py

Covers three things: the engine still reproduces the signed-off Excel workbook,
J.P. Morgan's published correlation matrix loads and behaves, and the sleeve
table's sync rules hold.
"""
import numpy as np
import pandas as pd

import sys
from pathlib import Path

# Run as a script (python tests/test_x.py): put the repo root on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from keepp_mc import jpm_ltcma as J
from keepp_mc.engine import return_ladder, simulate

ok = True


def check(label, cond):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")


# ==========================================================================
print("=" * 78)
print("1. ENGINE — feed the workbook's own inputs, must reproduce the workbook")
print("=" * 78)
# The original 5-asset workbook: FI, EQ, PE, CMD, Cash, with its own 5x5.
W_XL = [45, 20, 10, 8, 17]
R_XL = [5.00, 12.88, 11.90, 6.10, 3.80]          # as the workbook had them
V_XL = [7.02, 14.47, 19.80, 18.30, 0.67]
C_XL = [[1.00, 0.68, 0.30, 0.10, 0.00],
        [0.68, 1.00, 0.80, 0.30, 0.00],
        [0.30, 0.80, 1.00, 0.25, 0.00],
        [0.10, 0.30, 0.25, 1.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 1.00]]
EXCEL = {3: dict(p50=7.589, pl=4.79, wm=1.2454),
         5: dict(p50=7.470, pl=1.32, wm=1.4336),
         10: dict(p50=7.517, pl=0.09, wm=2.0643)}

res, rep = simulate(W_XL, R_XL, V_XL, C_XL, 10000, 10, 20260722)
check(f"analytic return {res.exp_arith*100:.2f}% == workbook 7.80%",
      abs(res.exp_arith * 100 - 7.80) < 0.01)
check(f"analytic vol    {res.analytic_vol*100:.2f}% == workbook 7.40%",
      abs(res.analytic_vol * 100 - 7.40) < 0.01)
for r in res.rows:
    e = EXCEL[r["_years"]]
    check(f"{r['_years']:>2}y median {r['Median']*100:5.2f}% (xl {e['p50']:.2f}) · "
          f"wealth {r['Median x']:.4f} (xl {e['wm']:.4f})",
          abs(r["Median"] * 100 - e["p50"]) < 0.25 and abs(r["Median x"] - e["wm"]) < 0.02)

# ==========================================================================
print()
print("=" * 78)
print("2. CORRELATIONS — J.P. Morgan 2027 LTCMA, USD, as published")
print("=" * 78)
raw, C = J.CORR_RAW, J.CORR
check(f"matrix is {C.shape[0]}x{C.shape[1]} (58 investable lines + the "
      f"excluded Americas Core Real Estate + U.S. Inflation)",
      C.shape == (60, 60))
check("as published: symmetric", np.abs(raw - raw.T).max() == 0)
check("as published: unit diagonal", np.allclose(np.diag(raw), 1.0))
check("as published: all values within [-1, 1]", raw.min() >= -1 and raw.max() <= 1)
check(f"as published: NOT positive semi-definite (min eig {J.CORR_RAW_MIN_EIG:+.4f})",
      J.CORR_RAW_MIN_EIG < 0)
check(f"after Higham repair: positive semi-definite "
      f"(min eig {np.linalg.eigvalsh(C).min():+.2e})",
      np.linalg.eigvalsh(C).min() > -1e-10)
check(f"repair is cosmetic: largest cell moves {J.CORR_MAX_REPAIR:.4f} (< 0.01)",
      J.CORR_MAX_REPAIR < 0.01)
off = ~np.eye(len(C), dtype=bool)
check("repair moves no pair by 0.01 or more", (np.abs(C - raw)[off] < 0.01).all())
check("repaired matrix stays symmetric with a unit diagonal",
      np.allclose(C, C.T) and np.allclose(np.diag(C), 1.0))

names = {n for n, *_ in J.LINES}
published = names - set(J.HOUSE_LINES)
check(f"every one of the {len(published)} published lines has a matrix row",
      published <= set(J.CORR_NAMES))
# A house line has no row of its own by construction — that is what makes it
# a house line — so it must NOT be in the matrix, and the row it borrows must.
check(f"the {len(J.HOUSE_LINES)} house line(s) have no row of their own, and "
      f"the row each borrows exists",
      not (set(J.HOUSE_LINES) & set(J.CORR_NAMES))
      and all(p in J.CORR_NAMES for p in J.HOUSE_LINES.values()))
check("U.S. Inflation is in the matrix but is not an investable line",
      J.INFLATION_ROW in J.CORR_NAMES and J.INFLATION_ROW not in names)
# 2027 adds Americas Core Real Estate, which is U.S. Core Real Estate under a
# second name: same published return, vol and arithmetic figure, rho = 1.00.
# It keeps its matrix row and is kept out of the catalogue, so no allocation
# can hold the same real estate twice.
check("the excluded duplicate has a matrix row but is not an investable line",
      all(n in J.CORR_NAMES for n in J.EXCLUDED_DUPLICATES)
      and not (set(J.EXCLUDED_DUPLICATES) & names))
for dup, orig in J.EXCLUDED_DUPLICATES.items():
    check(f"{dup} is excluded because rho({dup}, {orig}) = "
          f"{raw[J.CORR_INDEX[dup], J.CORR_INDEX[orig]]:.2f}",
          abs(raw[J.CORR_INDEX[dup], J.CORR_INDEX[orig]] - 1.0) < 1e-9
          and J.BY_NAME[orig][:3] == (8.10, 8.69, 11.40))

# spot-check a few published values straight out of the CSV
for a, b, want in [("U.S. Large Cap", "AC World Equity", 0.97),
                   ("Private Equity", "Diversified Hedge Funds", 0.76),
                   ("U.S. Aggregate Bonds", "AC World Equity", 0.34),
                   ("Commodities", "Gold", 0.33),
                   ("Macro Hedge Funds", "U.S. Aggregate Bonds", -0.09)]:
    got = raw[J.CORR_INDEX[a], J.CORR_INDEX[b]]
    check(f"published {a} x {b} = {want:+.2f}", abs(got - want) < 1e-9)

# ==========================================================================
print()
print("=" * 78)
print("3. SUB-MATRICES — what the app actually simulates with")
print("=" * 78)
sub = J.corr_for(["U.S. Aggregate Bonds", "AC World Equity", "Private Equity"])
check("corr_for returns the right shape", sub.shape == (3, 3))
check("corr_for pulls published values",
      abs(sub[1, 2] - C[J.CORR_INDEX["AC World Equity"],
                        J.CORR_INDEX["Private Equity"]]) < 1e-12)
check("corr_for accepts starred labels and legacy prefixes",
      np.allclose(J.corr_for(["Private Equity ★", "ALT - Commodities"]),
                  J.corr_for(["Private Equity", "Commodities"])))

# Every principal sub-matrix of a PSD matrix is PSD, so no allocation the app
# can build should ever need repairing. Check that over many random draws.
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(300):
    k = int(rng.integers(2, 13))
    pick = list(rng.choice(J.CORR_NAMES, size=k, replace=False))
    worst = min(worst, float(np.linalg.eigvalsh(J.corr_for(pick)).min()))
check(f"300 random sub-matrices all PSD (worst min eig {worst:+.2e})", worst > -1e-10)

vols = [7.02, 14.47, 19.78]
_, repaired = simulate([50, 30, 20], [5.0, 7.0, 10.2], vols,
                       J.corr_for(["U.S. Aggregate Bonds", "AC World Equity",
                                   "Private Equity"]), 2000, 5, 1)
check("a real allocation simulates without triggering the repair path", not repaired)

# ==========================================================================
print()
print("=" * 78)
print("4. CATEGORIES — J.P. Morgan's own three")
print("=" * 78)
counts = {k: len(J.CHOICES[k]) for k in J.CATEGORY_KEYS}
check(f"Fixed Income {counts['FI']} · Equities {counts['EQ']} · "
      f"Alternatives {counts['ALT']}",
      counts == {"FI": 21, "EQ": 18, "ALT": 20})
check("59 lines across the three categories, none missing or double-counted",
      sum(counts.values()) == 59)
reachable = {J.resolve(o) for v in J.CHOICES.values() for o in v}
check("every line is reachable from its category dropdown", reachable == names)
for k in J.CATEGORY_KEYS:
    check(f"{J.CATEGORY_NAME[k]} dropdown offers only {k} lines and nothing else",
          all(J.category_of(o) == k for o in J.CHOICES[k]))
check("no dropdown offers an option that is not a line — every sleeve has a "
      "correlation row",
      all(J.lookup(o) is not None for v in J.CHOICES.values() for o in v))
check("each category's default line belongs to that category",
      all(J.category_of(J.DEFAULT_LINE[k]) == k for k in J.CATEGORY_KEYS))
check("the hedge fund lines all sit in Alternatives",
      all(J.category_of(n) == "ALT" for n in J.BY_NAME if "Hedge Fund" in n))
check("no cash line leaks into Equities",
      not any("Cash" in o for o in J.CHOICES["EQ"]))

# ==========================================================================
print()
print("=" * 78)
print("5. SLEEVE TABLE — sync rules")
print("=" * 78)
COLS = ["LTCMA line", "Weight %", "Return %", "Vol %"]


def grp(lines, w, r, v):
    return pd.DataFrame({"LTCMA line": [J.LABEL.get(x, x) for x in lines],
                         "Weight %": list(w), "Return %": list(r),
                         "Vol %": list(v)})


check("no sleeve can be anonymous — Manual entry is gone from the module",
      not hasattr(J, "MANUAL") and not hasattr(J, "proxy_choices"))
check("the table carries four columns and no derived status column",
      COLS == ["LTCMA line", "Weight %", "Return %", "Vol %"]
      and not hasattr(J, "source_of"))

# rule 2 — moving to a different line refills from the catalogue
p = grp(["AC World Equity"], [20], [12.88], [14.47])        # an overridden row
e = p.copy(); e.at[0, "LTCMA line"] = J.LABEL["EAFE Equity"]
o, d = J.sync_group(p, e, "EQ")
check("picking a different line refills Return/Vol, discarding the override",
      (o.at[0, "Return %"], o.at[0, "Vol %"]) == (7.70, 18.34) and d)

# rule 3 — the headline change: editing a number does NOT disturb the line
p = grp(["Private Equity"], [10], [9.80], [20.31])
e = p.copy(); e.at[0, "Return %"] = 14.0
o, d = J.sync_group(p, e, "ALT")
check("editing Return keeps the sleeve on its line",
      o.at[0, "LTCMA line"] == J.LABEL["Private Equity"]
      and o.at[0, "Return %"] == 14.0)
check("an overridden sleeve still uses its own line's correlation row",
      np.allclose(J.corr_for(list(o["LTCMA line"])), [[1.0]]))

# an overridden row must survive every later pass untouched
o2, d2 = J.sync_group(o, o.copy(), "ALT")
check("an overridden sleeve is stable across reruns and does not loop",
      o2.at[0, "Return %"] == 14.0
      and o2.at[0, "LTCMA line"] == J.LABEL["Private Equity"] and not d2)

# a weight change touches nothing else
p = grp(["Commodities"], [8], [4.60], [18.32])
e = p.copy(); e.at[0, "Weight %"] = 12.0
o, d = J.sync_group(p, e, "ALT")
check("a weight change leaves the line alone and does not rerun",
      o.at[0, "LTCMA line"] == J.LABEL["Commodities"] and not d)
o, d = J.sync_group(p, p.copy(), "ALT")
check("a no-op edit does not trigger a rerun loop", not d)

# several sleeves in one category
two = grp(["U.S. Large Cap", "Emerging Markets Equity"],
          [12, 8], [6.70, 7.80], [16.47, 20.93])
o, d = J.sync_group(two, two.copy(), "EQ")
check("two sleeves in one category are stable", len(o) == 2 and not d)

# rule 1 — a blank row from the editor
blank = two.copy()
blank.loc[len(blank)] = [None, float("nan"), float("nan"), float("nan")]
o, d = J.sync_group(two, blank, "EQ")
check("a new blank row lands on the category default at 0% with JPM numbers",
      d and o.at[2, "LTCMA line"] == J.LABEL[J.DEFAULT_LINE["EQ"]]
      and o.at[2, "Weight %"] == 0.0
      and (o.at[2, "Return %"], o.at[2, "Vol %"]) == J.lookup(J.DEFAULT_LINE["EQ"]))
check("adding a row does not disturb the rows above it",
      o.at[0, "LTCMA line"] == J.LABEL["U.S. Large Cap"]
      and o.at[1, "LTCMA line"] == J.LABEL["Emerging Markets Equity"])

# deleting a row must not reset the edits of the rows beneath it
three = grp(["U.S. Large Cap", "EAFE Equity", "Emerging Markets Equity"],
            [10, 10, 10], [6.70, 7.50, 9.99], [16.47, 17.63, 20.93])
after = three.drop(index=0).reset_index(drop=True)
o, d = J.sync_group(three, after, "EQ")
check("deleting a row leaves the survivors' lines and overrides intact",
      len(o) == 2 and o.at[0, "LTCMA line"] == J.LABEL["EAFE Equity"]
      and o.at[1, "LTCMA line"] == J.LABEL["Emerging Markets Equity"]
      and o.at[1, "Return %"] == 9.99)

# an out-of-category value cannot stick
stale = grp(["Diversified Hedge Funds"], [10], [5.30], [5.75])
o, d = J.sync_group(grp(["AC World Equity"], [10], [7.00], [16.78]), stale, "EQ")
check("an Alternatives line cannot stick in the Equities table",
      o.at[0, "LTCMA line"] == J.LABEL[J.DEFAULT_LINE["EQ"]] and d)

# a retired "Manual entry" row left in a saved session must recover
legacy = grp(["Manual entry"], [10], [5.00], [7.02])
o, d = J.sync_group(legacy, legacy.copy(), "FI")
check("a legacy Manual entry row falls back to the category default line",
      o.at[0, "LTCMA line"] == J.LABEL[J.DEFAULT_LINE["FI"]]
      and J.lookup(o.at[0, "LTCMA line"]) is not None and d)

# ==========================================================================
print()
print("=" * 78)
print("6. BASE CASE — the shipped default, on the published matrix")
print("=" * 78)
# These must stay a copy of app.py's BASE_SLEEVES. They had drifted: this
# fixture carried AC World Equity at 12.88 / 14.47 — the ORIGINAL WORKBOOK's
# equity figures — while the app ships Keepp's MSCI World history at
# 7.00 / 16.78, so the test pinned a portfolio the app never runs. The three
# J.P. Morgan sleeves are on the 2027 edition.
BASE = [("U.S. Inv Grade Corporate Bonds", 45, 5.00, 7.02),
        ("U.S. Cash", 17, 3.80, 0.67),
        ("AC World Equity", 20, 7.00, 16.78),
        ("Private Equity", 10, 9.80, 20.31),
        ("Diversified Hedge Funds", 0, 6.70, 5.94),
        ("Commodities", 8, 4.80, 17.97)]
bw = [b[1] for b in BASE]
br = [b[2] for b in BASE]
bv = [b[3] for b in BASE]
bc = J.corr_for([b[0] for b in BASE])
res, rep = simulate(bw, br, bv, bc, 10000, 10, 20260722)
t = res.rows[-1]
check(f"weights total 100% ({sum(bw)}%)", sum(bw) == 100)
check("no repair needed", not rep)
check(f"expected return {res.exp_arith*100:.2f}%  (unchanged by the matrix)",
      abs(res.exp_arith * 100 - 6.39) < 0.02)
check(f"portfolio volatility {res.analytic_vol*100:.2f}%  "
      f"(was 7.40% on the fabricated matrix)",
      abs(res.analytic_vol * 100 - 8.00) < 0.05)
print(f"       10y median {t['Median x']:.3f}x · P(loss) {t['P(loss)']*100:.2f}% · "
      f"5th pct {t['5th x']:.3f}x")

# ==========================================================================
print()
print("=" * 78)
print("7. RETURN LADDER — weights x returns, reconciled to the simulation")
print("=" * 78)
L = return_ladder(bw, br, bv, bc)
check(f"step 3 IS the app's card: ladder {L.arithmetic*100:.4f}% == "
      f"exp_arith {res.exp_arith*100:.4f}%",
      abs(L.arithmetic - res.exp_arith) < 1e-12)
check(f"and the vol it drags by IS the app's: {L.portfolio_vol*100:.4f}% == "
      f"{res.analytic_vol*100:.4f}%",
      abs(L.portfolio_vol - res.analytic_vol) < 1e-12)
check("the ladder adds up: simple + uplift - drag == compound",
      abs((L.weighted_simple + L.variance_uplift - L.variance_drag)
          - L.compound) < 1e-15)
check(f"the gap splits exactly in two: dispersion {L.dispersion*100:+.4f} + "
      f"diversification {L.diversification*100:+.4f} == "
      f"{(L.compound - L.weighted_simple)*100:+.4f}pp",
      abs(L.dispersion + L.diversification
          - (L.compound - L.weighted_simple)) < 1e-15)
check(f"weight_sum is as entered, not renormalised ({L.weight_sum:.0f}%)",
      L.weight_sum == sum(bw))
check("per-sleeve contributions total the first two steps",
      abs(L.per_sleeve[:, 0].sum() - L.weighted_simple) < 1e-15
      and abs(L.per_sleeve[:, 1].sum() - L.variance_uplift) < 1e-15)

# The headline claim the panel makes: the simulation can only land at or
# above a weighted average of the Return column, never below it. Both halves
# of the gap are non-negative by construction (Jensen twice), so this is a
# theorem, not a property of these inputs -- but check it on real ones.
rng = np.random.default_rng(20260722)
worst = 0.0
for _ in range(500):
    k = int(rng.integers(2, 13))
    pick = list(rng.choice(J.CORR_NAMES, size=k, replace=False))
    pick = [p for p in pick if p in J.BY_NAME]
    if len(pick) < 2:
        continue
    k = len(pick)
    ww = rng.dirichlet(np.ones(k)) * 100
    rr = [J.BY_NAME[p][0] for p in pick]
    vv = [J.BY_NAME[p][2] for p in pick]
    Lx = return_ladder(ww, rr, vv, J.corr_for(pick))
    worst = min(worst, Lx.dispersion, Lx.diversification)
check(f"on 500 random catalogue portfolios neither half of the gap is ever "
      f"negative (most negative {worst:+.2e})", worst > -1e-15)

# The two cases where a weighted average IS the right answer.
one = return_ladder([100.0], [7.0], [16.78], [[1.0]])
check("one holding: uplift and drag cancel, so the weighted average is exact",
      abs(one.compound - one.weighted_simple) < 1e-15
      and abs(one.dispersion) < 1e-15 and abs(one.diversification) < 1e-15)
lock = return_ladder([50.0, 50.0], [7.0, 5.0], [16.78, 16.78],
                     [[1.0, 1.0], [1.0, 1.0]])
check("identical vols at rho = 1.00: also exact, nothing to diversify",
      abs(lock.compound - lock.weighted_simple) < 1e-15)
spread = return_ladder([50.0, 50.0], [7.0, 5.0], [16.78, 0.67],
                       [[1.0, 1.0], [1.0, 1.0]])
check(f"unequal vols at rho = 1.00: the gap is dispersion ONLY "
      f"({spread.dispersion*100:+.3f}pp, diversification "
      f"{spread.diversification*100:+.3f}pp)",
      spread.dispersion > 1e-6 and abs(spread.diversification) < 1e-15)
same = return_ladder([50.0, 50.0], [7.0, 5.0], [16.78, 16.78],
                     [[1.0, 0.0], [0.0, 1.0]])
check(f"identical vols at rho = 0.00: the gap is diversification ONLY "
      f"({same.diversification*100:+.3f}pp, dispersion "
      f"{same.dispersion*100:+.3f}pp)",
      same.diversification > 1e-6 and abs(same.dispersion) < 1e-15)

# And the simulation really does land there, not just the algebra.
sim_med = res.rows[-1]["Median"]
check(f"the 10y simulated median {sim_med*100:.3f}% sits on the ladder's "
      f"compound {L.compound*100:.3f}% (gap {(sim_med-L.compound)*100:+.3f}pp, "
      f"sampling noise at 10,000 paths)",
      abs(sim_med - L.compound) < 0.001)
check(f"and it is above the weighted average {L.weighted_simple*100:.3f}%",
      sim_med > L.weighted_simple)

# ==========================================================================
print()
print("=" * 78)
print("8. HOUSE LINES — in the catalogue, not in the published tables")
print("=" * 78)
# Digital Assets is the only one. J.P. Morgan published compound 10.0%,
# arithmetic 20.9% and vol 55%, which do not reconcile under this app's
# convention; we take the vol IMPLIED by their own two return figures so the
# app reproduces their arithmetic instead of contradicting it.
g_d, a_d, v_d = J.BY_NAME["Digital Assets"][:3]
check(f"Digital Assets carries J.P. Morgan's compound return ({g_d}%)",
      g_d == 10.00)
check(f"its vol is the one implied by their own 10.0 / 20.9 pair "
      f"({v_d}% vs sqrt(2*(20.9-10.0)) = {np.sqrt(2*(a_d-g_d))*10:.2f}%)",
      abs(v_d - np.sqrt(2 * (a_d - g_d)) * 10) < 0.01)
check(f"so the trio reconciles under this app's own convention: "
      f"{g_d} + {v_d}^2/2 = {g_d + v_d**2/200:.4f}% vs published {a_d}%",
      abs(g_d + v_d**2 / 200 - a_d) < 0.01)
check("J.P. Morgan's published 55% is NOT what the app uses",
      abs(v_d - 55.0) > 5)
check("it is marked with a dagger, and resolve() strips it",
      J.LABEL["Digital Assets"].endswith("†")
      and J.resolve(J.LABEL["Digital Assets"]) == "Digital Assets"
      and J.lookup(J.LABEL["Digital Assets"]) == (g_d, v_d))
check("the prior-year column is nan, not an invented number",
      J.BY_NAME["Digital Assets"][3] != J.BY_NAME["Digital Assets"][3])

# The borrowed row: identical to Gold's against every other line.
both = J.corr_for(["Digital Assets", "Gold"])
check(f"Digital Assets x Gold is SELF_CORR {J.SELF_CORR}, not 1.00 — a 1.00 "
      f"would make the matrix singular",
      both[0, 1] == J.SELF_CORR and both[1, 0] == J.SELF_CORR)
check("two sleeves on the SAME line are still 1.00 — they are one asset held "
      "twice, which is not the same situation",
      J.corr_for(["Gold", "Gold"])[0, 1] == 1.0)
others = [n for n, *_ in J.LINES if n not in ("Digital Assets", "Gold")][:12]
d_row = J.corr_for(["Digital Assets"] + others)[0, 1:]
g_row = J.corr_for(["Gold"] + others)[0, 1:]
check("against every other line it IS Gold's row, unmodified",
      np.allclose(d_row, g_row))

for sel in (["Digital Assets"],
            ["Digital Assets", "Gold"],
            ["Digital Assets", "Gold", "Commodities", "U.S. Large Cap",
             "U.S. Cash"]):
    C = J.corr_for(sel)
    try:
        np.linalg.cholesky(C)
        factorises = True
    except np.linalg.LinAlgError:
        factorises = False
    check(f"holding {len(sel)} line(s) incl. Digital Assets factorises without "
          f"repair (min eig {np.linalg.eigvalsh(C).min():+.5f})", factorises)

# It must also simulate, with no repair flag raised.
dres, drep = simulate([45, 15, 20, 10, 0, 8, 2],
                      [5.00, 3.80, 7.00, 9.80, 6.70, 4.80, g_d],
                      [7.02, 0.67, 16.78, 20.31, 5.94, 17.97, v_d],
                      J.corr_for(["U.S. Inv Grade Corporate Bonds", "U.S. Cash",
                                  "AC World Equity", "Private Equity",
                                  "Diversified Hedge Funds", "Commodities",
                                  "Digital Assets"]), 10000, 10, 20260722)
check(f"Base + 2% digital funded from cash simulates with no repair: "
      f"return {dres.exp_arith*100:.2f}%, vol {dres.analytic_vol*100:.2f}%, "
      f"10y median wealth {dres.rows[-1]['Median x']:.4f}x", not drep)
check("and it raises both the return and the risk against Base's 6.39 / 8.00",
      dres.exp_arith * 100 > 6.39 and dres.analytic_vol * 100 > 8.00)

print()
print("RESULT:", "all checks pass" if ok else "FAILURES ABOVE — investigate")
raise SystemExit(0 if ok else 1)
