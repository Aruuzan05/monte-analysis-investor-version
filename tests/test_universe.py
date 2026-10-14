"""Checks for whole-universe optimization (universe.py).

Run after any change to universe.py, optimizer.py or the LTCMA data:

    python tests/test_universe.py

Covers four things: the 58-line universe frame carries the user's own numbers,
the tangency portfolio refined on a precomputed frontier agrees with
optimizer.max_sharpe(), the category/line breakdown adds up and hides nothing,
and applying an optimum produces an allocation the app will actually accept.
"""
import numpy as np
import pandas as pd

import sys
from pathlib import Path

# Run as a script (python tests/test_x.py): put the repo root on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from keepp_mc import jpm_ltcma as J
from keepp_mc import optimizer as opt
from keepp_mc import universe as U

ok = True


def check(label, cond):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")


# The app's Base case: six sleeves, three of them carrying Keepp's own Return
# and Vol rather than J.P. Morgan's.
BASE = pd.DataFrame([
    {"Category": "Fixed Income", "LTCMA line": J.LABEL["U.S. Inv Grade Corporate Bonds"],
     "Weight %": 45.0, "Return %": 5.00, "Vol %": 7.02},
    {"Category": "Fixed Income", "LTCMA line": J.LABEL["U.S. Cash"],
     "Weight %": 17.0, "Return %": 3.80, "Vol %": 0.67},
    {"Category": "Equities", "LTCMA line": J.LABEL["AC World Equity"],
     "Weight %": 20.0, "Return %": 12.88, "Vol %": 14.47},
    {"Category": "Alternatives", "LTCMA line": J.LABEL["Private Equity"],
     "Weight %": 10.0, "Return %": 10.20, "Vol %": 19.78},
    {"Category": "Alternatives", "LTCMA line": J.LABEL["Diversified Hedge Funds"],
     "Weight %": 0.0, "Return %": 5.30, "Vol %": 5.75},
    {"Category": "Alternatives", "LTCMA line": J.LABEL["Commodities"],
     "Weight %": 8.0, "Return %": 4.60, "Vol %": 18.32},
])

# ==========================================================================
print("=" * 78)
print("1. UNIVERSE FRAME — every investable line, with the user's own numbers")
print("=" * 78)
uni = U.universe_alloc(BASE)
check(f"one row per investable line ({len(uni)} rows)", len(uni) == 58)
check("58 investable lines, not the matrix's 59 — U.S. Inflation is not one",
      len(U.INVESTABLE) == 58 and J.INFLATION_ROW not in U.INVESTABLE)
check("the frame is in app.py's own allocation shape",
      list(uni.columns) == U.ALLOC_COLS)
check("every line resolves to a real catalogue line",
      all(J.resolve(x) in J.BY_NAME for x in uni["LTCMA line"]))
check("every line has a published correlation row",
      U.INVESTABLE == [J.resolve(x) for x in uni["LTCMA line"]]
      and J.corr_for(list(uni["LTCMA line"])).shape == (58, 58))

by_line = {J.resolve(r["LTCMA line"]): r for _, r in uni.iterrows()}
check("weights total what the current allocation totals (100%)",
      abs(uni["Weight %"].sum() - 100.0) < 1e-9)
check("a held line keeps its weight",
      by_line["U.S. Inv Grade Corporate Bonds"]["Weight %"] == 45.0)
check("an unheld line arrives at 0%",
      by_line["Japanese Equity"]["Weight %"] == 0.0)
check("a Keepp-sourced sleeve keeps the USER's return, not J.P. Morgan's "
      "(12.88 not 7.00 for AC World Equity)",
      by_line["AC World Equity"]["Return %"] == 12.88
      and J.BY_NAME["AC World Equity"][0] == 7.00)
check("a Keepp-sourced sleeve keeps the user's vol too",
      by_line["U.S. Cash"]["Vol %"] == 0.67
      and by_line["U.S. Inv Grade Corporate Bonds"]["Return %"] == 5.00)
check("an unheld line takes J.P. Morgan's published compound return and vol",
      (by_line["Japanese Equity"]["Return %"],
       by_line["Japanese Equity"]["Vol %"]) == (8.80, 15.77))
check("categories match the catalogue's own",
      all(J.CATEGORY_NAME[J.category_of(r["LTCMA line"])] == r["Category"]
          for _, r in uni.iterrows()))

# two sleeves on one line (possible in Custom) must merge, not duplicate
dup = pd.concat([BASE, pd.DataFrame([
    {"Category": "Equities", "LTCMA line": J.LABEL["AC World Equity"],
     "Weight %": 5.0, "Return %": 9.99, "Vol %": 11.11}])], ignore_index=True)
uni_dup = U.universe_alloc(dup)
d_line = uni_dup[uni_dup["LTCMA line"] == J.LABEL["AC World Equity"]].iloc[0]
check("two sleeves on one line still produce one row", len(uni_dup) == 58)
check("their weights are summed (20 + 5 = 25)", d_line["Weight %"] == 25.0)
check("the first occurrence supplies Return/Vol — no invented average",
      d_line["Return %"] == 12.88 and d_line["Vol %"] == 14.47)

# blank / NaN cells must not poison the sums
blank = BASE.copy()
blank.at[1, "Weight %"] = float("nan")
check("a blank weight counts as 0%, not NaN",
      abs(U.universe_alloc(blank)["Weight %"].sum() - 83.0) < 1e-9)

# a line that is not in the catalogue at all is skipped rather than crashing
junk = pd.concat([BASE, pd.DataFrame([
    {"Category": "Equities", "LTCMA line": "Not A Real Line",
     "Weight %": 5.0, "Return %": 1.0, "Vol %": 1.0}])], ignore_index=True)
check("an unknown line is ignored rather than crashing the frame",
      len(U.universe_alloc(junk)) == 58
      and abs(U.universe_alloc(junk)["Weight %"].sum() - 100.0) < 1e-9)

# ==========================================================================
print()
print("=" * 78)
print("2. TANGENCY ON A PRECOMPUTED FRONTIER — must agree with max_sharpe()")
print("=" * 78)
# Small problem: the reference max_sharpe() is affordable here, so the two can
# be compared directly. That is the whole point of this check.
S_NAMES = ["U.S. Cash", "U.S. Aggregate Bonds", "U.S. High Yield Bonds",
           "AC World Equity", "Emerging Markets Equity", "Private Equity",
           "Commodities"]
s_mu, s_vols = opt.inputs_from_percent([J.BY_NAME[n][0] for n in S_NAMES],
                                       [J.BY_NAME[n][2] for n in S_NAMES])
s_cov = opt.covariance(s_vols, J.corr_for(S_NAMES))
s_groups = {}
for i, n in enumerate(S_NAMES):
    s_groups.setdefault(J.CATEGORY_NAME[J.category_of(n)], []).append(i)
s_groups = {k: np.array(v) for k, v in s_groups.items()}
n_s = len(S_NAMES)
s_lim = opt.Limits(np.zeros(n_s), np.full(n_s, 0.60), s_groups,
                   {k: (0.0, 1.0) for k in s_groups})
s_w0 = opt.feasible_start(s_lim)
RF = 0.038

ref_w, ref_r, ref_v, ref_s = opt.max_sharpe(s_mu, s_cov, s_lim, s_w0, RF)
fr24 = opt.efficient_frontier(s_mu, s_cov, s_lim, s_w0, 24)
new_w, new_s = U.tangency_on_frontier(s_mu, s_cov, s_lim, fr24, RF)
grid_s = float(np.nanmax((fr24[0] - RF) / np.where(fr24[1] > 0, fr24[1], np.nan)))
check(f"refined Sharpe {new_s:.6f} matches max_sharpe's {ref_s:.6f} "
      f"(gap {abs(new_s - ref_s):.2e})", abs(new_s - ref_s) < 1e-5)
check(f"and beats the raw 24-point grid it started from ({grid_s:.6f})",
      new_s >= grid_s - 1e-12)
check(f"weights match max_sharpe's to under 0.5pp "
      f"(worst {np.abs(new_w - ref_w).max()*100:.3f}pp)",
      np.abs(new_w - ref_w).max() < 0.005)
check("the answer is a valid portfolio: long-only, sums to 100%",
      abs(new_w.sum() - 1.0) < 1e-9 and new_w.min() >= -1e-9)
check("it respects the per-line cap", new_w.max() <= 0.60 + 1e-7)

coarse_w, coarse_s = U.tangency_on_frontier(s_mu, s_cov, s_lim, fr24, RF, iters=0)
check("iters=0 falls back to the best grid point, no refinement",
      abs(coarse_s - grid_s) < 1e-12)
check("refining is what closes the gap to max_sharpe",
      abs(new_s - ref_s) <= abs(coarse_s - ref_s) + 1e-12)

# a degenerate problem — bounds pin every weight, so there is only one
# feasible portfolio and the whole frontier is that one point repeated.
pin = opt.Limits(np.full(n_s, 1.0 / n_s), np.full(n_s, 1.0 / n_s), s_groups,
                 {k: (0.0, 1.0) for k in s_groups})
pin_fr = opt.efficient_frontier(s_mu, s_cov, pin, opt.feasible_start(pin), 24)
pin_w, pin_s = U.tangency_on_frontier(s_mu, s_cov, pin, pin_fr, RF)
check("a pinned problem still returns the one portfolio that is feasible",
      abs(pin_w.sum() - 1.0) < 1e-7
      and np.abs(pin_w - 1.0 / n_s).max() < 1e-6 and np.isfinite(pin_s))

# ==========================================================================
print()
print("=" * 78)
print("3. BREAKDOWN — category totals, then lines, adding up and hiding nothing")
print("=" * 78)
LINES = ["U.S. Cash", "U.S. Aggregate Bonds", "AC World Equity",
         "Private Equity", "Commodities", "Gold"]
CATS = [J.CATEGORY_NAME[J.category_of(n)] for n in LINES]
#            cash  agg   equity  PE     cmd    gold
CUR = np.array([0.17, 0.00, 0.20, 0.10, 0.08, 0.45])
OPT = np.array([0.05, 0.60, 0.30, 0.0499, 0.0001, 0.00])   # dust on cmd, PE just above

cat_df = U.category_table(CATS, CUR, OPT)
check("one row per category, in J.P. Morgan's own order",
      list(cat_df["Category"]) == ["Fixed Income", "Equities", "Alternatives"])
check("category Optimal % sums to 100%",
      abs(cat_df["Optimal %"].sum() - 100.0) < 1e-9)
check("category Current % sums to 100%",
      abs(cat_df["Current %"].sum() - 100.0) < 1e-9)
check("Fixed Income totals 5 + 60 = 65%",
      abs(cat_df.loc[0, "Optimal %"] - 65.0) < 1e-9)
check("Change (pp) is Optimal less Current",
      abs(cat_df.loc[0, "Change (pp)"] - (65.0 - 17.0)) < 1e-9)

tabs = U.line_tables(LINES, CATS, CUR, OPT)
check("a table per category", set(tabs) == set(CATS))
alt = tabs["Alternatives"]
check("a line the optimizer zeroes out is STILL shown when we hold it today "
      "(Gold: 45% -> 0%)",
      any(r["LTCMA line"].startswith("Gold") and r["Optimal %"] == 0.0
          and r["Current %"] == 45.0 for _, r in alt.iterrows()))
check("a line just above the dust threshold is shown (Private Equity 4.99%)",
      any(r["LTCMA line"].startswith("Private Equity") for _, r in alt.iterrows()))
check("lines are ordered by optimal weight, heaviest first",
      list(alt["Optimal %"])[:2] == sorted(list(alt["Optimal %"])[:2], reverse=True))
check("each category's line table adds up to that category's total",
      all(abs(tabs[c]["Optimal %"].sum()
              - float(cat_df.loc[cat_df["Category"] == c, "Optimal %"].iloc[0])) < 1e-9
          for c in tabs))

# dust: a line below the threshold that we do not hold is folded, not dropped
DUSTY = np.array([0.05, 0.60, 0.30, 0.0499, 0.0001, 0.00])
dusty = U.line_tables(["U.S. Cash", "U.S. Aggregate Bonds", "AC World Equity",
                       "Private Equity", "Commodities", "Gold"],
                      CATS, np.zeros(6), DUSTY)
d_alt = dusty["Alternatives"]
resid = [r for _, r in d_alt.iterrows() if r["LTCMA line"].startswith("—")]
check("sub-threshold lines are folded into one residual row, not dropped",
      len(resid) == 1 and abs(resid[0]["Optimal %"] - 0.01) < 1e-9)
# Commodities holds the 0.01% and Gold holds nothing; the row stands for every
# line not shown, so shown rows plus the residual account for the whole
# category — which is what makes the column add up.
check("the residual row names how many lines it stands for",
      "2 lines" in resid[0]["LTCMA line"])
check("nothing is lost: the folded table still sums to the category total",
      abs(d_alt["Optimal %"].sum() - 5.0) < 1e-9)
check("a category with nothing above the threshold and nothing held is empty",
      len(U.line_tables(["Gold"], ["Alternatives"], np.array([0.0]),
                        np.array([0.0]))["Alternatives"]) == 0)

banded = U.line_tables(LINES, CATS, CUR, OPT,
                       band=(np.full(6, 0.01), np.full(6, 0.30)))
check("a stability band adds a Stable range column",
      "Stable range %" in banded["Equities"].columns
      and banded["Equities"].iloc[0]["Stable range %"] == "1 – 30")

# ==========================================================================
print()
print("=" * 78)
print("4. APPLYING AN OPTIMUM — an allocation the app will actually accept")
print("=" * 78)
uni_small = U.universe_alloc(BASE)
w = np.zeros(58)
idx = {J.resolve(x): i for i, x in enumerate(uni_small["LTCMA line"])}
w[idx["U.S. Cash"]] = 0.30
w[idx["AC World Equity"]] = 0.50
w[idx["Private Equity"]] = 0.1996
w[idx["Gold"]] = 0.0004                       # dust — must be dropped
applied = U.applied_sleeves(uni_small, w)
check("only the lines above the dust threshold become sleeves", len(applied) == 3)
check("weights are renormalized to exactly 100%, so the app will simulate it",
      abs(applied["Weight %"].sum() - 100.0) < 1e-6)
check("the frame is in the app's allocation shape",
      list(applied.columns) == U.ALLOC_COLS)
check("an applied sleeve keeps the optimized frame's OWN return, override "
      "and all (AC World Equity at 12.88, not the catalogue's 7.00)",
      float(applied.loc[applied["LTCMA line"] == J.LABEL["AC World Equity"],
                        "Return %"].iloc[0]) == 12.88)
check("every applied line is a valid dropdown option in its own category",
      all(r["LTCMA line"] in J.CHOICES[J.category_of(r["LTCMA line"])]
          for _, r in applied.iterrows()))
check("categories on the applied sleeves match the catalogue",
      all(J.CATEGORY_NAME[J.category_of(r["LTCMA line"])] == r["Category"]
          for _, r in applied.iterrows()))
check("an all-dust optimum still yields one sleeve rather than an empty table",
      len(U.applied_sleeves(uni_small, np.full(58, 1e-9))) == 1)

# ==========================================================================
print()
print("=" * 78)
print("5. END TO END — the real 58-line universe, as the panel runs it")
print("=" * 78)
u_lines = [J.resolve(x) for x in uni["LTCMA line"]]
u_cats = list(uni["Category"])
u_mu, u_vols = opt.inputs_from_percent(uni["Return %"], uni["Vol %"])
u_cov = opt.covariance(u_vols, J.corr_for(list(uni["LTCMA line"])))
cur = np.asarray(uni["Weight %"], dtype=float) / 100.0

u_groups = {}
for i, c in enumerate(u_cats):
    u_groups.setdefault(c, []).append(i)
u_groups = {k: np.array(v) for k, v in u_groups.items()}
cur_cat = {k: float(cur[i].sum()) for k, i in u_groups.items()}

lower, upper = np.zeros(58), np.full(58, 0.15)
cash_i = u_lines.index("U.S. Cash")
lower[cash_i] = 0.05
u_lim = opt.Limits(lower, upper, u_groups,
                   {k: (max(0.0, v - 0.10), min(1.0, v + 0.10))
                    for k, v in cur_cat.items()})

import time
t0 = time.perf_counter()
u_w0 = opt.feasible_start(u_lim)
u_fr = opt.efficient_frontier(u_mu, u_cov, u_lim, u_w0, 24)
u_w, u_s = U.tangency_on_frontier(u_mu, u_cov, u_lim, u_fr, RF)
elapsed = time.perf_counter() - t0
print(f"       solved 58 lines in {elapsed:.1f}s "
      f"({len(u_fr[0])} frontier points, Sharpe {u_s:.4f})")

check("the optimum is a valid portfolio: long-only, sums to 100%",
      abs(u_w.sum() - 1.0) < 1e-7 and u_w.min() >= -1e-9)
check("it respects the cash floor and the per-line cap",
      u_w[cash_i] >= 0.05 - 1e-7 and u_w.max() <= 0.15 + 1e-7)
check("it respects every category band",
      all(cur_cat[k] - 0.10 - 1e-7 <= float(u_w[i].sum()) <= cur_cat[k] + 0.10 + 1e-7
          for k, i in u_groups.items()))
check("it beats the current portfolio on Sharpe",
      u_s > opt.sharpe(cur, u_mu, u_cov, RF))

u_cat_df = U.category_table(u_cats, cur, u_w)
u_tabs = U.line_tables(u_lines, u_cats, cur, u_w)
shown = sum(len(t) for t in u_tabs.values())
print(f"       {int((u_w >= U.DUST_PCT/100).sum())} of 58 lines above the "
      f"{U.DUST_PCT:.2f}% threshold · {shown} rows shown across 3 categories")
check("the category table still totals 100%",
      abs(u_cat_df["Optimal %"].sum() - 100.0) < 1e-7)
check("the per-line tables total 100% too — the residual rows account for "
      "every basis point the threshold hides",
      abs(sum(t["Optimal %"].sum() for t in u_tabs.values()) - 100.0) < 1e-7)
check("the whole 58-line universe collapses to a readable table",
      shown < 25)
check("applying it produces sleeves totalling exactly 100%",
      abs(U.applied_sleeves(uni, u_w)["Weight %"].sum() - 100.0) < 1e-6)

# ---------------------------------------------------------------------------
# max_sharpe_qp — the one-QP tangency the app now uses across the catalogue
# ---------------------------------------------------------------------------
print("\nmax_sharpe_qp (homogenised single QP)")
t0 = time.perf_counter()
q_w, q_s = U.max_sharpe_qp(u_mu, u_cov, u_lim, RF)
q_elapsed = time.perf_counter() - t0
print(f"       solved 58 lines in {q_elapsed:.2f}s (Sharpe {q_s:.6f}) "
      f"vs {elapsed:.1f}s on the frontier path (Sharpe {u_s:.6f})")
check("it is a valid portfolio inside every constraint",
      opt._feasible(q_w, u_lim, tol=1e-7))
check("it is at least as good as the frontier search (to 1e-6 in Sharpe)",
      q_s >= u_s - 1e-6)
check("no frontier point beats it",
      all(q_s >= (r - RF) / v - 1e-9 for r, v in zip(u_fr[0], u_fr[1]) if v > 0))
check("it is at least 5x faster than the frontier path", q_elapsed * 5 < elapsed)

# and on the six sleeves, against the reference optimizer.max_sharpe
_rng = np.random.default_rng(7)
_worst_gap = 0.0
_all_feasible = True
for _ in range(6):
    _mu = u_mu[:6] + _rng.normal(0, 0.01, 6)
    _vol = np.sqrt(np.diag(u_cov))[:6]
    _cov = u_cov[:6, :6]
    _lim = opt.Limits(np.zeros(6), np.full(6, 0.6),
                      {"all": np.arange(6)}, {"all": (0.0, 1.0)})
    _wr, _, _, _sr = opt.max_sharpe(_mu, _cov, _lim, None, 0.02)
    _wq, _sq = U.max_sharpe_qp(_mu, _cov, _lim, 0.02)
    _worst_gap = min(_worst_gap, _sq - _sr)
    _all_feasible &= opt._feasible(_wq, _lim, tol=1e-7)
check("on random 6-line problems it is feasible and never below the "
      "reference max_sharpe (worst gap %.1e)" % _worst_gap,
      _all_feasible and _worst_gap > -1e-6)

try:
    U.max_sharpe_qp(u_mu, u_cov, u_lim, 0.50)
    check("an rf above every return is refused, not solved", False)
except opt.Infeasible:
    check("an rf above every return is refused, not solved", True)

print()
print("RESULT:", "all checks pass" if ok else "FAILURES ABOVE — investigate")
raise SystemExit(0 if ok else 1)
