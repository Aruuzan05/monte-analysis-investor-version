"""Checks for the mean-variance optimizer.

    python tests/test_optimizer.py

Covers the properties in the build spec: the closed-form tangency match, the
two-asset analytic minimum-variance weight, frontier monotonicity, the
reconciliation against the app's own stat cards, and that infeasible constraint
sets raise a named error rather than returning garbage.
"""
import numpy as np

import sys
from pathlib import Path

# Run as a script (python tests/test_x.py): put the repo root on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from keepp_mc import jpm_ltcma as J
from keepp_mc import optimizer as O
from keepp_mc.engine import simulate

ok = True


def check(label, cond):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")


def wide_limits(n, groups=None):
    """No cash floor, no caps, bands fully open — the unconstrained case."""
    return O.Limits(np.zeros(n), np.ones(n),
                    groups or {"all": np.arange(n)},
                    {"all": (0.0, 1.0)} if groups is None
                    else {k: (0.0, 1.0) for k in groups})


# ==========================================================================
print("=" * 78)
print("1. UNITS — the arithmetic vs compound conversion")
print("=" * 78)
# The optimizer must agree with THE ENGINE, which defines the arithmetic return
# as exactly g + sigma^2/2 (engine.py: exp_arith = w . (g + 0.5 s^2)). That is
# the quantity the stat cards show and the number the optimizer has to
# reproduce. J.P. Morgan's own published arithmetic column is close but NOT
# identical — their conversion is not exactly sigma^2/2 — so it is deliberately
# not the target here. The gap is printed so nobody mistakes one for the other.
for g_pct, v_pct, jpm_arith in [(10.20, 19.78, 11.91), (4.60, 18.32, 6.15)]:
    mu, s = O.inputs_from_percent([g_pct], [v_pct])
    want = g_pct + 0.5 * (v_pct / 100.0) ** 2 * 100
    check(f"compound {g_pct}% + vol {v_pct}% -> {mu[0]*100:.2f}% == engine's "
          f"g+s^2/2 ({want:.2f}%); JPM's own column prints {jpm_arith}%, "
          f"a {mu[0]*100 - jpm_arith:+.2f}pp difference",
          abs(mu[0] * 100 - want) < 1e-9)

mu, s = O.inputs_from_percent([3.80], [0.00])
check("a zero volatility is floored, not left to make Sigma singular", s[0] >= O.MIN_VOL)
mu, s = O.inputs_from_percent([5.00], [0.67])
check("a near-zero vol barely moves the arithmetic return",
      abs(mu[0] * 100 - 5.00) < 0.01)

# ==========================================================================
print()
print("=" * 78)
print("2. TWO ASSETS — against the analytic minimum-variance weight")
print("=" * 78)
# The analytic formula is the UNCONSTRAINED minimum-variance weight. At high
# correlation it calls for shorting the low-vol asset (rho 0.9 -> w1 1.45,
# w2 -0.45), which a long-only solver must refuse. So the check splits: match
# the formula where it is admissible, and pin to the boundary where it is not.
for rho in (-0.5, 0.0, 0.3, 0.9):
    s1, s2 = 0.08, 0.20
    cov = O.covariance([s1, s2], [[1, rho], [rho, 1]])
    m = np.array([0.05, 0.09])
    res = O.min_variance(m, cov, wide_limits(2))
    want = (s2**2 - rho * s1 * s2) / (s1**2 + s2**2 - 2 * rho * s1 * s2)
    if 0.0 <= want <= 1.0:
        check(f"rho {rho:+.1f}: w1 {res.x[0]:.6f} == analytic {want:.6f}",
              res.success and abs(res.x[0] - want) < 1e-6)
    else:
        check(f"rho {rho:+.1f}: analytic w1 {want:.4f} needs a short, so the "
              f"long-only solver clamps to {res.x[0]:.6f}",
              res.success and abs(res.x[0] - np.clip(want, 0.0, 1.0)) < 1e-6)

# ==========================================================================
print()
print("=" * 78)
print("3. UNCONSTRAINED TANGENCY — against inv(Sigma)(mu - rf), normalized")
print("=" * 78)
TOL = 1e-6                     # the build spec's bar for this check
rng = np.random.default_rng(7)
worst = 0.0
tested = 0
for trial in range(25):
    n = int(rng.integers(3, 7))
    A = rng.normal(size=(n, n))
    corr = A @ A.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    vols = rng.uniform(0.04, 0.25, n)
    cov = O.covariance(vols, corr)
    m = rng.uniform(0.03, 0.12, n)
    rf = 0.03
    closed = O.unconstrained_tangency(m, cov, rf)
    if closed.min() < 0.001:
        continue                       # closed form went short; bounds would bind
    # bounds must not bind, so open them past the closed-form solution
    lim = O.Limits(np.zeros(n), np.ones(n), {"all": np.arange(n)}, {"all": (0.0, 1.0)})
    w, r, v, sh = O.max_sharpe(m, cov, lim, rf=rf, points=60)
    err = float(np.abs(w - closed).max())
    worst = max(worst, err)
    tested += 1
    if err >= TOL:
        check(f"n={n}: constrained solver matches the closed form "
              f"(max |dw| {err:.2e})", False)
check(f"{tested} random long-only problems, worst weight deviation from the "
      f"closed form {worst:.2e} (bar {TOL:.0e})", tested >= 5 and worst < TOL)

# ==========================================================================
print()
print("=" * 78)
print("4. THE REAL PORTFOLIO — six sleeves on the published JPM matrix")
print("=" * 78)
LINES = ["U.S. Inv Grade Corporate Bonds", "U.S. Cash", "AC World Equity",
         "Private Equity", "Diversified Hedge Funds", "Commodities"]
W = np.array([45, 17, 20, 10, 0, 8], dtype=float) / 100.0
R_PCT = [5.00, 3.80, 12.88, 10.20, 5.30, 4.60]
V_PCT = [7.02, 0.67, 14.47, 19.78, 5.75, 18.32]
mu, vols = O.inputs_from_percent(R_PCT, V_PCT)
corr = J.corr_for(LINES)
cov = O.covariance(vols, corr)
n = len(LINES)

# --- 4a. reconciliation against the app's own stat cards -------------------
res_mc, _ = simulate(list(W * 100), R_PCT, V_PCT, corr, 10000, 10, 20260722)
r_now, v_now = O.portfolio(W, mu, cov)
check(f"optimizer return {r_now*100:.4f}% == stat card {res_mc.exp_arith*100:.4f}%",
      abs(r_now - res_mc.exp_arith) < 1e-12)
check(f"optimizer vol    {v_now*100:.4f}% == stat card {res_mc.analytic_vol*100:.4f}%",
      abs(v_now - res_mc.analytic_vol) < 1e-12)
check("both agree to the last displayed decimal (2dp)",
      f"{r_now*100:.2f}" == f"{res_mc.exp_arith*100:.2f}"
      and f"{v_now*100:.2f}" == f"{res_mc.analytic_vol*100:.2f}")

# --- 4b. category bands, as the UI builds them -----------------------------
cats = {}
for i, name in enumerate(LINES):
    cats.setdefault(J.CATEGORY_NAME[J.category_of(name)], []).append(i)
groups = {k: np.array(v) for k, v in cats.items()}
bands = {k: (0.0, 1.0) for k in groups}
lower = np.zeros(n); upper = np.ones(n)
lower[LINES.index("U.S. Cash")] = 0.05                # cash floor
limits = O.Limits(lower, upper, groups, bands)

w0 = O.feasible_start(limits)
check(f"feasible start sums to 1 ({w0.sum():.9f}) and respects the cash floor",
      abs(w0.sum() - 1) < 1e-9 and w0[LINES.index("U.S. Cash")] >= 0.05 - 1e-9)

mv = O.min_variance(mu, cov, limits, w0)
r_mv, v_mv = O.portfolio(mv.x, mu, cov)
check(f"minimum-variance vol {v_mv*100:.2f}% <= current {v_now*100:.2f}%",
      v_mv <= v_now + 1e-9)

rets, vols_f, ws = O.efficient_frontier(mu, cov, limits, w0, points=40)
check(f"frontier returned {len(rets)} points", len(rets) >= 30)
check("every frontier weight sums to 1", np.abs(ws.sum(axis=1) - 1).max() < 1e-9)
check("every frontier weight is long-only", ws.min() >= -1e-9)
check("every frontier point honours the cash floor",
      ws[:, LINES.index("U.S. Cash")].min() >= 0.05 - 1e-6)
d = np.diff(vols_f)
check(f"frontier volatility is monotonically increasing (worst step {d.min():+.2e})",
      d.min() > -1e-6)
check("frontier return is monotonically increasing", np.diff(rets).min() > -1e-12)

rf = mu[LINES.index("U.S. Cash")]
w_t, r_t, v_t, sh_t = O.max_sharpe(mu, cov, limits, w0, rf)
check(f"tangency sums to 1 and is long-only ({w_t.sum():.9f}, min {w_t.min():+.2e})",
      abs(w_t.sum() - 1) < 1e-9 and w_t.min() >= -1e-9)
check(f"tangency Sharpe {sh_t:.4f} beats the current portfolio "
      f"{O.sharpe(W, mu, cov, rf):.4f}", sh_t > O.sharpe(W, mu, cov, rf))
check("tangency Sharpe is the best on the frontier",
      sh_t >= ((rets - rf) / vols_f).max() - 1e-6)

g = O.max_growth(mu, cov, limits, w0)
r_g, v_g = O.portfolio(g.x, mu, cov)
check(f"max-growth solves and is long-only (ret {r_g*100:.2f}%, vol {v_g*100:.2f}%)",
      g.success and g.x.min() >= -1e-9 and abs(g.x.sum() - 1) < 1e-9)
check("max-growth objective beats the current portfolio",
      (r_g - 0.5 * v_g**2) > (r_now - 0.5 * v_now**2))

# --- 4c. the objectives the UI offers --------------------------------------
tgt = r_now + 0.005
mvt = O.min_variance(mu, cov, limits, w0, target_return=tgt)
check(f"min-vol at a target return of {tgt*100:.2f}% hits the target",
      mvt.success and abs(mvt.x @ mu - tgt) < 1e-8)
budget = v_now
mrb = O.max_return(mu, cov, limits, w0, vol_budget=budget)
r_b, v_b = O.portfolio(mrb.x, mu, cov)
check(f"max-return within a {budget*100:.2f}% vol budget respects it "
      f"(got {v_b*100:.2f}%)", mrb.success and v_b <= budget + 1e-6)
check("max-return within the budget beats the current return", r_b > r_now)

# ==========================================================================
print()
print("=" * 78)
print("5. BANDS AND THE CASH FLOOR ACTUALLY BIND")
print("=" * 78)
eq = "Equities"
tight = O.Limits(lower.copy(), upper.copy(), groups,
                 {**bands, eq: (0.10, 0.15)})
w, r, v, sh = O.max_sharpe(mu, cov, tight, rf=rf)
check(f"an Equities band of 10-15% is respected (got {w[groups[eq]].sum():.4%})",
      0.10 - 1e-6 <= w[groups[eq]].sum() <= 0.15 + 1e-6)

hi_cash = lower.copy(); hi_cash[LINES.index("U.S. Cash")] = 0.30
w2, *_ = O.max_sharpe(mu, cov, O.Limits(hi_cash, upper, groups, bands), rf=rf)
check(f"a 30% cash floor is respected (got {w2[LINES.index('U.S. Cash')]:.4%})",
      w2[LINES.index("U.S. Cash")] >= 0.30 - 1e-6)

cap = np.full(n, 0.25)
w3, *_ = O.max_sharpe(mu, cov, O.Limits(np.zeros(n), cap, groups, bands), rf=rf)
check(f"a 25% per-line cap is respected (max weight {w3.max():.4%})",
      w3.max() <= 0.25 + 1e-6)

# ==========================================================================
print()
print("=" * 78)
print("6. INFEASIBLE INPUTS RAISE A NAMED ERROR, NEVER GARBAGE")
print("=" * 78)


def expect_infeasible(label, lim, must_mention=None):
    try:
        O.feasible_start(lim)
    except O.Infeasible as e:
        hit = must_mention is None or must_mention.lower() in str(e).lower()
        check(f"{label} -> {str(e)[:66]}", hit)
        return
    check(f"{label} -> raised nothing", False)


bad = np.zeros(n); bad[0] = 0.7; bad[1] = 0.6
expect_infeasible("per-line floors summing above 100%",
                  O.Limits(bad, upper, groups, bands), "100%")
expect_infeasible("category minimums summing above 100%",
                  O.Limits(np.zeros(n), upper, groups,
                           {k: (0.5, 1.0) for k in groups}), "100%")
expect_infeasible("category maximums summing below 100%",
                  O.Limits(np.zeros(n), upper, groups,
                           {k: (0.0, 0.2) for k in groups}), "maximum")
expect_infeasible("a category minimum above its own maximum",
                  O.Limits(np.zeros(n), upper, groups,
                           {**bands, eq: (0.6, 0.2)}), "Equities")
expect_infeasible("a cash floor above the cash category's cap",
                  O.Limits(hi_cash, upper, groups,
                           {**bands, "Fixed Income": (0.0, 0.10)}),
                  "Fixed Income")
expect_infeasible("per-line caps too small to reach 100%",
                  O.Limits(np.zeros(n), np.full(n, 0.05), groups, bands), "caps")

# ==========================================================================
print()
print("=" * 78)
print("7. DEGENERATE INPUTS DO NOT PRODUCE NONSENSE")
print("=" * 78)
mu_z, s_z = O.inputs_from_percent([8.0, 5.0], [0.0, 10.0])
cov_z = O.covariance(s_z, np.eye(2))
check("a zero-vol line gives a non-singular, invertible covariance",
      np.linalg.matrix_rank(cov_z) == 2 and np.all(np.linalg.eigvalsh(cov_z) > 0))
w_z, r_z, v_z, _ = O.max_sharpe(mu_z, cov_z, wide_limits(2), rf=0.03)
check("the optimizer still returns a valid weight vector",
      abs(w_z.sum() - 1) < 1e-9 and v_z >= 0)

flat = O.covariance([0.10, 0.10], [[1.0, 1.0], [1.0, 1.0]])
check("a perfectly correlated pair still yields a real volatility",
      O.portfolio([0.5, 0.5], np.array([0.05, 0.05]), flat)[1] > 0)

# ==========================================================================
print()
print("=" * 78)
print("8. STABILITY — how much the answer depends on the return estimates")
print("=" * 78)
p25, p75 = O.stability(mu, cov, limits, rf, noise=0.005, trials=40, points=20)
check("the stability check returns a band per line",
      p25 is not None and len(p25) == n)
if p25 is not None:
    spread = p75 - p25
    widest = int(np.argmax(spread))
    print(f"       widest: {LINES[widest]} optimal weight ranges "
          f"{p25[widest]:.1%}-{p75[widest]:.1%} across +/-0.5% noise on mu")
    check("at least one line's optimal weight is visibly unstable — which is "
          "the point of showing it", spread.max() > 0.01)

# ==========================================================================
print()
print("=" * 78)
print("9. FEASIBLE SPANS — what the two constrained objectives can be asked for")
print("=" * 78)
STEP = 0.0001                     # 0.01pp, the slider step

span_r = O.return_span(mu, cov, limits)
span_v = O.vol_span(mu, cov, limits)
print(f"       default bands + 5% cash floor:")
print(f"         return {span_r.lo*100:.4f}% - {span_r.hi*100:.4f}%")
print(f"         vol    {span_v.lo*100:.4f}% - {span_v.hi*100:.4f}%")

check("return span is ordered and non-degenerate",
      span_r.hi > span_r.lo and not span_r.degenerate)
check("vol span is ordered and non-degenerate",
      span_v.hi > span_v.lo and not span_v.degenerate)
check("both span endpoints are feasible portfolios",
      all(O._feasible(w, limits, tol=1e-6)
          for w in (span_r.w_lo, span_r.w_hi, span_v.w_lo, span_v.w_hi)))

# the upper end is the same max-return portfolio for both spans
check("return span's top and vol span's top are the same portfolio",
      np.abs(span_r.w_hi - span_v.w_hi).max() < 1e-9)
wlo_e, rlo_e, whi_e, rhi_e = O._endpoints(mu, cov, limits)
check(f"return span top {span_r.hi*100:.4f}% >= _endpoints' max-return "
      f"{rhi_e*100:.4f}% (LP is exact; trust-constr leaves a vertex artifact)",
      span_r.hi >= rhi_e - 1e-9)
sliver = (rlo_e - span_r.lo) * 1e4
print(f"       dominated sliver below the min-variance return: {sliver:.2f} bp")
check("return span's floor is at or below the min-variance portfolio's return",
      span_r.lo <= rlo_e + 1e-12)
check(f"vol span's floor {span_v.lo*100:.4f}% == the min-variance portfolio's "
      f"vol {O.portfolio(wlo_e, mu, cov)[1]*100:.4f}%",
      abs(span_v.lo - O.portfolio(wlo_e, mu, cov)[1]) < 1e-9)
check("vol span's ceiling == the vol of the LP max-return portfolio",
      abs(span_v.hi - O.portfolio(span_r.w_hi, mu, cov)[1]) < 1e-12)

# with every band open the return span is exactly [min(mu), max(mu)]
wide = O.Limits(np.zeros(n), np.ones(n), groups, {k: (0.0, 1.0) for k in groups})
sw = O.return_span(mu, cov, wide)
check(f"bands open: return span is [min(mu), max(mu)] = "
      f"[{mu.min()*100:.4f}%, {mu.max()*100:.4f}%]",
      abs(sw.lo - mu.min()) < 1e-12 and abs(sw.hi - mu.max()) < 1e-12)

# ---- inward rounding ------------------------------------------------------
rng2 = np.random.default_rng(11)
bad = 0
for _ in range(400):
    a = float(rng2.uniform(-0.05, 0.30)); b = a + float(rng2.uniform(1e-6, 0.2))
    st_ = float(rng2.choice([1e-4, 5e-4, 1e-3, 1e-2]))
    sp = O.Span("return", a, b, np.zeros(1), np.zeros(1), False)
    lo, hi = O.slider_bounds(sp, st_)
    if lo < a - 1e-12 or hi > b + 1e-12:
        bad += 1
    if abs(round((hi - lo) / st_) - (hi - lo) / st_) > 1e-9:
        bad += 1
check("400 random spans: slider_bounds never lands outside the span, and the "
      "width is always a whole number of steps", bad == 0)

for sp, lbl in ((span_r, "return"), (span_v, "vol")):
    lo, hi = O.slider_bounds(sp, STEP)
    check(f"{lbl} slider ends are inside the achievable span "
          f"({lo*100:.4f}% - {hi*100:.4f}%)",
          lo >= sp.lo - 1e-15 and hi <= sp.hi + 1e-15)

# round() would have put three of four ends OUTSIDE — the whole reason for this
outside = sum(1 for sp in (span_r, span_v) for v, end in
              ((round(sp.lo / STEP) * STEP, "lo"), (round(sp.hi / STEP) * STEP, "hi"))
              if (end == "lo" and v < sp.lo) or (end == "hi" and v > sp.hi))
print(f"       round() would place {outside} of 4 ends outside the feasible set")

# ---- EVERY NOTCH SOLVES. The test that would have caught the original bugs.
lo_r, hi_r = O.slider_bounds(span_r, STEP)
grid = [lo_r, hi_r] + list(np.linspace(lo_r, hi_r, 20))
worst_sum, fails = 0.0, []
for t in grid:
    r_ = O.min_variance(mu, cov, limits, target_return=float(t))
    worst_sum = max(worst_sum, abs(r_.x.sum() - 1.0))
    if not r_.success or abs(r_.x.sum() - 1.0) > 1e-9:
        fails.append(t)
check(f"every target-return notch solves, pre-renormalization "
      f"(22 targets, worst |sum(w)-1| {worst_sum:.2e})", not fails)

lo_v, hi_v = O.slider_bounds(span_v, STEP)
gridv = [lo_v, hi_v] + list(np.linspace(lo_v, hi_v, 20))
worst_sum, failsv = 0.0, []
for bgt in gridv:
    r_ = O.max_return(mu, cov, limits, vol_budget=float(bgt))
    worst_sum = max(worst_sum, abs(r_.x.sum() - 1.0))
    v_here = O.portfolio(r_.x, mu, cov)[1]
    if not r_.success or abs(r_.x.sum() - 1.0) > 1e-7 or v_here > bgt + 1e-6:
        failsv.append(bgt)
check(f"every vol-budget notch solves and respects its budget "
      f"(22 budgets, worst |sum(w)-1| {worst_sum:.2e})", not failsv)

# the three audited bad inputs are now unreachable from the slider
for bad_t, lbl in ((0.01, "1.00%"), (0.20, "20.00%")):
    check(f"a target return of {lbl} is outside the slider's range",
          not (lo_r <= bad_t <= hi_r))
check("a vol budget of 0.10% is outside the slider's range",
      not (lo_v <= 0.001 <= hi_v))

# ---- degenerate constraints -----------------------------------------------
pin = {k: (float(cur_cat := cur[i].sum()), float(cur_cat))
       for k, i in groups.items()} if False else None
cur = W
pinned = O.Limits(np.array([W[i] for i in range(n)]), np.array([W[i] for i in range(n)]),
                  groups, {k: (float(W[i].sum()), float(W[i].sum()))
                           for k, i in groups.items()})
sp_d = O.return_span(mu, cov, pinned)
check("constraints that pin every weight give degenerate=True, not a raise",
      sp_d.degenerate and abs(sp_d.hi - sp_d.lo) < O.SPAN_TOL)
lo_d, hi_d = O.slider_bounds(sp_d, STEP)
check("slider_bounds on a degenerate span returns lo == hi rather than hi < lo",
      lo_d == hi_d)

print()
print("RESULT:", "all checks pass" if ok else "FAILURES ABOVE — investigate")
raise SystemExit(0 if ok else 1)
