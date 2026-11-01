"""Mean-variance optimization over the allocation lines.

Pure numpy/scipy. No Streamlit import, so it is unit-testable and reusable,
mirroring engine.py.

UNITS — read this before calling anything.
Every function in this module works in DECIMALS: a 7.51% return is 0.0751 and a
14.47% volatility is 0.1447. The app stores percent points, so the boundary
conversion happens exactly once, in `inputs_from_percent()`. Doing it anywhere
else invites the sigma-squared-over-two bug below to reappear in new clothes.

THE ARITHMETIC vs COMPOUND TRAP — the single most likely way to get this wrong.
Mean-variance optimization is defined on ARITHMETIC expected returns. The
simulation engine is not: engine.py uses

    m = ln(1 + g) / 12

so the app's `Return %` column is consumed as a COMPOUND (geometric) return, and
engine.py derives the arithmetic figure internally as `g + sigma^2 / 2`. Feeding
the compound column straight into an optimizer would understate every risky
line's expected return in proportion to its variance — which is to say it would
systematically under-allocate to exactly the assets the optimizer exists to
size. `inputs_from_percent()` applies the conversion; nothing else should.

Weights are unitless, so nothing needs converting on the way back out.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import (Bounds, LinearConstraint, NonlinearConstraint,
                            linprog, minimize)

# A line with zero volatility and the highest return would take 100% of the
# portfolio and produce a singular covariance matrix. Floor it.
MIN_VOL = 1e-4          # 0.01% in decimals


class Infeasible(ValueError):
    """Raised when no weight vector can satisfy the constraints.

    Carries a message naming the offending constraint, so the UI can say which
    band to loosen rather than reporting a generic solver failure.
    """


@dataclass(frozen=True)
class Limits:
    """Box bounds per line, plus min/max bands on groups of lines.

    `sleeve_index` maps a group name to the integer positions of its lines;
    in this app the groups are J.P. Morgan's three categories. `lower`/`upper`
    are per-line and carry the cash floor and the max-weight-per-line cap.
    """
    lower: np.ndarray
    upper: np.ndarray
    sleeve_index: dict
    sleeve_band: dict
    # Weighted exposure limits, e.g. geography or liquidity: (name, coef, lo, hi) with
    # lo <= coef'w <= hi. coef[i] is the share of line i counted (a line that
    # is 65% US counts 0.65). Empty by default, so every existing caller and
    # result is unchanged.
    exposure: tuple = ()

    @property
    def n(self) -> int:
        return len(self.lower)

    def band_rows(self):
        """Every grouped limit as (name, coef, lo, hi): the category bands as
        0/1 rows, then any weighted exposure limits."""
        for name, idx in self.sleeve_index.items():
            lo, hi = self.sleeve_band[name]
            a = np.zeros(self.n); a[idx] = 1.0
            yield name, a, float(lo), float(hi)
        for name, a, lo, hi in self.exposure:
            yield name, np.asarray(a, dtype=float), float(lo), float(hi)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def inputs_from_percent(compound_pct, vol_pct):
    """(mu_arithmetic, sigma) in decimals from the app's percent-point columns.

    THE conversion point for the whole module. mu = g + sigma^2 / 2.
    """
    g = np.asarray(compound_pct, dtype=float) / 100.0
    s = np.maximum(np.asarray(vol_pct, dtype=float) / 100.0, MIN_VOL)
    return g + 0.5 * s**2, s


def covariance(vols, corr) -> np.ndarray:
    """Sigma from vols and correlations. Caller must pass a PSD-repaired corr."""
    s = np.maximum(np.asarray(vols, dtype=float), MIN_VOL)
    cov = np.outer(s, s) * np.asarray(corr, dtype=float)
    return (cov + cov.T) / 2.0          # kill any asymmetry from rounding


def portfolio(w, mu, cov):
    """(expected return, volatility) for a weight vector, in decimals."""
    w = np.asarray(w, dtype=float)
    return float(w @ mu), float(np.sqrt(max(w @ cov @ w, 0.0)))


def sharpe(w, mu, cov, rf) -> float:
    r, v = portfolio(w, mu, cov)
    return (r - rf) / v if v > 0 else 0.0


# --------------------------------------------------------------------------
# Feasibility
# --------------------------------------------------------------------------
def _named_infeasibility(limits):
    """Cheap structural checks that can NAME the offending constraint.

    Run before the solver so the user is told which band to loosen instead of
    being shown a bare convergence failure.
    """
    lo, hi = limits.lower, limits.upper
    if np.any(lo > hi + 1e-12):
        i = int(np.argmax(lo - hi))
        return f"line {i}: its floor ({lo[i]:.2%}) is above its cap ({hi[i]:.2%})"
    if lo.sum() > 1.0 + 1e-9:
        return (f"the per-line floors add up to {lo.sum():.2%}, which is more "
                f"than 100% — lower the cash floor or the per-line minimums")
    if hi.sum() < 1.0 - 1e-9:
        return (f"the per-line caps add up to only {hi.sum():.2%} — raise the "
                f"max weight per line")

    band_lo = sum(b[0] for b in limits.sleeve_band.values())
    band_hi = sum(b[1] for b in limits.sleeve_band.values())
    if band_lo > 1.0 + 1e-9:
        return (f"the category minimums add up to {band_lo:.2%}, which is more "
                f"than 100% — lower one of them")
    if band_hi < 1.0 - 1e-9:
        return (f"the category maximums add up to only {band_hi:.2%} — raise "
                f"one of them")

    for name, idx in limits.sleeve_index.items():
        blo, bhi = limits.sleeve_band[name]
        if blo > bhi + 1e-12:
            return f"{name}: minimum {blo:.2%} is above maximum {bhi:.2%}"
        if hi[idx].sum() < blo - 1e-9:
            return (f"{name}: its minimum of {blo:.2%} cannot be reached — the "
                    f"per-line caps inside it only reach {hi[idx].sum():.2%}")
        if lo[idx].sum() > bhi + 1e-9:
            return (f"{name}: its maximum of {bhi:.2%} is already exceeded by "
                    f"the floors inside it ({lo[idx].sum():.2%})")

    for name, a, elo, ehi in limits.exposure:
        a = np.asarray(a, dtype=float)
        if elo > ehi + 1e-12:
            return f"{name}: minimum {elo:.2%} is above maximum {ehi:.2%}"
        # Fully invested, so a'w can never exceed the largest share on offer.
        reach = float(a.max()) if a.size else 0.0
        if elo > reach + 1e-9:
            return (f"{name}: its minimum of {elo:.2%} cannot be reached — "
                    f"the investments available reach at most {reach:.2%}")
        if float(a @ lo) > ehi + 1e-9:
            return (f"{name}: its maximum of {ehi:.2%} is already exceeded by "
                    f"the floors ({float(a @ lo):.2%})")
    return None


def feasible_start(limits) -> np.ndarray:
    """A feasible, INTERIOR w0 respecting bounds and category bands.

    Feasibility is established with a linear program rather than guessed: a
    heuristic start that violates a band makes SLSQP report failure on a problem
    that is perfectly solvable.

    But a plain LP returns a VERTEX of the feasible polytope, and that is a trap.
    SLSQP started at a vertex solves a linearised subproblem that offers no
    descent direction, declares success after ONE iteration, and hands back the
    vertex. Measured: from the vertex [0, 0, 1] it reported a minimum-variance
    portfolio at 22.07% volatility when 0.12% was achievable — a silent, total
    failure that then collapsed the entire frontier to a single point.

    So several vertices are collected with random objectives and averaged. Every
    constraint here is convex, so the average of feasible points is feasible,
    and the average of distinct vertices is interior. Raises Infeasible, naming
    the offending constraint where it can.
    """
    named = _named_infeasibility(limits)
    if named:
        raise Infeasible(named)

    n = limits.n
    A_ub, b_ub = [], []
    for name, a, blo, bhi in limits.band_rows():
        A_ub.append(-a); b_ub.append(-blo)           # -sum <= -min
        A_ub.append(a); b_ub.append(bhi)             #  sum <=  max
    A_ub = np.array(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None
    bounds = list(zip(limits.lower, limits.upper))

    def vertex(c):
        r = linprog(c=c, A_ub=A_ub, b_ub=b_ub,
                    A_eq=np.ones((1, n)), b_eq=np.array([1.0]),
                    bounds=bounds, method="highs")
        return np.clip(r.x, limits.lower, limits.upper) if r.success else None

    first = vertex(np.zeros(n))
    if first is None:
        raise Infeasible(
            "no allocation satisfies every constraint at once. Check the cash "
            "floor against the category bands"
            + (" and the geography / liquidity limits" if limits.exposure else "")
            + ", and that the minimums leave room for 100% in total.")

    rng = np.random.default_rng(20260722)
    pts = [first]
    for _ in range(min(2 * n, 16)):
        v = vertex(rng.normal(size=n))
        if v is not None:
            pts.append(v)
    w = np.mean(pts, axis=0)
    w = np.clip(w, limits.lower, limits.upper)
    s = w.sum()
    if s > 0:
        w = w / s
    return w if _feasible(w, limits, tol=1e-7) else first


# --------------------------------------------------------------------------
# Solvers
# --------------------------------------------------------------------------
# Every problem below is convex (or concave, for growth) over a convex feasible
# set defined by linear constraints, so the optimum is unique and a single solve
# from any feasible start reaches it. That is why none of these use multi-start.
# The one exception is max_sharpe: the Sharpe ratio is scale-invariant and NOT
# convex, so it is seeded from a frontier scan rather than an arbitrary point.

def _constraints(mu, limits, target_return=None, vol_budget=None, cov=None):
    """Every linear constraint as ONE LinearConstraint, plus the vol budget.

    All of the structure here — the budget constraint, the target return, and
    the category bands — is linear in w, so it belongs in a single matrix with
    lower and upper bound vectors rather than a list of scalar callbacks.
    """
    n = limits.n
    A = [np.ones(n)]
    lb = [1.0]
    ub = [1.0]
    if target_return is not None:
        A.append(np.asarray(mu, dtype=float))
        lb.append(float(target_return))
        ub.append(float(target_return))
    for name, row, lo, hi in limits.band_rows():
        A.append(row); lb.append(float(lo)); ub.append(float(hi))

    cons = [LinearConstraint(np.array(A), np.array(lb), np.array(ub))]
    if vol_budget is not None:
        cons.append(NonlinearConstraint(
            lambda w: float(w @ cov @ w), -np.inf, float(vol_budget) ** 2,
            jac=lambda w: 2.0 * cov @ w,
            hess=lambda w, v: 2.0 * v[0] * cov))
    return cons


def _solve(objective, jac, mu, limits, w0, hess=None, **kw):
    """Constrained solve with trust-constr and an exact Hessian.

    NOT SLSQP, and the difference is not academic. Every problem here is a
    quadratic program, so the Hessian is a constant matrix that can be handed to
    the solver outright — trust-constr uses it, SLSQP cannot. Measured side by
    side on the same seven random long-only problems, worst-case weight error
    against the closed-form tangency:

        SLSQP          9.05e-02
        trust-constr   3.14e-08

    SLSQP also fails in a way that is silent rather than loud: it reports
    "Optimization terminated successfully" after a single iteration when started
    near a vertex, returning the start.

    The tolerances here are deliberately loose (gtol 1e-8, 200 iterations, ~17ms
    a solve). They do not set the accuracy of the answer: the caller finishes
    with `_kkt_polish`, which needs only the correct ACTIVE SET and then solves
    the remaining equality-constrained problem exactly. Measured across solver
    tolerances from 1e-10 down to 1e-4, the polished error stayed at 5.6e-16
    while the raw error grew from 4e-6 to 3e-2. Precision comes from the polish;
    the solver only has to find the right face of the polytope.
    """
    n = limits.n
    hess = hess if hess is not None else (lambda w: np.zeros((n, n)))
    return minimize(objective, np.asarray(w0, dtype=float), jac=jac, hess=hess,
                    method="trust-constr",
                    bounds=Bounds(limits.lower, limits.upper),
                    constraints=_constraints(mu, limits, **kw),
                    options={"maxiter": 200, "gtol": 1e-8, "xtol": 1e-10,
                             "verbose": 0})


def _feasible(w, limits, mu=None, target_return=None, tol=1e-7):
    if w.min() < limits.lower.min() - tol:
        return False
    if np.any(w < limits.lower - tol) or np.any(w > limits.upper + tol):
        return False
    if abs(w.sum() - 1.0) > tol:
        return False
    for name, a, lo, hi in limits.band_rows():
        if not (lo - tol <= float(a @ w) <= hi + tol):
            return False
    if target_return is not None and abs(w @ mu - target_return) > tol:
        return False
    return True


def _kkt_polish(w, mu, cov, limits, target_return=None, tol=1e-8):
    """Solve the QP exactly on the variables SLSQP left off their bounds.

    SLSQP stops on the objective, not the iterate, so on a flat QP it can be
    accurate to 1e-7 in variance while still 2e-3 out in individual weights.
    Once the active set is known the remaining problem is an equality-
    constrained quadratic, whose KKT system is a single linear solve — exact to
    machine precision. Returns None if the active set guess does not produce a
    feasible, better point, in which case the caller keeps the SLSQP answer.
    """
    n = len(w)
    fixed = (w <= limits.lower + tol) | (w >= limits.upper - tol)
    free = ~fixed
    if free.sum() < 2:
        return None
    wx = np.where(w <= limits.lower + tol, limits.lower, limits.upper)
    wx = np.where(fixed, wx, 0.0)

    rows = [np.ones(n)]
    rhs = [1.0 - wx[fixed].sum()]
    if target_return is not None:
        rows.append(mu.copy())
        rhs.append(target_return - mu[fixed] @ wx[fixed])
    for name, m, lo, hi in limits.band_rows():
        s = float(m @ w)
        near_lo, near_hi = abs(s - lo) < 1e-7, abs(s - hi) < 1e-7
        if near_lo or near_hi:
            rows.append(m)
            rhs.append((lo if near_lo else hi) - m[fixed] @ wx[fixed])

    A = np.array(rows)[:, free]
    b = np.array(rhs)

    # Drop linearly dependent constraint rows. A category band that is active at
    # exactly 100% duplicates the budget row, and a duplicated row makes the KKT
    # matrix singular — which is what silently disabled this whole polish step
    # on every test problem until it was caught. The rows are consistent (the
    # incoming w satisfies all of them), so dropping the redundant ones changes
    # nothing except the conditioning.
    keep = []
    basis = np.zeros((0, A.shape[1]))
    for i in range(A.shape[0]):
        r = A[i]
        if basis.size:
            r = r - basis.T @ (basis @ r)
        nrm = np.linalg.norm(r)
        if nrm > 1e-9 * max(np.linalg.norm(A[i]), 1.0):
            keep.append(i)
            basis = np.vstack([basis, r / nrm])
    A, b = A[keep], b[keep]

    C = cov[np.ix_(free, free)]
    k = A.shape[0]
    K = np.block([[2.0 * C, A.T], [A, np.zeros((k, k))]])
    g = -2.0 * cov[np.ix_(free, fixed)] @ wx[fixed]
    try:
        sol = np.linalg.solve(K, np.concatenate([g, b]))[:int(free.sum())]
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(sol)):
        return None

    out = wx.copy()
    out[free] = sol
    if not _feasible(out, limits, mu, target_return):
        return None
    if out @ cov @ out > w @ cov @ w + 1e-14:
        return None
    return out


def min_variance(mu, cov, limits, w0=None, target_return=None):
    """Minimum-variance portfolio, optionally at a target return.

    Two stages: a fast trust-constr solve to find the active set, then an exact
    KKT solve on the variables it left off their bounds.

    If the supplied start fails, it retries from a generic feasible point. A
    warm start carried over from a NEARBY target return can violate this call's
    return constraint badly enough that SLSQP gives up — measured at 10 failures
    in 50 target returns across one bracket, clustered exactly around the
    optimum, which is the worst possible place to lose points.
    """
    # SCALE THE OBJECTIVE. SLSQP's `ftol` is an ABSOLUTE tolerance on the change
    # in the objective. Portfolio variance is a small number — 0.0057 for this
    # portfolio, and ~1e-6 for a near-riskless one — so against ftol=1e-12 the
    # solver can decide it has converged before it has moved, and hand back its
    # own starting corner. That failure is silent and total: measured on an
    # ill-conditioned 3-asset problem it reported a minimum-variance portfolio
    # with 22% volatility when 0.1% was available, which then collapsed the
    # whole frontier to a single point. Dividing by the mean variance makes the
    # objective and its gradient O(1), so ftol means what it is meant to mean.
    k = 1.0 / max(float(np.mean(np.diag(cov))), 1e-300)
    obj, jac = (lambda w: k * (w @ cov @ w)), (lambda w: k * 2.0 * cov @ w)
    hess = lambda w: k * 2.0 * cov
    starts = [w0] if w0 is not None else []
    starts.append(None)                      # generic feasible point, computed lazily

    res = None
    for s in starts:
        s = feasible_start(limits) if s is None else s
        r = _solve(obj, jac, mu, limits, s, hess=hess,
                   target_return=target_return)
        if r.success and _feasible(np.clip(r.x, limits.lower, limits.upper),
                                   limits, mu, target_return, tol=1e-6):
            res = r
            break
        if res is None or (r.success and not res.success):
            res = r
    if not res.success:
        return res

    polished = _kkt_polish(res.x, mu, cov, limits, target_return)
    if polished is not None:
        res.x = polished
        res.fun = k * float(polished @ cov @ polished)
    return res


def _blend_start(w_lo, r_lo, w_hi, r_hi, target):
    """A start that satisfies the target-return constraint by construction.

    Bounds, the budget constraint and the category bands are all convex, so any
    convex combination of two feasible portfolios is feasible; and expected
    return is linear in w, so the blend hits `target` exactly. That makes this a
    strictly better warm start than the previous grid point, which satisfies the
    OLD target and can start SLSQP outside the new feasible set.
    """
    if abs(r_hi - r_lo) < 1e-15:
        return w_lo
    t = float(np.clip((target - r_lo) / (r_hi - r_lo), 0.0, 1.0))
    return (1.0 - t) * w_lo + t * w_hi


def max_return(mu, cov, limits, w0=None, vol_budget=None):
    w0 = feasible_start(limits) if w0 is None else w0
    n = limits.n
    return _solve(lambda w: -(w @ mu), lambda w: -mu, mu, limits, w0,
                  hess=lambda w: np.zeros((n, n)),
                  vol_budget=vol_budget, cov=cov)


def max_growth(mu, cov, limits, w0=None):
    """maximize w'mu - 0.5 w'Sigma w — the continuous-time log growth rate.

    This is the allocation with the highest LONG-RUN MEDIAN wealth, which is not
    generally the maximum-Sharpe portfolio: Sharpe rewards the mean, growth
    charges for the variance drag the mean ignores. Since the app's headline is
    a 10-year median wealth multiple, this is often the objective the reader
    actually wants, so it sits alongside Sharpe rather than beneath it.
    """
    w0 = feasible_start(limits) if w0 is None else w0
    return _solve(lambda w: -(w @ mu - 0.5 * w @ cov @ w),
                  lambda w: -(mu - cov @ w), mu, limits, w0,
                  hess=lambda w: cov)


def _endpoints(mu, cov, limits, w0=None):
    """(w_lo, r_lo, w_hi, r_hi) — the minimum-variance and maximum-return ends."""
    w0 = feasible_start(limits) if w0 is None else w0
    lo_res = min_variance(mu, cov, limits, w0)
    hi_res = max_return(mu, cov, limits, w0)
    if not (lo_res.success and hi_res.success):
        raise Infeasible("could not bracket the frontier under these constraints")
    return lo_res.x, float(lo_res.x @ mu), hi_res.x, float(hi_res.x @ mu)


def efficient_frontier(mu, cov, limits, w0=None, points: int = 60):
    """(returns, vols, weights) along the frontier, low risk to high return."""
    w_lo, r_lo, w_hi, r_hi = _endpoints(mu, cov, limits, w0)
    if r_hi - r_lo < 1e-9:                      # bands pin the portfolio
        r, v = portfolio(w_lo, mu, cov)
        return np.array([r]), np.array([v]), w_lo.reshape(1, -1)

    rets, vols, ws = [], [], []
    for t in np.linspace(r_lo, r_hi, int(points)):
        res = min_variance(mu, cov, limits,
                           _blend_start(w_lo, r_lo, w_hi, r_hi, t),
                           target_return=t)
        if not res.success:
            continue                            # skip, never report unconverged
        w = np.clip(res.x, limits.lower, limits.upper)
        w = w / w.sum()
        r, v = portfolio(w, mu, cov)
        rets.append(r); vols.append(v); ws.append(w)
    if not rets:
        raise Infeasible("no frontier point converged under these constraints")
    return np.array(rets), np.array(vols), np.array(ws)


def max_sharpe(mu, cov, limits, w0=None, rf: float = 0.0, points: int = 60,
               refine: bool = True):
    """Frontier scan for a bracket, then a golden-section search on it.

    Returns (weights, ret, vol, sharpe). `refine=False` skips the golden-section
    stage and returns the best point of the scan — a few basis points coarser,
    but ~10x cheaper, which is what makes the stability loop affordable.

    WHY NOT SLSQP ON THE SHARPE RATIO DIRECTLY. The obvious approach — seed from
    the best frontier point and polish with SLSQP on -Sharpe — was measured and
    does not work: on random 6-asset problems it left a gap of 3e-4 in Sharpe
    and 9e-3 in individual weights against the closed form, and the error fell
    only when the frontier GRID was refined, which is proof the polish was
    contributing nothing. The Sharpe ratio is scale-invariant, so its Hessian is
    singular along the ray through w and SLSQP stalls.

    What works instead: Sharpe is unimodal along the efficient frontier, and the
    frontier is parameterised by a single number — the target return. So this
    reduces to a one-dimensional maximisation, and every evaluation of it is an
    exact convex QP. Golden-section needs no derivatives, cannot overshoot a
    bracket, and converges linearly; 60 iterations shrink the interval by
    0.618^60, which is far below the precision of the QP underneath it. The
    result now matches the closed form to ~1e-9.
    """
    rets, vols, ws = efficient_frontier(mu, cov, limits, w0, points)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(vols > 0, (rets - rf) / vols, -np.inf)
    best = int(np.nanargmax(s))
    best_w, best_s = ws[best], float(s[best])

    if refine and len(rets) >= 3:
        lo = rets[max(best - 1, 0)]
        hi = rets[min(best + 1, len(rets) - 1)]
        w_lo, r_lo, w_hi, r_hi = _endpoints(mu, cov, limits, w0)

        def at(r):
            """Sharpe of the minimum-variance portfolio at target return r.

            Started from the endpoint blend, never from the previous iterate:
            each evaluation must be independent, or a single bad solve poisons
            the rest of the search.
            """
            res = min_variance(mu, cov, limits,
                               _blend_start(w_lo, r_lo, w_hi, r_hi, r),
                               target_return=r)
            if not res.success:
                return -np.inf, None
            w = np.clip(res.x, limits.lower, limits.upper)
            tot = w.sum()
            if tot <= 0:
                return -np.inf, None
            return sharpe(w / tot, mu, cov, rf), w / tot

        INV_PHI = (np.sqrt(5.0) - 1.0) / 2.0
        a, b = lo, hi
        c, dpt = b - INV_PHI * (b - a), a + INV_PHI * (b - a)
        fc, wc = at(c)
        fd, wd = at(dpt)
        for _ in range(60):
            if b - a < 1e-12:
                break
            if fc > fd:
                b, dpt, fd, wd = dpt, c, fc, wc
                c = b - INV_PHI * (b - a)
                fc, wc = at(c)
            else:
                a, c, fc, wc = c, dpt, fd, wd
                dpt = a + INV_PHI * (b - a)
                fd, wd = at(dpt)
        for f, w in ((fc, wc), (fd, wd)):
            if w is not None and f > best_s:
                best_s, best_w = f, w

    w = np.clip(best_w, limits.lower, limits.upper)
    w = w / w.sum()
    r, v = portfolio(w, mu, cov)
    return w, r, v, sharpe(w, mu, cov, rf)


def stability(mu, cov, limits, rf, noise=0.005, trials=40, seed=20260722,
              points: int = 12):
    """Re-solve the tangency portfolio with mu perturbed by N(0, `noise`).

    Mean-variance weights are notoriously sensitive to the return estimates.
    Reporting the spread of each line's optimal weight across plausible
    estimation error is far more honest than a single set of decimals: a line
    whose optimal weight ranges 4%-38% is not a recommendation.

    Returns (p25, p75) arrays of optimal weight per line.
    """
    rng = np.random.default_rng(seed)
    out = []
    base = feasible_start(limits)
    for _ in range(int(trials)):
        mu_p = mu + rng.normal(0.0, noise, size=len(mu))
        try:
            w, *_ = max_sharpe(mu_p, cov, limits, base, rf, points=points,
                               refine=False)
            out.append(w)
        except (Infeasible, ValueError):
            continue
    if not out:
        return None, None
    A = np.array(out)
    return np.percentile(A, 25, axis=0), np.percentile(A, 75, axis=0)


def unconstrained_tangency(mu, cov, rf) -> np.ndarray:
    """Closed-form w proportional to inv(Sigma)(mu - rf), normalized.

    Only valid with no bounds and no bands. Kept because it is the strongest
    available check on the constrained solver: open every band and the two must
    agree.
    """
    x = np.linalg.solve(cov, mu - rf)
    return x / x.sum()


# --------------------------------------------------------------------------
# Feasible ranges for the two constrained objectives
#
# "Minimum volatility for a target return" and "Maximum return for a volatility
# budget" each take a parameter that is neither a bound nor a band, so
# _named_infeasibility() cannot see it. Ask for a target outside the achievable
# range and the constraint set is empty: the solver breaks the BUDGET row to
# chase the return row, comes back with weights summing to 1.0003, and a caller
# that renormalizes turns that into a clean-looking allocation. A 20% target
# rendered as a 13.21% portfolio that way.
#
# These functions compute what is actually reachable so the UI can offer a
# slider between the two ends rather than a box that accepts -20% to 60%.
# --------------------------------------------------------------------------
SPAN_TOL = 1e-7             # 0.001 bp — below this the range is a point


@dataclass(frozen=True)
class Span:
    """The achievable range of one objective's parameter, under the constraints."""
    metric: str             # "return" | "vol"
    lo: float               # decimals, not percent points
    hi: float
    w_lo: np.ndarray        # the portfolio at each end
    w_hi: np.ndarray
    degenerate: bool        # hi - lo below SPAN_TOL


def _lp(mu, limits, sense=+1):
    """min (sense=+1) or max (sense=-1) of mu'w over the full constraint set.

    HiGHS, not trust-constr. The objective is linear, so this is exact — and it
    also sidesteps a measured artifact: trust-constr's max-return solve lands on
    99.982% of a single line rather than 100%, which would put the top of the
    slider 0.17bp inside the true maximum.
    """
    n = limits.n
    A_ub, b_ub = [], []
    for name, a, blo, bhi in limits.band_rows():
        A_ub.append(-a); b_ub.append(-blo)
        A_ub.append(a); b_ub.append(bhi)
    res = linprog(c=sense * np.asarray(mu, dtype=float),
                  A_ub=np.array(A_ub) if A_ub else None,
                  b_ub=np.array(b_ub) if b_ub else None,
                  A_eq=np.ones((1, n)), b_eq=np.array([1.0]),
                  bounds=list(zip(limits.lower, limits.upper)), method="highs")
    if not res.success:
        raise Infeasible(_named_infeasibility(limits) or
                         "no allocation satisfies every constraint at once.")
    return np.clip(res.x, limits.lower, limits.upper)


def return_span(mu, cov, limits) -> Span:
    """Achievable arithmetic mu'w. Two LPs — cheap at any universe size.

    The low end is the minimum achievable RETURN, not the return of the
    minimum-variance portfolio. Between the two sits a thin band that is
    feasible but dominated — you can target it and get more risk for less
    return. Measured at 4.1bp with the bands open and 0.4bp with the default
    bands. It is included because it is genuinely reachable and excluding it
    would make the slider's own floor infeasible; it is not worth drawing.
    """
    w_lo = _lp(mu, limits, sense=+1)
    w_hi = _lp(mu, limits, sense=-1)
    lo, hi = float(w_lo @ mu), float(w_hi @ mu)
    return Span("return", lo, hi, w_lo, w_hi, hi - lo < SPAN_TOL)


def vol_span(mu, cov, limits) -> Span:
    """Volatility budgets that actually bind.

    Lower end: the global minimum-variance portfolio. Nothing feasible carries
    less risk, so a smaller budget is infeasible rather than conservative.

    Upper end: the volatility OF THE MAXIMUM-RETURN PORTFOLIO, not the highest
    feasible variance. Above that the budget stops binding — the answer is the
    max-return portfolio whatever the slider says — so a slider running further
    would have a dead zone at the top, which reads as a broken control.
    """
    w_lo = min_variance(mu, cov, limits, feasible_start(limits)).x
    w_lo = np.clip(w_lo, limits.lower, limits.upper)
    w_hi = _lp(mu, limits, sense=-1)
    lo = float(np.sqrt(max(w_lo @ cov @ w_lo, 0.0)))
    hi = float(np.sqrt(max(w_hi @ cov @ w_hi, 0.0)))
    if hi < lo:                      # only reachable from solver noise
        lo, hi = hi, lo
        w_lo, w_hi = w_hi, w_lo
    return Span("vol", lo, hi, w_lo, w_hi, hi - lo < SPAN_TOL)


def slider_bounds(span, step):
    """Round the achievable span INWARD onto the step grid.

    Never round(). Rounding outward puts the end notch OUTSIDE the feasible set,
    and at a 1bp step that happens at three of four measured Base-sleeve
    endpoints — by 0.22 to 0.41 bp. That is not a rounding nicety: a target
    0.2245 bp below the achievable minimum is what produced the empty constraint
    set, the non-converged solve and the renormalized 99.97%-cash allocation. It
    would put that bug at the DEFAULT POSITION of the control.

    Inward rounding also makes (hi - lo) an exact multiple of the step, so the
    top notch lands on the bound rather than somewhere short of it.
    """
    lo = math.ceil(span.lo / step - 1e-9) * step
    hi = math.floor(span.hi / step + 1e-9) * step
    if hi < lo:
        # The whole span is narrower than one step, so no grid point sits
        # inside it. Snapping to the nearest notch would land OUTSIDE the
        # feasible set — the exact failure this function exists to prevent —
        # so return the span's own midpoint, off-grid and certainly feasible.
        # The caller renders this as a pinned value, not a slider.
        mid = (span.lo + span.hi) / 2.0
        return mid, mid
    n = max(int(round((hi - lo) / step)), 0)
    hi = lo + n * step
    # Float drift on lo + n*step can put the ends a few ulps outside the span.
    return max(lo, span.lo), min(hi, span.hi) if hi > span.hi else hi
