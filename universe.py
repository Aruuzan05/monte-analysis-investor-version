"""
Whole-universe optimization — optimize over every J.P. Morgan LTCMA line
rather than only the sleeves already in the allocation, and present the answer
the way a target allocation is actually read: category totals first, then the
lines inside each category.

Pure numpy / pandas / optimizer. No Streamlit import, so it is unit-testable
headlessly like engine.py, optimizer.py and run_store.py. It drives
optimizer.py; it does not reimplement any of it.

WHY THIS IS NOT SIMPLY "PASS MORE LINES TO optimizer.py".
Two things change at 58 lines that do not matter at six.

  1. COST. Every quadratic program trust-constr solves costs about a second at
     n=58 against roughly a millisecond at n=6 — measured at ~31 iterations a
     solve, all converging, so the bill is the NUMBER of solves and nothing
     else. The panel's Sharpe path solves a 60-point frontier for the chart,
     then max_sharpe() solves its own 60-point frontier again and golden-
     sections it 60 times: about 180 solves, or three minutes. Here the
     frontier is solved ONCE at a coarser grid and the tangency portfolio is
     refined on that same frontier, which is ~35 solves, or about 20 seconds.

  2. DUST. A 58-line optimum comes back with a long tail of lines carrying
     0.01% — solver residue, not an allocation. Measured on the Base case with
     no per-line cap: 43 of 58 lines "carry weight", but 35 of them hold less
     than 0.05% between them. Printing all 43 rows buries the eight that
     matter. `breakdown()` folds everything below a threshold into a single
     residual row rather than dropping it silently, so each column still adds
     up to its category total and nothing is hidden.
"""

import numpy as np
import pandas as pd

import jpm_ltcma as jpm
import optimizer as opt

# Every investable line. The published correlation matrix carries a 59th row,
# U.S. Inflation, which is not investable and so is never a candidate weight.
INVESTABLE = [name for name, *_ in jpm.LINES]

ALLOC_COLS = ["Category", "LTCMA line", "Weight %", "Return %", "Vol %"]

# Weights below this (in percent points) print as 0.00% at the two decimals
# this app shows everywhere, and at 58 lines they are solver residue rather
# than an allocation. Folded into a residual row, never silently dropped.
DUST_PCT = 0.05

RESIDUAL_LINE = "— below {pct:.2f}%, {n} line{s} —"


def _num(x, default=0.0) -> float:
    """A float, treating NaN and unparseable cells as `default`.

    `float(pd.to_numeric(x) or default)` does NOT do this: NaN is truthy, so
    a blank cell would sail through as NaN and poison every sum downstream.
    """
    v = pd.to_numeric(x, errors="coerce")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return default if v != v else v


def universe_alloc(alloc: pd.DataFrame) -> pd.DataFrame:
    """All 58 investable lines as an allocation frame, carrying the user's own
    numbers wherever they have them.

    The frame is deliberately in exactly the shape app.py's `alloc` already
    has, so the whole optimizer panel — correlations, categories, bands, the
    current-portfolio point — works against it unchanged.

    A line the current allocation already holds keeps that allocation's
    Weight, Return and Vol, INCLUDING any override the user typed. That is
    what stops Base's three Keepp-sourced sleeves (the global IG blend, MSCI
    World realised history and the live 3-month T-bill) from silently
    reverting to J.P. Morgan's published figures the moment the scope widens.
    Every other line arrives at 0% on J.P. Morgan's compound return and vol.

    Two sleeves sitting on one line — possible in Custom — are summed into
    that line's weight, and the first occurrence supplies Return and Vol:
    averaging two hand-typed overrides would invent a third number that
    neither sleeve asked for.
    """
    held = {}
    for _, row in alloc.iterrows():
        name = jpm.resolve(row["LTCMA line"])
        if name not in jpm.BY_NAME:
            continue
        w = _num(row.get("Weight %"))
        if name in held:
            held[name]["Weight %"] += w
        else:
            held[name] = {"Weight %": w,
                          "Return %": _num(row.get("Return %")),
                          "Vol %": _num(row.get("Vol %"))}

    rows = []
    for name, comp, _arith, vol, _c25, cat in jpm.LINES:
        h = held.get(name)
        rows.append({
            "Category": jpm.CATEGORY_NAME[cat],
            "LTCMA line": jpm.LABEL[name],
            "Weight %": h["Weight %"] if h else 0.0,
            "Return %": h["Return %"] if h else comp,
            "Vol %": h["Vol %"] if h else vol,
        })
    return pd.DataFrame(rows, columns=ALLOC_COLS)


def _blend(w_lo, r_lo, w_hi, r_hi, target):
    """A start that satisfies the target-return constraint by construction.

    Bounds, the budget constraint and the category bands are all convex, so
    any convex combination of two feasible portfolios is feasible; expected
    return is linear in w, so the blend hits `target` exactly. Same reasoning
    as optimizer._blend_start, kept local so this module does not reach into
    another module's private API.
    """
    if abs(r_hi - r_lo) < 1e-15:
        return w_lo
    t = float(np.clip((target - r_lo) / (r_hi - r_lo), 0.0, 1.0))
    return (1.0 - t) * w_lo + t * w_hi


def tangency_on_frontier(mu, cov, limits, frontier, rf, iters: int = 12):
    """(weights, sharpe) for the maximum-Sharpe portfolio, refined on a
    frontier that has ALREADY been solved.

    optimizer.max_sharpe() is the reference implementation and stays the
    reference — test_universe.py checks this agrees with it. The difference is
    only which frontier gets scanned: max_sharpe() solves one of its own,
    which at n=58 is the single most expensive thing the panel does and which
    the panel has already paid for once to draw the chart.

    The search itself is max_sharpe()'s: Sharpe is unimodal along the
    efficient frontier and the frontier is parameterised by one number, the
    target return, so this is a one-dimensional maximisation whose every
    evaluation is an exact convex QP. The QP is NOT reimplemented here —
    optimizer.min_variance() solves each one. Only the search loop is local.

    `iters` golden-section steps shrink the starting bracket by 0.618^iters.
    A 24-point grid over a 4.7pp return range brackets the optimum within
    ~0.4pp, so 12 steps take that below 0.002pp — far finer than the return
    estimates underneath it, and 12 solves rather than 60.
    """
    rets, vols, ws = frontier
    rets = np.asarray(rets, dtype=float)
    vols = np.asarray(vols, dtype=float)
    ws = np.asarray(ws, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(vols > 0, (rets - rf) / vols, -np.inf)
    best = int(np.nanargmax(s))
    best_w, best_s = ws[best], float(s[best])
    if len(rets) < 3 or iters <= 0:
        return best_w, best_s

    # ws is ordered low-risk to high-return, so its ends ARE the frontier's
    # endpoints — no need to re-solve _endpoints() for the blend start.
    w_lo, r_lo = ws[0], float(rets[0])
    w_hi, r_hi = ws[-1], float(rets[-1])

    def at(r):
        """Sharpe of the minimum-variance portfolio at target return r.

        Started from the endpoint blend, never from the previous iterate: each
        evaluation must be independent, or one bad solve poisons the search.
        """
        res = opt.min_variance(mu, cov, limits,
                               _blend(w_lo, r_lo, w_hi, r_hi, r),
                               target_return=r)
        if not res.success:
            return -np.inf, None
        w = np.clip(res.x, limits.lower, limits.upper)
        tot = w.sum()
        if tot <= 0:
            return -np.inf, None
        w = w / tot
        return opt.sharpe(w, mu, cov, rf), w

    INV_PHI = (np.sqrt(5.0) - 1.0) / 2.0
    a = float(rets[max(best - 1, 0)])
    b = float(rets[min(best + 1, len(rets) - 1)])
    c, d = b - INV_PHI * (b - a), a + INV_PHI * (b - a)
    fc, wc = at(c)
    fd, wd = at(d)
    for _ in range(int(iters)):
        if b - a < 1e-12:
            break
        if fc > fd:
            b, d, fd, wd = d, c, fc, wc
            c = b - INV_PHI * (b - a)
            fc, wc = at(c)
        else:
            a, c, fc, wc = c, d, fd, wd
            d = a + INV_PHI * (b - a)
            fd, wd = at(d)
    for f, w in ((fc, wc), (fd, wd)):
        if w is not None and f > best_s:
            best_s, best_w = f, w
    return best_w, best_s


def category_table(cats, cur_w, opt_w) -> pd.DataFrame:
    """Target allocation by J.P. Morgan category — the top-level answer.

    Weights in decimals; the frame is in percent points.
    """
    cur_w = np.asarray(cur_w, dtype=float)
    opt_w = np.asarray(opt_w, dtype=float)
    order = [jpm.CATEGORY_NAME[k] for k in jpm.CATEGORY_KEYS]
    seen = [c for c in order if c in set(cats)]
    seen += [c for c in dict.fromkeys(cats) if c not in seen]

    rows = []
    for cat in seen:
        idx = np.array([i for i, c in enumerate(cats) if c == cat])
        cur = float(cur_w[idx].sum()) * 100
        new = float(opt_w[idx].sum()) * 100
        rows.append({"Category": cat, "Current %": cur, "Optimal %": new,
                     "Change (pp)": new - cur})
    return pd.DataFrame(rows,
                        columns=["Category", "Current %", "Optimal %", "Change (pp)"])


def line_tables(lines, cats, cur_w, opt_w, band=None, dust_pct: float = DUST_PCT):
    """{category: per-line frame} — the detail under each category total.

    A line appears when EITHER the optimum or the current portfolio puts at
    least `dust_pct` on it. "Omit the zero weights" cannot mean "omit what we
    already hold": a line the optimizer zeroes out is the most important row
    on the page when the portfolio is sitting on it today.

    Everything below the threshold is summed into one residual row rather than
    dropped, so each column still adds up to that category's total. `band` is
    the optional (p25, p75) stability pair, per line, in decimals.
    """
    cur_w = np.asarray(cur_w, dtype=float)
    opt_w = np.asarray(opt_w, dtype=float)
    p25, p75 = band if band is not None else (None, None)
    floor = dust_pct / 100.0

    out = {}
    for cat in dict.fromkeys(cats):
        idx = [i for i, c in enumerate(cats) if c == cat]
        keep = [i for i in idx if opt_w[i] >= floor or cur_w[i] >= floor]
        drop = [i for i in idx if i not in set(keep)]
        keep.sort(key=lambda i: (-opt_w[i], -cur_w[i]))

        rows = []
        for i in keep:
            name = lines[i]
            row = {"LTCMA line": jpm.LABEL.get(name, name),
                   "Current %": cur_w[i] * 100,
                   "Optimal %": opt_w[i] * 100,
                   "Change (pp)": (opt_w[i] - cur_w[i]) * 100}
            if p25 is not None:
                row["Stable range %"] = f"{p25[i]*100:.0f} – {p75[i]*100:.0f}"
            rows.append(row)

        resid_opt = float(opt_w[drop].sum()) if drop else 0.0
        resid_cur = float(cur_w[drop].sum()) if drop else 0.0
        if drop and (resid_opt > 0 or resid_cur > 0):
            row = {"LTCMA line": RESIDUAL_LINE.format(
                       pct=dust_pct, n=len(drop), s="s" if len(drop) != 1 else ""),
                   "Current %": resid_cur * 100,
                   "Optimal %": resid_opt * 100,
                   "Change (pp)": (resid_opt - resid_cur) * 100}
            if p25 is not None:
                row["Stable range %"] = ""
            rows.append(row)

        cols = ["LTCMA line", "Current %", "Optimal %", "Change (pp)"]
        if p25 is not None:
            cols.append("Stable range %")
        out[cat] = pd.DataFrame(rows, columns=cols)
    return out


def applied_sleeves(uni_alloc: pd.DataFrame, opt_w,
                    dust_pct: float = DUST_PCT) -> pd.DataFrame:
    """The optimum as an allocation frame, ready to become the Custom sleeves.

    Only lines at or above the dust threshold survive, and the survivors are
    RENORMALIZED to total exactly 100%. Both halves matter: without the
    threshold the user inherits 40-odd sleeves holding 0.01% each, and without
    the renormalization the dropped dust leaves the table at 99.4%, which the
    app refuses to simulate until it is fixed by hand.

    Return and Vol are taken from `uni_alloc` — the frame that was actually
    optimized — and NOT re-read from the catalogue, so a line carrying the
    user's own override keeps it. Re-reading would hand back an allocation
    whose numbers differ from the ones the optimum was computed on.
    """
    opt_w = np.asarray(opt_w, dtype=float)
    floor = dust_pct / 100.0
    keep = [i for i in range(len(uni_alloc)) if opt_w[i] >= floor]
    if not keep:
        keep = [int(np.argmax(opt_w))]
    total = float(opt_w[keep].sum())

    src = uni_alloc.reset_index(drop=True)
    rows = []
    for i in sorted(keep, key=lambda j: (src.at[j, "Category"], -opt_w[j])):
        rows.append({"Category": src.at[i, "Category"],
                     "LTCMA line": src.at[i, "LTCMA line"],
                     "Weight %": round(float(opt_w[i]) / total * 100, 4),
                     "Return %": _num(src.at[i, "Return %"]),
                     "Vol %": _num(src.at[i, "Vol %"])})
    return pd.DataFrame(rows, columns=ALLOC_COLS)
