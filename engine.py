"""
Monte Carlo engine for the Keepp Capital target allocation.

Ported 1:1 from Keepp_Capital_Target_Allocation_Monte_Carlo.xlsx:
  - Cholesky decomposition of the correlation matrix -> correlated normals
  - monthly log drift  m_i = ln(1 + g_i) / 12
  - monthly log vol    s_i = sigma_i / sqrt(12)
  - asset monthly simple return  r_i = exp(m_i + s_i * z_i) - 1
  - portfolio return  sum_i w_i * r_i   (i.e. rebalanced to target weights monthly)

No I/O, no Streamlit imports — kept separate so it can be unit-tested headlessly.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

PCTS = (0.05, 0.25, 0.50, 0.75, 0.95)


@dataclass(frozen=True)
class Result:
    rows: list          # per-horizon summary dicts
    fan: np.ndarray     # (n_snapshots, 5) wealth percentiles
    fan_years: np.ndarray
    sim_vol: float
    analytic_vol: float
    exp_arith: float
    paths: int
    years: int
    horizons: list
    elapsed_ms: float


def cholesky_safe(corr: np.ndarray) -> np.ndarray:
    """Cholesky factor, repairing a matrix that is not quite positive definite.

    A user editing correlations by hand can easily produce an inconsistent
    matrix (e.g. A-B 0.9, B-C 0.9, A-C -0.9). Rather than failing, we clip the
    negative eigenvalues to a small positive floor and rescale to unit diagonal,
    which is the nearest usable correlation matrix.
    """
    corr = np.asarray(corr, dtype=float)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    try:
        return np.linalg.cholesky(corr), False
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(corr)
        vals = np.clip(vals, 1e-8, None)
        repaired = vecs @ np.diag(vals) @ vecs.T
        d = np.sqrt(np.diag(repaired))
        repaired = repaired / np.outer(d, d)
        np.fill_diagonal(repaired, 1.0)
        return np.linalg.cholesky(repaired), True


def simulate(
    weights: Sequence[float],
    rets: Sequence[float],
    vols: Sequence[float],
    corr: Sequence[Sequence[float]],
    paths: int = 10_000,
    years: int = 10,
    seed: int = 20260722,
) -> Result:
    """All inputs in percent (weights 45.0, returns 5.0, vols 7.02)."""
    import time
    t0 = time.perf_counter()

    w = np.asarray(weights, dtype=float) / 100.0
    g = np.asarray(rets, dtype=float) / 100.0
    s_ann = np.asarray(vols, dtype=float) / 100.0
    n = len(w)
    months = years * 12

    L, repaired = cholesky_safe(np.asarray(corr, dtype=float))

    m = np.log1p(g) / 12.0            # monthly log drift
    s = s_ann / np.sqrt(12.0)         # monthly log vol

    # snapshot months: coarse enough to keep memory flat, always including horizons
    horizons = [h for h in (3, 5, 10, 15, 20, 25, 30) if h <= years]
    if not horizons or horizons[-1] != years:
        horizons.append(years)
    step = 12 if months > 180 else (6 if months > 72 else 3)
    snap = sorted({0, months} | set(range(step, months + 1, step))
                  | {h * 12 for h in horizons})
    snap_pos = {t: i for i, t in enumerate(snap)}

    rng = np.random.default_rng(seed)
    W = np.ones(paths, dtype=float)
    store = np.empty((len(snap), paths), dtype=float)
    store[0] = 1.0

    sum_r = 0.0
    sum_r2 = 0.0

    # Vectorised across paths, looped over months. 120 iterations of a
    # (paths x n) matmul is far cheaper in numpy than a per-path Python loop.
    for t in range(1, months + 1):
        z = rng.standard_normal((paths, n)) @ L.T
        r = np.exp(m + s * z) - 1.0
        pr = r @ w
        sum_r += pr.sum()
        sum_r2 += np.square(pr).sum()
        W *= 1.0 + pr
        pos = snap_pos.get(t)
        if pos is not None:
            store[pos] = W

    cnt = paths * months
    mean_r = sum_r / cnt
    sim_vol = float(np.sqrt(max(sum_r2 / cnt - mean_r**2, 0.0) * 12.0))

    # analytics, matching the workbook's cells C13 / C14
    exp_arith = float(np.dot(w, g + 0.5 * s_ann**2))
    cov = np.outer(s_ann, s_ann) * np.asarray(corr, dtype=float)
    analytic_vol = float(np.sqrt(max(w @ cov @ w, 0.0)))

    store.sort(axis=1)
    q = np.quantile(store, PCTS, axis=1, method="linear").T  # (n_snap, 5)

    rows = []
    for h in horizons:
        col = store[snap_pos[h * 12]]
        ann = col ** (1.0 / h) - 1.0
        wq = np.quantile(col, PCTS, method="linear")
        rows.append({
            "Horizon": f"{h} years",
            "_years": h,
            "5th": float(wq[0] ** (1 / h) - 1),
            "25th": float(wq[1] ** (1 / h) - 1),
            "Median": float(wq[2] ** (1 / h) - 1),
            "75th": float(wq[3] ** (1 / h) - 1),
            "95th": float(wq[4] ** (1 / h) - 1),
            "Mean": float(ann.mean()),
            "P(loss)": float((col < 1.0).mean()),
            "Median x": float(wq[2]),
            "5th x": float(wq[0]),
            "95th x": float(wq[4]),
        })

    return Result(
        rows=rows,
        fan=q,
        fan_years=np.array(snap, dtype=float) / 12.0,
        sim_vol=sim_vol,
        analytic_vol=analytic_vol,
        exp_arith=exp_arith,
        paths=paths,
        years=years,
        horizons=horizons,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
    ), repaired


# expand_correlation() used to fabricate an N x N matrix from a 6x6 of general
# asset types plus a within-type guess. It is gone: jpm_ltcma.corr_for() now
# returns a sub-matrix of J.P. Morgan's published 59 x 59, so there is nothing
# left to fabricate.
