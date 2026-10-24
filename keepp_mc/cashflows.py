"""
Contributions and withdrawals — the "Contributions & withdrawals" tab.

A SEPARATE simulation beside engine.py, which is left untouched: the main tabs
keep their lump-sum, per-$1 model. This module runs the same Monte Carlo with
money moving in and out along the way.

THE ONE CHANGE TO THE MATH
engine.py grows $1 with no cash flows:          W_t = W_{t-1} (1 + r_t)
here the portfolio is tracked in DOLLARS and a cash flow C_t is added each
month (positive = contribution, negative = withdrawal):

    end of period    V_t = V_{t-1} (1 + r_t) + C_t
    start of period  V_t = (V_{t-1} + C_t)(1 + r_t)

Everything else is identical to engine.simulate(): the same assumptions, the
same Cholesky factor, the same monthly lognormal draws IN THE SAME ORDER from
the same seed, the same monthly rebalancing. So with no cash flows this
reproduces engine.simulate()'s wealth paths exactly (tests check it), and any
difference between the two tabs is caused by the cash flows alone.

RUNNING OUT
If a withdrawal is larger than what is left, what is left is paid out and the
path is DEPLETED in that month: its value stays at 0 and every later flow on
that path is skipped (contributions included). "Chance the money lasts" is the
share of paths never depleted.

CASH-FLOW SCHEDULE
A list of rows, each: kind (contribution / withdrawal), amount in TODAY's
dollars, frequency (monthly / yearly), first and last year (1-based,
inclusive), and a yearly growth % (e.g. 2.5 to keep pace with inflation).
In year y the amount paid is  amount * (1 + growth)^(y - 1)  — grown from
today, so a 2.5% indexed $50,000 is $51,250 in year 2 whether the row starts
in year 1 or year 2. Yearly flows fall in the last month of the year (end of
period) or the first (start of period).

Pure numpy. No Streamlit import.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .engine import cholesky_safe

PCTS = (0.05, 0.25, 0.50, 0.75, 0.95)
CONTRIBUTION, WITHDRAWAL = "Contribution", "Withdrawal"
MONTHLY, YEARLY = "Monthly", "Yearly"
END, START = "End of period", "Start of period"


@dataclass(frozen=True)
class Flow:
    kind: str            # CONTRIBUTION | WITHDRAWAL
    amount: float        # dollars per payment, in today's money (positive)
    frequency: str       # MONTHLY | YEARLY
    first_year: int      # 1-based, inclusive
    last_year: int       # inclusive
    growth_pct: float = 0.0


def schedule(flows: Sequence[Flow], years: int, timing: str = END) -> np.ndarray:
    """Monthly cash-flow vector c[0..12*years-1] in dollars (+ in, - out)."""
    c = np.zeros(int(years) * 12)
    for f in flows:
        amt = abs(float(f.amount))
        if amt == 0 or f.last_year < f.first_year:
            continue
        sign = 1.0 if f.kind == CONTRIBUTION else -1.0
        g = float(f.growth_pct) / 100.0
        for y in range(max(1, int(f.first_year)), min(int(years), int(f.last_year)) + 1):
            pay = sign * amt * (1.0 + g) ** (y - 1)
            if f.frequency == MONTHLY:
                c[(y - 1) * 12:(y - 1) * 12 + 12] += pay
            else:
                c[(y - 1) * 12 + (0 if timing == START else 11)] += pay
    return c


@dataclass(frozen=True)
class CashflowResult:
    years: int
    paths: int
    amount: float
    flows: np.ndarray            # the monthly schedule, dollars
    year_marks: np.ndarray       # 0..years
    value_pct: np.ndarray        # (years+1, 5) dollar percentiles at each year end
    p_depleted: np.ndarray       # (years+1,) share of paths depleted by each year end
    final: np.ndarray            # (paths,) ending dollar value
    depleted: np.ndarray         # (paths,) bool
    depletion_month: np.ndarray  # (paths,) month index (1-based), inf if never
    paid_in: float               # planned contributions, dollars
    paid_out_planned: float      # planned withdrawals, dollars
    paid_out_mean: float         # average actually withdrawn (less if depleted)
    twr_median: float            # typical yearly growth of the investments alone
    no_flow_median: float        # median ending value with no cash flows, dollars


def simulate_cashflows(weights, rets, vols, corr, amount: float,
                       flows: Sequence[Flow], paths: int = 10_000,
                       years: int = 10, seed: int = 20260722,
                       timing: str = END) -> CashflowResult:
    """All return inputs in percent, as engine.simulate(). `amount` in dollars."""
    w = np.asarray(weights, dtype=float) / 100.0
    g = np.asarray(rets, dtype=float) / 100.0
    s_ann = np.asarray(vols, dtype=float) / 100.0
    n = len(w)
    months = int(years) * 12
    L, _ = cholesky_safe(np.asarray(corr, dtype=float))
    m = np.log1p(g) / 12.0
    s = s_ann / np.sqrt(12.0)
    c = schedule(flows, years, timing)

    rng = np.random.default_rng(seed)          # same stream as engine.simulate
    V = np.full(paths, float(amount))
    W = np.ones(paths)                         # $1 with no flows (investment growth)
    dead = np.zeros(paths, dtype=bool)
    dmonth = np.full(paths, np.inf)
    paid_out = np.zeros(paths)
    snaps = np.empty((int(years) + 1, paths))
    dsnap = np.zeros(int(years) + 1)
    snaps[0] = V

    for t in range(1, months + 1):
        z = rng.standard_normal((paths, n)) @ L.T
        r = np.exp(m + s * z) - 1.0
        pr = r @ w
        W *= 1.0 + pr
        ct = c[t - 1]
        alive = ~dead
        if timing == START:
            V, out = _apply_flow(V, ct, alive)
            V = V * (1.0 + pr)
        else:
            V = V * (1.0 + pr)
            V, out = _apply_flow(V, ct, alive)
        paid_out += out
        newly = alive & (V <= 0.0) & (ct < 0)
        dmonth[newly] = t
        dead |= newly
        V[dead] = 0.0
        if t % 12 == 0:
            snaps[t // 12] = V
            dsnap[t // 12] = dead.mean()

    value_pct = np.quantile(snaps, PCTS, axis=1, method="linear").T
    return CashflowResult(
        years=int(years), paths=int(paths), amount=float(amount), flows=c,
        year_marks=np.arange(int(years) + 1), value_pct=value_pct,
        p_depleted=dsnap, final=V, depleted=dead, depletion_month=dmonth,
        paid_in=float(c[c > 0].sum()), paid_out_planned=float(-c[c < 0].sum()) + 0.0,
        paid_out_mean=float(paid_out.mean()),
        twr_median=float(np.median(W) ** (1.0 / years) - 1.0),
        no_flow_median=float(np.median(W) * amount),
    )


def _apply_flow(V, ct, alive):
    """Add this month's flow to living paths. Returns (V, amount withdrawn)."""
    out = np.zeros_like(V)
    if ct == 0:
        return V, out
    V = V.copy()
    if ct > 0:
        V[alive] += ct
        return V, out
    take = np.minimum(V, -ct)                  # cannot withdraw more than is left
    take[~alive] = 0.0
    V -= take
    out += take
    return V, out
