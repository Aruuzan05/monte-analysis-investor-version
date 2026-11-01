"""
Two-view comparison — one portfolio, Keepp's assumptions vs a provider's.

Custom mode's Your mix tab lets the user build ONE portfolio from a
provider's catalogue (J.P. Morgan 2027 LTCMA or BlackRock CMA, whichever the
sidebar selects) and give each line Keepp's own expected return — and,
optionally, its own volatility. The same portfolio is then simulated twice:

    Keepp view   weights, Keepp returns, Keepp vols (default: the provider's)
    Provider     weights, the provider's published compound returns and vols

Every function takes the provider as `src` — the jpm_ltcma module or a
blackrock_cma.BlackRockSource; both expose the same interface (CHOICES,
LABEL, lookup, resolve, category_of, corr_for). It defaults to J.P. Morgan.

WHAT IS HELD FIXED, so the gap between the two results is the assumptions and
nothing else:

  * the weights — identical on both sides;
  * the correlations — the provider's matrix on both sides (Keepp publishes
    no matrix of its own; personal lines borrow their proxy's row, see
    personal_lines.py);
  * the random numbers — both runs use the same seed and the same list of
    assets, so engine.simulate draws the SAME standard-normal shocks z for
    both. Only the drift ln(1+g)/12 and the scale sigma/sqrt(12) applied to
    them differ ("common random numbers"): the difference between the two
    results carries almost no simulation noise, even at 1,000 paths.

BLACKROCK SPECIFICS. Its returns depend on the horizon and the scenario, so
`src` must be the BlackRockSource for the run's years and scenario — the app
passes its own SRC. Its Custom-only extras (J.P. Morgan's Commodities, Gold,
Digital Assets) are offered too; their "provider" figures are J.P. Morgan's,
exactly as everywhere else in BlackRock Custom mode.

PERSONAL LINES. "Personal Private Equity" / "Personal Private Credit" have no
provider figure of their own, so the provider side uses the proxy line's
published figures (J.P. Morgan Private Equity / Direct Lending, BlackRock US
private equity (buyout) / Global direct lending (unlevered)).

Returns are COMPOUND (geometric), like everywhere else in the app.
Pure numpy / pandas, no Streamlit import.
"""

import io

import numpy as np
import pandas as pd

from . import jpm_ltcma as jpm
from . import personal_lines as pl

CATEGORY_KEYS = list(jpm.CATEGORY_KEYS)
CATEGORY_NAME = dict(jpm.CATEGORY_NAME)

FRAME_COLS = ["Category", "Line", "Weight %",
              "Provider return %", "Keepp return %",
              "Provider vol %", "Keepp vol %"]


def choices(category_key, src=jpm):
    """Dropdown labels for one category: the provider's lines, its Custom-only
    extras, and Keepp's two personal lines under Alternatives."""
    return pl.choices(src, category_key)


def resolve(label, src=jpm):
    """Bare line name for a dropdown label (personal lines pass through)."""
    s = str(label).strip()
    return s if pl.is_personal(s) else src.resolve(s)


def label_of(name, src=jpm):
    return name if pl.is_personal(name) else src.LABEL.get(name, name)


def category_of(name, src=jpm):
    return pl.CATEGORY if pl.is_personal(name) else src.category_of(name)


def provider_figures(name, src=jpm):
    """(compound return %, vol %) the provider publishes for a line. A
    personal line takes its proxy's figures."""
    rv = pl.lookup(src, name)
    if rv is None:
        raise KeyError(f"not a line of {getattr(src, 'KEY', 'jpm')}: {name!r}")
    return float(rv[0]), float(rv[1])


def provider_reference(name, src=jpm):
    """The provider line whose figures the provider side uses."""
    return pl.proxy_of(src, name)


# Names used before BlackRock was supported.
jpm_figures = provider_figures
jpm_reference = provider_reference


def _cols(side):
    if side == "keepp":
        return "Keepp return %", "Keepp vol %"
    if side in ("provider", "jpm"):
        return "Provider return %", "Provider vol %"
    raise ValueError(f"side must be 'keepp' or 'provider', not {side!r}")


def build_frame(lines, weights, own_ret, own_vol, src=jpm) -> pd.DataFrame:
    """One row per listed line, ordered by category then the order added.

    `own_ret` / `own_vol` map line -> Keepp figure; a missing or blank entry
    falls back to the provider's (so an untouched row is identical on both
    sides and contributes nothing to the difference).
    """
    rows = []
    for key in CATEGORY_KEYS:
        for ln in lines:
            if category_of(ln, src) != key:
                continue
            jr, jv = provider_figures(ln, src)
            kr = own_ret.get(ln)
            kv = own_vol.get(ln)
            rows.append({
                "Category": CATEGORY_NAME[key],
                "Line": ln,
                "Weight %": float(weights.get(ln, 0.0) or 0.0),
                "Provider return %": jr,
                "Keepp return %": jr if kr is None or pd.isna(kr) else float(kr),
                "Provider vol %": jv,
                "Keepp vol %": jv if kv is None or pd.isna(kv) else float(kv),
            })
    return pd.DataFrame(rows, columns=FRAME_COLS)


def held(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["Weight %"] > 0].reset_index(drop=True)


def sim_inputs(frame: pd.DataFrame, side: str, src=jpm):
    """(weights, returns, vols, flat corr) tuples for run_sim.
    `side` is "keepp" or "provider". The provider's correlations on both.

    0% rows are KEPT: they change nothing in the portfolio, but the engine
    draws one shock per listed asset, so dropping them would change the
    random draws — and the Keepp result would no longer match the app's
    main run (Overview etc.) for the very same inputs."""
    h = frame.reset_index(drop=True)
    r_col, v_col = _cols(side)
    corr = pl.corr_for(src, list(h["Line"]))
    return (tuple(h["Weight %"]), tuple(h[r_col]), tuple(h[v_col]),
            tuple(np.asarray(corr, dtype=float).flatten()))


def attribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Per line: how much its return / vol view moves the weighted inputs.

    `Return effect pp` = w_i (g_keepp - g_jpm): the line's share of the
    difference in the weighted-average compound return. They sum exactly to
    the gap in the back-of-envelope portfolio return; the simulated median
    gap also includes the volatility effects (see engine.return_ladder).
    """
    h = held(frame).copy()
    w = h["Weight %"] / 100.0
    h["Return Δ pp"] = h["Keepp return %"] - h["Provider return %"]
    h["Vol Δ pp"] = h["Keepp vol %"] - h["Provider vol %"]
    h["Return effect pp"] = w * h["Return Δ pp"]
    return h


def headline(res, amount: float) -> dict:
    """The figures the comparison shows for one run (decimals, dollars)."""
    t = res.rows[-1]
    return {
        "Typical value": t["Median x"] * amount,
        "Weak markets (1 in 20)": t["5th x"] * amount,
        "Strong markets (1 in 20)": t["95th x"] * amount,
        "Typical return / yr": t["Median"],
        "Average return / yr": res.exp_arith,
        "Volatility": res.analytic_vol,
        "Chance of a loss": t["P(loss)"],
        "Bad year (1 in 20)": res.exp_arith - 1.645 * res.analytic_vol,
    }


# Higher is better for every headline except these two.
LOWER_IS_BETTER = {"Volatility", "Chance of a loss"}
MONEY_ROWS = {"Typical value", "Weak markets (1 in 20)",
              "Strong markets (1 in 20)"}


def summary(res_keepp, res_jpm, amount: float,
            name: str = "J.P. Morgan") -> pd.DataFrame:
    """Metric | Keepp | <provider> | Difference (Keepp − provider)."""
    a, b = headline(res_keepp, amount), headline(res_jpm, amount)
    return pd.DataFrame([{"Metric": k, "Keepp": a[k], name: b[k],
                          "Difference": a[k] - b[k]} for k in a])


def horizons(res_keepp, res_jpm, amount: float,
             name: str = "J.P. Morgan") -> pd.DataFrame:
    """Typical value and return at every horizon both runs report."""
    out = []
    for ra, rb in zip(res_keepp.rows, res_jpm.rows):
        out.append({
            "Horizon": ra["Horizon"],
            "Keepp typical value": ra["Median x"] * amount,
            f"{name} typical value": rb["Median x"] * amount,
            "Keepp typical return / yr": ra["Median"],
            f"{name} typical return / yr": rb["Median"],
            "Keepp chance of loss": ra["P(loss)"],
            f"{name} chance of loss": rb["P(loss)"],
        })
    return pd.DataFrame(out)


def fan_long(res_keepp, res_jpm, amount: float,
             name: str = "J.P. Morgan") -> pd.DataFrame:
    """Both fans in long form for one overlaid chart."""
    parts = []
    for view, res in (("Keepp", res_keepp), (name, res_jpm)):
        f = pd.DataFrame(np.asarray(res.fan) * amount,
                         columns=["p05", "p25", "p50", "p75", "p95"])
        f["Year"] = res.fan_years
        f["View"] = view
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def to_excel_bytes(frame, summ, hz, meta: dict,
                   name: str = "J.P. Morgan") -> bytes:
    """Inputs, headline comparison and horizons in one workbook."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame(list(meta.items()), columns=["Setting", "Value"]).to_excel(
            xw, sheet_name="Settings", index=False)
        att = attribution(frame).rename(columns=lambda c: c.replace(
            "Provider", name))
        att.to_excel(xw, sheet_name="Assumptions", index=False)
        summ.to_excel(xw, sheet_name="Comparison", index=False)
        hz.to_excel(xw, sheet_name="By horizon", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(
                    max(10, width + 2), 48)
    return buf.getvalue()


# --------------------------------------------------------------------------
# Custom mode — the Your mix tab edits the app's own Custom
# portfolio, `custom_alloc` (Category, LTCMA line, Weight %, Return %, Vol %),
# where Return % / Vol % ARE the Keepp figures. These two helpers translate
# between that table and the comparison frame above.
# --------------------------------------------------------------------------
def frame_from_alloc(alloc: pd.DataFrame, src=jpm) -> pd.DataFrame:
    """Comparison frame from a Custom-mode allocation table.

    Rows whose line the provider does not know are left out (the caller warns).
    A line listed twice is merged: weights added, first row's figures kept.
    """
    lines, w, r, v = [], {}, {}, {}
    for _, row in alloc.iterrows():
        ln = resolve(row["LTCMA line"], src)
        if category_of(ln, src) is None:
            continue
        wt = pd.to_numeric(row["Weight %"], errors="coerce")
        wt = 0.0 if pd.isna(wt) else float(wt)
        if ln in w:
            w[ln] += wt
            continue
        lines.append(ln)
        w[ln] = wt
        r[ln] = pd.to_numeric(row["Return %"], errors="coerce")
        v[ln] = pd.to_numeric(row["Vol %"], errors="coerce")
    return build_frame(lines, w, r, v, src)


def to_alloc(frame: pd.DataFrame, side: str = "keepp", src=jpm) -> pd.DataFrame:
    """Back to the app's allocation table, on either side's figures — the
    Keepp side is what Custom mode simulates, the provider side is what
    gets saved as the second scenario."""
    r_col, v_col = _cols(side)
    return pd.DataFrame({
        "Category": frame["Category"],
        "LTCMA line": frame["Line"].map(lambda n: label_of(n, src)),
        "Weight %": frame["Weight %"],
        "Return %": frame[r_col],
        "Vol %": frame[v_col],
    }).reset_index(drop=True)


# --------------------------------------------------------------------------
# Saved scenarios — which view a saved run is, and pairing the two views.
#
# A run saved with "Save both scenarios" is two independent records with no
# link stored between them. Both facts are DERIVED instead, so they also work
# for every run saved before this existed:
#
#   view  — "provider" if every holding's Return % and Vol % equal the
#           provider's published figures (for that run's source, scenario and
#           horizon), else "keepp". Standard-mode runs are always "provider".
#   pair  — two runs are the two views of one portfolio when they share the
#           provider, horizon, number of futures, seed and the same holdings
#           at the same weights, in the same order.
# --------------------------------------------------------------------------
TOL = 1e-6


def classify(record, src) -> str:
    """'keepp' or 'provider' for a saved RunRecord."""
    if record.mode == "Base":
        return "provider"
    for s in record.sleeves:
        ln = resolve(s["LTCMA line"], src)
        if category_of(ln, src) is None:
            return "keepp"                   # not a provider line: own figures
        pr, pv = provider_figures(ln, src)
        if (abs(float(s["Return %"]) - pr) > TOL
                or abs(float(s["Vol %"]) - pv) > TOL):
            return "keepp"
    return "provider"


def pair_key(record, src) -> tuple:
    return (str(record.source), int(record.years), int(record.paths),
            int(record.seed),
            tuple((resolve(s["LTCMA line"], src), round(float(s["Weight %"]), 6))
                  for s in record.sleeves))


def run_metrics(record=None, result=None) -> dict:
    """The return figures the comparison shows, from a saved record or from
    an engine Result. Decimals per year; 'Typical ×' is terminal wealth."""
    if result is not None:
        t = result.rows[-1]
        return {"Average return / yr": result.exp_arith,
                "Typical return / yr": t["Median"],
                "Volatility": result.analytic_vol,
                "Typical ×": t["Median x"],
                "Chance of a loss": t["P(loss)"]}
    t = record.horizon_rows[-1] if record.horizon_rows else {}
    return {"Average return / yr": record.exp_return_pct / 100.0,
            "Typical return / yr": t.get("Median", float("nan")),
            "Volatility": record.analytic_vol_pct / 100.0,
            "Typical ×": record.median_wealth_x,
            "Chance of a loss": record.p_loss_pct / 100.0}


def pair_up(items):
    """Group (run, view, key) triples into comparison rows.

    `items` is a sequence of (run, view, key) where `run` is anything (the
    app passes StoredRun), `view` comes from classify() and `key` from
    pair_key(). Returns a list of {"keepp": run|None, "provider": run|None}
    in the order the runs first appear; a key with several runs of one view
    keeps the first (newest, as the app lists them).
    """
    rows, where = [], {}
    for run, view, key in items:
        i = where.get(key)
        if i is None:
            where[key] = len(rows)
            rows.append({"keepp": None, "provider": None})
            i = where[key]
        if rows[i][view] is None:
            rows[i][view] = run
        elif rows[i]["keepp" if view == "provider" else "provider"] is None:
            pass                         # duplicate of the same view: ignore
    return rows
