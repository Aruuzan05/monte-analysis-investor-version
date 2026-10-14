"""
Saved run files — run record, session history, and Excel export.

Captures a single simulation run (its inputs and its summary outputs) as a
RunRecord, keeps a growing in-session list of them, and can write that list
out as a multi-sheet .xlsx workbook a PM can actually open and archive.

No Streamlit import: st.session_state is just a list the caller (app.py)
owns and passes in, so this module is unit-testable headlessly, the same way
engine.py and optimizer.py are. It does not touch either of them beyond
reading the Result object engine.simulate() already returns.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Optional, Sequence

import pandas as pd

# The allocation table's columns, in the order every sheet and record uses
# them. Matches app.py's `alloc` DataFrame (COLS plus the Category it
# inserts), so a saved run can be replayed by eye against the app.
ALLOC_COLS = ["Category", "LTCMA line", "Weight %", "Return %", "Vol %"]

# engine.Result.rows carries one dict per simulated horizon, keyed by
# "_years" plus these columns. "_years" is renamed to "Years" on the way in
# — a leading underscore belongs in Python, not in a workbook a PM opens.
HORIZON_COLS = ["Years", "Horizon", "5th", "25th", "Median", "75th", "95th",
                "Mean", "P(loss)", "Median x", "5th x", "95th x"]

# Characters Excel refuses in a sheet name, and its length cap.
_INVALID_SHEET_CHARS = set("[]:*?/\\")
_MAX_SHEET_LEN = 31


@dataclass(frozen=True)
class RunRecord:
    """One saved simulation run: what was run, and what it produced.

    `seq` is the record's 1-based position in save order for this session —
    it is what makes every Excel sheet name unique, not a persistent ID.
    """
    seq: int
    label: str
    created_at: str              # ISO 8601, second precision
    mode: str                    # "Base" or "Custom"
    sleeves: list                # list[dict], ALLOC_COLS keys, one per sleeve
    paths: int
    years: int
    seed: int
    exp_return_pct: float        # arithmetic, per annum
    analytic_vol_pct: float
    sim_vol_pct: float
    median_wealth_x: float       # terminal (longest-horizon) median wealth
    p_loss_pct: float            # terminal P(loss), in percent
    elapsed_ms: float
    horizon_rows: list           # list[dict], HORIZON_COLS keys, one per horizon
    # Which market assumptions: "jpm" (J.P. Morgan 2026 LTCMA) or a BlackRock
    # id such as "blackrock-2026-08:Starting point". Defaults to J.P. Morgan so
    # records made before BlackRock was added read back unchanged.
    source: str = "jpm"


def run_digest(mode: str, sleeves: Sequence[dict], paths, years, seed,
               source: str = "jpm") -> str:
    """SHA-256 over everything that determines a run's numbers.

    Two runs with the same digest produce identical results, which is what
    lets auto-save skip a re-run of an already-saved scenario. Line order is
    kept (the Cholesky draw depends on it); floats are rounded to 6dp so a
    weight typed as 45 and one restored as 45.000000001 hash the same.
    """
    def _n(x):
        try:
            return round(float(x), 6)
        except (TypeError, ValueError):
            return str(x)
    body = {
        "mode": mode, "paths": int(paths), "years": int(years),
        "seed": int(seed),
        "sleeves": [[str(s["LTCMA line"]), _n(s["Weight %"]),
                     _n(s["Return %"]), _n(s["Vol %"])] for s in sleeves],
    }
    # Only non-J.P. Morgan runs carry the source, so every digest saved before
    # BlackRock was added still matches its own inputs.
    if source and source != "jpm":
        body["source"] = str(source)
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def auto_label(seq: int, mode: str, years: int, source: str = "jpm") -> str:
    """The default label offered in the Saved runs panel."""
    if source and source != "jpm":
        scen = source_label(source).partition(" · ")[2]
        src = f"BlackRock ({scen})" if scen else source_label(source)
        return f"Run {seq} · {src} · {mode} · {years}y"
    return f"Run {seq} · {mode} · {years}y"


def source_label(source: str) -> str:
    """'J.P. Morgan' or 'BlackRock · <scenario>' for tables and exports."""
    s = str(source or "jpm")
    if s == "jpm":
        return "J.P. Morgan"
    if s.startswith("blackrock"):
        scen = s.partition(":")[2]
        scen = {"Starting point": "Base case"}.get(scen, scen)
        return f"BlackRock · {scen}" if scen else "BlackRock"
    return s


def make_record(seq: int, label: str, mode: str, alloc: pd.DataFrame,
                 paths, years, seed, result, created_at: Optional[str] = None,
                 source: str = "jpm") -> RunRecord:
    """Build a RunRecord from an allocation table and an engine.Result.

    `alloc` is the app's sleeve DataFrame; `result` is the first element of
    what engine.simulate() returns. Every number here is read straight off
    `result` — nothing is recomputed, so a saved run can never disagree with
    what the app showed when it was saved.
    """
    term = result.rows[-1]
    sleeves = alloc[ALLOC_COLS].to_dict("records")
    horizon_rows = [
        {"Years": r["_years"], **{c: r[c] for c in HORIZON_COLS[1:]}}
        for r in result.rows
    ]
    clean_label = label.strip() or f"Run {seq}"
    return RunRecord(
        seq=int(seq),
        label=clean_label,
        created_at=created_at or datetime.now().isoformat(timespec="seconds"),
        mode=mode,
        sleeves=sleeves,
        paths=int(paths), years=int(years), seed=int(seed),
        exp_return_pct=float(result.exp_arith * 100),
        analytic_vol_pct=float(result.analytic_vol * 100),
        sim_vol_pct=float(result.sim_vol * 100),
        median_wealth_x=float(term["Median x"]),
        p_loss_pct=float(term["P(loss)"] * 100),
        elapsed_ms=float(result.elapsed_ms),
        horizon_rows=horizon_rows,
        source=str(source or "jpm"),
    )


def history_table(records: Sequence[RunRecord]) -> pd.DataFrame:
    """One row per saved run — what the Saved runs panel lists."""
    cols = ["#", "Label", "Saved", "Assumptions", "Mode", "Sleeves", "Paths",
            "Years",
            "Seed", "Return %", "Volatility %", "Median wealth", "P(loss) %"]
    if not records:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame([{
        "#": r.seq, "Label": r.label, "Saved": r.created_at,
        "Assumptions": source_label(r.source), "Mode": r.mode,
        "Sleeves": len(r.sleeves), "Paths": r.paths, "Years": r.years,
        "Seed": r.seed, "Return %": r.exp_return_pct,
        "Volatility %": r.analytic_vol_pct, "Median wealth": r.median_wealth_x,
        "P(loss) %": r.p_loss_pct,
    } for r in records])[cols]


def _sheet_name(prefix: str, seq: int, label: str) -> str:
    """An Excel-legal sheet name: <= 31 chars, none of []:*?/\\.

    Always starts with `prefix` + `seq`, which is what keeps every run's
    sheet unique regardless of what the label truncates to.
    """
    safe = "".join(c for c in label if c not in _INVALID_SHEET_CHARS).strip()
    name = f"{prefix}{seq} {safe}".strip() if safe else f"{prefix}{seq}"
    return name[:_MAX_SHEET_LEN] or f"{prefix}{seq}"


def to_excel_bytes(records: Sequence[RunRecord]) -> bytes:
    """A multi-sheet .xlsx: a Summary sheet, plus one sheet per saved run
    with that run's allocation table and its horizon-percentile table.

    Returns raw bytes, ready for st.download_button — nothing here touches
    disk or Streamlit. Exporting an empty history still returns an openable
    workbook (just the Summary sheet, with headers and no rows).
    """
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        history_table(records).to_excel(writer, sheet_name="Summary", index=False)
        for r in records:
            sheet = _sheet_name("Run ", r.seq, r.label)
            alloc_df = pd.DataFrame(r.sleeves, columns=ALLOC_COLS)
            alloc_df.to_excel(writer, sheet_name=sheet, index=False, startrow=0)
            gap = len(alloc_df) + 2          # one blank row between tables
            horizon_df = pd.DataFrame(r.horizon_rows, columns=HORIZON_COLS)
            horizon_df.to_excel(writer, sheet_name=sheet, index=False, startrow=gap)
    return buf.getvalue()
