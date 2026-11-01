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
from dataclasses import dataclass, replace
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


def renumber(records: Sequence[RunRecord]) -> list[RunRecord]:
    """Number `records` (oldest first) 1..N for display.

    The database never reuses a sequence number, so after deletions the list
    would read 3, 7, 12, ... This gives the runs still in the list a gapless
    1..N instead. An untouched auto label ("Run 7 · ...") follows its new
    number; a label the user typed is kept as it is.
    """
    out = []
    for i, r in enumerate(records, 1):
        label = r.label
        if label in (auto_label(r.seq, r.mode, r.years, r.source),
                     f"Run {r.seq}"):
            label = auto_label(i, r.mode, r.years, r.source)
        out.append(replace(r, seq=i, label=label))
    return out


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


# Number formats for a caller-supplied Summary, by header. Values stay
# numbers in Excel (sortable, usable in formulas); only the display changes.
_SUMMARY_FORMATS = {"Avg return %": "0.00", "Volatility %": "0.00",
                    "Chance of loss %": "0.00", "Typical ×": '0.000"×"',
                    "Futures": "#,##0", "Seed": "0"}


def _style_summary(ws, df: pd.DataFrame) -> None:
    from openpyxl.styles import Font
    for c in ws[1]:
        c.font = Font(bold=True)
    ws.freeze_panes = "A2"
    for j, col in enumerate(df.columns, start=1):
        letter = ws.cell(row=1, column=j).column_letter
        fmt = _SUMMARY_FORMATS.get(col)
        if fmt:
            for i in range(2, len(df) + 2):
                ws.cell(row=i, column=j).number_format = fmt
        width = max([len(str(col))] + [len(str(v)) for v in df[col]])
        ws.column_dimensions[letter].width = min(max(8, width + 2), 48)


def _write_grouped(ws, df: pd.DataFrame) -> None:
    """Write a frame with two-level columns (group, sub) the way the page
    shows it: a merged group header over Keepp / Provider / Difference, and
    single columns (sub == "") merged down both header rows."""
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter
    bold, center = Font(bold=True), Alignment(horizontal="center",
                                              vertical="center",
                                              wrap_text=True)
    cols = list(df.columns)
    j = 0
    while j < len(cols):
        g, sub = cols[j]
        if sub == "":
            ws.cell(row=1, column=j + 1, value=g)
            ws.merge_cells(start_row=1, start_column=j + 1, end_row=2,
                           end_column=j + 1)
            j += 1
            continue
        k = j
        while k + 1 < len(cols) and cols[k + 1][0] == g:
            k += 1
        ws.cell(row=1, column=j + 1, value=g)
        if k > j:
            ws.merge_cells(start_row=1, start_column=j + 1, end_row=1,
                           end_column=k + 1)
        for c in range(j, k + 1):
            ws.cell(row=2, column=c + 1, value=cols[c][1])
        j = k + 1
    for row in (1, 2):
        for c in range(1, len(cols) + 1):
            ws.cell(row=row, column=c).font = bold
            ws.cell(row=row, column=c).alignment = center
    for i, rec in enumerate(df.itertuples(index=False), start=3):
        for c, v in enumerate(rec, start=1):
            cell = ws.cell(row=i, column=c,
                           value=None if pd.isna(v) else v)
            g, sub = cols[c - 1]
            if isinstance(v, float):
                diff = sub.startswith("Difference")
                if "USD" in g:
                    cell.number_format = ('+"$"#,##0;-"$"#,##0;"$"0' if diff
                                          else '"$"#,##0')
                else:
                    cell.number_format = "+0.00;-0.00;0.00" if diff else "0.00"
    ws.freeze_panes = "B3"
    for c, (g, sub) in enumerate(cols, start=1):
        width = max([len(str(sub)), 0 if sub else len(str(g))]
                    + [len(str(v)) for v in df.iloc[:, c - 1].astype(str)])
        ws.column_dimensions[get_column_letter(c)].width = min(
            max(11, width + 2), 44)
    ws.row_dimensions[1].height = 32


def to_excel_bytes(records: Sequence[RunRecord],
                   summary: Optional[pd.DataFrame] = None,
                   scenarios: Optional[pd.DataFrame] = None) -> bytes:
    """A multi-sheet .xlsx: a Summary sheet, plus one sheet per saved run
    with that run's allocation table and its horizon-percentile table.

    `summary`, when given, is written as the Summary sheet exactly as passed.
    The app passes its Keepp vs provider returns table, with two-level
    columns (metric, Keepp / Provider / Difference) written as a grouped
    header. A frame with ordinary columns is written as a plain table.
    Without it, history_table(records) is used.

    `scenarios`, when given, follows as a "Scenarios" sheet — the app passes
    the Saved scenarios list as it is on screen. Run sheets follow, in the
    order of `records`.

    Returns raw bytes, ready for st.download_button — nothing here touches
    disk or Streamlit. Exporting an empty history still returns an openable
    workbook (just the Summary sheet, with headers and no rows).
    """
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        if summary is not None and isinstance(summary.columns,
                                              pd.MultiIndex):
            ws = writer.book.create_sheet("Summary")
            _write_grouped(ws, summary)
        elif summary is not None:
            summary.to_excel(writer, sheet_name="Summary", index=False)
            _style_summary(writer.sheets["Summary"], summary)
        else:
            history_table(records).to_excel(writer, sheet_name="Summary",
                                            index=False)
        if scenarios is not None:
            scenarios.to_excel(writer, sheet_name="Scenarios", index=False)
            _style_summary(writer.sheets["Scenarios"], scenarios)
        for r in records:
            sheet = _sheet_name("Run ", r.seq, r.label)
            alloc_df = pd.DataFrame(r.sleeves, columns=ALLOC_COLS)
            alloc_df.to_excel(writer, sheet_name=sheet, index=False, startrow=0)
            gap = len(alloc_df) + 2          # one blank row between tables
            horizon_df = pd.DataFrame(r.horizon_rows, columns=HORIZON_COLS)
            horizon_df.to_excel(writer, sheet_name=sheet, index=False, startrow=gap)
    return buf.getvalue()
