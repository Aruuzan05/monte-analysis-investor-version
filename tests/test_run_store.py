"""Checks for the Saved run files feature (run_store.py).

Run after any change to run_store.py:

    python tests/test_run_store.py

Covers three things: a RunRecord is built correctly from an allocation table
and an engine.Result, the session-history table lists saved runs faithfully,
and the Excel export produces a workbook that actually opens and matches
what was saved.
"""
from io import BytesIO

import openpyxl
import pandas as pd

import sys
from pathlib import Path

# Run as a script (python tests/test_x.py): put the repo root on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from keepp_mc import run_store as rs
from keepp_mc.engine import simulate

ok = True


def check(label, cond):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")


# A small, known allocation — same six sleeves as the app's Base case.
ALLOC = pd.DataFrame([
    {"Category": "Fixed Income", "LTCMA line": "U.S. Inv Grade Corporate Bonds",
     "Weight %": 45.0, "Return %": 5.00, "Vol %": 7.02},
    {"Category": "Fixed Income", "LTCMA line": "U.S. Cash",
     "Weight %": 17.0, "Return %": 3.80, "Vol %": 0.67},
    {"Category": "Equities", "LTCMA line": "AC World Equity",
     "Weight %": 20.0, "Return %": 12.88, "Vol %": 14.47},
    {"Category": "Alternatives", "LTCMA line": "Private Equity",
     "Weight %": 10.0, "Return %": 10.20, "Vol %": 19.78},
    {"Category": "Alternatives", "LTCMA line": "Diversified Hedge Funds",
     "Weight %": 0.0, "Return %": 5.30, "Vol %": 5.75},
    {"Category": "Alternatives", "LTCMA line": "Commodities",
     "Weight %": 8.0, "Return %": 4.60, "Vol %": 18.32},
])
CORR = [[1.00, 0.00, 0.10, 0.05, 0.00, 0.00],
        [0.00, 1.00, 0.00, 0.00, 0.00, 0.00],
        [0.10, 0.00, 1.00, 0.68, 0.00, 0.30],
        [0.05, 0.00, 0.68, 1.00, 0.00, 0.25],
        [0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
        [0.00, 0.00, 0.30, 0.25, 0.00, 1.00]]

res, _ = simulate(list(ALLOC["Weight %"]), list(ALLOC["Return %"]),
                  list(ALLOC["Vol %"]), CORR, 2000, 10, 20260722)

# ==========================================================================
print("=" * 78)
print("1. RECORD — building a RunRecord from an alloc table and a Result")
print("=" * 78)
rec = rs.make_record(1, "  My Base Run  ", "Base", ALLOC, 2000, 10, 20260722,
                     res, created_at="2026-07-22T10:00:00")
check("label is trimmed", rec.label == "My Base Run")
check("seq carries through", rec.seq == 1)
check("mode carries through", rec.mode == "Base")
check("paths/years/seed carry through as ints",
      (rec.paths, rec.years, rec.seed) == (2000, 10, 20260722))
check("sleeves captured with all five columns, in order",
      list(rec.sleeves[0].keys()) == rs.ALLOC_COLS)
check("sleeve count matches the allocation", len(rec.sleeves) == len(ALLOC))
check("a sleeve's own numbers are carried faithfully",
      rec.sleeves[2]["LTCMA line"] == "AC World Equity"
      and rec.sleeves[2]["Weight %"] == 20.0)
check("expected return matches the engine's own figure",
      abs(rec.exp_return_pct - res.exp_arith * 100) < 1e-9)
check("analytic vol matches the engine's own figure",
      abs(rec.analytic_vol_pct - res.analytic_vol * 100) < 1e-9)
check("median wealth is the terminal (longest-horizon) figure",
      abs(rec.median_wealth_x - res.rows[-1]["Median x"]) < 1e-9)
check("P(loss) is the terminal figure, in percent",
      abs(rec.p_loss_pct - res.rows[-1]["P(loss)"] * 100) < 1e-9)
check("one horizon row per simulated horizon",
      len(rec.horizon_rows) == len(res.rows))
check("horizon rows use 'Years', not the internal '_years' key",
      "Years" in rec.horizon_rows[0] and "_years" not in rec.horizon_rows[0])
check("horizon rows carry every HORIZON_COLS field",
      list(rec.horizon_rows[0].keys()) == rs.HORIZON_COLS)

blank = rs.make_record(2, "   ", "Custom", ALLOC, 500, 5, 1, res)
check("a blank label falls back to 'Run <seq>'", blank.label == "Run 2")
check("auto_label reads naturally",
      rs.auto_label(3, "Base", 10) == "Run 3 · Base · 10y")

# ==========================================================================
print()
print("=" * 78)
print("2. HISTORY TABLE — the Saved runs panel's list")
print("=" * 78)
empty = rs.history_table([])
check("an empty history is an empty DataFrame, not an error",
      len(empty) == 0 and list(empty.columns)[:2] == ["#", "Label"])

hist = rs.history_table([rec, blank])
check("one row per record, in the order given",
      len(hist) == 2 and list(hist["#"]) == [1, 2])
check("Label and Mode show up as saved",
      list(hist["Label"]) == ["My Base Run", "Run 2"]
      and list(hist["Mode"]) == ["Base", "Custom"])
check("Sleeves counts the allocation size", hist.at[0, "Sleeves"] == len(ALLOC))
check("Return % matches the record", abs(hist.at[0, "Return %"] - rec.exp_return_pct) < 1e-9)

# ==========================================================================
print()
print("=" * 78)
print("3. EXCEL EXPORT — a workbook a PM can actually open")
print("=" * 78)
xbytes = rs.to_excel_bytes([rec, blank])
check("produces non-trivial bytes", isinstance(xbytes, bytes) and len(xbytes) > 1000)

wb = openpyxl.load_workbook(BytesIO(xbytes))
check("has a Summary sheet plus one sheet per run",
      wb.sheetnames[0] == "Summary" and len(wb.sheetnames) == 3)
check("every run's sheet name carries its own seq number",
      wb.sheetnames[1].startswith("Run 1") and wb.sheetnames[2].startswith("Run 2"))

summary_ws = wb["Summary"]
header = [c.value for c in next(summary_ws.iter_rows(max_row=1))]
check("Summary sheet header matches history_table's columns",
      header == list(hist.columns))
check("Summary sheet has one data row per run",
      summary_ws.max_row == len(hist) + 1)

run_ws = wb[wb.sheetnames[1]]
# The sheet's overall width is set by the wider horizon table below, so
# iter_rows pads the narrower allocation header with trailing blanks —
# bound max_col to the allocation table's own width to check it in isolation.
n_alloc_cols = len(rs.ALLOC_COLS)
alloc_header = [c.value for c in
               next(run_ws.iter_rows(max_row=1, max_col=n_alloc_cols))]
check("run sheet starts with the allocation table's own columns",
      alloc_header == rs.ALLOC_COLS)
alloc_row3 = [c.value for c in
             next(run_ws.iter_rows(min_row=3, max_row=3, max_col=n_alloc_cols))]
check("run sheet's allocation rows carry the saved sleeve data",
      alloc_row3[1] == "U.S. Cash" and alloc_row3[2] == 17.0)

# the horizon table sits after a one-row gap below the allocation rows
gap_row = len(rec.sleeves) + 2
horizon_header_row = list(run_ws.iter_rows(min_row=gap_row + 1, max_row=gap_row + 1))[0]
horizon_header = [c.value for c in horizon_header_row]
check("horizon table sits below the allocation table on the same sheet",
      horizon_header == rs.HORIZON_COLS)

empty_xbytes = rs.to_excel_bytes([])
wb2 = openpyxl.load_workbook(BytesIO(empty_xbytes))
check("exporting an empty history still produces an openable workbook "
      "with just a Summary sheet", wb2.sheetnames == ["Summary"])
check("the empty Summary sheet still has a header row",
      [c.value for c in next(wb2["Summary"].iter_rows(max_row=1))] == list(hist.columns))

# ==========================================================================
print()
print("=" * 78)
print("4. SHEET NAMES — Excel's own constraints")
print("=" * 78)
weird = rs.make_record(3, "Q3 [Stress] Test / v2?", "Custom", ALLOC, 100, 3, 1, res)
name = rs._sheet_name("Run ", weird.seq, weird.label)
check("invalid Excel sheet characters are stripped",
      not (set(name) & rs._INVALID_SHEET_CHARS))
check("sheet name never exceeds Excel's 31-character limit", len(name) <= 31)
check("sheet name still starts with the run's own seq", name.startswith("Run 3"))

long_label = "A" * 60
weird2 = rs.make_record(4, long_label, "Custom", ALLOC, 100, 3, 1, res)
name2 = rs._sheet_name("Run ", weird2.seq, weird2.label)
check("a very long label is truncated to fit", len(name2) <= 31)

no_label = rs._sheet_name("Run ", 5, "")
check("an empty label still yields a valid, unique sheet name",
      no_label == "Run 5")

# a workbook with several weird labels must still open without a name clash
wb3 = openpyxl.load_workbook(BytesIO(rs.to_excel_bytes([rec, blank, weird, weird2])))
check("a full export with weird labels has one uniquely-named sheet per run",
      len(set(wb3.sheetnames)) == len(wb3.sheetnames) == 5)

print()
print("RESULT:", "all checks pass" if ok else "FAILURES ABOVE — investigate")
raise SystemExit(0 if ok else 1)
