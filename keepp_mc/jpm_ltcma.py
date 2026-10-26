"""
J.P. Morgan 2026 Long-Term Capital Market Assumptions — lines and correlations.

Kept separate from the UI and the engine so the figures can be audited and
swapped for next year's LTCMA without touching either.

COLUMN CHOICE — read this before editing.
The simulator takes a GEOMETRIC (compound) return: engine.py uses
    m_i = ln(1 + g_i) / 12
and derives the arithmetic return internally as g + sigma^2 / 2. So the
`compound` column below is what feeds the model. Feeding JPM's `arith` column
into it instead double-counts the variance drag and overstates the sleeve.
The original workbook did exactly that for Private Equity (11.90 rather than
10.20) and Commodities (6.15 rather than 4.60); both are corrected here.

`arith` and `compound25` are carried for reference and year-over-year context.
They are not used by the simulation.

CATEGORIES follow J.P. Morgan's own three: Fixed Income (21 lines),
Equities (18) and Alternatives (19). Each category's dropdown offers exactly
its own lines.

THE LINE IS THE SLEEVE'S IDENTITY — read this before adding a "manual" escape.
Every sleeve sits on a named LTCMA line, always. Picking a line fills Return
and Vol from the compound column; overwriting either number keeps the line and
keeps that override. The line is therefore two things at once: the sleeve's
own return and volatility (until the user overrides them) AND the row of the
published correlation matrix the sleeve uses. There is no separate "correlation
proxy" to choose, because a sleeve can no longer become anonymous.

Earlier builds had a "Manual entry" option that detached a row from every line
and so needed a second, hand-picked column to say where its correlations came
from. That column was the single input on the correlation path that was neither
published nor derived, and it asked the user a question they could not answer
from the numbers in front of them. Both are gone. Do not reintroduce a sleeve
that has no line: it has no correlation row, and nothing can honestly supply one.
"""

from pathlib import Path

import numpy as np
import pandas as pd

FI, EQ, ALT = "FI", "EQ", "ALT"

CATEGORY_NAME = {FI: "Fixed Income", EQ: "Equities", ALT: "Alternatives"}
CATEGORY_KEYS = [FI, EQ, ALT]

# (name, compound 2026, arithmetic 2026, annualized vol, compound 2025, category)
# Names must match jpm_correlations.csv exactly — a mismatch fails at import.
LINES = [
    # ---- Fixed income (21) ------------------------------------------------
    ("U.S. Cash",                             3.10,  3.10,  0.67, 3.10, FI),
    ("U.S. Short Duration Government/Credit", 4.00,  4.01,  1.63, 3.90, FI),
    ("U.S. Intermediate Treasuries",          4.00,  4.06,  3.48, 3.80, FI),
    ("U.S. Long Treasuries",                  4.90,  5.69, 13.02, 4.30, FI),
    ("TIPS",                                  4.30,  4.47,  5.88, 4.10, FI),
    ("U.S. Aggregate Bonds",                  4.80,  4.91,  4.76, 4.60, FI),
    ("U.S. Securitized",                      5.20,  5.28,  4.19, 4.90, FI),
    ("U.S. Long Duration Government/Credit",  5.20,  5.81, 11.45, 4.70, FI),
    ("U.S. Inv Grade Corporate Bonds",        5.20,  5.46,  7.39, 5.00, FI),
    ("U.S. Long Corporate Bonds",             5.40,  6.10, 12.28, 4.90, FI),
    ("U.S. High Yield Bonds",                 6.10,  6.46,  8.74, 6.10, FI),
    ("U.S. Leveraged Loans",                  6.60,  6.88,  7.70, 6.60, FI),
    ("World Government Bonds hedged",         4.30,  4.38,  4.02, 3.90, FI),
    ("World Government Bonds",                4.30,  4.55,  7.27, 4.20, FI),
    ("World ex-U.S. Government Bonds hedged", 4.20,  4.27,  3.95, 3.80, FI),
    ("World ex-U.S. Government Bonds",        4.30,  4.69,  9.06, 4.20, FI),
    ("Emerging Markets Sovereign Debt",       6.30,  6.66,  8.83, 5.80, FI),
    ("Emerging Markets Local Currency Debt",  6.70,  7.38, 12.12, 6.10, FI),
    ("Emerging Markets Corporate Bonds",      6.10,  6.37,  7.64, 6.20, FI),
    ("U.S. Muni 1-15 Yr Blend",               3.80,  3.88,  4.16, 3.60, FI),
    ("U.S. Muni High Yield",                  5.30,  5.66,  8.72, 4.70, FI),
    # ---- Equities (18) ----------------------------------------------------
    ("U.S. Large Cap",                        6.70,  7.94, 16.47, 6.70, EQ),
    ("U.S. Mid Cap",                          7.00,  8.55, 18.56, 7.00, EQ),
    ("U.S. Small Cap",                        6.90,  8.89, 21.10, 6.90, EQ),
    ("Euro Area Large Cap",                   7.80,  9.94, 22.04, 8.50, EQ),
    ("UK Large Cap",                          6.60,  7.99, 17.52, 7.80, EQ),
    ("Japanese Equity",                       8.80,  9.91, 15.77, 9.00, EQ),
    ("Hong Kong Equity",                      7.40,  9.43, 21.37, 7.40, EQ),
    ("EAFE Equity",                           7.50,  8.90, 17.63, 8.10, EQ),
    ("Chinese Domestic Equity",               7.70, 11.23, 28.70, 7.80, EQ),
    ("Emerging Markets Equity",               7.80,  9.74, 20.93, 7.20, EQ),
    ("AC Asia ex-Japan Equity",               7.90,  9.83, 20.84, 7.20, EQ),
    ("AC World Equity",                       7.00,  8.28, 16.78, 7.10, EQ),
    ("U.S. Equity Value Factor",              7.70,  9.11, 17.70, 7.70, EQ),
    ("U.S. Equity Momentum Factor",           7.60,  8.91, 17.06, 7.60, EQ),
    ("U.S. Equity Quality Factor",            6.60,  7.64, 15.11, 6.70, EQ),
    ("U.S. Equity Minimum Volatility Factor", 7.00,  7.79, 13.16, 7.00, EQ),
    ("U.S. Equity Dividend Yield Factor",     7.50,  8.71, 16.36, 7.70, EQ),
    ("Global Convertible Bonds hedged",       6.70,  7.28, 11.17, 6.70, EQ),
    # ---- Alternatives (19) ------------------------------------------------
    ("U.S. Core Real Estate",                 8.20,  8.79, 11.39, 8.10, ALT),
    ("U.S. Value-Added Real Estate",         10.10, 11.72, 19.23, 10.10, ALT),
    ("European Core Real Estate",             6.90,  7.77, 13.75, 7.60, ALT),
    ("Asia Pacific Core Real Estate",         8.40,  9.55, 15.98, 8.10, ALT),
    ("U.S. REITs",                            8.80, 10.15, 17.40, 8.00, ALT),
    ("Commercial Mortgage Loans",             6.20,  6.50,  7.94, 6.40, ALT),
    ("Global Core Infrastructure",            6.50,  6.99, 10.25, 6.30, ALT),
    ("Global Core Transport",                 7.90,  8.64, 12.77, 7.80, ALT),
    ("Global Timberland",                     6.30,  6.71,  9.38, 5.30, ALT),
    ("Commodities",                           4.60,  6.15, 18.32, 3.80, ALT),
    ("Gold",                                  5.50,  6.78, 16.68, 4.00, ALT),
    ("Private Equity",                       10.20, 11.91, 19.78, 9.90, ALT),
    ("Venture Capital",                       8.50, 10.66, 22.20, 8.80, ALT),
    ("Diversified Hedge Funds",               5.30,  5.46,  5.75, 4.90, ALT),
    ("Event Driven Hedge Funds",              5.20,  5.50,  8.01, 4.90, ALT),
    ("Long Bias Hedge Funds",                 5.50,  6.08, 11.12, 5.00, ALT),
    ("Relative Value Hedge Funds",            5.70,  5.84,  5.36, 5.00, ALT),
    ("Macro Hedge Funds",                     4.10,  4.33,  6.97, 3.80, ALT),
    ("Direct Lending",                        7.70,  8.59, 14.01, 8.20, ALT),
]

US_INFLATION = ("U.S. Inflation", 2.50, 2.52, 1.77, 2.40)
INFLATION_ROW = "U.S. Inflation"     # in the matrix, but not an investable sleeve

# name -> (compound, arith, vol, compound25, category)
BY_NAME = {n: (c, a, v, c25, f) for n, c, a, v, c25, f in LINES}

# The 14 lines flagged in the expansion proposal. Marked with a trailing star so
# a proposal line is identifiable at a glance; the star is a SUFFIX so it never
# interferes with type-to-filter on the name.
EXPANSION = {
    "U.S. Cash", "U.S. Aggregate Bonds", "U.S. Inv Grade Corporate Bonds",
    "U.S. High Yield Bonds", "Emerging Markets Sovereign Debt",
    "U.S. Large Cap", "EAFE Equity", "Emerging Markets Equity",
    "U.S. Core Real Estate", "Global Core Infrastructure", "Commodities",
    "Private Equity", "Venture Capital", "Direct Lending",
}
assert EXPANSION <= {n for n, *_ in LINES}, "expansion names must exist in LINES"

# Each category gets its own dropdown, so the option list is already scoped and
# a family prefix on the label would be redundant noise.
LABEL = {n: n + (" ★" if n in EXPANSION else "") for n, *_ in LINES}
NAME_OF = {v: k for k, v in LABEL.items()}

CHOICES = {k: [LABEL[n] for n, *_, c in LINES if c == k] for k in CATEGORY_KEYS}


def lookup(label_or_name):
    """(compound return, vol) for a line, or None if the name is unknown.

    Accepts the dropdown label ("U.S. Large Cap ★"), the label without its
    star, or an older family-prefixed spelling ("EQ - U.S. Large Cap").
    """
    name = resolve(label_or_name)
    row = BY_NAME.get(name)
    return (row[0], row[2]) if row else None


def resolve(label_or_name):
    """Bare LTCMA line name for any spelling we have ever used, else the input."""
    s = str(label_or_name).strip()
    if s in NAME_OF:
        return NAME_OF[s]
    s = s.removesuffix("★").strip()
    if s in NAME_OF:
        return NAME_OF[s]
    if s in BY_NAME:
        return s
    if " - " in s:                      # legacy "EQ - Name" label
        tail = s.split(" - ", 1)[1]
        if tail in BY_NAME:
            return tail
    return s


def category_of(label_or_name):
    """'FI' / 'EQ' / 'ALT' for a line, or None if the name is unknown."""
    row = BY_NAME.get(resolve(label_or_name))
    return row[4] if row else None


# --------------------------------------------------------------------------
# Correlations — J.P. Morgan 2026 LTCMA, USD, as published
# --------------------------------------------------------------------------
# The CSV is the source of truth and is committed unmodified so it can be
# diffed against the report. It arrives symmetric with a unit diagonal but is
# NOT positive semi-definite: its smallest eigenvalue is about -0.0161, which
# is normal for a published matrix rounded to two decimals, and is made worse
# by U.S. Core Real Estate and U.S. Value-Added Real Estate sharing one
# underlying series (identical rows -> a zero eigenvalue by construction).
#
# Cholesky needs a positive definite matrix, so the matrix is repaired ONCE at
# import with Higham's nearest-correlation-matrix algorithm. The repair is
# cosmetic: the largest single cell moves 0.0066 and no pair moves by as much
# as 0.01. Repairing the whole matrix once — rather than each sub-matrix at
# run time — also means every sub-matrix the app can build is automatically
# valid, because every principal sub-matrix of a PSD matrix is itself PSD.
_CSV = Path(__file__).parent / "data" / "jpm_correlations.csv"


def _higham(A, iters=200, tol=1e-12):
    """Nearest correlation matrix by alternating projections (Higham 2002)."""
    Y = np.array(A, dtype=float)
    dS = np.zeros_like(Y)
    for _ in range(iters):
        R = Y - dS
        w, V = np.linalg.eigh((R + R.T) / 2.0)
        X = (V * np.clip(w, 0.0, None)) @ V.T        # project onto PSD
        dS = X - R
        Y = X.copy()
        np.fill_diagonal(Y, 1.0)                     # project onto unit diagonal
        if np.linalg.eigvalsh(Y).min() > tol:
            break
    Y = (Y + Y.T) / 2.0
    np.fill_diagonal(Y, 1.0)
    return Y


def _load_corr():
    df = pd.read_csv(_CSV, index_col=0)
    df.index = [str(i).strip() for i in df.index]
    df.columns = [str(c).strip() for c in df.columns]
    if list(df.index) != list(df.columns):
        raise ValueError("jpm_correlations.csv: row and column order differ")
    missing = [n for n, *_ in LINES if n not in df.index]
    if missing:
        raise ValueError(f"jpm_correlations.csv is missing lines: {missing}")
    raw = df.to_numpy(dtype=float)
    repaired = _higham(raw)
    return (list(df.index), raw, repaired,
            float(np.abs(repaired - raw).max()),
            float(np.linalg.eigvalsh(raw).min()))


CORR_NAMES, CORR_RAW, CORR, CORR_MAX_REPAIR, CORR_RAW_MIN_EIG = _load_corr()
CORR_INDEX = {n: i for i, n in enumerate(CORR_NAMES)}


def corr_for(lines):
    """Sub-matrix of the published correlations for a list of LTCMA line names.

    One line name per sleeve. Every sleeve sits on a line, so every sleeve has
    a published correlation row and nothing here is assumed or borrowed.
    """
    idx = []
    for p in lines:
        name = resolve(p)
        if name not in CORR_INDEX:
            raise KeyError(f"no correlation row for {p!r}")
        idx.append(CORR_INDEX[name])
    return CORR[np.ix_(idx, idx)]


# The line a newly added row starts on, per category. Deliberately the broadest
# line in each: a new sleeve should begin somewhere unremarkable that the user
# will immediately change, not on a niche line they might leave by accident.
DEFAULT_LINE = {FI: "U.S. Aggregate Bonds",
                EQ: "AC World Equity",
                ALT: "Diversified Hedge Funds"}

# The line the optimizer's cash floor and risk-free default use.
CASH_LINE = "U.S. Cash"

def sync_group(prev, edited, category_key):
    """Keep one category's sleeves internally consistent.

    Each category is rendered as its own table whose LTCMA dropdown holds only
    that category's lines, so there is no cross-category pick to guard against.

    Rules:
      1. Every row must sit on a line in THIS category. A row the editor just
         added arrives with NaNs, and stale session state can carry a line from
         an older catalogue or the retired "Manual entry"; either way the row
         falls back to the category's default line and is refilled from it.
      2. Moving a row to a different line refills Return and Vol from the
         compound column, discarding any previous override — picking a line is
         how you ask for J.P. Morgan's numbers.
      3. Typing over Return or Vol does NOT disturb the line. The sleeve keeps
         your numbers and stays on its line, which is what supplies its row of
         the correlation matrix.

    Rule 2 compares positionally against `prev`, which is only sound while the
    rows still line up. Deleting a row shifts every row below it up by one, so
    on a shrinking frame rule 2 is skipped entirely — otherwise a delete would
    silently reset the edits of every row underneath it.

    Returns (frame, dirty).
    """
    allowed = set(CHOICES[category_key])
    default_line = LABEL[DEFAULT_LINE[category_key]]

    out = edited.copy().reset_index(drop=True)
    deleted = len(out) < len(prev)
    dirty = False

    for i in range(len(out)):
        line = out.at[i, "LTCMA line"]
        forced = False
        if not isinstance(line, str) or line not in allowed:
            out.at[i, "LTCMA line"] = line = default_line       # rule 1
            forced = dirty = True

        was = prev.at[i, "LTCMA line"] if i < len(prev) else None
        if forced or (not deleted and line != was):             # rule 2
            out.at[i, "Return %"], out.at[i, "Vol %"] = lookup(line)
            dirty = True

        for col in ("Weight %", "Return %", "Vol %"):
            try:
                v = float(out.at[i, col])
            except (TypeError, ValueError):
                v = float("nan")
            if v != v:                                          # NaN
                out.at[i, col] = 0.0
                dirty = True
            else:
                out.at[i, col] = v

    return out, dirty
