"""
J.P. Morgan 2027 Long-Term Capital Market Assumptions — lines and correlations.

31st edition, USD, estimates as of 30 September 2026. Kept separate from the UI
and the engine so the figures can be audited and swapped for next year's LTCMA
without touching either. The 2026 (30th edition) set it replaced is not retained
anywhere in the app: there is one J.P. Morgan vintage, and this is it.

COLUMN CHOICE — read this before editing.
The simulator takes a GEOMETRIC (compound) return: engine.py uses
    m_i = ln(1 + g_i) / 12
and derives the arithmetic return internally as g + sigma^2 / 2. So the
`compound` column below is what feeds the model. Feeding JPM's `arith` column
into it instead double-counts the variance drag and overstates the sleeve.
The original workbook did exactly that for Private Equity and Commodities;
both take the compound figure here.

`arith` and `compound26` are carried for reference and year-over-year context.
They are not used by the simulation. Note that JPM's published `arith` sits up
to 0.55pp BELOW compound + sigma^2/2 (worst: Chinese Domestic Equity) because
they round compound to 0.1pp before publishing. That is a rounding artefact in
their table, not a different convention, and it is another reason `arith` is
reference only — opt.inputs_from_percent() derives mu from the compound column.

`compound26` is J.P. Morgan's own restatement of last year's figure in the 2027
table, not our 2026 extraction. On all 58 lines the two agree exactly, which is
how this edition was validated on the way in.

CATEGORIES follow J.P. Morgan's own three: Fixed Income (21 lines),
Equities (18) and Alternatives (20). Each category's dropdown offers exactly
its own lines.

LINE-UP CHANGES FROM 2026 — see EXCLUDED_DUPLICATES below for the third.
  - DROPPED by JPM: U.S. Value-Added Real Estate. Not held; no replacement.
  - NEW: Global Core Real Estate (7.90 / 14.50), placed in the slot
    Value-Added vacated so the Alternatives dropdown order barely moves.

DIGITAL ASSETS is the one line here that is NOT a row of the published tables.
See HOUSE_LINES below for its figures, its borrowed correlation row, and the
reconciliation problem in J.P. Morgan's own bitcoin numbers. It is marked with
a trailing dagger in the dropdown so it never passes for a published line.

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
from the numbers in front of them. Both are gone, and must not come back.

WHAT IS FORBIDDEN IS THE USER-CHOSEN PROXY, NOT EVERY PROXY. This docstring
used to end "nothing can honestly supply one", which the code now contradicts:
HOUSE_LINES below gives Digital Assets the Gold row, and personal_lines.py has
long done the same for Keepp's two private-markets lines. The distinction is
what matters. A proxy that is fixed in the module, named in the UI and argued
for in writing is auditable: there is one answer, it is the same for every user
and every run, and a reader can disagree with it in a meeting. A proxy the user
picks from a dropdown is a free parameter on the correlation path that nobody
can justify from the numbers on screen. The first is allowed, sparingly and
with the figures stated. The second stays gone.
"""

from pathlib import Path

import numpy as np
import pandas as pd

FI, EQ, ALT = "FI", "EQ", "ALT"

# Prior-year column for a line J.P. Morgan did not publish last year. Nothing
# reads that column; it is carried for year-over-year context only.
NA = float("nan")

CATEGORY_NAME = {FI: "Fixed Income", EQ: "Equities", ALT: "Alternatives"}
CATEGORY_KEYS = [FI, EQ, ALT]

# (name, compound 2027, arithmetic 2027, annualized vol, compound 2026, category)
# Names must match jpm_correlations.csv exactly — a mismatch fails at import.
LINES = [
    # ---- Fixed income (21) ----------------------------------------------
    ("U.S. Cash",                              3.20,  3.20,  0.68,  3.10, FI),
    ("U.S. Short Duration Government/Credit",  4.40,  4.41,  1.60,  4.00, FI),
    ("U.S. Intermediate Treasuries",           4.60,  4.66,  3.44,  4.00, FI),
    ("U.S. Long Treasuries",                   5.80,  6.57, 12.87,  4.90, FI),
    ("TIPS",                                   5.10,  5.26,  5.80,  4.30, FI),
    ("U.S. Aggregate Bonds",                   5.60,  5.70,  4.68,  4.80, FI),
    ("U.S. Securitized",                       6.00,  6.08,  4.16,  5.20, FI),
    ("U.S. Long Duration Government/Credit",   6.50,  7.08, 11.22,  5.20, FI),
    ("U.S. Inv Grade Corporate Bonds",         6.10,  6.34,  7.16,  5.20, FI),
    ("U.S. Long Corporate Bonds",              6.60,  7.26, 11.97,  5.40, FI),
    ("U.S. High Yield Bonds",                  6.60,  6.94,  8.54,  6.10, FI),
    ("U.S. Leveraged Loans",                   6.60,  6.93,  8.46,  6.60, FI),
    ("World Government Bonds hedged",          5.00,  5.08,  3.99,  4.30, FI),
    ("World Government Bonds",                 5.10,  5.35,  7.22,  4.30, FI),
    ("World ex-U.S. Government Bonds hedged",  4.80,  4.87,  3.93,  4.20, FI),
    ("World ex-U.S. Government Bonds",         5.10,  5.48,  9.03,  4.30, FI),
    ("Emerging Markets Sovereign Debt",        6.50,  6.86,  8.77,  6.30, FI),
    ("Emerging Markets Local Currency Debt",   7.30,  7.96, 11.99,  6.70, FI),
    ("Emerging Markets Corporate Bonds",       6.30,  6.56,  7.51,  6.10, FI),
    ("U.S. Muni 1-15 Yr Blend",                4.40,  4.48,  4.17,  3.80, FI),
    ("U.S. Muni High Yield",                   5.70,  6.05,  8.63,  5.30, FI),
    # ---- Equities (18) --------------------------------------------------
    ("U.S. Large Cap",                         7.50,  8.95, 17.96,  6.70, EQ),
    ("U.S. Mid Cap",                           7.70,  9.34, 19.16,  7.00, EQ),
    ("U.S. Small Cap",                         7.70,  9.66, 21.03,  6.90, EQ),
    ("Euro Area Large Cap",                    8.10, 10.36, 22.69,  7.80, EQ),
    ("UK Large Cap",                           6.40,  7.90, 18.17,  6.60, EQ),
    ("Japanese Equity",                        9.10, 10.37, 16.89,  8.80, EQ),
    ("Hong Kong Equity",                       6.90,  8.92, 21.27,  7.40, EQ),
    ("EAFE Equity",                            7.70,  9.21, 18.34,  7.50, EQ),
    ("Chinese Domestic Equity",                7.40, 10.84, 28.26,  7.70, EQ),
    ("Emerging Markets Equity",                7.10,  9.13, 21.36,  7.80, EQ),
    ("AC Asia ex-Japan Equity",                7.10,  9.14, 21.42,  7.90, EQ),
    ("AC World Equity",                        7.50,  8.94, 17.88,  7.00, EQ),
    ("U.S. Equity Value Factor",               8.50, 10.13, 19.16,  7.70, EQ),
    ("U.S. Equity Momentum Factor",            8.50, 10.06, 18.74,  7.60, EQ),
    ("U.S. Equity Quality Factor",             7.50,  8.74, 16.59,  6.60, EQ),
    ("U.S. Equity Minimum Volatility Factor",  7.60,  8.52, 14.25,  7.00, EQ),
    ("U.S. Equity Dividend Yield Factor",      8.30,  9.68, 17.57,  7.50, EQ),
    ("Global Convertible Bonds hedged",        7.50,  8.11, 11.53,  6.70, EQ),
    # ---- Alternatives (19) ----------------------------------------------
    ("U.S. Core Real Estate",                  8.10,  8.69, 11.40,  8.20, ALT),
    ("Global Core Real Estate",                7.90,  8.85, 14.50,  7.90, ALT),
    ("European Core Real Estate",              7.10,  8.39, 16.89,  6.90, ALT),
    ("Asia Pacific Core Real Estate",          8.70,  9.88, 16.22,  8.40, ALT),
    ("U.S. REITs",                             8.20,  9.55, 17.33,  8.80, ALT),
    ("Commercial Mortgage Loans",              6.30,  6.59,  7.91,  6.20, ALT),
    ("Global Core Infrastructure",             7.10,  7.59, 10.29,  6.50, ALT),
    ("Global Core Transport",                  8.10,  8.82, 12.57,  7.90, ALT),
    ("Global Timberland",                      6.20,  6.61,  9.40,  6.30, ALT),
    ("Commodities",                            4.80,  6.29, 17.97,  4.60, ALT),
    ("Gold",                                   5.50,  6.86, 17.24,  5.50, ALT),
    ("Digital Assets",                        10.00, 20.90, 46.70,   NA, ALT),
    ("Private Equity",                         9.80, 11.60, 20.31, 10.20, ALT),
    ("Venture Capital",                        8.80, 10.96, 22.20,  8.50, ALT),
    ("Diversified Hedge Funds",                6.70,  6.86,  5.94,  5.30, ALT),
    ("Event Driven Hedge Funds",               7.10,  7.44,  8.56,  5.20, ALT),
    ("Long Bias Hedge Funds",                  7.30,  7.89, 11.29,  5.50, ALT),
    ("Relative Value Hedge Funds",             6.10,  6.23,  5.17,  5.70, ALT),
    ("Macro Hedge Funds",                      5.60,  5.86,  7.39,  4.10, ALT),
    ("Direct Lending",                         7.80,  8.69, 13.98,  7.70, ALT),
]

US_INFLATION = ("U.S. Inflation", 2.40, 2.42, 1.87, 2.50)
INFLATION_ROW = "U.S. Inflation"     # in the matrix, but not an investable sleeve

# Lines J.P. Morgan publishes that are deliberately NOT offered as sleeves.
#
# Americas Core Real Estate is new in the 2027 edition and is identical to
# U.S. Core Real Estate in every published figure — same 8.10 compound, same
# 8.69 arithmetic, same 11.40 vol — with rho = 1.00 between the two. It is a
# relabelling of one underlying series, not a second asset. Offering both would
# let a portfolio hold the same real estate twice and understate its volatility,
# because a pair at rho = 1.00 contributes no diversification at all while
# looking like two sleeves.
#
# The row stays in the correlation CSV, which is shipped exactly as published.
# It is simply never selectable, so no allocation can reference it.
EXCLUDED_DUPLICATES = {"Americas Core Real Estate": "U.S. Core Real Estate"}
assert not (set(EXCLUDED_DUPLICATES) & {n for n, *_ in LINES}), \
    "an excluded duplicate must not also be an investable line"

# --------------------------------------------------------------------------
# House lines — in the catalogue, NOT in the published tables
# --------------------------------------------------------------------------
# {line: the published line whose correlation row it borrows}
#
# DIGITAL ASSETS. The 2027 edition puts a figure on bitcoin for the first time,
# and the figures do not form a usable set:
#
#     compound 10.0%     arithmetic 20.9%     volatility 55%
#     "we do not yet include bitcoin in our main covariance matrices"
#
# Two problems. First, those three numbers do not reconcile under this app's
# convention: 10.0 + 55^2/2 = 25.12%, against their own published 20.9%, off by
# 4.23pp. The worst equivalent gap across the 58 published lines is 0.55pp
# (Chinese Domestic Equity), so this is a difference of convention, not
# rounding. We therefore take the volatility IMPLIED by their own two return
# figures, sqrt(2 * (20.9 - 10.0) / 100) = 46.70%, and keep their 10.0%
# compound. The resulting trio (10.00, 20.90, 46.70) is self-consistent to
# 0.004pp, so the app reproduces J.P. Morgan's published arithmetic figure
# instead of contradicting it. Their 55% is not used; it is stated in the
# app's provenance panel so the choice is visible.
#
# Second, there is no correlation row to have, because J.P. Morgan excluded
# bitcoin from the matrices (they published a separate, smaller one against a
# subset of assets, which we do not hold). The line borrows GOLD's row: the
# closest published no-cashflow store of value, and the choice that leaves the
# matrix positive definite. Commodities as the proxy makes the engine's repair
# fire on any run holding both.
#
# These figures are from press coverage of the 2027 report, NOT read off the
# PDF. Confirm them against the source before this line informs an allocation.
HOUSE_LINES = {"Digital Assets": "Gold"}

# The pair (house line, the line it borrows from) when BOTH are held. A copied
# row would put exactly 1.0 there — two assets that always move identically —
# which makes the matrix singular and forces the engine to repair it. Same
# constant and same reason as personal_lines.SELF_CORR.
SELF_CORR = 0.99

assert set(HOUSE_LINES) <= {n for n, *_ in LINES}, \
    "a house line must be in LINES"
assert not (set(HOUSE_LINES) & set(EXCLUDED_DUPLICATES)), \
    "a house line cannot also be an excluded duplicate"

# name -> (compound, arith, vol, compound26, category)
BY_NAME = {n: (c, a, v, c26, f) for n, c, a, v, c26, f in LINES}

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
# ★ marks an expansion-proposal line, † a house line (not published by
# J.P. Morgan). Both are SUFFIXES so neither interferes with type-to-filter.
LABEL = {n: n + (" ★" if n in EXPANSION else "")
            + (" †" if n in HOUSE_LINES else "") for n, *_ in LINES}
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
    s = s.removesuffix("†").strip().removesuffix("★").strip()
    if s in NAME_OF:
        return NAME_OF[s]
    if s in BY_NAME:
        return s
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
# Correlations — J.P. Morgan 2027 LTCMA, USD, as published
# --------------------------------------------------------------------------
# 60 x 60: the 58 investable lines, the excluded Americas Core Real Estate row,
# and U.S. Inflation. The CSV is the source of truth and is committed
# unmodified so it can be diffed against the report. It arrives symmetric with
# a unit diagonal but is NOT positive semi-definite: its smallest eigenvalue is
# about -0.0169 and seven eigenvalues are negative. That is normal for a
# published matrix rounded to two decimals, and is made worse by Americas Core
# Real Estate and U.S. Core Real Estate sharing one underlying series at
# rho = 1.00. Note that their correlation ROWS are not identical — they differ
# by up to 0.02 — so the duplication shows up as several small negative
# eigenvalues rather than one clean zero.
#
# Cholesky needs a positive definite matrix, so the matrix is repaired ONCE at
# import with Higham's nearest-correlation-matrix algorithm. The repair is
# cosmetic: the largest single cell moves 0.0047 (U.S. Large Cap <-> AC World
# Equity, 0.97 -> 0.9653) and no pair moves by as much as 0.01 — slightly less
# movement than the 2026 matrix needed. Repairing the whole matrix once —
# rather than each sub-matrix at run time — also means every sub-matrix the app
# can build is automatically valid, because every principal sub-matrix of a PSD
# matrix is itself PSD.
#
# The repair reaches PSD, not positive definite: the full matrix still has a
# zero eigenvalue and np.linalg.cholesky still refuses it. Only Standard mode
# with every line selected hits that case, where engine.cholesky_safe clips to
# a positive floor and flags the run. Unchanged from 2026, which behaved the
# same way.
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
    # House lines have no published row by definition — they borrow one, so
    # they are excluded here and checked the other way round below.
    missing = [n for n, *_ in LINES
               if n not in df.index and n not in HOUSE_LINES]
    if missing:
        raise ValueError(f"jpm_correlations.csv is missing lines: {missing}")
    present = [n for n in HOUSE_LINES if n in df.index]
    if present:
        raise ValueError(f"these are published after all, so they should be "
                         f"ordinary lines rather than house lines: {present}")
    borrowed = [p for p in HOUSE_LINES.values() if p not in df.index]
    if borrowed:
        raise ValueError(f"house lines borrow rows that do not exist: "
                         f"{borrowed}")
    # The excluded duplicates are published rows we choose not to offer. If one
    # disappears from the CSV the exclusion is stale and should be revisited,
    # so fail loudly rather than carrying a comment that no longer applies.
    gone = [n for n in EXCLUDED_DUPLICATES if n not in df.index]
    if gone:
        raise ValueError(f"EXCLUDED_DUPLICATES not in jpm_correlations.csv: {gone}")
    raw = df.to_numpy(dtype=float)
    repaired = _higham(raw)
    return (list(df.index), raw, repaired,
            float(np.abs(repaired - raw).max()),
            float(np.linalg.eigvalsh(raw).min()))


CORR_NAMES, CORR_RAW, CORR, CORR_MAX_REPAIR, CORR_RAW_MIN_EIG = _load_corr()
CORR_INDEX = {n: i for i, n in enumerate(CORR_NAMES)}


def corr_for(lines):
    """Correlation sub-matrix for a list of LTCMA line names.

    One line name per sleeve. Every sleeve sits on a line, so every sleeve has
    a correlation row: its own published one, or — for the house lines in
    HOUSE_LINES — the row of the published line it borrows from, named there.

    Two sleeves that end up on the SAME row are left at 1.00 when they are the
    same line, because then they really are the same asset held twice. When one
    of them is a house line they are set to SELF_CORR instead: Digital Assets
    and Gold are closely related, not identical, and a 1.00 would make the
    matrix singular and force the engine to repair it.
    """
    names = [resolve(p) for p in lines]
    rows = [HOUSE_LINES.get(n, n) for n in names]
    idx = []
    for p, row in zip(lines, rows):
        if row not in CORR_INDEX:
            raise KeyError(f"no correlation row for {p!r}")
        idx.append(CORR_INDEX[row])
    C = np.array(CORR[np.ix_(idx, idx)], dtype=float, copy=True)
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j or rows[i] != rows[j] or names[i] == names[j]:
                continue
            if names[i] in HOUSE_LINES or names[j] in HOUSE_LINES:
                C[i, j] = SELF_CORR
    return C


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
