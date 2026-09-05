"""
J.P. Morgan 2026 Long-Term Capital Market Assumptions — asset line items.

Kept in its own module so the figures can be audited and updated (next year's
LTCMA) without touching the UI or the simulation engine.

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
"""

FI, EQ, ALT = "FI", "EQ", "ALT"

# (name, compound 2026, arithmetic 2026, annualized vol, compound 2025, family)
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
    ("World ex-U.S. Govt Bonds hedged",       4.20,  4.27,  3.95, 3.80, FI),
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

# name -> (compound, arith, vol, compound25, family)
BY_NAME = {n: (c, a, v, c25, f) for n, c, a, v, c25, f in LINES}

MANUAL = "Manual entry"

# Which LTCMA lines each of the six general buckets may be set to. Curated
# rather than "the whole family": the alternatives family mixes real assets,
# private markets and hedge funds, and offering all 19 in every alternatives
# row invites mismatches. Every row also gets MANUAL for hand-entered figures.
_HEDGE_FUNDS = ["Diversified Hedge Funds", "Event Driven Hedge Funds",
                "Long Bias Hedge Funds", "Relative Value Hedge Funds",
                "Macro Hedge Funds"]
_PRIVATE_MARKETS = ["Private Equity", "Venture Capital", "Direct Lending"]
_REAL_ASSETS = ["Commodities", "Gold", "U.S. Core Real Estate",
                "U.S. Value-Added Real Estate", "European Core Real Estate",
                "Asia Pacific Core Real Estate", "U.S. REITs",
                "Global Core Infrastructure", "Global Core Transport",
                "Global Timberland", "Commercial Mortgage Loans"]
_CASH_LIKE = ["U.S. Cash", "U.S. Short Duration Government/Credit",
              "U.S. Intermediate Treasuries", "U.S. Muni 1-15 Yr Blend"]


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
assert EXPANSION <= {n for n, *_ in LINES}, "expansion set names must exist in LINES"

# Each general asset type gets its OWN dropdown (its own Streamlit editor), so
# the option list is already scoped to that type. That makes a family prefix on
# the label redundant noise — "Diversified Hedge Funds" inside the Hedge Funds
# table is unambiguous.
LABEL = {n: n + (" ★" if n in EXPANSION else "") for n, *_ in LINES}
NAME_OF = {v: k for k, v in LABEL.items()}


def _family(fam):
    return [LABEL[n] for n, *_, f in LINES if f == fam]


CHOICES = {
    "FI":   [MANUAL] + _family(FI),
    "EQ":   [MANUAL] + _family(EQ),
    "PE":   [MANUAL] + [LABEL[n] for n in _PRIVATE_MARKETS],
    "HF":   [MANUAL] + [LABEL[n] for n in _HEDGE_FUNDS],
    "CMD":  [MANUAL] + [LABEL[n] for n in _REAL_ASSETS],
    "CASH": [MANUAL] + [LABEL[n] for n in _CASH_LIKE],
}


def lookup(label_or_name):
    """(compound return, vol) for an LTCMA line. None for MANUAL / unknown.

    Accepts the current dropdown label ("EQ - U.S. Large Cap ★"), the same
    label without its star, or the bare line name ("U.S. Large Cap").
    """
    s = str(label_or_name).strip()
    name = NAME_OF.get(s)
    if name is None:
        # Tolerate a label written by an earlier build: strip a trailing star
        # and/or the family prefix, then fall back to the bare line name. A
        # value stored in a live session must not stop resolving just because
        # the label format changed.
        s = s.removesuffix("★").strip()
        name = NAME_OF.get(s)
        if name is None:
            name = s.split(" - ", 1)[1] if " - " in s else s
    row = BY_NAME.get(name)
    if row is None:
        return None
    return row[0], row[2]


FAMILY_WORD = {FI: "fixed income", EQ: "equity", ALT: "alternatives"}

# Long display name for each group heading, and the order they appear in.
GROUP_KEYS = ["FI", "EQ", "PE", "HF", "CMD", "CASH"]


def family_of(label_or_name):
    """LTCMA family ('FI' / 'EQ' / 'ALT') for a line, or None for MANUAL."""
    s = str(label_or_name).strip()
    name = NAME_OF.get(s)
    if name is None:
        s = s.removesuffix("★").strip()
        name = NAME_OF.get(s) or (s.split(" - ", 1)[1] if " - " in s else s)
    row = BY_NAME.get(name)
    return row[4] if row else None


def sync_group(prev, edited, group_key):
    """Keep one asset type's sleeves consistent between label and figures.

    Each general asset type is rendered as its own table, so the LTCMA dropdown
    for that table already contains only that type's lines — there is no
    cross-family pick to guard against, which is why this is simpler than the
    single-table version it replaces.

    Rules:
      1. A row the editor just added arrives with NaNs: default it to Manual
         entry with a 0% weight, so it cannot silently poison the weight total.
      2. Newly picking an LTCMA line fills Return and Vol from the compound
         column.
      3. Typing over Return or Vol while a line is selected flips the label to
         Manual entry, so a sleeve never claims J.P. Morgan provenance for
         figures that are no longer J.P. Morgan's.
      4. Anything not in this group's option list falls back to Manual entry
         (only reachable from stale session state after a catalogue change).

    Returns (frame, dirty).
    """
    allowed = set(CHOICES[group_key])
    out = edited.copy().reset_index(drop=True)
    dirty = False

    for i in range(len(out)):
        line = out.at[i, "LTCMA line"]
        if not isinstance(line, str) or line not in allowed:
            out.at[i, "LTCMA line"] = MANUAL
            line = MANUAL
            dirty = True
        for col in ("Weight %", "Return %", "Vol %"):
            try:
                v = float(out.at[i, col])
            except (TypeError, ValueError):
                v = float("nan")
            if v != v:                      # NaN
                out.at[i, col] = 0.0
                dirty = True

        was = prev.at[i, "LTCMA line"] if i < len(prev) else None
        hit = lookup(line)
        if line != was:
            if hit is not None:
                out.at[i, "Return %"], out.at[i, "Vol %"] = hit
                dirty = True
        elif hit is not None:
            if (abs(float(out.at[i, "Return %"]) - hit[0]) > 1e-9
                    or abs(float(out.at[i, "Vol %"]) - hit[1]) > 1e-9):
                out.at[i, "LTCMA line"] = MANUAL
                dirty = True

    return out, dirty
