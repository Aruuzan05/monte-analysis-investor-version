"""
BlackRock Capital Market Assumptions (August 2026, data as of 30 June 2026).

A second set of market assumptions beside jpm_ltcma.py. It exposes the SAME
interface the app already uses on the J.P. Morgan module (LINES, BY_NAME,
LABEL, CHOICES, resolve, lookup, category_of, corr_for, sync_group, ...), so
app.py and universe.py work against either one without knowing which.

Pure numpy / pandas. No Streamlit import.

WHERE THE NUMBERS COME FROM
blackrock_cma.csv is extracted from BlackRock's own workbook
(blackrock-capital-market-assumptions.xlsx) by running this module:

    python -m keepp_mc.blackrock_cma path/to/blackrock-capital-market-assumptions.xlsx

Only the USD rows are kept (the app is in US dollars), from three sheets:
"Starting point" (BlackRock's base case) and its two scenarios, "AI
productivity boom" and "Global risk premia rise". The scenarios change
expected returns only; volatility and correlations are identical across all
three. The "Global 60/40 portfolio" row is a portfolio, not an asset class,
and is left out. That leaves 34 investable lines. BlackRock refreshes the
workbook every quarter: re-run the command and commit the new CSV.

HOW IT DIFFERS FROM J.P. MORGAN — read before comparing results.

  1. RETURNS DEPEND ON THE HORIZON. BlackRock publishes an expected return for
     each of 5, 7, 10, 15, 20, 25 and 30 years; J.P. Morgan publishes one
     long-term figure. A source is therefore built for a horizon: the app
     passes the simulation's own number of years, and the return is read
     straight off BlackRock's column, interpolated linearly between the two
     nearest published horizons (a 12-year run uses 2/5 of the way from the
     10-year to the 15-year figure). Horizons under 5 years use the 5-year
     figure — BlackRock publishes nothing shorter.
     The figures are geometric (compound) annual returns, which is what
     engine.py expects, exactly as for J.P. Morgan's compound column.

  2. BLACKROCK DOES NOT PUBLISH A FULL CORRELATION MATRIX. Each asset comes
     with only two correlations: to government bonds and to equities. The
     simulator needs every pair. The matrix is built in two layers:

       a. BlackRock's own two correlations set the common part. With the
          bond/equity factor correlation matrix F, asset i's factor loadings
          c_i = (corr to bonds, corr to equities) give a shared component
          c_i' F^-1 c_j for every pair. This reproduces BlackRock's published
          numbers exactly.
       b. Two factors cannot see "family" resemblance: levered vs unlevered
          direct lending, China A-shares vs broad China, US credit vs long US
          credit. Tested on J.P. Morgan's own 59x59 matrix, a two-factor rebuild
          is within 0.05 for the median pair but misses within-family pairs
          by up to 0.8 (U.S. Core vs Value-Added Real Estate: 1.00 actual,
          0.16 rebuilt) — and an optimizer reads a missed family link as free
          diversification. So the part two factors leave unexplained is
          filled from J.P. Morgan's published matrix: each BlackRock asset is
          mapped to its closest J.P. Morgan line (CORR_PROXY below), and the
          proxies' RESIDUAL correlation, after removing J.P. Morgan's own
          bond and equity factors, scales the leftover:
              corr_ij = c_i' F^-1 c_j + sqrt((1-h_i)(1-h_j)) * resid(p_i, p_j)
          where h_i = c_i' F^-1 c_i is the share explained by the factors.

     The result is positive semi-definite by construction (a sum of two PSD
     terms), is repaired once with the same Higham routine J.P. Morgan's matrix
     uses, and BlackRock's published bond and equity correlations survive the
     repair to within the reported tolerance (see test_blackrock.py).

  3. CASH VOLATILITY IS 0 and BlackRock gives cash no correlations. Cash takes
     its factor loadings from its J.P. Morgan proxy (U.S. Cash). The engine
     handles a zero volatility; the optimizer floors it at its MIN_VOL.

  4. CATEGORIES. BlackRock's "Fixed income", "Equities" and "Private markets"
     map onto the app's three categories Fixed Income, Equities and
     Alternatives, so the category tables, bands and colours work unchanged.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import jpm_ltcma as jpm

FI, EQ, ALT = jpm.FI, jpm.EQ, jpm.ALT
CATEGORY_NAME = dict(jpm.CATEGORY_NAME)
CATEGORY_KEYS = list(jpm.CATEGORY_KEYS)

KEY = "blackrock"
EDITION = "August 2026"
AS_OF = "30 June 2026"
NAME = "BlackRock CMA (Aug 2026)"
SHORT = "BlackRock"

_CSV = Path(__file__).parent / "data" / "blackrock_cma.csv"
HORIZONS = [5, 7, 10, 15, 20, 25, 30]

# Sheet name in the workbook -> how the app names it.
SCENARIOS = {
    "Starting point": "Base case",
    "AI productivity boom": "AI productivity boom",
    "Global risk premia rise": "Global risk premia rise",
}
DEFAULT_SCENARIO = "Starting point"

_ASSET_CLASS = {"Fixed income": FI, "Equities": EQ, "Private markets": ALT}
_EXCLUDE = {"Global 60/40 portfolio"}          # a portfolio, not an asset class

# BlackRock asset -> the J.P. Morgan line whose correlation row supplies the
# "family" part of its correlations (layer b above). Chosen once, here, for
# the closest economic match; never picked by the user.
CORR_PROXY = {
    # Equities
    "Europe large cap equities": "Euro Area Large Cap",
    "US large cap equities": "U.S. Large Cap",
    "Emerging large cap equities": "Emerging Markets Equity",
    "Global ex-US large cap equities": "EAFE Equity",
    "Global large cap equities": "AC World Equity",
    "All country large cap equities": "AC World Equity",
    "Listed REITs": "U.S. REITs",
    "US small cap equities": "U.S. Small Cap",
    "Listed infrastructure equities": "U.S. Equity Dividend Yield Factor",
    "China-Broad market": "AC Asia ex-Japan Equity",
    "China-Ashares": "Chinese Domestic Equity",
    # Fixed income
    "US aggregate bond index": "U.S. Aggregate Bonds",
    "US agency MBS": "U.S. Securitized",
    "US credit (all maturities)": "U.S. Inv Grade Corporate Bonds",
    "US credit (10+ years)": "U.S. Long Corporate Bonds",
    "US government (all maturities)": "U.S. Intermediate Treasuries",
    "US government (10+ years)": "U.S. Long Treasuries",
    "USD EM debt": "Emerging Markets Sovereign Debt",
    "US high yield": "U.S. High Yield Bonds",
    "US inflation-linked government": "TIPS",
    "Local-currency EM debt": "Emerging Markets Local Currency Debt",
    "US cash": "U.S. Cash",
    "Global ex-US treasuries (hedged)": "World ex-U.S. Government Bonds hedged",
    "Global aggregate bonds (hedged)": "World Government Bonds hedged",
    "US bank loans": "U.S. Leveraged Loans",
    "Chinese treasuries": "World ex-U.S. Government Bonds",
    # Private markets
    "HY infrastructure debt": "U.S. High Yield Bonds",
    "US private equity (buyout)": "Private Equity",
    "Global infrastructure equity": "Global Core Infrastructure",
    "Hedge funds (global)": "Diversified Hedge Funds",
    "US core real estate": "U.S. Core Real Estate",
    "Global direct lending (levered)": "Direct Lending",
    "Developed infrastructure debt": "Commercial Mortgage Loans",
    "Global direct lending (unlevered)": "Direct Lending",
}

# BlackRock's own bond factor is "Global ex-US treasuries (hedged)": its
# correlation to government bonds is exactly 1, and its correlation to
# equities is the bond/equity factor correlation.
_BR_BOND_FACTOR = "Global ex-US treasuries (hedged)"

DEFAULT_LINE = {FI: "US aggregate bond index",
                EQ: "All country large cap equities",
                ALT: "Hedge funds (global)"}

_RET_COLS = [f"ret_{h}y" for h in HORIZONS]
CSV_COLS = (["scenario", "asset_class", "asset", "index"] + _RET_COLS
            + ["vol", "corr_gov_bonds", "corr_equities"])


# --------------------------------------------------------------------------
# Extraction from BlackRock's workbook (run as a script, once per quarter)
# --------------------------------------------------------------------------
def extract(xlsx_path) -> pd.DataFrame:
    """USD rows of the three scenario sheets as one tidy frame, in percent."""
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    sheets = {ws.title.strip(): ws for ws in wb.worksheets}
    out = []
    for sheet in SCENARIOS:
        if sheet not in sheets:
            raise ValueError(f"workbook has no sheet {sheet!r}")
        rows = list(sheets[sheet].iter_rows(values_only=True))
        head = [str(c).strip() if c is not None else "" for c in rows[2]]
        if head[:4] != ["Currency", "Asset class", "Asset", "Index"] \
                or head[4:11] != [f"{h} year" for h in HORIZONS]:
            raise ValueError(f"{sheet}: unexpected header layout {head[:11]}")
        top = [str(c).strip() if c is not None else "" for c in rows[1]]
        if top[25:28] != ["Volatility", "Correlation", ""]:
            raise ValueError(f"{sheet}: volatility/correlation columns moved")
        for r in rows[3:]:
            if r[0] != "USD" or str(r[2]).strip() in _EXCLUDE:
                continue
            pct = lambda x: None if x is None else round(float(x) * 100, 6)
            raw = lambda x: None if x is None else round(float(x), 6)
            out.append([sheet, str(r[1]).strip(), str(r[2]).strip(),
                        str(r[3]).strip()]
                       + [pct(x) for x in r[4:11]]
                       + [pct(r[25]), raw(r[26]), raw(r[27])])
    return pd.DataFrame(out, columns=CSV_COLS)


# --------------------------------------------------------------------------
# Data + correlation matrix (scenario- and horizon-independent)
# --------------------------------------------------------------------------
def _load():
    df = pd.read_csv(_CSV)
    if list(df.columns) != CSV_COLS:
        raise ValueError("blackrock_cma.csv: unexpected columns")
    base = df[df["scenario"] == DEFAULT_SCENARIO].reset_index(drop=True)
    names = list(base["asset"])
    missing = [n for n in names if n not in CORR_PROXY]
    if missing:
        raise ValueError(f"blackrock_cma.csv: no CORR_PROXY for {missing}")
    for n in names:
        if base.loc[base["asset"] == n, "asset_class"].iloc[0] not in _ASSET_CLASS:
            raise ValueError(f"blackrock_cma.csv: unknown asset class for {n!r}")
    for sc in SCENARIOS:
        if sorted(df.loc[df["scenario"] == sc, "asset"]) != sorted(names):
            raise ValueError(f"blackrock_cma.csv: scenario {sc!r} lists "
                             f"different assets")
    return df, base, names


def _factor_inv(rho_be):
    return np.linalg.inv(np.array([[1.0, rho_be], [rho_be, 1.0]]))


_JPM_GOV_LINES = ["U.S. Intermediate Treasuries", "U.S. Long Treasuries",
                  "World Government Bonds hedged",
                  "World ex-U.S. Government Bonds hedged"]


def _jpm_factors(CJ, jn):
    """Weights (over J.P. Morgan lines) of a broad equity factor and a broad
    government-bond factor: the first principal component of all equity lines
    and of the government-bond lines. A composite rather than a single line,
    so no proxy is ever the factor itself (whose residual would be empty)."""
    def pc1(members):
        idx = [jn.index(m) for m in members]
        w, V = np.linalg.eigh(CJ[np.ix_(idx, idx)])
        v = V[:, -1] * np.sign(V[:, -1].sum())
        a = np.zeros(len(jn))
        a[idx] = v
        return a
    eq = [n for n, *_, c in jpm.LINES if c == EQ]
    return pc1(_JPM_GOV_LINES), pc1(eq)


def _build_corr(base, names):
    # J.P. Morgan residual correlations between the proxies, after removing
    # a broad bond factor and a broad equity factor.
    jn = [n for n, *_ in jpm.LINES]
    CJ = jpm.corr_for(jn)
    ij = {n: k for k, n in enumerate(jn)}
    a_b, a_e = _jpm_factors(CJ, jn)
    sd_b, sd_e = np.sqrt(a_b @ CJ @ a_b), np.sqrt(a_e @ CJ @ a_e)
    load_b, load_e = (CJ @ a_b) / sd_b, (CJ @ a_e) / sd_e
    FJi = _factor_inv(float(a_b @ CJ @ a_e) / (sd_b * sd_e))
    proxies = [CORR_PROXY[n] for n in names]
    P = np.array([ij[p] for p in proxies])
    cJ = np.stack([load_b[P], load_e[P]], axis=1)
    common_J = cJ @ FJi @ cJ.T
    hJ = np.clip(np.einsum("ij,jk,ik->i", cJ, FJi, cJ), 0.0, 1.0)
    sJ = np.sqrt(1.0 - hJ)
    ok = sJ > 1e-6
    denom = np.outer(np.where(ok, sJ, 1.0), np.where(ok, sJ, 1.0))
    resid = (CJ[np.ix_(P, P)] - common_J) / denom
    resid[~ok, :] = 0.0
    resid[:, ~ok] = 0.0
    resid = np.clip(resid, -1.0, 1.0)
    np.fill_diagonal(resid, 1.0)

    # BlackRock's own factor loadings; cash (no correlations published) takes
    # its proxy's J.P. Morgan loadings.
    gb = base.set_index("asset")
    rho_be = float(gb.loc[_BR_BOND_FACTOR, "corr_equities"])
    FBi = _factor_inv(rho_be)
    cB = np.empty((len(names), 2))
    for k, n in enumerate(names):
        g, q = gb.loc[n, "corr_gov_bonds"], gb.loc[n, "corr_equities"]
        if pd.isna(g) or pd.isna(q):
            cB[k] = cJ[k]
        else:
            cB[k] = (float(g), float(q))
    hB = np.clip(np.einsum("ij,jk,ik->i", cB, FBi, cB), 0.0, 1.0)
    sB = np.sqrt(1.0 - hB)
    C = cB @ FBi @ cB.T + np.outer(sB, sB) * resid
    C = (C + C.T) / 2.0
    np.fill_diagonal(C, 1.0)
    repaired = jpm._higham(C)
    return C, repaired, cB, float(np.abs(repaired - C).max())


# Loaded at import — except when this file is run as the extraction script,
# which must work before blackrock_cma.csv exists.
if __name__ != "__main__":
    DATA, _BASE, NAMES = _load()
    CORR_RAW, CORR, FACTOR_LOADINGS, CORR_MAX_REPAIR = _build_corr(_BASE, NAMES)
    CORR_INDEX = {n: i for i, n in enumerate(NAMES)}
    _CAT_OF = {n: _ASSET_CLASS[c]
               for n, c in zip(_BASE["asset"], _BASE["asset_class"])}
    _VOL = dict(zip(_BASE["asset"], _BASE["vol"].astype(float)))
    INDEX_OF = dict(zip(_BASE["asset"], _BASE["index"]))


def horizon_note(years: int) -> str:
    """One line saying which of BlackRock's horizons a run uses."""
    y = int(years)
    if y <= HORIZONS[0]:
        return f"BlackRock's {HORIZONS[0]}-year return assumptions"
    if y in HORIZONS:
        return f"BlackRock's {y}-year return assumptions"
    lo = max(h for h in HORIZONS if h < y)
    hi = min(h for h in HORIZONS if h > y)
    return (f"BlackRock's return assumptions for {y} years, interpolated "
            f"between their {lo}- and {hi}-year figures")


def returns_for(scenario: str, years: int) -> dict:
    """{asset: compound return %} for a scenario at a horizon in years."""
    if scenario not in SCENARIOS:
        raise KeyError(f"unknown BlackRock scenario {scenario!r}")
    sub = DATA[DATA["scenario"] == scenario].set_index("asset")
    y = float(np.clip(int(years), HORIZONS[0], HORIZONS[-1]))
    return {n: float(np.interp(y, HORIZONS, sub.loc[n, _RET_COLS].astype(float)))
            for n in NAMES}


# --------------------------------------------------------------------------
# The source object — same interface as the jpm_ltcma module
# --------------------------------------------------------------------------
class BlackRockSource:
    """BlackRock's assumptions for one scenario at one horizon.

    Built through `get()`, which caches one per (scenario, years)."""

    FI, EQ, ALT = FI, EQ, ALT
    CATEGORY_NAME = CATEGORY_NAME
    CATEGORY_KEYS = CATEGORY_KEYS
    DEFAULT_LINE = DEFAULT_LINE
    EXPANSION: set = set()
    KEY = KEY
    NAME = NAME
    SHORT = SHORT

    def __init__(self, scenario: str = DEFAULT_SCENARIO, years: int = 10):
        self.scenario = scenario
        self.years = int(years)
        ret = returns_for(scenario, years)
        r5 = returns_for(scenario, HORIZONS[0])
        order = [n for k in CATEGORY_KEYS for n in NAMES if _CAT_OF[n] == k]
        # (name, compound, arithmetic ~ g + s^2/2, vol, 5-year compound, cat)
        self.LINES = [(n, round(ret[n], 4),
                       round(ret[n] + (_VOL[n] ** 2) / 200.0, 4),
                       round(_VOL[n], 4), round(r5[n], 4), _CAT_OF[n])
                      for n in order]
        self.BY_NAME = {n: (c, a, v, c5, f) for n, c, a, v, c5, f in self.LINES}
        self.LABEL = {n: n for n, *_ in self.LINES}
        self.NAME_OF = dict(self.LABEL)
        self.CHOICES = {k: [n for n, *_, c in self.LINES if c == k]
                        for k in CATEGORY_KEYS}
        self.CORR, self.CORR_INDEX = CORR, CORR_INDEX
        self.SOURCE_ID = source_id(scenario)
        self.HORIZON_NOTE = horizon_note(years)

    def resolve(self, label_or_name):
        return str(label_or_name).strip()

    def lookup(self, label_or_name):
        row = self.BY_NAME.get(self.resolve(label_or_name))
        return (row[0], row[2]) if row else None

    def category_of(self, label_or_name):
        row = self.BY_NAME.get(self.resolve(label_or_name))
        return row[4] if row else None

    def corr_for(self, lines):
        idx = []
        for p in lines:
            name = self.resolve(p)
            if name not in CORR_INDEX:
                raise KeyError(f"no correlation row for {p!r}")
            idx.append(CORR_INDEX[name])
        return CORR[np.ix_(idx, idx)]

    def sync_group(self, prev, edited, category_key):
        """Same rules as jpm_ltcma.sync_group, on BlackRock's lines."""
        allowed = set(self.CHOICES[category_key])
        default_line = self.LABEL[DEFAULT_LINE[category_key]]
        out = edited.copy().reset_index(drop=True)
        deleted = len(out) < len(prev)
        dirty = False
        for i in range(len(out)):
            line = out.at[i, "LTCMA line"]
            forced = False
            if not isinstance(line, str) or line not in allowed:
                out.at[i, "LTCMA line"] = line = default_line
                forced = dirty = True
            was = prev.at[i, "LTCMA line"] if i < len(prev) else None
            if forced or (not deleted and line != was):
                out.at[i, "Return %"], out.at[i, "Vol %"] = self.lookup(line)
                dirty = True
            for col in ("Weight %", "Return %", "Vol %"):
                try:
                    v = float(out.at[i, col])
                except (TypeError, ValueError):
                    v = float("nan")
                if v != v:
                    out.at[i, col] = 0.0
                    dirty = True
                else:
                    out.at[i, col] = v
        return out, dirty


def source_id(scenario: str = DEFAULT_SCENARIO) -> str:
    """What a saved run records: provider, edition and scenario."""
    return f"{KEY}-2026-08:{scenario}"


def parse_source_id(sid: str):
    """'blackrock-2026-08:AI productivity boom' -> 'AI productivity boom'."""
    _, _, scenario = str(sid).partition(":")
    return scenario if scenario in SCENARIOS else DEFAULT_SCENARIO


@functools.lru_cache(maxsize=128)
def get(scenario: str = DEFAULT_SCENARIO, years: int = 10) -> BlackRockSource:
    return BlackRockSource(scenario, int(years))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m keepp_mc.blackrock_cma <blackrock-capital-market-"
                 "assumptions.xlsx>")
    frame = extract(sys.argv[1])
    frame.to_csv(_CSV, index=False)
    n = frame["asset"].nunique()
    print(f"Wrote {_CSV.name}: {n} USD asset classes x "
          f"{frame['scenario'].nunique()} scenarios.")
