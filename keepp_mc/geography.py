"""
Geography — where each line's money sits, by large region.

Regions: US, Europe, Asia (excluding Japan), Japan, Global.

LOOK-THROUGH, FIXED SPLITS
Most lines sit in one region by definition (U.S. Large Cap is 100% US,
Japanese Equity is 100% Japan). A few broad indices span several regions; each
of those gets ONE fixed split at the large-region level, taken from the index's
published country weights. Large-region weights move slowly, so the splits are
not refreshed (agreed with the PM); each is dated so its age is visible.

"Global" holds everything that is not one of the four named regions:
  - other countries inside a split (Canada, Australia, Israel, Latin America …)
  - Emerging Markets lines (they span Asia, Latin America, the Middle East and
    Africa, so they are not forced into Asia)
  - lines with no meaningful geography (commodities, gold, hedge funds)
  - global private-markets lines whose regional mix the providers don't publish

Geography is a LABEL for display and optimizer limits only. It never changes a
return, a volatility or a correlation, so no simulated number depends on it.

Pure numpy / pandas. No Streamlit import.
"""

import numpy as np
import pandas as pd

from . import personal_lines as pl

REGIONS = ["US", "Europe", "Asia", "Japan", "Global"]
US, EU, AS, JP, GL = (dict(US=1.0), dict(Europe=1.0), dict(Asia=1.0),
                      dict(Japan=1.0), dict(Global=1.0))

# --------------------------------------------------------------------------
# Fixed splits for the broad indices (percent; each row sums to 100)
# --------------------------------------------------------------------------
# MSCI ACWI. US 64.21% and Japan 5.20% from MSCI's factsheet (30 Sep 2026).
# Europe / Asia / Global estimated from its country weights: Europe incl. UK
# and Switzerland; Asia = Taiwan, China, Korea, India, Hong Kong, Singapore;
# Global = Canada, Australia and the rest of emerging markets.
ACWI = {"US": 64, "Europe": 14, "Asia": 11, "Japan": 5, "Global": 6}
# MSCI World (developed markets only) = ACWI without emerging markets.
WORLD = {"US": 72, "Europe": 15, "Asia": 1, "Japan": 6, "Global": 6}
# MSCI EAFE (developed ex US & Canada): Europe incl. UK, Japan, Pacific.
EAFE = {"US": 0, "Europe": 65, "Asia": 4, "Japan": 23, "Global": 8}
# MSCI World ex USA = EAFE plus Canada.
WORLD_EX_US = {"US": 0, "Europe": 55, "Asia": 4, "Japan": 21, "Global": 20}
# FTSE World Government Bond Index, 30 Sep 2026 (LSEG quarterly report):
# US 40.66, eurozone 25.11 + UK 5.23 + Poland/Nordics 1.18, China 12.21 +
# Korea/Malaysia/Singapore 2.50, Japan 8.49, Canada/Australia/other 4.59.
WGBI = {"US": 41, "Europe": 31, "Asia": 15, "Japan": 8, "Global": 5}
# The same index without the US, rescaled to 100.
WGBI_EX_US = {"US": 0, "Europe": 53, "Asia": 25, "Japan": 14, "Global": 8}

SPLIT_SOURCES = {
    "ACWI": "MSCI ACWI factsheet, 30 Sep 2026 (US, Japan); other regions estimated from country weights",
    "WORLD": "MSCI World: ACWI without emerging markets (derived)",
    "EAFE": "MSCI EAFE: derived from the developed-market part of ACWI",
    "WORLD_EX_US": "MSCI World ex USA: EAFE plus Canada (derived)",
    "WGBI": "FTSE World Government Bond Index, 30 Sep 2026",
    "WGBI_EX_US": "FTSE WGBI without the US, rescaled (derived)",
}

# --------------------------------------------------------------------------
# J.P. Morgan 2027 LTCMA — all 59 lines
# --------------------------------------------------------------------------
JPM = {
    # United States
    **{n: US for n in [
        "U.S. Cash", "U.S. Short Duration Government/Credit",
        "U.S. Intermediate Treasuries", "U.S. Long Treasuries", "TIPS",
        "U.S. Aggregate Bonds", "U.S. Securitized",
        "U.S. Long Duration Government/Credit",
        "U.S. Inv Grade Corporate Bonds", "U.S. Long Corporate Bonds",
        "U.S. High Yield Bonds", "U.S. Leveraged Loans",
        "U.S. Muni 1-15 Yr Blend", "U.S. Muni High Yield",
        "U.S. Large Cap", "U.S. Mid Cap", "U.S. Small Cap",
        "U.S. Equity Value Factor", "U.S. Equity Momentum Factor",
        "U.S. Equity Quality Factor", "U.S. Equity Minimum Volatility Factor",
        "U.S. Equity Dividend Yield Factor",
        "U.S. Core Real Estate", "U.S. REITs"]},
    # Europe
    "Euro Area Large Cap": EU, "UK Large Cap": EU,
    "European Core Real Estate": EU,
    # Japan
    "Japanese Equity": JP,
    # Asia (excluding Japan)
    "Hong Kong Equity": AS, "Chinese Domestic Equity": AS,
    "AC Asia ex-Japan Equity": AS,
    "Asia Pacific Core Real Estate": AS,   # named Asia Pacific; no split published
    # Broad indices: fixed look-through splits
    "AC World Equity": ACWI,
    "EAFE Equity": EAFE,
    "World Government Bonds hedged": WGBI,
    "World Government Bonds": WGBI,
    "World ex-U.S. Government Bonds hedged": WGBI_EX_US,
    "World ex-U.S. Government Bonds": WGBI_EX_US,
    # Global: emerging markets, no geography, or no published regional mix
    **{n: GL for n in [
        "Emerging Markets Equity", "Emerging Markets Sovereign Debt",
        "Emerging Markets Local Currency Debt",
        "Emerging Markets Corporate Bonds",
        "Global Convertible Bonds hedged", "Global Core Real Estate",
        "Global Core Infrastructure",
        "Global Core Transport", "Global Timberland",
        "Commodities", "Gold", "Digital Assets",
        "Diversified Hedge Funds", "Event Driven Hedge Funds",
        "Long Bias Hedge Funds", "Relative Value Hedge Funds",
        "Macro Hedge Funds",
        "Private Equity", "Venture Capital", "Direct Lending",
        "Commercial Mortgage Loans"]},
}

# --------------------------------------------------------------------------
# BlackRock CMA, August 2026 — all 34 lines, plus the three J.P. Morgan
# lines offered in BlackRock Custom mode (same treatment as their J.P. Morgan
# originals: no meaningful geography, so Global)
# --------------------------------------------------------------------------
BLACKROCK = {
    "Europe large cap equities": EU,
    "US large cap equities": US,
    "Emerging large cap equities": GL,
    "Global ex-US large cap equities": WORLD_EX_US,   # MSCI World ex-US
    "Global large cap equities": WORLD,               # MSCI World
    "All country large cap equities": ACWI,           # MSCI ACWI
    "Listed REITs": GL,
    "US small cap equities": US,
    "Listed infrastructure equities": GL,
    "China-Broad market": AS,
    "China-Ashares": AS,
    "US aggregate bond index": US,
    "US agency MBS": US,
    "US credit (all maturities)": US,
    "US credit (10+ years)": US,
    "US government (all maturities)": US,
    "US government (10+ years)": US,
    "USD EM debt": GL,
    "US high yield": US,
    "US inflation-linked government": US,
    "Local-currency EM debt": GL,
    "US cash": US,
    "Global ex-US treasuries (hedged)": WGBI_EX_US,
    "Global aggregate bonds (hedged)": WGBI,          # closest published split
    "US bank loans": US,
    "Chinese treasuries": AS,
    "HY infrastructure debt": GL,
    "US private equity (buyout)": US,
    "Global infrastructure equity": GL,
    "Hedge funds (global)": GL,
    "US core real estate": US,
    "Global direct lending (levered)": GL,
    "Developed infrastructure debt": GL,
    "Global direct lending (unlevered)": GL,
    "Commodities (J.P. Morgan)": GL,
    "Gold (J.P. Morgan)": GL,
    "Digital Assets (J.P. Morgan)": GL,
}


def _row(split) -> np.ndarray:
    v = np.array([float(split.get(r, 0.0)) for r in REGIONS])
    return v / v.sum()


def shares(src, label) -> np.ndarray:
    """Region shares (sum 1) of one line under the active source. Personal
    lines take their proxy's split; an unknown line counts as Global."""
    name = str(label).strip()
    if pl.is_personal(name):
        name = pl.proxy_of(src, name)
    name = src.resolve(name)
    table = BLACKROCK if getattr(src, "KEY", "jpm") == "blackrock" else JPM
    return _row(table.get(name, GL))


def share_matrix(src, lines) -> np.ndarray:
    """(n_lines, 5) region shares, rows in the order of `lines`."""
    return np.array([shares(src, ln) for ln in lines]).reshape(-1, len(REGIONS))


TYPE_OF = {"Fixed Income": "Bonds", "Equities": "Stocks",
           "Alternatives": "Alternatives"}
TYPES = ["Stocks", "Bonds", "Alternatives"]


def breakdown(src, lines, categories, weights_pct) -> pd.DataFrame:
    """Region x asset type, in percent of the whole portfolio, plus a Total
    column. Rows: the five regions; values sum to the portfolio's weights."""
    S = share_matrix(src, lines)
    w = np.asarray(weights_pct, dtype=float)
    types = [TYPE_OF.get(c, "Alternatives") for c in categories]
    out = pd.DataFrame(0.0, index=REGIONS, columns=TYPES)
    for i, t in enumerate(types):
        out[t] += S[i] * w[i]
    out["Total"] = out[TYPES].sum(axis=1)
    return out


SCOPES = ["All assets", "Stocks", "Bonds"]


def exposure_rows(src, lines, categories, limits_df):
    """Optimizer exposure limits from the geography-limits table.

    `limits_df` rows are regions; columns '<scope> min %' / '<scope> max %'.
    Blank cells mean no limit. Returns a tuple of (name, coef, lo, hi) in
    DECIMALS, coef[i] = line i's share of the region if it is in the scope.
    """
    S = share_matrix(src, lines)
    types = np.array([TYPE_OF.get(c, "Alternatives") for c in categories])
    out = []
    for _, row in limits_df.iterrows():
        region = row["Region"]
        if region not in REGIONS:
            continue
        r = REGIONS.index(region)
        for scope in SCOPES:
            lo = row.get(f"{scope} min %")
            hi = row.get(f"{scope} max %")
            lo = None if lo is None or pd.isna(lo) else float(lo)
            hi = None if hi is None or pd.isna(hi) else float(hi)
            if lo is None and hi is None:
                continue
            mask = (np.ones(len(lines)) if scope == "All assets"
                    else (types == scope).astype(float))
            name = (f"{region} (all assets)" if scope == "All assets"
                    else f"{region} {scope.lower()}")
            out.append((name, S[:, r] * mask,
                        (lo or 0.0) / 100.0, (100.0 if hi is None else hi) / 100.0))
    return tuple(out)


def check(exposures, weights_dec) -> list:
    """Limits a weight vector breaks: [(name, value, lo, hi)], decimals."""
    w = np.asarray(weights_dec, dtype=float)
    bad = []
    for name, a, lo, hi in exposures:
        v = float(np.asarray(a) @ w)
        if v < lo - 1e-6 or v > hi + 1e-6:
            bad.append((name, v, lo, hi))
    return bad
