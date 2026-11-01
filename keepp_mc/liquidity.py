"""
Liquidity — how quickly each line's money can become cash, and at what cost.

Every line gets ONE ordinal tier (what the app displays), and each tier carries
two numbers (what the cash analysis uses):

  Tier  Label                  Time to cash   Crisis haircut  Typical lines
  L1    Cash                   T+0–2          0%              cash, T-bills
  L2    Daily liquid           T+2–5          5–15%           DM equity, IG credit, gold
  L3    Liquid, wide spreads   1–4 weeks      15–30%          EM equity & debt, HY, loans
  L4    Semi-liquid / gated    1–12 months    20–40%          hedge funds, evergreen / core
                                                              private funds, direct real estate
  L5    Locked                 3–10 years     35–60%          drawdown PE, VC, private credit,
                                                              Keepp's personal PE / PC lines

The point numbers are the MIDPOINT of each range (time to cash in calendar
days, haircut in decimals), so the rule is visible rather than hand-picked.

HAIRCUTS ARE JUDGEMENT CALLS. The crisis haircut is the discount to NAV a
forced seller should expect in a stressed market — wider than today's prices.
The L5 range is anchored on the LP-portfolio secondary market:

  Jefferies, Global Secondary Market Review, July 2026 (LP pricing, % of NAV,
  H1 2026): all strategies 87, buyout 91, infrastructure 91, credit 89,
  venture 79, real estate 68; tail-end funds (10+ years) at 25%+ discounts;
  Asian / emerging-market funds 59–66.
  Lazard, Interim Secondary Market Report 2026: ~87% of single-asset
  continuation-fund volume priced above 90% of NAV, but 40% of multi-asset
  volume priced below 90% (16% in 2025) — discounts widen fast for anything
  that is not top quality.

Those are CALM-market discounts of 9–32%; a crisis seller sits further out,
hence 35–60% for L5. The L2–L4 ranges are the brief's own estimates.

SPLITS. A line can span tiers (a macro hedge-fund line is part daily-dealing
CTA funds, part monthly-gated funds); its shares then sum to 1, exactly like
geography's look-through splits.

PERSONAL LINES default to L5. In Custom mode their tier and haircut can be
overridden — `overrides` = {line: {"tier": "L4", "haircut": 0.30}}, haircut in
decimals, either key optional. Overrides are honoured for the personal lines
only; provider lines keep their fixed tier.

Liquidity is a LABEL for display, optimizer limits and the cash analysis. It
never changes a return, a volatility or a correlation.

Pure numpy / pandas. No Streamlit import.
"""

import numpy as np
import pandas as pd

from . import personal_lines as pl
from .geography import TYPE_OF, TYPES, check  # noqa: F401  (check re-exported)

TIERS = ["L1", "L2", "L3", "L4", "L5"]
LABEL = {"L1": "Cash", "L2": "Daily liquid", "L3": "Liquid, wide spreads",
         "L4": "Semi-liquid / gated", "L5": "Locked"}
TIME_TO_CASH = {"L1": "T+0–2 days", "L2": "T+2–5 days", "L3": "1–4 weeks",
                "L4": "1–12 months", "L5": "3–10 years"}
# Calendar days, midpoint of each range (T+0–2, T+2–5, 1–4 wk, 1–12 mo, 3–10 yr).
DAYS = {"L1": 1.0, "L2": 3.5, "L3": 17.5, "L4": 6.5 * 30.4, "L5": 6.5 * 365.0}
# Percent ranges; HAIRCUT (decimals) is the midpoint.
HAIRCUT_RANGE = {"L1": (0, 0), "L2": (5, 15), "L3": (15, 30),
                 "L4": (20, 40), "L5": (35, 60)}
HAIRCUT = {t: (lo + hi) / 200.0 for t, (lo, hi) in HAIRCUT_RANGE.items()}

L1, L2, L3, L4, L5 = ({t: 1.0} for t in TIERS)
PERSONAL_DEFAULT = "L5"

# --------------------------------------------------------------------------
# J.P. Morgan 2027 LTCMA — all 59 lines
# --------------------------------------------------------------------------
JPM = {
    "U.S. Cash": L1,
    # Daily liquid: government, IG and agency bonds, developed equity, gold,
    # commodity futures, listed REITs
    **{n: L2 for n in [
        "U.S. Short Duration Government/Credit",
        "U.S. Intermediate Treasuries", "U.S. Long Treasuries", "TIPS",
        "U.S. Aggregate Bonds", "U.S. Securitized",
        "U.S. Long Duration Government/Credit",
        "U.S. Inv Grade Corporate Bonds", "U.S. Long Corporate Bonds",
        "U.S. Muni 1-15 Yr Blend",
        "World Government Bonds hedged", "World Government Bonds",
        "World ex-U.S. Government Bonds hedged",
        "World ex-U.S. Government Bonds",
        "U.S. Large Cap", "U.S. Mid Cap", "U.S. Small Cap",
        "U.S. Equity Value Factor", "U.S. Equity Momentum Factor",
        "U.S. Equity Quality Factor", "U.S. Equity Minimum Volatility Factor",
        "U.S. Equity Dividend Yield Factor",
        "Euro Area Large Cap", "UK Large Cap", "Japanese Equity",
        "Hong Kong Equity", "AC World Equity", "EAFE Equity",
        "U.S. REITs", "Commodities", "Gold"]},
    # Liquid, wide spreads: EM, high yield, loans (T+7 to weeks to settle),
    # convertibles, high-yield munis, China onshore, digital assets
    **{n: L3 for n in [
        "U.S. High Yield Bonds", "U.S. Leveraged Loans", "U.S. Muni High Yield",
        "Emerging Markets Sovereign Debt",
        "Emerging Markets Local Currency Debt",
        "Emerging Markets Corporate Bonds",
        "Global Convertible Bonds hedged",
        "Emerging Markets Equity", "AC Asia ex-Japan Equity",
        "Chinese Domestic Equity", "Digital Assets"]},
    # Semi-liquid / gated: hedge funds, open-ended core private funds
    **{n: L4 for n in [
        "Diversified Hedge Funds", "Event Driven Hedge Funds",
        "Long Bias Hedge Funds", "Relative Value Hedge Funds",
        "U.S. Core Real Estate", "European Core Real Estate",
        "Asia Pacific Core Real Estate", "Global Core Real Estate",
        "Global Core Infrastructure", "Commercial Mortgage Loans"]},
    # Macro: roughly half daily-dealing CTA / UCITS vehicles, half gated funds
    "Macro Hedge Funds": {"L3": 0.5, "L4": 0.5},
    # Locked: closed-end drawdown funds
    **{n: L5 for n in [
        "Private Equity", "Venture Capital", "Direct Lending",
        "Global Core Transport", "Global Timberland"]},
}

# --------------------------------------------------------------------------
# BlackRock CMA, August 2026 — all 34 lines, plus the three J.P. Morgan lines
# offered in BlackRock Custom mode (same tier as their J.P. Morgan originals)
# --------------------------------------------------------------------------
BLACKROCK = {
    "US cash": L1,
    **{n: L2 for n in [
        "Europe large cap equities", "US large cap equities",
        "Global ex-US large cap equities", "Global large cap equities",
        "All country large cap equities", "US small cap equities",
        "Listed REITs", "Listed infrastructure equities",
        "US aggregate bond index", "US agency MBS",
        "US credit (all maturities)", "US credit (10+ years)",
        "US government (all maturities)", "US government (10+ years)",
        "US inflation-linked government",
        "Global ex-US treasuries (hedged)", "Global aggregate bonds (hedged)",
        "Commodities (J.P. Morgan)", "Gold (J.P. Morgan)"]},
    **{n: L3 for n in [
        "Emerging large cap equities", "China-Broad market", "China-Ashares",
        "USD EM debt", "Local-currency EM debt", "US high yield",
        "US bank loans", "Chinese treasuries", "Digital Assets (J.P. Morgan)"]},
    **{n: L4 for n in ["Hedge funds (global)", "US core real estate"]},
    # Private infrastructure equity: open-ended core funds and closed-end funds
    "Global infrastructure equity": {"L4": 0.5, "L5": 0.5},
    **{n: L5 for n in [
        "US private equity (buyout)",
        "Global direct lending (levered)", "Global direct lending (unlevered)",
        "HY infrastructure debt", "Developed infrastructure debt"]},
}


def _row(split) -> np.ndarray:
    v = np.array([float(split.get(t, 0.0)) for t in TIERS])
    return v / v.sum()


def _override(label, overrides):
    """The override dict for a personal line, else {}."""
    name = str(label).strip()
    if not pl.is_personal(name) or not overrides:
        return {}
    return overrides.get(name) or {}


def tier_shares(src, label, overrides=None) -> np.ndarray:
    """Tier shares (sum 1) of one line under the active source. Personal
    lines sit in their override tier, else L5; an unknown line counts as L5
    (assume the worst about liquidity nobody has classified)."""
    name = str(label).strip()
    if pl.is_personal(name):
        t = _override(name, overrides).get("tier") or PERSONAL_DEFAULT
        return _row({t if t in TIERS else PERSONAL_DEFAULT: 1.0})
    name = src.resolve(name)
    table = BLACKROCK if getattr(src, "KEY", "jpm") == "blackrock" else JPM
    return _row(table.get(name, L5))


def tier_label(src, label, overrides=None) -> str:
    """The one tier label a line displays: 'L2', or 'L3/L4' for a split."""
    s = tier_shares(src, label, overrides)
    return "/".join(t for t, x in zip(TIERS, s) if x > 0)


def haircut(src, label, overrides=None) -> float:
    """Crisis haircut of one line, decimals: a personal line's own override,
    else its tier shares times the tier haircuts."""
    h = _override(label, overrides).get("haircut")
    if h is not None and h == h:
        return float(h)
    return float(tier_shares(src, label, overrides) @ np.array([HAIRCUT[t] for t in TIERS]))


def share_matrix(src, lines, overrides=None) -> np.ndarray:
    """(n_lines, 5) tier shares, rows in the order of `lines`."""
    return np.array([tier_shares(src, ln, overrides)
                     for ln in lines]).reshape(-1, len(TIERS))


def exposure(src, lines, weights_pct, overrides=None) -> pd.Series:
    """Share of the portfolio in each tier, same units as the weights."""
    w = np.asarray(weights_pct, dtype=float)
    return pd.Series(w @ share_matrix(src, lines, overrides), index=TIERS)


def breakdown(src, lines, categories, weights_pct, overrides=None) -> pd.DataFrame:
    """Tier x asset type, in percent of the whole portfolio, plus a Total
    column. Rows: the five tiers; values sum to the portfolio's weights."""
    S = share_matrix(src, lines, overrides)
    w = np.asarray(weights_pct, dtype=float)
    types = [TYPE_OF.get(c, "Alternatives") for c in categories]
    out = pd.DataFrame(0.0, index=TIERS, columns=TYPES)
    for i, t in enumerate(types):
        out[t] += S[i] * w[i]
    out["Total"] = out[TYPES].sum(axis=1)
    return out


def days_to_cash(src, lines, weights, overrides=None) -> float:
    """Weighted-average time to cash in calendar days (weights in any units;
    they are normalised). 0 for an empty portfolio."""
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        return 0.0
    d = share_matrix(src, lines, overrides) @ np.array([DAYS[t] for t in TIERS])
    return float(w @ d / w.sum())


def portfolio_haircut(src, lines, weights, overrides=None) -> float:
    """Weighted-average crisis haircut, decimals (weights normalised)."""
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        return 0.0
    h = np.array([haircut(src, ln, overrides) for ln in lines])
    return float(w @ h / w.sum())


def ladder_sentence(exp_pct) -> str:
    """'68% can be cash within a week, 12% within a year, 20% is locked for
    3+ years.' from a tier exposure in percent; empty parts are dropped."""
    e = pd.Series(exp_pct, index=TIERS).fillna(0.0)
    tot = float(e.sum())
    if tot <= 0:
        return ""
    week = (e["L1"] + e["L2"]) / tot * 100
    year = (e["L3"] + e["L4"]) / tot * 100
    lock = e["L5"] / tot * 100
    parts = []
    if week >= 0.05:
        parts.append(f"{week:.0f}% can be cash within a week")
    if year >= 0.05:
        parts.append(f"{year:.0f}% within a year" if parts
                     else f"{year:.0f}% can be cash within a year")
    if lock >= 0.05:
        parts.append(f"{lock:.0f}% is locked for 3+ years")
    s = ", ".join(parts)
    return s[0].upper() + s[1:] + "."


# Liquidity limits act on groups of tiers, as a share of the whole portfolio.
GROUPS = {"L1–L2 · cash within a week": ["L1", "L2"],
          "L4–L5 · gated or locked": ["L4", "L5"],
          "L5 · locked": ["L5"]}


def exposure_rows(src, lines, limits_df, overrides=None):
    """Optimizer exposure limits from the liquidity-limits table.

    `limits_df` rows are GROUPS keys under 'Tiers', columns 'Min %' / 'Max %';
    blank cells mean no limit. Returns a tuple of (name, coef, lo, hi) in
    DECIMALS, coef[i] = line i's share in those tiers — the same shape as
    geography.exposure_rows, so optimizer.Limits.exposure takes both.
    """
    S = share_matrix(src, lines, overrides)
    out = []
    for _, row in limits_df.iterrows():
        g = row["Tiers"]
        if g not in GROUPS:
            continue
        lo, hi = row.get("Min %"), row.get("Max %")
        lo = None if lo is None or pd.isna(lo) else float(lo)
        hi = None if hi is None or pd.isna(hi) else float(hi)
        if lo is None and hi is None:
            continue
        cols = [TIERS.index(t) for t in GROUPS[g]]
        out.append((g.split(" · ")[0] + " (liquidity)", S[:, cols].sum(axis=1),
                    (lo or 0.0) / 100.0, (100.0 if hi is None else hi) / 100.0))
    return tuple(out)
