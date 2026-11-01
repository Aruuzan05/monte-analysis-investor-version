"""
Keepp's own lines: Personal Private Equity and Personal Private Credit.

Two extra lines the user can add in CUSTOM mode, with their own return and
volatility typed in by hand. They are not part of either provider's catalogue,
so Standard mode never shows them, and the providers' modules are untouched.

CORRELATIONS. Each personal line borrows the correlation row of its asset type
in the active provider's matrix:

    Personal Private Equity -> J.P. Morgan "Private Equity"
                               BlackRock   "US private equity (buyout)"
    Personal Private Credit -> J.P. Morgan "Direct Lending"
                               BlackRock   "Global direct lending (unlevered)"

so its correlation with every other line is exactly that asset type's. The one
exception is the pair (personal line, its own proxy) when BOTH are held: a
copied row would put exactly 1.0 there, which makes the matrix singular (two
assets that always move identically) and forces the engine to repair it. That
single pair is set to SELF_CORR = 0.99 instead.

The return and volatility a personal line starts with when it is picked are
its proxy's published figures — a placeholder to type over, so a freshly added
row never sits at 0% volatility (which the optimizer would treat as risk-free).

Pure numpy / pandas, no Streamlit import.
"""

import numpy as np

PERSONAL_PE = "Personal Private Equity"
PERSONAL_PC = "Personal Private Credit"

# personal line -> {source key: proxy line in that source's catalogue}
PROXY = {
    PERSONAL_PE: {"jpm": "Private Equity",
                  "blackrock": "US private equity (buyout)"},
    PERSONAL_PC: {"jpm": "Direct Lending",
                  "blackrock": "Global direct lending (unlevered)"},
}
NAMES = list(PROXY)
CATEGORY = "ALT"                 # both sit under Alternatives
SELF_CORR = 0.99


def _key(src) -> str:
    return getattr(src, "KEY", "jpm")


def is_personal(label) -> bool:
    return str(label).strip() in PROXY


def proxy_of(src, label):
    """The provider line whose correlations a line uses (itself if not personal)."""
    name = str(label).strip()
    return PROXY[name][_key(src)] if name in PROXY else label


def choices(src, category_key):
    """The Custom-mode dropdown for one category, in three parts:

      1. the provider's own catalogue lines (also what Standard mode offers);
      2. any lines the provider exposes for Custom only — on BlackRock that is
         the three J.P. Morgan real-asset lines, since BlackRock's CMA has no
         commodities, gold or digital-asset line to offer;
      3. Keepp's two personal lines, under Alternatives.

    Standard mode reads src.CHOICES directly and so sees only part 1, which is
    what keeps it a single provider's catalogue.
    """
    base = list(src.CHOICES[category_key])
    base += list(getattr(src, "EXTRA_CHOICES", {}).get(category_key, []))
    return base + NAMES if category_key == CATEGORY else base


def lookup(src, label):
    """(return %, vol %) a line starts with when picked."""
    if is_personal(label):
        return src.lookup(proxy_of(src, label))
    return src.lookup(label)


def corr_for(src, lines):
    """Correlation sub-matrix for `lines`, personal lines on their proxy's row."""
    lines = list(lines)
    proxies = [proxy_of(src, ln) for ln in lines]
    C = np.array(src.corr_for(proxies), dtype=float, copy=True)
    names = [str(ln).strip() for ln in lines]
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i == j:
                continue
            # exactly one of the pair is personal and they share a proxy row
            if (is_personal(names[i]) != is_personal(names[j])
                    and src.resolve(proxies[i]) == src.resolve(proxies[j])):
                C[i, j] = SELF_CORR
    return C


def sync_group(src, prev, edited, category_key):
    """The source's own sync_group rules, with the personal lines allowed in
    Alternatives. Same three rules (see jpm_ltcma.sync_group):
      1. every row sits on a line of this category (else the default line);
      2. changing a row's line refills Return and Vol;
      3. typing over Return or Vol keeps the line.
    """
    allowed = set(choices(src, category_key))
    default_line = src.LABEL[src.DEFAULT_LINE[category_key]]
    out = edited.copy().reset_index(drop=True)
    deleted = len(out) < len(prev)
    dirty = False
    for i in range(len(out)):
        line = out.at[i, "LTCMA line"]
        forced = False
        if not isinstance(line, str) or line not in allowed:
            out.at[i, "LTCMA line"] = line = default_line          # rule 1
            forced = dirty = True
        was = prev.at[i, "LTCMA line"] if i < len(prev) else None
        if forced or (not deleted and line != was):                # rule 2
            out.at[i, "Return %"], out.at[i, "Vol %"] = lookup(src, line)
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
