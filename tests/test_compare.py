"""Compare assumptions tab — keepp_mc/compare.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from keepp_mc import compare as cm
from keepp_mc import jpm_ltcma as jpm
from keepp_mc import personal_lines as pl
from keepp_mc.engine import simulate


def _run(frame, side, paths=2000, years=10, seed=7, src=jpm):
    w, r, v, c = cm.sim_inputs(frame, side, src)
    n = len(w)
    return simulate(w, r, v, np.array(c).reshape(n, n), paths, years, seed)[0]


LINES = ["U.S. Large Cap", "U.S. Aggregate Bonds", pl.PERSONAL_PE]
W = {"U.S. Large Cap": 50.0, "U.S. Aggregate Bonds": 30.0, pl.PERSONAL_PE: 20.0}


def test_untouched_rows_equal_jpm_on_both_sides():
    f = cm.build_frame(LINES, W, {}, {})
    assert (f["Keepp return %"] == f["Provider return %"]).all()
    assert (f["Keepp vol %"] == f["Provider vol %"]).all()
    k, j = _run(f, "keepp"), _run(f, "provider")
    assert k.rows[-1]["Median x"] == j.rows[-1]["Median x"]


def test_personal_line_uses_proxy_figures():
    assert cm.jpm_figures(pl.PERSONAL_PE) == cm.jpm_figures("Private Equity")
    assert cm.jpm_reference(pl.PERSONAL_PE) == "Private Equity"
    assert pl.PERSONAL_PE in cm.choices(pl.CATEGORY)


def test_frame_ordered_by_category_and_figures_published():
    f = cm.build_frame(LINES, W, {}, {})
    assert list(f["Category"]) == ["Fixed Income", "Equities", "Alternatives"]
    r, v = jpm.lookup("U.S. Large Cap")
    row = f[f["Line"] == "U.S. Large Cap"].iloc[0]
    assert (row["Provider return %"], row["Provider vol %"]) == (r, v)


def test_higher_keepp_return_raises_every_percentile_common_shocks():
    f = cm.build_frame(LINES, W, {pl.PERSONAL_PE: 14.0}, {})
    k, j = _run(f, "keepp"), _run(f, "provider")
    # same shocks, higher drift on one asset -> every quantile is higher
    assert (np.asarray(k.fan)[1:] > np.asarray(j.fan)[1:]).all()
    s = cm.summary(k, j, 1_000_000)
    assert s.set_index("Metric").loc["Typical value", "Difference"] > 0


def test_attribution_sums_to_weighted_return_gap():
    own = {"U.S. Large Cap": 8.0, pl.PERSONAL_PE: 15.0}
    f = cm.build_frame(LINES, W, own, {})
    a = cm.attribution(f)
    w = f["Weight %"] / 100
    gap = float((w * (f["Keepp return %"] - f["Provider return %"])).sum())
    assert a["Return effect pp"].sum() == pytest.approx(gap)


def test_zero_weight_rows_kept_so_draws_match_main_run():
    f = cm.build_frame(LINES + ["Gold"], {**W, "Gold": 0.0}, {}, {})
    w, *_ = cm.sim_inputs(f, "keepp")
    assert len(w) == 4 and sum(w) == 100.0


def test_excel_export_has_all_sheets():
    import io
    import openpyxl
    f = cm.build_frame(LINES, W, {pl.PERSONAL_PE: 14.0}, {"U.S. Large Cap": 18.0})
    k, j = _run(f, "keepp", paths=500), _run(f, "jpm", paths=500)
    b = cm.to_excel_bytes(f, cm.summary(k, j, 1e6), cm.horizons(k, j, 1e6),
                          {"Years": 10})
    wb = openpyxl.load_workbook(io.BytesIO(b))
    assert wb.sheetnames == ["Settings", "Assumptions", "Comparison",
                             "By horizon"]


def test_alloc_round_trip_and_jpm_side():
    import pandas as pd
    alloc = pd.DataFrame([
        {"Category": "Equities", "LTCMA line": jpm.LABEL["U.S. Large Cap"],
         "Weight %": 60.0, "Return %": 8.0, "Vol %": 18.0},
        {"Category": "Alternatives", "LTCMA line": pl.PERSONAL_PE,
         "Weight %": 40.0, "Return %": 14.0, "Vol %": 22.0},
    ])
    f = cm.frame_from_alloc(alloc)
    back = cm.to_alloc(f, "keepp")
    assert list(back["Return %"]) == [8.0, 14.0]
    j = cm.to_alloc(f, "provider")
    assert tuple(j.loc[1, ["Return %", "Vol %"]]) == cm.jpm_figures("Private Equity")


def test_duplicate_lines_merge_and_unknown_dropped():
    import pandas as pd
    alloc = pd.DataFrame([
        {"Category": "Equities", "LTCMA line": "U.S. Large Cap",
         "Weight %": 30.0, "Return %": 8.0, "Vol %": 18.0},
        {"Category": "Equities", "LTCMA line": "U.S. Large Cap",
         "Weight %": 20.0, "Return %": 9.0, "Vol %": 18.0},
        {"Category": "Equities", "LTCMA line": "Not a line",
         "Weight %": 50.0, "Return %": 9.0, "Vol %": 18.0},
    ])
    f = cm.frame_from_alloc(alloc)
    assert len(f) == 1 and f.loc[0, "Weight %"] == 50.0
    assert f.loc[0, "Keepp return %"] == 8.0


# ---- BlackRock ------------------------------------------------------------
from keepp_mc import blackrock_cma as brc


def test_blackrock_frame_uses_blackrock_figures_and_horizon():
    src10, src20 = brc.get(brc.DEFAULT_SCENARIO, 10), brc.get(brc.DEFAULT_SCENARIO, 20)
    ln = brc.DEFAULT_LINE[brc.EQ]
    f10 = cm.build_frame([ln], {ln: 100.0}, {}, {}, src10)
    f20 = cm.build_frame([ln], {ln: 100.0}, {}, {}, src20)
    assert f10.loc[0, "Provider return %"] == pytest.approx(src10.lookup(ln)[0])
    assert f20.loc[0, "Provider return %"] == pytest.approx(src20.lookup(ln)[0])
    assert (f10["Keepp return %"] == f10["Provider return %"]).all()


def test_blackrock_personal_line_proxy_and_round_trip():
    src = brc.get(brc.DEFAULT_SCENARIO, 10)
    assert cm.provider_reference(pl.PERSONAL_PE, src) == "US private equity (buyout)"
    assert cm.provider_figures(pl.PERSONAL_PE, src) == cm.provider_figures(
        "US private equity (buyout)", src)
    ln = brc.DEFAULT_LINE[brc.FI]
    f = cm.build_frame([ln, pl.PERSONAL_PE], {ln: 70.0, pl.PERSONAL_PE: 30.0},
                       {pl.PERSONAL_PE: 15.0}, {}, src)
    back = cm.frame_from_alloc(cm.to_alloc(f, "keepp", src), src)
    assert list(back["Keepp return %"]) == list(f["Keepp return %"])
    w, r, v, c = cm.sim_inputs(f, "provider", src)
    assert len(c) == 4 and r[1] == cm.provider_figures(pl.PERSONAL_PE, src)[0]


def test_blackrock_offers_jpm_real_asset_extras():
    src = brc.get(brc.DEFAULT_SCENARIO, 10)
    alts = cm.choices(brc.ALT, src)
    assert any("J.P. Morgan" in a for a in alts) and pl.PERSONAL_PC in alts


# ---- saved scenarios: classify + pair -----------------------------------
def _record(alloc, res, mode="Custom", source="jpm", years=10):
    from keepp_mc import run_store as rs
    return rs.make_record(0, "x", mode, alloc, 2000, years, 7, res, source=source)


def test_classify_and_pair_the_two_saved_views():
    f = cm.build_frame(LINES, W, {pl.PERSONAL_PE: 14.0}, {})
    rk, rj = _run(f, "keepp"), _run(f, "provider")
    k = _record(cm.to_alloc(f, "keepp"), rk)
    j = _record(cm.to_alloc(f, "provider"), rj)
    assert cm.classify(k, jpm) == "keepp" and cm.classify(j, jpm) == "provider"
    assert cm.pair_key(k, jpm) == cm.pair_key(j, jpm)
    rows = cm.pair_up([("K", "keepp", cm.pair_key(k, jpm)),
                       ("J", "provider", cm.pair_key(j, jpm)),
                       ("X", "keepp", ("other",))])
    assert rows == [{"keepp": "K", "provider": "J"},
                    {"keepp": "X", "provider": None}]
    mk, mj = cm.run_metrics(k), cm.run_metrics(result=rj)
    assert mk["Average return / yr"] > mj["Average return / yr"]
    assert cm.run_metrics(j)["Typical ×"] == pytest.approx(mj["Typical ×"])


def test_classify_blackrock_and_standard():
    src = brc.get(brc.DEFAULT_SCENARIO, 10)
    ln = brc.DEFAULT_LINE[brc.EQ]
    f = cm.build_frame([ln], {ln: 100.0}, {}, {}, src)
    r = _run(f, "provider", paths=300, src=src)
    sid = brc.source_id(brc.DEFAULT_SCENARIO)
    assert cm.classify(_record(cm.to_alloc(f, "provider", src), r,
                               source=sid), src) == "provider"
    f2 = cm.build_frame([ln], {ln: 100.0}, {ln: 3.0}, {}, src)
    assert cm.classify(_record(cm.to_alloc(f2, "keepp", src), r,
                               source=sid), src) == "keepp"
    assert cm.classify(_record(cm.to_alloc(f2, "keepp", src), r, mode="Base",
                               source=sid), src) == "provider"


def test_excel_summary_is_the_table_as_passed():
    import io
    import openpyxl
    import pandas as pd
    from keepp_mc import run_store as rs
    f = cm.build_frame(LINES, W, {}, {})
    rec = _record(cm.to_alloc(f, "keepp"), _run(f, "keepp", paths=300))
    view = pd.DataFrame([{"Current": "●", "#": 7, "View": "J.P. Morgan",
                          "Scenario": "x", "Avg return %": 6.5}])
    wb = openpyxl.load_workbook(io.BytesIO(rs.to_excel_bytes([rec],
                                                             summary=view)))
    ws = wb["Summary"]
    assert [c.value for c in ws[1]] == list(view.columns)
    assert ws["E2"].number_format == "0.00" and ws["B2"].value == 7


def test_excel_grouped_summary_header():
    import io
    import openpyxl
    import pandas as pd
    from keepp_mc import run_store as rs
    f = cm.build_frame(LINES, W, {}, {})
    rec = _record(cm.to_alloc(f, "keepp"), _run(f, "keepp", paths=300))
    df = pd.DataFrame([{("Scenario", ""): "x", ("Avg (%)", "Keepp"): 6.8,
                        ("Avg (%)", "Provider"): 6.9,
                        ("Avg (%)", "Difference (pp)"): -0.1}])
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    wb = openpyxl.load_workbook(io.BytesIO(rs.to_excel_bytes(
        [rec], summary=df, scenarios=pd.DataFrame([{"#": 1}]))))
    assert wb.sheetnames[:2] == ["Summary", "Scenarios"]
    ws = wb["Summary"]
    assert ws["B1"].value == "Avg (%)" and ws["D2"].value == "Difference (pp)"
    assert ws["A3"].value == "x" and ws["D3"].value == -0.1


if __name__ == "__main__":                 # same convention as the other tests
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
