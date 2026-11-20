import pandas as pd

import htmek_processing as hp


def test_variant_iqr_and_page(tmp_path):
    rows = [dict(tag=t, km=k, vmax=v, r2=0.95) for t, base in (("K24A", 900), ("L87F", 1500), ("V9M", 600))
            for k, v in ((base * 0.9, 1e-5), (base, 1.2e-5), (base * 1.1, 1.4e-5))]
    rows += [dict(tag="WT", km=800, vmax=2e-5, r2=0.98), dict(tag="T26A", km=700, vmax=1e-5, r2=0.95)]   # T26A: one chamber
    pv = hp.viz.variant_iqr(pd.DataFrame(rows), scale=1e6, min_r2=0.9)
    assert list(pv.index) == ["K24A", "L87F", "V9M"] and pv.loc["L87F", "km"] == 1500 and abs(pv.loc["K24A", "vg"] - 12) < 1e-9
    out = hp.viz.structure_interactive(pv, tmp_path / "page.html", title="test",
                                       references=[dict(km=800, vg=20, label="WT r1 c1", group="WT (1)")], km_line=(800, "WT"),
                                       s_range_uM=(156, 5000))
    html = out.read_text()
    assert "PDBeMolstarPlugin" in html and "Plotly.newPlot" in html and "WT r1 c1" in html
    assert '"pos": 24' in html and "K24A" in html


def test_sequence_check(tmp_path):
    pv = hp.viz.variant_iqr(pd.DataFrame([dict(tag="Q24A", km=1000, vmax=1e-5)] * 3), scale=1e6)   # 2VH7 has K24
    try:
        hp.viz.structure_interactive(pv, tmp_path / "x.html")
    except ValueError as e:
        assert "Q24A" in str(e)
    else:
        raise AssertionError("a wrong wild-type residue should be caught")
