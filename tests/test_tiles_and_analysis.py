import numpy as np
import pandas as pd

import htmek_processing as hp


def _medians(dark=None):
    rng = np.random.default_rng(0)
    rows = []
    for s in range(6):
        for i in range(5):
            for j in range(5):
                m = 1000 * (1 + 0.1 * i) * (1 + rng.normal(0, 0.003)) * (2 if s == 3 else 1)   # one scan brighter overall
                if dark == (s, i, j):
                    m *= 0.85
                rows.append(dict(scan=f"s{s}", frame=0, tile_row=i, tile_col=j, y0=100 * i, y1=100 * (i + 1),
                                 x0=100 * j, x1=100 * (j + 1), median=m))
    return pd.DataFrame(rows)


def test_tile_deviations_flags_one_dark_tile():
    d = hp.tile_deviations(_medians(dark=(5, 4, 1)))
    bad = d[d["flagged"]]
    assert bad[["scan", "tile_row", "tile_col"]].values.tolist() == [["s5", 4, 1]]
    assert abs(bad["deviation"].iloc[0] + 0.15) < 0.02


def test_tile_of():
    t = _medians()
    r, c = hp.tile_of([50, 150, 450], [50, 250, 499], t[t["scan"] == "s0"])
    assert r.tolist() == [0, 2, 4] and c.tolist() == [0, 1, 4]


def test_analysis_skips_without_exports(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    an = hp.Analysis.from_config({"EXPERIMENT": "test"})
    for step in (an.call_expression, an.call_buttons, an.check_diffusion, an.calibrate, an.fit_rates,
                 an.rates_vs_enzyme, an.fit_mm, an.experiment_report):
        step()
    out = capsys.readouterr().out
    assert out.count("skipped") == 8


def test_call_buttons_leaves_out_flagged_tile(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "4_exports/gfp").mkdir(parents=True)
    (tmp_path / "4_exports/qc").mkdir(parents=True)
    rng = np.random.default_rng(1)
    rows = [dict(mark_row=r, mark_col=c, tag="BLANK" if c == 0 else "WT", scan="SDS wash 1", x=c * 15 + 5, y=r * 9 + 5,
                 button_sum_net=rng.normal(0, 10) + (0 if c == 0 else 500)) for r in range(56) for c in range(32)]
    pd.DataFrame(rows).to_csv(tmp_path / "4_exports/gfp/gfp_per_chamber.csv", index=False)
    t = hp.tile_deviations(_medians(dark=(5, 4, 1))).assign(scan=lambda d: d["scan"].replace({"s5": "SDS wash 1"}))
    t.to_csv(tmp_path / "4_exports/qc/tile_medians.csv", index=False)
    an = hp.Analysis.from_config({"EXPERIMENT": "test"})
    an.call_buttons()
    e = an.enzyme
    bad = e["in_flagged_tile"]
    assert bad.sum() > 0 and e.loc[bad, "gfp"].isna().all() and not e.loc[bad, "expressing"].any()
    assert (e["y"][bad] >= 400).all() and e["x"][bad].between(100, 200).all()
