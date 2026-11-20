"""More than one assay series (e.g. two substrates): SERIES in the process notebook, and the analysis steps run per series."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import htmek_processing as hp

PINLIST = Path(__file__).parent / "data/pinlist_print2_tag.csv"


def _experiment(tmp_path, **cfg):
    drive = tmp_path / "T7"
    drive.mkdir(exist_ok=True)
    base = dict(EXPERIMENT="MBO-099", DRIVE=drive, PINLIST_FILE=str(PINLIST), CAMERA={"rotation_deg": 0.0})
    return hp.Experiment.from_config({**base, **cfg})


def test_series_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exp = _experiment(tmp_path, ASSAY_RUNS={"bzp_5mM": 5000},
                      SERIES={"pnpp": dict(dir="Assays/pNPP", substrate="pnpp_uM", runs={"p1": 100, "p10": 1000}),
                              "inhib": dict(dir="Assays/I", runs={"i0": {"acp_uM": 500, "inhibitor": "none"}})})
    assert exp.series["pnpp"]["runs"] == {"p1": {"pnpp_uM": 100}, "p10": {"pnpp_uM": 1000}}
    assert exp.series["inhib"]["substrate"] == "acp_uM"            # the first numeric column
    assert exp.scans["Assays/pNPP/p10"] is False and exp._is_assay("Assays/pNPP/p10") and not exp._is_assay("GFP/x")
    assert exp.reference_run == "Assays/bzp_5mM"
    assert exp._label("Assays/pNPP/p10") == "pnpp 1000 µM"          # the analysis notebook's tile-check label


def test_series_only_places_the_grid_on_its_highest_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exp = _experiment(tmp_path, SERIES={"pnpp": dict(dir="Assays/pNPP", substrate="pnpp_uM", runs={"p10": 1000, "p1": 100})})
    assert exp.reference_run == "Assays/pNPP/p10"


@pytest.mark.parametrize("bad", [dict(runs={"p1": 100}),                                  # no dir
                                 dict(dir="Assays/pNPP", runs={"p1": 100}),               # numbers, no substrate
                                 dict(dir="Assays/pNPP", runs={"p1": {"note": "x"}})])    # no [S] at all
def test_series_config_errors(tmp_path, monkeypatch, bad):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="SERIES"):
        _experiment(tmp_path, SERIES={"pnpp": bad})


def _runs(substrate, concs, vmax, km, n=4):
    """A small chip's progress curves: v0 = vmax [S] / (km + [S]), BLANKs in column 0 flat."""
    rows = []
    for s in concs:
        v0 = vmax * s / (km + s)
        for r in range(n):
            for c in range(n):
                tag = "BLANK" if c == 0 else "WT" if c == 1 else f"A{c}G"
                for f in range(12):
                    t = 30.0 * f
                    rows.append(dict(mark_row=r, mark_col=c, tag=tag, x=10 * c, y=10 * r, frame=f, seconds=t, minutes=t / 60,
                                     run=f"{substrate}_{s:g}", tile_imaged=True, complete_pass=True, **{substrate: s},
                                     chamber_ROI_median=1000.0 + (0 if tag == "BLANK" else v0 * t * (1 + 0.1 * r))))
    return pd.DataFrame(rows)


def test_analysis_runs_each_series(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assay = tmp_path / "4_exports/assay"
    assay.mkdir(parents=True)
    _runs("bzp_uM", [250, 1000, 4000], vmax=2.0, km=800).to_csv(assay / "nadh_per_chamber.csv", index=False)
    _runs("pnpp_uM", [50, 200, 800, 3200], vmax=1.0, km=300).to_csv(assay / "nadh_per_chamber_pnpp.csv", index=False)
    (assay / "series.json").write_text(json.dumps({
        "": dict(file="nadh_per_chamber.csv", substrate="bzp_uM", dir="Assays", runs=3),
        "pnpp": dict(file="nadh_per_chamber_pnpp.csv", substrate="pnpp_uM", dir="Assays/pNPP", runs=4)}))

    an = hp.Analysis.from_config({"EXPERIMENT": "test", "N_ROWS": 4, "N_COLS": 4})
    assert [n for n, _ in an.series_views()] == [None, "pnpp"]
    out = an.fit_rates()
    assert set(out) == {None, "pnpp"}
    an.fit_mm()
    printed = capsys.readouterr().out
    assert "Assay runs (`bzp_uM`)" in printed and "Series pnpp (`pnpp_uM`)" in printed

    p = an.series("pnpp")
    assert an.rates["bzp_uM"].nunique() == 3 and "pnpp_uM" not in an.rates
    assert p.rates["pnpp_uM"].nunique() == 4 and p.substrate == "pnpp_uM"
    for f in ("nadh_rates_per_chamber", "mm_per_chamber"):
        assert (assay / f"{f}.csv").exists() and (assay / f"{f}_pnpp.csv").exists()
    assert (tmp_path / "5_figures/qc/5_assay/5d-rates_pnpp/5d_bleaching.pdf").exists()
    lib = lambda mm: mm[mm["tag"] == "A2G"]["km"].median()
    assert lib(an.mm_raw) == pytest.approx(800, rel=0.1) and lib(p.mm_raw) == pytest.approx(300, rel=0.1)
    with pytest.raises(KeyError, match="pnpp"):
        an.series("nope")


def test_one_series_runs_as_before(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assay = tmp_path / "4_exports/assay"
    assay.mkdir(parents=True)
    _runs("bzp_uM", [250, 1000, 4000], vmax=2.0, km=800).to_csv(assay / "nadh_per_chamber.csv", index=False)
    an = hp.Analysis.from_config({"EXPERIMENT": "test", "N_ROWS": 4, "N_COLS": 4})
    assert an.fit_rates() is None and an.series_views() == [(None, an)]
    assert (assay / "nadh_rates_per_chamber.csv").exists()
