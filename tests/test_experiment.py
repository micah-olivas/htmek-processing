"""Experiment: built from a Configure cell's names; its folders, scans, seam scan and grid-window rule."""

from pathlib import Path

import pandas as pd

import htmek_processing as hp


PINLIST = Path(__file__).parent / "data/pinlist_print2_tag.csv"   # Array-Print, print 2 (TAG)


def test_from_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    drive = tmp_path / "T7"; drive.mkdir()
    cfg = dict(EXPERIMENT="MBO-099", DRIVE=drive, PINLIST_FILE=str(PINLIST), CAMERA={"rotation_deg": 0.0},
               N_ROWS=56, N_COLS=32, MIRROR_COLS=True, EXPRESSION={"gfp_60min": 60, "gfp_20min": 20},
               SDS_WASHES={"GFP/w1": "SDS wash 1", "GFP/w4": "SDS wash 4"}, ASSAY_RUNS={"r5": 5000, "r2": 2000},
               STANDARDS={"kinetix": {"s0": 0}}, PLACE_DUMBBELLS=False, unrelated=1)
    exp = hp.Experiment.from_config(cfg)
    assert exp.experiment == "MBO-099" and exp.open_dumbbell_window is False and exp.tags.shape == (56, 32) and (exp.tags == 'BLANK').sum() == 24
    assert exp.scans["GFP/expression-time-course/gfp_60min"] is True and exp.scans["Assays/r5"] is False
    assert exp.scans["Standards/kinetix/s0"] is False and "GFP/gfp_bg_kinetix_500ms" in exp.scans
    assert exp.seam_scan == "GFP/expression-time-course/gfp_60min" == exp.flatfield_scan
    assert exp.expression_keys[-1].endswith("gfp_60min") and exp.button_scan == "GFP/w4" and exp.reference_run == "Assays/r5"
    assert all(Path(d).is_dir() for d in ("0_config", "2_masks", "3_intermediate", "4_exports", "5_figures"))
    assert hp.Experiment.place(None, tmp_path / "missing.json") and not hp.Experiment.place(False, tmp_path / "missing.json")
    (tmp_path / "c.json").write_text("[]")
    assert not hp.Experiment.place(None, tmp_path / "c.json") and hp.Experiment.place(True, tmp_path / "c.json")
