"""Paired reaction / DNA grids: per-chamber DNA centers in dumbbell_intensities, and GridAdjuster's paired mode
driven headlessly (switching grids, dragging a corner and a chamber in each, the saved files)."""

import json
from types import SimpleNamespace

import matplotlib
import numpy as np

matplotlib.use("Agg")

import htmek_processing as hp
from htmek_processing.processing.grid_adjuster import GridAdjuster, neck_lines

SHAPE = dict(dna_dx=40.0, dna_dy=0.0, dna_radius=8.0, neck_width=6.0)


def test_per_chamber_dna_centers_match_a_fixed_offset():
    rng = np.random.default_rng(0)
    img = rng.normal(100, 10, (300, 400))
    cy, cx = np.meshgrid([60.0, 150.0, 240.0], [60.0, 200.0, 320.0], indexing="ij")
    fixed = hp.dumbbell_intensities(img, cx, cy, None, radius=12, dumbbell=SHAPE, ring=(2.0, 8.0))
    paired = hp.dumbbell_intensities(img, cx, cy, None, radius=12, dumbbell=dict(dna_radius=8.0, neck_width=6.0), ring=(2.0, 8.0),
                                     dna_x=cx + 40, dna_y=cy)
    cols = ["dumbbell_sum", "dumbbell_n_px", "dumbbell_bg", "dumbbell_sum_net"]
    np.testing.assert_allclose(paired[cols].to_numpy(float), fixed[cols].to_numpy(float))
    moved = hp.dumbbell_intensities(img, cx, cy, None, radius=12, dumbbell=dict(dna_radius=8.0, neck_width=6.0),
                                    dna_x=cx + 40, dna_y=cy + np.where(cx > 100, 20, 0))   # the right two columns' DNA chambers lower
    assert not np.allclose(moved["dumbbell_sum"].to_numpy()[1::3], fixed["dumbbell_sum"].to_numpy()[1::3])


def test_neck_lines_join_the_circles():
    xs, ys = neck_lines((np.array([0.0]), np.array([0.0])), (np.array([100.0]), np.array([0.0])), 10, 20, 6)
    assert np.allclose(xs[[0, 1, 3, 4]], [np.sqrt(100 - 9), 100 - np.sqrt(400 - 9)] * 2)
    assert np.allclose(ys[[0, 1, 3, 4]], [3, 3, -3, -3])


def _event(ga, x, y, button=1, key=None):
    return SimpleNamespace(inaxes=ga.ax, xdata=x, ydata=y, button=button, key=key)


def test_paired_adjuster(tmp_path):
    img = np.zeros((600, 800), np.float32)
    rc, rw, dc, dw = (tmp_path / f for f in ("rc.json", "rw.json", "dc.json", "dw.json"))
    ga = GridAdjuster(img, n_rows=4, n_cols=5, corners=[[100, 100], [600, 100], [600, 450], [100, 450]], radius=20,
                      corners_path=rc, warp_path=rw, dna=dict(corners_path=dc, warp_path=dw, radius=12, offset=(60, 0)),
                      neck_width=8, block=False, local_fixes=True)
    r0, d0 = json.loads(rc.read_text()), json.loads(dc.read_text())
    assert r0["neck_width"] == 8 and r0["role"] == "reaction" and d0["radius"] == 12
    assert np.allclose(np.array(d0["corners"]) - np.array(r0["corners"]), [60, 0])
    assert len(ga._necks.get_xdata()) == 4 * 5 * 2 * 3   # two edges per chamber, NaN-separated

    # drag the DNA grid's top-left corner: only the DNA file changes
    ga._on_key(_event(ga, None, None, key="tab"))
    assert ga._active == "dna"
    ga._on_press(_event(ga, 160, 100)); ga._on_motion(_event(ga, 170, 110)); ga._on_release(_event(ga, 170, 110))
    assert json.loads(dc.read_text())["corners"][0] == [170, 110] and json.loads(rc.read_text())["corners"] == r0["corners"]

    # a local correction on a DNA chamber, then back to the reaction grid and drag one of its chambers
    cx, cy = ga.centers()
    ga._on_press(_event(ga, cx[1, 2], cy[1, 2])); ga._on_motion(_event(ga, cx[1, 2] + 5, cy[1, 2])); ga._on_release(_event(ga, 0, 0))
    assert len(hp.load_warp(dw)) == 1
    ga.switch_grid("reaction")
    assert ga.radius == 20 and ga._active == "reaction"
    rx, ry = ga.centers()
    ga._on_press(_event(ga, rx[2, 3], ry[2, 3])); ga._on_motion(_event(ga, rx[2, 3], ry[2, 3] + 4)); ga._on_release(_event(ga, 0, 0))
    assert len(hp.load_warp(rw)) == 1 and len(hp.load_warp(dw)) == 1

    # the neck slider saves with the reaction grid; reopening resumes both grids
    ga._dumbbell_sliders[-1].set_val(11)
    assert json.loads(rc.read_text())["neck_width"] == 11
    gb = GridAdjuster(img, n_rows=4, n_cols=5, radius=99, corners_path=rc, warp_path=rw,
                      dna=dict(corners_path=dc, warp_path=dw, radius=99, offset=(0, 0)), block=False)
    assert gb.radius == 20 and gb._grids["dna"]["radius"] == 12 and gb.neck_width == 11
    assert gb._grids["dna"]["corners"][0].tolist() == [170, 110]


def test_single_chambers_stay_put_by_default(tmp_path):
    img = np.zeros((600, 800), np.float32)
    ga = GridAdjuster(img, n_rows=4, n_cols=5, corners=[[100, 100], [600, 100], [600, 450], [100, 450]], radius=20,
                      corners_path=tmp_path / "c.json", warp_path=tmp_path / "w.json", block=False)
    cx, cy = ga.centers()
    ga._on_press(_event(ga, cx[1, 2], cy[1, 2])); ga._on_motion(_event(ga, cx[1, 2] + 9, cy[1, 2])); ga._on_release(_event(ga, 0, 0))
    assert not ga.controls and np.allclose(ga.centers()[0], cx)
