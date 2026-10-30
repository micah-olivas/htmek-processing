"""The standard steps: grid placement with local corrections, stitching a run, and the
per-timepoint chamber readout."""

import json

import matplotlib

matplotlib.use("agg")

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import htmek_processing as hp
from htmek_processing.processing import stitch


# --- local corrections


def test_warp_offsets_are_local_and_pin_the_corners():
    dx, dy = hp.warp_offsets([[50, 28, 10.0, -6.0]], 56, 32)
    assert np.isclose(dx[50, 28], 10.0) and np.isclose(dy[50, 28], -6.0)   # exact at the control
    assert np.allclose([dx[0, 0], dx[0, 31], dx[55, 0], dx[55, 31]], 0, atol=1e-6)   # corners pinned
    assert abs(dx[5, 5]) < 0.01                                            # nothing far away
    assert 0 < dx[52, 30] < 10                                             # neighbors follow, less


def test_warped_grid_round_trip(tmp_path):
    corners = [[100, 100], [1100, 100], [1100, 1900], [100, 1900]]
    hp.save_corners(tmp_path / "c.json", corners, radius=10)
    # no corrections yet: the bilinear grid
    cx, cy = hp.warped_grid(tmp_path / "c.json", tmp_path / "w.json", 56, 32)
    assert np.allclose((cx, cy), hp.bilinear_grid(corners, 56, 32))
    hp.save_warp(tmp_path / "w.json", {(20, 10): (4.0, 2.0)})
    assert hp.load_warp(tmp_path / "w.json") == [[20, 10, 4.0, 2.0]]
    cx2, cy2 = hp.warped_grid(tmp_path / "c.json", tmp_path / "w.json", 56, 32)
    assert np.isclose(cx2[20, 10] - cx[20, 10], 4.0) and np.isclose(cy2[20, 10] - cy[20, 10], 2.0)


class _Event:
    def __init__(self, ax, x=None, y=None, button=1, key=None):
        self.inaxes, self.xdata, self.ydata, self.button, self.key = ax, x, y, button, key


@pytest.fixture
def adjuster(tmp_path):
    img = np.random.default_rng(0).random((4, 2000, 1200)).astype(np.float32)   # (time, y, x)
    corners = [[100, 100], [1100, 100], [1100, 1900], [100, 1900]]
    ga = hp.GridAdjuster(img, n_rows=56, n_cols=32, corners=corners, radius=10, frame=-1,
                         corners_path=tmp_path / "c.json", warp_path=tmp_path / "w.json")
    return ga, tmp_path


def test_grid_adjuster_saves_on_open_and_starts_on_the_last_frame(adjuster):
    ga, d = adjuster
    assert (d / "c.json").exists()          # a later cell finds the placement without any drag
    assert ga._frame == 3


def test_grid_adjuster_local_correction_drag_and_remove(adjuster):
    ga, d = adjuster
    cx, cy = ga.centers()
    x0, y0 = cx[30, 15], cy[30, 15]
    ga._on_press(_Event(ga.ax, x0, y0))
    assert ga._drag[0] == "chamber"
    ga._last_drag_draw = 0.0
    ga._on_motion(_Event(ga.ax, x0 + 12, y0 - 5))
    ga._on_release(_Event(ga.ax, x0 + 12, y0 - 5))
    assert hp.load_warp(d / "w.json") == [[30, 15, 12.0, -5.0]]
    cx2, cy2 = hp.warped_grid(d / "c.json", d / "w.json", 56, 32)
    assert np.allclose((cx2, cy2), ga.centers())                     # later cells see the same grid
    ga._on_press(_Event(ga.ax, cx2[30, 15], cy2[30, 15], button=3))  # right-click: removed
    assert hp.load_warp(d / "w.json") == []


def test_grid_adjuster_corner_drag_still_moves_the_corner(adjuster):
    ga, d = adjuster
    tr = ga.corners[1].copy()
    ga._on_press(_Event(ga.ax, *tr))
    assert ga._drag == 1
    ga._last_drag_draw = 0.0
    ga._on_motion(_Event(ga.ax, tr[0] + 7, tr[1]))
    ga._on_release(_Event(ga.ax, tr[0] + 7, tr[1]))
    assert np.isclose(hp.load_mask(d / "c.json")["corners"][1][0], tr[0] + 7)


def test_grid_adjuster_zoom_keys(adjuster):
    ga, _ = adjuster
    ga._on_key(_Event(ga.ax, key="+"))
    x0, x1 = ga.ax.get_xlim()
    assert np.isclose(x1 - x0, 600) and np.isclose((x0 + x1) / 2, 599.5)   # 2x, on the center
    for _ in range(4):
        ga._on_key(_Event(ga.ax, key="-"))
    assert np.allclose(ga.ax.get_xlim(), (-0.5, 1199.5))                   # never past the chip
    ga._on_key(_Event(ga.ax, key="3"))
    ga._on_key(_Event(ga.ax, key="h"))
    assert np.allclose(ga.ax.get_xlim(), (-0.5, 1199.5))


def test_grid_adjuster_key_panel_fits_and_clears_the_sliders(adjuster):
    ga, _ = adjuster
    for size in [(10, 6.5), (12.5, 8.5)]:
        ga.fig.set_size_inches(*size)
        ga._draw_key()
        ga.fig.canvas.draw()
        r = ga.fig.canvas.get_renderer()
        panel = ga._key_ax.get_window_extent(r)
        assert panel.y0 >= ga.fig.bbox.y0 and panel.y1 <= ga.fig.bbox.y1
        assert all(t.get_window_extent(r).x1 <= panel.x1 + 1 for t in ga._key_ax.texts)
        for s in (ga._frame_slider, ga._vmin_slider, ga._radius_slider):
            assert not s.label.get_window_extent(r).overlaps(panel)


# --- stitching a run


def test_stitch_geometry_measures_once_and_checks_the_seams(tmp_path, monkeypatch):
    calls = []

    def fake_measure(src, *, rotation_deg, frame):
        calls.append(frame)
        return dict(overlap_y=[300.0], overlap_x=[301.0], cross_dy=0.2, cross_dx=-0.4,
                    spread_y=[0.1], spread_x=[0.2])

    tiled = xr.Dataset({"tile": (("time", "row", "col", "y", "x"), np.zeros((5, 2, 2, 4, 4)))})
    monkeypatch.setattr(stitch, "measure_tile_overlaps", fake_measure)
    monkeypatch.setattr(stitch, "open_tiled", lambda src: tiled)
    g = hp.stitch_geometry("run", tmp_path / "g.json", rotation_deg=-2.15)
    assert calls == [4]                      # the last frame
    assert g["overlap_x"] == [301.0] and g["rotation_deg"] == -2.15
    hp.stitch_geometry("run", tmp_path / "g.json", rotation_deg=-2.15)
    assert calls == [4]                      # reused, not measured again

    bad = dict(json.loads((tmp_path / "g.json").read_text()), cross_dx=-8.0)
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="rotation"):
        hp.stitch_geometry("run", tmp_path / "bad.json", rotation_deg=-1.94)


def test_fit_flatfield_skips_an_existing_flatfield(tmp_path, monkeypatch):
    out = tmp_path / "flat.tif"
    out.write_bytes(b"")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("refitted an existing flatfield"))
    assert hp.fit_flatfield("run", out) == out


# --- per-timepoint intensities


def test_chamber_timecourse_times_each_chamber_by_its_tile_and_subtracts_frame0():
    # a 2 x 2 chamber grid on a 2 x 2-tile image; each chamber brightens by 10 counts a frame
    n_t, tile = 3, 100
    img = np.zeros((n_t, 2 * tile, 2 * tile), np.float32)
    cx = np.array([[50.0, 150.0], [50.0, 150.0]])
    cy = np.array([[50.0, 50.0], [150.0, 150.0]])
    for t in range(n_t):
        img[t] = 100 + 10 * t
    seconds = np.array([[[0, 1], [2, 3]], [[60, 61], [62, 63]], [[120, 121], [122, 123]]], float)
    tiled = xr.Dataset({"seconds": (("time", "row", "col"), seconds)})
    df = hp.chamber_timecourse(xr.DataArray(img, dims=("time", "y", "x")), cx, cy,
                               np.full((2, 2), "WT", dtype=object), tiled, (tile, tile),
                               radius=10, ring=(20, 30))
    one = df[(df.mark_row == 1) & (df.mark_col == 1)].sort_values("frame")
    assert one["seconds"].tolist() == [3.0, 63.0, 123.0]           # its own tile's time
    assert np.allclose(one["median"], [100, 110, 120])            # raw, nothing subtracted
    assert np.allclose(one["median_minus_frame0"], [0, 10, 20])
    assert {"bg", "median_net", "n_px"} <= set(df.columns)


# --- progress-curve rates


def _traces(rng, n_wt=6, n_blank=6, frames=20, dt=60.0, bleach=0.002):
    """Synthetic progress curves: WT chambers follow a 4PL (rate known), BLANKs are flat, and
    every chamber loses `bleach` of its signal per frame."""
    rows = []
    a, d, c, b = 5000.0, 15000.0, 300.0, 1.4
    t = np.arange(frames) * dt + 20
    loss = (1 - bleach) ** np.arange(frames)
    for k in range(n_wt + n_blank):
        wt = k < n_wt
        y = hp.four_pl(t, a, d, c, b) if wt else np.full(frames, 5000.0)
        y = y * loss + rng.normal(0, 15, frames)
        rows += [dict(mark_row=k, mark_col=0, tag="WT" if wt else "BLANK", frame=f, seconds=t[f], median=y[f])
                 for f in range(frames)]
    return pd.DataFrame(rows), (a, d, c, b)


def test_fit_progress_curves_recovers_the_4pl_rate_and_leaves_blanks_to_the_line():
    rng = np.random.default_rng(1)
    traces, p = _traces(rng)
    out, rates = hp.fit_progress_curves(traces, "median", n_jobs=1)
    assert "median_bc" in out and out.attrs["bleach_curve"]["all"][19] < 0.97   # bleaching seen
    wt, blank = rates[rates.tag == "WT"], rates[rates.tag == "BLANK"]
    assert (wt.model == "4pl").all()
    _, v_true = hp.peak_4pl(*p)
    assert np.allclose(wt.v0, v_true, rtol=0.05)
    assert (blank.model == "linear").all() and (blank.v0.abs() < 0.2).all()


def test_fit_progress_curves_substrate_rules_only_with_a_substrate():
    rng = np.random.default_rng(2)
    traces, _ = _traces(rng, n_blank=2)
    # in counts, the [S] rules are off: a rise of 10,000 "counts" vs [S] = 5 doesn't matter
    _, rates = hp.fit_progress_curves(traces, "median", n_jobs=1)
    assert (rates[rates.tag == "WT"].model == "4pl").all()
    # with a substrate column in the same units, rule 6 (rise <= 1.2 x [S]) applies
    _, rates = hp.fit_progress_curves(traces.assign(s_uM=5.0), "median", substrate="s_uM", n_jobs=1)
    assert rates[rates.tag == "WT"].reason.str.contains(r"\[S\]").all()


def test_progress_fit_overlays_and_examples():
    rng = np.random.default_rng(3)
    traces, _ = _traces(rng, n_wt=2, n_blank=2)
    out, rates = hp.fit_progress_curves(traces, "median", n_jobs=1)
    ov = hp.progress_fit_overlays(out, rates, "median_bc")
    assert set(ov["curves"]) == set(ov["tangents"]) == {(0, 0), (1, 0)}
    assert set(ov["lines"]) == {(2, 0), (3, 0)}
    fig = hp.show_progress_fits(out, rates, [(0, 0), (2, 0)], "median_bc")
    assert len([a for a in fig.axes if a.get_visible()]) == 2


def test_fit_bleaching_recovers_an_exponential_loss_and_ignores_leaky_blanks():
    rng = np.random.default_rng(4)
    frames, rows = 20, []
    true = hp.bleach_model(np.arange(frames), 0.02, 0.4, 0.98)
    for k in range(40):
        leaky = k < 5                                   # product leaking in: rising
        y = 5000 * true * (1 + (0.004 * np.arange(frames) if leaky else 0)) + rng.normal(0, 3, frames)
        rows += [dict(mark_row=k, mark_col=0, tag="BLANK", frame=f, seconds=60.0 * f, median=y[f]) for f in range(frames)]
    traces = pd.DataFrame(rows)
    fit = hp.fit_bleaching(traces, "median", max_slope=0.001)
    assert fit["n_left_out"] == 5
    assert np.isclose(fit["amplitude"], 0.02, atol=0.003) and np.isclose(fit["rate"], 0.4, rtol=0.25)
    assert np.allclose(list(fit["curve"].values()), true, atol=0.002)
    # fit_progress_curves uses the model by default; corrected BLANKs come out flat
    out, _ = hp.fit_progress_curves(traces, "median", n_jobs=1)
    flat = out[out.mark_row >= 5].groupby("frame")["median_bc"].median()
    assert abs(flat.iloc[-1] / flat.iloc[0] - 1) < 0.002
    assert "bleach_fit" in out.attrs


def test_fit_progress_curves_keys_bleach_fits_by_the_run_value():
    rng = np.random.default_rng(5)
    traces, _ = _traces(rng, n_wt=2, n_blank=6)
    traces = pd.concat([traces.assign(s_uM=5000.0), traces.assign(s_uM=1000.0)], ignore_index=True)
    out, rates = hp.fit_progress_curves(traces, "median", per=("s_uM", "mark_row", "mark_col"), n_jobs=1)
    assert set(out.attrs["bleach_fit"]) == {5000.0, 1000.0}
    assert len(rates) == 2 * 8
