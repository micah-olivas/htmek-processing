"""Hot-row check: sensor rows off from their neighbors in every raw tile, and where they land in the stitch."""

import numpy as np
import pandas as pd
import xarray as xr

import htmek_processing as hp

HOT, DARK = (40, 104, 168), (38, 102, 166)   # every 64 rows; a dark row two above each


def _raw(tmp_path, passes=3, R=2, C=2, h=200, w=120, offset=80.0):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:h, :w]
    base = 1000 + 400 * np.exp(-((xx - 60) ** 2 + (yy - 100) ** 2) / 4000)   # a smooth spot, like illumination
    t = np.empty((passes, R, C, h, w), np.float32)
    for p in range(passes):
        for r in range(R):
            for c in range(C):
                img = base * (1 + 0.05 * r + 0.02 * c) * (1 - 0.01 * p)
                img[list(HOT)] *= 2.5 - 0.1 * p    # the gain moves a little between passes
                img[list(DARK)] *= 0.92
                t[p, r, c] = img + offset + rng.normal(0, 3, (h, w))
    ds = xr.Dataset({"tile": (("time", "row", "col", "y", "x"), t.astype(np.uint16)),
                     "seconds": (("time", "row", "col"), np.arange(passes * R * C, dtype=float).reshape(passes, R, C))})
    src = tmp_path / "acq" / "tiled_images"
    ds.to_zarr(src)
    return src


def test_row_gains_finds_the_hot_and_dark_rows(tmp_path):
    g = hp.row_gains(_raw(tmp_path), offset=80.0, frames="all").assign(camera="Kinetix")
    assert sorted(g["frame"].unique()) == [0, 1, 2]
    f = hp.hot_rows(g)
    assert f["row"].tolist() == sorted(HOT + DARK)
    hot = f[f["row"].isin(HOT)]
    assert np.allclose(hot["gain"], 2.4, atol=0.05) and (hot["gain_hi"] - hot["gain_lo"] > 0.15).all()
    assert np.allclose(f[f["row"].isin(DARK)]["gain"], 0.92, atol=0.01)
    assert hp.row_period(hot["row"]) == 64
    assert hp.row_period(f["row"], f["gain"]) == 64   # a hot row and its dark neighbor: one block
    edges = g[(g["row"] < 7) | (g["row"] > 192)]
    assert edges["gain"].isna().all()                  # no centered neighbors at the sensor's edges
    clean = g[~g["row"].isin(HOT + DARK) & g["row"].between(10, 189)]
    assert (clean["gain"] - 1).abs().max() < 0.02


def test_ends_reads_the_first_and_last_pass(tmp_path):
    g = hp.row_gains(_raw(tmp_path), offset=80.0)
    assert sorted(g["frame"].unique()) == [0, 2]


def test_row_period_needs_one_spacing():
    assert hp.row_period([103, 231, 359, 487, 615]) == 128
    assert hp.row_period([5, 40, 300]) is None
    assert hp.row_period([101, 102, 103, 229, 230, 231, 357, 358, 359], [0.9, 0.95, 2.0] * 3) == 128
    # a faint row between the hot ones (the Kinetix: 2% dark every 256) does not halve the spacing
    assert hp.row_period([103, 231, 295, 359, 487, 551, 615], [2.0, 2.5, 0.98, 2.6, 2.7, 0.98, 2.8]) == 128


def _store(R=2, C=3, h=100, w=80, rotation=0.0, overlap=(10, 8)):
    oy, ox = overlap
    img = hp.stitch_tiles(np.zeros((R, C, h, w), np.float32), (oy, ox), split=bool(rotation))
    attrs = dict(rotation_deg=rotation, overlap_px=[oy, ox], tile_y0=[0] * R, tile_x0=[0] * C)
    return {"store": type("S", (), {"attrs": attrs})(), "image": img}


def test_hot_row_mask_without_rotation_is_the_rows_of_every_tile():
    s = _store()
    m = hp.hot_row_mask(s, [20, 60], tile_shape=(100, 80))
    assert m.shape == s["image"].shape and m.dtype == bool
    rows = np.where(m.all(axis=1))[0]   # tile rows keep 0-89 (the trailing 10 px trimmed): rows 20, 60, 110, 150
    assert rows.tolist() == [20, 60, 110, 150]
    assert m.sum() == 4 * m.shape[1]


def test_hot_row_mask_follows_the_rotation():
    s = _store(rotation=-2.15)
    m = hp.hot_row_mask(s, [50], tile_shape=(100, 80))
    ys, xs = np.nonzero(m[:, :70])   # the first tile column
    first = ys < 95
    slope = np.polyfit(xs[first], ys[first], 1)[0]
    assert abs(slope - np.tan(np.deg2rad(2.15))) < 0.01   # a line tilted by the rotation, as in the stitched image
    assert 1 <= m[:95, 35].sum() <= 3                      # 1-3 px thick
