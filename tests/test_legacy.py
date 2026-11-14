from datetime import datetime, timedelta

import numpy as np
import pytest
import tifffile

import htmek_processing as hp

R, C, H, W = 2, 3, 16, 16
STEP = 3300.0


def _write_pass(folder, t0, *, lag=False):
    """A legacy pass: tile (r, c) holds the value 100*r + 10*c; its ImageDescription records the stage position, one
    tile behind (in acquisition order) when `lag`, as on the 2024-08-30 run."""
    folder.mkdir(parents=True)
    order = [(r, c) for r in range(R) for c in range(C)]
    for i, (r, c) in enumerate(order):
        sr, sc = order[i - 1] if lag else (r, c)
        when = (t0 + timedelta(seconds=i)).strftime("%Y-%m-%d %H:%M:%S")
        tifffile.imwrite(
            folder / f"4-Pos{c:03d}_{r:03d}_50.tif", np.full((H, W), 100 * r + 10 * c, np.uint16),
            description=f"{{Channel: dapi, Index:{i}, Pos:({sc * STEP - 5000.0},{sr * STEP + 1000.0})}}",
            extratags=[(306, "s", 20, when, False), (33434, 5, 1, (50, 1000), False)])


def test_timecourse_by_filename(tmp_path):
    t0 = datetime(2024, 8, 30, 16, 0, 0)
    for k in range(3):
        _write_pass(tmp_path / "run" / f"2024083{k}-pass_{k}", t0 + timedelta(seconds=75 * k), lag=True)
    (tmp_path / "run" / "StitchedImages").mkdir()                    # ignored
    dst = hp.read_legacy_tiles(tmp_path / "run", tmp_path / "out", overlap=0.01, mirror_cols=False)
    ds = hp.open_tiled(dst)
    assert ds["tile"].dims == ("time", "row", "col", "y", "x") and ds["tile"].shape == (3, R, C, H, W)
    assert ds["tile"].dtype == np.uint16
    tile = ds["tile"].values
    for r in range(R):
        for c in range(C):
            assert (tile[:, r, c] == 100 * r + 10 * c).all()           # placed by file name despite the lag
    assert ds["seconds"].shape == (3, R, C)
    assert ds["seconds"].values[0, 0, 0] == 0 and ds["seconds"].values[2, 0, 0] == 150
    assert ds.attrs["overlap"] == 0.01 and ds.attrs["channel"] == "dapi" and ds.attrs["exposure_ms"] == 50
    assert hp.overlap_px(ds) == 0                                       # 1% of 16 px rounds to 0 here; real tiles 1024


def test_snapshot_and_stage_positions(tmp_path):
    _write_pass(tmp_path / "scan", datetime(2024, 8, 30, 12, 40, 30))
    ds = hp.open_tiled(hp.read_legacy_tiles(tmp_path / "scan", tmp_path / "out", overlap=0.1, positions="stage",
                                            mirror_cols=False))
    assert ds["tile"].dims == ("row", "col", "y", "x") and "timestamp" in ds
    assert (ds["tile"].values[1, 2] == 120).all()
    assert np.allclose(ds["col"].values, np.arange(C) * STEP - 5000.0)
    assert np.allclose(ds["row"].values, np.arange(R) * STEP + 1000.0)


def test_requires_overlap(tmp_path):
    _write_pass(tmp_path / "scan", datetime(2024, 8, 30))
    with pytest.raises(ValueError, match="overlap"):
        hp.read_legacy_tiles(tmp_path / "scan", tmp_path / "out", overlap=-0.01)


def test_incomplete_pass_raises(tmp_path):
    t0 = datetime(2024, 8, 30)
    _write_pass(tmp_path / "run" / "p0", t0)
    _write_pass(tmp_path / "run" / "p1", t0)
    next((tmp_path / "run" / "p1").glob("*Pos002_001*")).unlink()
    with pytest.raises(ValueError, match="differ"):
        hp.read_legacy_tiles(tmp_path / "run", tmp_path / "out", overlap=0.1)


def test_stitch_run_with_legacy_geometry(tmp_path):
    t0 = datetime(2024, 8, 30, 16, 0, 0)
    for k in range(2):
        _write_pass(tmp_path / "run" / f"p{k}", t0 + timedelta(seconds=75 * k))
    src = hp.read_legacy_tiles(tmp_path / "run", tmp_path / "out", overlap=0)   # mirror_cols: file column 0 at right
    hp.legacy_geometry(tmp_path / "geometry.json", src)
    s = hp.stitch_run(src, tmp_path / "stitched.zarr", camera={"rotation_deg": 0.0, "offset": 0.0},
                      geometry_path=tmp_path / "geometry.json", flatfield_path=None)
    img = s["image"].values
    assert img.shape == (2, R * H, C * W)
    assert (img[:, H:, :W] == 120).all() and (img[:, :H, 2 * W:] == 0).all()   # butted, columns mirrored


def test_float_tiles_without_tags(tmp_path):
    """A processed copy: float32 tiles with no TIFF tags (MBO-R01's "_rebinned" GFP scans)."""
    scan = tmp_path / "20240830-124030_egfp_flushed_500_egfp"
    scan.mkdir()
    for r in range(R):
        for c in range(C):
            tifffile.imwrite(scan / f"4-Pos{c:03d}_{r:03d}_500.tif",
                             np.full((H, W), 100.25 * r + 10.5 * c + 0.125, np.float32))
    ds = hp.open_tiled(hp.read_legacy_tiles(scan, tmp_path / "out", overlap=0, mirror_cols=False))
    assert ds["tile"].dtype == np.float32
    assert np.allclose(ds["tile"].values[1, 2], 100.25 + 21.0 + 0.125)             # not truncated to an integer
    assert (ds["seconds"].values == 0).all() and ds.attrs["times_recorded"] is False
    assert ds.attrs["start_time"] == "2024-08-30T12:40:30" and ds.attrs["start_time_source"] == "folder name"
    s = hp.stitch_run(tmp_path / "out", tmp_path / "stitched.zarr", camera={"rotation_deg": 0.0, "offset": 0.0},
                      geometry_path=hp.legacy_geometry(tmp_path / "g.json", tmp_path / "out") and tmp_path / "g.json",
                      flatfield_path=None)
    assert np.isfinite(s["image"].values).all()                                     # no tile blanked
    assert ds["row"].values.tolist() == [0.0, 1.0] and ds["col"].values.tolist() == [0.0, 1.0, 2.0]
    with pytest.raises(ValueError, match="stage"):
        hp.read_legacy_tiles(scan, tmp_path / "out2", overlap=0, positions="stage")
