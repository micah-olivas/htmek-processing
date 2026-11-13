"""record_raw_source: one row per acquisition, updated on a re-run, and a README listing them."""

import numpy as np
import pandas as pd
import xarray as xr

import htmek_processing as hp


def _acquisition(root, group, name, camera="kinetix", passes=3):
    src = root / "1_raw-images" / group / name
    (src / "tiled_images").mkdir(parents=True)
    (src / "tiled_images" / "chunk").write_bytes(b"\0" * 2000)
    ds = xr.Dataset({"tile": (("time", "row", "col", "y", "x"), np.zeros((passes, 2, 3, 1, 1)))})
    ds.attrs.update(setup="Setup7", filter="NADHTi", camera={"name": camera, "exposure_ms": 500.0},
                    start_time="2026-11-06T19:54:47.5")
    return src, ds


def test_rows_are_added_once_and_updated(tmp_path):
    rec = tmp_path / "exp" / "1_raw-images"
    a, ta = _acquisition(tmp_path, "Assays/NaK-series", "run_1xK")
    b, tb = _acquisition(tmp_path, "Standards/zyla", "NADH_standard_0uM_500ms_zyla", camera="zyla", passes=1)
    hp.record_raw_source(a, tmp_path / "a.zarr", ta, rec)
    hp.record_raw_source(b, tmp_path / "b.zarr", tb, rec)
    t = hp.record_raw_source(a, tmp_path / "a2.zarr", ta, rec)   # re-run: same row, new store path
    assert len(t) == 2
    row = t.set_index("acquisition").loc["run_1xK"]
    assert row["group"] == "Assays/NaK-series" and row["stitched_path"].endswith("a2.zarr")
    assert row["tiles"] == "2 x 3" and int(row["passes"]) == 3 and row["camera"] == "kinetix"
    assert float(row["size_gb"]) >= 0
    assert len(pd.read_csv(rec / "raw_sources.csv")) == 2


def test_readme_lists_every_acquisition_by_group(tmp_path):
    rec = tmp_path / "1_rec"
    a, ta = _acquisition(tmp_path, "GFP", "gfp_bg_zyla_500ms", camera="zyla", passes=1)
    b, tb = _acquisition(tmp_path, "Assays/Standard-MM-series", "NADH_kinetics_5mM")
    hp.record_raw_source(a, "a.zarr", ta, rec)
    hp.record_raw_source(b, "b.zarr", tb, rec)
    text = (rec / "README.md").read_text()
    assert "## GFP" in text and "## Assays/Standard-MM-series" in text
    assert "`gfp_bg_zyla_500ms`" in text and "`NADH_kinetics_5mM`" in text
    assert str(a) in text


def test_a_zarr_group_inside_the_acquisition_records_the_acquisition(tmp_path):
    rec = tmp_path / "rec"
    src, ds = _acquisition(tmp_path, "Assays/Standard-MM-series", "NADH_kinetics_5mM")
    (src / "tiled_images" / "tile").mkdir()
    t = hp.record_raw_source(src / "tiled_images", "s.zarr", ds, rec)
    row = t.iloc[0]
    assert row["acquisition"] == "NADH_kinetics_5mM" and row["group"] == "Assays/Standard-MM-series"
    assert row["raw_path"] == str(src)
