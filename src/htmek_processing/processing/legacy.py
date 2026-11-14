"""Legacy tiled TIFF scans (the 2024 acquisition software, e.g. Setup 3) -> the tiled zarr layout open_tiled reads.

A legacy scan is a folder of tiles named `<n>-Pos<CCC>_<RRR>_<exposure>.tif` (CCC the column, RRR the row of the
tile grid), each with an ImageDescription tag `{Channel: ..., Index: ..., Pos:(x, y)}` (stage position, µm) and a
DateTime tag (to the second). A kinetic run is a folder of such folders, one per pass. read_legacy_tiles writes the
same layout as a current acquisition (`tile`, `seconds`, `row` / `col` coordinates, `overlap` attr), so stitch_run and
everything after it work unchanged.
"""
import re
from datetime import datetime
from pathlib import Path

import numpy as np

_TILE = re.compile(r"Pos(\d+)_(\d+)")
_POS = re.compile(r"Pos:\(\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*\)")
_CHANNEL = re.compile(r"Channel:\s*([^,}]+)")
_FOLDER_TIME = re.compile(r"(\d{8})-(\d{6})")   # the acquisition software names each scan folder YYYYMMDD-HHMMSS_...


def _tiles_in(folder):
    """{(row, col): path} for the tiles in one folder (macOS `._` sidecar files skipped)."""
    out = {}
    for p in Path(folder).glob("*.tif"):
        m = _TILE.search(p.name)
        if m and not p.name.startswith("._"):
            out[(int(m.group(2)), int(m.group(1)))] = p
    return out


def _meta(path):
    """Channel, stage position (x, y), acquisition time and exposure (ms) of one legacy tile."""
    import tifffile

    with tifffile.TiffFile(path) as t:
        tags = t.pages[0].tags
        desc = str(tags["ImageDescription"].value) if "ImageDescription" in tags else ""
        when = tags["DateTime"].value if "DateTime" in tags else None
        exposure = tags["ExposureTime"].value if "ExposureTime" in tags else None
    if isinstance(exposure, tuple):   # a TIFF RATIONAL read as (numerator, denominator)
        exposure = exposure[0] / exposure[1]
    pos = _POS.search(desc)
    channel = _CHANNEL.search(desc)
    return dict(channel=channel.group(1).strip() if channel else None,
                stage=(float(pos.group(1)), float(pos.group(2))) if pos else (np.nan, np.nan),
                time=datetime.strptime(when, "%Y-%m-%d %H:%M:%S") if when else None,
                exposure_ms=float(exposure) * 1000 if exposure is not None else None)


def legacy_passes(src):
    """The pass folders of a legacy scan, in acquisition order: `src` itself if it holds tiles (a snapshot), else its
    subfolders that hold tiles (a kinetic run), sorted by name (they start with the pass time); stitched-image folders
    are left out."""
    src = Path(src)
    if _tiles_in(src):
        return [src]
    passes = sorted(d for d in src.iterdir()
                    if d.is_dir() and not d.name.startswith(".") and "stitched" not in d.name.lower() and _tiles_in(d))
    if not passes:
        raise FileNotFoundError(f"no legacy tiles (*Pos<col>_<row>*.tif) in {src} or its subfolders")
    return passes


def read_legacy_tiles(src, dst, *, overlap, positions="filename", mirror_cols=True, group="tiled_images", overwrite=False):
    """Convert a legacy tiled TIFF scan to the tiled zarr layout of a current acquisition; returns the folder to pass
    to stitch_run / open_tiled as `src`.

    `src` is a pass folder of tiles (a snapshot: `tile` is (row, col, y, x)) or a run folder of pass folders (a
    timecourse: `tile` is (time, row, col, y, x)). Tiles are placed by their file name (`positions="filename"`, the
    default) or by the stage position in their ImageDescription (`positions="stage"`). Use the file name for scans whose
    tiles were renamed to repair their labels (MBO-R01, 2024-08-30: the images lag the recorded stage positions by one
    tile, and the renamed "Corrected-labels" files carry the true positions); the recorded stage positions are kept as
    `stage_x` / `stage_y` either way.

    With `mirror_cols` (default), grid column 0 is the tile with the highest column number in its file name, which
    is how the 2024 pipeline stitched these scans: the camera's x runs opposite to the stage's (MBO-R01: the converted
    and stitched pass is pixel-identical to the 2024 StitchedImg, 2026-11-13). Check a new setup against one of its
    stitched images before relying on it.

    `overlap` (the fraction of a tile shared with its neighbor) is not recorded in legacy files, so it is required. The
    2024 Setup 3 scans step about one field of view (no measurable overlap): pass `overlap=0` and write their seam
    geometry with legacy_geometry, which stitch_run then reads instead of measuring (measure_tile_overlaps needs an
    overlap to register). `seconds` is each tile's DateTime tag (1 s resolution) minus the first tile's.

    Raw tiles (integers) are written as uint16. Tiles stored as floats (processed copies, e.g. MBO-R01's "_rebinned"
    GFP scans) are kept as float32, never cast to integers. Such copies often lack the TIFF tags: a snapshot with no
    tile times gets `seconds` 0 for every tile (attr times_recorded=False; NaN would mark its tiles as never acquired,
    and stitch_to_zarr would blank them), and a timecourse without them raises (its kinetics need the times); with no
    stage positions the `row` / `col` coordinates are the grid indices. Without tile times, `start_time` is read from the
    pass folder's name (YYYYMMDD-HHMMSS, the scan's start; attr start_time_source="folder name").

    Written to `dst/<group>`; an existing store is reused unless `overwrite`."""
    import dask
    import dask.array as da
    import tifffile
    import xarray as xr

    if overlap is None or overlap < 0 or overlap >= 1:
        raise ValueError("overlap must be a fraction in [0, 1): legacy files don't record it (0 for tiles that abut, "
                         "with legacy_geometry)")
    if positions not in ("filename", "stage"):
        raise ValueError(f"positions must be 'filename' or 'stage', got {positions!r}")
    out = Path(dst) / group
    if out.exists() and not overwrite:
        return Path(dst)

    passes = legacy_passes(src)
    snapshot = len(passes) == 1 and passes[0] == Path(src)
    tiles = [_tiles_in(p) for p in passes]
    meta = [{k: _meta(path) for k, path in t.items()} for t in tiles]

    if positions == "stage":
        # rank each pass's stage x and y positions into the grid's columns and rows
        def by_stage(t, m):
            xs = np.unique(np.round([v["stage"][0] for v in m.values()], 1))
            ys = np.unique(np.round([v["stage"][1] for v in m.values()], 1))
            key = lambda k: (int(np.searchsorted(ys, round(m[k]["stage"][1], 1))),
                             int(np.searchsorted(xs, round(m[k]["stage"][0], 1))))
            return {key(k): t[k] for k in t}, {key(k): m[k] for k in m}
        if any(np.isnan(v["stage"]).any() for m in meta for v in m.values()):
            raise ValueError("positions='stage' needs every tile's stage position (ImageDescription Pos); use 'filename'")
        tiles, meta = map(list, zip(*(by_stage(t, m) for t, m in zip(tiles, meta))))

    if mirror_cols:
        n_cols = max(k[1] for k in tiles[0]) + 1
        tiles = [{(r, n_cols - 1 - c): v for (r, c), v in t.items()} for t in tiles]
        meta = [{(r, n_cols - 1 - c): v for (r, c), v in m.items()} for m in meta]
    keys = set(tiles[0])
    for p, t in zip(passes, tiles):
        if set(t) != keys:
            raise ValueError(f"{p.name}: tiles {sorted(set(t) ^ keys)} differ from the first pass's grid")
    R, C = max(k[0] for k in keys) + 1, max(k[1] for k in keys) + 1
    if len(keys) != R * C:
        raise ValueError(f"tile grid is incomplete: {len(keys)} tiles for a {R} x {C} grid")

    first = tifffile.imread(tiles[0][(0, 0)])
    h, w = first.shape
    dtype = np.uint16 if np.issubdtype(first.dtype, np.integer) else np.float32   # never truncate processed float tiles
    stage = np.array([[meta[0][(r, c)]["stage"] for c in range(C)] for r in range(R)])   # (R, C, 2), µm
    times = np.array([[[m[(r, c)]["time"] for c in range(C)] for r in range(R)] for m in meta])
    known = [t for t in times.ravel() if t is not None]
    start = min(known) if known else None
    times_recorded = all(t is not None for t in times.ravel())
    start_source = "tiles" if start is not None else ""
    if start is None:   # processed copies drop the tags; the scan folder's name still holds its start
        m = _FOLDER_TIME.search(passes[0].name)
        if m:
            start, start_source = datetime.strptime("".join(m.groups()), "%Y%m%d%H%M%S"), "folder name"
    if not times_recorded and not snapshot:
        raise ValueError(f"{src}: tiles without a DateTime tag in a timecourse; its kinetics need every tile's time")
    seconds = (np.vectorize(lambda t: (t - start).total_seconds(), otypes=[float])(times) if times_recorded
               else np.zeros(times.shape))   # a snapshot without times: 0, not NaN (NaN marks a tile never acquired)

    read = dask.delayed(lambda p: tifffile.imread(p).astype(dtype))
    stack = da.stack([da.stack([da.stack([da.from_delayed(read(t[(r, c)]), (h, w), dtype) for c in range(C)])
                                for r in range(R)]) for t in tiles])            # (time, row, col, y, x)
    channel = meta[0][(0, 0)]["channel"]
    exposure = meta[0][(0, 0)]["exposure_ms"]
    if np.isnan(stage).all():   # no stage positions recorded: grid indices
        coords = dict(row=np.arange(R, dtype=float), col=np.arange(C, dtype=float))
    else:
        coords = dict(row=np.nanmedian(stage[:, :, 1], axis=1), col=np.nanmedian(stage[:, :, 0], axis=0))
    dims = ("time", "row", "col")
    ds = xr.Dataset(
        dict(tile=(dims + ("y", "x"), stack), seconds=(dims, seconds),
             stage_x=(("row", "col"), stage[:, :, 0]), stage_y=(("row", "col"), stage[:, :, 1])),
        coords=coords,
        attrs=dict(overlap=float(overlap), start_time=start.isoformat() if start else "", channel=channel or "",
                   exposure_ms=exposure if exposure is not None else "", legacy=True, positions=positions, mirror_cols=bool(mirror_cols), times_recorded=times_recorded, start_time_source=start_source,
                   source=str(Path(src)), passes=[p.name for p in passes],
                   info=f"legacy tiled TIFF scan, {len(passes)} pass(es) of {R} x {C} tiles, {channel} {exposure:g} ms"
                   if exposure is not None else f"legacy tiled TIFF scan, {len(passes)} pass(es) of {R} x {C} tiles"))
    if snapshot:
        ds = ds.isel(time=0, drop=True)
        ds["timestamp"] = (("row", "col"), np.array([[np.datetime64(t, "ns") if t is not None else np.datetime64("NaT", "ns")
                                                      for t in row] for row in times[0]]))
    ds["tile"].encoding["chunks"] = (1,) * (ds["tile"].ndim - 2) + (h, w)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(out, mode="w", consolidated=False)
    return Path(dst)


def legacy_geometry(path, src, *, overlap_px=0, rotation_deg=0.0):
    """Write the seam geometry of a legacy scan to `path` (the JSON stitch_geometry reads), so stitch_run places its
    tiles `overlap_px` apart (0: butted, as the 2024 pipeline did) instead of measuring the seams, which needs an
    overlap. `src` is the converted store (read_legacy_tiles' return value), for the grid size. Returns the dict."""
    import json

    from .stitch import open_tiled

    tiled = open_tiled(src)
    R, C = tiled.sizes["row"], tiled.sizes["col"]
    geometry = dict(overlap_y=[float(overlap_px)] * (R - 1), overlap_x=[float(overlap_px)] * (C - 1),
                    cross_dy=0.0, cross_dx=0.0, spread_y=[0.0] * (R - 1), spread_x=[0.0] * (C - 1),
                    measured_on=f"not measured: legacy scan, tiles placed {overlap_px} px apart (legacy_geometry)",
                    rotation_deg=rotation_deg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(geometry, indent=1))
    return geometry
