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
    tiles with a fixed overlap instead of measuring the seams. `overlap_px` is one number for both axes (0: butted, as
    the 2024 pipeline did) or (overlap_y, overlap_x): px shared at the row seams and at the column seams. Use it where
    the seams can't be measured: no overlap, or an overlap too small to register against a repeating chamber lattice
    (MBO-R02: 3 px at the rows and 11 px at the columns; a seam search there locks onto the 115 px row pitch). `src` is
    the converted store (read_legacy_tiles' return value), for the grid size. Returns the dict."""
    import json

    from .stitch import open_tiled

    tiled = open_tiled(src)
    R, C = tiled.sizes["row"], tiled.sizes["col"]
    oy, ox = (overlap_px, overlap_px) if np.isscalar(overlap_px) else overlap_px
    geometry = dict(overlap_y=[float(oy)] * (R - 1), overlap_x=[float(ox)] * (C - 1),
                    cross_dy=0.0, cross_dx=0.0, spread_y=[0.0] * (R - 1), spread_x=[0.0] * (C - 1),
                    measured_on=f"not measured: legacy scan, tiles overlapping {oy:g} px at the rows and {ox:g} px at the "
                                "columns (legacy_geometry)",
                    rotation_deg=rotation_deg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(geometry, indent=1))
    return geometry


def _lattice_phase(profile, coords, period):
    """The phase (px, in [0, period)) of a periodic profile sampled at `coords`: where its fundamental peaks."""
    p = np.asarray(profile, float) - np.mean(profile)
    z = (p * np.exp(-2j * np.pi * np.asarray(coords, float) / period)).sum()
    return (-np.angle(z) / (2 * np.pi) * period) % period


def lattice_period(profile, lo, hi, n=2001):
    """The period (px) of a periodic profile, searched continuously in [lo, hi]: the period whose fundamental has the most
    power (an FFT's bins are too coarse: 1024 / k px)."""
    p = np.asarray(profile, float) - np.mean(profile)
    x = np.arange(len(p))
    periods = np.linspace(lo, hi, n)
    power = [abs((p * np.exp(-2j * np.pi * x / per)).sum()) for per in periods]
    return float(periods[int(np.argmax(power))])


def lattice_tile_origins(src, *, pitch_x, pitch_y, overlap_px=0.0, frame=-1, strip=400, cross_strip=200):
    """Each tile's origin in the stitched image, from the chip's own chamber lattice: for tiles whose seams can't be
    registered directly (an overlap too small for phase correlation, or a repeating lattice the registration locks onto
    a pitch away; MBO-R02, Setup 7 2025: 2-16 px overlaps, a seam search found the 115 px row pitch).

    At every seam the lattice's phase is read on both tiles near the seam (`strip` px; the cross-seam phase on
    `cross_strip` px): along the seam's axis it gives the tiles' spacing modulo the pitch, taken within half a pitch of
    the nominal `overlap_px`; across it, the sideways shift, within half a pitch of 0. The 2 x (R (C - 1) + (R - 1) C)
    spacings are then solved for the R x C origins by least squares (tile (0, 0) at 0), refit once without seams off
    by more than 3 robust SDs. `pitch_x`, `pitch_y` (px) bound the lattice period search, each a (lo, hi) range or a
    number (searched +-3%); the chip's lattice must be visible on `frame` (a bright, uniformly filled scan: a standard).

    Returns a dict for stitch_to_zarr(overlap=...): tile_origin ((R, C, 2) list, (y0, x0) px, the smallest 0),
    overlap_y / overlap_x (the median per gap, for the stores' attrs), pitch, the seams' residuals (residual_px:
    median and max), and the seams measured (seams: column and row seams' along / cross values)."""
    from .stitch import open_tiled

    tiled = open_tiled(src)
    tiles = tiled["tile"].squeeze([d for d in tiled["tile"].dims[:-4] if tiled.sizes[d] == 1])
    if tiles.ndim == 5:
        tiles = tiles.isel({tiles.dims[0]: frame})
    out = tile_origins_from_lattice(np.asarray(tiles.values, float), pitch_x=pitch_x, pitch_y=pitch_y,
                                    overlap_px=overlap_px, strip=strip, cross_strip=cross_strip)
    out["measured_on"] = f"{Path(src).name}, frame {frame}: tiles placed on the chamber lattice (lattice_tile_origins)"
    return out


def tile_origins_from_lattice(T, *, pitch_x, pitch_y, overlap_px=0.0, strip=400, cross_strip=200, rounds=2):
    """lattice_tile_origins on a tile array `T` (row, col, y, x) in memory; see there. The lattice pitch is first
    read on the tiles (a few periods each), then on the mosaic the first placement gives (`rounds` - 1 times): the
    seams' phases sit about a pitch apart, so a pitch error enters every seam once."""
    from .stitch import place_tiles

    T = np.asarray(T, float)
    R, C, h, w = T.shape
    rng = lambda p: (p[0], p[1]) if np.ndim(p) else (p * 0.97, p * 1.03)
    px = np.median([lattice_period(T[r, c].mean(0), *rng(pitch_x)) for r in range(R) for c in range(C)])
    py = np.median([lattice_period(T[r, c].mean(1), *rng(pitch_y)) for r in range(R) for c in range(C)])
    for k in range(rounds):
        out = _origins_one_pass(T, px, py, overlap_px, strip, cross_strip)
        if k < rounds - 1:
            o = np.asarray(out["tile_origin"])
            mosaic = place_tiles(T.astype(np.float32), o, (int(o[..., 0].max() + h), int(o[..., 1].max() + w)))
            mosaic = np.nan_to_num(mosaic, nan=float(np.nanmedian(mosaic)))
            px = lattice_period(mosaic.mean(0), px * 0.99, px * 1.01, n=4001)
            py = lattice_period(mosaic.mean(1), py * 0.99, py * 1.01, n=4001)
    return out


def _origins_one_pass(T, px, py, overlap_px, strip, cross_strip):
    """One placement of the tiles at a given lattice pitch (px, py): the seams' phases, then least squares."""
    R, C, h, w = T.shape
    wrap = lambda v, per, centre: (v - centre + per / 2) % per - per / 2 + centre
    xs, ys = np.arange(w), np.arange(h)
    eqs = {"x": [], "y": []}   # (tile a, tile b, measured origin_b - origin_a)
    seams = {"columns": np.full((R, C - 1, 2), np.nan), "rows": np.full((R - 1, C, 2), np.nan)}
    for r in range(R):
        for c in range(C):
            a = T[r, c]
            if c < C - 1:   # column seam: x spacing (along), y shift (across)
                b = T[r, c + 1]
                # phases read from the seam (a: x - w, b: x), so a slightly wrong pitch acts over the strip only
                fa = _lattice_phase(a[:, w - strip:].mean(0), xs[w - strip:] - w, px)
                fb = _lattice_phase(b[:, :strip].mean(0), xs[:strip], px)
                ov = wrap(fb - fa, px, overlap_px)              # px shared: origin_b - origin_a = w - ov
                ga = _lattice_phase(a[:, w - cross_strip:].mean(1), ys, py)
                gb = _lattice_phase(b[:, :cross_strip].mean(1), ys, py)
                d = wrap(ga - gb, py, 0.0)                     # origin_b - origin_a in y
                eqs["x"].append(((r, c), (r, c + 1), w - ov))
                eqs["y"].append(((r, c), (r, c + 1), d))
                seams["columns"][r, c] = (ov, d)
            if r < R - 1:   # row seam: y spacing (along), x shift (across)
                b = T[r + 1, c]
                fa = _lattice_phase(a[h - strip:].mean(1), ys[h - strip:] - h, py)
                fb = _lattice_phase(b[:strip].mean(1), ys[:strip], py)
                ov = wrap(fb - fa, py, overlap_px)
                ga = _lattice_phase(a[h - cross_strip:].mean(0), xs, px)
                gb = _lattice_phase(b[:cross_strip].mean(0), xs, px)
                d = wrap(ga - gb, px, 0.0)
                eqs["y"].append(((r, c), (r + 1, c), h - ov))
                eqs["x"].append(((r, c), (r + 1, c), d))
                seams["rows"][r, c] = (ov, d)
    idx = lambda rc: rc[0] * C + rc[1]
    origin, resid_all = np.zeros((R, C, 2)), []
    for k, axis in enumerate(("y", "x")):
        rows = eqs[axis]
        A = np.zeros((len(rows) + 1, R * C))
        y = np.zeros(len(rows) + 1)
        for i, (ta, tb, v) in enumerate(rows):
            A[i, idx(tb)], A[i, idx(ta)], y[i] = 1, -1, v
        A[-1, 0] = 1   # tile (0, 0) at 0
        keep = np.ones(len(y), bool)
        for _ in range(2):
            sol, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
            res = A @ sol - y
            mad = 1.4826 * np.median(np.abs(res[:-1] - np.median(res[:-1]))) or 1.0
            keep = np.r_[np.abs(res[:-1]) <= max(3 * mad, 1.0), True]
        origin[..., k] = sol.reshape(R, C)
        resid_all.append(np.abs(res[:-1]))
    origin -= origin.reshape(-1, 2).min(0)
    origin = np.rint(origin)
    resid = np.concatenate(resid_all)
    return dict(tile_origin=origin.astype(int).tolist(),
                overlap_y=(h - np.median(np.diff(origin[..., 0], axis=0), axis=1)).tolist(),
                overlap_x=(w - np.median(np.diff(origin[..., 1], axis=1), axis=0)).tolist(),
                cross_dy=0.0, cross_dx=0.0, pitch=[float(py), float(px)],
                residual_px=dict(median=float(np.median(resid)), max=float(resid.max())),
                seams={k: v.round(2).tolist() for k, v in seams.items()},
                measured_on="tile array: tiles placed on the chamber lattice (tile_origins_from_lattice)",
                rotation_deg=0.0)
