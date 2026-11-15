"""Tiled acquisitions: open a tile scan, flatfield-correct and stitch it, and time each
chamber; flatfield estimation and correction."""

from __future__ import annotations

from pathlib import Path

import os

import numpy as np

from .io import load_image


# A large chip is imaged as a grid of overlapping camera tiles, stored by Prismo
# as a zarr group (named 'tiled_images', or after the channel, e.g. 'egfp_500ms'):
#     tile      (..., row, col, y, x)   raw camera frames; leading dims such as time
#     seconds   (..., row, col)         elapsed time each tile was snapped
#     attrs     overlap                 fractional tile overlap (e.g. 0.1)
# A kinetic run leads with `time`; a snapshot has no leading dims (or only size-1 ones, such
# as a single `exposure`). Tiles are snapped one after another, so tiles in one pass are
# imaged at different times; chamber_seconds() recovers each chamber's own time.


def open_tiled(path, channel=None):
    """Open a tiled acquisition as a lazy xarray Dataset (nothing is read until indexed).

    `path` is the zarr group holding 'tile', or the acquisition folder containing it --
    the group's name varies ('tiled_images', 'egfp_500ms', ...), so it is found by looking
    for the one subfolder with a 'tile' array.

    An acquisition of several channels at every tile (a 'channel' dim, e.g. prismo's
    tiled_kinetics: "cy5", "gfp") needs `channel`, which selects one. Passes a timecourse
    never acquired (NaN 'pass_start_s': the run stopped before its last planned pass) are
    dropped, so they are never stitched."""
    import xarray as xr

    path = Path(path)
    if not (path / "tile").is_dir():
        groups = [g for g in path.iterdir() if (g / "tile").is_dir()]
        if len(groups) != 1:
            raise FileNotFoundError(
                f"expected one tiled group (a folder with a 'tile' array) in {path}, "
                f"found {[g.name for g in groups]}"
            )
        path = groups[0]
    ds = xr.open_zarr(path, consolidated=False)
    if "channel" in ds.dims:
        if channel is None:
            raise ValueError(f"{path} holds channels {list(ds['channel'].values)}: pass channel=")
        ds = ds.sel(channel=channel, drop=True)
    elif channel is not None:
        raise ValueError(f"{path} has no channel dim; got channel={channel!r}")
    if "pass_start_s" in ds and "time" in ds["pass_start_s"].dims:
        acquired = np.isfinite(ds["pass_start_s"].values)
        if not acquired.all():
            ds = ds.isel(time=np.flatnonzero(acquired))
    return ds


def overlap_px(tiled, overlap=None):
    """Overlap in pixels between neighboring tiles: `overlap` (a fraction) or the dataset's
    'overlap' attr, times the tile edge. Read from the data, never hardcoded, because the
    tile size differs between cameras."""
    frac = overlap if overlap is not None else tiled.attrs["overlap"]
    return int(round(frac * tiled.sizes["x"]))


def _gap_trims(overlap, n, split=False):
    """How many px each of n tiles along one axis loses: `overlap` is one value for every gap,
    or one per gap (n - 1 values), rounded to whole px.

    split=False: each tile loses its trailing overlap, the last tile the mean overlap
    (magnify's convention). Returns (lead, trail), lead all 0.
    split=True: each overlap is split between the two tiles that share it, and the outer
    edges lose half the mean overlap, so every tile keeps its middle and loses both edges
    (where a rotated tile has no data)."""
    gaps = np.rint(np.broadcast_to(np.asarray(overlap, float), (max(n - 1, 0),))).astype(int)
    mean = int(np.rint(gaps.mean())) if len(gaps) else int(np.rint(np.mean(overlap)))
    if not split:
        return np.zeros(n, int), np.r_[gaps, mean]
    lead = np.r_[mean // 2, gaps - gaps // 2]
    trail = np.r_[gaps // 2, mean - mean // 2]
    return lead, trail


def stitch_tiles(tiles, overlap, split=False):
    """Stitch a grid of tiles into one image by trimming each tile's trailing overlap and
    butting the tiles together (magnify's stitch convention).

    `overlap` is one number for both axes, or (overlap_y, overlap_x), each one number or one
    per gap between neighboring tiles (rows - 1 / cols - 1 values, e.g. from
    measure_tile_overlaps, when the stage steps are uneven). Overlaps are rounded to whole px.
    With `split`, each overlap is shared between its two tiles instead of cut from the first,
    so every tile loses both edges (see _gap_trims): use it for rotated tiles.

    `tiles` is (row, col, y, x), or (time, row, col, y, x) to stitch every frame at once.
    Returns (Y, X) or (time, Y, X). No interpolation, so pixel values are untouched."""
    tiles = np.asarray(tiles)
    *lead, R, C, h, w = tiles.shape
    oy, ox = overlap if isinstance(overlap, (tuple, list)) and len(overlap) == 2 else (overlap, overlap)
    (ly, ty), (lx, tx) = _gap_trims(oy or 0, R, split), _gap_trims(ox or 0, C, split)
    if not split and len(set(ty)) == 1 and len(set(tx)) == 1:  # pure reshaping
        tiles = tiles[..., : h - ty[0], : w - tx[0]]
        moved = np.moveaxis(tiles, -2, -3)  # (..., row, y, col, x): rows of tiles -> pixels
        return moved.reshape(*lead, R * (h - ty[0]), C * (w - tx[0]))
    rows = [np.concatenate([tiles[..., r, c, ly[r]: h - ty[r], lx[c]: w - tx[c]]
                            for c in range(C)], axis=-1) for r in range(R)]
    return np.concatenate(rows, axis=-2)


def measure_tile_overlaps(src, *, rotation_deg=0.0, frame=0, overlap=None, band=300, smooth=20, channel=None):
    """The overlap between neighboring tiles of a tiled acquisition, gap by gap, measured on
    the chip itself: each pair of neighbors' overlap strips (trailing strip of one, leading
    strip of the next, at the nominal overlap) are registered by phase correlation, after
    rotating the tiles by `rotation_deg` (as stitch_to_zarr does) and removing the smooth
    illumination (Gaussian `smooth` px), `band` px in from the strip ends.

    Stage steps can be uneven (MBO-041's Kinetix scans: 12-21 px between gaps), but they
    repeat from row to row and pass to pass, so one frame measures them. Returns a dict:
      overlap_y / overlap_x   per row gap / column gap (px), the median over the other axis;
      cross_dy / cross_dx     the leftover offset across the seams (px): sideways for
                              left-right neighbors (dy) and up-down ones (dx); near 0 when the
                              rotation is right;
      spread_y / spread_x     the SD over the other axis (px), how repeatable each gap is.
    `channel` picks one channel of a multi-channel acquisition (open_tiled)."""
    from concurrent.futures import ThreadPoolExecutor

    from scipy import ndimage
    from skimage.registration import phase_cross_correlation

    tiled = open_tiled(src, channel)
    ov = overlap_px(tiled, overlap)
    tiles = tiled["tile"].squeeze([d for d in tiled["tile"].dims[:-4] if tiled.sizes[d] == 1])
    if tiles.ndim == 5:
        tiles = tiles.isel({tiles.dims[0]: frame})
    raw = tiles.values.astype(np.float32)
    R, C, h, w = raw.shape

    def prep(t):
        if rotation_deg:
            t = ndimage.rotate(t, rotation_deg, reshape=False, order=1, mode="nearest")
        return t - ndimage.gaussian_filter(t, smooth)

    with ThreadPoolExecutor() as ex:
        t = np.array(list(ex.map(prep, raw.reshape(R * C, h, w)))).reshape(R, C, h, w)
    across = np.full((R, C - 1, 2), np.nan)  # left-right neighbors: (dy, dx)
    down = np.full((R - 1, C, 2), np.nan)    # up-down neighbors: (dy, dx)
    for r in range(R):
        for c in range(C):
            if c < C - 1:
                across[r, c] = phase_cross_correlation(
                    t[r, c][band:h - band, w - ov:], t[r, c + 1][band:h - band, :ov],
                    upsample_factor=10)[0]
            if r < R - 1:
                down[r, c] = phase_cross_correlation(
                    t[r, c][h - ov:, band:w - band], t[r + 1, c][:ov, band:w - band],
                    upsample_factor=10)[0]
    # a measured shift s means the next tile starts s px later than the nominal overlap puts it
    return dict(
        overlap_y=(ov - np.median(down[..., 0], axis=1)).tolist(),
        overlap_x=(ov - np.median(across[..., 1], axis=0)).tolist(),
        cross_dy=float(np.median(across[..., 0])), cross_dx=float(np.median(down[..., 1])),
        spread_y=down[..., 0].std(axis=1).tolist(), spread_x=across[..., 1].std(axis=0).tolist(),
    )


def _stitch_frame(job):
    """Stitch one frame of an acquisition into its slot of an existing zarr array (stitch_to_zarr's worker):
    offset, flatfield or background, rotation, then the tiles butted together."""
    import zarr

    tiled = open_tiled(job["src"], job["channel"])
    singleton = [d for d in tiled["tile"].dims[:-4] if tiled.sizes[d] == 1]
    tiles = tiled["tile"].squeeze(singleton)
    seconds = tiled["seconds"].squeeze([d for d in singleton if d in tiled["seconds"].dims])
    lead, idx, dtype = job["lead"], job["idx"], np.dtype(job["dtype"])
    offset, flatfield, bg_tiles = job["offset"], job["flatfield"], job["bg_tiles"]
    frame = tiles[dict(zip(lead, idx))].values  # (row, col, y, x)
    if offset is not None and np.any(offset):   # a scalar, or a per-pixel dark image (one camera tile)
        frame = frame.astype(np.float32) - np.asarray(offset, np.float32)
    if flatfield is not None:  # per tile, before the overlap is trimmed
        frame = flatfield_correct(frame, flatfield, darkfield=job["darkfield"], background=bg_tiles).astype(dtype)
    elif bg_tiles is not None:  # background only, no flatfield (none available yet)
        frame = (frame.astype(np.float32) - bg_tiles).astype(dtype)
    if job["rotation_deg"]:  # square each tile to the stage axes before butting them together
        from scipy.ndimage import rotate

        frame = np.stack([
            np.stack([rotate(t.astype(np.float32), job["rotation_deg"], reshape=False, order=1, mode="nearest") for t in row])
            for row in frame
        ])
        frame = (frame if np.issubdtype(dtype, np.floating) else np.rint(frame)).astype(dtype)
    # tiles of a pass the run stopped in the middle of were never acquired (no time):
    # NaN in a float store, not a dark tile
    times = seconds[dict(zip(lead, idx))].values if lead else seconds.values
    missing = ~np.isfinite(np.asarray(times, float))
    if missing.any() and np.issubdtype(dtype, np.floating):
        frame = frame.astype(dtype, copy=True)
        frame[missing] = np.nan
    zarr.open_array(job["dst"], mode="r+")[idx] = stitch_tiles(frame, job["ov"], split=job["split"])


def _offset_value(offset):
    """The camera offset as a scalar, or a per-pixel dark image (a 2-D array, or a path to one)."""
    if offset is None:
        return 0.0
    if isinstance(offset, (str, Path)):
        from .io import load_image

        return load_image(offset).astype(np.float32)
    a = np.asarray(offset, np.float32)
    return float(a) if a.ndim == 0 else a


def stitch_to_zarr(
    src,
    dst,
    *,
    overlap=None,
    flatfield=None,
    darkfield=0.0,
    background=None,
    background_shift=(0.0, 0.0),
    rotation_deg=0.0,
    offset=0.0,
    overwrite=False,
    channel=None,
    n_jobs=None,
):
    """Stitch a tiled acquisition and save it as a zarr store at `dst`.

    The store holds 'image' -- (time, y, x) for a kinetic run, (y, x) for a snapshot --
    chunked one camera tile per chunk, plus the per-tile 'seconds' and the stitched tile
    pitch (attr 'tile_px') that chamber_seconds() needs. Size-1 leading dims (e.g. a single
    `exposure`) are dropped. One frame (all its tiles) is in memory at a time.

    With `flatfield` (a path or array the size of one camera tile), every tile is
    flatfield-corrected BEFORE stitching -- illumination falls off toward each tile's edge,
    so it has to be undone per tile, not on the mosaic -- via flatfield_correct with
    `darkfield` (camera offset: scalar, array or path). The image is then float32, and the
    flatfield and darkfield are recorded in the store's attrs.

    With `background` (a tile scan of the same stage grid without the signal, e.g. the chip
    before expression), each background tile is subtracted from its tile -- before the
    flatfield division if there is one (see flatfield_correct), or on its own if not. `background_shift` (dy, dx) in pixels moves
    every background tile onto the signal tiles first, if the chip moved slightly between
    the scans (measure it with tile_shift).

    `overlap` is the tiles' overlap: a fraction of the tile (default: the acquisition's own
    'overlap' attr), (overlap_y, overlap_x) in px, each one value or one per gap, or the dict
    measure_tile_overlaps returns, when the stage's steps are uneven. The store records each
    tile row's and column's origin in the stitched image (attrs tile_y0, tile_x0) and where
    the tiles meet (attrs seams_y, seams_x).

    `offset` (RFU) is the camera's offset, subtracted from every tile first: a scalar, or a per-pixel dark
    image the size of one camera tile (array or path; e.g. MBO-053's measured darks). Use it with a
    flatfield scaled to mean 1 (e.g. BaSiC's), which has no offset of its own. (`darkfield` is
    the offset inside a raw flatfield image, see flatfield_correct.)

    With `rotation_deg`, every tile is rotated by that angle about its center (scipy.ndimage
    .rotate's convention, bilinear) before stitching: for a camera mounted slightly rotated
    relative to the stage axes, whose tiles then don't butt together (MBO-041: the Kinetix,
    -1.94 deg). The angle is recorded in the store's attrs; with rotation alone the image
    keeps the camera's integer type (interpolated values rounded), which halves the store.

    Returns the stitched store as a lazy xarray Dataset. If `dst` already holds a finished
    stitch it is reopened instead of rebuilt, unless `overwrite=True`. A store left
    half-written by an interrupted run is marked unfinished and is rebuilt. `channel` picks
    one channel of a multi-channel acquisition (open_tiled). Tiles never acquired (no time
    in 'seconds', e.g. the rest of a pass the run stopped in) are NaN in a float store. Frames
    are stitched in parallel across `n_jobs` processes (default half the cores, at most 6; each
    holds one frame, ~2 GB for 25 Kinetix tiles); 1 = serial."""
    import dask.array as da
    import xarray as xr
    import zarr

    dst = Path(dst)
    if dst.exists() and not overwrite:
        done = xr.open_zarr(dst, consolidated=False)
        # stores written before this marker existed have no attr: trust those
        if done.attrs.get("stitch_complete", True):
            return done

    dst.parent.mkdir(parents=True, exist_ok=True)
    tiled = open_tiled(src, channel)
    singleton = [d for d in tiled["tile"].dims[:-4] if tiled.sizes[d] == 1]
    tiles = tiled["tile"].squeeze(singleton)
    seconds = tiled["seconds"].squeeze(
        [d for d in singleton if d in tiled["seconds"].dims]
    )

    lead = tiles.dims[:-4]  # e.g. ("time",) for kinetics, () for a snapshot
    lead_shape = tuple(tiled.sizes[d] for d in lead)
    R, C = tiled.sizes["row"], tiled.sizes["col"]
    if isinstance(overlap, dict):  # measure_tile_overlaps' result: per-gap overlaps
        ov = (overlap["overlap_y"], overlap["overlap_x"])
    elif isinstance(overlap, (tuple, list)):
        ov = tuple(overlap)
    else:
        ov = overlap_px(tiled, overlap)
    oy, ox = ov if isinstance(ov, tuple) else (ov, ov)
    split = bool(rotation_deg)  # rotated tiles have no data at their edges: drop both edges
    (lead_y, trail_y), (lead_x, trail_x) = _gap_trims(oy, R, split), _gap_trims(ox, C, split)
    kept_y = tiled.sizes["y"] - lead_y - trail_y
    kept_x = tiled.sizes["x"] - lead_x - trail_x
    # where each tile's own pixel (0, 0) lands in the stitched image
    tile_y0 = np.r_[0, np.cumsum(kept_y)[:-1]] - lead_y
    tile_x0 = np.r_[0, np.cumsum(kept_x)[:-1]] - lead_x
    height, width = int(kept_y.sum()), int(kept_x.sum())
    step = int(round(height / R)), int(round(width / C))  # mean stitched tile pitch (y, x)

    # lay out the store (metadata only, nothing computed), then fill it frame by frame
    offset_src = offset if isinstance(offset, (str, Path)) else None
    offset = _offset_value(offset)
    corrected = flatfield is not None or background is not None or bool(np.any(offset))
    dtype = np.float32 if corrected else tiles.dtype
    image = xr.DataArray(
        da.zeros(
            lead_shape + (height, width),
            dtype=dtype,
            chunks=(1,) * len(lead) + step,
        ),
        dims=lead + ("y", "x"),
    )
    ds = xr.Dataset({"image": image, "seconds": seconds.load()})
    ds.attrs.update(tiled.attrs)
    ds.attrs.update(tile_px=list(step), source=str(src),
                    overlap_px=ov if np.isscalar(ov) else [np.asarray(v, float).tolist() for v in ov],
                    tile_y0=tile_y0.tolist(), tile_x0=tile_x0.tolist(),
                    seams_y=np.cumsum(kept_y)[:-1].tolist(),  # where tile rows meet (px)
                    seams_x=np.cumsum(kept_x)[:-1].tolist())  # where tile columns meet
    if flatfield is not None:
        ds.attrs.update(
            flatfield=str(flatfield) if isinstance(flatfield, (str, Path)) else "array",
            darkfield=str(darkfield),
        )
    if background is not None:
        ds.attrs.update(
            background=str(background), background_shift=list(background_shift)
        )
    if rotation_deg:
        ds.attrs["rotation_deg"] = float(rotation_deg)
    if np.ndim(offset) == 0 and offset:
        ds.attrs["offset"] = float(offset)
    elif np.ndim(offset) == 2:   # a dark image: its median, and where it came from
        ds.attrs["offset"] = float(np.median(offset))
        ds.attrs["offset_image"] = str(offset_src) if offset_src is not None else "array"
    ds.attrs["stitch_complete"] = False  # flipped only once every frame is written
    ds.to_zarr(dst, mode="w", compute=False, consolidated=False)

    # background tiles (row, col, y, x), moved onto the signal tiles if the chip shifted
    bg_tiles = None
    if background is not None:
        from scipy.ndimage import shift as nd_shift

        bg = open_tiled(background)
        bg_tiles = bg["tile"].squeeze(
            [d for d in bg["tile"].dims[:-4] if bg.sizes[d] == 1]
        )
        bg_tiles = bg_tiles.values.astype(np.float32)
        if any(background_shift):
            bg_tiles = nd_shift(
                bg_tiles, (0, 0, *background_shift), order=1, mode="nearest"
            )

    job = dict(src=str(src), channel=channel, lead=lead, offset=offset, flatfield=flatfield, darkfield=darkfield,
               bg_tiles=bg_tiles, rotation_deg=rotation_deg, dtype=np.dtype(dtype).str, ov=ov, split=split,
               dst=str(dst / "image"))
    frames = list(np.ndindex(*lead_shape))   # a single empty index for a snapshot
    n_jobs = min(n_jobs or max(1, min(6, (os.cpu_count() or 2) // 2)), len(frames))
    if n_jobs > 1:   # frames are independent and fill separate chunks: one worker process per frame
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp

        from ..workers import headless_workers

        with headless_workers(), ProcessPoolExecutor(n_jobs, mp_context=mp.get_context("spawn")) as ex:
            list(ex.map(_stitch_frame, [dict(job, idx=idx) for idx in frames]))
    else:
        for idx in frames:
            _stitch_frame(dict(job, idx=idx))
    zarr.open_group(dst, mode="r+").attrs["stitch_complete"] = True
    return xr.open_zarr(dst, consolidated=False)


def stitch_geometry(src, path, *, rotation_deg, frame=-1, max_cross_px=3.0, channel=None):
    """The seam geometry of a tiled acquisition, measured once and kept: measure_tile_overlaps
    on `frame` (default the last complete pass; a textured frame fits best), saved to `path` (JSON) with what
    it was measured on, and read back from there on later runs (delete it to measure again).

    Raises if the tiles are left more than `max_cross_px` apart across the seams (cross_dy,
    cross_dx): the camera's rotation is then off (Kinetix: -2.15 deg on MBO-049, 2026-10-23;
    -1.94 on MBO-041). The per-gap spread is not checked: on dim images (GFP) it is noisy at
    single gaps while the overlaps themselves agree. Returns the geometry dict, for
    stitch_to_zarr(overlap=...)."""
    import json

    path = Path(path)
    if not path.exists():
        tiled = open_tiled(src, channel)
        lead = [d for d in tiled["tile"].dims[:-4] if tiled.sizes[d] > 1]
        n = tiled.sizes[lead[0]] if lead else 1
        if lead and frame < 0 and "seconds" in tiled:
            # count back from the last complete pass: a run stopped early leaves its planned
            # passes with no tiles (seconds 0), and seams fit on those fall back to the nominal
            # overlap (MBO-054 run 1, frame 19 of 10 acquired)
            sec = tiled["seconds"].values.reshape(n, -1)
            complete = np.flatnonzero((sec > 0).all(axis=1))
            n = int(complete[-1]) + 1 if complete.size else n
        k = frame % n
        geometry = measure_tile_overlaps(src, rotation_deg=rotation_deg, frame=k, channel=channel)
        on = Path(src).name + (f" ({channel})" if channel else "")
        geometry.update(measured_on=f"{on}, frame {k}", rotation_deg=rotation_deg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(geometry, indent=1))
    geometry = json.loads(path.read_text())
    worst = max(abs(geometry["cross_dy"]), abs(geometry["cross_dx"]))
    if worst > max_cross_px:
        raise ValueError(f"{path}: tiles left {worst:.1f} px apart across the seams "
                         f"(measured on {geometry.get('measured_on', '?')}): check rotation_deg")
    return geometry


def fit_flatfield(src, path, *, camera_offset=80.0, channel=None):
    """The BaSiC pseudo flatfield of one acquisition (one camera tile, mean 1), fitted from
    its own tiles by the package's fit_flatfield.py in a throwaway uv environment (basicpy
    needs its own pins), unless `path` (.tif, with a .json of stats beside it) exists.
    `camera_offset` (RFU) is subtracted from the tiles first; `channel` picks one channel of a
    multi-channel acquisition. Returns `path`."""
    import os
    import subprocess

    path = Path(path)
    if not path.exists():
        script = Path(__file__).with_name("fit_flatfield.py")
        subprocess.run(["uv", "run", str(script), str(src), str(path), str(camera_offset),
                        *([channel] if channel else [])],
                       check=True, env={**os.environ, "PYTHONNOUSERSITE": "1"})
    return path


def stitch_run(src, dst, *, camera, geometry_path, flatfield_path, seam_frame=-1, max_cross_px=3.0,
               channel=None, record_dir=None):
    """The standard way in for a tiled acquisition (an assay's kinetic run, a GFP or Cy5 scan):
    seams measured or reused (stitch_geometry), flatfield fitted or reused (fit_flatfield; flatfield_path=None for none),
    then flatfield-corrected, rotated and stitched to the zarr store `dst` (stitch_to_zarr;
    reopened if it exists). `camera` is {"rotation_deg": ..., "offset": ...}, optionally with "dark": a measured
    per-pixel dark image (path or array, one camera tile), subtracted instead of the scalar offset.

    Returns a dict: image (the lazy (time, y, x) or (y, x) stitched image), store (the
    stitched dataset, with its attrs: seams_y / seams_x, tile_px, ...), tiled (the raw
    acquisition, with its attrs: substrate, filter, camera, start_time, ...), geometry, and
    pass_s (each frame's first tile, in s since the run started; None for a snapshot).
    `channel` picks one channel of a multi-channel acquisition (each channel gets its own
    flatfield and store; the seams can be measured on either and reused).
    `record_dir` (e.g. the experiment's "1_raw-images") adds the acquisition to that folder's record of where the
    raw images are (provenance.record_raw_source: raw_sources.csv and README.md)."""
    geometry = stitch_geometry(src, geometry_path, rotation_deg=camera["rotation_deg"],
                               frame=seam_frame, max_cross_px=max_cross_px, channel=channel)
    # flatfield_path=None: no flatfield correction (assay runs: the chambers' own signal is the readout, MBO-054)
    # the offset: camera["dark"] (a measured per-pixel dark image, path or array) when given, else camera["offset"];
    # BaSiC's fit takes a scalar, the dark image's median
    offset = camera.get("dark", camera.get("offset", 0.0))
    scalar = float(np.median(_offset_value(offset)))
    flat = None if flatfield_path is None else fit_flatfield(src, flatfield_path, camera_offset=scalar, channel=channel)
    store = stitch_to_zarr(src, dst, flatfield=flat, offset=offset,
                           rotation_deg=camera["rotation_deg"], overlap=geometry, channel=channel)
    tiled = open_tiled(src, channel)
    seconds = tiled["seconds"].values
    pass_s = np.nanmin(seconds.reshape(seconds.shape[0], -1), axis=1) if seconds.ndim == 3 else None
    if record_dir is not None:
        from .provenance import record_raw_source

        record_raw_source(src, dst, tiled, record_dir)
    from ..analysis.steps import tally_step

    own = Path(src).name in Path(geometry_path).stem
    tally_step("stitch_run", "Stitch tile scans",
               ["per acquisition: BaSiC flatfield (none on assay runs), camera offset",
                f"subtracted ({'measured dark image' if 'dark' in camera else 'scalar'}), rotated {camera['rotation_deg']:g}°, tiles joined at",
                f"measured seams (≤ {max_cross_px:g} px off across a seam)",
                "a dim scan reuses a bright scan's seams"],
               Path(src).name + (f" ({channel})" if channel else ""),
               label="seams measured on itself" if own else "seams from another scan",
               level="pass" if own else "info", unit="acquisitions")
    return dict(image=store["image"], store=store, tiled=tiled, geometry=geometry, pass_s=pass_s)


def tile_seam_flags(x, y, radius, pitch_x, pitch_y=None):
    """Which chambers' ROIs touch a tile seam in a stitched image.

    Tiles are butted every `pitch_x` px across and `pitch_y` px down (default: the same),
    so the seams sit at multiples of the pitch. A chamber at (x, y) with ROI `radius`
    touches a seam when its center is closer than `radius` to one:

      seam_x  the ROI crosses a VERTICAL seam (a tile's left/right edge)
      seam_y  the ROI crosses a HORIZONTAL seam (a tile's top/bottom edge)

    Chambers on a seam are measured across two tiles, imaged at different times (and
    the overlap strip twice), so their signal and bleaching differ from the rest.
    Returns (seam_x, seam_y) as boolean arrays shaped like x."""
    pitch_y = pitch_x if pitch_y is None else pitch_y
    x, y = np.asarray(x, float), np.asarray(y, float)

    # distance from the center to the nearest seam, across and down
    dist_x = np.minimum(x % pitch_x, pitch_x - x % pitch_x)
    dist_y = np.minimum(y % pitch_y, pitch_y - y % pitch_y)

    return dist_x < radius, dist_y < radius


def chamber_seconds(seconds, cx, cy, tile_px):
    """Per-chamber imaging time for a stitched tile scan: (time, n_rows, n_cols).

    `seconds` is the per-tile time array (time, row, col); `cx`, `cy` are chamber centers
    in stitched-image pixels; `tile_px` is the stitched tile pitch (y, x), e.g.
    stitched.attrs['tile_px']. Each chamber takes the time of the tile its center falls
    in. Pass the result to kinetic_traces(seconds=...)."""
    seconds = np.asarray(seconds, float)
    _, R, C = seconds.shape
    tile_r = np.clip(np.asarray(cy) // tile_px[0], 0, R - 1).astype(int)
    tile_c = np.clip(np.asarray(cx) // tile_px[1], 0, C - 1).astype(int)
    return seconds[:, tile_r, tile_c]


# --------------------------------------------------------------------------- #
# 4. Flatfield correction
# --------------------------------------------------------------------------- #
def tile_shift(reference, moving, *, tiles=None, upsample=4):
    """(dy, dx) in pixels that moves stitched image `moving` onto `reference`: the median of
    per-tile phase-correlation shifts measured on edge images (chip walls and channels), so
    a dim scan still registers. `tiles` is the tile edge in px (default: from the store's
    'tile_px' attr if `reference` is a stitched Dataset, else 922). Pass the result as
    stitch_to_zarr(background_shift=...)."""
    from scipy.ndimage import gaussian_filter
    from skimage.registration import phase_cross_correlation

    step = tiles or (
        reference.attrs["tile_px"][0]
        if hasattr(reference, "attrs") and "tile_px" in reference.attrs
        else 922
    )
    ref = np.asarray(getattr(reference, "image", reference), float)
    mov = np.asarray(getattr(moving, "image", moving), float)

    def edges(a):
        return np.hypot(*np.gradient(gaussian_filter(a, 3)))

    shifts = []
    for ty in range(ref.shape[0] // step):
        for tx in range(ref.shape[1] // step):
            sl = (slice(ty * step, (ty + 1) * step), slice(tx * step, (tx + 1) * step))
            s, _, _ = phase_cross_correlation(
                edges(ref[sl]),
                edges(mov[sl]),
                upsample_factor=upsample,
                normalization=None,
            )
            shifts.append(s)
    return tuple(float(v) for v in np.median(shifts, axis=0))


def scan_shift(ref, mov, *, n=4, half=500, sigma=3, upsample=10):
    """(dy, dx) of `mov`'s chip relative to `ref`'s (two stitched 2-D images of one chip): the median, over an
    n x n grid of (2 half)-px crops, of phase correlation on edge maps (Gaussian gradient magnitude, `sigma`
    px), so chamber walls register even where the signal is dim. The crops stay 2 `half` px clear of the
    top, bottom and left edges and 3 `half` of the right. Add the result to a grid placed on `ref` to read
    `mov` (MBO-051 / MBO-052's scan_shift)."""
    from scipy import ndimage
    from skimage.registration import phase_cross_correlation

    def edges(im):
        return ndimage.gaussian_gradient_magnitude(np.nan_to_num(np.asarray(im, np.float32)), sigma)

    H, W = np.shape(ref)
    shifts = []
    for y in np.linspace(2 * half, H - 2 * half, n).astype(int):
        for x in np.linspace(2 * half, W - 3 * half, n).astype(int):
            a = edges(ref[y - half:y + half, x - half:x + half])
            b = edges(mov[y - half:y + half, x - half:x + half])
            shifts.append(phase_cross_correlation(a, b, upsample_factor=upsample, normalization=None)[0])
    return tuple(float(v) for v in -np.median(shifts, axis=0))


def cached_shift(path, measure):
    """A (dy, dx) shift measured once and kept: `measure()` (a callable returning (dy, dx)) runs only when
    `path` (JSON) does not exist yet; afterwards the saved values are read back. Delete the file to measure
    again."""
    import json

    path = Path(path)
    if not path.exists():
        dy, dx = measure()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"dy": float(dy), "dx": float(dx)}, indent=1))
    saved = json.loads(path.read_text())
    return saved["dy"], saved["dx"]


def split_channel(name):
    """An acquisition name with an optional channel, "run|channel" (one channel of a multi-channel
    acquisition, MBO-052), as (run, channel or None, store name): the store name is "run_channel", or just
    "run" without a channel."""
    run, _, channel = str(name).partition("|")
    return run, channel or None, f"{run}_{channel}" if channel else run


def flatfield_from_tiles(src, *, sigma=40):
    """Estimate a flatfield from tile scans of the chip itself: the median over every tile
    of `src` (one acquisition, or a list of them), so chip features -- which sit at
    different places in different tiles -- average out, then smoothed with a Gaussian of
    `sigma` px. Use scans with signal but no specific features, e.g. the chip before
    expression, or free dye filling the chip. An APPROXIMATION for objectives/channels with
    no reference-slide flatfield. Returns a float32 array the size of one camera tile."""
    from scipy.ndimage import gaussian_filter

    sources = src if isinstance(src, (list, tuple)) else [src]
    stacks = []
    for one in sources:
        tiles = open_tiled(one)["tile"]
        stacks.append(
            tiles.squeeze().values.astype(np.float32).reshape(-1, *tiles.shape[-2:])
        )
    return gaussian_filter(np.median(np.concatenate(stacks), axis=0), sigma).astype(
        np.float32
    )


def flatfield_correct(
    img, flatfield, *, darkfield=0.0, background=None, normalize=True
):
    """Flatfield-correct `img` (2-D, or a 3-D (time, y, x) stack) with a reference
    `flatfield` (path or array), flattening illumination/vignetting:

        corrected = (img - dark) / (flat - dark) * mean(flat - dark)

    `darkfield` is a scalar camera offset or a 2-D dark image / path (default 0). With
    `background` (an image of the same field without the signal, e.g. the chip before
    expression; it already contains the camera offset), the numerator becomes
    img - background, which also removes the chip's own autofluorescence:

        corrected = (img - background) / (flat - dark) * mean(flat - dark)

    With `normalize=False` the `* mean(flat-dark)` rescale is skipped (output ~O(1)).
    Returns a float array the same shape as `img`; the flatfield must match the frame
    (y, x) shape."""
    a = np.asarray(img, float)
    flat = load_image(flatfield).astype(float)
    dark = (
        load_image(darkfield).astype(float)
        if isinstance(darkfield, (str, Path))
        else np.asarray(darkfield, float)
    )

    fd = flat - dark
    if a.shape[-2:] != fd.shape:
        raise ValueError(
            f"flatfield shape {fd.shape} != image frame shape {a.shape[-2:]}."
        )

    subtract = dark if background is None else np.asarray(background, float)
    corr = (a - subtract) / fd
    return corr * fd.mean() if normalize else corr
