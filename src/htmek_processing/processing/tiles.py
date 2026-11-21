"""Per-tile brightness check on stitched scans: a tile imaged darker or brighter than in the other scans (a lamp or
focus problem on one acquisition) shows up as a deviation of its median from that tile's usual share of the scan."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def tile_bounds(store):
    """Each tile's region of a stitched store, from its seams: [(tile_row, tile_col, y0, y1, x0, x1)]."""
    a, shape = store["store"].attrs, store["image"].shape[-2:]
    ys, xs = [0, *a["seams_y"], shape[0]], [0, *a["seams_x"], shape[1]]
    return [(i, j, ys[i], ys[i + 1], xs[j], xs[j + 1]) for i in range(len(ys) - 1) for j in range(len(xs) - 1)]


def tile_medians(store, *, scan, frames="all", stride=8):
    """The median pixel of every tile (every `stride`-th pixel) of a stitched scan, per pass (`frames`: "all", "last",
    or a list). Returns one row per frame and tile: scan, frame, tile_row, tile_col, y0, y1, x0, x1, median."""
    im = store["image"]
    n = im.sizes.get("time", 1) if hasattr(im, "sizes") else (im.shape[0] if im.ndim == 3 else 1)
    frames = range(n) if frames == "all" else [n - 1] if frames == "last" else frames
    bounds = tile_bounds(store)
    rows = []
    for f in frames:
        img = np.asarray(im[f] if im.ndim == 3 else im)
        for i, j, y0, y1, x0, x1 in bounds:
            rows.append(dict(scan=scan, frame=int(f), tile_row=i, tile_col=j, y0=y0, y1=y1, x0=x0, x1=x1,
                             median=float(np.median(img[y0:y1:stride, x0:x1:stride]))))
    return pd.DataFrame(rows)


def tile_deviations(medians, *, group="group", n_sd=5.0, min_deviation=0.02, window=5):
    """Each tile's deviation from its usual brightness. Per scan and frame, a tile's median over the median of all its
    tiles (`rel`); the tile's usual `rel` is its median over every scan and frame of the same `group` (e.g. the
    flatfielded GFP scans; for a time series, the tile's median over `window` passes around it); `deviation` = rel / usual, over its median across that scan's tiles, - 1. A tile
    is `flagged` when its |deviation| is
    above `n_sd` robust SDs (1.4826 x MAD) of every deviation, pooled over the groups,
    and by at least `min_deviation` (a fraction). Returns `medians` with rel, usual, deviation,
    flagged."""
    m = medians.copy()
    if group not in m:
        m[group] = "all"
    m["rel"] = m["median"] / m.groupby(["scan", "frame"])["median"].transform("median")
    m["usual"] = m.groupby([group, "tile_row", "tile_col"])["rel"].transform("median")
    # a time series (an assay run): the tile's share in its neighboring passes, so slow changes (product building up
    # unevenly over the chip) aren't flagged and a one-pass blip is
    series = m.groupby("scan")["frame"].transform("nunique") > 2
    if series.any():
        ts = m[series].sort_values("frame")
        roll = ts.groupby(["scan", "tile_row", "tile_col"])["rel"].transform(
            lambda r: r.rolling(window, center=True, min_periods=1).median())
        m.loc[roll.index, "usual"] = roll
    ratio = m["rel"] / m["usual"]
    m["deviation"] = ratio / ratio.groupby([m["scan"], m["frame"]]).transform("median") - 1   # one bad tile can't shift the rest
    d = m["deviation"]
    m["flagged"] = d.abs() > max(n_sd * 1.4826 * (d - d.median()).abs().median(), min_deviation)   # one spread: one camera and lamp
    return m


def tile_of(x, y, tiles):
    """The (tile_row, tile_col) each point (x, y) falls in, from tile_medians' bounds (one scan's)."""
    b = tiles.drop_duplicates(["tile_row", "tile_col"])
    ys, xs = np.sort(b["y0"].unique())[1:], np.sort(b["x0"].unique())[1:]
    return np.searchsorted(ys, np.asarray(y), side="right"), np.searchsorted(xs, np.asarray(x), side="right")


def tile_figure(dev, *, scans=None, title="tile median vs its usual share of the scan"):
    """The deviation of every tile, one small chip-shaped grid per scan (an assay: its worst pass), flagged tiles
    outlined."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    worst = dev.loc[dev.groupby(["scan", "tile_row", "tile_col"])["deviation"].apply(lambda d: d.abs().idxmax())]
    scans = scans or list(dict.fromkeys(dev["scan"]))
    ncol = min(len(scans), 6)
    nrow = int(np.ceil(len(scans) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.9 * ncol + 1, 2.2 * nrow + 0.6), squeeze=False)
    lim = max(0.02, float(np.nanpercentile(worst["deviation"].abs(), 99)))
    for ax, s in zip(axes.flat, scans):
        w = worst[worst["scan"] == s]
        g = w.pivot(index="tile_row", columns="tile_col", values="deviation")
        im = ax.imshow(100 * g.to_numpy(), cmap="RdBu_r", vmin=-100 * lim, vmax=100 * lim)
        for r in w[w["flagged"]].itertuples():
            ax.add_patch(Rectangle((r.tile_col - 0.5, r.tile_row - 0.5), 1, 1, fill=False, ec="k", lw=1.6))
            ax.text(r.tile_col, r.tile_row, f"{100 * r.deviation:+.0f}", ha="center", va="center", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(s if len(s) <= 22 else s[:21] + "…", fontsize=8)
    for ax in list(axes.flat)[len(scans):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="deviation (%)")
    fig.suptitle(title, fontsize=10, x=0.02, ha="left")
    return fig


# ---------------------------------------------------------------------- hot rows
# Some sensors have rows whose gain is off from their neighbors': the Kinetix (MBO-054, 2026-11) has one row in
# every 128 (rows 103, 231, 359, ...) at ~1.3-3x its neighbors, the two rows above it ~5-10% dark. Fixed sensor
# rows, in every tile and pass and in the standards; a smooth flatfield does not remove them. Stitching rotates
# each tile, so a hot row turns into a tilted line 2-4 px wide across the mosaic, through the chambers it crosses.


def row_gains(src, *, offset=0.0, frames="ends", window=15, stride=4, channel=None):
    """Each sensor row's gain against its neighbors', from the raw tiles of one acquisition: every tile's row
    medians (every `stride`-th column, less the camera `offset`) over a running median of `window` rows, one value
    per frame (the median over the tiles). `frames`: "ends" (the first and last pass), "all", or a list; a snapshot
    has one. The `window // 2` rows at each edge of the sensor have no centered neighbors and get NaN (stitching trims
    them anyway). Returns one row per frame and sensor row: frame, row, gain, tile_lo, tile_hi (the 10th and 90th
    percentile over tiles)."""
    from scipy.ndimage import median_filter

    from .stitch import open_tiled

    tiled = open_tiled(src, channel)
    tiles = tiled["tile"]
    lead = [d for d in tiles.dims[:-4]]
    tiles = tiles.squeeze([d for d in lead if tiled.sizes[d] == 1])
    lead = list(tiles.dims[:-4])
    if len(lead) > 1:
        raise ValueError(f"row_gains takes one leading dim (time) at most, not {lead}")
    n = tiles.sizes[lead[0]] if lead else 1
    frames = [0, n - 1] if frames == "ends" else range(n) if frames == "all" else frames
    rows = []
    for f in sorted(set(int(f) for f in frames)):
        fr = tiles[{lead[0]: f}] if lead else tiles
        prof = []
        for r in range(fr.sizes["row"]):
            for c in range(fr.sizes["col"]):
                t = np.asarray(fr[{"row": r, "col": c}].values[:, ::stride], np.float32) - offset
                prof.append(np.median(t, axis=1))
        prof = np.asarray(prof)                                   # (tiles, y)
        ref = median_filter(prof, size=(1, window), mode="nearest")
        g = prof / np.where(np.abs(ref) > 1e-6, ref, np.nan)
        g[:, : window // 2] = g[:, prof.shape[1] - window // 2:] = np.nan
        with warnings.catch_warnings():   # the edge rows: NaN in every tile
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, med, hi = np.nanpercentile(g, [10, 50, 90], axis=0)
        rows.append(pd.DataFrame(dict(frame=f, row=np.arange(prof.shape[1]), gain=med, tile_lo=lo, tile_hi=hi)))
    return pd.concat(rows, ignore_index=True)


def hot_rows(gains, *, threshold=0.02, by=("camera",)):
    """The rows off by more than `threshold` (2%) in the median over every scan and frame of a camera (`gains`:
    row_gains' tables with a `camera` column). Returns one row per camera and flagged sensor row: row, gain (that
    median), gain_lo, gain_hi (its range over the scans and passes: a gain that moves between passes does not
    cancel out of a rate)."""
    by = list(by)
    g = gains.groupby([*by, "row"])["gain"].agg(gain="median", gain_lo="min", gain_hi="max").reset_index()
    return g[(g["gain"] - 1).abs() > threshold].reset_index(drop=True)


def row_period(rows, gains=None, *, gap=3):
    """The spacing of a set of sensor rows, if one spacing covers most of them (the Kinetix: 128), else None. Rows
    within `gap` of each other are one block (a hot row and its dark neighbors), counted once at its row furthest
    from a gain of 1 (`gains`, one per row; without them, the block's first row). With `gains`, only the blocks off
    by at least a quarter of the largest are spaced (a faint pattern of its own, e.g. the Kinetix's 2% dark row in
    256, would otherwise halve the spacing)."""
    rows = np.asarray(rows)
    order = np.argsort(rows)
    rows = rows[order]
    off = np.abs(np.asarray(gains, float)[order] - 1) if gains is not None else -np.arange(len(rows), dtype=float)
    if not len(rows):
        return None
    blocks = np.split(np.arange(len(rows)), np.flatnonzero(np.diff(rows) > gap) + 1)
    peaks = [b[np.argmax(off[b])] for b in blocks]
    if gains is not None:
        peaks = [i for i in peaks if off[i] >= 0.25 * off[peaks].max()]
    d = np.diff(rows[peaks])
    if len(d) < 2:
        return None
    vals, counts = np.unique(d, return_counts=True)
    return int(vals[counts.argmax()]) if counts.max() >= 0.6 * len(d) else None


def hot_row_mask(store, rows, *, tile_shape, min_weight=0.25):
    """Where the given sensor rows land in a stitched store: one tile with the rows set, put through the store's own
    rotation and placement (the same for every tile), True where a stitched pixel takes at least `min_weight` of
    its value from them. `tile_shape` is the raw tile's (y, x)."""
    from scipy.ndimage import rotate

    from .stitch import place_tiles, stitch_tiles

    a = store["store"].attrs if "store" in store else store.attrs
    h, w = tile_shape
    t = np.zeros((h, w), np.float32)
    t[np.asarray(rows, int)] = 1.0
    rot = a.get("rotation_deg", 0.0) or 0.0
    if rot:
        t = rotate(t, rot, reshape=False, order=1, mode="nearest")
    t = t >= min_weight
    R, C = len(a["tile_y0"]), len(a["tile_x0"])
    grid = np.broadcast_to(t, (R, C, h, w))
    if a.get("tile_origin") is not None:
        shape = (store["image"] if "image" in store else store).shape[-2:]
        return place_tiles(grid, np.asarray(a["tile_origin"]), shape, fill_gaps=True)
    ov = a["overlap_px"]
    ov = tuple(ov) if isinstance(ov, list) else ov   # (overlap_y, overlap_x), each one number or one per gap
    return stitch_tiles(grid, ov, split=bool(rot))


def hot_row_figure(gains, flagged, *, threshold=0.02, title="sensor row gain vs its neighbors (raw tiles)"):
    """Per camera: every sensor row's gain (the median over scans and passes, the range shaded), the flagged rows
    marked; beside it, a close-up on the worst flagged row."""
    import matplotlib.pyplot as plt

    cams = list(dict.fromkeys(gains["camera"]))
    fig, axes = plt.subplots(len(cams), 2, figsize=(11, 2.6 * len(cams) + 0.4), squeeze=False,
                             gridspec_kw=dict(width_ratios=(3, 1)))
    for (ax, zx), cam in zip(axes, cams):
        g = gains[gains["camera"] == cam].groupby("row")["gain"].agg(["median", "min", "max"])
        f = flagged[flagged["camera"] == cam]
        for a in (ax, zx):
            a.fill_between(g.index, g["min"], g["max"], color="0.75", lw=0, step="mid")
            a.plot(g.index, g["median"], color="0.2", lw=0.6, drawstyle="steps-mid")
            a.axhline(1 + threshold, color="tab:red", lw=0.6, ls=":")
            a.axhline(1 - threshold, color="tab:red", lw=0.6, ls=":")
            a.scatter(f["row"], f["gain"], s=10, color="tab:red", zorder=3)
        lo, hi = np.nanpercentile(g["median"], [0.05, 99.95])
        ax.set_ylim(min(0.85, lo - 0.05), max(1.15, hi + 0.1))
        ax.set_xlim(g.index.min(), g.index.max())
        ax.set_xlabel("sensor row")
        ax.set_ylabel("gain")
        per = row_period(f["row"], f["gain"]) if len(f) else None
        ax.set_title(f"{cam}: {len(f)} rows off by > {100 * threshold:g}%" + (f", every {per} rows" if per else ""),
                     fontsize=9, loc="left")
        if len(f):
            worst = int(f.loc[(f["gain"] - 1).abs().idxmax(), "row"])
            zx.set_xlim(worst - 12, worst + 12)
            zx.set_title(f"rows around {worst}", fontsize=9, loc="left")
            zx.set_xlabel("sensor row")
        else:
            zx.axis("off")
    fig.suptitle(title, fontsize=10, x=0.02, ha="left")
    fig.tight_layout()
    return fig
