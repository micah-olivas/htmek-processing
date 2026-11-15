"""Per-tile brightness check on stitched scans: a tile imaged darker or brighter than in the other scans (a lamp or
focus problem on one acquisition) shows up as a deviation of its median from that tile's usual share of the scan."""
from __future__ import annotations

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
