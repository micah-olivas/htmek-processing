"""Quantification: per-chamber intensities for one frame or a whole stack."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .layout import _column_items, chamber_tags, column_tags
from .io import n_used_frames, time_coord, to_2d_frame
from .grid import bilinear_grid, grid_centers


def _circ_mask(radius):
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return xx**2 + yy**2 <= radius**2


def _resolve_bg(radius, bg, bg_inner, bg_outer):
    """Resolve annulus radii. Returns (bg_inner, bg_outer) or (None, None) if bg is off.
    `bg=True` (or either radius given) enables it with defaults: a ring from the ROI's edge
    out to 11 px beyond it (inner = radius, outer = radius + 11)."""
    if not (bg or bg_inner is not None or bg_outer is not None):
        return None, None
    return (
        bg_inner if bg_inner is not None else radius,
        bg_outer if bg_outer is not None else radius + 11,
    )


def intensities_at(
    img,
    cx,
    cy,
    radius=12,
    aggs=("median", "sum"),
    bg_inner=None,
    bg_outer=None,
    bg_agg="median",
):
    """Aggregate intensity in a circular ROI of `radius` px around each (cx, cy) center.
    `aggs` is a sequence of numpy reducer names (e.g. 'median', 'sum', 'mean', 'max',
    'std'). Returns {name: array shaped like cx} (NaN where the window runs off-image), plus
    'n_px'.

    If `bg_inner`/`bg_outer` are given, a local background is estimated as `bg_agg` over the
    annulus (bg_inner < r <= bg_outer) and returned as 'bg', along with
    background-subtracted '<agg>_net' columns (sum subtracts bg*n_px; per-pixel reducers
    subtract bg directly)."""
    img = np.asarray(img).squeeze()
    H, W = img.shape
    cx = np.asarray(cx, float)
    cy = np.asarray(cy, float)
    use_bg = bg_inner is not None and bg_outer is not None
    win = int(bg_outer if use_bg else radius)

    # pixel offsets (dy, dx) of the ROI disc and the background annulus around a center
    yy, xx = np.mgrid[-win : win + 1, -win : win + 1]
    d2 = xx**2 + yy**2
    roi = d2 <= radius**2
    n_px = int(roi.sum())

    # chambers whose whole window lies on the image; the rest stay NaN
    x = np.rint(cx).astype(int).ravel()
    y = np.rint(cy).astype(int).ravel()
    ok = (x >= win) & (x < W - win) & (y >= win) & (y < H - win)

    def gather(mask):
        """(n_ok, n_mask_px) pixel values: one fancy-index read for every chamber at
        once."""
        return img[y[ok, None] + yy[mask], x[ok, None] + xx[mask]].astype(float)

    def scatter(vals):
        """Place per-chamber values (n_ok,) back into a cx-shaped array, NaN off-image."""
        full = np.full(x.shape, np.nan)
        full[ok] = vals
        return full.reshape(cx.shape)

    # ROI stats per chamber
    px = gather(roi)
    stats = {a: getattr(np, a)(px, axis=1) for a in aggs}
    out = {a: scatter(v) for a, v in stats.items()}
    out["n_px"] = scatter(np.full(ok.sum(), n_px, float))

    if use_bg:
        bg = getattr(np, bg_agg)(
            gather((d2 > bg_inner**2) & (d2 <= bg_outer**2)), axis=1
        )
        out["bg"] = scatter(bg)
        for a, v in stats.items():
            # a sum covers n_px pixels, so it loses n_px x the per-pixel background
            out[f"{a}_net"] = scatter(v - bg * (n_px if a == "sum" else 1))
    return out


def centers_to_df(
    cx,
    cy,
    columns=None,
    n_rows=None,
    *,
    img=None,
    radius=12,
    aggs=("median", "sum"),
    bg=False,
    bg_inner=None,
    bg_outer=None,
    bg_agg="median",
):
    """Tidy DataFrame from explicit centers, one row per chamber.

    Columns: mark_row, mark_col, tag, x, y, one column per entry in `aggs`, and n_px.
    With `bg=True` (or explicit bg_inner/bg_outer) it also adds the local background 'bg'
    and background-subtracted '<agg>_net' columns. If `img` is None the intensity columns
    are NaN.

    `columns` is any identity layout chamber_tags() accepts: a column list/dict
    (minichip), a pinlist DataFrame or .csv path, or a 2-D tag array. The centers are
    stashed in df.attrs['centers'] for show_grid()."""
    cx = np.asarray(cx, float)
    cy = np.asarray(cy, float)
    nr, nc = cx.shape
    tags = chamber_tags(columns, nr, nc)
    bi, bo = _resolve_bg(radius, bg, bg_inner, bg_outer)

    if img is not None:
        stats = intensities_at(
            img, cx, cy, radius, aggs, bg_inner=bi, bg_outer=bo, bg_agg=bg_agg
        )
    else:
        keys = list(aggs) + ["n_px"]
        if bi is not None:
            keys += ["bg"] + [f"{a}_net" for a in aggs]
        stats = {k: np.full((nr, nc), np.nan) for k in keys}

    # build column-wise: row-major ravel matches (mark_row, mark_col) order
    rows, cols = np.indices((nr, nc))
    df = pd.DataFrame(
        {
            "mark_row": rows.ravel(),
            "mark_col": cols.ravel(),
            "tag": tags.ravel(),
            "x": cx.ravel(),
            "y": cy.ravel(),
            **{k: np.asarray(v).ravel() for k, v in stats.items()},
        }
    )
    df.attrs["centers"] = (
        cx.tolist(),
        cy.tolist(),
        float(radius),
    )  # lists: pd.concat-safe
    df.attrs["bg_annulus"] = (bi, bo)
    return df


def manual_grid(
    img,
    columns,
    *,
    x0,
    y0,
    col_pitch,
    row_pitch,
    angle=0.0,
    radius=12,
    n_rows=8,
    aggs=("median", "sum"),
    bg=False,
    bg_inner=None,
    bg_outer=None,
    bg_agg="median",
):
    """Place a fixed (optionally rotated) grid and aggregate intensity per button.
    `img` is a 2-D array (y, x). Returns a tidy DataFrame; see centers_to_df for
    columns/bg."""
    img = np.asarray(img).squeeze()
    if img.ndim != 2:
        raise ValueError(
            f"manual_grid expects a 2-D image, got shape {img.shape}. "
            "Select a single frame (e.g. img.isel(time=0)) first."
        )

    _, n_cols = column_tags(columns, n_rows)
    cx, cy = grid_centers(x0, y0, col_pitch, row_pitch, angle, n_rows, n_cols)
    return centers_to_df(
        cx,
        cy,
        columns,
        n_rows,
        img=img,
        radius=radius,
        aggs=aggs,
        bg=bg,
        bg_inner=bg_inner,
        bg_outer=bg_outer,
        bg_agg=bg_agg,
    )


def df_from_corners(
    img,
    corners,
    columns=None,
    *,
    n_rows=8,
    n_cols=None,
    radius=12,
    frame=None,
    reducer="max",
    **kw,
):
    """Apply a saved 4-corner placement to any image (or (time, y, x) stack) and quantify
    per chamber. Stacks are reduced to one 2-D frame via to_2d_frame(frame, reducer).
    Extra kwargs (aggs, bg, bg_inner/bg_outer, bg_agg) pass through to centers_to_df."""
    frame2d = to_2d_frame(img, frame=frame, reducer=reducer)
    if n_cols is None:
        n_cols = len(_column_items(columns)) if columns is not None else n_rows

    cx, cy = bilinear_grid(np.asarray(corners, float), n_rows, n_cols)
    df = centers_to_df(cx, cy, columns, n_rows, img=frame2d, radius=radius, **kw)
    df.attrs["corners"] = np.asarray(corners, float).tolist()
    return df


def kinetic_traces(
    stack,
    cx,
    cy,
    columns=None,
    n_rows=None,
    *,
    radius=12,
    aggs=("median",),
    bg=False,
    bg_inner=None,
    bg_outer=None,
    bg_agg="median",
    seconds=None,
    time_name="seconds",
    drop_empty=True,
):
    """Apply one fixed grid (centers `cx`, `cy`) to every frame of a (time, y, x) stack.

    Returns a *long* tidy DataFrame, one row per (chamber, frame): frame, seconds,
    mark_row, mark_col, tag, x, y, then each `aggs` column (+ n_px, and bg / '<agg>_net'
    if a background annulus is on). Ready for seaborn per variant, or fit_rates(df).

    `seconds` is either
      - per frame, shape (time,): every chamber in a frame shares one time, or
      - per chamber, shape (time, n_rows, n_cols): each chamber gets its own time, e.g.
        the time its tile was imaged (see chamber_seconds for stitched tile scans).
    If None it is read from the stack's xarray coord `time_name`, falling back to the
    frame index. `drop_empty=True` trims trailing unused acquisition slots (seconds == 0).

    Frames are read one at a time, so a lazy (zarr / xarray) stack never has to fit in
    memory as a whole."""
    if stack.ndim != 3:
        raise ValueError(
            f"kinetic_traces expects a 3-D (time, y, x) stack, got shape "
            f"{tuple(stack.shape)}. For a single frame use centers_to_df."
        )
    cx, cy = np.asarray(cx, float), np.asarray(cy, float)

    if seconds is None:
        seconds = time_coord(stack, time_name)
    if seconds is None:
        seconds = np.arange(stack.shape[0], dtype=float)
    seconds = np.asarray(seconds, float)

    # a frame counts as acquired if any of its (per-chamber) times is > 0
    n_frames = stack.shape[0]
    if drop_empty:
        first_t = seconds.reshape(len(seconds), -1).max(axis=1)
        n_frames = min(n_frames, n_used_frames(first_t))

    out = []
    for t in range(n_frames):
        frame = np.asarray(stack[t])  # one frame in memory at a time
        d = centers_to_df(
            cx,
            cy,
            columns,
            n_rows,
            img=frame,
            radius=radius,
            aggs=aggs,
            bg=bg,
            bg_inner=bg_inner,
            bg_outer=bg_outer,
            bg_agg=bg_agg,
        )
        d.insert(0, "frame", t)
        # per-frame time broadcasts to every chamber; per-chamber time ravels row-major,
        # which matches centers_to_df's row order
        d.insert(1, "seconds", np.broadcast_to(seconds[t], cx.shape).ravel())
        out.append(d)

    df = pd.concat(out, ignore_index=True)
    df.attrs["centers"] = (
        cx.tolist(),
        cy.tolist(),
        float(radius),
    )  # lists: pd.concat-safe
    return df


def bleach_correct(
    traces,
    value="median",
    *,
    ref_tags=("BLANK",),
    mode="ratio",
    by="frame",
    per=("mark_row", "mark_col"),
):
    """Remove the signal loss shared by every chamber (photobleaching) using reference
    chambers with no enzyme (BLANK by default), within one run of kinetic_traces.

    Each reference chamber's trace is taken relative to its first frame, and the median
    across references at each `by` step (frame = exposure count, which is what bleaches)
    is the bleach curve. Then, for every chamber:

      mode="ratio"       value / curve, where curve = median(ref / ref at frame 0).
                         Bleaching removes a fixed FRACTION per exposure, so this suits
                         raw intensities, where chambers differ in brightness.
      mode="difference"  value - curve, where curve = median(ref - ref at frame 0).
                         An additive shift, as MBO-039 applied in [NADH] (µM) space.

    Returns a copy with `<value>_bc` added and the curve in .attrs['bleach_curve'] as a
    plain {step: factor} dict (a dict, not a Series, so pd.concat of corrected runs works:
    it compares .attrs, and Series can't be compared that way). Reference chambers come out flat by construction, so
    judge orientation and blanks on the uncorrected `value`."""
    keys = list(per)
    ref = traces[traces["tag"].isin(ref_tags)]
    if ref.empty:
        raise ValueError(f"no reference chambers tagged {ref_tags} in the traces")

    # each reference chamber relative to its own first frame, then the median per step
    first = ref.sort_values(by).groupby(keys)[value].transform("first")
    rel = ref[value] / first if mode == "ratio" else ref[value] - first
    curve = rel.groupby(ref[by]).median()

    out = traces.copy()
    step = out[by].map(curve)
    if mode == "ratio":
        out[f"{value}_bc"] = out[value] / step
    elif mode == "difference":
        out[f"{value}_bc"] = out[value] - step
    else:
        raise ValueError(f"mode must be 'ratio' or 'difference', got {mode!r}")
    out.attrs["bleach_curve"] = curve.to_dict()
    return out
