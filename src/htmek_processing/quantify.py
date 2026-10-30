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

    # the ROI is read wherever it lies on the image; the annulus, which reaches further,
    # only where it does too (a chamber at the edge keeps its ROI and gets a NaN background)
    x = np.rint(cx).astype(int).ravel()
    y = np.rint(cy).astype(int).ravel()

    def fits(r):
        r = int(np.ceil(r))
        return (x >= r) & (x < W - r) & (y >= r) & (y < H - r)

    ok_roi = fits(radius)
    ok_bg = fits(bg_outer) if use_bg else ok_roi

    def gather(mask, ok):
        """(n_ok, n_mask_px) pixel values: one fancy-index read for every chamber at
        once."""
        return img[y[ok, None] + yy[mask], x[ok, None] + xx[mask]].astype(float)

    def scatter(vals, ok):
        """Place per-chamber values (n_ok,) back into a cx-shaped array, NaN off-image."""
        full = np.full(x.shape, np.nan)
        full[ok] = vals
        return full.reshape(cx.shape)

    # ROI stats per chamber
    px = gather(roi, ok_roi)
    stats = {a: scatter(getattr(np, a)(px, axis=1), ok_roi) for a in aggs}
    out = dict(stats)
    out["n_px"] = scatter(np.full(ok_roi.sum(), n_px, float), ok_roi)

    if use_bg:
        bg = scatter(getattr(np, bg_agg)(gather((d2 > bg_inner**2) & (d2 <= bg_outer**2), ok_bg), axis=1), ok_bg)
        out["bg"] = bg
        for a, v in stats.items():
            # a sum covers n_px pixels, so it loses n_px x the per-pixel background
            out[f"{a}_net"] = v - bg * (n_px if a == "sum" else 1)
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
    prefix=None,
):
    """Tidy DataFrame from explicit centers, one row per chamber.

    Columns: mark_row, mark_col, tag, x, y, one column per entry in `aggs`, and n_px.
    With `bg=True` (or explicit bg_inner/bg_outer) it also adds the local background 'bg'
    and background-subtracted '<agg>_net' columns. If `img` is None the intensity columns
    are NaN.

    `prefix` names the mask the image was read with, per the naming convention: "button"
    for a button mask, "chamber_ROI" for a chamber ROI. With it every intensity column
    carries the prefix (`button_sum`, `button_median_net`, `button_n_px`) and the annulus
    background is `<prefix>_annulus_<bg_agg>`; without it the names stay unprefixed
    (`sum`, `bg`, `sum_net`), as notebooks before MBO-051 expect.

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
    return _with_prefix(df, prefix, aggs, bg_agg)


def _with_prefix(df, prefix, aggs, bg_agg="median"):
    """Rename centers_to_df's intensity columns to the `<prefix>_...` convention (None: as is)."""
    if not prefix:
        return df
    names = {a: f"{prefix}_{a}" for a in aggs}
    names.update({f"{a}_net": f"{prefix}_{a}_net" for a in aggs})
    names.update({"n_px": f"{prefix}_n_px", "bg": f"{prefix}_annulus_{bg_agg}"})
    out = df.rename(columns={k: v for k, v in names.items() if k in df.columns})
    out.attrs = df.attrs
    return out


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
    prefix=None,
):
    """Apply one fixed grid (centers `cx`, `cy`) to every frame of a (time, y, x) stack.

    Returns a *long* tidy DataFrame, one row per (chamber, frame): frame, seconds,
    mark_row, mark_col, tag, x, y, then each `aggs` column (+ n_px, and bg / '<agg>_net'
    if a background annulus is on). Ready for seaborn per variant, or fit_rates(df).
    `prefix` names the columns by mask, as centers_to_df's.

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
            prefix=prefix,
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


def bleach_model(n, amplitude, rate, floor):
    """Fraction of the first exposure's signal left after exposure `n` (0 = the first): an
    exponential loss to a floor, amplitude * exp(-rate n) + floor, with amplitude + floor = 1."""
    return amplitude * np.exp(-rate * np.asarray(n, float)) + floor


def fit_bleaching(traces, value="median", *, ref_tags=("BLANK",), by="frame",
                  per=("mark_row", "mark_col"), max_slope=None):
    """Model photobleaching from reference chambers (no enzyme: BLANK by default), within one
    run: each reference chamber's signal relative to its first exposure, the median over
    them at each exposure (`by`), and an exponential loss to a floor fitted to that median
    (bleach_model, per exposure: bleaching follows the light received, not the clock).

    `max_slope` (fraction of the first exposure per exposure) leaves out reference chambers
    that rise faster than that over the run (e.g. 0.001: product leaking in from a
    neighbor), so they don't pull the curve. Returns dict(amplitude, rate, floor, curve =
    {step: the model's factor}, median = {step: the reference median}, n_ref, n_left_out).
    Pass `curve` on, or use fit_progress_curves(bleach="model")."""
    from scipy.optimize import curve_fit

    keys = list(per)
    ref = traces[traces["tag"].isin(ref_tags)].sort_values(by)
    if ref.empty:
        raise ValueError(f"no reference chambers tagged {ref_tags} in the traces")
    rel = ref[value] / ref.groupby(keys)[value].transform("first")
    ref = ref.assign(_rel=rel)
    n_left_out = 0
    if max_slope is not None:
        slope = ref.groupby(keys).apply(lambda g: np.polyfit(g[by], g["_rel"], 1)[0], include_groups=False)
        rising = slope[slope > max_slope].index
        n_left_out = len(rising)
        ref = ref[~ref.set_index(keys).index.isin(rising)]
    med = ref.groupby(by)["_rel"].median()
    steps = med.index.to_numpy(float)
    n = steps - steps.min()

    def model(nn, amplitude, rate):
        return bleach_model(nn, amplitude, rate, 1 - amplitude)   # 1 at the first exposure

    try:
        (amp, rate), _ = curve_fit(model, n, med.to_numpy(float), p0=[0.01, 0.2],
                                   bounds=([-0.5, 0.0], [0.5, 10.0]), maxfev=20_000)
    except (RuntimeError, ValueError):
        amp, rate = 0.0, 0.0   # no bleaching resolved
    curve = model(n, amp, rate)
    return dict(amplitude=float(amp), rate=float(rate), floor=float(1 - amp),
                curve=dict(zip(med.index.tolist(), curve.tolist())), median=med.to_dict(),
                n_ref=int(ref[keys].drop_duplicates().shape[0]), n_left_out=n_left_out)


def chamber_timecourse(stack, cx, cy, columns, tiled, tile_px, *, radius, ring=None,
                       aggs=("median", "mean", "sum"), prefix=None):
    """Per-timepoint intensities of every chamber of a kinetic run, the standard readout:
    one row per chamber x frame with `seconds` (the time of the tile the chamber's center
    falls in, since the run started: chambers in one pass are imaged up to a pass apart),
    the ROI's `aggs` (raw, nothing subtracted) and n_px, and each agg minus the chamber's own
    value at frame 0 (`<agg>_minus_frame0`).

    `ring` = (inner, outer) adds a reference ring's median (`bg`, and `<agg>_net`); for an
    assay put it outside the chamber: a ring on the chamber itself rises with the product
    and would cancel the signal. `tiled` is the raw acquisition (its per-tile 'seconds'),
    `tile_px` the stitched tile pitch (the store's attrs['tile_px']). `prefix` names the
    columns by mask, as centers_to_df's (e.g. "chamber_ROI": `chamber_ROI_median`,
    `chamber_ROI_median_minus_frame0`)."""
    from .stitch import chamber_seconds

    seconds = chamber_seconds(tiled["seconds"].values, cx, cy, tile_px)
    inner, outer = ring if ring is not None else (None, None)
    df = kinetic_traces(stack, cx, cy, columns, radius=radius, aggs=aggs, bg=ring is not None,
                        bg_inner=inner, bg_outer=outer, seconds=seconds, drop_empty=False, prefix=prefix)
    keys = ["mark_row", "mark_col"]
    first = df[df["frame"] == df["frame"].min()].set_index(keys)
    idx = df.set_index(keys).index
    for a in aggs:
        col = f"{prefix}_{a}" if prefix else a
        df[f"{col}_minus_frame0"] = df[col].to_numpy() - idx.map(first[col]).to_numpy()
    return df
