"""QC plots: grid overlays, chamber crops over time, chip heatmaps and sparklines; the
shared plot style."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .io import load_image, n_used_frames, time_coord


def _display_path(p):
    """Shorten a path for display: the basename for a file, but 'parent/basename' for a
    directory, since a zarr store's own name is boilerplate
    ('.../125uM_500ms/tiled_images')."""
    p = Path(str(p))
    return f"{p.parent.name}/{p.name}" if p.is_dir() and p.parent.name else p.name


def _image_name(img):
    """Best-effort display name for an image: the path when `img` is one, else where an
    xarray DataArray records the file it was opened from (`source`/`path`/`filename` attr,
    or `.encoding['source']`), else its `.name`. None if nothing usable fits."""
    if isinstance(img, (str, Path)):
        return _display_path(img)

    attrs = getattr(img, "attrs", None) or {}
    encoding = getattr(img, "encoding", None) or {}
    for src in (
        attrs.get("source"),
        attrs.get("path"),
        attrs.get("filename"),
        encoding.get("source"),
    ):
        if src:
            return _display_path(src)

    # .name is the *variable* name, not an image identity — "tile" is this module's own
    # zarr key (see load_standard_images), so it names nothing; pass name= for those.
    nm = getattr(img, "name", None)
    return str(nm) if nm and str(nm) != "tile" else None


def show_grid(
    img,
    df,
    *,
    vmax_pct=99,
    figsize=(9, 9),
    show_tags=True,
    tag_rotation=0,
    per_button=False,
    show_masks=True,
    tag_mode="number",
    name=None,
):
    """Overlay the grid ROIs (from manual_grid/GridAdjuster) on the image.

    `img` is an array or a path (loaded via load_image). The title is the image's `name`:
    pass it explicitly, or leave it None to infer — the file's basename when `img` is a
    path, else wherever an xarray DataArray records what it was opened from. With no name to
    show, the title is left empty rather than filled with boilerplate. Pixel ticks are
    hidden.

    If `show_tags` and df has a 'tag' column, identities are labeled just *above* the
    image, one label per *run of consecutive columns sharing a tag* (so a repeated identity
    isn't printed over every column). `tag_mode` picks the label:

      "number" (default) — a short lane index plus a "n  tag" legend to the right of the
                           image. Each distinct tag keeps one number wherever it recurs, so
                           long or repeated names don't crowd the top of the figure.
      "text"             — the tag name itself over each run (the old behavior).

    Empty tags (blanks) are skipped and break a run. Set `per_button=True` for the tag name
    on every button instead (unaffected by `tag_mode`).

    A live `vmax` slider (below the image) sets the display ceiling — drag it down to
    brighten a dim image. A "show masks" checkbox (top-left, like GridAdjuster's) hides
    everything drawn *over* the buttons — ROI circles, center dots, and `per_button` tags
    — so you can inspect the raw image; the column labels and legend sit outside the grid
    and stay put. `show_masks=False` starts with the overlay hidden. The widgets are stashed
    on the returned figure to keep them alive."""
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.widgets import Slider, CheckButtons

    if tag_mode not in ("number", "text"):
        raise ValueError(f"tag_mode must be 'number' or 'text', got {tag_mode!r}")

    # white text with a black outline stays legible over both dark and bright buttons
    halo = [pe.withStroke(linewidth=2, foreground="black")]
    if name is None:
        name = _image_name(img)
    img = load_image(img) if isinstance(img, (str, Path)) else np.asarray(img).squeeze()
    cx, cy, radius = df.attrs["centers"]

    fig, ax = plt.subplots(figsize=figsize)
    # clamp the display floor at 0 so sub-zero (negative) pixels render black;
    # scale vmax off the positive signal, since a mostly-negative (background-
    # subtracted) image can have a negative percentile that would invert vmin/vmax
    gray = plt.cm.gray.copy()
    gray.set_under("black")
    pos = img[img > 0]
    data_hi = float(pos.max()) if pos.size else 1.0
    vmax = float(np.percentile(pos, vmax_pct)) if pos.size else 1.0
    im = ax.imshow(img, cmap=gray, vmin=0, vmax=vmax)
    ax.set_xticks([])  # pixel indices aren't the point here; the frame is enough
    ax.set_yticks([])

    # everything drawn over the buttons goes in `mask_artists` so the checkbox can hide it
    mask_artists = []
    lane_labels = (
        False  # set when lane labels land above the axes (title needs clearance)
    )
    for _, r in df.iterrows():
        circ = plt.Circle((r.x, r.y), radius, ec="cyan", fc="none", lw=0.8)
        ax.add_patch(circ)
        (dot,) = ax.plot(r.x, r.y, "r.", ms=2)
        mask_artists += [circ, dot]

    if show_tags and "tag" in df.columns:
        if per_button:
            for _, r in df.iterrows():
                if str(r.tag):
                    mask_artists.append(
                        ax.annotate(
                            str(r.tag),
                            (r.x, r.y),
                            color="white",
                            fontsize=6,
                            ha="center",
                            va="center",
                            path_effects=halo,
                        )
                    )
        else:
            # top button of each column, left->right; merge consecutive same-tag columns
            tops = df.loc[df.groupby("mark_col")["mark_row"].idxmin()].sort_values(
                "mark_col"
            )
            groups = []
            for _, r in tops.iterrows():
                tag = str(r.tag)
                if tag and groups and groups[-1]["tag"] == tag:
                    groups[-1]["xs"].append(r.x)
                elif tag:
                    groups.append({"tag": tag, "xs": [r.x]})
                else:
                    groups.append({"tag": "", "xs": []})  # blank breaks the run

            # number each *distinct* tag once, in left->right order of first appearance,
            # so a tag that recurs in a later block reuses its number and the legend
            # stays short
            numbers = {}
            if tag_mode == "number":
                for grp in groups:
                    if grp["tag"] and grp["tag"] not in numbers:
                        numbers[grp["tag"]] = len(numbers) + 1

            for grp in groups:
                if not grp["tag"]:
                    continue
                xc = float(np.mean(grp["xs"]))
                text = str(numbers[grp["tag"]]) if tag_mode == "number" else grp["tag"]
                # x tracks the lane in data coords, y is pinned to the top of the axes
                # (get_xaxis_transform blends the two), so labels sit off the image entirely
                # — no halo needed, and nothing is drawn over the buttons
                ax.annotate(
                    text,
                    (xc, 1.0),
                    xycoords=ax.get_xaxis_transform(),
                    textcoords="offset points",
                    xytext=(0, 4),
                    fontsize=9 if tag_mode == "number" else 8,
                    rotation=tag_rotation,
                    ha="center",
                    va="bottom",
                    annotation_clip=False,
                )
                lane_labels = True

            if numbers:
                # label-only legend (no handles) outside the axes: "1  WT", "2  S92A", ...
                handles = [
                    Line2D([], [], ls="", label=f"{n}  {tag}")
                    for tag, n in numbers.items()
                ]
                ax.legend(
                    handles=handles,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    handlelength=0,
                    handletextpad=0,
                    borderaxespad=0,
                    frameon=False,
                    fontsize=8,
                )
                fig.subplots_adjust(right=0.80)  # room for the legend beside the image

    # the title names the image being inspected; the how-to-use hint is a small caption down
    # by the controls, where it doesn't compete with the figure
    ax.set_title(name or "", fontsize=10, pad=18 if lane_labels else 6)
    fig.subplots_adjust(bottom=0.12)

    # bottom-left corner, below the slider row: centering it under the axes would land the
    # caption on top of the x tick labels
    fig.text(
        0.02,
        0.012,
        "adjust x0/y0/pitch/angle/radius until circles sit on buttons",
        ha="left",
        va="bottom",
        fontsize=7,
        color="0.45",
    )

    # live vmax slider (vmin stays pinned at 0); lower vmax => brighter display
    ax_vmax = fig.add_axes([0.25, 0.04, 0.5, 0.02])
    vmax_slider = Slider(ax_vmax, "vmax", 0.0, data_hi, valinit=vmax)

    def _on_vmax(val):
        im.set_clim(
            vmax=max(val, 1e-9)
        )  # keep vmax just above vmin=0 so the range never inverts
        fig.canvas.draw_idle()

    vmax_slider.on_changed(_on_vmax)

    # checkbox to hide the overlay drawn over the buttons, to inspect the raw image
    if not show_masks:
        for a in mask_artists:
            a.set_visible(False)
    ax_chk = fig.add_axes([0.01, 0.90, 0.14, 0.06])
    ax_chk.set_axis_off()
    mask_check = CheckButtons(ax_chk, ["show masks"], [show_masks])

    def _on_toggle_masks(label):
        vis = mask_check.get_status()[0]
        for a in mask_artists:
            a.set_visible(vis)
        fig.canvas.draw_idle()

    mask_check.on_clicked(_on_toggle_masks)

    # keep refs so the widgets aren't garbage-collected
    fig._vmax_slider = vmax_slider
    fig._mask_check = mask_check
    return fig, ax


def show_chamber_timeseries(
    stack,
    cx,
    cy,
    *,
    chambers=None,
    frames=None,
    radius=12,
    bg_inner=None,
    bg_outer=None,
    pad=6,
    n_chambers=5,
    n_frames=5,
    seconds=None,
    time_name="seconds",
    vmax_pct=99.5,
    cmap="gray",
    figsize=None,
    drop_empty=True,
    labels=None,
):
    """QC grid (rows = chambers, cols = timepoints) of chamber crops with the ROI
    (+ annulus) overlay, to check mask alignment over time / drift. `cx`,`cy` are grid
    centers (e.g. bilinear_grid(corners, n_rows, n_cols)); pass the same
    `radius`/`bg_inner`/`bg_outer` you'll quantify with. Only the displayed frames are read,
    so it's safe on huge stacks.

    `chambers` is a list of (mark_row, mark_col) (default: n_chambers sampled across the
    chip); `frames` is a list of frame indices (default: n_frames spread over time).
    `labels` titles the columns, one per stack frame (e.g. acquisition names); the default
    is each frame's time. Returns (fig, axs)."""
    cx = np.asarray(cx, float)
    cy = np.asarray(cy, float)
    if stack.ndim != 3:
        raise ValueError(
            f"expected a 3-D (time, y, x) stack, got shape {tuple(stack.shape)}."
        )

    T, H, W = stack.shape
    if seconds is None:
        seconds = time_coord(stack, time_name)
    if drop_empty and seconds is not None:  # trim trailing unused (seconds==0) slots
        T = min(T, n_used_frames(seconds))

    # default picks: frames spread over time, chambers spread down a diagonal of the chip
    if frames is None:
        frames = np.unique(np.linspace(0, T - 1, n_frames).round().astype(int))
    frames = [int(t) for t in frames if 0 <= int(t) < T]
    if chambers is None:
        nr, nc = cx.shape
        ri = np.unique(np.linspace(0, nr - 1, n_chambers).round().astype(int))
        ci = np.linspace(0, nc - 1, len(ri)).round().astype(int)
        chambers = list(zip(ri.tolist(), ci.tolist()))

    # read only the displayed frames (xarray lazy-friendly)
    if hasattr(stack, "isel"):
        sel = np.asarray(stack.isel({stack.dims[0]: frames}))
    else:
        sel = np.asarray(stack)[frames]

    w = int(round((bg_outer if bg_outer else radius) + pad))
    nrows, ncols = len(chambers), len(frames)
    figsize = figsize or (2.0 * ncols, 2.0 * nrows)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for i, (rr, cc) in enumerate(chambers):
        xc, yc = cx[rr, cc], cy[rr, cc]
        xi, yi = int(round(xc)), int(round(yc))
        x0, y0 = max(0, xi - w), max(0, yi - w)
        x1, y1 = min(W, xi + w + 1), min(H, yi + w + 1)
        crops = sel[:, y0:y1, x0:x1]
        vmin, vmax = np.percentile(crops, 1), np.percentile(crops, vmax_pct)
        for j, t in enumerate(frames):
            ax = axs[i][j]
            ax.imshow(crops[j], cmap=cmap, vmin=vmin, vmax=vmax)
            cxl, cyl = xc - x0, yc - y0
            ax.add_patch(plt.Circle((cxl, cyl), radius, ec="yellow", fc="none", lw=1.0))
            if bg_inner and bg_outer:
                ax.add_patch(
                    plt.Circle(
                        (cxl, cyl), bg_inner, ec="orange", fc="none", lw=0.6, ls=":"
                    )
                )
                ax.add_patch(
                    plt.Circle(
                        (cxl, cyl), bg_outer, ec="orange", fc="none", lw=0.6, ls=":"
                    )
                )
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                if labels is not None:
                    title = str(labels[t])
                elif seconds is not None:
                    title = f"{seconds[t]:.0f}s"
                else:
                    title = f"t={t}"
                ax.set_title(title, fontsize=9)
        axs[i][0].set_ylabel(f"({cc},{rr})", fontsize=9)  # col,row

    fig.tight_layout()
    return fig, axs


# Plot style shared by every figure. 4PL fits and their vmax tangent share an orange/peach
# palette; straight-line fits are gray, so the two models are told apart at a glance.
# 4PL curve light, its vmax tangent dark, so the two read apart at a glance
FIT_COLORS = {"4pl": "#F7C59F", "vmax": "#B54708", "line": "0.45"}


def set_plot_style(font="Arial"):
    """Global matplotlib style for every plot in the pipeline: `font` for all text, and no
    top/right spines. Call once, after importing matplotlib (e.g. in the notebook setup)."""
    plt.rcParams.update(
        {
            "font.family": font,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,  # editable text in saved PDFs
        }
    )


def show_masks(
    img,
    cx,
    cy,
    *,
    radius,
    bg_inner=None,
    bg_outer=None,
    zoom=None,
    title="",
    max_px=2048,
    vmax_pct=99.5,
    zoom_px=70,
):
    """QC figure of chamber masks on an image: the whole chip (strided to <= `max_px`) with
    every ROI (yellow) and background annulus (orange), plus full-resolution zooms of the
    chambers in `zoom` (list of (mark_row, mark_col); default: the 4 corners and the
    center). The chip view's contrast comes from the chambers' area; each zoom has its own.
    One collection per ring type, so a 1,792-chamber chip draws fast. Returns fig."""
    from matplotlib.collections import EllipseCollection

    img = np.asarray(img)
    cx, cy = np.asarray(cx, float), np.asarray(cy, float)
    nr, nc = cx.shape
    zoom = zoom or [
        (0, 0),
        (0, nc - 1),
        (nr // 2, nc // 2),
        (nr - 1, 0),
        (nr - 1, nc - 1),
    ]

    # contrast from the chambers' area only: bright chip edges and debris elsewhere would
    # otherwise set the ceiling and render the chambers black
    pad = int(bg_outer or radius)
    y0, y1 = int(max(0, cy.min() - pad)), int(min(img.shape[0], cy.max() + pad))
    x0, x1 = int(max(0, cx.min() - pad)), int(min(img.shape[1], cx.max() + pad))
    lo, hi = np.percentile(img[y0:y1:4, x0:x1:4], [1, vmax_pct])

    fig = plt.figure(figsize=(11, 7.5))
    grid = fig.add_gridspec(len(zoom), 2, width_ratios=[2.6, 0.55])
    ax = fig.add_subplot(grid[:, 0])
    step = max(1, int(np.ceil(max(img.shape) / max_px)))
    shown = img[::step, ::step]
    ax.imshow(
        shown,
        cmap="gray",
        vmin=lo,
        vmax=hi,
        interpolation="nearest",
        extent=(-0.5, shown.shape[1] * step - 0.5, shown.shape[0] * step - 0.5, -0.5),
    )

    def rings(ax, pts, r, color, lw, ls="-"):
        d = np.full(len(pts), 2.0 * r)
        ax.add_collection(
            EllipseCollection(
                d,
                d,
                np.zeros(len(pts)),
                units="xy",
                offsets=pts,
                offset_transform=ax.transData,
                facecolors="none",
                edgecolors=color,
                linewidths=lw,
                linestyles=ls,
            )
        )

    pts = np.column_stack([cx.ravel(), cy.ravel()])
    rings(ax, pts, radius, "yellow", 0.4)
    for r in (bg_inner, bg_outer):
        if r:
            rings(ax, pts, r, "orange", 0.3)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # full-resolution crops around the zoom chambers, with the same contrast
    for k, (r, c) in enumerate(zoom):
        zax = fig.add_subplot(grid[k, 1])
        x, y = int(round(cx[r, c])), int(round(cy[r, c]))
        y0, x0 = max(0, y - zoom_px), max(0, x - zoom_px)
        crop = img[y0 : y + zoom_px + 1, x0 : x + zoom_px + 1]
        # each zoom on its own contrast, so a dim corner chamber is still readable
        zlo, zhi = np.percentile(crop, [1, vmax_pct])
        zax.imshow(crop, cmap="gray", vmin=zlo, vmax=zhi, interpolation="nearest")
        center = np.array([[cx[r, c] - x0, cy[r, c] - y0]])
        rings(zax, center, radius, "yellow", 1.0)
        for rad in (bg_inner, bg_outer):
            if rad:
                rings(zax, center, rad, "orange", 0.8, ":")
        zax.set_xticks([])
        zax.set_yticks([])
        zax.set_ylabel(f"({c},{r})", fontsize=8)  # col,row
    fig.tight_layout()
    return fig


def chip_heatmaps(
    df,
    value,
    *,
    panel=None,
    n_rows=None,
    n_cols=None,
    cmap="viridis",
    pct=(2, 98),
    center=None,
    title="",
    ncols=6,
    blanks=True,
):
    """QC figure: one chip-shaped heatmap (mark_row x mark_col) of `value` per level of
    `panel` (e.g. stage, bzp_uM, frame), on a shared color scale clipped to the `pct`
    percentiles (or symmetric about `center`, e.g. 0 for signed rates). BLANK chambers are
    outlined in red. Returns fig."""
    n_rows = n_rows or int(df["mark_row"].max()) + 1
    n_cols = n_cols or int(df["mark_col"].max()) + 1
    groups = list(df.groupby(panel, sort=True)) if panel else [("", df)]
    vals = df[value].to_numpy(float)
    lo, hi = np.nanpercentile(vals, pct)
    if center is not None:
        half = max(abs(lo - center), abs(hi - center))
        lo, hi = center - half, center + half

    n = len(groups)
    cols = min(ncols, n)
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(
        rows, cols, figsize=(2.0 * cols + 1, 3.3 * rows), squeeze=False
    )
    for ax, (level, g) in zip(axs.flat, groups):
        grid = np.full((n_rows, n_cols), np.nan)
        grid[g["mark_row"].to_numpy(int), g["mark_col"].to_numpy(int)] = g[value]
        im = ax.imshow(grid, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        if blanks and "tag" in g.columns:
            b = g[g["tag"] == "BLANK"]
            ax.scatter(
                b["mark_col"],
                b["mark_row"],
                s=14,
                marker="s",
                fc="none",
                ec="red",
                lw=0.8,
            )
        ax.set_title(
            f"{level:g}" if isinstance(level, (int, float, np.number)) else str(level),
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axs.flat[n:]:
        ax.set_visible(False)
    fig.colorbar(im, ax=axs, shrink=0.6, label=value)
    fig.suptitle(title or value, fontsize=10)
    return fig


def chip_sparklines(
    df,
    value,
    *,
    x="seconds",
    curves=None,
    lines=None,
    tangents=None,
    reference=None,
    color=None,
    labels=None,
    rows=None,
    cols=None,
    n_rows=None,
    n_cols=None,
    title="",
    logx=False,
    sharey=True,
    pct=(1, 99),
    ylim=None,
    span=None,
    cell_w=0.42,
    cell_h=0.32,
):
    """QC figure: every chamber's `value` vs `x` drawn as a small plot at its position on the
    chip, so a whole chip (or a block of it) fits on one page.

    `rows` / `cols` = (first, last) mark_row / mark_col to draw only a block of the chip
    (default: all); save_sparkline_pages() pages through the chip in readable blocks.
    y scaling, via `sharey`:
      True     one absolute y range for every cell (`ylim`, or the `pct` percentiles of all
               values): levels compare, but small changes on different baselines look flat;
      "span"   one y SPAN for every cell (`span`, or the 99th percentile of the per-chamber
               ranges), centered on each chamber's own median: changes and slopes compare
               across the chip and stay visible;
      False    each chamber on its own range (shows shape, exaggerates noise).
    Points outside a cell's range are pinned to its edge. Fits are drawn behind the points.

    `curves` is an optional {(mark_row, mark_col): (xs, ys)} of fitted curves drawn over the
    points; `lines` the same, drawn thinner (e.g. straight-line fits); `tangents` the same
    for thin solid lines in FIT_COLORS["vmax"] (e.g. the tangent at a 4PL's vmax). Fitted lines
    end where they leave the y range;
    `reference` an optional DataFrame with columns `x` and `value` drawn faint gray behind
    the data in EVERY cell, on that cell's scale (e.g. all BLANK chambers' rates, showing the
    floor); `color` an optional {(mark_row, mark_col): color} (e.g. by model); `labels` an optional
    {(mark_row, mark_col): text} written in each cell (e.g. variant and rate). Every cell
    also shows its position "col,row" small in the bottom-right corner and, if `df` has a
    'tag' column, its variant in the bottom-left. Points have a thin black outline. Points, curves
    and tangents are drawn as one collection each, so it stays fast. Returns fig."""
    from matplotlib.collections import LineCollection

    n_rows = n_rows or int(df["mark_row"].max()) + 1
    n_cols = n_cols or int(df["mark_col"].max()) + 1
    r0, r1 = rows if rows is not None else (0, n_rows - 1)
    c0, c1 = cols if cols is not None else (0, n_cols - 1)
    nr, nc = r1 - r0 + 1, c1 - c0 + 1
    shared = (
        ylim if ylim is not None else np.nanpercentile(df[value].to_numpy(float), pct)
    )
    if sharey == "span" and span is None:
        ranges = df.groupby(["mark_row", "mark_col"])[value].agg(
            lambda v: v.max() - v.min()
        )
        span = float(np.nanpercentile(ranges, 99))

    fig, ax = plt.subplots(figsize=(cell_w * nc + 0.5, cell_h * nr + 0.6))
    big = cell_w >= 0.8  # paged view: bigger marks and per-cell labels
    block = df[df["mark_row"].between(r0, r1) & df["mark_col"].between(c0, c1)]
    TEXT_BOX = dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.4)  # faint
    pts_xy, pts_c, segs, seg_c, tan_segs, line_segs, line_c = [], [], [], [], [], [], []
    ref_xy = []
    if reference is not None:
        ref_x = reference[x].to_numpy(float)
        ref_x = np.log10(ref_x) if logx else ref_x
        ref_y = reference[value].to_numpy(float)
    for (r, c), g in block.groupby(["mark_row", "mark_col"]):
        xs, ys = g[x].to_numpy(float), g[value].to_numpy(float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.sum() < 2:
            continue
        xs, ys = xs[ok], ys[ok]
        if logx:
            xs = np.log10(xs)
        x0, x1 = xs.min(), xs.max()
        fit = curves.get((r, c)) if curves else None
        if fit is None and lines:
            fit = lines.get((r, c))
            fit_is_line = fit is not None
        else:
            fit_is_line = False
        if reference is not None:  # the cell's x range covers the reference too
            x0, x1 = min(x0, np.nanmin(ref_x)), max(x1, np.nanmax(ref_x))
        if (
            fit is not None
        ):  # the cell's x range covers the fitted curve too (e.g. from 0)
            fx_all = np.log10(fit[0]) if logx else np.asarray(fit[0], float)
            x0, x1 = min(x0, np.nanmin(fx_all)), max(x1, np.nanmax(fx_all))
        if sharey == "span":
            mid = np.nanmedian(ys)
            y0, y1 = mid - span / 2, mid + span / 2
        elif sharey:
            y0, y1 = shared
        else:
            y_all = np.concatenate([ys, fit[1]]) if fit is not None else ys
            y0, y1 = np.nanmin(y_all), np.nanmax(y_all)

        def to_cell(u, v, clip=True):
            # the y range maps onto the cell with a margin (room for a label on top).
            # Points beyond it are pinned to the cell edge (clip=True); fitted lines are
            # cut where they leave the range instead (clip=False), so a line ends at the
            # edge rather than running along it.
            uu = (c - c0) + 0.08 + 0.84 * (np.asarray(u, float) - x0) / ((x1 - x0) or 1)
            frac = (np.asarray(v, float) - y0) / ((y1 - y0) or 1)
            if clip:
                frac = np.clip(frac, 0, 1)
            else:
                inside = (frac >= 0) & (frac <= 1)
                uu, frac = uu[inside], frac[inside]
            vv = (r - r0) + 0.92 - (0.66 if big else 0.84) * frac
            return np.column_stack([uu, vv])

        def fit_xy(curve):
            """A fitted (xs, ys) in cell coordinates, densified so the cut at the edge is
            tight; None if fewer than 2 points stay inside the range."""
            fx = np.log10(curve[0]) if logx else np.asarray(curve[0], float)
            fy = np.asarray(curve[1], float)
            dense = np.linspace(fx.min(), fx.max(), 200)
            xy = to_cell(dense, np.interp(dense, fx, fy), clip=False)
            return xy if len(xy) >= 2 else None

        col = (color or {}).get((r, c), "0.2")
        if reference is not None:
            ref_xy.append(to_cell(ref_x, ref_y))
        pts_xy.append(to_cell(xs, ys))
        pts_c += [col] * len(xs)
        fit_cell = fit_xy(fit) if fit is not None else None
        if fit_cell is not None:
            (line_segs if fit_is_line else segs).append(fit_cell)
            (line_c if fit_is_line else seg_c).append(col)
        tan = tangents.get((r, c)) if tangents else None
        tan_cell = fit_xy(tan) if tan is not None else None
        if tan_cell is not None:
            tan_segs.append(tan_cell)
        if labels and (r, c) in labels:
            ax.text(
                (c - c0) + 0.06,
                (r - r0) + 0.05,
                labels[(r, c)],
                fontsize=5 if big else 3,
                va="top",
                ha="left",
                color="0.25",
                bbox=TEXT_BOX,
            )
        # variant in the bottom-left corner
        if "tag" in g.columns:
            ax.text(
                (c - c0) + 0.03,
                (r - r0) + 0.97,
                str(g["tag"].iloc[0]),
                fontsize=4 if big else 2,
                va="bottom",
                ha="left",
                color="0.35",
                bbox=TEXT_BOX,
            )
        # position "col,row" in the bottom-right corner of every cell
        ax.text(
            (c - c0) + 0.97,
            (r - r0) + 0.97,
            f"{c},{r}",
            fontsize=4 if big else 2,
            va="bottom",
            ha="right",
            color="0.55",
            bbox=TEXT_BOX,
        )

    if ref_xy:  # the reference (e.g. blank rates): faint gray, behind everything
        rxy = np.concatenate(ref_xy)
        ax.scatter(
            rxy[:, 0], rxy[:, 1], s=4 if big else 0.4, c="0.8", linewidths=0, zorder=1
        )
    if pts_xy:
        xy = np.concatenate(pts_xy)
        # thin black outline on every point
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=7 if big else 0.8,
            c=pts_c,
            edgecolors="k",
            linewidths=0.25 if big else 0.08,
        )
    lw = 0.9 if big else 0.5
    if segs:
        ax.add_collection(LineCollection(segs, colors=seg_c, linewidths=lw))
    if line_segs:  # straight-line fits: thinner
        ax.add_collection(
            LineCollection(line_segs, colors=line_c, linewidths=lw * 0.45)
        )
    if tan_segs:
        ax.add_collection(
            LineCollection(tan_segs, colors=FIT_COLORS["vmax"], linewidths=lw * 0.5)
        )

    # z-order: fits (2) behind points (3), labels (5) on top
    from matplotlib.collections import PathCollection

    for coll in ax.collections:
        coll.set_zorder(3 if isinstance(coll, PathCollection) else 2)
    if ref_xy:
        ax.collections[0].set_zorder(1)  # the reference scatter was drawn first
    for txt in ax.texts:  # labels above everything
        txt.set_zorder(5)
    ax.set_xlim(0, nc)
    ax.set_ylim(nr, 0)
    tick_fs = 6 if big else 4
    ax.set_xticks(
        np.arange(nc) + 0.5, [str(i) for i in range(c0, c1 + 1)], fontsize=tick_fs
    )
    ax.set_yticks(
        np.arange(nr) + 0.5, [str(i) for i in range(r0, r1 + 1)], fontsize=tick_fs
    )
    ax.set_xticks(np.arange(nc + 1), minor=True)
    ax.set_yticks(np.arange(nr + 1), minor=True)
    ax.grid(which="minor", color="0.85", lw=0.4)
    ax.tick_params(which="both", length=0)
    ax.set_xlabel("mark_col", fontsize=tick_fs)
    ax.set_ylabel("mark_row", fontsize=tick_fs)
    if sharey == "span":
        note = f"shared y span: {span:.3g}, centered per chamber"
    elif sharey:
        note = f"shared y: {shared[0]:.3g} to {shared[1]:.3g}"
    else:
        note = "y scaled per chamber"
    title = f"{title}   ({note})" if title else note
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def save_sparkline_pages(
    df,
    value,
    path,
    *,
    rows_per_page=8,
    cols_per_page=16,
    title="",
    cell_w=1.1,
    cell_h=0.95,
    **kwargs,
):
    """Page through the chip in readable blocks of chip_sparklines (`rows_per_page` x
    `cols_per_page` chambers, large cells with labels) and save them as one multi-page PDF
    at `path`. The y range is computed once over the whole chip, so pages compare. Extra
    kwargs go to chip_sparklines. Returns the number of pages."""
    from matplotlib.backends.backend_pdf import PdfPages

    n_rows = kwargs.pop("n_rows", None) or int(df["mark_row"].max()) + 1
    n_cols = kwargs.pop("n_cols", None) or int(df["mark_col"].max()) + 1
    if kwargs.get("sharey", True) == "span" and kwargs.get("span") is None:
        ranges = df.groupby(["mark_row", "mark_col"])[value].agg(
            lambda v: v.max() - v.min()
        )
        kwargs["span"] = float(np.nanpercentile(ranges, 99))
    elif kwargs.get("sharey", True) and kwargs.get("ylim") is None:
        kwargs["ylim"] = np.nanpercentile(
            df[value].to_numpy(float), kwargs.pop("pct", (1, 99))
        )
    pages = 0
    # write under a temporary name and rename when complete, so a half-written PDF
    # (unreadable until closed) never sits at `path`
    path = Path(path)
    partial = path.with_suffix(".partial.pdf")
    with PdfPages(partial) as pdf:
        for r0 in range(0, n_rows, rows_per_page):
            for c0 in range(0, n_cols, cols_per_page):
                r1, c1 = (
                    min(r0 + rows_per_page, n_rows) - 1,
                    min(c0 + cols_per_page, n_cols) - 1,
                )
                fig = chip_sparklines(
                    df,
                    value,
                    rows=(r0, r1),
                    cols=(c0, c1),
                    n_rows=n_rows,
                    n_cols=n_cols,
                    cell_w=cell_w,
                    cell_h=cell_h,
                    title=f"{title}  rows {r0}-{r1}, cols {c0}-{c1}",
                    **kwargs,
                )
                pdf.savefig(fig)
                plt.close(fig)
                pages += 1
    partial.replace(path)
    return pages
