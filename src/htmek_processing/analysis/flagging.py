"""Flagging chambers: the shared JSON format and the interactive flaggers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ..processing.io import _STD_RE, load_standard_images, n_used_frames, time_coord


def save_flags(path, flagged, tags=None):
    """Write a set of flagged (mark_row, mark_col) to JSON (records: mark_row, mark_col,
    tag)."""
    import json

    tags = tags or {}
    recs = [
        {"mark_row": r, "mark_col": c, "tag": tags.get((r, c), "")}
        for (r, c) in sorted(flagged)
    ]
    Path(path).write_text(json.dumps(recs, indent=2))
    return str(path)


def load_flags(path):
    """Read a flags JSON written by save_flags / ChamberFlagger -> set of
    (mark_row, mark_col)."""
    import json

    return {
        (int(d["mark_row"]), int(d["mark_col"]))
        for d in json.loads(Path(path).read_text())
    }


def apply_flags(df, flagged, *, drop=False):
    """Add a boolean 'flagged' column to df from a set of (mark_row, mark_col); drop=True
    filters."""
    df = df.copy()
    df["flagged"] = [
        (int(t.mark_row), int(t.mark_col)) in flagged for t in df.itertuples()
    ]
    return df[~df["flagged"]].copy() if drop else df


def _thumb_png(arr2d, vmin, vmax, size=64, circle=None, annulus=None):
    """Normalize a 2-D crop to a grayscale PNG (bytes), upscaled to `size` px. If
    `circle=(cx, cy, r)` (crop coords) draw the ROI outline (cyan);
    `annulus=(inner, outer)` adds the background-ring outlines (orange) at the same
    center."""
    import io
    from PIL import Image as _PILImage, ImageDraw as _ImageDraw

    arr2d = np.asarray(arr2d, float)
    h, w = arr2d.shape
    a = np.clip((arr2d - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    im = _PILImage.fromarray((a * 255).astype(np.uint8)).convert("RGB")

    sx = sy = 1.0
    if size:
        sx, sy = size / w, size / h
        im = im.resize((size, size), _PILImage.NEAREST)

    if circle is not None:
        cxr, cyr, rr = circle
        cx, cy = cxr * sx, cyr * sy
        draw = _ImageDraw.Draw(im)

        def _ring(rad, color):
            rx, ry = rad * sx, rad * sy
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=color)

        _ring(rr, (0, 255, 255))
        for rad in annulus or ():
            _ring(rad, (255, 165, 0))

    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


class ChamberFlagger:
    """In-notebook clickable grid of chamber thumbnails for manual QC (e.g. flag glare).

    Renders inline via ipywidgets (independent of the matplotlib backend, so it coexists
    with `%matplotlib inline`). Thumbnails use a global intensity scale, so glare chambers
    read as blown-out.

        flagger = ChamberFlagger(img, df)  # df from GridAdjuster.to_df() / manual_grid()
        flagger                            # displays the grid; click a tile to flag (red)
        clean = flagger.apply(df, drop=True)  # df with flagged rows removed

    `flagger.flagged` is the set of flagged (mark_row, mark_col); `flagger.apply(df)` adds
    a boolean 'flagged' column (drop=True returns only the kept rows)."""

    def __init__(
        self,
        img,
        df,
        *,
        save_path=None,
        crop_radius=None,
        thumb_px=64,
        vmin_pct=1,
        vmax_pct=99,
        ncols=None,
        show_mask=True,
    ):
        import ipywidgets as W

        self.img = np.asarray(img).squeeze()
        self.df = df
        self.save_path = Path(save_path) if save_path else None
        self._loading = False

        cx, cy, radius = df.attrs["centers"]
        self.ncols = ncols or int(df["mark_col"].max()) + 1
        r = int(crop_radius if crop_radius is not None else round(1.6 * radius))
        H, Wd = self.img.shape
        vmin = np.percentile(self.img, vmin_pct)
        vmax = np.percentile(self.img, vmax_pct)

        # one thumbnail + toggle button per chamber, in row-major order
        self._toggles, self._tags = {}, {}
        cells = []
        for row in df.sort_values(["mark_row", "mark_col"]).itertuples():
            key = (int(row.mark_row), int(row.mark_col))
            self._tags[key] = str(getattr(row, "tag", ""))
            x, y = int(round(row.x)), int(round(row.y))
            x0, y0 = max(0, x - r), max(0, y - r)
            crop = self.img[y0 : min(H, y + r + 1), x0 : min(Wd, x + r + 1)]
            circle = (x - x0, y - y0, radius) if show_mask else None
            image = W.Image(
                value=_thumb_png(crop, vmin, vmax, thumb_px, circle),
                format="png",
                layout=W.Layout(width=f"{thumb_px}px", height=f"{thumb_px}px"),
            )
            tog = W.ToggleButton(
                value=False,
                description=f"{key[1]},{key[0]}",  # col,row
                tooltip=self._tags[key],
                layout=W.Layout(width=f"{thumb_px}px", height="22px"),
            )
            tog.observe(self._on_toggle, names="value")
            self._toggles[key] = tog
            cells.append(W.VBox([image, tog], layout=W.Layout(margin="1px")))

        self._count = W.HTML()
        grid = W.GridBox(
            children=cells,
            layout=W.Layout(
                grid_template_columns=f"repeat({self.ncols}, max-content)",
                grid_gap="2px",
            ),
        )
        self._box = W.VBox(
            [
                W.HTML(
                    "<b>Click chambers to flag for removal</b> (red = flagged; auto-saves)"
                ),
                self._count,
                grid,
            ]
        )

        if self.save_path and self.save_path.exists():
            self.load()
        self._update_count()

    def _on_toggle(self, change):
        change["owner"].button_style = "danger" if change["new"] else ""
        self._update_count()
        if self.save_path is not None and not self._loading:
            self.save()

    def _update_count(self):
        suffix = f" → {self.save_path.name}" if self.save_path else ""
        self._count.value = (
            f"flagged: {len(self.flagged)} / {len(self._toggles)}{suffix}"
        )

    @property
    def flagged(self):
        return {k for k, t in self._toggles.items() if t.value}

    def save(self, path=None):
        """Write flagged decisions to JSON (records: mark_row, mark_col, tag). Returns the
        path."""
        import json

        p = Path(path) if path else self.save_path
        if p is None:
            raise ValueError(
                "No save_path set; pass save_path=... to ChamberFlagger or save()."
            )

        recs = [
            {"mark_row": rr, "mark_col": cc, "tag": self._tags.get((rr, cc), "")}
            for (rr, cc) in sorted(self.flagged)
        ]

        p.write_text(json.dumps(recs, indent=2))
        self.save_path = p
        return str(p)

    def load(self, path=None):
        """Load flagged decisions from JSON and set the toggles accordingly."""
        import json

        p = Path(path) if path else self.save_path
        flagged = {
            (int(d["mark_row"]), int(d["mark_col"]))
            for d in json.loads(Path(p).read_text())
        }

        self._loading = True
        try:
            for key, tog in self._toggles.items():
                tog.value = key in flagged
        finally:
            self._loading = False

        self._update_count()
        return self.flagged

    def apply(self, df=None, *, drop=False):
        df = (self.df if df is None else df).copy()
        flagged = self.flagged
        df["flagged"] = [
            (int(t.mark_row), int(t.mark_col)) in flagged for t in df.itertuples()
        ]
        return df[~df["flagged"]].copy() if drop else df

    def _ipython_display_(self):
        from IPython.display import display

        display(self._box)


class ChamberFlaggerMpl:
    """matplotlib/Qt clickable chamber grid — use when inline ipywidgets won't render.
    Run `%matplotlib qt` first; a window opens with one thumbnail per chamber. Click a tile
    to toggle its flag (red border). Same JSON save format as ChamberFlagger, so the two
    are interchangeable via the file.

        %matplotlib qt
        fl = ChamberFlaggerMpl(img, df, save_path=f"{expt}/glare_flags.json")
        clean = fl.apply(df, drop=True)      # after flagging in the window
    """

    def __init__(
        self,
        img,
        df,
        *,
        save_path=None,
        crop_radius=None,
        vmin_pct=1,
        vmax_pct=99,
        figsize=(10, 10),
        show_mask=True,
    ):
        self.img = np.asarray(img).squeeze()
        self.df = df
        self.save_path = Path(save_path) if save_path else None

        cx, cy, radius = df.attrs["centers"]
        nr = int(df["mark_row"].max()) + 1
        nc = int(df["mark_col"].max()) + 1
        r = int(crop_radius if crop_radius is not None else round(1.6 * radius))
        H, Wd = self.img.shape
        vmin = np.percentile(self.img, vmin_pct)
        vmax = np.percentile(self.img, vmax_pct)
        cen = {
            (int(t.mark_row), int(t.mark_col)): (t.x, t.y, str(getattr(t, "tag", "")))
            for t in df.itertuples()
        }

        self._flagged = set()
        self._tags = {k: v[2] for k, v in cen.items()}
        self.fig, axes = plt.subplots(nr, nc, figsize=figsize)
        axes = np.atleast_2d(axes).reshape(nr, nc)
        self._ax2key = {}

        # one subplot per chamber position; positions missing from df stay blank
        for i in range(nr):
            for j in range(nc):
                ax = axes[i, j]
                x, y, _ = cen.get((i, j), (np.nan, np.nan, ""))
                if np.isfinite(x):
                    xi, yi = int(round(x)), int(round(y))
                    x0, y0 = max(0, xi - r), max(0, yi - r)
                    crop = self.img[y0 : min(H, yi + r + 1), x0 : min(Wd, xi + r + 1)]
                    ax.imshow(crop, cmap="gray", vmin=vmin, vmax=vmax)
                    if show_mask:
                        ax.add_patch(
                            plt.Circle(
                                (xi - x0, yi - y0), radius, ec="cyan", fc="none", lw=0.8
                            )
                        )
                ax.set_xticks([])
                ax.set_yticks([])
                self._ax2key[ax] = (i, j)

        self.fig.subplots_adjust(wspace=0.05, hspace=0.05)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        if self.save_path and self.save_path.exists():
            self._flagged = load_flags(self.save_path)
        self._refresh()

    def _on_click(self, e):
        key = self._ax2key.get(e.inaxes)
        if key is None:
            return
        self._flagged ^= {key}
        if self.save_path is not None:
            save_flags(self.save_path, self._flagged, self._tags)
        self._refresh()

    def _refresh(self):
        for ax, key in self._ax2key.items():
            on = key in self._flagged
            for sp in ax.spines.values():
                sp.set_color("red" if on else "0.6")
                sp.set_linewidth(3 if on else 0.5)
        suffix = f" → {self.save_path.name}" if self.save_path else ""
        self.fig.suptitle(
            f"click to flag (red = flagged) — "
            f"{len(self._flagged)}/{len(self._ax2key)}{suffix}"
        )
        self.fig.canvas.draw_idle()

    @property
    def flagged(self):
        return set(self._flagged)

    def save(self, path=None):
        return save_flags(path or self.save_path, self._flagged, self._tags)

    def apply(self, df=None, *, drop=False):
        return apply_flags(self.df if df is None else df, self._flagged, drop=drop)


class ChamberGridFlagger:
    """Clickable QC + flagging grid (use under %matplotlib qt). Laid out like
    show_chamber_timeseries: rows = chambers (all by default, one per row), columns = a
    few images (e.g. standard concentrations or assay timepoints), each crop with the ROI
    (+ annulus) overlay. Click any tile to flag/unflag that whole chamber (red border).

        flagger = ChamberGridFlagger(std_imgs, cx, cy, radius=R, bg_inner=BI, bg_outer=BO,
                                     save_path=f"{expt}/standards_flags.json")
        cal = mu.calibrate_chambers(flagger.apply(standards_df, drop=True))

    `images` is an ordered dict {label: 2-D image} or a list of 2-D images (`labels=`
    optional); if more than `n_cols` are given, that many are sampled across the range.
    Auto-saves to JSON (same format as ChamberFlagger); `.flagged` is the set of flagged
    (mark_row, mark_col)."""

    def __init__(
        self,
        images,
        cx,
        cy,
        *,
        labels=None,
        chambers=None,
        n_cols=5,
        radius=12,
        bg_inner=None,
        bg_outer=None,
        pad=6,
        save_path=None,
        vmax_pct=99.5,
        cmap="gray",
        cell=1.0,
    ):
        cx = np.asarray(cx, float)
        cy = np.asarray(cy, float)
        if isinstance(images, dict):
            labels, imgs = list(images.keys()), list(images.values())
        else:
            imgs = list(images)
            labels = list(labels) if labels is not None else list(range(len(imgs)))

        imgs = [np.asarray(im).squeeze() for im in imgs]
        if len(imgs) > n_cols:  # sample n_cols columns across the range
            idx = np.unique(np.linspace(0, len(imgs) - 1, n_cols).round().astype(int))
            imgs, labels = [imgs[k] for k in idx], [labels[k] for k in idx]

        nr, nc = cx.shape
        chambers = (
            chambers
            if chambers is not None
            else [(r, c) for r in range(nr) for c in range(nc)]
        )
        self.save_path = Path(save_path) if save_path else None
        self._flagged = set()
        w = int(round((bg_outer if bg_outer else radius) + pad))
        H, W = imgs[0].shape

        def crop(im, r, c):
            xi, yi = int(round(cx[r, c])), int(round(cy[r, c]))
            x0, y0 = max(0, xi - w), max(0, yi - w)
            sub = im[y0 : min(H, yi + w + 1), x0 : min(W, xi + w + 1)]
            return sub, (cx[r, c] - x0, cy[r, c] - y0)

        # per-image contrast limits, pooled over every shown chamber's crop
        vlims = [
            (
                np.percentile(
                    np.concatenate([crop(im, r, c)[0].ravel() for r, c in chambers]), 1
                ),
                np.percentile(
                    np.concatenate([crop(im, r, c)[0].ravel() for r, c in chambers]),
                    vmax_pct,
                ),
            )
            for im in imgs
        ]

        ncol = len(imgs)
        self.fig, axs = plt.subplots(
            len(chambers),
            ncol,
            squeeze=False,
            figsize=(cell * ncol, cell * len(chambers)),
        )
        self._ax2key, self._key2axes = {}, {}

        for i, (r, c) in enumerate(chambers):
            self._key2axes[(r, c)] = []
            for j, im in enumerate(imgs):
                ax = axs[i][j]
                sub, (cxl, cyl) = crop(im, r, c)
                ax.imshow(sub, cmap=cmap, vmin=vlims[j][0], vmax=vlims[j][1])
                ax.add_patch(
                    plt.Circle((cxl, cyl), radius, ec="yellow", fc="none", lw=0.8)
                )
                if bg_inner and bg_outer:
                    ax.add_patch(
                        plt.Circle(
                            (cxl, cyl), bg_inner, ec="orange", fc="none", lw=0.5, ls=":"
                        )
                    )
                    ax.add_patch(
                        plt.Circle(
                            (cxl, cyl), bg_outer, ec="orange", fc="none", lw=0.5, ls=":"
                        )
                    )
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(str(labels[j]), fontsize=8)
                self._ax2key[ax] = (r, c)
                self._key2axes[(r, c)].append(ax)
            axs[i][0].set_ylabel(
                f"{c},{r}", fontsize=7, rotation=0, ha="right", va="center"
            )  # col,row

        self.fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0.07, top=0.97)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        if self.save_path and self.save_path.exists():
            self._flagged = load_flags(self.save_path)
        self._refresh()

    def _on_click(self, e):
        key = self._ax2key.get(e.inaxes)
        if key is None:
            return
        self._flagged ^= {key}
        if self.save_path is not None:
            save_flags(self.save_path, self._flagged)
        self._refresh()

    def _refresh(self):
        for key, axes in self._key2axes.items():
            on = key in self._flagged
            for ax in axes:
                for sp in ax.spines.values():
                    sp.set_color("red" if on else "0.6")
                    sp.set_linewidth(2 if on else 0.4)

        self.fig.suptitle(
            f"click a chamber to flag (red) — {len(self._flagged)} flagged"
            + (f" → {self.save_path.name}" if self.save_path else "")
        )
        self.fig.canvas.draw_idle()

    @property
    def flagged(self):
        return set(self._flagged)

    def save(self, path=None):
        return save_flags(path or self.save_path, self._flagged)

    def apply(self, df, *, drop=False):
        return apply_flags(df, self._flagged, drop=drop)

    def __repr__(self):
        return f"<ChamberGridFlagger: {len(self._flagged)} flagged>"


class KineticFlagger:
    """Inline (ipywidgets) kinetics flagger: one row per chamber, a strip of timepoint
    thumbnails (ROI overlaid) plus a flag toggle that labels the chamber. Renders inline
    (no Qt) — run under `%matplotlib inline`. Same JSON flag format as the other flaggers.

        kf = KineticFlagger(stack, cx, cy, radius=R, save_path=f"{expt}/assay_flags.json")
        kf  # scrollable grid; click a chamber's button to flag
        clean = kf.apply(assays_df, drop=True)

    Only the displayed frames are read (safe on big stacks); trailing unused (seconds==0)
    frames are trimmed. `.flagged` is the set of flagged (mark_row, mark_col)."""

    def __init__(
        self,
        stack,
        cx,
        cy,
        *,
        frames=None,
        n_frames=5,
        chambers=None,
        radius=12,
        bg_inner=None,
        bg_outer=None,
        pad=6,
        thumb_px=70,
        seconds=None,
        time_name="seconds",
        vmax_pct=99.5,
        save_path=None,
        drop_empty=True,
        height="700px",
    ):
        import ipywidgets as W

        cx = np.asarray(cx, float)
        cy = np.asarray(cy, float)
        if stack.ndim != 3:
            raise ValueError(
                f"expected a 3-D (time, y, x) stack, got shape {tuple(stack.shape)}."
            )

        T, H, Wd = stack.shape
        if seconds is None:
            seconds = time_coord(stack, time_name)
        if drop_empty and seconds is not None:
            T = min(T, n_used_frames(seconds))
        if frames is None:
            frames = np.unique(np.linspace(0, T - 1, n_frames).round().astype(int))
        frames = [int(t) for t in frames if 0 <= int(t) < T]

        nr, nc = cx.shape
        chambers = (
            chambers
            if chambers is not None
            else [(r, c) for r in range(nr) for c in range(nc)]
        )
        self.save_path = Path(save_path) if save_path else None
        self._loading = False
        self._flagged = set()

        # read only the displayed frames (xarray lazy-friendly)
        sel = (
            np.asarray(stack.isel({stack.dims[0]: frames}))
            if hasattr(stack, "isel")
            else np.asarray(stack)[frames]
        )
        col_labels = [
            f"{seconds[t]:.0f}s" if seconds is not None else f"t={t}" for t in frames
        ]
        w = int(round((bg_outer if bg_outer else radius) + pad))

        def crop(j, r, c):
            xi, yi = int(round(cx[r, c])), int(round(cy[r, c]))
            x0, y0 = max(0, xi - w), max(0, yi - w)
            sub = sel[j, y0 : min(H, yi + w + 1), x0 : min(Wd, xi + w + 1)]
            return sub, (cx[r, c] - x0, cy[r, c] - y0)

        # column headers (frame times), offset by the toggle-button column
        header = W.HBox(
            [W.HTML("<div style='width:62px'></div>")]
            + [
                W.HTML(
                    f"<div style='width:{thumb_px}px;text-align:center;font-weight:bold'>{lab}</div>"
                )
                for lab in col_labels
            ]
        )

        # one row per chamber: flag toggle + a strip of thumbnails on a shared contrast
        self._toggles, rows = {}, []
        for r, c in chambers:
            crops = [crop(j, r, c) for j in range(len(frames))]
            vmin = min(np.percentile(s, 1) for s, _ in crops)
            vmax = max(np.percentile(s, vmax_pct) for s, _ in crops)
            imgs = [
                W.Image(
                    value=_thumb_png(
                        s, vmin, vmax, thumb_px, circle=(cxl, cyl, radius)
                    ),
                    format="png",
                    layout=W.Layout(width=f"{thumb_px}px", height=f"{thumb_px}px"),
                )
                for s, (cxl, cyl) in crops
            ]
            tog = W.ToggleButton(
                value=False,
                description=f"{c},{r}",  # col,row
                layout=W.Layout(width="60px", height=f"{thumb_px}px"),
            )
            tog.observe(
                lambda ch, key=(r, c): self._toggle(key, ch["new"], ch["owner"]),
                names="value",
            )
            self._toggles[(r, c)] = tog
            rows.append(
                W.HBox(
                    [tog] + imgs,  # don't let the row shrink
                    layout=W.Layout(flex="0 0 auto", height=f"{thumb_px + 6}px"),
                )
            )

        self._count = W.HTML()
        body = W.VBox(
            rows,
            layout=W.Layout(
                height=height,
                overflow="auto",
                flex_flow="column",
                align_items="flex-start",
            ),
        )
        self._box = W.VBox(
            [
                W.HTML("<b>click a chamber's button to flag it (kinetics QC)</b>"),
                self._count,
                header,
                body,
            ]
        )

        if self.save_path and self.save_path.exists():
            self.load()
        self._update_count()

    def _toggle(self, key, on, owner):
        owner.button_style = "danger" if on else ""
        (self._flagged.add if on else self._flagged.discard)(key)
        self._update_count()
        if self.save_path is not None and not self._loading:
            self.save()

    def _update_count(self):
        suffix = f" → {self.save_path.name}" if self.save_path else ""
        self._count.value = (
            f"flagged: {len(self._flagged)} / {len(self._toggles)}{suffix}"
        )

    @property
    def flagged(self):
        return set(self._flagged)

    def save(self, path=None):
        return save_flags(path or self.save_path, self._flagged)

    def load(self, path=None):
        flagged = load_flags(path or self.save_path)
        self._loading = True
        try:
            for key, tog in self._toggles.items():
                tog.value = key in flagged
        finally:
            self._loading = False
        self._update_count()
        return self.flagged

    def apply(self, df, *, drop=False):
        return apply_flags(df, self._flagged, drop=drop)

    def _ipython_display_(self):
        from IPython.display import display

        display(self._box)


class ImageGridFlagger:
    """Inline (ipywidgets) standards flagger: one row per chamber, columns = images (e.g.
    concentrations), grayscale thumbnails with the ROI (+ annulus) overlaid. Click a
    chamber's button to flag it. Renders inline (no Qt) — run under %matplotlib inline.

        flagger = mu.ImageGridFlagger(f"{expt}/Standards_corrected", cx, cy,
                                      radius=R, bg_inner=BI, bg_outer=BO,
                                      save_path=f"{expt}/standards_flags.json")
        flagger
        cal = mu.calibrate_chambers(flagger.apply(standards_df, drop=True))

    `images` may be a **directory** (loaded via load_standard_images — set it once), a
    dict {label: 2-D image}, or a list of 2-D images (`labels=` optional). Same JSON flag
    format as the other flaggers; `.flagged` is the set of flagged (mark_row, mark_col)."""

    def __init__(
        self,
        images,
        cx,
        cy,
        *,
        labels=None,
        chambers=None,
        n_cols=5,
        radius=12,
        bg_inner=None,
        bg_outer=None,
        pad=6,
        thumb_px=70,
        vmax_pct=99.5,
        save_path=None,
        height="700px",
        subpath="tiled_images",
        glob_pattern="*",
        pattern=_STD_RE,
    ):
        import ipywidgets as W

        cx = np.asarray(cx, float)
        cy = np.asarray(cy, float)
        if isinstance(images, (str, Path)):
            images = load_standard_images(
                images, glob_pattern=glob_pattern, subpath=subpath, pattern=pattern
            )

        if isinstance(images, dict):
            labels, imgs = list(images.keys()), list(images.values())
        else:
            imgs = list(images)
            labels = list(labels) if labels is not None else list(range(len(imgs)))

        imgs = [np.asarray(im).squeeze() for im in imgs]
        if len(imgs) > n_cols:
            idx = np.unique(np.linspace(0, len(imgs) - 1, n_cols).round().astype(int))
            imgs, labels = [imgs[k] for k in idx], [labels[k] for k in idx]

        nr, nc = cx.shape
        chambers = (
            chambers
            if chambers is not None
            else [(r, c) for r in range(nr) for c in range(nc)]
        )
        self.save_path = Path(save_path) if save_path else None
        self._loading = False
        self._flagged = set()
        ann = (bg_inner, bg_outer) if (bg_inner and bg_outer) else None
        w = int(round((bg_outer if bg_outer else radius) + pad))
        H, Wd = imgs[0].shape

        def crop(im, r, c):
            xi, yi = int(round(cx[r, c])), int(round(cy[r, c]))
            x0, y0 = max(0, xi - w), max(0, yi - w)
            return im[y0 : min(H, yi + w + 1), x0 : min(Wd, xi + w + 1)], (
                cx[r, c] - x0,
                cy[r, c] - y0,
            )

        # per-image contrast limits, pooled over every shown chamber's crop
        vlims = [
            (
                float(
                    np.percentile(
                        np.concatenate(
                            [crop(im, r, c)[0].ravel() for r, c in chambers]
                        ),
                        1,
                    )
                ),
                float(
                    np.percentile(
                        np.concatenate(
                            [crop(im, r, c)[0].ravel() for r, c in chambers]
                        ),
                        vmax_pct,
                    )
                ),
            )
            for im in imgs
        ]

        header = W.HBox(
            [W.HTML("<div style='width:62px'></div>")]
            + [
                W.HTML(
                    f"<div style='width:{thumb_px}px;text-align:center;font-weight:bold'>{lab}</div>"
                )
                for lab in labels
            ]
        )

        # one row per chamber: flag toggle + one thumbnail per image
        self._toggles, rows = {}, []
        for r, c in chambers:
            imgws = []
            for j, im in enumerate(imgs):
                sub, (cxl, cyl) = crop(im, r, c)
                imgws.append(
                    W.Image(
                        value=_thumb_png(
                            sub,
                            vlims[j][0],
                            vlims[j][1],
                            thumb_px,
                            circle=(cxl, cyl, radius),
                            annulus=ann,
                        ),
                        format="png",
                        layout=W.Layout(width=f"{thumb_px}px", height=f"{thumb_px}px"),
                    )
                )
            tog = W.ToggleButton(
                value=False,
                description=f"{c},{r}",  # col,row
                layout=W.Layout(width="60px", height=f"{thumb_px}px"),
            )
            tog.observe(
                lambda ch, key=(r, c): self._toggle(key, ch["new"], ch["owner"]),
                names="value",
            )
            self._toggles[(r, c)] = tog
            rows.append(
                W.HBox(
                    [tog] + imgws,
                    layout=W.Layout(flex="0 0 auto", height=f"{thumb_px + 6}px"),
                )
            )

        self._count = W.HTML()
        body = W.VBox(
            rows,
            layout=W.Layout(
                height=height,
                overflow="auto",
                flex_flow="column",
                align_items="flex-start",
            ),
        )
        self._box = W.VBox(
            [
                W.HTML("<b>click a chamber's button to flag it (standards QC)</b>"),
                self._count,
                header,
                body,
            ]
        )

        if self.save_path and self.save_path.exists():
            self.load()
        self._update_count()

    def _toggle(self, key, on, owner):
        owner.button_style = "danger" if on else ""
        (self._flagged.add if on else self._flagged.discard)(key)
        self._update_count()
        if self.save_path is not None and not self._loading:
            self.save()

    def _update_count(self):
        suffix = f" → {self.save_path.name}" if self.save_path else ""
        self._count.value = (
            f"flagged: {len(self._flagged)} / {len(self._toggles)}{suffix}"
        )

    @property
    def flagged(self):
        return set(self._flagged)

    def save(self, path=None):
        return save_flags(path or self.save_path, self._flagged)

    def load(self, path=None):
        flagged = load_flags(path or self.save_path)
        self._loading = True
        try:
            for key, tog in self._toggles.items():
                tog.value = key in flagged
        finally:
            self._loading = False
        self._update_count()
        return self.flagged

    def apply(self, df, *, drop=False):
        return apply_flags(df, self._flagged, drop=drop)

    def _ipython_display_(self):
        from IPython.display import display

        display(self._box)
