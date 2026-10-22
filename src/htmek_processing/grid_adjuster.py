"""GridAdjuster: drag the grid's four corners onto the image (interactive)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .layout import chamber_tags
from .io import time_coord
from .grid import bilinear_grid, load_mask, save_corners
from .quantify import centers_to_df, kinetic_traces


class GridAdjuster:
    """Drag 4 corner handles to place the grid; interior buttons are interpolated
    (bilinear -> handles rotation, shear, slight keystone).

    Requires the ipympl backend: run `%matplotlib widget` in the notebook first.

        ga = GridAdjuster(img, columns={1: "WT", 2: "S92A", ...})  # drag the cyan handles
        df = ga.to_df()  # tidy DataFrame when happy

    Corner order is [top-left, top-right, bottom-right, bottom-left] = the corner-button
    centers. Pass `corners=` to seed a first placement (e.g. another acquisition's corners);
    once `corners_path` exists, the saved placement is used instead.

    `img` may be a (time, y, x) stack, e.g. a kinetic run: one frame is shown (`frame`,
    default 0) and a "frame" slider switches between them. Frames are read only when shown,
    so a lazy zarr/xarray stack is never loaded whole. `to_kinetic_df()` uses the stack."""

    LABELS = ("TL", "TR", "BR", "BL")
    MAX_DISPLAY_PX = 2048  # longest image edge drawn at once (see _render_view)

    def __init__(
        self,
        img,
        columns=None,
        n_rows=None,
        n_cols=None,
        corners=None,
        corners_path=None,
        radius=12,
        bg_outer=None,
        vmin_pct=0,
        vmax_pct=99,
        figsize=(9, 9),
        pick_radius=None,
        frame=0,
    ):
        # A (time, y, x) stack stays lazy: only the frame on screen is read
        a = img.squeeze() if hasattr(img, "isel") else np.asarray(img).squeeze()
        self.stack = a if a.ndim == 3 else None
        self._seconds = time_coord(img) if self.stack is not None else None
        self._frame = int(frame or 0)
        self.img = (
            self._read_frame(self._frame) if self.stack is not None else np.asarray(a)
        )

        self.columns = columns
        if columns is not None:
            # the identity layout fixes the grid shape (columns list / pinlist / tag array)
            n_rows, n_cols = chamber_tags(columns, n_rows, n_cols).shape
        self.n_rows, self.n_cols = n_rows or 8, n_cols or 8

        # how close (px) a click must land to grab a corner; scales with the image so the
        # handles stay grabbable on a stitched mosaic as on a single 1024 px frame
        self.pick_radius = pick_radius or 0.025 * max(self.img.shape)
        self.corners_path = Path(corners_path) if corners_path else None

        # resume a saved placement (corners + radius + annulus); saved values win when
        # present, constructor args are the fallback / first-run defaults.
        saved = (
            load_mask(self.corners_path)
            if (self.corners_path and self.corners_path.exists())
            else {}
        )
        self.radius = saved.get("radius") if saved.get("radius") is not None else radius
        self.bg_inner = self.radius  # the annulus starts at the button edge
        self.bg_outer = (
            saved.get("bg_outer")
            if saved.get("bg_outer") is not None
            else bg_outer
            if bg_outer is not None
            else self.radius + 11
        )

        H, W = self.img.shape
        # a saved placement wins; `corners=` only seeds the first run (e.g. from another
        # acquisition's corners file), so rerunning the cell never discards your work
        if saved.get("corners") is not None:
            corners = saved["corners"]
        if corners is None:
            corners = [
                [0.2 * W, 0.2 * H],
                [0.8 * W, 0.2 * H],
                [0.8 * W, 0.8 * H],
                [0.2 * W, 0.8 * H],
            ]

        self.corners = np.asarray(corners, float)
        self._drag = None
        self._build(vmin_pct, vmax_pct, figsize)

    def _read_frame(self, t):
        """One 2-D frame of the stack as a numpy array (reads just that frame)."""
        return np.asarray(self.stack[int(t)])

    def centers(self):
        return bilinear_grid(self.corners, self.n_rows, self.n_cols)

    def to_df(
        self,
        aggs=("median", "sum"),
        bg=True,
        bg_inner=None,
        bg_outer=None,
        bg_agg="median",
    ):
        cx, cy = self.centers()
        bi = (bg_inner if bg_inner is not None else self.bg_inner) if bg else None
        bo = (bg_outer if bg_outer is not None else self.bg_outer) if bg else None

        df = centers_to_df(
            cx,
            cy,
            self.columns,
            self.n_rows,
            img=self.img,
            radius=self.radius,
            aggs=aggs,
            bg=bg,
            bg_inner=bi,
            bg_outer=bo,
            bg_agg=bg_agg,
        )
        df.attrs["corners"] = (
            self.corners.copy().tolist()
        )  # placement travels with the df
        return df

    def to_kinetic_df(
        self, aggs=("median",), bg=True, bg_inner=None, bg_outer=None, bg_agg="median"
    ):
        """Long tidy DataFrame of per-button traces across every frame of the stack the
        adjuster was built from (frame, seconds, mark_row, mark_col, tag, x, y, <aggs>...).
        Requires a 3-D stack; raises if the adjuster was given a single 2-D frame."""
        if self.stack is None:
            raise ValueError(
                "GridAdjuster was built from a single 2-D frame; pass the full "
                "(time, y, x) stack to GridAdjuster to extract kinetic traces."
            )

        cx, cy = self.centers()
        bi = (bg_inner if bg_inner is not None else self.bg_inner) if bg else None
        bo = (bg_outer if bg_outer is not None else self.bg_outer) if bg else None

        df = kinetic_traces(
            self.stack,
            cx,
            cy,
            self.columns,
            self.n_rows,
            radius=self.radius,
            aggs=aggs,
            bg=bg,
            bg_inner=bi,
            bg_outer=bo,
            bg_agg=bg_agg,
            seconds=self._seconds,
        )
        df.attrs["corners"] = self.corners.copy().tolist()
        return df

    def save_corners(self, path=None):
        """Save the full mask (corners + radius + annulus) to JSON so the next GridAdjuster
        with the same corners_path resumes exactly where you left off."""
        return save_corners(
            path or self.corners_path,
            self.corners,
            radius=self.radius,
            bg_inner=self.bg_inner,
            bg_outer=self.bg_outer,
        )

    def _build(self, vmin_pct, vmax_pct, figsize):
        from matplotlib.widgets import Slider, CheckButtons

        self.fig, self.ax = plt.subplots(figsize=figsize)

        # Display at most ~2048 px of image: redrawing a full stitched mosaic (~7k x 7k) on
        # every drag is what makes the widget lag. The full chip is shown strided; zooming
        # in re-renders just the visible region, at full resolution once it fits (see
        # _render_view). Axes stay in full-resolution pixels throughout, so corners and
        # centers are unaffected by what is displayed.
        H, W = self.img.shape
        step = max(1, int(np.ceil(max(H, W) / self.MAX_DISPLAY_PX)))
        shown = self.img[::step, ::step]

        # full data range bounds the contrast sliders; initial cutoffs are percentiles
        self._data_lo, self._data_hi = float(np.min(shown)), float(np.max(shown))
        self.vmin = float(np.percentile(shown, vmin_pct))
        self.vmax = float(np.percentile(shown, vmax_pct))
        self._im = self.ax.imshow(
            shown,
            cmap="gray",
            vmin=self.vmin,
            vmax=self.vmax,
            # each shown pixel spans `step` full-res pixels
            extent=(
                -0.5,
                shown.shape[1] * step - 0.5,
                shown.shape[0] * step - 0.5,
                -0.5,
            ),
            interpolation="nearest",
        )

        # re-render on zoom/pan; autoscale off so swapping the image never moves the view
        self.ax.set_autoscale_on(False)
        self._view_key = None
        self.ax.callbacks.connect("xlim_changed", self._render_view)
        self.ax.callbacks.connect("ylim_changed", self._render_view)

        # One EllipseCollection per ring type (ROI, annulus inner, annulus outer) instead
        # of one patch per chamber: a 56x32 chip is 1,792 chambers, and moving thousands
        # of separate patches on every drag makes the widget lag.
        from matplotlib.collections import EllipseCollection

        cx, cy = self.centers()
        pts = np.column_stack([cx.ravel(), cy.ravel()])

        def rings(r, color, lw, ls="-"):
            d = np.full(len(pts), 2.0 * r)  # diameters, in data (pixel) units
            coll = EllipseCollection(
                d,
                d,
                np.zeros(len(pts)),
                units="xy",
                offsets=pts,
                offset_transform=self.ax.transData,
                facecolors="none",
                edgecolors=color,
                linewidths=lw,
                linestyles=ls,
            )
            self.ax.add_collection(coll)
            return coll

        self._roi = rings(self.radius, "yellow", 0.8)
        self._bg_in = rings(self.bg_inner, "orange", 0.6, ":")
        self._bg_out = rings(self.bg_outer, "orange", 0.6, ":")
        (self._grid_pts,) = self.ax.plot(cx.ravel(), cy.ravel(), "r.", ms=3)

        # "col,row" labels beside each ROI -- only on small chips, where they stay legible
        nr, nc = cx.shape
        show_labels = nr * nc <= 100
        self._index_labels = (
            [
                self.ax.annotate(
                    f"{j},{i}",
                    (cx[i, j], cy[i, j]),
                    textcoords="offset points",
                    xytext=(3, 3),
                    color="yellow",
                    fontsize=6,
                    ha="left",
                    va="bottom",
                    annotation_clip=False,
                    clip_on=True,
                    zorder=6,
                )
                for i in range(nr)
                for j in range(nc)
            ]
            if show_labels
            else []
        )

        # draggable corner handles, labeled TL/TR/BR/BL
        (self._corner_pts,) = self.ax.plot(
            self.corners[:, 0],
            self.corners[:, 1],
            "o",
            mfc="none",
            mec="cyan",
            ms=15,
            mew=2,
        )
        self._labels = [
            self.ax.annotate(lab, (x, y), color="cyan", fontsize=9)
            for lab, (x, y) in zip(self.LABELS, self.corners)
        ]
        self.ax.set_title(
            "drag corners; keys 1-4 zoom to TL/TR/BR/BL, 0 = whole chip\n"
            "sliders: display gain (vmin/vmax), ROI radius (yellow), bg annulus (orange)",
            fontsize=9,
        )

        # live sliders: (frame,) display vmin/vmax (gain/cutoffs), radius (also the annulus inner edge), bg outer
        self.fig.subplots_adjust(bottom=0.34 if self.stack is not None else 0.30)
        if self.stack is not None:
            ax_t = self.fig.add_axes([0.25, 0.27, 0.5, 0.02])
            self._frame_slider = Slider(
                ax_t,
                "frame",
                0,
                self.stack.shape[0] - 1,
                valinit=self._frame,
                valstep=1,
                valfmt="%d",
            )
            self._frame_slider.on_changed(self._on_frame)
        rmax = max(60, int(6 * self.radius))
        ax_vmin = self.fig.add_axes([0.25, 0.23, 0.5, 0.02])
        ax_vmax = self.fig.add_axes([0.25, 0.19, 0.5, 0.02])
        ax_r = self.fig.add_axes([0.25, 0.12, 0.5, 0.02])
        ax_bo = self.fig.add_axes([0.25, 0.075, 0.5, 0.02])
        self._vmin_slider = Slider(
            ax_vmin, "vmin", self._data_lo, self._data_hi, valinit=self.vmin
        )
        self._vmax_slider = Slider(
            ax_vmax, "vmax", self._data_lo, self._data_hi, valinit=self.vmax
        )
        self._radius_slider = Slider(
            ax_r, "radius", 1, rmax, valinit=self.radius, valstep=1
        )
        self._bg_outer_slider = Slider(
            ax_bo, "bg outer", 1, rmax, valinit=self.bg_outer, valstep=1
        )

        self._vmin_slider.on_changed(self._on_vmin)
        self._vmax_slider.on_changed(self._on_vmax)
        self._radius_slider.on_changed(self._on_radius)
        self._bg_outer_slider.on_changed(self._on_bg_outer)

        # checkbox to hide the mask overlays (corner handles stay so you can still place the
        # grid)
        ax_chk = self.fig.add_axes([0.01, 0.90, 0.14, 0.06])
        ax_chk.set_axis_off()
        self._mask_check = CheckButtons(ax_chk, ["show masks"], [True])
        self._mask_check.on_clicked(self._on_toggle_masks)

        # mouse events drive corner dragging
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _render_view(self, ax=None):
        """Show the visible region at the finest stride that keeps it <= MAX_DISPLAY_PX,
        so zooming in reveals full-resolution pixels without ever drawing the whole
        mosaic at full resolution."""
        H, W = self.img.shape
        (xa, xb), (ya, yb) = sorted(self.ax.get_xlim()), sorted(self.ax.get_ylim())
        x0, x1 = max(0, int(xa)), min(W, int(np.ceil(xb)) + 1)
        y0, y1 = max(0, int(ya)), min(H, int(np.ceil(yb)) + 1)
        if x1 <= x0 or y1 <= y0:
            return

        step = max(1, int(np.ceil(max(x1 - x0, y1 - y0) / self.MAX_DISPLAY_PX)))
        key = (x0, x1, y0, y1, step)
        if key == self._view_key:  # x- and y-limit callbacks both fire on one zoom
            return
        self._view_key = key

        shown = self.img[y0:y1:step, x0:x1:step]
        self._im.set_data(shown)
        self._im.set_extent(
            (
                x0 - 0.5,
                x0 + shown.shape[1] * step - 0.5,
                y0 + shown.shape[0] * step - 0.5,
                y0 - 0.5,
            )
        )
        self.fig.canvas.draw_idle()

    def _on_key(self, e):
        """Keys 1-4 zoom onto the TL/TR/BR/BL corner handle; 0 shows the whole chip."""
        H, W = self.img.shape
        if e.key == "0":
            self.ax.set_xlim(-0.5, W - 0.5)
            self.ax.set_ylim(H - 0.5, -0.5)
        elif e.key in ("1", "2", "3", "4"):
            x, y = self.corners[int(e.key) - 1]
            # a window ~4 chamber pitches across, so the corner chamber and its
            # neighbors are both visible
            half = 2.5 * max(
                np.ptp(self.corners[:, 0]) / max(self.n_cols - 1, 1),
                np.ptp(self.corners[:, 1]) / max(self.n_rows - 1, 1),
            )
            self.ax.set_xlim(x - half, x + half)
            self.ax.set_ylim(y + half, y - half)
        else:
            return
        self.fig.canvas.draw_idle()

    def _persist(self):
        if self.corners_path is not None:
            self.save_corners()

    def _on_toggle_masks(self, label):
        vis = self._mask_check.get_status()[0]
        for coll in (self._roi, self._bg_in, self._bg_out, self._grid_pts):
            coll.set_visible(vis)
        for lab in self._index_labels:
            lab.set_visible(vis)
        self.fig.canvas.draw_idle()

    def _on_frame(self, val):
        """Show another frame of the stack; the view (zoom/pan) and the grid stay put."""
        self._frame = int(val)
        self.img = self._read_frame(self._frame)
        self._view_key = None  # force _render_view to redraw the current view
        self._render_view()

    def _apply_clim(self):
        # matplotlib requires vmin < vmax; keep them ordered without fighting the sliders
        lo, hi = sorted((self.vmin, self.vmax))
        if lo == hi:
            hi = lo + 1e-9
        self._im.set_clim(lo, hi)
        self.fig.canvas.draw_idle()

    def _on_vmin(self, val):
        self.vmin = float(val)
        self._apply_clim()

    def _on_vmax(self, val):
        self.vmax = float(val)
        self._apply_clim()

    def _on_radius(self, val):
        self.radius = self.bg_inner = int(val)  # the annulus starts at the button edge
        self._set_diameter(self._roi, self.radius)
        self._set_diameter(self._bg_in, self.bg_inner)
        self._persist()
        self.fig.canvas.draw_idle()

    def _on_bg_outer(self, val):
        self.bg_outer = int(val)
        self._set_diameter(self._bg_out, self.bg_outer)
        self._persist()
        self.fig.canvas.draw_idle()

    def _nearest(self, x, y):
        d = np.hypot(self.corners[:, 0] - x, self.corners[:, 1] - y)
        k = int(np.argmin(d))
        return k if d[k] <= self.pick_radius else None

    def _on_press(self, e):
        # while the toolbar's zoom/pan tool is active, clicks belong to it, not the corners
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if toolbar is not None and str(getattr(toolbar, "mode", "")):
            return
        if e.inaxes is self.ax and e.xdata is not None:
            self._drag = self._nearest(e.xdata, e.ydata)

    def _on_motion(self, e):
        if self._drag is not None and e.inaxes is self.ax and e.xdata is not None:
            self.corners[self._drag] = [e.xdata, e.ydata]
            self._redraw()

    def _on_release(self, e):
        if self._drag is not None and self.corners_path is not None:
            self.save_corners()  # auto-persist placement after each drag
        self._drag = None

    @staticmethod
    def _set_diameter(coll, r):
        """Resize every ring in an EllipseCollection to radius `r` (pixels)."""
        d = np.full(len(coll.get_offsets()), 2.0 * r)
        coll.set_widths(d)  # matplotlib >= 3.6
        coll.set_heights(d)

    def _redraw(self):
        cx, cy = self.centers()
        pts = np.column_stack([cx.ravel(), cy.ravel()])
        self._grid_pts.set_data(cx.ravel(), cy.ravel())

        for coll in (self._roi, self._bg_in, self._bg_out):
            coll.set_offsets(pts)  # one array update per ring type
        for lab, p in zip(self._index_labels, pts):  # index labels track their buttons
            lab.xy = p

        self._corner_pts.set_data(self.corners[:, 0], self.corners[:, 1])
        for lab, (x, y) in zip(self._labels, self.corners):
            lab.set_position((x, y))

        self.fig.canvas.draw_idle()
