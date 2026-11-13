"""GridAdjuster: drag the grid's four corners onto the image, then any chamber to correct
the grid locally (interactive)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .layout import chamber_tags
from .io import time_coord
from .grid import (DUMBBELL_FIELDS, WARP_REACH, bilinear_grid, dumbbell_outline, load_mask, load_warp,
                   save_corners, save_warp, warp_offsets)

# dumbbell mode: the background ring's outer edge, this many px beyond the outline
DUMBBELL_BG_MARGIN = 10
from .quantify import centers_to_df, kinetic_traces

CTRL_COLOR = "#FF4FD8"  # local corrections (magenta: apart from the cyan corners and yellow ROIs)

# The window's colors, light and dark (the image itself is always gray on black). `rc` is
# applied while the window and its sliders are built; the rest colors the "How to use" panel.
THEMES = {
    "light": dict(rc={}, panel="#F5F7FA", panel_edge="0.75", key="white", key_edge="0.35",
                  line="0.3", rule="0.8", text="black", note="0.4"),
    "dark": dict(rc={"figure.facecolor": "#1E1E1E", "axes.facecolor": "#2A2A2A", "axes.edgecolor": "0.6",
                     "text.color": "0.9", "axes.labelcolor": "0.9", "xtick.color": "0.8",
                     "ytick.color": "0.8"},
                 panel="#2A2D31", panel_edge="0.4", key="#3A3D42", key_edge="0.65",
                 line="0.75", rule="0.4", text="0.92", note="0.65"),
}


def os_dark_mode():
    """Whether the OS is set to a dark appearance: macOS (AppleInterfaceStyle), Windows
    (AppsUseLightTheme), GNOME (color-scheme). False when it can't tell."""
    import platform
    import subprocess

    system = platform.system()
    try:
        if system == "Darwin":
            out = subprocess.run(["defaults", "read", "-g", "AppleInterfaceStyle"],
                                 capture_output=True, text=True, timeout=2)
            return out.stdout.strip().lower() == "dark"
        if system == "Windows":
            import winreg

            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                 r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
            return winreg.QueryValueEx(key, "AppsUseLightTheme")[0] == 0
        out = subprocess.run(["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
                             capture_output=True, text=True, timeout=2)
        return "dark" in out.stdout.lower()
    except Exception:   # no such setting, no gsettings, ...: light
        return False

# the on-screen key: mouse actions (button, marker, color, size, what) and hotkeys (keys, what)
KEY_MOUSE = [
    ("left", "o", "cyan", 12, "drag a corner: place the grid"),
    ("left", "D", CTRL_COLOR, 7, "drag any circle: fix it locally (the grid around it follows)"),
    ("right", "D", CTRL_COLOR, 7, "right-click a fix: remove it"),
]
KEY_KEYS = [
    ("z", "zoom box (one zoom, then back to dragging)"),
    ("+ −", "zoom in / out where the mouse is"),
    ("2-finger", "scroll to pan (trackpad)"),
    ("h 0", "home: the whole chip"),
    ("p", "pan (press again to stop)"),
    ("esc", "leave zoom / pan"),
    ("1–4", "jump to the TL / TR / BR / BL corner"),
]


class GridAdjuster:
    """Drag 4 corner handles to place the grid; interior chambers are interpolated
    (bilinear -> handles rotation, shear, slight keystone). Then drag any chamber that is
    still off onto its spot: it becomes a local correction, and the grid around it follows,
    fading out within ~WARP_REACH chambers (grid.warp_offsets); the corners stay put.

    Needs an interactive backend (`%matplotlib qt`, or ipympl's `%matplotlib widget`).

        ga = GridAdjuster(img, columns={1: "WT", 2: "S92A", ...},
                          corners_path="2_masks/corners.json", warp_path="2_masks/warp.json")
        df = ga.to_df()  # tidy DataFrame when happy

    Corner order is [top-left, top-right, bottom-right, bottom-left] = the corner-button
    centers. Pass `corners=` to seed a first placement (e.g. another acquisition's corners);
    once `corners_path` exists, the saved placement is used instead. The placement is saved
    when the window opens and after every change. Local corrections go to `warp_path` (and
    are resumed from it); without one they last only for the session. In later cells,
    grid.warped_grid(corners_path, warp_path, n_rows, n_cols) gives the same centers.

    `img` may be a (time, y, x) stack, e.g. a kinetic run: one frame is shown (`frame`,
    default 0; negative counts from the end) and a "frame" slider switches between them.
    Frames are read only when shown, so a lazy zarr/xarray stack is never loaded whole.
    `to_kinetic_df()` uses the stack.

    With `dumbbell` = {dna_dx, dna_dy, dna_radius, neck_width} (px; grid.DUMBBELL_FIELDS),
    each grid point is a dumbbell instead of a circle: the reaction chamber (of `radius`, at
    the grid point), its DNA chamber at (dna_dx, dna_dy) from it, and the neck between them,
    with sliders for each and `bg_outer` as the background ring's margin beyond the outline.
    The shape is saved with the corners (and wins over `dumbbell=` once saved); read it with
    quantify.dumbbell_timecourse.

    The window follows the OS's light or dark appearance (os_dark_mode); `dark=True` or
    `dark=False` overrides it.

    A "How to use" panel left of the image lists the mouse actions and the hotkeys
    (KEY_KEYS): z zoom box, + / − zoom where the mouse is, two-finger scroll pans, h or 0 home, p pan, esc leave
    zoom / pan, 1-4 the corners. It refits when the window is resized.

    In a desktop window (`%matplotlib qt`) the call blocks until the window is closed (`block=True`,
    the default), so the next cell only runs once the grid is placed; `block=False` returns at once
    (then call `wait()` before running on)."""

    LABELS = ("TL", "TR", "BR", "BL")
    MAX_DISPLAY_PX = 1024  # longest edge of the VIEW drawn at once (see _render_view)
    RENDER_MARGIN = 0.5    # the image drawn reaches this fraction of the view past each side
    OUTLINE_POINTS = 64    # points per drawn dumbbell outline (1,792 x 2 outlines on a chip)
    PAN_SETTLE_MS = 150    # after the last scroll, wait this long before the full redraw (_pan_px)
    DRAG_FPS = 30          # at most this many redraws per second while dragging
    PANEL_RIGHT = 0.27     # figure fraction: the key panel ends here; image and sliders start right of it

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
        figsize=(12.5, 8.5),
        pick_radius=None,
        frame=0,
        warp_path=None,
        dark=None,
        dumbbell=None,
        block=True,
    ):
        # the window's theme: the OS's appearance unless `dark` is given
        self.theme = THEMES["dark" if (os_dark_mode() if dark is None else dark) else "light"]
        # A (time, y, x) stack stays lazy: only the frame on screen is read
        a = img.squeeze() if hasattr(img, "isel") else np.asarray(img).squeeze()
        self.stack = a if a.ndim == 3 else None
        self._seconds = time_coord(img) if self.stack is not None else None
        self._frame = int(frame or 0)
        if self.stack is not None and self._frame < 0:  # e.g. -1: the last frame
            self._frame += self.stack.shape[0]

        # local corrections {(row, col): (dx, dy)}, resumed from warp_path
        self.warp_path = Path(warp_path) if warp_path else None
        self.controls = {(int(r), int(c)): (dx, dy) for r, c, dx, dy in load_warp(self.warp_path)}
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
        # dumbbell mode: each grid point is a reaction chamber joined to its DNA chamber
        # (grid.dumbbell_mask); the saved shape wins, `dumbbell=` seeds a first run
        self.dumbbell = saved.get("dumbbell") or ({k: float(dumbbell[k]) for k in DUMBBELL_FIELDS}
                                                   if dumbbell is not None else None)
        self.bg_outer = (
            saved.get("bg_outer")
            if saved.get("bg_outer") is not None
            else bg_outer
            if bg_outer is not None
            else DUMBBELL_BG_MARGIN if self.dumbbell is not None
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
        with plt.rc_context(self.theme["rc"]):
            self._build(vmin_pct, vmax_pct, figsize)
        # saved as soon as the window opens (and after every change), so a later cell always
        # finds the placement, even when only local corrections were made
        self._persist()
        if block:
            self.wait()

    def wait(self):
        """Block until the window is closed, so the next notebook cell cannot run against a window that is
        still open (with `%matplotlib qt`, running on crashed the kernel; MBO-054, 2026-11-09). Only in a
        desktop window (Qt, macOS, Tk): with ipympl or a non-interactive backend (e.g. a notebook run by
        nbconvert) it returns at once."""
        import matplotlib

        backend = matplotlib.get_backend().lower()
        if any(k in backend for k in ("qt", "macosx", "tk")) and plt.fignum_exists(self.fig.number):
            plt.show(block=True)

    def _read_frame(self, t):
        """One 2-D frame of the stack as a numpy array (reads just that frame)."""
        return np.asarray(self.stack[int(t)])

    def centers(self):
        """Chamber centers (cx, cy): the corners' bilinear grid plus the local corrections."""
        cx, cy = bilinear_grid(self.corners, self.n_rows, self.n_cols)
        if not self.controls:
            return cx, cy
        dx, dy = warp_offsets([[*rc, *d] for rc, d in self.controls.items()], self.n_rows, self.n_cols)
        return cx + dx, cy + dy

    def save_warp(self, path=None):
        """Save the local corrections to JSON (grid.load_warp and warped_grid read them)."""
        return save_warp(path or self.warp_path, self.controls)

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
        """Save the full mask (corners + radius + annulus, and the dumbbell's shape) to JSON so
        the next GridAdjuster with the same corners_path resumes exactly where you left off."""
        return save_corners(
            path or self.corners_path,
            self.corners,
            radius=self.radius,
            bg_inner=self.bg_inner,
            bg_outer=self.bg_outer,
            dumbbell=self.dumbbell,
        )

    def _build(self, vmin_pct, vmax_pct, figsize):
        from matplotlib.widgets import Slider, CheckButtons

        self.fig, self.ax = plt.subplots(figsize=figsize)
        # ipympl shows its toolbar (zoom, pan, home) only on hover by default: keep it visible
        if hasattr(self.fig.canvas, "toolbar_visible"):
            self.fig.canvas.toolbar_visible = True

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
        self._fast, self._pan_timer, self._pan_region = False, None, None   # see _pan_px
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

        if self.dumbbell is not None:
            # one Line2D per outline type, every dumbbell's outline in it (NaN between them):
            # a single artist, like the ring collections, so a drag stays fast
            from matplotlib.lines import Line2D

            self._roi = self.ax.add_line(Line2D([], [], color="yellow", lw=0.8))
            self._bg_in = self.ax.add_line(Line2D([], [], visible=False))   # the ring starts at the outline
            self._bg_out = self.ax.add_line(Line2D([], [], color="orange", lw=0.6, ls=":"))
            self._dumbbell_templates()
        else:
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
        # local corrections: magenta diamonds, moving with the grid
        (self._ctrl_pts,) = self.ax.plot([], [], "D", ms=8, mfc="none", mec=CTRL_COLOR, mew=1.8, zorder=7)

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

        # mouse events drive corner and chamber dragging
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("button_release_event", self._after_zoom)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._connect_scroll_pan()

        # the "How to use" panel left of the image (refit on resize), the sliders right of it
        self.fig.subplots_adjust(left=self.PANEL_RIGHT + 0.06)
        x0 = self.PANEL_RIGHT + 0.14  # room for the slider labels ("bg outer") left of the bars
        right = 0.9 if self.dumbbell is None else 0.58   # dumbbell mode: a second column of sliders
        for s in (getattr(self, "_frame_slider", None), self._vmin_slider, self._vmax_slider,
                  self._radius_slider, self._bg_outer_slider):
            if s is not None:
                b = s.ax.get_position()
                s.ax.set_position([x0, b.y0, right - x0, b.height])
        self._dumbbell_sliders = []
        if self.dumbbell is not None:
            # where the DNA chamber sits from the reaction chamber, its radius, the neck's width
            specs = (("dna_dx", "DNA dx", -400, 400, 0.23), ("dna_dy", "DNA dy", -200, 200, 0.19),
                     ("dna_radius", "DNA r", 2, 150, 0.12), ("neck_width", "neck", 1, 150, 0.075))
            for key, label, lo, hi, y in specs:
                slider = Slider(self.fig.add_axes([0.70, y, 0.2, 0.02]), label, lo, hi,
                                valinit=self.dumbbell[key], valstep=1)
                slider.on_changed(lambda val, key=key: self._on_dumbbell(key, val))
                self._dumbbell_sliders.append(slider)
            self._bg_outer_slider.label.set_text("bg margin")
            self._radius_slider.label.set_text("chamber r")
        self._key_ax = None
        self._draw_key()
        self.fig.canvas.mpl_connect("resize_event", lambda e: self._draw_key())
        for s in (getattr(self, "_frame_slider", None), self._vmin_slider, self._vmax_slider,
                  self._radius_slider, self._bg_outer_slider, *self._dumbbell_sliders):
            if s is not None:   # slider labels and values follow the theme's text color
                s.label.set_color(self.theme["text"])
                s.valtext.set_color(self.theme["text"])
        for label in self._mask_check.labels:
            label.set_color(self.theme["text"])
        if hasattr(self._mask_check, "set_frame_props"):   # matplotlib >= 3.7: the box follows the theme
            self._mask_check.set_frame_props({"edgecolor": self.theme["text"]})
            self._mask_check.set_check_props({"facecolor": self.theme["text"]})
        self._update_artists()

    def _render_view(self, ax=None):
        """Show the visible region at the finest stride that keeps it within the axes' screen pixels,
        so zooming in reveals full-resolution pixels without ever drawing the whole
        mosaic at full resolution."""
        H, W = self.img.shape
        (xa, xb), (ya, yb) = sorted(self.ax.get_xlim()), sorted(self.ax.get_ylim())
        if min(xb, W) <= max(xa, 0) or min(yb, H) <= max(ya, 0):
            return

        # The stride keeps the view at no more image pixels than the axes have screen pixels:
        # matplotlib resamples every pixel it is given on each draw, and shrinking an image
        # larger than the screen is what makes drawing slow (MBO-052, a Retina screen: 73 ms
        # per draw at 1.7x the screen's pixels, 15 ms at or below them, 2026-10-30). Zoomed in
        # until the view fits the screen, it is full resolution.
        # The region drawn is the view plus RENDER_MARGIN of it on every side, so a pan within
        # it only moves the axes (no new image); the image is redrawn when the view leaves
        # that region or the stride changes (a zoom).
        screen_px = max(1.0, self.ax.bbox.width, self.ax.bbox.height)
        step = max(1, int(np.ceil(max(xb - xa, yb - ya) / screen_px)))
        if self._view_key is not None:
            kx0, kx1, ky0, ky1, kstep = self._view_key
            inside = (xa >= kx0 or kx0 == 0) and (xb <= kx1 or kx1 == W) and \
                     (ya >= ky0 or ky0 == 0) and (yb <= ky1 or ky1 == H)
            if kstep == step and inside:
                return
        mx, my = self.RENDER_MARGIN * (xb - xa), self.RENDER_MARGIN * (yb - ya)
        x0, x1 = max(0, int(xa - mx)), min(W, int(np.ceil(xb + mx)) + 1)
        y0, y1 = max(0, int(ya - my)), min(H, int(np.ceil(yb + my)) + 1)
        self._view_key = (x0, x1, y0, y1, step)

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
        if not self._fast:   # mid-pan, the pan draws the image itself (_pan_px)
            self.fig.canvas.draw_idle()

    def _on_key(self, e):
        """z: one zoom box; + / −: zoom on the view's center; h or 0: the whole chip;
        esc: leave zoom / pan (p is matplotlib's pan); 1-4: zoom onto a corner handle."""
        H, W = self.img.shape
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        mode = str(getattr(toolbar, "mode", "")) if toolbar is not None else ""
        if e.key == "z" and toolbar is not None:
            if "zoom" not in mode:
                toolbar.zoom()
            return
        if e.key == "escape" and toolbar is not None and mode:
            (toolbar.zoom if "zoom" in mode else toolbar.pan)()
            return
        if e.key in ("+", "=", "-", "_"):
            at = (e.xdata, e.ydata) if e.inaxes is self.ax and e.xdata is not None else None
            self._zoom_center(0.5 if e.key in ("+", "=") else 2.0, at=at)
            return
        if e.key in ("0", "h"):
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

    def _zoom_center(self, factor, at=None):
        """Zoom by `factor` (0.5: in 2x, 2: out 2x) about `at` (x, y), the point that stays put
        under the mouse, or the view's center; never past the whole chip."""
        H, W = self.img.shape
        (x0, x1), (y1, y0) = self.ax.get_xlim(), self.ax.get_ylim()  # y is inverted
        ax_, ay_ = at if at is not None else ((x0 + x1) / 2, (y0 + y1) / 2)
        hw = min((x1 - x0) * factor / 2, W / 2)
        hh = min((y1 - y0) * factor / 2, H / 2)
        # the new view keeps `at` at the same fraction across it
        fx = (ax_ - x0) / (x1 - x0) if x1 > x0 else 0.5
        fy = (ay_ - y0) / (y1 - y0) if y1 > y0 else 0.5
        xc = ax_ + (0.5 - fx) * 2 * hw
        yc = ay_ + (0.5 - fy) * 2 * hh
        self._set_view(xc, yc, hw, hh)

    def _set_view(self, xc, yc, hw, hh):
        """Show the window centered at (xc, yc), half-sizes (hw, hh), kept on the chip."""
        H, W = self.img.shape
        xc = min(max(xc, hw - 0.5), W - 0.5 - hw)
        yc = min(max(yc, hh - 0.5), H - 0.5 - hh)
        self.ax.set_xlim(xc - hw, xc + hw)
        self.ax.set_ylim(yc + hh, yc - hh)
        self.fig.canvas.draw_idle()

    def _pan_px(self, dx_px, dy_px):
        """Pan the view by a scroll of (dx_px, dy_px) screen points: the image follows the fingers.

        Redrawing the window costs ~100-150 ms on a Retina screen (the image's resampling, the
        key panel, sliders, 1,792 x 3 rings), too slow to follow a trackpad. So while
        scrolling, what is already on screen is moved: the axes' last drawn pixels are cached
        once and put back shifted by the scroll (canvas.restore_region with an offset, well
        under a millisecond), the uncovered strip left in the axes' background color; the view
        limits move with it. PAN_SETTLE_MS after the last scroll, the window is redrawn
        properly (the new strip, the ticks, the image at its stride)."""
        canvas = self.fig.canvas
        (x0, x1), (y1, y0) = self.ax.get_xlim(), self.ax.get_ylim()
        bbox = self.ax.get_window_extent()
        ratio = getattr(canvas, "device_pixel_ratio", 1) or 1   # screen points -> device pixels
        ppx = (x1 - x0) / max(bbox.width, 1)   # data units per device pixel
        ppy = (y1 - y0) / max(bbox.height, 1)
        H, W = self.img.shape
        hw, hh = (x1 - x0) / 2, (y1 - y0) / 2
        xc = min(max((x0 + x1) / 2 - dx_px * ratio * ppx, hw - 0.5), W - 0.5 - hw)
        yc = min(max((y0 + y1) / 2 - dy_px * ratio * ppy, hh - 0.5), H - 0.5 - hh)
        if not getattr(canvas, "supports_blit", False):
            self._set_view(xc, yc, hw, hh)
            return
        if not self._fast:   # cache what is on screen now
            self._fast = True
            self._pan_region = canvas.copy_from_bbox(self.ax.bbox)
            self._pan_shift = [0.0, 0.0]
            if self._pan_timer is None:
                self._pan_timer = canvas.new_timer(interval=self.PAN_SETTLE_MS)
                self._pan_timer.single_shot = True
                self._pan_timer.add_callback(self._end_fast_pan)
        # the content moves opposite to the view: right / down on screen, in device pixels
        self._pan_shift[0] += ((x0 + x1) / 2 - xc) / ppx
        self._pan_shift[1] += ((y0 + y1) / 2 - yc) / ppy
        self.ax.set_xlim(xc - hw, xc + hw)
        self.ax.set_ylim(yc + hh, yc - hh)
        sx, sd = (int(round(v)) for v in self._pan_shift)
        rx1, ry1, rx2, ry2 = self._pan_region.get_extents()   # buffer pixels, y down
        self.ax.draw_artist(self.ax.patch)                     # the uncovered strip: background
        canvas.restore_region(self._pan_region,
                              bbox=(rx1 + max(0, -sx), ry1 + max(0, -sd), rx2 - max(0, sx), ry2 - max(0, sd)),
                              xy=(rx1 + sx, ry1 + sd))
        canvas.blit(self.ax.bbox)
        self._pan_timer.stop()
        self._pan_timer.start()

    def _end_fast_pan(self):
        """The scrolling has stopped: redraw the window at the new view."""
        self._fast = False
        self._pan_region = None
        self.fig.canvas.draw_idle()

    def _connect_scroll_pan(self):
        """Two-finger (trackpad) scrolling pans the view. In a Qt window the raw scroll is read
        from the widget (matplotlib's scroll_event keeps only its vertical steps); elsewhere
        (ipympl) scroll pans up / down and shift-scroll left / right."""
        canvas = self.fig.canvas
        try:
            from matplotlib.backends.qt_compat import QtCore
        except ImportError:
            QtCore = None
        if QtCore is not None and hasattr(canvas, "installEventFilter"):
            adjuster = self

            class _ScrollPan(QtCore.QObject):
                def eventFilter(self, obj, event):
                    if event.type() != QtCore.QEvent.Type.Wheel:
                        return False
                    d = event.pixelDelta()
                    if d.isNull():   # a mouse wheel, not a trackpad: one notch ~ 40 px
                        a = event.angleDelta()
                        dx, dy = a.x() / 3, a.y() / 3
                    else:
                        dx, dy = d.x(), d.y()
                    # Qt's deltas are already in the content's direction (positive y: the
                    # content moves down the screen), as the image's rows grow down too
                    adjuster._pan_px(dx, dy)
                    return True

            self._scroll_filter = _ScrollPan()
            canvas.installEventFilter(self._scroll_filter)
            return

        def on_scroll(e):
            if e.inaxes is not self.ax:
                return
            step = 60 * (e.step or 0)
            if e.key == "shift":
                self._pan_px(step, 0)
            else:
                self._pan_px(0, -step)

        canvas.mpl_connect("scroll_event", on_scroll)

    def _after_zoom(self, e):
        """One zoom per `z`: once the box is drawn, back to dragging."""
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if toolbar is None or "zoom" not in str(getattr(toolbar, "mode", "")) or e.inaxes is not self.ax:
            return
        timer = self.fig.canvas.new_timer(interval=60)  # after the toolbar finishes its zoom
        timer.single_shot = True
        timer.add_callback(lambda: "zoom" in str(toolbar.mode) and toolbar.zoom())
        timer.start()
        self._zoom_timer = timer  # keep a reference until it fires

    def _draw_key(self):
        """The "How to use" panel, laid out in inches from its top: each row is as tall as its
        wrapped text, the text wraps to the panel's width, and the whole panel scales down to
        fit the window's height. Redrawn on resize."""
        import textwrap
        from matplotlib.patches import FancyBboxPatch

        if self._key_ax is not None:
            self._key_ax.remove()
        W_in, H_in = self.fig.get_size_inches()
        left, top = 0.15, 0.75  # inches from the figure's top-left
        width = (self.PANEL_RIGHT - 0.012) * W_in - left
        pad, line = 0.18, 0.19  # line height (in)
        icon_x, text_x = 0.2, 1.05  # inches from the panel's left edge
        wrap = int(np.clip((width - text_x - 0.2) / 0.075, 16, 30))  # characters that fit

        rows = [("mouse", m) for m in KEY_MOUSE] + [("rule", None)] + [("keys", k) for k in KEY_KEYS]
        heights = [
            0.18 if kind == "rule"
            else max(0.42 if kind == "mouse" else 0.32, len(textwrap.wrap(item[-1], wrap)) * line + 0.12)
            for kind, item in rows
        ]
        note = textwrap.fill(f"A fix fades out within ~{WARP_REACH} chambers; the corners stay "
                             "put. Every change saves.", wrap + 8)
        height = pad + 0.35 + sum(heights) + 0.1 + note.count("\n") * 0.16 + 0.3 + pad
        scale = min(1.0, (H_in - top - 0.2) / height)  # laid out at full size, then scaled to fit
        fs = lambda size: size * scale

        ax = self.fig.add_axes([left / W_in, 1 - (top + height * scale) / H_in,
                                width / W_in, height * scale / H_in])
        width = width / scale
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # unscaled inches from the panel's top
        ax.axis("off")
        th = self.theme
        ax.add_patch(FancyBboxPatch((0.02, 0.02), width - 0.04, height - 0.04,
                                    boxstyle="round,pad=0,rounding_size=0.12", fc=th["panel"], ec=th["panel_edge"],
                                    lw=0.8))
        ax.text(icon_x, pad + 0.12, "How to use", fontsize=fs(12), weight="bold", va="center", color=th["text"])

        def mouse(y, which):
            """A small mouse, its left or right button shaded."""
            w, h = 0.26, 0.34
            ax.add_patch(FancyBboxPatch((icon_x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.09",
                                        fc=th["key"], ec=th["line"], lw=0.9))
            bx = icon_x if which == "left" else icon_x + w / 2
            ax.add_patch(FancyBboxPatch((bx + 0.025, y - h / 2 + 0.025), w / 2 - 0.05, h / 2 - 0.05,
                                        boxstyle="round,pad=0,rounding_size=0.03", fc="#4A90D9", ec="none"))
            ax.plot([icon_x + w / 2] * 2, [y - h / 2, y], color=th["line"], lw=0.8)
            ax.plot([icon_x, icon_x + w], [y, y], color=th["line"], lw=0.8)

        def keycap(x, y, k):
            w = 0.16 + 0.07 * len(k)
            ax.add_patch(FancyBboxPatch((x, y - 0.12), w, 0.24, boxstyle="round,pad=0,rounding_size=0.04",
                                        fc=th["key"], ec=th["key_edge"], lw=0.9))
            ax.text(x + w / 2, y, k, fontsize=fs(9), weight="bold", ha="center", va="center", family="monospace",
                    color=th["text"])
            return x + w + 0.06

        y = pad + 0.35
        for (kind, item), h in zip(rows, heights):
            yc = y + h / 2
            if kind == "rule":
                ax.plot([icon_x, width - icon_x], [yc, yc], color=th["rule"], lw=0.8)
            else:
                if kind == "mouse":
                    which, marker, color, ms, text = item
                    mouse(yc, which)
                    ax.plot([icon_x + 0.5], [yc], marker, ms=ms * scale, mfc="none", mec=color, mew=1.8)
                    x = text_x
                else:
                    keys, text = item
                    x = icon_x
                    for k in keys.split():
                        x = keycap(x, yc, k)
                    x = max(text_x, x + 0.04)   # the text starts after the keys, never under them
                ax.text(x, yc, textwrap.fill(text, wrap), fontsize=fs(9.5), va="center", linespacing=1.25,
                        color=th["text"])
            y += h
        ax.text(icon_x, y + 0.12, note, fontsize=fs(8.5), color=th["note"], va="top", linespacing=1.3)
        self._key_ax = ax

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
        if self.dumbbell is not None:
            self._dumbbell_templates()
            self._update_artists()
        else:
            self._set_diameter(self._roi, self.radius)
            self._set_diameter(self._bg_in, self.bg_inner)
        self._persist()
        self.fig.canvas.draw_idle()

    def _on_bg_outer(self, val):
        self.bg_outer = int(val)
        if self.dumbbell is not None:   # the margin of the background ring beyond the outline
            self._dumbbell_templates()
            self._update_artists()
        else:
            self._set_diameter(self._bg_out, self.bg_outer)
        self._persist()
        self.fig.canvas.draw_idle()

    def _on_dumbbell(self, key, val):
        self.dumbbell[key] = float(val)
        self._dumbbell_templates()
        self._update_artists()
        self._persist()
        self.fig.canvas.draw_idle()

    def _dumbbell_templates(self):
        """The dumbbell's outline and its background ring's outer edge, relative to the
        reaction chamber's center, thinned to OUTLINE_POINTS each."""
        def thin(outline):
            keep = np.linspace(0, len(outline) - 1, self.OUTLINE_POINTS).astype(int)
            return outline[keep]

        self._db_outline = thin(dumbbell_outline(self.radius, self.dumbbell))
        self._db_bg = thin(dumbbell_outline(self.radius, self.dumbbell, margin=self.bg_outer))

    @staticmethod
    def _place(template, pts):
        """Every chamber's copy of an outline, as one (x, y) line with NaN between copies."""
        xs = pts[:, :1] + template[None, :, 0]
        ys = pts[:, 1:] + template[None, :, 1]
        gap = np.full((len(pts), 1), np.nan)
        return np.hstack([xs, gap]).ravel(), np.hstack([ys, gap]).ravel()

    def _nearest(self, x, y):
        d = np.hypot(self.corners[:, 0] - x, self.corners[:, 1] - y)
        k = int(np.argmin(d))
        return k if d[k] <= self.pick_radius else None

    def _chamber_at(self, x, y):
        """The (row, col) of the chamber whose ROI contains (x, y); None elsewhere."""
        cx, cy = self.centers()
        d = np.hypot(cx - x, cy - y)
        r, c = np.unravel_index(np.argmin(d), d.shape)
        return (int(r), int(c)) if d[r, c] <= self.radius else None

    def _on_press(self, e):
        # while the toolbar's zoom/pan tool is active, clicks belong to it, not the grid
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if toolbar is not None and str(getattr(toolbar, "mode", "")):
            return
        if e.inaxes is not self.ax or e.xdata is None:
            return
        # a chamber other than the corner ones: a local correction (right-click removes it)
        rc = self._chamber_at(e.xdata, e.ydata)
        corners = {(0, 0), (0, self.n_cols - 1), (self.n_rows - 1, self.n_cols - 1), (self.n_rows - 1, 0)}
        if rc is not None and rc not in corners:
            if e.button == 3:
                if self.controls.pop(rc, None) is not None:
                    self._persist_warp()
                    self._redraw()
            elif e.button == 1:
                cx, cy = self.centers()
                bx, by = bilinear_grid(self.corners, self.n_rows, self.n_cols)
                self._drag = ("chamber", rc, (e.xdata, e.ydata), (cx[rc] - bx[rc], cy[rc] - by[rc]))
                self._start_blit()
            return
        if e.button == 1:  # a corner handle
            self._drag = self._nearest(e.xdata, e.ydata)
            if self._drag is not None:
                self._start_blit()

    def _persist_warp(self):
        """Save the corrections, and the corners they are offsets from."""
        self._persist()
        if self.warp_path is not None:
            self.save_warp()

    def _moving(self):
        """The artists that follow a drag: the grid, its rings and labels, the handles and the
        local corrections."""
        return [self._grid_pts, self._roi, self._bg_in, self._bg_out, *self._index_labels,
                self._corner_pts, *self._labels, self._ctrl_pts]

    def _start_blit(self):
        """For a drag: draw everything once without the moving artists and keep that as the
        background, so each move redraws only the grid (blitting). A full redraw on every
        move is what makes dragging lag. Falls back to full redraws where blitting is
        unsupported."""
        self._last_drag_draw = time.monotonic()
        self._background = None
        if not getattr(self.fig.canvas, "supports_blit", False):
            return
        for a in self._moving():
            a.set_animated(True)
        self.fig.canvas.draw()
        self._background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def _end_blit(self):
        if getattr(self, "_background", None) is not None:
            for a in self._moving():
                a.set_animated(False)
            self._background = None
        self.fig.canvas.draw_idle()

    def _on_motion(self, e):
        if self._drag is not None and e.inaxes is self.ax and e.xdata is not None:
            if isinstance(self._drag, tuple):  # a local correction
                _, rc, (x0, y0), (dx0, dy0) = self._drag
                self.controls[rc] = (dx0 + e.xdata - x0, dy0 + e.ydata - y0)
            else:
                self.corners[self._drag] = [e.xdata, e.ydata]
            now = time.monotonic()
            if now - getattr(self, "_last_drag_draw", 0.0) < 1.0 / self.DRAG_FPS:
                return  # throttled: the release (or the next move) draws the latest position
            self._last_drag_draw = now
            self._update_artists()
            if getattr(self, "_background", None) is not None:
                self.fig.canvas.restore_region(self._background)
                for a in self._moving():
                    self.ax.draw_artist(a)
                self.fig.canvas.blit(self.ax.bbox)
            else:
                self.fig.canvas.draw_idle()

    def _on_release(self, e):
        if self._drag is not None:
            self._update_artists()
            self._end_blit()
            if isinstance(self._drag, tuple):
                self._persist_warp()
            else:
                self._persist()  # auto-persist placement after each drag
        self._drag = None

    @staticmethod
    def _set_diameter(coll, r):
        """Resize every ring in an EllipseCollection to radius `r` (pixels)."""
        d = np.full(len(coll.get_offsets()), 2.0 * r)
        coll.set_widths(d)  # matplotlib >= 3.6
        coll.set_heights(d)

    def _redraw(self):
        self._update_artists()
        self.fig.canvas.draw_idle()

    def _update_artists(self):
        """Move the grid, its rings and labels, and the handles to the current corners."""
        cx, cy = self.centers()
        pts = np.column_stack([cx.ravel(), cy.ravel()])
        self._grid_pts.set_data(cx.ravel(), cy.ravel())

        if self.dumbbell is not None:
            self._roi.set_data(*self._place(self._db_outline, pts))
            self._bg_out.set_data(*self._place(self._db_bg, pts))
        else:
            for coll in (self._roi, self._bg_in, self._bg_out):
                coll.set_offsets(pts)  # one array update per ring type
        for lab, p in zip(self._index_labels, pts):  # index labels track their buttons
            lab.xy = p

        self._corner_pts.set_data(self.corners[:, 0], self.corners[:, 1])
        for lab, (x, y) in zip(self._labels, self.corners):
            lab.set_position((x, y))

        if hasattr(self, "_ctrl_pts"):  # local corrections, and their count above the image
            rc = list(self.controls)
            self._ctrl_pts.set_data([cx[r, c] for r, c in rc], [cy[r, c] for r, c in rc])
            n = len(rc)
            self.ax.set_title(
                "drag the cyan corners to place the grid, then any circle to fix it locally\n"
                f"{n} local correction{'s' if n != 1 else ''}"
                + (f" → {self.warp_path.name}" if self.warp_path else " (not saved: no warp_path)"),
                fontsize=10,
            )
