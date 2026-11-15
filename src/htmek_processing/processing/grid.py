"""Grid geometry: chamber centers from a placement, saving and loading placements, and
snapping or centering a grid onto buttons, spots or chambers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def grid_centers(x0, y0, col_pitch, row_pitch, angle=0.0, n_rows=8, n_cols=8):
    """Centers for an n_rows x n_cols grid anchored at the (row1, col1) center,
    rotated by `angle` degrees (clockwise in image coords) about that anchor.

    Returns (cx, cy), each shaped (n_rows, n_cols).
    """
    th = np.deg2rad(angle)
    c, s = np.cos(th), np.sin(th)

    j = np.arange(n_cols)[None, :]  # column index -> rotated x-axis
    i = np.arange(n_rows)[:, None]  # row index    -> rotated y-axis
    cx = x0 + j * col_pitch * c - i * row_pitch * s
    cy = y0 + j * col_pitch * s + i * row_pitch * c
    return (
        np.broadcast_to(cx, (n_rows, n_cols)).astype(float),
        np.broadcast_to(cy, (n_rows, n_cols)).astype(float),
    )


def bilinear_grid(corners, n_rows=8, n_cols=8):
    """Interpolate an n_rows x n_cols grid of centers from 4 corner-button centers.

    `corners` is 4 (x, y) points ordered [top-left, top-right, bottom-right, bottom-left]
    (the centers of the corner buttons). Bilinear interpolation handles rotation, shear,
    and slight keystone. Returns (cx, cy), each (n_rows, n_cols)."""
    TL, TR, BR, BL = (np.asarray(c, float) for c in corners)
    U = np.broadcast_to(np.linspace(0, 1, n_cols)[None, :], (n_rows, n_cols))[..., None]
    V = np.broadcast_to(np.linspace(0, 1, n_rows)[:, None], (n_rows, n_cols))[..., None]

    # interpolate along the top and bottom edges, then between them
    top = (1 - U) * TL + U * TR
    bot = (1 - U) * BL + U * BR
    P = (1 - V) * top + V * bot
    return P[..., 0], P[..., 1]


def save_corners(path, corners, *, radius=None, bg_inner=None, bg_outer=None, dumbbell=None, extra=None):
    """Save the 4 grid corners [[x,y],...] to JSON. If any of radius/bg_inner/bg_outer is
    given, save the full mask as a dict {corners, radius, bg_inner, bg_outer}; otherwise
    save a bare corners list. `dumbbell` (the shape, see dumbbell_mask) is saved with them,
    and `extra` (a dict, e.g. a paired grid's neck_width) is merged in. Returns the path."""
    import json

    corners = np.asarray(corners, float).tolist()
    if radius is None and bg_inner is None and bg_outer is None:
        payload = corners
    else:
        payload = {
            "corners": corners,
            "radius": radius,
            "bg_inner": bg_inner,
            "bg_outer": bg_outer,
        }
        if dumbbell is not None:
            payload["dumbbell"] = {k: float(v) for k, v in dumbbell.items()}
        payload.update(extra or {})

    Path(path).write_text(json.dumps(payload, indent=2))
    return str(path)


def load_mask(path):
    """Read a saved placement -> {corners, radius, bg_inner, bg_outer, dumbbell} (missing
    fields None), plus any other saved field (e.g. neck_width). Accepts both the bare
    corners-list format and the full-mask dict format.

    bg_inner is always the button radius: the background annulus starts at the button's
    edge, so it is never set by hand (any saved value is ignored)."""
    import json

    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        mask = {**data, **{k: data.get(k) for k in ("corners", "radius", "bg_inner", "bg_outer", "dumbbell")}}
        mask["bg_inner"] = mask["radius"]
        return mask
    return {"corners": data, "radius": None, "bg_inner": None, "bg_outer": None, "dumbbell": None}


# --------------------------------------------------------------------------- #
# Dumbbells: a reaction chamber and its DNA chamber, joined by the neck
# --------------------------------------------------------------------------- #

# The shape's fields, in pixels: where the DNA chamber's center sits from the reaction
# chamber's (dna_dx, dna_dy), the DNA chamber's radius, and the neck's width. The reaction
# chamber's radius is the placement's `radius`.
DUMBBELL_FIELDS = ("dna_dx", "dna_dy", "dna_radius", "neck_width")


def dumbbell_mask(radius, dumbbell, margin=0.0):
    """The dumbbell as a boolean stamp: a disc of `radius` (the reaction chamber) at the
    origin, a disc of dumbbell["dna_radius"] at (dna_dx, dna_dy), and a strip
    dumbbell["neck_width"] wide joining their centers; grown by `margin` px on every side
    (for the background ring). Returns (mask, (x0, y0)): the stamp and the image offset of
    its pixel (0, 0) from the reaction chamber's center (pixel units, the center at 0, 0)."""
    from scipy import ndimage

    dx, dy = float(dumbbell["dna_dx"]), float(dumbbell["dna_dy"])
    r_dna, half_neck = float(dumbbell["dna_radius"]), float(dumbbell["neck_width"]) / 2.0
    reach = max(radius, r_dna) + margin + 2
    x0 = int(np.floor(min(0.0, dx) - reach))
    y0 = int(np.floor(min(0.0, dy) - reach))
    x1 = int(np.ceil(max(0.0, dx) + reach))
    y1 = int(np.ceil(max(0.0, dy) + reach))
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1].astype(float)
    shape = (np.hypot(xx, yy) <= radius) | (np.hypot(xx - dx, yy - dy) <= r_dna)
    length = np.hypot(dx, dy)
    if length > 0:   # the neck: within half its width of the segment between the centers
        t = np.clip((xx * dx + yy * dy) / length**2, 0.0, 1.0)
        shape |= np.hypot(xx - t * dx, yy - t * dy) <= half_neck
    if margin > 0:
        shape = ndimage.distance_transform_edt(~shape) <= margin
    return shape, (x0, y0)


def dumbbell_outline(radius, dumbbell, margin=0.0):
    """The dumbbell's outline, (n, 2) points (x, y) from the reaction chamber's center, for
    drawing: the boundary of dumbbell_mask (grown by `margin`)."""
    from skimage import measure

    shape, (x0, y0) = dumbbell_mask(radius, dumbbell, margin)
    contour = max(measure.find_contours(np.pad(shape, 1).astype(float), 0.5), key=len)
    return np.column_stack([contour[:, 1] - 1 + x0, contour[:, 0] - 1 + y0])


def load_corners(path):
    """Read 4 grid corners from JSON (bare list or full-mask dict) -> (4, 2) float array."""
    return np.asarray(load_mask(path)["corners"], float)


WARP_REACH = 8  # chambers: how far a local correction's offset reaches (see warp_offsets)


def warp_offsets(controls, n_rows, n_cols, reach=WARP_REACH):
    """Local corrections to a 4-corner grid: (dx, dy), each (n_rows, n_cols), the offset at
    every chamber, interpolated from `controls` [[row, col, dx, dy], ...] over the grid's
    (row, col) with a Gaussian radial basis, so each correction fades to ~0 within about
    `reach` chambers. The four corner chambers are pinned at 0 unless they are controls."""
    from scipy.interpolate import RBFInterpolator

    pts = {(int(r), int(c)): (dx, dy) for r, c, dx, dy in controls}
    for rc in ((0, 0), (0, n_cols - 1), (n_rows - 1, 0), (n_rows - 1, n_cols - 1)):
        pts.setdefault(rc, (0.0, 0.0))
    rc = np.array(list(pts), float)
    d = np.array(list(pts.values()), float)
    rows, cols = np.mgrid[0:n_rows, 0:n_cols]
    off = RBFInterpolator(rc, d, kernel="gaussian", epsilon=1.5 / reach, degree=-1)(
        np.c_[rows.ravel(), cols.ravel()]
    )
    return off[:, 0].reshape(n_rows, n_cols), off[:, 1].reshape(n_rows, n_cols)


def load_warp(path):
    """The local corrections saved by GridAdjuster(warp_path=...): [[row, col, dx, dy], ...],
    or [] when `path` is None or doesn't exist."""
    import json

    path = Path(path) if path else None
    return json.loads(path.read_text())["controls"] if path and path.exists() else []


def save_warp(path, controls):
    """Save local corrections, {(row, col): (dx, dy)} or [[row, col, dx, dy], ...], to JSON."""
    import json

    items = (controls.items() if isinstance(controls, dict)
             else [((r, c), (dx, dy)) for r, c, dx, dy in controls])
    rows = [[int(r), int(c), float(dx), float(dy)] for (r, c), (dx, dy) in sorted(items)]
    Path(path).write_text(json.dumps({"controls": rows, "reach": WARP_REACH}, indent=1))
    return str(path)


def warped_grid(corners_path, warp_path, n_rows, n_cols):
    """Chamber centers (cx, cy) of a saved placement: the 4-corner bilinear grid from
    `corners_path` plus the local corrections in `warp_path` (none if it doesn't exist)."""
    cx, cy = bilinear_grid(load_mask(corners_path)["corners"], n_rows, n_cols)
    controls = load_warp(warp_path)
    from ..analysis.steps import tally_step

    name = Path(corners_path).stem.replace("_corners", "")
    tally_step("warped_grid", "Place the grid",
               ["4 corners placed manually (GridAdjuster);",
                f"a bilinear {n_rows} × {n_cols} grid between them,",
                "plus local corrections placed manually"],
               name, label=f"`{name}`: {len(controls)} local corrections", unit="grids")
    if not controls:
        return cx, cy
    dx, dy = warp_offsets(controls, n_rows, n_cols)
    return cx + dx, cy + dy


def unit_cell_average(image, cx, cy, *, half_px=(80, 200)):
    """The typical unit cell of a chip image: the median, over every chamber, of the window
    centered on its grid position (cx, cy).

    Each window is first scaled to its own median (0) and 99th percentile (1), so bright
    and dim chambers count alike. `half_px` is (half height, half width) in px; windows
    that run off the image are skipped. Returns a (2 * hy + 1, 2 * hx + 1) array whose
    center pixel is the grid position, so structures beside the button (e.g. the DNA
    chamber) show up at their offset from it."""
    hy, hx = half_px
    image = np.asarray(image, np.float32)
    windows = []
    for x, y in zip(np.ravel(cx), np.ravel(cy)):
        yi, xi = int(round(y)), int(round(x))
        if yi - hy < 0 or xi - hx < 0:
            continue
        w = image[yi - hy : yi + hy + 1, xi - hx : xi + hx + 1]
        if w.shape != (2 * hy + 1, 2 * hx + 1):
            continue  # runs off the bottom or right edge
        lo, hi = np.median(w), np.percentile(w, 99)
        windows.append((w - lo) / (hi - lo + 1e-6))
    return np.median(windows, axis=0)


def fit_chamber_wall(average, *, dx_range, dy_range, r_range):
    """Find a round chamber in a unit-cell average (unit_cell_average): the circle along
    which the image gradient, i.e. the chamber wall, is strongest on average.

    The center is searched over dx_range x dy_range (px from the average's center pixel,
    inclusive) and the radius over r_range, first on a 2 px grid and then on a 0.5 px grid
    around the best circle. Returns dict(dx_px, dy_px, radius_px, wall_contrast), with
    wall_contrast the mean gradient on the circle over the mean gradient of the whole
    average (> 1: the circle follows an edge)."""
    from scipy.ndimage import gaussian_filter, map_coordinates

    hy, hx = (np.asarray(average.shape) - 1) // 2
    gradient = np.hypot(*np.gradient(gaussian_filter(np.asarray(average, float), 1.5)))
    theta = np.linspace(0, 2 * np.pi, 180, endpoint=False)

    def wall(dx, dy, r):
        ys, xs = hy + dy + r * np.sin(theta), hx + dx + r * np.cos(theta)
        return map_coordinates(gradient, [ys, xs], order=1).mean()

    def search(dxs, dys, rs):
        return max((wall(dx, dy, r), dx, dy, r) for dx in dxs for dy in dys for r in rs)

    # coarse, then fine around the best coarse circle
    _, dx, dy, r = search(np.arange(dx_range[0], dx_range[1] + 1, 2.0),
                          np.arange(dy_range[0], dy_range[1] + 1, 2.0),
                          np.arange(r_range[0], r_range[1] + 1, 2.0))
    fine = np.arange(-2, 2.01, 0.5)
    score, dx, dy, r = search(dx + fine, dy + fine, r + fine)
    return dict(dx_px=float(dx), dy_px=float(dy), radius_px=float(r),
                wall_contrast=float(score / gradient.mean()))


def find_buttons(image, cx, cy, *, seed=(0, 0), half=70, reach=60, smooth=40, n_iter=3,
                 min_score=0.5):
    """Each chamber's button center, found by template matching near its grid point: for a
    chip whose array is not a perfect parallelogram (MBO-041: bowed by up to ~35 px in the
    middle), which a grid set by four corners can't follow.

    The template starts as the `seed` chamber's crop (2 * `half` + 1 px, at its grid point;
    pick a chamber whose grid point sits on its button, e.g. a corner) and is then remade
    `n_iter` times as the mean of the best-matched buttons, aligned. Each chamber is searched
    within `reach` px of its grid point by normalized cross-correlation, with a parabolic
    subpixel peak. The image's smooth illumination (Gaussian `smooth` px) is removed first, so
    tile brightness steps don't bias the match.

    Returns (bx, by, score): the centers and each match's correlation. Where the score is
    below `min_score` (no button seen) the grid point is kept."""
    from scipy import ndimage
    from skimage.feature import match_template

    img = np.asarray(image, np.float32)
    img = img - ndimage.gaussian_filter(img, smooth)
    gx, gy = np.asarray(cx, float), np.asarray(cy, float)

    def crop(x, y, h):
        x, y = int(round(x)), int(round(y))
        return img[y - h : y + h + 1, x - h : x + h + 1]

    template = crop(gx[seed], gy[seed], half)
    bx, by, score = gx.copy(), gy.copy(), np.zeros(gx.shape)
    for _ in range(n_iter):
        for idx in np.ndindex(gx.shape):
            win = crop(gx[idx], gy[idx], half + reach)
            if win.shape != (2 * (half + reach) + 1,) * 2:
                continue  # too close to the image edge
            m = match_template(win, template)
            j = np.unravel_index(np.argmax(m), m.shape)
            sub = [0.0, 0.0]
            for axis in (0, 1):  # parabola through the peak and its two neighbors
                if 0 < j[axis] < m.shape[axis] - 1:
                    lo = m[(j[0] - 1, j[1]) if axis == 0 else (j[0], j[1] - 1)]
                    hi = m[(j[0] + 1, j[1]) if axis == 0 else (j[0], j[1] + 1)]
                    den = lo - 2 * m[j] + hi
                    sub[axis] = 0.5 * (lo - hi) / den if den else 0.0
            by[idx] = round(gy[idx]) - reach + j[0] + sub[0]
            bx[idx] = round(gx[idx]) - reach + j[1] + sub[1]
            score[idx] = m[j]
        best = score > np.percentile(score, 30)  # remake the template from the best matches
        template = np.mean([
            ndimage.shift(crop(bx[i], by[i], half + 2),
                          (round(by[i]) - by[i], round(bx[i]) - bx[i]), order=1)[2:-2, 2:-2]
            for i in zip(*np.nonzero(best))], axis=0)
    weak = score < min_score
    bx[weak], by[weak] = gx[weak], gy[weak]
    return bx, by, score


def snap_to_spots(image, cx, cy, *, reach=50, spot_radius=22, min_snr=8,
                  area_px=(200, 2800), n_iter=8):
    """Move each grid point onto the bright, compact spot near it, e.g. a labeled spot that
    sits off the button center (MBO-041). `image` is the signal image (e.g. last pass minus
    first, lightly smoothed); `cx`, `cy` the grid (bilinear_grid).

    For each point: the brightest pixel within `reach` px, above the window's median by at
    least `min_snr` x the image noise (robust SD); its half-maximum region must cover
    `area_px` (min, max) pixels, so a chamber-sized edge or a single hot pixel doesn't count.
    The center is then refined by a weighted centroid within `spot_radius` px, `n_iter`
    times. Points without such a spot stay where they are.

    Returns (sx, sy, snapped): the new centers and which points moved."""
    from scipy import ndimage

    img = np.asarray(image, np.float32)
    sub = img[::7, ::7]
    noise = 1.4826 * np.median(np.abs(sub - np.median(sub)))
    yy, xx = np.mgrid[-reach : reach + 1, -reach : reach + 1]
    disc = np.hypot(yy, xx) <= reach
    sx, sy = np.array(cx, float), np.array(cy, float)
    snapped = np.zeros(sx.shape, bool)
    for idx in np.ndindex(sx.shape):
        x0, y0 = int(round(sx[idx])), int(round(sy[idx]))
        w = img[y0 - reach : y0 + reach + 1, x0 - reach : x0 + reach + 1]
        if w.shape != disc.shape:
            continue  # too close to the image edge
        w = np.where(disc, w - np.median(w[disc]), -np.inf)
        j = np.unravel_index(np.argmax(w), w.shape)
        if w[j] < min_snr * noise:
            continue
        region, _ = ndimage.label(w > w[j] / 2)
        if not area_px[0] <= (region == region[j]).sum() <= area_px[1]:
            continue
        px, py = xx[j], yy[j]
        for _ in range(n_iter):  # weighted centroid of the spot, recentered each time
            m = disc & (np.hypot(xx - px, yy - py) <= spot_radius)
            wt = np.clip(w, 0, None) * m
            px, py = (wt * xx).sum() / wt.sum(), (wt * yy).sum() / wt.sum()
        sx[idx], sy[idx], snapped[idx] = x0 + px, y0 + py, True
    return sx, sy, snapped


def center_on_spots(image, cx, cy, radius, *, reach=10, threshold_frac=0.3, min_snr=15,
                    area_frac=(0.6, 1.6)):
    """Move each grid point onto the center of its spot's shape, e.g. a labeled button in
    the post-labeling Cy5 scan (MBO-052). `image` is the signal image, lightly smoothed;
    `cx`, `cy` the grid; `radius` the button mask's radius (px).

    For each point: the local background is the median of a ring outside the search area
    (radius + reach/2 to radius + reach), the spot's height the window's 99th percentile above
    it. The spot is the connected region above `threshold_frac` x that height which overlaps
    the grid circle most, and its center is the region's unweighted centroid, so a bright rim
    on one side doesn't pull the center (snap_to_spots, a weighted centroid, does). A point
    moves only when the height is >= `min_snr` x the image noise (robust SD), the region
    covers `area_frac` (min, max) x the circle's area (round, button-sized), and the move is
    <= `reach` px. Points that don't move can take their neighbors' shift (shift_from_neighbors).

    Returns (sx, sy, moved)."""
    from scipy import ndimage

    img = np.asarray(image, np.float32)
    sub = img[::7, ::7]
    noise = 1.4826 * np.median(np.abs(sub - np.median(sub)))
    half = int(np.ceil(radius + reach)) + 2
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
    rr = np.hypot(yy, xx)
    window, ring = rr <= half, (rr > radius + reach / 2) & (rr <= half)
    circle_area = np.pi * radius**2
    sx, sy = np.array(cx, float), np.array(cy, float)
    moved = np.zeros(sx.shape, bool)
    for idx in np.ndindex(sx.shape):
        x0, y0 = int(round(sx[idx])), int(round(sy[idx]))
        w = img[y0 - half : y0 + half + 1, x0 - half : x0 + half + 1]
        if w.shape != window.shape:
            continue  # too close to the image edge
        local = np.median(w[ring])
        height = np.percentile(w[window], 99) - local
        if height < min_snr * noise:
            continue
        labels, _ = ndimage.label((w - local > threshold_frac * height) & window)
        in_circle = np.hypot(xx - (sx[idx] - x0), yy - (sy[idx] - y0)) <= radius
        counts = np.bincount(labels[in_circle & (labels > 0)])
        if counts.size < 2:
            continue
        region = labels == np.argmax(counts)
        if not area_frac[0] * circle_area <= region.sum() <= area_frac[1] * circle_area:
            continue
        px, py = xx[region].mean(), yy[region].mean()
        if np.hypot(x0 + px - sx[idx], y0 + py - sy[idx]) > reach:
            continue
        sx[idx], sy[idx], moved[idx] = x0 + px, y0 + py, True
    return sx, sy, moved


def _ring_kernels(radius, width=1.5):
    """Kernels that, correlated with an image's x and y gradients, sum the gradient pointing
    toward the center along a ring of `radius` px (high for a bright disc on a dark field)."""
    h = int(np.ceil(radius + 3 * width))
    yy, xx = np.mgrid[-h : h + 1, -h : h + 1].astype(float)
    rr = np.hypot(xx, yy)
    ring = np.exp(-0.5 * ((rr - radius) / width) ** 2)
    ring /= ring.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        ux, uy = np.nan_to_num(xx / rr), np.nan_to_num(yy / rr)
    return -ring * ux, -ring * uy


def center_on_discs(image, cx, cy, radius, *, reach=12, smooth=2.0, min_contrast=1.8,
                    inner=(22, 33), outer=(42, 50)):
    """Move each grid point onto the center of the bright disc around it, e.g. a reaction
    chamber filled with dye in the last SNAP-labeling frame (MBO-052). `cx`, `cy` the grid;
    `radius` the disc's radius (px).

    A ring Hough transform at a known radius: within `reach` px of each point, the offset
    whose ring collects the most edge pointing toward its center (the image smoothed by
    `smooth` px), refined to a sub-pixel peak. Flow channels and necks break only short arcs
    of the edge, so they barely move the peak. A fit counts when the disc stands out, its
    median between `inner` radii >= `min_contrast` x the median between `outer` radii (the
    field outside), and the peak is not at the edge of the search; points that fail stay put
    and can take their neighbors' shift (shift_from_neighbors).

    Returns (sx, sy, fitted, contrast)."""
    from scipy import ndimage, signal

    img = ndimage.gaussian_filter(np.asarray(image, np.float32), smooth)
    gy, gx = np.gradient(img)
    kx, ky = _ring_kernels(radius)
    pad = kx.shape[0] // 2 + reach + 2
    sy_, sx_ = np.mgrid[-reach : reach + 1, -reach : reach + 1]
    in_reach = np.hypot(sx_, sy_) <= reach
    m = int(np.ceil(outer[1])) + 1
    yy, xx = np.mgrid[-m : m + 1, -m : m + 1]
    rr = np.hypot(xx, yy)
    inner_m, outer_m = (rr >= inner[0]) & (rr <= inner[1]), (rr >= outer[0]) & (rr <= outer[1])

    def vertex(a, b, c):  # sub-pixel peak of a parabola through three points
        den = a - 2 * b + c
        return 0.0 if den == 0 or not np.isfinite(den) else float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))

    sx, sy = np.array(cx, float), np.array(cy, float)
    fitted = np.zeros(sx.shape, bool)
    contrast = np.full(sx.shape, np.nan)
    for idx in np.ndindex(sx.shape):
        x0, y0 = int(round(sx[idx])), int(round(sy[idx]))
        if min(y0, x0) - pad < 0 or y0 + pad >= img.shape[0] or x0 + pad >= img.shape[1]:
            continue  # too close to the image edge
        ys, xs = slice(y0 - pad, y0 + pad + 1), slice(x0 - pad, x0 + pad + 1)
        score = (signal.correlate(gx[ys, xs], kx, mode="same", method="fft")
                 + signal.correlate(gy[ys, xs], ky, mode="same", method="fft"))
        win = np.where(in_reach, score[pad - reach : pad + reach + 1, pad - reach : pad + reach + 1], -np.inf)
        py, px = np.unravel_index(np.argmax(win), win.shape)
        if not (0 < py < win.shape[0] - 1 and 0 < px < win.shape[1] - 1) or not (
                np.isfinite(win[py - 1, px]) and np.isfinite(win[py + 1, px])
                and np.isfinite(win[py, px - 1]) and np.isfinite(win[py, px + 1])):
            continue  # the peak is at the edge of the search
        nx = x0 + px - reach + vertex(win[py, px - 1], win[py, px], win[py, px + 1])
        ny = y0 + py - reach + vertex(win[py - 1, px], win[py, px], win[py + 1, px])
        w = img[int(round(ny)) - m : int(round(ny)) + m + 1, int(round(nx)) - m : int(round(nx)) + m + 1]
        if w.shape != rr.shape:
            continue
        contrast[idx] = np.median(w[inner_m]) / max(float(np.median(w[outer_m])), 1e-6)
        if contrast[idx] >= min_contrast:
            sx[idx], sy[idx], fitted[idx] = nx, ny, True
    return sx, sy, fitted, contrast


def shift_from_neighbors(cx, cy, sx, sy, moved, *, span=2, min_neighbors=3):
    """Give each point that didn't move (`moved` False) the median shift of the points that
    did within `span` rows and columns, when it has >= `min_neighbors` of them: a grid's
    placement error changes slowly across the chip, so a dim chamber follows its neighbors.

    Returns (fx, fy, filled): the centers and which points took their neighbors' shift."""
    cx, cy = np.asarray(cx, float), np.asarray(cy, float)
    dx, dy = np.asarray(sx, float) - cx, np.asarray(sy, float) - cy
    fx, fy = np.array(sx, float), np.array(sy, float)
    filled = np.zeros(cx.shape, bool)
    for r, c in np.ndindex(cx.shape):
        if moved[r, c]:
            continue
        rs, cs = slice(max(r - span, 0), r + span + 1), slice(max(c - span, 0), c + span + 1)
        near = moved[rs, cs]
        if near.sum() >= min_neighbors:
            fx[r, c] = cx[r, c] + np.median(dx[rs, cs][near])
            fy[r, c] = cy[r, c] + np.median(dy[rs, cs][near])
            filled[r, c] = True
    return fx, fy, filled


def save_centers(path, cx, cy, source):
    """Write centers (n_rows x n_cols) to a CSV: mark_row, mark_col, x, y, and `source`, an
    array of the same shape naming how each was set (e.g. "fit", "neighbors", "grid")."""
    import pandas as pd

    r, c = np.indices(np.shape(cx))
    pd.DataFrame({"mark_row": r.ravel(), "mark_col": c.ravel(), "x": np.ravel(cx), "y": np.ravel(cy),
                  "source": np.ravel(source)}).to_csv(path, index=False)


def load_centers(path):
    """(cx, cy), each n_rows x n_cols, from a save_centers CSV."""
    import pandas as pd

    t = pd.read_csv(path).sort_values(["mark_row", "mark_col"])
    shape = (t["mark_row"].max() + 1, t["mark_col"].max() + 1)
    return t["x"].to_numpy().reshape(shape), t["y"].to_numpy().reshape(shape)


def center_on_disc_sum(image, cx, cy, radius, *, reach=16, frac=0.995, smooth=1.0, ring=0):
    """Move each grid point to where a disc of `radius` px holds the most signal, searched within `reach` px of the
    grid point. The disc sum is computed at every offset (one convolution), and the point moves to the middle of the
    connected region within `frac` of the window's maximum; a chamber longer than the disc in one direction (the
    reaction chamber's wristwatch face, MBO-054) has a flat ridge there, and the middle of the ridge centers it.
    With `ring` > 0 the score is the disc mean minus the mean of a `ring`-px annulus just outside it, so a disc
    that spills onto a bright neighbor (a side channel, the next chamber) loses score. A point whose maximum lies on
    the edge of the window keeps the grid position.

    Returns (sx, sy, matched)."""
    from scipy import ndimage, signal

    img = ndimage.gaussian_filter(np.nan_to_num(np.asarray(image, np.float32)), smooth)
    r = int(np.ceil(radius + ring))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    rr = np.hypot(xx, yy)
    kernel = (rr <= radius) / (rr <= radius).sum()
    if ring:
        annulus = (rr > radius) & (rr <= radius + ring)
        kernel = kernel - annulus / annulus.sum()
    total = signal.fftconvolve(img, kernel.astype(np.float32), mode="same")
    sx, sy = np.array(cx, float), np.array(cy, float)
    matched = np.zeros(sx.shape, bool)
    for idx in np.ndindex(sx.shape):
        x0, y0 = int(round(sx[idx])), int(round(sy[idx]))
        w = total[max(y0 - reach, 0):y0 + reach + 1, max(x0 - reach, 0):x0 + reach + 1]
        if w.shape != (2 * reach + 1, 2 * reach + 1):
            continue
        py, px = np.unravel_index(np.argmax(w), w.shape)
        if not (0 < py < 2 * reach and 0 < px < 2 * reach):
            continue
        lab, _ = ndimage.label(w >= w.min() + frac * (w.max() - w.min()))
        my, mx = ndimage.center_of_mass(lab == lab[py, px])
        sx[idx], sy[idx] = x0 - reach + mx, y0 - reach + my
        matched[idx] = True
    return sx, sy, matched


def center_on_template(image, cx, cy, *, half, reach=12, n_iter=2, min_score=0.5, smooth=1.0):
    """Move each grid point onto its chamber by matching a template of the chamber's own shape, e.g. the reaction
    chamber's wristwatch face (round on the sides, squared and extended top and bottom), which a circle fit
    centers poorly (MBO-054). The template is the median of every chamber's crop (`half` px either side), rebuilt
    `n_iter` times from the re-centered crops; each point then moves to the normalized cross-correlation peak
    within `reach` px (sub-pixel). A point moves only when its peak score is >= `min_score` and not at the edge of
    the search; the rest can take their neighbors' shift (shift_from_neighbors). The template is built from crops at
    the grid points, so it keeps the grid's offset (about 5 px left on MBO-054); center_on_disc_sum does not.

    Returns (sx, sy, matched, score, template)."""
    from scipy import ndimage
    from skimage.feature import match_template

    img = ndimage.gaussian_filter(np.nan_to_num(np.asarray(image, np.float32)), smooth)
    sx, sy = np.array(cx, float), np.array(cy, float)
    big = half + reach

    def crop(x, y, h):
        x0, y0 = int(round(x)), int(round(y))
        if y0 - h < 0 or x0 - h < 0 or y0 + h + 1 > img.shape[0] or x0 + h + 1 > img.shape[1]:
            return None
        sub = ndimage.shift(img[y0 - h - 1:y0 + h + 2, x0 - h - 1:x0 + h + 2], (y0 - y, x0 - x), order=1)
        return sub[1:-1, 1:-1]

    def vertex(a, b, c):
        den = a - 2 * b + c
        return 0.0 if den == 0 or not np.isfinite(den) else float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))

    matched = np.zeros(sx.shape, bool)
    score = np.full(sx.shape, np.nan)
    template = None
    for _ in range(n_iter + 1):
        crops = [c for c in (crop(sx[i], sy[i], half) for i in np.ndindex(sx.shape)) if c is not None]
        template = np.median(np.stack(crops), axis=0)
        nx, ny = np.array(cx, float), np.array(cy, float)
        matched[:] = False
        for idx in np.ndindex(sx.shape):
            window = crop(cx[idx], cy[idx], big)   # always searched from the grid point
            if window is None:
                continue
            ncc = match_template(window, template)          # (2 reach + 1)^2 offsets
            py, px = np.unravel_index(np.argmax(ncc), ncc.shape)
            score[idx] = ncc[py, px]
            if score[idx] < min_score or not (0 < py < ncc.shape[0] - 1 and 0 < px < ncc.shape[1] - 1):
                continue
            nx[idx] = cx[idx] + px - reach + vertex(ncc[py, px - 1], ncc[py, px], ncc[py, px + 1])
            ny[idx] = cy[idx] + py - reach + vertex(ncc[py - 1, px], ncc[py, px], ncc[py + 1, px])
            matched[idx] = True
        sx, sy = nx, ny
    return sx, sy, matched, score, template
