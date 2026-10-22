"""Grid geometry: chamber centers from a placement, saving and loading placements, and
snapping a grid onto buttons or spots."""

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


def save_corners(path, corners, *, radius=None, bg_inner=None, bg_outer=None):
    """Save the 4 grid corners [[x,y],...] to JSON. If any of radius/bg_inner/bg_outer is
    given, save the full mask as a dict {corners, radius, bg_inner, bg_outer}; otherwise
    save a bare corners list. Returns the path."""
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

    Path(path).write_text(json.dumps(payload, indent=2))
    return str(path)


def load_mask(path):
    """Read a saved placement -> {corners, radius, bg_inner, bg_outer} (missing fields
    None). Accepts both the bare corners-list format and the full-mask dict format.

    bg_inner is always the button radius: the background annulus starts at the button's
    edge, so it is never set by hand (any saved value is ignored)."""
    import json

    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        mask = {k: data.get(k) for k in ("corners", "radius", "bg_inner", "bg_outer")}
        mask["bg_inner"] = mask["radius"]
        return mask
    return {"corners": data, "radius": None, "bg_inner": None, "bg_outer": None}


def load_corners(path):
    """Read 4 grid corners from JSON (bare list or full-mask dict) -> (4, 2) float array."""
    return np.asarray(load_mask(path)["corners"], float)


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
