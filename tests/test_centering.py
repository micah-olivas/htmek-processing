import numpy as np

import htmek_processing as hp


def _grid(n=3, pitch=150, start=120):
    v = start + pitch * np.arange(n, dtype=float)
    return np.meshgrid(v, v)


def test_center_on_spots_finds_offset_spots_despite_a_rim():
    cx, cy = _grid()
    img = np.random.default_rng(0).normal(100, 2, (480, 480)).astype(np.float32)
    yy, xx = np.mgrid[:480, :480]
    true_x, true_y = cx + 5, cy - 4
    for x, y in zip(true_x.ravel(), true_y.ravel()):
        r = np.hypot(xx - x, yy - y)
        img[r <= 12] += 200
        img[(r <= 12) & (xx > x + 8)] += 400  # a bright rim on the right
    sx, sy, moved = hp.center_on_spots(img, cx, cy, 12)
    assert moved.all()
    assert np.allclose(sx, true_x, atol=1.0) and np.allclose(sy, true_y, atol=1.0)


def test_center_on_discs_and_neighbors():
    cx, cy = _grid()
    img = np.full((480, 480), 50, np.float32)
    yy, xx = np.mgrid[:480, :480]
    true_x, true_y = cx - 6, cy + 3
    for i, (x, y) in enumerate(zip(true_x.ravel(), true_y.ravel())):
        if i != 4:  # the middle chamber stays dark
            img[np.hypot(xx - x, yy - y) <= 37] = 200
        img[(np.abs(xx - x) < 15) & (np.abs(yy - y) < 70) & (np.hypot(xx - x, yy - y) > 37)] = 120  # a channel
    sx, sy, fitted, contrast = hp.center_on_discs(img, cx, cy, 37)
    assert fitted.sum() == 8 and not fitted[1, 1]
    assert np.allclose(sx[fitted], true_x[fitted], atol=0.7) and np.allclose(sy[fitted], true_y[fitted], atol=0.7)
    fx, fy, filled = hp.shift_from_neighbors(cx, cy, sx, sy, fitted)
    assert filled[1, 1] and abs(fx[1, 1] - true_x[1, 1]) < 0.7 and abs(fy[1, 1] - true_y[1, 1]) < 0.7


def test_centers_round_trip(tmp_path):
    cx, cy = _grid()
    hp.save_centers(tmp_path / "c.csv", cx, cy, np.full(cx.shape, "fit"))
    lx, ly = hp.load_centers(tmp_path / "c.csv")
    assert np.allclose(lx, cx) and np.allclose(ly, cy)


def test_chamber_outside_button_skips_the_off_center_button():
    img = np.full((200, 200), 10.0)
    yy, xx = np.mgrid[:200, :200]
    img[np.hypot(xx - 100, yy - 100) <= 37] = 50.0            # dye in the chamber
    img[np.hypot(xx - 92, yy - 95) <= 14] = 1000.0            # a bright button, off center
    med, n = hp.chamber_outside_button(img, np.array([[100.0]]), np.array([[100.0]]), np.array([[92.0]]),
                                       np.array([[95.0]]), chamber_radius=37, button_radius=14)
    assert med[0, 0] == 50.0 and n[0, 0] > 3000
    run, _ = hp.chamber_outside_button(np.stack([img, img * 2]), np.array([[100.0]]), np.array([[100.0]]),
                                       np.array([[92.0]]), np.array([[95.0]]), chamber_radius=37, button_radius=14)
    assert run.shape == (2, 1, 1) and run[1, 0, 0] == 100.0
