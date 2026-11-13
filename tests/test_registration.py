import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

import htmek_processing as hp


def _chip(shape=(2400, 2400), seed=0):
    """A chip-like image: a grid of bright rings (chamber walls) on noise."""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    img = np.zeros(shape, np.float32)
    for cy in range(60, shape[0], 120):
        for cx in range(60, shape[1], 120):
            r = np.hypot(yy[cy - 50:cy + 50, cx - 50:cx + 50] - cy, xx[cy - 50:cy + 50, cx - 50:cx + 50] - cx)
            img[cy - 50:cy + 50, cx - 50:cx + 50] += np.exp(-((r - 35) ** 2) / 8)
    return img + np.random.default_rng(seed).normal(0, 0.02, shape).astype(np.float32)


def test_scan_shift_recovers_a_known_shift():
    ref = _chip()
    mov = ndimage.shift(ref, (7.0, -12.0), order=1)   # mov's chip sits 7 px lower, 12 px left
    dy, dx = hp.scan_shift(ref, mov, n=3, half=400)
    assert abs(dy - 7.0) < 0.5 and abs(dx + 12.0) < 0.5


def test_cached_shift_measures_once(tmp_path):
    calls = []

    def measure():
        calls.append(1)
        return 1.5, -2.25

    p = tmp_path / "sub" / "shift.json"
    assert hp.cached_shift(p, measure) == (1.5, -2.25)
    assert hp.cached_shift(p, measure) == (1.5, -2.25)
    assert len(calls) == 1 and p.exists()


def test_split_channel():
    assert hp.split_channel("ester_hydrolysis_zyla|cy5") == ("ester_hydrolysis_zyla", "cy5", "ester_hydrolysis_zyla_cy5")
    assert hp.split_channel("cy5_bg_500ms") == ("cy5_bg_500ms", None, "cy5_bg_500ms")


def test_save_qc_fig(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = hp.save_qc_fig(fig, "2_buttons/2b-gfp", "2b_test", root=tmp_path)
    assert (out / "2b_test.pdf").exists() and (out / "2b_test.png").exists()


def test_offset_value_scalar_or_dark_image(tmp_path):
    import tifffile
    from htmek_processing.processing.stitch import _offset_value

    assert _offset_value(None) == 0.0 and _offset_value(111.5) == 111.5
    dark = np.full((8, 8), 111.5, np.float32)
    tifffile.imwrite(tmp_path / "dark.tif", dark)
    assert np.allclose(_offset_value(tmp_path / "dark.tif"), dark) and _offset_value(dark).shape == (8, 8)
