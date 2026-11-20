import numpy as np

from htmek_processing.processing.legacy import tile_origins_from_lattice
from htmek_processing.processing.stitch import place_tiles


def _chip(H, W, px=200.5, py=115.5):
    """A chamber lattice: bright discs on a dark field, with a little unevenness."""
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    dy = (yy % py) - py / 2
    dx = (xx % px) - px / 2
    img = 100 + 900 * ((dx ** 2 + dy ** 2) < 35 ** 2) + 300 * (((dx + px / 2) ** 2 + dy ** 2) < 25 ** 2)
    return img * (1 + 0.1 * np.sin(yy / 900) * np.cos(xx / 700))


def test_origins_recovered_and_placed():
    rng = np.random.default_rng(3)
    R = C = 4
    h = w = 1024
    step = 1024 - 6   # ~6 px nominal overlap
    truth = np.stack(np.meshgrid(np.arange(R) * step, np.arange(C) * step, indexing="ij"), -1).astype(int)
    truth += rng.integers(-5, 6, truth.shape)   # uneven stage: +-5 px per tile
    truth -= truth.reshape(-1, 2).min(0)
    H, W = truth[..., 0].max() + h, truth[..., 1].max() + w
    chip = _chip(H, W)
    tiles = np.stack([np.stack([chip[y0:y0 + h, x0:x0 + w] for (y0, x0) in row]) for row in truth])
    g = tile_origins_from_lattice(tiles, pitch_x=(190, 210), pitch_y=(108, 123), overlap_px=6)
    got = np.asarray(g["tile_origin"])
    assert np.abs(got - truth).max() <= 1
    img = place_tiles(tiles.astype(np.float32), got, (H, W))
    assert np.nanmedian(np.abs(img - chip)) < 0.05   # float32 rounding of values ~100-1000
