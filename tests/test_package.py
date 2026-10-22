"""The package imports, exposes its names, and stitches as chip_utils did by default."""

import numpy as np

import htmek_processing as hp


def test_every_public_name_is_exported():
    missing = [name for name in hp.__all__ if not hasattr(hp, name)]
    assert not missing


def test_stitch_tiles_scalar_overlap_is_plain_reshape():
    # chip_utils (MBO-040) stitched by trimming each tile's trailing overlap and butting tiles;
    # with one overlap for both axes and no rotation the merged stitch_tiles must match it
    rng = np.random.default_rng(0)
    tiles = rng.integers(0, 4096, size=(3, 4, 20, 25), dtype=np.uint16)   # (row, col, y, x)
    overlap = 3
    trimmed = tiles[..., :-overlap, :-overlap]
    R, C, h, w = trimmed.shape
    expected = np.moveaxis(trimmed, -2, -3).reshape(R * h, C * w)
    assert np.array_equal(hp.stitch_tiles(tiles, overlap), expected)


def test_stitch_tiles_stacks_frames():
    tiles = np.arange(2 * 2 * 2 * 6 * 6).reshape(2, 2, 2, 6, 6)   # (time, row, col, y, x)
    out = hp.stitch_tiles(tiles, 1)
    assert out.shape == (2, 10, 10)
    assert np.array_equal(out[1], hp.stitch_tiles(tiles[1], 1))
