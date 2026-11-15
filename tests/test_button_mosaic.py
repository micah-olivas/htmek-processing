"""button_mosaic: one tile per chamber, labeled, and crops off the image edge handled."""

import matplotlib
import numpy as np

matplotlib.use("Agg")

import htmek_processing as hp


def test_mosaic_tiles_and_labels():
    img = np.random.default_rng(0).normal(100, 5, (200, 300)).astype(np.float32)
    cy, cx = np.meshgrid([5.0, 100.0, 195.0], [10.0, 150.0], indexing="ij")   # 3 x 2 chambers, some at the edge
    tags = [["WT", "N41A"], ["F22*", "BLANK"], ["A5M", "G6F"]]
    fig = hp.button_mosaic(img, cx, cy, tags, np.full((3, 2), 2.3e6), half=10, radius=4, title="t")
    ax = fig.axes[0]
    assert ax.images[0].get_array().shape == (60, 40)
    texts = [t.get_text() for t in ax.texts]
    assert "F22tag" in texts and "r3 c2" in texts and texts.count("2.3M") == 6


def test_tag_suffix():
    assert hp.display_tag("A4*") == "A4tag" and hp.display_tag("WT") == "WT"
    hp.set_tag_suffix("φ")
    try:
        assert hp.display_tag("F22*") == "F22φ"
    finally:
        hp.set_tag_suffix("tag")
