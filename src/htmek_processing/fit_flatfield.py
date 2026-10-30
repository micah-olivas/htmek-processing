# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "basicpy==2.0.0",
#     "scipy>=1.11,<1.13",  # basicpy needs < 1.13; without a floor uv picks an unbuildable 1.6
#     "zarr>=3",
#     "tifffile",
#     "numpy",
# ]
# ///
"""Pseudo flatfield for the Kinetix tiles of one acquisition, estimated from the tiles
themselves with BaSiC
(Peng et al. 2017, Nat Commun 8:14836; basicpy, the Python port of the Fiji plugin). There
is no measured Kinetix flatfield yet; replace this one when there is.

The method is 261014-On-chip/fit_flatfield.py's, unchanged; this copy takes the acquisition
and the output as arguments (the raw images live on the T7) and leaves out tiles that are all
zero (the frames a timecourse never acquired).

BaSiC models every tile as  tile = true image x F + D:
  - F is the flatfield, the illumination and vignetting every tile shares (scaled to mean 1);
  - D is an additive darkfield, not fitted here: a fitted D can soak up real signal.
F is estimated from the stack of every tile: what all tiles share at the same pixel is
shading, and the chip's pattern averages out because it falls differently on each tile.

F is smooth, so the tiles are first averaged down by DOWNSAMPLE (3,200 -> 400 px; BaSiC
works at 128 px internally anyway), and the fitted F is smoothed (SMOOTH_PX) and interpolated
back to the full tile.

It runs in its own throwaway environment (uv reads the dependencies above), so the analysis
env is untouched. The process notebook runs it when the file is missing:

    PYTHONNOUSERSITE=1 uv run fit_flatfield.py <tile group or acquisition folder> <output.tif> [camera offset]

In the package it is run by stitch.fit_flatfield(), which skips it when the output exists.

Writes  <output.tif>   F (float32, 3,200 x 3,200, mean 1)
        <output.json>  settings and stats
"""

import json
import sys
from pathlib import Path

import numpy as np
import tifffile
import zarr
from basicpy import BaSiC
from scipy import ndimage

CAMERA_DARK = 80  # Kinetix offset (counts), subtracted before fitting. Measured 2026-10-09
#                   from the GFP scans at 500 and 1,000 ms (same tiles, same chip): if signal
#                   scales with exposure, offset = 2 x I(500 ms) - I(1,000 ms); median over
#                   the 25 tiles 79 counts (interquartile 77-83).
DOWNSAMPLE = 8    # block mean before fitting
SMOOTH_PX = 150   # Gaussian smoothing of F (full-tile px): the shading is smooth, but the
#                   chip's columns leave a faint imprint (the tile step is ~7.2 columns, so
#                   they don't average out completely)


def read_tiles(folder, camera_dark=CAMERA_DARK):
    """Every acquired camera tile of one acquisition, averaged down, as (n_tiles, y, x)
    float32, and the number of all-zero tiles left out. `folder` is the group holding 'tile',
    or the acquisition folder with one such group inside."""
    group = folder if (folder / "tile").exists() else next(p for p in folder.iterdir() if (p / "tile").exists())
    tile = zarr.open_group(str(group), mode="r")["tile"]
    *lead, h, w = tile.shape
    out, empty = [], 0
    for idx in np.ndindex(*lead):  # one tile in memory at a time
        raw = np.asarray(tile[idx], np.float32)
        if not raw.any():          # never acquired (a timecourse's unused frames)
            empty += 1
            continue
        t = raw - camera_dark
        out.append(t.reshape(h // DOWNSAMPLE, DOWNSAMPLE, w // DOWNSAMPLE, DOWNSAMPLE).mean(axis=(1, 3)))
    return np.stack(out), (h, w), empty


def fit(src, out, camera_dark=CAMERA_DARK):
    stack, shape, empty = read_tiles(src, camera_dark)
    basic = BaSiC(get_darkfield=False)
    basic.fit(stack)
    small = np.asarray(basic.flatfield, np.float32)
    small = ndimage.gaussian_filter(small, SMOOTH_PX / DOWNSAMPLE, mode="nearest")
    flat = ndimage.zoom(small, DOWNSAMPLE, order=3).astype(np.float32)[: shape[0], : shape[1]]
    flat /= flat.mean()  # mean 1: correcting keeps the overall intensity
    out.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(out.with_suffix(".tif"), flat)
    stats = dict(
        method="BaSiC (basicpy 2.0.0), over every acquired tile of one acquisition",
        source=str(src), n_tiles=len(stack), empty_tiles_left_out=empty, downsample=DOWNSAMPLE,
        smooth_px=SMOOTH_PX, camera_dark_counts=camera_dark,
        get_darkfield=False, flatfield_min=float(flat.min()), flatfield_max=float(flat.max()),
        flatfield_p1_p99=[float(v) for v in np.percentile(flat, [1, 99])],
        corner_over_center=float(flat[:200, :200].mean() / flat[1500:1700, 1500:1700].mean()),
    )
    out.with_suffix(".json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"{len(stack)} tiles ({empty} empty left out); F {stats['flatfield_min']:.2f}-"
          f"{stats['flatfield_max']:.2f}, corner / center {stats['corner_over_center']:.2f} -> {out.with_suffix('.tif')}")


def main():
    src, out = (Path(a) for a in sys.argv[1:3])
    fit(src, out, float(sys.argv[3]) if len(sys.argv) > 3 else CAMERA_DARK)


if __name__ == "__main__":
    main()
