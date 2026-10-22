# HT-MEK Processing

Image processing and analysis for HT-MEK data.

## Overview

Takes the tiled images (Prismo zarr stores) and a chip layout (pinlist), and performs stitching, per-chamber intensity extraction, kinetic fitting, and QC figures. Supports minichip (8 x 8 chambers) and PS1.8K (56 x 32 chambers) architectures.

## Installation

### Install from source

```bash
# Install into the environment the notebooks run in, editable so changes apply on reload
uv pip install --python ~/.venvs/htmek/bin/python -e ~/repos/htmek-processing

# With the notebook widgets (GridAdjuster, the flaggers) and magnify button-finding
uv pip install --python ~/.venvs/htmek/bin/python -e "$HOME/repos/htmek-processing[interactive,magnify]"
```

### Requirements

- Python >=3.12
- Core dependencies: numpy, pandas, matplotlib, scipy, xarray, zarr (>=3), dask, scikit-image, tifffile, pillow
- `interactive` extra: ipython, ipywidgets. `GridAdjuster` and the Qt flaggers also need a Qt backend for `%matplotlib qt`
- `magnify` extra: magnify, for `buttons_pipe_minichip`

## Package Structure

```
htmek-processing/
├── pyproject.toml
├── dev/
│   └── split_from_chip_utils.py  # Builds src/ from the experiments' chip_utils.py (provenance)
└── src/htmek_processing/
    ├── __init__.py       # Re-exports every function, class and constant
    ├── layout.py         # Chamber identities: pinlists, column layouts, mirroring
    ├── io.py             # Images, time axes, NADH standards
    ├── stitch.py         # Tile scans: open, flatfield-correct, stitch, per-chamber times
    ├── grid.py           # Chamber centers from 4 corners; saved placements; button snapping
    ├── grid_adjuster.py  # GridAdjuster: interactive 4-corner placement
    ├── quantify.py       # Per-chamber ROI and background-annulus statistics; bleach correction
    ├── plots.py          # QC figures and the shared plot style
    ├── fit.py            # Lines, 4PL rates, calibration, Michaelis-Menten fits
    ├── report.py         # Chamber report PDFs
    ├── flagging.py       # Flag files and the interactive chamber flaggers
    └── buttons.py        # magnify button-finding for the minichip
```

## Usage

### In a notebook

The package exposes every name at the top level, making all utilities available from a single import:

```python
import htmek_processing as hp        # was: import chip_utils as cu

# Stitch a kinetics run once; later calls reopen the finished store
kin = hp.stitch_to_zarr(raw_dir, raw_dir / "stitched_images")

# Place the grid once (interactive); the placement is saved to JSON
ga = hp.GridAdjuster(kin.image.isel(time=1).values, columns="pinlist.csv",
                     corners_path="corners.json")

# Chamber centers, each chamber's imaging time, and its trace
m = hp.load_mask("corners.json")
cx, cy = hp.bilinear_grid(m["corners"], 56, 32)
sec = hp.chamber_seconds(kin["seconds"], cx, cy, kin.attrs["tile_px"])
traces = hp.kinetic_traces(kin.image, cx, cy, "pinlist.csv", radius=m["radius"],
                           bg=True, bg_inner=m["bg_inner"], bg_outer=m["bg_outer"],
                           seconds=sec)

# Initial rates
rates = hp.fit_rates(traces, value="median")
```

Modules can also be imported on their own, e.g. `from htmek_processing.fit import fit_mm_table`.