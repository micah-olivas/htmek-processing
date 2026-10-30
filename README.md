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
│   ├── run_notebook.py           # Runs a notebook headlessly, cell by cell, with a log
│   ├── compare_outputs.py        # Compares a notebook's outputs with a reference run
│   └── split_from_chip_utils.py  # Built src/ from the experiments' chip_utils.py (provenance)
├── tests/
└── src/htmek_processing/
    ├── __init__.py               # Re-exports every function, class and constant (hp.*)
    ├── processing/               # Images -> per-chamber numbers
    │   ├── layout.py             # Chamber identities: pinlists, column layouts, mirroring
    │   ├── io.py                 # Images, time axes, NADH standard images
    │   ├── stitch.py             # Tile scans: geometry, flatfields, stitching, per-chamber times
    │   ├── fit_flatfield.py      # BaSiC flatfield script, run by stitch.fit_flatfield (own uv env)
    │   ├── grid.py               # Chamber centers from 4 corners; warps; saved placements
    │   ├── grid_adjuster.py      # GridAdjuster: interactive grid placement
    │   ├── buttons.py            # magnify button-finding for the minichip
    │   └── quantify.py           # Per-chamber ROI and annulus statistics; kinetic traces; bleaching
    ├── analysis/                 # Per-chamber numbers -> rates, fits, QC
    │   ├── rates.py              # Lines and 4PL fits, the rate rules, depletion / r² flags, BLANK subtraction
    │   ├── mm.py                 # Michaelis-Menten fits, per chamber or pooled
    │   ├── standards.py          # NADH standard line per chamber; signal -> [NADH]
    │   ├── flagging.py           # Flag files and the interactive chamber flaggers
    │   ├── plots.py              # QC figures, chip maps, the shared plot style and colors
    │   └── kinetics_qc.py        # Rate and MM QC figures (orientation, fits, Km by scan, rate definitions)
    └── report/                   # PDFs and per-variant statistics
        ├── chamber.py            # Chamber report: one page per chamber
        ├── volcano.py            # Per-variant fold changes, bootstrap p-values, volcano plots
        └── experiment.py         # Experiment report: summary, chip maps, rate maps, trends, distributions
```

Every name is available as `hp.<name>` (`import htmek_processing as hp`); the reports also
as `hr.<name>` (`from htmek_processing import report as hr`). Import from a submodule
(`from htmek_processing.analysis.rates import four_pl`) only when the flat name is not enough.

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