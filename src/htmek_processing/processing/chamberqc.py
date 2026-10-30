"""Button image QC with chamberqc (a separate package, ~/repos/chamberqc, with its own PyTorch
environment): score how unusual each button looks in an image, from the grid the pipeline
already placed (warps and per-scan shifts included).

chamberqc runs in its own uv environment, as stitch.fit_flatfield runs BaSiC, so PyTorch never
becomes a dependency here. Its scores go to the chamber report's QC tabs
(report.read_button_qc, report.button_qc_tabs).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

CHAMBERQC_REPO = Path("~/repos/chamberqc").expanduser()


def chamberqc_inputs(cx, cy, *, tags=None):
    """The crop-centers table chamberqc reads: mark_row, mark_col, x_px, y_px, tag, one row
    per chamber of the (n_rows, n_cols) center grids `cx`, `cy` (already moved by any scan
    shift). `tags`: a table with mark_row, mark_col and tag (e.g. any per-chamber export)."""
    cx, cy = np.asarray(cx, float), np.asarray(cy, float)
    rows, cols = np.indices(cx.shape)
    centers = pd.DataFrame({"mark_row": rows.ravel(), "mark_col": cols.ravel(),
                            "x_px": cx.ravel(), "y_px": cy.ravel()})
    if tags is not None:
        t = tags[["mark_row", "mark_col", "tag"]].drop_duplicates(["mark_row", "mark_col"])
        centers = centers.merge(t, on=["mark_row", "mark_col"], how="left")
    if "tag" not in centers:
        centers["tag"] = ""
    centers["tag"] = centers["tag"].fillna("")
    return centers


def run_chamberqc(images, *, chip, out_dir, tags=None, repo=CHAMBERQC_REPO, review=True, device=None):
    """Score the buttons of one or more images with chamberqc and return its tables.

    `images` = {name: (image, cx, cy, radius)} or {name: (image, cx, cy, radius, options)},
    with `image` a 2-D array (or a 2-D xarray), `cx`, `cy` the chamber centers in that image
    (n_rows x n_cols), `radius` the button ROI radius (px), and `options` a dict of
    target ("button" or "dna_chamber"), tile_pitch (px), description.
    Writes chamberqc's outputs under <out_dir>/<chip>/ (chamberqc_<name>.csv, review_<name>.pdf,
    score_map_<name>.png, run_<name>.json) in one chamberqc run, and returns
    {name: scores table} (one row per chamber: anomaly_score, anomaly_rank, ...)."""
    repo = Path(repo).expanduser()
    if not (repo / "pyproject.toml").exists():
        raise FileNotFoundError(f"chamberqc not found at {repo} (set repo=)")
    out_dir = Path(out_dir).resolve()
    with tempfile.TemporaryDirectory(prefix="chamberqc_") as tmp:
        tmp = Path(tmp)
        items = []
        for name, spec in images.items():
            image, cx, cy, radius, *rest = spec
            options = rest[0] if rest else {}
            np.save(tmp / f"{name}.npy", np.asarray(image, np.float32))
            chamberqc_inputs(cx, cy, tags=tags).to_csv(tmp / f"{name}_centers.csv", index=False)
            items.append({"name": name, "image": f"{name}.npy", "centers": f"{name}_centers.csv",
                          "radius": float(radius), **options})
        manifest = tmp / "manifest.json"
        manifest.write_text(json.dumps({"chip": chip, "out_dir": str(out_dir), "images": items}))
        cmd = ["uv", "run", "--project", str(repo), "chamberqc", "score-manifest", str(manifest)]
        if not review:
            cmd.append("--no-review")
        if device:
            cmd += ["--device", device]
        subprocess.run(cmd, check=True, env={**os.environ, "PYTHONNOUSERSITE": "1"})
    return {name: pd.read_csv(out_dir / chip / f"chamberqc_{name}.csv") for name in images}
