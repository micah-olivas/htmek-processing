"""A record of where an experiment's raw images live.

Raw acquisitions stay on the external drive; the experiment folder keeps `1_raw-images/`
with `raw_sources.csv` (one row per acquisition) and a `README.md` generated from it.
`stitch_run(..., record_dir=...)` calls `record_raw_source` for every acquisition it
stitches, so the record fills in as the process notebook runs."""

import os
from datetime import date
from pathlib import Path

COLUMNS = ["group", "acquisition", "camera", "channel", "exposure_ms", "tiles", "passes", "start_time",
           "size_gb", "raw_path", "stitched_path", "recorded"]


def _folder_gb(path):
    """Total size of the files under `path` (GB), skipping macOS '._' sidecar files."""
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            if not f.startswith("._"):
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    return round(total / 1e9, 2)


def _group(src):
    """The acquisition's folder relative to `1_raw-images/` (e.g. 'Assays/NaK-series'), else its parent's name."""
    parts = Path(src).parts
    if "1_raw-images" in parts:
        rel = parts[parts.index("1_raw-images") + 1:-1]
        return "/".join(rel) if rel else "."
    return Path(src).parent.name


def _camera(attrs, name):
    """The camera, from the acquisition's metadata, else from a '_zyla' / '_kinetix' suffix in its name."""
    cam = attrs.get("camera")
    if isinstance(cam, dict) and cam.get("name"):
        return str(cam["name"]).lower()
    if attrs.get("lightpath"):
        return str(attrs["lightpath"]).lower()
    for c in ("kinetix", "zyla"):
        if c in name.lower():
            return c
    return ""


def raw_source_row(src, stitched, tiled, measure_size=True):
    """One row of the record for acquisition `src` (its folder), stitched to `stitched`; `tiled` is its open_tiled
    dataset (sizes and attrs). `measure_size=False` leaves size_gb empty (walking a large acquisition takes time)."""
    attrs = dict(tiled.attrs)
    cam = attrs.get("camera") if isinstance(attrs.get("camera"), dict) else {}
    name = Path(src).name
    return {
        "group": _group(src),
        "acquisition": name,
        "camera": _camera(attrs, name),
        "channel": attrs.get("filter") or attrs.get("channel") or "",
        "exposure_ms": attrs.get("exposure_ms", cam.get("exposure_ms", "")),
        "tiles": f"{tiled.sizes.get('row', '')} x {tiled.sizes.get('col', '')}",
        "passes": int(tiled.sizes.get("time", 1)),
        "start_time": str(attrs.get("start_time", ""))[:19],
        "size_gb": _folder_gb(src) if measure_size else "",
        "raw_path": str(Path(src)),
        "stitched_path": str(Path(stitched)),
        "recorded": date.today().isoformat(),
    }


def _acquisition(src):
    """The acquisition's folder: `src`, or its parent when `src` is the zarr group inside it (e.g. .../<run>/tiled_images,
    a folder with the 'tile' array)."""
    src = Path(src)
    return src.parent if (src / "tile").is_dir() else src


def record_raw_source(src, stitched, tiled, record_dir="1_raw-images"):
    """Add or update acquisition `src` in `<record_dir>/raw_sources.csv` (keyed by its raw path, so a re-run
    updates its row instead of adding one), then rewrite `<record_dir>/README.md` from the table. The folder's
    size is measured once, when the row is new. Returns the table."""
    import pandas as pd

    src = _acquisition(src)
    record_dir = Path(record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)
    csv = record_dir / "raw_sources.csv"
    table = pd.read_csv(csv, dtype=str, keep_default_na=False) if csv.exists() else pd.DataFrame(columns=COLUMNS)
    key = str(Path(src))
    old = table[table["raw_path"] == key]
    if old.empty:
        row = raw_source_row(src, stitched, tiled)
    else:   # the row exists: refresh it, keeping its measured size
        row = {**raw_source_row(src, stitched, tiled, measure_size=False), "size_gb": old.iloc[0]["size_gb"]}
    table = pd.concat([table[table["raw_path"] != key], pd.DataFrame([row])], ignore_index=True)
    table = table[COLUMNS].sort_values(["group", "acquisition"]).reset_index(drop=True)
    table.to_csv(csv, index=False)
    write_raw_readme(table, record_dir / "README.md")
    return table


def write_raw_readme(table, path):
    """`README.md` for `1_raw-images/`: where the raw acquisitions are, one table per group."""
    lines = ["# Raw images", "",
             "The raw acquisitions are stored on a separate drive, at the paths below;",
             "The same record is provided as a table in `raw_sources.csv`. This record is written by `htmek_processing` (`record_raw_source`,",
             "called from `stitch_run`) each time an acquisition is stitched.", ""]
    roots = sorted({str(Path(p).parent) for p in table["raw_path"]})
    if roots:
        common = os.path.commonpath(roots) if len(roots) > 1 else roots[0]
        lines += [f"Raw root: `{common}`", ""]
    for group, g in table.groupby("group", sort=True):
        lines += [f"## {group}", "",
                  "| acquisition | camera | channel | exposure (ms) | tiles | passes | started | size (GB) | raw path | stitched |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in g.itertuples():
            lines.append(f"| `{r.acquisition}` | {r.camera} | {r.channel} | {r.exposure_ms} | {r.tiles} | {r.passes} | "
                         f"{r.start_time} | {r.size_gb} | `{r.raw_path}` | `{r.stitched_path}` |")
        lines.append("")
    Path(path).write_text("\n".join(lines))
