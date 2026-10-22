"""Loading images, time axes, and NADH standards."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def load_image(src):
    """Load a 2-D image from a path (.tif/.tiff via tifffile, .npy via numpy) or pass an
    array / xarray DataArray straight through. Returns a squeezed numpy array."""
    if isinstance(src, (str, Path)):
        p = str(src)
        if p.lower().endswith((".tif", ".tiff")):
            import tifffile

            return np.asarray(tifffile.imread(p)).squeeze()
        if p.lower().endswith(".npy"):
            return np.asarray(np.load(p)).squeeze()
        raise ValueError(f"unsupported image file (expected .tif/.tiff/.npy): {p}")
    return np.asarray(src).squeeze()


def to_2d_frame(img, frame=None, reducer="max"):
    """Collapse a (possibly 3-D time-stack) to a single 2-D frame for display/grid
    placement.

    2-D input is returned unchanged. For a 3-D (time, y, x) stack, `frame=<int>` picks that
    timepoint; `frame=None` projects over time with `reducer` ('max'|'mean'|'median'|'min').
    A max projection usually gives the crispest buttons to place the grid against."""
    a = np.asarray(img).squeeze()
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        return a[int(frame)] if frame is not None else getattr(np, reducer)(a, axis=0)
    raise ValueError(
        f"expected a 2-D image or 3-D (time, y, x) stack, got shape {a.shape}."
    )


def time_coord(stack, time_coord="seconds"):
    """Elapsed-time values for a stack's frames, or None. Reads the xarray coord named
    `time_coord` (default 'seconds') if present; falls back to frame index otherwise."""
    coords = getattr(stack, "coords", None)
    if coords is not None and time_coord in coords:
        return np.asarray(coords[time_coord].values, float)
    return None


def n_used_frames(seconds):
    """Number of frames actually acquired = 1 + index of the last frame with seconds > 0.
    Trailing unused acquisition slots (seconds == 0, blank/fill-value frames) are excluded.
    Returns len(seconds) if no positive times are present (can't tell)."""
    sec = np.asarray(seconds, float)
    nz = np.flatnonzero(sec > 0)
    return int(nz.max()) + 1 if nz.size else len(sec)


# NADH standards are named {type}_{conc}{unit}_{exposure}ms_{power}power[_{replicate}]
_CONC_UNIT_TO_UM = {"nM": 1e-3, "uM": 1.0, "mM": 1e3, "M": 1e6}


# concentration+unit immediately before '_<exposure>ms', with optional '_<power>power' and
# trailing '_<replicate>'. Tolerates extra descriptor tokens in between (e.g.
# SOLA_swappedObjective).
_STD_RE = re.compile(
    r"(?P<conc>\d[\d_]*?)(?P<unit>nM|uM|mM|M)_(?P<exposure>\d+)ms"
    r"(?:_(?P<power>\d+)power)?(?:_(?P<replicate>\d+))?$"
)


def parse_standard_name(name, pattern=_STD_RE):
    """Parse a standard's folder/file name into {type, conc, conc_unit, conc_uM,
    exposure_ms, power_pct, replicate}. The default handles e.g.
    'NADH_125uM_500ms_100power', 'NADH_15_625uM_500ms_100power_2', and
    'NADH_SOLA_swappedObjective_1mM_1000ms'.

    `pattern` is a regex (str or compiled) searched against the basename; it must define
    named groups `conc` and `unit`, and may define `exposure`, `power`, `replicate`, `type`.
    If `type` isn't captured it defaults to the text before the match. Set it to match any
    naming convention, e.g. parse_standard_name(name, r'(?P<conc>\\d+)(?P<unit>uM)_...')."""
    import os

    rx = pattern if hasattr(pattern, "search") else re.compile(pattern)
    base = os.path.basename(str(name).rstrip("/"))
    m = rx.search(base)
    if not m:
        raise ValueError(
            f"name {base!r} does not match pattern {getattr(rx, 'pattern', rx)!r}"
        )

    g = m.groupdict()
    conc = float(g["conc"].replace("_", "."))  # underscore = decimal point
    unit = g["unit"]

    return {
        "type": g.get("type") or base[: m.start()].rstrip("_"),
        "conc": conc,
        "conc_unit": unit,
        "conc_uM": conc * _CONC_UNIT_TO_UM[unit],
        "exposure_ms": int(g["exposure"]) if g.get("exposure") else None,
        "power_pct": int(g["power"]) if g.get("power") else None,
        "replicate": int(g["replicate"]) if g.get("replicate") else 1,
    }


def parse_concentration(name, label=None, *, after=None):
    """Concentration in µM from a run's name, with '_' as the decimal point. Either
      - `label`: the number and unit right BEFORE '_<label>', e.g.
        parse_concentration("NADH_kinetics_156_25uM_BzP_zyla_500ms", "BzP") -> 156.25, or
      - `after`: the number and unit right AFTER '<after>_', e.g.
        parse_concentration("NADH_standard_62_5uM_zyla_500ms", after="standard") -> 62.5.
    Read from the name rather than the acquisition's 'info' attr, which can carry the
    wrong unit."""
    import os

    base = os.path.basename(str(name).rstrip("/"))
    conc = r"(\d[\d_]*?)(nM|uM|mM|M)"
    if after is not None:
        m = re.search(rf"{re.escape(after)}_{conc}(?:_|$)", base)
        what = f"'{after}_<conc><unit>'"
    else:
        m = re.search(rf"{conc}_{re.escape(label)}(?:_|$)", base)
        what = f"'<conc><unit>_{label}'"
    if not m:
        raise ValueError(f"no {what} in {base!r}")
    return float(m.group(1).replace("_", ".")) * _CONC_UNIT_TO_UM[m.group(2)]


def load_standard_images(
    folder,
    *,
    glob_pattern="*",
    subpath="tiled_images",
    consolidated=False,
    pattern=_STD_RE,
    sort_by_conc=True,
):
    """Build an ordered {label: 2-D image} dict from a standards directory (set the dir
    once). Each subfolder matching `glob_pattern` is parsed with
    parse_standard_name(pattern=...); its `subpath` (a zarr 'tiled_images' store, or a
    .tif/.npy file) is loaded. Sorted low->high concentration; labels are '{conc:g}uM'
    (with '#n' appended on collisions/replicates)."""
    import glob as _glob
    import os

    items = []
    for f in sorted(_glob.glob(os.path.join(str(folder), glob_pattern))):
        if not os.path.isdir(f):
            continue
        try:
            meta = parse_standard_name(f, pattern)
        except ValueError:
            continue
        store = os.path.join(f, subpath) if subpath else f
        if os.path.isdir(store):
            import xarray as xr

            img = np.asarray(
                xr.open_zarr(store, consolidated=consolidated)["tile"]
            ).squeeze()
        else:
            img = load_image(store)
        items.append((meta["conc_uM"], meta["replicate"], img))

    if sort_by_conc:
        items.sort(key=lambda t: (t[0], t[1]))

    # label by concentration; repeats of a label get '#2', '#3', ...
    out, seen = {}, {}
    for conc, _rep, img in items:
        lab = f"{conc:g}uM"
        if lab in out:
            seen[lab] = seen.get(lab, 1) + 1
            lab = f"{lab}#{seen[lab]}"
        out[lab] = img
    return out
