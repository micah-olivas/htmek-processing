"""Build src/htmek_processing/ from the experiments' chip_utils.py (run once, kept for the record).

The base is MBO-040's chip_utils.py (report v1.9.4). Its sections 3 and 4 (tiled acquisitions,
flatfield) are taken from MBO-042/044's copy instead, which adds per-gap overlaps, camera rotation
and offset to stitching (_gap_trims, measure_tile_overlaps) and button snapping (find_buttons,
snap_to_spots, moved to grid.py). Every function and class is copied verbatim, with the comments
above it; only module docstrings and the imports between modules are new. Sections 3 and 4 share
a module because they call each other.

    python dev/split_from_chip_utils.py      (from the repo root; refuses to overwrite modules)
"""

import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

EXPERIMENTS = Path.home() / "Research/Experiments"
BASE = EXPERIMENTS / "MBO-040-hAcyP1-AFMtag-Canonical-261002/chip_utils.py"
STITCH_SOURCE = EXPERIMENTS / "MBO-042-Print-Buffer-Testing/261014-On-chip/chip_utils.py"
OUT = Path("src/htmek_processing")

# module -> (docstring, [chip_utils sections]); in import order
MODULES = {
    "layout": "Chamber identities: which variant is in each chamber (pinlists, column layouts).",
    "io": "Loading images, time axes, and NADH standards.",
    "stitch": "Tiled acquisitions: open a tile scan, flatfield-correct and stitch it, and time each\nchamber; flatfield estimation and correction.",
    "grid": "Grid geometry: chamber centers from a placement, saving and loading placements, and\nsnapping a grid onto buttons or spots.",
    "quantify": "Quantification: per-chamber intensities for one frame or a whole stack.",
    "grid_adjuster": "GridAdjuster: drag the grid's four corners onto the image (interactive).",
    "plots": "QC plots: grid overlays, chamber crops over time, chip heatmaps and sparklines; the\nshared plot style.",
    "fit": "Fitting: initial rates (lines, 4PL), per-chamber calibration, Michaelis-Menten fits.",
    "report": "Chamber reports: one PDF page per chamber, and the page key.",
    "flagging": "Flagging chambers: the shared JSON format and the interactive flaggers.",
    "buttons": "magnify button-finding for the minichip (8x8 adaptation of htmek.buttons_pipe).",
}
# first line of each chip_utils section in BASE -> module ("report" starts inside section 8)
SECTIONS = [(46, "layout"), (127, "io"), (301, "stitch"), (509, "stitch"), (604, "grid"),
            (742, "grid_adjuster"), (1224, "quantify"), (1569, "plots"), (2399, "report"),
            (4052, "flagging"), (4942, "fit"), (5609, "buttons")]
STITCH_SECTIONS = (3, 4)            # taken from STITCH_SOURCE
TO_GRID = {"find_buttons", "snap_to_spots"}
HEADER_IMPORTS = {                  # chip_utils' own top-level imports, added where used
    "re": "import re", "tempfile": "import tempfile", "Path": "from pathlib import Path",
    "np": "import numpy as np", "pd": "import pandas as pd", "plt": "import matplotlib.pyplot as plt",
}
STDLIB = {"re", "tempfile", "Path"}
BANNER = re.compile(r"^# -{20,} #\s*$")
TITLE = re.compile(r"^# (\d+)\. (.*)$")


def chunks(path):
    """[(name(s), first line, text)] for every top-level def/class/assignment after the imports:
    each with the comments and blank lines above it."""
    text = Path(path).read_text()
    lines = text.splitlines(keepends=True)
    body = ast.parse(text).body
    last_import = max(n.end_lineno for n in body if isinstance(n, (ast.Import, ast.ImportFrom)))
    out, prev = [], last_import
    for n in body:
        if n.end_lineno <= last_import:
            continue
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            names = [n.name]
        elif isinstance(n, ast.Assign):
            names = [t.id for t in n.targets if isinstance(t, ast.Name)]
        elif isinstance(n, ast.AnnAssign):
            names = [n.target.id]
        else:
            raise ValueError(f"{path}:{n.lineno}: unexpected top-level {type(n).__name__}")
        start = n.decorator_list[0].lineno if getattr(n, "decorator_list", None) else n.lineno
        out.append((names, start, "".join(lines[prev:n.end_lineno])))
        prev = n.end_lineno
    return out


def section_of(path, line):
    """chip_utils section number of `line` in `path` (from its banners)."""
    number = 0
    for k, text in enumerate(Path(path).read_text().splitlines()[:line], 1):
        m = TITLE.match(text)
        if m:
            number = int(m.group(1))
    return number


def used_names(code):
    """Every name `code` uses, except in a function that imports that name itself (e.g. a
    local `import tempfile`), so it isn't imported again at the top."""
    used = set()
    for node in ast.parse(code).body:
        local = {(a.asname or a.name).split(".")[0] for n in ast.walk(node)
                 if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        used |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} - local
    return used


def main():
    if OUT.exists() and any(OUT.glob("*.py")):
        sys.exit(f"{OUT} already has modules: move them aside to rebuild")

    # ---- collect the chunks: BASE, with sections 3-4 from STITCH_SOURCE --------------------
    module_of_line = lambda line: [m for start, m in SECTIONS if line >= start][-1]
    parts = defaultdict(list)        # module -> [(names, text)]
    for names, line, text in chunks(BASE):
        if section_of(BASE, line) not in STITCH_SECTIONS:
            parts[module_of_line(line)].append((names, text))
    grid_extra = []
    for names, line, text in chunks(STITCH_SOURCE):
        if section_of(STITCH_SOURCE, line) in STITCH_SECTIONS:
            (grid_extra if set(names) & TO_GRID else parts["stitch"]).append((names, text))
    parts["grid"] += grid_extra

    owner = {}
    for module, items in parts.items():
        for names, _ in items:
            for name in names:
                if name in owner:
                    raise ValueError(f"{name} defined in both {owner[name]} and {module}")
                owner[name] = module

    # ---- write each module ------------------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    deps = defaultdict(set)
    for module in MODULES:
        items = parts[module]
        code = "".join(text for _, text in items)
        # drop the module's first section banner (it becomes the docstring); later banners stay
        lines = code.splitlines(keepends=True)
        k = 0
        while k < len(lines) and (not lines[k].strip() or BANNER.match(lines[k]) or TITLE.match(lines[k])):
            k += 1
        code = "".join(lines[k:])

        used = used_names(code)
        own = {n for names, _ in items for n in names}
        imports = ["from __future__ import annotations", ""]
        std = [HEADER_IMPORTS[n] for n in ("re", "tempfile", "Path") if n in used]
        third = [HEADER_IMPORTS[n] for n in ("np", "pd", "plt") if n in used]
        imports += std + ([""] if std else []) + third
        local = defaultdict(list)
        for name in sorted(used & set(owner) - own, key=str.lower):
            local[owner[name]].append(name)
            deps[module].add(owner[name])
        if local:
            imports.append("")
        for other in MODULES:
            if local.get(other):
                imports.append(f"from .{other} import " + ", ".join(local[other]))
        header = f'"""{MODULES[module]}"""\n\n' + "\n".join(imports).rstrip() + "\n\n\n"
        (OUT / f"{module}.py").write_text(header + code.rstrip() + "\n")
        print(f"{module:14s} {len(own):3d} names, {code.count(chr(10)):5d} lines, "
              f"imports from {sorted(local) or '-'}")

    # ---- cycles would fail at import ---------------------------------------------------------
    def visit(node, path):
        for nxt in deps[node]:
            if nxt in path:
                sys.exit(f"import cycle: {' -> '.join(path[path.index(nxt):] + [nxt])}")
            visit(nxt, path + [nxt])
    for module in MODULES:
        visit(module, [module])

    # ---- the package: every name, private ones too, so it replaces chip_utils one for one ----
    lines = ['"""HT-MEK chip image processing: from raw tile scans to per-chamber rates and fits.',
             "",
             "Modules follow the pipeline: layout (chamber identities), io (images, standards), stitch",
             "(tile scans, flatfield), grid and grid_adjuster (chamber positions), quantify (per-chamber",
             "intensities), plots (QC), fit (rates, calibration, Michaelis-Menten), report (chamber",
             "reports), flagging (excluding chambers) and buttons (magnify button-finding).",
             "",
             "Every function is also available from the package itself, as it was from chip_utils:",
             "",
             "    import htmek_processing as hp",
             '"""', "", '__version__ = "0.1.0"', ""]
    public = []
    for module in MODULES:
        names = [n for names, _ in parts[module] for n in names]
        lines.append(f"from .{module} import (  # noqa: F401")
        lines += [f"    {n}," for n in names]
        lines.append(")")
        public += [n for n in names if not n.startswith("_")]
    lines += ["", "__all__ = ["] + [f'    "{n}",' for n in public] + ["]", ""]
    (OUT / "__init__.py").write_text("\n".join(lines))
    print(f"__init__: {len(owner)} names ({len(public)} public)")


if __name__ == "__main__":
    main()
