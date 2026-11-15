"""Start an experiment's workup from the package's notebook templates (htmek_processing/templates/)."""

import json
from datetime import date as _date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

TEMPLATES = Path(__file__).parent / "templates"
NOTEBOOKS = {"process.ipynb": "1-Process.ipynb", "analysis.ipynb": "2-Analysis.ipynb"}
QC_FOLDERS = ("1_stitch", "2_expression", "3_buttons", "4_standards", "5_assay")   # 5_figures/qc/, in pipeline order
FOLDERS = ("0_config", "2_masks", "3_intermediate", "4_exports", *(f"5_figures/qc/{q}" for q in QC_FOLDERS), "6_interpretation")


def _package_version():
    try:
        return version("htmek-processing")
    except PackageNotFoundError:
        return "unknown"


def create_experiment_notebooks(folder, experiment, *, date=None, drive=None, pinlist=None, overwrite=False):
    """Write a new experiment's `1-Process.ipynb` and `2-Analysis.ipynb` into `folder` from the package's templates,
    and create the standard folders (0_config, 2_masks, 3_intermediate, 4_exports, 5_figures/qc/<1_stitch, 2_expression,
    3_buttons, 4_standards, 5_assay>, 6_interpretation).

    The templates' placeholders are filled: {{EXPERIMENT}} with `experiment` (e.g. "MBO-057"), {{DATE}} with `date`
    (default today, YYYY-MM-DD), {{DRIVE}} with `drive` (the experiment's folder on the external drive) and {{PINLIST}}
    with `pinlist` (default "../_config/_pinlists/pinlist.csv"). The header markdown is left for the experiment's own
    description. Each notebook records the package version it came from in its metadata (`htmek_template_version`).
    An existing notebook is not overwritten unless `overwrite`. Returns the paths written."""
    folder = Path(folder)
    for sub in FOLDERS:
        (folder / sub).mkdir(parents=True, exist_ok=True)
    fill = {"{{EXPERIMENT}}": experiment, "{{DATE}}": date or _date.today().isoformat(),
            "{{DRIVE}}": str(drive) if drive else "/Volumes/Data_T7/HTMEK-data/uVAST/" + experiment,
            "{{PINLIST}}": str(pinlist) if pinlist else "../_config/_pinlists/pinlist.csv"}
    out = []
    for template, name in NOTEBOOKS.items():
        dst = folder / name
        if dst.exists() and not overwrite:
            raise FileExistsError(f"{dst} exists (pass overwrite=True to replace it)")
        nb = json.loads((TEMPLATES / template).read_text())
        for cell in nb["cells"]:
            src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
            for key, value in fill.items():
                src = src.replace(key, value)
            lines = src.split("\n")
            cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        nb.setdefault("metadata", {})["htmek_template_version"] = _package_version()
        dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
        out.append(dst)
    return out
