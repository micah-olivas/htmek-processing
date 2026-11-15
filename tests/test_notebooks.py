"""create_experiment_notebooks: both notebooks from the templates, placeholders filled, folders made, no overwrite."""

import json

import pytest

import htmek_processing as hp


def test_notebooks_and_folders(tmp_path):
    paths = hp.create_experiment_notebooks(tmp_path / "exp", "MBO-099", date="2026-11-13", drive="/Volumes/X/MBO-099",
                                           pinlist="../_config/_pinlists/p.csv")
    assert [p.name for p in paths] == ["1-Process.ipynb", "2-Analysis.ipynb"]
    for p in paths:
        nb = json.loads(p.read_text())
        text = "".join("".join(c["source"]) for c in nb["cells"])
        assert "{{" not in text and "MBO-099" in text
        assert nb["metadata"]["htmek_template_version"]
    process = "".join("".join(c["source"]) for c in json.loads(paths[0].read_text())["cells"])
    assert 'DRIVE = Path("/Volumes/X/MBO-099")' in process and 'PINLIST_FILE = "../_config/_pinlists/p.csv"' in process
    assert (tmp_path / "exp/5_figures/qc/2_expression").is_dir()
    with pytest.raises(FileExistsError):
        hp.create_experiment_notebooks(tmp_path / "exp", "MBO-099")
