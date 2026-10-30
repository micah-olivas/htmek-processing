"""run_chamberqc: the files it hands chamberqc, and the tables it reads back."""

import json

import numpy as np
import pandas as pd

import htmek_processing as hp


def test_inputs_and_outputs(tmp_path, monkeypatch):
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / "pyproject.toml").write_text("")
    cx, cy = np.meshgrid([10.0, 20.0], [5.0, 15.0])
    seen = {}

    def fake_run(cmd, check, env):  # stands in for chamberqc: checks the manifest, writes a table
        manifest = json.loads(open(cmd[-1]).read())
        item = manifest["images"][0]
        seen.update(manifest=manifest, image=np.load(f"{cmd[-1].rsplit('/', 1)[0]}/{item['image']}"),
                    centers=pd.read_csv(f"{cmd[-1].rsplit('/', 1)[0]}/{item['centers']}"))
        out = tmp_path / "out" / manifest["chip"]
        out.mkdir(parents=True)
        seen["centers"].assign(anomaly_score=1.0, anomaly_rank=1).to_csv(out / f"chamberqc_{item['name']}.csv")

    monkeypatch.setattr("subprocess.run", fake_run)
    tags = pd.DataFrame({"mark_row": [0, 1], "mark_col": [0, 1], "tag": ["WT", "BLANK"]})
    out = hp.run_chamberqc({"gfp": (np.ones((30, 30)), cx, cy, 4, {"tile_pitch": 2896})}, chip="X",
                           out_dir=tmp_path / "out", tags=tags, repo=tmp_path / "repo")
    item = seen["manifest"]["images"][0]
    assert item["radius"] == 4 and item["tile_pitch"] == 2896 and seen["image"].shape == (30, 30)
    c = seen["centers"].set_index(["mark_row", "mark_col"])
    assert c.loc[(1, 0), "x_px"] == 10.0 and c.loc[(1, 0), "y_px"] == 15.0
    assert c.loc[(0, 0), "tag"] == "WT"
    assert pd.isna(c.loc[(0, 1), "tag"])  # no tag given: written empty
    assert len(out["gfp"]) == 4
