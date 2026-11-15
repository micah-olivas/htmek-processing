"""expression_calls: gain from first to last time point, the BLANK floor, and the variant call."""

import numpy as np
import pandas as pd

import htmek_processing as hp


def test_calls():
    rows = []
    gains = {(0, 0, "BLANK"): 0, (0, 1, "BLANK"): 10, (0, 2, "BLANK"): -10, (1, 0, "A5M"): 500, (1, 1, "A5M"): 0, (1, 2, "G6F"): 5}
    for (r, c, tag), gain in gains.items():
        rows += [{"mark_row": r, "mark_col": c, "tag": tag, "minutes": 20, "dumbbell_sum_net": 100},
                 {"mark_row": r, "mark_col": c, "tag": tag, "minutes": 60, "dumbbell_sum_net": 100 + gain}]
    ch, var, floor = hp.expression_calls(pd.DataFrame(rows), value="dumbbell_sum_net", floor="sd")
    assert floor == 30.0   # median 0 + 3 x SD 10
    assert np.isclose(hp.expression_calls(pd.DataFrame(rows), value="dumbbell_sum_net")[2], 3 * 1.4826 * 10)   # robust: MAD 10
    v = var.set_index("tag")
    assert bool(v.loc["A5M", "expressed"]) and v.loc["A5M", "chambers_expressed"] == 1
    assert not bool(v.loc["G6F", "expressed"]) and "BLANK" not in v.index


def test_report_pages(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    rows = []
    for c in range(12):
        tag = "BLANK" if c < 4 else "WT" if c < 6 else "A5M" if c < 9 else "G6F"
        gain = 0 if tag in ("BLANK", "G6F") else 500
        for m in (20, 40, 60):
            rows.append({"mark_row": 0, "mark_col": c, "tag": tag, "minutes": m, "dumbbell_sum_net": 100 + gain * (m - 20) / 40 + c})
    e = pd.DataFrame(rows)
    ch, var, floor = hp.expression_calls(e, value="dumbbell_sum_net")
    out = hp.expression_report(tmp_path / "r.pdf", e, ch, var, floor, value="dumbbell_sum_net", experiment="test", n_rows=1, n_cols=12)
    assert out.exists() and out.stat().st_size > 10_000


def test_blank_leakage():
    tags = ["BLANK" if (r, c) in ((1, 1), (3, 3)) else "A5M" for r in range(5) for c in range(5)]
    ch = pd.DataFrame({"mark_row": np.repeat(range(5), 5), "mark_col": np.tile(range(5), 5), "tag": tags,
                       "gain": [5.0 if t == "BLANK" and i == 6 else 0.1 if t == "BLANK" else 100.0 for i, t in enumerate(tags)]})
    t, s = hp.blank_leakage(ch, floor=1.0, n_rows=5, n_cols=5)
    assert s["blanks"] == 2 and s["contaminated"] == 1 and np.isclose(s["clean_median_ratio"], 0.001)


def test_dna_chamber_diffusion_flags_rising_dna_chambers():
    import htmek_processing as hp

    rng = np.random.default_rng(0)
    rows = []
    for r in range(4):
        for c in range(10):
            tag = "BLANK" if c == 0 else "WT"
            leak = 2.0 if (r, c) == (1, 5) else 0.0
            for f, sec in enumerate(range(0, 600, 60)):
                rows.append({"run": "s1", "mark_row": r, "mark_col": c, "tag": tag, "frame": f, "seconds": sec, "complete_pass": True,
                             "chamber_ROI_median": 1000 + (0 if tag == "BLANK" else 5) * sec + rng.normal(0, 5),
                             "dna_chamber_median": 500 + leak * sec + rng.normal(0, 5)})
    t, s = hp.dna_chamber_diffusion(pd.DataFrame(rows), time="seconds")
    assert t.loc[t["rising"], ["mark_row", "mark_col"]].values.tolist() == [[1, 5]]
    assert s.loc[0, "rising"] == 1 and abs(t.loc[t["rising"], "ratio"].iloc[0] - 0.4) < 0.05
