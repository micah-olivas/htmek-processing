import matplotlib

matplotlib.use("Agg")
import pandas as pd

import htmek_processing as hp
from htmek_processing.report import flowchart_page, report_steps


def test_leave_out_points_records_its_step(tmp_path):
    hp.clear_steps()
    traces = pd.DataFrame({"bzp_uM": [78.125] * 4 + [5000.0] * 4, "frame": [0, 1, 2, 3] * 2,
                           "nadh_uM": [0, -30, 5, 6, 0, 1, 2, -25]})
    out, keep = hp.leave_out_points(traces, "nadh_uM", floor=-20, last_frame={78.125: 2})
    assert keep.tolist() == [True, False, True, False, True, True, True, False]
    assert out["nadh_uM_fit"].isna().sum() == 3
    (step,) = hp.pipeline_steps()
    assert step["n"] == 8 and [b["n"] for b in step["branches"]] == [2, 1]
    # rerunning replaces the step; the record survives a save and load
    hp.leave_out_points(traces, "nadh_uM", floor=-20)
    hp.save_steps(tmp_path / "steps.json")
    assert len(hp.load_steps(tmp_path / "steps.json")) == 1


def test_flowchart_page_draws_recorded_and_report_steps():
    mm_raw = pd.DataFrame({"mark_row": [0, 0, 1], "mark_col": [0, 1, 0], "r2": [0.95, 0.5, 0.99]})
    steps = [dict(step="s", title="A step", rule=["a rule"], unit="runs", n=10,
                  branches=[dict(text="left out", n=2, level="fail")])]
    steps += report_steps(mm_raw=mm_raw)
    assert [b["n"] for b in steps[-1]["branches"]] == [2, 1]
    fig = flowchart_page(steps, experiment="test")
    assert len(fig.patches) > 4


def test_tally_counts_each_item_once():
    hp.clear_steps()
    for name, own in (("a", True), ("b", False), ("a", True)):
        hp.tally_step("stitch", "Stitch", ["rule"], name, label="own" if own else "reused")
    (step,) = hp.pipeline_steps()
    assert step["n"] == 2 and step["stage"] == "images"
    assert {b["text"]: b["n"] for b in step["branches"]} == {"own": 1, "reused": 1}


def test_flowchart_pages_split_by_stage():
    from htmek_processing.report import flowchart_pages

    steps = [dict(step="a", title="A", rule=["r"], unit="images", n=1, branches=[], stage="images"),
             dict(step="b", title="B", rule=["r"], unit="runs", n=1, branches=[])]
    assert len(flowchart_pages(steps)) == 2
