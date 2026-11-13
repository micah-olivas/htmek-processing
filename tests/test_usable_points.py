"""usable_points: flags keep or drop points, and minus_frame0 restarts at each chamber's first kept point."""

import pandas as pd

import htmek_processing as hp


def test_flags_and_baseline():
    t = pd.DataFrame({"mark_row": 0, "mark_col": [0, 0, 0, 1, 1, 1], "frame": [0, 1, 2, 0, 1, 2],
                      "chamber_ROI_median": [5.0, 100.0, 110.0, 50.0, 60.0, 70.0],
                      "light_off": [True, False, False, False, False, False],
                      "complete_pass": [True, True, True, True, True, False], "tile_imaged": True})
    u = hp.usable_points(t)
    assert len(u) == 4
    a = u[u["mark_col"] == 0].sort_values("frame")
    assert list(a["chamber_ROI_median_minus_frame0"]) == [0.0, 10.0]   # from pass 1, its first lit point
    assert list(u[u["mark_col"] == 1]["frame"]) == [0, 1]              # the incomplete pass dropped


def test_without_flags_everything_is_kept():
    t = pd.DataFrame({"mark_row": 0, "mark_col": 0, "frame": [0, 1], "chamber_ROI_median": [1.0, 3.0]})
    assert list(hp.usable_points(t)["chamber_ROI_median_minus_frame0"]) == [0.0, 2.0]
