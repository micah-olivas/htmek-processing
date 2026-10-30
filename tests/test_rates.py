"""Rate post-processing: forcing the 4PL on runs depleted early, and subtracting the BLANK rate."""

import numpy as np
import pandas as pd

import htmek_processing as hp


def test_subtract_blank_rate_per_run():
    rates = pd.DataFrame({"bzp_uM": [100, 100, 100, 200, 200, 200],
                          "tag": ["BLANK", "BLANK", "A1B", "BLANK", "BLANK", "A1B"],
                          "v0": [1.0, 3.0, 10.0, 5.0, 7.0, 20.0]})
    out = hp.subtract_blank_rate(rates)
    np.testing.assert_allclose(out["blank_median_v0"], [2, 2, 2, 6, 6, 6])
    np.testing.assert_allclose(out["v0"], [-1, 1, 8, -1, 1, 14])
    np.testing.assert_allclose(out["v0_unsubtracted"], rates["v0"])


def test_force_4pl_uses_slope_inside_the_window():
    # b < 1: the 4PL is steepest at t -> 0, so inside [6, 600] s its steepest slope is at 6 s
    a, d, c, b = 100.0, 1000.0, 150.0, 0.8
    traces = pd.DataFrame({"mark_row": 0, "mark_col": [0] * 3 + [1] * 3, "seconds": [6.0, 300.0, 600.0] * 2})
    rates = pd.DataFrame({"mark_row": [0, 0], "mark_col": [0, 1], "a": a, "d": d, "c": c, "b": b,
                          "v0": [1.0, 1.0], "model": "linear", "depleted_early": [True, False]})
    out = hp.force_4pl_on_depleted(rates, traces)
    assert out.loc[0, "model"] == "4pl_forced" and out.loc[1, "model"] == "linear"
    np.testing.assert_allclose(out.loc[0, "v0"], hp.dfour_pl(6.0, a, d, c, b), rtol=1e-6)
    assert out.loc[1, "v0"] == 1.0
