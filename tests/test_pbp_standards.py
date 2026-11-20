import numpy as np
import pandas as pd

from htmek_processing.analysis.standards import bound_complex, fit_pbp_standards, free_from_bound, predict_conc_pbp


def test_inverse():
    L = np.array([0, 5, 20, 50])
    assert np.allclose(free_from_bound(bound_complex(L, 60.0, 0.3), 60.0, 0.3), L)


def test_calibrate_and_invert_recovers_sensor_and_conc():
    rng = np.random.default_rng(0)
    P, Kd, concs = 60.0, 0.3, [0, 6, 15, 30, 75, 150]
    rows = []
    for r in range(6):
        for c in range(6):
            b, a = rng.uniform(800, 1500), rng.uniform(60, 120)
            for L in concs:
                rows.append(dict(mark_row=r, mark_col=c, tag="X", conc_uM=L,
                                 median=b + a * bound_complex(L, P, Kd) + rng.normal(0, 5)))
    std = pd.DataFrame(rows)
    cal = fit_pbp_standards(std, "median", x="conc_uM", keep=("tag",))
    assert abs(cal.attrs["p_total"] - P) < 1.5
    assert cal["r2"].min() > 0.99
    # a chamber's signal at 20 µM Pi reads back as ~20 µM; one on the plateau gets NaN
    one = cal.iloc[0]
    test = pd.DataFrame(dict(mark_row=[one.mark_row] * 2, mark_col=[one.mark_col] * 2,
                             median=[one.intercept + one.slope * bound_complex(20, P, Kd), one.intercept + one.slope * P]))
    out = predict_conc_pbp(test, cal, "median")
    assert abs(out["conc_uM"].iloc[0] - 20) < 1.0
    assert np.isnan(out["conc_uM"].iloc[1]) and out["conc_uM_saturated"].iloc[1]
