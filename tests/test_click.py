import numpy as np
import pandas as pd

from htmek_processing.analysis.labeling import click_changes, click_figures


def _click():
    rows = []
    for r, c, tag in [(0, 0, "BLANK"), (0, 1, "A1F"), (1, 0, "WT")]:
        for t in range(3):
            rows.append(dict(mark_row=r, mark_col=c, tag=tag, series="s1", **{"pass": t}, buttons_open_s=60.0 * t,
                             button_median=100 + 10 * t + r, button_annulus_median=500 - 50 * t, button_sum_net=0.0,
                             dna_median=200 + 5 * t, reaction_median=400 - 20 * t))
    return pd.DataFrame(rows)


def test_click_changes_from_pass_0_and_groups():
    enz = pd.DataFrame(dict(mark_row=[0, 0, 1], mark_col=[0, 1, 0], gfp=[0, 5e5, 1e3], expressing=[False, True, False]))
    d = click_changes(_click(), enz)
    assert np.allclose(d.loc[d["pass"] == 0, "d_button_median"], 0)
    assert np.allclose(d.loc[d["pass"] == 2, "d_button_median"], 20)
    assert np.allclose(d.loc[d["pass"] == 2, "d_button_annulus_median"], -100)
    g = d.drop_duplicates(["mark_row", "mark_col"]).set_index(["mark_row", "mark_col"])["group"]
    assert g[(0, 0)] == "BLANK" and g[(0, 1)] == "library, GFP above floor" and g[(1, 0)] == "WT"


def test_click_figures_build():
    import matplotlib
    matplotlib.use("Agg")
    enz = pd.DataFrame(dict(mark_row=[0, 0, 1], mark_col=[0, 1, 0], gfp=[0, 5e5, 1e3], expressing=[False, True, False]))
    figs = click_figures(_click(), enz, n_rows=2, n_cols=2)
    assert set(figs) == {"3e_click_timecourse", "3e_click_last_pass_vs_gfp"}
