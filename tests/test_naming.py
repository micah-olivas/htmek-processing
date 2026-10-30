"""The mask-prefix naming convention: `<prefix>_<agg>` columns, unprefixed by default."""

import numpy as np

import htmek_processing as hp


def _image_and_grid():
    img = np.full((200, 200), 100.0, np.float32)
    cx, cy = np.meshgrid([50.0, 150.0], [50.0, 150.0])
    for x, y in zip(cx.ravel(), cy.ravel()):
        img[int(y) - 5:int(y) + 6, int(x) - 5:int(x) + 6] = 500.0
    return img, cx, cy


def test_centers_to_df_default_names_unchanged():
    img, cx, cy = _image_and_grid()
    df = hp.centers_to_df(cx, cy, img=img, radius=8, aggs=("median", "sum"), bg=True, bg_inner=12, bg_outer=20)
    assert {"median", "sum", "n_px", "bg", "median_net", "sum_net"} <= set(df.columns)


def test_centers_to_df_prefix():
    img, cx, cy = _image_and_grid()
    plain = hp.centers_to_df(cx, cy, img=img, radius=8, aggs=("median", "sum"), bg=True, bg_inner=12, bg_outer=20)
    df = hp.centers_to_df(cx, cy, img=img, radius=8, aggs=("median", "sum"), bg=True, bg_inner=12, bg_outer=20,
                          prefix="button")
    expected = {"button_median", "button_sum", "button_n_px", "button_annulus_median", "button_median_net", "button_sum_net"}
    assert expected <= set(df.columns)
    assert not {"median", "sum", "bg", "n_px"} & set(df.columns)
    np.testing.assert_allclose(df["button_sum_net"], plain["sum_net"])
    assert df.attrs["centers"] == plain.attrs["centers"]


def test_kinetic_traces_prefix():
    img, cx, cy = _image_and_grid()
    stack = np.stack([img, img * 2])
    df = hp.kinetic_traces(stack, cx, cy, radius=8, aggs=("median",), seconds=np.array([0.0, 30.0]),
                           drop_empty=False, prefix="chamber_ROI")
    assert {"frame", "seconds", "chamber_ROI_median", "chamber_ROI_n_px"} <= set(df.columns)
    first, second = (df.loc[df["frame"] == f, "chamber_ROI_median"].to_numpy() for f in (0, 1))
    np.testing.assert_allclose(second, 2 * first)


def test_plot_control_traces_one_panel_per_substrate():
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd
    rows = [dict(bzp_uM=s, mark_row=r, mark_col=0, tag=tag, frame=f, seconds=30.0 * f, value=float(f * (r + 1)))
            for s in (1250, 2500, 5000) for r, tag in enumerate(["WT", "N41A", "BLANK"]) for f in range(5)]
    fig = hp.plot_control_traces(pd.DataFrame(rows), "value")
    assert sum(ax.get_visible() for ax in fig.axes) == 3


def test_roi_read_when_only_the_annulus_runs_off_the_image():
    img = np.full((200, 200), 100.0, np.float32)
    cx, cy = np.array([[15.0, 100.0]]), np.array([[100.0, 100.0]])
    df = hp.centers_to_df(cx, cy, img=img, radius=10, aggs=("median",), bg=True, bg_inner=12, bg_outer=30)
    edge, middle = df.iloc[0], df.iloc[1]
    assert edge["median"] == 100.0 and np.isnan(edge["bg"]) and np.isnan(edge["median_net"])
    assert middle["median"] == 100.0 and middle["bg"] == 100.0
