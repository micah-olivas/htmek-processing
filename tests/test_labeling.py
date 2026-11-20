import numpy as np
import pandas as pd

import htmek_processing as hp


def test_labeling_table_and_figure():
    rng = np.random.default_rng(0)
    rows_c, rows_g = [], []
    for r in range(56):
        for c in range(32):
            tag = "BLANK" if (r % 8 == 0 and c == 0) else "WT" if (r, c) == (3, 3) else "A4*"
            gfp = 0 if tag == "BLANK" else rng.uniform(1e5, 1e7) * (3 if c < 16 else 0.3)
            rows_g.append(dict(mark_row=r, mark_col=c, tag=tag, scan="SDS wash 4", button_sum_minus_bg=gfp + rng.normal(0, 1e4)))
            rows_c.append(dict(mark_row=r, mark_col=c, tag=tag, scan="post-labeling", button_sum_minus_bg=0.02 * gfp + rng.normal(0, 500)))
    t = hp.labeling_table(pd.DataFrame(rows_c), pd.DataFrame(rows_g), gfp_scan="SDS wash 4")
    lib = t[(t.tag == "A4*") & t.gfp_above_floor]
    assert abs(lib.cy5_over_gfp.median() - 0.02) < 0.002 and set(t.half) == {"left", "right"}
    fig = hp.labeling_figure(t, title="test")
    assert len(fig.axes) == 2
