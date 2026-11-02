"""Volcano statistics: the bootstrap test and the reference fallback."""

import numpy as np
import pandas as pd

from htmek_processing.report import volcano as V


def test_bootstrap_p_separates_shifted_samples():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 0.3, 200)
    same = rng.normal(0, 0.3, 5)
    shifted = rng.normal(2, 0.3, 5)
    assert V.bootstrap_p(same, ref, np.random.default_rng(1)) > 0.05
    assert V.bootstrap_p(shifted, ref, np.random.default_rng(1)) < 0.001


def _tables(n_wt):
    rows = [dict(mark_row=i, mark_col=0, tag="WT" if i < n_wt else ("A1F" if i >= 24 else "B2M"))
            for i in range(30)]
    gfp = pd.DataFrame(rows).assign(button_sum_minus_bg=1e6)
    km = np.where(gfp["tag"] == "A1F", 400.0, 1000.0)
    mm = gfp[["mark_row", "mark_col", "tag"]].assign(vmax=10.0, km=km, r2=0.99)
    rates = pd.concat([gfp[["mark_row", "mark_col", "tag"]].assign(bzp_uM=s, v0=5.0) for s in (100, 5000)])
    return V.volcano_inputs(gfp, mm, mm, rates, floor=0)


def test_reference_is_wt_or_the_library():
    q = V.quantities()[0]  # Km
    t, ref, _ = V.variant_table(q, *_tables(n_wt=5))
    assert t["reference"].iloc[0] == "WT" and len(ref) == 5
    t, ref, _ = V.variant_table(q, *_tables(n_wt=2))  # too few WT: the library
    assert t["reference"].iloc[0] == "library median" and len(ref) == 28
    a1f = t.set_index("tag").loc["A1F"]
    assert a1f["log2_fc"] < 0 and a1f["p_value"] < 0.05  # 6 chambers vs a pool 21 % alike: p ~ 0.02


def test_bootstrap_p_fixed_separates_far_values():
    import numpy as np
    from htmek_processing.report.volcano import bootstrap_p_fixed

    rng = np.random.default_rng(0)
    far = np.log2(np.array([3000.0, 3200, 2800, 3100]))
    near = np.log2(np.array([690.0, 720, 700, 710]))
    assert bootstrap_p_fixed(far, np.log2(704), rng) < 0.05
    assert bootstrap_p_fixed(near, np.log2(704), rng) > 0.1
