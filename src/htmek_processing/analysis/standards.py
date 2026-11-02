"""NADH standards: a standard line per chamber through the on-chip standards, and signal ->
[NADH] with it."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rates import fit_lines


def calibrate_chambers(
    standards_df,
    value="median",
    *,
    x="conc_uM",
    per=("mark_row", "mark_col"),
    keep=("tag", "x", "y"),
    conc_max=None,
):
    """Per-chamber calibration: linear fit of `value` vs concentration `x` for each
    chamber. Returns one row per chamber: the `per` keys, `keep` columns, slope, intercept,
    r2, n.

    Each chamber gets its own intensity->[NADH] line (illumination varies across the chip);
    invert it later as conc = (signal - intercept) / slope. `conc_max` restricts the fit to
    the linear range (e.g. conc_max=500 to drop high-end self-absorption curvature)."""
    from .steps import record_step

    window = (-np.inf, conc_max) if conc_max is not None else None
    cal = fit_lines(
        standards_df, value, x, per=per, keep=keep, window=window, slope_name="slope"
    )
    conc = sorted(standards_df[x].unique())
    used = [c for c in conc if conc_max is None or c <= conc_max]
    ok = cal["slope"].notna()
    record_step("calibrate_chambers", "Calibrate per chamber",
                [f"{len(used)} standards, {used[0]:g}–{used[-1]:g} µM; a line per chamber",
                 f"of `{value}` vs [NADH] (median r² {cal.loc[ok, 'r2'].median():.3f})",
                 "signal → [NADH] through the chamber's own line"],
                n=len(cal), unit="chambers", stage="images",
                branches=[dict(text="calibrated", n=ok.sum(), level="pass")]
                + ([dict(text="no line", n=(~ok).sum(), level="fail")] if (~ok).any() else []))
    return cal


def predict_conc(
    df,
    cal,
    value="median",
    *,
    per=("mark_row", "mark_col"),
    slope="slope",
    intercept="intercept",
    out="conc_uM",
):
    """Invert a per-chamber calibration (from calibrate_chambers) to convert a signal
    column to concentration, matched per chamber on `per`:
    out = (value - intercept) / slope.

    Works on a single-frame df or a kinetic_traces df (adds `out` per row, so a trace
    becomes a [NADH]-over-time curve). Then `fit_rates(result, value=out)` gives rates in
    conc/second directly. Chambers missing from `cal` (e.g. dropped for low r2) get NaN.
    Returns a copy."""
    keys = list(per)
    c = cal[keys + [slope, intercept]].rename(
        columns={slope: "_cal_m", intercept: "_cal_b"}
    )

    m = df.merge(c, on=keys, how="left")
    m[out] = (m[value] - m["_cal_b"]) / m["_cal_m"]
    return m.drop(columns=["_cal_m", "_cal_b"])
