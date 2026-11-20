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


def fit_standard_lines(*args, **kwargs):
    """A line per chamber through its standards (calibrate_chambers, under the name the analysis uses)."""
    return calibrate_chambers(*args, **kwargs)


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


def bound_complex(conc, p_total, kd):
    """The bound complex of a tight-binding sensor (µM): B = ((P + L + Kd) - sqrt((P + L + Kd)^2 - 4 P L)) / 2, for
    ligand `conc` L, sensor `p_total` P and `kd` (µM)."""
    L = np.asarray(conc, float)
    s = p_total + L + kd
    return (s - np.sqrt(np.maximum(s * s - 4 * p_total * L, 0.0))) / 2


def free_from_bound(bound, p_total, kd):
    """Invert bound_complex: the total ligand that leaves `bound` µM of complex, L = B + Kd B / (P - B); NaN at or
    beyond the sensor's saturation (B >= P)."""
    B = np.asarray(bound, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.where(B < p_total, B + kd * B / (p_total - B), np.nan)
    return L


def _chamber_lines(design, y):
    """Per chamber, the least-squares (intercept, amplitude) of y = intercept + amplitude * design, both (n_chambers,
    n_points) with NaN where missing; returns (intercept, amplitude, rss, n)."""
    m = np.isfinite(design) & np.isfinite(y)
    n = m.sum(1)
    d, v = np.where(m, design, 0.0), np.where(m, y, 0.0)
    dm, vm = d.sum(1) / np.maximum(n, 1), v.sum(1) / np.maximum(n, 1)
    dd = np.where(m, design - dm[:, None], 0.0)
    vv = np.where(m, y - vm[:, None], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        amp = (dd * vv).sum(1) / (dd * dd).sum(1)
    icpt = vm - amp * dm
    rss = np.where(m, (y - icpt[:, None] - amp[:, None] * design) ** 2, 0.0).sum(1)
    return icpt, amp, rss, n


def fit_pbp_standards(standards_df, value="median", *, x="conc_uM", per=("mark_row", "mark_col"), keep=("tag", "x", "y"),
                  p_guess=100.0, kd_guess=0.5):
    """Per-chamber standards fit for a tight-binding sensor (PBP for phosphate): signal = intercept + amplitude * B(L),
    B the bound complex (bound_complex) for ligand L, with the sensor's active concentration P and Kd shared by every
    chamber and each chamber's own intercept and amplitude (its illumination). P and Kd are fit chip-wide (variable
    projection: for each P, Kd the chambers' intercepts and amplitudes are linear; their summed residuals are
    minimized), then each chamber's line in B.

    For a sensor in excess, B ~ L below saturation, so `amplitude` is the low-[L] slope (RFU per µM) and the
    standards' fit is linear there; above P the signal plateaus. Returns one row per chamber: the `per` keys, `keep`
    columns, intercept, slope (= amplitude, RFU per µM bound), r2, n; attrs p_total, kd (µM). Invert with
    predict_conc_pbp."""
    from scipy.optimize import minimize

    from .steps import record_step

    d = standards_df.dropna(subset=[value, x])
    concs = np.sort(d[x].unique())
    wide = d.pivot_table(index=list(per), columns=x, values=value).reindex(columns=concs)
    Y = wide.to_numpy(float)
    L = np.broadcast_to(concs, Y.shape)

    def cost(theta):
        P, Kd = np.exp(theta)
        _, _, rss, _ = _chamber_lines(bound_complex(L, P, Kd), Y)
        return np.nansum(rss)

    best = min((minimize(cost, np.log([p0, kd0]), method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6, maxiter=2000))
                for p0 in (p_guess * 0.5, p_guess, p_guess * 1.5) for kd0 in (kd_guess * 0.1, kd_guess, kd_guess * 10)),
               key=lambda r: r.fun)
    P, Kd = (float(v) for v in np.exp(best.x))
    icpt, amp, rss, n = _chamber_lines(bound_complex(L, P, Kd), Y)
    tss = np.nansum((Y - np.nanmean(Y, 1, keepdims=True)) ** 2, 1)
    cal = wide.index.to_frame(index=False)
    meta = d.drop_duplicates(list(per)).set_index(list(per))
    for k in keep:
        if k in meta and k not in per:
            cal[k] = meta[k].reindex(wide.index).to_numpy()
    cal["intercept"], cal["slope"], cal["n"] = icpt, amp, n
    with np.errstate(divide="ignore", invalid="ignore"):
        cal["r2"] = 1 - rss / tss
    cal.attrs.update(p_total=P, kd=Kd, model="tight binding")
    record_step("fit_pbp_standards", "Standards per chamber (PBP)",
                ["tight-binding isotherm: signal = intercept + amplitude × bound,",
                 f"sensor {P:.1f} µM active, Kd {Kd:.2g} µM (chip-wide fit);",
                 f"each chamber its own intercept and amplitude, on {len(concs)} standards"],
                n=len(cal), branches=[dict(text="fit", n=int(np.isfinite(amp).sum()), level="pass")])
    return cal


def predict_conc_pbp(df, cal, value="median", *, per=("mark_row", "mark_col"), out="conc_uM", max_frac=0.9):
    """Invert fit_pbp_standards per chamber: bound = (value - intercept) / slope, then L = free_from_bound(bound). Points
    whose bound complex is above `max_frac` of the sensor (near its plateau, where L is no longer determined) get NaN;
    the column f"{out}_saturated" marks them. Returns a copy."""
    keys = list(per)
    P, Kd = cal.attrs["p_total"], cal.attrs["kd"]
    m = df.merge(cal[keys + ["slope", "intercept"]].rename(columns={"slope": "_m", "intercept": "_b"}), on=keys, how="left")
    bound = (m[value] - m["_b"]) / m["_m"]
    sat = bound > max_frac * P
    m[out] = np.where(sat, np.nan, free_from_bound(np.minimum(bound, max_frac * P), P, Kd))
    m[f"{out}_saturated"] = sat.to_numpy()
    return m.drop(columns=["_m", "_b"])
