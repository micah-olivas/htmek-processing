"""Initial rates from progress curves: straight lines and 4PL fits, the rules that choose
between them, early-depletion and fit-quality flags, BLANK subtraction, and rates per enzyme."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..processing.quantify import bleach_correct, fit_bleaching


def fit_lines(
    df,
    value,
    x,
    *,
    per=("mark_row", "mark_col"),
    keep=("tag", "x", "y"),
    window=None,
    slope_name="slope",
):
    """Per-group linear fit of `value` vs `x`. Groups by `per` and returns one row per
    group: the `per` keys, the first value of each `keep` column, `slope_name` (slope),
    intercept, r2, and n (points fit). `window=(lo, hi)` restricts `x` before fitting.
    Groups with < 2 points or no spread in `x` get NaN fits."""
    rows = []
    for key, g in df.groupby(list(per)):
        g = g.dropna(subset=[value, x])
        if window is not None:
            g = g[(g[x] >= window[0]) & (g[x] <= window[1])]

        rec = dict(zip(per, key if isinstance(key, tuple) else (key,)))
        for c in keep:
            if c in g.columns and len(g):
                rec[c] = g[c].iloc[0]

        xv, yv = g[x].to_numpy(float), g[value].to_numpy(float)
        if len(xv) >= 2 and np.ptp(xv) > 0:
            slope, intercept = np.polyfit(xv, yv, 1)
            resid = yv - (slope * xv + intercept)
            ss_tot = np.sum((yv - yv.mean()) ** 2)
            rec.update(
                {
                    slope_name: slope,
                    "intercept": intercept,
                    "r2": (1 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else np.nan,
                    "n_points": len(xv),
                }
            )
        else:
            rec.update(
                {slope_name: np.nan, "intercept": np.nan, "r2": np.nan, "n_points": len(xv)}
            )
        rows.append(rec)

    return pd.DataFrame(rows)


def fit_rates(
    df,
    value="median",
    *,
    t="seconds",
    start=None,
    end=None,
    window=None,
    per=("mark_row", "mark_col"),
    keep=("tag", "x", "y"),
):
    """Per-button initial-rate fit: slope of `value` vs `t` from a kinetic_traces
    DataFrame.

    Restrict the fit to a time window (in `t`'s units, e.g. seconds) with `start`/`end`
    (either may be omitted to leave that side open) or an explicit `window=(t0, t1)`;
    default fits all frames. Use it to isolate the linear initial-velocity phase of the
    curve. Returns one row per button: the `per` keys, `keep` columns, plus rate (slope),
    intercept, r2, n.
    A negative `rate` means the signal falls over time (NADH consumed)."""
    if window is None and (start is not None or end is not None):
        window = (
            start if start is not None else -np.inf,
            end if end is not None else np.inf,
        )
    return fit_lines(df, value, t, per=per, keep=keep, window=window, slope_name="rate")


# Sigmoidal (4PL) progress curves, ported from MBO-039 (scratch/3a_rate_fits.py):
#     y(t) = d + (a - d) / (1 + (t / c) ** b)
# a = starting level, d = final plateau, c = inflection time, b = steepness. A 4PL follows
# a whole progress curve (lag, linear phase, approach to the plateau), so its peak slope
# is a rate estimate that doesn't depend on choosing a linear window by hand.


def four_pl(t, a, d, c, b):
    """4PL sigmoid: a = start, d = end plateau, c = inflection time, b = steepness."""
    return d + (a - d) / (1 + (t / c) ** b)


def dfour_pl(t, a, d, c, b):
    """First derivative of four_pl: the instantaneous slope at time t (signed)."""
    u = (t / c) ** b
    return -(a - d) * b * (t / c) ** (b - 1) / (c * (1 + u) ** 2)


def peak_4pl(a, d, c, b):
    """(t_peak, vmax): time and signed value of the steepest slope.

    The slope has an interior maximum only when b > 1; for b <= 1 it is steepest at t -> 0
    and both are NaN. vmax keeps its sign (negative = signal falling). MBO-039 reported
    abs(vmax), which can't show a downward drift and reads positive even on pure noise."""
    if not b > 1:
        return np.nan, np.nan
    t_peak = c * ((b - 1) / (b + 1)) ** (1 / b)
    return t_peak, dfour_pl(t_peak, a, d, c, b)


def slope_4pl_in_window(a, d, c, b, t_min, t_max, n=200):
    """Steepest slope of the 4PL inside the observed window [t_min, t_max] (signed: the
    largest |slope|), for each row of parameter arrays. Unlike peak_4pl this never
    extrapolates: when the 4PL's own peak lies outside the data, it returns the slope at
    the window edge nearest to it. Use it for an 'always 4PL' rate definition."""
    a, d, c, b, t_min, t_max = (
        np.asarray(v, float)[:, None] for v in (a, d, c, b, t_min, t_max)
    )
    t = t_min + (t_max - t_min) * np.linspace(0, 1, n)[None, :]
    with np.errstate(all="ignore"):
        s = dfour_pl(np.maximum(t, 1e-9), a, d, c, b)
    k = np.nanargmax(np.where(np.isfinite(s), np.abs(s), -1), axis=1)
    return s[np.arange(len(s)), k]


def guess_4pl(t, y):
    """Default 4PL starting point [a, d, c, b] (MBO-039): a = mean of the first 3 points,
    d = mean of the last 3, c = median time, b = 2."""
    return [np.mean(y[:3]), np.mean(y[-3:]), max(np.median(t), 1), 2]


# Default 4PL bounds, as (lower, upper) over [a, d, c, b]. a and d are plateau LEVELS: after
# background subtraction they can sit at or below zero, so only c and b are bounded
# positive (bounding a, d >= 0 silently dropped 141 of 547 chambers in MBO-039).
BOUNDS_4PL = ([-np.inf, -np.inf, 1e-9, 1e-9], [np.inf, np.inf, np.inf, np.inf])


def fit_4pl(
    t,
    y,
    *,
    p0=guess_4pl,
    bounds=BOUNDS_4PL,
    rising=False,
    min_points=5,
    max_fev=20_000,
):
    """Fit four_pl to one trace -> array [a, d, c, b], or None if it can't be fit
    (too few points, or no convergence).

    `p0` is the starting point [a, d, c, b], or a function (t, y) -> [a, d, c, b];
    `bounds` is (lower, upper) over [a, d, c, b]. The starting point is clipped into the
    bounds.

    rising=True constrains the curve to go UP: the end plateau d can't sit below the start
    a, so the steepest slope (vmax) can never be negative. It is enforced by fitting the
    rise (d - a) itself, bounded below by 0, instead of d; `bounds` on d are then ignored.
    Use it when the product signal can only grow (e.g. NADH made by the enzyme)."""
    from scipy.optimize import curve_fit

    t, y = np.asarray(t, float), np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) < min_points:
        return None

    lower = np.asarray(bounds[0], float)
    upper = np.asarray(bounds[1], float)
    start = np.clip(p0(t, y) if callable(p0) else p0, lower, upper)

    if rising:
        # Fit [a, rise, c, b] with rise = d - a >= 0 (a constraint between two parameters,
        # which curve_fit's per-parameter bounds can't express directly).
        def model(tt, a_, rise_, c_, b_):
            return four_pl(tt, a_, a_ + rise_, c_, b_)

        start = start.copy()
        start[1] = max(start[1] - start[0], 1e-9)  # starting rise: positive
        lower = lower.copy()
        upper = upper.copy()
        lower[1], upper[1] = 0.0, np.inf  # the rise can't be negative
    else:
        model = four_pl

    # curve_fit probes large b, where (t/c)**b overflows; the optimizer handles the inf
    # itself, so those warnings are noise rather than a sign of a bad fit
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            popt, _ = curve_fit(
                model, t, y, p0=start, bounds=(lower, upper), maxfev=max_fev
            )
    except (RuntimeError, ValueError):
        return None

    if rising:
        popt = np.array(
            [popt[0], popt[0] + popt[1], popt[2], popt[3]]
        )  # back to [a, d, c, b]
    return popt


def _bic(rss, n, k):
    """Bayesian information criterion of a least-squares fit with k parameters."""
    return n * np.log(max(rss, 1e-300) / n) + k * np.log(n)


def _fit_4pl_summary(job):
    """Fit one chamber with both a 4PL and a straight line, and pick the rate.

    The 4PL's steepest slope (vmax) is used only if EVERY rule below holds; otherwise the
    rate is the straight line's slope (model "linear"). The 4PL is the default, so `reason`
    says why it was not used: the first rule that fails (empty when the 4PL is used).

      1. better than the line: the 4PL beats the line by BIC, i.e. its curvature is worth
         its two extra parameters;
      2. has a steepest point: b > 1 (for b <= 1 the slope is steepest at t -> 0);
      3. steepest point inside the data: beyond the measured time range the slope is
         extrapolated, not measured;
      4. rise not too fast: the 10-90 % rise spans at least one sampling interval, so the
         steepest slope is measured by data on both sides, not drawn as a step between two
         neighboring points;
      5. rise above the noise (when min_rise_snr is set): the rise of the fitted 4PL across
         the measured time range is at least min_rise_snr x the scatter of the points about
         the fit. A flat trace with a noise bump fits as a small, sharp step whose slope
         says nothing about the enzyme.

    Two more rules compare the 4PL with the run's substrate concentration [S] (when it is
    given). They assume the signal is in the same units as [S] and that each substrate
    turned over makes one product (e.g. NADH in µM vs [BzP] in µM):

      6. no more product than substrate (when max_rise_frac is set): the rise seen across
         the measured time range is at most max_rise_frac x [S];
      7. a plateau means the substrate ran out (when plateau_min_frac is set): if the 4PL
         reaches 90 % of its rise inside the measured time range, that full rise (d - a)
         is at least plateau_min_frac x [S]. A curve that levels off far below [S] is a
         step, not substrate depletion.

    With rising=True the 4PL itself is constrained to go up (fit_4pl), so a 4PL vmax is
    never negative; a falling trace is then left to the line. Module-level so worker
    processes can import it."""
    (tv, yv, min_points, t_rate, p0, bounds,
     rising, min_rise_snr, s_conc, max_rise_frac, plateau_min_frac) = job

    ok = np.isfinite(tv) & np.isfinite(yv)
    tv, yv = tv[ok], yv[ok]
    n = len(tv)
    rec = {"n_points": n}
    if n < min_points:
        return rec

    ss_tot = np.sum((yv - yv.mean()) ** 2)

    def r2(rss):
        return 1 - rss / ss_tot if ss_tot > 0 else np.nan

    # ---- the straight line: the fallback rate, and the yardstick for the 4PL ----------
    slope, intercept = np.polyfit(tv, yv, 1)
    rss_line = np.sum((yv - (slope * tv + intercept)) ** 2)
    rec.update(slope_line=slope, model="linear", vmax=slope, r2=r2(rss_line))

    # ---- the 4PL ------------------------------------------------------------------------
    p = fit_4pl(tv, yv, p0=p0, bounds=bounds, rising=rising, min_points=min_points)
    if p is None:
        rec["reason"] = "4PL did not converge"
        return rec

    a, d, c, b = p
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        t_peak, vmax_4pl = peak_4pl(a, d, c, b)  # time and value of the steepest slope
        rss_4pl = np.sum((yv - four_pl(tv, *p)) ** 2)
        width = c * (9 ** (1 / b) - 9 ** (-1 / b))  # time from 10 % to 90 % of the rise
        t90 = c * 9 ** (1 / b)  # when the curve reaches 90 % of its rise
        if t_rate is not None:
            rec["rate_at_t"] = dfour_pl(t_rate, *p)

    # what the rules look at
    dt = np.median(np.diff(np.sort(tv)))  # sampling interval
    rise = four_pl(tv.max(), *p) - four_pl(tv.min(), *p)  # rise seen within the data
    scatter = np.sqrt(rss_4pl / max(n - 4, 1))  # residual SD about the 4PL (4 parameters)
    rise_snr = rise / scatter if scatter > 0 else np.inf
    plateaus = bool(t90 <= tv.max())  # the curve levels off inside the run

    rec.update(
        a=a,
        d=d,
        c=c,
        b=b,
        t_peak=t_peak,
        vmax_4pl=vmax_4pl,
        width_4pl=width,
        rise_4pl=rise,
        rise_snr=rise_snr,
        plateaus=plateaus,
        r2_4pl=r2(rss_4pl),  # the 4PL's fit over the whole run, whichever model gives the rate
    )

    # ---- the rules, in order: the first that fails keeps the line ---------------------
    rules = []

    # 1. the curvature is worth the two extra parameters
    rules.append((_bic(rss_4pl, n, 4) < _bic(rss_line, n, 2), "4PL no better than a line (BIC)"))

    # 2. there is a steepest point
    rules.append((b > 1, "4PL has no steepest point (b ≤ 1)"))

    # 3. ... and it is inside the measured time range (a NaN t_peak fails too)
    rules.append((tv.min() <= t_peak <= tv.max(), "4PL's steepest point is outside the run"))

    # 4. the rise is resolved by the sampling
    rules.append((width >= dt, "4PL rises faster than the sampling"))

    # 5. the rise stands out from the noise
    if min_rise_snr is not None:
        rules.append((rise_snr >= min_rise_snr, f"4PL rise < {min_rise_snr} × noise"))

    # 6. no more product than there was substrate
    if max_rise_frac is not None and s_conc is not None:
        rules.append((rise <= max_rise_frac * s_conc, f"4PL rise > {max_rise_frac} × [S]"))

    # 7. a curve that levels off inside the run has used up its substrate
    if plateau_min_frac is not None and s_conc is not None:
        used_up = (not plateaus) or (d - a) >= plateau_min_frac * s_conc
        rules.append((used_up, f"4PL plateaus below {plateau_min_frac} × [S]"))

    for passed, why in rules:
        if not passed:
            rec["reason"] = why
            return rec

    rec.update(model="4pl", vmax=vmax_4pl, r2=r2(rss_4pl), reason=None)  # no reason needed
    return rec


def fit_4pl_rates(
    df,
    value="median",
    *,
    t="seconds",
    end=None,
    t_rate=None,
    per=("mark_row", "mark_col"),
    keep=("tag", "x", "y"),
    p0=guess_4pl,
    bounds=BOUNDS_4PL,
    rising=False,
    min_rise_snr=None,
    substrate=None,
    max_rise_frac=None,
    plateau_min_frac=None,
    min_points=5,
    n_jobs=None,
):
    """Per-chamber rate from a 4PL fit of `value` vs `t` (kinetic_traces DataFrame), with
    a straight line as the fallback when the 4PL is fitting noise.

    `end` restricts the fit to t <= end (the early phase; MBO-039 used 1000 s). Returns
    one row per chamber: the `per` keys, `keep` columns, then
      a, d, c, b, t_peak   4PL parameters and the time of its steepest slope
      vmax_4pl             the 4PL's signed steepest slope (NaN when b <= 1)
      width_4pl            the 4PL's 10-90 % rise time
      slope_line           slope of a straight-line fit
      model, vmax          which model gives the rate ('4pl' or 'line'), and that rate
      r2, n                fit quality of the chosen model, and points fit
    See _fit_4pl_summary for how the model is chosen. With `t_rate`, also `rate_at_t` =
    the 4PL's slope at that time (MBO-039's rate_4pl). Every chamber gets a row.

    `p0` (starting point [a, d, c, b], or a function (t, y) -> [a, d, c, b]) and `bounds`
    ((lower, upper) over [a, d, c, b]) go to fit_4pl; defaults guess_4pl and BOUNDS_4PL.
    A p0 function is evaluated here, before the fits are farmed out, so one defined in a
    notebook works with the parallel workers too.

    Two constraints on the 4PL (see _fit_4pl_summary for every rule):
      rising=True        the 4PL can only go up, so its vmax is never negative;
      min_rise_snr=k     a 4PL is used only if its rise across the run is >= k x the
                         points' scatter about it; otherwise the rate is the line's slope.
    and two that compare it with the run's substrate concentration, read from the column
    `substrate` of `df` (e.g. "bzp_uM"; same units as `value`, one product per substrate):
      max_rise_frac=f    the rise across the run is at most f x [S];
      plateau_min_frac=f a 4PL that plateaus inside the run rises by at least f x [S].
    The output also has rise_4pl, rise_snr, plateaus, `model` ("4pl" or "linear") and
    `reason`: why the 4PL was not used (empty when it was).

    Chambers are fit in parallel across `n_jobs` processes (default: half the cores; 1 = serial,
    which is also used for small chips, where starting the workers costs more than it
    saves). Notebooks need nothing extra; a script must call this under
    `if __name__ == "__main__":`, since each worker process re-imports the script."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    if end is not None:
        df = df[df[t] <= end]

    # one job per chamber: its keys, kept columns, and (t, y) arrays
    heads, jobs = [], []
    for key, g in df.groupby(list(per)):
        rec = dict(zip(per, key if isinstance(key, tuple) else (key,)))
        rec.update({c: g[c].iloc[0] for c in keep if c in g.columns})
        heads.append(rec)
        tv, yv = g[t].to_numpy(float), g[value].to_numpy(float)
        ok = np.isfinite(tv) & np.isfinite(yv)
        # evaluate a p0 function in this process: workers can't import notebook functions
        start = p0(tv[ok], yv[ok]) if callable(p0) and ok.sum() >= min_points else p0
        s_conc = float(g[substrate].iloc[0]) if substrate is not None else None
        jobs.append(
            (tv, yv, min_points, t_rate, start, bounds,
             rising, min_rise_snr, s_conc, max_rise_frac, plateau_min_frac)
        )

    # default: half the cores, so the machine stays usable while it runs
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) // 2)
    if n_jobs == 1 or len(jobs) < 200:
        fits = [_fit_4pl_summary(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            fits = list(pool.map(_fit_4pl_summary, jobs, chunksize=32))

    # fixed column order; columns a chamber's fit didn't produce are NaN
    cols = list(per) + [c for c in keep if c in df.columns]
    cols += [
        "a",
        "d",
        "c",
        "b",
        "t_peak",
        "vmax_4pl",
        "width_4pl",
        "rise_4pl",
        "rise_snr",
        "plateaus",
        "r2_4pl",
    ]
    cols += ["slope_line"] + (["rate_at_t"] if t_rate is not None else [])
    cols += ["model", "reason", "vmax", "r2", "n_points"]
    return pd.DataFrame([{**h, **f} for h, f in zip(heads, fits)]).reindex(columns=cols)


# The standard 4PL rules for progress curves (MBO-040, 2026-10-05; see _fit_4pl_summary): the 4PL
# can only rise; its rise across the run is >= 10 x the points' scatter about it; and, when the
# signal is in the substrate's units (calibrated NADH vs [BzP], µM), it rises at most 1.2 x [S]
# and, if it levels off inside the run, by at least 0.5 x [S]
PROGRESS_4PL_RULES = dict(rising=True, min_rise_snr=10, max_rise_frac=1.2, plateau_min_frac=0.5)


def flag_early_depletion(rates, traces, *, t="seconds", per=("mark_row", "mark_col"), min_rise_snr=10,
                         max_half_rise=1.0):
    """Flag runs whose curve is already slowing at the first frame: column `depleted_early`.

    A run is flagged when the 4PL was not used (model "linear") although it beats the line
    by BIC and rises above the noise (rise_snr >= `min_rise_snr`), and its steepest slope is
    at or before the first frame: b <= 1 (steepest at t -> 0) or t_peak before the run's
    first time point; and its 4PL midpoint c falls within the first `max_half_rise` of the
    run (default 1.0: anywhere in it; a slow, slightly bent trace whose midpoint is beyond
    the run is not flagged). On MBO-051 (2026-10-29), c within the run gave the most
    consistent Km: within variants (median CV 0.27) and between the high- and low-pulldown
    halves (median Km ratio 1.02), vs 0.29 / 1.04 with c in the first half. Its substrate is
    largely used up within the first frames, so no frame
    is on the initial slope: the line underestimates v0 and the 4PL's steepest slope is an
    extrapolation (MBO-051, 2026-10-28). Leave these runs out of the MM fits.

    `rates` is fit_progress_curves / fit_4pl_rates output with its `reason` column; `per`
    keys a run in `traces` (e.g. ("bzp_uM", "mark_row", "mark_col"))."""
    span = traces.groupby(list(per))[t].agg(_t0="min", _t1="max").reset_index()
    out = rates.merge(span, on=list(per), how="left")
    reason = out["reason"].fillna("")
    early_peak = reason.str.contains("no steepest point") | (
        reason.str.contains("outside the run") & (out["t_peak"] < out["_t0"]))
    half_done = out["c"] <= out["_t0"] + max_half_rise * (out["_t1"] - out["_t0"])
    out["depleted_early"] = ((out["model"] == "linear") & early_peak & half_done
                             & (out["rise_snr"] >= min_rise_snr))
    return out.drop(columns=["_t0", "_t1"])

def force_4pl_on_depleted(rates, traces, *, t="seconds", per=("mark_row", "mark_col"), rate="v0"):
    """For runs flagged `depleted_early` (flag_early_depletion), set `rate` to the 4PL's
    steepest slope inside the run's time window (slope_4pl_in_window: the 4PL's own steepest
    point when it lies in the data, else the slope at the first frame), and `model` to
    "4pl_forced". The line kept for these runs underestimates v0; this never extrapolates
    before the first frame (MBO-051, 2026-10-29). Other runs are unchanged."""
    span = traces.groupby(list(per))[t].agg(_t0="min", _t1="max").reset_index()
    out = rates.merge(span, on=list(per), how="left")
    sel = out["depleted_early"].fillna(False).astype(bool).to_numpy()
    if sel.any():
        d = out.loc[sel]
        out.loc[sel, rate] = slope_4pl_in_window(d["a"], d["d"], d["c"], d["b"], d["_t0"], d["_t1"])
        out.loc[sel, "model"] = "4pl_forced"
    return out.drop(columns=["_t0", "_t1"])


def flag_poor_4pl(rates, *, min_r2_4pl=0.98, name="left_out_of_mm"):
    """Mark runs whose 4PL fit over the whole run has r² below `min_r2_4pl` (column
    `r2_4pl`, from fit_4pl_rates), in the boolean column `name`, to leave them out of the
    MM fits. A progress curve the 4PL cannot follow (an imaging step mid-run, a flat trace
    with no rise above the noise) gives no usable rate under either model. On MBO-051
    (2026-10-29) every run outside 156.25 µM had r2_4pl >= 0.993 (1st percentile); at
    156.25 µM, 0.98 caught 533 of the 576 runs in the rows with a mid-run dip."""
    out = rates.copy()
    out[name] = ~(out["r2_4pl"] >= min_r2_4pl)  # NaN (no 4PL) is left out too
    return out

def subtract_blank_rate(rates, *, run="bzp_uM", rate="v0", ref_tags=("BLANK",)):
    """Subtract, from every chamber's `rate`, the median `rate` of the reference chambers
    (`ref_tags`, default the BLANKs) in the same run (`run`, e.g. the [S] of each assay): the
    background rate with no enzyme (product carry-over, photo-products). Adds
    f"{rate}_unsubtracted" and f"blank_median_{rate}"; `rate` becomes the difference."""
    out = rates.copy()
    ref = out[out["tag"].isin(ref_tags)].groupby(run)[rate].median()
    out[f"{rate}_unsubtracted"] = out[rate]
    out[f"blank_median_{rate}"] = out[run].map(ref)
    out[rate] = out[rate] - out[f"blank_median_{rate}"]
    return out


def fit_progress_curves(
    traces,
    value="median",
    *,
    t="seconds",
    per=("mark_row", "mark_col"),
    ref_tags=("BLANK",),
    bleach="model",
    max_ref_slope=0.001,
    substrate=None,
    end=None,
    p0=guess_4pl,
    bounds=BOUNDS_4PL,
    n_jobs=None,
    **rules,
):
    """The standard rate step for an assay's progress curves (chamber_timecourse output):

      1. bleach correction, from the `ref_tags` (BLANK) chambers, per exposure (frame):
         `bleach`="model" (default) divides every chamber by a fitted exponential loss to a
         floor (fit_bleaching; reference chambers rising faster than `max_ref_slope` per
         exposure, e.g. product leaking in, are left out of it); "ratio" by the reference
         median itself, exposure by exposure (bleach_correct); None skips it;
      2. a 4PL and a straight line per chamber (fit_4pl_rates), with PROGRESS_4PL_RULES:
         the initial rate `v0` is the 4PL's steepest slope when every rule holds, else the
         line's slope; `model` and `reason` record which and why.

    `per` keys one fit (e.g. ("bzp_uM", "mark_row", "mark_col") for several runs; the bleach
    curve is then computed per run, i.e. per `per` key other than the chamber's). The [S]
    rules (max_rise_frac, plateau_min_frac) apply only with `substrate`, the column holding
    each run's [S] in the signal's units: leave it None while the signal is in counts.
    `rules` override PROGRESS_4PL_RULES (e.g. min_rise_snr=5).

    Returns (traces, rates): the traces with `<value>_bc` (the fitted signal; = `value` when
    `bleach` is None) and the bleach curve in .attrs['bleach_curve'] ({run: {frame: factor}};
    with "model" also .attrs['bleach_fit'], {run: fit_bleaching's dict}),
    and fit_4pl_rates' table, one row per `per` key, with its `vmax` / `vmax_4pl` named `v0` /
    `v0_4pl`: an initial rate from the progress curve, not a Michaelis-Menten Vmax."""
    chamber = {"mark_row", "mark_col"}
    run_keys = [k for k in per if k not in chamber]
    fitted = f"{value}_bc"

    curves, bleach_fits = {}, {}
    if bleach is None:
        out = traces.assign(**{fitted: traces[value]})
    else:
        parts = []
        groups = traces.groupby(run_keys, sort=False) if run_keys else [((), traces)]
        for key, g in groups:
            # one run key: its plain value (e.g. 5000), not a 1-tuple
            run = (key[0] if isinstance(key, tuple) and len(key) == 1 else key) if run_keys else "all"
            if bleach == "model":
                fit = fit_bleaching(g, value, ref_tags=ref_tags, per=[k for k in per if k in chamber],
                                    max_slope=max_ref_slope)
                bleach_fits[run], curves[run] = fit, fit["curve"]
                bc = g.assign(**{fitted: g[value] / g["frame"].map(fit["curve"])})
            else:
                bc = bleach_correct(g, value=value, ref_tags=ref_tags, mode=bleach,
                                    per=[k for k in per if k in chamber])
                curves[run] = bc.attrs["bleach_curve"]
            parts.append(bc)
        out = pd.concat(parts, ignore_index=True)
    out.attrs = {"bleach_curve": curves, **({"bleach_fit": bleach_fits} if bleach_fits else {})}

    kw = dict(PROGRESS_4PL_RULES, **rules)
    if substrate is None:   # the [S] rules need the signal in [S]'s units
        kw["max_rise_frac"] = kw["plateau_min_frac"] = None
    rates = fit_4pl_rates(out, value=fitted, t=t, end=end, per=per, p0=p0, bounds=bounds,
                          substrate=substrate, n_jobs=n_jobs, **kw)
    return out, rates.rename(columns={"vmax": "v0", "vmax_4pl": "v0_4pl"})


def normalize_rates(
    rates,
    signal,
    *,
    rate="vmax",
    by="sum_net",
    min_signal=None,
    per=("mark_row", "mark_col"),
):
    """Divide each chamber's `rate` by its `by` signal (e.g. summed, background-subtracted
    GFP: enzyme amount), matching chambers on `per`. Returns a copy of `rates` with
    columns `by` and `<rate>_per_<by>`.

    `signal` is a one-row-per-chamber DataFrame (e.g. centers_to_df, or one stage of a
    stacked GFP quantification). Chambers whose signal is <= `min_signal` (default 0) get
    NaN: dividing by a signal at the noise floor flips the sign or blows the ratio up,
    which says nothing about the enzyme. Set `min_signal` from the blanks, e.g. their
    median + 3 SD."""
    keys = list(per)
    if signal.duplicated(keys).any():
        raise ValueError(
            "`signal` has more than one row per chamber; select one stage first."
        )

    out = rates.merge(signal[keys + [by]], on=keys, how="left")
    floor = 0 if min_signal is None else min_signal
    enough = out[by] > floor
    out[f"{rate}_per_{by}"] = np.where(enough, out[rate] / out[by], np.nan)
    return out


# Michaelis-Menten, ported from MBO-039 (scratch/4a, 4b): v = Vmax * [S] / (Km + [S]),
# Vmax and Km bounded >= 0, at least 3 distinct [S] per fit.
