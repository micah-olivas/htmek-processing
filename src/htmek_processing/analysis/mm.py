"""Michaelis-Menten fits: one curve, one chamber at a time or a table of them, with an optional
one-run-left-out rescue."""

from __future__ import annotations

import numpy as np
import pandas as pd


def michaelis_menten(s, vmax, km):
    """v = Vmax * [S] / (Km + [S])."""
    return vmax * s / (km + s)


def guess_mm(s, v):
    """Default Michaelis-Menten starting point [Vmax, Km] (MBO-039): Vmax = the largest
    rate, Km = the median positive [S]."""
    s, v = np.asarray(s, float), np.asarray(v, float)
    return [np.max(v), np.median(s[s > 0]) if (s > 0).any() else 1.0]


# Default Michaelis-Menten bounds, as (lower, upper) over [Vmax, Km]: both non-negative.
BOUNDS_MM = (
    [0, 0], 
    [np.inf, np.inf]
    )


def fit_mm(s, v, *, p0=guess_mm, bounds=BOUNDS_MM):
    """Fit michaelis_menten to one set of (substrate, rate) points.

    Returns a dict: vmax, km, their standard errors, r2, n and km_note. Km is unbounded
    above (by default) and reported as fitted; a Km beyond the highest [S] just means the
    data never reach half-saturation, so its standard error (km_se) is large. NaN fits when
    there are fewer than 3 distinct [S] or the fit fails.

    Km is undefined when Vmax ends on its lower bound (0 by default): the fitted curve is
    then flat, Km has no effect on it, and the optimizer leaves Km at its starting guess. That
    happens for chambers with no activity (rates ~0 or negative). Their Km is NaN, with
    km_note = "Vmax at its lower bound: no Km", so it can't be mistaken for a measurement.

    `p0` is the starting point [Vmax, Km], or a function (s, v) -> [Vmax, Km]; `bounds` is
    (lower, upper) over [Vmax, Km]. Both are in the rates' own units.

    Rates are rescaled to ~1 before fitting and scaled back after: normalized rates can be
    ~1e-5, where curve_fit's default tolerances stop the optimizer early. Km is unchanged
    by the rescaling; Vmax, its guess and its bounds are scaled with the rates."""
    from scipy.optimize import curve_fit

    s, v = np.asarray(s, float), np.asarray(v, float)
    ok = np.isfinite(s) & np.isfinite(v)
    s, v = s[ok], v[ok]
    out = dict(
        vmax=np.nan,
        km=np.nan,
        vmax_se=np.nan,
        km_se=np.nan,
        r2=np.nan,
        n=len(s),
        km_note="",
    )
    if len(np.unique(s)) < 3:  # fit requires more than 2 unique [S]'s
        return out

    scale = np.nanmax(np.abs(v)) or 1.0
    y = v / scale
    unit = np.array([scale, 1.0])  # [Vmax, Km] in fitting units = value / unit
    lower = np.asarray(bounds[0], float) / unit
    upper = np.asarray(bounds[1], float) / unit
    start = np.asarray(p0(s, v) if callable(p0) else p0, float) / unit
    start = np.clip(start, lower, upper)
    
    try:
        popt, pcov = curve_fit(
            michaelis_menten, s, y, p0=start, bounds=(lower, upper), maxfev=10_000
        )
    except (RuntimeError, ValueError):
        return out

    se = np.sqrt(np.diag(pcov))
    resid = y - michaelis_menten(s, *popt)
    ss_tot = np.sum((y - y.mean()) ** 2)
    out.update(
        vmax=popt[0] * scale,
        km=popt[1],
        vmax_se=se[0] * scale,
        km_se=se[1],
        r2=1 - np.sum(resid**2) / ss_tot if ss_tot > 0 else np.nan,
    )

    # Vmax on its lower bound: the curve is flat, so Km was never fitted (it stays at the
    # starting guess). Don't report it as a number.
    vmax_floor = lower[0]
    if np.isfinite(vmax_floor) and popt[0] <= vmax_floor + 1e-9:
        out.update(km=np.nan, km_se=np.nan, km_note="Vmax at its lower bound: no Km")
    return out


def fit_mm_one_out(s, v, *, min_r2, p0=guess_mm, bounds=BOUNDS_MM):
    """fit_mm, with one [S] left out when that is what lets the fit pass.

    The fit uses every point first. If its r² is below `min_r2`, it is redone once per
    [S] with that [S] left out (all its points), and the best of those (highest r²) is used
    instead, but only if it reaches `min_r2` (so it also improves on the full fit). A single
    run gone wrong (e.g. a rate read too high) then no longer sinks the chamber, and the
    chamber is marked as a conditional pass. Leaving out a point needs at least 4 distinct
    [S], so the fit keeps 3.

    Returns fit_mm's dict plus:
      r2_all  r² of the fit to every point;
      left_out  the [S] left out (NaN when every point is used);
      r2_status  "pass" (every point, r² >= min_r2), "conditional pass" (one [S] left
                 out) or "fail"."""
    s, v = np.asarray(s, float), np.asarray(v, float)
    full = fit_mm(s, v, p0=p0, bounds=bounds)
    out = dict(full, r2_all=full["r2"], left_out=np.nan,
               r2_status="pass" if full["r2"] >= min_r2 else "fail")
    levels = np.unique(s[np.isfinite(s)])
    if out["r2_status"] == "pass" or len(levels) < 4:
        return out

    best = None
    for level in levels:
        keep = s != level
        trial = fit_mm(s[keep], v[keep], p0=p0, bounds=bounds)
        if np.isfinite(trial["r2"]) and (best is None or trial["r2"] > best[1]["r2"]):
            best = (level, trial)
    if best is not None and best[1]["r2"] >= min_r2:
        out = dict(best[1], r2_all=full["r2"], left_out=float(best[0]),
                   r2_status="conditional pass")
    return out


def fit_mm_table(
    df,
    rate,
    *,
    s="bzp_uM",
    per=("mark_row", "mark_col"),
    keep=("tag",),
    p0=guess_mm,
    bounds=BOUNDS_MM,
    one_out_min_r2=None,
):
    """One Michaelis-Menten fit per group of `per` (default: per chamber; per=("tag",)
    pools every chamber of a variant). Returns the `per` keys, `keep` columns, and the
    fit_mm columns, one row per group. `p0` and `bounds` go to fit_mm.

    With `one_out_min_r2`, a fit whose r² falls below it may leave out one [S]
    (fit_mm_one_out), adding the columns r2_all, left_out and r2_status."""
    rows = []
    for key, g in df.groupby(list(per)):
        rec = dict(zip(per, key if isinstance(key, tuple) else (key,)))
        rec.update({c: g[c].iloc[0] for c in keep if c in g.columns and c not in per})
        if one_out_min_r2 is None:
            rec.update(fit_mm(g[s], g[rate], p0=p0, bounds=bounds))
        else:
            rec.update(fit_mm_one_out(g[s], g[rate], min_r2=one_out_min_r2, p0=p0,
                                      bounds=bounds))
        rows.append(rec)
    return pd.DataFrame(rows)
