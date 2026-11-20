"""A rate that relaxes during a run: e.g. a bound monovalent cation (MVC) exchanging with the new buffer's after the
buttons open, fit jointly over one chamber's runs.

At [S] below K_m the rate is first order in substrate, v = a(t) S(t), with a = (kcat/K_m)[E]. In run b:

    P_b(t) = P0_b + S [1 - exp(-A_b(t))]
    single rate:  A_b = a_b t
    exchange:     A_b = a_inf t + (a_0 - a_inf)(1 - exp(-k_x t)) / k_x

so a relaxes from a_0 to a_inf at rate constant k_x, and S(t) depletes. S, the chamber's substrate in its own signal
units, is shared by its runs (the same [S] read through the same calibration), so depletion is pinned by every run
together and need not be known in absolute units. Each run is tested by dropping its exchange from the fit that allows
it in every run (AIC)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

KEYS = ["mark_row", "mark_col"]
LN2 = np.log(2)


def _A_exchange(t, a0, ainf, kx):
    return ainf * t + (a0 - ainf) * (1 - np.exp(-kx * t)) / kx


def substrate_left(A, sigma):
    """S(t) / S0 under Michaelis-Menten turnover with a time-varying kcat/K_m: dS/dt = -a(t) S K_m / (K_m + S) separates to
    ln(S/S0) + (S - S0)/K_m = -A(t), A = the integral of a, so S/S0 = W(sigma exp(sigma - A)) / sigma with sigma = S0 / K_m
    (W: Lambert W). sigma = 0 (K_m >> S0) is first order, exp(-A)."""
    from scipy.special import lambertw

    A, sigma = np.broadcast_arrays(np.asarray(A, float), np.asarray(sigma, float))
    first = sigma <= 1e-9
    sg = np.where(first, 1.0, sigma)
    z = np.log(sg) + sg - A   # log of W's argument, kept in logs: exp(sigma - A) can underflow
    w = np.where(z > -30, np.real(lambertw(np.exp(np.minimum(z, 700)))), np.exp(z))   # W(x) ~ x for tiny x
    return np.where(first, np.exp(-A), w / sg)


def fit_exchange_joint(curves, order=None, *, half_life_s=(20.0, 3600.0), sigma=0.0, run_scale=None):
    """One chamber: `curves` = {run: (t, p)} (t from when the run starts, s; p the product). Fits the chamber with
    exchange allowed in every run (one shared S), then once per run with that run forced to a single rate (the others
    still free): d_aic for a run is the AIC of that reduced fit minus the full fit's, so > 0 favors exchange in that run,
    and another run's curvature cannot pass for it through S. Returns one dict per run: S_hat, a0, a_inf, k_x (the full
    fit), a_single (that run in its reduced fit), d_aic. k_x is bounded to half-lives within `half_life_s`. `sigma` =
    [S] / K_m (the chamber's K_m, e.g. from its MM series) makes the turnover Michaelis-Menten (substrate_left); 0: first
    order. `run_scale` = {run: factor}: each run's substrate is S x its factor (e.g. each buffer's plateau relative to
    one, when the runs' substrate stocks differ); S then stands for the chamber's calibration, shared by its runs."""
    from scipy.optimize import least_squares

    runs = [r for r in (order or curves) if r in curves]
    scale = {r: float((run_scale or {}).get(r, 1.0)) for r in runs}
    n = sum(len(curves[r][0]) for r in runs)
    rise = max(max(float(p[-1] - p[0]), 1e-3) for t, p in curves.values())
    kx_lo, kx_hi = LN2 / half_life_s[1], LN2 / half_life_s[0]

    def unpack(x, ex):
        s, i, par = np.exp(x[0]), 1, {}
        for r in runs:
            if r in ex:
                par[r] = ("ex", x[i], np.exp(x[i + 1]), np.exp(x[i + 2]), np.exp(x[i + 3])); i += 4
            else:
                par[r] = ("single", x[i], np.exp(x[i + 1])); i += 2
        return s, par

    def resid(x, ex):
        s, par = unpack(x, ex)
        out = []
        for r in runs:
            t, p = curves[r]
            q = par[r]
            A = _A_exchange(t, q[2], q[3], q[4]) if q[0] == "ex" else q[2] * t
            out.append(p - (q[1] + s * scale[r] * (1 - substrate_left(A, sigma))))
        return np.concatenate(out)

    def bounds(ex):
        lo, hi = [np.log(rise * 0.5)], [np.log(1e7)]
        for r in runs:
            if r in ex:
                lo += [-np.inf, np.log(1e-10), np.log(1e-10), np.log(kx_lo)]; hi += [np.inf, 0.0, 0.0, np.log(kx_hi)]
            else:
                lo += [-np.inf, np.log(1e-10)]; hi += [np.inf, 0.0]
        return np.array(lo), np.array(hi)

    def solve(ex, starts):
        lo, hi = bounds(ex)
        best = None
        for x0 in starts:
            x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
            try:
                r = least_squares(resid, x0, args=(ex,), bounds=(lo, hi), max_nfev=6000)
            except ValueError:
                continue
            if best is None or r.cost < best.cost:
                best = r
        return best

    def aic(fit, ex):
        return n * np.log(max(2 * fit.cost, 1e-300) / n) + 2 * (1 + 2 * len(runs) + 2 * len(ex))

    def start(logs, par, ex, kx0, r0):
        x = [logs]
        for r in runs:
            p0, a = par[r][1], par[r][2]
            x += [p0, np.log(a * r0), np.log(a), np.log(kx0)] if r in ex else [p0, np.log(a)]
        return np.array(x)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        starts = []
        for m in (1.5, 4.0, 30.0):   # S at a few multiples of the largest rise: depleting to nearly linear
            s0, x = rise * m, [np.log(rise * m)]
            for r in runs:
                t, p = curves[r]
                a = max(-np.log(max(1 - (p[-1] - p[0]) / s0, 1e-3)) / max(t[-1], 1.0), 1e-9)
                x += [p[0], np.log(a)]
            starts.append(np.array(x))
        single = solve(set(), starts)
        if single is None:
            return []
        _, par0 = unpack(single.x, set())
        full_set = set(runs)
        # the full fit from the single fit, each run starting flat or relaxing (fast, medium, slow; falling or rising)
        st = [start(single.x[0] + d, par0, full_set, kx0, r0) for d in (0.0, 1.0) for kx0 in (LN2 / 60, LN2 / 240, LN2 / 600)
              for r0 in (0.5, 1.0, 2.0)]
        full = solve(full_set, st)
        if full is None:
            return []
        s_hat, par = unpack(full.x, full_set)
        aic_full = aic(full, full_set)
        rows = {}
        for b in runs:
            red = full_set - {b}
            # the reduced fit from the full one: run b as a single rate at its average a, plus the single fit
            pb = dict(par)
            pb[b] = ("single", par[b][1], max((par[b][2] + par[b][3]) / 2, 1e-9))
            x = [full.x[0]]
            for r in runs:
                x += [pb[r][1], np.log(pb[r][2]), np.log(pb[r][3]), np.log(pb[r][4])] if r in red else [pb[r][1], np.log(pb[r][2])]
            fit = solve(red, [np.array(x), start(single.x[0], par0, red, LN2 / 240, 1.0)])
            q = par[b]
            row = dict(run=b, S_hat=s_hat, a0=q[2], a_inf=q[3], k_x=q[4], p0=q[1], n=len(curves[b][0]),
                       P_rise=float(curves[b][1][-1] - curves[b][1][0]), a_single=par0[b][2], p0_single=par0[b][1],
                       S_hat_single=np.exp(single.x[0]))
            if fit is not None:
                row["d_aic"] = aic(fit, red) - aic_full
            rows[b] = row
        return [rows[r] for r in runs]


def _fit_one(args):
    key, curves, order, half_life_s, sigma, run_scale = args
    return [dict(zip(KEYS, key), sigma=sigma, run_scale=(run_scale or {}).get(r["run"], 1.0), **r)
            for r in fit_exchange_joint(curves, order, half_life_s=half_life_s, sigma=sigma, run_scale=run_scale)]


def fit_exchange_table(traces, value, *, run="condition", t="seconds", order=None, min_points=8, half_life_s=(20.0, 3600.0),
                       d_aic_min=4.0, change_min=0.25, half_life_max_s=600.0, half_life_min_s=60.0, sigma=None, run_scale=None, n_jobs=None):
    """fit_exchange_joint for every chamber of `traces` (one row per chamber, run and pass; `value` the product) that
    has every run with at least `min_points` points; in parallel over `n_jobs` processes (default half the cores).
    Returns one row per chamber and run, with ratio = a_inf / a_0, half_life_s = ln2 / k_x, turnover = the run's rise
    / S_hat, and `exchange`: the exchange fit kept (d_aic >= d_aic_min, a rate change of at least `change_min` either
    way, and a half-life from `half_life_min_s`, about one pass (faster happens before the first image), to
    `half_life_max_s`). `sigma`: {(mark_row, mark_col): [S] / K_m} for Michaelis-Menten turnover per chamber (missing
    chambers, or None: first order)."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    from ..workers import headless_workers

    order = list(order or sorted(traces[run].unique()))
    jobs = []
    for key, g in traces.sort_values(t).groupby(KEYS):
        curves = {}
        for r, h in g.groupby(run):
            tt, pp = h[t].to_numpy(float), h[value].to_numpy(float)
            ok = np.isfinite(tt) & np.isfinite(pp)
            if ok.sum() >= min_points:
                curves[r] = (tt[ok], pp[ok])
        if len(curves) == len(order):
            jobs.append((key, curves, order, half_life_s, float((sigma or {}).get(key, 0.0)), run_scale))
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) // 2)
    if n_jobs == 1 or len(jobs) < 20:
        rows = [r for j in jobs for r in _fit_one(j)]
    else:
        with headless_workers(), ProcessPoolExecutor(max_workers=n_jobs) as pool:
            rows = [r for out in pool.map(_fit_one, jobs, chunksize=8) for r in out]
    fits = pd.DataFrame(rows).rename(columns={"run": run})
    fits["ratio"] = fits["a_inf"] / fits["a0"]
    fits["half_life_s"] = LN2 / fits["k_x"]
    fits["turnover"] = fits["P_rise"] / (fits["S_hat"] * fits.get("run_scale", 1.0))
    fits["exchange"] = ((fits["d_aic"] >= d_aic_min) & ((fits["ratio"] <= 1 / (1 + change_min)) | (fits["ratio"] >= 1 + change_min))
                        & fits["half_life_s"].between(half_life_min_s, half_life_max_s))
    fits[run] = pd.Categorical(fits[run], order, ordered=True)
    return fits.sort_values([run, *KEYS]).reset_index(drop=True)
