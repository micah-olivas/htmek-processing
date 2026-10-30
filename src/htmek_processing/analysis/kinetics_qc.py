"""The standard kinetics QC steps after the per-chamber traces and rate fits, ported from MBO-040
(its 3e and 3i-3k): the orientation check, example rate fits, rate vs enzyme, Michaelis-Menten
examples, fits per variant, Km by substitution scan, Km vs Vmax, the [E] of good fits, and the
comparison of three rate definitions. Each function takes the column names, so it serves a raw
signal (RFU/s) and a calibrated one (µM/s) alike, and returns a figure or tables.

Conventions: traces and rates carry mark_row, mark_col and tag; the substrate column is `s`
(default bzp_uM); one rate per chamber per run is `rate` (default v0)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from .mm import BOUNDS_MM, fit_mm_table, guess_mm, michaelis_menten
from .rates import four_pl, slope_4pl_in_window
from .plots import CONTROL_COLORS, FIT_COLORS

KEYS = ["mark_row", "mark_col"]
SCAN_NAMES = {"A": "Ala", "F": "Phe", "M": "Met"}   # the substitution a tag's last letter names
SCAN_ORDER = ["WT", "Ala", "Phe", "Met", "N41A"]
HIGHLIGHT = "#C0392B"   # WT on the Km plots, as on MBO-040's
RATE_DEFINITIONS = ["linear", "4pl", "dynamic"]
DEF_COLORS = {"linear": FIT_COLORS["line"], "4pl": FIT_COLORS["vmax"], "dynamic": "#1F4E79"}


def _conc_label(s):
    return f"{s / 1000:g} mM" if s >= 1000 else f"{s:g} µM"


def scan_group(tags, controls=("WT", "N41A"), scan_names=SCAN_NAMES):
    """The substitution scan of each tag (its last letter: A -> Ala, F -> Phe, M -> Met), or the
    tag itself for the controls."""
    tags = pd.Series(tags)
    return pd.Series(np.where(tags.isin(controls), tags, tags.str[-1].map(scan_names)), index=tags.index)


def example_chambers(rates, *, rate="v0", s="bzp_uM", controls=("WT", "N41A", "BLANK"), n_library=2):
    """Chambers to show as examples: for each control tag, its chamber with the median rate at the
    highest [S]; then the library's fastest chamber and its median one. Returns
    [(mark_row, mark_col, label)]."""
    top = rates[rates[s] == rates[s].max()]
    out = []
    for tag in controls:
        d = top[top["tag"] == tag].sort_values(rate)
        if len(d):
            f = d.iloc[len(d) // 2]
            out.append((int(f.mark_row), int(f.mark_col), tag))
    lib = top[~top["tag"].isin(controls)].dropna(subset=[rate]).sort_values(rate)
    for f, why in ((lib.iloc[-1], "fastest"), (lib.iloc[len(lib) // 2], "median"))[:n_library]:
        out.append((int(f.mark_row), int(f.mark_col), f"{f.tag} ({why})"))
    return out


def plot_orientation_check(traces, value, *, s="bzp_uM", s_ref=None, ref_tags=("BLANK",)):
    """3e: the signal kept over the run (last / first pass) per chamber at `s_ref` (default the
    highest [S]), with the reference chambers circled. They should look alike, and like a dead
    control, not scatter across the chip's range; if they do, the pinlist is flipped or rotated
    relative to the image."""
    s_ref = traces[s].max() if s_ref is None else s_ref
    ref = traces[traces[s] == s_ref]
    first = ref[ref["frame"] == ref["frame"].min()].set_index(KEYS)
    last = ref[ref["frame"] == ref["frame"].max()].set_index(KEYS)
    kept = (last[value] / first[value]).unstack()
    fig, ax = plt.subplots(figsize=(4.6, 7.2))
    im = ax.imshow(kept, cmap="viridis", vmin=np.nanpercentile(kept, 2), vmax=np.nanpercentile(kept, 98),
                   interpolation="nearest")
    blanks = first[first["tag"].isin(ref_tags)].index
    ax.scatter([c for _, c in blanks], [r for r, _ in blanks], s=60, fc="none", ec="red", lw=1.4)
    ax.set(xlabel="column", ylabel="row")
    ax.set_title(f"{_conc_label(s_ref)}: signal kept (last / first pass); {', '.join(ref_tags)} circled",
                 loc="left", fontsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(ax).append_axes("bottom", size="2.5%", pad=0.5)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("signal kept (last / first)")
    cb.outline.set_linewidth(0.8)
    fig.tight_layout()
    return fig


def plot_example_rate_fits(traces, rates, chambers, value, *, s="bzp_uM", rate="v0", unit="RFU/s",
                           value_label=None):
    """3i: example progress curves with the fit that gave each rate. Rows = `chambers`
    [(row, col, label)], columns = runs from low to high [S], one y-axis per chamber: light
    peach = 4PL with its tangent at the rate (dark orange), gray = the straight line."""
    runs = sorted(traces[s].unique())
    fits = rates.set_index([s] + KEYS)
    fig, axs = plt.subplots(len(chambers), len(runs), figsize=(2.3 * len(runs), 1.9 * len(chambers)),
                            sharey="row", squeeze=False)
    for j, run in enumerate(runs):
        tr = traces[traces[s] == run]
        for i, (r, c, label) in enumerate(chambers):
            ax = axs[i, j]
            g = tr[(tr["mark_row"] == r) & (tr["mark_col"] == c)].sort_values("seconds")
            ax.plot(g["seconds"], g[value], "o", ms=3, color="0.6", mec="k", mew=0.3, zorder=3)
            if (run, r, c) in fits.index:
                f = fits.loc[(run, r, c)]
                t = np.linspace(g["seconds"].min(), g["seconds"].max(), 200)
                if f["model"] == "4pl":
                    ax.plot(t, four_pl(t, f["a"], f["d"], f["c"], f["b"]), color=FIT_COLORS["4pl"])
                    yp = four_pl(f["t_peak"], f["a"], f["d"], f["c"], f["b"])
                    ax.plot(t, yp + f[rate] * (t - f["t_peak"]), color=FIT_COLORS["vmax"], lw=0.7)
                elif np.isfinite(f["slope_line"]):
                    ax.plot(t, g[value].mean() + f["slope_line"] * (t - g["seconds"].mean()),
                            color=FIT_COLORS["line"], lw=0.8)
                ax.text(0.04, 0.93, f"{f[rate]:.3g} {unit}, {f['model']}", transform=ax.transAxes,
                        fontsize=6.5, va="top")
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(_conc_label(run), fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{label}\n(row {r + 1}, col {c + 1})", fontsize=7.5)
    fig.supxlabel("time (s)")
    fig.supylabel(value_label or value)
    fig.tight_layout()
    return fig


def plot_enzyme_vs_rate(rates, enzyme_col, *, rate="v0", s="bzp_uM", s_ref=None, floor=None,
                        unit="RFU/s", enzyme_label=None):
    """3j: the enzyme readout vs the rate per chamber at `s_ref` (default the highest [S]), the
    controls colored; the gray band below `floor` gets no normalized rate."""
    s_ref = rates[s].max() if s_ref is None else s_ref
    d = rates[rates[s] == s_ref]
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    lib = ~d["tag"].isin(CONTROL_COLORS)
    ax.scatter(d.loc[lib, enzyme_col], d.loc[lib, rate], s=6, color="#8FB3D9", lw=0, alpha=0.7, label="library")
    for tag, color in CONTROL_COLORS.items():
        sel = d["tag"] == tag
        if sel.any():
            ax.scatter(d.loc[sel, enzyme_col], d.loc[sel, rate], s=16, color=color, ec="k", lw=0.3, label=tag)
    if floor is not None:
        ax.axvspan(ax.get_xlim()[0], floor, color="0.92", zorder=0, label="below the [E] floor")
    ax.axhline(0, color="0.7", lw=0.6)
    ax.set(xlabel=enzyme_label or enzyme_col, ylabel=f"{rate} ({unit})")
    ax.set_title(_conc_label(s_ref), loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


def plot_mm_examples(points, fits, chambers, rate, *, s="bzp_uM", unit="RFU/s", reference=None):
    """3k: example Michaelis-Menten fits, one square panel per chamber [(row, col, label)]: its
    rates vs [S] and its fitted curve (`fits`: fit_mm_table output). `reference` (rows with `s`
    and `rate`, e.g. the BLANKs' rates) is drawn faint gray in every panel: the floor a rate has
    to clear."""
    fit_by = fits.set_index(KEYS)
    s_grid = np.linspace(0, points[s].max(), 100)
    fig, axs = plt.subplots(1, len(chambers), figsize=(2.9 * len(chambers), 3.3), sharex=True, squeeze=False)
    for ax, (r, c, label) in zip(axs[0], chambers):
        if reference is not None:
            ax.scatter(reference[s], reference[rate], s=8, c="0.85", lw=0, zorder=1)
        pts = points[(points["mark_row"] == r) & (points["mark_col"] == c)]
        ax.plot(pts[s], pts[rate], "o", ms=4, color="0.6", mec="k", mew=0.3, zorder=3)
        if (r, c) in fit_by.index and np.isfinite(fit_by.loc[(r, c), "km"]):
            f = fit_by.loc[(r, c)]
            ax.plot(s_grid, michaelis_menten(s_grid, f["vmax"], f["km"]), color="#1F4E79", zorder=2)
            note = f"K$_m$ {f['km']:,.0f} µM, r² {f['r2']:.2f}"
        else:
            note = "not fit"
        ax.axhline(0, color="0.85", lw=0.6, zorder=0)
        ax.set_title(f"{label} (row {r + 1}, col {c + 1})\n{note}", fontsize=8)
        ax.set_box_aspect(1)
    fig.supxlabel(f"[{s.split('_')[0].upper() if s == 'bzp_uM' else s}] (µM)".replace("[BZP]", "[BzP]"))
    fig.supylabel(f"{rate} ({unit})")
    fig.tight_layout()
    return fig


def plot_mm_per_variant(points, fits_variant, rate, *, s="bzp_uM", unit="RFU/s", n_show=28, ncols=7,
                        controls=("WT", "N41A"), pct=(1, 99)):
    """3k: pooled MM fits per variant (`fits_variant`: fit_mm_table(per=("tag",)) with a
    `chambers` column), the controls first, then the best-replicated variants; each chamber's
    points in its own color, one shared y-axis. The axis spans the `pct` percentiles of the
    points shown, so one chamber with a tiny [E] (a huge normalized rate) can't flatten every
    panel; points beyond it sit at the edge as triangles. The axis is rescaled to ×10ⁿ."""
    is_control = fits_variant["tag"].isin(controls)
    show = pd.concat([fits_variant[is_control], fits_variant[~is_control]]).head(n_show)
    shown = points[points["tag"].isin(show["tag"])]
    lo, hi = np.nanpercentile(shown[rate], pct)
    lo, hi = min(lo, 0.0), hi * 1.1
    exp = int(np.floor(np.log10(abs(hi)))) if hi > 0 else 0
    k = 10.0 ** -exp
    s_grid = np.linspace(0, points[s].max(), 100)
    nrows = int(np.ceil(len(show) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.15 * nrows), sharex=True, sharey=True,
                            squeeze=False)
    for ax, f in zip(axs.flat, show.itertuples()):
        pts = points[points["tag"] == f.tag]
        for j, (_, ch) in enumerate(pts.groupby(KEYS)):
            y = ch[rate].to_numpy(float)
            inside = (y >= lo) & (y <= hi)
            ax.plot(ch[s][inside], y[inside] * k, "o", ms=2.5, color=f"C{j % 10}", alpha=0.8)
            ax.plot(ch[s][y > hi], np.full((y > hi).sum(), hi * k), "^", ms=3.5, color=f"C{j % 10}")
            ax.plot(ch[s][y < lo], np.full((y < lo).sum(), lo * k), "v", ms=3.5, color=f"C{j % 10}")
        if np.isfinite(f.km):
            ax.plot(s_grid, michaelis_menten(s_grid, f.vmax, f.km) * k, color="k", lw=1)
        ax.set_title(f"{f.tag} (n={f.chambers})\nK$_m$ {f.km:,.0f} µM, r² {f.r2:.2f}", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_box_aspect(1)
    axs[0, 0].set_ylim(lo * k, hi * k)
    for ax in axs.flat[len(show):]:
        ax.set_visible(False)
    fig.supxlabel("[BzP] (µM)" if s == "bzp_uM" else s)
    fig.supylabel(f"{rate} (×10$^{{{exp}}}$ {unit}; ▲ above the axis)" if exp else f"{rate} ({unit})")
    fig.tight_layout()
    return fig


def plot_km_vs_vmax(fits_variant, *, unit="RFU/s"):
    """3k: Km vs Vmax of the pooled per-variant fits, sized by the variant's chambers."""
    d = fits_variant[np.isfinite(fits_variant["km"]) & (fits_variant["km"] > 0)]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sizes = 8 + 6 * d["chambers"].fillna(1)
    ax.scatter(d["km"], d["vmax"], s=sizes, color="#1F4E79", alpha=0.6, lw=0)
    for tag, color in CONTROL_COLORS.items():
        sel = d["tag"] == tag
        if sel.any():
            ax.scatter(d.loc[sel, "km"], d.loc[sel, "vmax"], s=40, color=color, ec="k", lw=0.5, label=tag)
    ax.set(xscale="log", xlabel="K$_m$ (µM)", ylabel=f"v$_{{max}}$ ({unit})")
    ax.set_title("per-variant MM fits (size: chambers)", loc="left", fontsize=10)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_e_of_good_fits(enzyme, mm_raw, readouts, *, floor_sd=3, good_r2=0.9, labels=None):
    """3k: is the [E] floor too strict? For each enzyme readout column in `readouts`, the
    readout of the chambers whose raw MM fit is good (r² > `good_r2`) against all library
    chambers and the BLANKs, with the floor that readout would get (BLANK median + `floor_sd` SD)
    and how many good fits it would throw out."""
    good = mm_raw.loc[mm_raw["r2"] > good_r2, KEYS].assign(good=True)
    e = enzyme.merge(good, on=KEYS, how="left")
    e["good"] = e["good"].fillna(False).astype(bool)
    is_blank = e["tag"] == "BLANK"
    fig, axs = plt.subplots(1, len(readouts), figsize=(5.2 * len(readouts), 3.6), squeeze=False)
    for ax, col in zip(axs[0], readouts):
        blank = e.loc[is_blank, col]
        floor = blank.median() + floor_sd * blank.std()
        lib, good_e = e.loc[~is_blank, col], e.loc[~is_blank & e["good"], col]
        lo, hi = np.nanpercentile(e[col], (0.5, 99.5))
        bins = np.linspace(lo, hi, 60)
        ax.hist(lib.clip(lo, hi), bins=bins, color="0.85", label=f"library chambers (n={len(lib)})")
        ax.hist(good_e.clip(lo, hi), bins=bins, color="#1F4E79", alpha=0.85, label=f"raw MM r² > {good_r2} (n={len(good_e)})")
        ax.hist(blank.clip(lo, hi), bins=bins, histtype="step", color="k", lw=1, label=f"BLANK (n={len(blank)})")
        ax.axvline(floor, color=HIGHLIGHT, ls="--", lw=1.2, label=f"floor: BLANK median + {floor_sd} SD")
        ax.set(xlabel=(labels or {}).get(col, col), ylabel="chambers")
        ax.set_title(f"{(good_e <= floor).mean():.0%} of good fits below the floor", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


def plot_km_by_scan(fits, *, km_max, km_min=None, min_r2=0.9, order=SCAN_ORDER, ax=None, title=None):
    """3k: Km per chamber by substitution scan, WT highlighted (red; dashed line = WT median),
    black bars = group medians. Fits with r² < `min_r2` are left out; Km beyond `km_max` (the
    highest [S]) sits at the axis edge as a triangle. With `km_min` (e.g. the lowest [S]) the
    axis is log2 from km_min to km_max, ticked at each doubling from km_min (the [S] series of a
    two-fold dilution), and Km below km_min sits at the left edge as a triangle; else linear from 0."""
    km = fits[fits["r2"] >= min_r2].copy()
    km["group"] = scan_group(km["tag"]).to_numpy()
    order = [g for g in order if g in set(km["group"])]
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6, 0.55 * len(order) + 1.3))
    rng = np.random.default_rng(0)
    for i, grp in enumerate(order):
        d = km[km["group"] == grp]
        is_wt = grp == "WT"
        y = i + rng.uniform(-0.25, 0.25, len(d))
        side = np.where(d["km"] > km_max, ">", np.where((km_min is not None) & (d["km"] < (km_min or 0)), "<", "o"))
        for marker in ("o", ">", "<"):
            sel = side == marker
            ax.scatter(d["km"][sel].clip(km_min, km_max), y[sel], marker=marker, s=26 if is_wt else 10,
                       c=HIGHLIGHT if is_wt else "0.55", ec="k" if is_wt else "none", lw=0.4,
                       alpha=1 if is_wt else 0.6, zorder=3 if is_wt else 2)
        if len(d):
            ax.plot([np.clip(d["km"].median(), km_min, km_max)] * 2, [i - 0.35, i + 0.35], color="k", lw=1.5, zorder=4)
    wt = km.loc[km["group"] == "WT", "km"]
    if len(wt):
        ax.axvline(np.clip(wt.median(), km_min, km_max), color=HIGHLIGHT, ls="--", lw=0.8, zorder=1,
                   label=f"WT median ({wt.median():,.0f} µM)")
    ax.scatter([], [], marker=">", s=10, c="0.55", label=f"K$_m$ > {km_max:,.0f} µM (off scale)")
    if km_min is not None:
        ax.scatter([], [], marker="<", s=10, c="0.55", label=f"K$_m$ < {km_min:,g} µM (off scale)")
    ax.set(yticks=range(len(order)), yticklabels=[f"{g} (n={(km['group'] == g).sum()})" for g in order],
           ylim=(len(order) - 0.5, -0.5), xlabel="K$_m$ (µM)")
    if km_min is None:
        ax.set_xlim(0, km_max * 1.03)
    else:  # log2, ticked at the doublings from km_min
        ax.set_xscale("log", base=2)
        ax.set_xlim(km_min / 1.1, km_max * 1.1)
        ticks = km_min * 2.0 ** np.arange(int(np.floor(np.log2(km_max / km_min))) + 1)
        ax.set_xticks(ticks, [f"{t:,g}" for t in ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())  # unlabeled minor ticks
    ax.set_title(title or f"K$_m$ per chamber (r² ≥ {min_r2})", loc="left", fontsize=10)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    if own:
        fig.tight_layout()
        return fig
    return ax.figure


def compare_rate_definitions(traces, rates, *, s="bzp_uM", rate="v0", min_r2=0.9,
                             p0=guess_mm, bounds=BOUNDS_MM):
    """3k: the rate under three definitions per chamber and run: `linear` (the straight line's
    slope over the run), `4pl` (the 4PL's steepest slope inside the data window,
    slope_4pl_in_window, so it never extrapolates) and `dynamic` (the fit's own choice,
    `rate`); then one MM fit per chamber per definition (raw rates, BLANKs left out).

    Returns (rate_defs: long, one row per chamber, run and definition; mm_defs: the MM fits with
    `definition` and `good` = r² >= min_r2; summary: per definition, fit counts, median r², Km
    beyond the highest [S], WT's Km median and CV, and the median Km CV within variants with at
    least 3 good chambers)."""
    window = traces.groupby([s] + KEYS)["seconds"].agg(["min", "max"]).reset_index()
    rd = rates.merge(window, on=[s] + KEYS)
    rd["linear"] = rd["slope_line"]
    rd["4pl"] = slope_4pl_in_window(rd["a"], rd["d"], rd["c"], rd["b"], rd["min"], rd["max"])
    rd["dynamic"] = rd[rate]
    rate_defs = rd.melt(id_vars=[s] + KEYS + ["tag"], value_vars=RATE_DEFINITIONS,
                        var_name="definition", value_name="rate")
    not_blank = rate_defs[rate_defs["tag"] != "BLANK"]
    mm_defs = pd.concat([fit_mm_table(g, "rate", s=s, p0=p0, bounds=bounds).assign(definition=name)
                         for name, g in not_blank.groupby("definition", sort=False)], ignore_index=True)
    mm_defs["good"] = mm_defs["r2"] >= min_r2
    s_max = rate_defs[s].max()

    def summary(g):
        good = g[g["good"]]
        wt = good.loc[good["tag"] == "WT", "km"]
        reps = good.groupby("tag")["km"].agg(["count", "std", "mean"])
        reps = reps[reps["count"] >= 3]
        return pd.Series({
            "chambers fit": int(g["r2"].gt(0).sum()),
            "median r²": g.loc[g["r2"] > 0, "r2"].median(),
            "Km > highest [S]": int(g["km"].gt(s_max).sum()),
            f"good (r² ≥ {min_r2})": int(g["good"].sum()),
            "WT Km median (µM)": wt.median(),
            "WT Km CV": wt.std() / wt.mean() if len(wt) > 1 else np.nan,
            "median Km CV within variants (≥3 good)": (reps["std"] / reps["mean"]).median(),
        })

    table = mm_defs.groupby("definition", sort=False).apply(summary, include_groups=False).T
    return rate_defs, mm_defs, table[[d for d in RATE_DEFINITIONS if d in table.columns]]


def plot_rate_definition_examples(rate_defs, mm_defs, chambers, *, s="bzp_uM", unit="RFU/s"):
    """3k: each example chamber's rates and MM fit under the three definitions, overlaid."""
    s_grid = np.linspace(0, rate_defs[s].max(), 100)
    fits = mm_defs.set_index(["definition"] + KEYS)
    fig, axs = plt.subplots(1, len(chambers), figsize=(2.9 * len(chambers), 3.4), squeeze=False)
    for ax, (r, c, label) in zip(axs[0], chambers):
        for k, name in enumerate(RATE_DEFINITIONS):
            pts = rate_defs[(rate_defs["definition"] == name) & (rate_defs["mark_row"] == r) & (rate_defs["mark_col"] == c)]
            ax.plot(pts[s] * (1 + (k - 1) * 0.015), pts["rate"], "o", ms=4, color=DEF_COLORS[name],
                    mec="k", mew=0.3, zorder=3, label=name)
            if (name, r, c) in fits.index and np.isfinite(fits.loc[(name, r, c), "km"]):
                f = fits.loc[(name, r, c)]
                ax.plot(s_grid, michaelis_menten(s_grid, f["vmax"], f["km"]), lw=1.2, color=DEF_COLORS[name], zorder=2)
        ax.axhline(0, color="0.85", lw=0.6, zorder=0)
        ax.set_title(f"{label} (row {r + 1}, col {c + 1})", fontsize=8.5)
        ax.set_box_aspect(1)
        ax.set_xlabel("[BzP] (µM)" if s == "bzp_uM" else s)
    axs[0, 0].set_ylabel(f"rate ({unit})")
    axs[0, 0].legend(fontsize=7, loc="upper left", frameon=False)
    fig.tight_layout()
    return fig


def plot_rate_definition_km_scatter(mm_defs, *, km_max):
    """3k: Km per chamber, each definition against another (chambers good under both; WT red;
    dotted = equal)."""
    wide = mm_defs[mm_defs["good"]].pivot_table(index=KEYS + ["tag"], columns="definition", values="km").reset_index()
    lim = (0, km_max)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, (x, y) in zip(axs, (("linear", "dynamic"), ("4pl", "dynamic"), ("linear", "4pl"))):
        if x not in wide or y not in wide:
            ax.set_visible(False)
            continue
        d = wide.dropna(subset=[x, y])
        is_wt = d["tag"] == "WT"
        ax.scatter(d.loc[~is_wt, x].clip(*lim), d.loc[~is_wt, y].clip(*lim), s=7, c="0.6", lw=0)
        ax.scatter(d.loc[is_wt, x].clip(*lim), d.loc[is_wt, y].clip(*lim), s=28, c=HIGHLIGHT, ec="k", lw=0.5, label="WT")
        ax.plot(lim, lim, color="0.3", lw=0.8, ls=":", zorder=0)
        rho = d[[x, y]].corr("spearman").iloc[0, 1] if len(d) > 2 else np.nan
        ax.set(xlim=lim, ylim=lim, xlabel=f"K$_m$, {x} (µM)", ylabel=f"K$_m$, {y} (µM)")
        ax.set_title(f"n={len(d)}, Spearman ρ={rho:.2f}", loc="left", fontsize=9)
        ax.set_box_aspect(1)
    fig.tight_layout()
    return fig


def plot_rate_definition_km_by_scan(mm_defs, *, km_max, km_min=None, min_r2=0.9):
    """3k: Km by substitution scan under each rate definition, side by side."""
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, name in zip(axs, RATE_DEFINITIONS):
        d = mm_defs[mm_defs["definition"] == name]
        plot_km_by_scan(d, km_max=km_max, km_min=km_min, min_r2=min_r2, ax=ax, title=f"{name} (n={int((d['r2'] >= min_r2).sum())})")
    fig.tight_layout()
    return fig
