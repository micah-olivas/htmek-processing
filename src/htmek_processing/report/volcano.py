"""Per-variant effects with p-values, and the volcano plot: how much each variant's Km, its
Vmax per enzyme and its kcat/Km differ from the reference, and how sure we are.

Ported from the MBO-040 / MBO-051 downstream scripts (7_downstream/volcano, 2026-10-29).
The functions take tidy tables, so any experiment can use them:

    fits, rates = volcano_inputs(gfp, mm_raw, mm_per_gfp, rates, floor=gfp_floor(gfp))
    for q in quantities(rate_unit="RFU/s"):
        table, ref, s_range = variant_table(q, fits, rates)
        fig = plot_volcano(q, table, ref, s_range)

The method:
  - Chambers: MM fits with r² >= `min_r2`, in chambers whose GFP clears the floor
    (`E_above_floor`). Without enzyme, a "fit" is background.
      Km:      the raw fit (rate vs [S]); Km does not depend on GFP.
      Vmax:    the per-GFP fit (rate / GFP vs [S]), so that expression does not count.
      kcat/Km: (Vmax / GFP) / Km of the per-GFP fit, the initial slope; it is set even when
               Km is beyond the highest [S].
  - Values the [S] range only bounds: Km outside [lowest, highest] [S] is clipped to that
    range; Vmax, when Km is above the highest [S], is replaced by the chamber's rate / GFP at
    the highest [S], a lower bound; kcat/Km with Km below the lowest [S] is a lower bound. A
    variant whose chambers mostly sit on a bound is drawn as a limit (triangle).
  - Fold change: the median of the variant's chambers minus the reference's, in log2. The
    reference is WT's chambers, or, with fewer than `min_ref_chambers` WT chambers, every
    library chamber (all variants but WT and BLANK) except the tested variant's own, so a
    variant is never part of its own reference (as WT never holds the mutant).
  - p: bootstrap hypothesis test (Markin et al. 2021, Science, supplement "Testing
    statistical significance"): pool the variant's m chambers with the reference's n, draw m
    and n from the pool with replacement, record t* = |median(m draws) - median(n draws)|;
    p = (#(t* >= t_obs) + 1) / (N + 1), with t_obs the observed |difference of medians|.
    N steps through BOOT_STEPS (100 -> 1,000 -> 100,000) while p is small.
  - A fixed reference (`fixed_reference`, e.g. a Km measured on another chip when WT did not
    express here) replaces the reference chambers with one value: fold change = the variant's
    median minus it, in log2; p from bootstrap_p_fixed (the variant's chambers, shifted to
    the reference's median, resampled). Marked as a placeholder on the plot when it is one.
  - The 95 % interval of the fold change resamples the variant's chambers alone.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

KEYS = ["mark_row", "mark_col"]
SCANS = {"A": "Ala scan", "F": "Phe scan", "M": "Met scan"}  # mutant residue -> panel
BOOT_STEPS = ((100, 0.1), (1_000, 0.01), (100_000, None))  # (iterations, go on if p <= this)
COLORS = {"significant": "#E8833A", "not significant": "0.72", "edge": "0.15"}
SIZE_PER_REPLICATE = 14  # volcano marker area (pt²) per chamber
SIZE_LEGEND = (2, 4, 6)  # replicate counts shown in the size legend


@dataclass(frozen=True)
class Quantity:
    """One volcano: which MM fit and value, how it is labeled."""

    name: str      # "km", "vmax" or "kcat_km": file stems, table columns
    fit: str       # "raw" or "per_E" (the `fit` column of volcano_inputs' fits)
    label: str     # axis label (mathtext)
    unit: str      # unit of the value
    column: str    # table column stem for the value
    markers: dict  # limit -> marker: "" a circle, else an isosceles triangle


def quantities(rate_unit="µM/s", s_unit="µM"):
    """Km, Vmax / GFP and kcat/Km, with units from the rate's (`rate_unit`) and [S]'s."""
    per_gfp = f"{rate_unit} per GFP RFU"
    return [
        # Km beyond the [S] range: > highest points right, < lowest points left
        Quantity("km", "raw", "K$_m$", s_unit, "km", {"": "o", "lower": ">", "upper": "<"}),
        # Vmax where Km > highest [S] (the rate at the highest [S]): a left-pointing triangle
        Quantity("vmax", "per_E", "v$_{max}$ / GFP", per_gfp, "vmax_per_gfp",
                 {"": "o", "lower": "<"}),
        Quantity("kcat_km", "per_E", "k$_{cat}$/K$_m$", f"{per_gfp} per {s_unit}", "kcat_km",
                 {"": "o", "lower": ">"}),
    ]


def gfp_floor(gfp, column="button_sum_minus_bg", n_sd=3, ref_tags=("BLANK",)):
    """The GFP floor: the reference chambers' (BLANK) median + `n_sd` SD of `column`."""
    ref = gfp.loc[gfp["tag"].isin(ref_tags), column]
    return float(ref.median() + n_sd * ref.std())


def volcano_inputs(gfp, mm_raw, mm_per_gfp, rates, *, floor, gfp_col="button_sum_minus_bg",
                   rate="v0", min_r2=0.9, s="bzp_uM", left_out="left_out_of_mm"):
    """The two tables the volcano functions read, from tidy per-chamber tables:
      gfp        one row per chamber (one GFP scan), with `gfp_col`
      mm_raw     raw MM fits per chamber (fit_mm_table: vmax, km, r2)
      mm_per_gfp per-GFP MM fits per chamber
      rates      one row per chamber and run, with `rate` and `s`
    Returns (fits: one row per chamber and fit, `fit` = "raw" / "per_E", km_uM, vmax,
    passes_r2_filter, E_above_floor; rates: with gfp, E_above_floor and rate_per_gfp_count).
    Runs marked in `left_out` (when present) are dropped from rates, as from the MM fits."""
    e = gfp[KEYS + [gfp_col]].rename(columns={gfp_col: "gfp"})
    e["E_above_floor"] = e["gfp"] > floor
    fits = pd.concat([mm_raw.assign(fit="raw"), mm_per_gfp.assign(fit="per_E")],
                     ignore_index=True).rename(columns={"km": "km_uM"})
    fits["passes_r2_filter"] = fits["r2"] >= min_r2
    fits = fits.merge(e[KEYS + ["E_above_floor"]], on=KEYS, how="left")
    rates = rates.copy()
    if left_out in rates:
        rates = rates[~rates[left_out].fillna(False).astype(bool)]
    rates = rates.merge(e, on=KEYS, how="left")
    rates["rate_per_gfp_count"] = rates[rate] / rates["gfp"]
    if s != "bzp_uM":  # the functions below read the run's [S] as bzp_uM
        rates["bzp_uM"] = rates[s]
    return fits, rates


def pooled_replicate_sd(log2_value, tags):
    """The SD of log2 value between replicate chambers of the same variant, pooled over
    every variant (WT included, BLANK not) with >= 2 chambers. Returns (SD, degrees of
    freedom). Reported with the table; the p-values come from the bootstrap."""
    d = pd.DataFrame({"l": np.asarray(log2_value, float), "tag": np.asarray(tags)})
    d = d[d["tag"] != "BLANK"]
    n = d.groupby("tag")["l"].transform("size")
    dev = (d["l"] - d.groupby("tag")["l"].transform("mean"))[n >= 2]
    dof = int((d[n >= 2].groupby("tag").size() - 1).sum())
    return (float(np.sqrt((dev**2).sum() / dof)) if dof else np.nan), dof


def chamber_values(q, fits, rates, s_range, *, require_expressed=True):
    """The chambers that count for quantity `q`, with the value used (`value`), whether it
    is a bound (`bound`: "lower", "upper" or ""), and its log2. `s_range` = (lowest, highest)
    [S]."""
    s_min, s_max = s_range
    ok = fits[(fits["fit"] == q.fit) & fits["passes_r2_filter"]].copy()
    if require_expressed:
        ok = ok[ok["E_above_floor"].fillna(False).astype(bool)]
    if q.name == "kcat_km":
        ok = ok[(ok["km_uM"] > 0) & (ok["vmax"] > 0)]
        ok["value"] = ok["vmax"] / ok["km_uM"]
        ok["bound"] = np.where(ok["km_uM"] < s_min, "lower", "")
    elif q.name == "km":
        ok = ok[ok["km_uM"].notna()]
        ok["value"] = ok["km_uM"].clip(s_min, s_max)  # beyond the [S] range: only bounded
        ok["bound"] = np.where(ok["km_uM"] > s_max, "lower",
                               np.where(ok["km_uM"] < s_min, "upper", ""))
    else:
        # when Km is above the highest [S] the fitted Vmax is extrapolated, so the chamber's
        # rate / GFP at the highest [S] (below Vmax at any [S]) is used
        at_top = rates.loc[rates["bzp_uM"] == s_max, KEYS + ["rate_per_gfp_count"]]
        ok = ok.merge(at_top, on=KEYS, how="left")
        beyond = ok["km_uM"].isna() | (ok["km_uM"] > s_max)
        ok["value"] = np.where(beyond, ok["rate_per_gfp_count"], ok["vmax"])
        ok["bound"] = np.where(beyond, "lower", "")
        ok = ok[ok["value"] > 0]  # a log needs a positive rate
    ok["log2_value"] = np.log2(ok["value"])
    return ok


def bootstrap_p(v, ref, rng, *, steps=BOOT_STEPS, chunk=10_000):
    """Bootstrap hypothesis test of a difference of medians (Markin et al. 2021): pool the
    m variant and n reference values, draw m and n from the pool with replacement, and count
    how often |median(m) - median(n)| reaches the observed one. Iterations step up through
    `steps` while p stays small; p = (count + 1) / (N + 1)."""
    t_obs = abs(np.median(v) - np.median(ref))
    pool, m, n = np.concatenate([v, ref]), len(v), len(ref)
    count = done = 0
    p = 1.0
    for n_iter, go_on in steps:
        while done < n_iter:
            k = min(chunk, n_iter - done)
            a = np.median(pool[rng.integers(0, len(pool), (k, m))], axis=1)
            b = np.median(pool[rng.integers(0, len(pool), (k, n))], axis=1)
            count += int((np.abs(a - b) >= t_obs).sum())
            done += k
        p = (count + 1) / (done + 1)
        if go_on is None or p > go_on:
            return p
    return p


def bootstrap_p_fixed(v, ref, rng, *, steps=BOOT_STEPS, chunk=10_000):
    """Bootstrap test of a median against a fixed value `ref` (log2 units, as `v`): shift
    the variant's values so their median is `ref` (the null), draw m of them with
    replacement, and count how often |median - ref| reaches the observed one. Iterations
    step up through `steps` while p stays small; p = (count + 1) / (N + 1)."""
    t_obs = abs(np.median(v) - ref)
    null, m = v - np.median(v) + ref, len(v)
    count = done = 0
    p = 1.0
    for n_iter, go_on in steps:
        while done < n_iter:
            k = min(chunk, n_iter - done)
            a = np.median(null[rng.integers(0, m, (k, m))], axis=1)
            count += int((np.abs(a - ref) >= t_obs).sum())
            done += k
        p = (count + 1) / (done + 1)
        if go_on is None or p > go_on:
            return p
    return p


def variant_table(q, fits, rates, *, reference="WT", exclude=("WT", "BLANK"), min_replicates=2,
                  min_ref_chambers=3, p_significant=0.01, require_expressed=True, scans=SCANS,
                  steps=BOOT_STEPS, seed=0, fixed_reference=None, reference_lines=None):
    """One row per variant (not in `exclude`): replicates, fold change vs the reference,
    bootstrap p and 95 % interval. Returns (table, ref: the reference's log2 values,
    s_range).

    `fixed_reference` = {quantity name: dict(value=..., name=..., placeholder=bool)} tests
    that quantity against one value (in its unit, e.g. Km in µM) instead of reference
    chambers, e.g. {"km": dict(value=704, name="MBO-040 WT", placeholder=True)}.
    `reference_lines`, in the same form, only marks such a value on the plot (a dashed
    line at its fold change from the reference), leaving the test as it is."""
    s_range = (rates["bzp_uM"].min(), rates["bzp_uM"].max())
    ok = chamber_values(q, fits, rates, s_range, require_expressed=require_expressed)
    sd, _ = pooled_replicate_sd(ok["log2_value"], ok["tag"])
    ref = ok.loc[ok["tag"] == reference, "log2_value"].to_numpy()
    ref_label = reference
    library = ok[~ok["tag"].isin(exclude)]
    fixed = (fixed_reference or {}).get(q.name)
    if fixed is not None:   # one value, e.g. from another chip
        ref = np.array([np.log2(fixed["value"])])
        ref_label = fixed.get("name", f"{fixed['value']:g}")
    elif len(ref) < min_ref_chambers:  # e.g. WT did not express: every library chamber
        ref = library["log2_value"].to_numpy()
        ref_label = "library median"

    raw_col = {"km": "km_uM", "vmax": "vmax", "kcat_km": "value"}[q.name]
    rng = np.random.default_rng(seed)
    rows = []
    for tag, g in ok[~ok["tag"].isin(exclude)].groupby("tag"):
        n = len(g)
        tested = n >= min_replicates
        v = g["log2_value"].to_numpy()
        # the library reference leaves out the tested variant's own chambers
        ref_v = (ref if fixed is not None or ref_label == reference
                 else library.loc[library["tag"] != tag, "log2_value"].to_numpy())
        fc = np.median(v) - np.median(ref_v)
        test = (lambda: bootstrap_p_fixed(v, ref_v[0], rng, steps=steps)) if fixed is not None else \
            (lambda: bootstrap_p(v, ref_v, rng, steps=steps))
        p = test() if tested else np.nan
        boot = np.median(v[rng.integers(0, n, (10_000, n))], axis=1) - np.median(ref_v)
        lo95, hi95 = np.percentile(boot, [2.5, 97.5])
        rows.append({
            "tag": tag,
            "scan": scans.get(tag[-1], "other"),
            "wt_residue": tag[0],
            "position": int(tag[1:-1]) if tag[1:-1].isdigit() else np.nan,
            "mutant_residue": tag[-1],
            "n_chambers": n,
            f"{q.column}_fit_median": g[raw_col].median(),  # as fitted, before any bound
            f"{q.column}_median_used": 2 ** np.median(v),
            "log2_fc": fc,
            "se_log2_fc": boot.std(ddof=1),  # SD of the variant's bootstrap fold changes
            "log2_fc_ci95_low": lo95,
            "log2_fc_ci95_high": hi95,
            "p_value": p,
            "neg_log10_p": -np.log10(p),
            "significant": bool(tested and p < p_significant),
            # a limit when most of its chambers sit on a bound
            "limit": ("lower" if (g["bound"] == "lower").mean() > 0.5
                      else "upper" if (g["bound"] == "upper").mean() > 0.5 else ""),
            "tested": tested,
        })
    table = pd.DataFrame(rows).sort_values(["scan", "position"], ignore_index=True)
    table["quantity"] = q.name
    table["unit"] = q.unit
    table["reference"] = ref_label
    table["reference_median"] = 2 ** np.median(ref)
    table["reference_n_chambers"] = 0 if fixed is not None else len(ref)
    table["reference_placeholder"] = bool(fixed is not None and fixed.get("placeholder", False))
    table["replicate_sd_log2"] = sd
    table["bzp_range_uM"] = f"{s_range[0]:g}-{s_range[1]:g}"
    table.attrs.update(p_significant=p_significant, min_replicates=min_replicates,
                       reference=reference, steps=steps, fixed=fixed,
                       reference_line=(reference_lines or {}).get(q.name))
    return table, ref, s_range


def label_points(ax, d, *, top_n=4, fontsize=7):
    """Name, on each side of the reference, the `top_n` variants with the smallest p (dark
    gray if significant, light gray if not), each at the first of a few offsets that stays
    inside the axes and overlaps no earlier label, no other text and no marker, with a thin
    leader line to its point. A label with no free spot is left out."""
    from matplotlib.transforms import Bbox

    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    ax_box = ax.get_window_extent(renderer)
    near = [(12, 8), (12, -12), (-12, 8), (-12, -12), (14, 16), (-14, 16), (14, -20),
            (-14, -20), (0, 18), (0, -22)]
    far = [(dx, dy) for r in (22, 32, 44) for dx, dy in
           ((r, 0), (-r, 0), (r, r * 0.6), (-r, r * 0.6), (r, -r * 0.6), (-r, -r * 0.6),
            (0, r), (0, -r))]
    placed = [t.get_window_extent(renderer) for t in ax.texts]
    to_px = ax.transData.transform
    for row in d.itertuples():
        x, y = to_px((row.log2_fc, row.y))
        r = np.sqrt(SIZE_PER_REPLICATE * row.n_chambers) / 2 * fig.dpi / 72
        placed.append(Bbox([[x - r, y - r], [x + r, y + r]]))
    ranked = d.sort_values("neg_log10_p", ascending=False)
    top = [*ranked[ranked["log2_fc"] < 0].index[:top_n], *ranked[ranked["log2_fc"] > 0].index[:top_n]]
    for row in ranked[ranked.index.isin(top)].itertuples():
        for dx, dy in near + far:
            t = ax.annotate(row.tag, (row.log2_fc, row.y), xytext=(dx, dy), textcoords="offset points",
                            ha="left" if dx > 0 else "right" if dx < 0 else "center", va="center",
                            fontsize=fontsize, color="0.2" if row.significant else "0.5",
                            arrowprops=dict(arrowstyle="-", color="0.55", lw=0.5, shrinkA=1, shrinkB=3))
            t.update_positions(renderer)  # the text alone: the leader always touches its marker
            bb = matplotlib.text.Text.get_window_extent(t, renderer).expanded(1.05, 1.1)
            inside = (bb.x0 >= ax_box.x0 and bb.x1 <= ax_box.x1
                      and bb.y0 >= ax_box.y0 and bb.y1 <= ax_box.y1)
            if inside and not any(bb.overlaps(o) for o in placed):
                placed.append(bb)
                break
            t.remove()


def limit_text(q, s_range, s_unit="µM"):
    """{limit: legend text} for quantity `q`'s bound markers."""
    lo, hi = s_range
    if q.name == "km":
        return {"lower": f"lower limit (K$_m$ > {hi:,.0f} {s_unit})",
                "upper": f"upper limit (K$_m$ < {lo:g} {s_unit})"}
    if q.name == "kcat_km":
        return {"lower": f"lower limit (K$_m$ < {lo:g} {s_unit})"}
    return {"lower": f"K$_m$ > {hi:,.0f} {s_unit}: rate at {hi:,.0f} {s_unit}"}


def _sci(v):
    mantissa, exponent = f"{v:.2e}".split("e")
    return f"{mantissa}×10$^{{{int(exponent)}}}$"


def plot_volcano(q, table, ref, s_range, *, label_top_n=4, min_r2=0.9):
    """One panel per scan (Ala, Phe, Met), a histogram of fold changes on top, points sized
    by replicates, limits as triangles, a caption with the reference and the method."""
    p_sig = table.attrs.get("p_significant", 0.01)
    reference = table.attrs.get("reference", "WT")
    ref_label = table["reference"].iloc[0]
    tested = table[table["tested"]].copy()
    tested["y"] = tested["neg_log10_p"]
    y_top = max(tested["y"].max(), -np.log10(p_sig)) * 1.15
    scans = [s for s in SCANS.values() if s in set(tested["scan"])]
    with plt.rc_context({"font.family": "Arial", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, len(scans), figsize=(3.4 * len(scans), 4.6), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1, 3.2], hspace=0.06, wspace=0.12),
                                 squeeze=False)
        lim = np.ceil(tested["log2_fc"].abs().max() * 2) / 2 + 0.25
        bins = np.arange(-lim, lim + 0.25, 0.25)
        hist_max = 0
        for k, scan in enumerate(scans):
            d = tested[tested["scan"] == scan]
            ax_h, ax = axes[0, k], axes[1, k]
            sig = d["significant"]
            counts, _, _ = ax_h.hist([d.loc[~sig, "log2_fc"], d.loc[sig, "log2_fc"]], bins=bins,
                                     stacked=True, color=[COLORS["not significant"], COLORS["significant"]],
                                     edgecolor=COLORS["edge"], linewidth=0.5)
            hist_max = max(hist_max, counts[-1].max())
            ax_h.set_title(f"{scan}  (n = {len(d)})", fontsize=10.5)
            ax_h.tick_params(labelbottom=False)
            if k > 0:
                ax_h.tick_params(labelleft=False)
            for row in d.itertuples():
                ax.scatter([row.log2_fc], [row.y], s=SIZE_PER_REPLICATE * row.n_chambers,
                           marker=q.markers[row.limit],
                           color=COLORS["significant" if row.significant else "not significant"],
                           edgecolor=COLORS["edge"], linewidth=0.4, zorder=3 + row.significant,
                           clip_on=False)
            ax.axvline(0, color="0.5", lw=0.6, zorder=1)
            ax.axhline(-np.log10(p_sig), color="0.3", lw=0.8, ls="--", zorder=1)
            line = table.attrs.get("reference_line")
            if line is not None:   # a value from elsewhere (e.g. another chip's WT), marked only
                x_line = np.log2(line["value"]) - np.median(ref)
                ax.axvline(x_line, color="#B9770E", lw=1.2, ls="--", zorder=2)
                if k == 0:
                    ax.annotate(f"{line['name']} {line['value']:,.0f} {q.unit}"
                                + (" (PLACEHOLDER)" if line.get("placeholder") else ""),
                                (x_line, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -3),
                                textcoords="offset points", ha="left", va="top", fontsize=7.5,
                                color="#B9770E", weight="bold")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(0, y_top)
            ax.set_xlabel(f"log$_2$ fold change in {q.label}\n(variant / {ref_label})")
            if k == 0:
                ax.set_ylabel("−log$_{10}$ p")
                ax.annotate(f"p = {p_sig:g}", (-lim, -np.log10(p_sig)), xytext=(3, -3),
                            textcoords="offset points", va="top", fontsize=7.5, color="0.3")
            else:
                ax.tick_params(labelleft=False)
            ax.sharey(axes[1, 0])
        for ax_h in axes[0]:
            ax_h.set_ylim(0, hist_max * 1.1)
        axes[0, 0].set_ylabel("variants")
        fig.canvas.draw()  # labels need the final layout to check overlaps
        for k, scan in enumerate(scans):
            label_points(axes[1, k], tested[tested["scan"] == scan], top_n=label_top_n)

        size_handles = [plt.scatter([], [], s=SIZE_PER_REPLICATE * n, color=COLORS["not significant"],
                                    edgecolor=COLORS["edge"], linewidth=0.4, label=f"{n}")
                        for n in SIZE_LEGEND]
        shape_handles = [Line2D([], [], ls="", marker=q.markers[kind], ms=6, mfc="white", mec="0.15",
                                label=text) for kind, text in limit_text(q, s_range).items()]
        shape_handles.append(Line2D([], [], ls="", marker="o", ms=6, mfc=COLORS["significant"],
                                    mec="0.15", label=f"p < {p_sig:g}"))
        last = axes[1, -1]
        leg = last.legend(handles=size_handles, title="replicates", loc="upper left",
                          bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8, title_fontsize=8.5,
                          labelspacing=1.1, borderaxespad=0)
        last.add_artist(leg)
        last.legend(handles=shape_handles, loc="lower left", bbox_to_anchor=(1.02, 0.0),
                    frameon=False, fontsize=8, borderaxespad=0)

        ref_value = 2 ** np.median(ref)
        lo, hi = s_range
        value_text = {"km": f"K$_m$ {ref_value:,.0f} µM",
                      "vmax": f"v$_{{max}}$/GFP {_sci(ref_value)} {q.unit}",
                      "kcat_km": f"k$_{{cat}}$/K$_m$ {_sci(ref_value)} {q.unit}"}[q.name]
        how = {"km": f"raw MM fit r² ≥ {min_r2:g}; K$_m$ clipped to {lo:g}–{hi:,.0f} µM",
               "vmax": f"per-GFP MM fit r² ≥ {min_r2:g}; K$_m$ > {hi:,.0f} µM: rate / GFP at {hi:,.0f} µM "
                       "(lower bound)",
               "kcat_km": f"per-GFP MM fit r² ≥ {min_r2:g}; k$_{{cat}}$/K$_m$ = (v$_{{max}}$/GFP) / K$_m$"}[q.name]
        n_ref = len(ref)
        fixed = table.attrs.get("fixed")
        placeholder = bool(table["reference_placeholder"].iloc[0]) if "reference_placeholder" in table else False
        if fixed is not None:
            ref_text = (f"Reference: {ref_label}, {value_text}, a fixed value"
                        + (" (PLACEHOLDER: not measured on this chip)." if placeholder else "."))
            p_text = ("p: bootstrap test of the variant's median against the fixed reference (its chambers "
                      "shifted to the reference and resampled).")
        else:
            ref_text = (f"Reference: {reference}, {value_text} ({n_ref} chambers)." if ref_label == reference else
                        f"Reference: library median, {value_text} ({n_ref:,} chambers; {reference} did not express).")
            p_text = "p: bootstrap test of the difference of medians, variant and reference pooled (Markin et al. 2021)."
        lines = [ref_text,
                 f"Chambers: GFP above the floor; {how}. Variants with ≥ {table.attrs.get('min_replicates', 2)} chambers.",
                 p_text]
        if placeholder:   # a tag nobody can miss, top right
            fig.text(0.99, 1.0, f"PLACEHOLDER reference: {ref_label}, {value_text}", ha="right", va="bottom",
                     fontsize=9, weight="bold", bbox=dict(facecolor="#FBE7C6", edgecolor="#B9770E", lw=0.8, pad=3))
        # one line each, wrapped to the figure's width (about 15 characters per inch at 7.5 pt)
        width = int(fig.get_size_inches()[0] * 15)
        fig.text(0.01, -0.02, "\n".join(textwrap.fill(line, width) for line in lines), fontsize=7.5,
                 color="0.35", va="top", linespacing=1.4)
    return fig
