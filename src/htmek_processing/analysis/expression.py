"""Which chambers and variants expressed, from the in-solution expression time course (GFP over each dumbbell)."""

import numpy as np
import pandas as pd

KEYS = ["mark_row", "mark_col"]


def expression_calls(expression, value="dna_chamber_sum_net", time="minutes", *, blank="BLANK", n_sd=3.0, floor="mad",
                     controls=("BLANK", "WT", "N41A")):
    """Call each chamber expressed or not from its expression time course, and each variant from its chambers.

    `expression` has one row per chamber and time point: mark_row, mark_col, tag, `time` and `value` (default: the GFP
    in the DNA chamber, where it is expressed, minus background). A chamber's gain is `value` at its last time point
    minus its first. The floor is the `blank` chambers' median gain + `n_sd` robust SDs (`floor="mad"`: 1.4826 x the
    median absolute deviation, so a few contaminated BLANKs don't raise it) or SDs (`floor="sd"`); a chamber expressed
    when its gain is above it. A variant did not express when
    none of its chambers did (`controls` are reported per chamber but not called as variants).

    Returns (chambers, variants, floor): one row per chamber (keys, tag, first, last, gain, expressed) and one per
    variant (tag, chambers, chambers_expressed, expressed)."""
    t = expression.sort_values(time)
    g = t.groupby(KEYS + ["tag"])[value]
    ch = pd.DataFrame({"first": g.first(), "last": g.last()}).reset_index()
    ch["gain"] = ch["last"] - ch["first"]
    b = ch.loc[ch["tag"] == blank, "gain"]
    spread = 1.4826 * np.median(np.abs(b - b.median())) if floor == "mad" else b.std()
    floor = float(b.median() + n_sd * spread) if len(b) > 1 else np.nan
    ch["expressed"] = ch["gain"] > floor
    lib = ch[~ch["tag"].isin(controls)]
    var = (lib.groupby("tag").agg(chambers=("expressed", "size"), chambers_expressed=("expressed", "sum"))
           .assign(expressed=lambda v: v["chambers_expressed"] > 0).reset_index())
    return ch, var, floor


def blank_leakage(chambers, floor, *, blank="BLANK", n_rows=56, n_cols=32):
    """GFP in the BLANK chambers (no DNA printed): leakage from their neighbors, or contamination.

    `chambers` and `floor` are expression_calls' outputs. For each BLANK: its gain, its neighbors' (the 8 around it,
    BLANKs left out) median gain, the same row's two neighbors' mean (they share a flow line), and the ratio gain /
    neighbors. A BLANK above the floor is flagged `contaminated` (its own GFP, e.g. DNA carried into it at printing).
    Leakage, GFP reaching a BLANK from its neighbors, would make the clean BLANKs' gain follow their neighbors': the
    summary gives the clean BLANKs' median ratio (the leak fraction, at most), the through-origin slope of gain vs
    neighbors, and its Spearman correlation.

    Returns (per-BLANK table, summary dict)."""
    g = chambers.set_index(KEYS)["gain"].unstack().reindex(index=range(n_rows), columns=range(n_cols)).to_numpy()
    tags = chambers.set_index(KEYS)["tag"].unstack().reindex(index=range(n_rows), columns=range(n_cols)).to_numpy()
    rows = []
    for r, c in np.argwhere(tags == blank):
        around = [g[i, j] for i in range(r - 1, r + 2) for j in range(c - 1, c + 2)
                  if (i, j) != (r, c) and 0 <= i < n_rows and 0 <= j < n_cols and tags[i, j] != blank and np.isfinite(g[i, j])]
        row = [g[r, j] for j in (c - 1, c + 1) if 0 <= j < n_cols and tags[r, j] != blank and np.isfinite(g[r, j])]
        nb = float(np.median(around)) if around else np.nan
        rows.append({"mark_row": int(r), "mark_col": int(c), "gain": g[r, c], "neighbors_gain": nb,
                     "row_neighbors_gain": float(np.mean(row)) if row else np.nan, "ratio": g[r, c] / nb if nb else np.nan})
    t = pd.DataFrame(rows)
    t["contaminated"] = t["gain"] > floor
    clean = t[~t["contaminated"] & t["neighbors_gain"].notna()]
    slope = float((clean["gain"] * clean["neighbors_gain"]).sum() / (clean["neighbors_gain"] ** 2).sum()) if len(clean) else np.nan
    summary = {"blanks": len(t), "contaminated": int(t["contaminated"].sum()), "clean_median_ratio": float(clean["ratio"].median()),
               "clean_slope": slope, "clean_spearman": float(clean["gain"].corr(clean["neighbors_gain"], method="spearman"))}
    return t, summary


def dna_chamber_diffusion(nadh, *, time="minutes", reaction="chamber_ROI_median", dna="dna_chamber_median", per=("run",),
                          blank="BLANK", n_sd=3.0):
    """Signal in the DNA chambers during an assay. The neck is closed, so product reaches a DNA chamber only by
    diffusing under it.

    `nadh` is the process notebook's per-pass table, with `dna` (the DNA chamber's median at every pass); passes not
    flagged `complete_pass` are left out. Per chamber and run (`per`): the least-squares slope over time of the DNA
    chamber's signal and of the reaction chamber's, and their ratio. A DNA chamber is `rising` when its slope is above
    the BLANK chambers' median + `n_sd` robust SDs (1.4826 x MAD) for that run.

    Returns (per-chamber table, per-run summary: floor, chambers, rising, the library's median ratio)."""
    if dna not in nadh:
        raise KeyError(f"no {dna} column: the assay was quantified without the DNA chambers (1-Process, quantify_assay)")
    d = nadh[nadh["complete_pass"]] if "complete_pass" in nadh else nadh
    per = [c for c in per if c in d]

    def slope(x, y):
        ok = np.isfinite(x) & np.isfinite(y)
        return np.polyfit(x[ok], y[ok], 1)[0] if ok.sum() >= 3 else np.nan

    groups = per + KEYS + ["tag"]
    rows = [{**dict(zip(groups, k)), "dna_slope": slope(g[time].to_numpy(float), g[dna].to_numpy(float)),
             "reaction_slope": slope(g[time].to_numpy(float), g[reaction].to_numpy(float))}
            for k, g in d.groupby(groups)]
    t = pd.DataFrame(rows)
    t["ratio"] = t["dna_slope"] / t["reaction_slope"]
    t["rising"] = False
    summary = []
    for k, g in (t.groupby(per) if per else [((), t)]):
        b = g.loc[g["tag"] == blank, "dna_slope"].dropna()
        floor = b.median() + n_sd * 1.4826 * (b - b.median()).abs().median() if len(b) > 1 else np.nan
        t.loc[g.index, "rising"] = g["dna_slope"] > floor
        k = k if isinstance(k, tuple) else (k,)
        summary.append({**dict(zip(per, k)), "floor": floor, "chambers": len(g), "rising": int((g["dna_slope"] > floor).sum()),
                        "median_ratio": float(g.loc[g["tag"] != blank, "ratio"].median())})
    return t, pd.DataFrame(summary)


def diffusion_figure(table, summary, *, run=None, runs=None, per="run", blank="BLANK", controls=("WT", "N41A"), n_rows=56, n_cols=32):
    """QC for dna_chamber_diffusion, a row per run (`runs`, or one `run`; default all): each chamber's DNA-chamber slope
    against its reaction-chamber slope (RFU/min) with the BLANK floor, and the DNA-chamber slope over the chip, rising
    chambers outlined."""
    import matplotlib.pyplot as plt

    from .plots import set_plot_style
    from ..processing.layout import display_tag

    set_plot_style()
    runs = [run] if run is not None else list(runs) if runs is not None else (list(summary[per]) if per in summary else [None])
    fig, axes = plt.subplots(len(runs), 2, figsize=(11, 3.9 * len(runs)), gridspec_kw=dict(width_ratios=[1.5, 1]), squeeze=False)
    for (ax, axm), rn in zip(axes, runs):
        t = table[table[per] == rn] if rn is not None else table
        floor = float(summary.loc[summary[per] == rn, "floor"].iloc[0]) if rn is not None else float(summary["floor"].iloc[0])
        name = f"{rn:g} µM" if isinstance(rn, (int, float, np.number)) else str(rn or "")
        lib = ~t["tag"].isin([*controls, blank])
        styles = [("library", lib, "#8E44AD", 6), *[(c, t["tag"] == c, col, 14) for c, col in zip(controls, ("#2E8B57", "#1F4E79"))],
                  (blank, t["tag"] == blank, "0.45", 14)]
        for nm, m, color, size in styles:
            ax.scatter(t.loc[m, "reaction_slope"], t.loc[m, "dna_slope"], s=size, color=color, alpha=0.6 if nm == "library" else 0.9,
                       lw=0, label=display_tag(nm) if nm in controls else nm)
        r = t[t["rising"]]
        ax.scatter(r["reaction_slope"], r["dna_slope"], s=26, facecolor="none", edgecolor="k", lw=0.7, label=f"rising ({len(r)})")
        ax.axhline(floor, color="k", lw=0.9, ls="--", label="BLANK median + 3 robust SD")
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set(xlabel="reaction chamber slope (RFU/min)", ylabel="DNA chamber slope (RFU/min)")
        ax.set_title(name, loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        g = t.set_index(KEYS)["dna_slope"].unstack().reindex(index=range(n_rows), columns=range(n_cols)).to_numpy()
        lim = np.nanpercentile(np.abs(g), 99) or 1.0
        im = axm.imshow(g, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        axm.scatter(r["mark_col"], r["mark_row"], s=14, facecolor="none", edgecolor="k", lw=0.6)
        fig.colorbar(im, ax=axm, fraction=0.05, pad=0.03, label="DNA chamber slope (RFU/min)")
        axm.set_xticks([0, n_cols // 2 - 1, n_cols - 1], ["1", str(n_cols // 2), str(n_cols)])
        axm.set_yticks([0, n_rows // 2 - 1, n_rows - 1], ["1", str(n_rows // 2), str(n_rows)])
        axm.set(xlabel="column", ylabel="row")
        axm.set_title("DNA chamber slope, rising outlined", fontsize=9)
    fig.tight_layout()
    return fig
