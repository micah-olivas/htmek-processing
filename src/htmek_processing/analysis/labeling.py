"""Cy5 labeling against the enzyme on each button: the Cy5 button sum on the post-labeling scan (minus the Cy5
background scan) against the bound GFP (the button sum on a GFP scan, e.g. the last SDS wash, minus the GFP background
scan), and their ratio."""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["mark_row", "mark_col"]
CONTROLS = {"WT": "#2E8B57", "N41A": "#1F4E79", "BLANK": "0.45"}
HALVES = {"left": "#C0392B", "right": "#2E86C1"}


def labeling_table(cy5, gfp, *, gfp_scan, cy5_scan="post-labeling", value="button_sum_minus_bg", n_sd=3.0, n_cols=32):
    """One row per button: tag, cy5 and gfp (`value` on `cy5_scan` and `gfp_scan`; tables with a `scan` column, or one
    row per button), cy5_over_gfp, the chip half (left: the first n_cols / 2 columns), and whether each readout is above
    its BLANK floor (median + `n_sd` SD; the floors are in .attrs)."""
    pick = lambda d, s: (d[d["scan"] == s] if "scan" in d and s is not None else d).set_index(KEYS)
    c, g = pick(cy5, cy5_scan), pick(gfp, gfp_scan)
    t = c[["tag", value]].rename(columns={value: "cy5"}).join(g[value].rename("gfp"), how="inner").reset_index()
    blank = t[t["tag"] == "BLANK"]
    floors = {k: float(blank[k].median() + n_sd * blank[k].std()) for k in ("cy5", "gfp")}
    t["cy5_over_gfp"] = t["cy5"] / t["gfp"]
    t["half"] = np.where(t["mark_col"] < n_cols // 2, "left", "right")
    t["gfp_above_floor"], t["cy5_above_floor"] = t["gfp"] > floors["gfp"], t["cy5"] > floors["cy5"]
    t.attrs.update(floors=floors, gfp_scan=gfp_scan, cy5_scan=cy5_scan)
    return t


def labeling_figure(t, *, title="", gfp_label=None, n_cols=32, axes=None, ratio_range=None):
    """Cy5 against GFP per button (log-log; library by chip half, the controls marked; dashed: the GFP floor, dotted:
    the Cy5 floor, noted instead when one is <= 0), and the Cy5 / GFP ratio of library buttons above the GFP floor
    with Cy5 > 0, per half (dashed: each half's median; ticks: WT, N41A). Returns the figure."""
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    gf, cf = t.attrs["floors"]["gfp"], t.attrs["floors"]["cy5"]
    lib = t[~t["tag"].isin(CONTROLS)]
    names = {"left": f"left (cols 1-{n_cols // 2})", "right": f"right (cols {n_cols // 2 + 1}-{n_cols})"}
    if axes is None:
        fig, (ax, axh) = plt.subplots(1, 2, figsize=(14, 5.0), gridspec_kw=dict(width_ratios=[1.1, 1], wspace=0.22))
    else:
        ax, axh = axes
        fig = ax.figure
    for h, col in HALVES.items():
        x = lib[(lib["half"] == h) & (lib["gfp"] > 0) & (lib["cy5"] > 0)]
        ax.scatter(x["gfp"], x["cy5"], s=6, lw=0, alpha=0.45, color=col, label=f"library, {names[h]} ({len(x)})")
    for tag, col in CONTROLS.items():
        x = t[(t["tag"] == tag) & (t["gfp"] > 0) & (t["cy5"] > 0)]
        ax.scatter(x["gfp"], x["cy5"], s=28, color=col, ec="k", lw=0.4, marker="D" if tag == "BLANK" else "o",
                   label=f"{tag} ({len(x)})", zorder=3)
    lines = [title, f"Spearman ρ {spearmanr(lib['gfp'], lib['cy5'])[0]:.2f} (library) · dashed: GFP floor, dotted: Cy5 floor (BLANK + 3 SD) · "
                    "points with both > 0"]
    notes = []
    if gf > 0:
        ax.axvline(gf, color="k", ls="--", lw=0.8)
    else:
        notes.append(f"GFP floor ≤ 0 ({gf:,.0f})")
    if cf > 0:
        ax.axhline(cf, color="k", ls=":", lw=0.8)
    else:
        notes.append(f"Cy5 floor ≤ 0 ({cf:,.0f})")
    neg = int(((lib["gfp"] > max(gf, 0)) & (lib["cy5"] <= 0)).sum())
    if neg:
        notes.append(f"{neg} library buttons above the GFP floor have Cy5 ≤ 0")
    if notes:
        lines.append("; ".join(notes))
    ax.set(xscale="log", yscale="log", xlabel=gfp_label or f"GFP button sum, {t.attrs['gfp_scan']} − GFP background scan (RFU)",
           ylabel=f"Cy5 button sum, {t.attrs['cy5_scan']} − Cy5 background scan (RFU)")
    ax.set_title("\n".join(l for l in lines if l), loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    above = lib[lib["gfp"] > max(gf, 0)]
    rl = above[above["cy5_over_gfp"] > 0]
    if len(rl):
        lo, hi = ratio_range or (rl["cy5_over_gfp"].quantile(0.005), rl["cy5_over_gfp"].quantile(0.995))
        bins = np.logspace(np.log10(lo), np.log10(hi), 50)
        for h, col in HALVES.items():
            x = rl.loc[rl["half"] == h, "cy5_over_gfp"]
            if x.empty:
                continue
            axh.hist(x.clip(bins[0], bins[-1]), bins=bins, histtype="stepfilled", alpha=0.45, color=col,
                     label=f"{names[h]}: {len(x)}, median {x.median():.3g}")
            axh.axvline(x.median(), color=col, lw=1.2, ls="--")
        for tag in ("WT", "N41A"):
            x = t[(t["tag"] == tag) & (t["gfp"] > max(gf, 0)) & (t["cy5"] > 0)]
            for v in x["cy5_over_gfp"]:
                axh.axvline(v, color=CONTROLS[tag], lw=1.3, ymax=0.12)
        axh.set_xscale("log")
        axh.legend(frameon=False, fontsize=8, loc="upper right")
    axh.set(xlabel=f"Cy5 / GFP ({t.attrs['gfp_scan']})", ylabel="library buttons")
    axh.set_title(f"Cy5 / GFP: library buttons above the GFP floor with Cy5 > 0 ({len(rl)} of {len(above)})\n"
                  "dashed: each half's median · ticks: WT (green), N41A (blue)", loc="left", fontsize=9)
    if axes is None:
        fig.tight_layout()
    return fig


def labeling_states_figure(cy5, gfp, scans, *, title="", cy5_scan="post-labeling", n_cols=32, **table_kw):
    """labeling_figure for several GFP scans of one chip (e.g. post-flush, SDS wash 4, SDS clean), a row each, to compare
    which state of the bound GFP tracks the Cy5 labeling: the scatter and the Cy5 / GFP ratio, against that scan.
    Returns (figure, {scan: labeling_table})."""
    import matplotlib.pyplot as plt

    tables = {s: labeling_table(cy5, gfp, gfp_scan=s, cy5_scan=cy5_scan, n_cols=n_cols, **table_kw) for s in scans}
    fig, axes = plt.subplots(len(scans), 2, figsize=(14, 5.0 * len(scans)), squeeze=False, sharex="col",
                             gridspec_kw=dict(width_ratios=[1.1, 1], wspace=0.22, hspace=0.55))
    # one ratio range over every state (the histograms share their x axis and their bins)
    r = pd.concat([t.loc[~t["tag"].isin(CONTROLS) & t["gfp_above_floor"] & (t["cy5_over_gfp"] > 0), "cy5_over_gfp"] for t in tables.values()])
    rng = (r.quantile(0.005), r.quantile(0.995)) if len(r) else None
    for (s, t), row in zip(tables.items(), axes):
        labeling_figure(t, title=f"GFP: {s}", n_cols=n_cols, axes=row, ratio_range=rng)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    return fig, tables


CLICK_GROUPS = {"library, GFP above floor": "#8E44AD", "library, GFP below floor": "#C39BD3", "WT": "#2E8B57", "N41A": "#1F4E79",
                "BLANK": "0.45"}
# the button's own median: with the buttons closed for imaging, the ring inside the rim sits in the chamber's dye, so
# button sum − ring tracks the dye leaving the solution, not dye on the button (MBO-056, 2026-11-17)
CLICK_METRICS = [("button_median", "button median (RFU)", 1), ("button_annulus_median", "ring around the button, in solution (RFU)", 1),
                 ("dna_median", "DNA chamber median (RFU)", 1), ("reaction_median", "reaction chamber median (RFU)", 1)]


def click_changes(click, enz=None):
    """The click time course table with d_<metric>: each chamber's change from its series' pass 0 (before the buttons
    first opened: removes the tile shading and the chip's own Cy5), and `group` (controls; library above / below the
    GFP floor of `enz`: mark_row, mark_col, gfp, expressing)."""
    d = click.copy()
    if enz is not None:
        d = d.drop(columns=[c for c in ("gfp", "expressing") if c in d]).merge(enz, on=KEYS, how="left")
    for col, _, _ in CLICK_METRICS:
        if col in d:
            d[f"d_{col}"] = d[col] - d.sort_values("pass").groupby(["series"] + KEYS)[col].transform("first")
    expressing = d["expressing"].fillna(False).astype(bool) if "expressing" in d else pd.Series(False, index=d.index)
    d["group"] = np.where(d["tag"].isin(["WT", "N41A", "BLANK"]), d["tag"],
                          np.where(expressing, "library, GFP above floor", "library, GFP below floor"))
    return d


def click_figures(click, enz=None, *, title="", n_rows=56, n_cols=32):
    """QC for the click labeling time courses: {"3e_click_timecourse": Cy5 vs the buttons' cumulative open time, a
    row per region (the button's median, the ring around it in solution, DNA chamber, reaction chamber; change from
    pass 0), a column per series, median per group with the IQR; "3e_click_last_pass_vs_gfp": the first series' last
    pass, the button median's change vs the button GFP, and over the chip}."""
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    d = click_changes(click, enz)
    metrics = [(f"d_{c}", f"{lab}, change from pass 0", sc) for c, lab, sc in CLICK_METRICS if f"d_{c}" in d]
    series = list(dict.fromkeys(d["series"]))
    figs = {}
    fig, axes = plt.subplots(len(metrics), len(series), figsize=(6.2 * len(series), 3.6 * len(metrics)), sharey="row", sharex=True, squeeze=False)
    for j, s in enumerate(series):
        e = d[d["series"] == s]
        for i, (col, lab, sc) in enumerate(metrics):
            ax = axes[i, j]
            for g, c in CLICK_GROUPS.items():
                q = e[e["group"] == g].groupby("pass").agg(t=("buttons_open_s", "first"), m=(col, "median"),
                                                         lo=(col, lambda v: v.quantile(.25)), hi=(col, lambda v: v.quantile(.75)), n=(col, "size"))
                if q.empty:
                    continue
                x = q["t"] / 60
                ax.plot(x, q["m"] / sc, "o-", ms=3, lw=1.2, color=c, label=f"{g} ({int(q['n'].iloc[0])})")
                ax.fill_between(x, q["lo"] / sc, q["hi"] / sc, color=c, alpha=0.15, lw=0)
            ax.axhline(0, color="0.6", lw=0.7)
            ax.set_xscale("symlog", linthresh=1)
            if j == 0:
                ax.set_ylabel(lab, fontsize=9)
            if i == 0:
                ax.set_title(s, loc="left", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("buttons open, cumulative (min; linear below 1)")
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    H = fig.get_size_inches()[1]
    fig.subplots_adjust(top=1 - 0.55 / H)
    fig.suptitle(f"{title}: Cy5 vs the time the buttons were open; median per group, band IQR".strip(": "), x=0.01, y=1 - 0.1 / H,
                 ha="left", va="top", fontsize=11)
    figs["3e_click_timecourse"] = fig

    s0 = d[d["series"] == series[0]]
    s1 = s0[s0["pass"] == s0["pass"].max()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    if "gfp" in s1:
        for g, c in CLICK_GROUPS.items():
            e = s1[s1["group"] == g]
            ax.scatter(e["gfp"].clip(lower=1e3), e["d_button_median"], s=7 if g.startswith("library") else 30, color=c, lw=0, alpha=0.6, label=g)
        lib = s1[s1["group"] == "library, GFP above floor"]
        rho = spearmanr(lib["gfp"], lib["d_button_median"])[0] if len(lib) > 2 else np.nan
        ax.set(xscale="log", xlabel="button GFP (button_sum_net; RFU, clipped at 10$^3$)")
        ax.set_title(f"{series[0]}, last pass (buttons open {s1['buttons_open_s'].iloc[0] / 60:.0f} min in all)\n"
                     f"Spearman ρ {rho:.2f} (library above the GFP floor)", loc="left", fontsize=9)
        ax.legend(frameon=False, fontsize=7.5)
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_ylabel("button median, change from pass 0 by the last pass (RFU)")
    ax = axes[1]
    g = s1.pivot_table(index="mark_row", columns="mark_col", values="d_button_median").reindex(index=range(n_rows), columns=range(n_cols))
    lo, hi = np.nanpercentile(g, [2, 98])
    m = ax.imshow(g, cmap="RdBu_r", vmin=-max(abs(lo), abs(hi)), vmax=max(abs(lo), abs(hi)))
    ax.set_title("button median, change from pass 0 by the last pass (RFU)", loc="left", fontsize=9)
    ax.set(xlabel="column", ylabel="row")
    fig.colorbar(m, ax=ax, shrink=0.8)
    fig.tight_layout()
    figs["3e_click_last_pass_vs_gfp"] = fig
    return figs
