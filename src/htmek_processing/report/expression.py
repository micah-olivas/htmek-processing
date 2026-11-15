"""The expression report: which variants did not express, from the in-solution expression time course (GFP over each
dumbbell; analysis.expression.expression_calls). One PDF: a summary page, every variant's chambers by residue
position, and each non-expressing variant's time courses against the controls."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["mark_row", "mark_col"]
CONTROLS = ("WT", "N41A", "BLANK")
COLORS = {"WT": "#2E8B57", "N41A": "#1F4E79", "BLANK": "0.55", "library": "#8E44AD", "not": "#D35400", "yes": "#7FB77E"}
PAGE = (11, 8.5)


def _position(tag):
    import re

    m = re.search(r"\d+", str(tag))
    return int(m.group()) if m else -1


def _frame(fig, title, experiment):
    fig.text(0.04, 0.955, title, fontsize=15, weight="bold", va="top")
    fig.text(0.96, 0.955, experiment, fontsize=9, color="0.4", ha="right", va="top")
    fig.text(0.96, 0.02, f"expression report · built {date.today():%Y-%m-%d}", fontsize=7, color="0.5", ha="right")


def expression_report(path, expression, chambers, variants, floor, *, value="dna_chamber_sum_net", time="minutes",
                      experiment="", n_rows=56, n_cols=32, per_page=12):
    """Write the expression report to `path` (PDF). `expression` is the time-course table (one row per chamber and time
    point); `chambers`, `variants`, `floor` are expression_calls' outputs. Pages:

      1. summary: the floor and the counts, each group's gain against the floor, the chip (expressed / not), and the
         variants that did not express (no chamber above the floor);
      2. the BLANK chambers: leakage (does their GFP follow their neighbors'?) and contamination (BLANKs above the
         floor; analysis.expression.blank_leakage);
      3. every variant's chambers' gains in residue order (filled: above the floor), and the variants where only some
         chambers expressed;
      3+. each non-expressing variant's chambers over time, against the WT and BLANK chambers' ranges.

    Returns the path."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import ListedColormap

    import textwrap

    from ..analysis.plots import set_plot_style
    from ..processing.layout import display_tag

    set_plot_style()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ch, var = chambers.copy(), variants.copy()
    lib = ch[~ch["tag"].isin(CONTROLS)]
    none = var[~var["expressed"]].assign(pos=lambda t: t["tag"].map(_position)).sort_values("pos")
    partial = var[var["expressed"] & (var["chambers_expressed"] < var["chambers"])].assign(
        frac=lambda t: t["chambers_expressed"] / t["chambers"], pos=lambda t: t["tag"].map(_position)).sort_values(["frac", "pos"])
    ctrl = ch[ch["tag"].isin(CONTROLS)].groupby("tag")["expressed"].agg(["sum", "size"]).reindex(list(CONTROLS))
    unit, scale = "10$^6$ RFU", 1e6

    with PdfPages(path) as pdf:
        # ---- 1. summary
        fig = plt.figure(figsize=PAGE)
        _frame(fig, "Expression: variants that did not express", experiment)
        lines = [f"A chamber expressed when its GFP gain over the time course (in its DNA chamber, minus background; last time point minus "
                 f"first) is above the BLANKs' median + 3 robust SDs (1.4826 x MAD): {floor / scale:.3f} {unit}.",
                 f"{int(ch['expressed'].sum()):,} of {len(ch):,} chambers expressed; controls: "
                 + ", ".join(f"{t} {int(r['sum'])} / {int(r['size'])}" for t, r in ctrl.dropna().iterrows()) + ".",
                 f"{len(none)} of {len(var)} variants had no chamber above the floor; {len(partial)} more had some chambers below it."]
        y = 0.905
        for ln in lines:
            wrapped = textwrap.fill(ln, 150)
            fig.text(0.04, y, wrapped, fontsize=10, va="top")
            y -= 0.03 * (wrapped.count("\n") + 1)
        if "WT" in ctrl.dropna().index and ctrl.loc["WT", "sum"] < ctrl.loc["WT", "size"] / 2:
            fig.text(0.04, 0.815, "Fewer than half the WT chambers clear the floor: check the time course before trusting these calls.",
                     fontsize=10, color=COLORS["not"], weight="bold", va="top")
        ax = fig.add_axes([0.06, 0.12, 0.42, 0.6])
        lo, hi = np.nanpercentile(ch["gain"], [0.5, 99.5]) / scale
        bins = np.linspace(min(lo, floor / scale * 1.2), hi, 60)
        groups = {"library": ~ch["tag"].isin(CONTROLS), **{t: ch["tag"] == t for t in CONTROLS}}
        for name, m in groups.items():
            ax.hist((ch.loc[m, "gain"] / scale).clip(bins[0], bins[-1]), bins=bins, density=True, color=COLORS[name],
                    histtype="stepfilled" if name == "library" else "step", alpha=0.45 if name == "library" else 1, lw=1.3,
                    label=f"{name} ({int(m.sum())})")
        ax.axvline(floor / scale, color="k", lw=1, ls="--", label="floor: BLANK median + 3 robust SD")
        ax.set(xlabel=f"GFP gained in the DNA chamber over the time course ({unit})", ylabel="density")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax = fig.add_axes([0.52, 0.12, 0.17, 0.6])
        grid = ch.set_index(KEYS)["expressed"].astype(float).unstack().reindex(index=range(n_rows), columns=range(n_cols))
        ax.imshow(grid.to_numpy(), cmap=ListedColormap([COLORS["not"], COLORS["yes"]]), vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        ax.set_title("expressed (green) / not (orange)", fontsize=9)
        ax.set_xticks([0, n_cols // 2 - 1, n_cols - 1], ["1", str(n_cols // 2), str(n_cols)])
        ax.set_yticks([0, n_rows // 2 - 1, n_rows - 1], ["1", str(n_rows // 2), str(n_rows)])
        ax.set(xlabel="column", ylabel="row")
        ax = fig.add_axes([0.73, 0.12, 0.23, 0.6])
        ax.axis("off")
        ax.set_title(f"did not express ({len(none)})", fontsize=10, loc="left", weight="bold")
        if len(none):
            rows = [[display_tag(r.tag), f"{int(r.chambers)}", f"{lib.loc[lib['tag'] == r.tag, 'gain'].max() / scale:.2f}"]
                    for r in none.itertuples()][:28]
            t = ax.table(cellText=rows, colLabels=["variant", "chambers", f"best gain\n({unit})"], loc="upper left", cellLoc="center",
                         colWidths=[0.34, 0.28, 0.38])
            t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.25)
            for (r, c), cell in t.get_celld().items():
                cell.set_edgecolor("0.85")
                if r == 0:
                    cell.set_facecolor("#EEF1F4"); cell.set_text_props(weight="bold"); cell.set_height(cell.get_height() * 1.8)
        else:
            ax.text(0, 0.95, "every variant had at least one chamber above the floor", fontsize=9, va="top")
        pdf.savefig(fig); plt.close(fig)

        # ---- 2. the BLANKs: leakage and contamination
        from ..analysis.expression import blank_leakage

        bt, bs = blank_leakage(ch, floor, n_rows=n_rows, n_cols=n_cols)
        fig = plt.figure(figsize=PAGE)
        _frame(fig, "BLANK chambers: leakage and contamination", experiment)
        txt = (f"{bs['blanks']} BLANK chambers (no DNA printed). {bs['contaminated']} above the floor: contaminated, their own GFP "
               f"(e.g. DNA carried in at printing). The other {bs['blanks'] - bs['contaminated']} gain a median "
               f"{100 * bs['clean_median_ratio']:.1f}% of their neighbors' gain, with no relation to it (Spearman {bs['clean_spearman']:.2f}; "
               f"through-origin slope {bs['clean_slope']:.3f}): leakage from the neighbors is at most that.")
        fig.text(0.04, 0.905, textwrap.fill(txt, 150), fontsize=10, va="top")
        ax = fig.add_axes([0.06, 0.12, 0.3, 0.62])
        clean = bt[~bt["contaminated"]]
        ax.scatter(clean["neighbors_gain"] / scale, clean["gain"] / scale, s=24, color=COLORS["BLANK"], ec="k", lw=0.3, label="clean")
        cont = bt[bt["contaminated"]]
        ax.scatter(cont["neighbors_gain"] / scale, cont["gain"] / scale, s=34, color=COLORS["not"], ec="k", lw=0.4, label="contaminated")
        for r in cont.itertuples():
            ax.annotate(f"r{r.mark_row + 1} c{r.mark_col + 1}", (r.neighbors_gain / scale, r.gain / scale), xytext=(5, 3),
                        textcoords="offset points", fontsize=8)
        xs = np.array([0, np.nanmax(bt["neighbors_gain"]) / scale * 1.05])
        ax.plot(xs, xs * bs["clean_slope"], color="k", lw=0.8, ls=":", label=f"clean BLANKs' slope {bs['clean_slope']:.3f}")
        ax.axhline(floor / scale, color="k", lw=0.9, ls="--", label="floor")
        ax.set(xlabel=f"neighbors' median gain ({unit})", ylabel=f"BLANK chamber's gain ({unit})")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax = fig.add_axes([0.42, 0.12, 0.14, 0.62])
        gridb = np.full((n_rows, n_cols), np.nan)
        for r in bt.itertuples():
            gridb[r.mark_row, r.mark_col] = 1.0 if r.contaminated else 0.0
        ax.imshow(np.zeros((n_rows, n_cols)), cmap=ListedColormap(["#F2F2F2"]), aspect="auto")
        ax.imshow(gridb, cmap=ListedColormap([COLORS["BLANK"], COLORS["not"]]), vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        ax.set_title("BLANKs on the chip\n(orange: contaminated)", fontsize=9)
        ax.set_xticks([0, n_cols // 2 - 1, n_cols - 1], ["1", str(n_cols // 2), str(n_cols)])
        ax.set_yticks([0, n_rows // 2 - 1, n_rows - 1], ["1", str(n_rows // 2), str(n_rows)])
        ax.set(xlabel="column", ylabel="row")
        ax = fig.add_axes([0.63, 0.12, 0.33, 0.62])
        e0 = expression.copy()
        wtb = e0[e0["tag"] == "WT"].groupby(time)[value].quantile([0.1, 0.5, 0.9]).unstack() / scale
        ax.fill_between(wtb.index, wtb[0.1], wtb[0.9], color=COLORS["WT"], alpha=0.2, lw=0)
        ax.plot(wtb.index, wtb[0.5], color=COLORS["WT"], lw=1.6, label="WT (median, 10-90th pct)")
        contaminated = {(r.mark_row, r.mark_col) for r in cont.itertuples()}
        for (rr, cc), cg in e0[e0["tag"] == "BLANK"].groupby(KEYS):
            cg = cg.sort_values(time)
            bad = (rr, cc) in contaminated
            ax.plot(cg[time], cg[value] / scale, "o-" if bad else "-", ms=2.5, lw=1.1 if bad else 0.7,
                    color=COLORS["not"] if bad else COLORS["BLANK"], alpha=1 if bad else 0.7)
        ax.plot([], [], color=COLORS["BLANK"], label="BLANK, clean"); ax.plot([], [], "o-", ms=2.5, color=COLORS["not"], label="BLANK, contaminated")
        ax.set(xlabel=time, ylabel=f"GFP in the DNA chamber ({unit})")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        pdf.savefig(fig); plt.close(fig)

        # ---- 3. every variant, by residue position
        fig = plt.figure(figsize=PAGE)
        _frame(fig, "Every variant's chambers: GFP gained over the time course", experiment)
        ax = fig.add_axes([0.06, 0.42, 0.9, 0.45])
        tags = sorted(lib["tag"].unique(), key=lambda t: (_position(t), str(t)))
        xpos = {t: i for i, t in enumerate(tags)}
        rng = np.random.default_rng(0)
        x = lib["tag"].map(xpos).to_numpy() + rng.uniform(-0.25, 0.25, len(lib))
        g = lib["gain"].to_numpy() / scale
        ax.scatter(x[lib["expressed"]], g[lib["expressed"]], s=9, color=COLORS["library"], lw=0, alpha=0.75, label="above the floor")
        ax.scatter(x[~lib["expressed"]], g[~lib["expressed"]], s=11, facecolor="white", edgecolor=COLORS["not"], lw=0.8, label="at or below the floor")
        for t in none["tag"]:
            ax.axvspan(xpos[t] - 0.5, xpos[t] + 0.5, color=COLORS["not"], alpha=0.12, lw=0)
        ax.axhline(floor / scale, color="k", lw=0.9, ls="--")
        step = max(1, len(tags) // 48)
        ax.set_xticks(range(0, len(tags), step), [display_tag(t) for t in tags[::step]], rotation=90, fontsize=6.5)
        ax.set(xlim=(-0.7, len(tags) - 0.3), ylabel=f"GFP gained ({unit})")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("shaded: no chamber above the floor; dashed: the floor", fontsize=9, loc="left")
        ax = fig.add_axes([0.06, 0.05, 0.9, 0.25])
        ax.axis("off")
        ax.set_title(f"some chambers below the floor ({len(partial)})", fontsize=10, loc="left", weight="bold")
        if len(partial):
            cells = [f"{display_tag(r.tag)}  {int(r.chambers_expressed)}/{int(r.chambers)}" for r in partial.itertuples()]
            ncol = 8
            rows = [cells[i:i + ncol] + [""] * (ncol - len(cells[i:i + ncol])) for i in range(0, min(len(cells), ncol * 7), ncol)]
            t = ax.table(cellText=rows, loc="upper left", cellLoc="left")
            t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.3)
            for cell in t.get_celld().values():
                cell.set_edgecolor("white")
        pdf.savefig(fig); plt.close(fig)

        # ---- 4+. each non-expressing variant over time
        e = expression.copy()
        band = lambda tag: e[e["tag"] == tag].groupby(time)[value].quantile([0.1, 0.5, 0.9]).unstack() / scale
        wt, bl = band("WT"), band("BLANK")
        for start in range(0, len(none), per_page):
            fig = plt.figure(figsize=PAGE)
            _frame(fig, "Variants that did not express: each chamber over time", experiment)
            fig.text(0.04, 0.905, "Green: the WT chambers (median and 10th-90th percentile); gray: the BLANKs; orange: the variant's chambers, one line each.",
                     fontsize=9, color="0.3", va="top")
            sub = none.iloc[start:start + per_page]
            ncol = 4
            for k, r in enumerate(sub.itertuples()):
                ax = fig.add_axes([0.06 + (k % ncol) * 0.235, 0.62 - (k // ncol) * 0.27, 0.19, 0.2])
                for b, color in ((wt, COLORS["WT"]), (bl, COLORS["BLANK"])):
                    ax.fill_between(b.index, b[0.1], b[0.9], color=color, alpha=0.2, lw=0)
                    ax.plot(b.index, b[0.5], color=color, lw=1.6)
                for _, cg in e[e["tag"] == r.tag].groupby(KEYS):
                    cg = cg.sort_values(time)
                    ax.plot(cg[time], cg[value] / scale, "o-", ms=2.5, lw=0.9, color=COLORS["not"])
                ax.set_title(f"{display_tag(r.tag)} ({int(r.chambers)} chambers)", fontsize=9, loc="left")
                ax.tick_params(labelsize=7)
                ax.spines[["top", "right"]].set_visible(False)
                if k % ncol == 0:
                    ax.set_ylabel(unit, fontsize=8)
                if k // ncol == (len(sub) - 1) // ncol:
                    ax.set_xlabel(time, fontsize=8)
            pdf.savefig(fig); plt.close(fig)
    return path
