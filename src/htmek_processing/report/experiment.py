"""The experiment report: one PDF with the chip-level picture of an HT-MEK experiment.

Ported from MBO-040's 7_downstream/full_report/make_full_report.py (2026-10-29). Pages follow
the chamber report's grid (report.chamber._PAGE: 6 columns, the same type sizes, grid tables,
header and footer); the numbers come from report.volcano, so they match the volcano plots:

  1. summary: key numbers; chip maps of expression, MM fit status, Km and Vmax / GFP; the
     chamber funnel; the Km and Vmax / GFP volcanoes
  2. chip maps: expression (GFP), GFP vs floor, MM fit status, Km, Vmax / GFP
  3. every [S] run over the chip: the rate, and the rate / GFP (positional artifacts)
  4. position trends: GFP, Km and Vmax / GFP down the rows and across the columns
  5. distributions: GFP (with the BLANKs and the floor), Km, Vmax / GFP, Km vs Vmax / GFP

    experiment_report_pdf(path, gfp=..., rates=..., mm_raw=..., mm_per_gfp=..., floor=...,
                          experiment="hAcyP1 AFMtag canonical · 2026-10-27", rate_unit="RFU/s")
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, TwoSlopeNorm

from . import volcano as V
from .chamber import REPORT_COLORS, _PAGE, _REPORT_STYLE, _grid_table, _log_minor_ticks

EXPERIMENT_REPORT_VERSION = "2.0"  # 1.x: MBO-040's full report script; bump on layout changes
KEYS = V.KEYS
GRID = _PAGE  # 11 in wide; 6 columns of col_w, gap apart, from `left`
PAGE = (11, 8.5)  # inches, landscape
CONTROLS = {"WT": "o", "N41A": "s", "BLANK": "^"}  # control -> marker on the chip maps
SCAN_COLORS = {"Ala scan": "#4E79A7", "Phe scan": "#E1812C", "Met scan": "#59A14F"}
STATUS = {  # MM fit status on the chip map: (color, label)
    "pass": ("#1F4E79", "pass: r² ≥ {min_r2:g}"),
    "fail": ("#E8A3A0", "fail: r² < {min_r2:g}"),
    "below floor": ("0.85", "not analyzed: GFP below floor"),
    "BLANK": ("white", "BLANK (no DNA)"),
}
FLOOR_MAP = {  # GFP vs floor on the chip map: (color, label)
    "above": ("#4E9A6B", "above floor"),
    "below, mixed": (STATUS["below floor"][0], "below floor; variant partly above"),
    "below, all": ("#C0392B", "below floor; all of variant below"),
}
OUTLINE = {False: 0.35, True: 1.0}  # volcano point outlines (pt): p >= cutoff, p < cutoff


@dataclass
class Context:
    """What every page needs: names, units and settings of this experiment."""

    experiment: str  # on every page, top right (e.g. library · date)
    rate_unit: str  # "µM/s" (calibrated) or "RFU/s"
    gfp_label: str  # what the GFP readout is, e.g. "button sum − background scan"
    floor: float
    s_range: tuple
    s_ticks: list  # [S] of the runs: the Km axes' ticks
    min_r2: float
    p_significant: float
    reference: str  # "WT", "library median", or a fixed reference's name (e.g. "MBO-040 WT")
    n_rows: int
    n_cols: int
    rate_note: str = ""
    built: str = field(default_factory=lambda: date.today().isoformat())
    km_label: str = "K$_m$"
    vmax_label: str = "v$_{max}$ / GFP"

    @property
    def vmax_unit(self):
        return f"{self.rate_unit} per GFP RFU"

    @property
    def ref_text(self):
        return self.reference


# ---- small helpers ------------------------------------------------------------------------
def counts_format(axis, values=None):
    """Tick labels for GFP RFU in k or M (with a true minus sign), by their size."""
    big = values is not None and np.nanmax(np.abs(values)) >= 2e6
    div, suffix = (1e6, "M") if big else (1e3, "k")
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v / div:g}{suffix}".replace("-", "−") if v else "0"))


def counts_text(v, big):
    return f"{v / 1e6:.2f}M" if big else f"{v / 1e3:.1f}k"


def mark_floor(bar, floor, big):
    """The GFP floor on an expression color bar: a white line across the bar, labeled above."""
    bar.ax.axvline(floor, color="white", lw=1.4)
    bar.ax.axvline(floor, color="0.15", lw=0.5)
    bar.ax.annotate(f"floor {counts_text(floor, big)}".replace("-", "−"), (floor, 1),
                    xycoords=("data", "axes fraction"), xytext=(0, 1.5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5, color="0.2")


def km_ticks(axis, ticks, lo=None, hi=None):
    """A log Km axis labeled at `ticks` (within lo-hi), in full: 156.25, 312.5, 1,250."""
    ticks = [t for t in ticks if (lo is None or t >= lo) and (hi is None or t <= hi)]
    axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.6g}"))
    _log_minor_ticks(axis)


def grid_col(k):
    """Left edge (in) of grid column k (0-5)."""
    return GRID["left"] + k * (GRID["col_w"] + GRID["gap"])


def span(k0, k1):
    """Width (in) of grid columns k0..k1, inclusive."""
    return grid_col(k1) + GRID["col_w"] - grid_col(k0)


def box(fig, x, top, w, h):
    """Axes x inches from the left and `top` inches from the top of the page."""
    W, H = fig.get_size_inches()
    return fig.add_axes((x / W, 1 - (top + h) / H, w / W, h / H))


def page_frame(fig, ctx, title, size=15):
    """Title top left, the experiment top right (the grid's right edge), version and build
    date at the bottom."""
    W, H = fig.get_size_inches()
    fig.text(GRID["left"] / W, 1 - 0.45 / H, title, fontsize=size, weight="bold", va="center_baseline")
    fig.text((grid_col(5) + GRID["col_w"]) / W, 1 - 0.28 / H, ctx.experiment, ha="right", va="top",
             fontsize=12, color="0.35")
    fig.text(1 - 0.15 / W, 0.12 / H,
             f"experiment report v{EXPERIMENT_REPORT_VERSION} · built {ctx.built}",
             ha="right", va="bottom", fontsize=8, color="0.55")


def grid(chambers, column, ctx):
    """A chamber column as an (n_rows, n_cols) array, for imshow."""
    g = chambers.pivot(index="mark_row", columns="mark_col", values=column)
    return g.reindex(index=range(ctx.n_rows), columns=range(ctx.n_cols)).to_numpy(float)


def map_axes(fig, ctx, x, top, width):
    """Axes for one chip map, every chamber a square. Returns (axes, height in inches)."""
    w, h = fig.get_size_inches()
    height = width * ctx.n_rows / ctx.n_cols
    return fig.add_axes((x / w, 1 - (top + height) / h, width / w, height / h)), height


def full_frame(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)


def chip_map(ax, values, chambers, ctx, *, cmap, norm, title, marks=True, shows=None):
    """One chip map in chamber index space. `shows`: a small gray line under the title."""
    shown = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    full_frame(ax)
    shown.cmap.set_bad("0.93")
    ax.set_title(title, loc="left", pad=13 if shows else None)
    if shows:
        ax.annotate(shows, (0, 1), xycoords="axes fraction", xytext=(0, 3), textcoords="offset points",
                    va="bottom", fontsize=8, color="0.4")
    ax.set_xticks([0, ctx.n_cols // 2 - 1, ctx.n_cols - 1], [1, ctx.n_cols // 2, ctx.n_cols])
    ax.set_yticks([0, ctx.n_rows // 2 - 1, ctx.n_rows - 1], [1, ctx.n_rows // 2, ctx.n_rows])
    ax.tick_params(labelsize=7, length=2)
    if marks:
        for tag, marker in CONTROLS.items():
            c = chambers[chambers["tag"] == tag]
            ax.scatter(c["mark_col"], c["mark_row"], marker=marker, s=9, facecolor="none",
                       edgecolor="k", lw=0.6, label=tag)
    return shown


def floor_kind(chambers):
    """Above the GFP floor, or below it split by whether every chamber of its variant is."""
    above = chambers["E_above_floor"].fillna(False).astype(bool)
    any_above = above.groupby(chambers["tag"]).transform("any")
    return np.select([above, ~any_above], ["above", "below, all"], "below, mixed")


def status_legend(ctx):
    return {k: (c, l.format(min_r2=ctx.min_r2)) for k, (c, l) in STATUS.items()}


# ---- the inputs ---------------------------------------------------------------------------
def report_tables(gfp, rates, mm_raw, mm_per_gfp, *, floor, gfp_col="button_sum_minus_bg", rate="v0",
                  min_r2=0.9, rate_unit="µM/s", volcano_kw=None):
    """One row per chamber (tag, GFP, floor verdict, MM fit status, the Km and Vmax / GFP the
    volcano uses, their fit SEs), the rates with rate / GFP, and the volcano tables.
    Returns (chambers, rates, tables {quantity name: table}, quantities, refs)."""
    fits, vrates = V.volcano_inputs(gfp, mm_raw, mm_per_gfp, rates, floor=floor, gfp_col=gfp_col,
                                    rate=rate, min_r2=min_r2)
    s_range = (vrates["bzp_uM"].min(), vrates["bzp_uM"].max())
    chambers = gfp[KEYS + ["tag", gfp_col]].rename(columns={gfp_col: "gfp"})
    chambers["E_above_floor"] = chambers["gfp"] > floor
    raw = mm_raw.set_index(KEYS)["r2"]
    chambers["r2_raw"] = chambers.set_index(KEYS).index.map(raw)
    chambers["status"] = np.select(
        [chambers["tag"] == "BLANK", ~chambers["E_above_floor"], chambers["r2_raw"] >= min_r2],
        ["BLANK", "below floor", "pass"], default="fail")
    qs = [q for q in V.quantities(rate_unit=rate_unit) if q.name in ("km", "vmax")]
    for q in qs:
        used = V.chamber_values(q, fits, vrates, s_range)
        chambers = chambers.merge(used[KEYS + ["value", "bound"]].rename(
            columns={"value": q.name, "bound": f"{q.name}_bound"}), on=KEYS, how="left")
    se = (mm_raw[KEYS + ["km_se"]].merge(mm_per_gfp[KEYS + ["vmax_se"]], on=KEYS, how="outer"))
    chambers = chambers.merge(se, on=KEYS, how="left")
    tables, refs = {}, {}
    for q in qs:
        tables[q.name], refs[q.name], _ = V.variant_table(q, fits, vrates, **(volcano_kw or {}))
    return chambers, vrates, tables, qs, refs


# ---- pages ----------------------------------------------------------------------------------
def page_chip_maps(pdf, chambers, ctx):
    fig = plt.figure(figsize=PAGE)
    page_frame(fig, ctx, "Chip maps")
    n_maps, gap = 5, 0.26
    width = grid_col(5) + GRID["col_w"] - GRID["left"]
    side = (width - (n_maps - 1) * gap) / n_maps
    lefts = [GRID["left"] + k * (side + gap) for k in range(n_maps)]
    top = 1.0
    e = grid(chambers, "gfp", ctx)
    big = np.nanmax(np.abs(e)) >= 2e6

    def bar_below(m, x, h, kind="linear", label=""):
        cax = box(fig, x, top + h + 0.35, side, 0.09)
        bar = fig.colorbar(m, cax=cax, orientation="horizontal")
        bar.ax.tick_params(labelsize=7.5, length=2)
        bar.set_label(label, fontsize=7.5, color="0.3", labelpad=2)
        if kind == "km":
            km_ticks(bar.ax.xaxis, ctx.s_ticks[::2] + [ctx.s_ticks[-1]], *ctx.s_range)
        elif kind == "log":
            bar.ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
            _log_minor_ticks(bar.ax.xaxis)
        else:
            bar.ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            counts_format(bar.ax.xaxis, e)
        return bar

    lo, hi = np.nanpercentile(e, [1, 99])
    ax, h = map_axes(fig, ctx, lefts[0], top, side)
    m = chip_map(ax, e, chambers, ctx, cmap="viridis",
                 norm=TwoSlopeNorm(vcenter=0, vmin=min(lo, -0.05 * hi), vmax=hi),
                 title="Expression (GFP)", shows=ctx.gfp_label)
    mark_floor(bar_below(m, lefts[0], h, label="RFU"), ctx.floor, big)

    kind = pd.Series(floor_kind(chambers), index=chambers.index)
    codes = {k: i for i, k in enumerate(FLOOR_MAP)}
    ax, _ = map_axes(fig, ctx, lefts[1], top, side)
    chip_map(ax, grid(chambers.assign(code=kind.map(codes)), "code", ctx), chambers, ctx,
             cmap=ListedColormap([c for c, _ in FLOOR_MAP.values()]),
             norm=BoundaryNorm(np.arange(len(FLOOR_MAP) + 1) - 0.5, len(FLOOR_MAP)),
             title="GFP relative to floor", shows=f"floor: {ctx.floor:,.0f} RFU")
    counts = kind.value_counts()
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=c, ec="0.3", lw=0.5,
                                     label=textwrap.fill(f"{l} ({counts.get(k, 0):,})", 32))
                       for k, (c, l) in FLOOR_MAP.items()],
              loc="upper left", bbox_to_anchor=(-0.04, -0.04), fontsize=7.5, frameon=False,
              labelspacing=0.45)

    status = status_legend(ctx)
    codes = {k: i for i, k in enumerate(status)}
    ax, _ = map_axes(fig, ctx, lefts[2], top, side)
    chip_map(ax, grid(chambers.assign(code=chambers["status"].map(codes)), "code", ctx), chambers, ctx,
             cmap=ListedColormap([c for c, _ in status.values()]),
             norm=BoundaryNorm(np.arange(len(status) + 1) - 0.5, len(status)),
             title="MM fit status", shows="per chamber, raw rates")
    counts = chambers["status"].value_counts()
    controls = [plt.Line2D([], [], ls="", marker=mk, mfc="none", mec="k", label=f"{t} control chamber")
                for t, mk in CONTROLS.items()]
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=c, ec="0.3", lw=0.5,
                                     label=textwrap.fill(f"{l} ({counts.get(k, 0):,})", 32))
                       for k, (c, l) in status.items()] + controls,
              loc="upper left", bbox_to_anchor=(-0.04, -0.04), fontsize=7.5, frameon=False,
              labelspacing=0.45)

    ax, _ = map_axes(fig, ctx, lefts[3], top, side)
    m = chip_map(ax, grid(chambers, "km", ctx), chambers, ctx, cmap="magma",
                 norm=LogNorm(vmin=ctx.s_range[0], vmax=ctx.s_range[1]), title=ctx.km_label,
                 shows="raw MM fit")
    bar_below(m, lefts[3], h, kind="km", label="µM (clipped to the [S] range), log scale")

    vm = grid(chambers, "vmax", ctx)
    lo, hi = np.nanpercentile(vm, [2, 98]) if np.isfinite(vm).any() else (1e-9, 1)
    ax, _ = map_axes(fig, ctx, lefts[4], top, side)
    m = chip_map(ax, vm, chambers, ctx, cmap="magma", norm=LogNorm(vmin=lo, vmax=hi),
                 title=ctx.vmax_label, shows="MM fit of rate / GFP")
    bar_below(m, lefts[4], h, kind="log", label=ctx.vmax_unit)

    note = ("Light gray on the K$_m$ and v$_{max}$ / GFP maps: chamber not analyzed. Red chambers on "
            "the GFP floor map belong to variants whose chambers are all below the GFP floor (the "
            "BLANK chambers, triangles, are colored by their own GFP). No GFP standard curve has "
            "been applied: GFP is in RFU, not enzyme concentration. K$_m$ and v$_{max}$ / "
            "GFP are the values used in the volcano plots: K$_m$ clipped to the [S] range, and "
            "v$_{max}$ / GFP replaced by the rate / GFP at the highest [S] where K$_m$ exceeds it.")
    fig.text(GRID["left"] / PAGE[0], 0.04, textwrap.fill(note, 200), fontsize=7.5, color="0.35", va="bottom")
    pdf.savefig(fig)
    plt.close(fig)


def page_rate_maps(pdf, chambers, rates, ctx, rate="v0"):
    """Every run over the chip, one run per grid column: the rate, and below it the rate /
    GFP of the chambers above the GFP floor."""
    runs = np.sort(rates["bzp_uM"].unique())[:6]
    fig = plt.figure(figsize=PAGE)
    page_frame(fig, ctx, "Rate by [S] run")
    side = GRID["col_w"]
    rows = [(rate, "Rate (BLANK median subtracted)" if "blank_median_v0" in rates else "Rate",
             ctx.rate_unit, 1.15, 1),
            ("rate_per_gfp_count", "Rate / GFP, chambers with GFP above floor",
             f"10$^{{-6}}$ {ctx.vmax_unit}", 4.75, 1e6)]
    for column, label, unit, top, scale in rows:
        fig.text(GRID["left"] / PAGE[0], 1 - (top - 0.3) / PAGE[1], label, fontsize=11, weight="bold")
        for k, b in enumerate(runs):
            x = grid_col(k)
            d = chambers[KEYS + ["tag"]].merge(
                rates[rates["bzp_uM"] == b][KEYS + [column, "E_above_floor"]], on=KEYS, how="left")
            if column == "rate_per_gfp_count":
                d.loc[~d["E_above_floor"].fillna(False).astype(bool), column] = np.nan
            v = grid(d, column, ctx) * scale
            hi = np.nanpercentile(np.abs(v), 99) if np.isfinite(v).any() else 1
            ax, h = map_axes(fig, ctx, x, top, side)
            m = chip_map(ax, v, d, ctx, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-0.3 * hi, vmax=hi),
                         title="", marks=False)
            ax.set_title(f"{b:g} µM", fontsize=9.5)
            ax.set_xticks([])
            ax.set_yticks([])
            cax = box(fig, x, top + h + 0.12, side, 0.07)
            bar = fig.colorbar(m, cax=cax, orientation="horizontal")
            bar.ax.tick_params(labelsize=6.5, length=2)
            bar.ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            bar.set_label(unit, fontsize=6.5, color="0.3", labelpad=1.5)
    fig.text(GRID["left"] / PAGE[0], 0.03, ("Each map is scaled to its own 99th percentile; blue "
             "indicates a decreasing signal. " + ctx.rate_note).strip(), fontsize=7.5, color="0.35")
    pdf.savefig(fig)
    plt.close(fig)


def page_position_trends(pdf, chambers, ctx):
    fig, axes = plt.subplots(2, 3, figsize=PAGE, gridspec_kw=dict(hspace=0.4, wspace=0.3))
    page_frame(fig, ctx, "Position trends (median, IQR)")
    fig.subplots_adjust(top=0.86, left=0.08, right=0.94)
    library = chambers[~chambers["tag"].isin(CONTROLS)]
    for j, (col, name, log) in enumerate([("gfp", f"GFP ({textwrap.shorten(ctx.gfp_label, 30)})", False),
                                          ("km", f"{ctx.km_label} (µM, analyzed)", True),
                                          ("vmax", f"{ctx.vmax_label} ({ctx.vmax_unit}, analyzed)", True)]):
        for i, by in enumerate(["mark_row", "mark_col"]):
            ax = axes[i, j]
            q = library.groupby(by)[col].quantile([0.25, 0.5, 0.75]).unstack()
            ax.fill_between(q.index + 1, q[0.25], q[0.75], color="#4E79A7", alpha=0.25, lw=0)
            ax.plot(q.index + 1, q[0.5], color="#1F4E79", lw=1.5)
            if col == "km":
                ax.set_yscale("log", base=2)
                km_ticks(ax.yaxis, ctx.s_ticks)
            elif log:
                ax.set_yscale("log")
                _log_minor_ticks(ax.yaxis)
            else:
                counts_format(ax.yaxis, library["gfp"])
            ax.set(xlabel="chip row (top → bottom)" if by == "mark_row" else "chip column", ylabel=name)
    pdf.savefig(fig)
    plt.close(fig)


def page_distributions(pdf, chambers, ctx):
    fig, axes = plt.subplots(2, 2, figsize=PAGE, gridspec_kw=dict(hspace=0.4, wspace=0.3))
    page_frame(fig, ctx, "Distributions")
    fig.subplots_adjust(top=0.86, left=0.08, right=0.94)
    analyzed = chambers["status"] == "pass"

    ax = axes[0, 0]
    lo, hi = np.nanpercentile(chambers["gfp"], [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)
    ax.hist(chambers.loc[chambers["tag"] != "BLANK", "gfp"].clip(lo, hi), bins=bins, color="0.7",
            edgecolor="0.3", lw=0.3, label="chambers with DNA")
    ax.hist(chambers.loc[chambers["tag"] == "BLANK", "gfp"].clip(lo, hi), bins=bins, color="#E1812C",
            edgecolor="0.3", lw=0.3, label="BLANK")
    ax.axvline(ctx.floor, color="k", ls="--", lw=0.8, label=f"GFP floor ({ctx.floor:,.0f})")
    ax.set(xlabel=f"GFP ({ctx.gfp_label})", ylabel="chambers")
    counts_format(ax.xaxis, chambers["gfp"])
    ax.legend(frameon=False, fontsize=8)

    # Km on log2 (ticked at the runs' [S]); Vmax / GFP on log10
    for ax, col, name, base in ((axes[0, 1], "km", f"{ctx.km_label} (µM), clipped to the [S] range", 2),
                                (axes[1, 0], "vmax", f"{ctx.vmax_label} ({ctx.vmax_unit})", 10)):
        v = chambers.loc[analyzed & (chambers["tag"] != "BLANK"), col].dropna()
        v = v[v > 0]
        if len(v):
            edges = np.logspace(np.log(v.min()) / np.log(base), np.log(v.max()) / np.log(base) + 1e-9,
                                41, base=base)
            ax.hist(v, bins=edges, color="#4E79A7", edgecolor="0.3", lw=0.3)
        ax.set_xscale("log", base=base)
        _log_minor_ticks(ax.xaxis)
        if col == "km":
            km_ticks(ax.xaxis, ctx.s_ticks)
        for w in chambers.loc[analyzed & (chambers["tag"] == "WT"), col].dropna():
            ax.axvline(w, color="#C0392B", lw=1.2)
        ax.set(xlabel=name + (" (log$_2$ scale)" if base == 2 else ""), ylabel="analyzed chambers")
        ax.set_title("red: WT chambers", loc="right", fontsize=8, color="#C0392B")

    ax = axes[1, 1]
    d = chambers[analyzed & ~chambers["tag"].isin(CONTROLS)].copy()
    d["scan"] = d["tag"].str[-1].map(V.SCANS)

    def bars(e, color, z):
        """± 1 SE of each chamber's fit, where its value is a fitted estimate (not a bound)."""
        fitted = e[(e["km_bound"] == "") & (e["vmax_bound"] == "")]
        x, y = fitted["km"].to_numpy(), fitted["vmax"].to_numpy()
        xe, ye = fitted["km_se"].to_numpy(), fitted["vmax_se"].to_numpy()
        ax.errorbar(x, y, xerr=[np.minimum(xe, 0.9 * x), xe], yerr=[np.minimum(ye, 0.9 * y), ye],
                    fmt="none", ecolor=color, elinewidth=0.5, alpha=0.35, zorder=z)

    for scan, color in SCAN_COLORS.items():
        e = d[d["scan"] == scan]
        bars(e, color, 1)
        ax.scatter(e["km"], e["vmax"], s=8, color=color, alpha=0.8, lw=0, label=scan, zorder=2)
    wt = chambers[analyzed & (chambers["tag"] == "WT")]
    if len(wt):
        bars(wt, "#C0392B", 3)
        ax.scatter(wt["km"], wt["vmax"], s=40, color="#C0392B", ec="k", lw=0.5, label="WT", zorder=4)
    ax.set_title("error bars: ± 1 SE of each chamber's fit (omitted where the value is a bound)",
                 loc="left", fontsize=8, color="0.35")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    km_ticks(ax.xaxis, ctx.s_ticks)
    _log_minor_ticks(ax.yaxis)
    ax.set(xlabel=f"{ctx.km_label} (µM, log$_2$ scale)", ylabel=f"{ctx.vmax_label} ({ctx.vmax_unit})")
    ax.legend(frameon=False, fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)


def volcano_size(n_chambers):
    return 4 * n_chambers + 6


def volcano_key(ax, ctx, sizes):
    """The volcano markers' key in the upper right of `ax`, and marker size per chambers."""
    def dot(label, **kw):
        style = dict(ls="", marker="o", ms=5, mfc="0.75", mec="0.25", mew=OUTLINE[False])
        return plt.Line2D([], [], label=label, **{**style, **kw})

    main = ax.legend(handles=[dot(sc, mfc=c) for sc, c in SCAN_COLORS.items()]
                     + [dot(f"p < {ctx.p_significant:g}", mfc="white", mec="k", mew=OUTLINE[True]),
                        dot(f"{ctx.km_label} outside [S] range", marker=">"),
                        dot(f"{ctx.vmax_label}: lower bound", marker="<")],
                     loc="upper right", fontsize=7.5, frameon=False, handletextpad=0.3, labelspacing=0.3,
                     borderaxespad=0.3, title="markers", title_fontsize=7.5, alignment="left")
    ax.add_artist(main)
    ax.figure.canvas.draw()
    bb = main.get_window_extent().transformed(ax.transAxes.inverted())
    ax.legend(handles=[dot(f"{n}", ms=np.sqrt(volcano_size(n))) for n in sizes], loc="upper right",
              bbox_to_anchor=(1, bb.y0 - 0.01), ncol=len(sizes), fontsize=7.5, frameon=False,
              handletextpad=0.2, columnspacing=0.8, borderaxespad=0.3, title="chambers per variant",
              title_fontsize=7.5, alignment="left")


def mini_volcano(ax, q, table, ctx):
    """A compact volcano over all scans: color by scan, size by chambers, heavier outline at
    p < cutoff, the most significant few on each side named."""
    t = table[table["tested"]]
    for row in t.itertuples():
        ax.scatter([row.log2_fc], [row.neg_log10_p], s=volcano_size(row.n_chambers),
                   marker=q.markers[row.limit], color=SCAN_COLORS.get(row.scan, "0.6"),
                   edgecolor="k" if row.significant else "0.25", linewidth=OUTLINE[row.significant],
                   alpha=0.9, zorder=2 + row.significant)
    ax.axhline(-np.log10(ctx.p_significant), color="0.3", ls="--", lw=0.8)
    ax.axvline(0, color="0.6", lw=0.6)
    line = table.attrs.get("reference_line")
    if line is not None:   # e.g. another chip's WT Km, marked only
        x_line = np.log2(line["value"] / table["reference_median"].iloc[0])
        ax.axvline(x_line, color="#B9770E", lw=1.1, ls="--")
        ax.annotate(f"{line['name']}" + (" (PLACEHOLDER)" if line.get("placeholder") else ""), (x_line, 1.0),
                    xycoords=("data", "axes fraction"), xytext=(3, -3), textcoords="offset points",
                    va="top", fontsize=7, color="#B9770E", weight="bold")
    lim = np.ceil(t["log2_fc"].abs().max()) + 0.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(0, max(t["neg_log10_p"].max(), 2.0) * 1.15)
    ref = table["reference"].iloc[0]   # this quantity's own (a fixed one for Km, say)
    placeholder = bool(table.get("reference_placeholder", pd.Series([False])).iloc[0])
    ax.set_xlabel(f"log$_2$ fold change in {q.label}\n(vs {ref}{', PLACEHOLDER' if placeholder else ''})")
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    placed = []
    ranked = t.sort_values("neg_log10_p", ascending=False)
    named = pd.concat([ranked[ranked["log2_fc"] < 0].head(3), ranked[ranked["log2_fc"] > 0].head(3)])
    spots = [(8, 6), (8, -8), (-8, 6), (-8, -8)] + [
        (dx, dy) for r in (18, 28, 40) for dx, dy in
        ((r, 0), (-r, 0), (r, r * 0.6), (-r, r * 0.6), (r, -r * 0.6), (-r, -r * 0.6))]
    for row in named.itertuples():
        for dx, dy in spots:
            label = ax.annotate(row.tag, (row.log2_fc, row.neg_log10_p), xytext=(dx, dy),
                                textcoords="offset points", fontsize=8, va="center",
                                ha="left" if dx > 0 else "right", color="0.15" if row.significant else "0.5",
                                arrowprops=dict(arrowstyle="-", color="0.55", lw=0.5, shrinkA=1, shrinkB=3))
            label.update_positions(renderer)
            bb = matplotlib.text.Text.get_window_extent(label, renderer)
            inside = frame.x0 <= bb.x0 and bb.x1 <= frame.x1 and bb.y1 <= frame.y1
            if inside and not any(bb.overlaps(o) for o in placed):
                placed.append(bb)
                break
            label.remove()


def page_summary(pdf, chambers, tables, qs, ctx):
    """Everything important on one page, on the chamber report's 6-column grid."""
    with plt.rc_context(_REPORT_STYLE):
        fig = plt.figure(figsize=(GRID["w"], GRID["h"]))
        page_frame(fig, ctx, "Summary", size=19)
        top_a = 0.9
        analyzed = chambers["status"] == "pass"
        library = ~chambers["tag"].isin(CONTROLS)
        per_variant = chambers[analyzed & library].groupby("tag").size()
        km_t, vm_t = tables["km"], tables["vmax"]
        wt = chambers[(chambers["tag"] == "WT") & analyzed]
        ref_km = km_t["reference_median"].iloc[0]
        placeholder = bool(km_t.get("reference_placeholder", pd.Series([False])).iloc[0])
        ref_note = ("" if ctx.reference == "WT" else " (library)" if ctx.reference == "library median"
                    else f" ({ctx.reference}{', PLACEHOLDER' if placeholder else ''})")
        rows = [
            ["chambers with DNA", f"{int((chambers['tag'] != 'BLANK').sum()):,}"],
            ["GFP above the floor", f"{int((~chambers['status'].isin(['BLANK', 'below floor'])).sum()):,}"],
            [f"MM fit r² ≥ {ctx.min_r2:g}", f"{int(analyzed.sum()):,}"],
            ["variants, ≥ 2 analyzed", f"{int((per_variant >= 2).sum())} of {chambers.loc[library, 'tag'].nunique()}"],
            [f"median {ctx.km_label} (µM)", f"{np.nanmedian(chambers.loc[analyzed, 'km']):,.0f}"],
            [f"reference {ctx.km_label} (µM)", f"{ref_km:,.0f}" + ref_note],
            [f"{ctx.km_label} hits (p < {ctx.p_significant:g})",
             f"{int(km_t['significant'].sum())} of {int(km_t['tested'].sum())}"],
            [f"{ctx.vmax_label} hits (p < {ctx.p_significant:g})",
             f"{int(vm_t['significant'].sum())} of {int(vm_t['tested'].sum())}"],
        ]
        fills = {}
        if ctx.reference != "WT":  # no WT reference: flag it
            fills[(5, 1)] = REPORT_COLORS["warn"]
        row_h = 0.21
        ax = box(fig, grid_col(0), top_a, span(0, 1), row_h * (len(rows) + 1))
        _grid_table(ax, ["summary", ""], rows, label_cols=(0,), col_widths=[0.62, 0.38], fills=fills, size=10.5)
        big = np.nanmax(np.abs(chambers["gfp"])) >= 2e6
        note = (f"GFP floor: {counts_text(ctx.floor, big)} RFU\n".replace("-", "−")
                + f"[S]: {ctx.s_range[0]:g}–{ctx.s_range[1]:,.0f} µM; rates in {ctx.rate_unit}")
        if placeholder:
            note += (f"\nAmber: the {ctx.km_label} reference is a PLACEHOLDER\n({ctx.reference}, not measured on this chip).")
        elif ctx.reference != "WT":
            note += "\nAmber: too few WT chambers; fold changes\nare vs the library median."
        ax.annotate(note, (0, 0), xycoords="axes fraction", xytext=(0, -3), textcoords="offset points",
                    va="top", fontsize=9, color="0.45")

        side = GRID["col_w"]
        maps = []
        e = grid(chambers, "gfp", ctx)
        big = np.nanmax(np.abs(e)) >= 2e6
        lo, hi = np.nanpercentile(e, [1, 99])
        status = status_legend(ctx)
        codes = {k: i for i, k in enumerate(status)}
        vm = grid(chambers, "vmax", ctx)
        v_lo, v_hi = np.nanpercentile(vm, [2, 98]) if np.isfinite(vm).any() else (1e-9, 1)
        specs = [
            ("Expression (GFP)", ctx.gfp_label, e, "viridis",
             TwoSlopeNorm(vcenter=0, vmin=min(lo, -0.05 * hi), vmax=hi), "RFU"),
            ("MM fit", "per chamber, raw rates", grid(chambers.assign(code=chambers["status"].map(codes)), "code", ctx),
             ListedColormap([c for c, _ in status.values()]),
             BoundaryNorm(np.arange(len(status) + 1) - 0.5, len(status)), ""),
            (ctx.km_label, "raw MM fit", grid(chambers, "km", ctx), "magma",
             LogNorm(vmin=ctx.s_range[0], vmax=ctx.s_range[1]), "µM, log scale"),
            (ctx.vmax_label, "MM fit of rate / GFP", vm, "magma", LogNorm(vmin=v_lo, vmax=v_hi), ctx.vmax_unit),
        ]
        for k, (title, shows, values, cmap, norm, unit) in enumerate(specs):
            ax, h = map_axes(fig, ctx, grid_col(2 + k), top_a, side)
            m = chip_map(ax, values, chambers, ctx, cmap=cmap, norm=norm, title="", marks=k != 1)
            ax.set_title(title, fontsize=11, loc="left", pad=13)
            ax.annotate(shows, (0, 1), xycoords="axes fraction", xytext=(0, 3), textcoords="offset points",
                        va="bottom", fontsize=7.5, color="0.4")
            ax.set_xticks([])
            ax.set_yticks([])
            maps.append((ax, m, title, h))
            if title == "MM fit":
                counts = chambers["status"].value_counts()
                short = {"pass": "pass", "fail": "fail", "below floor": "GFP < floor", "BLANK": "BLANK"}
                ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=c, ec="0.3", lw=0.4,
                                                 label=f"{short[key]} {counts.get(key, 0):,}")
                                   for key, (c, _) in status.items()],
                          loc="upper left", bbox_to_anchor=(-0.06, -0.055), fontsize=7, ncol=2, frameon=False,
                          handlelength=0.8, handletextpad=0.3, labelspacing=0.15, columnspacing=0.6,
                          borderaxespad=0)
            else:
                cax = box(fig, grid_col(2 + k), top_a + h + 0.24, side, 0.08)
                bar = fig.colorbar(m, cax=cax, orientation="horizontal")
                bar.ax.tick_params(labelsize=8, length=2)
                bar.set_label(unit, fontsize=7.5, color="0.3", labelpad=1)
                if title == ctx.km_label:  # a narrow bar: three labels
                    km_ticks(bar.ax.xaxis, [ctx.s_ticks[0], ctx.s_ticks[len(ctx.s_ticks) // 2], ctx.s_ticks[-1]],
                             *ctx.s_range)
                elif isinstance(norm, LogNorm):
                    bar.ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
                    _log_minor_ticks(bar.ax.xaxis)
                else:
                    bar.ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                    counts_format(bar.ax.xaxis, e)
                    mark_floor(bar, ctx.floor, big)
        map_h = maps[0][3]
        handles = [plt.Line2D([], [], ls="", marker=mk, ms=4, mfc="none", mec="k", label=t)
                   for t, mk in CONTROLS.items()]
        table_h = row_h * (len(rows) + 1)
        legend_ax = box(fig, grid_col(0), top_a + table_h + 0.82, span(0, 1), 0.3)
        legend_ax.axis("off")
        legend_ax.legend(handles=handles, loc="upper left", ncol=3, frameon=False, fontsize=8.5,
                         handletextpad=0.2, columnspacing=1.0, borderaxespad=0,
                         title="control chambers (markers on the chip maps)", title_fontsize=8.5,
                         alignment="left")

        top_c = top_a + max(table_h + 1.2, map_h + 0.55) + 0.35
        h_c = 3.2
        ax = box(fig, grid_col(0), top_c, span(0, 1) - 0.2, h_c)
        steps = [("all chambers", len(chambers)),
                 ("DNA printed", int((chambers["tag"] != "BLANK").sum())),
                 ("GFP above floor", int((~chambers["status"].isin(["BLANK", "below floor"])).sum())),
                 (f"MM fit r² ≥ {ctx.min_r2:g}", int(analyzed.sum()))]
        colors = ["0.75", "0.68", "0.6", STATUS["pass"][0]]
        for k, ((name, n), c) in enumerate(zip(steps, colors)):
            y = len(steps) - 1 - k
            ax.barh(y, n, color=c, edgecolor="0.2", lw=0.4, height=0.75)
            inside = n > 0.55 * len(chambers)
            pad = 0.015 * len(chambers)
            ax.text(pad if inside else n + pad, y, f"{name}: {n:,}", va="center", fontsize=9,
                    color="white" if inside and c == STATUS["pass"][0] else "0.1")
        ax.set_xlim(0, len(chambers))
        ax.set_yticks([])
        ax.set_xlabel("chambers")
        ax.set_title("Chambers retained at each step", loc="left", fontsize=11)
        ax.spines["left"].set_visible(False)

        for k, q in enumerate(qs):
            ax = box(fig, grid_col(2 + 2 * k), top_c, span(0, 1), h_c)
            mini_volcano(ax, q, tables[q.name], ctx)
            ax.set_title(f"{q.label} volcano: −log$_{{10}}$ p", loc="left", fontsize=11)
            if q.name == "vmax":
                n = tables[q.name].loc[tables[q.name]["tested"], "n_chambers"]
                volcano_key(ax, ctx, sorted({int(n.min()), int(round(n.median())), int(n.max())}))
        pdf.savefig(fig)
        plt.close(fig)


def experiment_report_pdf(path, *, gfp, rates, mm_raw, mm_per_gfp, floor, experiment,
                          rate_unit="µM/s", gfp_col="button_sum_minus_bg", gfp_label="button sum",
                          rate="v0", min_r2=0.9, n_rows=56, n_cols=32, rate_note="", volcano_kw=None):
    """Write the experiment report (5 pages) to `path`.

    gfp        one row per chamber: mark_row, mark_col, tag and `gfp_col` (one GFP scan)
    rates      one row per chamber and [S] run (bzp_uM), with `rate` (e.g. fit_progress_curves'
               v0, BLANK-subtracted); runs marked left_out_of_mm are dropped
    mm_raw     raw MM fits per chamber (fit_mm_table: vmax, km, r2, vmax_se, km_se)
    mm_per_gfp per-GFP MM fits per chamber
    floor      the GFP floor (e.g. volcano.gfp_floor(gfp))
    experiment the text at the top right of every page (library · date)
    volcano_kw settings for volcano.variant_table (reference, min_replicates, ...)
    Returns {"km": table, "vmax": table}, the volcano tables behind the summary page."""
    volcano_kw = dict(volcano_kw or {})
    chambers, vrates, tables, qs, _ = report_tables(gfp, rates, mm_raw, mm_per_gfp, floor=floor,
                                                    gfp_col=gfp_col, rate=rate, min_r2=min_r2,
                                                    rate_unit=rate_unit, volcano_kw=volcano_kw)
    s_ticks = sorted(float(s) for s in vrates["bzp_uM"].unique())
    ctx = Context(experiment=experiment, rate_unit=rate_unit, gfp_label=gfp_label, floor=floor,
                  s_range=(s_ticks[0], s_ticks[-1]), s_ticks=s_ticks, min_r2=min_r2,
                  p_significant=volcano_kw.get("p_significant", 0.01),
                  reference=tables["km"]["reference"].iloc[0], n_rows=n_rows, n_cols=n_cols,
                  rate_note=rate_note)
    with plt.rc_context({"font.family": "Arial", "font.size": 9, "axes.titlesize": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}):
        with PdfPages(path) as pdf:
            page_summary(pdf, chambers, tables, qs, ctx)
            page_chip_maps(pdf, chambers, ctx)
            page_rate_maps(pdf, chambers, vrates, ctx, rate=rate)
            page_position_trends(pdf, chambers, ctx)
            page_distributions(pdf, chambers, ctx)
    return tables
