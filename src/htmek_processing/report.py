"""Chamber reports: one PDF page per chamber, and the page key."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .plots import FIT_COLORS
from .fit import four_pl, michaelis_menten


# Version of the chamber report's layout and content. Bump it whenever the report changes
# (what a page shows, how it is drawn, or a rule it displays), and add an entry to
# chamber_report_changelog.md next to the report PDF (5_figures/4_report/).
REPORT_VERSION = "1.9.4"

REPORT_COLORS = {
    "this": "#1F4E79",  # this chamber
    "replicate": "#8FB3D9",  # the variant's other chambers
    "blank": "0.82",  # BLANK chambers / everything else
    "header": "#D5DFEA",  # table header row
    "label": "#EEF2F7",  # table label columns
    "fail": "#F6D3CF",  # a value that fails a filter
    "pass": "#D7EBD5",  # ... and one that passes
    "warn": "#FBE7C6",  # a flag that doesn't exclude the chamber (e.g. on a tile seam)
    # button-QC tabs on the image crops: lighter and brighter than the table fills
    "tab_fail": "#FFCFC7",
    "tab_warn": "#FFE7B3",
    "tab_pass": "#CFF5C9",
    "wt": "#C0392B",  # WT reference lines (Km strip, normalized MM panel)
    "seam": "#E8A33D",  # tile-seam notches on the image crops (the "warn" amber, darker
    #                     so it shows on an image)
}


def read_button_qc(folder, image):
    """Read chamberqc's per-button scores for one image (e.g. "gfp_washed") from its output
    folder for this experiment (e.g. ~/repos/chamberqc/out/<experiment>). Returns None if
    there is no file, so the report simply draws no QC tab.

    The file is one row per button: mark_row, mark_col, anomaly_score, anomaly_rank
    (1 = most unusual), and, once a classifier is trained, qc_prob, qc_flag, qc_reason.
    Accepts the current file name (chamberqc_<image>.csv) and the earlier one
    (pedestal_qc_<image>.csv)."""
    folder = Path(folder)
    for name in (f"chamberqc_{image}.csv", f"pedestal_qc_{image}.csv"):
        if (folder / name).exists():
            return pd.read_csv(folder / name)
    return None


def button_qc_tabs(qc, *, flag_top_frac=0.05, warn_top_frac=0.15):
    """Turn chamberqc scores into a report tab per button: {(mark_row, mark_col): (text,
    level)}, with level "fail" (red), "warn" (amber) or "pass" (green).

    With a trained classifier (a `qc_flag` column), the tab says whether the button
    looks bad, with its probability and reason. Without one (stage 1: anomaly scores
    only), it says how unusual the button looks among this image's buttons:
      the `flag_top_frac` most unusual             -> "fail", e.g. "QC: anomaly 99th pct"
      next, up to the `warn_top_frac` most unusual -> "warn", e.g. "QC: anomaly 91st pct"
      the rest                                     -> "pass", "QC ok"
    The percentile is the anomaly score's: the share of this image's buttons that look
    less unusual (99th = more unusual than 99 % of them), rounded down so the most unusual
    button reads 99th, not 100th.
    An unusual button is not necessarily a bad one; the classifier separates the two."""
    if qc is None:
        return {}
    tabs = {}
    n = len(qc)
    for row in qc.itertuples():
        key = (int(row.mark_row), int(row.mark_col))
        if "qc_flag" in qc.columns and pd.notna(getattr(row, "qc_flag")):
            if bool(row.qc_flag):
                reason = getattr(row, "qc_reason", "") or ""
                text = f"QC: bad {row.qc_prob:.2f}" + (f" · {reason}" if reason else "")
                tabs[key] = (text, "fail")
            else:
                tabs[key] = ("QC ok", "pass")
            continue
        top = row.anomaly_rank / n  # fraction of buttons at least this unusual
        percentile = int(np.floor(100 * (n - row.anomaly_rank) / n))  # share less unusual
        if top <= flag_top_frac:
            tabs[key] = (f"QC: anomaly {_ordinal(percentile)} pct", "fail")
        elif top <= warn_top_frac:
            tabs[key] = (f"QC: anomaly {_ordinal(percentile)} pct", "warn")
        else:
            tabs[key] = ("QC ok", "pass")
    return tabs


def _ordinal(k):
    """1 -> "1st", 2 -> "2nd", 11 -> "11th", 91 -> "91st"."""
    suffix = "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{suffix}"


def _fmt(v):
    """Format one table value: yes/no for booleans, thousands separators for big numbers,
    3 significant figures otherwise, and ×10ⁿ for tiny ones."""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if not isinstance(v, (int, float, np.number)):
        return str(v)
    if not np.isfinite(v):
        return "–"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if isinstance(v, (int, np.integer)):
        return str(v)
    if v == 0 or abs(v) >= 1e-3:
        return f"{v:#.3g}".rstrip(".")  # keep trailing zeros: 0.110, not 0.11
    exp = int(np.floor(np.log10(abs(v))))
    return f"{v / 10**exp:.2f}×10$^{{{exp}}}$"


def _grid_table(
    ax, header, rows, *, label_cols=(0,), col_widths=None, fills=None, size=11
):
    """Draw a table filling `ax` (axis hidden): thin grid lines, a shaded bold header row,
    and lightly shaded label columns (`label_cols`). `fills` maps (row, col) of a body cell
    (0-based, header excluded) to a background color, e.g. to flag a failing value.
    Values go through _fmt."""
    fills = fills or {}
    ax.axis("off")
    tab = ax.table(
        cellText=[[_fmt(v) for v in row] for row in rows],
        colLabels=header,
        colWidths=col_widths,
        bbox=(0, 0, 1, 1),
        cellLoc="left",
        colLoc="left",
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(size)
    for (i, j), cell in tab.get_celld().items():
        cell.set_edgecolor("0.7")
        cell.set_linewidth(0.6)
        cell.PAD = 0.07
        if i == 0:  # header row
            cell.set_facecolor(REPORT_COLORS["header"])
            cell.set_text_props(weight="bold")
        elif (i - 1, j) in fills:
            cell.set_facecolor(fills[(i - 1, j)])
        elif j in label_cols:
            cell.set_facecolor(REPORT_COLORS["label"])
            cell.set_text_props(color="0.25")


def _left_out(fit):
    """The [S] a fit left out (fit_mm_one_out), or None when it used every point."""
    if fit is None:
        return None
    value = fit.get("left_out", np.nan) if hasattr(fit, "get") else np.nan
    return float(value) if value is not None and np.isfinite(value) else None


def _mm_panel(ax, s_max, this, this_fit, reps, rep_fits, y, good=frozenset(), this_good=True):
    """One rate-vs-[BzP] panel: the variant's other chambers (light blue, each with its MM
    curve) and this chamber (dark blue, with its MM curve on top).

    A chamber whose MM fit passes the filter (its (mark_row, mark_col) is in `good`; for
    this chamber, `this_good`) is drawn with filled circles and a solid curve. One that
    fails is drawn with open circles and a dashed curve, as in the Km strip. A point its
    fit left out (a conditional pass, fit_mm_one_out) is drawn as an x."""
    s_grid = np.linspace(0, s_max, 100)  # linear [S], from 0
    rep_left = {}  # (mark_row, mark_col) -> the [S] its fit left out
    if rep_fits is not None and "left_out" in rep_fits.columns:
        for f in rep_fits.itertuples():
            if np.isfinite(f.left_out):
                rep_left[(f.mark_row, f.mark_col)] = f.left_out

    # ---- the variant's other chambers: points, then their MM curves ----
    if reps is not None and len(reps):
        out = np.asarray([rep_left.get((r_, c_)) == b for r_, c_, b in
                          zip(reps["mark_row"], reps["mark_col"], reps["bzp_uM"])], bool)
        ax.scatter(  # left out of their chamber's fit: x
            reps.loc[out, "bzp_uM"],
            reps.loc[out, y],
            s=16,
            marker="x",
            c=REPORT_COLORS["replicate"],
            lw=1.0,
            zorder=2,
        )
        reps = reps[~out]
        passes = [
            (r_, c_) in good for r_, c_ in zip(reps["mark_row"], reps["mark_col"])
        ]
        passes = np.asarray(passes, bool)
        ax.scatter(  # passing: filled
            reps.loc[passes, "bzp_uM"],
            reps.loc[passes, y],
            s=14,
            c=REPORT_COLORS["replicate"],
            lw=0,
            zorder=2,
        )
        ax.scatter(  # failing: open
            reps.loc[~passes, "bzp_uM"],
            reps.loc[~passes, y],
            s=14,
            c="white",
            ec=REPORT_COLORS["replicate"],
            lw=1.0,
            zorder=2,
        )
    for f in [] if rep_fits is None else rep_fits.itertuples():
        ok = (f.mark_row, f.mark_col) in good
        ax.plot(
            s_grid,
            michaelis_menten(s_grid, f.vmax, f.km),
            color=REPORT_COLORS["replicate"],
            lw=0.8,
            ls="-" if ok else "--",
            zorder=2,
        )

    # ---- this chamber: its MM curve, then its points on top ----
    if this_fit is not None:
        ax.plot(
            s_grid,
            michaelis_menten(s_grid, this_fit["vmax"], this_fit["km"]),
            color=REPORT_COLORS["this"],
            lw=1.8,
            ls="-" if this_good else "--",
            zorder=3,
        )
    left = _left_out(this_fit)
    if this is not None and left is not None:  # the point the fit left out, as an x
        out = this[this["bzp_uM"] == left]
        ax.plot(out["bzp_uM"], out[y], "x", ms=7, mew=1.6, color=REPORT_COLORS["this"],
                zorder=4)
        this = this[this["bzp_uM"] != left]
    if this is not None:
        ax.plot(
            this["bzp_uM"],
            this[y],
            "o",
            ms=6,
            color=REPORT_COLORS["this"],
            mfc=REPORT_COLORS["this"] if this_good else "white",
            mec="k" if this_good else REPORT_COLORS["this"],
            mew=0.5 if this_good else 1.3,
            zorder=4,
        )
    ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    ax.set_xlim(0, s_max * 1.04)
    ax.set_xlabel("[BzP] (µM)")


# chamber report page geometry (inches): one grid of 6 equal columns that every panel snaps
# to, so edges line up down the page
_PAGE = dict(w=11.0, h=8.5, left=0.9, col_w=1.32, gap=0.3)
_REPORT_STYLE = {
    "font.size": 12.5,
    "axes.titlesize": 12.5,
    "axes.labelsize": 12.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11.5,
}
_AXIS_LABELS = {"nadh_uM": "[NADH] (µM)", "median_bc": "NADH signal (a.u.)"}


def _report_page(page, value, norm_col):
    """Draw one chamber's report page from its `page` payload (built by chamber_report_pdf)
    and return the figure. Module-level so worker processes can import it.

    Landscape, on a 6-column grid (every panel's edges sit on grid lines):
      top:    one table, MM fits + chamber details (cols 1-3), a small chip map marking
              this chamber and its variant's other chambers | image crops (cols 4-6)
      middle: progress curve + rate fit per run (cols 1-6), one y-axis
      bottom: raw rate vs [BzP] (from col 1) and normalized rate (from col 4), square, each
              with its MM fit, then the key (col 6) and the variant's other replicates overlaid; the key (col 6)
      last:   a thin strip of every chamber's Km (cols 1-4, key in 5-6), with this variant's other
              chambers and this chamber marked (if raw MM fits were given)."""
    P = _PAGE
    runs, fits = page["runs"], page["fits"]
    r, c, tag = page["row"], page["col"], page["tag"]
    enzyme = norm_col.split("_per_")[-1]
    col = [P["left"] + k * (P["col_w"] + P["gap"]) for k in range(6)]  # left edges
    km = page.get("km")  # Km strip, when raw MM fits were given
    # rows, as (top, height) in inches from the top of the page
    top_a = 0.9  # table + image crops
    top_b, h_b = top_a + P["col_w"] + 0.6, 1.15  # progress curves
    top_c, h_c = top_b + h_b + 1.0, 2.5  # rate-vs-[BzP] panels (square) and key
    std = page.get("standards")  # this chamber's NADH standards and calibration line, if given
    # with the standard curve as a third square, the key moves to a band under the row
    h_key = 0.95 if std is not None else 0.0
    top_d, h_d = top_c + h_c + 0.85 + h_key, 0.75  # Km and Vmax / [E] strips
    vm = page.get("vmax")  # the Vmax / [E] strip, when normalized fits were given
    legend_h = 0.4 if km is not None and vm is not None else 0.0  # their shared key
    H = (top_d + h_d + legend_h if km is not None else top_c + h_c) + 0.65  # page height

    with plt.rc_context(_REPORT_STYLE):
        fig = plt.figure(figsize=(P["w"], H))
        # the page's elements by name, so the key page (_report_key_page) can point at them
        fig.key = dict(crops={}, tabs={}, runs={}, insets={})

        def box(x, top, w, h):
            """Axes at x inches from the left, `top` inches from the top of the page."""
            return fig.add_axes((x / P["w"], 1 - (top + h) / H, w / P["w"], h / H))

        # page title: a shaded pill with row | col, then the variant name. The pill has a
        # fixed width, so the name starts at the same place on every page
        from matplotlib.patches import FancyBboxPatch

        inch = fig.dpi_scale_trans  # positions in inches from the bottom-left
        pill_w, pill_h, yc = 0.9, 0.36, H - 0.45  # pill size, title center line
        x = P["left"]
        fig.key["pill"] = FancyBboxPatch(
            (x, yc - pill_h / 2),
            pill_w,
            pill_h,
            transform=inch,
            boxstyle=f"round,pad=0,rounding_size={pill_h / 2}",
            fc=REPORT_COLORS["header"],
            ec="none",
        )
        fig.patches.append(fig.key["pill"])
        fig.add_artist(
            plt.Line2D(
                [x + pill_w / 2] * 2,
                [yc - pill_h * 0.3, yc + pill_h * 0.3],
                transform=inch,
                color="0.55",
                lw=0.8,
            )
        )
        for k, num in enumerate((r, c)):  # row in the left half, col in the right
            fig.text(
                x + pill_w * (0.25 + 0.5 * k),
                yc,
                f"{num}",
                transform=inch,
                fontsize=14,
                weight="bold",
                ha="center",
                va="center_baseline",
            )
        fig.key["variant"] = fig.text(
            x + pill_w + 0.15,
            yc,
            tag,
            transform=inch,
            fontsize=19,
            weight="bold",
            va="center_baseline",
        )

        # top row: one table (cols 1-3), MM fits on the left and chamber details on the
        # right sharing rows; then the image crops (cols 4-6)
        def mm_cell(f, key):
            """One MM value. A Km beyond the highest [BzP] is shown as "> <highest [BzP]>"
            (display only: the fitted value is unchanged in the tables)."""
            if f is None:
                return np.nan
            if key == "km" and np.isfinite(f["km"]) and f["km"] > max(runs):
                return f"> {_fmt(max(runs))}"
            return f[key]

        fits_mm = (page["mm_raw"], page["mm"])
        mm_rows = [
            [name, *(mm_cell(f, key) for f in fits_mm)]
            for name, key in [("K$_m$ (µM)", "km"), ("v$_{max}$", "vmax"), ("r²", "r2")]
        ]
        # which runs each fit uses: all, or all but the one a conditional pass left out
        mm_rows.append(["fit", *("" if f is None else "all runs" if _left_out(f) is None
                                 else f"w/o {_fmt(_left_out(f))}" for f in fits_mm)])
        # the Km filter (r² of the raw fit, as in the Km strip): flag failing r² cells
        # and report pass/fail for the chamber
        min_r2 = page["km_min_r2"]
        fills = {
            (2, j + 1): REPORT_COLORS["fail"]
            for j, f in enumerate(fits_mm)
            if f is not None and not f["r2"] >= min_r2
        }
        raw_ok = page["mm_raw"] is not None and page["mm_raw"]["r2"] >= min_r2
        conditional = raw_ok and _left_out(page["mm_raw"]) is not None
        verdict = "conditional" if conditional else "pass" if raw_ok else "fail"
        info = [*page["info"], [f"r² ≥ {min_r2}", verdict]]
        fills[(len(info) - 1, 4)] = REPORT_COLORS[
            "warn" if conditional else "pass" if raw_ok else "fail"]
        for j, f in enumerate(fits_mm):  # a fit that left a run out: amber
            if _left_out(f) is not None:
                fills[(3, j + 1)] = REPORT_COLORS["warn"]

        # chamber details that fail a filter (red) or carry a flag (amber), in the value
        # column (col 4 of the table)
        for i, (label, cell_value) in enumerate(info):
            floor = page["info_floors"].get(label)  # e.g. the [E] floor for "GFP sum"
            if floor is not None and isinstance(cell_value, (int, float, np.number)):
                if not cell_value > floor:  # below the floor (NaN fails too)
                    fills[(i, 4)] = REPORT_COLORS["fail"]
            if label == "tile seam" and cell_value not in ("no", "", False, None):
                fills[(i, 4)] = REPORT_COLORS["warn"]
        n = max(len(mm_rows), len(info))
        mm_rows += [["", "", ""]] * (n - len(mm_rows))
        info = info + [["", ""]] * (n - len(info))
        widths = [0.74, 0.8, 0.86, 0.8, 0.88]  # inches, snug around the text ("w/o 2,500",
        #                                       "conditional")
        row_h = 0.23
        ax = box(col[0], top_a, sum(widths), row_h * (n + 1))
        fig.key["table"] = ax
        fig.key["table_widths"] = [w / sum(widths) for w in widths]
        fig.key["table_rows"] = n + 1  # header + n rows
        _grid_table(
            ax,
            ["MM fit", "raw", "per [E]", "chamber", ""],
            [m + i for m, i in zip(mm_rows, info)],
            label_cols=(0, 3),
            col_widths=[w / sum(widths) for w in widths],
            fills=fills,
        )
        ax.annotate(
            f"v$_{{max}}$ in µM/s (raw), µM/s per {enzyme} count (per [E])",
            (0, 0),
            xycoords="axes fraction",
            xytext=(0, -3),
            textcoords="offset points",
            va="top",
            fontsize=10,
            color="0.45",
        )
        # where this chamber and its variant's other chambers sit on the chip: in the gap
        # between the table and the image crops, top-aligned with them
        # The map is drawn in chamber-index space (rows x cols, evenly spaced), so it is
        # tall: as high as the image crops, and as wide as the rows:cols ratio allows.
        gap_l, gap_r = col[0] + sum(widths) + 0.1, col[3] - 0.1
        n_map_rows, n_map_cols = np.ptp(page["chip_rc"], axis=0) + 1
        map_h = P["col_w"]
        map_w = min(map_h * n_map_cols / n_map_rows, gap_r - gap_l)
        fig.key["chip"] = box((gap_l + gap_r - map_w) / 2, top_a, map_w, map_h)
        _chip_map(
            fig.key["chip"],
            page["chip_rc"],
            page["tag_rc"],
            (r, c),
            good=page["good_raw"],
        )
        for k, (label, crop, (xr, yr, radius), (lo, hi), seams) in enumerate(
            page["crops"]
        ):
            ax = box(col[3 + k], top_a, P["col_w"], P["col_w"])
            fig.key["crops"][label] = ax
            ax.key_seams = seams  # for the key: where the notches are
            ax.imshow(crop, cmap="gray", vmin=lo, vmax=hi, aspect="auto")
            ax.add_patch(plt.Circle((xr, yr), radius, ec="yellow", fc="none", lw=1))
            _seam_notches(ax, crop.shape, *seams)
            ax.set_title(label.split(",")[0], fontsize=11.5)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():  # a full black frame round each image
                sp.set_visible(True)
                sp.set_color("black")
                sp.set_linewidth(0.8)
            ax.keep_frame = True  # exempt from the page-wide top/right spine removal

            # button-QC tab (chamberqc), bottom left of the crop, outlined in black
            tab = page.get("qc_tabs", {}).get(label)
            if tab is not None:
                text, level = tab
                fig.key["tabs"][label] = ax.text(0.04, 0.04, text, transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=8.5, color="black", zorder=10,
                        bbox=dict(facecolor=REPORT_COLORS[f"tab_{level}"],
                                  edgecolor="black", linewidth=0.6, pad=1.5))

        # middle row: progress curve + rate fit per run, shared y-axis. Points are open
        # when this chamber's (raw) MM fit fails the filter, as everywhere on the page
        passes_filter = (r, c) in page["good_raw"]
        g = page["traces"]
        lo, hi = np.nanmin(g[value]), np.nanmax(g[value])
        t_max = g["seconds"].max()
        axs = []
        for k, bzp in enumerate(runs):
            ax = box(col[k], top_b, P["col_w"], h_b)
            fig.key["runs"][bzp] = ax
            if axs:
                ax.sharey(axs[0])
                ax.tick_params(labelleft=False)
            axs.append(ax)
            d = g[g["bzp_uM"] == bzp]
            ax.plot(
                d["seconds"],
                d[value],
                "o",
                ms=3,
                color="0.6",
                mfc="0.6" if passes_filter else "white",  # open: MM fit fails the filter
                mec="k" if passes_filter else "0.35",
                mew=0.3 if passes_filter else 0.7,
                zorder=3,
            )
            title = f"{bzp:g} µM"
            if bzp in fits.index:
                f = fits.loc[bzp]
                t = np.linspace(d["seconds"].min(), d["seconds"].max(), 100)
                if f["model"] == "4pl":
                    ax.plot(
                        t,
                        four_pl(t, f["a"], f["d"], f["c"], f["b"]),
                        color=FIT_COLORS["4pl"],
                        lw=1.5,
                        zorder=2,
                    )
                    yp = four_pl(f["t_peak"], f["a"], f["d"], f["c"], f["b"])
                    ax.plot(
                        t,
                        yp + f["vmax"] * (t - f["t_peak"]),
                        color=FIT_COLORS["vmax"],
                        lw=0.8,
                        zorder=2,
                    )
                elif np.isfinite(f["slope_line"]):
                    tm, ym = d["seconds"].mean(), d[value].mean()
                    ax.plot(
                        t,
                        ym + f["slope_line"] * (t - tm),
                        color=FIT_COLORS["line"],
                        lw=0.9,
                        zorder=2,
                    )
                # the rate and the model it comes from, in whichever corner the curve
                # leaves free: top-left, unless the curve starts in the upper half
                early = d.loc[d["seconds"] <= d["seconds"].quantile(0.3), value].mean()
                high = (early - lo) / (hi - lo or 1) > 0.45
                ax.key_model = f["model"]
                fig.key["insets"][bzp] = ax.text(
                    0.95 if high else 0.05,
                    0.06 if high else 0.94,
                    ("4PL" if f["model"] == "4pl" else "linear")
                    + f"\n{_fmt(f['vmax'])} µM/s",
                    transform=ax.transAxes,
                    ha="right" if high else "left",
                    va="bottom" if high else "top",
                    fontsize=10,
                    linespacing=1.15,
                    color=FIT_COLORS["vmax"] if f["model"] == "4pl" else "0.4",
                    zorder=6,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
                )
            ax.set_title(title, fontsize=11.5)
            ax.set_xlim(0, t_max * 1.04)  # every run on the same time axis, from 0
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=2, steps=[1, 2, 5, 10]))
        pad = 0.06 * (hi - lo or 1)
        axs[0].set_ylim(lo - pad, hi + pad)  # from the data: fit lines end at the edge
        axs[0].set_ylabel(_AXIS_LABELS.get(value, value))
        row_mid = (col[0] + col[5] + P["col_w"]) / 2  # centered under the progress row
        fig.text(
            row_mid / P["w"],
            1 - (top_b + h_b + 0.32) / H,
            "time (s)",
            ha="center",
            va="top",
        )

        # bottom row: the two rate-vs-[BzP] panels (square), then the key. Each panel's
        # y label sits above it (left-aligned) instead of a title, so the normalized panel
        # can sit closer.
        s_max = max(runs)

        # raw rate, from col 1
        ax = box(col[0], top_c, h_c, h_c)  # MM plots are square
        fig.key["mm_raw"] = ax
        _mm_panel(
            ax,
            s_max,
            fits.reset_index(),
            page["mm_raw"],
            page["rep_raw"],
            page["rep_raw_fits"],
            "vmax",
            good=page["good_raw"],
            this_good=(r, c) in page["good_raw"],
        )
        ax.set_title("v$_0$ (µM/s)", loc="left", fontsize=11.5)  # each run's initial rate
        fig.key["count"] = ax.text(  # how many chambers of this variant are overlaid
            0.04,
            0.96,
            f"{page['n_replicates'] + 1} {tag} chambers",
            transform=ax.transAxes,
            va="top",
            fontsize=10.5,
            zorder=6,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5),
        )

        # normalized rate, from col 3. Only the variant's other chambers that PASS the
        # filter are shown (this chamber always is). Values are shown x 10^-exp so the
        # tick labels stay short (no offset text to collide with the label).
        good_norm = page["good_norm"]

        def passing_only(df):
            if df is None or not len(df):
                return df
            keep = [(r_, c_) in good_norm for r_, c_ in zip(df["mark_row"], df["mark_col"])]
            return df[np.asarray(keep, bool)]

        nn = page["norm"]
        has_norm = nn is not None and nn[norm_col].notna().any()
        rep_norm = passing_only(page["rep_norm"])
        rep_mm = passing_only(page["rep_mm"])
        shown = [d[norm_col] for d in (nn if has_norm else None, rep_norm) if d is not None]
        top = np.nanmax(np.abs(np.concatenate(shown))) if shown and len(np.concatenate(shown)) else 0
        exp = int(np.floor(np.log10(top))) if np.isfinite(top) and top > 0 else 0
        k = 10.0 ** -exp  # multiply rates by k for display

        def scaled(df, rate_col):
            return None if df is None else df.assign(**{rate_col: df[rate_col] * k})

        this_mm = None  # this chamber's normalized MM fit, its Vmax scaled for display
        if has_norm and page["mm"] is not None:
            this_mm = page["mm"].copy()
            this_mm["vmax"] = this_mm["vmax"] * k
        ax = box(col[2], top_c, h_c, h_c)
        fig.key["mm_norm"] = ax
        _mm_panel(
            ax,
            s_max,
            scaled(nn, norm_col) if has_norm else None,
            this_mm,
            scaled(rep_norm, norm_col),
            scaled(rep_mm, "vmax"),
            norm_col,
            good=good_norm,
            this_good=(r, c) in good_norm,
        )
        # WT's Vmax / [E] (median of the passing WT chambers, as on the Vmax / [E] strip),
        # as a reference line across the normalized panel
        vm_page = page.get("vmax")
        wt_vmax = vm_page.get("wt", np.nan) if vm_page else np.nan
        if np.isfinite(wt_vmax):
            fig.key["mm_norm_wt"] = ax.axhline(wt_vmax * k, color=REPORT_COLORS["wt"], ls="--",
                                               lw=1.0, zorder=1)
            ax.text(0.98, wt_vmax * k, "WT v$_{max}$", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=9, color=REPORT_COLORS["wt"], zorder=6,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.8))
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(lo, 0), max(hi, wt_vmax * k * 1.12))
        unit = "" if exp == 0 else f"×10$^{{{exp}}}$ "
        ax.set_title(
            f"v$_0$ / {enzyme} ({unit}µM/s per count)",
            loc="left",
            fontsize=11.5,
        )
        # this chamber's NADH standard curve: its button median at every standard, with the
        # calibration line that converts its signal to [NADH] (third square, from col 5)
        if std is not None:
            ax = box(col[4], top_c, h_c, h_c)
            fig.key["standards"] = ax
            pts, line = std["points"], std["line"]
            if line is not None:
                x_line = np.array([0.0, pts["conc_uM"].max() * 1.04])
                ax.plot(x_line, line["slope"] * x_line + line["intercept"],
                        color=FIT_COLORS["line"], lw=0.9, zorder=2)
            ax.plot(pts["conc_uM"], pts[std["value"]], "o", ms=6, color=REPORT_COLORS["this"],
                    mec="k", mew=0.5, zorder=3)
            y = pts[std["value"]].to_numpy(float)
            y_pad = 0.05 * (np.nanmax(y) - np.nanmin(y) or 1)
            ax.set_xlim(0, pts["conc_uM"].max() * 1.06)
            ax.set_ylim(np.nanmin(y) - y_pad, np.nanmax(y) + y_pad)

            # inset, top left: the lowest standards (up to zoom_max), a true zoom: the
            # panel's proportions and one magnification in x and y, so the line keeps its
            # angle and its box on the panel has the inset's shape. If the low points don't
            # fit at that magnification, the inset fits its own y range instead.
            low = pts[pts["conc_uM"] <= std["zoom_max"]]
            if len(low) >= 2:
                zoom = ax.inset_axes([0.06, 0.56, 0.4, 0.4])
                fig.key["standards_zoom"] = zoom
                if line is not None:
                    zoom.plot(x_line, line["slope"] * x_line + line["intercept"],
                              color=FIT_COLORS["line"], lw=0.9, zorder=2)
                zoom.plot(low["conc_uM"], low[std["value"]], "o", ms=5,
                          color=REPORT_COLORS["this"], mec="k", mew=0.5, zorder=3)
                X0, X1 = ax.get_xlim()
                Y0, Y1 = ax.get_ylim()
                # room around the points for their labels: the window reaches past the top
                # standard, and the points sit between 18 % and 80 % of its height
                z_x1 = low["conc_uM"].max() * 1.3
                z_x0 = -0.12 * z_x1
                y_low = low[std["value"]].to_numpy(float)
                z_span = (Y1 - Y0) * (z_x1 - z_x0) / (X1 - X0)  # same magnification in y
                z_y0 = np.nanmin(y_low) - 0.18 * z_span
                if np.nanmax(y_low) > z_y0 + 0.8 * z_span:  # doesn't fit: its own y range
                    z_pad = 0.3 * (np.ptp(y_low) or 1)
                    z_y0, z_span = np.nanmin(y_low) - z_pad, np.ptp(y_low) + 2 * z_pad
                zoom.set_xlim(z_x0, z_x1)
                zoom.set_ylim(z_y0, z_y0 + z_span)
                # each point labeled with its [NADH] (µM), on the side of the line that is
                # free: upper left, except the first (at the left edge), to its right
                for i, (conc, val) in enumerate(zip(low["conc_uM"], low[std["value"]])):
                    first = i == 0
                    zoom.annotate(f"{conc:g}", (conc, val), xytext=(5, -1) if first else (-4, 3),
                                  textcoords="offset points", ha="left" if first else "right",
                                  va="center" if first else "bottom", fontsize=7.5, zorder=6)
                zoom.set_xticks([])  # the points carry their [NADH]
                zoom.yaxis.set_major_locator(plt.MaxNLocator(2))
                zoom.yaxis.tick_right()  # clear of the panel's own y labels
                zoom.tick_params(labelsize=8, length=2, pad=1.5)
                zoom.set_facecolor("white")
                for sp in zoom.spines.values():
                    sp.set_visible(True)
                    sp.set_linewidth(0.6)
                zoom.keep_frame = True  # exempt from the page-wide top/right spine removal
            ax.set_xlabel("[NADH] (µM)")
            ax.set_title("NADH standards (counts)", loc="left", fontsize=11.5)
            if line is not None:
                fig.key["standards_fit"] = ax.text(
                    0.96, 0.05,
                    f"slope {_fmt(line['slope'])} / µM\nr² {_fmt(line['r2'])}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
                    linespacing=1.2, zorder=6,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5))

        # the key: right of the normalized panel, or in a band under the row when the
        # standard curve takes that place
        if std is not None:
            key = box(col[0], top_c + h_c + 0.75, P["w"] - col[0] - 0.1, h_key - 0.25)
        else:
            key_l = col[2] + h_c + 0.45
            key = box(key_l, top_c, P["w"] - key_l - 0.1, h_c)
        fig.key["legend"] = key
        key.axis("off")
        handles = [
            plt.Line2D(
                [],
                [],
                marker="o",
                color=REPORT_COLORS["this"],
                mec="k",
                mew=0.5,
                label="this chamber",
            ),
            plt.Line2D(
                [],
                [],
                marker="o",
                ms=4,
                lw=0.8,
                color=REPORT_COLORS["replicate"],
                label=f"other {tag} chambers",
            ),
            plt.Line2D(
                [],
                [],
                marker="o",
                ls="--",
                ms=5,
                lw=0.8,
                color=REPORT_COLORS["replicate"],
                mfc="white",
                mew=1.0,
                label=f"open, dashed: r² < {page['km_min_r2']}",
            ),
            plt.Line2D([], [], color=FIT_COLORS["4pl"], lw=1.5, label="4PL fit"),
            plt.Line2D(
                [], [], color=FIT_COLORS["vmax"], lw=0.8, label="v$_0$ (4PL tangent)"
            ),
            plt.Line2D([], [], color=FIT_COLORS["line"], lw=0.9, label="line fit"),
            plt.Line2D([], [], ls="", marker="x", ms=6, mew=1.4, color=REPORT_COLORS["this"],
                       label="x: left out of the fit"),
            plt.Line2D([], [], ls="--", lw=1.0, color=REPORT_COLORS["wt"],
                       label="WT v$_{max}$ / [E] (median, passing)"),
        ]
        key.legend(
            handles=handles,
            loc="upper left",
            frameon=False,
            borderaxespad=0,
            handlelength=1.6,
            ncol=4 if std is not None else 1,  # a band under the row, or a column beside it
            columnspacing=1.2,
        )
        if km is not None and vm is not None:
            # Km (cols 1-3) and Vmax / [E] (cols 4-6) side by side, one key below both
            strip_w = col[2] + P["col_w"] - col[0]
            fig.key["km"] = box(col[0], top_d, strip_w, h_d)
            fig.key["km_data"] = km
            _km_row(fig.key["km"], km, tag, legend=False)
            fig.key["vmax_strip"] = box(col[3], top_d, strip_w, h_d)
            fig.key["vmax_data"] = vm
            _km_row(fig.key["vmax_strip"], vm, tag, name="v$_{max}$ / [E]",
                    unit=f"µM/s per {enzyme} count", log=True, legend=False)
            # no key under the strips: the gray points and off-scale triangles are in the key page
        elif km is not None:
            fig.key["km"] = box(col[0], top_d, col[3] + P["col_w"] - col[0], h_d)
            fig.key["km_data"] = km
            _km_row(fig.key["km"], km, tag)  # its own key: cols 5-6
        # header, top right: the experiment (name, library, date)
        if page.get("experiment"):
            right_edge = col[5] + P["col_w"]  # aligned with the right edge of the crops
            fig.key["experiment"] = fig.text(right_edge / P["w"], 1 - 0.28 / H,
                                             page["experiment"], ha="right", va="top",
                                             fontsize=12, color="0.35")

        # footer: report version and build date, bottom right
        fig.key["stamp"] = fig.text(1 - 0.15 / P["w"], 0.12 / H, page["stamp"], ha="right",
                                    va="bottom", fontsize=8, color="0.55")
        for a in fig.get_axes():
            if not getattr(a, "keep_frame", False):  # plots: no top/right spines
                a.spines[["top", "right"]].set_visible(False)
    return fig


def _seams_in_window(start, length, pitch):
    """Tile seams (multiples of `pitch`) strictly inside the window [start, start + length)
    of a stitched image, in window pixels. A seam at stitched pixel p is the edge between
    pixels p - 1 and p, i.e. at p - 0.5 in imshow's pixel-centered coordinates."""
    k = np.arange(np.floor(start / pitch) + 1, np.ceil((start + length) / pitch))
    return [float(seam - start - 0.5) for seam in k * pitch]


def _seam_notches(ax, shape, seams_x, seams_y, depth=0.08):
    """Mark tile seams on an image crop (drawn with imshow, origin upper left) with a pair
    of amber notches on its edges, pointing in along the seam: top and bottom for each
    vertical seam at x in `seams_x`, left and right for each horizontal seam at y in
    `seams_y`. `depth` is the notch's length as a fraction of the crop."""
    h, w = shape
    top, bottom, left, right = -0.5, h - 0.5, -0.5, w - 0.5  # the crop's edges
    style = dict(fc=REPORT_COLORS["seam"], ec="black", lw=0.6, zorder=6, clip_on=False)

    # vertical seam: a notch on the top edge pointing down, and one on the bottom pointing up
    dy, half = depth * h, depth * w / 2
    for x in seams_x:
        for edge, inward in ((top, dy), (bottom, -dy)):
            tri = [(x - half, edge), (x + half, edge), (x, edge + inward)]
            ax.add_patch(plt.Polygon(tri, **style))

    # horizontal seam: a notch on the left edge pointing right, and one on the right
    dx, half = depth * w, depth * h / 2
    for y in seams_y:
        for edge, inward in ((left, dx), (right, -dx)):
            tri = [(edge, y - half), (edge, y + half), (edge + inward, y)]
            ax.add_patch(plt.Polygon(tri, **style))

    # keep the view on the crop (the notches must not widen it)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)


def _chip_map(ax, chip_rc, tag_rc, this, good=frozenset()):
    """A small map of the chip (rows down, cols across, as in the stitched image): every
    chamber a gray dot; this variant's chambers light blue and this one dark blue. As
    everywhere on the page, a chamber whose MM fit passes the filter (in `good`) is filled
    and outlined in black; one that fails is open (white, outlined in its color)."""
    rows, cols = np.asarray(chip_rc).T
    ax.scatter(cols, rows, s=1.2, c=REPORT_COLORS["blank"], lw=0, zorder=1)

    # this variant's other chambers
    for r_, c_ in np.asarray(tag_rc).reshape(-1, 2):
        ok = (r_, c_) in good
        ax.scatter(
            [c_],
            [r_],
            s=14,
            c=REPORT_COLORS["replicate"] if ok else "white",
            ec="k" if ok else REPORT_COLORS["replicate"],
            lw=0.5 if ok else 1.0,
            zorder=2,
            clip_on=False,  # edge chambers aren't cut off by the frame
        )

    # this chamber
    ok = tuple(this) in good
    ax.scatter(
        [this[1]],
        [this[0]],
        s=34,
        c=REPORT_COLORS["this"] if ok else "white",
        ec="k" if ok else REPORT_COLORS["this"],
        lw=0.6 if ok else 1.3,
        zorder=3,
        clip_on=False,
    )
    ax.set(
        xlim=(cols.min() - 1, cols.max() + 1),
        ylim=(rows.max() + 1, rows.min() - 1),
        xticks=[],
        yticks=[],
    )
    for sp in ax.spines.values():  # no frame: the gray dots outline the chip
        sp.set_visible(False)
    ax.set_title("chip", fontsize=11.5)  # like the image-crop titles


def _km_row(ax, km, tag, *, name="K$_m$", unit="µM", log=False, legend=True):
    """Every chamber's value (Km by default) as a thin horizontal strip. The axis runs to
    km["xmax"] (for Km, the highest [S]), linear from 0, or with `log` from km["xmin"];
    values beyond it are off scale and sit at the edge as triangles (> above, < below).
    Other variants are gray: every chamber with a value, all as the same small point. This
    variant's other chambers are light blue and this chamber dark blue, both outlined in
    black and always shown; their fits below the r² filter are drawn open. The WT median
    (red) and this variant's median (dark blue, passing chambers only) are labeled dashed
    lines. Each point is (value, jitter) with the chamber's fixed jitter, so every chamber
    sits in the same place on every page. With `legend`, the key sits to the right."""
    xmax = km["xmax"]
    xmin = km.get("xmin", 0.0) if log else 0.0

    # the axis first, so the median labels below are placed against the final scale
    if log:
        ax.set_xscale("log")
        ax.set_xlim(xmin / 1.15, xmax * 1.15)
    else:
        ax.set_xlim(0, xmax * 1.03)
    ax.set_ylim(-0.32, 0.75)  # points jitter within +/- 0.2; the band above is for labels

    def placed(values):
        """Each value, or the axis edge it lies beyond, and which side ('', '>', '<')."""
        values = np.asarray(values, float)
        side = np.where(values > xmax, ">", np.where(log & ~(values >= xmin), "<", ""))
        return np.clip(values, xmin, xmax), side

    def strip(points, size, color, edge, z, label=None):
        """points: rows of (value, jitter, passes the r² filter). Fits that miss the filter
        are drawn open (white fill, colored outline)."""
        points = np.asarray(points, float).reshape(-1, 3)
        x, side = placed(points[:, 0])
        y, good = points[:, 1], points[:, 2].astype(bool)
        first = True
        for where, marker, s in [("", "o", size), (">", ">", size * 1.4), ("<", "<", size * 1.4)]:
            for ok in (True, False):
                sel = (side == where) & (good == ok)
                if not sel.any():
                    continue
                ax.scatter(
                    x[sel],
                    y[sel],
                    s=s,
                    marker=marker,
                    c=color if ok else "white",
                    ec=(edge or "none") if ok else color,
                    lw=(0.6 if edge else 0) if ok else 1.3,
                    zorder=z,
                    label=label if first else None,
                )
                first = False

    # other variants: every chamber with a value, all as the same small gray point, whether
    # its fit passes the r² filter or not
    GRAY_SIZE = 3
    others = np.asarray(km["others"], float).reshape(-1, 3)
    x_others, side_others = placed(others[:, 0])
    for where, marker, s in [("", "o", GRAY_SIZE), (">", ">", GRAY_SIZE * 1.4),
                             ("<", "<", GRAY_SIZE * 1.4)]:
        sel = side_others == where
        if sel.any():
            ax.scatter(
                x_others[sel],
                others[sel, 1],
                s=s,
                marker=marker,
                c=REPORT_COLORS["blank"],
                ec=REPORT_COLORS["blank"],  # same gray edge: small points render solid
                lw=0.4,
                zorder=1,
                label="other variants" if where == "" else None,
            )

    # Medians as dashed lines, each labeled on the plot (not in the legend), above the
    # points:
    #   WT       red, top row, just LEFT of its line;
    #   variant  dark blue, a row lower (so WT-like medians don't collide), just RIGHT of
    #            its line, unless that would run into the WT label or line: then just LEFT
    #            of its line, on a white background.
    from matplotlib.transforms import blended_transform_factory

    on_x = blended_transform_factory(ax.transData, ax.transAxes)
    renderer = ax.figure.canvas.get_renderer()

    ax.key_lines = {}  # the median lines by label, for the key page

    def median_line(value, label, color, row_y, side, white_bg=False):
        x_line = float(np.clip(value, xmin if log else -np.inf, xmax))
        ax.key_lines[label] = ax.axvline(x_line, color=color, ls="--", lw=1.4, zorder=2)
        return ax.annotate(
            label,
            (x_line, row_y),
            xycoords=on_x,
            xytext=(-3 if side == "left" else 3, 0),
            textcoords="offset points",
            ha="right" if side == "left" else "left",
            va="top",
            fontsize=10,
            color=color,
            weight="bold",
            zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5) if white_bg else None,
        )

    wt_label = None
    if np.isfinite(km["wt"]):
        wt_label = median_line(km["wt"], "WT", "#C0392B", 0.98, "left")

    if tag != "WT" and np.isfinite(km["variant_median"]):
        var_label = median_line(
            km["variant_median"], tag, REPORT_COLORS["this"], 0.80, "right"
        )
        if wt_label is not None:
            box_var = var_label.get_window_extent(renderer)
            wt_x = ax.transData.transform((float(np.clip(km["wt"], xmin if log else -np.inf, xmax)), 0))[0]
            runs_into_wt = (
                box_var.overlaps(wt_label.get_window_extent(renderer))
                or box_var.x0 <= wt_x <= box_var.x1
            )
        else:
            box_var, runs_into_wt = var_label.get_window_extent(renderer), False
        past_edge = box_var.x1 > ax.get_window_extent(renderer).x1  # e.g. off scale
        if runs_into_wt or past_edge:  # flip it to the left of its own line, on white
            var_label.set_ha("right")
            var_label.xyann = (-3, 0)
            var_label.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.5))

    strip(km["reps"], 40, REPORT_COLORS["replicate"], "k", 3, f"other {tag} chambers")
    if km["this"] is not None:
        this = km["this"]
        value = (f"> {_fmt(xmax)}" if this[0] > xmax
                 else f"< {_fmt(xmin)}" if log and not this[0] >= xmin else _fmt(this[0]))
        strip([this], 90, REPORT_COLORS["this"], "k", 4,
              f"this chamber ({value}{' ' + unit if unit else ''})")
    ax.scatter([], [], marker=">", s=20, c=REPORT_COLORS["blank"],
               label=f"{name} > {_fmt(xmax)}{' ' + unit if unit else ''} (off scale)"
               if not log else "off scale (at the edge)")
    ax.scatter([], [], s=20, c="white", ec=REPORT_COLORS["this"], lw=1.3,
               label=f"open: r² < {km['min_r2']}")
    ax.set_yticks([])
    ax.set_xlabel(f"{name} ({unit})" if unit else name)
    if log:
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.LogFormatterSciNotation())
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.spines["left"].set_visible(False)
    ax.set_title(
        f"{name}, all chambers (n={km['n_all']}; {km['n']} with r² ≥ {km['min_r2']})",
        loc="left",
    )
    if legend:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.03, 0.5),
            frameon=False,
            borderaxespad=0,
            handletextpad=0.3,
            labelspacing=0.25,
            fontsize=10,
        )

def _report_chunk(job):
    """Render a chunk of report pages into one PDF (a worker process's share). With a
    `key` payload, the key page (_report_key_page, drawn from it) is added at the end."""
    from matplotlib.backends.backend_pdf import PdfPages

    pages, out, value, norm_col, style, key = job
    plt.rcParams.update(style)  # worker processes start with matplotlib's defaults
    with PdfPages(out) as pdf:
        for page in pages:
            fig = _report_page(page, value, norm_col)
            pdf.savefig(fig)
            plt.close(fig)
        if key is not None:
            fig = _report_key_page(key, value, norm_col)
            pdf.savefig(fig)
            plt.close(fig)
    return len(pages)


# The key page: an example page, shrunk, with every element numbered. The badge color is
# used nowhere on a report page (blues, oranges, red, amber, green, gray, yellow), so a
# badge never reads as part of the page.
KEY_ACCENT = "#E6007E"
KEY_SAMPLE_DPI = 400  # the example page is embedded as an image at this resolution


def _report_key_page(page, value, norm_col):
    """The report's key: one example page (drawn from `page`, a normal page payload),
    shrunk to the upper left, with a numbered badge beside each element. The numbers are
    explained, in reading order, in a column on the right and a band below the example,
    with the conventions that hold across the whole page at the foot. Returns the figure,
    the same size as a report page."""
    import io
    import textwrap

    # ---- the example page, and where its elements are -------------------------------
    sample = _report_page(page, value, norm_col)
    sample.canvas.draw()
    renderer = sample.canvas.get_renderer()
    to_frac = sample.transFigure.inverted()
    K = sample.key
    W, H = sample.get_size_inches()

    def box_of(thing):
        """Bounding box (x0, y0, x1, y1) of an artist or axes, in figure fraction."""
        bb = thing.get_window_extent(renderer) if not isinstance(thing, plt.Axes) \
            else thing.get_window_extent(renderer)
        (x0, y0), (x1, y1) = to_frac.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        return x0, y0, x1, y1


    def title_of(ax):
        """The axes' title text, whichever position (left or center) it was set in."""
        return ax._left_title if ax._left_title.get_text() else ax.title

    def cell_box(row, col):
        """Bounding box of a table cell (row 0 = header), in figure fraction."""
        x0, y0, x1, y1 = box_of(K["table"])
        widths, n = K["table_widths"], K["table_rows"]
        cx0 = x0 + sum(widths[:col]) * (x1 - x0)
        cx1 = cx0 + widths[col] * (x1 - x0)
        cy1 = y1 - row / n * (y1 - y0)
        return cx0, cy1 - (y1 - y0) / n, cx1, cy1

    runs, fits = page["runs"], page["fits"]
    km = K.get("km_data")

    def union(*boxes):
        boxes = [b for b in boxes if b is not None]
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))

    def point(ax, x, y):
        """Data point (x, y) of `ax`, in figure fraction."""
        return tuple(to_frac.transform(ax.transData.transform((x, y))))

    def line_path(line):
        """A drawn line's points, in figure fraction, kept to its axes (what is visible)."""
        xy = line.get_transform().transform(np.asarray(line.get_xydata(), float))
        frame = line.axes.get_window_extent(renderer)
        inside = ((xy[:, 0] >= frame.x0 - 0.5) & (xy[:, 0] <= frame.x1 + 0.5)
                  & (xy[:, 1] >= frame.y0 - 0.5) & (xy[:, 1] <= frame.y1 + 0.5))
        return [tuple(p) for p in to_frac.transform(xy[inside])]

    def text_box(ax, text):
        """Bounding box of the first text on `ax` that reads `text` (None if none)."""
        hits = [t for t in ax.texts if t.get_text() == text]
        return box_of(hits[0]) if hits else None

    # Each item: (shape, title, text), in reading order. The shape is what gets outlined:
    #   ("box", bbox, corners, inset)  a rounded box; the badge on the first free spot
    #                                  (corners tl, tr, bl, br, or tc, bc: top or bottom
    #                                  center). inset > 0 draws it inside the
    #                                  element's edge (for elements that touch each other)
    #   ("ring", center, radius_in)    a circle round a single point or a small feature
    #   ("trace", points[, spots])     a highlight along a single line or curve; spots
    #                                  orders the badge's places (ur, ul: right or left
    #                                  of the upper end; br, below, above)
    # All coordinates in the example's figure fraction; sizes in key-page inches.
    items = []

    def add(shape, title, text):
        if shape is not None and shape[1] is not None and len(shape[1]):
            items.append((shape, title, text))

    n_rows = K["table_rows"]
    add(("box", box_of(K["pill"]), ("tl", "bl"), 0), "Position",
        "Row | column of the chamber on the chip (0-based; row 0 at the top).")
    add(("box", box_of(K["variant"]), ("tr", "br"), 0), "Variant",
        "The variant printed in this chamber (from the pinlist).")
    if "experiment" in K:
        add(("box", box_of(K["experiment"]), ("tl", "bl"), 0), "Experiment",
            "The experiment: its library and start date.")
    add(("box", union(cell_box(0, 0), cell_box(n_rows - 1, 2)), ("bl", "tl"), 0.02),
        "Michaelis–Menten fits",
        "raw: rate vs [BzP]; per [E]: rate ÷ GFP. K$_m$ “> 5,000” lies beyond the "
        "highest [BzP], so it is a lower bound; “–”: no K$_m$ (v$_{max}$ ended at 0). fit: the "
        "runs it uses (“w/o” one run: a conditional pass).")
    add(("box", union(cell_box(0, 3), cell_box(n_rows - 3, 4)), ("tc", "tr"), 0.02),
        "Chamber",
        "GFP and Cy5: button sums minus the BLANK median. Tile seam: whether the NADH "
        "ROI crosses a camera tile seam.")
    add(("box", union(cell_box(n_rows - 2, 4), cell_box(n_rows - 1, 4)), ("br", "tr"), 0.02),
        "Shading",
        "Red: fails a filter (r² < 0.9, or GFP at or below the [E] floor). Amber: a flag "
        "only (tile seam), or a conditional pass (r² ≥ 0.9 only with one run left out). "
        "Green: passes.")
    add(("box", union(box_of(K["chip"]), box_of(title_of(K["chip"]))), ("tl", "bl"), 0.01),
        "Chip map",
        "Where the chamber sits: this chamber dark blue, the variant's others light blue, "
        "every chamber a gray dot.")
    crops = list(K["crops"].items())
    add(("box", union(*(union(box_of(ax), box_of(title_of(ax))) for _, ax in crops)),
         ("tr", "tl"), 0), "Image crops",
        "151 px around the chamber at full resolution, one contrast per image across all "
        "pages. Yellow circle: the region measured.")
    for label, ax in reversed(crops):  # a notch on the rightmost crop with a seam
        seams_x, seams_y = ax.key_seams
        if seams_x or seams_y:
            h, w = ax.images[0].get_array().shape
            center = (point(ax, w - 0.5 - 0.04 * w, seams_y[0]) if seams_y
                      else point(ax, seams_x[0], h - 0.5 - 0.04 * h))
            add(("ring", center, 0.07), "Tile seam notches",
                "A camera tile seam runs through the crop: notches top and bottom for a "
                "vertical seam, left and right for a horizontal one.")
            break
    if K["tabs"]:
        tab = next(iter(K["tabs"].values()))
        patch = tab.get_bbox_patch()
        add(("box", box_of(patch) if patch is not None else box_of(tab), ("bl", "br"), 0),
            "Button QC tab",
            "Automated button check (chamberqc). Green “QC ok”; amber or red "
            "“anomaly Nth pct”: the button looks more unusual than N % of this "
            "chip's buttons. Unusual is not necessarily bad.")
    run_axes = list(K["runs"].values())
    add(("box", union(*(union(box_of(ax), box_of(title_of(ax))) for ax in run_axes)),
         ("tl", "bl"), 0), "Progress curves",
        "NADH per frame, bleach-corrected: one panel per [BzP] run, all on one y axis. "
        "Points are open when the chamber's MM fit fails.")
    line_runs = [b for b, ax in K["runs"].items()
                 if getattr(ax, "key_model", None) == "linear" and b in K["insets"]]
    if line_runs:
        add(("box", box_of(K["insets"][line_runs[0]]), ("br", "bl"), 0), "Rate",
            "The run's rate and the model it came from: the 4PL (orange), or the straight "
            "line (gray) whenever a 4PL rule fails.")
    pl_runs = [b for b, ax in K["runs"].items() if getattr(ax, "key_model", None) == "4pl"]
    if pl_runs:
        ax = K["runs"][pl_runs[len(pl_runs) // 2]]
        add(("trace", line_path(ax.lines[1])), "4PL fit",  # lines: points, 4PL, tangent
            "Orange: the 4PL fit; thin dark line: its steepest tangent, which is the rate. "
            "A gray line is a straight-line fit.")
    add(("box", union(box_of(K["mm_raw"]), box_of(title_of(K["mm_raw"]))), ("bl", "tl"), 0),
        "v$_0$ vs [BzP]",
        "Each run's initial rate v$_0$ (the rate on the traces above) for this chamber and "
        "the variant's other chambers, each with its Michaelis–Menten fit, whose plateau is "
        "v$_{max}$ (BLANKs not shown). ×: a point left out of its chamber's fit.")
    add(("box", box_of(K["count"]), ("tr", "br"), 0), "Replicates",
        "How many chambers of this variant are on the chip.")
    add(("box", union(box_of(K["mm_norm"]), box_of(title_of(K["mm_norm"]))), ("br", "tr"), 0),
        "v$_0$ ÷ GFP",
        "v$_0$ normalized by the chamber's [E] (GFP); only chambers whose per-[E] fit passes "
        "are shown. Dashed red: WT's v$_{max}$ / [E].")
    legend = K["legend"].get_legend()
    if legend is not None:
        add(("box", box_of(legend), ("tr", "br"), 0), "Legend",
            "The curves and markers of the panels above.")
    if km is not None:
        add(("box", union(box_of(K["km"]), box_of(title_of(K["km"]))), ("tr", "br"), 0),
            "K$_m$ of every chamber",
            "Gray: every chamber with a K$_m$; the title counts them, and those with "
            "r² ≥ 0.9. Each chamber keeps its height on every page.")
        if km["this"] is not None:
            add(("ring", point(K["km"], min(km["this"][0], km["xmax"]), km["this"][1]), 0.06),
                "This chamber's K$_m$",
                "Dark blue; the variant's other chambers light blue (open: r² < 0.9).")
        lines = getattr(K["km"], "key_lines", {})
        if "WT" in lines:
            add(("trace", line_path(lines["WT"])), "WT median",
                "Red dashed line: median K$_m$ of the WT chambers that pass.")
        if page["tag"] in lines and page["tag"] != "WT":
            add(("trace", line_path(lines[page["tag"]]), ("below", "br")), "Variant median",
                "Blue dashed line: median K$_m$ of this variant's chambers that pass.")
        add(("trace", [point(K["km"], km["xmax"], y) for y in (-0.25, 0.25)]), "Off scale",
            "Triangles at the right edge: K$_m$ above the highest [BzP].")
    if "vmax_strip" in K:
        add(("box", union(box_of(K["vmax_strip"]), box_of(title_of(K["vmax_strip"]))),
             ("tr", "br"), 0), "v$_{max}$ / [E] of every chamber",
            "The same for v$_{max}$ per [E] (normalized fits, chambers above the [E] floor), on "
            "a log axis. Where K$_m$ lies beyond the [BzP] range, v$_{max}$ is extrapolated and sits "
            "off scale at the edge.")
    add(("box", box_of(K["stamp"]), ("tl", "bl"), 0), "Version",
        "Report version and build date.")

    buffer = io.BytesIO()
    sample.savefig(buffer, format="png", dpi=KEY_SAMPLE_DPI)
    plt.close(sample)
    buffer.seek(0)
    picture = plt.imread(buffer)

    # ---- the key page ------------------------------------------------------------------
    margin, top = 0.25, 0.95          # outer margin; room for the title (in)
    img_w = 6.8                       # the example's width on the key (in)
    img_h = img_w * H / W
    img_x0, img_y0 = margin, H - top - img_h
    col_x0, col_x1 = img_x0 + img_w + 0.3, W - margin  # the right column
    badge_r = 0.1  # badge radius (in)

    with plt.rc_context(_REPORT_STYLE):
        fig = plt.figure(figsize=(W, H))
        inch = fig.dpi_scale_trans
        fig.text(margin / W, 1 - 0.3 / H, "Key: how to read a chamber page", fontsize=17,
                 weight="bold", va="top")
        fig.text(margin / W, 1 - 0.66 / H,
                 f"Example: {page['tag']}, row {page['row']} | column {page['col']}. "
                 "Every page has this layout.", fontsize=10.5, color="0.35", va="top")

        ax_img = fig.add_axes((img_x0 / W, img_y0 / H, img_w / W, img_h / H))
        ax_img.imshow(picture, interpolation="none")  # embedded as is, not resampled
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for sp in ax_img.spines.values():  # the page's edge
            sp.set_visible(True)
            sp.set_color("0.6")
            sp.set_linewidth(0.8)

        def badge(x, y, number, size=7.5):
            """A numbered badge centered at (x, y) inches."""
            fig.patches.append(plt.Circle((x, y), badge_r, transform=inch, fc=KEY_ACCENT,
                                          ec="white", lw=1.0, zorder=10))
            fig.text(x, y, str(number), transform=inch, ha="center", va="center_baseline",
                     fontsize=size, weight="bold", color="white", zorder=11)

        # each element outlined (a box, a ring round a point, or a highlight along a
        # line), with its badge on the outline. A badge that would overlap an earlier one
        # tries the shape's other spots, then moves a little further out.
        from matplotlib.patches import FancyBboxPatch

        box_pad, lw = 0.012, 0.9  # outline padding (in) and width (pt)

        def to_key(p):  # example figure fraction -> key page inches
            return img_x0 + p[0] * img_w, img_y0 + p[1] * img_h

        placed = []

        def free(spots):
            for spot in spots:
                if all(np.hypot(spot[0] - px, spot[1] - py) > 2.15 * badge_r
                       for px, py in placed):
                    return spot
            return None

        for k, ((kind, geometry, *rest), _, _) in enumerate(items, start=1):
            if kind == "box":
                corners, inset = rest
                x0, y0 = to_key(geometry[:2])
                x1, y1 = to_key(geometry[2:])
                grow = box_pad - inset
                x0, y0, x1, y1 = x0 - grow, y0 - grow, x1 + grow, y1 + grow
                fig.patches.append(FancyBboxPatch(
                    (x0, y0), x1 - x0, y1 - y0, transform=inch, fc="none", ec=KEY_ACCENT,
                    lw=lw, zorder=9, boxstyle=f"round,pad=0,rounding_size="
                    f"{min(0.04, (y1 - y0) / 2, (x1 - x0) / 2)}"))
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                # just outside each corner, diagonally, so the badge touches the outline
                # without covering what it outlines
                out = 0.75 * badge_r
                spots = {"tl": (x0 - out, y1 + out), "tr": (x1 + out, y1 + out),
                         "bl": (x0 - out, y0 - out), "br": (x1 + out, y0 - out),
                         "tc": ((x0 + x1) / 2, y1 + badge_r), "bc": ((x0 + x1) / 2, y0 - badge_r)}
                order = list(corners) + [c for c in spots if c not in corners]
                near = [spots[c] for c in order]
            elif kind == "ring":
                (radius,) = rest
                cx, cy = to_key(geometry)
                fig.patches.append(plt.Circle((cx, cy), radius, transform=inch, fc="none",
                                              ec=KEY_ACCENT, lw=lw, zorder=9))
                diag = (radius + badge_r * 0.6) / np.sqrt(2)
                near = [(cx + sx * diag, cy + sy * diag)
                        for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1))]
            else:  # trace: a translucent highlight along the line
                xy = np.array([to_key(p) for p in geometry])
                fig.add_artist(plt.Line2D(xy[:, 0], xy[:, 1], transform=inch,
                                          color=KEY_ACCENT, lw=6, alpha=0.35,
                                          solid_capstyle="round", zorder=8))
                cx, cy = xy.mean(axis=0)
                top, bottom = xy[np.argmax(xy[:, 1])], xy[np.argmin(xy[:, 1])]
                right, left = xy[np.argmax(xy[:, 0])], xy[np.argmin(xy[:, 0])]
                gap = badge_r * 1.3
                upper = top if top[1] - bottom[1] > right[0] - left[0] else right
                spots = {"ur": (upper[0] + gap, upper[1] - 0.5 * gap),  # beside the upper end
                         "ul": (upper[0] - gap, upper[1] - 0.5 * gap),
                         "br": (bottom[0] + gap, bottom[1]),
                         "below": (bottom[0], bottom[1] - gap),
                         "above": (top[0], top[1] + gap)}
                order = list(rest[0]) if rest else []
                near = [spots[c] for c in order + [c for c in spots if c not in order]]
            spot = free(near)
            if spot is None:  # every spot taken: step outward from the shape's center
                further = [(x + (x - cx) / (np.hypot(x - cx, y - cy) or 1) * badge_r * 1.2,
                            y + (y - cy) / (np.hypot(x - cx, y - cy) or 1) * badge_r * 1.2)
                           for x, y in near]
                spot = free(further) or further[0]
            placed.append(spot)
            badge(*spot, k)

        # the explanations, in number order: down the right column, then across three
        # columns under the example; the conventions that hold across the page come last.
        # If it does not all fit, the type shrinks a step.
        band_w = (img_w - 2 * 0.2) / 3
        regions = [(col_x0, col_x1, H - 0.4, 0.3)] + [  # (x0, x1, top, bottom), inches
            (img_x0 + k * (band_w + 0.2), img_x0 + k * (band_w + 0.2) + band_w,
             img_y0 - 0.3, 0.3) for k in range(3)]
        conventions_h = 1.05

        def layout(title_size, body_size):
            """Where each block goes: [(region, top, wrapped body)], or None if it
            does not fit. The last block is the conventions (body None)."""
            line_in, title_in = body_size * 1.25 / 72, title_size * 1.35 / 72
            blocks = [(t, x) for _, t, x in items] + [(None, None)]
            out, region, y = [], 0, regions[0][2]
            for title, text in blocks:
                while True:
                    x0, x1, top_r, bottom_r = regions[region]
                    chars = int((x1 - x0 - 0.3) / (body_size * 0.5 / 72))
                    body = textwrap.wrap(text, chars) if text else None
                    height = (title_in + len(body) * line_in + 0.1) if text else conventions_h
                    if y - height >= bottom_r:
                        break
                    region += 1
                    if region == len(regions):
                        return None
                    y = regions[region][2]
                out.append((region, y, body))
                y -= height
            return out

        for title_size, body_size in [(9, 8), (8.5, 7.5), (8, 7), (7.5, 6.5)]:
            flow = layout(title_size, body_size)
            if flow is not None:
                break
        title_in = title_size * 1.35 / 72
        for k, ((_, title, _), (region, y, body)) in enumerate(zip(items, flow[:-1]), start=1):
            x0 = regions[region][0]
            badge(x0 + badge_r, y - title_in / 2, k)
            fig.text(x0 + 0.3, y, title, transform=inch, fontsize=title_size, weight="bold",
                     va="top")
            fig.text(x0 + 0.3, y - title_in, "\n".join(body), transform=inch,
                     fontsize=body_size, color="0.25", va="top", linespacing=1.25)

        # the conventions that hold across the page
        region, y, _ = flow[-1]
        x0, x1 = regions[region][:2]
        conv = fig.add_axes((x0 / W, (y - conventions_h) / H, (x1 - x0) / W, conventions_h / H))
        conv.axis("off")
        handles = [
            plt.Line2D([], [], ls="", marker="o", ms=7, color=REPORT_COLORS["this"],
                       mec="k", mew=0.5, label="this chamber"),
            plt.Line2D([], [], ls="", marker="o", ms=5.5, color=REPORT_COLORS["replicate"],
                       mec="k", mew=0.5, label="the variant's other chambers"),
            plt.Line2D([], [], ls="", marker="o", ms=4, color=REPORT_COLORS["blank"],
                       label="every other chamber"),
            plt.Line2D([], [], marker="o", ms=5.5, lw=1.2, color=REPORT_COLORS["replicate"],
                       mec="k", mew=0.5, label="filled, solid: MM fit r² ≥ 0.9"),
            plt.Line2D([], [], marker="o", ms=5.5, lw=1.0, ls="--",
                       color=REPORT_COLORS["replicate"], mfc="white", mew=1.0,
                       label="open, dashed: r² < 0.9"),
        ]
        conv.legend(handles=handles, loc="upper left", ncol=1, frameon=False,
                    fontsize=body_size, handlelength=1.8, borderaxespad=0, labelspacing=0.3,
                    title="Throughout the page", alignment="left",
                    title_fontproperties=dict(weight="bold", size=title_size))
        fig.text(1 - 0.15 / W, 0.12 / H, page["stamp"], ha="right", va="bottom",
                 fontsize=8, color="0.55")
    return fig


def chamber_report_pdf(
    path,
    chambers,
    *,
    images,
    traces,
    rates,
    value="median_bc",
    mm=None,
    mm_raw=None,
    norm=None,
    norm_col=None,
    info=None,
    km_min_r2=0.9,
    info_floors=None,
    crop_px=75,
    n_jobs=None,
    version=REPORT_VERSION,
    experiment=None,
    image_qc=None,
    tile_px=None,
    key_chamber=None,
    standards=None,
    calibration=None,
    standards_value="median",
    standards_zoom_um=125,
):
    """One PDF page per chamber, for reviewing chambers one by one.

    Each page shows:
      - top row: tables (the chamber's variant, position and `info`; its MM fits; the rate
        fit in every run), then a full-resolution crop of the chamber in every image of
        `images`, with its ROI circle;
      - middle row: its progress curve in every run of `traces` (x = seconds, y = `value`),
        with the fit from `rates` (fit_4pl_rates output, keyed by bzp_uM + chamber): 4PL
        with its vmax tangent (thin solid line), or the straight line;
      - bottom row: raw vmax vs [BzP] with its MM fit from `mm_raw`, and the normalized
        rate (`norm_col` of `norm`) with its MM fit from `mm` (fit_mm_table outputs). Both
        overlay the variant's other chambers (light blue). Chambers whose MM fit has
        r² >= `km_min_r2` are drawn filled with solid curves; the rest open with dashed
        curves (raw panel: raw fits; normalized panel: normalized fits).
      - last row (with `mm_raw`): the distribution of Km over all chambers whose raw fit
        has r² ≥ `km_min_r2`, with this variant's other chambers and this chamber marked;
        the axis ends at the highest [S], and Km beyond it sits at the edge as a triangle.

    Table cells that fail a filter are shaded red: r² below `km_min_r2`, and any `info`
    column named in `info_floors` ({column: floor}, e.g. {"GFP sum": E_FLOOR}) at or below
    its floor. A "tile seam" other than "no" is shaded amber (a flag, not a filter).

    `chambers` is a list of (mark_row, mark_col). `images` is {label: (2-D image, cx, cy,
    radius)} with cx/cy the grid centers for that image; each image gets one contrast for
    all pages (from the chambers' area), so pages compare. `info` is an optional
    DataFrame indexed by (mark_row, mark_col) whose columns are printed in the text box.

    Every page carries a footer with `version` (default REPORT_VERSION) and the build date,
    and, if given, `experiment` (e.g. "hAcyP1 AFMtag Canonical · 2026-10-02") at
    the top right. `image_qc` = {image label: tabs} adds a button-QC tab to the bottom left of
    that image's crop, where tabs = {(mark_row, mark_col): (text, level)} as made by
    button_qc_tabs from chamberqc's scores.

    `tile_px` = {image label: (pitch_y, pitch_x)}, the stitched tile pitch of that image
    (e.g. stitched.attrs["tile_px"]), marks every tile seam that runs through a crop with a
    pair of amber notches on the crop's edges: top and bottom for a vertical seam, left and
    right for a horizontal one. Seams sit at multiples of the pitch, as in tile_seam_flags.

    `standards` (one row per chamber and standard: mark_row, mark_col, conc_uM and the
    `standards_value` column, e.g. the button median) adds the chamber's NADH standard curve
    as a third square on the bottom row, with its line from `calibration` (one row per
    chamber: mark_row, mark_col, slope, intercept, r2; calibrate_chambers output) and the
    line's slope and r², and an inset zoomed on the standards up to `standards_zoom_um`.
    The key then moves to a band under that row.

    The last page is a key: one chamber's page, shrunk, with each element labeled
    (_report_key_page). `key_chamber` = (mark_row, mark_col) picks the example (default:
    the first of `chambers`); it need not be among `chambers`.

    Pages are drawn in parallel across `n_jobs` processes (default: half the cores), each
    writing a chunk that is then merged in chamber order with `pdfunite` (poppler); without
    pdfunite, or with n_jobs=1, pages are drawn serially. The PDF is written under a
    temporary name and renamed when complete. Returns the number of pages."""
    import os
    import shutil
    import subprocess
    import tempfile
    from concurrent.futures import ProcessPoolExecutor

    # one contrast per image, from its chambers' area, so every page is comparable
    limits = {}
    for label, (img, cx, cy, radius) in images.items():
        y0, y1 = int(max(0, np.min(cy) - crop_px)), int(np.max(cy) + crop_px)
        x0, x1 = int(max(0, np.min(cx) - crop_px)), int(np.max(cx) + crop_px)
        limits[label] = tuple(
            np.percentile(np.asarray(img)[y0:y1:4, x0:x1:4], [1, 99.5])
        )

    # one payload per page with only what that page needs: workers never see whole images
    runs = sorted(traces["bzp_uM"].unique())
    by_chamber = dict(list(traces.groupby(["mark_row", "mark_col"])))
    # chip layout for the small map: every chamber, and each variant's chambers
    chambers_df = traces[["mark_row", "mark_col", "tag"]].drop_duplicates(
        ["mark_row", "mark_col"]
    )
    chip_rc = chambers_df[["mark_row", "mark_col"]].to_numpy()
    tag_rc = {
        t: g[["mark_row", "mark_col"]].to_numpy() for t, g in chambers_df.groupby("tag")
    }
    chip_aspect = 1.0
    if len(images):  # physical aspect (width / height) of the chamber grid
        _img, gx, gy, _r = next(iter(images.values()))
        chip_aspect = float(np.ptp(gx) / np.ptp(gy))
    fits_by = dict(list(rates.groupby(["mark_row", "mark_col"])))
    norm_by = (
        dict(list(norm.groupby(["mark_row", "mark_col"]))) if norm is not None else {}
    )
    key = ["mark_row", "mark_col"]
    mm_fits = mm.set_index(key) if mm is not None else None
    mm_raw_fits = mm_raw.set_index(key) if mm_raw is not None else None
    # which chambers' MM fits pass the filter (r² >= km_min_r2), per panel: raw fits for
    # the raw-rate panel, normalized fits for the normalized one. The rest are drawn open.
    def passing(fits):
        if fits is None:
            return frozenset()
        ok = fits[fits["r2"] >= km_min_r2]
        return frozenset(zip(ok["mark_row"], ok["mark_col"]))

    std_by = dict(list(standards.groupby(key))) if standards is not None else {}
    cal_by = calibration.set_index(key) if calibration is not None else None

    good_raw = passing(mm_raw)
    good_norm = passing(mm)
    norm_by_tag = dict(list(norm.groupby("tag"))) if norm is not None else {}
    rates_by_tag = dict(list(rates.groupby("tag")))

    def others(df, r, c):
        """Rows of `df` from chambers other than (r, c); None if there is no df."""
        if df is None:
            return None
        return df[(df["mark_row"] != r) | (df["mark_col"] != c)]

    def of_tag(df, tag):
        return None if df is None else df[df["tag"] == tag]

    # Km distribution (raw fits: Km doesn't need [E]); fits with r² ≥ km_min_r2
    km_ok = None
    if mm_raw is not None:
        km_all = mm_raw.dropna(subset=["km"]).sort_values(key).copy()
        km_xmax = max(runs)  # axis limit: the highest [S]
        # one fixed vertical jitter per chamber, so the strip is identical on every page
        km_all["jitter"] = np.random.default_rng(0).uniform(-0.2, 0.2, len(km_all))
        km_all["passes"] = km_all["r2"] >= km_min_r2
        km_ok = km_all[km_all["passes"]]  # the gray background: good fits only
        km_wt = km_ok.loc[km_ok["tag"] == "WT", "km"].median()
        cols = ["km", "jitter", "passes"]
        km_by = km_all.set_index(key)[cols]

    # Vmax / [E] distribution (the normalized fits, chambers above the [E] floor), drawn on a
    # log axis from the 1st to the 99th percentile of the passing fits whose Km lies inside
    # the [S] range. Where Km is beyond it, Vmax is extrapolated (often far off): those sit
    # at the edge as off-scale triangles instead of stretching the axis.
    vm_ok = None
    if mm is not None and len(mm):
        vm_all = mm.dropna(subset=["vmax"]).sort_values(key).copy()
        vm_all["jitter"] = np.random.default_rng(1).uniform(-0.2, 0.2, len(vm_all))
        vm_all["passes"] = vm_all["r2"] >= km_min_r2
        vm_ok = vm_all[vm_all["passes"]]
        positive = vm_ok.loc[(vm_ok["vmax"] > 0) & (vm_ok["km"] <= max(runs)), "vmax"]
        vm_xmin, vm_xmax = np.percentile(positive, [1, 99]) if len(positive) else (1e-9, 1.0)
        vm_wt = vm_ok.loc[vm_ok["tag"] == "WT", "vmax"].median()
        vm_cols = ["vmax", "jitter", "passes"]
        vm_by = vm_all.set_index(key)[vm_cols]

    from datetime import date

    built = date.today().isoformat()

    key_chamber = tuple(key_chamber) if key_chamber is not None else tuple(chambers[0])
    extra = [] if key_chamber in {tuple(rc) for rc in chambers} else [key_chamber]

    payloads = []
    for r, c in list(chambers) + extra:
        g = by_chamber.get((r, c))
        if g is None:
            continue
        crops = []
        for label, (img, cx, cy, radius) in images.items():
            xc, yc = float(cx[r, c]), float(cy[r, c])
            xi, yi = int(round(xc)), int(round(yc))
            y0, x0 = max(0, yi - crop_px), max(0, xi - crop_px)
            crop = np.asarray(
                img[y0 : yi + crop_px + 1, x0 : xi + crop_px + 1], np.float32
            )
            # tile seams through the crop, in crop pixels (none without this image's pitch)
            seams = ([], [])
            if tile_px is not None and label in tile_px:
                pitch_y, pitch_x = tile_px[label]
                seams = (
                    _seams_in_window(x0, crop.shape[1], pitch_x),  # vertical seams (x)
                    _seams_in_window(y0, crop.shape[0], pitch_y),  # horizontal seams (y)
                )
            crops.append(
                (label, crop, (xc - x0, yc - y0, radius), limits[label], seams)
            )
        tag = str(g["tag"].iloc[0])
        rep_raw = others(rates_by_tag.get(tag), r, c)
        n_rep = 0 if rep_raw is None else rep_raw[key].drop_duplicates().shape[0]
        rows = []  # variant + position are in the page title; replicates on the MM plot
        if info is not None and (r, c) in info.index:
            rows += [[name, v] for name, v in info.loc[(r, c)].items()]

        nn = norm_by.get((r, c))
        payloads.append(
            dict(
                row=r,
                col=c,
                tag=tag,
                runs=runs,
                crops=crops,
                info=rows,
                traces=g,
                fits=fits_by[(r, c)].set_index("bzp_uM")
                if (r, c) in fits_by
                else pd.DataFrame(),
                mm=None
                if mm_fits is None or (r, c) not in mm_fits.index
                else mm_fits.loc[(r, c)],
                mm_raw=None
                if mm_raw_fits is None or (r, c) not in mm_raw_fits.index
                else mm_raw_fits.loc[(r, c)],
                norm=nn.sort_values("bzp_uM") if nn is not None else None,
                n_replicates=n_rep,
                rep_raw=rep_raw,
                rep_raw_fits=others(of_tag(mm_raw, tag), r, c),
                rep_norm=others(norm_by_tag.get(tag), r, c),
                rep_mm=others(of_tag(mm, tag), r, c),
                good_raw=good_raw,
                good_norm=good_norm,
                standards=None
                if standards is None or (r, c) not in std_by
                else dict(
                    points=std_by[(r, c)].sort_values("conc_uM"),
                    value=standards_value,
                    zoom_max=standards_zoom_um,
                    line=cal_by.loc[(r, c)]
                    if cal_by is not None and (r, c) in cal_by.index
                    else None,
                ),
                info_floors=dict(info_floors or {}),
                stamp=f"chamber report v{version} · built {built}",
                experiment=experiment or "",
                # the button-QC tab for each image crop (label -> (text, level)), if any
                qc_tabs={
                    label: tabs[(r, c)]
                    for label, tabs in (image_qc or {}).items()
                    if (r, c) in tabs
                },
                chip_rc=chip_rc,
                tag_rc=tag_rc.get(tag, np.empty((0, 2))),
                chip_aspect=chip_aspect,
                km_min_r2=km_min_r2,
                km=None
                if km_ok is None
                else dict(
                    n=len(km_ok),
                    n_all=len(km_all),
                    min_r2=km_min_r2,
                    # every other variant's chamber with a Km (passing or not)
                    others=km_all.loc[km_all["tag"] != tag, cols].to_numpy(float),
                    xmax=km_xmax,
                    wt=km_wt,
                    # median Km of this variant's chambers that pass the filter
                    variant_median=km_ok.loc[km_ok["tag"] == tag, "km"].median(),
                    # this variant's chambers are always shown; poor fits drawn open
                    reps=others(km_all[km_all["tag"] == tag], r, c)[cols].to_numpy(
                        float
                    ),
                    this=km_by.loc[(r, c)].to_numpy(float)
                    if (r, c) in km_by.index
                    else None,
                ),
                vmax=None
                if vm_ok is None
                else dict(
                    n=len(vm_ok),
                    n_all=len(vm_all),
                    min_r2=km_min_r2,
                    others=vm_all.loc[vm_all["tag"] != tag, vm_cols].to_numpy(float),
                    xmin=vm_xmin,
                    xmax=vm_xmax,
                    wt=vm_wt,
                    variant_median=vm_ok.loc[vm_ok["tag"] == tag, "vmax"].median(),
                    reps=others(vm_all[vm_all["tag"] == tag], r, c)[vm_cols].to_numpy(float),
                    this=vm_by.loc[(r, c)].to_numpy(float) if (r, c) in vm_by.index else None,
                ),
            )
        )

    # the key page's example; if it was only added for the key, it gets no page of its own
    key_payload = next((p for p in payloads if (p["row"], p["col"]) == key_chamber), None)
    if extra and key_payload is not None:
        payloads.remove(key_payload)

    path = Path(path)
    partial = path.with_suffix(".partial.pdf")
    style = {
        k: v for k, v in plt.rcParams.items() if k in ("font.family", "pdf.fonttype")
    }
    # default: half the cores, so the machine stays usable while it runs
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) // 2)
    if n_jobs == 1 or not shutil.which("pdfunite") or len(payloads) < 2 * n_jobs:
        _report_chunk((payloads, partial, value, norm_col, style, key_payload))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            # contiguous chunks keep chamber order when the parts are joined
            bounds = np.linspace(0, len(payloads), n_jobs + 1).round().astype(int)
            parts = [Path(tmp) / f"part_{k:03d}.pdf" for k in range(n_jobs)]
            jobs = [
                (payloads[bounds[k] : bounds[k + 1]], parts[k], value, norm_col, style,
                 key_payload if k == n_jobs - 1 else None)  # the key goes last
                for k in range(n_jobs)
            ]
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                list(pool.map(_report_chunk, jobs))
            subprocess.run(["pdfunite", *map(str, parts), str(partial)], check=True)
    partial.replace(path)
    return len(payloads) + (key_payload is not None)
