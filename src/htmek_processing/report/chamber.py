"""Chamber reports: one PDF page per chamber, and the page key."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..analysis.plots import FIT_COLORS
from ..analysis.rates import four_pl
from ..analysis.mm import michaelis_menten


# Version of the chamber report's layout and content. Bump it whenever the report changes
# (what a page shows, how it is drawn, or a rule it displays), and add an entry to
# docs/chamber_report_changelog.md in this repo.
REPORT_VERSION = "1.10.0"

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
    """Format one table value: yes/no for booleans, thousands separators up to a million,
    ×10ⁿ (3 significant figures) for larger and for tiny ones, so every value fits its cell,
    3 significant figures otherwise."""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if not isinstance(v, (int, float, np.number)):
        return str(v)
    if not np.isfinite(v):
        return "–"
    if abs(v) >= 1e6:   # wider than a table cell with separators
        exp = int(np.floor(np.log10(abs(v))))
        return f"{v / 10**exp:.2f}×10$^{{{exp}}}$"
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


# a run marked True in this column of the rates was left out of the MM fits: drawn as an x
# on the MM panels (runs depleted early are now kept, with a forced 4PL: force_4pl_on_depleted)
EXCLUDED_COL = "left_out_of_mm"


def _excluded(df):
    """Boolean mask of `df`'s rows left out of the MM fits (EXCLUDED_COL; none without it)."""
    if df is None or EXCLUDED_COL not in df.columns:
        return np.zeros(0 if df is None else len(df), bool)
    return df[EXCLUDED_COL].fillna(False).astype(bool).to_numpy()


ZOOM_FRAMES = 4  # the zoom inset on each progress curve shows its first ZOOM_FRAMES frames


def _early_zoom(ax, d, value, f, is_4pl, lo, hi):
    """Small inset on a progress-curve panel: the run's first ZOOM_FRAMES frames with the
    model that gives its rate, the 4PL and its tangent (orange; purple when forced) or the
    straight line (gray), so the quality of the initial rate is visible. It sits at the right,
    top or bottom, whichever the curve's right half leaves free (`lo`, `hi`: the y range)."""
    tm, ym = d["seconds"].mean(), d[value].mean()  # the line's center, as on the main panel
    late = d.loc[d["seconds"] >= d["seconds"].quantile(0.55), value]
    up = ((late.mean() - lo) / (hi - lo or 1)) < 0.5  # the curve's right half is low: inset up top
    ins = ax.inset_axes([0.63, 0.6 if up else 0.05, 0.34, 0.34])
    d = d.sort_values("seconds").head(ZOOM_FRAMES)
    if len(d) < 2:
        return
    t_end = d["seconds"].iloc[-1]
    t = np.linspace(d["seconds"].iloc[0], t_end, 60)
    ins.plot(d["seconds"], d[value], "o", ms=2.6, color="0.45", mec="k", mew=0.3, zorder=3)
    if not is_4pl and np.isfinite(f.get("slope_line", np.nan)):  # the line, when it gives the rate
        ins.plot(t, ym + f["slope_line"] * (t - tm), color=FIT_COLORS["line"], lw=0.7, zorder=2)
    if is_4pl and all(np.isfinite(f.get(k, np.nan)) for k in "adcb"):  # the 4PL, when it gives the rate
        ins.plot(t, four_pl(t, f["a"], f["d"], f["c"], f["b"]), color=FIT_COLORS["4pl"], lw=1.0, zorder=2)
        tp = f["t_peak"] if np.isfinite(f["t_peak"]) else t[0]
        tp = float(np.clip(tp, t[0], t_end))
        slope = f.get("v0_unsubtracted", f["vmax"])
        slope = f["vmax"] if not np.isfinite(slope) else slope
        ins.plot(t, four_pl(tp, f["a"], f["d"], f["c"], f["b"]) + slope * (t - tp),
                 color=FIT_COLORS["vmax_forced" if f["model"] == "4pl_forced" else "vmax"],
                 lw=0.6, zorder=2)
    y = d[value].to_numpy(float)
    pad = 0.25 * (np.ptp(y) or 1)
    ins.set_xlim(t[0] - 0.08 * (t_end - t[0]), t_end * 1.05)
    ins.set_ylim(y.min() - pad, y.max() + pad)
    ins.set_xticks([])
    ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)
        sp.set_color("0.4")
    ins.set_facecolor("white")
    ins.set_zorder(5)
    ins.keep_frame = True


# a faint white box behind a label that may sit on a line
LABEL_BOX = dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.8)


def _label_standards(ax, conc, val, line, fontsize=7.5):
    """Label the standards of the zoom inset with their [NADH]: each label anchored on the
    fitted line at its concentration (not on the point, which scatters about it), all at the
    same offset to the right of and below the line, so they form an even staircase in
    ascending order on every page. Where the line is nearly flat (a poor standard curve), the
    labels go above it instead, each one line height higher, so they never overlap. Without a line, beside
    each point (see the call in _report_page)."""
    offset = np.array([5.0, -5.0]) * ax.figure.dpi / 72  # right of and below the line (px)
    if line is None:
        for c, v in zip(conc, val):
            ax.annotate(f"{c:g}", (c, v), xytext=(5, -5), textcoords="offset points", ha="left",
                        va="top", fontsize=fontsize, zorder=6, bbox=LABEL_BOX)
        return
    to_px, to_data = ax.transData.transform, ax.transData.inverted().transform
    on_line = to_px(np.c_[conc, line["slope"] * np.asarray(conc) + line["intercept"]])
    rise = np.diff(on_line[:, 1]).min() if len(conc) > 1 else 0
    step = 1.15 * fontsize * ax.figure.dpi / 72  # one line height (px)
    flat = rise < step
    spots = on_line + (offset * [1, -1] + np.c_[np.zeros(len(conc)), step * np.arange(len(conc))] if flat else offset)
    if flat:  # the stack stays inside the inset: moved down by any overshoot of its top edge
        spots[:, 1] -= max(0.0, spots[:, 1].max() + step - ax.bbox.y1)
    for spot, (c, v) in zip(spots, sorted(zip(conc, val))):
        ax.annotate(f"{c:g}", (c, v), xytext=to_data(spot), textcoords="data", ha="left",
                    va="bottom" if flat else "top", fontsize=fontsize, zorder=6, bbox=LABEL_BOX)


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
    fit left out (a conditional pass, fit_mm_one_out, or a run marked in EXCLUDED_COL) is
    drawn as an x."""
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
        out |= _excluded(reps)
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
    if this is not None and _excluded(this).any():  # runs left out of every MM fit
        out = _excluded(this)
        ax.plot(this.loc[out, "bzp_uM"], this.loc[out, y], "x", ms=7, mew=1.6,
                color=REPORT_COLORS["this"], zorder=4)
        this = this[~out]
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
_AXIS_LABELS = {"nadh_uM": "[NADH] (µM)", "median_bc": "NADH signal (a.u.)",
                "chamber_ROI_median_bc": "chamber ROI median (RFU)", "chamber_ROI_median": "chamber ROI median (RFU)"}


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
    rate_unit = page.get("rate_unit", "µM/s")
    col = [P["left"] + k * (P["col_w"] + P["gap"]) for k in range(6)]  # left edges
    km = page.get("km")  # Km strip, when raw MM fits were given
    # rows, as (top, height) in inches from the top of the page
    top_a = 0.9  # table + image crops
    top_b, h_b = top_a + P["col_w"] + 0.85, 1.15  # progress curves, clear of the crops' captions
    top_c, h_c = top_b + h_b + 0.75, 2.5  # rate-vs-[BzP] panels (square) and key
    std = page.get("standards")  # this chamber's NADH standards and calibration line, if given
    # with the standard curve as a third square, the key moves to a band under the row
    show_legend = page.get("show_legend", False)  # only on the key page's sample
    h_key = 1.2 if std is not None and show_legend else 0.0  # the legend band: 3 rows
    top_d, h_d = top_c + h_c + 0.75 + h_key, 0.66  # Km and Vmax / [E] strips (title inside)
    vm = page.get("vmax")  # the Vmax / [E] strip, when normalized fits were given
    legend_h = 0.4 if km is not None and vm is not None else 0.0  # their shared key
    kc, v0t = page.get("kcat"), page.get("v0_top")  # the second strip row, when given
    top_e = top_d + h_d + 0.6  # kcat/Km and v0 strips, under Km and Vmax / [E]
    bottom = top_e + h_d if (kc is not None or v0t is not None) else top_d + h_d
    H = (bottom + legend_h if km is not None else top_c + h_c) + 0.65  # page height

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
        n_dep = int(_excluded(fits).sum()) if len(fits) else 0  # runs left out of the MM fits

        def runs_used(f):
            if f is None:
                return ""
            out = [_fmt(_left_out(f))] if _left_out(f) is not None else []
            out += [f"{n_dep} run{'s' * (n_dep > 1)}"] if n_dep else []
            return "w/o " + ", ".join(out) if out else "all runs"

        mm_rows.append(["fit", *(runs_used(f) for f in fits_mm)])
        # the Km filter (r² of the raw fit, as in the Km strip): flag failing r² cells
        # and report pass/fail for the chamber
        min_r2 = page["km_min_r2"]
        fills = {
            (2, j + 1): REPORT_COLORS["warn"]  # low r²: a flag (Km not set), not a fail
            for j, f in enumerate(fits_mm)
            if f is not None and not f["r2"] >= min_r2
        }
        raw_ok = page["mm_raw"] is not None and page["mm_raw"]["r2"] >= min_r2
        conditional = raw_ok and _left_out(page["mm_raw"]) is not None
        # a low r² is a flag, not a fail: the fit can still bound Vmax (amber)
        verdict = "conditional" if conditional else "pass" if raw_ok else "low"
        info = [*page["info"], [f"r² ≥ {min_r2}", verdict]]
        fills[(len(info) - 1, 4)] = REPORT_COLORS["pass" if raw_ok and not conditional else "warn"]
        # overall: the chamber fails only if its enzyme readout is not above its floor (every
        # `info_floors` entry, e.g. the GFP floor); a raw MM fit below the r² filter leaves
        # its Vmax ("vmax only"), not its Km
        e_ok = all(
            isinstance(v, (int, float, np.number)) and v > page["info_floors"][lab]
            for lab, v in page["info"] if lab in page["info_floors"]
        )
        # a chamber flagged manually (an `info` column "manual flag", e.g. "delam bridge") fails
        manual = next((v for lab, v in page["info"] if lab == "manual flag"), "")
        flagged = isinstance(manual, str) and manual not in ("", "no")
        overall = ("flagged" if flagged else "fail" if not e_ok else "vmax only" if not raw_ok
                   else "conditional" if conditional else "pass")
        info.append(["overall", overall])
        fills[(len(info) - 1, 4)] = REPORT_COLORS[
            {"pass": "pass", "conditional": "warn", "vmax only": "warn", "fail": "fail",
             "flagged": "fail"}[overall]]
        for j, f in enumerate(fits_mm):  # a fit that left a run out: amber
            if f is not None and (_left_out(f) is not None or n_dep):
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
            if label == "manual flag" and flagged:
                fills[(i, 4)] = REPORT_COLORS["fail"]
        n = max(len(mm_rows), len(info))
        mm_rows += [["", "", ""]] * (n - len(mm_rows))
        info = info + [["", ""]] * (n - len(info))
        widths = [0.74, 0.8, 0.95, 0.88, 0.8]  # inches, snug around the text ("w/o 2,500",
        #                                       "conditional"); the chamber labels get the room
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
        # a manual flag's label and reason are longer than the others: smaller, so they fit
        for i, (label, _) in enumerate(info):
            if label == "manual flag":
                ax.tables[0][i + 1, 3].get_text().set_fontsize(9.5)
                if flagged:
                    ax.tables[0][i + 1, 4].get_text().set_fontsize(8)
        ax.annotate(
            f"v$_{{max}}$ in {rate_unit} (raw), {rate_unit} per {enzyme} RFU (per [E])",
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
            caption = page.get("image_captions", {}).get(label)
            if caption:  # e.g. the source acquisition, under the crop
                ax.text(0.5, -0.03, caption, transform=ax.transAxes, ha="center", va="top",
                        fontsize=6.5, color="black")

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
        v = g[value].to_numpy(float)
        # a chamber with no finite signal (e.g. its ROI off the image) still gets a page
        lo, hi = (np.nanmin(v), np.nanmax(v)) if np.isfinite(v).any() else (0.0, 1.0)
        t_max = np.nanmax(g["seconds"]) if np.isfinite(g["seconds"]).any() else 1.0
        axs = []
        # one panel per run across the grid's width: the grid's columns for up to 6 runs,
        # narrower panels (same gap) for more
        if len(runs) <= 6:
            run_x, run_w = col[: len(runs)], P["col_w"]
        else:
            span_w = col[5] + P["col_w"] - col[0]
            run_w = (span_w - (len(runs) - 1) * P["gap"]) / len(runs)
            run_x = [col[0] + k * (run_w + P["gap"]) for k in range(len(runs))]
        for k, bzp in enumerate(runs):
            ax = box(run_x[k], top_b, run_w, h_b)
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
                is_4pl = f["model"] in ("4pl", "4pl_forced")
                if is_4pl:
                    ax.plot(
                        t,
                        four_pl(t, f["a"], f["d"], f["c"], f["b"]),
                        color=FIT_COLORS["4pl"],
                        lw=1.5,
                        zorder=2,
                    )
                    # the tangent where the rate was read: the 4PL's steepest point, or for a
                    # forced 4PL whose peak is before the run (or b <= 1), the first frame; its
                    # slope is the fitted one, before any BLANK subtraction
                    tp = f["t_peak"] if np.isfinite(f["t_peak"]) else t[0]
                    tp = float(np.clip(tp, t[0], t[-1]))
                    slope = f.get("v0_unsubtracted", f["vmax"])
                    slope = f["vmax"] if not np.isfinite(slope) else slope
                    yp = four_pl(tp, f["a"], f["d"], f["c"], f["b"])
                    ax.plot(
                        t,
                        yp + slope * (t - tp),
                        color=FIT_COLORS["vmax_forced" if f["model"] == "4pl_forced" else "vmax"],
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
                # the fitted endpoint, net of the run's background (the BLANKs' median at the
                # first frame): the fit's value at the end of the run, labeled at the
                # right edge just below the last points, or above them when the curve ends low
                # (in the trace's units)
                t_end = d["seconds"].max()
                if is_4pl:
                    y_end = four_pl(t_end, f["a"], f["d"], f["c"], f["b"])
                elif np.isfinite(f["slope_line"]):
                    y_end = d[value].mean() + f["slope_line"] * (t_end - d["seconds"].mean())
                else:
                    y_end = np.nan
                if np.isfinite(y_end):
                    unit = "µM" if value.startswith("nadh_uM") else "RFU"
                    net = y_end - page.get("background0", {}).get(bzp, 0.0)  # above the run's background
                    # below the points, unless that runs into the x-axis: then just above them
                    span = (hi - lo) or 1
                    low_end = (y_end - (lo - 0.06 * span)) / (1.12 * span) < 0.18
                    fig.key.setdefault("endpoints", {})[bzp] = ax.annotate(
                        f"{'+' if net >= 0 else '−'}{_fmt(abs(net))} {unit}", (1, y_end),
                        xycoords=("axes fraction", "data"),
                        xytext=(-1, 5 if low_end else -5), textcoords="offset points", ha="right",
                        va="bottom" if low_end else "top",
                        fontsize=7, zorder=6, annotation_clip=False,
                        color={"4pl": FIT_COLORS["vmax"], "4pl_forced": FIT_COLORS["vmax_forced"]}.get(
                            f["model"], "0.4"),
                    )
                # the rate on its own line just under the run's [S], colored by the model it
                # comes from (4PL, forced 4PL or line, as in the legend)
                ax.key_model = f["model"]
                _early_zoom(ax, d, value, f, is_4pl, lo, hi)
                fig.key["insets"][bzp] = ax.annotate(
                    f"{_fmt(f['vmax'])} {rate_unit}", (0, 1), xycoords="axes fraction",
                    xytext=(0, 3), textcoords="offset points", ha="left", va="bottom",
                    fontsize=8.5,
                    color={"4pl": FIT_COLORS["vmax"], "4pl_forced": FIT_COLORS["vmax_forced"]}.get(
                        f["model"], "0.4"),
                )
            ax.set_title(title, loc="left", fontsize=10.5 if len(runs) > 6 else 11, pad=15)  # the run's [S]
            ax.set_xlim(0, t_max * 1.04)  # every run on the same time axis, from 0
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=2, steps=[1, 2, 5, 10]))
            # the unit on the ticks ("1000 s"): no separate time label under the row
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g} s" if v else "0"))
        pad = 0.06 * (hi - lo or 1)
        axs[0].set_ylim(lo - pad, hi + pad)  # from the data: fit lines end at the edge
        axs[0].set_ylabel(_AXIS_LABELS.get(value, value))

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
        blank = page.get("blank_curve") or {}
        if blank:   # the BLANKs' median rate per run, under everything else
            ax.plot(list(blank), list(blank.values()), "s-", ms=3.5, lw=1.0, color="0.6", mec="0.35",
                    mew=0.4, zorder=1.5)
        ax.set_title(f"v$_0$ ({rate_unit})", loc="left", fontsize=11.5)  # each run's initial rate
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
            f"v$_0$ / {enzyme} ({unit}{rate_unit} per RFU)",
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
                z_x1 = low["conc_uM"].max() * 1.6  # room right of the top standard for its label
                z_x0 = -0.12 * z_x1
                y_low = low[std["value"]].to_numpy(float)
                z_span = (Y1 - Y0) * (z_x1 - z_x0) / (X1 - X0)  # same magnification in y
                z_y0 = np.nanmin(y_low) - 0.18 * z_span
                if np.nanmax(y_low) > z_y0 + 0.8 * z_span:  # doesn't fit: its own y range
                    z_pad = 0.3 * (np.ptp(y_low) or 1)
                    z_y0, z_span = np.nanmin(y_low) - z_pad, np.ptp(y_low) + 2 * z_pad
                zoom.set_xlim(z_x0, z_x1)
                zoom.set_ylim(z_y0, z_y0 + z_span)
                # each point labeled with its [NADH] (µM): first on the side of the line away
                # from the point (above it: upper left; below: lower right), else the first
                # of a few spots that stays inside the inset and overlaps no earlier label
                # or point, so close standards (e.g. 31.25 and 62.5) never collide
                _label_standards(zoom, low["conc_uM"].to_numpy(float), y_low, line)
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
            ax.set_title("NADH standards (RFU)", loc="left", fontsize=11.5)
            if line is not None:
                fig.key["standards_fit"] = ax.text(
                    0.96, 0.05,
                    f"slope {_fmt(line['slope'])} RFU/µM\nr² {_fmt(line['r2'])}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
                    linespacing=1.2, zorder=6,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5))

        # the key: right of the normalized panel, or in a band under the row when the
        # standard curve takes that place. Chamber pages draw none: the key page's sample does
        if show_legend and std is not None:
            key = box(col[0], top_c + h_c + 0.75, P["w"] - col[0] - 0.1, h_key - 0.25)
        elif show_legend:
            key_l = col[2] + h_c + 0.45
            key = box(key_l, top_c, P["w"] - key_l - 0.1, h_c)
        if show_legend:
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
                ls="--",
                ms=5,
                lw=0.8,
                color=REPORT_COLORS["replicate"],
                mfc="white",
                mew=1.0,
                label=f"open, dashed: r² < {page['km_min_r2']}",
            ),
            plt.Line2D(
                [], [], color=FIT_COLORS["vmax"], lw=0.8, label="v$_0$ (4PL tangent)"
            ),
            plt.Line2D([], [], color=FIT_COLORS["vmax_forced"], lw=0.8,
                       label="v$_0$ (forced 4PL: depleted early)"),
            plt.Line2D([], [], color=FIT_COLORS["line"], lw=0.9, label="line fit"),
            plt.Line2D([], [], ls="", marker="x", ms=6, mew=1.4, color=REPORT_COLORS["this"],
                       label="x: left out of the fit"),
            plt.Line2D([], [], ls="--", lw=1.0, color=REPORT_COLORS["wt"],
                       label="WT v$_{max}$ / [E] (median, passing)"),
        ]
        if show_legend and page.get("blank_curve"):
            handles.append(plt.Line2D([], [], marker="s", ms=3.5, lw=1.0, color="0.6", mec="0.35", mew=0.4,
                                      label="BLANK median v$_0$"))
        if show_legend:
            key.legend(
                handles=handles,
                loc="upper left",
                frameon=False,
                borderaxespad=0,
                handlelength=1.6,
                ncol=3 if std is not None else 1,  # a band under the row, or a column beside it
                columnspacing=1.2,
            )
        if km is not None and vm is not None:
            # Km (cols 1-3) and Vmax / [E] (cols 4-6) side by side, one key below both
            strip_w = col[2] + P["col_w"] - col[0]
            fig.key["km"] = box(col[0], top_d, strip_w, h_d)
            fig.key["km_data"] = km
            _km_row(fig.key["km"], km, tag, log=2, legend=False)
            fig.key["vmax_strip"] = box(col[3], top_d, strip_w, h_d)
            fig.key["vmax_data"] = vm
            _km_row(fig.key["vmax_strip"], vm, tag, name="v$_{max}$ / [E]",
                    unit=f"{rate_unit} per {enzyme} RFU", log=True, legend=False)
            # second row: kcat/Km (per [E]) under Km, v0 at the highest [S] under Vmax / [E]
            if kc is not None:
                fig.key["kcat_strip"] = box(col[0], top_e, strip_w, h_d)
                _km_row(fig.key["kcat_strip"], kc, tag, name="k$_{cat}$/K$_m$",
                        unit=page.get("kcat_unit") or f"{rate_unit} per {enzyme} RFU per µM", log=True,
                        legend=False, note=page.get("kcat_note"))
            if v0t is not None:
                fig.key["v0_strip"] = box(col[3], top_e, strip_w, h_d)
                _km_row(fig.key["v0_strip"], v0t, tag, name=f"v$_0$ at {page['s_top']:g} µM",
                        unit=rate_unit, log=True, legend=False, sci=False)
            # no key under the strips: the gray points and off-scale triangles are in the key page
        elif km is not None:
            fig.key["km"] = box(col[0], top_d, col[3] + P["col_w"] - col[0], h_d)
            fig.key["km_data"] = km
            _km_row(fig.key["km"], km, tag, log=2)  # its own key: cols 5-6
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


def _log_minor_ticks(axis):
    """Unlabeled minor ticks on a log axis, at 2-9 x each power of 10 (on base-2 axes too, so
    the minor ticks mark round numbers whatever the major ticks are)."""
    ticker = plt.matplotlib.ticker
    axis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
    axis.set_minor_formatter(ticker.NullFormatter())


def _km_row(ax, km, tag, *, name="K$_m$", unit="µM", log=False, legend=True, note=None, sci=True):
    """Every chamber's value (Km by default) as a thin horizontal strip. The axis runs to
    km["xmax"] (for Km, the highest [S]), linear from 0, or with `log` from km["xmin"]
    (log=True: base 10; log=2: base 2, ticked at round numbers, 1, 2 and 5 per decade);
    values beyond it are off scale and sit at the edge as triangles (> above, < below).
    Other variants are gray: every chamber with a value, all as the same small point. This
    variant's other chambers are light blue and this chamber dark blue, both outlined in
    black and always shown; their fits below the r² filter are drawn open. The WT median
    (red) and this variant's median (dark blue, passing chambers only) are labeled dashed
    lines. Each point is (value, jitter) with the chamber's fixed jitter, so every chamber
    sits in the same place on every page. With `legend`, the key sits to the right. `note`
    is a small grey caveat under the strip's right end, e.g. that its unit rests on a
    placeholder. `sci`=False labels a log axis's ticks as plain numbers (0.1, 1, 10)."""
    xmax = km["xmax"]
    xmin = km.get("xmin", 0.0) if log else 0.0

    # the axis first, so the median labels below are placed against the final scale
    if log == 2:
        ax.set_xscale("log", base=2)
        ax.set_xlim(xmin / 1.15, xmax * 1.15)
        # round numbers (1, 2, 5 per decade) inside the [S] range, not each [S]
        decades = 10.0 ** np.arange(np.floor(np.log10(xmin)), np.ceil(np.log10(xmax)) + 1)
        ticks = [t for t in np.outer(decades, (1, 2, 5)).ravel() if xmin / 1.15 <= t <= xmax * 1.15]
        ax.set_xticks(ticks, [f"{t:,g}" for t in ticks])
        _log_minor_ticks(ax.xaxis)
    elif log:
        ax.set_xscale("log")
        ax.set_xlim(xmin / 1.15, xmax * 1.15)
        _log_minor_ticks(ax.xaxis)
    else:
        ax.set_xlim(0, xmax * 1.03)
    # points jitter within +/- 0.2; above them, one row for the median labels, then the title
    ax.set_ylim(-0.27, 0.73)

    def placed(values):
        """Each value, or the axis edge it lies beyond, and which side ('', '>', '<')."""
        values = np.asarray(values, float)
        side = np.where(values > xmax, ">", np.where(bool(log) & ~(values >= xmin), "<", ""))
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
    #   WT       red, just LEFT of its line;
    #   variant  dark blue, in the same row, just RIGHT of its line, unless that would run
    #            into the WT label or line: then just LEFT of its line, on a white background.
    from matplotlib.transforms import blended_transform_factory

    on_x = blended_transform_factory(ax.transData, ax.transAxes)
    renderer = ax.figure.canvas.get_renderer()

    ax.key_lines = {}  # the median lines by label, for the key page

    def median_line(value, label, color, row_y, side, white_bg=False):
        x_line = float(np.clip(value, xmin if log else -np.inf, xmax))
        # up to the label row, clear of the title above it
        ax.key_lines[label] = ax.axvline(x_line, ymax=row_y, color=color, ls="--", lw=1.4, zorder=2)
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
        wt_label = median_line(km["wt"], "WT", "#C0392B", 0.72, "left")

    if tag != "WT" and np.isfinite(km["variant_median"]):
        var_label = median_line(
            km["variant_median"], tag, REPORT_COLORS["this"], 0.72, "right"
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
    ax.tick_params(axis="x", which="major", length=8, width=1.1)  # longer than the default: readable
    ax.tick_params(axis="x", which="minor", length=4.5, width=0.9)  # at the strip's small size
    ax.set_xlabel(unit)  # the name is the strip's title
    if log and log != 2:  # base 2 keeps its round-number tick labels
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.LogFormatterSciNotation() if sci else
                                     plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,g}"))
        _log_minor_ticks(ax.xaxis)
    ax.spines["left"].set_visible(False)
    # the measurement inside the strip's top row, at the left, above the median labels
    ax.set_title(name, loc="left", fontsize=12, y=0.97, va="top", x=0.008)
    if note:
        ax.annotate(note, (1.0, 0), xycoords="axes fraction", xytext=(0, -24), textcoords="offset points",
                    ha="right", va="top", fontsize=8, style="italic", color="#7F7F7F")
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
    `key` payload, the key page (_report_key_page, drawn from it) is added at the end; with
    a `lead` payload (the flowchart's steps and labels), the flowchart page goes first."""
    from matplotlib.backends.backend_pdf import PdfPages

    pages, out, value, norm_col, style, key, lead = job
    plt.rcParams.update(style)  # worker processes start with matplotlib's defaults
    with PdfPages(out) as pdf:
        if lead is not None:
            from .flowchart import flowchart_pages

            for fig in flowchart_pages(**lead):
                pdf.savefig(fig)
                plt.close(fig)
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
    sample = _report_page(dict(page, show_legend=True), value, norm_col)
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
    legend = K["legend"].get_legend() if "legend" in K else None
    if legend is not None:
        add(("box", box_of(legend), ("tr", "br"), 0), "Legend",
            "The curves and markers of the panels above (shown here only, not on the chamber pages).")
    if km is not None:
        add(("box", union(box_of(K["km"]), box_of(title_of(K["km"]))), ("tr", "br"), 0),
            "K$_m$ of every chamber",
            "Gray: every chamber with a K$_m$; filled where its fit has r² ≥ 0.9. Each "
            "chamber keeps its height on every page.")
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
    standards_zoom_um=62.5,
    rate_unit="µM/s",
    image_captions=None,
    image_pct_high=None,
    steps=None,
    enzyme_rfu_per_nm=None,
    enzyme_placeholder=False,
):
    """One PDF page per chamber, for reviewing chambers one by one.

    The first page is a flowchart of the filters (report.flowchart): `steps`, the pipeline's
    record of its own steps (default: analysis.steps.pipeline_steps(), what this session ran
    or loaded with load_steps), then the report's own (the MM fits' r² filter, the enzyme
    floor). With no recorded steps and no MM fits there is no flowchart page.

    `enzyme_rfu_per_nm` converts the enzyme readout to [E] (e.g. GFP button sum RFU per nM
    enzyme), so the kcat/Km strip reads M⁻¹ s⁻¹ (with the rate in µM/s and the normalized
    rate per RFU): kcat/Km = vmax / Km × enzyme_rfu_per_nm × 10⁹. `enzyme_placeholder`
    marks that conversion as a placeholder (not a calibration) on every page's strip and on
    the flowchart.

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

    `rate_unit` labels every rate on the page: "µM/s" once the signal is calibrated to [NADH]
    (the default, MBO-040), "RFU/s" while it is still raw. The normalized rate is labeled
    `rate_unit` per RFU of the enzyme readout.

    `image_captions` = {image label: text} prints a small caption under that crop, e.g.
    the acquisition it came from. Each image's contrast runs from the 1st to the 99.5th
    percentile of its chambers' area; `image_pct_high` = {image label: percentile} lowers
    the upper limit (more gain), e.g. for dim GFP buttons.

    The table's "overall" cell fails the chamber when any `info_floors` value is not above
    its floor, and reads "flagged" (red) when an `info` column "manual flag" holds a reason
    (apply_manual_flags); a raw MM fit below r² >= `km_min_r2` reads "vmax only" (amber): its Km is
    not determined, but its Vmax can still be reported.

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
            np.percentile(np.asarray(img)[y0:y1:4, x0:x1:4], [1, (image_pct_high or {}).get(label, 99.5)])
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
    # the BLANK chambers' median rate per run: the floor a rate has to clear, on the v0 panel
    blank_curve = (rates[rates["tag"] == "BLANK"].groupby("bzp_uM")["vmax"].median().sort_index().to_dict()
                   if "tag" in rates and (rates["tag"] == "BLANK").any() else {})
    # each run's background at its first frame: the BLANKs' median signal (e.g. free phosphate in
    # the mix, turned to NADH by the coupling enzymes before the first image); endpoints are
    # labeled net of it
    first = traces.groupby("bzp_uM")["frame"].transform("min") == traces["frame"]
    background0 = (traces[first & (traces["tag"] == "BLANK")].groupby("bzp_uM")[value].median().to_dict()
                   if "tag" in traces else {})
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
        km_xmin, km_xmax = min(runs), max(runs)  # axis limits: the lowest and highest [S]
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

    def strip_table(d, col, seed, passes):
        """One row per chamber with a positive `col`: value, its fixed jitter, and whether it
        passes (drawn filled); the axis spans the passing values' 1st-99th percentile."""
        d = d[key + ["tag", col]].dropna(subset=[col])
        d = d[d[col] > 0].sort_values(key).copy()
        d["jitter"] = np.random.default_rng(seed).uniform(-0.2, 0.2, len(d))
        d["passes"] = passes(d)
        ok = d[d["passes"]]
        lim = np.percentile(ok[col], [1, 99]) if len(ok) else (1e-9, 1.0)
        return d, ok, lim

    # kcat/Km (per [E]): Vmax / Km of the normalized fits, set even where Km is beyond the
    # [S] range (it is the initial slope); filled where that fit passes the r² filter
    kcat_scale, kcat_unit, kcat_note = 1.0, None, None
    if enzyme_rfu_per_nm is not None:   # µM/s per RFU / µM -> M⁻¹ s⁻¹ at [E] = RFU / enzyme_rfu_per_nm nM
        kcat_scale, kcat_unit = enzyme_rfu_per_nm * 1e9, "M$^{-1}$ s$^{-1}$"
        if enzyme_placeholder:
            kcat_note = f"Placeholder slope: {enzyme_rfu_per_nm / 1e3:,.0f}K RFU/nM"
    kc_data = None
    if mm is not None and len(mm):
        kc_src = mm.assign(kcat_km=mm["vmax"] / mm["km"] * kcat_scale)
        kc_r2 = mm.set_index(key)["r2"]
        kc_data = strip_table(kc_src, "kcat_km", 2,
                              lambda d: d.set_index(key).index.map(kc_r2).to_numpy() >= km_min_r2)
    # v0 at the highest [S] (raw rates): the measured rate, not a fit; filled where the
    # chamber's raw MM fit passes the r² filter
    v0_data = None
    if "vmax" in rates and mm_raw is not None:
        raw_r2 = mm_raw.set_index(key)["r2"]
        v0_data = strip_table(rates[rates["bzp_uM"] == max(runs)].rename(columns={"vmax": "v0_top"}),
                              "v0_top", 3,
                              lambda d: d.set_index(key).index.map(raw_r2).to_numpy() >= km_min_r2)

    def strip_page(data, col, tag, r, c):
        """The strip dict _km_row reads, for chamber (r, c) of variant `tag`."""
        if data is None:
            return None
        d, ok, (lo, hi) = data
        cols = [col, "jitter", "passes"]
        by = d.set_index(key)[cols]
        return dict(n=len(ok), n_all=len(d), min_r2=km_min_r2,
                    others=d.loc[d["tag"] != tag, cols].to_numpy(float), xmin=lo, xmax=hi,
                    wt=ok.loc[ok["tag"] == "WT", col].median(),
                    variant_median=ok.loc[ok["tag"] == tag, col].median(),
                    reps=others(d[d["tag"] == tag], r, c)[cols].to_numpy(float),
                    this=by.loc[(r, c)].to_numpy(float) if (r, c) in by.index else None)

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
                rate_unit=rate_unit,
                blank_curve=blank_curve,
                background0=background0,
                image_captions=image_captions or {},
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
                    xmin=km_xmin,
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
                kcat=strip_page(kc_data, "kcat_km", tag, r, c),
                kcat_unit=kcat_unit,
                kcat_note=kcat_note,
                v0_top=strip_page(v0_data, "v0_top", tag, r, c),
                s_top=max(runs),
            )
        )

    # the key page's example; if it was only added for the key, it gets no page of its own
    key_payload = next((p for p in payloads if (p["row"], p["col"]) == key_chamber), None)
    if extra and key_payload is not None:
        payloads.remove(key_payload)

    # the flowchart page: the pipeline's recorded steps, then the report's own
    from ..analysis.steps import pipeline_steps
    from .flowchart import report_steps

    flow = list(pipeline_steps() if steps is None else steps) + report_steps(
        rates=rates, mm_raw=mm_raw, mm=mm, norm=norm, km_min_r2=km_min_r2,
        info_floors=info_floors, rate_unit=rate_unit, enzyme_rfu_per_nm=enzyme_rfu_per_nm,
        enzyme_placeholder=enzyme_placeholder)
    lead = dict(steps=flow, experiment=experiment or "",
                stamp=f"chamber report v{version} · built {built}") if flow else None

    path = Path(path)
    partial = path.with_suffix(".partial.pdf")
    style = {
        k: v for k, v in plt.rcParams.items() if k in ("font.family", "pdf.fonttype")
    }
    # default: half the cores, so the machine stays usable while it runs
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) // 2)
    if n_jobs == 1 or not shutil.which("pdfunite") or len(payloads) < 2 * n_jobs:
        _report_chunk((payloads, partial, value, norm_col, style, key_payload, lead))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            # contiguous chunks keep chamber order when the parts are joined
            bounds = np.linspace(0, len(payloads), n_jobs + 1).round().astype(int)
            parts = [Path(tmp) / f"part_{k:03d}.pdf" for k in range(n_jobs)]
            jobs = [
                (payloads[bounds[k] : bounds[k + 1]], parts[k], value, norm_col, style,
                 key_payload if k == n_jobs - 1 else None,  # the key goes last
                 lead if k == 0 else None)  # the flowchart first
                for k in range(n_jobs)
            ]
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                list(pool.map(_report_chunk, jobs))
            subprocess.run(["pdfunite", *map(str, parts), str(partial)], check=True)
    partial.replace(path)
    n_flow = len({st.get("stage", "rates") for st in flow}) if lead is not None else 0
    return len(payloads) + (key_payload is not None) + n_flow
