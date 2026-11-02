"""The chamber report's first page: a flowchart of every filtering step the pipeline ran,
drawn from the steps the functions recorded (analysis.steps), with their rules and counts."""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from .chamber import REPORT_COLORS, _REPORT_STYLE

# geometry (inches)
_BOX_W, _BRANCH_W, _BRANCH_GAP = 3.05, 1.6, 0.22
_LINE, _HEAD, _PAD, _V_GAP = 0.175, 0.3, 0.08, 0.32
_RULE_SIZE, _TITLE_SIZE, _BRANCH_SIZE = 9, 10.5, 9
_LEVEL_FILL = {"fail": REPORT_COLORS["fail"], "warn": REPORT_COLORS["warn"],
               "pass": REPORT_COLORS["pass"], "info": REPORT_COLORS["label"]}


def report_steps(*, rates=None, mm_raw=None, mm=None, norm=None, km_min_r2=0.9, info_floors=None,
                 left_out="left_out_of_mm", rate_unit="µM/s", enzyme_rfu_per_nm=None, enzyme_placeholder=False):
    """The steps the chamber report itself applies, in the flowchart's format: the raw MM
    fits and their r² filter, the enzyme floor, and the normalized MM fits."""
    steps = []

    def chambers(df):
        return df[["mark_row", "mark_col"]].drop_duplicates().shape[0]

    if mm_raw is not None:
        ok = (mm_raw["r2"] >= km_min_r2).sum()
        rule = ["Michaelis-Menten fit of `v0` vs [S]"]
        if rates is not None and left_out in rates:
            rule.append(f"(without the {int(rates[left_out].sum()):,} runs left out)")
        rule.append(f"Km reported when r² ≥ {km_min_r2:g}")
        steps.append(dict(step="report_mm_raw", stage="rates", title="MM fits, raw v0", rule=rule,
                          unit="chambers", n=len(mm_raw),
                          branches=[dict(text=f"r² ≥ {km_min_r2:g}: Km, Vmax", n=ok, level="pass"),
                                    dict(text=f"r² < {km_min_r2:g}: Vmax only", n=len(mm_raw) - ok,
                                         level="warn")]))
    if norm is not None and rates is not None:
        n_all, n_norm = chambers(rates), chambers(norm)
        rule = [f"`{name}` > {floor:,.0f}" for name, floor in (info_floors or {}).items()] or ["enzyme above its floor"]
        steps.append(dict(step="report_enzyme_floor", stage="rates", title="Enzyme floor", rule=rule + ["below it: no rate per enzyme"],
                          unit="chambers", n=n_all,
                          branches=[dict(text="below the floor", n=n_all - n_norm, level="fail")]))
    if mm is not None:
        ok = (mm["r2"] >= km_min_r2).sum()
        steps.append(dict(step="report_mm_norm", stage="rates", title="MM fits, v0 / enzyme",
                          rule=["Michaelis-Menten fit of `v0` / enzyme",
                                f"kcat/Km reported when r² ≥ {km_min_r2:g}"],
                          unit="chambers", n=len(mm),
                          branches=[dict(text=f"r² ≥ {km_min_r2:g}", n=ok, level="pass"),
                                    dict(text=f"r² < {km_min_r2:g}", n=len(mm) - ok, level="warn")]))
    if enzyme_rfu_per_nm is not None:
        rule = [f"[E] = enzyme readout / {enzyme_rfu_per_nm:,.0f} RFU per nM",
                "kcat/Km in M⁻¹ s⁻¹ from v0 / enzyme"]
        if enzyme_placeholder:
            rule.append("PLACEHOLDER slope: not a calibration")
        n = len(mm) if mm is not None else 0
        steps.append(dict(step="report_enzyme_scale", title="Enzyme concentration", rule=rule, unit="chambers",
                          n=n, stage="rates",
                          branches=[dict(text="PLACEHOLDER [E]" if enzyme_placeholder else "[E] in nM", n=n,
                                         level="warn" if enzyme_placeholder else "pass")]))
    return steps


def _code(text):
    """`name` -> the name in code font (mathtext's monospace), for columns and other code names."""
    return re.sub(r"`([^`]+)`", lambda m: r"$\mathtt{" + m.group(1).replace("_", r"\_").replace(" ", r"\ ") + "}$", text)


def _line(text):
    """One rule line as drawn: code names in code font, "- " as a bullet."""
    return _code("  • " + text[2:] if text.startswith("- ") else text)


def _count(n, unit):
    """"1,507 chambers"; "1 grid" (singular for one)."""
    return f"{n:,} {unit[:-1] if n == 1 and unit.endswith('s') else unit}"


def _height(step):
    return _HEAD + 2 * _PAD + _LINE * len(step["rule"])


def flowchart_page(steps, *, title="Processing flowchart", h=8.5, w=11.0, experiment="", stamp=""):
    """Draw `steps` (pipeline_steps() + report_steps()) as a flowchart page: one box per
    step, top to bottom and then in a second column, each with its rule and the number
    that went in, and a side box per branch (left out red, changed amber, kept green)."""
    with plt.rc_context(_REPORT_STYLE):
        fig = plt.figure(figsize=(w, h))
        inch = fig.dpi_scale_trans
        margin, top = 0.5, 0.95
        fig.text(margin / w, 1 - 0.3 / h, title, fontsize=17,
                 va="top")
        if experiment:
            fig.text(1 - margin / w, 1 - 0.28 / h, experiment, ha="right", va="top", fontsize=11, color="0.35")
        if stamp:
            fig.text(1 - 0.15 / w, 0.12 / h, stamp, ha="right", va="bottom", fontsize=8, color="0.5")

        col_w = _BOX_W + _BRANCH_GAP + _BRANCH_W
        col_x = [margin, w - margin - col_w]
        bottom = 0.45
        col, y = 0, h - top
        placed = []
        for s in steps:
            hh = _height(s)
            nb = len(s["branches"])
            hh = max(hh, nb * 0.5 + (nb - 1) * 0.08)
            if y - hh < bottom and col == 0 and placed:
                col, y = 1, h - top
            placed.append((s, col_x[col], y, hh))
            y -= hh + _V_GAP

        renderer = fig.canvas.get_renderer()

        def width(text):
            """A text's width in inches."""
            return text.get_window_extent(renderer).width / fig.dpi

        def fit(text, max_w, smallest=7):
            """Shrink `text` until it is at most `max_w` inches wide."""
            while width(text) > max_w and text.get_fontsize() > smallest:
                text.set_fontsize(text.get_fontsize() - 0.5)

        def arrow(p0, p1, conn="arc3"):
            fig.patches.append(FancyArrowPatch(p0, p1, transform=inch, arrowstyle="-|>", mutation_scale=11,
                                               lw=0.9, color="0.35", connectionstyle=conn,
                                               shrinkA=0, shrinkB=1))

        for i, (s, x, y0, hh) in enumerate(placed):
            # the step: a header band (title, how many went in) over the rule
            fig.patches.append(Rectangle((x, y0 - hh), _BOX_W, hh, transform=inch, fc="white", ec="0.45", lw=0.8))
            fig.patches.append(Rectangle((x, y0 - _HEAD), _BOX_W, _HEAD, transform=inch,
                                         fc=REPORT_COLORS["header"], ec="0.45", lw=0.8))
            count = fig.text(x + _BOX_W - 0.1, y0 - _HEAD / 2, _count(s["n"], s["unit"]), transform=inch,
                             va="center", ha="right", fontsize=_RULE_SIZE, color="0.3")
            title = fig.text(x + 0.1, y0 - _HEAD / 2, s["title"], transform=inch, va="center", fontsize=_TITLE_SIZE,
                             weight="bold")
            fit(title, _BOX_W - 0.3 - width(count))
            fit(fig.text(x + 0.1, y0 - _HEAD - _PAD, "\n".join(map(_line, s["rule"])), transform=inch, va="top",
                         fontsize=_RULE_SIZE, linespacing=1.3), _BOX_W - 0.2)
            # its branches, stacked beside it
            nb = len(s["branches"])
            bh = 0.5
            stack = nb * bh + (nb - 1) * 0.08
            by = y0 - (hh - stack) / 2
            bx = x + _BOX_W + _BRANCH_GAP
            for b in s["branches"]:
                fig.patches.append(FancyBboxPatch((bx, by - bh), _BRANCH_W, bh, transform=inch,
                                                  boxstyle="round,pad=0,rounding_size=0.06",
                                                  fc=_LEVEL_FILL.get(b["level"], "white"), ec="0.45", lw=0.6))
                share = f" ({b['n'] / s['n']:.1%})" if s["n"] else ""
                fit(fig.text(bx + 0.08, by - 0.07, _code(b["text"]), transform=inch, va="top",
                             fontsize=_BRANCH_SIZE), _BRANCH_W - 0.16)
                fit(fig.text(bx + 0.08, by - bh + 0.07, f"{_count(b['n'], s['unit'])}{share}", transform=inch,
                             va="bottom", fontsize=_BRANCH_SIZE, weight="bold"), _BRANCH_W - 0.16)
                arrow((x + _BOX_W, by - bh / 2), (bx, by - bh / 2))
                by -= bh + 0.08
            # on to the next step: straight down, or up and over to the next column
            if i + 1 < len(placed):
                _s, nx, ny, _h = placed[i + 1]
                if nx == x:
                    arrow((x + _BOX_W / 2, y0 - hh), (x + _BOX_W / 2, ny))
                else:
                    mid = (x + col_w + nx) / 2   # the gap between the columns
                    xs = [x + _BOX_W / 2, x + _BOX_W / 2, mid, mid, nx + _BOX_W / 2]
                    ys = [y0 - hh, y0 - hh - 0.15, y0 - hh - 0.15, ny + 0.15, ny + 0.15]
                    fig.add_artist(plt.Line2D(xs, ys, transform=inch, lw=0.9, color="0.35"))
                    arrow((nx + _BOX_W / 2, ny + 0.15), (nx + _BOX_W / 2, ny))
    return fig


STAGES = {"images": "image processing", "rates": "rates and fits"}


def flowchart_pages(steps, **kw):
    """One flowchart page per stage of `steps` (STAGES: image processing, then rates and
    fits), each titled with its stage; keyword arguments go to flowchart_page."""
    figs = []
    for stage, label in STAGES.items():
        part = [st for st in steps if st.get("stage", "rates") == stage]
        if part:
            figs.append(flowchart_page(part, title=f"Processing flowchart: {label}", **kw))
    return figs
