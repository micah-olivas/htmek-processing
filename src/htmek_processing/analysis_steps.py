"""The analysis notebook's steps (2-Analysis), one method per section, on the process notebook's exports.

    import htmek_processing as hp
    an = hp.Analysis.from_config(globals())
    an.call_expression()
    an.call_buttons()
    an.call_labeling()
    an.check_diffusion()
    an.fit_standards()
    an.fit_rates()
    an.rates_vs_enzyme()
    an.fit_mm()
    an.chamber_report()
    an.experiment_report()

Every step reads what the process notebook wrote to 4_exports, skips with a one-line note when its input is missing,
shows a short summary, saves its QC figures under 5_figures/qc and its tables under 4_exports, and keeps its results on
the Analysis (e.g. `an.rates`)."""
from __future__ import annotations

import copy
import functools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .experiment import KEYS, note, table

CONTROLS = ("WT", "N41A", "BLANK")
COLORS = {"library": "#8E44AD", "WT": "#2E8B57", "N41A": "#1F4E79", "BLANK": "0.5"}

# Configure-cell names -> Analysis fields
CONFIG_NAMES = {
    "EXPERIMENT": "experiment", "SUBSTRATE": "substrate", "N_ROWS": "n_rows", "N_COLS": "n_cols",
    "N_SD": "n_sd", "FLOOR": "floor", "BUTTON_SCAN": "button_scan", "BUTTON_VALUE": "button_value",
    "STANDARDS_CAMERA": "standards_camera", "CONC_MAX": "conc_max", "RATE_RULES": "rate_rules",
    "MIN_R2": "min_r2", "DROP_FLAGGED_TILES": "drop_flagged_tiles", "MIN_BLEACH_BLANKS": "min_bleach_blanks", "TAG_SUFFIX": "tag_suffix", "MIN_REF_CHAMBERS": "min_ref_chambers", "REF_NEEDS_GFP_FLOOR": "ref_needs_gfp_floor",
    "DEPLETED_EARLY": "depleted_early", "STANDARDS_MODEL": "standards_model", "PBP_MAX_FRAC": "pbp_max_frac",
    "CALIBRATION": "standards_model",   # the name before 2026-11-20
    "BUTTON_SCAN_NOTE": "button_scan_note", "VOLCANO_REFERENCE": "volcano_reference", "VOLCANO_REFERENCE_NOTE": "volcano_reference_note",
}


def hp_max_ref_slope():
    """fit_progress_curves' default: BLANKs rising faster than this (per exposure) are left out of the bleaching model."""
    import inspect

    from .analysis.rates import fit_progress_curves

    return inspect.signature(fit_progress_curves).parameters["max_ref_slope"].default


def robust_floor(values, n_sd=3.0, how="mad"):
    """The BLANKs' median + `n_sd` SDs: robust (1.4826 x MAD, "mad") or the plain SD ("sd")."""
    v = pd.Series(values, dtype=float).dropna()
    if len(v) < 2:
        return np.nan
    m = v.median()
    sd = 1.4826 * (v - m).abs().median() if how == "mad" else v.std()
    return float(m + n_sd * sd)


def _chip_grid(df, value, n_rows, n_cols):
    return df.set_index(KEYS)[value].astype(float).unstack().reindex(index=range(n_rows), columns=range(n_cols)).to_numpy()


def _chip_axes(ax, n_rows, n_cols):
    ax.set_xticks([0, n_cols // 2 - 1, n_cols - 1], ["1", str(n_cols // 2), str(n_cols)])
    ax.set_yticks([0, n_rows // 2 - 1, n_rows - 1], ["1", str(n_rows // 2), str(n_rows)])
    ax.set(xlabel="column", ylabel="row")


def calls_figure(df, value, floor, called, *, xlabel, floor_label, n_rows=56, n_cols=32, map_title="above the floor (green) / not (orange)"):
    """A call against a BLANK floor: `value`'s distribution per group (library and the controls) with the floor, and
    the chip map of `called`."""
    import matplotlib
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axes[0]
    groups = {"library": ~df["tag"].isin(CONTROLS), **{c: df["tag"] == c for c in CONTROLS}}
    v = df[value].astype(float)
    bins = np.linspace(v.quantile(0.005), v.quantile(0.995), 60)
    for name, m in groups.items():
        if m.any():
            ax.hist(v[m].clip(bins[0], bins[-1]), bins=bins, histtype="stepfilled" if name == "library" else "step",
                    color=COLORS[name], alpha=0.5 if name == "library" else 1, lw=1.2, density=True, label=f"{name} ({int(m.sum())})")
    ax.axvline(floor, color="k", lw=0.9, ls="--", label=floor_label)
    ax.set(xlabel=xlabel, ylabel="density")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    ax.imshow(_chip_grid(df.assign(_c=df[called].astype(float)), "_c", n_rows, n_cols),
              cmap=matplotlib.colors.ListedColormap(["#D35400", "#D5E8D4"]), vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(map_title, fontsize=10)
    _chip_axes(ax, n_rows, n_cols)
    fig.tight_layout()
    return fig


# results every assay series shares: the expression and button calls (a series view copies them before each step)
SHARED = ("enzyme", "gfp_floor", "enzyme_scan", "blank_chambers", "expression_chambers", "expression_variants",
          "expression_floor", "labeling")
# a series' own results: reset in its view
PER_SERIES = ("rates", "mm", "mm_raw", "traces", "norm", "diffusion", "standards_fit", "rate_unit")


def each_series(step):
    """An assay step that runs once per assay series (Analysis.series_views): ASSAY_RUNS' table and each of 1-Process's
    SERIES, under its own heading, each writing its own files. With one series it runs as is. Returns the step's result,
    or {series name (None: ASSAY_RUNS): result} with more than one."""
    @functools.wraps(step)
    def run(self, *args, **kw):
        if self.series_name is not None:   # a series' view: the step itself
            return step(self, *args, **kw)
        views = self.series_views()
        if not views or views == [(None, self)]:
            return step(self, *args, **kw)
        out = {}
        for name, view in views:
            if view is not self:
                for k in SHARED:
                    setattr(view, k, getattr(self, k, None))
            note(f"**▸ {view.series_title}**")
            out[name] = step(view, *args, **kw)
        return out
    return run


@dataclass
class Analysis:
    """The analysis notebook's state: settings from its Configure cell, the process notebook's tables, and each
    step's results."""

    experiment: str = ""
    substrate: str = "bzp_uM"
    n_rows: int = 56
    n_cols: int = 32
    n_sd: float = 3.0
    floor: str = "mad"                 # "mad": BLANK median + n_sd robust SDs; "sd": + n_sd SDs
    button_scan: str | None = None     # the button GFP scan (a `scan` label); None: the last SDS wash
    button_value: str = "button_sum_net"   # the button GFP readout: sum minus the ring inside the button rim
    button_scan_note: str = ""             # why that scan: printed on the experiment report's summary page
    volcano_reference: str = "WT"          # the volcanoes' reference: "WT", or "library" (the library median, e.g. WT dead on the chip)
    volcano_reference_note: str = ""       # why, when it isn't WT: printed on the summary page
    standards_camera: str | None = None    # the standards' camera folder; None: the only one
    conc_max: float | None = None          # fit the standards up to this [NADH] (µM)
    standards_model: str = "line"          # the standards' fit: "line", a line per chamber (NADH); "pbp", a tight-binding sensor (phosphate binding protein, Pi)
    pbp_max_frac: float = 0.9              # "pbp": passes whose bound sensor is above this fraction of it (near the plateau) get no [Pi]
    rate_rules: dict = field(default_factory=dict)   # overrides of hp.PROGRESS_4PL_RULES
    min_r2: float = 0.9                    # MM fits below this r² are left out of the report
    min_bleach_blanks: int = 5             # a run's bleaching model needs this many flat BLANKs; fewer: no correction
    ref_needs_gfp_floor: bool = True       # False: every WT chamber with a passing fit is the volcanoes' reference, its GFP above the floor or not
    min_ref_chambers: int = 3              # the volcanoes test against WT with at least this many WT chambers, else the library median
    tag_suffix: str | None = None          # how figures write a TAG ("*") substitution: None "A4tag"; e.g. "AHA": "A4AHA"
    depleted_early: str | None = None      # "force": runs already slowing at the first image (hp.flag_early_depletion) take the 4PL's slope inside the run (hp.force_4pl_on_depleted); None: they keep the line
    drop_flagged_tiles: bool = True        # leave out buttons and chamber passes in a tile the process notebook flagged
    series_name: str | None = None         # which assay series this is (a view, Analysis.series_views); None: ASSAY_RUNS'

    @classmethod
    def from_config(cls, namespace, **overrides):
        """An Analysis from the Configure cell's names (pass `globals()`); `overrides` set any field directly."""
        kw = {f: namespace[n] for n, f in CONFIG_NAMES.items() if n in namespace}
        kw.update(overrides)
        return cls(**kw)

    def __post_init__(self):
        import logging

        import matplotlib.pyplot as plt

        from .analysis.plots import set_plot_style

        from .processing.layout import set_tag_suffix

        set_plot_style()
        set_tag_suffix(self.tag_suffix)
        plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
        self.exports, self.figures = Path("4_exports"), Path("5_figures")
        read = lambda rel: pd.read_csv(self.exports / rel) if (self.exports / rel).exists() else None
        self.expression = read("expression/gfp_expression_per_dumbbell.csv")
        self.gfp = read("gfp/gfp_per_chamber.csv")
        self.nadh = read("assay/nadh_per_chamber.csv")
        # every assay series' table and [S] column (1-Process quantify_assay); "": ASSAY_RUNS'
        idx = self.exports / "assay/series.json"
        import json

        self.series_index = json.loads(idx.read_text()) if idx.exists() else {}
        self._views = None
        std = sorted((self.exports / "assay/standards").glob("*/nadh_standards_per_chamber.csv"))
        if self.standards_camera:
            std = [p for p in std if p.parent.name == self.standards_camera]
        self.standards = pd.read_csv(std[0]) if std else None
        self.tiles = read("qc/tile_medians.csv")   # 1-Process check_tiles: every tile's deviation, per scan and pass
        self.value = "chamber_ROI_median"   # the progress-curve signal; fit_standards() switches it to nadh_uM
        self.enzyme = self.rates = self.mm = self.mm_raw = None
        rows = {"expression time course": self.expression, "button GFP": self.gfp, "assay": self.nadh, "NADH standards": self.standards}
        have = [f"{k} ({len(v):,} rows)" for k, v in rows.items() if v is not None]
        runs = f"; [S] runs: {', '.join(f'{s:g}' for s in sorted(self.nadh[self.substrate].unique()))} µM" if self.nadh is not None else ""
        extra = {k: v for k, v in self.series_index.items() if k}
        if extra:
            runs += "; more assay series: " + ", ".join(f"{k} ({v['runs']} runs, `{v['substrate']}`)" for k, v in extra.items())
        note(f"**{self.experiment}** · from `4_exports`: {', '.join(have) or 'nothing yet'}{runs}")

    def __repr__(self):
        return f"Analysis({self.experiment}" + (f", series {self.series_name})" if self.series_name else ")")

    # ------------------------------------------------------------------ assay series
    def series_views(self):
        """[(name, Analysis)]: one per assay series the assay steps run on. ASSAY_RUNS' table is this Analysis (name
        None; left out without a table), each 1-Process SERIES a view: a copy with that series' table, [S] column
        (`substrate`) and its own results (`an.series("pnpp").rates`), sharing the expression and button calls."""
        if self._views is None:
            self._views = {}
            for name, ent in self.series_index.items():
                path = self.exports / "assay" / ent["file"]
                if not name or not path.exists():
                    continue
                v = copy.copy(self)
                v.series_name, v.substrate, v.nadh, v.value, v._views = name, ent["substrate"], pd.read_csv(path), "chamber_ROI_median", {}
                for k in PER_SERIES:
                    setattr(v, k, None)
                self._views[name] = v
        return ([(None, self)] if self.nadh is not None else []) + list(self._views.items())

    def series(self, name):
        """One assay series' view (its results: `an.series("pnpp").rates`); None: ASSAY_RUNS' (this Analysis)."""
        views = dict(self.series_views())
        if name not in views:
            raise KeyError(f"No assay series {name!r}; there are: {', '.join(str(k) for k in views) or 'none'}")
        return views[name]

    @property
    def series_title(self):
        return f"Series {self.series_name} (`{self.substrate}`)" if self.series_name else f"Assay runs (`{self.substrate}`)"

    @property
    def _sfx(self):
        """The series' file-name suffix: "" for ASSAY_RUNS', "_<name>" for a series."""
        return f"_{self.series_name}" if self.series_name else ""

    @property
    def _report_title(self):
        return f"{self.experiment} · {self.series_name}" if self.series_name else self.experiment

    def _run_label(self, s_uM):
        """A run's scan label in 1-Process's tables (the tile check), from its [S]."""
        return f"{self.series_name or 'assay'} {s_uM:g} µM"

    def save_fig(self, fig, folder, name, show=True, **kw):
        """Save a QC figure in 5_figures/qc/<folder>/<name> (PDF and PNG), show it inline, close it."""
        import matplotlib.pyplot as plt

        from .analysis.plots import save_qc_fig

        save_qc_fig(fig, folder, name, **kw)
        if show:
            plt.show()
        plt.close(fig)

    def _floor(self, values):
        return robust_floor(values, self.n_sd, self.floor)

    @property
    def _floor_label(self):
        return f"BLANK median + {self.n_sd:g} {'robust SD' if self.floor == 'mad' else 'SD'}"

    def _in_flagged_tile(self, df, scan, frame=None):
        """Whether each row's (x, y) falls in a tile flagged on `scan` (and `frame`, a column of df, for a time series)."""
        from .processing.tiles import tile_of

        if self.tiles is None:
            return pd.Series(False, index=df.index)
        t = self.tiles[self.tiles["scan"] == scan]
        if t.empty:
            return pd.Series(False, index=df.index)
        tr, tc = tile_of(df["x"], df["y"], t)
        bad = t[t["flagged"]]
        keys = set(zip(bad["frame"], bad["tile_row"], bad["tile_col"])) if frame else set(zip(bad["tile_row"], bad["tile_col"]))
        rows = zip(df[frame], tr, tc) if frame else zip(tr, tc)
        return pd.Series([k in keys for k in rows], index=df.index)

    @property
    def _runs(self):
        return sorted(self.nadh[self.substrate].unique()) if self.nadh is not None else []

    # ------------------------------------------------------------------ 1a. expression
    def call_expression(self):
        """Which variants expressed, from the expression time course: a chamber expressed when its GFP gain in the DNA
        chamber (background subtracted; last time point minus first) is above the BLANK floor; a variant did not
        express when none of its chambers did (hp.expression_calls). The BLANKs: leakage from their neighbors and
        contamination (hp.blank_leakage). Writes 4_exports/expression/expression_calls_per_{chamber,variant}.csv and
        blank_chambers.csv; QC 2_expression/2c_expression_calls and the report 2_expression/expression_report.pdf."""
        from .analysis.expression import blank_leakage, expression_calls
        from .processing.layout import display_tag
        from .report.expression import expression_report

        if self.expression is None:
            return note("No expression time course (`4_exports/expression/gfp_expression_per_dumbbell.csv`): skipped")
        ch, var, floor = expression_calls(self.expression, value="dna_chamber_sum_net", time="minutes", n_sd=self.n_sd, floor=self.floor)
        blanks, leak = blank_leakage(ch, floor, n_rows=self.n_rows, n_cols=self.n_cols)
        out = self.exports / "expression"
        ch.to_csv(out / "expression_calls_per_chamber.csv", index=False)
        var.to_csv(out / "expression_calls_per_variant.csv", index=False)
        blanks.to_csv(out / "blank_chambers.csv", index=False)
        self.expression_chambers, self.expression_variants, self.expression_floor, self.blank_chambers = ch, var, floor, blanks

        none = var.loc[~var["expressed"], "tag"]
        bad = blanks[blanks["contaminated"]]
        note(f"**Expression calls** · floor {floor:,.0f} RFU ({self._floor_label} of the gain) · "
             f"{int(ch['expressed'].sum()):,} of {len(ch):,} chambers expressed · "
             f"{len(none)} of {len(var)} variants did not" + (f": {', '.join(display_tag(t) for t in none)}" if len(none) else ""))
        ctrl = ch.groupby("tag")["expressed"].agg(["sum", "size"]).reindex(list(CONTROLS)).dropna()
        table(ctrl.rename(columns={"sum": "expressed", "size": "chambers"}), caption="controls")
        note(f"**BLANKs** · {leak['contaminated']} of {leak['blanks']} contaminated"
             + (f" ({', '.join(f'r{r.mark_row + 1} c{r.mark_col + 1}' for r in bad.itertuples())})" if len(bad) else "")
             + f" · the rest gain a median {100 * leak['clean_median_ratio']:.1f}% of their neighbors' (Spearman {leak['clean_spearman']:.2f})")
        if "WT" in ctrl.index and ctrl.loc["WT", "sum"] < ctrl.loc["WT", "size"] / 2:
            note("**Warning:** fewer than half the WT chambers clear the floor; check the time course before trusting these calls")
        self.save_fig(calls_figure(ch, "gain", floor, "expressed", xlabel="GFP gained in the DNA chamber over the time course (RFU)",
                                   floor_label=self._floor_label, n_rows=self.n_rows, n_cols=self.n_cols,
                                   map_title="expressed (green) / not (orange)"),
                      "2_expression", "2c_expression_calls")
        report = expression_report(self.figures / "qc/2_expression/expression_report.pdf", self.expression, ch, var, floor,
                                   experiment=self.experiment, n_rows=self.n_rows, n_cols=self.n_cols)
        note(f"Report: `{report}`")

    # ------------------------------------------------------------------ 1b. buttons
    def call_buttons(self):
        """The enzyme on each button: `button_value` on `button_scan` (default the last SDS wash: GFP left on the buttons
        only), and whether it is above the BLANK buttons' floor (`expressing`). Writes 4_exports/gfp/enzyme_per_chamber.csv;
        QC 3_buttons/3b-calls/3b_expressing_buttons. Kept as `an.enzyme`, the GFP the rates are normalized by."""
        if self.gfp is None:
            return note("No button GFP (`4_exports/gfp/gfp_per_chamber.csv`): skipped")
        scans = list(dict.fromkeys(self.gfp["scan"]))
        scan = self.button_scan or next((s for s in reversed(scans) if s.startswith("SDS wash")), scans[-1])
        e = self.gfp[self.gfp["scan"] == scan][KEYS + ["tag", "x", "y", self.button_value]].rename(columns={self.button_value: "gfp"})
        e["in_flagged_tile"] = self._in_flagged_tile(e, scan).to_numpy()
        n_bad = int(e["in_flagged_tile"].sum())
        if n_bad and self.drop_flagged_tiles:
            e.loc[e["in_flagged_tile"], "gfp"] = np.nan
        # the floor from the clean BLANK buttons: a BLANK the expression calls flagged contaminated has GFP of its own
        bad = getattr(self, "blank_chambers", None)
        contaminated = set(zip(bad.loc[bad["contaminated"], "mark_row"], bad.loc[bad["contaminated"], "mark_col"])) if bad is not None else set()
        clean = (e["tag"] == "BLANK") & ~pd.Series([k in contaminated for k in zip(e["mark_row"], e["mark_col"])], index=e.index)
        floor = self._floor(e.loc[clean, "gfp"])
        e["expressing"] = (e["tag"] != "BLANK") & (e["gfp"] > floor)
        e.to_csv(self.exports / "gfp/enzyme_per_chamber.csv", index=False)
        self.enzyme, self.gfp_floor, self.enzyme_scan = e, floor, scan

        med = e.groupby("tag")["gfp"].median()
        lib = e[~e["tag"].isin(CONTROLS)]
        note(f"**Enzyme on the buttons** · `{self.button_value}` on {scan} · floor {floor:,.0f} RFU ({self._floor_label}"
             + (f"; {len(contaminated)} contaminated BLANKs left out" if contaminated else "") + ") · "
             f"{int(e['expressing'].sum()):,} of {int((e['tag'] != 'BLANK').sum()):,} buttons above it")
        if self.button_scan_note:
            note(f"Why {scan}: {self.button_scan_note}")
        if self.tiles is None:
            note("No tile check (`4_exports/qc/tile_medians.csv`; 1-Process `check_tiles`): buttons in a bad tile are not flagged")
        elif n_bad:
            note(f"**{n_bad} buttons sit in a tile flagged on {scan}** (1-Process tile check): "
                 + ("left out (no GFP, not called)" if self.drop_flagged_tiles else "kept; `in_flagged_tile` marks them"))
        rows = {g: {"buttons": int(m.sum()), "above the floor": int(e.loc[m, "expressing"].sum()), "median (RFU)": e.loc[m, "gfp"].median()}
                for g, m in {"library": ~e["tag"].isin(CONTROLS), **{c: e["tag"] == c for c in CONTROLS}}.items() if m.any()}
        table(pd.DataFrame(rows).T)
        if "WT" in med and "BLANK" in med and med["WT"] < floor:
            note(f"**Warning:** the median WT button ({med['WT']:,.0f} RFU) is below the floor: the button GFP barely separates "
                 f"from the BLANKs on this scan, so v0 / [E] is not reliable here")
        self.save_fig(calls_figure(e, "gfp", floor, "expressing", xlabel=f"button GFP, {scan} (`{self.button_value}`, RFU)".replace("`", ""),
                                   floor_label=self._floor_label, n_rows=self.n_rows, n_cols=self.n_cols),
                      "3_buttons/3b-calls", "3b_expressing_buttons")

    # ------------------------------------------------------------------ 1b'. labeling
    def call_labeling(self, gfp_scan=None):
        """Cy5 labeling against the enzyme on each button (hp.labeling_table / labeling_figure): the post-labeling Cy5
        button sum (minus the Cy5 background scan) against the button GFP on `gfp_scan` (default the button scan of
        call_buttons, else the last SDS wash; minus the GFP background scan),
        and their ratio. Needs 4_exports/gfp/cy5_per_chamber.csv. Writes 4_exports/gfp/labeling_per_chamber.csv; QC
        3_buttons/3d-labeling/3d_cy5_vs_gfp."""
        from .analysis.labeling import labeling_figure, labeling_table

        path = self.exports / "gfp/cy5_per_chamber.csv"
        if not path.exists() or self.gfp is None:
            return note("No Cy5 per button (`4_exports/gfp/cy5_per_chamber.csv`) or no button GFP: skipped")
        scans = list(dict.fromkeys(self.gfp["scan"]))
        scan = gfp_scan or getattr(self, "enzyme_scan", None) or next((s for s in reversed(scans) if s.startswith("SDS wash")), scans[-1])
        t = labeling_table(pd.read_csv(path), self.gfp, gfp_scan=scan, n_cols=self.n_cols)
        t.to_csv(self.exports / "gfp/labeling_per_chamber.csv", index=False)
        self.labeling = t
        lib = t[~t["tag"].isin(CONTROLS) & t["gfp_above_floor"] & (t["cy5"] > 0)]
        note(f"**Labeling** · Cy5 vs GFP ({scan}) · Cy5 / GFP of library buttons above the GFP floor: median "
             f"{lib['cy5_over_gfp'].median():.3g} (left {lib.loc[lib['half'] == 'left', 'cy5_over_gfp'].median():.3g}, "
             f"right {lib.loc[lib['half'] == 'right', 'cy5_over_gfp'].median():.3g}); {int(t['cy5_above_floor'].sum()):,} buttons above the Cy5 floor")
        self.save_fig(labeling_figure(t, title=self.experiment, n_cols=self.n_cols), "3_buttons/3d-labeling", "3d_cy5_vs_gfp")

    # ------------------------------------------------------------------ 1c. diffusion
    @each_series
    def check_diffusion(self):
        """Signal in the DNA chambers during the assay: the neck is closed, so a DNA chamber rises only when product
        diffuses under it (hp.dna_chamber_diffusion: each DNA chamber's slope against the BLANKs' floor, and its ratio to
        the reaction chamber's). Writes 4_exports/assay/dna_chamber_diffusion.csv; QC 5_assay/5c-diffusion."""
        from .analysis.expression import diffusion_figure, dna_chamber_diffusion
        from .processing.layout import display_tag

        if self.nadh is None or "dna_chamber_median" not in self.nadh:
            return note("No DNA-chamber readout in the assay table: skipped (1-Process: `quantify_assay` with the dumbbells placed)")
        t, s = dna_chamber_diffusion(self.nadh, time="minutes", per=(self.substrate,), n_sd=self.n_sd)
        t.to_csv(self.exports / f"assay/dna_chamber_diffusion{self._sfx}.csv", index=False)
        self.diffusion = t
        note(f"**Signal in the DNA chambers during the assay** · {int(t['rising'].sum())} of {len(t):,} chamber-runs rising "
             f"(DNA-chamber slope above the {self._floor_label}). A rise is product diffusing under the neck, or activity "
             "in the DNA chamber itself (enzyme left there after expression): a DNA chamber rising faster than its "
             "reaction chamber (ratio > 1) can't be diffusion from it")
        table(s.set_index(self.substrate).rename(index=lambda v: f"{v:g} µM").rename(columns={
            "floor": "floor (RFU/min)", "median_ratio": "library median, DNA / reaction slope"}),
            fmt={"floor (RFU/min)": "{:.2f}", "chambers": "{:,.0f}", "rising": "{:,.0f}", "library median, DNA / reaction slope": "{:.3f}"})
        top = t[t["rising"]].sort_values("ratio", ascending=False).head(10)
        if len(top):
            table(pd.DataFrame({"[S] (µM)": top[self.substrate], "chamber": [f"r{r + 1} c{c + 1}" for r, c in zip(top["mark_row"], top["mark_col"])],
                                "tag": top["tag"].map(display_tag), "DNA slope (RFU/min)": top["dna_slope"],
                                "% of reaction slope": 100 * top["ratio"]}).set_index("chamber"),
                  caption="the rising chambers, highest ratio first", fmt={"[S] (µM)": "{:g}", "DNA slope (RFU/min)": "{:.2f}", "% of reaction slope": "{:.1f}"})
        self.save_fig(diffusion_figure(t, s, runs=list(s[self.substrate]), per=self.substrate, n_rows=self.n_rows, n_cols=self.n_cols),
                      "5_assay/5c-diffusion", f"5c_dna_chamber_diffusion{self._sfx}")

    # ------------------------------------------------------------------ 2. standards
    @each_series
    def fit_standards(self):
        """Per-chamber standard curves, then every pass converted to µM (`nadh_uM`, the product's concentration); the
        rates are then fit in µM with the [S] rules on. `standards_model` "line": a line of the chamber ROI median vs
        [NADH] per chamber, up to `conc_max`. "pbp": the phosphate binding protein's tight-binding isotherm
        (analysis.standards.fit_pbp_standards: the sensor's active concentration and Kd chip-wide, each chamber's intercept
        and amplitude), inverted to [Pi]; passes near the sensor's plateau (`pbp_max_frac`) get NaN. Writes
        4_exports/assay/standards_fit_per_chamber.csv. QC 4_standards: the curves of example chambers."""
        from .analysis.standards import fit_pbp_standards, fit_standard_lines, predict_conc, predict_conc_pbp

        if self.standards is None:
            return note("No standards (`4_exports/assay/standards/<camera>/`): skipped; the rates are fit in RFU")
        if self.nadh is None:
            return note("No assay table: skipped")
        if self.standards_model == "pbp":
            cal = fit_pbp_standards(self.standards, "chamber_ROI_median", x="nadh_uM", keep=("tag",))
            cal.to_csv(self.exports / "assay/standards_fit_per_chamber.csv", index=False)
            self.standards_fit = cal
            self.nadh = predict_conc_pbp(self.nadh, cal, "chamber_ROI_median", out="nadh_uM", max_frac=self.pbp_max_frac)
            self.value = "nadh_uM"
            ok = cal["slope"].notna()
            sat = self.nadh["nadh_uM_saturated"]
            note(f"**Standards** (PBP, [Pi]) · {int(ok.sum()):,} of {len(cal):,} chambers, {self.standards['nadh_uM'].nunique()} standards · "
                 f"sensor {cal.attrs['p_total']:.1f} µM active, Kd {cal.attrs['kd']:.2g} µM (chip-wide) · median low-[Pi] slope "
                 f"{cal.loc[ok, 'slope'].median():,.1f} RFU/µM, r² {cal.loc[ok, 'r2'].median():.3f} · {int(sat.sum()):,} of {len(sat):,} "
                 f"passes above {self.pbp_max_frac:.0%} of the sensor (near its plateau): no [Pi]")
            self.save_fig(self._pbp_figure(cal), "4_standards", "4a_pbp_standards")
            return None
        if self.standards_model != "line":
            raise ValueError(f"STANDARDS_MODEL: 'line' or 'pbp', not {self.standards_model!r}")
        cal = fit_standard_lines(self.standards, "chamber_ROI_median", x="nadh_uM", keep=("tag",), conc_max=self.conc_max)
        cal.to_csv(self.exports / "assay/standards_fit_per_chamber.csv", index=False)
        self.standards_fit = cal
        self.nadh = predict_conc(self.nadh, cal, "chamber_ROI_median", out="nadh_uM")
        self.value = "nadh_uM"
        ok = cal["slope"].notna()
        note(f"**Standards** (NADH, a line per chamber) · {int(ok.sum()):,} of {len(cal):,} chambers, {self.standards['nadh_uM'].nunique()} standards · "
             f"median slope {cal.loc[ok, 'slope'].median():,.1f} RFU/µM, r² {cal.loc[ok, 'r2'].median():.3f}")

    calibrate = fit_standards   # the name before 2026-11-20

    def _pbp_figure(self, cal):
        """The PBP standards' fit: the chip's median standard curve (normalized per chamber) with the isotherm, and example
        chambers (best, median and worst r²) with their own fits."""
        import matplotlib.pyplot as plt

        from .analysis.standards import bound_complex

        P, Kd = cal.attrs["p_total"], cal.attrs["kd"]
        st = self.standards.merge(cal[KEYS + ["intercept", "slope", "r2"]], on=KEYS)
        st["norm"] = (st["chamber_ROI_median"] - st["intercept"]) / st["slope"]   # µM bound, per chamber
        xs = np.linspace(0, st["nadh_uM"].max() * 1.05, 300)
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
        ax = axes[0]
        g = st.groupby("nadh_uM")["norm"]
        ax.errorbar(g.median().index, g.median(), yerr=[g.median() - g.quantile(.25), g.quantile(.75) - g.median()], fmt="o", color="k", ms=4)
        ax.plot(xs, bound_complex(xs, P, Kd), color="#2E86C1", label=f"isotherm: {P:.1f} µM sensor, Kd {Kd:.2g} µM")
        ax.plot(xs, xs, color="0.6", ls=":", lw=0.8, label="linear (no saturation)")
        ax.axhline(self.pbp_max_frac * P, color="#C0392B", ls="--", lw=0.8, label=f"{self.pbp_max_frac:.0%} of the sensor: no [Pi] above")
        ax.set(xlabel="[Pi] (µM)", ylabel="signal − intercept, / slope (µM bound)", ylim=(0, None))
        ax.set_title("every chamber (median, IQR)", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=7)
        ok = cal.dropna(subset=["r2"]).sort_values("r2")
        for ax, (name, row) in zip(axes[1:], (("worst r²", ok.iloc[0]), ("median r²", ok.iloc[len(ok) // 2]), ("best r²", ok.iloc[-1]))):
            d = st[(st["mark_row"] == row["mark_row"]) & (st["mark_col"] == row["mark_col"])]
            ax.plot(d["nadh_uM"], d["chamber_ROI_median"], "o", color="k", ms=4)
            ax.plot(xs, row["intercept"] + row["slope"] * bound_complex(xs, P, Kd), color="#2E86C1")
            ax.set(xlabel="[Pi] (µM)", ylabel="chamber ROI median (RFU)")
            ax.set_title(f"{name}: {row['tag']} r{int(row['mark_row']) + 1} c{int(row['mark_col']) + 1}, r² {row['r2']:.3f}", loc="left", fontsize=9)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ 3. rates
    @each_series
    def fit_rates(self):
        """Initial rates: photobleaching from the BLANKs per run (exponential loss to a floor, per exposure), then a 4PL
        and a line per chamber and run (hp.fit_progress_curves, hp.PROGRESS_4PL_RULES with `rate_rules`); complete
        passes only. Writes 4_exports/assay/nadh_rates_per_chamber.csv; QC 5_assay/5d-rates: the bleaching model, v0
        over the chip, every chamber's fit, examples."""
        from .analysis.plots import chip_heatmaps, chip_sparklines, progress_fit_overlays, show_progress_fits
        from .analysis.rates import fit_progress_curves

        if self.nadh is None:
            return note("No assay table (`4_exports/assay/nadh_per_chamber.csv`): skipped")
        d = self.nadh[self.nadh["complete_pass"].astype(bool)] if "complete_pass" in self.nadh else self.nadh
        if self.tiles is not None and self.drop_flagged_tiles:
            bad = pd.concat([self._in_flagged_tile(g, self._run_label(s), frame="frame") for s, g in d.groupby(self.substrate)])
            if bad.any():
                note(f"{int(bad.sum()):,} chamber passes in a tile flagged on that pass (1-Process tile check): left out of the fits")
                d = d[~bad.reindex(d.index).to_numpy()]
        # one run at a time: a run whose bleaching model rests on too few BLANKs (most left out as rising) is fit
        # without a bleach correction
        kw = dict(per=(self.substrate, *KEYS), substrate=self.substrate if self.value == "nadh_uM" else None, **self.rate_rules)
        parts, fits, curves, uncorrected = [], {}, {}, []
        for s_uM, g in d.groupby(self.substrate):
            tr, ra = fit_progress_curves(g, self.value, progress=f"{s_uM:g} µM: line and 4PL", **kw)
            bf = tr.attrs.get("bleach_fit", {}).get(s_uM)
            # too few flat BLANKs, or a fit on its bound (the "flat" BLANKs still rising): no correction
            if bf is not None and (bf["n_ref"] < self.min_bleach_blanks or abs(bf["amplitude"]) >= 0.49):
                tr, ra = fit_progress_curves(g, self.value, bleach=None, progress=f"{s_uM:g} µM: refit, no bleach correction", **kw)
                uncorrected.append((s_uM, bf))
            elif bf is not None:
                fits[s_uM] = bf
            curves.update(tr.attrs.get("bleach_curve", {}))
            parts.append((tr, ra))
        traces = pd.concat([t for t, _ in parts], ignore_index=True)
        traces.attrs = {"bleach_curve": curves, "bleach_fit": fits}
        rates = pd.concat([r for _, r in parts], ignore_index=True)
        if self.depleted_early == "force":
            # a fast chamber whose substrate is largely used up by the first image keeps the line, which reads it low:
            # flagged, its v0 becomes the 4PL's steepest slope inside the run (model "4pl_forced"; never extrapolated)
            from .analysis.rates import flag_early_depletion, force_4pl_on_depleted
            per = (self.substrate, *KEYS)
            rates = force_4pl_on_depleted(flag_early_depletion(rates, d, per=per), d, per=per, rate="v0")
            n_forced = rates[rates["model"] == "4pl_forced"].groupby(self.substrate).size()
            note("**Depleted early** (slowing at the first image; v0 from the 4PL inside the run): "
                 + (", ".join(f"{s:g} µM: {n}" for s, n in n_forced.items()) or "none"))
        elif self.depleted_early is not None:
            raise ValueError(f"DEPLETED_EARLY: None or 'force', not {self.depleted_early!r}")
        all_bleach = {**fits, **{s_uM: dict(bf, used=False) for s_uM, bf in uncorrected}}
        for s_uM, bf in uncorrected:
            why = (f"only {bf['n_ref']} BLANKs flat enough to model it ({bf['n_left_out']} rising faster than "
                   f"{hp_max_ref_slope():g} per exposure were left out)" if bf["n_ref"] < self.min_bleach_blanks else
                   f"the model from {bf['n_ref']} BLANKs sits on its bound ({100 * bf['amplitude']:+.0f}%): they still rise")
            note(f"**{s_uM:g} µM: no bleach correction** · {why}")
        if self.enzyme is not None:
            rates = rates.merge(self.enzyme[KEYS + ["gfp", "expressing"]], on=KEYS, how="left")
        rates.to_csv(self.exports / f"assay/nadh_rates_per_chamber{self._sfx}.csv", index=False)
        self.traces, self.rates = traces, rates
        unit = "µM/s" if self.value == "nadh_uM" else "RFU/s"
        self.rate_unit = unit

        bleach = traces.attrs.get("bleach_fit", {})
        note(f"**Initial rates** · `{self.value}`, {len(self._runs)} runs, complete passes only · v0 from the 4PL "
             f"where every rule holds, else the line")
        summ = rates.assign(grp=rates["tag"].where(rates["tag"].isin(CONTROLS), "library")).groupby([self.substrate, "grp"]).agg(
            chambers=("v0", "size"), by_4pl=("model", lambda m: int(m.isin(["4pl", "4pl_forced"]).sum())), median_v0=("v0", "median")).unstack("grp")
        summ.index = [f"{v:g} µM" for v in summ.index]
        for s_uM, bf in bleach.items():
            summ.loc[f"{s_uM:g} µM", ("bleaching", "% lost")] = 100 * bf["amplitude"]
        table(summ["median_v0"], caption=f"median v0 ({unit})", fmt="{:.3g}")
        table(summ["by_4pl"].astype(float).div(summ["chambers"]).mul(100), caption="% of chambers with v0 from the 4PL", fmt="{:.0f}")
        if bleach:
            table(pd.DataFrame({f"{s:g} µM": {"lost (%)": 100 * b["amplitude"], "rate (per exposure)": b["rate"], "BLANKs used": b["n_ref"],
                                              "BLANKs left out (rising)": b["n_left_out"]} for s, b in bleach.items()}).T,
                  caption="photobleaching, from the BLANKs", fmt={"lost (%)": "{:.2f}", "rate (per exposure)": "{:.2f}",
                                                                 "BLANKs used": "{:,.0f}", "BLANKs left out (rising)": "{:,.0f}"})
        # one figure per kind, a panel per run; every chamber's fit: one PDF per run in traces/
        runs = sorted(traces[self.substrate].unique())
        folder = f"5_assay/5d-rates{self._sfx}"
        self.save_fig(self._bleach_figure(traces, runs, all_bleach), folder, "5d_bleaching")
        self.save_fig(chip_heatmaps(rates, "v0", panel=self.substrate, cmap="Blues", blanks=False, ncols=len(runs),
                                    title=f"initial rate per chamber (v0, {unit}), by [S] (µM)"), folder, "5d_v0_maps")
        self.save_fig(self._examples_figure(traces, rates, runs), folder, "5d_example_fits")
        from .progress import track

        for s_uM in track(runs, "every chamber's fit (PDF per run)", unit="run"):
            run, r = traces[traces[self.substrate] == s_uM], rates[rates[self.substrate] == s_uM]
            fig = chip_sparklines(run, f"{self.value}_bc", sharey="span", **progress_fit_overlays(run, r, f"{self.value}_bc"),
                                  title=f"progress curves and the fits that gave each initial rate, {s_uM:g} µM")
            self.save_fig(fig, f"{folder}/traces", f"5d_traces_and_fits_{s_uM:g}uM", show=False, exts=("pdf",))
        note(f"Every chamber's fit, one PDF per run: `5_figures/qc/{folder}/traces/`")

    def _bleach_figure(self, traces, runs, fits):
        """The BLANKs relative to their first exposure, per run, with the bleaching model where one was used."""
        import matplotlib.pyplot as plt

        from .processing.quantify import bleach_model

        fig, axes = plt.subplots(1, len(runs), figsize=(3.4 * len(runs), 3.4), sharey=True, squeeze=False)
        for ax, s_uM in zip(axes[0], runs):
            b = traces[(traces[self.substrate] == s_uM) & (traces["tag"] == "BLANK")].sort_values("frame")
            b = b.assign(rel=b[self.value] / b.groupby(KEYS)[self.value].transform("first"))
            for _, g in b.groupby(KEYS):
                ax.plot(g["frame"] + 1, g["rel"], color="0.8", lw=0.5)
            med = b.groupby("frame")["rel"].median()
            ax.plot(med.index + 1, med.values, "o", color="k", ms=3, label="BLANK median")
            bf = fits.get(s_uM)
            if bf is not None and bf.get("used", True):
                nn = np.linspace(0, med.index.max(), 200)
                ax.plot(nn + 1, bleach_model(nn, bf["amplitude"], bf["rate"], bf["floor"]), color="#C0392B", lw=1.4,
                        label=f"model: {100 * bf['amplitude']:.1f}% lost")
            else:
                ax.text(0.04, 0.04, "no correction:\nthe BLANKs rise", transform=ax.transAxes, fontsize=8, color="#C0392B")
            ax.set_title(f"{s_uM:g} µM", loc="left", fontsize=10)
            ax.set_xlabel("pass")
            ax.legend(frameon=False, fontsize=7, loc="upper left")
        axes[0, 0].set_ylabel("BLANK signal / first pass")
        lo, hi = traces.loc[traces["tag"] == "BLANK"].pipe(lambda b: (b[self.value] / b.groupby([self.substrate, *KEYS])[self.value].transform("first")).quantile([0.02, 0.98]))
        axes[0, 0].set_ylim(lo - 0.01, hi + 0.01)
        fig.suptitle("photobleaching from the BLANK chambers", x=0.01, ha="left", fontsize=11)
        fig.tight_layout()
        return fig

    def _examples_figure(self, traces, rates, runs):
        """Five example chambers per run (rows): the fastest, median and slowest 4PL fits above the GFP floor, and the
        median and highest BLANK."""
        from .analysis.plots import show_progress_fits

        tr, ra, chambers, labels = [], [], [], []
        for i, s_uM in enumerate(runs):   # a row per run: its chambers keyed apart by a row offset
            r = rates[rates[self.substrate] == s_uM]
            ok = r[(r["model"] == "4pl") & ((r["expressing"] == True) if "expressing" in r else (r["tag"] != "BLANK"))].sort_values("v0")
            bl = r[r["tag"] == "BLANK"].sort_values("v0")
            pick = [(int(d_.iloc[j]["mark_row"]), int(d_.iloc[j]["mark_col"]), name) for d_, j, name in
                    ((ok, -1, "fastest"), (ok, len(ok) // 2, "median"), (ok, 0, "slowest"), (bl, len(bl) // 2, "BLANK, median"),
                     (bl, -1, "BLANK, highest")) if len(d_)]
            off = 1000 * i
            tr.append(traces[traces[self.substrate] == s_uM].assign(mark_row=lambda d: d["mark_row"] + off))
            ra.append(r.assign(mark_row=r["mark_row"] + off))
            chambers += [(a + off, b) for a, b, _ in pick]
            labels += [f"{s_uM:g} µM, {n}: r{a + 1} c{b + 1}" for a, b, n in pick]
        return show_progress_fits(pd.concat(tr), pd.concat(ra), chambers, f"{self.value}_bc", labels=labels, ncols=5,
                                  unit="µM" if self.value == "nadh_uM" else "RFU")

    # ------------------------------------------------------------------ 3b. rate vs enzyme
    @each_series
    def rates_vs_enzyme(self):
        """Initial rate against the enzyme on the button, per run (Spearman ρ over the buttons above the floor).
        QC 5_assay/5d-rates/5d_v0_vs_enzyme_<[S]>."""
        import matplotlib.pyplot as plt
        from scipy import stats

        if self.rates is None or "gfp" not in self.rates:
            return note("Needs `call_buttons` and `fit_rates`: skipped")
        rows = {}
        groups = list(self.rates.groupby(self.substrate))
        fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups), 3.6), squeeze=False)
        for ax, (s_uM, r) in zip(axes[0], groups):
            e = r[r["expressing"] == True]
            rho = stats.spearmanr(e["gfp"], e["v0"])[0] if len(e) > 2 else np.nan
            rows[f"{s_uM:g} µM"] = {"buttons above the floor": len(e), "Spearman ρ": rho}
            for name, m in {"BLANK": r["tag"] == "BLANK", "below the floor": (r["tag"] != "BLANK") & (r["expressing"] != True),
                            "above the floor": r["expressing"] == True}.items():
                ax.scatter(r.loc[m, "gfp"], r.loc[m, "v0"], s=5, lw=0, alpha=0.8,
                           color={"BLANK": "0.6", "below the floor": "#D35400", "above the floor": "#1F4E79"}[name], label=name)
            ax.axvline(self.gfp_floor, color="k", lw=0.9, ls="--")
            ax.set_title(f"{s_uM:g} µM · ρ = {rho:.2f}", loc="left", fontsize=10)
            ax.set_xlabel("button GFP (RFU)")
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))
        axes[0, 0].set_ylabel(f"initial rate v0 ({self.rate_unit})")
        axes[0, 0].legend(frameon=False, fontsize=7, markerscale=2)
        fig.suptitle("initial rate vs the enzyme on the button (ρ: Spearman, buttons above the floor)", x=0.01, ha="left", fontsize=11)
        fig.tight_layout()
        self.save_fig(fig, f"5_assay/5d-rates{self._sfx}", "5d_v0_vs_enzyme")
        note("**Initial rate vs the enzyme on the button**")
        table(pd.DataFrame(rows).T, fmt={"buttons above the floor": "{:,.0f}", "Spearman ρ": "{:.2f}"})

    # ------------------------------------------------------------------ 4. Michaelis–Menten
    @each_series
    def fit_mm(self):
        """v0 / [E] (buttons above the floor only) and one Michaelis–Menten fit per chamber across the [S] runs, raw and
        per enzyme. Needs ≥ 3 [S] runs. Writes 4_exports/assay/mm_per_chamber.csv and mm_per_gfp_per_chamber.csv."""
        from .analysis.mm import fit_mm_table
        from .analysis.rates import normalize_rates

        if self.rates is None:
            return note("Needs `fit_rates`: skipped")
        if len(self._runs) < 3:
            return note(f"{len(self._runs)} [S] run(s): Michaelis–Menten needs ≥ 3; skipped")
        self.mm_raw = fit_mm_table(self.rates.rename(columns={"v0": "vmax"}), "vmax", s=self.substrate, progress="Michaelis–Menten, raw")
        self.mm_raw.to_csv(self.exports / f"assay/mm_per_chamber{self._sfx}.csv", index=False)
        if self.enzyme is not None:
            # per GFP: the buttons above the floor; and WT's below it too when it is the reference without the floor
            use = self.enzyme["expressing"] | ((self.enzyme["tag"] == "WT") & ~self.ref_needs_gfp_floor & (self.enzyme["gfp"] > 0))
            norm = normalize_rates(self.rates.drop(columns=["gfp", "expressing"], errors="ignore"), self.enzyme[use],
                                   rate="v0", by="gfp")
            self.mm = fit_mm_table(norm.dropna(subset=["v0_per_gfp"]), "v0_per_gfp", s=self.substrate, progress="Michaelis–Menten, per GFP")
            self.norm = norm
            self.mm.to_csv(self.exports / f"assay/mm_per_gfp_per_chamber{self._sfx}.csv", index=False)
        good = self.mm_raw["r2"] >= self.min_r2
        note(f"**Michaelis–Menten** · {len(self.mm_raw):,} chambers fit across {len(self._runs)} [S] runs · "
             f"{int(good.sum()):,} with r² ≥ {self.min_r2:g}")
        table(self.mm_raw.loc[good, ["km", "vmax", "r2"]].describe().loc[["count", "25%", "50%", "75%"]],
              fmt={"km": "{:,.0f}", "vmax": "{:.3g}", "r2": "{:.3f}"})

    # ------------------------------------------------------------------ 5. chamber report
    @each_series
    def chamber_report(self, chambers=None, n_jobs=None):
        """One page per chamber (hp.report.chamber_report_pdf): crops of its button (the button GFP scan) and its chamber
        (the highest [S], last pass), every run's progress curve and fit, and the MM fits, raw and per GFP; buttons below
        the GFP floor shaded red. `chambers`: (mark_row, mark_col) pairs (default every chamber). Reads the stitched
        images on the drive (1-Process wrote where: 4_exports/stitched_scans.json). One copy in 5_figures/4_report
        (older ones are deleted)."""
        import json
        import os
        from datetime import date

        import xarray as xr

        from .processing.grid import load_centers
        from .report.chamber import chamber_report_pdf

        idx_path = self.exports / "stitched_scans.json"
        if self.rates is None:
            return note("Needs `fit_rates`: skipped")
        if not idx_path.exists():
            return note("No `4_exports/stitched_scans.json` (1-Process `stitch_all` writes it): skipped")
        idx = json.loads(idx_path.read_text())
        # the chamber crop: this series' highest [S] run, last pass (else the chamber grid's run)
        ref = self._run_label(max(self._runs)) if self._runs else None
        ref = ref if ref in idx["scans"] else idx["reference_run"]
        missing = [l for l in (idx["button_scan"], ref) if l and not Path(idx["scans"][l]["store"]).exists()]
        if missing:
            return note(f"The stitched images aren't reachable ({', '.join(missing)}): is the drive plugged in? Skipped")
        images, tile_px, captions = {}, {}, {}
        scan = getattr(self, "enzyme_scan", None) or idx["button_scan"]
        if scan not in idx["scans"] and f"GFP {scan}" in idx["scans"]:   # the GFP table's "post-binding" is the index's "GFP post-binding"
            scan = f"GFP {scan}"
        for label, mask, what in ((scan, "button", "GFP, " + str(scan)), (ref, "chamber", f"NADH, {(ref or '').replace('assay ', '')}, last pass")):
            if not label:
                continue
            z = xr.open_zarr(idx["scans"][label]["store"], consolidated=False)
            im = z["image"].isel(time=-1) if "time" in z["image"].dims else z["image"]
            cx, cy = load_centers(idx["masks"][mask]["centers"])
            images[what] = (im.values.astype(np.float32), cx, cy, idx["masks"][mask]["radius"])
            tile_px[what] = tuple(z.attrs["tile_px"])
            captions[what] = label
        info = None
        if self.enzyme is not None:
            info = self.enzyme.set_index(KEYS)[["gfp"]].rename(columns={"gfp": "GFP"}).round(0)
        folder = self.figures / "4_report"
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob(f"*_chamber_report{self._sfx}_??????.pdf"):   # this series' older copies only
            old.unlink()
        path = folder / f"{self.experiment}_chamber_report{self._sfx}_{date.today():%y%m%d}.pdf"
        chambers = chambers or [(r, c) for r in range(self.n_rows) for c in range(self.n_cols)]
        norm = getattr(self, "norm", None)
        n = chamber_report_pdf(path, chambers, images=images, traces=self.traces, value=f"{self.value}_bc",
                               rates=self.rates.rename(columns={"v0": "vmax", "v0_4pl": "vmax_4pl"}),   # the report's names
                               mm_raw=self.mm_raw, mm=self.mm, norm=norm.dropna(subset=["v0_per_gfp"]) if norm is not None else None,
                               norm_col="v0_per_gfp" if norm is not None else None, info=info,
                               info_floors={"GFP": self.gfp_floor} if info is not None else None, km_min_r2=self.min_r2,
                               tile_px=tile_px, image_captions=captions, rate_unit=self.rate_unit, experiment=self._report_title,
                               n_jobs=n_jobs, substrate=self.substrate)   # default: half the cores; each gets only its chambers' crops
        note(f"**Chamber report** · {n:,} pages · `{path}` ({path.stat().st_size / 1e6:.0f} MB)")

    # ------------------------------------------------------------------ 6. experiment report
    @each_series
    def experiment_report(self):
        """The chip-level report (hp.report.experiment_report_pdf) in 5_figures/4_report; one copy is kept (older ones
        are deleted). Needs fit_mm with the enzyme."""
        from datetime import date

        from .report.experiment import experiment_report_pdf

        if self.mm is None or self.mm_raw is None:
            return note("Needs `fit_mm` with the button GFP: skipped")
        folder = self.figures / "4_report"
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob(f"*_experiment_report{self._sfx}_??????.pdf"):   # this series' older copies only
            old.unlink()
        path = folder / f"{self.experiment}_experiment_report{self._sfx}_{date.today():%y%m%d}.pdf"
        if self.volcano_reference not in ("WT", "library"):
            raise ValueError(f"VOLCANO_REFERENCE: 'WT' or 'library', not {self.volcano_reference!r}")
        lib_ref = self.volcano_reference == "library"
        scan = self.enzyme_scan
        experiment_report_pdf(path, substrate=self.substrate, gfp=self.enzyme, rates=self.rates.drop(columns=["gfp", "expressing"], errors="ignore"), mm_raw=self.mm_raw, mm_per_gfp=self.mm, floor=self.gfp_floor,
                              gfp_col="gfp", gfp_label=f"{scan} GFP", experiment=self._report_title,
                              rate_unit=self.rate_unit, min_r2=self.min_r2,
                              gfp_scan=scan, gfp_note=f"{self.button_value}. {self.button_scan_note}" if self.button_scan_note else "",
                              reference_note=self.volcano_reference_note or ("set to the library median (VOLCANO_REFERENCE)" if lib_ref else ""),
                              volcano_kw=dict(reference="WT", min_ref_chambers=10 ** 9 if lib_ref else self.min_ref_chambers,   # 'library': never enough WT
                                              exclude=CONTROLS,   # controls aren't variants
                                              ref_require_expressed=self.ref_needs_gfp_floor))
        note(f"Report: `{path}`")
