"""One experiment's processing, step by step: the process notebook's Configure cell feeds an `Experiment`, and each
step is one method call that writes its tables and QC figures (htmek_processing/templates/process.ipynb).

    exp = hp.Experiment.from_config(globals())
    exp.stitch_all()
    exp.place_dumbbells(); exp.center_dumbbells(); exp.quantify_expression()
    exp.find_buttons(); exp.quantify_buttons()
    exp.place_chambers(); exp.quantify_assay(); exp.quantify_standards()

Raw images to intensities only: nothing is fitted or filtered here (the analysis notebook does that)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["mark_row", "mark_col"]


def _in_notebook():
    try:
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"   # noqa: F821
    except NameError:
        return False


def note(text):
    """A one-line step summary: Markdown in a notebook (bold allowed), plain text elsewhere."""
    if _in_notebook():
        from IPython.display import Markdown, display

        display(Markdown(text))
    else:
        print(text.replace("**", ""))


def table(df, caption=None, fmt=None):
    """A small summary table: styled in a notebook (one number format, no index clutter), plain text elsewhere."""
    df = df.rename_axis(index=None, columns=None)   # no extra header row for the index's name
    if _in_notebook():
        from IPython.display import display

        sty = df.style.format(fmt or "{:,.0f}", na_rep="–")
        if caption:
            sty = sty.set_caption(caption).set_table_styles([{"selector": "caption", "props": "caption-side: top; text-align: left; font-weight: bold; padding-bottom: 4px;"}])
        display(sty)
    else:
        if caption:
            print(caption)
        print(df.to_string(float_format=lambda v: f"{v:,.1f}"))


def _num(x):
    """x as a float when it is a number, else -inf (an [S] that isn't a number never ranks highest)."""
    return float(x) if isinstance(x, (int, float)) else float("-inf")


def _ctrl_counts(tags):
    n = lambda t: int((tags == t).sum())
    tag = int(np.char.endswith(tags.astype(str), "*").sum())
    return n("WT"), n("N41A"), n("BLANK"), tag

# the Configure cell's names -> Experiment fields
CONFIG_NAMES = {
    "EXPERIMENT": "experiment", "LIBRARY": "library", "EXPERIMENT_DATE": "date", "DRIVE": "drive", "CAMERA": "camera",
    "MAX_CROSS_PX": "max_cross_px", "EXPRESSION_DIR": "expression_dir", "EXPRESSION": "expression",
    "GFP_BACKGROUND": "gfp_background", "SDS_WASHES": "sds_washes", "STANDARDS": "standards", "ASSAY_DIR": "assay_dir",
    "ASSAY_RUNS": "assay_runs", "SUBSTRATE": "substrate", "SEAM_SCAN": "seam_scan", "FLATFIELD_SCAN": "flatfield_scan",
    "N_ROWS": "n_rows", "N_COLS": "n_cols", "PINLIST_FILE": "pinlist_file", "MIRROR_COLS": "mirror_cols",
    "REACTION_R": "reaction_r", "DNA_R": "dna_r", "DNA_OFFSET": "dna_offset", "NECK_WIDTH": "neck_width",
    "DUMBBELL_BG": "dumbbell_bg", "BUTTON_R": "button_r", "BUTTON_RING": "button_ring", "CHAMBER_R": "chamber_r",
    "CHAMBER_RING": "chamber_ring", "CENTER_REACH": "center_reach", "BUTTON_REACH": "button_reach",
    "BUTTON_SPOT_R": "button_spot_r", "SEED_CORNERS": "seed_corners", "PLACE_DUMBBELLS": "open_dumbbell_window",
    "PLACE_CHAMBERS": "open_chamber_window",
    "TAG_SUFFIX": "tag_suffix",
    "CAMERAS": "cameras", "SCAN_CAMERAS": "scan_cameras", "FLATFIELD_FILE": "flatfield_file", "GFP_SCANS": "gfp_scans",
    "CY5_SCANS": "cy5_scans", "SERIES": "series", "STANDARD_MASKS": "standard_masks", "MASK_FILES": "mask_files",
    "CENTER_ON": "center_on", "EXPORT_TIFFS": "export_tiffs", "PLACE_CY5_GRID": "open_cy5_window",
    "CLICK_SERIES": "click_series", "PLACE_CLICK_GRID": "open_click_window", "CLICK_BUTTON_R": "click_button_r",
    "CLICK_CENTERS": "click_centers", "CLICK_REACH": "click_reach",
}


@dataclass
class Experiment:
    """An experiment's settings (the process notebook's Configure cell), its folders, its pinlist, and the steps that
    turn its raw images into intensities. Build it with `Experiment.from_config(globals())`. Paths are relative to the
    notebook's folder; the raw and stitched images are under `drive`."""

    experiment: str
    drive: Path
    pinlist_file: str
    camera: dict
    library: str = "LIBRARY"
    date: str = ""
    max_cross_px: float = 25.0
    expression_dir: str = "GFP/expression-time-course"
    expression: dict = field(default_factory=dict)
    gfp_background: tuple = ("GFP", "gfp_bg_kinetix_500ms")
    sds_washes: dict = field(default_factory=dict)
    standards: dict = field(default_factory=dict)
    assay_dir: str = "Assays"
    assay_runs: dict = field(default_factory=dict)
    substrate: str = "bzp_uM"
    seam_scan: str | None = None
    flatfield_scan: str | None = None
    n_rows: int = 56
    n_cols: int = 32
    mirror_cols: bool = True
    reaction_r: float = 90
    dna_r: float = 65
    dna_offset: tuple = (234, 0)
    neck_width: float = 36
    dumbbell_bg: float = 18
    button_r: float = 34
    button_ring: tuple = (38, 52)
    chamber_r: float = 65
    chamber_ring: tuple = (95, 115)
    center_reach: int = 30
    button_reach: int = 30
    button_spot_r: float = 20
    seed_corners: str | None = None
    open_dumbbell_window: bool | None = None   # None: open each grid window until its corners files exist
    open_chamber_window: bool | None = None
    tag_suffix: str | None = None   # how figures write a TAG ("*") substitution: None "A4tag"; e.g. "AHA": "A4AHA"
    # a chip with more than one camera, series, or its own mask files (e.g. MBO-054)
    cameras: dict = field(default_factory=dict)         # more cameras: {key: camera dict, + "seams" (a geometry JSON to reuse), "max_cross_px"}
    scan_cameras: dict = field(default_factory=dict)    # {scan folder prefix: camera key}; other scans: `camera`
    flatfield_file: str | None = None                   # the GFP flatfield, when not 0_config/flatfield_basic_<flatfield_scan>.tif
    gfp_scans: dict = field(default_factory=dict)       # more GFP snapshots read on the buttons: {folder: label}
    cy5_scans: dict = field(default_factory=dict)       # Cy5 on the buttons: {folder: label}, the background scan first
    series: dict = field(default_factory=dict)          # more assay series (e.g. another substrate): {name: dict(dir=..., substrate=..., runs={folder: [S] or {column: value}})}
    standard_masks: dict = field(default_factory=dict)  # per camera: dict(centers=CSV, radius, ring) for its standards
    mask_files: dict = field(default_factory=dict)      # names in 2_masks: buttons, chamber_corners, chamber_warp, chamber_centers
    center_on: str = "last"                             # the chamber masks centered on the reference run's "last" pass or "mean"
    export_tiffs: bool = False                          # write the Cy5 scans as TIFFs to <drive>/4_stitched-tiffs/
    open_cy5_window: bool | None = None                 # the Cy5 button grid window: None opens it until its corners exist
    click_series: dict = field(default_factory=dict)    # on-chip click labeling time courses: {folder: label}, Cy5 at every pass
    open_click_window: bool | None = None               # the click button grid window: None opens it until its corners exist
    click_button_r: float | None = None                 # the click series' button ROI radius (px); None: button_r. MBO-056: 18, the GFP spot
    click_centers: str = "gfp"                          # "gfp": the GFP buttons mapped onto the click grid (one affine fit); "dark": then moved
                                                        # by the smoothed offset onto the closed buttons' dark cores (first pass); "grid": the grid
    click_reach: int = 15                               # "dark" centering: how far the template match searches (px)

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_config(cls, namespace, **overrides):
        """An Experiment from the Configure cell's names (pass `globals()`); `overrides` set any field directly."""
        kw = {f: namespace[n] for n, f in CONFIG_NAMES.items() if n in namespace}
        kw.update(overrides)
        return cls(**kw)

    def __post_init__(self):
        import matplotlib.pyplot as plt

        from .analysis.plots import set_plot_style
        from .processing.layout import chamber_tags, mirror_pinlist

        import logging

        from .processing.layout import set_tag_suffix

        set_plot_style()
        set_tag_suffix(self.tag_suffix)
        plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # Arial's weights: harmless, but noisy
        self.drive = Path(self.drive)
        if not self.drive.exists():
            raise FileNotFoundError(f"{self.drive} not found: is the T7 plugged in?")
        self.series = {name: self._series_entry(name, v) for name, v in self.series.items()}
        self.raw, self.stitched = self.drive / "1_raw-images", self.drive / "3_stitched"
        self.config, self.masks, self.intermediate = Path("0_config"), Path("2_masks"), Path("3_intermediate")
        self.exports, self.figures, self.raw_record = Path("4_exports"), Path("5_figures"), Path("1_raw-images")
        for d in (self.config, self.masks, self.intermediate, self.exports, self.figures):
            d.mkdir(parents=True, exist_ok=True)
        self.pinlist = mirror_pinlist(self.pinlist_file, n_cols=self.n_cols) if self.mirror_cols else pd.read_csv(self.pinlist_file)
        self.tags = chamber_tags(self.pinlist, self.n_rows, self.n_cols)
        # every scan: key (its folder under 1_raw-images) -> flatfield-corrected? Assays and standards never are
        self.scans = {**{f"{self.expression_dir}/{n}": True for n in self.expression},
                      **({"/".join(self.gfp_background): True} if (self.sds_washes or self.expression or self.gfp_scans) else {}),
                      **{g: True for g in self.gfp_scans},
                      **{w: True for w in self.sds_washes},
                      **{c: False for c in self.cy5_scans},
                      **{c: False for c in self.click_series},
                      **{f"Standards/{cam}/{n}": False for cam, runs in self.standards.items() for n in runs},
                      **{f"{self.assay_dir}/{n}": False for n in self.assay_runs},
                      **{f"{ser['dir']}/{n}": False for ser in self.series.values() for n in ser["runs"]}}
        self.seam_scan = self.seam_scan or next(iter(self.scans), None)
        self.flatfield_scan = self.flatfield_scan or self.seam_scan
        self.stores = {}
        self.expression_keys = [f"{self.expression_dir}/{n}" for n in sorted(self.expression, key=self.expression.get)]
        self.button_scan = list(self.sds_washes)[-1] if self.sds_washes else None
        # the chamber grid's run: the highest [S] of ASSAY_RUNS, else of the first series
        self.reference_run = (f"{self.assay_dir}/{max(self.assay_runs, key=self.assay_runs.get)}" if self.assay_runs else
                              next((f"{v['dir']}/{max(v['runs'], key=lambda r, v=v: _num(v['runs'][r].get(v['substrate'])))}"
                                    for v in self.series.values() if v["runs"]), None))
        # the mask files
        m = self.masks
        self.reaction_corners, self.reaction_warp = m / "reaction_chamber_corners.json", m / "reaction_chamber_warp.json"
        self.dna_corners, self.dna_warp = m / "dna_chamber_corners.json", m / "dna_chamber_warp.json"
        mf = {"buttons": "button_centers.csv", "chamber_corners": "assay_chamber_corners.json",
              "chamber_warp": "assay_chamber_warp.json", "chamber_centers": "assay_chamber_centers.csv", **self.mask_files}
        self.button_centers = m / mf["buttons"]
        self.chamber_corners, self.chamber_warp = m / mf["chamber_corners"], m / mf["chamber_warp"]
        self.chamber_centers = m / mf["chamber_centers"]
        self.cy5_corners, self.cy5_warp = m / "cy5_button_corners.json", m / "cy5_button_warp.json"
        self.click_corners, self.click_warp = m / "click_button_corners.json", m / "click_button_warp.json"
        wt, n41a, blank, tag = _ctrl_counts(self.tags)
        groups = {"expression": len(self.expression), "background": int(bool(self.sds_washes or self.expression or self.gfp_scans)),
                  "GFP": len(self.gfp_scans), "SDS washes": len(self.sds_washes), "Cy5": len(self.cy5_scans), "click series": len(self.click_series),
                  "standards": sum(len(r) for r in self.standards.values()), "assay runs": len(self.assay_runs),
                  **{f"{k} runs": len(v["runs"]) for k, v in self.series.items()}}
        note(f"**{self.experiment}** · {self.tags.size:,} chambers, {len(pd.unique(self.tags.ravel()))} tags "
             f"({wt} WT, {n41a} N41A, {blank} BLANK" + (f", {tag:,} TAG" if tag else "") + ") · "
             f"{len(self.scans)} scans: " + ", ".join(f"{v} {k}" for k, v in groups.items() if v))

    @staticmethod
    def _series_entry(name, v):
        """One SERIES entry, normalized: dir, substrate (the series' [S] column, e.g. "pnpp_uM") and runs
        {folder: {column: value}}. A run given as a number is its [S] in `substrate`; without `substrate`, the first
        numeric column of the runs' dicts is."""
        if not str(name) or not str(name).replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"SERIES name {name!r}: letters, digits, - and _ only (it names the series' files)")
        if not isinstance(v, dict) or "dir" not in v or "runs" not in v:
            raise ValueError(f"SERIES[{name!r}] needs dir= (its folder under 1_raw-images) and runs= ({{folder: [S] (µM)}})")
        sub, runs = v.get("substrate"), {}
        for run, meta in v["runs"].items():
            if not isinstance(meta, dict):
                if sub is None:
                    raise ValueError(f"SERIES[{name!r}]: runs given as numbers need substrate= (their [S] column, e.g. 'pnpp_uM')")
                meta = {sub: meta}
            runs[run] = meta
        if sub is None:
            sub = next((k for meta in runs.values() for k, x in meta.items() if isinstance(x, (int, float))), None)
        if sub is None or any(sub not in meta for meta in runs.values()):
            raise ValueError(f"SERIES[{name!r}]: every run needs its [S] ({sub or 'substrate='})")
        return {**v, "substrate": sub, "runs": runs}

    def _is_assay(self, key):
        """Whether a scan key is an assay run (ASSAY_RUNS or a series)."""
        return key.startswith(self.assay_dir + "/") or any(key.startswith(v["dir"] + "/") for v in self.series.values())

    def __repr__(self):
        return f"Experiment({self.experiment}, {len(self.scans)} scans, drive={self.drive})"

    def save_fig(self, fig, folder, name, show=True, **kw):
        """Save a QC figure in 5_figures/qc/<folder>/<name> (PDF and PNG, or `exts=`), show it inline, close it."""
        import matplotlib.pyplot as plt

        from .analysis.plots import save_qc_fig

        save_qc_fig(fig, folder, name, **kw)
        if show:
            plt.show()
        plt.close(fig)

    @staticmethod
    def place(flag, *corners):
        """Whether a grid window opens: the flag, or (None) until every one of its corners files exists."""
        return flag or (flag is None and not all(Path(c).exists() for c in corners))

    @staticmethod
    def _interactive():
        """Switch the notebook to a desktop window for a grid (`%matplotlib qt`), when in IPython."""
        try:
            get_ipython().run_line_magic("matplotlib", "qt")   # noqa: F821
        except NameError:
            pass

    @staticmethod
    def _inline():
        try:
            get_ipython().run_line_magic("matplotlib", "inline")   # noqa: F821
        except NameError:
            pass

    # ------------------------------------------------------------------ 1. stitch
    def stitch(self, key):
        """One scan: the shared seams, the camera's offset and rotation; the GFP flatfield where `scans` says so."""
        from .processing.stitch import stitch_run

        # the raw scan, or a download that landed beside 1_raw-images or beside the stitched stores
        raw = next((p for p in (self.raw / key, self.drive / key, self.stitched / key) if p.exists()), self.raw / key)
        flat = (Path(self.flatfield_file) if self.flatfield_file else self.config / f"flatfield_basic_{Path(self.flatfield_scan).name}.tif") \
            if self.scans[key] else None
        cam_key = next((v for prefix, v in self.scan_cameras.items() if key.startswith(prefix)), None)
        cam = dict(self.cameras[cam_key]) if cam_key else dict(self.camera)
        seams = cam.pop("seams", None)
        max_cross = cam.pop("max_cross_px", self.max_cross_px)
        geometry = Path(seams) if seams else self.intermediate / f"stitch_geometry_{Path(self.seam_scan).name}.json"
        self.stores[key] = stitch_run(raw, self.stitched / f"{key}.zarr", camera=cam, geometry_path=geometry,
                                      flatfield_path=flat, max_cross_px=max_cross, record_dir=self.raw_record)
        return self.stores[key]

    def image(self, key):
        """A scan's stitched image (the last pass of a time series), float32."""
        if key not in self.stores:   # stitched already: reopens it
            self.stitch(key)
        da = self.stores[key]["image"]
        da = da.isel(time=-1) if "time" in da.dims else da   # select the pass first: never load the whole series
        return np.asarray(da.values, dtype=np.float32)

    def stitch_all(self):
        """Stitch every scan: the seam scan first (its seams measured, the GFP flatfield fit on `flatfield_scan`), then
        every other scan reusing both; existing stores are reopened. QC: every tile corner on the seam scan."""
        from .analysis.plots import show_seams

        if not self.scans:   # every scan list in Configure empty: nothing to stitch
            return note("**No scans configured** (the expression, GFP, SDS wash, Cy5, standards and assay lists in Configure are "
                        f"empty): nothing to stitch. Raw scans go in `{self.raw}/` as zarr folders, named in those lists")
        for k in sorted(self.scans, key=lambda k: (k != self.seam_scan, k != self.flatfield_scan)):
            self.stitch(k)
        groups = {"expression time course": [f"{self.expression_dir}/{n}" for n in self.expression],
                  "GFP background": ["/".join(self.gfp_background)] if "/".join(self.gfp_background) in self.scans else [],
                  "GFP scans": list(self.gfp_scans),
                  "SDS washes": list(self.sds_washes),
                  "Cy5 scans": list(self.cy5_scans),
                  "click series": list(self.click_series),
                  **{f"standards, {cam}": [f"Standards/{cam}/{n}" for n in runs] for cam, runs in self.standards.items()},
                  "assay runs": [f"{self.assay_dir}/{n}" for n in self.assay_runs],
                  **{f"{k} runs": [f"{v['dir']}/{n}" for n in v["runs"]] for k, v in self.series.items()}}
        labels = {**{f"{self.expression_dir}/{n}": f"{m:g} min" for n, m in self.expression.items()}, **self.sds_washes,
                  **self.gfp_scans, **self.cy5_scans, **self.click_series,
                  **{f"{v['dir']}/{n}": ", ".join(f"{x:g} µM" if isinstance(x, (int, float)) else str(x) for x in meta.values())
                     for v in self.series.values() for n, meta in v["runs"].items()},
                  **{f"Standards/{cam}/{n}": f"{c:g} µM" for cam, runs in self.standards.items() for n, c in runs.items()},
                  **{f"{self.assay_dir}/{n}": f"{c:g} µM" for n, c in self.assay_runs.items()}}
        rows = []
        for g, keys in groups.items():
            if not keys:
                continue
            z = [self.stores[k]["image"].sizes for k in keys]
            sizes = sorted({f"{d['y']:,} x {d['x']:,}" for d in z})
            rows.append({"": g, "scans": len(keys), "which": ", ".join(labels.get(k, Path(k).name) for k in keys) if len(keys) > 1 or g.startswith(("expression", "SDS", "GFP scans", "Cy5")) else Path(keys[0]).name,
                         "passes": "/".join(sorted({str(d.get("time", 1)) for d in z})), "size (px)": "; ".join(sizes),
                         "flatfield": "yes" if self.scans[keys[0]] else "no"})
        geo = self.stores[self.seam_scan]["geometry"]
        note(f"**Stitched {len(self.scans)} scans** · seams from `{Path(self.seam_scan).name}` (neighbors across a seam: "
             f"{abs(geo.get('cross_dy', 0)):.1f}, {abs(geo.get('cross_dx', 0)):.1f} px apart); flatfield from `{Path(self.flatfield_scan).name}`; "
             "raw sources recorded in `1_raw-images/`")
        table(pd.DataFrame(rows).set_index(""), fmt={"scans": "{:,.0f}"})
        s = self.stores[self.seam_scan]
        self.save_fig(show_seams(self.image(self.seam_scan), s["store"].attrs["seams_y"], s["store"].attrs["seams_x"],
                                 title=f"every tile corner, {self.seam_scan}"), "1_stitch", "1a_seams")
        self.write_scan_index()

    def write_scan_index(self):
        """4_exports/stitched_scans.json: every scan's label, stitched store, tile pitch and passes, and the masks' files
        and radii, so the analysis notebook (its chamber report) can find the images."""
        import json

        scans = {self._label(k): dict(key=k, store=str((self.stitched / f"{k}.zarr").resolve()),
                                      tile_px=list(self.stores[k]["store"].attrs["tile_px"]) if k in self.stores else None,
                                      kind="assay" if self._is_assay(k) else "standards" if k.startswith("Standards") else "GFP")
                 for k in self.scans}
        masks = dict(button=dict(centers=str(self.button_centers), radius=self.button_r),
                     chamber=dict(centers=str(self.chamber_centers), radius=self.chamber_r),
                     reaction=dict(centers=str(self.masks / "reaction_chamber_centers.csv"), radius=self.reaction_r),
                     dna=dict(centers=str(self.masks / "dna_chamber_centers.csv"), radius=self.dna_r))
        (self.exports / "stitched_scans.json").write_text(json.dumps(dict(experiment=self.experiment, scans=scans, masks=masks,
                                                                          button_scan=self._label(self.button_scan) if self.button_scan else None,
                                                                          reference_run=self._label(self.reference_run) if self.reference_run else None), indent=1))

    def _label(self, key):
        labels = {**{f"{self.expression_dir}/{n}": f"expression {m:g} min" for n, m in self.expression.items()},
                  "/".join(self.gfp_background): "GFP background", **self.sds_washes,
                  **{g: f"GFP {lab}" for g, lab in self.gfp_scans.items()}, **{c: f"Cy5 {lab}" for c, lab in self.cy5_scans.items()}, **self.click_series,
                  **{f"{v['dir']}/{n}": f"{k} " + ", ".join(f"{x:g} µM" if isinstance(x, (int, float)) else str(x) for x in meta.values())
                     for k, v in self.series.items() for n, meta in v["runs"].items()},
                  **{f"Standards/{cam}/{n}": f"standard {c:g} µM ({cam})" for cam, runs in self.standards.items() for n, c in runs.items()},
                  **{f"{self.assay_dir}/{n}": f"assay {c:g} µM" for n, c in self.assay_runs.items()}}
        return labels.get(key, Path(key).name)

    def check_tiles(self, n_sd=5.0, min_deviation=0.02, stride=8, hot_rows=True, hot_row_threshold=0.02):
        """Every tile's median pixel on every scan (every pass of an assay), against that tile's usual share of the scan
        (its median over the scans of the same kind: the expression and background scans, the SDS washes, the assay
        passes, the standards); a tile
        off by more than `n_sd` robust SDs of every tile's deviation is flagged (hp.tile_deviations; 5, since hundreds of
        tiles are tested: 3 would flag a few by chance) and by at least `min_deviation` (2%: tiles agree within
        ~0.3% on a good scan). A flagged tile is
        an acquisition problem on that scan (a darker or brighter tile), not chip content. Writes
        4_exports/qc/tile_medians.csv (the analysis notebook flags buttons and chambers in a flagged tile). QC:
        1_stitch/1b_tile_check.

        With `hot_rows` (default), also the sensor rows off from their neighbors in every raw tile, by more than
        `hot_row_threshold` (see _check_hot_rows). QC: 1_stitch/1c_hot_rows."""
        from .processing.tiles import tile_deviations, tile_figure, tile_medians

        # a scan's medians are reused from the last check while its stitched store is unchanged
        old_path = self.exports / "qc/tile_medians.csv"
        old = pd.read_csv(old_path) if old_path.exists() else None
        parts, reused = [], 0
        for k, flat in self.scans.items():
            kind = ("assay" if self._is_assay(k)
                    else "standards" if k.startswith("Standards") else "SDS washes" if k in self.sds_washes
                    else "GFP background" if k == "/".join(self.gfp_background)
                    else "Cy5" if k in self.cy5_scans else "Cy5 click" if k in self.click_series else "GFP" if flat else "other")   # washes: less GFP, their own pattern
            cam_key = next((v for prefix, v in self.scan_cameras.items() if k.startswith(prefix)), None)
            if cam_key:   # another camera: its own tile grid, its own group
                kind = f"{kind} ({cam_key})"
            f = self.stitched / f"{k}.zarr" / "zarr.json"
            mtime = round(f.stat().st_mtime) if f.exists() else -1
            prev = old[(old["key"] == k) & (old["store_mtime"] == mtime)] if old is not None and "store_mtime" in old else None
            if prev is not None and len(prev):
                parts.append(prev[["scan", "frame", "tile_row", "tile_col", "y0", "y1", "x0", "x1", "median", "key", "store_mtime"]]
                             .assign(group=kind))
                reused += 1
                continue
            if k not in self.stores:
                self.stitch(k)
            parts.append(tile_medians(self.stores[k], scan=self._label(k), frames="all" if kind == "assay" else "last",
                                      stride=stride).assign(key=k, group=kind, store_mtime=mtime))
        dev = tile_deviations(pd.concat(parts, ignore_index=True), n_sd=n_sd, min_deviation=min_deviation)
        (self.exports / "qc").mkdir(parents=True, exist_ok=True)
        dev.to_csv(self.exports / "qc/tile_medians.csv", index=False)
        self.tiles = dev
        bad = dev[dev["flagged"]].sort_values("deviation", key=abs, ascending=False)
        d = dev["deviation"]
        spread = 100 * 1.4826 * (d - d.median()).abs().median()
        note(f"**Tile check** · {dev[['scan', 'frame']].drop_duplicates().shape[0]} scans and passes x "
             f"{dev[['tile_row', 'tile_col']].drop_duplicates().shape[0]} tiles · typical deviation {spread:.2f}% (robust SD) · "
             f"{len(bad)} flagged (off by > {max(n_sd * spread, 100 * min_deviation):.1f}%: {n_sd:g} robust SD, at least {100 * min_deviation:g}%) · "
             f"{len(parts) - reused} scans read, {reused} unchanged since the last check · "
             "`4_exports/qc/tile_medians.csv`")
        if len(bad):
            table(pd.DataFrame({"scan": bad["scan"], "pass": bad["frame"] + 1, "tile (row, col)": [f"{r + 1}, {c + 1}" for r, c in zip(bad["tile_row"], bad["tile_col"])],
                                "deviation (%)": 100 * bad["deviation"]}).head(20).set_index("scan"),
                  caption="flagged tiles, largest first (rows and columns of the tile grid, from the top left)",
                  fmt={"pass": "{:,.0f}", "deviation (%)": "{:+.1f}"})
        self.save_fig(tile_figure(dev), "1_stitch", "1b_tile_check")
        if hot_rows:
            self._check_hot_rows(threshold=hot_row_threshold)

    def _camera_of(self, key):
        """A scan's camera: (its key in `cameras`, or "main" for `camera`; that camera's dict)."""
        cam_key = next((v for prefix, v in self.scan_cameras.items() if key.startswith(prefix)), None)
        return (cam_key, self.cameras[cam_key]) if cam_key else ("main", self.camera)

    def _check_hot_rows(self, threshold=0.02, frames="ends", stride=4):
        """Sensor rows whose gain is off from their neighbors', from the raw tiles of every scan (an assay's first and
        last pass, or `frames`): each row's median over the tiles against a running median of the 15 rows around it
        (hp.row_gains). A row off by more than `threshold` (2%; rows agree within ~0.3%) in the median over a camera's
        scans is flagged (hp.hot_rows): a fixed pattern of the camera, in every tile, which stitching turns into a
        tilted line 2-4 px wide. The flagged rows are mapped through the stitch (rotation, placement) onto the assay
        chambers and the buttons, to show which ROIs they cross. A QC output only: nothing downstream changes.
        Writes 4_exports/qc/hot_rows.csv (every scan's row gains), 4_exports/qc/hot_row_rois.csv (each ROI's share
        of px on a flagged row). QC: 1_stitch/1c_hot_rows."""
        from .processing.grid import load_centers
        from .processing.quantify import intensities_at
        from .processing.stitch import open_tiled
        from .processing.tiles import hot_row_figure, hot_row_mask, hot_rows, row_gains, row_period

        # a scan's gains are reused from the last check while its raw scan is unchanged
        old_path = self.exports / "qc/hot_rows.csv"
        old = pd.read_csv(old_path) if old_path.exists() else None
        old = old if old is not None and "raw_mtime" in old else None
        parts, reused, shapes, skipped = [], 0, {}, []
        for k in self.scans:
            raw = next((p for p in (self.raw / k, self.drive / k) if p.exists()), None)
            if raw is None:
                continue
            cam_key, cam = self._camera_of(k)
            meta = next((p for p in (raw / "zarr.json", *raw.glob("*/zarr.json")) if p.exists()), None)
            mtime = round(meta.stat().st_mtime) if meta else -1
            try:
                t = open_tiled(raw)["tile"]
            except ValueError:   # several channels per tile: not a single-channel scan
                skipped.append(self._label(k))
                continue
            shapes[cam_key] = (t.sizes["y"], t.sizes["x"])
            prev = old[(old["key"] == k) & (old["raw_mtime"] == mtime)] if old is not None else None
            if prev is not None and len(prev):
                parts.append(prev)
                reused += 1
                continue
            offset = cam.get("offset", 0.0)
            try:
                g = row_gains(raw, offset=offset if np.isscalar(offset) else 0.0, frames=frames if self._is_assay(k) else "ends",
                              stride=stride)
            except ValueError:   # more than one leading dim (e.g. passes and exposures)
                skipped.append(self._label(k))
                continue
            parts.append(g.assign(key=k, scan=self._label(k), camera=cam.get("name", cam_key), raw_mtime=mtime))
        if not parts:
            note("**Hot-row check** · no raw scans found")
            return None
        gains = pd.concat(parts, ignore_index=True)
        (self.exports / "qc").mkdir(parents=True, exist_ok=True)
        gains.to_csv(old_path, index=False)
        flagged = hot_rows(gains, threshold=threshold)
        self.hot_rows = flagged

        # which ROIs the flagged rows cross: the chambers on the reference run's stitch, the buttons on the button scan's
        rois = []
        for what, key, centers, radius in (("chamber", self.reference_run, self.chamber_centers, self.chamber_r),
                                            ("button", self.button_scan, self.button_centers, self.button_r)):
            if not key or not Path(centers).exists():
                continue
            cam_key, cam = self._camera_of(key)
            rows = flagged.loc[flagged["camera"] == cam.get("name", cam_key), "row"]
            if not len(rows) or cam_key not in shapes:
                continue
            if key not in self.stores:
                self.stitch(key)
            mask = hot_row_mask(self.stores[key], rows, tile_shape=shapes[cam_key]).astype(np.uint8)
            cx, cy = load_centers(centers)
            frac = intensities_at(mask, cx, cy, radius=radius, aggs=("mean",))["mean"]
            r, c = np.indices(cx.shape)
            rois.append(pd.DataFrame(dict(roi=what, scan=self._label(key), mark_row=r.ravel(), mark_col=c.ravel(),
                                          hot_row_px_frac=np.ravel(frac))))
        rois = pd.concat(rois, ignore_index=True) if rois else pd.DataFrame(columns=["roi", "scan", "mark_row", "mark_col", "hot_row_px_frac"])
        rois.to_csv(self.exports / "qc/hot_row_rois.csv", index=False)

        lines = []
        for cam, f in flagged.groupby("camera", sort=False):
            per = row_period(f["row"], f["gain"])
            worst = f.loc[(f["gain"] - 1).abs().idxmax()]
            lines.append(f"{cam}: {len(f)} rows" + (f" (every {per})" if per else "")
                         + f", worst row {int(worst['row'])} at {worst['gain']:.2f}x ({worst['gain_lo']:.2f}-{worst['gain_hi']:.2f}x over scans and passes)")
        crossed = ", ".join(f"{(g['hot_row_px_frac'] > 0).sum():,} of {len(g):,} {w}s (median {100 * g['hot_row_px_frac'].median():.1f}% of the ROI)"
                            for w, g in rois.groupby("roi", sort=False))
        note(f"**Hot-row check** · {gains['key'].nunique()} scans ({len(parts) - reused} read, {reused} unchanged since the last check) · "
             f"rows off by > {100 * threshold:g}% from their neighbors: " + ("; ".join(lines) if lines else "none")
             + (f" · on a flagged row: {crossed}" if crossed else "")
             + (f" · not checked (several channels or leading dims): {', '.join(skipped)}" if skipped else "")
             + " · `4_exports/qc/hot_rows.csv`, `4_exports/qc/hot_row_rois.csv`")
        self.save_fig(hot_row_figure(gains, flagged, threshold=threshold), "1_stitch", "1c_hot_rows")
        return flagged

    # ------------------------------------------------------------------ 2. expression time course
    def place_dumbbells(self, force=None):
        """The reaction- and DNA-chamber grids (one window, Tab switches), on the time course's last scan. Opens until
        both corners files exist (or as `force` / `open_dumbbell_window` says); blocks until the window is closed."""
        from .processing.grid import load_mask
        from .processing.grid_adjuster import GridAdjuster

        flag = self.open_dumbbell_window if force is None else force
        if not self.expression_keys or not self.place(flag, self.reaction_corners, self.dna_corners):
            return
        self._interactive()
        seed = None if self.reaction_corners.exists() or not self.seed_corners else load_mask(self.seed_corners)["corners"]
        gd = GridAdjuster(self.stores[self.expression_keys[-1]]["image"], columns=self.pinlist, frame=-1, corners=seed,
                          radius=self.reaction_r, bg_outer=self.dumbbell_bg, corners_path=self.reaction_corners,
                          warp_path=self.reaction_warp, vmax_pct=99.5, neck_width=self.neck_width,
                          dna=dict(corners_path=self.dna_corners, warp_path=self.dna_warp, radius=self.dna_r, offset=self.dna_offset))
        self._inline()
        self.dumbbell_window = gd

    def _visible(self, image, x, y, r, inner, outer, tags):
        """Where a disc stands out: its median minus a ring's, above the BLANKs' median + 3 SD of that contrast."""
        from .processing.quantify import centers_to_df

        t = centers_to_df(x, y, self.pinlist, img=image, radius=r, aggs=("median",), bg=True, bg_inner=inner, bg_outer=outer)
        contrast = (t["median"] - t["bg"]).to_numpy().reshape(np.shape(x))
        blank = contrast[tags == "BLANK"]
        return contrast > np.median(blank) + 3 * np.std(blank)

    def _centering_qc(self, image, grid, centered, r, ok, fill, folder, name, title, half):
        from .analysis.plots import centering_examples

        d = np.hypot(centered[0] - grid[0], centered[1] - grid[1])
        rng = np.random.default_rng(0)
        first = sorted(map(tuple, np.argwhere(ok)), key=lambda k: -d[k])[:8] if name.endswith("chambers") else \
            [tuple(k) for k in rng.permutation(np.argwhere(ok))[:8]]
        groups = [("largest moves" if name.endswith("chambers") else "centered", first),
                  ("random" if name.endswith("chambers") else "from neighbors",
                   [tuple(k) for k in rng.permutation(np.argwhere(ok if name.endswith("chambers") else fill))[:8]])]
        self.save_fig(centering_examples(image, grid, centered, r, self.tags, groups, half=half, ncols=8, title=title),
                      folder, name, dpi=130)

    def center_dumbbells(self):
        """Center each reaction and DNA chamber on the last expression scan (`center_reach` px, hp.center_on_disc_sum); a
        fit counts only where the chamber stands out from the ring around it (above the BLANKs' median + 3 SD of that
        contrast), and the rest take their neighbors' shift. Writes 2_masks/<reaction|dna>_chamber_centers.csv. QC:
        examples, the placed grid dashed and the centered circle solid. The centers are kept in `exp.centered`."""
        from .processing.grid import center_on_disc_sum, load_mask, save_centers, shift_from_neighbors, warped_grid

        if not self.expression_keys:
            return None
        img = self.image(self.expression_keys[-1])
        self.centered, rows, qc = {}, [], []
        for name, corners, warp in (("reaction", self.reaction_corners, self.reaction_warp), ("dna", self.dna_corners, self.dna_warp)):
            r = load_mask(corners)["radius"]
            gx, gy = warped_grid(corners, warp, self.n_rows, self.n_cols)
            sx, sy, ok = center_on_disc_sum(img, gx, gy, r, reach=self.center_reach)
            ok &= self._visible(img, sx, sy, r, r + 4, r + 20, self.tags)
            fx, fy, fill = shift_from_neighbors(gx, gy, sx, sy, ok)
            save_centers(self.masks / f"{name}_chamber_centers.csv", fx, fy, np.select([ok, fill], ["fit", "neighbors"], "grid"))
            d = np.hypot(fx - gx, fy - gy)
            rows.append({"": f"{'DNA' if name == 'dna' else name} chambers", "radius (px)": r, "fit": ok.sum(), "from neighbors": fill.sum(),
                         "grid": (~ok & ~fill).sum(), "moved, median (px)": np.median(d), "90th (px)": np.percentile(d, 90),
                         "max (px)": d.max()})
            qc.append((img, (gx, gy), (fx, fy), r, ok, fill, "2_expression", f"2b_centering_{name}_chambers",
                       f"{name} chambers centered on the last expression scan: dashed the placed grid, solid the centered circle",
                       int(1.6 * r)))
            self.centered[name] = (fx, fy)
        note(f"**Centered the dumbbells** on `{Path(self.expression_keys[-1]).name}` (search ±{self.center_reach} px); "
             "a fit counts where the chamber stands out from its ring (BLANK median + 3 SD)")
        table(pd.DataFrame(rows).set_index(""), fmt={"radius (px)": "{:.0f}", "fit": "{:,.0f}", "from neighbors": "{:,.0f}", "grid": "{:,.0f}",
                                                    "moved, median (px)": "{:.1f}", "90th (px)": "{:.1f}", "max (px)": "{:.1f}"})
        for args in qc:
            self._centering_qc(*args)

    def _dumbbell_centers(self):
        from .processing.grid import load_centers

        if not getattr(self, "centered", None):
            self.centered = {n: load_centers(self.masks / f"{n}_chamber_centers.csv") for n in ("reaction", "dna")}
        return self.centered["reaction"], self.centered["dna"]

    def quantify_expression(self):
        """GFP on every scan of the time course, per chamber: in its DNA chamber, where it is expressed (the DNA chamber's
        disc: dna_chamber_sum, _median, _n_px, and dna_chamber_sum_net = the sum minus the dumbbell's background ring's
        median x its pixels), and over the whole dumbbell (its reaction chamber, DNA chamber and the neck between their
        centered circles, minus a ring around the outline; hp.dumbbell_intensities). Writes
        4_exports/expression/gfp_expression_per_dumbbell.csv. QC: the reaction chambers on the last scan, the controls'
        DNA-chamber GFP over time. Returns the table."""
        import matplotlib.pyplot as plt

        from .analysis.plots import show_masks
        from .processing.grid import load_mask
        from .processing.quantify import centers_to_df, dumbbell_intensities

        if not self.expression_keys:
            return None
        (dbx, dby), (dnx, dny) = self._dumbbell_centers()
        rmask, dmask = load_mask(self.reaction_corners), load_mask(self.dna_corners)
        shape = dict(dna_radius=dmask["radius"], neck_width=rmask["neck_width"])
        parts = []
        for k in self.expression_keys:
            img = self.image(k)
            d = dumbbell_intensities(img, dbx, dby, self.pinlist, radius=rmask["radius"], dumbbell=shape,
                                     ring=(2.0, float(rmask["bg_outer"])), dna_x=dnx, dna_y=dny)
            dna = centers_to_df(dnx, dny, self.pinlist, img=img, radius=dmask["radius"], aggs=("sum", "median"), prefix="dna_chamber")
            for col in ("dna_chamber_sum", "dna_chamber_median", "dna_chamber_n_px"):
                d[col] = dna[col].to_numpy()
            # the background: the median of the ring around the whole dumbbell (clear of the neck and the reaction chamber)
            d["dna_chamber_sum_net"] = d["dna_chamber_sum"] - d["dumbbell_bg"] * d["dna_chamber_n_px"]
            parts.append(d.assign(scan=Path(k).name, minutes=self.expression[Path(k).name]))
        expression = pd.concat(parts, ignore_index=True)
        (self.exports / "expression").mkdir(parents=True, exist_ok=True)
        expression.to_csv(self.exports / "expression/gfp_expression_per_dumbbell.csv", index=False)
        self.save_fig(show_masks(self.image(self.expression_keys[-1]), dbx, dby, radius=rmask["radius"],
                                 title="reaction chambers, last expression scan"), "2_expression", "2a_dumbbell_grid", show=False, dpi=150)
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for tag, color in (("WT", "#C0392B"), ("N41A", "#1F4E79"), ("BLANK", "0.5")):
            for _, g in expression[expression["tag"] == tag].groupby(KEYS):
                ax.plot(g["minutes"], g["dna_chamber_sum_net"] / 1e6, "o-", ms=3, lw=0.8, color=color, alpha=0.8)
            ax.plot([], [], color=color, label=tag)
        ax.set(xlabel="minutes", ylabel="GFP in the DNA chamber, minus background (10$^6$ RFU)")
        ax.legend(frameon=False)
        note(f"**GFP in the DNA chambers** (where it is expressed) and over each dumbbell, {len(self.expression_keys)} time points · reaction chamber r {rmask['radius']:g} px, "
             f"DNA chamber r {dmask['radius']:g} px (median {np.median(dnx - dbx):.0f}, {np.median(dny - dby):.0f} px away), "
             f"neck {rmask['neck_width']:g} px, background ring to {rmask['bg_outer']:g} px · `4_exports/expression/gfp_expression_per_dumbbell.csv`")
        ctrl = (expression[expression["tag"].isin(["WT", "N41A", "BLANK"])].groupby(["tag", "minutes"])["dna_chamber_sum_net"]
                .median().unstack("minutes").reindex(["WT", "N41A", "BLANK"]) / 1e6)
        ctrl.columns = [f"{m:g} min" for m in ctrl.columns]
        table(ctrl, caption="controls, median GFP in the DNA chamber minus background (10⁶ RFU)", fmt="{:.2f}")
        self.save_fig(fig, "2_expression", "2b_expression_time_course_controls")
        self.expression_table = expression
        return expression

    # ------------------------------------------------------------------ 3. buttons
    def find_buttons(self, force=False):
        """Each button from its centered reaction chamber plus an offset (first the one that puts the most last-wash
        signal under button-sized discs, searched ±120 px, then 1 px; then an affine map from the reaction chambers to the
        buttons found, refit over three rounds, since the offset drifts over the chip; then each button from its grid place and
        the misses of its 8 nearest found buttons); buttons with a GFP spot that stands out (above the
        BLANKs' median + 3 SD of the spot's contrast) are centered on it with a disc the spot's size (`button_spot_r`,
        within `button_reach`), the rest take their neighbors' shift. Writes 2_masks/button_centers.csv. QC: examples.
        The centers are kept in `exp.buttons`."""
        from scipy import ndimage

        from .processing.grid import center_on_disc_sum, save_centers, shift_from_neighbors

        if not self.button_scan:
            return None
        if self.button_centers.exists() and not force:   # found before (or by an older notebook): reused
            from .processing.grid import load_centers

            self.buttons = load_centers(self.button_centers)
            src = pd.read_csv(self.button_centers)["source"].value_counts() if "source" in pd.read_csv(self.button_centers, nrows=1) else None
            note(f"**Buttons** from `{self.button_centers}` (found before; `find_buttons(force=True)` finds them again)"
                 + (" · " + ", ".join(f"{v:,} {k}" for k, v in src.items()) if src is not None else ""))
            return None
        wash = self.image(self.button_scan)
        (dbx, dby), _ = self._dumbbell_centers()
        blur = ndimage.uniform_filter(np.nan_to_num(wash), 2 * self.button_r)
        score = lambda dy, dx: ndimage.map_coordinates(blur, [dby.ravel() + dy, dbx.ravel() + dx], order=1).mean()
        best = max((score(dy, dx), dy, dx) for dy in range(-120, 121, 6) for dx in range(-120, 121, 6))
        best = max((score(dy, dx), dy, dx) for dy in range(best[1] - 6, best[1] + 7) for dx in range(best[2] - 6, best[2] + 7))
        ox, oy = dbx + best[2], dby + best[1]
        # the offset drifts over the chip (the scans' small rotation and scale): an affine map from the reaction chambers to
        # the buttons found so far, refit a few rounds (fits off by more than 3 robust SDs left out of each)
        affine_res = None
        for _ in range(3):
            # a disc the size of the GFP spot (a disc larger than the spot has a flat top and lands off-center)
            sx, sy, ok = center_on_disc_sum(wash, ox, oy, self.button_spot_r, reach=self.button_reach)
            bright = self._visible(wash, sx, sy, self.button_spot_r, self.button_spot_r + 4, self.button_spot_r + 16, self.tags)
            ok &= bright
            if ok.sum() < 20:
                break
            A = np.c_[dbx[ok], dby[ok], np.ones(ok.sum())]
            keep = np.ones(ok.sum(), bool)
            for _ in range(2):
                cx_, *_ = np.linalg.lstsq(A[keep], sx[ok][keep], rcond=None)
                cy_, *_ = np.linalg.lstsq(A[keep], sy[ok][keep], rcond=None)
                res = np.hypot(A @ cx_ - sx[ok], A @ cy_ - sy[ok])
                keep = res <= np.median(res) + 3 * 1.4826 * np.median(np.abs(res - np.median(res)))
            full = np.stack([dbx, dby, np.ones_like(dbx)], -1)
            ox, oy = full @ cx_, full @ cy_
            affine_res = np.median(res[keep])
        # every button predicted from its place on the grid (an affine map of row and column fit to the buttons found),
        # plus the median miss of its 8 nearest found buttons: local distortion, without leaning on each reaction chamber's
        # own center (left-one-out on MBO-056: 4.8 px median, 12 px 90th percentile)
        sx, sy, ok = center_on_disc_sum(wash, ox, oy, self.button_spot_r, reach=self.button_reach)
        ok &= self._visible(wash, sx, sy, self.button_spot_r, self.button_spot_r + 4, self.button_spot_r + 16, self.tags)
        if ok.sum() >= 20:
            rr, cc = np.indices(ox.shape).astype(float)
            G = np.stack([rr, cc, np.ones_like(rr)], -1)
            gx = G @ np.linalg.lstsq(G[ok], sx[ok], rcond=None)[0]
            gy = G @ np.linalg.lstsq(G[ok], sy[ok], rcond=None)[0]
            fr, fc = rr[ok], cc[ok]
            mx, my = (sx - gx)[ok], (sy - gy)[ok]
            d = np.hypot(rr.ravel()[:, None] - fr[None], cc.ravel()[:, None] - fc[None])
            near = np.argsort(d, axis=1)[:, :8]
            ox = gx + np.median(mx[near], axis=1).reshape(ox.shape)
            oy = gy + np.median(my[near], axis=1).reshape(oy.shape)
        sx, sy, ok = center_on_disc_sum(wash, ox, oy, self.button_spot_r, reach=self.button_reach)
        bright = self._visible(wash, sx, sy, self.button_spot_r, self.button_spot_r + 4, self.button_spot_r + 16, self.tags)
        ok &= bright
        bx, by, fill = shift_from_neighbors(ox, oy, sx, sy, ok)
        save_centers(self.button_centers, bx, by, np.select([ok, fill], ["fit", "neighbors"], "offset"))
        d = np.hypot(bx - ox, by - oy)
        note(f"**Found the buttons** on `{Path(self.button_scan).name}` · {best[2]:+d}, {best[1]:+d} px (x, y) from the reaction "
             f"chambers on average; each predicted from its grid place and its 8 nearest found buttons" +
             f"; {bright.sum():,} with a GFP spot (contrast above the BLANK median + 3 SD)")
        table(pd.DataFrame([{"centered on the spot": ok.sum(), "from neighbors": fill.sum(), "at the offset": (~ok & ~fill).sum(),
                             "moved, median (px)": np.median(d), "max (px)": d.max()}], index=["buttons"]),
              fmt={"centered on the spot": "{:,.0f}", "from neighbors": "{:,.0f}", "at the offset": "{:,.0f}",
                   "moved, median (px)": "{:.1f}", "max (px)": "{:.1f}"})
        self._centering_qc(wash, (ox, oy), (bx, by), self.button_r, ok, fill, "3_buttons/3a-masks", "3a_button_centering",
                           f"buttons on {self.sds_washes[self.button_scan]}: dashed the reaction chamber + offset, solid the centered button",
                           int(2.2 * self.button_r))
        self.buttons = (bx, by)

    def quantify_buttons(self):
        """Every GFP scan (time course, background, washes): the button sum and median, the ring's median, and the button
        sum minus the same button on the background scan. Writes 4_exports/gfp/gfp_per_chamber.csv. QC: the button ROIs
        on the last wash, and the button mosaic (every button as a crop, laid out as the chip; one PDF page). Returns
        the table."""
        from .analysis.plots import button_mosaic, show_masks
        from .processing.grid import load_centers
        from .processing.quantify import centers_to_df

        if not self.button_scan:
            return None
        bx, by = getattr(self, "buttons", None) or load_centers(self.button_centers)
        labels = {**{f"{self.expression_dir}/{n}": f"expression {m:g} min" for n, m in self.expression.items()},
                  "/".join(self.gfp_background): "background", **self.gfp_scans, **self.sds_washes}
        lo, hi = self.button_ring
        gfp = pd.concat([centers_to_df(bx, by, self.pinlist, img=self.image(k), radius=self.button_r, aggs=("median", "sum"), bg=True,
                                       bg_inner=lo, bg_outer=hi, prefix="button", name=lab).assign(scan=lab)
                         for k, lab in labels.items()], ignore_index=True)
        bg_sum = gfp[gfp["scan"] == "background"].set_index(KEYS)["button_sum"]
        gfp["button_sum_minus_bg"] = gfp["button_sum"].to_numpy() - gfp.set_index(KEYS).index.map(bg_sum).to_numpy()
        gfp = gfp[KEYS + ["tag", "scan"] + [c for c in gfp if c not in KEYS + ["tag", "scan"]]]
        (self.exports / "gfp").mkdir(parents=True, exist_ok=True)
        gfp.to_csv(self.exports / "gfp/gfp_per_chamber.csv", index=False)
        cols = list(self.sds_washes.values())   # the background scan itself reads 0 by definition
        ctrl = gfp.groupby(["tag", "scan"])["button_sum_minus_bg"].median().unstack("scan").reindex(["WT", "N41A", "BLANK"])[cols] / 1e3
        note(f"**GFP per button**, {len(labels)} scans · button r {self.button_r:g} px, ring {lo:g}-{hi:g} px · "
             "`4_exports/gfp/gfp_per_chamber.csv`; the mosaic: `5_figures/qc/3_buttons/3a-masks/3a_button_mosaic.pdf`")
        table(ctrl, caption="controls, median button sum minus the background scan (10³ RFU)", fmt="{:,.0f}")
        wash = self.image(self.button_scan)
        label = self.sds_washes[self.button_scan]
        self.save_fig(show_masks(wash, bx, by, radius=self.button_r, bg_inner=lo, bg_outer=hi, title=f"button ROIs, {label}"),
                      "3_buttons/3a-masks", "3a_masks_buttons", show=False, dpi=150)
        vals = gfp[gfp["scan"] == label].set_index(KEYS)["button_sum_minus_bg"].unstack().reindex(index=range(self.n_rows), columns=range(self.n_cols))
        self.save_fig(button_mosaic(wash, bx, by, self.tags, vals.to_numpy(), half=int(1.9 * self.button_r), radius=self.button_r,
                                    title=f"{self.experiment} GFP buttons, {label}", value_label="bg-subtracted button sum"),
                      "3_buttons/3a-masks", "3a_button_mosaic", show=False, exts=("pdf",))
        self.gfp_table = gfp
        return gfp

    # ------------------------------------------------------------------ 3c. Cy5 labeling
    def _cy5_shift(self):
        from .processing.stitch import cached_shift, scan_shift

        bg, post = list(self.cy5_scans)[:2]
        return cached_shift(self.intermediate / f"{Path(post).name}_vs_sds_shift.json",
                            lambda: scan_shift(self.image(self.button_scan), self.image(post)))

    def place_cy5_grid(self, force=None):
        """The Cy5 button grid (interactive) on the post-labeling scan: the GFP buttons moved by one chip-wide shift can
        miss toward the edges when the chip sat differently for the labeling. Opens until 2_masks/cy5_button_corners.json
        exists (or as `force` / `open_cy5_window` says), seeded with the moved GFP buttons; every change saves."""
        from .processing.grid import load_centers
        from .processing.grid_adjuster import GridAdjuster

        if len(self.cy5_scans) < 2:
            return None
        flag = self.open_cy5_window if force is None else force
        if not self.place(flag, self.cy5_corners):
            if self.cy5_corners.exists():
                return note(f"**Cy5 button grid** from `{self.cy5_corners}` (`place_cy5_grid(True)` opens it again)")
            return note("**Cy5 button grid** not placed (the window is off): Cy5 is read on the GFP buttons moved by one shift")
        bx, by = getattr(self, "buttons", None) or load_centers(self.button_centers)
        dy, dx = self._cy5_shift()
        gx, gy = bx + dx, by + dy
        seed = [[gx[0, 0], gy[0, 0]], [gx[0, -1], gy[0, -1]], [gx[-1, -1], gy[-1, -1]], [gx[-1, 0], gy[-1, 0]]]   # TL, TR, BR, BL
        self._interactive()
        post = list(self.cy5_scans)[1]
        if post not in self.stores:
            self.stitch(post)
        self.cy5_window = GridAdjuster(self.stores[post]["image"], columns=self.pinlist, corners=seed, radius=self.button_r,
                                       bg_outer=self.button_ring[1] - self.button_r, corners_path=self.cy5_corners,
                                       warp_path=self.cy5_warp, vmax_pct=99.5)

    def quantify_cy5(self):
        """Cy5 on every button: each Cy5 scan read on the Cy5 button grid (place_cy5_grid; until it is placed, the GFP
        buttons moved by the post-labeling scan's shift from the button scan), button sum minus the same button on the
        background scan (the first of `cy5_scans`). Writes 4_exports/gfp/cy5_per_chamber.csv and labeling_per_chamber.csv.
        QC 3_buttons/3c-cy5: the ROIs, Cy5 over the chip, its distribution, and Cy5 against the bound GFP (the button
        scan, and every GFP state) with the Cy5 / GFP ratio. With `export_tiffs`, the scans and the post-labeling minus
        background as TIFFs in <drive>/4_stitched-tiffs/2_cy5/."""
        from .analysis.labeling import labeling_figure, labeling_states_figure, labeling_table
        from .analysis.plots import button_sum_distribution, plot_chip_map, show_masks
        from .processing.grid import load_centers, warped_grid
        from .processing.quantify import centers_to_df

        if len(self.cy5_scans) < 2:
            return None
        self._inline()
        if self.cy5_corners.exists():
            cx, cy = warped_grid(self.cy5_corners, self.cy5_warp, self.n_rows, self.n_cols)
            where = f"the Cy5 button grid (`{self.cy5_corners.name}`)"
        else:
            bx, by = getattr(self, "buttons", None) or load_centers(self.button_centers)
            dy, dx = self._cy5_shift()
            cx, cy = bx + dx, by + dy
            where = f"the GFP buttons moved by ({dy:.1f}, {dx:.1f}) px (no Cy5 grid placed yet)"
        lo, hi = self.button_ring
        cy5 = pd.concat([centers_to_df(cx, cy, self.pinlist, img=self.image(k), radius=self.button_r, aggs=("median", "sum"), bg=True,
                                       bg_inner=lo, bg_outer=hi, prefix="button", name=lab).assign(scan=lab)
                         for k, lab in self.cy5_scans.items()], ignore_index=True)
        bg_label = next(iter(self.cy5_scans.values()))
        bg_sum = cy5[cy5["scan"] == bg_label].set_index(KEYS)["button_sum"]
        cy5["button_sum_minus_bg"] = cy5["button_sum"].to_numpy() - cy5.set_index(KEYS).index.map(bg_sum).to_numpy()
        cy5 = cy5[KEYS + ["tag", "scan"] + [c for c in cy5 if c not in KEYS + ["tag", "scan"]]]
        (self.exports / "gfp").mkdir(parents=True, exist_ok=True)
        cy5.to_csv(self.exports / "gfp/cy5_per_chamber.csv", index=False)
        post_label = list(self.cy5_scans.values())[1]
        post = cy5[cy5["scan"] == post_label]
        blank = post.loc[post["tag"] == "BLANK", "button_sum_minus_bg"]
        floor = blank.median() + 3 * blank.std()
        note(f"**Cy5 per button** on {where} · {post_label} minus {bg_label} · floor (BLANK median + 3 SD) {floor:,.0f} RFU; "
             f"{int((post.loc[post['tag'] != 'BLANK', 'button_sum_minus_bg'] > floor).sum()):,} of {int((post['tag'] != 'BLANK').sum()):,} "
             "buttons above it · `4_exports/gfp/cy5_per_chamber.csv`")
        table(post.groupby("tag")["button_sum_minus_bg"].median().reindex(["WT", "N41A", "BLANK"]).div(1e3).to_frame("median (10³ RFU)").T,
              fmt="{:,.1f}")
        post_key = list(self.cy5_scans)[1]
        self.save_fig(show_masks(self.image(post_key), cx, cy, radius=self.button_r, bg_inner=lo, bg_outer=hi,
                                 title=f"button ROIs, Cy5 {post_label}"), "3_buttons/3c-cy5", "3c_masks_cy5", show=False, dpi=150)
        self.save_fig(plot_chip_map(post.set_index(KEYS)["button_sum_minus_bg"] / 1e6, f"Cy5 button sum, {post_label}",
                                    "button sum − background scan (10$^6$ RFU)", cmap="Reds", n_rows=self.n_rows, n_cols=self.n_cols),
                      "3_buttons/3c-cy5", "3c_cy5_button_sum_minus_bg")
        self.save_fig(button_sum_distribution(post, label="Cy5 button sum − background scan", floor=floor, floor_label="BLANK median + 3 SD"),
                      "3_buttons/3c-cy5", "3c_cy5_button_sum_distribution")
        # against the bound GFP: the button scan, and every GFP state after expression
        gfp = getattr(self, "gfp_table", None)
        gfp = gfp if gfp is not None else pd.read_csv(self.exports / "gfp/gfp_per_chamber.csv")
        scan = self.sds_washes[self.button_scan]
        lab = labeling_table(cy5, gfp, gfp_scan=scan, cy5_scan=post_label, n_cols=self.n_cols)
        lab.to_csv(self.exports / "gfp/labeling_per_chamber.csv", index=False)
        self.save_fig(labeling_figure(lab, title=f"{self.experiment} · Cy5 {post_label} vs GFP {scan}", n_cols=self.n_cols),
                      "3_buttons/3c-cy5", "3c_cy5_vs_gfp")
        states = [s for s in dict.fromkeys(gfp["scan"]) if s != "background"]
        fig, _ = labeling_states_figure(cy5, gfp, states, cy5_scan=post_label, n_cols=self.n_cols,
                                        title=f"{self.experiment}: Cy5 {post_label} vs the bound GFP at each state")
        self.save_fig(fig, "3_buttons/3c-cy5", "3c_cy5_vs_gfp_by_gfp_state", show=False)
        if self.export_tiffs:
            import tifffile

            from .processing.stitch import export_stitched_tiff

            out = self.drive / "4_stitched-tiffs/2_cy5"
            for k in self.cy5_scans:
                export_stitched_tiff(self.stitched / f"{k}.zarr", out / f"{Path(k).name}_stitched.tif")
            bg_key = next(iter(self.cy5_scans))
            tifffile.imwrite(out / f"{Path(post_key).name.replace('post_labeling', 'post_labeling_minus_bg')}_stitched.tif",
                             (self.image(post_key) - self.image(bg_key)).astype(np.float32), imagej=True,
                             metadata={"Info": f"{post_key} minus {bg_key}, pixel by pixel (same session, no shift between them)"})
        self.cy5_table = cy5
        return cy5

    # ------------------------------------------------------------------ 3e. on-chip click labeling
    def _click_seed(self):
        """The click grid's first placement: the Cy5 grid if placed, else the GFP buttons (TL, TR, BR, BL corners)."""
        from .processing.grid import load_centers, warped_grid

        if self.cy5_corners.exists():
            gx, gy = warped_grid(self.cy5_corners, self.cy5_warp, self.n_rows, self.n_cols)
        else:
            gx, gy = getattr(self, "buttons", None) or load_centers(self.button_centers)
        gx, gy = np.asarray(gx), np.asarray(gy)
        return [[gx[0, 0], gy[0, 0]], [gx[0, -1], gy[0, -1]], [gx[-1, -1], gy[-1, -1]], [gx[-1, 0], gy[-1, 0]]]

    def place_click_grid(self, force=None, frame=-1):
        """The button grid for the click labeling time courses (interactive), on a pass (`frame`, default the last) of the
        first click series: the chip can sit differently for the labeling than for the GFP or Cy5 scans. Opens until
        2_masks/click_button_corners.json exists (or as `force` / `open_click_window` says), seeded with the Cy5 grid
        (else the GFP buttons); every change saves."""
        from .processing.grid_adjuster import GridAdjuster

        if not self.click_series:
            return None
        flag = self.open_click_window if force is None else force
        if not self.place(flag, self.click_corners):
            if self.click_corners.exists():
                return note(f"**Click button grid** from `{self.click_corners}` (`place_click_grid(True)` opens it again)")
            return note("**Click button grid** not placed (the window is off): the click series are read on the Cy5 grid, else the GFP buttons")
        key = next(iter(self.click_series))
        if key not in self.stores:
            self.stitch(key)
        seed = None if self.click_corners.exists() else self._click_seed()
        self._interactive()
        self.click_window = GridAdjuster(self.stores[key]["image"], columns=self.pinlist, frame=frame, corners=seed,
                                         radius=self.button_r, bg_outer=self.button_ring[1] - self.button_r,
                                         corners_path=self.click_corners, warp_path=self.click_warp, vmax_pct=99.5)

    def quantify_click(self):
        """Cy5 at every pass of every click series: on the click button grid (place_click_grid; until it is placed, the
        Cy5 grid, else the GFP buttons) the button sum and median and the ring inside the rim (with the buttons closed for
        imaging, the ring sits in the chamber's dye: the QC plots the button median and the ring apart); the DNA
        chambers' and the reaction (assay) chambers' medians. Each pass carries the
        buttons' cumulative open time (buttons_open_s, from the acquisition). Writes
        4_exports/gfp/cy5_click_timecourse_per_chamber.csv. QC 3_buttons/3e-click: the ROIs on the last pass, Cy5 vs
        open time per group (change from each series' pass 0), and the last pass's button median change vs the button GFP."""
        import zarr

        from .analysis.plots import show_masks
        from .processing.grid import load_centers, warped_grid
        from .processing.quantify import centers_to_df

        if not self.click_series:
            return None
        self._inline()
        if self.click_corners.exists():
            bx, by = warped_grid(self.click_corners, self.click_warp, self.n_rows, self.n_cols)
            where = f"the click button grid (`{self.click_corners.name}`)"
            if self.click_centers in ("gfp", "dark") and self.button_centers.exists():
                # the GFP buttons, each where it sat on the GFP scan, mapped onto the click images by one affine fit of the
                # GFP buttons to the click grid (keeps each button's own offset from the lattice)
                gx, gy = load_centers(self.button_centers)
                gx, gy, bx, by = (np.asarray(v, float) for v in (gx, gy, bx, by))
                A = np.c_[gx.ravel(), gy.ravel(), np.ones(gx.size)]
                coef, *_ = np.linalg.lstsq(A, np.c_[bx.ravel(), by.ravel()], rcond=None)
                fit = A @ coef
                resid = np.hypot(fit[:, 0] - bx.ravel(), fit[:, 1] - by.ravel())
                bx, by = fit[:, 0].reshape(gx.shape), fit[:, 1].reshape(gx.shape)
                where = (f"the GFP buttons mapped onto the click grid (`{self.click_corners.name}`; affine, median distance to the grid "
                         f"{np.median(resid):.1f} px)")
        elif self.cy5_corners.exists():
            bx, by = warped_grid(self.cy5_corners, self.cy5_warp, self.n_rows, self.n_cols)
            where = "the Cy5 button grid (no click grid placed yet)"
        else:
            bx, by = getattr(self, "buttons", None) or load_centers(self.button_centers)
            where = "the GFP buttons (no click or Cy5 grid placed yet)"
        regions = {}
        if (self.masks / "dna_chamber_centers.csv").exists():
            regions["dna"] = (*load_centers(self.masks / "dna_chamber_centers.csv"), self.dna_r)
        if self.chamber_centers.exists():
            regions["reaction"] = (*load_centers(self.chamber_centers), self.chamber_r)
        lo, hi = self.button_ring
        r_click = self.click_button_r or self.button_r
        if self.click_centers == "dark":
            # the GFP and Cy5 images sit a few px apart, unevenly over the chip: each ROI moves by the smoothed offset to
            # where the first series' first pass best matches the closed button's mean pattern (its dark core in the dye)
            from .processing.grid import template_offsets

            first = next(iter(self.click_series))
            if first not in self.stores:
                self.stitch(first)
            da0 = self.stores[first]["image"]
            ref = np.asarray((da0.isel(time=0) if "time" in da0.dims else da0).values, np.float32)
            dx, dy, _, _ = template_offsets(ref, bx, by, reach=self.click_reach)
            bx, by = np.asarray(bx) + dx, np.asarray(by) + dy
            where += (f", moved onto the dark button cores of the first pass (offset field smoothed over 5 x 5 chambers; median "
                      f"{np.median(np.hypot(dx, dy)):.1f} px, at most {np.max(np.hypot(dx, dy)):.1f} px)")
        parts = []
        for key, label in self.click_series.items():
            if key not in self.stores:
                self.stitch(key)
            da = self.stores[key]["image"]
            raw = next((p for p in (self.raw / key, self.drive / key) if p.exists()), None)
            grp = zarr.open(str(next(raw.glob("*/zarr.json")).parent), mode="r") if raw is not None and any(raw.glob("*/zarr.json")) else None
            open_s = np.asarray(grp["buttons_open_s"]) if grp is not None and "buttons_open_s" in grp else None
            secs = np.asarray(grp["seconds"]) if grp is not None and "seconds" in grp else None
            for t in range(da.sizes.get("time", 1)):
                im = np.asarray((da.isel(time=t) if "time" in da.dims else da).values, dtype=np.float32)
                d = centers_to_df(bx, by, self.pinlist, img=im, radius=r_click, aggs=("median", "sum"), bg=True,
                                  bg_inner=lo, bg_outer=hi, prefix="button")
                for name, (rx, ry, r) in regions.items():
                    e = centers_to_df(rx, ry, self.pinlist, img=im, radius=r, aggs=("median",), prefix=name)
                    d = d.merge(e[KEYS + [f"{name}_median"]], on=KEYS, how="left")
                d.insert(3, "series", label)
                d.insert(4, "pass", t)
                d.insert(5, "buttons_open_s", float(open_s[t]) if open_s is not None else np.nan)
                d.insert(6, "minutes", float(secs[t].mean() / 60) if secs is not None else np.nan)
                parts.append(d)
        click = pd.concat(parts, ignore_index=True)
        (self.exports / "gfp").mkdir(parents=True, exist_ok=True)
        click.to_csv(self.exports / "gfp/cy5_click_timecourse_per_chamber.csv", index=False)
        n_pass = click.groupby("series")["pass"].nunique()
        note(f"**Click labeling** on {where}, button ROI r = {r_click:g} px · " + ", ".join(f"{s}: {n} passes" for s, n in n_pass.items())
             + " · `4_exports/gfp/cy5_click_timecourse_per_chamber.csv`")
        last = next(iter(self.click_series))
        self.save_fig(show_masks(np.asarray(self.stores[last]["image"].isel(time=-1).values, np.float32), bx, by, radius=r_click,
                                 bg_inner=lo, bg_outer=hi, title=f"button ROIs, {self.click_series[last]}, last pass"),
                      "3_buttons/3e-click", "3e_masks_click", show=False, dpi=150)
        from .analysis.labeling import click_figures

        enz = self._enzyme_call()
        figs = click_figures(click, enz, title=f"{self.experiment} click labeling", n_rows=self.n_rows, n_cols=self.n_cols)
        for name, fig in figs.items():
            self.save_fig(fig, "3_buttons/3e-click", name)
        self.click_centers_xy = (bx, by)
        self.click_table = click
        return click

    def _enzyme_call(self):
        """The button GFP per chamber and its floor call, for the click QC: the analysis notebook's call
        (4_exports/gfp/enzyme_per_chamber.csv) when it exists, else the button scan's button_sum_net vs the BLANKs."""
        p = self.exports / "gfp/enzyme_per_chamber.csv"
        if p.exists():
            return pd.read_csv(p)[KEYS + ["gfp", "expressing"]]
        g = pd.read_csv(self.exports / "gfp/gfp_per_chamber.csv")
        g = g[g["scan"] == self.sds_washes.get(self.button_scan, self.button_scan)].rename(columns={"button_sum_net": "gfp"})
        b = g.loc[g["tag"] == "BLANK", "gfp"]
        g["expressing"] = g["gfp"] > b.median() + 3 * 1.4826 * (b - b.median()).abs().median()
        return g[KEYS + ["gfp", "expressing"]]

    # ------------------------------------------------------------------ 4. assay
    def place_chambers(self, force=None):
        """The assay's chamber grid (interactive) on the highest [S] run's last pass, seeded from the reaction chambers
        (else `seed_corners`); then each ROI centered on its chamber's shape (`center_reach`). Writes
        2_masks/assay_chamber_centers.csv. QC: the ROIs on the run's last pass. The centers are kept in `exp.chambers`."""
        from .analysis.plots import show_masks
        from .processing.grid import center_on_disc_sum, load_mask, save_centers, shift_from_neighbors, warped_grid
        from .processing.grid_adjuster import GridAdjuster

        if not self.reference_run:
            return None
        flag = self.open_chamber_window if force is None else force
        if self.reference_run not in self.stores:
            self.stitch(self.reference_run)
        if self.place(flag, self.chamber_corners):
            self._interactive()
            seed = None if self.chamber_corners.exists() else (load_mask(self.reaction_corners)["corners"] if self.reaction_corners.exists()
                                                               else load_mask(self.seed_corners)["corners"] if self.seed_corners else None)
            GridAdjuster(self.stores[self.reference_run]["image"], columns=self.pinlist, frame=-1, corners=seed, radius=self.chamber_r,
                         bg_outer=self.chamber_ring[1] - self.chamber_r, corners_path=self.chamber_corners, warp_path=self.chamber_warp,
                         vmax_pct=99.5)
            self._inline()
        gx, gy = warped_grid(self.chamber_corners, self.chamber_warp, self.n_rows, self.n_cols)
        if self.center_on == "mean":   # every pass averaged (as MBO-054 placed them)
            ref = np.asarray(self.stores[self.reference_run]["image"].mean("time").values, np.float32)
        else:
            ref = self.image(self.reference_run)   # the last pass: the most product, the chambers' outlines clearest
        sx, sy, ok = center_on_disc_sum(ref, gx, gy, self.chamber_r, reach=self.center_reach)
        kx, ky, fill = shift_from_neighbors(gx, gy, sx, sy, ok)
        save_centers(self.chamber_centers, kx, ky, np.select([ok, fill], ["fit", "neighbors"], "grid"))
        note(f"**Chamber masks** on `{Path(self.reference_run).name}` · centered {ok.sum():,}, from neighbors {fill.sum():,}, "
             f"grid {(~ok & ~fill).sum():,}")
        self.save_fig(show_masks(self.image(self.reference_run), kx, ky, radius=self.chamber_r, bg_inner=self.chamber_ring[0],
                                 bg_outer=self.chamber_ring[1], title=f"chamber ROIs, {self.reference_run}, last pass"),
                      "5_assay/5a-chambers", "5a_chamber_masks", show=False, dpi=150)
        self.chambers = (kx, ky)

    def _chamber_centers(self):
        from .processing.grid import load_centers

        return getattr(self, "chambers", None) or load_centers(self.chamber_centers)

    def quantify_assay(self, force=False):
        """The chamber ROI median (RFU) at every pass of every run, each chamber timed by its own tile; every pass kept,
        flagged `complete_pass` and `tile_imaged`. With the expression time course's dumbbells, also the DNA chamber's
        median at every pass (`dna_chamber_median`; the neck is closed during an assay, so a rise means diffusion under
        it, or activity in the DNA chamber). Each frame is read once for both masks. A run is quantified once and kept in
        3_intermediate/assay/<run>.csv, reused while its masks, radii and stitched store are unchanged (`force=True`
        re-reads every run). Writes 4_exports/assay/nadh_per_chamber.csv, one nadh_per_chamber_<name>.csv per SERIES (each
        with its own [S] column) and assay/series.json, their index for the analysis notebook. QC: the control traces, per
        series. Returns ASSAY_RUNS' table (else the first series'); every series' is kept in `exp.nadh_series`."""
        import hashlib

        from .analysis.plots import plot_control_traces
        from .processing.grid import load_mask
        from .processing.quantify import chamber_timecourse

        if not self.reference_run:
            return None
        kx, ky = self._chamber_centers()
        # each chamber's DNA chamber, at its own offset from the reaction chamber (from the expression centering)
        dna = None
        if (self.masks / "reaction_chamber_centers.csv").exists() and (self.masks / "dna_chamber_centers.csv").exists():
            (rx, ry), (dx_, dy_) = self._dumbbell_centers()
            dna = (np.asarray(kx) + (np.asarray(dx_) - np.asarray(rx)), np.asarray(ky) + (np.asarray(dy_) - np.asarray(ry)))
            dna_r = load_mask(self.dna_corners)["radius"]
        cache = self.intermediate / "assay"
        cache.mkdir(parents=True, exist_ok=True)
        masks_key = hashlib.sha1(np.round(np.stack([kx, ky, *(dna if dna is not None else ())]), 2).tobytes()
                                 + repr((self.chamber_r, self.chamber_ring, dna_r if dna is not None else None)).encode()).hexdigest()
        reused = 0
        series = {**({None: (self.assay_dir, {run: {self.substrate: s_uM} for run, s_uM in self.assay_runs.items()})} if self.assay_runs else {}),
                  **{name: (v["dir"], v["runs"]) for name, v in self.series.items() if v["runs"]}}
        outputs = {}
        for name, (folder, runs) in series.items():
          tables = []
          for run, meta in runs.items():
            key = f"{folder}/{run}"
            if key not in self.stores:
                self.stitch(key)
            s = self.stores[key]
            sec = s["tiled"]["seconds"].values.reshape(s["tiled"].sizes["time"], -1)
            complete = (sec > 0).all(axis=1)
            store_file = self.stitched / f"{key}.zarr" / "zarr.json"
            run_key = masks_key + (f"{store_file.stat().st_mtime:.0f}" if store_file.exists() else "")
            here = cache if name is None else cache / name   # a series' runs apart: run folders can share names across series
            here.mkdir(exist_ok=True)
            csv, keyfile = here / f"{run}.csv", here / f"{run}.key"
            if not force and csv.exists() and keyfile.exists() and keyfile.read_text() == run_key:
                t = pd.read_csv(csv)
                reused += 1
            else:
                also = [dict(cx=dna[0], cy=dna[1], radius=dna_r, prefix="dna_chamber")] if dna is not None else ()
                t = chamber_timecourse(s["image"], kx, ky, self.pinlist, s["tiled"], s["store"].attrs["tile_px"],
                                       radius=self.chamber_r, ring=self.chamber_ring, aggs=("median",), prefix="chamber_ROI",
                                       name=run, also=also)
                t.to_csv(csv, index=False)
                keyfile.write_text(run_key)
            tables.append(t.assign(run=run, minutes=t["seconds"] / 60, tile_imaged=t["seconds"] > 0,
                                   complete_pass=complete[t["frame"].to_numpy()], **meta))
            label = ", ".join(f"{x:g} µM" if isinstance(x, (int, float)) else str(x) for x in meta.values())
            self._passes = getattr(self, "_passes", {}) | {(f"{name}: " if name else "") + label: (int(complete.sum()), len(complete))}
          outputs[name] = pd.concat(tables, ignore_index=True)
        (self.exports / "assay").mkdir(parents=True, exist_ok=True)
        n_runs = len(self.assay_runs) + sum(len(v["runs"]) for v in self.series.values())
        files = []
        for name, t in outputs.items():
            f = "nadh_per_chamber.csv" if name is None else f"nadh_per_chamber_{name}.csv"
            t.to_csv(self.exports / "assay" / f, index=False)
            files.append(f"`4_exports/assay/{f}`")
            sub = self.substrate if name is None else self.series[name]["substrate"]
            self.save_fig(plot_control_traces(t, "chamber_ROI_median", s=sub), "5_assay/5b-nadh",
                          "5b_control_traces" + (f"_{name}" if name else ""))
        # the analysis notebook finds each series' table and [S] column here
        index = {(name or ""): dict(file=("nadh_per_chamber.csv" if name is None else f"nadh_per_chamber_{name}.csv"),
                                    substrate=self.substrate if name is None else self.series[name]["substrate"],
                                    dir=folder, runs=len(runs))
                 for name, (folder, runs) in series.items()}
        import json

        (self.exports / "assay/series.json").write_text(json.dumps(index, indent=1))
        note(f"**NADH per chamber**, {n_runs} runs ({n_runs - reused} read, {reused} unchanged since last time) · " + ", ".join(files))
        table(pd.DataFrame(self._passes, index=["complete passes", "passes"]).T, fmt="{:,.0f}")
        self.nadh_series = outputs
        return outputs.get(None, next(iter(outputs.values())))

    # ------------------------------------------------------------------ 5. standards
    def quantify_standards(self):
        """Each camera's NADH standards read with the assay's chamber masks and ROI, no flatfield (the calibration is fit
        in the analysis notebook). Writes 4_exports/assay/standards/<camera>/nadh_standards_per_chamber.csv. Returns
        {camera: table}."""
        from .processing.quantify import centers_to_df

        if not self.standards:
            return None
        from .processing.grid import load_centers

        out = {}
        for cam, runs in self.standards.items():
            folder = self.exports / "assay/standards" / cam
            folder.mkdir(parents=True, exist_ok=True)
            own = self.standard_masks.get(cam)   # this camera's own chamber masks, else the assay's
            kx, ky = load_centers(own["centers"]) if own else self._chamber_centers()
            r, (lo, hi) = (own["radius"], tuple(own["ring"])) if own else (self.chamber_r, self.chamber_ring)
            std = pd.concat([centers_to_df(kx, ky, self.pinlist, img=self.image(f"Standards/{cam}/{n}"), radius=r,
                                           aggs=("median",), bg=True, bg_inner=lo, bg_outer=hi,
                                           prefix="chamber_ROI", name=n).assign(camera=cam, nadh_uM=conc, standard=n)
                             for n, conc in runs.items()], ignore_index=True)
            std.to_csv(folder / "nadh_standards_per_chamber.csv", index=False)
            out[cam] = std
        med = pd.DataFrame({cam: t.groupby("nadh_uM")["chamber_ROI_median"].median() for cam, t in out.items()})
        med.index = [f"{v:g} µM" for v in med.index]
        note(f"**NADH standards**, {', '.join(out)} · `4_exports/assay/standards/<camera>/nadh_standards_per_chamber.csv`")
        table(med, caption="median chamber ROI (RFU)", fmt="{:,.0f}")
        return out
