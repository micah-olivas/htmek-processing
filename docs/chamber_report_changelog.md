# Chamber report: changelog

The version is `REPORT_VERSION` in `src/htmek_processing/report/chamber.py`, printed in each
page's footer with the build date. **When the report changes, bump `REPORT_VERSION` and add
an entry at the top.** Entries up to v1.9.4 were written in MBO-040, when the report lived in
that experiment's `chip_utils.py` (`cu.`); notebook cell numbers there are MBO-040's.

---

## v1.10.0 (2026-10-30)

- **Flowchart first page:** every filter the pipeline ran, in order, with its rule, the
  parameters it ran with, and how many points, runs or chambers it left out, changed or
  kept. Drawn from the steps the functions record as they run (`hp.pipeline_steps()`,
  saved by the process notebook as `4_exports/assay/nadh_rate_steps.json`), then the
  report's own: the MM fits' r² filter and the enzyme floor. Two pages: image processing
  (stitching, grid placement, per-chamber readouts, NADH calibration), then rates and fits.
- **k<sub>cat</sub>/K<sub>m</sub> in M⁻¹ s⁻¹** with `enzyme_rfu_per_nm` (enzyme readout per
  nM enzyme). With `enzyme_placeholder=True` the strip carries an amber "PLACEHOLDER [E]"
  tag and the flowchart an amber step. MBO-051 uses a placeholder 150,000 RFU (GFP button
  sum) per nM.
- **Strips:** k<sub>cat</sub>/K<sub>m</sub> and v<sub>0</sub> at the highest [S] under the
  K<sub>m</sub> and v<sub>max</sub> strips. Each strip's measurement name sits inside it, top
  left, above the median labels; the axis label is the unit alone; the chamber counts are
  gone. The strips are closer together.
- **Progress curves:** any number of runs across the page; v<sub>0</sub> under each
  run's [S]; x ticks in seconds; the endpoint [NADH] (net of the BLANKs' first frame) by
  the last point.
- **Standards:** labels on the fitted line, in ascending order; inset up to 62.5 µM.

## v1.9.4 (2026-10-13)

- **No key under the strips:** the line with "other variants" and "off scale (at the edge)"
  at the bottom of each page is gone. Both are explained on the key page.
- **v<sub>max</sub>:** written with a subscript (was "Vmax") in the table, the normalized MM
  panel's WT line and legend, the strip title and axis, and the key texts.

## v1.9.3 (2026-10-12)

- **v₀ on the MM panels:** their y axes read "v₀ (µM/s)" and "v₀ / gfp", each run's
  initial rate, not Vmax (the fit's plateau). The trace legend's tangent is "v₀ (4PL
  tangent)", and the key page says so too.
- **K<sub>m</sub>:** written with a subscript everywhere on the page and the key (table,
  strip title and axis, key texts).
- **One legend per symbol:** the key under the strips lists only what the strips add (other
  variants, off scale). This chamber, its variant's chambers and open markers are in the
  legend beside the MM panels.

## v1.9.2 (2026-10-09)

- **WT reference on the normalized MM panel:** a dashed red line at WT's Vmax / [E] (the
  median of the passing WT chambers, the same value as the WT mark on the Vmax / [E] strip),
  labeled "WT Vmax" and listed in the key.

## v1.9.1 (2026-10-07)

- **Spelling:** American throughout the page and key text (gray, color, center, normalized,
  analyzed, neighbor, labeled).

## v1.9 (2026-10-07)

- **Vmax / [E] strip:** next to the Km strip, every chamber's Vmax per [E] (normalized
  fits, chambers above the [E] floor) on a log axis, with the same marks:
  - gray for the other variants;
  - blue for this variant (open: r² < 0.9);
  - dashed WT and variant medians.

  The axis runs from the 1st to the 99th percentile of the passing fits whose Km lies
  inside the [BzP] range. Where Km is beyond it, Vmax is extrapolated and sits at the edge
  as an off-scale triangle.
- **Layout:** Km now spans columns 1–3 and Vmax / [E] columns 4–6, with one shared key in
  a row below both.
- **Key page:** a new entry for the Vmax / [E] strip.

## v1.8.1 (2026-10-07)

- **Experiment name:** the top right reads "hAcyP1 AFMtag Canonical · 2026-10-02" (the
  library and start date). MBO-040 is a personal index and stays in the file name only.

## v1.8 (2026-10-07)

- **Conditional passes:** a chamber whose MM fit to all six runs misses r² ≥ 0.9, but
  passes with one run left out, now uses that fit (notebook 3k, `KM_RESCUE_ONE_OUT`). Its
  page shows:
  - the left-out point as an × in the MM panels (also for the variant's other chambers);
  - the curve fitted without it;
  - a "fit" row in the table ("all runs", or "w/o 2,500" in amber);
  - an amber "conditional" verdict.

  Conditional passes count as passing everywhere (the Km strip, the medians, the open or
  filled markers).
- **Key page:** describes the "fit" row, the amber verdict and the ×.

## v1.7 (2026-10-07)

- **Key page:** the last page is a guide to the page. One chamber's page (set by
  `REPORT_KEY_CHAMBER` in notebook 3l; K72M, row 13 | column 5) is shown shrunk, with every
  element outlined in magenta (a color the pages never use) and numbered. Text and groups
  get a snug box, single points a ring, and lines a highlight along their path. The
  numbers are explained in reading order beside and under the example, with the
  conventions that hold across the page. The report now has one page per chamber plus the
  key.
- **Button-QC tab wording:** "QC: oddest n%" now reads "QC: anomaly 99th pct": the
  percentile of the button's anomaly score among the chip's buttons in that image (red
  95th–99th, amber 85th–94th). Same thresholds as before.

## v1.6 (2026-10-06)

- **Tile-seam notches:** a tile seam running through an image crop is marked by a pair of
  amber notches on the crop's edges, top and bottom for a vertical seam, left and right
  for a horizontal one. Each image's seams come from its own stitched tile pitch
  (`tile_px`).
- **Button-QC tab:** moved to the bottom left of the crop, outlined in black, in lighter,
  brighter colors. "QC: top n%" now reads "QC: oddest n%", so the direction is clear.
- **File name:** the version's dots are written as dashes (`chamber_report_v1-6.pdf`).

## v1.5 (2026-10-06)

- **Button-QC tabs:** the GFP and Cy5 crops carry a tab in their upper right, from
  chamberqc (`methods/button_qc.md`). For now it ranks how unusual the button looks among
  that image's buttons: red for the top 5 %, amber for the next 10 %, green otherwise. Once
  chamberqc has a classifier, the tab will say "bad", with a probability and a reason.

## v1.4 (2026-10-06)

- **Experiment on every page:** "MBO-040 · hAcyP1 AFMtag Canonical · 2026-10-02" (name,
  library, date of the kinetics) at the top right. It is set by `EXPERIMENT`, `LIBRARY`
  and `EXPERIMENT_DATE` in the notebook's setup cell.

## v1.3.3 (2026-10-06)

- **Km strip:** every gray point (other variants) has the same small size and color,
  whether its fit passes r² ≥ 0.9 or not. The title still gives both counts.

## v1.3.2 (2026-10-06)

- **Normalized MM panel:** dropped the "(this chamber: [E] < floor)" note from its label.
  The table already shades a below-floor [E] red.

## v1.3.1 (2026-10-06)

- **Km strip:** all gray points render the same gray. The small ones (failing fits) were
  rendering lighter through anti-aliasing; they now have an edge in the same color.
- **Km strip:** a variant's median label flips left of its line when it would run past
  the plot's right edge (a median off scale, at 5 mM).

## v1.3 (2026-10-06)

- **Km strip:**
  - Every chamber with a Km is plotted. Fits with r² ≥ 0.9 are full-size gray points and
    the rest small gray points; the title gives both counts.
  - The variant's median label sits a row below WT's, so WT-like medians don't collide. It
    stays right of its line, unless it would run into the WT label or line: then it moves
    left of its own line, on a white background.
- **Table:** cells that fail a filter are shaded red: r² < 0.9, as before, and now the [E]
  readout in use (GFP sum) when it is below the [E] floor. A tile seam is shaded amber (a
  flag; seam chambers stay in the analysis).
- **Image crops:** full black frames.

## v1.2 (2026-10-06)

- **Tile seams in both directions.** The table's "tile seam" row now says whether the
  chamber's NADH ROI crosses a vertical seam, a horizontal one, both, or neither (it used
  to cover horizontal seams only, and used the GFP grid). 357 chambers are on a seam: 155
  vertical, 220 horizontal.
- **MM panels:**
  - The y-axis labels sit above the panels, and the titles are gone, so the normalized
    panel moves from col 4 to col 3.
  - The normalized panel shows only the variant's other chambers that pass the filter
    (this chamber is always shown).
  - Its values are shown ×10^n, in the label, instead of matplotlib's offset text.
- **Km beyond the highest [BzP]** shows as "> 5,000" in the table and in the Km strip's
  legend. This is display only; the fitted value is unchanged in the exports.
- **Km strip:** if the variant's median label would overlap WT's, it drops to a second row.
- **Chip map:** no frame.
- **Key:** moved left, next to the normalized panel, so its longest line isn't cut off.

## v1.1 (2026-10-06)

- **No Km for chambers with no activity.** When an MM fit ends with Vmax on its lower
  bound (0), the curve is flat and Km was never fitted: it stayed at the starting guess,
  the median [BzP] (937.5 µM). That affected 372 chambers (all with r² < 0), which showed up
  as a pile of open markers at ~940 µM. Their Km is now undefined (NaN, `km_note` says
  why): the table shows "–" and the Km strip has no point for them.
- **Km strip:** the points sit in a shorter band (jitter ±0.2), leaving room above for the
  WT and variant median labels.

## v1.0 (2026-10-06)

The first versioned report. A page shows, top to bottom:

- **Title:** row | col pill, then the variant.
- **Table:**
  - MM fit (raw and per [E]): Km, Vmax, r².
  - Chamber: GFP sum, Cy5 sum, tile seam.
  - A pass/fail row for r² ≥ 0.9; failing r² cells are shaded red.
- **Chip map:** this chamber and the variant's other chambers marked.
- **Image crops:** GFP, washed Cy5, NADH 5 mM, each with the ROI.
- **Progress curves**, one panel per [BzP]:
  - NADH vs time with the rate fit (4PL with its vmax tangent, or a line).
  - An inset with the model and rate.
- **MM panels:**
  - Raw and normalized rate vs [BzP], with this chamber's fit and the variant's other
    chambers overlaid.
  - No BLANK chambers.
- **Km strip:**
  - Every chamber's Km (r² ≥ 0.9), with this variant's chambers and this chamber
    highlighted.
  - Labeled WT and variant median lines.
  - Km > 5 mM shown as triangles at the edge.

Conventions:

- A chamber whose MM fit fails the filter (r² < 0.9) is drawn **open** (dashed curves)
  everywhere on the page: chip map, progress points, MM panels and Km strip.
- [BzP] axes are linear from 0. MM panels are square.
- Km is never capped.

Rules behind the numbers on the page are in `methods/` (3i rate fits, 3j normalization,
3k MM fits).

### Before v1.0 (unversioned, 2026-10-03 to 10-06)
- **Layout:** grew from a first draft (crops, info text, progress curves, raw and
  normalized rate) to the gridded layout above. Font sizes went up, the table was merged
  into one gridded table, and the Km strip, chip map and open/filled styling were added.
- **Rate rules changed under it:** the 4PL is constrained to rise, plus the noise and
  substrate rules (`methods/3i_rate_fits.md`).
- **Other changes:**
  - [E] switched from Cy5 to GFP.
  - The Km cap was removed.
  - The Km comparison cutoff was raised from r² ≥ 0.8 to 0.9.
