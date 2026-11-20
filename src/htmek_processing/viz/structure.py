"""Per-variant kinetics beside the protein structure, as one standalone HTML page.

The plot (plotly) shows each variant's K_m against V_max / [E] (median over its chambers, bars the IQR); the structure
(PDBe Mol*, pdbe-molstar 3.12.0, Apache-2.0, vendored in `viz/data/`) shows the protein, with fixed regions as sticks.
They are linked both ways: hovering a variant shows its residue as sticks; clicking a variant or a residue pins that
position (its variants enlarged and labeled, the rest faded) until it is clicked again, the background is clicked, Esc,
or "clear". Everything (plotly.js, the viewer, the PDB) is inside the file, so it opens offline (~11 MB)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
VENDOR = DATA / "pdbe-molstar-3.12.0"

# human acylphosphatase 1 (PDB 2VH7, residues 5-98; numbering matches the AFMtag libraries)
ACYP1_PDB = DATA / "2VH7.pdb"
ACYP1_REGIONS = [dict(name="P-loop", residues=list(range(17, 24)), color="#FF8C00"),        # 17-23 (MBO-046)
                 dict(name="MVC loop", residues=list(range(69, 76)), color="#2CA02C"),      # G69-I75
                 dict(name="active site", residues=[23, 41], color="#6495ED", labels=["R23", "N41"])]   # drawn last: R23 blue

# a substitution's color, by the tag's last character (the pinlist's substituted residue; "*" a TAG codon)
SUBSTITUTIONS = {"A": ("Ala", "#87D1D5"), "F": ("Phe", "#CE95EC"), "M": ("Met", "#E2B345"), "*": (None, "#8E44AD")}
AA = dict(zip("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(), "ARNDCQEGHILKMFPSTWYV"))


def variant_iqr(per_chamber, *, km="km", vg="vmax", scale=1.0, min_chambers=3, min_r2=None, exclude=("WT", "N41A", "BLANK")):
    """Per variant: the median and IQR over its chambers of K_m (`km`, µM) and V_max / [E] (`vg` x `scale`), and n.
    Chambers need a positive K_m and rate (and r2 >= `min_r2` when given); variants at least `min_chambers` chambers;
    `exclude` tags are left out. Returns a table indexed by tag: n, km, km_lo, km_hi, vg, vg_lo, vg_hi."""
    d = per_chamber.dropna(subset=[km, vg])
    d = d[(d[km] > 0) & (d[vg] > 0) & ~d["tag"].isin(exclude)]
    if min_r2 is not None:
        d = d[d["r2"] >= min_r2]
    d = d.assign(_km=d[km], _vg=d[vg] * scale)
    q = lambda x: (lambda v: v.quantile(x))
    pv = d.groupby("tag").agg(n=("_km", "size"), km=("_km", "median"), km_lo=("_km", q(.25)), km_hi=("_km", q(.75)),
                              vg=("_vg", "median"), vg_lo=("_vg", q(.25)), vg_hi=("_vg", q(.75)))
    return pv[pv["n"] >= min_chambers]


def _position(tag):
    import re

    m = re.search(r"\d+", str(tag))
    return int(m.group()) if m else None


def structure_interactive(variants, path, *, title="V_max / [E] vs K_m on the structure", pdb=ACYP1_PDB, structure_name="PDB 2VH7",
                          regions=ACYP1_REGIONS, substitutions=None, references=None, km_line=None, s_range_uM=None,
                          y_label="V<sub>max</sub> / GFP, variant median", check_sequence=True):
    """Write the linked plot-and-structure page to `path` (HTML) and return it.

    variants        variant_iqr's table (indexed by the pinlist tag, e.g. "K98*"; K_m in µM)
    pdb             the structure file (default PDB 2VH7, human ACYP1); `regions` its residues always drawn as sticks
                    (list of dict(name, residues, color[, labels]))
    substitutions   {tag's last character: (legend name or None for the tag display name, color)} (default SUBSTITUTIONS)
    references      reference chambers drawn as stars (e.g. WT): list of dict(km (µM), vg, label, color="#E67E22")
    km_line         (K_m in µM, legend note) for a dashed vertical line, e.g. WT's median K_m
    s_range_uM      (lowest, highest) [S]: the K_m axis stays within a quarter of the lowest to four times the highest
    check_sequence  every variant's wild-type residue must match the structure's at its position"""
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.offline import get_plotlyjs

    from ..processing.layout import TAG_SUFFIX, display_tag

    pv = variants.copy()
    subs = dict(SUBSTITUTIONS if substitutions is None else substitutions)
    subs = {k: ((n if n is not None else (TAG_SUFFIX if k == "*" else k)), c) for k, (n, c) in subs.items()}
    references = list(references or [])

    # ---- the structure: protein atoms and the deposited secondary structure
    pdb_text = "".join(l for l in Path(pdb).read_text().splitlines(True) if l.startswith(("CRYST1", "HELIX", "SHEET", "ATOM", "TER", "END")))
    wt = {int(l[22:26]): AA.get(l[17:20], "X") for l in pdb_text.splitlines() if l.startswith("ATOM")}   # residue -> one letter
    resolved = sorted(wt)
    if check_sequence:
        bad = sorted(t for t in pv.index if _position(t) in wt and wt[_position(t)] != str(t)[0])
        if bad:
            raise ValueError(f"the variants' wild-type residue differs from the structure at {bad}")

    # ---- the plot: a trace per substitution, the reference stars, and (last) an empty trace for the pinned position
    hover = ("<b>%{customdata[0]}</b><br>K<sub>m</sub> %{x:.2f} mM (IQR %{customdata[1]:.2f}–%{customdata[2]:.2f})<br>"
             "V<sub>max</sub>/[E] %{y:.2f} (IQR %{customdata[3]:.2f}–%{customdata[4]:.2f})<br>%{customdata[5]} chambers<extra></extra>")
    cd = lambda e: np.c_[e.index.map(display_tag), e["km_lo"] / 1000, e["km_hi"] / 1000, e["vg_lo"], e["vg_hi"], e["n"]]
    fig = go.Figure()
    for sb, (name, color) in subs.items():
        e = pv[pv.index.str[-1] == sb]
        if e.empty:
            continue
        fig.add_trace(go.Scatter(
            x=e["km"] / 1000, y=e["vg"], mode="markers", name=f"{name} ({len(e)})", showlegend=True,
            marker=dict(size=9, color=color, line=dict(color="black", width=0.6)),
            error_x=dict(type="data", symmetric=False, array=(e["km_hi"] - e["km"]) / 1000, arrayminus=(e["km"] - e["km_lo"]) / 1000,
                         color=color, thickness=0.8, width=0),
            error_y=dict(type="data", symmetric=False, array=e["vg_hi"] - e["vg"], arrayminus=e["vg"] - e["vg_lo"], color=color, thickness=0.8, width=0),
            customdata=cd(e), hovertemplate=hover))
    if references:
        ref = pd.DataFrame(references)
        for color, g in ref.groupby(ref.get("color", pd.Series(["#E67E22"] * len(ref))).fillna("#E67E22")):
            fig.add_trace(go.Scatter(x=g["km"] / 1000, y=g["vg"], mode="markers", name=g["group"].iloc[0] if "group" in g else "reference",
                                     marker=dict(symbol="star", size=18, color=color, line=dict(color="black", width=0.8)),
                                     customdata=np.c_[g["label"]],
                                     hovertemplate="<b>%{customdata[0]}</b><br>K<sub>m</sub> %{x:.2f} mM<br>V<sub>max</sub>/[E] %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers+text", name="pinned", showlegend=False, hovertemplate=hover, cliponaxis=False,
                             textposition="top center", textfont=dict(size=13, color="black"),
                             marker=dict(size=15, color=[], line=dict(color="black", width=2)),
                             error_x=dict(type="data", symmetric=False, array=[], arrayminus=[], color="#333333", thickness=1.4, width=0),
                             error_y=dict(type="data", symmetric=False, array=[], arrayminus=[], color="#333333", thickness=1.4, width=0)))
    if km_line is not None:
        fig.add_vline(x=km_line[0] / 1000, line=dict(color="#E67E22", width=1, dash="dash"))
    # K_m on log2 (a tick per doubling, three minor ticks between); V_max / [E] on log10 (minor ticks at 2-9 x each decade)
    km_major = [2.0 ** k for k in range(-6, 7)]
    km_minor = [2.0 ** (k + f) for k in range(-6, 7) for f in (0.25, 0.5, 0.75)]
    lo, hi = pv["km_lo"].min() / 1000 / 1.15, pv["km_hi"].max() / 1000 * 1.15
    if s_range_uM is not None:
        lo, hi = max(lo, s_range_uM[0] / 1000 / 4), min(hi, s_range_uM[1] / 1000 * 4)
    vgs = pd.concat([pv["vg"], pd.Series([r["vg"] for r in references], dtype=float)])
    vg_minor = [m * 10.0 ** e for e in range(-4, 5) for m in range(2, 10)]
    fig.update_xaxes(type="log", title="K<sub>m</sub> (mM), variant median (log<sub>2</sub> scale)", tickvals=km_major,
                     ticktext=[f"{v:g}" for v in km_major], range=np.log10([lo, hi]).tolist(),   # IQR bars beyond run off the edge
                     minor=dict(tickvals=km_minor, ticks="outside", ticklen=3), showline=True, linecolor="black", ticks="outside", ticklen=6)
    fig.update_yaxes(type="log", title=y_label, tickvals=[10.0 ** k for k in range(-4, 5)], ticktext=[f"{10.0 ** k:g}" for k in range(-4, 5)],
                     minor=dict(tickvals=vg_minor, ticks="outside", ticklen=3), range=np.log10([vgs.min() / 2.5, vgs.max() * 2.5]).tolist(),
                     showline=True, linecolor="black", ticks="outside", ticklen=6)
    fig.update_layout(template="simple_white", font=dict(family="Arial", size=13), autosize=True, hovermode="closest",
                      margin=dict(l=70, r=20, t=30, b=60), legend=dict(x=0.99, y=0.02, xanchor="right", yanchor="bottom"))
    # every variant once, by position, for the pinned trace
    points = [dict(pos=_position(t), sub=str(t)[-1], x=r.km / 1000, y=r.vg,
                   cd=[display_tag(t), r.km_lo / 1000, r.km_hi / 1000, r.vg_lo, r.vg_hi, int(r.n)]) for t, r in pv.iterrows()]
    js = lambda o: json.dumps(o).replace("</", "<\\/")   # safe inside <script>
    key = " ".join(f'<span style="color:{g["color"]}">■ {g["name"]} ({", ".join(g["labels"]) if g.get("labels") else f"{min(g["residues"])}–{max(g["residues"])}"})</span>'
                   for g in regions)
    min_n = int(pv["n"].min()) if len(pv) else 0

    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>{(VENDOR / "pdbe-molstar-light.css").read_text()}</style>
<style>body{{margin:0;font-family:Arial,sans-serif;background:#fff;color:#111}} /* the plot left, the structure right, side by side at any window width */
.row{{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:12px;padding:12px 12px 0;align-items:start}}
#plot{{height:min(76vh,72vw)}} #mol{{width:100%;height:min(70vh,40vw);position:relative;margin-top:24px;border:1px solid #ddd}} #molstar{{position:absolute;inset:0}}
#info{{font-size:13px;margin:6px 4px;min-height:2.8em;line-height:1.4}} #info:empty::before{{content:"Click a residue or a variant to pin it; click again or press Esc to release";color:#888}}
#key{{font-size:12px;color:#555;margin:0 4px}} #info button{{font:inherit;font-size:12px;margin-left:8px;cursor:pointer}}</style>
<script>{get_plotlyjs()}</script><script>{(VENDOR / "pdbe-molstar-plugin.js").read_text()}</script></head><body>
<div class="row"><div id="plot"></div><div><div id="mol"><div id="molstar"></div></div><div id="info"></div><div id="key">Always as sticks: {key} · {structure_name}, residues {resolved[0]}–{resolved[-1]}</div></div></div>
<script>
const fig = {pio.to_json(fig)};
const points = {js(points)}, wt = {js(wt)};
const COLORS = {js({sb: color for sb, (name, color) in subs.items()})}, PIN_COLOR = "#222222";
const REGIONS = {js([dict(res=g["residues"], color=g["color"]) for g in regions])};
const pdb = {js(pdb_text)};

const plot = document.getElementById("plot"), info = document.getElementById("info"), mol = document.getElementById("mol");
const PIN = fig.data.length - 1, OTHERS = [...Array(PIN).keys()];
const byPos = {{}};
for (const p of points) (byPos[p.pos] = byPos[p.pos] || []).push(p);
const posOf = tag => parseInt((String(tag).match(/\\d+/) || [NaN])[0]);
const subOf = {{}};
for (const p of points) subOf[p.cd[0]] = p.sub;   // a variant's substitution, by its displayed name
const colorOf = tag => COLORS[subOf[tag]] || PIN_COLOR;
Plotly.newPlot(plot, fig.data, fig.layout, {{displaylogo: false, responsive: true}});

// ---- structure
const viewer = new PDBeMolstarPlugin();
let ready = false, pinned = null;   // pinned: a residue position, or null
// a residue as sticks: its cartoon and its carbons in `color`, the heteroatoms (N, O, S) in the element colors
const sticks = (resi, color, focus) => [
  {{auth_seq_id: resi, color: color, sideChain: true, focus: !!focus}},
  {{auth_seq_id: resi, type_symbol: "C", color: null, representationColor: color}}];
const BASE = REGIONS.flatMap(g => g.res.flatMap(r => sticks(r, g.color)));   // always shown; later regions drawn over earlier
function paintStructure(pos, color, focus) {{
  if (!ready) return Promise.resolve();
  const data = BASE.slice();
  if (pos != null && wt[pos]) data.push(...sticks(pos, color, focus));
  return viewer.visual.select({{data: data, nonSelectedColor: "#d9d9d9"}});
}}

// ---- plot
function paintScatter(pos) {{
  const pts = pos == null ? [] : (byPos[pos] || []);
  Plotly.restyle(plot, {{opacity: pos == null ? 1 : 0.18}}, OTHERS);
  Plotly.restyle(plot, {{x: [pts.map(p => p.x)], y: [pts.map(p => p.y)], text: [pts.map(p => p.cd[0])],
                        customdata: [pts.map(p => p.cd)], "marker.color": [pts.map(p => COLORS[p.sub])],
                        // the IQR bars: cd = [tag, km_lo, km_hi, vg_lo, vg_hi, n], K_m in mM as x
                        "error_x.array": [pts.map(p => p.cd[2] - p.x)], "error_x.arrayminus": [pts.map(p => p.x - p.cd[1])],
                        "error_y.array": [pts.map(p => p.cd[4] - p.y)], "error_y.arrayminus": [pts.map(p => p.y - p.cd[3])]}}, [PIN]);
}}
function describe(pos) {{
  const name = (wt[pos] || "?") + pos, pts = byPos[pos] || [];
  if (!wt[pos]) return name + ": not resolved in the structure";
  if (!pts.length) return name + ": no variants with ≥ {min_n} chambers";
  return name + ": " + pts.map(p => "<b>" + p.cd[0] + "</b> K<sub>m</sub> " + p.x.toFixed(2) + " mM, V<sub>max</sub>/[E] " + p.y.toFixed(2)).join("; ");
}}

// ---- pin / release
function pin(pos, color, focus) {{
  pinned = pos;
  paintScatter(pos);
  paintStructure(pos, color || PIN_COLOR, focus);
  info.innerHTML = "<b>Pinned</b> " + describe(pos) + ' <button id="clear">clear</button>';
  document.getElementById("clear").onclick = release;
}}
function release() {{
  if (pinned == null) return;
  pinned = null;
  paintScatter(null);
  paintStructure(null);
  info.textContent = "";
}}
const togglePin = (pos, color, focus) => (pinned === pos ? release() : pin(pos, color, focus));

plot.on("plotly_hover", e => {{
  if (pinned != null) return;
  const tag = e.points[0].customdata[0];
  if (!subOf[tag]) return;   // a reference star: no residue
  paintStructure(posOf(tag), colorOf(tag));
  info.innerHTML = describe(posOf(tag));
}});
plot.on("plotly_unhover", () => {{ if (pinned == null) {{ paintStructure(null); info.textContent = ""; }} }});
plot.on("plotly_click", e => {{ const tag = e.points[0].customdata[0]; if (subOf[tag]) togglePin(posOf(tag), colorOf(tag), true); }});
mol.addEventListener("PDB.molstar.click", e => {{ const pos = e.eventData && e.eventData.auth_seq_id; if (pos != null) togglePin(pos); }});
document.addEventListener("keydown", e => {{ if (e.key === "Escape") release(); }});

viewer.events.loadComplete.subscribe(ok => {{
  if (!ok) {{ info.textContent = "the structure failed to load"; return; }}
  ready = true;
  paintStructure(null);
  viewer.plugin.behaviors.interaction.click.subscribe(e => {{   // a click on the background releases
    if (e.button === 1 && (!e.current || !e.current.loci || e.current.loci.kind === "empty-loci")) release();
  }});
}});
viewer.render(document.getElementById("molstar"), {{
  customData: {{url: URL.createObjectURL(new Blob([pdb], {{type: "text/plain"}})), format: "pdb", binary: false}},
  bgColor: "#ffffff", visualStyle: {{polymer: "cartoon"}}, hideStructure: ["water", "het"],
  hideControls: true, sequencePanel: false, leftPanel: false, rightPanel: false, logPanel: false,
  pdbeLink: false, loadingOverlay: false,
  selectInteraction: false,   // a click zooms the residue; no ball-and-stick surroundings
  hideCanvasControls: ["controlToggle", "controlInfo", "selection", "animation", "trajectory", "snapshotControls", "snapshotDescription"],
}});
</script></body></html>"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page)
    return path
