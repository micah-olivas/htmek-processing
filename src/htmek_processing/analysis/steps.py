"""The pipeline's record of its own filtering decisions, drawn as the chamber report's
flowchart (report.flowchart).

Each filtering function (leave_out_points, fit_progress_curves, flag_poor_4pl,
flag_early_depletion, force_4pl_on_depleted, subtract_blank_rate) records one step when it
runs: its rule, the parameters it ran with, and how many points or runs it kept, flagged or
changed. Steps live in one session record, keyed by function, so rerunning a cell replaces
its step in place rather than adding a second one.

The process and analysis notebooks run in separate sessions: the process notebook saves the
record next to its rates table (save_steps), and the analysis notebook loads it (load_steps)
before the report draws it.

A step is a dict:
    step      the function that recorded it (the record's key)
    title     a short name for the flowchart box
    rule      lines describing the rule, with the parameters it ran with; `name` marks a
              column or other code name (drawn in code font); a line starting "- " is a
              bullet
    unit      what the counts count ("points", "runs")
    n         how many went in
    branches  [{"text", "n", "level"}]: what the step did to how many; level "fail" (left
              out), "warn" (changed or flagged, kept), "pass" (kept as is)
    stage     "images" (image processing) or "rates" (the numbers it reads): one flowchart
              page each
    when      ISO date it ran

Steps that run once per image (stitching, grid placement) are tallies (tally_step): each
call adds its image as an item, and the step counts its items, so rerunning a cell for the
same image doesn't count it twice.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

_STEPS: dict = {}


def record_step(step, title, rule, *, n, unit="runs", branches=(), stage="rates"):
    """Add (or replace) the step `step` in the session record. See the module docstring
    for the fields."""
    _STEPS[step] = dict(step=step, title=title, rule=list(rule), unit=unit, n=int(n),
                        branches=[dict(b, n=int(b["n"])) for b in branches], stage=stage,
                        when=date.today().isoformat())


def tally_step(step, title, rule, item, *, label=None, level="info", unit="images", stage="images"):
    """Count `item` (e.g. an acquisition's name) under the step `step`: its branches are
    the items' `label`s (e.g. "seams measured on itself"), each with how many items carry
    it. An item tallied again replaces its earlier label. The rule is the latest call's."""
    items = dict(_STEPS.get(step, {}).get("items", {}))
    items[str(item)] = [label, level]
    counts = {}
    for lab, lev in items.values():
        if lab is not None:
            counts.setdefault((lab, lev), 0)
            counts[(lab, lev)] += 1
    record_step(step, title, rule, n=len(items), unit=unit, stage=stage,
                branches=[dict(text=lab, n=k, level=lev) for (lab, lev), k in counts.items()])
    _STEPS[step]["items"] = items


def pipeline_steps():
    """The session record: a list of steps, in the order they first ran."""
    return list(_STEPS.values())


def clear_steps():
    """Empty the session record."""
    _STEPS.clear()


def save_steps(path):
    """Write the session record to `path` (JSON), e.g. next to the rates export."""
    Path(path).write_text(json.dumps(pipeline_steps(), indent=1, ensure_ascii=False) + "\n")


def load_steps(path, *, replace=True):
    """Read a record written by save_steps into the session record (replacing it unless
    `replace` is False) and return it as a list of steps."""
    if replace:
        clear_steps()
    for s in json.loads(Path(path).read_text()):
        _STEPS[s["step"]] = s
    return pipeline_steps()
