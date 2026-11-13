"""Read the chip-control protocol logs (e.g. patterning_log.json): one JSON list of runs, each with its protocol,
the valve / inlet assignments, the planned step times, an `info` dict, and the steps actually run (`step`, `start`,
`seconds`); `completed` is False for a run stopped early."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# keys of a run that are not settings (everything else, e.g. inlets and `info`, is kept as the run's settings)
_RUN_KEYS = {"protocol", "chip", "steps", "completed", "start", "end", "times", "sides", "info"}


def read_protocol_log(path):
    """(runs, steps) tables from a protocol log.

    runs   one row per run, in log order: run (0-based), protocol, chip, start, end, minutes, completed, n_steps,
           info (a "key: value; ..." summary of `info`), settings (inlets, sides, flags; the same form)
    steps  one row per step actually run: run, protocol, step, stage (the step's "Step N" prefix, or the protocol
           for protocols without one), start, end, seconds
    Timestamps are naive datetimes, as logged."""
    runs_raw = json.loads(Path(path).read_text())
    runs, steps = [], []
    for i, r in enumerate(runs_raw):
        start, end = pd.to_datetime(r.get("start")), pd.to_datetime(r.get("end"))
        settings = {k: v for k, v in r.items() if k not in _RUN_KEYS}
        runs.append(dict(run=i, protocol=r.get("protocol", ""), chip=r.get("chip", ""), start=start, end=end,
                         minutes=(end - start).total_seconds() / 60 if pd.notna(start) and pd.notna(end) else float("nan"),
                         completed=bool(r.get("completed", False)), n_steps=len(r.get("steps", [])),
                         info="; ".join(f"{k}: {v}" for k, v in (r.get("info") or {}).items()),
                         settings="; ".join(f"{k}: {v}" for k, v in settings.items())))
        for s in r.get("steps", []):
            t0 = pd.to_datetime(s["start"])
            name = s.get("step", "")
            stage = name.split(":")[0] if name.lower().startswith("step ") else r.get("protocol", "")
            steps.append(dict(run=i, protocol=r.get("protocol", ""), step=name, stage=stage, start=t0,
                              end=t0 + pd.Timedelta(seconds=s.get("seconds", 0)), seconds=s.get("seconds", 0)))
    return pd.DataFrame(runs), pd.DataFrame(steps)


def protocol_summary(runs):
    """Short lines for a report: one per completed run (protocol, start, duration, info); runs stopped early are
    counted, not listed."""
    lines = []
    for r in runs[runs["completed"]].itertuples():
        info = f" · {r.info}" if r.info else ""
        lines.append(f"{r.protocol} · {r.start:%Y-%m-%d %H:%M} · {r.minutes:.0f} min{info}")
    stopped = int((~runs["completed"]).sum())
    if stopped:
        lines.append(f"{stopped} run{'s' if stopped > 1 else ''} stopped early (restarted)")
    return lines
