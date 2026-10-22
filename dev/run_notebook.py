"""Run a notebook's code cells top to bottom in one IPython shell, headless, logging each cell.

    python dev/run_notebook.py NOTEBOOK LOG [--skip 10 32] [--sub 'OLD=>NEW' ...]

Run from the experiment folder (the notebooks import from there). `--skip` leaves out cells by
index (e.g. interactive GridAdjuster cells); `--sub` replaces text in the cells before they run
(each must match somewhere, or the run stops); the notebook file is only read. `%matplotlib`
lines are dropped (figures render with Agg and are closed after each cell). Stops at the first
cell that raises.
"""

import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("notebook", type=Path)
    p.add_argument("log", type=Path)
    p.add_argument("--skip", type=int, nargs="*", default=[])
    p.add_argument("--sub", nargs="*", default=[])
    args = p.parse_args()

    os.environ["MPLBACKEND"] = "Agg"
    sys.path.insert(0, os.getcwd())   # as Jupyter does: the notebook's folder
    import matplotlib.pyplot as plt
    from IPython.core.interactiveshell import InteractiveShell

    cells = json.loads(args.notebook.read_text())["cells"]
    subs = [s.split("=>", 1) for s in args.sub]
    used = {old: 0 for old, _ in subs}
    shell = InteractiveShell.instance()
    shell.colors = "nocolor"   # plain tracebacks in the log
    args.log.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(args.log, "w") as log:
        for i, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            text = "".join(cell["source"])
            if i in args.skip:
                log.write(f"\n===== [{i}] SKIPPED =====\n")
                continue
            for old, new in subs:
                if old in text:
                    text = text.replace(old, new)
                    used[old] += 1
            text = "\n".join(line for line in text.splitlines()
                             if not line.lstrip().startswith("%matplotlib"))
            t0 = time.time()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                result = shell.run_cell(text, store_history=False)
            plt.close("all")
            took = time.time() - t0
            log.write(f"\n===== [{i}] {took:.1f} s =====\n{buf.getvalue()}")
            log.flush()
            print(f"[{i}] {took:6.1f} s{'  ERROR' if not result.success else ''}", flush=True)
            if not result.success:
                print(buf.getvalue()[-3000:], flush=True)
                sys.exit(f"stopped at cell {i}")
    missing = [old for old, n in used.items() if n == 0]
    if missing:
        sys.exit(f"--sub never matched: {missing}")
    print(f"done in {(time.time() - start) / 60:.1f} min; log: {args.log}")


if __name__ == "__main__":
    main()
