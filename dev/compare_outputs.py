"""Compare the tables (and figure list) of two experiment output folders, file by file.

    python dev/compare_outputs.py A B [--rtol 0]

A and B each hold 3_intermediate/ and 4_exports/. For every CSV in A: the same columns and rows
in B, numbers equal within `rtol` (relative, NaN = NaN; 0 = bit for bit), everything else exactly
equal. CSVs are read exactly (float_precision="round_trip"). Also lists the figures
(5_figures/**/*.pdf) present in only one of the two.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compare(a, b, rtol):
    """Problems found comparing tables a and b, as text; empty if they agree."""
    problems = []
    if list(a.columns) != list(b.columns):
        only_a = [c for c in a.columns if c not in b.columns]
        only_b = [c for c in b.columns if c not in a.columns]
        problems.append(f"columns differ (only in A: {only_a}, only in B: {only_b})")
    if len(a) != len(b):
        return problems + [f"rows: {len(a)} vs {len(b)}"]
    for col in [c for c in a.columns if c in b.columns]:
        x, y = a[col], b[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            x, y = x.to_numpy(float), y.to_numpy(float)
            same = np.isclose(x, y, rtol=rtol, atol=0, equal_nan=True) | (x == y)
            if not same.all():
                with np.errstate(invalid="ignore", divide="ignore"):
                    d = np.abs(x - y)[~same]
                    rel = (np.abs(x - y) / np.maximum(np.abs(x), np.abs(y)))[~same]
                problems.append(f"{col}: {(~same).sum()} of {len(x)} values differ "
                                f"(max |diff| {np.nanmax(d):.3g}, max rel {np.nanmax(rel):.3g})")
        else:
            xs, ys = x.astype("string").fillna("<NA>"), y.astype("string").fillna("<NA>")
            n = int((xs != ys).sum())
            if n:
                k = (xs != ys).to_numpy().argmax()
                problems.append(f"{col}: {n} of {len(x)} values differ (e.g. {xs.iloc[k]!r} vs {ys.iloc[k]!r})")
    return problems


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--rtol", type=float, default=0.0)
    args = p.parse_args()
    read = lambda f: pd.read_csv(f, float_precision="round_trip")

    n_same = n_diff = 0
    for folder in ("3_intermediate", "4_exports"):
        for fa in sorted((args.a / folder).glob("*.csv")):
            fb = args.b / folder / fa.name
            name = f"{folder}/{fa.name}"
            if not fb.exists():
                print(f"MISSING in B  {name}")
                n_diff += 1
                continue
            problems = compare(read(fa), read(fb), args.rtol)
            n_same, n_diff = (n_same, n_diff + 1) if problems else (n_same + 1, n_diff)
            print(f"{'DIFFERS' if problems else 'same':14s}{name}")
            for line in problems:
                print(f"                {line}")
        for fb in sorted((args.b / folder).glob("*.csv")):
            if not (args.a / folder / fb.name).exists():
                print(f"only in B     {folder}/{fb.name}")

    figs = {d: {str(f.relative_to(d / "5_figures")) for f in (d / "5_figures").rglob("*.pdf")}
            for d in (args.a, args.b)}
    print(f"\nfigures: {len(figs[args.a])} in A, {len(figs[args.b])} in B, "
          f"{len(figs[args.a] & figs[args.b])} in both")
    for f in sorted(figs[args.a] - figs[args.b]):
        print(f"  only in A: {f}")
    for f in sorted(figs[args.b] - figs[args.a]):
        print(f"  only in B: {f}")
    print(f"\ntables: {n_same} the same, {n_diff} different or missing (rtol {args.rtol:g})")


if __name__ == "__main__":
    main()
