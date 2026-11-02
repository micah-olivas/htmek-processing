"""Chamber identities: which variant is in each chamber (pinlists, column layouts)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def _column_items(columns):
    """Normalize `columns` (list or {1-based col: variant}) to
    [(col_1based, variant), ...]."""
    if isinstance(columns, dict):
        return sorted(columns.items())
    return list(enumerate(columns, start=1))


def column_tags(columns, n_rows=8):
    """(n_rows, n_cols) array of variant tags; every row in a column shares its identity."""
    names = [v for _, v in _column_items(columns)]
    return np.tile(np.array(names, dtype="<U200"), (n_rows, 1)), len(names)


def pinlist_tags(pinlist, n_rows=None, n_cols=None, *, blank=""):
    """(n_rows, n_cols) array of variant tags from a magnify-style pinlist.

    `pinlist` is a CSV path or DataFrame with `Indices` ("(col, row)", 1-based) and
    `MutantID` -- the format Array-Print's fld_to_pinlist writes. mark_row = row - 1 and
    mark_col = col - 1, so pinlist (1, 1) is the grid's top-left chamber. The shape defaults
    to the largest indices; chambers the pinlist doesn't list get `blank`."""
    pl = pd.read_csv(pinlist) if isinstance(pinlist, (str, Path)) else pinlist
    # Indices may be "(c, r)" strings (CSV) or (c, r) tuples (in-memory DataFrame)
    idx = np.array([tuple(map(int, re.findall(r"\d+", str(i)))) for i in pl["Indices"]])
    cols, rows = idx[:, 0] - 1, idx[:, 1] - 1

    n_rows = n_rows or int(rows.max()) + 1
    n_cols = n_cols or int(cols.max()) + 1
    if rows.max() >= n_rows or cols.max() >= n_cols:
        raise ValueError(
            f"pinlist reaches (col {cols.max() + 1}, row {rows.max() + 1}), outside "
            f"the {n_rows} x {n_cols} grid; pass n_rows/n_cols to match the chip."
        )

    tags = np.full((n_rows, n_cols), blank, dtype="<U200")
    tags[rows, cols] = pl["MutantID"].fillna(blank).astype(str).to_numpy()
    return tags


def mirror_pinlist(pinlist, *, n_cols=None, n_rows=None):
    """Reflect a pinlist to match how the chip appears in the image. Returns a new
    DataFrame (the file is untouched); pass it anywhere a pinlist is accepted.

    `n_cols=N` mirrors left-right (col -> N + 1 - col): a reflection about the chip's long
    axis when rows are the long side. `n_rows=N` mirrors top-bottom (row -> N + 1 - row).
    Give both to rotate 180 degrees. Use it when the printed layout and the imaged chip are
    mirror images -- e.g. the print's frame vs the camera's."""
    pl = (pd.read_csv(pinlist) if isinstance(pinlist, (str, Path)) else pinlist).copy()
    idx = np.array([tuple(map(int, re.findall(r"\d+", str(i)))) for i in pl["Indices"]])
    cols, rows = idx[:, 0], idx[:, 1]
    if n_cols is not None:
        cols = n_cols + 1 - cols
    if n_rows is not None:
        rows = n_rows + 1 - rows
    pl["Indices"] = [f"({c}, {r})" for c, r in zip(cols, rows)]
    return pl


def chamber_tags(layout, n_rows=None, n_cols=None):
    """Resolve any identity `layout` to an (n_rows, n_cols) tag array:

      - None                       -> all "" (no identities)
      - list / {col: variant}      -> one variant per column (minichip), via column_tags
      - pinlist DataFrame or .csv  -> per-chamber identities, via pinlist_tags
      - 2-D array                  -> used as-is

    `n_rows`/`n_cols` set the shape where the layout doesn't fix it."""
    if layout is None:
        return np.full((n_rows or 8, n_cols or 8), "", dtype="<U200")
    if isinstance(layout, pd.DataFrame) or (
        isinstance(layout, (str, Path)) and str(layout).endswith(".csv")
    ):
        return pinlist_tags(layout, n_rows, n_cols)
    if isinstance(layout, np.ndarray) and layout.ndim == 2:
        return layout.astype("<U200")
    return column_tags(layout, n_rows or 8)[0]


def display_tag(tag):
    """A variant's name for figures: an ester (TAG, amber-suppression) substitution, "*" in the
    pinlist (e.g. "F22*"), is written with phi ("F22φ"). Other tags are unchanged; tables keep
    the pinlist's names."""
    tag = str(tag)
    return tag[:-1] + "φ" if tag.endswith("*") else tag

