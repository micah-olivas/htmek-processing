"""magnify button-finding for the minichip (8x8 adaptation of htmek.buttons_pipe)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from .layout import _column_items


def column_pinlist(columns, n_rows=8, path=None):
    """Build a minichip pinlist CSV where each column holds one variant.

    Returns the CSV path (a temp file if `path` is None). Indices are (col, row),
    1-indexed, matching magnify's identify_buttons parser.
    """
    df = pd.DataFrame(
        {"Indices": f"({col}, {row})", "MutantID": variant}
        for col, variant in _column_items(columns)
        for row in range(1, n_rows + 1)
    )
    path = (
        Path(path)
        if path
        else Path(tempfile.gettempdir()) / "minichip_column_pinlist.csv"
    )

    df.to_csv(path, index=False)
    return str(path)


def buttons_pipe_minichip(
    data,
    columns=None,
    pinlist=None,
    *,
    n_rows=8,
    shape=(8, 8),
    overlap=0,
    rotation=3,
    blank="BLANK",
    row_dist=125,
    col_dist=125,
    post_hflip=True,
    basic_correct=False,
    interactive=False,
    **kwargs,
):
    """htmek.buttons_pipe's detection recipe, adapted to the 8x8 minichip.

    htmek.buttons_pipe is hardcoded to the PS 56x32 chip, so we rebuild the pipe with
    minichip geometry. Provide identities the conserved way via `columns` (one variant
    per column) or pass an explicit `pinlist` CSV path.

    The PS default pitch (~233 px) is too big to fit 8 buttons in a 1024 px frame, so
    row_dist/col_dist default to 125; tune to your scope. Pass left_chamber/top_chamber
    (kwargs) to anchor the grid manually.
    """
    import magnify as mg

    if columns is not None:
        if pinlist is not None:
            raise ValueError("Pass either `columns` or `pinlist`, not both.")
        pinlist = column_pinlist(columns, n_rows=n_rows)

    pipe = mg.microfluidic_chip_pipe(
        shape=shape,
        pinlist=pinlist,
        overlap=overlap,
        rotation=rotation,
        blank=blank,
        row_dist=row_dist,
        col_dist=col_dist,
        high_edge_quantile=0.999,
        min_button_diameter=20,
        max_button_diameter=32,
        chamber_diameter=75,
        min_roundness=0.10,
        roi_length=None,
        search_channel=["egfp"],
        progress_bar=True,
        interactive=interactive,
        **kwargs,
    )

    if basic_correct:
        pipe.add_pipe("basic_correct", before="stitch")
    if post_hflip:
        pipe.add_pipe("horizontal_flip", after="stitch")
    return pipe(data)
