"""Reports: the chamber report (one page per chamber), the per-variant volcano statistics,
and the experiment report (the chip-level picture).

    chamber      chamber_report_pdf and its page helpers (was report.py)
    volcano      per-variant fold changes with bootstrap p-values, and the volcano plot
    experiment   experiment_report_pdf: summary, chip maps, rate maps, trends, distributions
    flowchart    the chamber report's first page: the filters the pipeline ran, as a flowchart
"""

from .chamber import (  # noqa: F401
    REPORT_VERSION,
    REPORT_COLORS,
    EXCLUDED_COL,
    read_button_qc,
    button_qc_tabs,
    _ordinal,
    _fmt,
    _grid_table,
    _left_out,
    _mm_panel,
    _PAGE,
    _REPORT_STYLE,
    _AXIS_LABELS,
    _report_page,
    _seams_in_window,
    _seam_notches,
    _chip_map,
    _km_row,
    _report_chunk,
    KEY_ACCENT,
    KEY_SAMPLE_DPI,
    _report_key_page,
    chamber_report_pdf,
)
from . import volcano  # noqa: F401,E402
from .experiment import experiment_report_pdf  # noqa: F401,E402
from .flowchart import flowchart_page, flowchart_pages, report_steps  # noqa: F401,E402
