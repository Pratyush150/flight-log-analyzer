"""flightlog -- automated health reports from PX4 and ArduPilot flight logs.

Quickstart
----------
>>> from flightlog import analyze_log, generate_defective
>>> report = analyze_log(generate_defective())
>>> report.counts["critical"] > 0
True

Reading a real log::

    from flightlog import load, analyze_log, write_html
    log = load("flight.ulg")
    report = analyze_log(log)
    write_html(report, "report.html", log)

The package is deliberately layered:

``flightlog.readers``
    Anything that touches a file format. The only place optional dependencies
    (pyulog, pymavlink) appear, and both are guarded.
``flightlog.analysis``
    Pure numpy analyzers over the normalised :class:`~flightlog.types.FlightLog`.
``flightlog.report``
    Ranking and rendering. Terminal, self-contained HTML, JSON.
``flightlog.svgplot``
    A small hand-written SVG plotter so the report path needs no matplotlib.
"""

from __future__ import annotations

from .analysis import ANALYZERS
from .channels import UNITS, units_for
from .readers import (
    DefectSpec,
    SyntheticConfig,
    generate,
    generate_clean,
    generate_defective,
    load,
    read_csv,
    read_csv_text,
    supported_formats,
)
from .report import (
    Report,
    build_report,
    rank_findings,
    render_html,
    render_json,
    render_terminal,
    write_html,
    write_json,
)
from .types import Event, FlightLog, Finding, ModeInterval, Series, Severity

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "FlightLog",
    "Series",
    "Event",
    "ModeInterval",
    "Finding",
    "Severity",
    "load",
    "read_csv",
    "read_csv_text",
    "supported_formats",
    "generate",
    "generate_clean",
    "generate_defective",
    "DefectSpec",
    "SyntheticConfig",
    "analyze_log",
    "build_report",
    "rank_findings",
    "Report",
    "render_terminal",
    "render_html",
    "render_json",
    "write_html",
    "write_json",
    "ANALYZERS",
    "UNITS",
    "units_for",
]


def analyze_log(log: FlightLog, source: str = "") -> Report:
    """Run every analyzer over ``log`` and return a ranked :class:`Report`.

    Thin wrapper over :func:`flightlog.report.build_report`; it exists so the
    common case is one import and one call.
    """
    return build_report(log, source=source)
