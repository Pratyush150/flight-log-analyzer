"""Log readers.  One job: turn any supported log into a :class:`FlightLog`.

The only entry point most callers need is :func:`load`, which picks a reader
from the file extension and gives a specific, actionable error if the optional
dependency for that format is missing.
"""

from __future__ import annotations

import os
from typing import Dict, List

from ..types import FlightLog
from . import csv_reader, dataflash_reader, ulog_reader
from .dataflash_reader import read_dataflash
from .csv_reader import read_csv, read_csv_text
from .synthetic import DefectSpec, SyntheticConfig, generate, generate_clean, generate_defective
from .ulog_reader import MissingDependencyError, read_ulog

__all__ = [
    "load",
    "supported_formats",
    "read_ulog",
    "read_dataflash",
    "read_csv",
    "read_csv_text",
    "generate",
    "generate_clean",
    "generate_defective",
    "DefectSpec",
    "SyntheticConfig",
    "MissingDependencyError",
]


def supported_formats() -> List[Dict[str, object]]:
    """Table of formats, extensions, backing library and availability.

    Used by ``flightlog-analyze --formats`` and by the README table, so the
    documented support matrix cannot drift from reality.
    """
    return [
        {
            "format": "PX4 ULog",
            "extensions": [".ulg", ".ulog"],
            "library": "pyulog",
            "available": ulog_reader.AVAILABLE,
            "install": ulog_reader.INSTALL_HINT,
        },
        {
            "format": "ArduPilot dataflash",
            "extensions": [".bin", ".log", ".px4log"],
            "library": "pymavlink",
            "available": dataflash_reader.AVAILABLE,
            "install": dataflash_reader.INSTALL_HINT,
        },
        {
            "format": "CSV / TSV export",
            "extensions": [".csv", ".tsv", ".txt"],
            "library": "stdlib",
            "available": True,
            "install": "",
        },
        {
            "format": "Synthetic (built in)",
            "extensions": ["--demo"],
            "library": "stdlib",
            "available": True,
            "install": "",
        },
    ]


def load(path: str) -> FlightLog:
    """Load a flight log, choosing the reader from the file extension.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the extension is not recognised.
    MissingDependencyError
        If the format needs an optional package that is not installed.  The
        message includes the exact pip command.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if ulog_reader.can_read(path):
        return read_ulog(path)
    if csv_reader.can_read(path) and os.path.splitext(path)[1].lower() in (".csv", ".tsv"):
        return read_csv(path)
    if dataflash_reader.can_read(path):
        return read_dataflash(path)
    if csv_reader.can_read(path):
        return read_csv(path)
    known = ", ".join(
        e for f in supported_formats() for e in f["extensions"] if str(e).startswith(".")
    )
    raise ValueError(f"unrecognised log extension for {path!r}; supported: {known}")
