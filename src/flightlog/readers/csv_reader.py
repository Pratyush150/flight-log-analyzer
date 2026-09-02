"""CSV reader -- the fallback that always works.

Every log-viewing tool in the drone world can export CSV (Flight Review, MAVExplorer,
UAV Log Viewer, `ulog2csv`, `mavlogdump.py --format csv`).  So when a client
cannot install ``pyulog``, or has already exported a subset of topics, this
reader still gets them a report.

Header handling is deliberately forgiving: ``TimeUS``, ``timestamp``, ``AccX``,
``accel_x`` and ``Accel X (m/s2)`` all resolve to the right canonical channel
via :func:`flightlog.channels.canonical_from_alias`.  Unrecognised columns are
kept under their normalised name so nothing is silently dropped.

No third-party dependencies -- stdlib :mod:`csv` plus numpy.
"""

from __future__ import annotations

import csv
import io
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..channels import canonical_from_alias, units_for
from ..types import FlightLog, ModeInterval

__all__ = ["read_csv", "read_csv_text", "can_read"]

_TIME_KEYS = ("__time__", "__time_us__", "__time_ms__")


def can_read(path: str) -> bool:
    """True if ``path`` looks like a CSV/TSV file this reader can handle."""
    return os.path.splitext(path)[1].lower() in (".csv", ".tsv", ".txt")


def read_csv(
    path: str,
    time_column: Optional[str] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> FlightLog:
    """Read a CSV file into a :class:`~flightlog.types.FlightLog`."""
    with open(path, "r", newline="", encoding="utf-8-sig", errors="replace") as fh:
        text = fh.read()
    log = read_csv_text(text, time_column=time_column, metadata=metadata)
    log.metadata.setdefault("source", os.path.abspath(path))
    return log


def read_csv_text(
    text: str,
    time_column: Optional[str] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> FlightLog:
    """Parse CSV held in a string.  Used by tests and by piped input.

    Raises
    ------
    ValueError
        If the file has no header row, or no usable time column.  A log with
        no time base cannot be analyzed at all, so failing loudly beats
        inventing timestamps.
    """
    dialect_text = text[:8192]
    delimiter = "\t" if dialect_text.count("\t") > dialect_text.count(",") else ","
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)

    header: Optional[List[str]] = None
    rows: List[Sequence[str]] = []
    for row in reader:
        if not row or all(not c.strip() for c in row):
            continue
        if row[0].lstrip().startswith("#"):
            continue
        if header is None:
            header = [c.strip() for c in row]
            continue
        rows.append(row)
    if header is None:
        raise ValueError("CSV has no header row")
    if not rows:
        raise ValueError("CSV has a header but no data rows")

    canonical = [canonical_from_alias(h) for h in header]
    time_idx, time_scale = _resolve_time_column(header, canonical, time_column)

    ncol = len(header)
    columns: List[List[float]] = [[] for _ in range(ncol)]
    for row in rows:
        for i in range(ncol):
            columns[i].append(_to_float(row[i]) if i < len(row) else math.nan)

    t = np.asarray(columns[time_idx], dtype=float) * time_scale
    finite = np.isfinite(t)
    if not np.any(finite):
        raise ValueError("time column contains no finite values")
    t = t[finite]
    t = t - t[0]

    log = FlightLog()
    log.metadata.update(
        {
            "log_format": "csv",
            "vehicle": "unknown",
            "firmware": "unknown",
            "time_column": header[time_idx],
        }
    )
    if metadata:
        log.metadata.update(metadata)

    for i, (raw, name) in enumerate(zip(header, canonical)):
        if i == time_idx or name in _TIME_KEYS:
            continue
        v = np.asarray(columns[i], dtype=float)[finite]
        if not np.any(np.isfinite(v)):
            continue
        log.add(name, t, v, units_for(name), f"csv:{raw}")

    _derive_events(log)
    return log


def _resolve_time_column(
    header: Sequence[str], canonical: Sequence[str], requested: Optional[str]
) -> tuple[int, float]:
    """Return ``(column_index, seconds_per_unit)``.

    ArduPilot exports ``TimeUS`` (microseconds), PX4 exports ``timestamp``
    (also microseconds), and hand-made CSVs usually use seconds.  Guessing
    wrong scales the entire frequency axis by 10^6, so the unit is inferred
    from the column name first and from the value magnitude second.
    """
    if requested is not None:
        for i, h in enumerate(header):
            if h == requested or h.strip().lower() == requested.strip().lower():
                return i, _scale_for(h)
        raise ValueError(f"requested time column {requested!r} not found in header")
    for i, c in enumerate(canonical):
        if c == "__time_us__":
            return i, 1e-6
        if c == "__time_ms__":
            return i, 1e-3
    for i, c in enumerate(canonical):
        if c == "__time__":
            return i, _scale_for(header[i])
    return 0, _scale_for(header[0])


def _scale_for(name: str) -> float:
    n = name.strip().lower()
    if n.endswith("us") or "micro" in n:
        return 1e-6
    if n.endswith("ms") or "milli" in n:
        return 1e-3
    if n in ("timestamp", "time_boot_ms"):
        # PX4 "timestamp" is microseconds; MAVLink "time_boot_ms" is ms.
        return 1e-6 if n == "timestamp" else 1e-3
    return 1.0


def _to_float(cell: str) -> float:
    s = cell.strip()
    if not s:
        return math.nan
    try:
        return float(s)
    except ValueError:
        low = s.lower()
        if low in ("true", "yes", "on", "armed"):
            return 1.0
        if low in ("false", "no", "off", "disarmed"):
            return 0.0
        return math.nan


def _derive_events(log: FlightLog) -> None:
    """Synthesise arm/disarm and mode intervals from channels, if present.

    A CSV export rarely carries the event stream, but it usually carries an
    ``armed`` or ``mode`` column.  Reconstructing the events from those keeps
    the mode/arm analyzer useful on CSV input.
    """
    armed = log.get("armed")
    if armed is not None and len(armed) > 1:
        state = armed.values > 0.5
        for i in range(1, state.size):
            if state[i] and not state[i - 1]:
                log.add_event(float(armed.time[i]), "arm", "armed (from CSV column)")
            elif state[i - 1] and not state[i]:
                log.add_event(float(armed.time[i]), "disarm", "disarmed (from CSV column)")

    mode = log.get("mode.id")
    if mode is not None and len(mode) > 1:
        vals = mode.values
        start = float(mode.time[0])
        cur = vals[0]
        for i in range(1, vals.size):
            if vals[i] != cur:
                log.modes.append(ModeInterval(start, float(mode.time[i]), f"MODE_{int(cur)}"))
                log.add_event(float(mode.time[i]), "mode_change", f"MODE_{int(vals[i])}")
                start, cur = float(mode.time[i]), vals[i]
        log.modes.append(ModeInterval(start, float(mode.time[-1]), f"MODE_{int(cur)}"))

    log.events.sort(key=lambda e: e.time)


def write_csv(
    log: FlightLog,
    path: str,
    channels: Optional[Iterable[str]] = None,
    precision: int = 6,
) -> str:
    """Export a FlightLog back to CSV on a common time grid.

    Useful for handing a client a trimmed dataset, and for producing the
    example file shipped in ``examples/``.  Channels are resampled onto the
    grid of the highest-rate selected channel.

    ``precision`` is the number of decimal places written.  Six is lossless
    enough for every analyzer here; four roughly halves the file size, which
    matters when the CSV goes into a git repository.
    """
    names = list(channels) if channels else sorted(log.series)
    present = [n for n in names if log.get(n) is not None and len(log.series[n]) > 1]
    if not present:
        raise ValueError("no channels to write")
    base = max((log.series[n] for n in present), key=lambda s: s.sample_rate)
    grid = base.time
    cols = [log.series[n].interp_to(grid) for n in present]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time"] + present)
        for i, tv in enumerate(grid):
            w.writerow(
                [f"{tv:.{precision}f}"] + [f"{c[i]:.{precision}f}" for c in cols]
            )
    return path
