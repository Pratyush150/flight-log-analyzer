"""Core data model shared by every reader and every analyzer.

The whole point of this package is that a PX4 ULog, an ArduPilot dataflash log
and a hand-rolled CSV all get normalised into the *same* object before any
analysis runs.  Analyzers never import ``pyulog`` or ``pymavlink``; they only
ever see :class:`FlightLog`.  That keeps the optional dependencies confined to
:mod:`flightlog.readers` and makes every analyzer unit-testable against
synthetic data.

Canonical channel names are defined in :mod:`flightlog.channels`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Severity",
    "Series",
    "Event",
    "ModeInterval",
    "FlightLog",
    "Finding",
]


class Severity(str, Enum):
    """Ordered severity levels.

    The string values are what land in JSON/HTML output.  ``rank`` gives the
    sort key used by :func:`flightlog.report.rank_findings` (higher = worse).
    """

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"info": 0, "warning": 1, "critical": 2}[self.value]

    def __lt__(self, other: object) -> bool:  # pragma: no cover - trivial
        if not isinstance(other, Severity):
            return NotImplemented
        return self.rank < other.rank


@dataclass
class Series:
    """A single named time-series with units.

    Attributes
    ----------
    name:
        Canonical channel name, e.g. ``"accel.x"``.
    time:
        Monotonic timestamps in **seconds since log start**.  Readers are
        responsible for converting native units (ULog microseconds, dataflash
        TimeUS) and for rebasing to zero.
    values:
        Sample values, same length as ``time``.
    units:
        Free-form unit string, e.g. ``"m/s^2"``.  Used for report labelling
        only; no automatic conversion is ever performed.
    source:
        Native topic/message and field the data came from, e.g.
        ``"sensor_accel.x"`` or ``"IMU.AccX"``.  Kept so a report can tell the
        reader exactly where to look in their own log.
    """

    name: str
    time: np.ndarray
    values: np.ndarray
    units: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float).reshape(-1)
        self.values = np.asarray(self.values, dtype=float).reshape(-1)
        if self.time.size != self.values.size:
            raise ValueError(
                f"series {self.name!r}: time has {self.time.size} samples but "
                f"values has {self.values.size}"
            )

    def __len__(self) -> int:
        return int(self.time.size)

    @property
    def duration(self) -> float:
        """Span of the series in seconds (0.0 if fewer than two samples)."""
        if self.time.size < 2:
            return 0.0
        return float(self.time[-1] - self.time[0])

    @property
    def sample_rate(self) -> float:
        """Median sample rate in Hz.

        Median rather than mean because logs routinely contain dropouts: a
        single 2-second gap would drag a mean rate far below the true rate and
        silently corrupt every FFT frequency axis downstream.
        """
        if self.time.size < 2:
            return 0.0
        dt = np.diff(self.time)
        dt = dt[dt > 0]
        if dt.size == 0:
            return 0.0
        return float(1.0 / np.median(dt))

    def slice_time(self, t0: Optional[float] = None, t1: Optional[float] = None) -> "Series":
        """Return a new :class:`Series` restricted to ``[t0, t1]``."""
        mask = np.ones(self.time.shape, dtype=bool)
        if t0 is not None:
            mask &= self.time >= t0
        if t1 is not None:
            mask &= self.time <= t1
        return Series(self.name, self.time[mask], self.values[mask], self.units, self.source)

    def gaps(self, factor: float = 5.0) -> List[Tuple[float, float]]:
        """Return ``(start, end)`` intervals where sampling stalled.

        A gap is any inter-sample interval longer than ``factor`` times the
        median interval.  Logging dropouts matter: they invalidate spectral
        analysis and are themselves a finding (SD card too slow, log rate too
        high, or the FC running out of CPU).
        """
        if self.time.size < 3:
            return []
        dt = np.diff(self.time)
        med = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 0.0
        if med <= 0:
            return []
        idx = np.flatnonzero(dt > factor * med)
        return [(float(self.time[i]), float(self.time[i + 1])) for i in idx]

    def interp_to(self, t: np.ndarray) -> np.ndarray:
        """Linearly resample onto timestamps ``t`` (no extrapolation flair).

        Used whenever two channels logged at different rates must be compared
        sample-for-sample, e.g. battery voltage (10 Hz) against throttle
        (50 Hz).
        """
        t = np.asarray(t, dtype=float)
        if self.time.size == 0:
            return np.full(t.shape, np.nan)
        if self.time.size == 1:
            return np.full(t.shape, float(self.values[0]))
        return np.interp(t, self.time, self.values)

    def stats(self) -> Dict[str, float]:
        """Basic descriptive statistics, NaN-safe."""
        v = self.values[np.isfinite(self.values)]
        if v.size == 0:
            return {"n": 0, "min": math.nan, "max": math.nan, "mean": math.nan, "std": math.nan}
        return {
            "n": float(v.size),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
        }


@dataclass
class Event:
    """A point-in-time occurrence: arm, disarm, failsafe, EKF reset, mode change."""

    time: float
    kind: str
    detail: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModeInterval:
    """A contiguous stretch of one flight mode."""

    start: float
    end: float
    mode: str

    @property
    def duration(self) -> float:
        return float(self.end - self.start)


@dataclass
class FlightLog:
    """Normalised flight log: named series + metadata + events.

    Readers populate this; analyzers consume it.  Nothing else in the package
    knows what a ULog or a dataflash block looks like.
    """

    series: Dict[str, Series] = field(default_factory=dict)
    events: List[Event] = field(default_factory=list)
    modes: List[ModeInterval] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- construction -----------------------------------------------------
    def add(
        self,
        name: str,
        time: Sequence[float] | np.ndarray,
        values: Sequence[float] | np.ndarray,
        units: str = "",
        source: str = "",
    ) -> Series:
        """Add (or replace) a channel and return it."""
        s = Series(name, np.asarray(time, dtype=float), np.asarray(values, dtype=float), units, source)
        self.series[name] = s
        return s

    def add_event(self, time: float, kind: str, detail: str = "", **data: Any) -> Event:
        ev = Event(float(time), kind, detail, dict(data))
        self.events.append(ev)
        return ev

    # -- access -----------------------------------------------------------
    def get(self, name: str) -> Optional[Series]:
        """Return a channel or ``None``.  Analyzers must degrade gracefully."""
        return self.series.get(name)

    def require(self, *names: str) -> bool:
        """True only if every named channel exists and is non-empty."""
        return all(
            (s := self.series.get(n)) is not None and len(s) > 0 for n in names
        )

    def first_present(self, *names: str) -> Optional[Series]:
        """Return the first of ``names`` that exists, else ``None``."""
        for n in names:
            s = self.series.get(n)
            if s is not None and len(s) > 0:
                return s
        return None

    def matching(self, prefix: str) -> List[Series]:
        """All channels whose name starts with ``prefix``, name-sorted."""
        return [self.series[k] for k in sorted(self.series) if k.startswith(prefix)]

    def events_of(self, *kinds: str) -> List[Event]:
        return [e for e in self.events if e.kind in kinds]

    # -- derived ----------------------------------------------------------
    @property
    def duration(self) -> float:
        """Wall-clock span covered by the longest channel, in seconds."""
        if "duration" in self.metadata:
            return float(self.metadata["duration"])
        spans = [(s.time[0], s.time[-1]) for s in self.series.values() if len(s) > 1]
        if not spans:
            return 0.0
        return float(max(b for _, b in spans) - min(a for a, _ in spans))

    @property
    def armed_intervals(self) -> List[Tuple[float, float]]:
        """``(arm_time, disarm_time)`` pairs derived from arm/disarm events.

        An unmatched final arm is closed at the end of the log, because that is
        exactly what a crash or a battery brownout looks like: the log stops
        while still armed.
        """
        out: List[Tuple[float, float]] = []
        open_t: Optional[float] = None
        for ev in sorted(self.events, key=lambda e: e.time):
            if ev.kind == "arm":
                open_t = ev.time
            elif ev.kind == "disarm" and open_t is not None:
                out.append((open_t, ev.time))
                open_t = None
        if open_t is not None:
            out.append((open_t, self.duration))
        return out

    def flight_window(self) -> Tuple[float, float]:
        """Best guess at the in-flight window.

        Prefers arm/disarm events; falls back to the full log.  Analyzers use
        this so that pre-arm bench noise and post-landing handling do not get
        reported as flight problems.
        """
        iv = self.armed_intervals
        if iv:
            return (min(a for a, _ in iv), max(b for _, b in iv))
        return (0.0, self.duration)

    def summary(self) -> Dict[str, Any]:
        return {
            "vehicle": self.metadata.get("vehicle", "unknown"),
            "firmware": self.metadata.get("firmware", "unknown"),
            "log_format": self.metadata.get("log_format", "unknown"),
            "duration_s": round(self.duration, 2),
            "channels": len(self.series),
            "events": len(self.events),
            "modes": [m.mode for m in self.modes],
        }


@dataclass
class Finding:
    """One diagnosed problem (or one clean-bill-of-health note).

    A finding is only useful if it survives the "so what do I do on Saturday
    morning?" test, so ``action`` is mandatory and must be concrete.
    """

    analyzer: str
    severity: Severity
    title: str
    explanation: str
    action: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    confidence: float = 1.0
    plot: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "analyzer": self.analyzer,
            "severity": self.severity.value,
            "title": self.title,
            "explanation": self.explanation,
            "action": self.action,
            "evidence": _jsonify(self.evidence),
            "confidence": round(float(self.confidence), 3),
        }
        if self.t_start is not None:
            d["t_start"] = round(float(self.t_start), 3)
        if self.t_end is not None:
            d["t_end"] = round(float(self.t_end), 3)
        return d


def _jsonify(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays into plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonify(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return round(obj, 6)
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return str(obj)


def concat_events(*groups: Iterable[Event]) -> List[Event]:
    """Merge event iterables into one time-sorted list."""
    out: List[Event] = []
    for g in groups:
        out.extend(g)
    out.sort(key=lambda e: e.time)
    return out
