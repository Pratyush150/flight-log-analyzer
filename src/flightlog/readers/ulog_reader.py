"""PX4 ULog reader (optional dependency: ``pyulog``).

The import is guarded.  If ``pyulog`` is absent the module still imports, still
reports :data:`AVAILABLE` as ``False``, and :func:`read_ulog` raises a
:class:`MissingDependencyError` with the exact pip command needed.  Nothing
else in the package -- tests included -- depends on pyulog being installed.

What this reader does beyond a straight field copy
--------------------------------------------------
* Converts ULog microsecond timestamps to seconds and rebases to log start.
* Flattens vector fields (``gyro_rad[0]``) into scalar channels.
* Handles multi-instance topics: ``sensor_accel`` instance 0 becomes
  ``accel.*``, instance 1 becomes ``accel.*@1`` so a dual-IMU log can be
  compared instead of silently overwriting itself.
* Flips ``vehicle_local_position.z`` from NED-down to altitude-up, because
  every human reading a report expects "altitude" to increase upwards.
* Scales GPS lat/lon from 1e-7 degrees and altitude from millimetres.
* Extracts arm/disarm, mode changes and logged messages into events.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from ..channels import ULOG_MAP, units_for
from ..types import FlightLog, ModeInterval

try:  # pragma: no cover - depends on the host environment
    from pyulog import ULog as _ULog  # type: ignore

    AVAILABLE = True
except Exception:  # pragma: no cover - the whole point of the guard
    _ULog = None  # type: ignore
    AVAILABLE = False

__all__ = ["AVAILABLE", "MissingDependencyError", "read_ulog", "can_read"]

INSTALL_HINT = "pip install pyulog"

#: PX4 nav_state enum -> readable mode name.  Kept local rather than imported
#: from PX4 so the reader does not depend on a firmware checkout.
NAV_STATE_NAMES: Dict[int, str] = {
    0: "MANUAL",
    1: "ALTCTL",
    2: "POSCTL",
    3: "AUTO.MISSION",
    4: "AUTO.LOITER",
    5: "AUTO.RTL",
    10: "ACRO",
    12: "DESCEND",
    13: "TERMINATION",
    14: "OFFBOARD",
    15: "STABILIZED",
    17: "AUTO.TAKEOFF",
    18: "AUTO.LAND",
    19: "AUTO.FOLLOW_TARGET",
    20: "AUTO.PRECLAND",
}


class MissingDependencyError(ImportError):
    """Raised when a reader needs an optional package that is not installed."""


def can_read(path: str) -> bool:
    """True if the path has a ULog extension (does not check availability)."""
    return os.path.splitext(path)[1].lower() in (".ulg", ".ulog")


def read_ulog(path: str, topics: Optional[List[str]] = None) -> FlightLog:
    """Read a ``.ulg`` file into a :class:`~flightlog.types.FlightLog`.

    Parameters
    ----------
    path:
        Path to the ULog file.
    topics:
        Optional whitelist of ULog topic names.  Restricting topics on a large
        log (a 20-minute flight at full logging is easily 200 MB) cuts parse
        time by an order of magnitude.  Default: every topic this package
        knows how to map, plus the ones needed for events.

    Raises
    ------
    MissingDependencyError
        If ``pyulog`` is not installed.
    """
    if not AVAILABLE:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            f"reading {path!r} requires pyulog. Install it with: {INSTALL_HINT}"
        )
    if topics is None:
        topics = sorted({t for t, _ in ULOG_MAP} | {"vehicle_status", "vehicle_land_detected"})

    ulog = _ULog(path, message_name_filter_list=topics)  # type: ignore[misc]
    log = FlightLog()
    t0 = _log_start(ulog)

    log.metadata.update(
        {
            "log_format": "ulog",
            "source": os.path.abspath(path),
            "vehicle": _info(ulog, "sys_name", "PX4"),
            "firmware": _firmware_string(ulog),
            "hardware": _info(ulog, "ver_hw", "unknown"),
            "sys_uuid": _info(ulog, "sys_uuid", ""),
            "dropouts": len(getattr(ulog, "dropouts", []) or []),
        }
    )

    for dataset in ulog.data_list:
        _ingest_dataset(log, dataset, t0)

    _extract_events(log, ulog, t0)
    _post_process(log)
    log.metadata["duration"] = log.duration
    return log


def _log_start(ulog: Any) -> float:
    """Earliest timestamp across all datasets, in microseconds."""
    starts = [
        float(ds.data["timestamp"][0])
        for ds in ulog.data_list
        if "timestamp" in ds.data and len(ds.data["timestamp"])
    ]
    return min(starts) if starts else 0.0


def _info(ulog: Any, key: str, default: str) -> str:
    val = getattr(ulog, "msg_info_dict", {}).get(key, default)
    return str(val)


def _firmware_string(ulog: Any) -> str:
    info = getattr(ulog, "msg_info_dict", {})
    ver = info.get("ver_sw_release")
    git = str(info.get("ver_sw", ""))[:9]
    if ver is not None:
        try:
            v = int(ver)
            major, minor, patch = (v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF
            return f"PX4 v{major}.{minor}.{patch} ({git})" if git else f"PX4 v{major}.{minor}.{patch}"
        except (TypeError, ValueError):
            pass
    return f"PX4 ({git})" if git else "PX4 (unknown)"


def _ingest_dataset(log: FlightLog, dataset: Any, t0: float) -> None:
    """Copy every mapped field of one ULog topic into canonical channels."""
    topic = dataset.name
    inst = int(getattr(dataset, "multi_id", 0) or 0)
    data = dataset.data
    if "timestamp" not in data:
        return
    t = (np.asarray(data["timestamp"], dtype=float) - t0) * 1e-6

    for field, values in data.items():
        if field == "timestamp":
            continue
        canonical = ULOG_MAP.get((topic, field))
        if canonical is None:
            continue
        v = np.asarray(values, dtype=float)
        if v.size != t.size:
            continue
        name = canonical if inst == 0 else f"{canonical}@{inst}"
        log.add(name, t, v, units_for(canonical), f"{topic}[{inst}].{field}")


def _extract_events(log: FlightLog, ulog: Any, t0: float) -> None:
    """Build arm/disarm, mode and failsafe events from ``vehicle_status``."""
    status = next((d for d in ulog.data_list if d.name == "vehicle_status"), None)
    if status is not None and "timestamp" in status.data:
        t = (np.asarray(status.data["timestamp"], dtype=float) - t0) * 1e-6
        arming = np.asarray(status.data.get("arming_state", []), dtype=float)
        if arming.size == t.size and arming.size > 1:
            # PX4 arming_state: 2 == ARMED in the long-standing enum.
            armed = arming >= 2
            log.add("armed", t, armed.astype(float), "bool", "vehicle_status.arming_state")
            for i in range(1, armed.size):
                if armed[i] and not armed[i - 1]:
                    log.add_event(float(t[i]), "arm", "armed")
                elif armed[i - 1] and not armed[i]:
                    log.add_event(float(t[i]), "disarm", "disarmed")

        nav = np.asarray(status.data.get("nav_state", []), dtype=float)
        if nav.size == t.size and nav.size > 0:
            log.add("mode.id", t, nav, "enum", "vehicle_status.nav_state")
            start, cur = float(t[0]), int(nav[0])
            for i in range(1, nav.size):
                if int(nav[i]) != cur:
                    log.modes.append(ModeInterval(start, float(t[i]), _mode_name(cur)))
                    log.add_event(float(t[i]), "mode_change", _mode_name(int(nav[i])))
                    start, cur = float(t[i]), int(nav[i])
            log.modes.append(ModeInterval(start, float(t[-1]), _mode_name(cur)))

        fs = np.asarray(status.data.get("failsafe", []), dtype=float)
        if fs.size == t.size and fs.size > 1:
            for i in range(1, fs.size):
                if fs[i] > 0 and fs[i - 1] <= 0:
                    log.add_event(float(t[i]), "failsafe", "vehicle_status.failsafe set")

    for msg in getattr(ulog, "logged_messages", []) or []:
        ts = (float(getattr(msg, "timestamp", 0)) - t0) * 1e-6
        text = str(getattr(msg, "message", ""))
        level = int(getattr(msg, "log_level", 6))
        kind = "error" if level <= 3 else "message"
        if "reset" in text.lower() and "ekf" in text.lower():
            kind = "ekf_reset"
        log.add_event(ts, kind, text, level=level)

    log.events.sort(key=lambda e: e.time)


def _mode_name(nav_state: int) -> str:
    return NAV_STATE_NAMES.get(nav_state, f"NAV_STATE_{nav_state}")


def _post_process(log: FlightLog) -> None:
    """Unit fixes that only make sense once the raw fields are in place."""
    z = log.get("alt.ekf")
    if z is not None:
        # vehicle_local_position.z is NED down; report altitude up.
        log.add("alt.ekf", z.time, -z.values, "m", z.source + " (negated NED z)")
    for name in ("gps.lat", "gps.lon"):
        s = log.get(name)
        if s is not None and np.nanmax(np.abs(s.values)) > 400.0:
            log.add(name, s.time, s.values * 1e-7, "deg", s.source + " (1e-7 deg)")
    alt_mm = log.get("gps.alt_mm")
    if alt_mm is not None:
        log.add("alt.gps", alt_mm.time, alt_mm.values * 1e-3, "m", alt_mm.source + " (mm)")
        log.series.pop("gps.alt_mm", None)
    hdop = log.get("gps.hdop")
    if hdop is not None and np.nanmedian(hdop.values) > 50.0:
        log.add("gps.hdop", hdop.time, hdop.values * 0.01, "-", hdop.source + " (cm scale)")

    # actuator_outputs on PWM hardware is in microseconds; actuator_motors is
    # already 0..1.  Normalise so downstream code has one convention.
    for s in list(log.matching("motor.")):
        if s.values.size and np.nanmax(s.values) > 2.0:
            norm = np.clip((s.values - 1000.0) / 1000.0, 0.0, 1.0)
            log.add(s.name, s.time, norm, "fraction", s.source + " (PWM 1000-2000 normalised)")
