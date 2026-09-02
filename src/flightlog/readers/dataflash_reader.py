"""ArduPilot dataflash reader (optional dependency: ``pymavlink``).

Reads both binary ``.bin`` logs and text ``.log`` dumps through
``pymavlink.DFReader``.  The import is guarded exactly like
:mod:`flightlog.readers.ulog_reader`: no pymavlink, no crash, just a clear
error at call time.

ArduPilot quirks handled here
-----------------------------
* ``TimeUS`` is microseconds since boot -- converted and rebased.
* ``ATT`` angles are **degrees**; ``RATE`` is deg/s.  Converted to radians so
  every analyzer sees SI.
* ``RCOU.C1..C6`` are PWM microseconds (1000-2000).  Normalised to 0..1.
* ``BAT.RemPct`` is a percentage.  Converted to a 0..1 fraction.
* ``GPS.Status`` is ArduPilot's own fix enum (0 none, 1 no-fix, 2 2D, 3 3D,
  4 DGPS, 5/6 RTK), which happens to line up with the MAVLink ``fix_type``
  values this package uses for anything >= 3.
* ``MODE`` messages carry the mode name directly, which is why an ArduPilot
  mode timeline is more readable than a PX4 one.
* ``ERR`` messages are the failsafe/subsystem error stream -- the single most
  useful message type in a crash log, so they are all promoted to events.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from ..channels import DATAFLASH_MAP, units_for
from ..types import Event, FlightLog, ModeInterval

try:  # pragma: no cover - depends on the host environment
    from pymavlink import DFReader as _DFReader  # type: ignore

    AVAILABLE = True
except Exception:  # pragma: no cover
    _DFReader = None  # type: ignore
    AVAILABLE = False

__all__ = ["AVAILABLE", "MissingDependencyError", "read_dataflash", "can_read"]

INSTALL_HINT = "pip install pymavlink"

#: ArduPilot ``ERR`` subsystem ids that matter for a health report.
ERR_SUBSYSTEMS: Dict[int, str] = {
    2: "radio failsafe",
    5: "battery failsafe",
    6: "GPS failsafe",
    9: "GCS failsafe",
    10: "fence breach",
    11: "flight mode change denied",
    12: "GPS glitch",
    15: "EKF check",
    16: "EKF failsafe",
    17: "barometer glitch",
    18: "CPU load",
    24: "terrain data missing",
    27: "vibration failsafe",
}

#: ArduPilot ``GPS.Status`` values.
GPS_FIX_NAMES: Dict[int, str] = {
    0: "NO_GPS",
    1: "NO_FIX",
    2: "2D_FIX",
    3: "3D_FIX",
    4: "DGPS",
    5: "RTK_FLOAT",
    6: "RTK_FIXED",
}


class MissingDependencyError(ImportError):
    """Raised when a reader needs an optional package that is not installed."""


def can_read(path: str) -> bool:
    """True if the path looks like a dataflash log."""
    return os.path.splitext(path)[1].lower() in (".bin", ".log", ".px4log")


def read_dataflash(path: str, types: Optional[List[str]] = None) -> FlightLog:
    """Read an ArduPilot ``.bin``/``.log`` into a :class:`FlightLog`.

    Parameters
    ----------
    path:
        Log file path.
    types:
        Optional whitelist of dataflash message names.  Defaults to the set
        this package maps plus ``MODE``, ``EV`` and ``ERR``.  Filtering
        matters: a full ``.bin`` from a long flight contains millions of
        messages, and parsing everything wastes minutes.

    Raises
    ------
    MissingDependencyError
        If ``pymavlink`` is not installed.
    """
    if not AVAILABLE:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            f"reading {path!r} requires pymavlink. Install it with: {INSTALL_HINT}"
        )
    wanted = set(types) if types else {m for m, _ in DATAFLASH_MAP} | {"MODE", "EV", "ERR", "MSG"}

    ext = os.path.splitext(path)[1].lower()
    if ext == ".bin":
        reader = _DFReader.DFReader_binary(path)  # type: ignore[union-attr]
    else:
        reader = _DFReader.DFReader_text(path)  # type: ignore[union-attr]

    raw: Dict[str, Dict[str, List[float]]] = {}
    events: List[Event] = []
    mode_marks: List[tuple] = []
    t0: Optional[float] = None

    while True:
        msg = reader.recv_match(type=list(wanted))
        if msg is None:
            break
        mtype = msg.get_type()
        t_us = getattr(msg, "TimeUS", None)
        if t_us is None:
            t_us = float(getattr(msg, "_timestamp", 0.0)) * 1e6
        t_us = float(t_us)
        if t0 is None:
            t0 = t_us
        t = (t_us - t0) * 1e-6

        if mtype == "MODE":
            name = str(getattr(msg, "Mode", getattr(msg, "ModeNum", "?")))
            mode_marks.append((t, name))
            events.append(Event(t, "mode_change", name))
            continue
        if mtype == "ERR":
            sub = int(getattr(msg, "Subsys", 0))
            code = int(getattr(msg, "ECode", 0))
            label = ERR_SUBSYSTEMS.get(sub, f"subsystem {sub}")
            kind = "failsafe" if code != 0 else "failsafe_cleared"
            events.append(Event(t, kind, f"{label} (ECode={code})", {"subsys": sub, "ecode": code}))
            continue
        if mtype == "EV":
            evid = int(getattr(msg, "Id", 0))
            # 10 = ARMED, 11 = DISARMED in ArduPilot's LogEvent enum.
            if evid == 10:
                events.append(Event(t, "arm", "armed"))
            elif evid == 11:
                events.append(Event(t, "disarm", "disarmed"))
            else:
                events.append(Event(t, "event", f"EV id={evid}", {"id": evid}))
            continue
        if mtype == "MSG":
            text = str(getattr(msg, "Message", ""))
            kind = "ekf_reset" if "ekf" in text.lower() and "reset" in text.lower() else "message"
            events.append(Event(t, kind, text))
            continue

        bucket = raw.setdefault(mtype, {"__t__": []})
        bucket["__t__"].append(t)
        for field in msg.get_fieldnames():
            if (mtype, field) not in DATAFLASH_MAP:
                continue
            try:
                val = float(getattr(msg, field))
            except (TypeError, ValueError):
                val = float("nan")
            bucket.setdefault(field, []).append(val)

    log = FlightLog()
    log.metadata.update(
        {
            "log_format": "dataflash",
            "source": os.path.abspath(path),
            "vehicle": _vehicle_name(reader),
            "firmware": _firmware_name(reader),
        }
    )

    for mtype, fields in raw.items():
        t = np.asarray(fields["__t__"], dtype=float)
        for field, values in fields.items():
            if field == "__t__":
                continue
            v = np.asarray(values, dtype=float)
            if v.size != t.size:
                continue
            canonical = DATAFLASH_MAP[(mtype, field)]
            log.add(canonical, t, v, units_for(canonical), f"{mtype}.{field}")

    log.events.extend(events)
    log.events.sort(key=lambda e: e.time)
    _build_modes(log, mode_marks)
    _post_process(log)
    log.metadata["duration"] = log.duration
    return log


def _vehicle_name(reader: Any) -> str:
    for attr in ("vehicle_type_string", "vehicle_type"):
        val = getattr(reader, attr, None)
        if val:
            return str(val)
    return "ArduPilot"


def _firmware_name(reader: Any) -> str:
    params = getattr(reader, "params", {}) or {}
    ver = params.get("SYSID_SW_MREV")
    msgs = getattr(reader, "messages", {}) or {}
    vrec = msgs.get("VER")
    if vrec is not None:
        fw = getattr(vrec, "FWS", None)
        if fw:
            return str(fw)
    return f"ArduPilot (SYSID_SW_MREV={ver})" if ver else "ArduPilot (unknown)"


def _build_modes(log: FlightLog, marks: List[tuple]) -> None:
    if not marks:
        return
    marks = sorted(marks, key=lambda m: m[0])
    end_of_log = log.duration or marks[-1][0]
    for i, (start, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else end_of_log
        if end > start:
            log.modes.append(ModeInterval(start, end, name))


def _post_process(log: FlightLog) -> None:
    """Apply ArduPilot's unit conventions."""
    for name in ("att.roll", "att.pitch", "att.yaw",
                 "att.roll_sp", "att.pitch_sp", "att.yaw_sp"):
        s = log.get(name)
        if s is not None:
            log.add(name, s.time, np.deg2rad(s.values), "rad", s.source + " (deg->rad)")
    for name in ("rate.roll", "rate.pitch", "rate.yaw",
                 "rate.roll_sp", "rate.pitch_sp", "rate.yaw_sp"):
        s = log.get(name)
        if s is not None:
            log.add(name, s.time, np.deg2rad(s.values), "rad/s", s.source + " (deg/s->rad/s)")

    # DFReader applies the format multipliers when reading a binary .bin, but
    # a text .log dump can arrive with the raw integers. Both cases are handled
    # by testing the magnitude: real degrees never exceed 180, and a real HDOP
    # is never in the tens.
    for name in ("gps.lat", "gps.lon"):
        s_ = log.get(name)
        if s_ is not None and s_.values.size and np.nanmax(np.abs(s_.values)) > 400.0:
            log.add(name, s_.time, s_.values * 1e-7, "deg", s_.source + " (1e-7 deg)")
    hdop = log.get("gps.hdop")
    if hdop is not None and hdop.values.size and np.nanmedian(hdop.values) > 50.0:
        log.add("gps.hdop", hdop.time, hdop.values * 0.01, "-", hdop.source + " (cm scale)")

    rem = log.get("bat.remaining")
    if rem is not None and rem.values.size and np.nanmax(rem.values) > 1.5:
        log.add("bat.remaining", rem.time, rem.values / 100.0, "fraction",
                rem.source + " (percent->fraction)")

    for s in list(log.matching("motor.")):
        if s.values.size and np.nanmax(s.values) > 2.0:
            norm = np.clip((s.values - 1000.0) / 1000.0, 0.0, 1.0)
            log.add(s.name, s.time, norm, "fraction", s.source + " (PWM 1000-2000 normalised)")

    rssi = log.get("rc.rssi")
    if rssi is not None and rssi.values.size and np.nanmax(rssi.values) > 1.5:
        log.add("rc.rssi", rssi.time, np.clip(rssi.values / 255.0, 0, 1), "fraction",
                rssi.source + " (0-255 normalised)")

    load = log.get("cpu.load")
    if load is not None and load.values.size and np.nanmax(load.values) > 1.5:
        log.add("cpu.load", load.time, load.values / 1000.0, "fraction",
                load.source + " (permille->fraction)")

    # ESC RPM is revolutions per minute; analyzers want Hz.
    for s in list(log.matching("rpm.")):
        if s.values.size and np.nanmedian(np.abs(s.values)) > 500.0:
            log.add(s.name, s.time, s.values / 60.0, "Hz", s.source + " (RPM->Hz)")
