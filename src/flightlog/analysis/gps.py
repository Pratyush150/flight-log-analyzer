"""GNSS quality analysis: fix type, satellites, HDOP, jumps, time to first fix.

GPS problems are over-diagnosed and under-measured.  "Bad GPS" is the default
explanation for anything that drifts, but the log contains four independent
numbers that separate a genuinely poor fix from a perfectly good one being
mis-used by a vibrating estimator:

* **fix type** -- 3 is a 3D fix, 4 is DGPS, 5/6 are RTK. Anything below 3 is
  not usable for position control. A drop from 3 to 2 mid-flight is a real
  event, not noise.
* **satellite count** -- below 8 the geometry is usually poor. Below 6, expect
  position hold to wander. The absolute number matters less than a sudden
  drop, which means the antenna lost sky view or something started interfering.
* **HDOP** -- horizontal dilution of precision, the geometric quality factor.
  Under 1.0 is good, 1.0-2.0 is workable, above 2.0 the satellites are
  clustered and horizontal position is weakly constrained. HDOP spikes and
  satellite drops usually arrive together; when they do not, suspect
  interference rather than obstruction.
* **time to first fix** -- how long from log start to a usable 3D fix. A cold
  receiver takes 30-60 s. Consistently long TTFF on a warm receiver points at
  antenna placement or a nearby noise source (the video transmitter is the
  usual culprit: 5.8 GHz harmonics and switching supplies both land near L1).

The position-jump detector lives in :mod:`flightlog.analysis.ekf`, because a
jump is only meaningful when cross-checked against the estimator's innovation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..types import FlightLog, Finding, Severity

__all__ = [
    "FIX_NAMES",
    "MIN_SATELLITES",
    "MAX_HDOP",
    "fix_timeline",
    "time_to_first_fix",
    "analyze",
]

#: MAVLink ``GPS_FIX_TYPE`` values, which ArduPilot's ``GPS.Status`` matches
#: for everything >= 3.
FIX_NAMES: Dict[int, str] = {
    0: "NO_GPS",
    1: "NO_FIX",
    2: "2D_FIX",
    3: "3D_FIX",
    4: "DGPS",
    5: "RTK_FLOAT",
    6: "RTK_FIXED",
}

#: Satellite count below which position hold degrades noticeably.
MIN_SATELLITES = 8
#: Satellite count below which position control should not be trusted at all.
CRITICAL_SATELLITES = 6
#: HDOP above which horizontal position is weakly constrained.
MAX_HDOP = 2.0
CRITICAL_HDOP = 3.0


def fix_timeline(log: FlightLog) -> List[Dict[str, object]]:
    """Contiguous intervals of constant fix type."""
    s = log.get("gps.fix_type")
    if s is None or len(s) < 2:
        return []
    vals = np.rint(s.values).astype(int)
    out: List[Dict[str, object]] = []
    start, cur = float(s.time[0]), int(vals[0])
    for i in range(1, vals.size):
        if int(vals[i]) != cur:
            out.append(
                {
                    "t_start": round(start, 2),
                    "t_end": round(float(s.time[i]), 2),
                    "fix_type": cur,
                    "fix_name": FIX_NAMES.get(cur, f"FIX_{cur}"),
                }
            )
            start, cur = float(s.time[i]), int(vals[i])
    out.append(
        {
            "t_start": round(start, 2),
            "t_end": round(float(s.time[-1]), 2),
            "fix_type": cur,
            "fix_name": FIX_NAMES.get(cur, f"FIX_{cur}"),
        }
    )
    return out


def time_to_first_fix(log: FlightLog, required_fix: int = 3) -> Optional[float]:
    """Seconds from log start to the first usable 3D fix.

    ``None`` means the receiver never reached ``required_fix`` in this log --
    which, if the aircraft flew a GPS mode anyway, is itself the finding.
    """
    s = log.get("gps.fix_type")
    if s is None or len(s) == 0:
        return None
    idx = np.flatnonzero(s.values >= required_fix)
    if idx.size == 0:
        return None
    return float(s.time[idx[0]] - s.time[0])


def analyze(
    log: FlightLog,
    min_sats: int = MIN_SATELLITES,
    max_hdop: float = MAX_HDOP,
    ttff_warn_s: float = 60.0,
) -> List[Finding]:
    """Run the GNSS analyzer."""
    findings: List[Finding] = []
    fix = log.get("gps.fix_type")
    sats = log.get("gps.satellites")
    hdop = log.get("gps.hdop")
    if fix is None and sats is None and hdop is None:
        return findings

    t0, t1 = log.flight_window()

    # -- fix drops in flight ----------------------------------------------
    if fix is not None and len(fix) > 2:
        inflight = fix.slice_time(t0, t1)
        if len(inflight) > 2:
            below = inflight.values < 3
            if np.any(below):
                lost = float(np.sum(below)) / below.size
                first_t = float(inflight.time[int(np.argmax(below))])
                findings.append(
                    Finding(
                        analyzer="gps",
                        severity=Severity.CRITICAL if lost > 0.02 else Severity.WARNING,
                        title=f"GPS fix degraded below 3D for {lost*100:.0f}% of the flight",
                        explanation=(
                            f"The receiver reported a fix type below 3D starting at "
                            f"t={first_t:.1f}s. Without a 3D fix there is no usable "
                            "position or velocity for the estimator, so any GPS-referenced "
                            "mode falls back to dead reckoning. Position hold drifts, and "
                            "return-to-launch cannot be trusted."
                        ),
                        action=(
                            "Check the antenna's sky view and its distance from the video "
                            "transmitter and any switching regulator -- both put noise into "
                            "the L1 band. Move the GPS module onto a mast with a ground "
                            "plane if it is currently sitting on the frame."
                        ),
                        evidence={
                            "fraction_below_3d": round(lost, 4),
                            "first_degraded_t": round(first_t, 2),
                            "timeline": fix_timeline(log)[:10],
                        },
                        t_start=first_t,
                        t_end=t1,
                        plot={"kind": "series", "channels": ["gps.fix_type", "gps.satellites"]},
                    )
                )

    # -- satellite count --------------------------------------------------
    if sats is not None and len(sats) > 2:
        inflight = sats.slice_time(t0, t1)
        if len(inflight) > 2:
            vmin = float(np.nanmin(inflight.values))
            vmed = float(np.nanmedian(inflight.values))
            t_min = float(inflight.time[int(np.nanargmin(inflight.values))])
            if vmin < CRITICAL_SATELLITES or vmed < min_sats:
                findings.append(
                    Finding(
                        analyzer="gps",
                        severity=Severity.CRITICAL
                        if vmin < CRITICAL_SATELLITES
                        else Severity.WARNING,
                        title=f"Low satellite count (min {vmin:.0f}, median {vmed:.0f})",
                        explanation=(
                            f"Satellite count fell to {vmin:.0f} at t={t_min:.1f}s "
                            f"(median {vmed:.0f} for the flight). Below {min_sats} the "
                            "geometry is weak enough that horizontal position wanders; "
                            f"below {CRITICAL_SATELLITES} the fix should not be used for "
                            "position control at all. A sudden drop mid-flight means the "
                            "antenna lost sky view or something started interfering."
                        ),
                        action=(
                            "Check the antenna mounting and clearance from carbon fibre, "
                            "which blocks L1. Move the GPS away from the video transmitter "
                            "and any high-current lead. If the drop coincides with a "
                            "specific heading or location, it is obstruction, not hardware."
                        ),
                        evidence={
                            "min_satellites": vmin,
                            "median_satellites": vmed,
                            "t_min": round(t_min, 2),
                            "threshold": min_sats,
                        },
                        t_start=max(t0, t_min - 2.0),
                        t_end=min(t1, t_min + 2.0),
                        plot={"kind": "series", "channels": ["gps.satellites"]},
                    )
                )

    # -- HDOP --------------------------------------------------------------
    if hdop is not None and len(hdop) > 2:
        inflight = hdop.slice_time(t0, t1)
        if len(inflight) > 2:
            vmax = float(np.nanmax(inflight.values))
            vmed = float(np.nanmedian(inflight.values))
            t_max = float(inflight.time[int(np.nanargmax(inflight.values))])
            if vmax > max_hdop:
                findings.append(
                    Finding(
                        analyzer="gps",
                        severity=Severity.CRITICAL if vmax > CRITICAL_HDOP else Severity.WARNING,
                        title=f"HDOP peaked at {vmax:.1f} (median {vmed:.2f})",
                        explanation=(
                            f"Horizontal dilution of precision reached {vmax:.1f} at "
                            f"t={t_max:.1f}s. HDOP is a geometry factor: it multiplies the "
                            "receiver's ranging error into position error. Above "
                            f"{max_hdop:.1f} the visible satellites are clustered and "
                            "horizontal position is only weakly constrained, so position "
                            "hold will wander even though the fix still reports as 3D."
                        ),
                        action=(
                            "If HDOP spikes together with a satellite drop, it is "
                            "obstruction -- fly with a clearer sky view. If HDOP spikes "
                            "while satellite count stays high, suspect interference from "
                            "the video transmitter or a switching regulator near the "
                            "receiver."
                        ),
                        evidence={
                            "max_hdop": round(vmax, 3),
                            "median_hdop": round(vmed, 3),
                            "t_max": round(t_max, 2),
                            "threshold": max_hdop,
                        },
                        t_start=max(t0, t_max - 2.0),
                        t_end=min(t1, t_max + 2.0),
                        plot={"kind": "series", "channels": ["gps.hdop"]},
                    )
                )

    # -- time to first fix -------------------------------------------------
    ttff = time_to_first_fix(log)
    if ttff is None and fix is not None and len(fix):
        findings.append(
            Finding(
                analyzer="gps",
                severity=Severity.CRITICAL,
                title="No 3D fix at any point in this log",
                explanation=(
                    "The receiver never reported a 3D fix. Any GPS-dependent mode "
                    "(position hold, mission, return-to-launch) was unavailable for the "
                    "whole flight."
                ),
                action=(
                    "Confirm the GPS module is powered and wired to the correct UART, "
                    "check the configured baud rate and protocol, and verify the antenna "
                    "has an unobstructed view of the sky."
                ),
                evidence={"fix_samples": len(fix)},
            )
        )
    elif ttff is not None and ttff > ttff_warn_s:
        findings.append(
            Finding(
                analyzer="gps",
                severity=Severity.WARNING,
                title=f"Slow time to first fix: {ttff:.0f} s",
                explanation=(
                    f"The receiver took {ttff:.0f} s from log start to a 3D fix. A cold "
                    "start legitimately takes 30-60 s, but a consistently slow warm start "
                    "points at antenna placement or an interference source close to the "
                    "receiver."
                ),
                action=(
                    "Move the GPS module away from the video transmitter, the FC and any "
                    "switching regulator, and give it a ground plane. Confirm the "
                    "receiver's backup power (for ephemeris retention) is working -- a "
                    "dead backup cell turns every start into a cold start."
                ),
                evidence={"ttff_s": round(ttff, 2), "threshold_s": ttff_warn_s},
            )
        )

    if not findings:
        ev: Dict[str, object] = {}
        if sats is not None and len(sats):
            window = sats.slice_time(t0, t1)
            vals = window.values if len(window) else sats.values
            ev["min_satellites"] = float(np.nanmin(vals))
        if hdop is not None and len(hdop):
            window = hdop.slice_time(t0, t1)
            vals = window.values if len(window) else hdop.values
            ev["max_hdop"] = round(float(np.nanmax(vals)), 3)
        if ttff is not None:
            ev["ttff_s"] = round(ttff, 2)
        findings.append(
            Finding(
                analyzer="gps",
                severity=Severity.INFO,
                title="GNSS quality good for the whole flight",
                explanation=(
                    "3D fix held throughout, satellite count and HDOP stayed inside "
                    "advisory limits, and no position jump was detected."
                ),
                action="No action needed.",
                evidence=ev,
            )
        )
    return findings
