"""Estimator (EKF) health analysis.

What an innovation is, in one paragraph
---------------------------------------
An extended Kalman filter carries a prediction of what each sensor *should*
read.  The difference between that prediction and the actual measurement is the
**innovation**.  The filter also carries a variance for that difference.  The
ratio ``innovation^2 / innovation_variance`` -- what PX4 and ArduPilot log as
``*_test_ratio`` -- is the quantity both firmwares threshold at 1.0: below 1.0
the measurement is consistent with the filter's own uncertainty and gets fused;
above 1.0 it is rejected.  So a test ratio that sits near 1.0 means the
estimator is on the edge of throwing away the sensor it needs to hold position.

Reading test ratios correctly
-----------------------------
* < 0.3 -- healthy, no comment needed.
* 0.3-0.5 -- worth noting; margin is shrinking.
* 0.5-1.0 -- the filter is stressed. Almost always a *symptom*: vibration,
  a bad compass calibration, or a GPS with a poor view of the sky.
* > 1.0 -- measurements are being rejected. Position hold degrades, then the
  aircraft falls back to a lower mode or triggers failsafe.

Which ratio is high tells you where to look:

* ``vel``/``pos`` high, ``mag`` fine  -> GPS quality or vibration-induced
  velocity error.
* ``mag`` high, others fine           -> compass: calibration, interference
  from power leads, or a nearby ferrous payload.
* ``hgt`` high                        -> baro disagrees with GPS/rangefinder;
  prop wash over the baro, or a genuine baro drift.
* everything high at once             -> vibration, or an IMU that is failing.

This analyzer never says "your EKF is broken".  It says which residual went
out of range, when, and which physical cause is consistent with that pattern.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..types import Event, FlightLog, Finding, Series, Severity

__all__ = [
    "RATIO_WARN",
    "RATIO_CRITICAL",
    "analyze",
    "ratio_excursions",
    "detect_gps_glitches",
    "height_source_disagreement",
    "magnetometer_consistency",
    "detect_ekf_resets",
]

#: Test ratio above which the estimator is stressed but still fusing.
RATIO_WARN = 0.5
#: Test ratio above which the estimator is rejecting measurements.
RATIO_CRITICAL = 1.0
#: Minimum duration (s) of an excursion before it is reported. Single-sample
#: spikes are common and harmless; sustained excursions are the problem.
MIN_EXCURSION_S = 0.5

_RATIO_MEANING: Dict[str, Tuple[str, str]] = {
    "vel": (
        "velocity innovations",
        "GPS velocity disagrees with the inertial prediction. The usual causes, in "
        "order of likelihood: accelerometer vibration corrupting the prediction, GPS "
        "multipath near buildings or trees, and a GPS-to-IMU lever-arm offset that "
        "has not been configured.",
    ),
    "pos": (
        "position innovations",
        "GPS position disagrees with the dead-reckoned estimate. Look for a GPS "
        "glitch, a satellite count drop, or a jump in HDOP at the same timestamp.",
    ),
    "hgt": (
        "height innovations",
        "The height source disagrees with the inertial prediction. On a multirotor "
        "this is usually the barometer: prop wash or cabin pressure changes move the "
        "baro reading independently of real altitude.",
    ),
    "mag": (
        "magnetometer innovations",
        "The compass disagrees with the filter's heading. Either the calibration is "
        "stale, or a high-current lead is running close to the compass and the field "
        "distorts in proportion to throttle.",
    ),
}

_RATIO_ACTION: Dict[str, str] = {
    "vel": (
        "Fix vibration first (see the vibration section). If vibration is already "
        "clean, check the GPS antenna's sky view and set the GPS lever-arm parameters "
        "(PX4 EKF2_GPS_POS_*, ArduPilot GPS_POS*)."
    ),
    "pos": (
        "Cross-check the GPS section of this report for a satellite-count or HDOP "
        "event at the same time. If the GPS is clean, the cause is vibration feeding "
        "false acceleration into the prediction."
    ),
    "hgt": (
        "Foam over the barometer port and shield it from prop wash. If the aircraft "
        "has a rangefinder, confirm it is configured as the primary height source "
        "below its maximum range."
    ),
    "mag": (
        "Re-run the compass calibration away from steel and rebar, then re-fly. If the "
        "distortion tracks throttle, physically move the compass away from the power "
        "leads and the ESCs -- a mast or a GPS/compass module on a stalk is the fix."
    ),
}


def _excursions(
    s: Series, threshold: float, min_duration: float = MIN_EXCURSION_S
) -> List[Tuple[float, float, float]]:
    """Return ``(start, end, peak)`` for each sustained threshold crossing."""
    if len(s) < 2:
        return []
    over = s.values > threshold
    out: List[Tuple[float, float, float]] = []
    i = 0
    n = over.size
    while i < n:
        if not over[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and over[j + 1]:
            j += 1
        start, end = float(s.time[i]), float(s.time[j])
        if end - start >= min_duration or (j > i and j - i >= 2):
            out.append((start, end, float(np.max(s.values[i : j + 1]))))
        i = j + 1
    return out


def ratio_excursions(
    log: FlightLog, threshold: float = RATIO_WARN
) -> Dict[str, List[Tuple[float, float, float]]]:
    """Sustained innovation test-ratio excursions per estimator channel.

    Named ``ratio_excursions`` rather than ``test_ratio_excursions`` on
    purpose: a public function whose name starts with ``test_`` gets collected
    as a test case by pytest in any project that imports it.
    """
    out: Dict[str, List[Tuple[float, float, float]]] = {}
    t0, t1 = log.flight_window()
    for key in ("vel", "pos", "hgt", "mag"):
        s = log.get(f"ekf.test_ratio.{key}")
        if s is None or len(s) < 4:
            continue
        ex = _excursions(s.slice_time(t0, t1), threshold)
        if ex:
            out[key] = ex
    return out


def detect_gps_glitches(
    log: FlightLog,
    jump_m: float = 5.0,
    max_speed_mps: float = 30.0,
    merge_window: float = 6.0,
) -> List[Dict[str, object]]:
    """Detect GPS position jumps that no aircraft could physically have flown.

    Works on lat/lon when available (converted to metres about the median
    latitude), falling back to the estimator's own north/east position.  The
    test is an implied speed test: if two consecutive fixes imply a speed above
    ``max_speed_mps``, the receiver jumped, the aircraft did not.

    A jump also shows in the EKF position innovation at the same timestamp,
    which is what raises confidence from "maybe" to "yes"; that cross-check is
    applied here.
    """
    lat, lon = log.get("gps.lat"), log.get("gps.lon")
    if lat is not None and lon is not None and len(lat) > 3 and len(lat) == len(lon):
        lat0 = float(np.nanmedian(lat.values))
        m_lat = 111_320.0
        m_lon = m_lat * float(np.cos(np.deg2rad(lat0)))
        north = (lat.values - lat0) * m_lat
        east = (lon.values - float(np.nanmedian(lon.values))) * m_lon
        t = lat.time
    else:
        n_s, e_s = log.get("pos.north"), log.get("pos.east")
        if n_s is None or e_s is None or len(n_s) < 4:
            return []
        t = n_s.time
        north = n_s.values
        east = e_s.interp_to(t)

    if t.size < 3:
        return []
    dt = np.diff(t)
    dt[dt <= 0] = np.nan
    step = np.hypot(np.diff(north), np.diff(east))
    speed = step / dt

    innov = log.first_present("ekf.innov.vel_n", "ekf.innov.pos_d")
    glitches: List[Dict[str, object]] = []
    flagged = (step > jump_m) & (speed > max_speed_mps)
    idx = np.flatnonzero(flagged)
    for i in idx:
        t_ev = float(t[i + 1])
        entry: Dict[str, object] = {
            "time": round(t_ev, 3),
            "jump_m": round(float(step[i]), 2),
            "implied_speed_mps": round(float(speed[i]), 1),
        }
        if innov is not None and len(innov):
            near = np.abs(innov.time - t_ev) < 1.0
            if np.any(near):
                entry["innovation_at_event"] = round(float(np.max(np.abs(innov.values[near]))), 3)
        sats = log.get("gps.satellites")
        if sats is not None and len(sats):
            near = np.abs(sats.time - t_ev) < 1.5
            if np.any(near):
                entry["satellites_at_event"] = int(np.min(sats.values[near]))
        glitches.append(entry)

    # A glitch produces two jumps: one when the fix leaves the true position and
    # one when it snaps back. Reporting them separately would double-count a
    # single receiver event, so jumps within `merge_window` seconds are folded
    # into one finding that keeps the largest jump and records the span.
    merged: List[Dict[str, object]] = []
    for g in glitches:
        if merged and float(g["time"]) - float(merged[-1]["t_end"]) <= merge_window:
            prev = merged[-1]
            prev["t_end"] = g["time"]
            prev["jump_count"] = int(prev.get("jump_count", 1)) + 1
            if float(g["jump_m"]) > float(prev["jump_m"]):
                prev["jump_m"] = g["jump_m"]
                prev["implied_speed_mps"] = g["implied_speed_mps"]
            continue
        entry = dict(g)
        entry["t_end"] = g["time"]
        entry["jump_count"] = 1
        merged.append(entry)
    for m in merged:
        m["duration_s"] = round(float(m["t_end"]) - float(m["time"]), 2)
    return merged


def height_source_disagreement(log: FlightLog) -> Optional[Dict[str, object]]:
    """Compare baro, GPS and rangefinder altitude after removing bias.

    Each source has a different datum: the barometer is relative to whatever
    pressure it saw at boot, GPS altitude is relative to the WGS-84 ellipsoid,
    and a rangefinder measures to the ground under the aircraft.  Comparing raw
    values is meaningless, so the constant offset is removed first (median
    difference over the flight) and only the *drift* is reported.

    A growing divergence is the signal that matters.  A baro drifting up at
    0.1 m/s does not look alarming in the log, but over a five-minute flight it
    is 30 m of altitude error, and in altitude-hold the aircraft will fly it.
    """
    baro = log.get("alt.baro")
    others = {
        "gps": log.get("alt.gps"),
        "rangefinder": log.get("alt.rangefinder"),
        "ekf": log.get("alt.ekf"),
    }
    if baro is None or len(baro) < 20:
        return None
    t0, t1 = log.flight_window()
    b = baro.slice_time(t0, t1)
    if len(b) < 20:
        return None

    worst: Optional[Dict[str, object]] = None
    for name, s in others.items():
        if s is None or len(s) < 20:
            continue
        other = s.interp_to(b.time)
        diff = b.values - other
        good = np.isfinite(diff)
        if good.sum() < 20:
            continue
        diff = diff[good]
        tt = b.time[good]
        diff = diff - float(np.median(diff))  # remove datum offset
        # Least-squares drift rate over the flight.
        slope = float(np.polyfit(tt, diff, 1)[0])
        spread = float(np.max(diff) - np.min(diff))
        entry = {
            "reference": "baro",
            "compared_to": name,
            "drift_mps": round(slope, 4),
            "total_divergence_m": round(abs(slope) * (tt[-1] - tt[0]), 2),
            "peak_to_peak_m": round(spread, 2),
        }
        if worst is None or abs(slope) > abs(float(worst["drift_mps"])):
            worst = entry
    return worst


def magnetometer_consistency(log: FlightLog) -> Optional[Dict[str, object]]:
    """Check the magnetometer for throttle-correlated distortion.

    The earth's field has constant magnitude.  Any change in the measured field
    *norm* during flight is therefore distortion, not navigation.  Correlating
    that change against throttle separates the two common causes: a constant
    offset is a bad calibration or a nearby ferrous object, while a
    throttle-correlated change is current flowing in a lead near the compass.

    The second one is the interesting diagnosis, because it explains the "yaw
    slowly rotates" and "toilet-bowling" complaints exactly: heading error
    grows with throttle, so the aircraft's idea of north shifts as it works
    harder.
    """
    mx, my, mz = log.get("mag.x"), log.get("mag.y"), log.get("mag.z")
    if mx is None or my is None or mz is None or len(mx) < 50:
        return None
    t0, t1 = log.flight_window()
    m = mx.slice_time(t0, t1)
    if len(m) < 50:
        return None
    norm = np.sqrt(m.values**2 + my.interp_to(m.time) ** 2 + mz.interp_to(m.time) ** 2)
    base = float(np.median(norm))
    if base <= 0:
        return None
    variation = float(np.max(norm) - np.min(norm)) / base

    corr = 0.0
    thr = log.get("throttle")
    if thr is not None and len(thr) > 10:
        th = thr.interp_to(m.time)
        if np.std(th) > 1e-6 and np.std(norm) > 1e-12:
            corr = float(np.corrcoef(th, norm)[0, 1])
    return {
        "field_norm_median": round(base, 4),
        "variation_fraction": round(variation, 4),
        "throttle_correlation": round(corr, 3),
    }


def detect_ekf_resets(log: FlightLog) -> List[Event]:
    """Collect EKF reset events, from the event stream or a reset counter.

    A reset means the estimator gave up on continuity and jumped its state.
    Every reset is a discontinuity the position controller has to absorb, and
    a burst of resets is how a flyaway starts.
    """
    events = list(log.events_of("ekf_reset"))
    counter = log.get("ekf.reset_count")
    if counter is not None and len(counter) > 1:
        d = np.diff(counter.values)
        for i in np.flatnonzero(d > 0):
            t_ev = float(counter.time[i + 1])
            if not any(abs(e.time - t_ev) < 0.5 for e in events):
                events.append(
                    Event(t_ev, "ekf_reset", f"reset counter incremented by {int(d[i])}")
                )
    events.sort(key=lambda e: e.time)
    return events


def analyze(
    log: FlightLog,
    ratio_warn: float = RATIO_WARN,
    ratio_critical: float = RATIO_CRITICAL,
    gps_jump_m: float = 5.0,
    height_drift_mps: float = 0.05,
    mag_variation: float = 0.15,
) -> List[Finding]:
    """Run the estimator-health analyzer."""
    findings: List[Finding] = []

    # -- test ratios -----------------------------------------------------
    excursions = ratio_excursions(log, ratio_warn)
    for key, spans in excursions.items():
        peak = max(s[2] for s in spans)
        total = sum(e - s for s, e, _ in spans)
        label, cause = _RATIO_MEANING.get(key, (f"{key} innovations", ""))
        critical = peak >= ratio_critical
        findings.append(
            Finding(
                analyzer="ekf",
                severity=Severity.CRITICAL if critical else Severity.WARNING,
                title=(
                    f"EKF {label} {'rejected' if critical else 'stressed'} "
                    f"(peak ratio {peak:.2f})"
                ),
                explanation=(
                    f"The {key} test ratio peaked at {peak:.2f} and spent {total:.1f} s "
                    f"above {ratio_warn:.1f}, across {len(spans)} excursion(s). "
                    + (
                        "Above 1.0 the estimator rejects the measurement outright, so "
                        "the aircraft is dead-reckoning until the ratio comes back down. "
                        if critical
                        else "Below 1.0 the measurement is still fused, but the margin is "
                        "small and shrinking. "
                    )
                    + cause
                ),
                action=_RATIO_ACTION.get(key, "Investigate the sensor feeding this residual."),
                evidence={
                    "channel": f"ekf.test_ratio.{key}",
                    "peak_ratio": round(peak, 3),
                    "seconds_above_threshold": round(total, 2),
                    "excursions": [
                        {"t_start": round(s, 2), "t_end": round(e, 2), "peak": round(p, 3)}
                        for s, e, p in spans[:5]
                    ],
                    "warn_threshold": ratio_warn,
                    "reject_threshold": ratio_critical,
                },
                t_start=spans[0][0],
                t_end=spans[-1][1],
                plot={"kind": "series", "channels": [f"ekf.test_ratio.{key}"]},
            )
        )

    # -- GPS glitches ----------------------------------------------------
    for g in detect_gps_glitches(log, gps_jump_m):
        findings.append(
            Finding(
                analyzer="ekf",
                severity=Severity.CRITICAL,
                title=f"GPS position glitch at t={float(g['time']):.1f}s ({g['jump_m']} m jump)",
                explanation=(
                    f"GPS position moved {g['jump_m']} m between consecutive fixes, an "
                    f"implied ground speed of {g['implied_speed_mps']} m/s. No multirotor "
                    "accelerates like that, so the receiver jumped rather than the "
                    "aircraft. "
                    + (
                        f"The fix jumped {int(g['jump_count'])} times over "
                        f"{g['duration_s']} s -- once away from the true position and once "
                        "back, which is the shape of a single receiver glitch. "
                        if int(g.get("jump_count", 1)) > 1
                        else ""
                    )
                    + "In a GPS-referenced mode the position controller chases "
                    "the jump: the aircraft lurches, then lurches back when the fix "
                    "recovers. This is the signature behind most 'it suddenly darted "
                    "sideways' reports."
                ),
                action=(
                    "Check the GPS antenna's sky view and its distance from the video "
                    "transmitter and any switching regulator -- both radiate in the L1 "
                    "band. If the glitch repeats at the same place, it is multipath from "
                    "a building or tree line; fly elsewhere to confirm. Raising "
                    "EKF2_GPS_P_NOISE / setting a tighter GPS accuracy gate makes the "
                    "estimator reject these instead of following them."
                ),
                evidence=dict(g),
                t_start=float(g["time"]) - 1.0,
                t_end=float(g.get("t_end", g["time"])) + 1.0,
                plot={"kind": "series", "channels": ["gps.satellites", "gps.hdop"]},
            )
        )

    # -- height source disagreement --------------------------------------
    hgt = height_source_disagreement(log)
    if hgt and abs(float(hgt["drift_mps"])) >= height_drift_mps:
        drift = float(hgt["drift_mps"])
        findings.append(
            Finding(
                analyzer="ekf",
                severity=Severity.WARNING
                if abs(drift) < 4 * height_drift_mps
                else Severity.CRITICAL,
                title=(
                    f"Height sources disagree: baro vs {hgt['compared_to']} drifting "
                    f"{drift:+.2f} m/s"
                ),
                explanation=(
                    f"After removing the constant datum offset, the barometer and the "
                    f"{hgt['compared_to']} altitude diverged by "
                    f"{hgt['total_divergence_m']} m over the flight "
                    f"({drift:+.3f} m/s). The two sensors cannot both be right. In "
                    "altitude-hold the aircraft flies whichever one the estimator "
                    "trusts, so a drifting baro becomes a real, slow altitude error the "
                    "pilot has to keep correcting."
                ),
                action=(
                    "Cover the barometer with open-cell foam and make sure the airframe "
                    "shell does not channel prop wash over it. If the drift is thermal "
                    "(worst in the first two minutes after power-up), let the FC warm up "
                    "before arming. Check the baro is not near a heat source such as a "
                    "regulator or a companion computer."
                ),
                evidence=dict(hgt),
                plot={"kind": "series", "channels": ["alt.baro", "alt.gps", "alt.ekf"]},
            )
        )

    # -- magnetometer ----------------------------------------------------
    mag = magnetometer_consistency(log)
    if mag and float(mag["variation_fraction"]) >= mag_variation:
        corr = float(mag["throttle_correlation"])
        throttle_driven = abs(corr) > 0.5
        findings.append(
            Finding(
                analyzer="ekf",
                severity=Severity.WARNING if not throttle_driven else Severity.CRITICAL,
                title=(
                    "Magnetometer field varies by "
                    f"{float(mag['variation_fraction'])*100:.0f}% during flight"
                    + (" and tracks throttle" if throttle_driven else "")
                ),
                explanation=(
                    "The earth's magnetic field has a constant magnitude, so any change "
                    "in the measured field norm is distortion. "
                    + (
                        f"The variation correlates with throttle (r={corr:+.2f}), which "
                        "means current flowing in a power lead near the compass is bending "
                        "the field. Heading error then grows with throttle: the aircraft's "
                        "idea of north shifts as it works harder. That is the mechanism "
                        "behind slow yaw rotation in position hold and toilet-bowling."
                        if throttle_driven
                        else "The variation does not track throttle, which points at a stale "
                        "calibration or a ferrous object near the compass rather than at "
                        "current-induced distortion."
                    )
                ),
                action=(
                    "Move the compass away from the power distribution board and the "
                    "battery leads -- a GPS/compass module on a mast is the standard fix. "
                    "Then re-run the compass calibration outdoors, away from rebar."
                    if throttle_driven
                    else "Re-run the compass calibration outdoors away from steel and "
                    "rebar, and check for magnets or ferrous hardware near the compass "
                    "(speaker magnets and steel screws are common offenders)."
                ),
                evidence=dict(mag),
                plot={"kind": "series", "channels": ["mag.x", "mag.y", "mag.z"]},
            )
        )

    # -- resets ----------------------------------------------------------
    resets = detect_ekf_resets(log)
    if resets:
        findings.append(
            Finding(
                analyzer="ekf",
                severity=Severity.CRITICAL if len(resets) > 2 else Severity.WARNING,
                title=f"{len(resets)} EKF reset event(s)",
                explanation=(
                    "The estimator discarded state continuity and jumped. Each reset is "
                    "a step change the position controller has to absorb, visible to the "
                    "pilot as a twitch or a lurch. Repeated resets mean the filter never "
                    "reaches agreement with its sensors -- look at whichever test ratio "
                    "was high at the same timestamps."
                ),
                action=(
                    "Correlate each reset timestamp with the vibration and GPS sections "
                    "of this report and fix the underlying sensor problem. Resets are "
                    "never the root cause; they are the estimator's response to one."
                ),
                evidence={
                    "count": len(resets),
                    "times": [round(e.time, 2) for e in resets[:10]],
                    "details": [e.detail for e in resets[:10]],
                },
                t_start=resets[0].time,
                t_end=resets[-1].time,
            )
        )

    if not findings and log.first_present(
        "ekf.test_ratio.vel", "ekf.test_ratio.pos", "ekf.test_ratio.hgt"
    ):
        peaks = {
            k: round(float(np.nanmax(log.series[f"ekf.test_ratio.{k}"].values)), 3)
            for k in ("vel", "pos", "hgt", "mag")
            if log.get(f"ekf.test_ratio.{k}") is not None
        }
        findings.append(
            Finding(
                analyzer="ekf",
                severity=Severity.INFO,
                title="Estimator healthy: all innovation test ratios within limits",
                explanation=(
                    "No test ratio exceeded the 0.5 advisory level, no GPS glitch was "
                    "detected, and no EKF resets occurred. The estimator agreed with its "
                    "sensors for the whole flight."
                ),
                action="No action needed.",
                evidence={"peak_test_ratios": peaks, "warn_threshold": ratio_warn},
            )
        )
    return findings
