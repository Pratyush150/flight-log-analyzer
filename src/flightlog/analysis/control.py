"""Control-loop analysis: tracking error, oscillation, saturation.

The question this module answers is "is the aircraft doing what it was told,
and if not, which gain is wrong?"

Setpoint versus actual
----------------------
Every attitude controller logs both the commanded and the achieved angle.  The
difference is the tracking error.  A well-tuned multirotor holds attitude to
roughly 1-2 degrees RMS in calm hover.  Larger errors are not automatically bad
-- an aggressive stick input or a gust produces a transient -- so this module
reports RMS over the flight window *and* separates steady bias from
oscillation, because they have completely different causes.

Oscillation frequency tells you which gain
------------------------------------------
This is the single most useful mapping in practical multirotor tuning, and it
falls straight out of a spectrum of the tracking error.  The bands below are
the same ones used by the PID tuning notes in the companion
``drone-control-toolkit`` repository:

===============  =========================================================
Frequency        Most likely cause
===============  =========================================================
0.2 - 2 Hz       Slow wallow. Attitude (angle) P too high for the airframe,
                 or position-controller gains fighting the attitude loop.
                 Also what an over-large rate I term looks like.
2 - 8 Hz         Classic "wobble". Attitude P too high, or rate I too high.
8 - 20 Hz        Fast oscillation. Rate P too high -- the most common
                 over-tune, and the one that eats flight time.
20 - 60 Hz       Rate D too high, or D-term amplifying gyro noise. Motors
                 run hot and the airframe buzzes audibly.
> 60 Hz          Not a gain problem: filtering. Gyro noise is reaching the
                 D term. Lower the gyro cutoff or enable the notch filter.
===============  =========================================================

Why the direction matters: raising a gain that is already too high makes the
oscillation *worse*, and the usual instinct on seeing poor tracking is to raise
P.  Knowing the frequency stops that.

Motor saturation and desync
---------------------------
When a rate controller demands more differential thrust than the motors can
deliver, outputs clip.  The aircraft stops responding to attitude commands even
though the controller is still asking.  One motor pinned at maximum while the
others sit low is the classic desync/failure signature: the mixer is trying to
compensate for a motor that has stopped producing thrust, so it commands the
opposite motor to full.  That pattern -- one motor at 100%, its diagonal
partner at minimum -- is worth interrupting the flight for.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..types import FlightLog, Finding, Series, Severity
from .spectral import dominant_frequency, welch_psd

__all__ = [
    "TRACKING_RMS_WARN_DEG",
    "TRACKING_RMS_CRITICAL_DEG",
    "OSCILLATION_BANDS",
    "tracking_error",
    "detect_oscillation",
    "classify_oscillation",
    "integrator_saturation",
    "motor_saturation",
    "analyze",
]

#: Attitude tracking RMS (degrees) worth commenting on.
TRACKING_RMS_WARN_DEG = 4.0
#: Tracking RMS at which the aircraft is not following commands.
TRACKING_RMS_CRITICAL_DEG = 10.0
#: Oscillation amplitude (degrees peak) below which a spectral peak is noise.
OSC_AMP_WARN_DEG = 1.0
OSC_AMP_CRITICAL_DEG = 3.0

#: ``(f_low, f_high, key, cause, action)`` -- see the module docstring.
OSCILLATION_BANDS: List[Tuple[float, float, str, str, str]] = [
    (
        0.2,
        2.0,
        "slow_wallow",
        "A slow wallow in this band is the attitude (angle) loop fighting the "
        "position loop, or an over-large rate integral term. The aircraft looks like "
        "it is 'breathing' rather than buzzing.",
        "Reduce the attitude/angle P gain by 20% (PX4 MC_ROLL_P / MC_PITCH_P; "
        "ArduPilot ATC_ANG_RLL_P / ATC_ANG_PIT_P) and re-fly. If it persists, reduce "
        "the rate I term next, not P.",
    ),
    (
        2.0,
        8.0,
        "attitude_p_high",
        "Oscillation in the 2-8 Hz band is the classic multirotor wobble: attitude "
        "(angle) P too high for the airframe's inertia, or rate I wound too tight.",
        "Reduce attitude P by 25% (PX4 MC_ROLL_P / MC_PITCH_P; ArduPilot "
        "ATC_ANG_RLL_P / ATC_ANG_PIT_P). If the wobble only appears after a stick "
        "input and then decays slowly, reduce rate I instead.",
    ),
    (
        8.0,
        20.0,
        "rate_p_high",
        "Oscillation in the 8-20 Hz band is rate P set too high. This is the most "
        "common over-tune. It is often inaudible from a distance but shows clearly "
        "as ripple on the gyro trace, and it costs flight time because the motors "
        "are constantly correcting.",
        "Reduce rate P by 25-30% on the affected axis (PX4 MC_ROLLRATE_P / "
        "MC_PITCHRATE_P; ArduPilot ATC_RAT_RLL_P / ATC_RAT_PIT_P) and re-fly. "
        "Do not raise D to damp it -- that moves the problem to a higher frequency.",
    ),
    (
        20.0,
        60.0,
        "rate_d_high",
        "Oscillation above 20 Hz is the rate D term, either set too high or "
        "amplifying gyro noise. D is a differentiator, so it multiplies "
        "high-frequency content. Symptoms on the bench: hot motors after a short "
        "hover and an audible buzz.",
        "Reduce rate D by 30% (PX4 MC_ROLLRATE_D / MC_PITCHRATE_D; ArduPilot "
        "ATC_RAT_RLL_D / ATC_RAT_PIT_D). Then lower the gyro low-pass cutoff "
        "(PX4 IMU_GYRO_CUTOFF; ArduPilot INS_GYRO_FILTER) so D is not fed noise in "
        "the first place.",
    ),
    (
        60.0,
        1000.0,
        "filter_problem",
        "Content this fast is not a tuning problem -- no airframe responds at this "
        "frequency. It is gyro noise passing through the rate loop, usually because "
        "the low-pass cutoff is set too high for the vibration present.",
        "Fix the vibration first, then lower the gyro low-pass cutoff and enable the "
        "dynamic notch filter (PX4 IMU_GYRO_DNF_EN; ArduPilot INS_HNTCH_ENABLE) "
        "so the motor fundamental never reaches the controller.",
    ),
]


def tracking_error(log: FlightLog, axis: str) -> Optional[Series]:
    """Return ``setpoint - actual`` for one axis as a :class:`Series`.

    Supports ``roll``, ``pitch``, ``yaw`` (attitude, radians) and ``alt``
    (altitude, metres).  Yaw is unwrapped before subtraction so a wrap from
    +pi to -pi does not register as a 360-degree tracking error.
    """
    if axis == "alt":
        actual = log.first_present("alt.ekf", "alt.baro")
        setpoint = log.get("alt.sp")
    else:
        actual = log.get(f"att.{axis}")
        setpoint = log.get(f"att.{axis}_sp")
    if actual is None or setpoint is None or len(actual) < 10:
        return None
    a = actual.values
    sp = setpoint.interp_to(actual.time)
    if axis in ("roll", "pitch", "yaw"):
        a = np.unwrap(a)
        sp = np.unwrap(sp)
    err = sp - a
    t0, t1 = log.flight_window()
    s = Series(f"err.{axis}", actual.time, err, actual.units, f"{axis} setpoint - actual")
    sl = s.slice_time(t0, t1)
    return sl if len(sl) >= 10 else s


def detect_oscillation(
    err: Series, fmin: float = 0.3, fmax: float = 100.0
) -> Optional[Dict[str, float]]:
    """Find the dominant oscillation in a tracking-error series.

    Returns frequency, amplitude and a "peakiness" figure: the ratio of the
    peak's power to the median power across the band.  A high ratio means a
    genuine narrowband oscillation; a low one means broadband noise that
    happens to have a maximum somewhere, which is not a tuning problem.
    """
    if len(err) < 64:
        return None
    fmax = min(fmax, 0.45 * err.sample_rate)
    if fmax <= fmin:
        return None
    freq, amp = dominant_frequency(err.time, err.values, fmin, fmax)
    if freq <= 0:
        return None
    spec = welch_psd(err.time, err.values)
    if spec.freq.size == 0:
        return None
    band = (spec.freq >= fmin) & (spec.freq <= fmax)
    if not np.any(band):
        return None
    med = float(np.median(spec.psd[band]))
    peak = float(np.max(spec.psd[band]))
    return {
        "freq_hz": freq,
        "amplitude": amp,
        "peakiness": peak / med if med > 0 else 0.0,
        "resolution_hz": spec.resolution,
        "sample_rate_hz": spec.fs,
    }


def classify_oscillation(freq: float) -> Tuple[str, str, str]:
    """Map an oscillation frequency to ``(key, cause, action)``."""
    for lo, hi, key, cause, action in OSCILLATION_BANDS:
        if lo <= freq < hi:
            return key, cause, action
    return (
        "unclassified",
        f"Oscillation at {freq:.1f} Hz falls outside the bands this analyzer "
        "recognises for a multirotor.",
        "Compare against a known-good log from the same airframe before changing gains.",
    )


def integrator_saturation(
    log: FlightLog, limit: float = 0.9, min_duration: float = 1.0
) -> List[Dict[str, object]]:
    """Windows where a rate-loop integrator sat near its limit.

    A saturated integrator means the controller has run out of authority and is
    accumulating error it cannot correct.  On a multirotor the usual causes are
    a physical trim problem (bent arm, offset CG, a motor that is weaker than
    the rest) or wind the aircraft cannot fight.  Either way the axis has no
    reserve left for a gust.
    """
    out: List[Dict[str, object]] = []
    t0, t1 = log.flight_window()
    for axis in ("roll", "pitch", "yaw"):
        s = log.get(f"pid.{axis}_i")
        if s is None or len(s) < 10:
            continue
        sl = s.slice_time(t0, t1)
        if len(sl) < 10:
            continue
        scale = max(float(np.nanpercentile(np.abs(sl.values), 99)), 1e-9)
        norm = np.abs(sl.values) / max(scale, 1.0)
        sat = norm >= limit
        i = 0
        while i < sat.size:
            if not sat[i]:
                i += 1
                continue
            j = i
            while j + 1 < sat.size and sat[j + 1]:
                j += 1
            dur = float(sl.time[j] - sl.time[i])
            if dur >= min_duration:
                out.append(
                    {
                        "axis": axis,
                        "t_start": round(float(sl.time[i]), 2),
                        "t_end": round(float(sl.time[j]), 2),
                        "duration_s": round(dur, 2),
                        "peak_value": round(float(np.max(np.abs(sl.values[i : j + 1]))), 4),
                    }
                )
            i = j + 1
    return out


def motor_saturation(
    log: FlightLog, high: float = 0.95, low: float = 0.15, min_duration: float = 0.5
) -> Optional[Dict[str, object]]:
    """Detect sustained motor-output saturation and the desync pattern.

    Returns ``None`` when nothing saturates.  When something does, the return
    value distinguishes two very different situations:

    * ``pattern == "all_high"`` -- the aircraft is simply out of thrust
      (overweight, or climbing hard). Not a fault, but a limit.
    * ``pattern == "one_pinned"`` -- one motor at maximum while at least one
      other sits low. The mixer is compensating for a motor that is not
      producing thrust: ESC desync, a de-soldered motor lead, or a stripped
      prop adapter. This is the pattern that precedes a flip.
    """
    motors = [s for s in log.matching("motor.") if len(s) > 10]
    if len(motors) < 3:
        return None
    t0, t1 = log.flight_window()
    grid = motors[0].slice_time(t0, t1).time
    if grid.size < 10:
        return None
    stacked = np.vstack([m.interp_to(grid) for m in motors])
    names = [m.name for m in motors]

    sat_mask = stacked >= high
    any_sat = np.any(sat_mask, axis=0)
    if not np.any(any_sat):
        return None

    # Longest contiguous saturated window.
    best: Tuple[int, int] = (0, -1)
    i = 0
    while i < any_sat.size:
        if not any_sat[i]:
            i += 1
            continue
        j = i
        while j + 1 < any_sat.size and any_sat[j + 1]:
            j += 1
        if j - i > best[1] - best[0]:
            best = (i, j)
        i = j + 1
    i, j = best
    duration = float(grid[j] - grid[i]) if j > i else 0.0
    if duration < min_duration:
        return None

    window = stacked[:, i : j + 1]
    per_motor_sat = np.mean(window >= high, axis=1)
    per_motor_low = np.mean(window <= low, axis=1)
    pinned = [names[k] for k in range(len(names)) if per_motor_sat[k] > 0.6]
    starved = [names[k] for k in range(len(names)) if per_motor_low[k] > 0.6]

    if len(pinned) >= len(names) - 1:
        pattern = "all_high"
    elif len(pinned) == 1 and starved:
        pattern = "one_pinned"
    elif pinned:
        pattern = "partial"
    else:
        pattern = "transient"
    return {
        "pattern": pattern,
        "t_start": round(float(grid[i]), 2),
        "t_end": round(float(grid[j]), 2),
        "duration_s": round(duration, 2),
        "pinned_motors": pinned,
        "starved_motors": starved,
        "saturated_fraction": {
            names[k]: round(float(per_motor_sat[k]), 3) for k in range(len(names))
        },
    }


def analyze(
    log: FlightLog,
    rms_warn_deg: float = TRACKING_RMS_WARN_DEG,
    rms_critical_deg: float = TRACKING_RMS_CRITICAL_DEG,
    osc_amp_warn_deg: float = OSC_AMP_WARN_DEG,
) -> List[Finding]:
    """Run the control-loop analyzer."""
    findings: List[Finding] = []
    rms_summary: Dict[str, float] = {}

    for axis in ("roll", "pitch", "yaw"):
        err = tracking_error(log, axis)
        if err is None:
            continue
        vals = err.values[np.isfinite(err.values)]
        if vals.size < 10:
            continue
        rms_deg = float(np.rad2deg(np.sqrt(np.mean(vals**2))))
        bias_deg = float(np.rad2deg(np.mean(vals)))
        rms_summary[axis] = round(rms_deg, 3)

        if rms_deg >= rms_warn_deg:
            findings.append(
                Finding(
                    analyzer="control",
                    severity=Severity.CRITICAL
                    if rms_deg >= rms_critical_deg
                    else Severity.WARNING,
                    title=f"Poor {axis} tracking: {rms_deg:.1f} deg RMS error",
                    explanation=(
                        f"The {axis} axis missed its setpoint by {rms_deg:.1f} degrees RMS "
                        f"with a steady bias of {bias_deg:+.1f} degrees. "
                        + (
                            "A large steady bias with small variation is a physical trim "
                            "problem -- offset CG, a bent arm, or a weak motor -- not a "
                            "gain problem."
                            if abs(bias_deg) > 0.6 * rms_deg
                            else "The error is mostly dynamic rather than a steady offset, "
                            "so look at the oscillation findings below before touching trim."
                        )
                    ),
                    action=(
                        "Level the aircraft on a flat surface and re-run the accelerometer "
                        "level calibration, then check CG and arm alignment."
                        if abs(bias_deg) > 0.6 * rms_deg
                        else "Address the oscillation finding for this axis first; tracking "
                        "error usually collapses once the loop stops ringing."
                    ),
                    evidence={
                        "axis": axis,
                        "rms_deg": round(rms_deg, 3),
                        "bias_deg": round(bias_deg, 3),
                        "peak_deg": round(float(np.rad2deg(np.max(np.abs(vals)))), 2),
                        "samples": int(vals.size),
                    },
                    t_start=float(err.time[0]),
                    t_end=float(err.time[-1]),
                    plot={"kind": "series", "channels": [f"att.{axis}", f"att.{axis}_sp"]},
                )
            )

        osc = detect_oscillation(err)
        if osc is None:
            continue
        amp_deg = float(np.rad2deg(osc["amplitude"]))
        if amp_deg < osc_amp_warn_deg or osc["peakiness"] < 8.0:
            continue
        key, cause, action = classify_oscillation(float(osc["freq_hz"]))
        findings.append(
            Finding(
                analyzer="control",
                severity=Severity.CRITICAL
                if amp_deg >= OSC_AMP_CRITICAL_DEG
                else Severity.WARNING,
                title=(
                    f"{axis.capitalize()} oscillation at {float(osc['freq_hz']):.1f} Hz "
                    f"({amp_deg:.1f} deg) - {key.replace('_', ' ')}"
                ),
                explanation=(
                    f"The {axis} tracking error contains a {amp_deg:.1f} degree "
                    f"oscillation at {float(osc['freq_hz']):.1f} Hz, standing "
                    f"{float(osc['peakiness']):.0f}x above the surrounding noise floor. "
                    + cause
                ),
                action=action,
                evidence={
                    "axis": axis,
                    "freq_hz": round(float(osc["freq_hz"]), 2),
                    "amplitude_deg": round(amp_deg, 3),
                    "peakiness": round(float(osc["peakiness"]), 1),
                    "spectrum_resolution_hz": round(float(osc["resolution_hz"]), 3),
                    "classification": key,
                },
                t_start=float(err.time[0]),
                t_end=float(err.time[-1]),
                plot={"kind": "series", "channels": [f"att.{axis}", f"att.{axis}_sp"]},
            )
        )

    # -- altitude tracking ------------------------------------------------
    alt_err = tracking_error(log, "alt")
    if alt_err is not None and len(alt_err) > 10:
        vals = alt_err.values[np.isfinite(alt_err.values)]
        if vals.size > 10:
            rms_m = float(np.sqrt(np.mean(vals**2)))
            rms_summary["alt_m"] = round(rms_m, 3)
            if rms_m > 1.5:
                findings.append(
                    Finding(
                        analyzer="control",
                        severity=Severity.WARNING if rms_m < 4.0 else Severity.CRITICAL,
                        title=f"Altitude tracking error {rms_m:.1f} m RMS",
                        explanation=(
                            f"Commanded and achieved altitude differ by {rms_m:.1f} m RMS. "
                            "On a multirotor this is almost never the altitude controller's "
                            "gains: it is the height *estimate* moving, driven by vibration "
                            "or barometer disturbance. Check the vibration and EKF height "
                            "findings before changing any altitude gain."
                        ),
                        action=(
                            "Fix vibration and shield the barometer first. Only if both are "
                            "clean should you touch the altitude controller gains."
                        ),
                        evidence={
                            "rms_m": round(rms_m, 3),
                            "peak_m": round(float(np.max(np.abs(vals))), 2),
                        },
                        t_start=float(alt_err.time[0]),
                        t_end=float(alt_err.time[-1]),
                        plot={"kind": "series", "channels": ["alt.ekf", "alt.sp"]},
                    )
                )

    # -- integrator saturation -------------------------------------------
    for win in integrator_saturation(log)[:3]:
        findings.append(
            Finding(
                analyzer="control",
                severity=Severity.WARNING,
                title=(
                    f"{str(win['axis']).capitalize()} integrator saturated for "
                    f"{win['duration_s']} s"
                ),
                explanation=(
                    f"The {win['axis']} rate integrator sat at its limit from "
                    f"t={win['t_start']}s to t={win['t_end']}s. A saturated integrator "
                    "means the controller ran out of authority and kept accumulating an "
                    "error it could not correct. Until it unwinds, that axis has no "
                    "reserve left for a gust."
                ),
                action=(
                    "Look for a physical asymmetry first: CG offset, a bent arm, a weak "
                    "motor, or a prop of the wrong pitch. If the airframe is straight, the "
                    "aircraft was fighting wind beyond its authority -- reduce payload or "
                    "fly in calmer conditions."
                ),
                evidence=dict(win),
                t_start=float(win["t_start"]),
                t_end=float(win["t_end"]),
                plot={"kind": "series", "channels": [f"pid.{win['axis']}_i"]},
            )
        )

    # -- motor saturation / desync ----------------------------------------
    sat = motor_saturation(log)
    if sat is not None:
        pattern = str(sat["pattern"])
        if pattern == "one_pinned":
            findings.append(
                Finding(
                    analyzer="control",
                    severity=Severity.CRITICAL,
                    title=(
                        f"Motor desync signature: {sat['pinned_motors'][0]} pinned at full "
                        f"for {sat['duration_s']} s"
                    ),
                    explanation=(
                        f"From t={sat['t_start']}s to t={sat['t_end']}s, "
                        f"{sat['pinned_motors'][0]} was commanded to maximum while "
                        f"{', '.join(str(m) for m in sat['starved_motors'])} sat near "
                        "minimum. The mixer only produces that pattern when it is trying to "
                        "correct an attitude error that will not go away -- which happens "
                        "when a motor stops producing the thrust it is being commanded to "
                        "produce. ESC desync, a broken motor lead, a stripped prop adapter "
                        "or a shed prop all look like this."
                    ),
                    action=(
                        "Do not fly again until you find the mechanical or electrical fault. "
                        "Check the motor and ESC at the pinned position, its solder joints, "
                        "and the prop adapter. Re-flash the ESC with a lower motor timing / "
                        "higher demag compensation if it is a desync-prone ESC."
                    ),
                    evidence=dict(sat),
                    t_start=float(sat["t_start"]),
                    t_end=float(sat["t_end"]),
                    plot={"kind": "motors"},
                )
            )
        elif pattern in ("all_high", "partial"):
            findings.append(
                Finding(
                    analyzer="control",
                    severity=Severity.WARNING,
                    title=f"Motor outputs saturated for {sat['duration_s']} s",
                    explanation=(
                        f"Motor outputs hit their upper limit between t={sat['t_start']}s "
                        f"and t={sat['t_end']}s. While saturated the controller has no "
                        "authority left: attitude commands are simply not executed. On a "
                        "multirotor this means the aircraft is at or beyond its thrust "
                        "limit -- overweight, climbing too hard, or flying on a sagging "
                        "pack."
                    ),
                    action=(
                        "Reduce all-up weight or fit higher-thrust propulsion so hover "
                        "throttle sits near 50%. Check the power findings: a sagging pack "
                        "reduces available thrust exactly when it is needed."
                    ),
                    evidence=dict(sat),
                    t_start=float(sat["t_start"]),
                    t_end=float(sat["t_end"]),
                    plot={"kind": "motors"},
                )
            )

    if not findings and rms_summary:
        findings.append(
            Finding(
                analyzer="control",
                severity=Severity.INFO,
                title="Attitude tracking within limits, no oscillation detected",
                explanation=(
                    "Setpoint tracking stayed inside the advisory limits on every axis "
                    "and no narrowband oscillation stood out above the noise floor. The "
                    "control loop is not the source of any symptom in this log."
                ),
                action="No action needed.",
                evidence={"tracking_rms": rms_summary, "warn_threshold_deg": rms_warn_deg},
            )
        )
    return findings
