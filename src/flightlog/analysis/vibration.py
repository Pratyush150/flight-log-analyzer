"""Vibration analysis: FFT of the accelerometers, and what the peaks mean.

Why this module is first in the report
--------------------------------------
Most "my EKF variance goes red", "altitude climbs on its own", "it twitches in
hover" and "position hold wanders" reports are vibration problems wearing a
different hat.  The mechanism is not mysterious:

1. The IMU is a strapdown accelerometer.  It measures specific force, which
   includes every bit of frame vibration.
2. The estimator integrates accelerometer output to get velocity, and again to
   get position.  Integration is a low-pass operation on the *signal*, but
   vibration is not zero-mean once it aliases.
3. Accelerometers are sampled at a finite rate and pass through an anti-alias
   filter that is never perfect.  A vibration tone above Nyquist folds down to
   a low frequency the estimator cannot distinguish from real motion.  A 92 Hz
   prop tone sampled at 100 Hz *becomes* an 8 Hz acceleration the EKF believes.
4. Worse, if vibration is large enough to clip the accelerometer (typically
   +/-16 g), the clipped waveform is asymmetric.  Its mean is no longer zero,
   so the estimator sees a constant false acceleration.  The classic result is
   an aircraft that slowly climbs in altitude-hold: clipping on the Z axis
   biases measured specific force, the estimator thinks it is descending, and
   the controller adds throttle.

So: fix vibration before touching a single PID gain or EKF parameter.  Tuning
around a mechanical problem never works.

What this analyzer does
-----------------------
* Welch-averaged PSD per accelerometer axis (see :mod:`.spectral`).
* Peak detection against a *local* noise floor.
* Peak classification -- motor/prop harmonic (using RPM if logged, an estimate
  from throttle if not), frame resonance, or loose FC mount.
* Broadband vibration RMS above 5 Hz, compared against field thresholds.
* Accelerometer clipping detection from the cumulative clip counters.
* Per-motor output asymmetry, which is how a damaged prop or bent arm shows up
  even when the vibration itself looks symmetric.

Thresholds
----------
The RMS thresholds match the numbers the ArduPilot and PX4 communities settled
on for ``VIBE.VibeX/Y/Z``: comfortable below 15 m/s^2, worth fixing between 15
and 30, and expect estimator problems above 30.  They are stated as module
constants so a caller can override them for an unusual airframe rather than
having to fork the analyzer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..types import FlightLog, Finding, Severity
from .spectral import Peak, Spectrum, find_peaks, highpass_rms, welch_psd

__all__ = [
    "VIBE_RMS_WARN",
    "VIBE_RMS_CRITICAL",
    "AxisResult",
    "VibrationResult",
    "analyze",
    "analyze_axes",
    "classify_peak",
    "estimate_motor_frequency",
    "motor_asymmetry",
]

#: Broadband vibration RMS (m/s^2, >5 Hz) at which to warn.
VIBE_RMS_WARN = 15.0
#: RMS at which estimator problems become likely rather than possible.
VIBE_RMS_CRITICAL = 30.0
#: Peak amplitude (m/s^2) of a single spectral line that is worth reporting.
PEAK_AMP_WARN = 2.0
#: Peak amplitude at which a single tone alone can destabilise the estimator.
PEAK_AMP_CRITICAL = 6.0
#: High-pass cutoff separating "flight" from "vibration".
HP_CUTOFF_HZ = 5.0


@dataclass
class AxisResult:
    """Per-axis vibration result."""

    axis: str
    rms: float
    spectrum: Spectrum
    peaks: List[Peak]

    def to_dict(self) -> Dict[str, object]:
        return {
            "axis": self.axis,
            "rms_m_s2": round(self.rms, 3),
            "sample_rate_hz": round(self.spectrum.fs, 1),
            "peaks": [p.to_dict() for p in self.peaks],
        }


@dataclass
class VibrationResult:
    """Everything the vibration analyzer computed, for reuse by the report."""

    axes: List[AxisResult]
    motor_hz: Optional[float]
    motor_hz_source: str
    clip_events: Dict[str, float]
    asymmetry: Optional[Dict[str, object]]

    @property
    def worst_rms(self) -> float:
        return max((a.rms for a in self.axes), default=0.0)

    def to_dict(self) -> Dict[str, object]:
        return {
            "axes": [a.to_dict() for a in self.axes],
            "motor_hz": round(self.motor_hz, 2) if self.motor_hz else None,
            "motor_hz_source": self.motor_hz_source,
            "clip_events": self.clip_events,
            "asymmetry": self.asymmetry,
        }


# ---------------------------------------------------------------------------
# peak classification
# ---------------------------------------------------------------------------


def classify_peak(
    freq: float,
    bandwidth: float = 0.0,
    motor_hz: Optional[float] = None,
    n_blades: int = 2,
) -> Tuple[str, str, str]:
    """Map a peak frequency to a likely mechanical source.

    Returns ``(source_key, plain_english_cause, recommended_action)``.

    With RPM telemetry the classification is close to certain: prop imbalance
    puts energy at exactly 1x motor rotation frequency, and blade-pass effects
    land at ``n_blades`` x that.  A peak at 2x on a two-blade prop is
    aerodynamic (blade passing the arm) and is normal at low amplitude; a peak
    at 1x is mass imbalance and is never normal above a couple of m/s^2.

    Without RPM the classification falls back to frequency bands, which are
    genuinely useful but genuinely less certain -- the report says so rather
    than pretending otherwise.
    """
    if motor_hz and motor_hz > 0:
        for n in (1, 2, 3, 4):
            target = motor_hz * n
            tol = max(2.0, 0.08 * target)
            if abs(freq - target) <= tol:
                if n == 1:
                    return (
                        "prop_imbalance",
                        f"{freq:.0f} Hz is 1x motor rotation ({motor_hz:.0f} Hz). "
                        "Energy at exactly one cycle per revolution is mass imbalance: "
                        "a chipped, bent, water-logged or mismatched prop, or a bent "
                        "motor shaft.",
                        "Swap props one at a time and re-fly a 30-second hover after "
                        "each swap; the peak collapses when you replace the bad one. "
                        "If no prop fixes it, spin each motor bare and check for shaft "
                        "runout.",
                    )
                if n == n_blades:
                    return (
                        "blade_pass",
                        f"{freq:.0f} Hz is blade-pass frequency ({n_blades} blades x "
                        f"{motor_hz:.0f} Hz). Some energy here is normal -- it is each "
                        "blade passing the arm. At this amplitude it is not.",
                        "Check prop-to-arm clearance and arm stiffness. Raising the "
                        "motors on taller standoffs, or stiffening the arms, moves this "
                        "peak out of the flight-control band.",
                    )
                return (
                    "motor_harmonic",
                    f"{freq:.0f} Hz is the {n}x harmonic of motor rotation "
                    f"({motor_hz:.0f} Hz). Harmonics grow when something in the "
                    "rotating assembly is worn -- typically a dry or notchy bearing.",
                    "Spin each motor by hand with the prop off. A rough or gritty one "
                    "is the source; replace the bearings or the motor.",
                )

    if freq < 10.0:
        return (
            "airframe_flex",
            f"{freq:.0f} Hz is below the range of anything rotating. Low-frequency "
            "energy this large is the airframe itself flexing, a payload or battery "
            "swinging on soft straps, or the control loop exciting a structural mode.",
            "Check that the battery is strapped tight and nothing is dangling. If the "
            "frame itself is flexing, add arm bracing. If it started after a tune, "
            "lower rate P before touching the frame.",
        )
    if freq < 30.0:
        return (
            "soft_mount_resonance",
            f"{freq:.0f} Hz is the classic anti-vibration-mount resonance band. A "
            "gel/o-ring mount that is too soft for the flight controller's mass will "
            "amplify -- not attenuate -- energy here.",
            "Stiffen the FC mount: fewer or harder gel pads, or a thinner double-sided "
            "tape stack. Counterintuitively, going softer makes this peak worse.",
        )
    if freq < 60.0:
        return (
            "frame_resonance",
            f"{freq:.0f} Hz is in the frame-resonance band for a typical multirotor "
            "arm. It can also be the motor fundamental on a large, low-KV setup "
            "(15-inch props and up).",
            "Log RPM (ESC telemetry or a bidirectional-DShot capable ESC) to separate "
            "these two cases -- it is a 10-minute change that removes the ambiguity. "
            "Meanwhile, check arm bolts and motor-mount screws for looseness.",
        )
    if freq < 200.0:
        return (
            "prop_imbalance_likely",
            f"{freq:.0f} Hz is the usual motor-fundamental band for 5-10 inch props at "
            "hover throttle. A sharp peak here is almost always prop imbalance or a "
            "damaged blade.",
            "Balance or replace the props. Cheapest test: swap all four props for a "
            "known-good set and re-fly a hover -- if the peak disappears, it was the "
            "props.",
        )
    if freq < 400.0:
        return (
            "motor_bell_or_bearing",
            f"{freq:.0f} Hz is above the prop fundamental for most airframes. Peaks up "
            "here come from the motor bell, worn bearings, or blade-pass harmonics.",
            "Check each motor for bearing roughness and bell wobble. Also confirm the "
            "gyro/accel low-pass filters are not set higher than this frequency, or "
            "this noise is reaching the rate controller.",
        )
    return (
        "high_frequency_noise",
        f"{freq:.0f} Hz is high-frequency mechanical or electrical noise. It rarely "
        "affects the estimator directly, but it does reach the rate D term and heats "
        "the motors.",
        "Lower the gyro low-pass cutoff (PX4 IMU_GYRO_CUTOFF, ArduPilot INS_GYRO_FILTER) "
        "or enable the dynamic notch filter, so this energy never reaches the D term.",
    )


def estimate_motor_frequency(log: FlightLog) -> Tuple[Optional[float], str]:
    """Best available estimate of motor rotation frequency in Hz.

    Preference order, most to least trustworthy:

    1. Logged RPM telemetry (``rpm.*``), converted to Hz by the reader.
    2. A ``motor_hz_hint`` in the log metadata (set by the synthetic generator
       or supplied by the caller who knows their KV and pack voltage).
    3. Nothing.  Returning ``None`` is a legitimate answer, and the report says
       "no RPM telemetry" rather than inventing a number from throttle.

    Deriving RPM from throttle alone would need KV, pack voltage under load,
    prop load constant and ESC timing -- four unknowns to guess one number.
    That is exactly the kind of fabricated precision that makes a diagnostic
    report worthless, so this function refuses to do it.
    """
    rpms = log.matching("rpm.")
    if rpms:
        medians = [float(np.nanmedian(s.values)) for s in rpms if len(s)]
        medians = [m for m in medians if np.isfinite(m) and m > 0]
        if medians:
            return float(np.median(medians)), "ESC RPM telemetry"
    hint = log.metadata.get("motor_hz_hint")
    if hint:
        return float(hint), "caller-supplied motor frequency"
    return None, "unavailable (no RPM telemetry logged)"


def motor_asymmetry(log: FlightLog, threshold: float = 0.06) -> Optional[Dict[str, object]]:
    """Detect one motor working harder than the rest.

    A level multirotor in hover should command near-identical output on all
    motors.  Persistent asymmetry has a short list of causes, and the *pattern*
    tells you which:

    * one motor high  -> that motor is producing less thrust for the same
      command: damaged/bent prop, weak motor, dragging bearing, or a bent arm
      changing the thrust vector.
    * one diagonal pair high -> centre of gravity is offset toward the other
      pair. Move the battery.
    * all four ramping up over the flight -> the aircraft is getting heavier
      (it is not) or the pack is sagging (it is). See the power analyzer.

    Yaw authority note: on a quad, motors 0/1 spin one way and 2/3 the other,
    so a *diagonal pair* biased high also shows up when the frame has a
    persistent yaw torque -- a twisted arm or a motor mounted at an angle.
    """
    motors = [s for s in log.matching("motor.") if len(s) > 10]
    if len(motors) < 4:
        return None
    t0, t1 = log.flight_window()
    means: Dict[str, float] = {}
    for s in motors:
        sl = s.slice_time(t0, t1)
        v = sl.values[np.isfinite(sl.values)]
        v = v[v > 0.05]  # ignore disarmed samples
        if v.size < 10:
            return None
        means[s.name] = float(np.mean(v))
    if not means:
        return None
    overall = float(np.mean(list(means.values())))
    if overall <= 0:
        return None
    devs = {k: (v - overall) / overall for k, v in means.items()}
    worst = max(devs, key=lambda k: abs(devs[k]))
    spread = max(means.values()) - min(means.values())
    high = [k for k, d in devs.items() if d > threshold]
    if abs(devs[worst]) < threshold:
        return None
    if len(high) == 1:
        pattern = "single_motor"
    elif len(high) == 2:
        pattern = "diagonal_pair"
    else:
        pattern = "multiple_motors"
    return {
        "means": {k: round(v, 4) for k, v in means.items()},
        "deviation": {k: round(v, 4) for k, v in devs.items()},
        "worst_motor": worst,
        "worst_deviation": round(devs[worst], 4),
        "spread": round(spread, 4),
        "pattern": pattern,
    }


# ---------------------------------------------------------------------------
# main analysis
# ---------------------------------------------------------------------------


def analyze_axes(
    log: FlightLog,
    axes: Tuple[str, ...] = ("x", "y", "z"),
    max_peaks: int = 4,
) -> VibrationResult:
    """Compute spectra, peaks, RMS, clipping and asymmetry (no findings yet).

    Split out from :func:`analyze` so the HTML report can plot the spectra
    without recomputing them, and so tests can assert on raw numbers.
    """
    t0, t1 = log.flight_window()
    results: List[AxisResult] = []
    for axis in axes:
        s = log.get(f"accel.{axis}")
        if s is None or len(s) < 64:
            continue
        sl = s.slice_time(t0, t1)
        if len(sl) < 64:
            sl = s
        spec = welch_psd(sl.time, sl.values)
        peaks = find_peaks(spec, fmin=4.0, max_peaks=max_peaks)
        rms = highpass_rms(sl.time, sl.values, HP_CUTOFF_HZ)
        results.append(AxisResult(axis, rms, spec, peaks))

    motor_hz, motor_src = estimate_motor_frequency(log)
    clips: Dict[str, float] = {}
    for i in range(3):
        s = log.get(f"vibe.clip{i}")
        if s is not None and len(s) > 1:
            v = s.values[np.isfinite(s.values)]
            if v.size:
                clips[f"clip{i}"] = float(np.nanmax(v) - np.nanmin(v))
    return VibrationResult(results, motor_hz, motor_src, clips, motor_asymmetry(log))


def analyze(
    log: FlightLog,
    rms_warn: float = VIBE_RMS_WARN,
    rms_critical: float = VIBE_RMS_CRITICAL,
    peak_amp_warn: float = PEAK_AMP_WARN,
    peak_amp_critical: float = PEAK_AMP_CRITICAL,
) -> List[Finding]:
    """Run the vibration analyzer and return ranked findings."""
    result = analyze_axes(log)
    findings: List[Finding] = []
    if not result.axes:
        return findings

    t0, t1 = log.flight_window()
    worst_axis = max(result.axes, key=lambda a: a.rms)

    # -- broadband RMS ---------------------------------------------------
    rms_map = {a.axis: round(a.rms, 2) for a in result.axes}
    if worst_axis.rms >= rms_critical:
        findings.append(
            Finding(
                analyzer="vibration",
                severity=Severity.CRITICAL,
                title=f"Severe broadband vibration ({worst_axis.rms:.1f} m/s^2 on {worst_axis.axis})",
                explanation=(
                    f"High-pass (>{HP_CUTOFF_HZ:.0f} Hz) accelerometer RMS reached "
                    f"{worst_axis.rms:.1f} m/s^2, above the {rms_critical:.0f} m/s^2 level "
                    "where the estimator stops coping. At this level the position and "
                    "velocity estimates are being driven by frame noise rather than by "
                    "flight, which is what produces EKF variance warnings, altitude "
                    "creep and position-hold drift. Any control tuning done on this "
                    "airframe is tuning around a mechanical fault."
                ),
                action=(
                    "Ground the aircraft. Balance or replace all props, check every arm "
                    "and motor bolt, and confirm the flight controller is mounted on a "
                    "stiff, thin pad rather than thick foam. Re-fly a 30 s hover and "
                    "confirm RMS is under 15 m/s^2 before flying a mission or touching "
                    "any PID gain."
                ),
                evidence={"rms_by_axis_m_s2": rms_map, "threshold_m_s2": rms_critical},
                t_start=t0,
                t_end=t1,
                plot={"kind": "spectrum", "axis": worst_axis.axis},
            )
        )
    elif worst_axis.rms >= rms_warn:
        findings.append(
            Finding(
                analyzer="vibration",
                severity=Severity.WARNING,
                title=f"Elevated vibration ({worst_axis.rms:.1f} m/s^2 on {worst_axis.axis})",
                explanation=(
                    f"High-pass accelerometer RMS is {worst_axis.rms:.1f} m/s^2, in the "
                    f"{rms_warn:.0f}-{rms_critical:.0f} m/s^2 band. The aircraft will fly, "
                    "but the estimator has less margin than it should, and the problem "
                    "gets worse as props wear and mounts age."
                ),
                action=(
                    "Balance the props and re-check the FC mount before the next tuning "
                    "session. Re-measure after: the target is under 15 m/s^2."
                ),
                evidence={"rms_by_axis_m_s2": rms_map, "threshold_m_s2": rms_warn},
                t_start=t0,
                t_end=t1,
                plot={"kind": "spectrum", "axis": worst_axis.axis},
            )
        )
    else:
        findings.append(
            Finding(
                analyzer="vibration",
                severity=Severity.INFO,
                title=f"Vibration within limits (peak {worst_axis.rms:.1f} m/s^2)",
                explanation=(
                    "High-pass accelerometer RMS stayed below the 15 m/s^2 advisory "
                    "level on every axis. Vibration is not the cause of any other "
                    "symptom in this log."
                ),
                action="No action needed. Re-check after any prop, motor or mount change.",
                evidence={"rms_by_axis_m_s2": rms_map},
                t_start=t0,
                t_end=t1,
            )
        )

    # -- discrete peaks --------------------------------------------------
    seen: List[float] = []
    for axis_result in sorted(result.axes, key=lambda a: -a.rms):
        for peak in axis_result.peaks:
            if peak.amplitude < peak_amp_warn:
                continue
            if any(abs(peak.freq - f) < max(2.0, 0.05 * peak.freq) for f in seen):
                continue
            seen.append(peak.freq)
            key, cause, action = classify_peak(
                peak.freq, peak.bandwidth, result.motor_hz
            )
            severity = (
                Severity.CRITICAL if peak.amplitude >= peak_amp_critical else Severity.WARNING
            )
            width_note = (
                " The peak is broad, which points at a structural resonance rather "
                "than a single rotating part."
                if peak.bandwidth > 15.0
                else " The peak is narrow, consistent with one discrete rotating source."
            )
            findings.append(
                Finding(
                    analyzer="vibration",
                    severity=severity,
                    title=(
                        f"Vibration peak at {peak.freq:.0f} Hz "
                        f"({peak.amplitude:.1f} m/s^2, accel {axis_result.axis}) - {key}"
                    ),
                    explanation=cause + width_note,
                    action=action,
                    evidence={
                        "freq_hz": round(peak.freq, 2),
                        "amplitude_m_s2": round(peak.amplitude, 3),
                        "prominence_db": round(peak.prominence, 1),
                        "bandwidth_hz": round(peak.bandwidth, 2),
                        "axis": axis_result.axis,
                        "classification": key,
                        "motor_hz": result.motor_hz,
                        "motor_hz_source": result.motor_hz_source,
                        "spectrum_resolution_hz": round(axis_result.spectrum.resolution, 3),
                        "welch_segments": axis_result.spectrum.n_segments,
                    },
                    t_start=t0,
                    t_end=t1,
                    confidence=0.9 if result.motor_hz else 0.6,
                    plot={"kind": "spectrum", "axis": axis_result.axis},
                )
            )
            if len(seen) >= 3:
                break
        if len(seen) >= 3:
            break

    # -- clipping --------------------------------------------------------
    total_clips = sum(result.clip_events.values())
    if total_clips > 0:
        findings.append(
            Finding(
                analyzer="vibration",
                severity=Severity.CRITICAL if total_clips >= 5 else Severity.WARNING,
                title=f"Accelerometer clipping: {int(total_clips)} events",
                explanation=(
                    "The accelerometer hit its measurement range limit and the samples "
                    "were truncated. A clipped waveform is asymmetric, so its mean is no "
                    "longer zero: the estimator receives a constant false acceleration. "
                    "This is the mechanism behind unexplained altitude climb in "
                    "altitude-hold and sudden position-estimate jumps. Clipping means "
                    "the vibration is not merely noisy, it is off the scale of the "
                    "sensor."
                ),
                action=(
                    "Do not fly again until this is fixed. Replace the props, re-check "
                    "every motor for bearing damage, and soften the FC mount only if it "
                    "is currently rigid (hard-mounted). Confirm the clip counters stay "
                    "at zero on the next hover."
                ),
                evidence={"clip_counts": result.clip_events, "total": int(total_clips)},
                t_start=t0,
                t_end=t1,
            )
        )

    # -- per-motor asymmetry ---------------------------------------------
    asym = result.asymmetry
    if asym:
        dev_pct = abs(float(asym["worst_deviation"])) * 100.0
        pattern = str(asym["pattern"])
        if pattern == "single_motor":
            cause = (
                f"{asym['worst_motor']} ran {dev_pct:.0f}% above the average of the other "
                "motors for the whole flight. One motor working harder than its "
                "neighbours means it is producing less thrust per unit of command: a "
                "damaged or bent prop, a weak or dragging motor, or a bent arm."
            )
            action = (
                f"Swap the prop on {asym['worst_motor']} with a known-good one and re-fly "
                "a hover. If the asymmetry follows the prop, it was the prop; if it stays "
                "with the position, check that motor's bearings and the arm for a bend."
            )
        elif pattern == "diagonal_pair":
            cause = (
                f"Two motors ran {dev_pct:.0f}% above average. A diagonal pair working "
                "harder is a centre-of-gravity offset, not a motor fault: the aircraft is "
                "tilting to hold position and the low side has to push harder."
            )
            action = (
                "Move the battery toward the light side until hover motor outputs match "
                "within a few percent. Check payload mounting too."
            )
        else:
            cause = (
                f"Motor outputs are spread by {float(asym['spread'])*100:.0f}% of full "
                "range in hover, which is more than airframe build tolerance explains."
            )
            action = (
                "Check that all props are the same pitch and diameter, all motors are the "
                "same KV, and the frame is not twisted. Verify motor order and spin "
                "direction against the mixer."
            )
        findings.append(
            Finding(
                analyzer="vibration",
                severity=Severity.WARNING if dev_pct < 15 else Severity.CRITICAL,
                title=f"Motor output asymmetry ({pattern.replace('_', ' ')}, {dev_pct:.0f}%)",
                explanation=cause,
                action=action,
                evidence=asym,
                t_start=t0,
                t_end=t1,
                plot={"kind": "motors"},
            )
        )

    if result.motor_hz is None and any(
        f.severity is not Severity.INFO for f in findings
    ):
        findings.append(
            Finding(
                analyzer="vibration",
                severity=Severity.INFO,
                title="No RPM telemetry: peak classification is band-based, not exact",
                explanation=(
                    "This log contains no ESC RPM data, so vibration peaks were "
                    "classified by frequency band rather than matched against actual "
                    "motor rotation frequency. That distinguishes 'prop imbalance' from "
                    "'frame resonance' with much less certainty than an RPM-matched "
                    "analysis would."
                ),
                action=(
                    "Enable bidirectional DShot or ESC telemetry and log RPM "
                    "(PX4: esc_status; ArduPilot: ESC / SERVO_BLH_ options). One "
                    "parameter change turns every future vibration analysis from a "
                    "band guess into a direct harmonic match."
                ),
                evidence={"motor_hz_source": result.motor_hz_source},
                confidence=1.0,
            )
        )
    return findings
