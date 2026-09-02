"""Vibration analyzer: the injected-defect tests that prove detection works."""

from __future__ import annotations

import numpy as np
import pytest

from flightlog.analysis import vibration
from flightlog.analysis.vibration import (
    analyze,
    analyze_axes,
    classify_peak,
    estimate_motor_frequency,
    motor_asymmetry,
)
from flightlog.readers.synthetic import DefectSpec, generate
from flightlog.types import Severity


def test_injected_92hz_peak_is_found_near_92hz(vibration_log):
    """The headline claim: inject 92 Hz, find 92 Hz."""
    result = analyze_axes(vibration_log)
    x_axis = next(a for a in result.axes if a.axis == "x")
    assert x_axis.peaks, "no spectral peak detected at all"
    top = x_axis.peaks[0]
    assert top.freq == pytest.approx(92.0, abs=2.0), f"peak reported at {top.freq:.2f} Hz"


def test_injected_peak_amplitude_matches_injection(vibration_log):
    """18 m/s^2 peak amplitude is 12.7 m/s^2 RMS."""
    result = analyze_axes(vibration_log)
    top = next(a for a in result.axes if a.axis == "x").peaks[0]
    assert top.amplitude == pytest.approx(18.0 / np.sqrt(2), rel=0.20)


def test_92hz_peak_is_classified_and_reported(vibration_log):
    findings = analyze(vibration_log)
    peak_findings = [f for f in findings if "Hz" in f.title and "peak" in f.title.lower()]
    assert peak_findings, "no vibration-peak finding produced"
    f = peak_findings[0]
    assert f.evidence["freq_hz"] == pytest.approx(92.0, abs=2.0)
    assert f.evidence["classification"] == "prop_imbalance_likely"
    assert f.severity is Severity.CRITICAL
    assert f.action.strip(), "a finding without a concrete action is useless"


def test_rpm_telemetry_upgrades_classification_to_1x_harmonic(rpm_log):
    """With RPM logged, a peak at the motor frequency is prop imbalance, and
    the analyzer says so with higher confidence than the band heuristic."""
    motor_hz, source = estimate_motor_frequency(rpm_log)
    assert motor_hz == pytest.approx(92.0, rel=0.30)
    assert "RPM" in source
    findings = analyze(rpm_log)
    peak_findings = [f for f in findings if "prop_imbalance" in f.title]
    assert peak_findings
    assert peak_findings[0].evidence["classification"] == "prop_imbalance"
    assert peak_findings[0].confidence >= 0.9


def test_no_rpm_telemetry_is_reported_honestly(vibration_log):
    motor_hz, source = estimate_motor_frequency(vibration_log)
    assert motor_hz is None
    assert "no RPM telemetry" in source
    titles = [f.title for f in analyze(vibration_log)]
    assert any("No RPM telemetry" in t for t in titles)


def test_classify_peak_band_boundaries():
    assert classify_peak(5.0)[0] == "airframe_flex"
    assert classify_peak(22.0)[0] == "soft_mount_resonance"
    assert classify_peak(45.0)[0] == "frame_resonance"
    assert classify_peak(92.0)[0] == "prop_imbalance_likely"
    assert classify_peak(300.0)[0] == "motor_bell_or_bearing"
    assert classify_peak(600.0)[0] == "high_frequency_noise"


def test_classify_peak_with_rpm_identifies_harmonic_order():
    assert classify_peak(100.0, motor_hz=100.0)[0] == "prop_imbalance"
    assert classify_peak(200.0, motor_hz=100.0, n_blades=2)[0] == "blade_pass"
    assert classify_peak(300.0, motor_hz=100.0, n_blades=2)[0] == "motor_harmonic"


def test_every_classification_returns_a_concrete_action():
    for freq in (5.0, 22.0, 45.0, 92.0, 300.0, 600.0):
        key, cause, action = classify_peak(freq)
        assert len(action) > 40, f"{key} action is too vague to act on"
        assert len(cause) > 40


def test_clipping_is_detected_and_critical(vibration_log):
    findings = analyze(vibration_log)
    clip = [f for f in findings if "clipping" in f.title.lower()]
    assert clip, "clip counters were incremented but no finding was raised"
    assert clip[0].evidence["total"] > 0


def test_high_rms_produces_critical_finding():
    log = generate(defects=DefectSpec(vibration_peak_hz=110.0, vibration_peak_amp=48.0))
    findings = analyze(log)
    rms = [f for f in findings if "vibration" in f.title.lower() and "m/s^2" in f.title]
    assert rms
    assert rms[0].severity is Severity.CRITICAL
    assert rms[0].evidence["rms_by_axis_m_s2"]["x"] > vibration.VIBE_RMS_CRITICAL


def test_single_motor_asymmetry_is_detected_and_named():
    log = generate(defects=DefectSpec(motor_asymmetry=0.12))
    asym = motor_asymmetry(log)
    assert asym is not None
    assert asym["worst_motor"] == "motor.0"
    assert asym["pattern"] == "single_motor"
    assert asym["worst_deviation"] > 0.06


def test_clean_flight_has_no_motor_asymmetry(clean_log):
    assert motor_asymmetry(clean_log) is None


def test_clean_flight_produces_no_critical_vibration_findings(clean_log):
    findings = analyze(clean_log)
    assert findings, "the analyzer must still report a clean bill of health"
    assert all(f.severity is Severity.INFO for f in findings)
    assert "within limits" in findings[0].title


def test_analyzer_degrades_gracefully_without_accelerometers(clean_log):
    from flightlog.types import FlightLog

    empty = FlightLog()
    empty.add("bat.voltage", [0, 1, 2], [22.0, 21.9, 21.8])
    assert analyze(empty) == []
