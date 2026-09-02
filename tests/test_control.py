"""Control analyzer: tracking, oscillation frequency, saturation, windup."""

from __future__ import annotations

import numpy as np
import pytest

from flightlog.analysis.control import (
    analyze,
    classify_oscillation,
    detect_oscillation,
    integrator_saturation,
    motor_saturation,
    tracking_error,
)
from flightlog.readers.synthetic import DefectSpec, generate
from flightlog.types import FlightLog, Severity


def test_roll_oscillation_dominant_frequency_is_identified(oscillation_log):
    """6.5 Hz was injected into roll; the analyzer must find 6.5 Hz."""
    err = tracking_error(oscillation_log, "roll")
    assert err is not None
    osc = detect_oscillation(err)
    assert osc is not None
    assert osc["freq_hz"] == pytest.approx(6.5, abs=0.5)
    assert osc["peakiness"] > 8.0


def test_oscillation_finding_names_the_gain_to_change(oscillation_log):
    findings = analyze(oscillation_log)
    osc = [f for f in findings if "oscillation" in f.title.lower()]
    assert osc
    f = osc[0]
    assert f.evidence["freq_hz"] == pytest.approx(6.5, abs=0.5)
    assert f.evidence["classification"] == "attitude_p_high"
    assert "MC_ROLL_P" in f.action or "ATC_ANG_RLL_P" in f.action


def test_oscillation_band_classification_matches_the_documented_table():
    assert classify_oscillation(1.0)[0] == "slow_wallow"
    assert classify_oscillation(5.0)[0] == "attitude_p_high"
    assert classify_oscillation(14.0)[0] == "rate_p_high"
    assert classify_oscillation(35.0)[0] == "rate_d_high"
    assert classify_oscillation(120.0)[0] == "filter_problem"


def test_high_frequency_oscillation_maps_to_rate_d():
    log = generate(defects=DefectSpec(roll_oscillation_hz=28.0, roll_oscillation_deg=3.0))
    osc = [f for f in analyze(log) if "oscillation" in f.title.lower()]
    assert osc
    assert osc[0].evidence["classification"] == "rate_d_high"
    assert "MC_ROLLRATE_D" in osc[0].action or "ATC_RAT_RLL_D" in osc[0].action


def test_clean_flight_shows_no_oscillation(clean_log):
    err = tracking_error(clean_log, "roll")
    osc = detect_oscillation(err)
    if osc is not None:
        assert np.rad2deg(osc["amplitude"]) < 1.0


def test_yaw_tracking_error_unwraps_across_pi():
    """Without unwrapping, a heading crossing +/-pi looks like 360 degrees of
    tracking error and swamps every real finding."""
    log = FlightLog()
    t = np.linspace(0, 20, 2001)
    yaw = np.arctan2(np.sin(0.5 * t), np.cos(0.5 * t))  # wrapped ramp
    log.add("att.yaw", t, yaw)
    log.add("att.yaw_sp", t, yaw)
    err = tracking_error(log, "yaw")
    assert err is not None
    assert float(np.max(np.abs(err.values))) < 0.05


def test_one_motor_pinned_is_reported_as_a_desync_signature(desync_log):
    sat = motor_saturation(desync_log)
    assert sat is not None
    assert sat["pattern"] == "one_pinned"
    assert sat["pinned_motors"] == ["motor.1"]
    assert float(sat["t_start"]) == pytest.approx(30.0, abs=1.0)
    findings = [f for f in analyze(desync_log) if "desync" in f.title.lower()]
    assert findings
    assert findings[0].severity is Severity.CRITICAL
    assert "ESC" in findings[0].explanation


def test_clean_flight_has_no_motor_saturation(clean_log):
    assert motor_saturation(clean_log) is None


def test_integrator_windup_window_is_located():
    log = generate(defects=DefectSpec(integrator_windup_at=25.0))
    windows = integrator_saturation(log)
    assert windows
    roll = [w for w in windows if w["axis"] == "roll"]
    assert roll
    assert float(roll[0]["t_start"]) == pytest.approx(25.0, abs=1.0)
    findings = [f for f in analyze(log) if "integrator saturated" in f.title.lower()]
    assert findings
    assert findings[0].severity is Severity.WARNING


def test_clean_flight_has_no_integrator_saturation(clean_log):
    assert integrator_saturation(clean_log) == []


def test_clean_flight_control_findings_are_all_informational(clean_log):
    findings = analyze(clean_log)
    assert findings
    assert all(f.severity is Severity.INFO for f in findings)
    assert "within limits" in findings[0].title


def test_analyzer_returns_nothing_without_attitude_channels():
    log = FlightLog()
    log.add("bat.voltage", [0, 1, 2], [22.0, 21.9, 21.8])
    assert analyze(log) == []
