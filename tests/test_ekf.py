"""Estimator analyzer: test ratios, GPS glitches, mag, height, resets."""

from __future__ import annotations

import pytest

from flightlog.analysis.ekf import (
    analyze,
    detect_ekf_resets,
    detect_gps_glitches,
    height_source_disagreement,
    magnetometer_consistency,
    ratio_excursions,
)
from flightlog.readers.synthetic import DefectSpec, generate
from flightlog.types import FlightLog, Severity


def test_gps_glitch_is_flagged_at_the_injected_timestamp(glitch_log):
    """The glitch was injected at t = 25.0 s; the detector must say so."""
    glitches = detect_gps_glitches(glitch_log)
    assert len(glitches) == 1, f"expected one merged glitch, got {glitches}"
    assert float(glitches[0]["time"]) == pytest.approx(25.0, abs=1.0)
    assert float(glitches[0]["jump_m"]) > 20.0
    assert float(glitches[0]["implied_speed_mps"]) > 30.0


def test_gps_glitch_finding_carries_the_timestamp_and_an_action(glitch_log):
    findings = analyze(glitch_log)
    glitch = [f for f in findings if "glitch" in f.title.lower()]
    assert glitch
    f = glitch[0]
    assert f.severity is Severity.CRITICAL
    assert f.t_start == pytest.approx(24.0, abs=1.5)
    assert "25.0s" in f.title
    assert "antenna" in f.action.lower()


def test_glitch_and_its_recovery_merge_into_one_event(glitch_log):
    """A glitch produces two jumps -- out and back. One event, not two."""
    glitches = detect_gps_glitches(glitch_log)
    assert glitches[0]["jump_count"] == 2
    assert float(glitches[0]["duration_s"]) == pytest.approx(4.0, abs=1.0)


def test_clean_flight_has_no_gps_glitches(clean_log):
    assert detect_gps_glitches(clean_log) == []


def test_test_ratio_excursion_is_found_with_its_window():
    log = generate(defects=DefectSpec(ekf_variance_at=20.0, ekf_variance_ratio=1.5))
    ex = ratio_excursions(log)
    assert "vel" in ex and "pos" in ex
    start, end, peak = ex["vel"][0]
    assert start == pytest.approx(20.0, abs=0.5)
    assert peak == pytest.approx(1.5, abs=0.05)


def test_ratio_above_one_is_critical_below_one_is_warning():
    high = generate(defects=DefectSpec(ekf_variance_at=20.0, ekf_variance_ratio=1.4))
    low = generate(defects=DefectSpec(ekf_variance_at=20.0, ekf_variance_ratio=0.7))
    hi_findings = [f for f in analyze(high) if "velocity innovations" in f.title]
    lo_findings = [f for f in analyze(low) if "velocity innovations" in f.title]
    assert hi_findings and hi_findings[0].severity is Severity.CRITICAL
    assert lo_findings and lo_findings[0].severity is Severity.WARNING


def test_throttle_correlated_mag_interference_is_identified():
    log = generate(defects=DefectSpec(mag_interference=0.25))
    mag = magnetometer_consistency(log)
    assert mag is not None
    assert mag["variation_fraction"] > 0.15
    assert mag["throttle_correlation"] > 0.5
    findings = [f for f in analyze(log) if "Magnetometer" in f.title]
    assert findings
    assert "tracks throttle" in findings[0].title
    assert "toilet-bowling" in findings[0].explanation


def test_clean_flight_magnetometer_is_consistent(clean_log):
    mag = magnetometer_consistency(clean_log)
    assert mag is not None
    assert mag["variation_fraction"] < 0.15


def test_height_source_drift_is_measured():
    log = generate(defects=DefectSpec(baro_drift_mps=0.15))
    hgt = height_source_disagreement(log)
    assert hgt is not None
    assert abs(float(hgt["drift_mps"])) == pytest.approx(0.15, rel=0.35)
    findings = [f for f in analyze(log) if "Height sources disagree" in f.title]
    assert findings
    assert "barometer" in findings[0].action.lower() or "baro" in findings[0].action.lower()


def test_ekf_reset_events_are_collected():
    log = generate(defects=DefectSpec(ekf_reset_at=18.0))
    resets = detect_ekf_resets(log)
    assert resets
    assert resets[0].time == pytest.approx(18.0, abs=0.5)
    findings = [f for f in analyze(log) if "reset" in f.title.lower()]
    assert findings
    assert findings[0].evidence["count"] >= 1


def test_clean_flight_reports_a_healthy_estimator(clean_log):
    findings = analyze(clean_log)
    assert len(findings) == 1
    assert findings[0].severity is Severity.INFO
    assert "healthy" in findings[0].title.lower()
    assert all(v < 0.5 for v in findings[0].evidence["peak_test_ratios"].values())


def test_analyzer_returns_nothing_on_an_empty_log():
    assert analyze(FlightLog()) == []
