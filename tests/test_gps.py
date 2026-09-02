"""GNSS analyzer: fix timeline, satellites, HDOP, time to first fix."""

from __future__ import annotations

import numpy as np
import pytest

from flightlog.analysis.gps import FIX_NAMES, analyze, fix_timeline, time_to_first_fix
from flightlog.types import FlightLog, Severity


def test_fix_timeline_splits_on_every_change(clean_log):
    tl = fix_timeline(clean_log)
    assert len(tl) >= 2
    assert tl[0]["fix_name"] == FIX_NAMES[1]
    assert tl[-1]["fix_name"] == "3D_FIX"
    assert tl[0]["t_end"] == tl[1]["t_start"]


def test_time_to_first_fix_matches_the_generated_acquisition(clean_log):
    ttff = time_to_first_fix(clean_log)
    assert ttff == pytest.approx(2.0, abs=0.5)


def test_time_to_first_fix_is_none_when_no_fix_is_ever_reached():
    log = FlightLog()
    t = np.linspace(0, 30, 151)
    log.add("gps.fix_type", t, np.ones(t.size))
    assert time_to_first_fix(log) is None
    findings = analyze(log)
    assert any("No 3D fix" in f.title for f in findings)
    assert findings[0].severity is Severity.CRITICAL


def test_satellite_drop_during_a_glitch_is_flagged(glitch_log):
    findings = analyze(glitch_log)
    sats = [f for f in findings if "satellite" in f.title.lower()]
    assert sats
    assert sats[0].evidence["min_satellites"] <= 5
    assert float(sats[0].evidence["t_min"]) == pytest.approx(25.0, abs=1.0)


def test_hdop_spike_is_flagged_with_its_timestamp(glitch_log):
    findings = analyze(glitch_log)
    hdop = [f for f in findings if "HDOP" in f.title]
    assert hdop
    assert hdop[0].evidence["max_hdop"] > 2.0
    assert float(hdop[0].evidence["t_max"]) == pytest.approx(25.0, abs=1.0)


def test_fix_degradation_in_flight_is_critical(glitch_log):
    findings = analyze(glitch_log)
    fix = [f for f in findings if "fix degraded" in f.title.lower()]
    assert fix
    assert fix[0].severity is Severity.CRITICAL
    assert 0 < fix[0].evidence["fraction_below_3d"] < 1


def test_clean_flight_gnss_is_reported_good(clean_log):
    findings = analyze(clean_log)
    assert len(findings) == 1
    assert findings[0].severity is Severity.INFO
    assert "good" in findings[0].title.lower()
    assert findings[0].evidence["min_satellites"] >= 8


def test_every_gps_finding_has_an_actionable_recommendation(glitch_log):
    for f in analyze(glitch_log):
        assert len(f.action) > 40
        assert f.explanation


def test_analyzer_returns_nothing_without_gnss_channels():
    log = FlightLog()
    log.add("accel.x", [0, 1, 2], [0.0, 0.1, 0.2])
    assert analyze(log) == []
