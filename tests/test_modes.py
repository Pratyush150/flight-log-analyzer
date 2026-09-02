"""Mode/arm/failsafe analyzer and its timeline."""

from __future__ import annotations

import numpy as np
import pytest

from flightlog.analysis.modes import analyze, mode_timeline, rc_signal_loss
from flightlog.readers.synthetic import generate
from flightlog.types import Event, FlightLog, Severity


def test_mode_timeline_is_contiguous_and_ordered(clean_log):
    tl = mode_timeline(clean_log)
    assert len(tl) >= 3
    for a, b in zip(tl, tl[1:]):
        assert a["t_end"] <= b["t_start"] + 1e-6
        assert a["duration_s"] > 0


def test_rc_loss_window_is_located(rc_loss_log):
    losses = rc_signal_loss(rc_loss_log)
    assert len(losses) == 1
    assert losses[0]["t_start"] == pytest.approx(20.0, abs=0.5)
    assert losses[0]["duration_s"] == pytest.approx(3.9, abs=0.5)


def test_rc_loss_and_failsafe_both_produce_findings(rc_loss_log):
    findings = analyze(rc_loss_log)
    titles = [f.title for f in findings]
    assert any("RC link lost" in t for t in titles)
    assert any("failsafe" in t.lower() for t in titles)
    rc = next(f for f in findings if "RC link lost" in f.title)
    assert rc.severity is Severity.CRITICAL
    assert "antenna" in rc.action.lower()


def test_rtl_mode_appears_in_the_timeline_after_rc_loss(rc_loss_log):
    modes = [m["mode"] for m in mode_timeline(rc_loss_log)]
    assert "AUTO.RTL" in modes


def test_clean_flight_has_no_rc_dropouts(clean_log):
    assert rc_signal_loss(clean_log) == []


def test_short_flight_is_called_out():
    log = generate(duration=12.0)
    findings = analyze(log)
    short = [f for f in findings if "short armed time" in f.title.lower()]
    assert short
    assert short[0].severity is Severity.WARNING


def test_log_ending_while_armed_is_critical():
    log = FlightLog()
    log.metadata["duration"] = 40.0
    log.add("accel.x", np.linspace(0, 40, 401), np.zeros(401))
    log.events.append(Event(5.0, "arm", "armed"))
    findings = analyze(log)
    still_armed = [f for f in findings if "still armed" in f.title.lower()]
    assert still_armed
    assert still_armed[0].severity is Severity.CRITICAL
    assert "SD card" in still_armed[0].action


def test_missing_arm_events_are_reported_not_guessed():
    log = FlightLog()
    log.add("accel.x", np.linspace(0, 30, 301), np.zeros(301))
    findings = analyze(log)
    assert any("No arm event" in f.title for f in findings)


def test_timeline_finding_is_always_present(clean_log):
    findings = analyze(clean_log)
    assert any("timeline" in f.title.lower() for f in findings)
    assert findings[-1].evidence["total_duration_s"] > 0
