"""Core data-model behaviour: Series, FlightLog, Severity ordering, JSON."""

from __future__ import annotations

import math

import numpy as np
import pytest

from flightlog.types import FlightLog, Finding, ModeInterval, Series, Severity, _jsonify


def test_series_length_mismatch_raises():
    with pytest.raises(ValueError):
        Series("bad", np.arange(5), np.arange(4))


def test_series_sample_rate_uses_median_not_mean():
    """A single logging dropout must not corrupt the reported sample rate.

    100 Hz data with one 2-second gap has a mean interval far above 0.01 s.
    Using the mean would shift every FFT frequency downstream.
    """
    t = np.concatenate([np.arange(0, 1.0, 0.01), np.arange(3.0, 4.0, 0.01)])
    s = Series("x", t, np.zeros(t.size))
    assert s.sample_rate == pytest.approx(100.0, rel=0.01)
    gaps = s.gaps()
    assert len(gaps) == 1
    assert gaps[0][0] == pytest.approx(0.99, abs=0.02)


def test_series_slice_and_interp():
    t = np.linspace(0, 10, 101)
    s = Series("v", t, t * 2.0, units="m")
    sl = s.slice_time(2.0, 4.0)
    assert len(sl) == 21
    assert sl.values[0] == pytest.approx(4.0)
    assert s.interp_to(np.array([1.5]))[0] == pytest.approx(3.0)
    assert s.duration == pytest.approx(10.0)


def test_series_stats_ignores_nan():
    s = Series("v", np.arange(5.0), np.array([1.0, np.nan, 3.0, np.nan, 5.0]))
    st = s.stats()
    assert st["n"] == 3
    assert st["min"] == 1.0 and st["max"] == 5.0


def test_flightlog_flight_window_prefers_arm_events():
    log = FlightLog()
    log.add("x", np.linspace(0, 100, 101), np.zeros(101))
    log.add_event(10.0, "arm")
    log.add_event(80.0, "disarm")
    assert log.flight_window() == (10.0, 80.0)
    assert log.armed_intervals == [(10.0, 80.0)]


def test_unmatched_arm_closes_at_end_of_log():
    """A log that ends while armed is what a crash or brownout looks like."""
    log = FlightLog()
    log.metadata["duration"] = 50.0
    log.add_event(5.0, "arm")
    assert log.armed_intervals == [(5.0, 50.0)]


def test_require_and_first_present():
    log = FlightLog()
    log.add("a", [0, 1], [1, 2])
    assert log.require("a")
    assert not log.require("a", "b")
    assert log.first_present("b", "a").name == "a"
    assert log.first_present("b", "c") is None


def test_matching_returns_sorted_prefix_group():
    log = FlightLog()
    for i in (2, 0, 1):
        log.add(f"motor.{i}", [0, 1], [0.5, 0.5])
    assert [s.name for s in log.matching("motor.")] == ["motor.0", "motor.1", "motor.2"]


def test_severity_rank_ordering():
    assert Severity.CRITICAL.rank > Severity.WARNING.rank > Severity.INFO.rank
    assert Severity.INFO < Severity.CRITICAL


def test_mode_interval_duration():
    assert ModeInterval(2.0, 7.5, "POSCTL").duration == pytest.approx(5.5)


def test_finding_to_dict_is_json_safe():
    f = Finding(
        analyzer="vibration",
        severity=Severity.CRITICAL,
        title="t",
        explanation="e",
        action="a",
        evidence={"arr": np.array([1.0, 2.0]), "n": np.int64(3), "bad": float("nan")},
        t_start=1.0,
        t_end=2.0,
    )
    d = f.to_dict()
    assert d["severity"] == "critical"
    assert d["evidence"]["arr"] == [1.0, 2.0]
    assert d["evidence"]["n"] == 3
    assert d["evidence"]["bad"] is None
    assert d["t_start"] == 1.0


def test_jsonify_handles_nested_and_infinite():
    out = _jsonify({"a": [np.float64(1.5), {"b": math.inf}]})
    assert out["a"][0] == 1.5
    assert out["a"][1]["b"] is None
