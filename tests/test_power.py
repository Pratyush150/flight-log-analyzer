"""Power analyzer: internal resistance, sag, brownout risk, capacity sanity."""

from __future__ import annotations

import numpy as np
import pytest

from flightlog.analysis.power import (
    analyze,
    estimate_cell_count,
    estimate_internal_resistance,
    sag_metrics,
    throttle_voltage_correlation,
)
from flightlog.readers.synthetic import DefectSpec, generate
from flightlog.types import FlightLog, Severity


def test_internal_resistance_recovers_the_injected_value(sag_log):
    """48 mohm was injected; the V/I fit must find it."""
    res = estimate_internal_resistance(sag_log)
    assert res is not None
    assert res["r_pack_ohm"] == pytest.approx(0.048, rel=0.20)
    assert res["fit_r2"] > 0.9


def test_internal_resistance_recovers_a_healthy_pack_too(clean_log):
    res = estimate_internal_resistance(clean_log)
    assert res is not None
    assert res["r_pack_ohm"] == pytest.approx(0.010, rel=0.30)


def test_resistance_fit_separates_state_of_charge_decline():
    """Voltage falls over a flight for two reasons: charge used, and load.
    If the fit does not separate them it roughly doubles the resistance."""
    log = generate(
        defects=DefectSpec(
            pack_resistance_ohm=0.020, start_cell_v=4.20, end_cell_v=3.55
        )
    )
    res = estimate_internal_resistance(log)
    assert res is not None
    assert res["r_pack_ohm"] == pytest.approx(0.020, rel=0.25)
    assert res["soc_decline_v_per_s"] < 0


def test_resistance_returns_none_without_current_variation():
    """A constant load makes the slope unidentifiable. Reporting a number from
    a degenerate fit would be worse than reporting nothing."""
    log = FlightLog()
    t = np.linspace(0, 60, 601)
    log.add("bat.voltage", t, 22.0 - 0.001 * t)
    log.add("bat.current", t, np.full(t.size, 30.0))
    assert estimate_internal_resistance(log) is None


def test_brownout_warning_is_raised_for_a_sagging_pack():
    log = generate(
        defects=DefectSpec(
            pack_resistance_ohm=0.055, start_cell_v=3.95, end_cell_v=3.55
        )
    )
    findings = analyze(log)
    brownout = [f for f in findings if "brownout" in f.title.lower()]
    assert brownout, "a pack dipping below 3.3 V/cell must raise a brownout warning"
    assert brownout[0].severity is Severity.CRITICAL
    assert brownout[0].evidence["v_min_cell"] < 3.30
    assert "failsafe" in brownout[0].action.lower()


def test_high_resistance_produces_a_sag_at_peak_finding(sag_log):
    findings = analyze(sag_log)
    res = [f for f in findings if "resistance" in f.title.lower()]
    assert res
    assert res[0].severity in (Severity.WARNING, Severity.CRITICAL)
    assert res[0].evidence["cell_sag_at_peak_v"] > 0.30


def test_cell_count_is_read_from_the_log(clean_log):
    count, source = estimate_cell_count(clean_log)
    assert count == 6
    assert source == "logged"


def test_cell_count_is_inferred_when_not_logged():
    log = FlightLog()
    t = np.linspace(0, 30, 301)
    log.add("bat.voltage", t, np.full(t.size, 16.6))  # 4S at 4.15 V/cell
    count, source = estimate_cell_count(log)
    assert count == 4
    assert "inferred" in source


def test_sag_metrics_report_the_minimum_and_its_timestamp(sag_log):
    m = sag_metrics(sag_log)
    assert m is not None
    assert m["v_min"] < m["v_max"]
    assert 0.0 <= m["t_v_min"] <= sag_log.duration
    assert m["i_max"] > m["i_mean"]


def test_voltage_dips_correlate_with_throttle(clean_log):
    corr = throttle_voltage_correlation(clean_log)
    assert corr is not None
    assert corr["throttle_voltage_corr"] < -0.4


def test_clean_flight_has_no_critical_power_findings(clean_log):
    findings = analyze(clean_log)
    assert not [f for f in findings if f.severity is Severity.CRITICAL]
    assert any("healthy" in f.title.lower() for f in findings)


def test_analyzer_returns_nothing_without_battery_channels():
    log = FlightLog()
    log.add("accel.x", [0, 1, 2], [0.0, 0.1, 0.2])
    assert analyze(log) == []
