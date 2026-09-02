"""The synthetic generator itself: determinism, coverage, defect bookkeeping."""

from __future__ import annotations

import numpy as np
import pytest

from flightlog.readers.synthetic import DefectSpec, SyntheticConfig, generate


def test_generation_is_deterministic_for_a_fixed_seed():
    a = generate(SyntheticConfig(seed=7))
    b = generate(SyntheticConfig(seed=7))
    assert np.array_equal(a.series["accel.x"].values, b.series["accel.x"].values)


def test_a_different_seed_gives_different_noise():
    a = generate(SyntheticConfig(seed=1))
    b = generate(SyntheticConfig(seed=2))
    assert not np.array_equal(a.series["accel.x"].values, b.series["accel.x"].values)


def test_every_analyzer_input_channel_is_generated(clean_log):
    required = [
        "accel.x", "accel.y", "accel.z", "gyro.x",
        "att.roll", "att.roll_sp", "rate.roll",
        "alt.baro", "alt.gps", "alt.ekf", "alt.sp",
        "bat.voltage", "bat.current", "bat.remaining",
        "gps.fix_type", "gps.satellites", "gps.hdop", "gps.lat", "gps.lon",
        "ekf.test_ratio.vel", "ekf.test_ratio.pos", "ekf.innov.vel_n",
        "mag.x", "motor.0", "motor.3", "throttle",
        "pid.roll_i", "rc.rssi", "mode.id", "armed",
    ]
    missing = [c for c in required if clean_log.get(c) is None]
    assert missing == [], f"generator is missing channels: {missing}"


def test_channels_carry_units_and_a_native_source_name(clean_log):
    accel = clean_log.series["accel.x"]
    assert accel.units == "m/s^2"
    assert "accelerometer" in accel.source


def test_sample_rates_match_the_configuration():
    log = generate(SyntheticConfig(imu_rate=200.0, attitude_rate=50.0))
    assert log.series["accel.x"].sample_rate == pytest.approx(200.0, rel=0.02)
    assert log.series["att.roll"].sample_rate == pytest.approx(50.0, rel=0.02)


def test_gravity_sits_on_the_z_accelerometer(clean_log):
    assert np.mean(clean_log.series["accel.z"].values) == pytest.approx(-9.81, abs=0.5)


def test_altitude_profile_takes_off_and_lands(clean_log):
    alt = clean_log.series["alt.ekf"].values
    assert alt[0] == pytest.approx(0.0, abs=1.0)
    assert np.max(alt) == pytest.approx(20.0, abs=1.5)
    assert alt[-1] == pytest.approx(0.0, abs=1.5)


def test_arm_and_disarm_events_bracket_the_flight(clean_log):
    intervals = clean_log.armed_intervals
    assert len(intervals) == 1
    start, end = intervals[0]
    assert start == pytest.approx(4.0, abs=0.1)
    assert end < clean_log.duration


def test_battery_voltage_falls_and_current_follows_throttle(clean_log):
    v = clean_log.series["bat.voltage"].values
    assert v[0] > v[-1]
    thr = clean_log.series["throttle"]
    cur = clean_log.series["bat.current"]
    corr = np.corrcoef(thr.interp_to(cur.time), cur.values)[0, 1]
    assert corr > 0.9


def test_injected_defects_are_recorded_in_metadata():
    log = generate(defects=DefectSpec(vibration_peak_hz=92.0, vibration_peak_amp=10.0))
    injected = log.metadata["injected_defects"]
    assert "vibration" in injected
    assert "92 Hz" in injected["vibration"]


def test_clean_log_records_no_injected_defects(clean_log):
    assert clean_log.metadata["injected_defects"] == {}


def test_defective_log_injects_the_documented_defect_set(defective_log):
    injected = defective_log.metadata["injected_defects"]
    for key in ("vibration", "clipping", "battery", "gps_glitch", "roll_oscillation"):
        assert key in injected


def test_keyword_overrides_reach_the_config():
    log = generate(duration=25.0)
    assert log.duration == pytest.approx(25.0, abs=0.1)


def test_rpm_channels_only_appear_when_requested(clean_log):
    assert clean_log.get("rpm.0") is None
    with_rpm = generate(SyntheticConfig(motor_hz=100.0))
    assert with_rpm.get("rpm.0") is not None
