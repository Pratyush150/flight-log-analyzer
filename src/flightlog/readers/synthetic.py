"""Synthetic flight logs with injectable defects.

This module exists so that the entire package -- tests, demo, HTML report --
runs with zero real log files and zero optional dependencies.  It is also the
only honest way to unit-test a diagnostic tool: if you inject a 92 Hz
vibration peak and a 0.05 ohm battery, you know exactly what the analyzers are
supposed to find, and a regression shows up as a failing assert rather than as
a subtly wrong report six months later.

The generated data is *physically plausible*, not a physics simulation.  Gravity
sits on the Z accelerometer, throttle tracks climb rate, battery voltage sags
with current draw through a real series resistance, and GPS position is
integrated from velocity.  That is enough structure for every analyzer in this
package to behave the way it would on a real log.

Usage
-----
>>> from flightlog.readers.synthetic import generate, DefectSpec
>>> log = generate(defects=DefectSpec(vibration_peak_hz=92.0,
...                                   vibration_peak_amp=9.0))
>>> "accel.x" in log.series
True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..channels import units_for
from ..types import Event, FlightLog, ModeInterval

__all__ = ["DefectSpec", "SyntheticConfig", "generate", "generate_clean", "generate_defective"]

# Physical constants for the notional airframe.  A 5 kg / 15-inch class quad:
# big enough that 92 Hz is a believable prop tone, small enough that 4S/6S
# battery numbers look normal.
_G = 9.80665


@dataclass
class DefectSpec:
    """Which defects to inject, and how hard.

    Every field defaults to "no defect".  A :class:`DefectSpec` with no
    arguments produces a healthy aircraft, which is what
    :func:`generate_clean` uses to prove the analyzers do not cry wolf.
    """

    # -- vibration -------------------------------------------------------
    vibration_peak_hz: Optional[float] = None
    """Inject a discrete vibration tone at this frequency (Hz)."""
    vibration_peak_amp: float = 0.0
    """Amplitude of that tone in m/s^2 on the accelerometer."""
    vibration_axes: Tuple[str, ...] = ("x", "y", "z")
    """Which accelerometer axes carry the tone."""
    clip_events: int = 0
    """Number of accelerometer clipping events to record on each axis."""

    # -- power -----------------------------------------------------------
    pack_resistance_ohm: float = 0.010
    """Series resistance of the whole pack.  ~0.01 ohm is a healthy 6S LiPo;
    0.05 ohm is a puffed, cold or wildly undersized pack."""
    start_cell_v: float = 4.15
    """Open-circuit per-cell voltage at the start of the log."""
    end_cell_v: float = 3.80
    """Open-circuit per-cell voltage at the end of the log."""
    brownout_at: Optional[float] = None
    """Time (s) of a sudden extra voltage collapse, e.g. a failing connector."""

    # -- GPS -------------------------------------------------------------
    gps_glitch_at: Optional[float] = None
    """Time (s) of a GPS position jump."""
    gps_glitch_duration: float = 3.0
    gps_glitch_jump_m: float = 25.0
    gps_sat_drop_to: int = 5
    """Satellite count during the glitch."""

    # -- control ---------------------------------------------------------
    roll_oscillation_hz: Optional[float] = None
    """Inject a sustained roll oscillation at this frequency."""
    roll_oscillation_deg: float = 0.0
    """Peak amplitude of that oscillation, in degrees."""
    motor_asymmetry: float = 0.0
    """Extra output fraction pushed onto motor 0 (damaged prop / bent arm)."""
    motor_saturation_at: Optional[float] = None
    """Time (s) at which motor 1 pins to full output and its diagonal partner
    drops to idle -- the ESC-desync / lost-thrust signature."""
    integrator_windup_at: Optional[float] = None
    """Time (s) at which the roll integrator saturates and stays there."""

    # -- estimator -------------------------------------------------------
    ekf_variance_at: Optional[float] = None
    """Time (s) of an EKF innovation-ratio excursion."""
    ekf_variance_ratio: float = 1.4
    ekf_reset_at: Optional[float] = None
    """Time (s) of an EKF yaw/position reset."""
    mag_interference: float = 0.0
    """Fraction of magnetic field distortion correlated with throttle.
    This is the classic symptom of power leads routed next to the compass."""
    baro_drift_mps: float = 0.0
    """Barometer drift rate (m/s) -- makes baro and GPS height disagree."""

    # -- link ------------------------------------------------------------
    rc_loss_at: Optional[float] = None
    """Time (s) at which the RC link drops and failsafe triggers."""
    rc_loss_duration: float = 4.0


@dataclass
class SyntheticConfig:
    """Sampling rates and flight profile for the generator."""

    duration: float = 60.0
    imu_rate: float = 250.0
    attitude_rate: float = 100.0
    actuator_rate: float = 50.0
    ekf_rate: float = 20.0
    battery_rate: float = 10.0
    gps_rate: float = 5.0
    cell_count: int = 6
    capacity_mah: float = 10000.0
    hover_throttle: float = 0.48
    cruise_alt_m: float = 20.0
    arm_time: float = 4.0
    takeoff_time: float = 6.0
    seed: int = 20260101
    vehicle: str = "quadrotor"
    firmware: str = "synthetic-1.0"
    motor_hz: Optional[float] = None
    """If set, RPM telemetry is generated at this motor frequency (Hz).
    Left at ``None`` by default because most real logs have no RPM data --
    and the analyzers must stay useful without it."""
    defects: DefectSpec = field(default_factory=DefectSpec)


def _grid(duration: float, rate: float) -> np.ndarray:
    n = max(2, int(round(duration * rate)) + 1)
    return np.arange(n) / rate


def _smoothstep(x: np.ndarray) -> np.ndarray:
    """Clamped cubic ease, used for takeoff/landing ramps."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _altitude_profile(t: np.ndarray, cfg: SyntheticConfig) -> np.ndarray:
    """Take off, hold, descend.  Metres above launch."""
    d = cfg.duration
    t_up0, t_up1 = cfg.takeoff_time, cfg.takeoff_time + 6.0
    t_dn0, t_dn1 = d - 8.0, d - 2.0
    climb = _smoothstep((t - t_up0) / max(t_up1 - t_up0, 1e-6))
    descend = 1.0 - _smoothstep((t - t_dn0) / max(t_dn1 - t_dn0, 1e-6))
    return cfg.cruise_alt_m * climb * descend


def _mission_lateral(t: np.ndarray, cfg: SyntheticConfig) -> Tuple[np.ndarray, np.ndarray]:
    """A slow lazy-eight so position, velocity and attitude are not constant."""
    w = 2.0 * np.pi / 40.0
    active = _smoothstep((t - (cfg.takeoff_time + 6.0)) / 4.0) * (
        1.0 - _smoothstep((t - (cfg.duration - 12.0)) / 4.0)
    )
    north = 18.0 * np.sin(w * t) * active
    east = 12.0 * np.sin(2.0 * w * t) * active
    return north, east


def generate(
    config: Optional[SyntheticConfig] = None,
    defects: Optional[DefectSpec] = None,
    **overrides,
) -> FlightLog:
    """Build a synthetic :class:`~flightlog.types.FlightLog`.

    Parameters
    ----------
    config:
        Sampling/profile configuration.  Defaults to :class:`SyntheticConfig`.
    defects:
        Defects to inject.  Overrides ``config.defects`` when given.
    **overrides:
        Convenience keyword overrides applied to ``config`` (e.g.
        ``duration=30``).

    The result is deterministic for a fixed ``seed``: every test in this repo
    depends on that.
    """
    cfg = config or SyntheticConfig()
    if overrides:
        cfg = SyntheticConfig(**{**cfg.__dict__, **overrides})
    if defects is not None:
        cfg.defects = defects
    d = cfg.defects
    rng = np.random.default_rng(cfg.seed)

    log = FlightLog()
    log.metadata.update(
        {
            "vehicle": cfg.vehicle,
            "firmware": cfg.firmware,
            "log_format": "synthetic",
            "source": "flightlog.readers.synthetic",
            "duration": float(cfg.duration),
            "cell_count": cfg.cell_count,
            "capacity_mah": cfg.capacity_mah,
            "synthetic": True,
            "injected_defects": _describe_defects(d),
        }
    )
    if cfg.motor_hz is not None:
        log.metadata["motor_hz_hint"] = float(cfg.motor_hz)

    _add_attitude(log, cfg, rng)
    _add_imu(log, cfg, rng)
    _add_position(log, cfg, rng)
    _add_actuators(log, cfg, rng)
    _add_power(log, cfg, rng)
    _add_gps(log, cfg, rng)
    _add_estimator(log, cfg, rng)
    _add_link_and_modes(log, cfg, rng)
    return log


def generate_clean(**overrides) -> FlightLog:
    """A healthy flight.  Must produce zero critical findings."""
    return generate(SyntheticConfig(**overrides))


def generate_defective(**overrides) -> FlightLog:
    """The canonical "bad day" log used by ``flightlog-analyze --demo``.

    Combines the five failures that account for most real support requests:
    prop-imbalance vibration, a sagging pack, a GPS glitch, a roll oscillation
    from over-tuned rate P, and the EKF variance excursion those cause.
    """
    defects = DefectSpec(
        vibration_peak_hz=92.0,
        vibration_peak_amp=24.0,
        clip_events=6,
        pack_resistance_ohm=0.048,
        start_cell_v=4.05,
        end_cell_v=3.62,
        gps_glitch_at=33.0,
        gps_glitch_duration=4.0,
        gps_glitch_jump_m=28.0,
        roll_oscillation_hz=14.0,
        roll_oscillation_deg=3.2,
        motor_asymmetry=0.11,
        ekf_variance_at=33.0,
        ekf_variance_ratio=1.6,
        mag_interference=0.22,
        baro_drift_mps=0.10,
    )
    cfg = SyntheticConfig(defects=defects, **overrides)
    return generate(cfg)


# ---------------------------------------------------------------------------
# channel builders
# ---------------------------------------------------------------------------


def _add(log: FlightLog, name: str, t: np.ndarray, v: np.ndarray, source: str) -> None:
    log.add(name, t, v, units_for(name), source)


def _add_attitude(log: FlightLog, cfg: SyntheticConfig, rng: np.random.Generator) -> None:
    t = _grid(cfg.duration, cfg.attitude_rate)
    d = cfg.defects
    north, east = _mission_lateral(t, cfg)

    # Attitude that would actually produce that lateral motion: bank into the
    # acceleration.  Second difference of position -> commanded tilt.
    dt = 1.0 / cfg.attitude_rate
    acc_n = np.gradient(np.gradient(north, dt), dt)
    acc_e = np.gradient(np.gradient(east, dt), dt)
    pitch_sp = np.clip(np.arctan2(acc_n, _G), -0.35, 0.35)
    roll_sp = np.clip(np.arctan2(acc_e, _G), -0.35, 0.35)
    yaw_sp = np.unwrap(0.35 * np.sin(2.0 * np.pi * t / 55.0))

    # Real attitude lags the setpoint by roughly one control time constant and
    # carries a little tracking noise.
    lag = 0.08
    roll = _first_order_lag(roll_sp, dt, lag) + rng.normal(0, 0.0025, t.size)
    pitch = _first_order_lag(pitch_sp, dt, lag) + rng.normal(0, 0.0025, t.size)
    yaw = _first_order_lag(yaw_sp, dt, 0.25) + rng.normal(0, 0.0035, t.size)

    if d.roll_oscillation_hz and d.roll_oscillation_deg > 0:
        # Injected as an error on the *measured* attitude: the setpoint is
        # smooth, the aircraft rings.  That is what an over-tuned rate loop
        # looks like in a log.
        env = _smoothstep((t - cfg.takeoff_time) / 3.0) * (
            1.0 - _smoothstep((t - (cfg.duration - 6.0)) / 3.0)
        )
        roll = roll + np.deg2rad(d.roll_oscillation_deg) * env * np.sin(
            2.0 * np.pi * d.roll_oscillation_hz * t
        )

    _add(log, "att.roll", t, roll, "vehicle_attitude.roll")
    _add(log, "att.pitch", t, pitch, "vehicle_attitude.pitch")
    _add(log, "att.yaw", t, yaw, "vehicle_attitude.yaw")
    _add(log, "att.roll_sp", t, roll_sp, "vehicle_attitude_setpoint.roll_body")
    _add(log, "att.pitch_sp", t, pitch_sp, "vehicle_attitude_setpoint.pitch_body")
    _add(log, "att.yaw_sp", t, yaw_sp, "vehicle_attitude_setpoint.yaw_body")

    for name, sig, sp in (
        ("roll", roll, roll_sp),
        ("pitch", pitch, pitch_sp),
        ("yaw", yaw, yaw_sp),
    ):
        _add(log, f"rate.{name}", t, np.gradient(sig, dt), "vehicle_angular_velocity")
        _add(log, f"rate.{name}_sp", t, np.gradient(sp, dt), "vehicle_rates_setpoint")


def _first_order_lag(x: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """Discrete first-order lag; models closed-loop tracking delay."""
    a = dt / max(tau, dt)
    y = np.empty_like(x)
    acc = float(x[0]) if x.size else 0.0
    for i, v in enumerate(x):
        acc += a * (v - acc)
        y[i] = acc
    return y


def _add_imu(log: FlightLog, cfg: SyntheticConfig, rng: np.random.Generator) -> None:
    t = _grid(cfg.duration, cfg.imu_rate)
    d = cfg.defects
    alt = _altitude_profile(t, cfg)
    dt = 1.0 / cfg.imu_rate
    climb_acc = np.gradient(np.gradient(alt, dt), dt)

    # Baseline broadband noise: a well-mounted FC on a soft-mounted frame sits
    # around 1-2 m/s^2 RMS of high-frequency content.
    base = 0.55
    ax = rng.normal(0, base, t.size)
    ay = rng.normal(0, base, t.size)
    az = -_G - climb_acc + rng.normal(0, base * 1.3, t.size)

    # Low-frequency frame flex that every airframe has and nobody worries about.
    for f, amp in ((7.5, 0.35), (18.0, 0.22)):
        ph = rng.uniform(0, 2 * np.pi)
        ax += amp * np.sin(2 * np.pi * f * t + ph)
        ay += amp * 0.8 * np.sin(2 * np.pi * f * t + ph + 1.1)
        az += amp * 0.6 * np.sin(2 * np.pi * f * t + ph + 2.2)

    if cfg.motor_hz:
        # Always-present, harmless motor fundamental when RPM is known.
        for k, amp in ((1, 0.9), (2, 0.4)):
            f = cfg.motor_hz * k
            if f < 0.45 * cfg.imu_rate:
                ax += amp * np.sin(2 * np.pi * f * t + 0.3 * k)
                ay += amp * np.sin(2 * np.pi * f * t + 1.7 * k)
                az += amp * 0.7 * np.sin(2 * np.pi * f * t + 2.9 * k)

    if d.vibration_peak_hz and d.vibration_peak_amp > 0:
        f = float(d.vibration_peak_hz)
        # A real prop-imbalance tone is not a pure sinusoid: RPM wanders with
        # throttle, so the peak has a couple of Hz of width.
        jitter = 0.2 * np.sin(2 * np.pi * 0.15 * t)
        phase = 2 * np.pi * (f * t + jitter)
        amp = d.vibration_peak_amp * _smoothstep((t - cfg.takeoff_time) / 2.0)
        if "x" in d.vibration_axes:
            ax += amp * np.sin(phase)
        if "y" in d.vibration_axes:
            ay += amp * 0.85 * np.sin(phase + 0.9)
        if "z" in d.vibration_axes:
            az += amp * 0.55 * np.sin(phase + 1.9)

    _add(log, "accel.x", t, ax, "sensor_combined.accelerometer_m_s2[0]")
    _add(log, "accel.y", t, ay, "sensor_combined.accelerometer_m_s2[1]")
    _add(log, "accel.z", t, az, "sensor_combined.accelerometer_m_s2[2]")

    for axis, base_amp in (("x", 0.02), ("y", 0.02), ("z", 0.015)):
        g = rng.normal(0, base_amp, t.size)
        if d.vibration_peak_hz and d.vibration_peak_amp > 0:
            g += 0.02 * d.vibration_peak_amp * np.sin(
                2 * np.pi * d.vibration_peak_hz * t + 0.4
            )
        _add(log, f"gyro.{axis}", t, g, f"sensor_combined.gyro_rad[{'xyz'.index(axis)}]")

    # Clip counters are cumulative in both PX4 and ArduPilot.
    tc = _grid(cfg.duration, 10.0)
    clip = np.zeros(tc.size)
    if d.clip_events > 0:
        onset = np.searchsorted(tc, cfg.takeoff_time + 8.0)
        steps = np.linspace(0, d.clip_events, max(tc.size - onset, 1))
        clip[onset:] = np.floor(steps[: tc.size - onset])
    for i in range(3):
        _add(log, f"vibe.clip{i}", tc, clip, f"sensor_accel.clip_counter[{i}]")


def _add_position(log: FlightLog, cfg: SyntheticConfig, rng: np.random.Generator) -> None:
    t = _grid(cfg.duration, cfg.ekf_rate)
    d = cfg.defects
    alt = _altitude_profile(t, cfg)
    north, east = _mission_lateral(t, cfg)
    dt = 1.0 / cfg.ekf_rate

    _add(log, "pos.north", t, north + rng.normal(0, 0.05, t.size), "vehicle_local_position.x")
    _add(log, "pos.east", t, east + rng.normal(0, 0.05, t.size), "vehicle_local_position.y")
    _add(log, "vel.north", t, np.gradient(north, dt), "vehicle_local_position.vx")
    _add(log, "vel.east", t, np.gradient(east, dt), "vehicle_local_position.vy")
    _add(log, "vel.down", t, -np.gradient(alt, dt), "vehicle_local_position.vz")

    _add(log, "alt.ekf", t, alt + rng.normal(0, 0.08, t.size), "vehicle_local_position.z")
    _add(log, "alt.sp", t, alt, "vehicle_local_position_setpoint.z")

    baro = alt + rng.normal(0, 0.25, t.size)
    if d.baro_drift_mps:
        baro = baro + d.baro_drift_mps * np.maximum(t - cfg.takeoff_time, 0.0)
    _add(log, "alt.baro", t, baro, "vehicle_air_data.baro_alt_meter")


def _add_actuators(log: FlightLog, cfg: SyntheticConfig, rng: np.random.Generator) -> None:
    t = _grid(cfg.duration, cfg.actuator_rate)
    d = cfg.defects
    alt = _altitude_profile(t, cfg)
    dt = 1.0 / cfg.actuator_rate
    climb_rate = np.gradient(alt, dt)

    armed = (t >= cfg.arm_time) & (t <= cfg.duration - 1.0)
    thr = np.where(armed, cfg.hover_throttle + 0.09 * climb_rate, 0.0)
    thr = np.clip(thr + rng.normal(0, 0.006, t.size), 0.0, 1.0)
    _add(log, "throttle", t, thr, "vehicle_thrust_setpoint / CTUN.ThO")

    roll_s = log.get("att.roll")
    pitch_s = log.get("att.pitch")
    roll = roll_s.interp_to(t) if roll_s else np.zeros(t.size)
    pitch = pitch_s.interp_to(t) if pitch_s else np.zeros(t.size)

    # Standard X-quad mixer signs (motor 0 front-right, CCW numbering).
    mix = ((1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0))
    for i, (mr, mp) in enumerate(mix):
        m = thr + 0.55 * (mr * roll + mp * pitch) + rng.normal(0, 0.004, t.size)
        if i == 0 and d.motor_asymmetry:
            # A damaged or unbalanced prop makes exactly one motor work harder
            # for the whole flight -- the classic single-motor signature.
            m = m + d.motor_asymmetry * (thr > 0.05)
        if d.motor_saturation_at is not None and i in (0, 1):
            # A real desync does not just pin one motor: the mixer drives the
            # diagonal opposite (motor 0 here) down to idle while it commands
            # the failing motor (motor 1) to maximum, trying to cancel an
            # attitude error that never goes away.
            hit = (t >= d.motor_saturation_at) & (t <= d.motor_saturation_at + 6.0)
            m = np.where(hit, 1.0 if i == 1 else 0.06, m)
        _add(log, f"motor.{i}", t, np.clip(m, 0.0, 1.0), f"actuator_outputs.output[{i}]")

    if cfg.motor_hz:
        # Rotation frequency scales with throttle and equals cfg.motor_hz at
        # hover; motors read zero when disarmed, as real ESC telemetry does.
        ratio = np.clip(thr / max(cfg.hover_throttle, 1e-6), 0.0, 1.8)
        base = cfg.motor_hz * (0.55 + 0.45 * ratio)
        for i in range(4):
            rpm = np.where(thr > 0.02, base + rng.normal(0, 0.5, t.size), 0.0)
            _add(log, f"rpm.{i}", t, rpm, "esc_status.rpm")

    for axis in ("roll", "pitch", "yaw"):
        integ = 0.05 * np.sin(2 * np.pi * t / 30.0) + rng.normal(0, 0.004, t.size)
        if d.integrator_windup_at is not None and axis == "roll":
            hit = t >= d.integrator_windup_at
            integ = np.where(hit, 0.99, integ)
        _add(log, f"pid.{axis}_i", t, integ, f"rate_ctrl_status.{axis}speed_integ")

    _add(log, "cpu.load", _grid(cfg.duration, 2.0),
         np.clip(0.42 + rng.normal(0, 0.02, _grid(cfg.duration, 2.0).size), 0, 1),
         "cpuload.load")


def _add_power(log: FlightLog, cfg: SyntheticConfig, rng: np.random.Generator) -> None:
    t = _grid(cfg.duration, cfg.battery_rate)
    d = cfg.defects
    thr_s = log.get("throttle")
    thr = thr_s.interp_to(t) if thr_s else np.zeros(t.size)

    # Current roughly follows thrust^1.5; add a little measurement noise.
    current = 3.0 + 95.0 * np.clip(thr, 0, 1) ** 1.5
    current = current + rng.normal(0, 0.8, t.size)
    current = np.where(thr > 0.02, current, 0.4)

    # Open-circuit voltage falls linearly with consumed charge; the pack's
    # series resistance turns current into sag on top of that.
    frac = np.clip(t / max(cfg.duration, 1e-6), 0, 1)
    ocv_cell = d.start_cell_v + (d.end_cell_v - d.start_cell_v) * frac
    voltage = cfg.cell_count * ocv_cell - current * d.pack_resistance_ohm
    if d.brownout_at is not None:
        hit = t >= d.brownout_at
        voltage = voltage - np.where(hit, 2.4, 0.0)
    voltage = voltage + rng.normal(0, 0.012, t.size)

    consumed = np.cumsum(current) / cfg.battery_rate / 3.6  # A*s -> mAh
    remaining = np.clip(1.0 - consumed / cfg.capacity_mah, 0.0, 1.0)

    _add(log, "bat.voltage", t, voltage, "battery_status.voltage_filtered_v")
    _add(log, "bat.current", t, current, "battery_status.current_filtered_a")
    _add(log, "bat.consumed", t, consumed, "battery_status.discharged_mah")
    _add(log, "bat.remaining", t, remaining, "battery_status.remaining")
    _add(log, "bat.cell_count", t, np.full(t.size, cfg.cell_count), "battery_status.cell_count")


def _add_gps(log: FlightLog, cfg: SyntheticConfig, rng: np.random.Generator) -> None:
    t = _grid(cfg.duration, cfg.gps_rate)
    d = cfg.defects
    north_s, east_s = log.get("pos.north"), log.get("pos.east")
    north = north_s.interp_to(t) if north_s else np.zeros(t.size)
    east = east_s.interp_to(t) if east_s else np.zeros(t.size)
    alt_s = log.get("alt.ekf")
    alt = alt_s.interp_to(t) if alt_s else np.zeros(t.size)

    sats = np.full(t.size, 16.0)
    hdop = np.full(t.size, 0.72) + rng.normal(0, 0.02, t.size)
    fix = np.full(t.size, 3.0)
    # Time to first fix: the receiver is still acquiring for the first seconds.
    acquiring = t < 2.0
    fix[acquiring] = 1.0
    sats[acquiring] = 4.0
    hdop[acquiring] = 4.5

    jump_n = np.zeros(t.size)
    jump_e = np.zeros(t.size)
    if d.gps_glitch_at is not None:
        hit = (t >= d.gps_glitch_at) & (t < d.gps_glitch_at + d.gps_glitch_duration)
        jump_n[hit] = d.gps_glitch_jump_m
        jump_e[hit] = -0.6 * d.gps_glitch_jump_m
        sats[hit] = float(d.gps_sat_drop_to)
        hdop[hit] = 3.8
        fix[hit] = 2.0

    # Convert local metres to degrees around a fixed reference point.
    lat0, lon0 = 47.397742, 8.545594
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = m_per_deg_lat * float(np.cos(np.deg2rad(lat0)))
    lat = lat0 + (north + jump_n) / m_per_deg_lat
    lon = lon0 + (east + jump_e) / m_per_deg_lon

    _add(log, "gps.fix_type", t, fix, "vehicle_gps_position.fix_type")
    _add(log, "gps.satellites", t, sats, "vehicle_gps_position.satellites_used")
    _add(log, "gps.hdop", t, hdop, "vehicle_gps_position.hdop")
    _add(log, "gps.lat", t, lat, "vehicle_gps_position.lat")
    _add(log, "gps.lon", t, lon, "vehicle_gps_position.lon")
    _add(log, "alt.gps", t, alt + rng.normal(0, 0.6, t.size), "vehicle_gps_position.alt")
    _add(log, "gps.speed", t, np.abs(np.gradient(north, 1.0 / cfg.gps_rate)),
         "vehicle_gps_position.vel_m_s")


def _add_estimator(log: FlightLog, cfg: SyntheticConfig, rng: np.random.Generator) -> None:
    t = _grid(cfg.duration, cfg.ekf_rate)
    d = cfg.defects
    thr_s = log.get("throttle")
    thr = thr_s.interp_to(t) if thr_s else np.zeros(t.size)

    ratios = {
        "vel": 0.11 + 0.03 * rng.random(t.size),
        "pos": 0.09 + 0.03 * rng.random(t.size),
        "hgt": 0.13 + 0.03 * rng.random(t.size),
        "mag": 0.10 + 0.03 * rng.random(t.size),
    }
    if d.ekf_variance_at is not None:
        hit = (t >= d.ekf_variance_at) & (t < d.ekf_variance_at + 5.0)
        for key in ("vel", "pos"):
            ratios[key] = np.where(hit, d.ekf_variance_ratio, ratios[key])
    if d.mag_interference:
        ratios["mag"] = ratios["mag"] + 2.0 * d.mag_interference * thr
    if d.baro_drift_mps:
        ratios["hgt"] = ratios["hgt"] + 0.9 * abs(d.baro_drift_mps) * np.maximum(
            t - cfg.takeoff_time, 0.0
        ) / max(cfg.duration, 1e-6)
    for key, val in ratios.items():
        _add(log, f"ekf.test_ratio.{key}", t, val, f"estimator_status.{key}_test_ratio")

    innov_n = rng.normal(0, 0.06, t.size)
    innov_e = rng.normal(0, 0.06, t.size)
    innov_d = rng.normal(0, 0.09, t.size)
    if d.gps_glitch_at is not None:
        hit = (t >= d.gps_glitch_at) & (t < d.gps_glitch_at + d.gps_glitch_duration)
        innov_n = np.where(hit, d.gps_glitch_jump_m * 0.25, innov_n)
        innov_e = np.where(hit, -d.gps_glitch_jump_m * 0.15, innov_e)
    _add(log, "ekf.innov.vel_n", t, innov_n, "estimator_innovations.gps_hvel[0]")
    _add(log, "ekf.innov.vel_e", t, innov_e, "estimator_innovations.gps_hvel[1]")
    _add(log, "ekf.innov.pos_d", t, innov_d, "estimator_innovations.gps_vpos")

    resets = np.zeros(t.size)
    if d.ekf_reset_at is not None:
        resets[t >= d.ekf_reset_at] = 1.0
        log.add_event(d.ekf_reset_at, "ekf_reset", "yaw reset", axis="yaw")
    _add(log, "ekf.reset_count", t, resets, "estimator_status.reset_count")

    # Magnetometer: earth field plus throttle-correlated distortion.
    tm = _grid(cfg.duration, 50.0)
    thr_m = thr_s.interp_to(tm) if thr_s else np.zeros(tm.size)
    yaw_s = log.get("att.yaw")
    yaw = yaw_s.interp_to(tm) if yaw_s else np.zeros(tm.size)
    field_mag = 0.48  # gauss, typical mid-latitude total field
    mx = field_mag * np.cos(yaw) + rng.normal(0, 0.004, tm.size)
    my = field_mag * -np.sin(yaw) + rng.normal(0, 0.004, tm.size)
    mz = np.full(tm.size, 0.30) + rng.normal(0, 0.004, tm.size)
    if d.mag_interference:
        mx = mx * (1.0 + d.mag_interference * thr_m)
        mz = mz + field_mag * d.mag_interference * thr_m
    _add(log, "mag.x", tm, mx, "sensor_mag.x")
    _add(log, "mag.y", tm, my, "sensor_mag.y")
    _add(log, "mag.z", tm, mz, "sensor_mag.z")


def _add_link_and_modes(log: FlightLog, cfg: SyntheticConfig, rng: np.random.Generator) -> None:
    t = _grid(cfg.duration, 10.0)
    d = cfg.defects
    rssi = np.full(t.size, 0.92) + rng.normal(0, 0.01, t.size)
    lost = np.zeros(t.size)
    if d.rc_loss_at is not None:
        hit = (t >= d.rc_loss_at) & (t < d.rc_loss_at + d.rc_loss_duration)
        rssi[hit] = 0.02
        lost[hit] = 1.0
        log.add_event(d.rc_loss_at, "failsafe", "RC link lost", source="rc")
    _add(log, "rc.rssi", t, rssi, "input_rc.rssi")
    _add(log, "rc.link_lost", t, lost, "input_rc.rc_lost")

    armed = ((t >= cfg.arm_time) & (t <= cfg.duration - 1.0)).astype(float)
    _add(log, "armed", t, armed, "vehicle_status.arming_state")

    log.events.append(Event(cfg.arm_time, "arm", "armed"))
    log.events.append(Event(cfg.duration - 1.0, "disarm", "disarmed"))

    mode_marks: List[Tuple[float, str]] = [
        (0.0, "MANUAL"),
        (cfg.arm_time, "STABILIZED"),
        (cfg.takeoff_time, "POSCTL"),
        (cfg.duration - 8.0, "AUTO.LAND"),
    ]
    if d.rc_loss_at is not None:
        mode_marks.append((d.rc_loss_at, "AUTO.RTL"))
        mode_marks.append((d.rc_loss_at + d.rc_loss_duration, "POSCTL"))
    mode_marks.sort(key=lambda m: m[0])

    mode_ids = {"MANUAL": 0, "STABILIZED": 1, "POSCTL": 2, "AUTO.RTL": 5, "AUTO.LAND": 6}
    for i, (start, mode) in enumerate(mode_marks):
        end = mode_marks[i + 1][0] if i + 1 < len(mode_marks) else cfg.duration
        if end > start:
            log.modes.append(ModeInterval(start, end, mode))
            log.events.append(Event(start, "mode_change", mode))

    tm = _grid(cfg.duration, 5.0)
    ids = np.zeros(tm.size)
    for iv in log.modes:
        ids[(tm >= iv.start) & (tm < iv.end)] = mode_ids.get(iv.mode, 0)
    _add(log, "mode.id", tm, ids, "vehicle_status.nav_state")
    log.events.sort(key=lambda e: e.time)


def _describe_defects(d: DefectSpec) -> Dict[str, object]:
    """Human-readable record of what was injected (used by the demo header)."""
    out: Dict[str, object] = {}
    if d.vibration_peak_hz:
        out["vibration"] = f"{d.vibration_peak_amp:.1f} m/s^2 tone at {d.vibration_peak_hz:.0f} Hz"
    if d.clip_events:
        out["clipping"] = f"{d.clip_events} accel clip events"
    if d.pack_resistance_ohm > 0.02:
        out["battery"] = f"pack resistance {d.pack_resistance_ohm*1000:.0f} mohm"
    if d.brownout_at is not None:
        out["brownout"] = f"voltage collapse at t={d.brownout_at:.1f}s"
    if d.gps_glitch_at is not None:
        out["gps_glitch"] = f"{d.gps_glitch_jump_m:.0f} m jump at t={d.gps_glitch_at:.1f}s"
    if d.roll_oscillation_hz:
        out["roll_oscillation"] = f"{d.roll_oscillation_deg:.1f} deg at {d.roll_oscillation_hz:.1f} Hz"
    if d.motor_asymmetry:
        out["motor_asymmetry"] = f"motor 0 +{d.motor_asymmetry*100:.0f}% output"
    if d.motor_saturation_at is not None:
        out["motor_saturation"] = f"motor 1 pinned from t={d.motor_saturation_at:.1f}s"
    if d.integrator_windup_at is not None:
        out["integrator_windup"] = f"roll I saturated from t={d.integrator_windup_at:.1f}s"
    if d.ekf_variance_at is not None:
        out["ekf_variance"] = f"ratio {d.ekf_variance_ratio:.2f} at t={d.ekf_variance_at:.1f}s"
    if d.ekf_reset_at is not None:
        out["ekf_reset"] = f"reset at t={d.ekf_reset_at:.1f}s"
    if d.mag_interference:
        out["mag_interference"] = f"{d.mag_interference*100:.0f}% throttle-correlated"
    if d.baro_drift_mps:
        out["baro_drift"] = f"{d.baro_drift_mps:.2f} m/s"
    if d.rc_loss_at is not None:
        out["rc_loss"] = f"link lost at t={d.rc_loss_at:.1f}s"
    return out
