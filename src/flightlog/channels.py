"""Canonical channel names and the native-topic maps that feed them.

Every reader translates its native message/field names into the canonical
names listed here.  Analyzers only ever ask for canonical names, so adding a
new log format means writing one mapping table -- not touching any analyzer.

Naming convention: ``group.item`` in lower snake case, SI units throughout
(metres, m/s, m/s^2, radians, volts, amps, seconds).  Angles are radians in
storage and are converted to degrees only at report time.
"""

from __future__ import annotations

from typing import Dict, Tuple

__all__ = [
    "UNITS",
    "ULOG_MAP",
    "DATAFLASH_MAP",
    "CSV_ALIASES",
    "canonical_from_alias",
    "units_for",
]

#: Canonical channel -> unit string.  Also serves as the authoritative list of
#: names an analyzer may ask for.
UNITS: Dict[str, str] = {
    # --- inertial -------------------------------------------------------
    "accel.x": "m/s^2",
    "accel.y": "m/s^2",
    "accel.z": "m/s^2",
    "gyro.x": "rad/s",
    "gyro.y": "rad/s",
    "gyro.z": "rad/s",
    "vibe.x": "m/s^2",          # pre-computed vibration RMS, if the FC logs it
    "vibe.y": "m/s^2",
    "vibe.z": "m/s^2",
    "vibe.clip0": "count",      # cumulative accelerometer clip counters
    "vibe.clip1": "count",
    "vibe.clip2": "count",
    # --- attitude -------------------------------------------------------
    "att.roll": "rad",
    "att.pitch": "rad",
    "att.yaw": "rad",
    "att.roll_sp": "rad",
    "att.pitch_sp": "rad",
    "att.yaw_sp": "rad",
    "rate.roll": "rad/s",
    "rate.pitch": "rad/s",
    "rate.yaw": "rad/s",
    "rate.roll_sp": "rad/s",
    "rate.pitch_sp": "rad/s",
    "rate.yaw_sp": "rad/s",
    # --- altitude / position -------------------------------------------
    "alt.baro": "m",
    "alt.gps": "m",
    "alt.rangefinder": "m",
    "alt.ekf": "m",
    "alt.sp": "m",
    "pos.north": "m",
    "pos.east": "m",
    "vel.north": "m/s",
    "vel.east": "m/s",
    "vel.down": "m/s",
    # --- power ----------------------------------------------------------
    "bat.voltage": "V",
    "bat.current": "A",
    "bat.remaining": "fraction",
    "bat.consumed": "mAh",
    "bat.cell_count": "count",
    # --- GPS ------------------------------------------------------------
    "gps.fix_type": "enum",
    "gps.satellites": "count",
    "gps.hdop": "-",
    "gps.vdop": "-",
    "gps.lat": "deg",
    "gps.lon": "deg",
    "gps.speed": "m/s",
    # --- estimator ------------------------------------------------------
    "ekf.test_ratio.vel": "-",
    "ekf.test_ratio.pos": "-",
    "ekf.test_ratio.hgt": "-",
    "ekf.test_ratio.mag": "-",
    "ekf.innov.vel_n": "m/s",
    "ekf.innov.vel_e": "m/s",
    "ekf.innov.pos_d": "m",
    "ekf.innov.mag_x": "gauss",
    "ekf.reset_count": "count",
    # --- magnetometer ---------------------------------------------------
    "mag.x": "gauss",
    "mag.y": "gauss",
    "mag.z": "gauss",
    # --- actuators ------------------------------------------------------
    "motor.0": "fraction",
    "motor.1": "fraction",
    "motor.2": "fraction",
    "motor.3": "fraction",
    "motor.4": "fraction",
    "motor.5": "fraction",
    "motor.6": "fraction",
    "motor.7": "fraction",
    "throttle": "fraction",
    "rpm.0": "Hz",
    "rpm.1": "Hz",
    "rpm.2": "Hz",
    "rpm.3": "Hz",
    # --- integrators ----------------------------------------------------
    "pid.roll_i": "fraction",
    "pid.pitch_i": "fraction",
    "pid.yaw_i": "fraction",
    # --- RC / system ----------------------------------------------------
    "rc.rssi": "fraction",
    "rc.link_lost": "bool",
    "mode.id": "enum",
    "armed": "bool",
    "cpu.load": "fraction",
}

#: PX4 ULog: ``(topic, field, multi_instance_index)`` -> canonical name.
#: Only the subset this package actually analyzes is listed; unknown topics are
#: ignored rather than guessed at.
ULOG_MAP: Dict[Tuple[str, str], str] = {
    ("sensor_combined", "accelerometer_m_s2[0]"): "accel.x",
    ("sensor_combined", "accelerometer_m_s2[1]"): "accel.y",
    ("sensor_combined", "accelerometer_m_s2[2]"): "accel.z",
    ("sensor_combined", "gyro_rad[0]"): "gyro.x",
    ("sensor_combined", "gyro_rad[1]"): "gyro.y",
    ("sensor_combined", "gyro_rad[2]"): "gyro.z",
    ("sensor_accel", "x"): "accel.x",
    ("sensor_accel", "y"): "accel.y",
    ("sensor_accel", "z"): "accel.z",
    ("sensor_accel", "clip_counter[0]"): "vibe.clip0",
    ("sensor_accel", "clip_counter[1]"): "vibe.clip1",
    ("sensor_accel", "clip_counter[2]"): "vibe.clip2",
    ("vehicle_attitude_setpoint", "roll_body"): "att.roll_sp",
    ("vehicle_attitude_setpoint", "pitch_body"): "att.pitch_sp",
    ("vehicle_attitude_setpoint", "yaw_body"): "att.yaw_sp",
    ("vehicle_rates_setpoint", "roll"): "rate.roll_sp",
    ("vehicle_rates_setpoint", "pitch"): "rate.pitch_sp",
    ("vehicle_rates_setpoint", "yaw"): "rate.yaw_sp",
    ("vehicle_angular_velocity", "xyz[0]"): "rate.roll",
    ("vehicle_angular_velocity", "xyz[1]"): "rate.pitch",
    ("vehicle_angular_velocity", "xyz[2]"): "rate.yaw",
    ("vehicle_local_position", "z"): "alt.ekf",       # NED: negated by reader
    ("vehicle_local_position", "x"): "pos.north",
    ("vehicle_local_position", "y"): "pos.east",
    ("vehicle_local_position", "vx"): "vel.north",
    ("vehicle_local_position", "vy"): "vel.east",
    ("vehicle_local_position", "vz"): "vel.down",
    ("vehicle_gps_position", "fix_type"): "gps.fix_type",
    ("vehicle_gps_position", "satellites_used"): "gps.satellites",
    ("vehicle_gps_position", "hdop"): "gps.hdop",
    ("vehicle_gps_position", "vdop"): "gps.vdop",
    ("vehicle_gps_position", "lat"): "gps.lat",       # 1e-7 deg: scaled by reader
    ("vehicle_gps_position", "lon"): "gps.lon",
    ("vehicle_gps_position", "alt"): "gps.alt_mm",    # mm: scaled by reader
    ("vehicle_air_data", "baro_alt_meter"): "alt.baro",
    ("distance_sensor", "current_distance"): "alt.rangefinder",
    ("battery_status", "voltage_v"): "bat.voltage",
    ("battery_status", "voltage_filtered_v"): "bat.voltage",
    ("battery_status", "current_a"): "bat.current",
    ("battery_status", "current_filtered_a"): "bat.current",
    ("battery_status", "remaining"): "bat.remaining",
    ("battery_status", "discharged_mah"): "bat.consumed",
    ("battery_status", "cell_count"): "bat.cell_count",
    ("estimator_status", "vel_test_ratio"): "ekf.test_ratio.vel",
    ("estimator_status", "pos_test_ratio"): "ekf.test_ratio.pos",
    ("estimator_status", "hgt_test_ratio"): "ekf.test_ratio.hgt",
    ("estimator_status", "mag_test_ratio"): "ekf.test_ratio.mag",
    ("estimator_innovations", "gps_hvel[0]"): "ekf.innov.vel_n",
    ("estimator_innovations", "gps_hvel[1]"): "ekf.innov.vel_e",
    ("estimator_innovations", "gps_vpos"): "ekf.innov.pos_d",
    ("estimator_innovations", "mag_field[0]"): "ekf.innov.mag_x",
    ("sensor_mag", "x"): "mag.x",
    ("sensor_mag", "y"): "mag.y",
    ("sensor_mag", "z"): "mag.z",
    ("actuator_outputs", "output[0]"): "motor.0",
    ("actuator_outputs", "output[1]"): "motor.1",
    ("actuator_outputs", "output[2]"): "motor.2",
    ("actuator_outputs", "output[3]"): "motor.3",
    ("actuator_motors", "control[0]"): "motor.0",
    ("actuator_motors", "control[1]"): "motor.1",
    ("actuator_motors", "control[2]"): "motor.2",
    ("actuator_motors", "control[3]"): "motor.3",
    ("rate_ctrl_status", "rollspeed_integ"): "pid.roll_i",
    ("rate_ctrl_status", "pitchspeed_integ"): "pid.pitch_i",
    ("rate_ctrl_status", "yawspeed_integ"): "pid.yaw_i",
    ("cpuload", "load"): "cpu.load",
    ("input_rc", "rssi"): "rc.rssi",
    ("input_rc", "rc_lost"): "rc.link_lost",
}

#: ArduPilot dataflash: ``(message, field)`` -> canonical name.
DATAFLASH_MAP: Dict[Tuple[str, str], str] = {
    ("IMU", "AccX"): "accel.x",
    ("IMU", "AccY"): "accel.y",
    ("IMU", "AccZ"): "accel.z",
    ("IMU", "GyrX"): "gyro.x",
    ("IMU", "GyrY"): "gyro.y",
    ("IMU", "GyrZ"): "gyro.z",
    ("VIBE", "VibeX"): "vibe.x",
    ("VIBE", "VibeY"): "vibe.y",
    ("VIBE", "VibeZ"): "vibe.z",
    ("VIBE", "Clip0"): "vibe.clip0",
    ("VIBE", "Clip1"): "vibe.clip1",
    ("VIBE", "Clip2"): "vibe.clip2",
    ("ATT", "Roll"): "att.roll",        # degrees: converted by reader
    ("ATT", "Pitch"): "att.pitch",
    ("ATT", "Yaw"): "att.yaw",
    ("ATT", "DesRoll"): "att.roll_sp",
    ("ATT", "DesPitch"): "att.pitch_sp",
    ("ATT", "DesYaw"): "att.yaw_sp",
    ("RATE", "R"): "rate.roll",
    ("RATE", "P"): "rate.pitch",
    ("RATE", "Y"): "rate.yaw",
    ("RATE", "RDes"): "rate.roll_sp",
    ("RATE", "PDes"): "rate.pitch_sp",
    ("RATE", "YDes"): "rate.yaw_sp",
    ("CTUN", "Alt"): "alt.ekf",
    ("CTUN", "DAlt"): "alt.sp",
    ("CTUN", "BAlt"): "alt.baro",
    ("CTUN", "ThO"): "throttle",
    ("BARO", "Alt"): "alt.baro",
    ("RFND", "Dist"): "alt.rangefinder",
    ("GPS", "Status"): "gps.fix_type",
    ("GPS", "NSats"): "gps.satellites",
    ("GPS", "HDop"): "gps.hdop",
    ("GPS", "Lat"): "gps.lat",
    ("GPS", "Lng"): "gps.lon",
    ("GPS", "Alt"): "alt.gps",
    ("GPS", "Spd"): "gps.speed",
    ("BAT", "Volt"): "bat.voltage",
    ("BAT", "Curr"): "bat.current",
    ("BAT", "CurrTot"): "bat.consumed",
    ("BAT", "RemPct"): "bat.remaining",   # percent: scaled by reader
    ("MAG", "MagX"): "mag.x",
    ("MAG", "MagY"): "mag.y",
    ("MAG", "MagZ"): "mag.z",
    ("NKF4", "SV"): "ekf.test_ratio.vel",
    ("NKF4", "SP"): "ekf.test_ratio.pos",
    ("NKF4", "SH"): "ekf.test_ratio.hgt",
    ("NKF4", "SM"): "ekf.test_ratio.mag",
    ("XKF4", "SV"): "ekf.test_ratio.vel",
    ("XKF4", "SP"): "ekf.test_ratio.pos",
    ("XKF4", "SH"): "ekf.test_ratio.hgt",
    ("XKF4", "SM"): "ekf.test_ratio.mag",
    ("NKF3", "IVN"): "ekf.innov.vel_n",
    ("NKF3", "IVE"): "ekf.innov.vel_e",
    ("NKF3", "IPD"): "ekf.innov.pos_d",
    ("NKF3", "IMX"): "ekf.innov.mag_x",
    ("XKF3", "IVN"): "ekf.innov.vel_n",
    ("XKF3", "IVE"): "ekf.innov.vel_e",
    ("XKF3", "IPD"): "ekf.innov.pos_d",
    ("XKF3", "IMX"): "ekf.innov.mag_x",
    ("RCOU", "C1"): "motor.0",           # PWM us: normalised by reader
    ("RCOU", "C2"): "motor.1",
    ("RCOU", "C3"): "motor.2",
    ("RCOU", "C4"): "motor.3",
    ("RCOU", "C5"): "motor.4",
    ("RCOU", "C6"): "motor.5",
    ("ESC", "RPM"): "rpm.0",
    ("PIDR", "I"): "pid.roll_i",
    ("PIDP", "I"): "pid.pitch_i",
    ("PIDY", "I"): "pid.yaw_i",
    ("RSSI", "RXRSSI"): "rc.rssi",
    ("PM", "Load"): "cpu.load",
}

#: Loose CSV header aliases -> canonical name.  Matching is case-insensitive
#: and ignores spaces, hyphens and underscores, so ``"Accel X (m/s2)"`` and
#: ``"accel_x"`` both resolve to ``accel.x``.
CSV_ALIASES: Dict[str, str] = {
    "time": "__time__",
    "timestamp": "__time__",
    "times": "__time__",
    "t": "__time__",
    "timeus": "__time_us__",
    "timems": "__time_ms__",
    "accx": "accel.x",
    "accely": "accel.y",
    "accelx": "accel.x",
    "accelz": "accel.z",
    "accy": "accel.y",
    "accz": "accel.z",
    "gyrx": "gyro.x",
    "gyry": "gyro.y",
    "gyrz": "gyro.z",
    "roll": "att.roll",
    "pitch": "att.pitch",
    "yaw": "att.yaw",
    "desroll": "att.roll_sp",
    "despitch": "att.pitch_sp",
    "desyaw": "att.yaw_sp",
    "volt": "bat.voltage",
    "voltage": "bat.voltage",
    "curr": "bat.current",
    "current": "bat.current",
    "alt": "alt.baro",
    "altitude": "alt.baro",
    "nsats": "gps.satellites",
    "sats": "gps.satellites",
    "hdop": "gps.hdop",
    "fixtype": "gps.fix_type",
    "status": "gps.fix_type",
    "thr": "throttle",
    "thro": "throttle",
}


def _normalise(header: str) -> str:
    """Strip units, punctuation and case from a CSV header cell."""
    h = header.strip().lower()
    if "(" in h:
        h = h.split("(", 1)[0]
    for ch in " -_/[]":
        h = h.replace(ch, "")
    return h


def canonical_from_alias(header: str) -> str:
    """Map a free-form CSV header to a canonical name.

    Headers that are already canonical (``accel.x``) pass through untouched.
    Unknown headers are returned normalised, so custom channels survive into
    the :class:`~flightlog.types.FlightLog` even if no analyzer understands
    them -- they still get plotted and listed.
    """
    raw = header.strip()
    if raw in UNITS:
        return raw
    n = _normalise(raw)
    if n in CSV_ALIASES:
        return CSV_ALIASES[n]
    dotted = raw.strip().lower().replace(" ", "").replace("-", "").replace("_", ".")
    if dotted in UNITS:
        return dotted
    return n or raw


def units_for(name: str) -> str:
    """Unit string for a canonical channel, or ``""`` if unknown."""
    return UNITS.get(name, "")
