"""Shared fixtures.

Synthetic logs are generated once per session and reused. Generating one is
cheap (tens of milliseconds), but the vibration analysis over 15000 IMU samples
is not free, so caching keeps the whole suite well under a second.

Every fixture is deterministic: the generator is seeded, so an assertion that
passes today passes on any machine.
"""

from __future__ import annotations

import os
import sys

import pytest

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from flightlog.readers.synthetic import (  # noqa: E402
    DefectSpec,
    SyntheticConfig,
    generate,
    generate_clean,
    generate_defective,
)


@pytest.fixture(scope="session")
def clean_log():
    """A healthy 60-second flight. Must produce zero critical findings."""
    return generate_clean()


@pytest.fixture(scope="session")
def defective_log():
    """The canonical demo log: vibration, sag, GPS glitch, oscillation."""
    return generate_defective()


@pytest.fixture(scope="session")
def vibration_log():
    """Only a 92 Hz accelerometer tone injected, nothing else."""
    return generate(
        defects=DefectSpec(vibration_peak_hz=92.0, vibration_peak_amp=18.0, clip_events=4)
    )


@pytest.fixture(scope="session")
def rpm_log():
    """A 92 Hz tone plus RPM telemetry at 92 Hz, so it is a 1x harmonic."""
    cfg = SyntheticConfig(
        motor_hz=92.0,
        defects=DefectSpec(vibration_peak_hz=92.0, vibration_peak_amp=14.0),
    )
    return generate(cfg)


@pytest.fixture(scope="session")
def sag_log():
    """A pack with 48 mohm of series resistance and a low starting charge."""
    return generate(
        defects=DefectSpec(
            pack_resistance_ohm=0.048, start_cell_v=4.00, end_cell_v=3.60
        )
    )


@pytest.fixture(scope="session")
def glitch_log():
    """A single 30 m GPS glitch starting at t = 25.0 s."""
    return generate(
        defects=DefectSpec(
            gps_glitch_at=25.0, gps_glitch_duration=4.0, gps_glitch_jump_m=30.0
        )
    )


@pytest.fixture(scope="session")
def oscillation_log():
    """A 6.5 Hz roll oscillation, in the attitude-P band."""
    return generate(
        defects=DefectSpec(roll_oscillation_hz=6.5, roll_oscillation_deg=4.0)
    )


@pytest.fixture(scope="session")
def desync_log():
    """Motor 1 pinned at full output from t = 30 s (the desync signature)."""
    return generate(defects=DefectSpec(motor_saturation_at=30.0))


@pytest.fixture(scope="session")
def rc_loss_log():
    """RC link lost at t = 20 s for 4 s, triggering a failsafe."""
    return generate(defects=DefectSpec(rc_loss_at=20.0, rc_loss_duration=4.0))
