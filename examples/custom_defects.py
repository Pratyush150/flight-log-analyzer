#!/usr/bin/env python3
"""Build a synthetic log with exactly the defects you want to test against.

Useful for two things:

* checking that a threshold change still catches the case it was written for
* producing a worked example for a client -- "this is what your symptom looks
  like in a log, and this is what the tool says about it"

    python3 examples/custom_defects.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flightlog import analyze_log
from flightlog.readers.synthetic import DefectSpec, SyntheticConfig, generate

SCENARIOS = {
    "bent prop on motor 0": DefectSpec(
        vibration_peak_hz=118.0, vibration_peak_amp=20.0, motor_asymmetry=0.10
    ),
    "tired 6S pack": DefectSpec(
        pack_resistance_ohm=0.060, start_cell_v=3.95, end_cell_v=3.45
    ),
    "rate P too high on roll": DefectSpec(
        roll_oscillation_hz=15.0, roll_oscillation_deg=3.5
    ),
    "compass next to the power leads": DefectSpec(mag_interference=0.30),
    "ESC desync at t=30 s": DefectSpec(motor_saturation_at=30.0),
    "GPS glitch under a tree line": DefectSpec(
        gps_glitch_at=28.0, gps_glitch_duration=5.0, gps_glitch_jump_m=35.0
    ),
}


def main() -> None:
    for name, defects in SCENARIOS.items():
        log = generate(SyntheticConfig(duration=60.0, defects=defects))
        report = analyze_log(log)
        top = report.findings[0]
        print(f"{name}")
        print(f"  verdict : {report.verdict}")
        print(f"  top     : [{top.severity.value}] {top.title}")
        print(f"  action  : {top.action[:96]}...")
        print()


if __name__ == "__main__":
    main()
