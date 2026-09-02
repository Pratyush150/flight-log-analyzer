"""Analyzers.  Each one takes a :class:`FlightLog` and returns findings.

Every analyzer follows the same contract:

* signature ``analyze(log: FlightLog, **thresholds) -> list[Finding]``
* returns ``[]`` when the channels it needs are absent -- never raises
* never mutates the log
* every :class:`~flightlog.types.Finding` it emits carries evidence with real
  numbers and timestamps, plus a concrete recommended action

That contract is what lets :mod:`flightlog.report` run all of them blind and
still produce a coherent, ranked report on a log that is missing half its
topics.
"""

from __future__ import annotations

from . import control, ekf, gps, modes, power, spectral, vibration

__all__ = ["vibration", "ekf", "power", "control", "gps", "modes", "spectral", "ANALYZERS"]

#: Ordered registry used by :func:`flightlog.report.build_report`.
ANALYZERS = [
    ("vibration", vibration.analyze),
    ("power", power.analyze),
    ("control", control.analyze),
    ("ekf", ekf.analyze),
    ("gps", gps.analyze),
    ("modes", modes.analyze),
]
