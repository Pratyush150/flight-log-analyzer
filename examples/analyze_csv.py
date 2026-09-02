#!/usr/bin/env python3
"""Analyze a CSV export -- the path that needs no optional dependencies.

Every log tool in the ecosystem can export CSV (Flight Review, MAVExplorer,
UAV Log Viewer, ``ulog2csv``, ``mavlogdump.py --format csv``), so this works
even when a client cannot install pyulog or pymavlink.

    python3 examples/analyze_csv.py sample_flight.csv
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flightlog import analyze_log, render_terminal
from flightlog.readers.csv_reader import read_csv


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sample_flight.csv"
    )
    log = read_csv(path)

    print(f"loaded {len(log.series)} channels from {os.path.basename(path)}")
    for name in sorted(log.series):
        s = log.series[name]
        print(f"  {name:16s} {len(s):6d} samples @ {s.sample_rate:6.1f} Hz  [{s.units}]")
    print()

    report = analyze_log(log, source=path)
    print(render_terminal(report, color=True))
    return 1 if report.counts["critical"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
