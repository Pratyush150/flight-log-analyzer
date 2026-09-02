#!/usr/bin/env python3
"""Generate a defective synthetic flight, analyze it, write an HTML report.

This is the same thing ``flightlog-analyze --demo`` does, spelled out as
library calls so you can see where to hook your own code in.

    python3 examples/run_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flightlog import analyze_log, generate_defective, render_terminal, write_html, write_json
from flightlog.types import Severity


def main() -> int:
    log = generate_defective(duration=60.0)

    print("Injected defects:")
    for name, detail in log.metadata["injected_defects"].items():
        print(f"  {name:20s} {detail}")
    print()

    report = analyze_log(log, source="synthetic defective flight")
    print(render_terminal(report, color=True, max_findings=6))

    out_dir = os.path.dirname(os.path.abspath(__file__))
    html = write_html(report, os.path.join(out_dir, "demo_report.html"), log)
    js = write_json(report, os.path.join(out_dir, "demo_report.json"))
    print(f"\nHTML: {html}\nJSON: {js}")

    # Exit non-zero when something critical was found -- handy in CI, where a
    # regression test flight that develops a critical finding should fail the
    # build rather than quietly produce a report nobody reads.
    return 1 if report.by_severity(Severity.CRITICAL) else 0


if __name__ == "__main__":
    raise SystemExit(main())
