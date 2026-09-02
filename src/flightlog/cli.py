"""Command-line interface for flight-log-analyzer.

``flightlog-analyze flight.ulg --html report.html --json out.json``

The ``--demo`` flag is the one that matters for a first look: it generates a
synthetic flight with known defects and produces the full report, so the tool
can be evaluated in one command with no log file and no optional dependencies
installed.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Optional, Sequence

from . import __version__
from .readers import generate_clean, generate_defective, load, supported_formats
from .readers.ulog_reader import MissingDependencyError
from .report import build_report, render_json, render_terminal, write_html, write_json
from .types import FlightLog

__all__ = ["main", "build_parser"]

_EPILOG = """\
examples:
  flightlog-analyze --demo --html demo.html
      Generate a synthetic defective flight and write a full HTML report.

  flightlog-analyze flight.ulg --html report.html --json report.json
      Analyze a PX4 ULog and write both report formats.

  flightlog-analyze flight.bin --quiet --json - | jq '.findings[0]'
      Analyze an ArduPilot log and pipe JSON to another tool.

exit codes:
  0  no critical findings
  1  at least one critical finding
  2  the log could not be read
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="flightlog-analyze",
        description=(
            "Analyze a PX4 ULog or ArduPilot dataflash log and produce a ranked "
            "health report: vibration, power, control, estimator, GNSS, modes."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("log", nargs="?", help="path to .ulg, .bin, .log or .csv")
    p.add_argument("--demo", action="store_true",
                   help="analyze a synthetic flight with injected defects (no log needed)")
    p.add_argument("--demo-clean", action="store_true",
                   help="analyze a synthetic healthy flight (proves the analyzers do not "
                        "cry wolf)")
    p.add_argument("--html", metavar="PATH", help="write a self-contained HTML report")
    p.add_argument("--json", metavar="PATH",
                   help="write JSON findings ('-' writes to stdout)")
    p.add_argument("--formats", action="store_true",
                   help="list supported log formats and whether their readers are available")
    p.add_argument("--no-color", action="store_true", help="disable ANSI colour")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="suppress the terminal report (still writes --html/--json)")
    p.add_argument("--max-findings", type=int, default=0, metavar="N",
                   help="show only the top N findings in the terminal output")
    p.add_argument("--duration", type=float, default=60.0, metavar="S",
                   help="synthetic flight duration for --demo (default: 60)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="print a traceback when a log fails to load")
    p.add_argument("--version", action="version", version=f"flight-log-analyzer {__version__}")
    return p


def _print_formats(stream) -> None:
    rows = supported_formats()
    name_w = max(len(str(r["format"])) for r in rows) + 2
    print("supported log formats:", file=stream)
    for r in rows:
        exts = ", ".join(str(e) for e in r["extensions"])
        status = "available" if r["available"] else f"MISSING - {r['install']}"
        print(
            f"  {str(r['format']).ljust(name_w)}{exts.ljust(24)}"
            f"{str(r['library']).ljust(12)}{status}",
            file=stream,
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.formats:
        _print_formats(sys.stdout)
        return 0

    log: Optional[FlightLog] = None
    source = ""
    if args.demo:
        log = generate_defective(duration=args.duration)
        source = "synthetic defective flight"
    elif args.demo_clean:
        log = generate_clean(duration=args.duration)
        source = "synthetic clean flight"
    elif args.log:
        try:
            log = load(args.log)
            source = os.path.abspath(args.log)
        except MissingDependencyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: could not read {args.log!r}: {exc}", file=sys.stderr)
            if args.verbose:
                traceback.print_exc()
            return 2
        except Exception as exc:  # pragma: no cover - reader-specific failures
            print(f"error: failed to parse {args.log!r}: {exc}", file=sys.stderr)
            if args.verbose:
                traceback.print_exc()
            return 2
    else:
        parser.print_help(sys.stderr)
        print("\nerror: give a log path, or use --demo to see the tool run.",
              file=sys.stderr)
        return 2

    report = build_report(log, source=source)

    if args.demo and not args.quiet:
        injected = log.metadata.get("injected_defects", {})
        if injected:
            print("Synthetic flight with these defects injected:")
            for k, v in injected.items():
                print(f"  - {k}: {v}")
            print()

    if not args.quiet:
        print(
            render_terminal(
                report, color=not args.no_color, max_findings=args.max_findings
            )
        )

    if args.html:
        write_html(report, args.html, log)
        if not args.quiet:
            print(f"HTML report written to {os.path.abspath(args.html)}")
    if args.json:
        if args.json == "-":
            print(render_json(report))
        else:
            write_json(report, args.json)
            if not args.quiet:
                print(f"JSON findings written to {os.path.abspath(args.json)}")

    return 1 if report.counts["critical"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
