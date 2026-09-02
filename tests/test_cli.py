"""The command-line interface, including the one-command demo."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET

from flightlog.cli import build_parser, main


def test_demo_runs_end_to_end_and_writes_both_reports(tmp_path, capsys):
    html = tmp_path / "report.html"
    js = tmp_path / "report.json"
    code = main(["--demo", "--no-color", "--duration", "40",
                 "--html", str(html), "--json", str(js)])
    assert code == 1, "the demo log has critical findings, so the exit code is 1"
    out = capsys.readouterr().out
    assert "FLIGHT LOG HEALTH REPORT" in out
    assert "Synthetic flight with these defects injected" in out

    text = html.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    for svg in re.findall(r"<svg.*?</svg>", text, re.S):
        ET.fromstring(svg)

    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["counts"]["critical"] > 0
    assert data["findings"][0]["action"]


def test_clean_demo_exits_zero(capsys):
    assert main(["--demo-clean", "--no-color", "--duration", "40", "--quiet"]) == 0


def test_json_to_stdout(capsys):
    main(["--demo-clean", "--duration", "30", "--quiet", "--json", "-"])
    data = json.loads(capsys.readouterr().out)
    assert data["counts"]["critical"] == 0
    assert "verdict" in data


def test_formats_listing_reports_availability(capsys):
    assert main(["--formats"]) == 0
    out = capsys.readouterr().out
    assert "PX4 ULog" in out and "ArduPilot dataflash" in out
    assert "pyulog" in out and "pymavlink" in out


def test_missing_log_returns_exit_code_two(capsys):
    assert main(["/no/such/flight.ulg"]) == 2
    assert "error" in capsys.readouterr().err.lower()


def test_no_arguments_prints_help_and_fails(capsys):
    assert main([]) == 2
    assert "--demo" in capsys.readouterr().err


def test_quiet_suppresses_the_terminal_report(tmp_path, capsys):
    js = tmp_path / "o.json"
    main(["--demo-clean", "--duration", "30", "--quiet", "--json", str(js)])
    assert capsys.readouterr().out == ""
    assert js.exists()


def test_no_color_output_has_no_escape_sequences(capsys):
    main(["--demo-clean", "--duration", "30", "--no-color"])
    assert "\033[" not in capsys.readouterr().out


def test_csv_input_is_analyzed_from_the_command_line(clean_log, tmp_path, capsys):
    from flightlog.readers.csv_reader import write_csv

    path = tmp_path / "flight.csv"
    write_csv(clean_log, str(path), channels=["accel.x", "accel.y", "accel.z"])
    assert main([str(path), "--no-color", "--quiet"]) == 0


def test_parser_exposes_the_documented_flags():
    parser = build_parser()
    flags = {a.dest for a in parser._actions}
    assert {"demo", "html", "json", "formats", "quiet", "max_findings"} <= flags
