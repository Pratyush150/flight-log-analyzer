"""Report aggregation: ranking, the no-false-positive guarantee, renderers."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from flightlog import analyze_log
from flightlog.report import (
    ANALYZER_PRECEDENCE,
    build_report,
    rank_findings,
    render_html,
    render_json,
    render_terminal,
    write_html,
    write_json,
)
from flightlog.types import Finding, Severity


def _finding(analyzer="vibration", severity=Severity.INFO, title="t", confidence=1.0):
    return Finding(
        analyzer=analyzer,
        severity=severity,
        title=title,
        explanation="e",
        action="a",
        confidence=confidence,
    )


# --- ranking ---------------------------------------------------------------


def test_severity_ranking_orders_critical_first():
    findings = [
        _finding(severity=Severity.INFO, title="i"),
        _finding(severity=Severity.CRITICAL, title="c"),
        _finding(severity=Severity.WARNING, title="w"),
    ]
    ordered = [f.title for f in rank_findings(findings)]
    assert ordered == ["c", "w", "i"]


def test_equal_severity_orders_by_confidence_then_cause_before_symptom():
    findings = [
        _finding("ekf", Severity.CRITICAL, "symptom", confidence=1.0),
        _finding("vibration", Severity.CRITICAL, "cause", confidence=1.0),
        _finding("vibration", Severity.CRITICAL, "unsure", confidence=0.4),
    ]
    ordered = [f.title for f in rank_findings(findings)]
    assert ordered == ["cause", "symptom", "unsure"]
    assert ANALYZER_PRECEDENCE["vibration"] < ANALYZER_PRECEDENCE["ekf"]


def test_ranking_is_stable_across_repeated_calls():
    findings = [_finding(title=f"t{i}") for i in range(10)]
    assert [f.title for f in rank_findings(findings)] == [
        f.title for f in rank_findings(list(reversed(findings)))
    ]


# --- the no-false-positive guarantee ---------------------------------------


def test_clean_flight_produces_zero_critical_findings(clean_log):
    """The single most important test in this repository.

    A diagnostic tool that flags a healthy aircraft is worse than no tool: it
    sends people to replace parts that were never broken.
    """
    report = analyze_log(clean_log)
    critical = report.by_severity(Severity.CRITICAL)
    assert critical == [], f"false positives: {[f.title for f in critical]}"


def test_clean_flight_produces_zero_warnings_too(clean_log):
    report = analyze_log(clean_log)
    warnings = report.by_severity(Severity.WARNING)
    assert warnings == [], f"false positives: {[f.title for f in warnings]}"


def test_clean_flight_verdict_says_healthy(clean_log):
    report = analyze_log(clean_log)
    assert "healthy" in report.verdict.lower()
    assert report.counts["info"] > 0


def test_defective_flight_finds_every_injected_defect_class(defective_log):
    report = analyze_log(defective_log)
    titles = " | ".join(f.title.lower() for f in report.findings)
    for expected in ("92 hz", "brownout", "glitch", "oscillation", "magnetometer"):
        assert expected in titles, f"{expected!r} missing from: {titles}"
    assert report.counts["critical"] > 0


def test_every_finding_has_evidence_explanation_and_action(defective_log):
    for f in analyze_log(defective_log).findings:
        assert f.explanation.strip(), f.title
        assert f.action.strip(), f.title
        assert f.analyzer in ANALYZER_PRECEDENCE


def test_verdict_names_the_top_critical_finding(defective_log):
    report = analyze_log(defective_log)
    assert report.findings[0].title in report.verdict


# --- renderers -------------------------------------------------------------


def test_json_round_trips_and_carries_counts(defective_log):
    report = analyze_log(defective_log)
    data = json.loads(render_json(report))
    assert data["counts"]["critical"] == report.counts["critical"]
    assert len(data["findings"]) == len(report.findings)
    assert data["findings"][0]["severity"] == "critical"
    assert "channels" in data and len(data["channels"]) > 10


def test_json_contains_no_nan_or_infinity_tokens(defective_log):
    text = render_json(analyze_log(defective_log))
    assert "NaN" not in text and "Infinity" not in text


def test_terminal_output_is_plain_when_color_is_off(defective_log):
    text = render_terminal(analyze_log(defective_log), color=False)
    assert "\033[" not in text
    assert "FLIGHT LOG HEALTH REPORT" in text
    assert "ACTION:" in text


def test_terminal_output_can_be_truncated(defective_log):
    text = render_terminal(analyze_log(defective_log), color=False, max_findings=3)
    assert "[ 3]" in text
    assert "[ 4]" not in text


def test_html_is_self_contained_and_every_svg_parses(defective_log):
    html = render_html(analyze_log(defective_log), defective_log)
    assert html.startswith("<!DOCTYPE html>")
    assert "<script" not in html
    assert "<link" not in html
    assert 'src="http' not in html
    import re

    svgs = re.findall(r"<svg.*?</svg>", html, re.S)
    assert len(svgs) >= 3
    for svg in svgs:
        ET.fromstring(svg)


def test_html_escapes_finding_text(defective_log):
    report = analyze_log(defective_log)
    report.findings.insert(
        0, _finding(title="<script>alert(1)</script>", severity=Severity.INFO)
    )
    html = render_html(report, defective_log)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_write_html_and_json_produce_readable_files(defective_log, tmp_path):
    report = analyze_log(defective_log)
    h = write_html(report, str(tmp_path / "r.html"), defective_log)
    j = write_json(report, str(tmp_path / "r.json"))
    assert open(h, encoding="utf-8").read().startswith("<!DOCTYPE html>")
    assert json.load(open(j, encoding="utf-8"))["findings"]


def test_a_failing_analyzer_degrades_to_one_warning_not_a_crash(clean_log):
    def boom(_log):
        raise RuntimeError("synthetic failure")

    report = build_report(clean_log, analyzers=[("boom", boom)])
    assert len(report.findings) == 1
    assert report.findings[0].severity is Severity.WARNING
    assert "failed to run" in report.findings[0].title


def test_report_summary_carries_log_metadata(defective_log):
    report = analyze_log(defective_log)
    assert report.summary["duration_s"] > 0
    assert report.summary["channels"] > 20
    assert report.summary["log_format"] == "synthetic"
