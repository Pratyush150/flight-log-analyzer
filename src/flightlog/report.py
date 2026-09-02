"""Report generation: aggregate every analyzer into one ranked answer.

A pile of numbers is not a diagnosis.  What a client needs, in order:

1. **What is wrong**, ranked, with the worst thing first.
2. **The evidence** -- actual values and timestamps, so they can check the
   claim against their own log viewer rather than taking it on faith.
3. **Why it matters**, in plain English, with the mechanism spelled out.
4. **What to do about it**, concretely enough to act on this weekend.

Every :class:`~flightlog.types.Finding` carries all four.  This module puts
them in order and renders them three ways: a coloured terminal summary, a
self-contained HTML file with inline SVG plots, and JSON for anything
downstream.

Ranking rules
-------------
Findings sort by severity first (critical, warning, info), then by confidence,
then by a fixed analyzer precedence.  The analyzer precedence is not arbitrary:
vibration comes before the estimator because vibration *causes* estimator
problems, and telling someone to retune their EKF when their props are
unbalanced sends them in the wrong direction for a week.
"""

from __future__ import annotations

import datetime as _dt
import html
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import svgplot
from .analysis import ANALYZERS
from .analysis.vibration import analyze_axes
from .types import FlightLog, Finding, Severity, _jsonify

__all__ = [
    "Report",
    "ANALYZER_PRECEDENCE",
    "build_report",
    "rank_findings",
    "render_terminal",
    "render_html",
    "render_json",
    "write_html",
    "write_json",
]

#: Causal ordering. Lower index = closer to a root cause, so it is reported
#: first among findings of equal severity.
ANALYZER_PRECEDENCE: Dict[str, int] = {
    "vibration": 0,
    "power": 1,
    "control": 2,
    "ekf": 3,
    "gps": 4,
    "modes": 5,
}

_SEVERITY_COLOR = {
    Severity.CRITICAL: "\033[1;31m",
    Severity.WARNING: "\033[1;33m",
    Severity.INFO: "\033[1;36m",
}
_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"

_SEVERITY_HTML = {
    Severity.CRITICAL: ("#b00020", "#fdecef"),
    Severity.WARNING: ("#a86400", "#fff6e5"),
    Severity.INFO: ("#00628a", "#e9f4fa"),
}


@dataclass
class Report:
    """A finished analysis: the log summary plus ranked findings."""

    findings: List[Finding]
    summary: Dict[str, Any]
    generated_at: str = ""
    source: str = ""
    channel_index: List[Dict[str, Any]] = field(default_factory=list)

    def by_severity(self, severity: Severity) -> List[Finding]:
        return [f for f in self.findings if f.severity is severity]

    @property
    def counts(self) -> Dict[str, int]:
        return {
            s.value: len(self.by_severity(s))
            for s in (Severity.CRITICAL, Severity.WARNING, Severity.INFO)
        }

    @property
    def verdict(self) -> str:
        """One-line headline: the first thing a reader should see."""
        crit = self.by_severity(Severity.CRITICAL)
        warn = self.by_severity(Severity.WARNING)
        if crit:
            return f"{len(crit)} critical issue(s) found. Top: {crit[0].title}"
        if warn:
            return f"No critical issues. {len(warn)} warning(s). Top: {warn[0].title}"
        return "No critical or warning findings. This log looks healthy."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "source": self.source,
            "summary": _jsonify(self.summary),
            "verdict": self.verdict,
            "counts": self.counts,
            "findings": [f.to_dict() for f in self.findings],
            "channels": _jsonify(self.channel_index),
        }


def rank_findings(findings: Sequence[Finding]) -> List[Finding]:
    """Order findings worst-first.

    Sort key: severity descending, then confidence descending, then analyzer
    precedence ascending, then title for stability.  The last term matters --
    without it the order of two equally severe findings depends on dict
    iteration order, and a report that reshuffles between runs is impossible to
    diff.
    """
    return sorted(
        findings,
        key=lambda f: (
            -f.severity.rank,
            -float(f.confidence),
            ANALYZER_PRECEDENCE.get(f.analyzer, 99),
            f.title,
        ),
    )


def build_report(
    log: FlightLog,
    analyzers: Optional[Sequence[Tuple[str, Callable[[FlightLog], List[Finding]]]]] = None,
    source: str = "",
    generated_at: Optional[str] = None,
) -> Report:
    """Run every analyzer over ``log`` and return a ranked :class:`Report`.

    An analyzer that raises is caught and turned into a warning finding rather
    than being allowed to abort the report.  A malformed channel in one topic
    should cost you that one section, not the whole diagnosis.
    """
    analyzers = analyzers if analyzers is not None else ANALYZERS
    findings: List[Finding] = []
    for name, fn in analyzers:
        try:
            findings.extend(fn(log))
        except Exception as exc:  # pragma: no cover - defensive
            findings.append(
                Finding(
                    analyzer=name,
                    severity=Severity.WARNING,
                    title=f"Analyzer '{name}' failed to run",
                    explanation=(
                        f"The {name} analyzer raised {type(exc).__name__}: {exc}. The rest "
                        "of the report is unaffected, but this section is missing."
                    ),
                    action=(
                        f"Check that the channels the {name} analyzer needs are present and "
                        "sane in this log. Re-run with --verbose for the traceback."
                    ),
                    evidence={"exception": type(exc).__name__, "message": str(exc)},
                )
            )

    channel_index = [
        {
            "name": s.name,
            "units": s.units,
            "samples": len(s),
            "rate_hz": round(s.sample_rate, 1),
            "source": s.source,
        }
        for s in (log.series[k] for k in sorted(log.series))
    ]
    return Report(
        findings=rank_findings(findings),
        summary=log.summary(),
        generated_at=generated_at
        or _dt.datetime.now().replace(microsecond=0).isoformat(sep=" "),
        source=source or str(log.metadata.get("source", "")),
        channel_index=channel_index,
    )


# ---------------------------------------------------------------------------
# terminal
# ---------------------------------------------------------------------------


def _supports_color(stream_isatty: bool) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return stream_isatty


def _wrap(
    text: str, width: int, indent: str, subsequent_indent: Optional[str] = None
) -> List[str]:
    """Greedy word wrap.

    Kept local rather than using :mod:`textwrap` so that long parameter names
    such as ``ATC_RAT_RLL_P`` are never broken across lines -- a parameter name
    split in half is a parameter name the reader cannot paste into a ground
    station.
    """
    sub = indent if subsequent_indent is None else subsequent_indent
    words = text.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        pad = indent if not lines else sub
        candidate = f"{cur} {w}".strip()
        if len(candidate) + len(pad) > width and cur:
            lines.append(pad + cur)
            cur = w
        else:
            cur = candidate
    if cur:
        lines.append((indent if not lines else sub) + cur)
    return lines


def render_terminal(
    report: Report, color: bool = True, width: int = 92, max_findings: int = 0
) -> str:
    """Render a coloured terminal summary.

    ``color=False`` produces plain text with no escape sequences, which is what
    the tests assert on and what you want when piping to a file.
    """
    use_color = color and _supports_color(True)

    def c(code: str, text: str) -> str:
        return f"{code}{text}{_RESET}" if use_color else text

    out: List[str] = []
    rule = "=" * width
    out.append(rule)
    out.append(c(_BOLD, "FLIGHT LOG HEALTH REPORT"))
    out.append(rule)

    s = report.summary
    meta = [
        f"vehicle      : {s.get('vehicle', 'unknown')}",
        f"firmware     : {s.get('firmware', 'unknown')}",
        f"log format   : {s.get('log_format', 'unknown')}",
        f"duration     : {s.get('duration_s', 0)} s",
        f"channels     : {s.get('channels', 0)}",
        f"generated    : {report.generated_at}",
    ]
    if report.source:
        meta.append(f"source       : {report.source}")
    out.extend(meta)
    out.append("")

    counts = report.counts
    badge = (
        f"{c(_SEVERITY_COLOR[Severity.CRITICAL], str(counts['critical']) + ' critical')}  "
        f"{c(_SEVERITY_COLOR[Severity.WARNING], str(counts['warning']) + ' warning')}  "
        f"{c(_SEVERITY_COLOR[Severity.INFO], str(counts['info']) + ' info')}"
    )
    out.append(f"VERDICT: {report.verdict}")
    out.append(f"         {badge}")
    out.append("")

    findings = report.findings
    if max_findings:
        findings = findings[:max_findings]

    for i, f in enumerate(findings, 1):
        tag = f.severity.value.upper().ljust(8)
        head = f"[{i:>2}] {tag} {f.title}"
        out.append(c(_SEVERITY_COLOR[f.severity], head))
        span = ""
        if f.t_start is not None and f.t_end is not None:
            span = f"  t = {f.t_start:.1f}s .. {f.t_end:.1f}s"
        elif f.t_start is not None:
            span = f"  t = {f.t_start:.1f}s"
        out.append(c(_DIM, f"     analyzer={f.analyzer}  confidence={f.confidence:.2f}{span}"))
        out.extend(_wrap(f.explanation, width, "     "))
        action_lines = _wrap(f.action, width, "     ACTION: ", "       ")
        out.append(c(_BOLD, action_lines[0][:12]) + action_lines[0][12:])
        out.extend(action_lines[1:])
        ev = _format_evidence(f.evidence)
        if ev:
            out.append(c(_DIM, "     evidence: " + ev))
        out.append("")

    out.append(rule)
    out.append(
        "Findings are ranked worst-first, and by cause before symptom: fix vibration "
        "and power before touching estimator or controller settings."
    )
    out.append(rule)
    return "\n".join(out)


def _format_evidence(evidence: Dict[str, Any], limit: int = 5) -> str:
    """Compact one-line evidence rendering for the terminal."""
    if not evidence:
        return ""
    parts: List[str] = []
    for k, v in list(evidence.items())[:limit]:
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        elif isinstance(v, (int, str, bool)):
            parts.append(f"{k}={v}")
        elif isinstance(v, dict):
            inner = ", ".join(
                f"{ik}={iv:.3g}" if isinstance(iv, float) else f"{ik}={iv}"
                for ik, iv in list(v.items())[:4]
            )
            parts.append(f"{k}={{{inner}}}")
        elif isinstance(v, (list, tuple)):
            parts.append(f"{k}=[{len(v)} item(s)]")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def render_json(report: Report, indent: int = 2) -> str:
    """Serialise the report to JSON."""
    return json.dumps(report.to_dict(), indent=indent, sort_keys=False)


def write_json(report: Report, path: str, indent: int = 2) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_json(report, indent))
    return path


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: #f5f6f7; color: #1c1c1c;
  font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.wrap { max-width: 980px; margin: 0 auto; padding: 28px 20px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 16px; margin: 34px 0 12px; padding-bottom: 6px; border-bottom: 1px solid #dcdfe3; }
.sub { color: #5a6270; font-size: 13px; margin: 0 0 20px; }
.meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px; background: #fff; border: 1px solid #dcdfe3; border-radius: 8px; padding: 14px 16px; }
.meta div { font-size: 13px; }
.meta span { display: block; color: #6b7280; font-size: 11px; text-transform: uppercase;
  letter-spacing: .04em; }
.verdict { margin: 18px 0 6px; padding: 14px 16px; border-radius: 8px; border: 1px solid;
  font-weight: 600; }
.badges { margin: 10px 0 6px; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12px;
  font-weight: 600; margin-right: 8px; border: 1px solid; }
.finding { background: #fff; border: 1px solid #dcdfe3; border-left-width: 4px;
  border-radius: 8px; padding: 16px 18px; margin: 14px 0; }
.finding h3 { margin: 0 0 6px; font-size: 15px; }
.tag { font-size: 11px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
  padding: 2px 8px; border-radius: 4px; margin-right: 8px; vertical-align: 2px; }
.trace { color: #6b7280; font-size: 12px; margin: 0 0 10px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.action { background: #f0f7f2; border-left: 3px solid #0f7b4f; padding: 10px 12px;
  border-radius: 0 6px 6px 0; margin: 12px 0 0; }
.action b { color: #0f7b4f; }
table.ev { border-collapse: collapse; margin-top: 10px; font-size: 12px; width: 100%;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
table.ev td { border-top: 1px solid #eceef1; padding: 4px 8px; vertical-align: top; }
table.ev td.k { color: #6b7280; white-space: nowrap; width: 210px; }
.plot { background: #fff; border: 1px solid #dcdfe3; border-radius: 8px; margin: 12px 0;
  padding: 8px; overflow-x: auto; }
.channels { columns: 3; column-gap: 22px; font-size: 12px; color: #4b5563;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
footer { margin-top: 40px; padding-top: 14px; border-top: 1px solid #dcdfe3;
  color: #6b7280; font-size: 12px; }
@media (max-width: 720px) { .channels { columns: 1; } }
"""


def _plot_for_finding(log: FlightLog, finding: Finding, vib_cache: Dict[str, Any]) -> str:
    """Render the plot a finding asked for, or an empty string."""
    spec = finding.plot or {}
    kind = spec.get("kind")
    try:
        if kind == "spectrum":
            axis = str(spec.get("axis", "x"))
            result = vib_cache.get("result")
            if result is None:
                return ""
            match = next((a for a in result.axes if a.axis == axis), None)
            if match is None or match.spectrum.freq.size == 0:
                return ""
            peaks = [(p.freq, p.amplitude) for p in match.peaks]
            return svgplot.spectrum_plot(
                match.spectrum.freq,
                match.spectrum.psd,
                title=f"accel.{axis} spectrum (Welch, {match.spectrum.n_segments} segments, "
                f"{match.spectrum.resolution:.2f} Hz bins)",
                peaks=peaks,
            )
        if kind == "series":
            channels = [c for c in spec.get("channels", []) if log.get(c) is not None]
            if not channels:
                return ""
            specs = []
            for i, name in enumerate(channels):
                s = log.series[name]
                specs.append(
                    svgplot.SeriesSpec(
                        f"{name} [{s.units}]" if s.units else name,
                        s.time,
                        s.values,
                        dashed=name.endswith("_sp"),
                    )
                )
            highlight = None
            if finding.t_start is not None and finding.t_end is not None:
                highlight = (finding.t_start, finding.t_end)
            return svgplot.line_plot(
                specs, title=" / ".join(channels), highlight=highlight
            )
        if kind == "motors":
            motors = [s for s in log.matching("motor.") if len(s) > 5]
            if not motors:
                return ""
            t0, t1 = log.flight_window()
            labels, values = [], []
            for m in motors:
                v = m.slice_time(t0, t1).values
                v = v[np.isfinite(v) & (v > 0.05)]
                if v.size:
                    labels.append(m.name)
                    values.append(float(np.mean(v)))
            if not values:
                return ""
            return svgplot.bar_plot(
                labels,
                values,
                title="Mean motor output in the flight window",
                y_label="output (0-1)",
                reference=float(np.mean(values)),
            )
        if kind == "modes":
            from .analysis.modes import mode_timeline

            markers = [
                (e.time, e.kind) for e in log.events_of("arm", "disarm", "failsafe")
            ][:12]
            return svgplot.timeline_plot(
                mode_timeline(log), markers, title="Flight mode timeline"
            )
    except Exception:  # pragma: no cover - plotting must never break a report
        return ""
    return ""


def _overview_plots(log: FlightLog, vib_cache: Dict[str, Any]) -> List[Tuple[str, str]]:
    """The three plots worth showing regardless of what was found."""
    out: List[Tuple[str, str]] = []
    alt_channels = [c for c in ("alt.ekf", "alt.baro", "alt.gps") if log.get(c) is not None]
    if alt_channels:
        out.append(
            (
                "Altitude",
                svgplot.line_plot(
                    [
                        svgplot.SeriesSpec(c, log.series[c].time, log.series[c].values)
                        for c in alt_channels
                    ],
                    title="Altitude sources",
                    y_label="m",
                ),
            )
        )
    if log.get("bat.voltage") is not None:
        specs = [
            svgplot.SeriesSpec(
                "bat.voltage [V]", log.series["bat.voltage"].time, log.series["bat.voltage"].values
            )
        ]
        cur = log.get("bat.current")
        if cur is not None:
            # Scale current onto the voltage axis so both fit one plot; the
            # legend states the scaling so nobody misreads the numbers.
            v = log.series["bat.voltage"].values
            scale = (np.nanmax(v) - np.nanmin(v)) / max(np.nanmax(cur.values), 1e-6) or 1.0
            specs.append(
                svgplot.SeriesSpec(
                    f"bat.current [A x {scale:.3f} + offset]",
                    cur.time,
                    cur.values * scale + float(np.nanmin(v)),
                    dashed=True,
                )
            )
        out.append(("Power", svgplot.line_plot(specs, title="Battery", y_label="V")))
    result = vib_cache.get("result")
    if result is not None and result.axes:
        worst = max(result.axes, key=lambda a: a.rms)
        if worst.spectrum.freq.size:
            out.append(
                (
                    "Vibration spectrum",
                    svgplot.spectrum_plot(
                        worst.spectrum.freq,
                        worst.spectrum.psd,
                        title=f"accel.{worst.axis} spectrum "
                        f"(worst axis, RMS {worst.rms:.1f} m/s^2)",
                        peaks=[(p.freq, p.amplitude) for p in worst.peaks],
                    ),
                )
            )
    return out


def _evidence_table(evidence: Dict[str, Any]) -> str:
    if not evidence:
        return ""
    rows: List[str] = []
    for k, v in evidence.items():
        if isinstance(v, float):
            text = f"{v:.6g}"
        elif isinstance(v, (dict, list, tuple)):
            text = json.dumps(_jsonify(v), separators=(", ", ": "))
            if len(text) > 400:
                text = text[:397] + "..."
        else:
            text = str(v)
        rows.append(
            f'<tr><td class="k">{html.escape(str(k))}</td>'
            f"<td>{html.escape(text)}</td></tr>"
        )
    return '<table class="ev">' + "".join(rows) + "</table>"


def render_html(report: Report, log: Optional[FlightLog] = None, title: str = "") -> str:
    """Render a self-contained HTML report.

    Everything is inline: CSS, SVG, values.  No external stylesheet, no script
    tag, no font download, no network access at any point.  The file works from
    a USB stick on a machine that has never seen this package.
    """
    vib_cache: Dict[str, Any] = {}
    if log is not None:
        try:
            vib_cache["result"] = analyze_axes(log)
        except Exception:  # pragma: no cover - defensive
            vib_cache["result"] = None

    doc_title = title or "Flight log health report"
    s = report.summary
    parts: List[str] = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{html.escape(doc_title)}</title>",
        f"<style>{_CSS}</style></head><body><div class=\"wrap\">",
        f"<h1>{html.escape(doc_title)}</h1>",
        f'<p class="sub">Generated {html.escape(report.generated_at)}'
        + (f" &middot; {html.escape(report.source)}" if report.source else "")
        + "</p>",
    ]

    meta_items = [
        ("vehicle", str(s.get("vehicle", "unknown"))),
        ("firmware", str(s.get("firmware", "unknown"))),
        ("log format", str(s.get("log_format", "unknown"))),
        ("duration", f"{s.get('duration_s', 0)} s"),
        ("channels", str(s.get("channels", 0))),
        ("events", str(s.get("events", 0))),
    ]
    parts.append('<div class="meta">')
    for k, v in meta_items:
        parts.append(f"<div><span>{html.escape(k)}</span>{html.escape(v)}</div>")
    parts.append("</div>")

    counts = report.counts
    top_sev = (
        Severity.CRITICAL
        if counts["critical"]
        else (Severity.WARNING if counts["warning"] else Severity.INFO)
    )
    fg, bg = _SEVERITY_HTML[top_sev]
    parts.append(
        f'<div class="verdict" style="color:{fg};background:{bg};border-color:{fg}33">'
        f"{html.escape(report.verdict)}</div>"
    )
    parts.append('<div class="badges">')
    for sev in (Severity.CRITICAL, Severity.WARNING, Severity.INFO):
        f_, b_ = _SEVERITY_HTML[sev]
        parts.append(
            f'<span class="badge" style="color:{f_};background:{b_};border-color:{f_}44">'
            f"{counts[sev.value]} {sev.value}</span>"
        )
    parts.append("</div>")

    parts.append("<h2>Findings</h2>")
    if not report.findings:
        parts.append("<p>No findings were produced. The log may be missing key channels.</p>")
    for i, f in enumerate(report.findings, 1):
        fg, bg = _SEVERITY_HTML[f.severity]
        span = ""
        if f.t_start is not None and f.t_end is not None:
            span = f" &middot; t = {f.t_start:.1f}s .. {f.t_end:.1f}s"
        elif f.t_start is not None:
            span = f" &middot; t = {f.t_start:.1f}s"
        parts.append(f'<div class="finding" style="border-left-color:{fg}">')
        parts.append(
            f"<h3><span class=\"tag\" style=\"color:{fg};background:{bg}\">"
            f"{html.escape(f.severity.value)}</span>{i}. {html.escape(f.title)}</h3>"
        )
        parts.append(
            f'<p class="trace">analyzer={html.escape(f.analyzer)} &middot; '
            f"confidence={f.confidence:.2f}{span}</p>"
        )
        parts.append(f"<p>{html.escape(f.explanation)}</p>")
        parts.append(
            f'<div class="action"><b>Recommended action:</b> {html.escape(f.action)}</div>'
        )
        parts.append(_evidence_table(f.evidence))
        if log is not None and f.plot:
            svg = _plot_for_finding(log, f, vib_cache)
            if svg:
                parts.append(f'<div class="plot">{svg}</div>')
        parts.append("</div>")

    if log is not None:
        overview = _overview_plots(log, vib_cache)
        if overview:
            parts.append("<h2>Overview</h2>")
            for name, svg in overview:
                parts.append(f'<div class="plot">{svg}</div>')

    if report.channel_index:
        parts.append("<h2>Channels in this log</h2>")
        parts.append('<div class="channels">')
        for ch in report.channel_index:
            unit = f" [{ch['units']}]" if ch.get("units") else ""
            parts.append(
                f"<div>{html.escape(str(ch['name']))}{html.escape(unit)} "
                f"&middot; {ch['samples']} @ {ch['rate_hz']} Hz</div>"
            )
        parts.append("</div>")

    parts.append(
        "<footer>Generated by flight-log-analyzer. Findings are ranked worst-first and "
        "cause-before-symptom: vibration and power problems are reported ahead of the "
        "estimator and controller symptoms they produce. Thresholds used are documented "
        "in docs/INTERPRETING_LOGS.md.</footer>"
    )
    parts.append("</div></body></html>")
    return "".join(parts)


def write_html(
    report: Report, path: str, log: Optional[FlightLog] = None, title: str = ""
) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_html(report, log, title))
    return path
