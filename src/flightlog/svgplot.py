"""A minimal SVG plotter, written by hand.

Why not matplotlib?  Because the report path should have no plotting
dependency at all.  A client who is handed ``report.html`` opens one file in a
browser; a client who wants to *generate* one should not have to install a
50 MB scientific stack on a field laptop.  Everything here is string building
over numpy arrays, and the output is embedded directly in the HTML report -- no
external files, no base64 images, no network fetches.

The output is well-formed XML.  Every text value is escaped, every attribute is
quoted, and the root carries an explicit ``xmlns``, so the same string can be
inlined into HTML *or* saved as a standalone ``.svg``.  There is a test that
parses the output with :mod:`xml.etree.ElementTree` to keep it that way.

Plot types
----------
:func:`line_plot`      one or more time-series on a shared axis
:func:`spectrum_plot`  log-frequency PSD with annotated peaks
:func:`bar_plot`       categorical bars, used for per-motor output
:func:`timeline_plot`  labelled intervals, used for the flight-mode timeline
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape, quoteattr

import numpy as np

__all__ = [
    "Style",
    "SeriesSpec",
    "line_plot",
    "spectrum_plot",
    "bar_plot",
    "timeline_plot",
    "PALETTE",
]

#: Colour-blind-safe qualitative palette (Okabe-Ito), which also survives
#: printing in greyscale reasonably well.
PALETTE: Tuple[str, ...] = (
    "#0072b2",
    "#d55e00",
    "#009e73",
    "#cc79a7",
    "#e69f00",
    "#56b4e9",
    "#5c5c5c",
)

#: Maximum points drawn per series. Beyond this the polyline path gets large
#: enough to slow a browser down with no visible benefit, so the series is
#: min/max decimated -- which preserves spikes, unlike naive subsampling.
MAX_POINTS = 1400


@dataclass
class Style:
    """Colours and sizes for one figure."""

    width: int = 860
    height: int = 260
    pad_left: int = 62
    pad_right: int = 16
    pad_top: int = 28
    pad_bottom: int = 40
    bg: str = "#ffffff"
    grid: str = "#e6e6e6"
    axis: str = "#3a3a3a"
    text: str = "#222222"
    font: str = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
    font_size: int = 11
    title_size: int = 13

    @property
    def plot_w(self) -> int:
        return self.width - self.pad_left - self.pad_right

    @property
    def plot_h(self) -> int:
        return self.height - self.pad_top - self.pad_bottom


@dataclass
class SeriesSpec:
    """One line on a plot."""

    label: str
    x: np.ndarray
    y: np.ndarray
    color: Optional[str] = None
    dashed: bool = False
    width: float = 1.3


def _fmt(v: float) -> str:
    """Format a coordinate compactly without scientific notation surprises."""
    if not math.isfinite(v):
        return "0"
    if abs(v) >= 1000:
        return f"{v:.0f}"
    return f"{v:.2f}".rstrip("0").rstrip(".") or "0"


def _tick_label(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1000 or a < 0.01:
        return f"{v:.3g}"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _nice_ticks(lo: float, hi: float, count: int = 5) -> List[float]:
    """Choose round tick values covering ``[lo, hi]``.

    Standard 1/2/5 x 10^n selection.  Ticks at 0.37 and 0.74 make a reader
    work; ticks at 0.5 and 1.0 do not.
    """
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return [lo, hi] if hi > lo else [lo]
    raw = (hi - lo) / max(count, 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = mult * mag
        if raw <= step:
            break
    start = math.ceil(lo / step) * step
    ticks: List[float] = []
    v = start
    while v <= hi + step * 1e-9 and len(ticks) < 20:
        ticks.append(round(v, 10))
        v += step
    return ticks or [lo, hi]


def _decimate(x: np.ndarray, y: np.ndarray, max_points: int = MAX_POINTS) -> Tuple[np.ndarray, np.ndarray]:
    """Min/max decimation: keep the extremes of each bucket.

    Naive subsampling of a vibration trace hides exactly the spikes the reader
    needs to see.  Taking both the minimum and the maximum of each bucket keeps
    the visual envelope intact at a fraction of the point count.
    """
    n = x.size
    if n <= max_points:
        return x, y
    buckets = max_points // 2
    edges = np.linspace(0, n, buckets + 1).astype(int)
    xs: List[float] = []
    ys: List[float] = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = y[a:b]
        if not np.any(np.isfinite(seg)):
            continue
        i_min = a + int(np.nanargmin(seg))
        i_max = a + int(np.nanargmax(seg))
        lo, hi = (i_min, i_max) if i_min <= i_max else (i_max, i_min)
        xs.extend([float(x[lo]), float(x[hi])])
        ys.extend([float(y[lo]), float(y[hi])])
    return np.asarray(xs), np.asarray(ys)


def _open_svg(style: Style, title: str) -> List[str]:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {style.width} {style.height}" '
        f'width="100%" height={quoteattr(str(style.height))} '
        f'role="img" aria-label={quoteattr(title)} '
        f'font-family={quoteattr(style.font)}>',
        f'<rect x="0" y="0" width="{style.width}" height="{style.height}" fill="{style.bg}"/>',
    ]
    if title:
        parts.append(
            f'<text x="{style.pad_left}" y="{style.pad_top - 12}" '
            f'font-size="{style.title_size}" fill="{style.text}" '
            f'font-weight="600">{escape(title)}</text>'
        )
    return parts


def _axes(
    style: Style,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    x_label: str,
    y_label: str,
    log_x: bool = False,
) -> Tuple[List[str], object, object]:
    """Draw grid, ticks and axis labels; return the parts and the two mappers."""
    L, T, W, H = style.pad_left, style.pad_top, style.plot_w, style.plot_h

    if log_x:
        lx0 = math.log10(max(x0, 1e-6))
        lx1 = math.log10(max(x1, x0 * 1.0001 + 1e-6))

        def sx(v: float) -> float:
            return L + (math.log10(max(v, 1e-6)) - lx0) / max(lx1 - lx0, 1e-9) * W
    else:
        def sx(v: float) -> float:
            return L + (v - x0) / max(x1 - x0, 1e-9) * W

    def sy(v: float) -> float:
        return T + H - (v - y0) / max(y1 - y0, 1e-9) * H

    parts: List[str] = [
        f'<rect x="{L}" y="{T}" width="{W}" height="{H}" fill="none" '
        f'stroke="{style.axis}" stroke-width="0.8"/>'
    ]

    if log_x:
        decade_lo = math.floor(math.log10(max(x0, 1e-6)))
        decade_hi = math.ceil(math.log10(max(x1, 1e-6)))
        xticks = []
        for d in range(int(decade_lo), int(decade_hi) + 1):
            for m in (1, 2, 5):
                v = m * (10.0**d)
                if x0 <= v <= x1:
                    xticks.append(v)
    else:
        xticks = _nice_ticks(x0, x1)

    for v in xticks:
        px = sx(v)
        parts.append(
            f'<line x1="{_fmt(px)}" y1="{T}" x2="{_fmt(px)}" y2="{T+H}" '
            f'stroke="{style.grid}" stroke-width="0.7"/>'
        )
        parts.append(
            f'<text x="{_fmt(px)}" y="{T+H+15}" font-size="{style.font_size}" '
            f'fill="{style.text}" text-anchor="middle">{escape(_tick_label(v))}</text>'
        )

    for v in _nice_ticks(y0, y1):
        py = sy(v)
        parts.append(
            f'<line x1="{L}" y1="{_fmt(py)}" x2="{L+W}" y2="{_fmt(py)}" '
            f'stroke="{style.grid}" stroke-width="0.7"/>'
        )
        parts.append(
            f'<text x="{L-6}" y="{_fmt(py+3.5)}" font-size="{style.font_size}" '
            f'fill="{style.text}" text-anchor="end">{escape(_tick_label(v))}</text>'
        )

    if x_label:
        parts.append(
            f'<text x="{L+W/2}" y="{style.height-8}" font-size="{style.font_size}" '
            f'fill="{style.text}" text-anchor="middle">{escape(x_label)}</text>'
        )
    if y_label:
        parts.append(
            f'<text x="12" y="{T+H/2}" font-size="{style.font_size}" fill="{style.text}" '
            f'text-anchor="middle" transform="rotate(-90 12 {_fmt(T+H/2)})">'
            f"{escape(y_label)}</text>"
        )
    return parts, sx, sy


def _legend(style: Style, labels: Sequence[Tuple[str, str]]) -> List[str]:
    parts: List[str] = []
    x = style.pad_left + 6
    y = style.pad_top + 12
    for label, color in labels:
        parts.append(
            f'<rect x="{_fmt(x)}" y="{_fmt(y-8)}" width="10" height="3" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{_fmt(x+14)}" y="{_fmt(y)}" font-size="{style.font_size}" '
            f'fill="{style.text}">{escape(label)}</text>'
        )
        x += 16 + 7.0 * max(len(label), 1)
    return parts


def line_plot(
    series: Sequence[SeriesSpec],
    title: str = "",
    x_label: str = "time (s)",
    y_label: str = "",
    style: Optional[Style] = None,
    highlight: Optional[Tuple[float, float]] = None,
) -> str:
    """Render one or more time-series as an SVG string.

    ``highlight`` shades a time window -- used to mark the interval a finding
    refers to, so the reader's eye lands on the right part of the trace instead
    of hunting for a timestamp.
    """
    style = style or Style()
    clean = [
        SeriesSpec(
            s.label,
            np.asarray(s.x, dtype=float),
            np.asarray(s.y, dtype=float),
            s.color,
            s.dashed,
            s.width,
        )
        for s in series
        if len(np.asarray(s.x)) > 1
    ]
    parts = _open_svg(style, title)
    if not clean:
        parts.append(
            f'<text x="{style.width/2}" y="{style.height/2}" font-size="{style.font_size}" '
            f'fill="{style.text}" text-anchor="middle">no data</text>'
        )
        parts.append("</svg>")
        return "".join(parts)

    x0 = min(float(np.nanmin(s.x)) for s in clean)
    x1 = max(float(np.nanmax(s.x)) for s in clean)
    finite_y = [s.y[np.isfinite(s.y)] for s in clean]
    finite_y = [a for a in finite_y if a.size]
    if not finite_y:
        y0, y1 = 0.0, 1.0
    else:
        y0 = min(float(np.min(a)) for a in finite_y)
        y1 = max(float(np.max(a)) for a in finite_y)
    if y1 - y0 < 1e-9:
        y0, y1 = y0 - 0.5, y1 + 0.5
    margin = 0.06 * (y1 - y0)
    y0, y1 = y0 - margin, y1 + margin
    if x1 - x0 < 1e-9:
        x1 = x0 + 1.0

    axis_parts, sx, sy = _axes(style, x0, x1, y0, y1, x_label, y_label)
    parts.extend(axis_parts)

    if highlight is not None:
        hx0, hx1 = max(highlight[0], x0), min(highlight[1], x1)
        if hx1 > hx0:
            parts.append(
                f'<rect x="{_fmt(sx(hx0))}" y="{style.pad_top}" '
                f'width="{_fmt(sx(hx1)-sx(hx0))}" height="{style.plot_h}" '
                f'fill="#d55e00" fill-opacity="0.10"/>'
            )

    legend: List[Tuple[str, str]] = []
    for i, s in enumerate(clean):
        color = s.color or PALETTE[i % len(PALETTE)]
        xs, ys = _decimate(s.x, s.y)
        pts = " ".join(
            f"{_fmt(sx(px))},{_fmt(sy(py))}"
            for px, py in zip(xs, ys)
            if math.isfinite(py)
        )
        if not pts:
            continue
        dash = ' stroke-dasharray="5 3"' if s.dashed else ""
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="{s.width}" stroke-linejoin="round" '
            f'stroke-linecap="round"{dash}/>'
        )
        legend.append((s.label, color))

    parts.extend(_legend(style, legend))
    parts.append("</svg>")
    return "".join(parts)


def spectrum_plot(
    freq: np.ndarray,
    psd: np.ndarray,
    title: str = "",
    peaks: Sequence[Tuple[float, float]] = (),
    style: Optional[Style] = None,
    y_label: str = "amplitude (m/s^2)",
    fmin: float = 1.0,
) -> str:
    """Render a spectrum on a log frequency axis, with peaks annotated.

    Log frequency because vibration sources span two decades: an 8 Hz airframe
    mode and a 400 Hz bearing tone belong on the same picture, and on a linear
    axis the low end is unreadable.

    ``peaks`` are ``(frequency, amplitude)`` pairs; each gets a marker and a
    frequency label, because "there is a peak at 92 Hz" is the entire message
    of the plot.
    """
    style = style or Style(height=280)
    freq = np.asarray(freq, dtype=float)
    psd = np.asarray(psd, dtype=float)
    parts = _open_svg(style, title)
    mask = (freq >= fmin) & np.isfinite(psd)
    if freq.size < 4 or not np.any(mask):
        parts.append(
            f'<text x="{style.width/2}" y="{style.height/2}" font-size="{style.font_size}" '
            f'fill="{style.text}" text-anchor="middle">no spectrum available</text>'
        )
        parts.append("</svg>")
        return "".join(parts)

    f = freq[mask]
    amp = np.sqrt(np.maximum(psd[mask], 0.0))
    x0, x1 = float(f[0]), float(f[-1])
    y0, y1 = 0.0, float(np.max(amp)) * 1.12 or 1.0

    axis_parts, sx, sy = _axes(
        style, x0, x1, y0, y1, "frequency (Hz)", y_label, log_x=True
    )
    parts.extend(axis_parts)

    xs, ys = _decimate(f, amp)
    pts = " ".join(f"{_fmt(sx(px))},{_fmt(sy(py))}" for px, py in zip(xs, ys))
    parts.append(
        f'<polyline points="{pts}" fill="none" stroke="{PALETTE[0]}" stroke-width="1.2"/>'
    )

    for pf, pa in peaks:
        if pf < x0 or pf > x1:
            continue
        px, py = sx(pf), sy(pa)
        parts.append(
            f'<circle cx="{_fmt(px)}" cy="{_fmt(py)}" r="3.2" fill="{PALETTE[1]}"/>'
        )
        parts.append(
            f'<line x1="{_fmt(px)}" y1="{_fmt(py)}" x2="{_fmt(px)}" '
            f'y2="{style.pad_top + style.plot_h}" stroke="{PALETTE[1]}" '
            f'stroke-width="0.7" stroke-dasharray="3 3"/>'
        )
        parts.append(
            f'<text x="{_fmt(px)}" y="{_fmt(max(py - 8, style.pad_top + 10))}" '
            f'font-size="{style.font_size}" fill="{PALETTE[1]}" text-anchor="middle">'
            f"{escape(f'{pf:.0f} Hz')}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def bar_plot(
    labels: Sequence[str],
    values: Sequence[float],
    title: str = "",
    y_label: str = "",
    style: Optional[Style] = None,
    reference: Optional[float] = None,
) -> str:
    """Categorical bar chart, used for per-motor mean output.

    ``reference`` draws a dashed line at a value -- the mean across motors --
    so an asymmetric motor is visible at a glance rather than by reading four
    numbers.
    """
    style = style or Style(height=230)
    labels = list(labels)
    values = [float(v) for v in values]
    parts = _open_svg(style, title)
    if not labels:
        parts.append(
            f'<text x="{style.width/2}" y="{style.height/2}" font-size="{style.font_size}" '
            f'fill="{style.text}" text-anchor="middle">no data</text>'
        )
        parts.append("</svg>")
        return "".join(parts)

    y1 = max(values + ([reference] if reference is not None else []) + [1e-9]) * 1.15
    y0 = 0.0
    L, T, W, H = style.pad_left, style.pad_top, style.plot_w, style.plot_h

    def sy(v: float) -> float:
        return T + H - (v - y0) / max(y1 - y0, 1e-9) * H

    parts.append(
        f'<rect x="{L}" y="{T}" width="{W}" height="{H}" fill="none" '
        f'stroke="{style.axis}" stroke-width="0.8"/>'
    )
    for v in _nice_ticks(y0, y1):
        py = sy(v)
        parts.append(
            f'<line x1="{L}" y1="{_fmt(py)}" x2="{L+W}" y2="{_fmt(py)}" '
            f'stroke="{style.grid}" stroke-width="0.7"/>'
        )
        parts.append(
            f'<text x="{L-6}" y="{_fmt(py+3.5)}" font-size="{style.font_size}" '
            f'fill="{style.text}" text-anchor="end">{escape(_tick_label(v))}</text>'
        )

    slot = W / len(labels)
    bar_w = slot * 0.55
    for i, (label, value) in enumerate(zip(labels, values)):
        cx = L + slot * (i + 0.5)
        top = sy(value)
        color = PALETTE[i % len(PALETTE)]
        if reference is not None and reference > 0 and abs(value - reference) / reference > 0.06:
            color = PALETTE[1]
        parts.append(
            f'<rect x="{_fmt(cx-bar_w/2)}" y="{_fmt(top)}" width="{_fmt(bar_w)}" '
            f'height="{_fmt(T+H-top)}" fill="{color}" fill-opacity="0.85"/>'
        )
        parts.append(
            f'<text x="{_fmt(cx)}" y="{T+H+15}" font-size="{style.font_size}" '
            f'fill="{style.text}" text-anchor="middle">{escape(label)}</text>'
        )
        parts.append(
            f'<text x="{_fmt(cx)}" y="{_fmt(top-4)}" font-size="{style.font_size}" '
            f'fill="{style.text}" text-anchor="middle">'
            f"{escape(_tick_label(value))}</text>"
        )

    if reference is not None:
        py = sy(reference)
        parts.append(
            f'<line x1="{L}" y1="{_fmt(py)}" x2="{L+W}" y2="{_fmt(py)}" '
            f'stroke="{style.axis}" stroke-width="1" stroke-dasharray="6 4"/>'
        )
        parts.append(
            f'<text x="{L+W-4}" y="{_fmt(py-4)}" font-size="{style.font_size}" '
            f'fill="{style.axis}" text-anchor="end">mean</text>'
        )
    if y_label:
        parts.append(
            f'<text x="12" y="{T+H/2}" font-size="{style.font_size}" fill="{style.text}" '
            f'text-anchor="middle" transform="rotate(-90 12 {_fmt(T+H/2)})">'
            f"{escape(y_label)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def timeline_plot(
    intervals: Sequence[Dict[str, object]],
    markers: Sequence[Tuple[float, str]] = (),
    title: str = "",
    style: Optional[Style] = None,
) -> str:
    """Horizontal timeline of labelled intervals, plus point markers.

    Used for the flight-mode timeline with arm/disarm and failsafe markers, so
    a reader can see at a glance that (for instance) every EKF excursion in the
    report happened inside one ``POSCTL`` segment.
    """
    style = style or Style(height=150, pad_bottom=44, pad_top=30)
    parts = _open_svg(style, title)
    rows = [iv for iv in intervals if float(iv.get("t_end", 0)) > float(iv.get("t_start", 0))]
    if not rows:
        parts.append(
            f'<text x="{style.width/2}" y="{style.height/2}" font-size="{style.font_size}" '
            f'fill="{style.text}" text-anchor="middle">no mode data</text>'
        )
        parts.append("</svg>")
        return "".join(parts)

    x0 = min(float(iv["t_start"]) for iv in rows)
    x1 = max(float(iv["t_end"]) for iv in rows)
    for t, _ in markers:
        x0, x1 = min(x0, float(t)), max(x1, float(t))
    if x1 - x0 < 1e-9:
        x1 = x0 + 1.0
    L, T, W = style.pad_left, style.pad_top, style.plot_w

    def sx(v: float) -> float:
        return L + (v - x0) / max(x1 - x0, 1e-9) * W

    bar_top = T + 10
    bar_h = 30
    for i, iv in enumerate(rows):
        a, b = sx(float(iv["t_start"])), sx(float(iv["t_end"]))
        color = PALETTE[i % len(PALETTE)]
        parts.append(
            f'<rect x="{_fmt(a)}" y="{bar_top}" width="{_fmt(max(b-a, 1.0))}" '
            f'height="{bar_h}" fill="{color}" fill-opacity="0.75"/>'
        )
        if b - a > 44:
            parts.append(
                f'<text x="{_fmt((a+b)/2)}" y="{bar_top + bar_h/2 + 4}" '
                f'font-size="{style.font_size}" fill="#ffffff" text-anchor="middle">'
                f"{escape(str(iv.get('mode', '')))}</text>"
            )

    axis_y = bar_top + bar_h + 12
    parts.append(
        f'<line x1="{L}" y1="{axis_y}" x2="{L+W}" y2="{axis_y}" '
        f'stroke="{style.axis}" stroke-width="0.8"/>'
    )
    for v in _nice_ticks(x0, x1):
        px = sx(v)
        parts.append(
            f'<line x1="{_fmt(px)}" y1="{axis_y}" x2="{_fmt(px)}" y2="{axis_y+4}" '
            f'stroke="{style.axis}" stroke-width="0.8"/>'
        )
        parts.append(
            f'<text x="{_fmt(px)}" y="{axis_y+16}" font-size="{style.font_size}" '
            f'fill="{style.text}" text-anchor="middle">{escape(_tick_label(v))}</text>'
        )

    for t, label in markers:
        px = sx(float(t))
        parts.append(
            f'<line x1="{_fmt(px)}" y1="{bar_top-8}" x2="{_fmt(px)}" y2="{axis_y}" '
            f'stroke="#b00020" stroke-width="1.1"/>'
        )
        parts.append(
            f'<text x="{_fmt(px)}" y="{bar_top-11}" font-size="{style.font_size}" '
            f'fill="#b00020" text-anchor="middle">{escape(label)}</text>'
        )

    parts.append(
        f'<text x="{L+W/2}" y="{style.height-8}" font-size="{style.font_size}" '
        f'fill="{style.text}" text-anchor="middle">time (s)</text>'
    )
    parts.append("</svg>")
    return "".join(parts)
