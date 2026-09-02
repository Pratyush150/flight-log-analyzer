"""The SVG plotter must emit well-formed, self-contained XML."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from flightlog import svgplot
from flightlog.svgplot import (
    SeriesSpec,
    Style,
    bar_plot,
    line_plot,
    spectrum_plot,
    timeline_plot,
)

_SVG_NS = "{http://www.w3.org/2000/svg}"


def _parse(svg: str) -> ET.Element:
    """Parse and assert the root element is a namespaced <svg>."""
    root = ET.fromstring(svg)
    assert root.tag == _SVG_NS + "svg"
    return root


def test_line_plot_is_well_formed_svg():
    t = np.linspace(0, 10, 500)
    svg = line_plot(
        [SeriesSpec("alt", t, np.sin(t) * 5), SeriesSpec("sp", t, np.sin(t) * 5 + 0.4)],
        title="altitude",
        y_label="m",
    )
    root = _parse(svg)
    polylines = root.findall(f".//{_SVG_NS}polyline")
    assert len(polylines) == 2
    assert all(p.get("points") for p in polylines)


def test_spectrum_plot_annotates_peaks():
    freq = np.linspace(0, 125, 512)
    psd = np.exp(-((freq - 92.0) ** 2) / 4.0) + 0.001
    svg = spectrum_plot(freq, psd, title="accel.x", peaks=[(92.0, 3.4)])
    root = _parse(svg)
    texts = [e.text for e in root.findall(f".//{_SVG_NS}text") if e.text]
    assert "92 Hz" in texts
    assert root.findall(f".//{_SVG_NS}circle")


def test_bar_plot_draws_one_rect_per_bar_plus_background():
    svg = bar_plot(
        ["motor.0", "motor.1", "motor.2", "motor.3"],
        [0.60, 0.49, 0.49, 0.49],
        title="motors",
        reference=0.5175,
    )
    root = _parse(svg)
    rects = root.findall(f".//{_SVG_NS}rect")
    assert len(rects) >= 5  # background + plot frame + 4 bars
    texts = [e.text for e in root.findall(f".//{_SVG_NS}text") if e.text]
    assert "motor.0" in texts


def test_timeline_plot_renders_intervals_and_markers():
    intervals = [
        {"mode": "STABILIZED", "t_start": 0.0, "t_end": 20.0},
        {"mode": "POSCTL", "t_start": 20.0, "t_end": 60.0},
    ]
    svg = timeline_plot(intervals, markers=[(4.0, "arm"), (59.0, "disarm")], title="modes")
    root = _parse(svg)
    texts = [e.text for e in root.findall(f".//{_SVG_NS}text") if e.text]
    assert "POSCTL" in texts
    assert "arm" in texts


def test_special_characters_in_labels_are_escaped():
    """A channel called ``a<b & c"`` must not break the document."""
    t = np.linspace(0, 1, 20)
    svg = line_plot([SeriesSpec('a<b & c"', t, t)], title="<script>alert(1)</script>")
    root = _parse(svg)  # would raise if the escaping were wrong
    assert "<script>" not in svg
    texts = [e.text for e in root.findall(f".//{_SVG_NS}text") if e.text]
    assert 'a<b & c"' in texts


def test_empty_input_renders_a_placeholder_not_a_crash():
    for svg in (
        line_plot([]),
        spectrum_plot(np.zeros(0), np.zeros(0)),
        bar_plot([], []),
        timeline_plot([]),
    ):
        root = _parse(svg)
        texts = [e.text for e in root.findall(f".//{_SVG_NS}text") if e.text]
        assert any("no " in (t or "") for t in texts)


def test_large_series_is_decimated_but_keeps_its_extremes():
    """Naive subsampling would hide the spike; min/max decimation keeps it."""
    t = np.linspace(0, 60, 60_000)
    y = np.zeros_like(t)
    y[30_000] = 99.0
    svg = line_plot([SeriesSpec("accel.z", t, y)])
    root = _parse(svg)
    points = root.find(f".//{_SVG_NS}polyline").get("points")
    assert len(points.split()) <= svgplot.MAX_POINTS + 2
    # The spike must survive: it defines the y range, so it maps to the top of
    # the plot box.
    ys = [float(p.split(",")[1]) for p in points.split()]
    assert min(ys) < Style().pad_top + 12


def test_flat_series_does_not_divide_by_zero():
    t = np.linspace(0, 5, 100)
    svg = line_plot([SeriesSpec("const", t, np.full(t.size, 3.3))])
    root = _parse(svg)
    assert root.find(f".//{_SVG_NS}polyline") is not None


def test_nan_values_are_skipped_not_rendered():
    t = np.linspace(0, 5, 100)
    y = np.sin(t)
    y[40:50] = np.nan
    svg = line_plot([SeriesSpec("gappy", t, y)])
    root = _parse(svg)
    assert "nan" not in svg.lower()
    assert root.find(f".//{_SVG_NS}polyline") is not None


def test_output_has_no_external_references():
    """A report must open with no network access, from a USB stick."""
    t = np.linspace(0, 5, 50)
    svg = line_plot([SeriesSpec("x", t, t)], title="local only")
    for token in ("http://", "https://", "<image", "xlink:href"):
        if token == "http://":
            # the xmlns declaration is the one allowed occurrence
            assert svg.count(token) == 1
        else:
            assert token not in svg
