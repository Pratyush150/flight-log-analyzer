"""Spectral primitives: Welch PSD, peak finding, dominant frequency."""

from __future__ import annotations

import numpy as np
import pytest

from flightlog.analysis.spectral import (
    band_power,
    dominant_frequency,
    find_peaks,
    hann,
    highpass_rms,
    uniform_resample,
    welch_psd,
)


def _tone(freq, amp, fs=250.0, duration=40.0, offset=0.0, noise=0.0, seed=0):
    t = np.arange(0, duration, 1.0 / fs)
    rng = np.random.default_rng(seed)
    x = offset + amp * np.sin(2 * np.pi * freq * t)
    if noise:
        x = x + rng.normal(0, noise, t.size)
    return t, x


def test_hann_is_periodic_and_unit_peak():
    w = hann(8)
    assert w[0] == pytest.approx(0.0)
    assert np.max(w) == pytest.approx(1.0)
    assert len(hann(1)) == 1


def test_welch_finds_the_injected_frequency():
    t, x = _tone(92.0, 5.0)
    spec = welch_psd(t, x)
    assert spec.n_segments > 1, "Welch must average multiple segments"
    peak_freq = spec.freq[int(np.argmax(spec.psd))]
    assert peak_freq == pytest.approx(92.0, abs=1.0)


def test_welch_detrend_removes_gravity_offset():
    """Accelerometer Z sits at about -9.81 m/s^2. Without detrending that DC
    term dominates the spectrum and leaks across the low end."""
    t, x = _tone(60.0, 2.0, offset=-9.81)
    spec = welch_psd(t, x, detrend=True)
    dc_power = spec.psd[0]
    tone_power = np.max(spec.psd[spec.freq > 5])
    assert dc_power < tone_power * 1e-3


def test_welch_on_short_input_returns_empty_not_error():
    spec = welch_psd(np.arange(4) * 0.01, np.arange(4.0))
    assert spec.freq.size == 0
    assert spec.psd.size == 0


def test_peak_amplitude_recovers_sinusoid_rms():
    """A 6 m/s^2 sinusoid has an RMS of 6/sqrt(2) = 4.24."""
    t, x = _tone(80.0, 6.0, noise=0.4, seed=3)
    peaks = find_peaks(welch_psd(t, x))
    assert peaks, "a 6 m/s^2 tone in 0.4 noise must be detected"
    assert peaks[0].freq == pytest.approx(80.0, abs=1.0)
    assert peaks[0].amplitude == pytest.approx(6.0 / np.sqrt(2), rel=0.20)


def test_peak_finding_ignores_pure_noise():
    rng = np.random.default_rng(11)
    t = np.arange(0, 40, 1 / 250.0)
    peaks = find_peaks(welch_psd(t, rng.normal(0, 1.0, t.size)), min_prominence_db=8.0)
    assert len(peaks) <= 1, f"white noise should not yield peaks, got {len(peaks)}"


def test_two_separate_tones_are_both_found():
    t = np.arange(0, 40, 1 / 250.0)
    x = 4.0 * np.sin(2 * np.pi * 40 * t) + 3.0 * np.sin(2 * np.pi * 95 * t)
    peaks = find_peaks(welch_psd(t, x), max_peaks=4)
    found = sorted(round(p.freq) for p in peaks)
    assert 40 in found and 95 in found


def test_peak_search_stops_below_nyquist_by_default():
    """The top of the band is shaped by anti-alias filtering, so peaks above
    80% of Nyquist are not trustworthy and are excluded unless asked for."""
    t = np.arange(0, 40, 1 / 250.0)
    x = 5.0 * np.sin(2 * np.pi * 115.0 * t)
    assert find_peaks(welch_psd(t, x)) == []
    peaks = find_peaks(welch_psd(t, x), fmax=124.0)
    assert peaks and peaks[0].freq == pytest.approx(115.0, abs=1.0)


def test_dominant_frequency_respects_band_limits():
    t = np.arange(0, 40, 1 / 200.0)
    x = np.sin(2 * np.pi * 3.0 * t) + 5.0 * np.sin(2 * np.pi * 70.0 * t)
    f_low, _ = dominant_frequency(t, x, fmin=0.5, fmax=10.0)
    assert f_low == pytest.approx(3.0, abs=0.3)


def test_highpass_rms_rejects_low_frequency_motion():
    """Real flight motion lives below a few Hz; vibration lives above."""
    t = np.arange(0, 40, 1 / 250.0)
    slow = 8.0 * np.sin(2 * np.pi * 0.5 * t)
    fast = 2.0 * np.sin(2 * np.pi * 90.0 * t)
    assert highpass_rms(t, slow, cutoff=5.0) < 0.5
    assert highpass_rms(t, slow + fast, cutoff=5.0) == pytest.approx(
        2.0 / np.sqrt(2), rel=0.15
    )


def test_uniform_resample_uses_median_rate():
    t = np.concatenate([np.arange(0, 1, 0.01), np.arange(2.0, 3.0, 0.01)])
    y, fs = uniform_resample(t, np.zeros(t.size))
    assert fs == pytest.approx(100.0, rel=0.01)
    assert y.size > 200


def test_band_power_is_positive_and_localised():
    t, x = _tone(50.0, 3.0)
    spec = welch_psd(t, x)
    inside = band_power(spec, 45, 55)
    outside = band_power(spec, 100, 110)
    assert inside > 0
    assert inside > outside * 100
