"""Spectral primitives implemented on bare numpy.

Why hand-roll Welch instead of importing ``scipy.signal.welch``?  Two reasons.

1. Dependency weight.  A log analyzer that a client can ``pip install`` with
   only numpy is far easier to drop onto a field laptop or a Jetson than one
   that drags in scipy.
2. Control over resampling.  Flight logs are *not* uniformly sampled.  A raw
   ``rfft`` of a series whose timestamps jitter by 20% produces a smeared
   spectrum and peak frequencies that are simply wrong.  Everything here
   resamples onto a uniform grid first and reports the grid rate it used, so
   a caller can sanity-check the frequency axis.

All functions are pure and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# numpy renamed trapz -> trapezoid in 2.0; support both so the package works
# on the older numpy that ships with many Jetson/JetPack images.
_trapz = getattr(np, "trapezoid", None) or np.trapz

__all__ = [
    "Spectrum",
    "Peak",
    "uniform_resample",
    "hann",
    "welch_psd",
    "find_peaks",
    "band_power",
    "dominant_frequency",
    "highpass_rms",
]


@dataclass
class Spectrum:
    """One-sided power spectral density.

    Attributes
    ----------
    freq:
        Frequency bins in Hz, ``freq[0] == 0``.
    psd:
        Power spectral density in ``units^2/Hz``.
    fs:
        Uniform sample rate actually used (Hz).
    nperseg:
        Segment length used for the Welch average.
    n_segments:
        Number of averaged segments.  One segment means no averaging, which
        makes the noise floor unreliable -- peak detection thresholds account
        for this.
    """

    freq: np.ndarray
    psd: np.ndarray
    fs: float
    nperseg: int
    n_segments: int

    @property
    def resolution(self) -> float:
        """Frequency bin width in Hz."""
        return float(self.fs / self.nperseg) if self.nperseg else 0.0

    def amplitude(self) -> np.ndarray:
        """Approximate per-bin amplitude in the original units.

        ``sqrt(psd * df)`` recovers RMS-per-bin, which is what a person means
        when they ask "how big is the 92 Hz peak, in m/s^2?".
        """
        return np.sqrt(np.maximum(self.psd, 0.0) * self.resolution)


@dataclass
class Peak:
    """A detected spectral peak."""

    freq: float
    power: float
    amplitude: float
    prominence: float
    bandwidth: float = 0.0

    def to_dict(self) -> dict:
        return {
            "freq_hz": round(self.freq, 2),
            "amplitude": round(self.amplitude, 4),
            "prominence_db": round(self.prominence, 2),
            "bandwidth_hz": round(self.bandwidth, 2),
        }


def uniform_resample(
    t: np.ndarray, x: np.ndarray, fs: Optional[float] = None
) -> Tuple[np.ndarray, float]:
    """Resample ``x(t)`` onto a uniform grid.

    Returns ``(values, fs)``.  If ``fs`` is not given, the median sample rate
    of ``t`` is used -- median, not mean, so that logging dropouts do not drag
    the assumed rate down and shift every frequency in the result.
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    if t.size < 2:
        return x, 0.0
    order = np.argsort(t)
    t, x = t[order], x[order]
    good = np.isfinite(t) & np.isfinite(x)
    t, x = t[good], x[good]
    if t.size < 2:
        return x, 0.0
    if fs is None:
        dt = np.diff(t)
        dt = dt[dt > 0]
        if dt.size == 0:
            return x, 0.0
        fs = float(1.0 / np.median(dt))
    n = max(2, int(round((t[-1] - t[0]) * fs)) + 1)
    grid = t[0] + np.arange(n) / fs
    return np.interp(grid, t, x), float(fs)


def hann(n: int) -> np.ndarray:
    """Periodic Hann window.

    Periodic (``/ n``) rather than symmetric (``/ (n-1)``) because these
    windows feed an FFT, where the periodic form is the one that does not
    leak an extra half-bin.
    """
    if n <= 1:
        return np.ones(max(n, 1))
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def welch_psd(
    t: np.ndarray,
    x: np.ndarray,
    fs: Optional[float] = None,
    nperseg: Optional[int] = None,
    overlap: float = 0.5,
    detrend: bool = True,
) -> Spectrum:
    """Welch-averaged one-sided PSD.

    Parameters
    ----------
    t, x:
        Timestamps (seconds) and samples.  Need not be uniformly spaced.
    fs:
        Force a resampling rate.  Default: median rate of ``t``.
    nperseg:
        Segment length in samples.  Default: the largest power of two that
        still yields at least 4 averaged segments, clamped to
        ``[256, 4096]``.  Segment length is the frequency-resolution knob:
        1024 samples at 250 Hz gives ~0.24 Hz bins, which is plenty to
        separate a 92 Hz prop peak from a 96 Hz one.
    overlap:
        Fractional segment overlap, 0.5 is the standard Welch choice.
    detrend:
        Remove the per-segment mean.  Essential for accelerometer data, where
        the Z axis sits at about -9.81 m/s^2 and would otherwise dump all its
        energy into the DC bin and leak across the low end of the spectrum.

    Returns an empty :class:`Spectrum` (zero-length arrays) if there is not
    enough data, rather than raising -- a short log should degrade to "no
    result", not to a crash in the middle of a report.
    """
    y, rate = uniform_resample(t, x, fs)
    n = y.size
    if n < 16 or rate <= 0:
        return Spectrum(np.zeros(0), np.zeros(0), rate, 0, 0)

    if nperseg is None:
        nperseg = 1
        while nperseg * 2 <= n // 4 and nperseg < 4096:
            nperseg *= 2
        nperseg = max(256, nperseg)
    nperseg = int(min(nperseg, n))
    if nperseg < 16:
        return Spectrum(np.zeros(0), np.zeros(0), rate, 0, 0)

    step = max(1, int(nperseg * (1.0 - overlap)))
    win = hann(nperseg)
    # Window power normalisation so the PSD is density, not raw magnitude.
    scale = 1.0 / (rate * np.sum(win**2))

    starts = list(range(0, n - nperseg + 1, step))
    if not starts:
        starts = [0]
    acc = np.zeros(nperseg // 2 + 1)
    for s in starts:
        seg = y[s : s + nperseg]
        if detrend:
            seg = seg - np.mean(seg)
        spec = np.fft.rfft(seg * win)
        p = (np.abs(spec) ** 2) * scale
        # One-sided: double everything except DC and (for even n) Nyquist.
        p[1:-1] *= 2.0
        acc += p
    acc /= len(starts)
    freq = np.fft.rfftfreq(nperseg, d=1.0 / rate)
    return Spectrum(freq, acc, rate, nperseg, len(starts))


def find_peaks(
    spec: Spectrum,
    fmin: float = 1.0,
    fmax: Optional[float] = None,
    max_peaks: int = 6,
    min_prominence_db: float = 6.0,
) -> List[Peak]:
    """Locate prominent spectral peaks between ``fmin`` and ``fmax``.

    "Prominence" here is measured in dB above a *local* noise floor: the
    median PSD in a window around the candidate.  A global threshold does not
    work on flight logs, because real spectra fall off steeply with frequency
    and a fixed level would either miss a 400 Hz motor peak entirely or
    declare the whole low end a peak.

    ``fmin`` defaults to 1 Hz to skip residual DC/drift.  ``fmax`` defaults to
    80% of Nyquist, since anti-alias filtering makes the top of the band
    untrustworthy.
    """
    if spec.freq.size == 0:
        return []
    nyq = spec.fs / 2.0
    if fmax is None:
        fmax = 0.8 * nyq
    band = (spec.freq >= fmin) & (spec.freq <= fmax)
    if not np.any(band):
        return []
    f = spec.freq[band]
    p = spec.psd[band]
    if f.size < 5:
        return []

    # Local noise floor: rolling median over roughly a 10% relative window,
    # floored so log10 never sees zero.
    win = max(5, (f.size // 20) | 1)
    floor = _rolling_median(p, win)
    floor = np.maximum(floor, np.max(p) * 1e-9 + 1e-30)
    ratio_db = 10.0 * np.log10(np.maximum(p, 1e-30) / floor)

    cand: List[Peak] = []
    for i in range(1, f.size - 1):
        if p[i] <= p[i - 1] or p[i] < p[i + 1]:
            continue
        if ratio_db[i] < min_prominence_db:
            continue
        fq, pw = _parabolic_refine(f, p, i)
        amp = _peak_rms(f, p, floor, i)
        cand.append(Peak(fq, float(pw), amp, float(ratio_db[i]), _half_power_width(f, p, i)))

    cand.sort(key=lambda pk: pk.power, reverse=True)
    # Suppress near-duplicates. A real rotating source wanders in frequency as
    # throttle changes, so its peak arrives as a cluster of adjacent lobes
    # rather than one line. Merging within 3% of the peak frequency (or three
    # bins, whichever is larger) reports it as the single source it is, instead
    # of three separate "findings" at 91.6, 92.0 and 92.4 Hz.
    keep: List[Peak] = []
    for pk in cand:
        tol = max(spec.resolution * 3.0, 0.03 * pk.freq)
        if all(abs(pk.freq - k.freq) > max(tol, 0.03 * k.freq) for k in keep):
            keep.append(pk)
        if len(keep) >= max_peaks:
            break
    return keep


def _rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    """Edge-padded rolling median (odd window)."""
    if win <= 1 or x.size <= win:
        return np.full(x.shape, float(np.median(x)) if x.size else 0.0)
    half = win // 2
    padded = np.pad(x, (half, half), mode="edge")
    strided = np.lib.stride_tricks.sliding_window_view(padded, win)
    return np.median(strided, axis=-1)


def _peak_rms(f: np.ndarray, p: np.ndarray, floor: np.ndarray, i: int) -> float:
    """RMS amplitude of the tone under a peak, in the original units.

    A single bin's amplitude systematically under-reports a real vibration
    tone: propeller RPM wanders with throttle, so the energy spreads over
    several bins and no one bin holds it all. Integrating the PSD across the
    whole peak -- out to where it falls back into the local noise floor --
    recovers the quantity a person actually wants: "how many m/s^2 is that
    92 Hz tone?".

    For a sinusoid of amplitude A the integral equals A^2/2, so the square root
    is the RMS amplitude, which is the convention used everywhere else in this
    package (and the one ArduPilot's VIBE fields use).
    """
    n = f.size
    cutoff = np.maximum(floor * 2.0, p[i] * 0.02)
    lo = i
    while lo > 0 and p[lo - 1] <= p[lo] and p[lo - 1] > cutoff[lo - 1]:
        lo -= 1
    hi = i
    while hi < n - 1 and p[hi + 1] <= p[hi] and p[hi + 1] > cutoff[hi + 1]:
        hi += 1
    if hi <= lo:
        df = float(f[1] - f[0]) if n > 1 else 1.0
        return float(np.sqrt(max(p[i], 0.0) * df))
    return float(np.sqrt(max(_trapz(p[lo : hi + 1], f[lo : hi + 1]), 0.0)))


def _parabolic_refine(f: np.ndarray, p: np.ndarray, i: int) -> Tuple[float, float]:
    """Sub-bin peak location by fitting a parabola to three log-power points.

    Without this, a peak reported from a 0.24 Hz grid is quantised to the grid
    and a genuine 92.0 Hz prop tone gets reported as 91.8 or 92.2 depending on
    where the bins landed.
    """
    if i <= 0 or i >= f.size - 1:
        return float(f[i]), float(p[i])
    y0, y1, y2 = (np.log(max(v, 1e-30)) for v in (p[i - 1], p[i], p[i + 1]))
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-18:
        return float(f[i]), float(p[i])
    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -0.5, 0.5))
    df = float(f[1] - f[0])
    return float(f[i] + delta * df), float(p[i])


def _half_power_width(f: np.ndarray, p: np.ndarray, i: int) -> float:
    """Width of the peak at half power (-3 dB), in Hz.

    Wide peaks (>15 Hz) read as broadband structural resonance; narrow ones
    (<5 Hz) read as a discrete rotating source such as a prop or bell.
    """
    half = p[i] * 0.5
    lo = i
    while lo > 0 and p[lo] > half:
        lo -= 1
    hi = i
    while hi < f.size - 1 and p[hi] > half:
        hi += 1
    return float(f[hi] - f[lo])


def band_power(spec: Spectrum, f0: float, f1: float) -> float:
    """Integrated power in ``[f0, f1]`` (units^2)."""
    if spec.freq.size == 0:
        return 0.0
    m = (spec.freq >= f0) & (spec.freq <= f1)
    if not np.any(m):
        return 0.0
    return float(_trapz(spec.psd[m], spec.freq[m]))


def dominant_frequency(
    t: np.ndarray, x: np.ndarray, fmin: float = 0.3, fmax: float = 60.0
) -> Tuple[float, float]:
    """Return ``(freq_hz, amplitude)`` of the strongest tone in a band.

    Used for oscillation detection on tracking-error signals, where the whole
    question is "at what frequency is it ringing?".  Returns ``(0.0, 0.0)`` if
    the signal is too short or the band is empty.
    """
    spec = welch_psd(t, x)
    if spec.freq.size == 0:
        return 0.0, 0.0
    m = (spec.freq >= fmin) & (spec.freq <= min(fmax, 0.9 * spec.fs / 2.0))
    if not np.any(m):
        return 0.0, 0.0
    idx = int(np.argmax(spec.psd[m]))
    f_sel = spec.freq[m]
    p_sel = spec.psd[m]
    fq, pw = _parabolic_refine(f_sel, p_sel, idx) if 0 < idx < f_sel.size - 1 else (
        float(f_sel[idx]),
        float(p_sel[idx]),
    )
    return fq, float(np.sqrt(max(pw, 0.0) * spec.resolution))


def highpass_rms(t: np.ndarray, x: np.ndarray, cutoff: float = 5.0) -> float:
    """RMS of ``x`` above ``cutoff`` Hz, computed in the frequency domain.

    This is the quantity ArduPilot reports as ``VIBE.VibeX/Y/Z``: the vehicle's
    actual attitude motion lives below a few Hz, so anything above the cutoff
    is frame/prop vibration rather than flight.  Doing it spectrally avoids
    designing and warming up a time-domain filter.
    """
    y, fs = uniform_resample(t, x)
    if y.size < 16 or fs <= 0:
        return 0.0
    y = y - np.mean(y)
    spec = np.fft.rfft(y)
    freq = np.fft.rfftfreq(y.size, d=1.0 / fs)
    spec[freq < cutoff] = 0.0
    filtered = np.fft.irfft(spec, n=y.size)
    return float(np.sqrt(np.mean(filtered**2)))
