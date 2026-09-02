# flight-log-analyzer

**"The drone flew badly. Why?"**

That question usually gets answered by scrolling through a plot viewer, spotting
something that looks wrong, and guessing. This tool answers it from the log
itself: it parses PX4 ULog and ArduPilot dataflash logs, runs six analyzers over
the normalised data, and produces a ranked list of findings — each one with the
evidence, the mechanism in plain English, and a concrete thing to change.

It is built around one opinion: **most flight problems are misattributed.**
"EKF variance" is usually vibration. "Bad GPS" is usually vibration. "Needs a
retune" is often a bent prop or a sagging pack. So findings are ranked
cause-before-symptom — vibration and power are reported ahead of the estimator
and controller symptoms they produce — and the report says why.

---

## One-command demo

No log file needed, no optional dependencies, nothing to configure:

```bash
pip install numpy
python3 tools/flightlog-analyze --demo --html report.html
```

That generates a synthetic flight with known injected defects, analyzes it, and
writes a self-contained HTML report. It is also how the test suite proves the
analyzers actually detect what they claim to.

---

## What the output looks like

```
============================================================================================
FLIGHT LOG HEALTH REPORT
============================================================================================
vehicle      : quadrotor
firmware     : synthetic-1.0
log format   : synthetic
duration     : 60.0 s
channels     : 65

VERDICT: 12 critical issue(s) found. Top: Accelerometer clipping: 18 events
         12 critical  5 warning  3 info

[ 1] CRITICAL Accelerometer clipping: 18 events
     analyzer=vibration  confidence=1.00  t = 4.0s .. 59.0s
     The accelerometer hit its measurement range limit and the samples were truncated. A
     clipped waveform is asymmetric, so its mean is no longer zero: the estimator receives a
     constant false acceleration. This is the mechanism behind unexplained altitude climb in
     altitude-hold and sudden position-estimate jumps. Clipping means the vibration is not
     merely noisy, it is off the scale of the sensor.
     ACTION: Do not fly again until this is fixed. Replace the props, re-check every motor
       for bearing damage, and soften the FC mount only if it is currently rigid
       (hard-mounted). Confirm the clip counters stay at zero on the next hover.
     evidence: clip_counts={clip0=6, clip1=6, clip2=6}  total=18

[ 2] CRITICAL Motor output asymmetry (single motor, 15%)
     analyzer=vibration  confidence=1.00  t = 4.0s .. 59.0s
     motor.0 ran 15% above the average of the other motors for the whole flight. One motor
     working harder than its neighbours means it is producing less thrust per unit of
     command: a damaged or bent prop, a weak or dragging motor, or a bent arm.
     ACTION: Swap the prop on motor.0 with a known-good one and re-fly a hover. If the
       asymmetry follows the prop, it was the prop; if it stays with the position, check
       that motor's bearings and the arm for a bend.
     evidence: means={motor.0=0.595, motor.1=0.49, motor.2=0.488, motor.3=0.49}
       worst_motor=motor.0  worst_deviation=0.1534  spread=0.1068

[ 3] CRITICAL Brownout risk: cell voltage fell to 3.27 V under load
     analyzer=power  confidence=1.00  t = 7.0s .. 11.0s
     Minimum pack voltage was 19.65 V at t=9.0s, which is 3.27 V per cell across 6 cells.
     Below 3.30 V per cell the flight controller's regulator loses headroom. When it drops
     out the FC reboots in flight and the log simply ends -- which is exactly what an
     unexplained fall out of the sky looks like in a log file.
     ACTION: Stop flying this pack. Then, in order: land at a higher voltage (raise the
       low-battery failsafe threshold), fit a pack with a higher C rating or more capacity,
       and check every connector between the pack and the FC.
     evidence: v_min=19.65  v_max=22.55  v_min_cell=3.275  t_v_min=9

[13] WARNING  Roll oscillation at 14.0 Hz (1.7 deg) - rate p high
     analyzer=control  confidence=1.00  t = 4.0s .. 59.0s
     The roll tracking error contains a 1.7 degree oscillation at 14.0 Hz, standing 34096x
     above the surrounding noise floor. Oscillation in the 8-20 Hz band is rate P set too
     high. This is the most common over-tune. It is often inaudible from a distance but
     shows clearly as ripple on the gyro trace, and it costs flight time because the motors
     are constantly correcting.
     ACTION: Reduce rate P by 25-30% on the affected axis (PX4 MC_ROLLRATE_P /
       MC_PITCHRATE_P; ArduPilot ATC_RAT_RLL_P / ATC_RAT_PIT_P) and re-fly. Do not raise D
       to damp it -- that moves the problem to a higher frequency.
     evidence: axis=roll  freq_hz=14  amplitude_deg=1.672  classification=rate_p_high
```

The HTML report contains the same findings plus inline SVG plots — spectra with
the detected peaks annotated, per-motor output bars, the mode timeline, and the
time window each finding refers to shaded on the trace. It is a single file with
no external references, so it opens on a machine with no internet and nothing
installed.

---

## Usage

```bash
# analyze a PX4 log, write both report formats
flightlog-analyze flight.ulg --html report.html --json report.json

# ArduPilot dataflash
flightlog-analyze flight.bin --html report.html

# a CSV export from any tool (needs no optional dependencies)
flightlog-analyze exported.csv

# pipe JSON into something else
flightlog-analyze flight.bin --quiet --json - | jq '.findings[0].action'

# which readers are available in this environment
flightlog-analyze --formats
```

Exit code is `0` when nothing critical was found, `1` when something was, and
`2` when the log could not be read — so it drops straight into CI or a
post-flight script.

As a library:

```python
from flightlog import load, analyze_log, write_html

log = load("flight.ulg")
report = analyze_log(log)

for finding in report.findings:
    print(finding.severity.value, finding.title)
    print("  ->", finding.action)

write_html(report, "report.html", log)
```

---

## Supported log formats

| Format | Extensions | Reader | Dependency | Required? |
|---|---|---|---|---|
| PX4 ULog | `.ulg`, `.ulog` | `readers/ulog_reader.py` | `pyulog` | optional |
| ArduPilot dataflash | `.bin`, `.log`, `.px4log` | `readers/dataflash_reader.py` | `pymavlink` | optional |
| CSV / TSV export | `.csv`, `.tsv`, `.txt` | `readers/csv_reader.py` | stdlib | — |
| Synthetic (`--demo`) | — | `readers/synthetic.py` | stdlib | — |

Both optional imports are guarded. Without them the package still imports, the
whole test suite still passes, and `--demo` still runs; only that one file
format is unavailable, and the error names the exact `pip install` command.

The CSV path exists because every tool in the ecosystem can export CSV
(Flight Review, MAVExplorer, UAV Log Viewer, `ulog2csv`,
`mavlogdump.py --format csv`), so there is always a way to get a report even
when installing a parser is not an option. Headers are matched loosely:
`AccX`, `accel_x`, `Accel X (m/s2)` and `accel.x` all resolve to the same
channel, and `TimeUS` / `timestamp` / `time` are scaled correctly.

---

## What each analyzer looks for

| Analyzer | Detects |
|---|---|
| **vibration** | Welch-averaged accelerometer spectra, dominant peaks mapped to a source (prop imbalance, blade pass, bearing harmonic, frame resonance, soft-mount resonance), broadband RMS against field thresholds, clipping events, per-motor output asymmetry |
| **power** | Pack internal resistance from a V-vs-I fit (with the state-of-charge trend removed), per-cell voltage under load, brownout risk, capacity/coulomb-count disagreement, throttle-to-voltage correlation |
| **control** | Setpoint-vs-actual tracking error on roll/pitch/yaw/altitude, oscillation frequency mapped to the specific gain that is too high, integrator saturation windows, motor saturation and the one-motor-pinned desync signature |
| **ekf** | Innovation test-ratio excursions per channel, GPS position jumps cross-checked against innovations, magnetometer distortion (and whether it tracks throttle), baro-vs-GPS height drift, EKF reset events |
| **gps** | Fix-type timeline, satellite count, HDOP, time to first fix |
| **modes** | Flight-mode timeline, arm/disarm, failsafe events, RC signal loss, logs that end while still armed |

The reasoning behind every threshold, plus a symptom index ("altitude drifts",
"toilet-bowling", "sudden fall out of the sky") mapping complaints to log
signatures, is in **[docs/INTERPRETING_LOGS.md](docs/INTERPRETING_LOGS.md)**.

---

## Architecture

```
src/flightlog/
  types.py            FlightLog, Series, Finding, Severity -- the shared model
  channels.py         canonical channel names + ULog/dataflash/CSV mapping tables
  readers/
    ulog_reader.py       PX4 .ulg          (pyulog, guarded)
    dataflash_reader.py  ArduPilot .bin    (pymavlink, guarded)
    csv_reader.py        CSV/TSV           (stdlib only)
    synthetic.py         generator with injectable defects
  analysis/
    spectral.py       hand-rolled Welch PSD, peak finding, band power
    vibration.py  power.py  control.py  ekf.py  gps.py  modes.py
  svgplot.py          small SVG plotter -- no matplotlib in the report path
  report.py           ranking + terminal / HTML / JSON renderers
  cli.py              argument parsing and exit codes
```

Three deliberate choices:

**Readers normalise into one object.** Analyzers never import `pyulog` or
`pymavlink` — they only see a `FlightLog` of named SI-unit time-series. Adding a
log format means writing one mapping table, not touching any analyzer.

**Welch is implemented on bare numpy.** Not to avoid scipy for its own sake, but
because flight logs are *not* uniformly sampled. A raw FFT of jittery timestamps
produces a smeared spectrum and wrong peak frequencies. Everything here resamples
onto a uniform grid first (using the *median* interval, so a logging dropout does
not shift the whole frequency axis) and reports the rate it used.

**The report path has no plotting dependency.** `svgplot.py` builds SVG by hand,
so generating a report needs numpy and nothing else — which matters when the
machine is a field laptop or a Jetson.

---

## Tests

```bash
pip install numpy pytest
python3 -m pytest -q
```

169 tests across 14 files, fully offline, no network and no real log files.
They are written to prove detection rather than to exercise code paths:

- inject a 92 Hz vibration tone → assert the peak is found within 2 Hz, that its
  amplitude matches the injected RMS, and that it is classified as prop imbalance
- inject 48 mΩ of pack resistance → assert the V/I fit recovers it within 20%
  and that the brownout warning fires
- inject a GPS glitch at t = 25 s → assert it is flagged at that timestamp, and
  that the outbound and recovery jumps merge into one event
- inject a 6.5 Hz roll oscillation → assert the dominant frequency is identified
  and mapped to the attitude-P band
- **a clean synthetic flight must produce zero critical findings and zero
  warnings** — tested explicitly, because a diagnostic tool that flags a healthy
  aircraft is worse than no tool
- the SVG plotter's output parses as XML, escapes hostile labels, and contains
  no external references
- findings rank critical → warning → info, and cause before symptom within a
  severity level

The ArduPilot reader is exercised against a hand-built text dataflash log, and
those tests skip cleanly when `pymavlink` is not installed.

---

## What this is / isn't

**It is:** a log-analysis library and CLI with a defensible opinion about which
findings matter and in what order, honest about its own uncertainty, and
runnable with numpy alone.

**It isn't:**

- **A replacement for reading the log yourself** on anything unusual. It reports
  what it can measure; a human still owns the diagnosis.
- **Able to identify a damaged prop without RPM telemetry.** Without RPM it
  classifies vibration peaks by frequency band, which distinguishes "prop
  imbalance" from "frame resonance" with real but limited confidence. The report
  says so, and tells you which parameter to change to fix that for next time.
- **Able to separate wind from a bent airframe.** Sustained attitude bias and
  integrator saturation look identical in a log for both causes. Wind is not
  logged.
- **Better than the current sensor it is fed.** Consumed mAh, remaining
  percentage and the fitted internal resistance all inherit the calibration error
  of the current sensor, which is often 20% out of the box.
- **A trend tool.** It analyzes one flight. Most mechanical problems are far more
  visible as a trend across flights than in any single log.
- **A tuning autopilot.** It tells you which gain is too high and by roughly how
  much to reduce it. It does not fly the aircraft to find the new value.

Thresholds are module-level constants with the reasoning documented next to them.
If your airframe is unusual, override them at the call site rather than trusting
defaults chosen for a conventional multirotor.

---

## Related repositories

- **[drone-control-toolkit](https://github.com/Pratyush150/drone-control-toolkit)**
  — the PID/LQR/EKF implementations and sim harness the oscillation-frequency →
  gain mapping in `analysis/control.py` comes from.
- **[px4-mavlink-companion](https://github.com/Pratyush150/px4-mavlink-companion)**
  — MAVLink bridge and link watchdog between the flight controller and a
  companion computer, for pulling logs off the aircraft in the first place.

## License

MIT — see [LICENSE](LICENSE).
