"""Flight-mode, arming and failsafe timeline.

This analyzer produces very few findings, and that is intentional.  Its main
job is to build the timeline that gives every *other* finding its context.  A
vibration peak that only appears in ``POSCTL`` and not in ``STABILIZED`` is a
different problem from one that is always present; a voltage sag two seconds
after an ``AUTO.RTL`` transition is the RTL climb, not a failing pack.

The findings it does raise are the ones where the mode timeline itself is the
evidence:

* **failsafe triggered** -- the aircraft decided something was wrong. Whatever
  else is in the report, the firmware's own opinion is worth surfacing first.
* **RC signal loss** -- distinct from a failsafe, because a brief RC dropout
  that recovers before the failsafe timer expires leaves no failsafe event but
  is still the reason the aircraft twitched.
* **very short flight** -- a log that arms and disarms within a few seconds is
  usually a failed takeoff, and every downstream analyzer will be working with
  too little data to say anything useful. Better to say so plainly.
* **still armed at end of log** -- the log stopped while the aircraft was
  flying. That is a crash, a brownout, or an SD card that filled up. It is
  never nothing.
"""

from __future__ import annotations

from typing import Dict, List

from ..types import FlightLog, Finding, ModeInterval, Severity

__all__ = ["mode_timeline", "rc_signal_loss", "analyze"]

#: Minimum armed duration (s) below which analysis results are unreliable.
MIN_USEFUL_FLIGHT_S = 8.0
#: RSSI fraction below which the RC link is considered lost.
RSSI_LOSS_THRESHOLD = 0.15


def mode_timeline(log: FlightLog) -> List[Dict[str, object]]:
    """Flight-mode intervals as plain dicts, sorted by time."""
    intervals: List[ModeInterval] = list(log.modes)
    if not intervals:
        mode = log.get("mode.id")
        if mode is not None and len(mode) > 1:
            vals = mode.values
            start, cur = float(mode.time[0]), vals[0]
            for i in range(1, vals.size):
                if vals[i] != cur:
                    intervals.append(ModeInterval(start, float(mode.time[i]), f"MODE_{int(cur)}"))
                    start, cur = float(mode.time[i]), vals[i]
            intervals.append(ModeInterval(start, float(mode.time[-1]), f"MODE_{int(cur)}"))
    return [
        {
            "mode": iv.mode,
            "t_start": round(iv.start, 2),
            "t_end": round(iv.end, 2),
            "duration_s": round(iv.duration, 2),
        }
        for iv in sorted(intervals, key=lambda m: m.start)
    ]


def rc_signal_loss(
    log: FlightLog, threshold: float = RSSI_LOSS_THRESHOLD, min_duration: float = 0.2
) -> List[Dict[str, float]]:
    """Windows where the RC link was lost.

    Uses the explicit ``rc.link_lost`` flag when present, and falls back to an
    RSSI threshold otherwise.  Short dropouts matter even when no failsafe
    fires: the controller holds the last valid stick input during the gap, so a
    150 ms dropout during a stick movement leaves the aircraft executing a stale
    command.
    """
    out: List[Dict[str, float]] = []
    lost = log.get("rc.link_lost")
    rssi = log.get("rc.rssi")
    if lost is not None and len(lost) > 2:
        mask = lost.values > 0.5
        times = lost.time
    elif rssi is not None and len(rssi) > 2:
        mask = rssi.values < threshold
        times = rssi.time
    else:
        return out

    i = 0
    while i < mask.size:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < mask.size and mask[j + 1]:
            j += 1
        dur = float(times[j] - times[i])
        if dur >= min_duration:
            out.append(
                {"t_start": round(float(times[i]), 2), "t_end": round(float(times[j]), 2),
                 "duration_s": round(dur, 2)}
            )
        i = j + 1
    return out


def analyze(log: FlightLog, min_flight_s: float = MIN_USEFUL_FLIGHT_S) -> List[Finding]:
    """Run the mode/arming/failsafe analyzer."""
    findings: List[Finding] = []
    timeline = mode_timeline(log)
    arms = log.events_of("arm")
    disarms = log.events_of("disarm")
    intervals = log.armed_intervals

    # -- failsafes ---------------------------------------------------------
    failsafes = log.events_of("failsafe")
    if failsafes:
        findings.append(
            Finding(
                analyzer="modes",
                severity=Severity.CRITICAL,
                title=f"{len(failsafes)} failsafe event(s) triggered",
                explanation=(
                    "The firmware decided a safety condition was breached and took "
                    "control. Whatever else this report says, the aircraft's own opinion "
                    "of the flight is recorded here: "
                    + "; ".join(f"t={e.time:.1f}s {e.detail}" for e in failsafes[:5])
                    + "."
                ),
                action=(
                    "Work backwards from each failsafe timestamp. The cause is in the "
                    "seconds *before* it -- check the battery, GPS and RC sections of this "
                    "report at those times."
                ),
                evidence={
                    "count": len(failsafes),
                    "events": [
                        {"time": round(e.time, 2), "detail": e.detail} for e in failsafes[:10]
                    ],
                },
                t_start=failsafes[0].time,
                t_end=failsafes[-1].time,
            )
        )

    # -- RC loss -----------------------------------------------------------
    dropouts = rc_signal_loss(log)
    if dropouts:
        total = sum(float(d["duration_s"]) for d in dropouts)
        longest = max(float(d["duration_s"]) for d in dropouts)
        findings.append(
            Finding(
                analyzer="modes",
                severity=Severity.CRITICAL if longest > 1.0 else Severity.WARNING,
                title=f"RC link lost {len(dropouts)} time(s), longest {longest:.1f} s",
                explanation=(
                    f"The RC link dropped for a total of {total:.1f} s. During a dropout "
                    "the controller holds the last valid stick input, so the aircraft "
                    "keeps executing a stale command until the link returns or the "
                    "failsafe timer expires. Short dropouts that never trigger failsafe "
                    "are still the reason an aircraft 'twitched for no reason'."
                ),
                action=(
                    "Check receiver antenna placement -- two antennas at 90 degrees, "
                    "clear of carbon fibre and away from the video transmitter. Check "
                    "the receiver's telemetry-power setting is not overdriving a shared "
                    "supply. If dropouts correlate with distance or a specific heading, "
                    "it is antenna orientation, not the link budget."
                ),
                evidence={"dropouts": dropouts[:10], "total_s": round(total, 2)},
                t_start=float(dropouts[0]["t_start"]),
                t_end=float(dropouts[-1]["t_end"]),
                plot={"kind": "series", "channels": ["rc.rssi"]},
            )
        )

    # -- arming sanity -----------------------------------------------------
    if not arms:
        findings.append(
            Finding(
                analyzer="modes",
                severity=Severity.INFO,
                title="No arm event found in this log",
                explanation=(
                    "No arming event was recorded, so the analyzers used the whole log as "
                    "the analysis window. Ground handling and bench testing before or "
                    "after the flight are therefore included in every statistic in this "
                    "report."
                ),
                action=(
                    "If this log covers a real flight, check that the arming event stream "
                    "is being logged (PX4 vehicle_status; ArduPilot EV messages). Results "
                    "get noticeably sharper with a correct flight window."
                ),
                evidence={"duration_s": round(log.duration, 2)},
            )
        )
    else:
        flight_s = sum(b - a for a, b in intervals)
        if flight_s < min_flight_s:
            findings.append(
                Finding(
                    analyzer="modes",
                    severity=Severity.WARNING,
                    title=f"Very short armed time: {flight_s:.1f} s",
                    explanation=(
                        f"The aircraft was armed for only {flight_s:.1f} s in total. Every "
                        "spectral result in this report is computed from a short window, so "
                        "frequency resolution is poor and vibration numbers should be "
                        "treated as indicative rather than precise."
                    ),
                    action=(
                        "Fly a 60-second hover and re-run the analysis. A steady hover is "
                        "the single most useful log for diagnosis, because it removes pilot "
                        "input as a variable."
                    ),
                    evidence={
                        "armed_seconds": round(flight_s, 2),
                        "arm_count": len(arms),
                        "intervals": [
                            {"t_start": round(a, 2), "t_end": round(b, 2)} for a, b in intervals
                        ],
                    },
                )
            )
        if len(arms) > len(disarms):
            last_arm = max(e.time for e in arms)
            findings.append(
                Finding(
                    analyzer="modes",
                    severity=Severity.CRITICAL,
                    title="Log ends while still armed",
                    explanation=(
                        f"The aircraft armed at t={last_arm:.1f}s and the log ends without a "
                        "matching disarm. A log that stops mid-flight has three usual "
                        "causes: the flight controller lost power (brownout -- see the power "
                        "section), the SD card stopped accepting writes, or the aircraft hit "
                        "something hard enough to interrupt logging."
                    ),
                    action=(
                        "Check the power findings for voltage sag near the end of the log. "
                        "If voltage was healthy, reformat or replace the SD card -- a card "
                        "that stalls under write load ends logs exactly like this."
                    ),
                    evidence={
                        "last_arm_t": round(last_arm, 2),
                        "log_end_t": round(log.duration, 2),
                        "arm_events": len(arms),
                        "disarm_events": len(disarms),
                    },
                    t_start=last_arm,
                    t_end=log.duration,
                )
            )

    # -- timeline as an info finding, always present -----------------------
    findings.append(
        Finding(
            analyzer="modes",
            severity=Severity.INFO,
            title=f"Flight mode timeline: {len(timeline)} segment(s)",
            explanation=(
                "Mode, arm and failsafe timeline for the flight. Use it to place every "
                "other finding in context -- a symptom that appears only in one mode is "
                "a different problem from one that is always present."
                if timeline
                else "No flight-mode data was found in this log, so findings cannot be "
                "attributed to a particular mode."
            ),
            action=(
                "Cross-reference the timestamps of any critical finding against this "
                "timeline before changing anything."
            ),
            evidence={
                "modes": timeline[:20],
                "arm_events": [round(e.time, 2) for e in arms[:10]],
                "disarm_events": [round(e.time, 2) for e in disarms[:10]],
                "total_duration_s": round(log.duration, 2),
            },
            plot={"kind": "modes"},
        )
    )
    return findings
