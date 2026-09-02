"""Battery and power-system analysis.

The one number that matters: pack internal resistance
-----------------------------------------------------
A battery under load behaves, to first order, as an ideal source in series
with a resistance:

    V_measured = V_open_circuit - I * R_pack

Everything a pilot notices about a tired pack -- voltage that collapses on
throttle punches, a flight time that suddenly halves, a brownout on a hard
climb -- comes from ``R_pack`` growing.  And ``R_pack`` is directly measurable
from any log that records both voltage and current: it is the negative slope
of voltage against current.

This module fits that slope, which is more robust than eyeballing the minimum
voltage.  Minimum voltage depends on how hard the pilot happened to pull;
resistance is a property of the pack.

There is no single milliohm number that means "bad", because resistance scales
inversely with capacity: 8 mohm/cell is fine on a 1300 mAh racing pack and
alarming on a 10000 mAh survey pack.  What generalises is the *sag that
resistance causes at the current the aircraft actually draws*.  This module
therefore thresholds on volts-per-cell of resistive sag at the flight's own
peak current, and reports the milliohm figure alongside it as context:

* under 0.30 V/cell at peak  -- healthy
* 0.30-0.50 V/cell           -- working hard; the sag grows as the pack ages
* over 0.50 V/cell           -- no margin left for a climb or a gust recovery

Brownout is the failure mode to actually fear
---------------------------------------------
If the pack voltage under load drops below what the flight controller's
regulator needs, the FC reboots mid-air.  The log simply stops.  Every
"it fell out of the sky and the log ends with no error" case is either this or
a physical failure.  Predicting it is straightforward: take the measured
resistance, take the highest current the aircraft actually drew, and check the
resulting voltage against the per-cell floor.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..types import FlightLog, Finding, Severity

__all__ = [
    "CELL_NOMINAL_V",
    "CELL_SAG_WARN_V",
    "CELL_SAG_CRITICAL_V",
    "CELL_SAG_AT_PEAK_WARN",
    "CELL_SAG_AT_PEAK_CRITICAL",
    "estimate_cell_count",
    "estimate_internal_resistance",
    "sag_metrics",
    "throttle_voltage_correlation",
    "analyze",
]

#: Nominal per-cell voltage of a LiPo/Li-ion cell under light load.
CELL_NOMINAL_V = 3.7
#: Per-cell voltage under load below which the pack is being over-worked.
CELL_SAG_WARN_V = 3.50
#: Per-cell voltage under load at which brownout becomes likely.
CELL_SAG_CRITICAL_V = 3.30
#: Per-cell sag caused by internal resistance at the flight's own peak current.
#: This -- not an absolute milliohm figure -- is the criterion that generalises.
#: Resistance scales inversely with pack capacity (a 1300 mAh pack at 8 mohm/cell
#: and a 10000 mAh pack at 1.5 mohm/cell are both healthy), so a fixed milliohm
#: threshold would flag every large pack and clear every small one. What actually
#: matters is how many volts per cell that resistance costs at the current the
#: aircraft really draws.
CELL_SAG_AT_PEAK_WARN = 0.30
CELL_SAG_AT_PEAK_CRITICAL = 0.50
#: Resting (near-zero-current) per-cell voltage a full pack should show.
CELL_FULL_V = 4.15


def estimate_cell_count(log: FlightLog) -> Tuple[int, str]:
    """Determine the pack's cell count.

    Prefers the logged value; otherwise infers it from the highest observed
    voltage by picking the count whose implied per-cell voltage lands closest
    to a plausible charged cell.  Guessing matters because every per-cell
    threshold in this module depends on it, so the source of the number is
    returned alongside it and reported.
    """
    s = log.get("bat.cell_count")
    if s is not None and len(s):
        v = s.values[np.isfinite(s.values)]
        v = v[v > 0]
        if v.size:
            return int(round(float(np.median(v)))), "logged"
    meta = log.metadata.get("cell_count")
    if meta:
        return int(meta), "metadata"
    volt = log.get("bat.voltage")
    if volt is None or not len(volt):
        return 0, "unknown"
    vmax = float(np.nanpercentile(volt.values, 99))
    best, best_err = 0, 1e9
    for n in range(1, 15):
        err = abs(vmax / n - CELL_FULL_V)
        if err < best_err:
            best, best_err = n, err
    return best, f"inferred from {vmax:.2f} V peak"


def estimate_internal_resistance(
    log: FlightLog, min_current_span: float = 5.0
) -> Optional[Dict[str, float]]:
    """Fit ``V = a - R*I`` and return the pack resistance.

    Returns ``None`` when there is not enough current variation to fit
    anything: with a constant load the slope is unidentifiable, and reporting a
    number from a degenerate fit would be worse than reporting nothing.

    The fit is done on the flight window only, and the state-of-charge trend is
    removed first.  Without that removal the natural voltage decline over the
    flight is attributed to resistance, and the estimate comes out roughly
    double.  Concretely: voltage is regressed on ``[current, time, 1]``, and
    the current coefficient is the resistance.
    """
    v_s, i_s = log.get("bat.voltage"), log.get("bat.current")
    if v_s is None or i_s is None or len(v_s) < 20:
        return None
    t0, t1 = log.flight_window()
    v = v_s.slice_time(t0, t1)
    if len(v) < 20:
        v = v_s
    current = i_s.interp_to(v.time)
    volt = v.values
    good = np.isfinite(current) & np.isfinite(volt) & (current > 0.5)
    if good.sum() < 20:
        return None
    current, volt, t = current[good], volt[good], v.time[good]
    span = float(np.percentile(current, 95) - np.percentile(current, 5))
    if span < min_current_span:
        return None

    # Design matrix: [current, time, 1]. The time column absorbs the
    # state-of-charge decline so it does not masquerade as resistance.
    A = np.column_stack([current, t - t[0], np.ones_like(current)])
    coeffs, residuals, rank, _ = np.linalg.lstsq(A, volt, rcond=None)
    if rank < 3:
        return None
    r_pack = float(-coeffs[0])
    soc_slope = float(coeffs[1])
    v_open = float(coeffs[2])

    pred = A @ coeffs
    ss_res = float(np.sum((volt - pred) ** 2))
    ss_tot = float(np.sum((volt - np.mean(volt)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    cells, cell_src = estimate_cell_count(log)
    return {
        "r_pack_ohm": r_pack,
        "r_cell_ohm": r_pack / cells if cells else float("nan"),
        "v_open_circuit": v_open,
        "soc_decline_v_per_s": soc_slope,
        "fit_r2": r2,
        "current_span_a": span,
        "samples": float(current.size),
        "cell_count": float(cells),
        "cell_count_source": cell_src,  # type: ignore[dict-item]
    }


def sag_metrics(log: FlightLog) -> Optional[Dict[str, float]]:
    """Minimum/maximum voltage and current over the flight window."""
    v_s, i_s = log.get("bat.voltage"), log.get("bat.current")
    if v_s is None or len(v_s) < 5:
        return None
    t0, t1 = log.flight_window()
    v = v_s.slice_time(t0, t1)
    if len(v) < 5:
        v = v_s
    vals = v.values[np.isfinite(v.values)]
    if vals.size < 5:
        return None
    cells, _ = estimate_cell_count(log)
    out = {
        "v_min": float(np.min(vals)),
        "v_max": float(np.max(vals)),
        "v_min_cell": float(np.min(vals)) / cells if cells else float("nan"),
        "v_max_cell": float(np.max(vals)) / cells if cells else float("nan"),
        "t_v_min": float(v.time[int(np.argmin(v.values))]),
        "cell_count": float(cells),
    }
    if i_s is not None and len(i_s):
        cur = i_s.slice_time(t0, t1).values
        cur = cur[np.isfinite(cur)]
        if cur.size:
            out["i_max"] = float(np.max(cur))
            out["i_mean"] = float(np.mean(cur))
    return out


def throttle_voltage_correlation(log: FlightLog) -> Optional[Dict[str, float]]:
    """Correlate throttle spikes with voltage dips.

    A strong negative correlation confirms the sag is load-driven (a pack or
    wiring problem) rather than a failing sensor or a loose voltage-sense lead.
    The distinction matters: a loose sense lead shows dips uncorrelated with
    throttle, and replacing the pack would not fix it.
    """
    v_s = log.get("bat.voltage")
    thr = log.first_present("throttle", "motor.0")
    if v_s is None or thr is None or len(v_s) < 20:
        return None
    t0, t1 = log.flight_window()
    v = v_s.slice_time(t0, t1)
    if len(v) < 20:
        return None
    th = thr.interp_to(v.time)
    # Remove the slow state-of-charge trend so the correlation measures the
    # dynamic response, not the fact that both decline over the flight.
    vd = v.values - np.poly1d(np.polyfit(v.time, v.values, 1))(v.time)
    if np.std(th) < 1e-6 or np.std(vd) < 1e-9:
        return None
    corr = float(np.corrcoef(th, vd)[0, 1])
    return {
        "throttle_voltage_corr": corr,
        "detrended_v_std": float(np.std(vd)),
        "throttle_std": float(np.std(th)),
    }


def analyze(
    log: FlightLog,
    cell_sag_warn: float = CELL_SAG_WARN_V,
    cell_sag_critical: float = CELL_SAG_CRITICAL_V,
    peak_sag_warn: float = CELL_SAG_AT_PEAK_WARN,
    peak_sag_critical: float = CELL_SAG_AT_PEAK_CRITICAL,
) -> List[Finding]:
    """Run the power-system analyzer."""
    findings: List[Finding] = []
    metrics = sag_metrics(log)
    if metrics is None:
        return findings
    cells = int(metrics["cell_count"]) or 1
    t0, t1 = log.flight_window()

    # -- internal resistance ---------------------------------------------
    res = estimate_internal_resistance(log)
    if res is not None:
        r_cell = float(res["r_cell_ohm"])
        r_pack = float(res["r_pack_ohm"])
        fit_ok = float(res["fit_r2"]) > 0.5 and r_pack > 0
        i_max = float(metrics.get("i_max", 0.0))
        cell_sag = r_pack * i_max / cells if cells else float("nan")
        if fit_ok and np.isfinite(cell_sag) and cell_sag >= peak_sag_warn:
            critical = cell_sag >= peak_sag_critical
            findings.append(
                Finding(
                    analyzer="power",
                    severity=Severity.CRITICAL if critical else Severity.WARNING,
                    title=(
                        f"Pack resistance costs {cell_sag:.2f} V per cell at peak current "
                        f"({r_pack*1000:.0f} mohm pack)"
                    ),
                    explanation=(
                        f"Fitting V = Voc - I*R over {int(res['samples'])} samples with "
                        f"{res['current_span_a']:.0f} A of current variation gives a pack "
                        f"resistance of {r_pack*1000:.0f} mohm (R^2 = {res['fit_r2']:.2f}), "
                        f"i.e. {r_cell*1000:.1f} mohm per cell across {cells} cells. At the "
                        f"{i_max:.0f} A peak this flight actually drew, that resistance alone "
                        f"costs {r_pack*i_max:.2f} V of sag -- {cell_sag:.2f} V per cell. "
                        + (
                            "Above 0.50 V per cell of resistive sag the pack cannot hold "
                            "voltage through a climb or a gust recovery, and the margin "
                            "before the flight controller browns out is gone."
                            if critical
                            else "Above 0.30 V per cell the pack is working noticeably "
                            "harder than it should, and the sag will get worse as it ages."
                        )
                    ),
                    action=(
                        "Retire this pack, or restrict it to gentle flying. Before blaming "
                        "the cells, check the whole power path: XT60/XT90 contacts, the "
                        "solder joints on the power module, and any connector that feels "
                        "warm after landing. A bad connector reads as pack resistance in "
                        "this fit and is far cheaper to fix."
                        if critical
                        else "Log the fitted resistance each flight and watch the trend. "
                        "Check connector and solder-joint quality first -- a marginal XT60 "
                        "adds several milliohms and is indistinguishable from cell ageing "
                        "here."
                    ),
                    evidence={
                        k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in res.items()
                    }
                    | {
                        "peak_current_a": round(i_max, 1),
                        "sag_at_peak_v": round(r_pack * i_max, 3),
                        "cell_sag_at_peak_v": round(cell_sag, 3),
                        "warn_threshold_v_per_cell": peak_sag_warn,
                    },
                    t_start=t0,
                    t_end=t1,
                    confidence=min(1.0, float(res["fit_r2"])),
                    plot={"kind": "series", "channels": ["bat.voltage", "bat.current"]},
                )
            )
        elif fit_ok:
            findings.append(
                Finding(
                    analyzer="power",
                    severity=Severity.INFO,
                    title=(
                        f"Pack resistance healthy: {cell_sag:.2f} V/cell sag at "
                        f"{i_max:.0f} A peak"
                    ),
                    explanation=(
                        f"Fitted pack resistance is {r_pack*1000:.0f} mohm across {cells} "
                        f"cells ({r_cell*1000:.1f} mohm/cell, R^2 = {res['fit_r2']:.2f}). At "
                        f"the {i_max:.0f} A peak drawn this flight that is only "
                        f"{cell_sag:.2f} V per cell of resistive sag."
                    ),
                    action="No action needed. Re-measure every few flights to track ageing.",
                    evidence={
                        "r_pack_ohm": round(r_pack, 5),
                        "r_cell_ohm": round(r_cell, 5),
                        "cell_sag_at_peak_v": round(cell_sag, 4),
                        "peak_current_a": round(i_max, 1),
                        "fit_r2": round(float(res["fit_r2"]), 3),
                        "cell_count": cells,
                        "cell_count_source": res["cell_count_source"],
                    },
                    t_start=t0,
                    t_end=t1,
                )
            )

    # -- absolute sag / brownout risk ------------------------------------
    v_min_cell = float(metrics["v_min_cell"])
    if np.isfinite(v_min_cell) and v_min_cell <= cell_sag_critical:
        findings.append(
            Finding(
                analyzer="power",
                severity=Severity.CRITICAL,
                title=f"Brownout risk: cell voltage fell to {v_min_cell:.2f} V under load",
                explanation=(
                    f"Minimum pack voltage was {metrics['v_min']:.2f} V at "
                    f"t={metrics['t_v_min']:.1f}s, which is {v_min_cell:.2f} V per cell "
                    f"across {cells} cells. Below {cell_sag_critical:.2f} V per cell the "
                    "flight controller's regulator loses headroom. When it drops out the "
                    "FC reboots in flight and the log simply ends -- which is exactly what "
                    "an unexplained fall out of the sky looks like in a log file."
                ),
                action=(
                    "Stop flying this pack. Then, in order: land at a higher voltage "
                    "(raise the low-battery failsafe threshold), fit a pack with a higher "
                    "C rating or more capacity, and check every connector between the "
                    "pack and the FC. If the aircraft is at its all-up-weight limit, "
                    "reduce payload -- the current draw is the root cause."
                ),
                evidence={
                    k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()
                },
                t_start=max(t0, float(metrics["t_v_min"]) - 2.0),
                t_end=float(metrics["t_v_min"]) + 2.0,
                plot={"kind": "series", "channels": ["bat.voltage", "throttle"]},
            )
        )
    elif np.isfinite(v_min_cell) and v_min_cell <= cell_sag_warn:
        findings.append(
            Finding(
                analyzer="power",
                severity=Severity.WARNING,
                title=f"Deep voltage sag: {v_min_cell:.2f} V per cell under load",
                explanation=(
                    f"The pack dipped to {v_min_cell:.2f} V per cell at "
                    f"t={metrics['t_v_min']:.1f}s. That is below the "
                    f"{cell_sag_warn:.2f} V per cell level where a pack should be landed. "
                    "Repeatedly pulling cells this low shortens pack life and shrinks the "
                    "margin before a brownout."
                ),
                action=(
                    "Raise the low-battery failsafe so the aircraft lands before reaching "
                    "this voltage, and check whether the pack's C rating actually covers "
                    "the peak current recorded here."
                ),
                evidence={
                    k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()
                },
                t_start=max(t0, float(metrics["t_v_min"]) - 2.0),
                t_end=float(metrics["t_v_min"]) + 2.0,
                plot={"kind": "series", "channels": ["bat.voltage", "throttle"]},
            )
        )

    # -- throttle correlation --------------------------------------------
    corr = throttle_voltage_correlation(log)
    if corr is not None:
        c = float(corr["throttle_voltage_corr"])
        if c <= -0.5:
            findings.append(
                Finding(
                    analyzer="power",
                    severity=Severity.INFO,
                    title=f"Voltage dips track throttle (r = {c:+.2f})",
                    explanation=(
                        "Detrended voltage correlates negatively with throttle, which "
                        "confirms the sag is load-driven rather than a sensing fault. "
                        "That is expected behaviour for any pack; it is reported here "
                        "because the *absence* of this correlation would point at a loose "
                        "voltage-sense lead instead of a tired battery."
                    ),
                    action=(
                        "No action on its own. Use it to confirm that any resistance or "
                        "sag finding above is a real power-path problem."
                    ),
                    evidence={k: round(v, 4) for k, v in corr.items()},
                    t_start=t0,
                    t_end=t1,
                )
            )
        elif abs(c) < 0.15 and float(corr["detrended_v_std"]) > 0.15:
            findings.append(
                Finding(
                    analyzer="power",
                    severity=Severity.WARNING,
                    title="Voltage moves independently of throttle",
                    explanation=(
                        f"Voltage varies by {corr['detrended_v_std']:.2f} V (std) after "
                        f"detrending, but barely correlates with throttle (r={c:+.2f}). "
                        "Load-driven sag always tracks throttle. Voltage that moves on its "
                        "own points at the measurement path -- a loose voltage-sense wire, "
                        "a cracked solder joint on the power module, or an intermittent "
                        "main connector."
                    ),
                    action=(
                        "Inspect and re-seat the power-module connector and the "
                        "voltage-sense lead. Wiggle-test them while watching the live "
                        "voltage in the ground station."
                    ),
                    evidence={k: round(v, 4) for k, v in corr.items()},
                    t_start=t0,
                    t_end=t1,
                )
            )

    # -- capacity sanity --------------------------------------------------
    consumed = log.get("bat.consumed")
    remaining = log.get("bat.remaining")
    capacity = log.metadata.get("capacity_mah")
    if consumed is not None and len(consumed) > 2 and capacity:
        used = float(np.nanmax(consumed.values) - np.nanmin(consumed.values))
        frac = used / float(capacity)
        if frac > 0.85:
            findings.append(
                Finding(
                    analyzer="power",
                    severity=Severity.WARNING,
                    title=f"Pack discharged to {frac*100:.0f}% of rated capacity",
                    explanation=(
                        f"{used:.0f} mAh was drawn from a {float(capacity):.0f} mAh pack. "
                        "Routinely discharging a LiPo past ~85% shortens its life sharply "
                        "and leaves nothing in reserve for a go-around or a headwind on "
                        "the way home."
                    ),
                    action=(
                        "Shorten the mission or fit a larger pack. Set the return-to-launch "
                        "trigger at 30% remaining rather than 20%."
                    ),
                    evidence={"consumed_mah": round(used, 1), "capacity_mah": float(capacity)},
                    t_start=t0,
                    t_end=t1,
                )
            )
        if remaining is not None and len(remaining) > 2:
            reported = 1.0 - float(np.nanmin(remaining.values))
            if abs(reported - frac) > 0.25:
                findings.append(
                    Finding(
                        analyzer="power",
                        severity=Severity.WARNING,
                        title="Reported remaining capacity disagrees with coulomb count",
                        explanation=(
                            f"The consumed-charge counter says {frac*100:.0f}% of the pack "
                            f"was used, but the reported remaining figure implies "
                            f"{reported*100:.0f}%. One of the two is wrong, and the "
                            "failsafe logic is using whichever the firmware trusts."
                        ),
                        action=(
                            "Check the configured battery capacity parameter matches the "
                            "pack actually fitted, and calibrate the current sensor against "
                            "a known load. An uncalibrated current sensor makes every "
                            "remaining-capacity failsafe meaningless."
                        ),
                        evidence={
                            "coulomb_used_fraction": round(frac, 3),
                            "reported_used_fraction": round(reported, 3),
                        },
                        t_start=t0,
                        t_end=t1,
                    )
                )
    return findings
