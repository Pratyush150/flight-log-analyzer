# Interpreting flight logs

A reference for reading PX4 ULog and ArduPilot dataflash logs by hand: what each
message actually contains, which thresholds matter and why, and a symptom index
that maps a pilot complaint to the log signature that confirms it.

This is the reasoning `flight-log-analyzer` automates. It is written so that you
can check the tool's conclusions rather than take them on faith.

---

## 1. Message reference

### 1.1 Inertial measurement

| PX4 topic / field | ArduPilot message / field | Units | What it is |
|---|---|---|---|
| `sensor_combined.accelerometer_m_s2[0..2]` | `IMU.AccX/Y/Z` | m/s² | Specific force in body axes. Includes gravity: a level, stationary aircraft reads ≈ 0, 0, −9.81. |
| `sensor_combined.gyro_rad[0..2]` | `IMU.GyrX/Y/Z` | rad/s | Body angular rate. |
| `sensor_accel.clip_counter[0..2]` | `VIBE.Clip0/1/2` | count | Cumulative count of samples that hit the sensor's measurement limit. |
| — | `VIBE.VibeX/Y/Z` | m/s² | ArduPilot's own high-passed accelerometer RMS. PX4 does not log an equivalent, so this tool computes it. |

**Body axis convention.** Both stacks use FRD: X forward, Y right, Z down. So Z
accelerometer reads about −9.81 in level flight, and a positive Z gyro rate is a
nose-right yaw.

**Reading accelerometers.** Never look at raw accelerometer values to judge
vibration — gravity and real acceleration dominate. High-pass above ~5 Hz first.
Real flight motion is almost entirely below 5 Hz; anything above it is the frame.

### 1.2 Attitude and rates

| PX4 | ArduPilot | Units | Notes |
|---|---|---|---|
| `vehicle_attitude` (quaternion `q`) | `ATT.Roll/Pitch/Yaw` | PX4 quaternion, ArduPilot **degrees** | ArduPilot logs degrees; convert before comparing. |
| `vehicle_attitude_setpoint.roll_body` etc. | `ATT.DesRoll/DesPitch/DesYaw` | rad / deg | What the controller asked for. |
| `vehicle_angular_velocity.xyz` | `RATE.R/P/Y` | rad/s / deg/s | Measured body rates, filtered. |
| `vehicle_rates_setpoint` | `RATE.RDes/PDes/YDes` | rad/s / deg/s | Rate loop input. |

The gap between `Des*` and the measured value is the **tracking error**, and it is
the single most useful derived signal in a tuning investigation. Everything about
oscillation diagnosis comes from its spectrum.

### 1.3 Height

| PX4 | ArduPilot | Notes |
|---|---|---|
| `vehicle_air_data.baro_alt_meter` | `BARO.Alt`, `CTUN.BAlt` | Pressure altitude, relative to the pressure seen at boot. Drifts with weather and temperature. |
| `vehicle_gps_position.alt` (mm) | `GPS.Alt` | WGS-84 ellipsoid height. Noisy, but does not drift. |
| `distance_sensor.current_distance` | `RFND.Dist` | Distance to whatever is under the aircraft — not altitude above launch. |
| `vehicle_local_position.z` | `CTUN.Alt` | Estimator output. **PX4 `z` is NED down**: negate it to get altitude. |

Each source has a different datum, so comparing raw values is meaningless.
Remove the constant offset first and look only at the *drift*.

### 1.4 Power

| PX4 | ArduPilot | Notes |
|---|---|---|
| `battery_status.voltage_filtered_v` | `BAT.Volt` | Pack voltage at the power module, not at the cells. |
| `battery_status.current_filtered_a` | `BAT.Curr` | Requires a calibrated current sensor; many are 20% out of the box. |
| `battery_status.discharged_mah` | `BAT.CurrTot` | Integrated current. Only as good as the current calibration. |
| `battery_status.remaining` | `BAT.RemPct` | Firmware's state-of-charge estimate. ArduPilot logs a **percentage**. |

### 1.5 GNSS

| PX4 | ArduPilot | Notes |
|---|---|---|
| `vehicle_gps_position.fix_type` | `GPS.Status` | 0 none, 1 no fix, 2 2D, 3 3D, 4 DGPS, 5 RTK float, 6 RTK fixed. |
| `satellites_used` | `GPS.NSats` | Satellites actually in the solution. |
| `hdop` | `GPS.HDop` | Horizontal dilution of precision — geometry quality, not accuracy. |
| `lat` / `lon` (1e-7 deg) | `GPS.Lat/Lng` | PX4 logs integer 1e-7 degrees. |

### 1.6 Estimator

| PX4 | ArduPilot | Notes |
|---|---|---|
| `estimator_status.vel_test_ratio` | `NKF4.SV` / `XKF4.SV` | Velocity innovation test ratio. |
| `pos_test_ratio` | `NKF4.SP` / `XKF4.SP` | Position. |
| `hgt_test_ratio` | `NKF4.SH` / `XKF4.SH` | Height. |
| `mag_test_ratio` | `NKF4.SM` / `XKF4.SM` | Magnetometer. |
| `estimator_innovations.*` | `NKF3.*` / `XKF3.*` | Raw innovations, in sensor units. |

An **innovation** is measurement minus prediction. The **test ratio** is
`innovation² / innovation_variance`. Both firmwares reject a measurement when its
test ratio exceeds 1.0. A ratio near 1.0 therefore means the estimator is about
to stop using the sensor it depends on for position hold.

### 1.7 Actuators, RC, system

| PX4 | ArduPilot | Notes |
|---|---|---|
| `actuator_motors.control[i]` | `RCOU.C1..C8` | PX4 `actuator_motors` is 0..1; `actuator_outputs` and `RCOU` are PWM microseconds (1000–2000). |
| `esc_status.esc[i].esc_rpm` | `ESC.RPM` | Only present with bidirectional DShot or ESC telemetry. Worth enabling — see §4. |
| `rate_ctrl_status.*speed_integ` | `PIDR/PIDP/PIDY.I` | Rate-loop integrator state. |
| `input_rc.rssi`, `rc_lost` | `RSSI.RXRSSI` | RC link quality. |
| `cpuload.load` | `PM.Load` | Above ~0.85 the scheduler starts dropping work, including log writes. |
| `vehicle_status.nav_state`, `arming_state` | `MODE`, `EV` (10 = armed, 11 = disarmed) | Mode and arming timeline. |
| logged messages | `ERR`, `MSG` | ArduPilot `ERR` is the failsafe/subsystem error stream — the first thing to read in a crash log. |

---

## 2. Thresholds that matter, and why

### 2.1 Vibration (high-passed accelerometer RMS, >5 Hz)

| Value | Meaning |
|---|---|
| < 15 m/s² | Healthy. Vibration is not your problem. |
| 15–30 m/s² | Will fly, but the estimator has less margin than it should, and it gets worse as props wear. |
| > 30 m/s² | Expect EKF variance warnings, altitude creep and position drift. |
| any clipping | Stop. See below. |

**Why vibration causes "EKF" problems.** The IMU is strapdown, so it measures
frame vibration along with real motion. Three mechanisms turn that into estimator
error:

1. **Aliasing.** Accelerometers are sampled at a finite rate behind an imperfect
   anti-alias filter. A vibration tone above Nyquist folds down to a low frequency
   the estimator cannot distinguish from real acceleration. A 92 Hz prop tone
   sampled at 100 Hz *becomes* 8 Hz of acceleration the filter believes.
2. **Clipping.** At roughly ±16 g the accelerometer saturates. A clipped waveform
   is asymmetric, so its mean is no longer zero — the estimator receives a
   constant false acceleration. This is the mechanism behind unexplained climb in
   altitude-hold: the estimator thinks the aircraft is descending, so the
   controller adds throttle.
3. **Integration.** Velocity is the integral of acceleration and position the
   integral of that. A small bias becomes a large position error given time.

**Corollary: fix vibration before touching any gain or EKF parameter.** Tuning
around a mechanical problem does not work, and the tune you arrive at will be
wrong once the mechanical problem is fixed.

### 2.2 Vibration peaks by frequency

Without RPM telemetry, classification is by band and is genuinely less certain:

| Band | Most likely source |
|---|---|
| < 10 Hz | Airframe flex, a swinging battery or payload, or the control loop exciting a structural mode. |
| 10–30 Hz | Anti-vibration-mount resonance. A mount that is *too soft* amplifies here rather than attenuating. |
| 30–60 Hz | Frame resonance, or the motor fundamental on a large low-KV setup (15" props and up). |
| 60–200 Hz | Motor fundamental for 5–10" props at hover. A sharp peak is almost always prop imbalance. |
| 200–400 Hz | Motor bell, worn bearings, blade-pass harmonics. |
| > 400 Hz | High-frequency mechanical or electrical noise. Rarely reaches the estimator; does reach the D term. |

With RPM logged, this becomes near-certain instead of a band guess:

- energy at **1× rotation** = mass imbalance (chipped, bent, water-logged or
  mismatched prop; bent shaft). Never normal above a couple of m/s².
- energy at **blade-count × rotation** = blade pass. Some is normal; a lot means
  poor prop-to-arm clearance or flexible arms.
- energy at **3×, 4×** = worn bearings.

Peak **width** matters too: a narrow peak (< 5 Hz wide) is one discrete rotating
source; a broad peak (> 15 Hz) is structural resonance.

### 2.3 Motor output asymmetry

A level multirotor in hover commands near-identical output on every motor. The
*pattern* of asymmetry names the cause:

| Pattern | Cause |
|---|---|
| One motor high (> 6% above the others) | That motor produces less thrust per unit of command: damaged/bent prop, weak motor, dragging bearing, bent arm. |
| One diagonal pair high | Centre of gravity offset toward the other pair. Move the battery. |
| One motor pinned at max while another sits near idle | ESC desync, broken motor lead, stripped prop adapter, shed prop. **Do not fly again.** |
| All motors climbing over the flight | Pack sagging (see §2.4), not a motor problem. |

### 2.4 Battery

The pack behaves as an ideal source in series with a resistance:

```
V_measured = V_open_circuit − I × R_pack
```

`R_pack` is measurable from any log with voltage and current: it is the negative
slope of V against I. Fit it properly — regress voltage on `[current, time, 1]`
so the state-of-charge decline over the flight does not get attributed to
resistance (it roughly doubles the estimate if you skip this).

There is **no universal milliohm threshold**, because resistance scales inversely
with capacity: 8 mΩ/cell is fine on a 1300 mAh racing pack and alarming on a
10000 mAh survey pack. What generalises is the sag that resistance causes at the
current the aircraft actually draws:

| `R_pack × I_peak / cells` | Meaning |
|---|---|
| < 0.30 V/cell | Healthy. |
| 0.30–0.50 V/cell | Working hard; the sag grows as the pack ages. |
| > 0.50 V/cell | No margin left for a climb or a gust recovery. |

Absolute voltage under load, per cell:

| Value | Meaning |
|---|---|
| > 3.50 V | Fine. |
| 3.30–3.50 V | Land. Repeatedly pulling this low shortens pack life. |
| < 3.30 V | Brownout territory: the FC's regulator loses headroom. |

**The brownout failure mode.** If pack voltage under load drops below what the
flight controller's regulator needs, the FC reboots mid-air and the log simply
stops. Every "it fell out of the sky and the log ends with no error" case is
either this or a physical failure. Check the last few seconds of voltage first.

**Sanity check before blaming the cells.** A bad XT60, a cracked solder joint on
the power module, or a warm connector all read as pack resistance in this fit,
and all are cheaper to fix than a battery.

### 2.5 Estimator test ratios

| Value | Meaning |
|---|---|
| < 0.3 | Healthy. |
| 0.3–0.5 | Margin shrinking; worth noting. |
| 0.5–1.0 | Stressed. Almost always a *symptom* of vibration, compass calibration, or poor GPS sky view. |
| > 1.0 | Measurements rejected. Position hold degrades, then the aircraft falls back or triggers failsafe. |

Which ratio is high tells you where to look:

- `vel` / `pos` high, `mag` fine → GPS quality or vibration-induced velocity error
- `mag` high, others fine → compass calibration or current-induced interference
- `hgt` high → baro disagrees with GPS/rangefinder (prop wash, thermal drift)
- everything high at once → vibration, or an IMU that is failing

Single-sample spikes are common and harmless. Sustained excursions (> 0.5 s) are
the ones that matter.

### 2.6 GNSS

| Signal | Threshold | Notes |
|---|---|---|
| fix type | must be ≥ 3 | Below 3D there is no usable position for the estimator. |
| satellites | ≥ 8 workable, < 6 unusable | The absolute number matters less than a sudden drop. |
| HDOP | < 1.0 good, 1.0–2.0 workable, > 2.0 weak | Geometry factor, multiplies ranging error into position error. |
| time to first fix | 30–60 s cold is normal | Consistently slow warm start = antenna placement or interference. |
| position jump | implied speed > 30 m/s | No multirotor accelerates like that; the receiver jumped, the aircraft did not. |

If HDOP spikes *with* a satellite drop, it is obstruction. If HDOP spikes while
satellite count stays high, suspect interference — the video transmitter and
switching regulators both put noise into the L1 band.

### 2.7 Control loop

Attitude tracking RMS in calm hover:

| Value | Meaning |
|---|---|
| 1–2° | Well tuned. |
| 2–4° | Acceptable, especially in wind. |
| 4–10° | Something is wrong: check the oscillation spectrum before adjusting trim. |
| > 10° | The aircraft is not following commands. |

Distinguish **steady bias** from **oscillation**. A large steady bias with small
variation is a physical trim problem — CG offset, bent arm, weak motor — not a
gain problem, and raising gains will not fix it.

### 2.8 Oscillation frequency → which gain

This is the most useful mapping in practical multirotor tuning. Take the spectrum
of the tracking error and read the dominant peak:

| Frequency | Most likely cause | Parameter |
|---|---|---|
| 0.2–2 Hz | Slow wallow: attitude P too high for the airframe, or position loop fighting the attitude loop, or rate I too large. | `MC_ROLL_P` / `ATC_ANG_RLL_P` |
| 2–8 Hz | Classic wobble: attitude (angle) P too high, or rate I wound too tight. | `MC_ROLL_P` / `ATC_ANG_RLL_P` |
| 8–20 Hz | Rate P too high. The most common over-tune. Often inaudible; costs flight time. | `MC_ROLLRATE_P` / `ATC_RAT_RLL_P` |
| 20–60 Hz | Rate D too high, or D amplifying gyro noise. Hot motors, audible buzz. | `MC_ROLLRATE_D` / `ATC_RAT_RLL_D` |
| > 60 Hz | Not a gain problem — filtering. Gyro noise is reaching the rate loop. | `IMU_GYRO_CUTOFF` / `INS_GYRO_FILTER`, notch filter |

**Direction matters.** The instinct on seeing poor tracking is to raise P. If the
error is an oscillation in the 8–20 Hz band, raising P makes it worse. Read the
frequency first.

See the companion `drone-control-toolkit` repository for the PID/LQR
implementations and the sim harness these bands come from.

---

## 3. Symptom index

Each entry: the complaint, the log signature that confirms it, and what to change.

### "Drone twitches in hover"

**Look for:** a narrowband peak in the attitude tracking error spectrum, usually
8–20 Hz. Cross-check the gyro trace for visible ripple, and motor outputs for
constant small corrections.

**Confirms:** rate P too high.
**If the spectrum is flat instead:** look at RC dropouts (`rc.link_lost`, RSSI
below threshold). A 150 ms dropout leaves the controller executing a stale stick
input — that is a twitch with no oscillation.
**If both are clean:** check EKF reset events. Each reset is a state jump the
position controller has to absorb.

### "Altitude drifts / climbs on its own in altitude-hold"

**Look for, in order:**
1. Accelerometer clip counters. Any increment → clipping bias. Fix vibration.
2. `hgt` test ratio excursions.
3. Baro vs GPS altitude drift after removing the datum offset. A 0.1 m/s drift is
   30 m over five minutes.

**Confirms:** vibration-induced Z bias (if clipping), or barometer disturbance
(if the drift is baro-vs-GPS and clipping is zero).
**Fix:** foam over the baro port, shield it from prop wash and from heat sources
like regulators or a companion computer. Let the FC warm up before arming if the
drift is worst in the first two minutes.

### "Yaw slowly rotates in position hold"

**Look for:** magnetometer field norm varying during flight, correlated with
throttle. The earth's field magnitude is constant, so any change in the measured
norm is distortion.

**Confirms:** current in a power lead near the compass. Heading error grows with
throttle.
**Fix:** move the compass away from the power distribution board and battery
leads — a GPS/compass module on a mast is the standard fix. Recalibrate outdoors,
away from rebar. Recalibrating without moving it first only re-learns the same
bad geometry.

### "Toilet-bowling" (circling around the hold point, growing radius)

**Look for:** the same throttle-correlated magnetometer distortion as above, plus
`mag` test ratio excursions. Also check the compass calibration offsets are not
implausibly large.

**Confirms:** heading error. The position controller pushes toward a target it
has mis-rotated, so the aircraft orbits.
**Fix:** compass placement first, calibration second. Verify with a hover in
position hold at two different throttle settings.

### "Sudden fall out of the sky"

**Look at the last two seconds of the log, in this order:**

1. **Voltage.** Per-cell voltage under load below ~3.3 V → brownout. The log ends
   because the FC rebooted.
2. **Motor outputs.** One motor pinned at maximum with another near idle → ESC
   desync or lost thrust. The mixer was fighting an attitude error it could not
   correct.
3. **Log ends while still armed with healthy voltage** → SD card stopped
   accepting writes, or impact interrupted logging. Check CPU load too: above
   ~0.85 the scheduler drops log writes.
4. **`ERR` messages (ArduPilot)** or logged messages (PX4) in the final seconds.

### "It suddenly darted sideways"

**Look for:** a GPS position jump — consecutive fixes implying a ground speed
above 30 m/s — with a matching spike in the EKF position innovation, and usually a
satellite-count drop or HDOP spike at the same timestamp.

**Confirms:** GPS glitch. In a GPS-referenced mode the position controller chases
the jump, then chases it back when the fix recovers.
**Fix:** antenna sky view and separation from the video transmitter and switching
regulators. If it repeats at the same physical place, it is multipath from a
building or tree line.

### "Position hold wanders"

**Check in this order** (cause before symptom):
1. Vibration RMS and clipping.
2. Satellite count and HDOP.
3. `vel` and `pos` test ratios.

Vibration is the most common cause and the least suspected, because the symptom
looks like a GPS problem.

### "Flight time suddenly halved"

**Look for:** fitted pack resistance and the resulting sag at peak current;
consumed mAh against rated capacity; hover throttle. If hover throttle has crept
up over successive flights with the same weight, the pack is the variable.

**Also check:** an oscillation in the 8–20 Hz band. A motor that is constantly
correcting burns current with nothing to show for it.

### "Motors are hot after a short hover"

**Look for:** oscillation above 20 Hz in the tracking error, and high-frequency
content in the gyro spectrum.

**Confirms:** rate D too high, or D amplifying gyro noise.
**Fix:** reduce D, then lower the gyro low-pass cutoff so D is not fed noise in
the first place. Enable the dynamic notch filter if the airframe has a strong
motor fundamental.

### "It flipped on takeoff"

**Look for:** motor outputs in the first second. A quad that flips immediately has
a motor spinning the wrong way, in the wrong position, or with the wrong prop.
Compare the commanded mixer pattern against which motor actually responded.

**Also check:** the accelerometer trace before arming — if the aircraft was not
level and stationary during the pre-arm gyro calibration, the estimate starts
wrong.

---

## 4. Getting better logs

Small changes that make every future analysis sharper:

- **Log RPM.** Enable bidirectional DShot or ESC telemetry (PX4: `esc_status`;
  ArduPilot: `SERVO_BLH_*` / `ESC` messages). This single change turns vibration
  peak classification from a frequency-band guess into a direct harmonic match.
- **Fly a 60-second hover.** A steady hover with no stick input is the most
  diagnostically useful log there is, because it removes the pilot as a variable.
  Do it after any hardware change.
- **Log at a rate that survives.** High-rate IMU logging is what makes spectral
  analysis possible, but a slow SD card will drop samples. Gaps in the timestamps
  invalidate the frequency axis — check for them before trusting any spectrum.
- **Keep the pre-arm period in the log.** Bench noise before arming is a free
  baseline: vibration present before the motors spin is not coming from the props.
- **Note the conditions.** Wind, temperature and payload change what "normal"
  looks like. A log with no context is much harder to interpret.

---

## 5. What a log cannot tell you

Being honest about the limits is part of the analysis:

- **Prop damage without RPM telemetry** can only be inferred from a frequency
  band, not confirmed. The tool says so in its output rather than overstating.
- **Which specific motor has a bad bearing** is not visible unless the asymmetry
  shows in the outputs; a bearing that is merely noisy but not dragging looks like
  broadband vibration with no per-motor signature.
- **Current sensor accuracy** is unknown unless it has been calibrated. Consumed
  mAh, remaining percentage and the fitted resistance all inherit that error.
- **Wind** is not logged. Sustained attitude bias and integrator saturation are
  consistent with both a bent airframe and a steady crosswind, and the log alone
  cannot separate them. Fly the same profile in two directions to tell them apart.
- **Whether a mechanical part was already failing** cannot be read from one log.
  Trends across flights are far more informative than any single flight.
