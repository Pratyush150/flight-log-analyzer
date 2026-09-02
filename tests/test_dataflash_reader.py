"""ArduPilot dataflash reader.

These tests need ``pymavlink``, so they skip cleanly when it is absent -- which
is the whole point of the guarded import. The fixture builds a tiny ArduPilot
*text* log (the ``.log`` dump format) so no binary file has to live in the
repository.
"""

from __future__ import annotations

import numpy as np
import pytest

from flightlog.readers import dataflash_reader
from flightlog.readers.dataflash_reader import ERR_SUBSYSTEMS, can_read, read_dataflash

pytestmark = pytest.mark.skipif(
    not dataflash_reader.AVAILABLE, reason="pymavlink is not installed"
)

_MINI_LOG = """\
FMT, 128, 89, FMT, BBnNZ, Type,Length,Name,Format,Columns
FMT, 130, 42, ATT, QccccCC, TimeUS,DesRoll,Roll,DesPitch,Pitch,DesYaw,Yaw
FMT, 131, 34, BAT, Qfffff, TimeUS,Volt,VoltR,Curr,CurrTot,Temp
FMT, 132, 30, GPS, QBIHBcLLeeEe, TimeUS,Status,GMS,GWk,NSats,HDop,Lat,Lng,Alt,Spd,GCrs,VZ
FMT, 133, 30, MODE, QMBB, TimeUS,Mode,ModeNum,Rsn
FMT, 134, 20, EV, QB, TimeUS,Id
FMT, 135, 24, ERR, QBB, TimeUS,Subsys,ECode
FMT, 136, 32, RCOU, QHHHH, TimeUS,C1,C2,C3,C4
MODE, 1000000, STABILIZE, 0, 1
EV, 1100000, 10
ATT, 1000000, 100, 98, 0, 3, 0, 1000
BAT, 1000000, 22.5, 22.5, 30.0, 100.0, 25.0
GPS, 1000000, 3, 100, 2000, 14, 85, 473977420, 85455940, 500, 1.0, 90.0, 0.0
RCOU, 1000000, 1500, 1490, 1495, 1505
ATT, 2000000, 120, 118, 10, 12, 0, 1000
BAT, 2000000, 22.1, 22.1, 55.0, 102.0, 25.0
GPS, 2000000, 3, 200, 2000, 14, 88, 473977430, 85455950, 501, 1.1, 90.0, 0.0
RCOU, 2000000, 1600, 1590, 1595, 1605
MODE, 2500000, LOITER, 5, 1
ERR, 2600000, 5, 1
ATT, 3000000, 140, 139, 20, 21, 0, 1000
BAT, 3000000, 21.7, 21.7, 80.0, 104.0, 25.0
GPS, 3000000, 3, 300, 2000, 13, 92, 473977440, 85455960, 502, 1.2, 90.0, 0.0
RCOU, 3000000, 1700, 1690, 1695, 1705
EV, 3500000, 11
"""


@pytest.fixture(scope="module")
def mini_log(tmp_path_factory):
    path = tmp_path_factory.mktemp("df") / "mini.log"
    path.write_text(_MINI_LOG, encoding="utf-8")
    return read_dataflash(str(path))


def test_can_read_recognises_dataflash_extensions():
    assert can_read("flight.bin") and can_read("flight.log") and can_read("f.px4log")
    assert not can_read("flight.ulg")


def test_messages_map_to_canonical_channels(mini_log):
    for name in ("att.roll", "att.roll_sp", "bat.voltage", "bat.current",
                 "gps.fix_type", "gps.satellites", "gps.hdop", "motor.0"):
        assert mini_log.get(name) is not None, f"{name} missing"
    assert mini_log.metadata["log_format"] == "dataflash"


def test_timestamps_are_seconds_rebased_to_zero(mini_log):
    t = mini_log.series["att.roll"].time
    assert t[0] == 0.0
    assert t[-1] == pytest.approx(2.0, abs=0.01)


def test_attitude_is_converted_from_degrees_to_radians(mini_log):
    s = mini_log.series["att.roll"]
    assert s.units == "rad"
    assert "deg->rad" in s.source


def test_pwm_motor_outputs_are_normalised_to_0_1(mini_log):
    m = mini_log.series["motor.0"]
    assert m.units == "fraction"
    assert m.values[0] == pytest.approx(0.5, abs=0.01)  # 1500 us
    assert m.values[-1] == pytest.approx(0.7, abs=0.01)  # 1700 us


def test_raw_lat_lon_integers_are_scaled_to_degrees(mini_log):
    """A text dump can carry unscaled 1e-7 degrees; a .bin carries degrees.
    Both must come out as degrees."""
    lat = mini_log.series["gps.lat"].values
    assert np.all(np.abs(lat) < 180.0)
    assert lat[0] == pytest.approx(47.397742, abs=1e-5)


def test_raw_hdop_centi_units_are_scaled(mini_log):
    hdop = mini_log.series["gps.hdop"].values
    assert hdop[0] == pytest.approx(0.85, abs=0.01)


def test_mode_timeline_is_built_from_mode_messages(mini_log):
    modes = [m.mode for m in mini_log.modes]
    assert modes[0] == "STABILIZE"
    assert "LOITER" in modes


def test_arm_and_disarm_events_come_from_ev_ids(mini_log):
    kinds = [e.kind for e in mini_log.events]
    assert "arm" in kinds and "disarm" in kinds
    assert mini_log.armed_intervals


def test_err_messages_become_failsafe_events(mini_log):
    failsafes = mini_log.events_of("failsafe")
    assert failsafes
    assert "battery failsafe" in failsafes[0].detail
    assert 5 in ERR_SUBSYSTEMS


def test_a_report_can_be_built_from_a_dataflash_log(mini_log):
    from flightlog import analyze_log

    report = analyze_log(mini_log)
    assert report.findings
    assert any(f.analyzer == "modes" for f in report.findings)
