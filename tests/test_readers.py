"""Readers: CSV parsing, format dispatch, and guarded optional dependencies."""

from __future__ import annotations

import numpy as np
import pytest

from flightlog.readers import dataflash_reader, load, supported_formats, ulog_reader
from flightlog.readers.csv_reader import read_csv, read_csv_text, write_csv
from flightlog.channels import canonical_from_alias, units_for


def test_optional_readers_import_without_their_dependencies():
    """The guard is the point: these modules must import on a bare install."""
    assert isinstance(ulog_reader.AVAILABLE, bool)
    assert isinstance(dataflash_reader.AVAILABLE, bool)
    assert ulog_reader.INSTALL_HINT == "pip install pyulog"
    assert dataflash_reader.INSTALL_HINT == "pip install pymavlink"


def test_missing_dependency_error_names_the_pip_command():
    if ulog_reader.AVAILABLE:
        pytest.skip("pyulog is installed in this environment")
    with pytest.raises(ulog_reader.MissingDependencyError) as exc:
        ulog_reader.read_ulog("nonexistent.ulg")
    assert "pip install pyulog" in str(exc.value)


def test_supported_formats_table_covers_every_extension():
    rows = supported_formats()
    exts = {e for r in rows for e in r["extensions"]}
    assert {".ulg", ".bin", ".csv"} <= exts
    csv_row = next(r for r in rows if "CSV" in str(r["format"]))
    assert csv_row["available"] is True  # stdlib only, always works


def test_load_rejects_unknown_extensions(tmp_path):
    p = tmp_path / "flight.xyz"
    p.write_text("nothing")
    with pytest.raises(ValueError, match="unrecognised"):
        load(str(p))


def test_load_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load("/definitely/not/here.ulg")


def test_csv_alias_resolution_is_forgiving():
    assert canonical_from_alias("AccX") == "accel.x"
    assert canonical_from_alias("Accel X (m/s2)") == "accel.x"
    assert canonical_from_alias("accel_x") == "accel.x"
    assert canonical_from_alias("accel.x") == "accel.x"
    assert canonical_from_alias("DesRoll") == "att.roll_sp"
    assert units_for("accel.x") == "m/s^2"
    assert units_for("not.a.channel") == ""


def test_csv_reader_parses_headers_and_rebases_time():
    text = "time,AccX,AccY,Volt\n10.0,0.1,0.2,22.1\n10.1,0.3,0.4,22.0\n10.2,0.5,0.6,21.9\n"
    log = read_csv_text(text)
    assert log.get("accel.x") is not None
    assert log.get("bat.voltage") is not None
    assert log.series["accel.x"].time[0] == 0.0
    assert log.series["accel.x"].time[-1] == pytest.approx(0.2)
    assert log.metadata["log_format"] == "csv"


def test_csv_reader_scales_microsecond_timestamps():
    """ArduPilot exports TimeUS. Getting the scale wrong moves every frequency
    in the report by a factor of a million."""
    text = "TimeUS,AccX\n1000000,0.1\n1010000,0.2\n1020000,0.3\n"
    log = read_csv_text(text)
    assert log.series["accel.x"].time[-1] == pytest.approx(0.02)
    assert log.series["accel.x"].sample_rate == pytest.approx(100.0, rel=0.01)


def test_csv_reader_handles_tabs_comments_and_blank_lines():
    text = "# exported by some tool\n\ntime\tAccX\n0\t1.0\n\n1\t2.0\n"
    log = read_csv_text(text)
    assert len(log.series["accel.x"]) == 2


def test_csv_reader_keeps_unknown_columns():
    log = read_csv_text("time,my_custom_signal\n0,1\n1,2\n")
    assert "my.custom.signal" in log.series or "mycustomsignal" in log.series


def test_csv_reader_reconstructs_arm_events_from_a_column():
    text = "time,armed,AccX\n0,0,0.1\n1,1,0.2\n2,1,0.3\n3,0,0.4\n"
    log = read_csv_text(text)
    kinds = [e.kind for e in log.events]
    assert "arm" in kinds and "disarm" in kinds
    assert log.armed_intervals == [(1.0, 3.0)]


def test_csv_reader_rejects_a_file_with_no_data():
    with pytest.raises(ValueError):
        read_csv_text("time,AccX\n")


def test_csv_round_trip_preserves_channel_values(clean_log, tmp_path):
    path = str(tmp_path / "export.csv")
    write_csv(clean_log, path, channels=["accel.x", "accel.y", "accel.z", "bat.voltage"])
    back = read_csv(path)
    assert back.get("accel.x") is not None
    original = clean_log.series["accel.x"]
    restored = back.series["accel.x"]
    assert len(restored) == len(original)
    assert np.allclose(restored.values, original.values, atol=1e-4)


def test_analyzers_run_on_a_csv_derived_log(clean_log, tmp_path):
    """A CSV export with no optional dependencies must still produce a report."""
    from flightlog import analyze_log

    path = str(tmp_path / "export.csv")
    write_csv(
        clean_log,
        path,
        channels=["accel.x", "accel.y", "accel.z", "bat.voltage", "bat.current"],
    )
    report = analyze_log(read_csv(path))
    assert report.findings
    assert report.counts["critical"] == 0
