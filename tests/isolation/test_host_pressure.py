"""Global host file-table telemetry blocks admission without creating a probe."""

import pytest

from mikoshi.isolation import admission


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_supported_host_counters_and_unavailable_observations(monkeypatch, platform):
    monkeypatch.setattr(admission.sys, "platform", platform)
    sensor = "_darwin_file_table" if platform == "darwin" else "_linux_file_table"
    monkeypatch.setattr(admission, sensor, lambda: (950, 1000))
    assert admission.host_file_table_pressure() == {"host_file_table_percent": 95}

    def unavailable():
        raise OSError("sensor unavailable")

    monkeypatch.setattr(admission, sensor, unavailable)
    with pytest.raises(admission.PressureUnavailable, match="OSError"):
        admission.host_file_table_pressure()
    monkeypatch.setattr(admission, sensor, lambda: (10, 0))
    with pytest.raises(admission.PressureUnavailable, match="ValueError"):
        admission.host_file_table_pressure()


def test_unsupported_hosts_omit_file_table_check(monkeypatch):
    monkeypatch.setattr(admission.sys, "platform", "unsupported")
    assert admission.host_file_table_source() is None
    assert admission.host_file_table_pressure() is None


def test_linux_file_table_subtracts_unused_handles(monkeypatch):
    monkeypatch.setattr(admission.Path, "read_text", lambda _: "1200\t200\t2000\n")
    assert admission._linux_file_table() == (1000, 2000)
