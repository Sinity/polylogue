"""The worker-memory retention probe attributes growth, not the import floor.

``devtools.worker_memory.WORKER_PEAK_ANON_MIB`` is 4750 against a 12 GiB
``memory.high``, which is the whole reason ``CORPUS_MAX_WORKERS`` is 2 and a
complete corpus takes hours. Collection alone is ~583 MiB, so what a worker
accumulates *while running* is the term that decides width -- and the probe is
useless if it charges the import floor to whichever test happened to run
first, or reports a plateau as if every test had contributed to it.

Anti-vacuity:
- delete ``pytest_collection_finish`` and ``test_import_floor_is_not_a_test``
  goes red, because the first test absorbs the whole collection cost;
- charge the high-water raise to every test rather than only to the test
  during which the mark moved and ``test_only_the_peak_setter_is_charged``
  goes red, which is the distinction between "one heavy test set the ceiling"
  and "every test drifts upward".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.infra import retention_probe
from tests.infra.retention_probe import RetentionProbe, read_memory_kib


class _Memory:
    """A scripted ``/proc/self/status`` reader with a real high-water rule."""

    def __init__(self, anon_series: list[int]) -> None:
        self._series = list(anon_series)
        self._hwm = 0

    def __call__(self) -> dict[str, int]:
        anon = self._series.pop(0) if self._series else 0
        self._hwm = max(self._hwm, anon)
        return {"VmRSS": anon, "RssAnon": anon, "RssFile": 0, "RssShmem": 0, "VmHWM": self._hwm}


def _probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, anon_series: list[int]) -> RetentionProbe:
    monkeypatch.setattr(retention_probe, "read_memory_kib", _Memory(anon_series))
    monkeypatch.setattr(retention_probe, "_object_census", lambda: [])
    monkeypatch.setattr(retention_probe, "_malloc_trim", lambda: True)
    return RetentionProbe(report_path=tmp_path / "retention.json")


def _report(probe: RetentionProbe) -> dict[str, Any]:
    probe.pytest_sessionfinish(session=None, exitstatus=0)
    payload = json.loads(probe.report_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_read_memory_reports_the_live_process() -> None:
    """The unmocked reader is a real ``/proc/self/status`` read, not a stub."""
    values = read_memory_kib()

    assert values["RssAnon"] > 0
    assert values["VmHWM"] >= values["VmRSS"] >= values["RssAnon"]


def test_import_floor_is_not_a_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # sessionstart 100, collection_finish 700 (a 600 MiB import floor), then
    # one cheap test that adds 10.
    probe = _probe(tmp_path, monkeypatch, [100, 700, 710, 710, 710, 710, 710])
    probe.pytest_sessionstart(session=None)
    probe.pytest_collection_finish(session=None)
    probe.pytest_runtest_logfinish("tests/a.py::test_one", ("tests/a.py", 1, "test_one"))

    report = _report(probe)

    assert report["recovery_kib"]["import_floor"] == 600
    assert report["recovery_kib"]["accumulated_while_running"] == 10
    assert report["files"] == [
        {"path": "tests/a.py", "tests": 1, "anon_delta_kib": 10, "hwm_raise_kib": 10},
    ]


def test_only_the_peak_setter_is_charged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A heavy test raises the mark by 500; the two after it run inside the
    # space it already claimed and raise nothing.
    probe = _probe(tmp_path, monkeypatch, [0, 100, 600, 150, 200, 200, 200, 200, 200])
    probe.pytest_sessionstart(session=None)
    probe.pytest_collection_finish(session=None)
    for name in ("heavy", "light_one", "light_two"):
        probe.pytest_runtest_logfinish(f"tests/{name}.py::test_it", (f"tests/{name}.py", 1, "test_it"))

    report = _report(probe)

    assert report["files"][0] == {
        "path": "tests/heavy.py",
        "tests": 1,
        "anon_delta_kib": 500,
        "hwm_raise_kib": 500,
    }
    assert [row["hwm_raise_kib"] for row in report["files"][1:]] == [0, 0]
    assert report["peak_raising_tests"] == [{"hwm_raise_kib": 500, "nodeid": "tests/heavy.py::test_it"}]
