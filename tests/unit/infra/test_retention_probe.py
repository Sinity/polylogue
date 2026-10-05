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
  and "every test drifts upward";
- report the kernel's ``VmHWM`` as the peak instead of the probe's own
  maximum and ``test_a_watermark_reset_does_not_lower_the_peak`` goes red.
  That is not hypothetical: ``FinishedBuildResourceProbe.start`` writes
  ``/proc/self/clear_refs``, so any run containing it reports a watermark set
  after the reset, and two runs stop being comparable.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from tests.infra import retention_probe
from tests.infra.retention_probe import RetentionProbe, read_memory_kib


class _Memory:
    """A scripted ``/proc/self/status`` reader with a real high-water rule."""

    def __init__(self, anon_series: Sequence[int | tuple[int, int]]) -> None:
        # An entry is the occupancy at the read, or ``(occupancy, transient
        # peak since the previous read)`` for a test that rose and fell back.
        self._series = list(anon_series)
        self._hwm = 0

    def reset_watermark(self) -> None:
        """What ``/proc/self/clear_refs`` 5 does: drop the mark to current."""
        self._hwm = 0

    def __call__(self) -> dict[str, int]:
        entry = self._series.pop(0) if self._series else 0
        anon, peak = entry if isinstance(entry, tuple) else (entry, entry)
        self._hwm = max(self._hwm, anon, peak)
        return {"VmRSS": anon, "RssAnon": anon, "RssFile": 0, "RssShmem": 0, "VmHWM": self._hwm}


def _probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, anon_series: list[int]) -> RetentionProbe:
    # The unqualified report path is the non-xdist name; the managed runner
    # itself runs under xdist, so clear the worker id the probe would append.
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
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
    assert report["observed_anon_peak_kib"] == 600


def test_a_watermark_reset_does_not_lower_the_peak(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: resetting VmHWM between tests must not charge the later test as a new peak setter."""
    reader = _Memory([0, 100, 900, 200, 200, 200, 200, 200])
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.setattr(retention_probe, "read_memory_kib", reader)
    monkeypatch.setattr(retention_probe, "_object_census", lambda: [])
    monkeypatch.setattr(retention_probe, "_malloc_trim", lambda: True)
    probe = RetentionProbe(report_path=tmp_path / "retention.json")

    probe.pytest_sessionstart(session=None)
    probe.pytest_collection_finish(session=None)
    probe.pytest_runtest_logfinish("tests/heavy.py::test_it", ("tests/heavy.py", 1, "test_it"))
    # ``clear_refs`` 5 in a later test drops the kernel watermark to the
    # process's current occupancy, below the 900 this run already reached.
    reader.reset_watermark()
    probe.pytest_runtest_logfinish("tests/after.py::test_it", ("tests/after.py", 1, "test_it"))

    report = _report(probe)

    assert report["observed_anon_peak_kib"] == 900
    assert report["final_kib"]["VmHWM"] < report["observed_anon_peak_kib"]


def test_worker_output_is_qualified_and_retention_ranking_is_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: an exact .json path collides under xdist and HWM sorting hides later retainers."""
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw1")
    # peak.py spikes the high-water mark to 900 but settles at 200 (+100
    # retained); retainer.py stays below the mark but keeps 400 more.
    reader = _Memory([0, 100, (200, 900), 600, 600, 600, 600, 600])
    monkeypatch.setattr(retention_probe, "read_memory_kib", reader)
    monkeypatch.setattr(retention_probe, "_object_census", lambda: [])
    monkeypatch.setattr(retention_probe, "_malloc_trim", lambda: True)
    probe = RetentionProbe(report_path=tmp_path / "retention.json")
    probe.pytest_sessionstart(session=None)
    probe.pytest_collection_finish(session=None)
    probe.pytest_runtest_logfinish("tests/peak.py::test_it", ("tests/peak.py", 1, "test_it"))
    reader.reset_watermark()
    probe.pytest_runtest_logfinish("tests/retainer.py::test_it", ("tests/retainer.py", 1, "test_it"))
    probe.pytest_sessionfinish(session=None, exitstatus=0)

    payload = json.loads((tmp_path / "retention-gw1.json").read_text(encoding="utf-8"))
    assert payload["files"][0]["path"] == "tests/retainer.py"
    assert payload["peak_setter_files"][0]["path"] == "tests/peak.py"
    assert not (tmp_path / "retention.json").exists()
