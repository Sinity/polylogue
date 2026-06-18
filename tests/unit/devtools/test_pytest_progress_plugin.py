from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from devtools import pytest_progress_plugin


@pytest.fixture(autouse=True)
def _restore_plugin_state() -> Iterator[None]:
    selected_nodeids = list(pytest_progress_plugin._SELECTED_NODEIDS)
    deselected_nodeids = list(pytest_progress_plugin._DESELECTED_NODEIDS)
    slowest_reports = list(pytest_progress_plugin._SLOWEST_REPORTS)
    collection_started_at = pytest_progress_plugin._COLLECTION_STARTED_AT
    collection_duration_s = pytest_progress_plugin._COLLECTION_DURATION_S
    yield
    pytest_progress_plugin._SELECTED_NODEIDS[:] = selected_nodeids
    pytest_progress_plugin._DESELECTED_NODEIDS[:] = deselected_nodeids
    pytest_progress_plugin._SLOWEST_REPORTS[:] = slowest_reports
    pytest_progress_plugin._COLLECTION_STARTED_AT = collection_started_at
    pytest_progress_plugin._COLLECTION_DURATION_S = collection_duration_s


@dataclass(frozen=True)
class _Report:
    nodeid: str
    when: str
    outcome: str
    duration: float = 0.0
    longrepr: str = ""


def test_progress_plugin_records_call_and_setup_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("POLYLOGUE_PYTEST_EVENTS_PATH", str(events_path))

    pytest_progress_plugin.pytest_runtest_logreport(_Report("test_ok", "setup", "passed"))
    pytest_progress_plugin.pytest_runtest_logreport(_Report("test_body", "call", "passed", duration=0.25))
    pytest_progress_plugin.pytest_runtest_logreport(
        _Report("test_setup", "setup", "failed", duration=0.1, longrepr="fixture exploded")
    )

    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [(event["nodeid"], event["when"], event["outcome"]) for event in events] == [
        ("test_ok", "setup", "passed"),
        ("test_body", "call", "passed"),
        ("test_setup", "setup", "failed"),
    ]
    assert events[1]["duration_s"] == 0.25
    assert events[2]["longrepr"] == "fixture exploded"


def test_progress_plugin_records_node_start_and_finish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("POLYLOGUE_PYTEST_EVENTS_PATH", str(events_path))

    location = ("tests/unit/test_example.py", 12, "test_example")
    pytest_progress_plugin.pytest_runtest_logstart("tests/unit/test_example.py::test_example", location)
    pytest_progress_plugin.pytest_runtest_logfinish("tests/unit/test_example.py::test_example", location)

    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [(event["event"], event["nodeid"]) for event in events] == [
        ("test_started", "tests/unit/test_example.py::test_example"),
        ("test_finished", "tests/unit/test_example.py::test_example"),
    ]
    assert events[0]["location"] == ["tests/unit/test_example.py", 12, "test_example"]


def test_progress_plugin_write_failures_do_not_escape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYLOGUE_PYTEST_EVENTS_PATH", "/dev/null/events.jsonl")

    pytest_progress_plugin.pytest_runtest_logreport(_Report("test_body", "call", "failed", longrepr="boom"))


class _Item:
    def __init__(self, nodeid: str) -> None:
        self.nodeid = nodeid


class _Session:
    def __init__(self, nodeids: list[str]) -> None:
        self.items = [_Item(nodeid) for nodeid in nodeids]


def test_progress_plugin_records_final_selected_nodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events_path = tmp_path / "events.jsonl"
    selection_path = tmp_path / "selection.json"
    monkeypatch.setenv("POLYLOGUE_PYTEST_EVENTS_PATH", str(events_path))
    monkeypatch.setenv("POLYLOGUE_PYTEST_SELECTION_PATH", str(selection_path))
    pytest_progress_plugin._DESELECTED_NODEIDS.clear()

    pytest_progress_plugin.pytest_deselected([_Item("tests/a.py::test_skip")])
    pytest_progress_plugin.pytest_collection_modifyitems(
        _Session(["tests/a.py::test_keep"]),
        object(),
        [_Item("tests/a.py::test_keep")],
    )

    selection = json.loads(selection_path.read_text())
    assert selection["selected_count"] == 1
    assert selection["deselected_count"] == 1
    assert selection["selected_nodeids"] == ["tests/a.py::test_keep"]
    assert selection["deselected_nodeids"] == ["tests/a.py::test_skip"]
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert events[-1]["event"] == "collection_finished"
    assert events[-1]["selected_count"] == 1


def test_progress_plugin_records_collection_duration_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events_path = tmp_path / "events.jsonl"
    selection_path = tmp_path / "selection.json"
    summary_path = tmp_path / "summary.json"
    monkeypatch.setenv("POLYLOGUE_PYTEST_EVENTS_PATH", str(events_path))
    monkeypatch.setenv("POLYLOGUE_PYTEST_SELECTION_PATH", str(selection_path))
    monkeypatch.setenv("POLYLOGUE_PYTEST_SUMMARY_PATH", str(summary_path))
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr("devtools.pytest_progress_plugin.time.monotonic", lambda: next(ticks))

    pytest_progress_plugin.pytest_sessionstart(object())
    pytest_progress_plugin.pytest_collection(object())
    pytest_progress_plugin.pytest_deselected([_Item("tests/a.py::test_skip")])
    pytest_progress_plugin.pytest_collection_modifyitems(
        _Session(["tests/a.py::test_keep"]),
        object(),
        [_Item("tests/a.py::test_keep")],
    )
    pytest_progress_plugin.pytest_runtest_logreport(_Report("test_slow", "setup", "passed", duration=1.5))
    pytest_progress_plugin.pytest_runtest_logreport(_Report("test_fast", "call", "passed", duration=0.1))
    pytest_progress_plugin.pytest_sessionfinish(object(), 0)

    selection = json.loads(selection_path.read_text())
    assert selection["collection_duration_s"] == 2.5
    summary = json.loads(summary_path.read_text())
    assert summary["collection_duration_s"] == 2.5
    assert summary["selected_count"] == 1
    assert summary["deselected_count"] == 1
    assert [report["nodeid"] for report in summary["slowest_reports"]] == ["test_slow", "test_fast"]
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert events[0]["event"] == "collection_started"
    assert events[1]["event"] == "collection_finished"
    assert events[1]["duration_s"] == 2.5
