from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from devtools import pytest_progress_plugin


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
