from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from devtools import pytest_progress_plugin
from devtools.verify_runs import aggregate_pytest_statistics


@pytest.fixture(autouse=True)
def _restore_plugin_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # Unit tests own their event destinations; do not let a surrounding
    # managed verify invocation redirect them into its step artifacts.
    for name in (
        "POLYLOGUE_PYTEST_EVENTS_DIR",
        "POLYLOGUE_PYTEST_EVENTS_PATH",
        "POLYLOGUE_PYTEST_SELECTION_PATH",
        "POLYLOGUE_PYTEST_SUMMARY_PATH",
        "PYTEST_XDIST_WORKER",
    ):
        monkeypatch.delenv(name, raising=False)
    selected_count = pytest_progress_plugin._SELECTED_COUNT
    deselected_count = pytest_progress_plugin._DESELECTED_COUNT
    deselected_nodeids = list(pytest_progress_plugin._DESELECTED_NODEIDS_SAMPLE)
    slowest_reports = list(pytest_progress_plugin._SLOWEST_REPORTS)
    collection_started_at = pytest_progress_plugin._COLLECTION_STARTED_AT
    collection_duration_s = pytest_progress_plugin._COLLECTION_DURATION_S
    controller_collection_payload = pytest_progress_plugin._CONTROLLER_COLLECTION_PAYLOAD
    recorded_report_keys = set(pytest_progress_plugin._RECORDED_REPORT_KEYS)
    pytest_progress_plugin._RECORDED_REPORT_KEYS.clear()
    yield
    pytest_progress_plugin._SELECTED_COUNT = selected_count
    pytest_progress_plugin._DESELECTED_COUNT = deselected_count
    pytest_progress_plugin._DESELECTED_NODEIDS_SAMPLE[:] = deselected_nodeids
    pytest_progress_plugin._SLOWEST_REPORTS[:] = slowest_reports
    pytest_progress_plugin._COLLECTION_STARTED_AT = collection_started_at
    pytest_progress_plugin._COLLECTION_DURATION_S = collection_duration_s
    pytest_progress_plugin._CONTROLLER_COLLECTION_PAYLOAD = controller_collection_payload
    pytest_progress_plugin._RECORDED_REPORT_KEYS.clear()
    pytest_progress_plugin._RECORDED_REPORT_KEYS.update(recorded_report_keys)


@dataclass(frozen=True)
class _Report:
    nodeid: str
    when: str
    outcome: str
    duration: float = 0.0
    longrepr: str = ""
    worker_id: str | None = None
    wasxfail: str | None = None


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


def test_progress_plugin_preserves_xfail_and_xpass_in_durable_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = tmp_path / "step"
    step.mkdir()
    events_path = step / "events.jsonl"
    monkeypatch.setenv("POLYLOGUE_PYTEST_EVENTS_PATH", str(events_path))
    pytest_progress_plugin.pytest_sessionstart(object())

    pytest_progress_plugin.pytest_runtest_logstart("test_xfailed", ("tests/a.py", 1, "test_xfailed"))
    pytest_progress_plugin.pytest_runtest_logreport(
        _Report("test_xfailed", "call", "skipped", wasxfail="known failure")
    )
    pytest_progress_plugin.pytest_runtest_logstart("test_setup_xfailed", ("tests/a.py", 2, "test_setup_xfailed"))
    pytest_progress_plugin.pytest_runtest_logreport(
        _Report("test_setup_xfailed", "setup", "skipped", wasxfail="fixture calls pytest.xfail()")
    )
    pytest_progress_plugin.pytest_runtest_logstart("test_xpassed", ("tests/a.py", 3, "test_xpassed"))
    pytest_progress_plugin.pytest_runtest_logreport(_Report("test_xpassed", "call", "passed", wasxfail="known failure"))

    statistics = aggregate_pytest_statistics(step)

    assert statistics["outcomes"] == {"xfailed": 2, "xpassed": 1}


def test_progress_plugin_observes_real_pytest_xfail_outcome(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    test_path = tmp_path / "test_xfail.py"
    test_path.write_text(
        "import pytest\n\n@pytest.mark.xfail(reason='known failure')\ndef test_expected_failure():\n    assert False\n"
    )
    env = os.environ.copy()
    env["POLYLOGUE_PYTEST_EVENTS_PATH"] = str(events_path)
    checkout_root = Path(__file__).resolve().parents[3]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "devtools.pytest_progress_plugin",
            "-p",
            "no:testmon",
            str(test_path),
        ],
        cwd=checkout_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    reports = [
        json.loads(line)
        for line in events_path.read_text().splitlines()
        if json.loads(line).get("event") == "test_report"
    ]
    assert any(report["when"] == "call" and report["outcome"] == "xfailed" for report in reports)


def test_progress_plugin_skips_xdist_controller_forwarding_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setenv("POLYLOGUE_PYTEST_EVENTS_PATH", str(events_path))
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    pytest_progress_plugin.pytest_runtest_logreport(_Report("test_one", "call", "passed", worker_id="gw0"))

    monkeypatch.delenv("PYTEST_XDIST_WORKER")
    pytest_progress_plugin.pytest_runtest_logreport(_Report("test_one", "call", "passed", worker_id="gw0"))

    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [(event["nodeid"], event["when"], event["worker_id"]) for event in events] == [("test_one", "call", "gw0")]


def test_progress_plugin_keeps_xdist_worker_timings_in_controller_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events_path = tmp_path / "events.jsonl"
    summary_path = tmp_path / "summary.json"
    monkeypatch.setenv("POLYLOGUE_PYTEST_EVENTS_PATH", str(events_path))
    monkeypatch.setenv("POLYLOGUE_PYTEST_SUMMARY_PATH", str(summary_path))
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    pytest_progress_plugin.pytest_runtest_logreport(
        _Report("test_slow", "call", "passed", duration=1.5, worker_id="gw0")
    )
    pytest_progress_plugin.pytest_sessionfinish(object(), 0)

    monkeypatch.delenv("PYTEST_XDIST_WORKER")
    pytest_progress_plugin.pytest_sessionstart(object())
    pytest_progress_plugin.pytest_runtest_logreport(
        _Report("test_slow", "call", "passed", duration=1.5, worker_id="gw0")
    )
    pytest_progress_plugin.pytest_sessionfinish(object(), 0)

    events = [
        json.loads(line)
        for line in events_path.read_text().splitlines()
        if json.loads(line).get("event") == "test_report"
    ]
    summary = json.loads(summary_path.read_text())
    assert [(event["nodeid"], event["worker_id"]) for event in events] == [("test_slow", "gw0")]
    assert [report["nodeid"] for report in summary["slowest_reports"]] == ["test_slow"]


def test_managed_event_ledger_survives_test_host_environment_scrub(tmp_path: Path) -> None:
    events_dir = tmp_path / "events"
    checkout_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    for name in (
        "POLYLOGUE_PYTEST_BASETEMP_ROOT",
        "POLYLOGUE_PYTEST_TMPFS",
        "POLYLOGUE_PYTEST_RUN_ID",
        "POLYLOGUE_PYTEST_MANAGED_BASETEMP",
    ):
        env.pop(name, None)
    env.update(
        {
            # Keep the nested real pytest away from the host-only scratch
            # fallback while preserving the scrubbed event/testmon scenario.
            "POLYLOGUE_PYTEST_BASETEMP_ROOT": str(tmp_path / "pytest-basetemp"),
            "POLYLOGUE_PYTEST_TMPFS": "0",
            "POLYLOGUE_PYTEST_EVENTS_DIR": str(events_dir),
            "POLYLOGUE_PYTEST_SELECTION_PATH": str(tmp_path / "selection.json"),
            "POLYLOGUE_PYTEST_SUMMARY_PATH": str(tmp_path / "summary.json"),
            "POLYLOGUE_VERIFY_RUN_ID": "subprocess-regression",
            "TESTMON_DATAFILE": str(tmp_path / "testmon" / "testmon.sqlite"),
        }
    )
    Path(env["TESTMON_DATAFILE"]).parent.mkdir(parents=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "devtools.pytest_progress_plugin",
            "--testmon-noselect",
            "tests/unit/core/test_identity_law.py::test_session_id_is_origin_native_id",
        ],
        cwd=checkout_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    events = [json.loads(line) for path in events_dir.glob("*.jsonl") for line in path.read_text().splitlines()]
    reports = [event for event in events if event.get("event") == "test_report"]
    assert {event["when"] for event in reports} == {"setup", "call", "teardown"}


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


def test_progress_plugin_writes_worker_scoped_event_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events_dir = tmp_path / "events"
    monkeypatch.setenv("POLYLOGUE_PYTEST_EVENTS_DIR", str(events_dir))
    monkeypatch.setenv("POLYLOGUE_VERIFY_RUN_ID", "run-123")
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw3")

    location = ("tests/unit/test_example.py", 12, "test_example")
    pytest_progress_plugin.pytest_runtest_logstart("tests/unit/test_example.py::test_example", location)

    files = list(events_dir.glob("gw3-*.jsonl"))
    assert len(files) == 1
    event = json.loads(files[0].read_text().splitlines()[0])
    assert event["run_id"] == "run-123"
    assert event["worker_id"] == "gw3"
    assert event["event"] == "test_started"


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
    pytest_progress_plugin.pytest_sessionstart(object())

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
    assert selection["selected_nodeids_omitted"] == 0
    assert selection["deselected_nodeids_omitted"] == 0
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert events[-1]["event"] == "collection_finished"
    assert events[-1]["selected_count"] == 1


def test_progress_plugin_bounds_selection_nodeid_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_path = tmp_path / "selection.json"
    monkeypatch.setenv("POLYLOGUE_PYTEST_SELECTION_PATH", str(selection_path))
    monkeypatch.setenv("POLYLOGUE_PYTEST_SELECTION_NODEID_LIMIT", "2")
    pytest_progress_plugin.pytest_sessionstart(object())

    pytest_progress_plugin.pytest_deselected(
        [
            _Item("tests/a.py::test_skip_1"),
            _Item("tests/a.py::test_skip_2"),
            _Item("tests/a.py::test_skip_3"),
        ]
    )
    pytest_progress_plugin.pytest_collection_modifyitems(
        _Session(["unused"]),
        object(),
        [
            _Item("tests/a.py::test_keep_1"),
            _Item("tests/a.py::test_keep_2"),
            _Item("tests/a.py::test_keep_3"),
        ],
    )

    selection = json.loads(selection_path.read_text())
    assert selection["selected_count"] == 3
    assert selection["deselected_count"] == 3
    assert selection["selected_nodeids"] == ["tests/a.py::test_keep_1", "tests/a.py::test_keep_2"]
    assert selection["deselected_nodeids"] == ["tests/a.py::test_skip_1", "tests/a.py::test_skip_2"]
    assert selection["selected_nodeids_omitted"] == 1
    assert selection["deselected_nodeids_omitted"] == 1
    assert selection["nodeid_sample_limit"] == 2


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
    assert [event["event"] for event in events[:3]] == [
        "session_started",
        "collection_started",
        "collection_finished",
    ]
    assert events[2]["duration_s"] == 2.5


def test_progress_plugin_retains_controller_selection_through_session_finish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_path = tmp_path / "selection.json"
    monkeypatch.setenv("POLYLOGUE_PYTEST_SELECTION_PATH", str(selection_path))
    pytest_progress_plugin.pytest_sessionstart(object())
    pytest_progress_plugin.pytest_collection_modifyitems(
        _Session(["tests/a.py::test_keep"]),
        object(),
        [_Item("tests/a.py::test_keep")],
    )

    pytest_progress_plugin.pytest_sessionfinish(object(), 0)

    selection = json.loads(selection_path.read_text())
    assert selection["selected_nodeids"] == ["tests/a.py::test_keep"]
    assert selection["selected_nodeids_omitted"] == 0


def test_progress_plugin_merges_xdist_collection_facts_without_double_counting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events_dir = tmp_path / "events"
    selection_path = tmp_path / "selection.json"
    summary_path = tmp_path / "summary.json"
    monkeypatch.setenv("POLYLOGUE_PYTEST_EVENTS_DIR", str(events_dir))
    monkeypatch.setenv("POLYLOGUE_PYTEST_SELECTION_PATH", str(selection_path))
    monkeypatch.setenv("POLYLOGUE_PYTEST_SUMMARY_PATH", str(summary_path))

    for worker_id, duration in (("gw1", 1.5), ("gw0", 2.5)):
        ticks = iter([10.0, 10.0 + duration])
        monkeypatch.setattr("devtools.pytest_progress_plugin.time.monotonic", lambda ticks=ticks: next(ticks))
        monkeypatch.setenv("PYTEST_XDIST_WORKER", worker_id)
        pytest_progress_plugin.pytest_sessionstart(object())
        pytest_progress_plugin.pytest_collection(object())
        pytest_progress_plugin.pytest_deselected([_Item("tests/a.py::test_skip")])
        pytest_progress_plugin.pytest_collection_modifyitems(
            _Session(["tests/a.py::test_keep"]), object(), [_Item("tests/a.py::test_keep")]
        )
        pytest_progress_plugin.pytest_sessionfinish(object(), 0)

    monkeypatch.delenv("PYTEST_XDIST_WORKER")
    pytest_progress_plugin.pytest_sessionstart(object())
    pytest_progress_plugin.pytest_sessionfinish(object(), 0)

    selection = json.loads(selection_path.read_text())
    summary = json.loads(summary_path.read_text())
    assert selection["selected_count"] == 1
    assert selection["deselected_count"] == 1
    assert selection["selected_nodeids"] == ["tests/a.py::test_keep"]
    assert summary["selected_count"] == 1
    assert summary["deselected_count"] == 1
    assert summary["collection_duration_s"] == 2.5
