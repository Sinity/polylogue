from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from devtools.testmon_state import (
    BaselineStatus,
    GraphStatus,
    file_fingerprint,
    inspect_testmon_database,
    stamp_from_attempt,
    validate_stamp,
)
from devtools.testmon_state import (
    TestmonSeedStamp as _TestmonSeedStamp,
)

PROTOCOL = 4
NODEIDS = ("tests/test_seed.py::test_passed", "tests/test_seed.py::test_failed")


def _write_graph(path: Path, *, failed: bool = False, with_edges: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE environment (id INTEGER PRIMARY KEY, environment_name TEXT)")
        connection.execute("CREATE TABLE file_fp (id INTEGER PRIMARY KEY, filename TEXT, fsha TEXT)")
        connection.execute("CREATE TABLE test_execution (id INTEGER PRIMARY KEY, test_name TEXT, failed INTEGER)")
        connection.execute("CREATE TABLE test_execution_file_fp (test_execution_id INTEGER, fingerprint_id INTEGER)")
        for index, nodeid in enumerate(NODEIDS, start=1):
            connection.execute("INSERT INTO file_fp VALUES (?, ?, ?)", (index, f"file-{index}.py", f"sha-{index}"))
            connection.execute(
                "INSERT INTO test_execution VALUES (?, ?, ?)",
                (index, nodeid, int(failed and index == 2)),
            )
            if with_edges:
                connection.execute("INSERT INTO test_execution_file_fp VALUES (?, ?)", (index, index))


def _attempt(data: Path, *, outcomes: tuple[str, str] = ("passed", "failed")) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL,
        "status": "incomplete",
        "identity": {
            "git_head": "head",
            "worktree_fingerprint": "tree",
            "python": "python",
            "skip_slow": True,
            "lab": False,
        },
        "selection": {
            "selected_count": len(NODEIDS),
            "selected_nodeids_omitted": 0,
        },
        "expected_nodeids": list(NODEIDS),
        "expected_count": len(NODEIDS),
        "expected_digest": hashlib.sha256("\n".join(sorted(NODEIDS)).encode()).hexdigest(),
        "node_outcomes": [
            {"nodeid": nodeid, "outcome": outcome} for nodeid, outcome in zip(NODEIDS, outcomes, strict=True)
        ],
        "exit_code": 1,
        "run_id": "run-red",
        "artifact_dir": ".cache/verify/runs/run-red",
        "testmon_data": file_fingerprint(data),
    }


def test_failed_complete_graph_is_selection_only_and_rebindable(tmp_path: Path) -> None:
    data = tmp_path / "testmondata"
    _write_graph(data, failed=True)
    stamp = stamp_from_attempt(_attempt(data), data, checkout_root=tmp_path, protocol_version=PROTOCOL)

    assert stamp is not None
    assert stamp.baseline_status is BaselineStatus.RED
    assert stamp.affected_selection_allowed
    assert not stamp.release_baseline_allowed

    passed_outcomes = stamp_from_attempt(
        _attempt(data, outcomes=("passed", "passed")), data, checkout_root=tmp_path, protocol_version=PROTOCOL
    )
    assert passed_outcomes is not None
    assert passed_outcomes.baseline_status is BaselineStatus.RED
    assert not passed_outcomes.release_baseline_allowed

    stamp_path = tmp_path / "seed.json"
    stamp_path.write_text(json.dumps(stamp.as_dict()))
    assert validate_stamp(stamp_path, data, checkout_root=tmp_path, protocol_version=PROTOCOL) == stamp


def test_omitted_interrupted_and_uncovered_nodes_fail_closed(tmp_path: Path) -> None:
    data = tmp_path / "testmondata"
    _write_graph(data)
    omitted = _attempt(data)
    omitted["selection"] = {"selected_count": 1, "selected_nodeids_omitted": 1}
    assert stamp_from_attempt(omitted, data, checkout_root=tmp_path, protocol_version=PROTOCOL) is None
    interrupted = _attempt(data, outcomes=("passed", "interrupted"))
    assert stamp_from_attempt(interrupted, data, checkout_root=tmp_path, protocol_version=PROTOCOL) is None
    data.unlink()
    _write_graph(data, with_edges=False)
    assert stamp_from_attempt(_attempt(data), data, checkout_root=tmp_path, protocol_version=PROTOCOL) is None


def test_malformed_sqlite_and_stale_stamp_fail_closed(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed"
    malformed.write_bytes(b"not sqlite")
    inspection = inspect_testmon_database(malformed, NODEIDS)
    assert inspection.status is GraphStatus.INVALID

    data = tmp_path / "testmondata"
    _write_graph(data)
    stamp = stamp_from_attempt(_attempt(data), data, checkout_root=tmp_path, protocol_version=PROTOCOL)
    assert stamp is not None
    stamp_path = tmp_path / "seed.json"
    stamp_path.write_text(json.dumps(stamp.as_dict()))
    data.write_bytes(data.read_bytes() + b"stale")
    assert validate_stamp(stamp_path, data, checkout_root=tmp_path, protocol_version=PROTOCOL) is None


def test_malformed_sqlite_values_fail_closed(tmp_path: Path) -> None:
    data = tmp_path / "testmondata"
    _write_graph(data)
    with sqlite3.connect(data) as connection:
        connection.execute("update test_execution set failed = 'bad' where id = 1")

    inspection = inspect_testmon_database(data, NODEIDS)

    assert inspection.status is GraphStatus.INVALID


def test_sqlite_paths_with_uri_characters_are_inspected_safely(tmp_path: Path) -> None:
    data = tmp_path / "checkout?fragment#1" / "testmondata"
    _write_graph(data)

    inspection = inspect_testmon_database(data, NODEIDS)

    assert inspection.status is GraphStatus.COMPLETE


@pytest.mark.parametrize("filename", ["../outside.py", "/tmp/outside.py"])
def test_unsafe_testmon_fingerprint_paths_fail_closed(tmp_path: Path, filename: str) -> None:
    data = tmp_path / "testmondata"
    _write_graph(data)
    with sqlite3.connect(data) as connection:
        connection.execute("update file_fp set filename = ? where id = 1", (filename,))

    assert inspect_testmon_database(data, NODEIDS).status is GraphStatus.INVALID


def test_attempt_status_must_be_promotable(tmp_path: Path) -> None:
    data = tmp_path / "testmondata"
    _write_graph(data)
    attempt = _attempt(data)
    attempt["status"] = "running"

    assert stamp_from_attempt(attempt, data, checkout_root=tmp_path, protocol_version=PROTOCOL) is None

    attempt = _attempt(data)
    attempt["run_id"] = None
    assert stamp_from_attempt(attempt, data, checkout_root=tmp_path, protocol_version=PROTOCOL) is None


def test_stamp_parser_rejects_untyped_or_non_graph_state() -> None:
    try:
        _TestmonSeedStamp.from_mapping({"protocol_version": PROTOCOL, "status": "complete"}, protocol_version=PROTOCOL)
    except ValueError:
        pass
    else:
        raise AssertionError("legacy green-looking stamp must not be accepted")
