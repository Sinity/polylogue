from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

import devtools.testmon_state as testmon_state
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
    artifact_dir = data.parent / ".cache" / "verify" / "runs" / "run-red"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "run-red",
                "checkout_root": str(data.parent.resolve()),
                "artifact_dir": ".cache/verify/runs/run-red",
            }
        )
    )
    return {
        "protocol_version": PROTOCOL,
        "status": "reusable",
        "identity": {
            "git_head": "head",
            "git_tree": "tree-hash",
            "worktree_fingerprint": "tree",
            "python": "python",
            "skip_slow": True,
            "lab": False,
            "terminal_authorization": "narrow-terminal",
        },
        "selection": {
            "selected_count": len(NODEIDS),
            "selected_nodeids_omitted": 0,
        },
        "expected_nodeids": list(NODEIDS),
        "expected_count": len(NODEIDS),
        "expected_digest": hashlib.sha256("\n".join(sorted(NODEIDS)).encode()).hexdigest(),
        "verification_scope": "narrow-terminal",
        "release_baseline_allowed": False,
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
    assert validate_stamp(stamp_path, data, checkout_root=tmp_path, protocol_version=PROTOCOL) is None


def test_seed_shard_ledger_rejects_duplicate_nodes_across_shards() -> None:
    shard = {
        "index": 1,
        "nodeids": [NODEIDS[0]],
        "nodeid_count": 1,
        "nodeid_digest": hashlib.sha256(NODEIDS[0].encode()).hexdigest(),
        "status": "complete",
        "node_outcomes": [{"nodeid": NODEIDS[0], "outcome": "passed"}],
    }
    duplicate = {**shard, "index": 2}

    assert testmon_state.validate_seed_shard_ledger([shard, duplicate], expected_nodeids=[NODEIDS[0]]) is None


def test_seed_shard_ledger_accepts_expected_and_unexpected_xfail_outcomes() -> None:
    nodes = sorted(NODEIDS)
    shard = {
        "index": 1,
        "nodeids": nodes,
        "nodeid_count": len(nodes),
        "nodeid_digest": hashlib.sha256("\n".join(nodes).encode()).hexdigest(),
        "status": "complete",
        "node_outcomes": [
            {"nodeid": nodes[0], "outcome": "xfailed"},
            {"nodeid": nodes[1], "outcome": "xpassed"},
        ],
    }

    assert testmon_state.validate_seed_shard_ledger([shard], expected_nodeids=nodes) == [shard]


def test_expected_failure_and_non_strict_xpass_preserve_green_baseline(tmp_path: Path) -> None:
    data = tmp_path / "testmondata"
    _write_graph(data)
    attempt = _attempt(data, outcomes=("xfailed", "xpassed"))
    attempt.update(status="complete", exit_code=0, release_baseline_allowed=True, verification_scope="release-baseline")
    identity = attempt["identity"]
    assert isinstance(identity, dict)
    identity.update(skip_slow=False, terminal_authorization=None)

    green = stamp_from_attempt(attempt, data, checkout_root=tmp_path, protocol_version=PROTOCOL)

    assert green is not None
    assert green.baseline_status is BaselineStatus.GREEN
    serialized = green.as_dict()
    assert _TestmonSeedStamp.from_mapping(serialized, protocol_version=PROTOCOL).baseline_status is BaselineStatus.GREEN

    # pytest controls strict-xpass behavior through its process exit code.  A
    # strict xpass therefore remains reusable graph evidence but is red.
    attempt.update(status="reusable", exit_code=1, release_baseline_allowed=False)
    strict = stamp_from_attempt(attempt, data, checkout_root=tmp_path, protocol_version=PROTOCOL)

    assert strict is not None
    assert strict.baseline_status is BaselineStatus.RED
    assert strict.affected_selection_allowed


def test_testmon_database_canonicalizes_xdist_group_names(tmp_path: Path) -> None:
    data = tmp_path / "testmondata"
    _write_graph(data)
    with sqlite3.connect(data) as connection:
        connection.execute(
            "UPDATE test_execution SET test_name = ? WHERE test_name = ?",
            (f"{NODEIDS[0]}@web-reader", NODEIDS[0]),
        )

    graph = inspect_testmon_database(data, NODEIDS)

    assert graph.missing_nodeids == ()
    assert graph.recorded_count == len(NODEIDS)


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


def test_current_protocol_rejects_a_reusable_attempt_with_a_nonreusable_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "testmondata"
    _write_graph(data, failed=True)
    attempt = _attempt(data)
    attempt["protocol_version"] = 6
    identity = attempt["identity"]
    assert isinstance(identity, dict)
    identity["dependency_environment"] = "dependencies"
    identity["pytest_harness"] = "harness"
    monkeypatch.setattr(testmon_state, "testmon_runtime_identity", lambda _root: ("dependencies", "harness"))

    attempt["outcome"] = "resource-timeout"
    assert stamp_from_attempt(attempt, data, checkout_root=tmp_path, protocol_version=6) is None

    attempt["outcome"] = "red-baseline"
    assert stamp_from_attempt(attempt, data, checkout_root=tmp_path, protocol_version=6) is not None


def test_stamp_from_attempt_does_not_reopen_the_validated_database(tmp_path: Path) -> None:
    data = tmp_path / "testmondata"
    _write_graph(data)
    attempt = _attempt(data)

    with patch("devtools.testmon_state.file_fingerprint", return_value=attempt["testmon_data"]) as fingerprint:
        stamp = stamp_from_attempt(attempt, data, checkout_root=tmp_path, protocol_version=PROTOCOL)

    assert stamp is not None
    fingerprint.assert_called_once_with(data)


def test_incomplete_attempt_fails_closed(tmp_path: Path) -> None:
    data = tmp_path / "testmondata"
    _write_graph(data)
    attempt = _attempt(data, outcomes=("passed", "passed"))
    attempt["exit_code"] = 0
    attempt["release_baseline_allowed"] = True
    attempt["status"] = "incomplete"

    stamp = stamp_from_attempt(attempt, data, checkout_root=tmp_path, protocol_version=PROTOCOL)

    assert stamp is None

    attempt["status"] = "complete"
    completed = stamp_from_attempt(attempt, data, checkout_root=tmp_path, protocol_version=PROTOCOL)

    assert completed is not None
    assert completed.baseline_status is BaselineStatus.GREEN
    assert completed.release_baseline_allowed


def test_reusable_attempt_rejects_a_changed_dependency_or_pytest_harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reusable graphs belong to the environment that captured them."""
    data = tmp_path / "testmondata"
    _write_graph(data)
    attempt = _attempt(data)
    identity = attempt["identity"]
    assert isinstance(identity, dict)
    identity["dependency_environment"] = "dependency-environment"
    identity["pytest_harness"] = "pytest-harness"
    attempt["protocol_version"] = 5
    monkeypatch.setattr(
        testmon_state,
        "testmon_runtime_identity",
        lambda _root: ("dependency-environment", "pytest-harness"),
        raising=False,
    )

    assert stamp_from_attempt(attempt, data, checkout_root=tmp_path, protocol_version=5) is not None

    identity["dependency_environment"] = "different-environment"
    assert stamp_from_attempt(attempt, data, checkout_root=tmp_path, protocol_version=5) is None


def test_runtime_identity_includes_test_behavior_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(testmon_state, "_installed_distributions", lambda: (("pytest", "9"),))
    monkeypatch.setenv("HYPOTHESIS_PROFILE", "ci")
    monkeypatch.setenv("POLYLOGUE_CI", "1")
    first = testmon_state.testmon_runtime_identity(tmp_path)

    monkeypatch.setenv("HYPOTHESIS_PROFILE", "default")
    second = testmon_state.testmon_runtime_identity(tmp_path)

    assert first is not None
    assert second is not None
    assert first[0] == second[0]
    assert first[1] != second[1]


def test_green_skipped_slow_attempt_without_typed_terminal_authority_is_selection_only(tmp_path: Path) -> None:
    data = tmp_path / "testmondata"
    _write_graph(data)
    attempt = _attempt(data, outcomes=("passed", "passed"))
    attempt["status"] = "complete"
    attempt["exit_code"] = 0
    raw_identity = attempt["identity"]
    assert isinstance(raw_identity, dict)
    identity = dict(raw_identity)
    identity["terminal_authorization"] = None
    attempt["identity"] = identity
    attempt["verification_scope"] = "narrow-terminal"
    attempt["release_baseline_allowed"] = False

    stamp = stamp_from_attempt(attempt, data, checkout_root=tmp_path, protocol_version=PROTOCOL)

    assert stamp is not None
    assert stamp.baseline_status is BaselineStatus.RED
    assert stamp.affected_selection_allowed
    assert not stamp.release_baseline_allowed


def test_typed_complete_markerless_attempt_is_selection_only(tmp_path: Path) -> None:
    data = tmp_path / "testmondata"
    _write_graph(data, failed=False)
    attempt = _attempt(data, outcomes=("passed", "passed"))
    attempt["status"] = "complete"
    attempt["exit_code"] = 0
    raw_identity = attempt["identity"]
    assert isinstance(raw_identity, dict)
    identity = dict(raw_identity)
    identity["skip_slow"] = False
    identity["terminal_authorization"] = None
    attempt["identity"] = identity
    attempt["verification_scope"] = "release-baseline"
    attempt["release_baseline_allowed"] = True

    published = stamp_from_attempt(attempt, data, checkout_root=tmp_path, protocol_version=PROTOCOL)
    markerless = stamp_from_attempt(
        attempt,
        data,
        checkout_root=tmp_path,
        protocol_version=PROTOCOL,
        published_marker=False,
    )

    assert published is not None
    assert published.release_baseline_allowed
    assert markerless is not None
    assert markerless.baseline_status is BaselineStatus.RED
    assert markerless.affected_selection_allowed
    assert not markerless.release_baseline_allowed


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


def test_attempt_and_green_stamp_artifacts_fail_closed_when_malformed(tmp_path: Path) -> None:
    data = tmp_path / "testmondata"
    _write_graph(data)
    attempt = _attempt(data, outcomes=("passed", "passed"))
    attempt["exit_code"] = 0
    attempt["release_baseline_allowed"] = True
    attempt["artifact_dir"] = "/tmp/outside-testmon-run"
    assert stamp_from_attempt(attempt, data, checkout_root=tmp_path, protocol_version=PROTOCOL) is None

    attempt["artifact_dir"] = ".cache/verify/runs/run-red"
    attempt["status"] = "complete"
    stamp = stamp_from_attempt(attempt, data, checkout_root=tmp_path, protocol_version=PROTOCOL)
    assert stamp is not None
    receipt = tmp_path / ".cache" / "verify" / "runs" / "run-red" / "run.json"
    receipt.unlink()
    assert stamp_from_attempt(attempt, data, checkout_root=tmp_path, protocol_version=PROTOCOL) is None
    receipt.write_text(
        json.dumps(
            {
                "run_id": "wrong-run",
                "checkout_root": str(tmp_path.resolve()),
                "artifact_dir": ".cache/verify/runs/run-red",
            }
        )
    )
    assert stamp_from_attempt(attempt, data, checkout_root=tmp_path, protocol_version=PROTOCOL) is None
    receipt.write_text(
        json.dumps(
            {
                "run_id": "run-red",
                "checkout_root": str(tmp_path.resolve()),
                "artifact_dir": ".cache/verify/runs/run-red",
            }
        )
    )
    stamp_path = tmp_path / ".cache" / "testmon" / "seed.json"
    stamp_path.parent.mkdir(parents=True)
    payload = stamp.as_dict()
    payload["baseline"]["exit_code"] = 1
    stamp_path.write_text(json.dumps(payload))
    assert validate_stamp(stamp_path, data, checkout_root=tmp_path, protocol_version=PROTOCOL) is None

    payload = stamp.as_dict()
    payload["graph"]["failed_nodeids"] = [NODEIDS[0]]
    stamp_path.write_text(json.dumps(payload))
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
