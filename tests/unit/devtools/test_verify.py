from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import watchfiles

from devtools import run_tests, verify, verify_runs
from devtools.testmon_state import (
    BaselineStatus,
    BindingMode,
    CollectionStatus,
    GraphInspection,
    GraphStatus,
    file_fingerprint,
)
from devtools.testmon_state import (
    TestmonBinding as _TestmonBinding,
)
from devtools.testmon_state import (
    TestmonIdentity as _TestmonIdentity,
)
from devtools.testmon_state import (
    TestmonSeedStamp as _TestmonSeedStamp,
)
from devtools.testmon_state import (
    testmon_runtime_identity as _testmon_runtime_identity,
)
from devtools.verify import (
    PYTEST_CONTAINMENT_PATH,
    PYTEST_EVENTS_PATH,
    PYTEST_JUNIT_REPORT_PATH,
    PYTEST_OUTPUT_PATH,
    PYTEST_PROGRESS_PATH,
    PYTEST_REPORT_PATH,
    ROOT,
    TESTMON_AFFECTED_STAMP,
    TESTMON_DATA,
    TESTMON_SEED_ATTEMPT,
    TESTMON_SEED_PROTOCOL_VERSION,
    TESTMON_SEED_SHARD_SIZE,
    TESTMON_SEED_STAMP,
    _anchor_verification_paths,
    _checkpoint_testmon_seed_shard,
    _finalize_testmon_seed_attempt,
    _flatten_seed_outcomes,
    _format_completion_notification,
    _matching_testmon_coverage,
    _parse_pytest_test_count,
    _prepare_testmon_seed_attempt,
    _prepare_testmon_seed_shards,
    _pytest_command_metadata,
    _pytest_metadata_from_report,
    _pytest_stall_timeout_s,
    _pytest_timeout_s,
    _read_pytest_report,
    _record_testmon_affected_coverage,
    _run,
    _seed_node_outcomes_from_events,
    _seed_shard_command,
    _stop_after_failed_step,
    _testmon_database_state,
    _testmon_preflight,
    _testmon_seed_can_resume,
    build_verify_steps,
    main,
)
from devtools.verify_runs import (
    CheckoutMutationMonitor,
    CheckoutMutationObservation,
    PytestResourceError,
    PytestStepArtifacts,
    ResourceSampler,
    VerifyRun,
    adaptive_pytest_runtime_policy,
    adaptive_pytest_worker_count,
    aggregate_pytest_statistics,
    append_verify_history,
    apply_managed_pytest_runtime_policy,
    classify_pytest_result,
    cleanup_managed_pytest_basetemp,
    pytest_basetemp_known_roots,
    pytest_basetemp_path,
    pytest_tmpfs_budget_exceeded,
    pytest_tmpfs_budget_kb,
    resolve_pytest_basetemp_root,
    xdist_uninterruptible_stall_reason,
)
from devtools.verify_runs import (
    worktree_fingerprint as _worktree_fingerprint,
)


@pytest.fixture(autouse=True)
def _isolate_verify_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep supervisor and testmon receipts private to each test."""
    monkeypatch.chdir(tmp_path)
    for name in (
        "TESTMON_DATA",
        "TESTMON_SEED_STAMP",
        "TESTMON_SEED_ATTEMPT",
        "TESTMON_AFFECTED_STAMP",
    ):
        isolated = tmp_path / ".cache" / "testmon" / getattr(verify, name).name
        monkeypatch.setattr(verify, name, isolated)
        monkeypatch.setattr(sys.modules[__name__], name, isolated)


def _pytest_marker_expr(command: list[str]) -> str:
    marker_indexes = [idx for idx, item in enumerate(command) if item == "-m"]
    assert marker_indexes
    assert marker_indexes[-1] + 1 < len(command)
    return command[marker_indexes[-1] + 1]


def _testmon_runtime_identity_fields(checkout_root: Path = ROOT) -> dict[str, str]:
    runtime_identity = _testmon_runtime_identity(checkout_root)
    assert runtime_identity is not None
    dependency_environment, pytest_harness = runtime_identity
    return {"dependency_environment": dependency_environment, "pytest_harness": pytest_harness}


def _write_real_testmon_state(nodeids: tuple[str, ...] = ("tests/test_a.py::test_one",)) -> Path:
    TESTMON_DATA.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(TESTMON_DATA) as conn:
        conn.execute("CREATE TABLE environment (id INTEGER PRIMARY KEY, environment_name TEXT)")
        conn.execute("CREATE TABLE file_fp (id INTEGER PRIMARY KEY, filename TEXT, fsha TEXT)")
        conn.execute("CREATE TABLE test_execution (id INTEGER PRIMARY KEY, test_name TEXT, failed INTEGER)")
        conn.execute("CREATE TABLE test_execution_file_fp (test_execution_id INTEGER, fingerprint_id INTEGER)")
        for index, nodeid in enumerate(nodeids, start=1):
            conn.execute("INSERT INTO file_fp(id, filename, fsha) VALUES (?, ?, ?)", (index, nodeid, f"sha-{index}"))
            conn.execute("INSERT INTO test_execution(id, test_name, failed) VALUES (?, ?, 0)", (index, nodeid))
            conn.execute("INSERT INTO test_execution_file_fp VALUES (?, ?)", (index, index))
    stamp = _TestmonSeedStamp(
        TESTMON_SEED_PROTOCOL_VERSION,
        CollectionStatus.COMPLETE,
        nodeids,
        0,
        BaselineStatus.GREEN,
        True,
        0,
        GraphInspection(GraphStatus.COMPLETE, len(nodeids), len(nodeids), (), 0, 0, None, ()),
        _TestmonIdentity(
            "current-head",
            "covered",
            "python",
            True,
            False,
            None,
            "narrow-terminal",
            **_testmon_runtime_identity_fields(),
        ),
        _TestmonBinding(BindingMode.EXACT, str(ROOT.resolve())),
        file_fingerprint(TESTMON_DATA),
        "seed",
        ".cache/verify/runs/seed",
    )
    TESTMON_SEED_STAMP.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = ROOT / ".cache" / "verify" / "runs" / "seed"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "seed",
                "checkout_root": str(ROOT.resolve()),
                "artifact_dir": ".cache/verify/runs/seed",
            }
        )
    )
    TESTMON_SEED_STAMP.write_text(json.dumps(stamp.as_dict()))
    return TESTMON_DATA


def _write_run_receipt(root: Path, run_id: str) -> None:
    artifact_dir = root / ".cache" / "verify" / "runs" / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "checkout_root": str(root.resolve()),
                "artifact_dir": f".cache/verify/runs/{run_id}",
            }
        )
    )


def test_quick_verify_omits_pytest() -> None:
    steps = build_verify_steps(quick=True, lab=False, skip_slow=False)

    labels = [label for label, _command in steps]
    assert labels == [
        "ruff format",
        "ruff check",
        "mypy",
        "render all",
        "verify layering",
        "lab graph strict",
        "verify closure-matrix",
        "lab schema roundtrip",
        "verify manifests",
        "verify ci-workflows",
        "verify catalog-bypasses",
        "verify doc-commands",
        "verify docs-coverage",
        "verify test-infra-currency",
        "verify pytest-timeout-overrides",
        "verify degrade-loudly",
        "lab policy schema-versioning",
        "lab policy classifier-fingerprints",
        "lab policy demo-tour-freshness",
        "lab policy raw-payload-hash-purity",
        "lab policy position-derived-identity",
        "lab policy raw-authority-frontier-executability",
        "lab policy table-exists-duplication",
        "schema promotion audit",
        "incident coverage ledger",
    ]


def test_default_verify_uses_adaptive_pytest_testmon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("devtools.verify.adaptive_pytest_worker_count", lambda _env: 8)
    steps = build_verify_steps(quick=False, lab=False, skip_slow=False)

    label, command = steps[-1]
    assert label == "pytest testmon"
    assert "--testmon" in command
    assert "--testmon-noselect" not in command
    assert "--testmon-forceselect" in command
    assert "-n" in command
    assert command[command.index("-n") + 1] == "8"
    assert "--dist=loadgroup" in command


def test_broad_default_verify_uses_parallel_testmon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("devtools.verify.adaptive_pytest_worker_count", lambda _env: 8)
    steps = build_verify_steps(quick=False, lab=False, skip_slow=False, broad_testmon=True)

    label, command = steps[-1]
    assert label == "pytest testmon (broad)"
    assert "--testmon" in command
    assert "--testmon-forceselect" in command
    assert command[command.index("-n") + 1] == "8"


def test_pytest_step_requests_structured_json_report() -> None:
    """Every pytest invocation must emit the report consumed by verify and dashboards (#1026)."""
    for kwargs in (
        {"seed_testmon": True},
        {"full_pytest": True},
        {},  # default testmon
    ):
        steps = build_verify_steps(quick=False, lab=False, skip_slow=False, **kwargs)
        pytest_steps = [(label, command) for label, command in steps if label.startswith("pytest")]
        assert pytest_steps, kwargs
        # Every pytest lane emits a structured JSON report.
        for label, command in pytest_steps:
            assert "--json-report" in command, f"{label}: {command}"
            assert any(arg.startswith("--json-report-file=") for arg in command), label
            assert command[command.index("-p") + 1] == "devtools.pytest_progress_plugin"
        # The canonical report path consumed by verify/dashboards is emitted by
        # the primary lane; the #1775 isolated lane writes its own file.
        expected_target = f"--json-report-file={PYTEST_REPORT_PATH}"
        assert any(expected_target in command for _label, command in pytest_steps), kwargs


def test_seed_testmon_runs_full_collection_without_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYLOGUE_PYTEST_WORKERS", raising=False)
    monkeypatch.setattr("devtools.verify.adaptive_pytest_worker_count", lambda _env: 8)
    steps = build_verify_steps(quick=False, lab=False, skip_slow=False, seed_testmon=True)

    label, command = steps[-1]
    assert label == "pytest seed-testmon collect"
    assert "--collect-only" in command
    assert command[command.index("--ignore=tests/benchmarks")] == "--ignore=tests/benchmarks"
    assert "--testmon" not in command
    assert "-n" in command
    assert command[command.index("-n") + 1] == "0"


def test_seed_testmon_caps_adaptive_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYLOGUE_PYTEST_WORKERS", raising=False)
    monkeypatch.setattr("devtools.verify.adaptive_pytest_worker_count", lambda _env: 12)

    steps = build_verify_steps(quick=False, lab=False, skip_slow=False, seed_testmon=True)

    label, command = steps[-1]
    assert label == "pytest seed-testmon collect"
    assert command[command.index("-n") + 1] == "0"


def test_seed_shards_are_deterministic_and_use_managed_xdist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("devtools.verify.adaptive_pytest_worker_count", lambda _environment: 64)
    expected = sorted(f"tests/test_seed.py::test_{index:03d}" for index in range(TESTMON_SEED_SHARD_SIZE + 2))
    prepared = _prepare_testmon_seed_shards(
        {"resume": False, "expected_nodeids": []},
        selection={
            "selected_count": len(expected),
            "selected_nodeids": list(reversed(expected)),
            "selected_nodeids_omitted": 0,
        },
    )

    shards = prepared["shards"]
    assert [shard["nodeid_count"] for shard in shards] == [TESTMON_SEED_SHARD_SIZE, 2]
    assert shards[0]["nodeids"] == expected[:TESTMON_SEED_SHARD_SIZE]
    assert shards[1]["nodeids"] == expected[TESTMON_SEED_SHARD_SIZE:]
    nodeids_file = tmp_path / "seed-shard.args"
    command = _seed_shard_command(["pytest", "--collect-only", "-n", "0"], shards[0], nodeids_file=nodeids_file)
    assert "--collect-only" not in command
    assert command[command.index("-n") + 1] == "10"
    assert "--testmon" in command
    assert "--testmon-noselect" in command
    assert "--dist=loadgroup" in command
    assert command.count("-n") == 1
    assert command[command.index("-n") + 1] == "10"
    assert command[-1] == f"@{nodeids_file}"
    assert nodeids_file.read_text().splitlines() == expected[:TESTMON_SEED_SHARD_SIZE]


def test_seed_outcomes_normalize_xdist_group_suffix(tmp_path: Path) -> None:
    expected = ["tests/test_seed.py::test_grouped"]
    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps(
            {
                "event": "test_report",
                "nodeid": f"{expected[0]}@web-reader",
                "when": "call",
                "outcome": "passed",
            }
        )
        + "\n"
    )

    outcomes = _seed_node_outcomes_from_events(
        events,
        expected_nodeids=expected,
        database={"node_outcomes": {}},
        pytest_step=None,
    )

    assert outcomes == [
        {
            "nodeid": expected[0],
            "outcome": "passed",
            "reason": "test call passed",
            "started": False,
            "finished": False,
            "phases": [{"when": "call", "outcome": "passed", "duration_s": None}],
        }
    ]


def test_seed_shard_checkpoint_preserves_completed_shards_for_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    expected = ["tests/test_seed.py::test_b", "tests/test_seed.py::test_a"]
    ordered = sorted(expected)
    prepared = _prepare_testmon_seed_shards(
        {"resume": False, "expected_nodeids": []},
        selection={"selected_count": 2, "selected_nodeids": expected, "selected_nodeids_omitted": 0},
    )
    prepared["shards"] = [
        {
            **prepared["shards"][0],
            "nodeids": [ordered[0]],
            "nodeid_count": 1,
            "nodeid_digest": hashlib.sha256(ordered[0].encode()).hexdigest(),
        },
        {
            **prepared["shards"][0],
            "index": 2,
            "nodeids": [ordered[1]],
            "nodeid_count": 1,
            "nodeid_digest": hashlib.sha256(ordered[1].encode()).hexdigest(),
            "status": "pending",
            "node_outcomes": [],
        },
    ]
    _atomic_payload = {
        **prepared,
        "expected_nodeids": ordered,
        "expected_count": 2,
        "expected_digest": hashlib.sha256("\n".join(sorted(expected)).encode()).hexdigest(),
    }
    artifact_dir = tmp_path / "shard-1"
    artifact_dir.mkdir()
    (artifact_dir / "selection.json").write_text(
        json.dumps(
            {
                "selected_count": 1,
                "selected_nodeids": [f"{ordered[0]}@web-reader"],
                "selected_nodeids_omitted": 0,
            }
        )
    )
    (artifact_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "event": "test_report",
                "nodeid": f"{ordered[0]}@web-reader",
                "when": "call",
                "outcome": "xfailed",
            }
        )
        + "\n"
    )

    checkpointed = _checkpoint_testmon_seed_shard(
        prepared=_atomic_payload,
        shard_index=1,
        step={"name": "pytest seed-testmon shard 1/2", "exit": 0, "artifact_dir": str(artifact_dir)},
    )

    assert checkpointed["shards"][0]["status"] == "complete"
    assert checkpointed["shards"][1]["status"] == "pending"
    assert json.loads(TESTMON_SEED_ATTEMPT.read_text())["shards"][0]["node_outcomes"][0]["outcome"] == "xfailed"
    resumed = _prepare_testmon_seed_attempt(
        identity={
            "git_head": "head",
            "git_tree": "tree",
            "worktree_fingerprint": "fingerprint",
            "python": "python",
            "skip_slow": False,
            "lab": False,
            **_testmon_runtime_identity_fields(Path.cwd()),
        },
        run=VerifyRun(tier="seed-testmon", argv=[], git_head="head", polylogue_import_path="polylogue"),
        resume=True,
    )
    assert resumed["shards"][0]["status"] == "complete"
    assert resumed["shards"][1]["status"] == "pending"


def test_seed_shard_checkpoint_does_not_trust_preexisting_testmon_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    expected = ["tests/test_seed.py::test_only_database_row"]
    prepared = _prepare_testmon_seed_shards(
        {"resume": False, "expected_nodeids": []},
        selection={"selected_count": 1, "selected_nodeids": expected, "selected_nodeids_omitted": 0},
    )
    artifact_dir = tmp_path / "shard-1"
    artifact_dir.mkdir()
    (artifact_dir / "selection.json").write_text(
        json.dumps({"selected_count": 1, "selected_nodeids": expected, "selected_nodeids_omitted": 0})
    )
    (artifact_dir / "events.jsonl").write_text("")
    monkeypatch.setattr(
        "devtools.verify._testmon_database_state",
        lambda _nodeids: {
            "recorded_count": 1,
            "failed_count": 0,
            "dependency_edge_count": 0,
            "missing_nodeids": [],
            "failed_nodeids": [],
            "node_outcomes": {expected[0]: "passed"},
            "error": None,
            "graph_status": "complete",
            "orphan_execution_edges": 0,
            "orphan_fingerprint_edges": 0,
        },
    )

    checkpointed = _checkpoint_testmon_seed_shard(
        prepared=prepared,
        shard_index=1,
        step={"name": "pytest seed-testmon shard 1/1", "exit": 0, "artifact_dir": str(artifact_dir)},
    )

    shard = checkpointed["shards"][0]
    assert shard["status"] == "incomplete"
    assert shard["node_outcomes"][0]["outcome"] == "missing"


def test_seed_outcome_does_not_infer_call_success_from_teardown(
    tmp_path: Path,
) -> None:
    nodeid = "tests/test_seed.py::test_call_missing"
    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps({"event": "test_started", "nodeid": nodeid})
        + "\n"
        + json.dumps({"event": "test_report", "nodeid": nodeid, "when": "teardown", "outcome": "passed"})
        + "\n"
        + json.dumps({"event": "test_finished", "nodeid": nodeid})
        + "\n"
    )

    without_database = _seed_node_outcomes_from_events(
        events,
        expected_nodeids=[nodeid],
        database={"node_outcomes": {}},
        pytest_step={"exit": 0},
    )
    assert without_database[0]["outcome"] == "missing"

    with_failed_database = _seed_node_outcomes_from_events(
        events,
        expected_nodeids=[nodeid],
        database={"node_outcomes": {nodeid: "failed"}},
        pytest_step={"exit": 1},
    )
    assert with_failed_database[0]["outcome"] == "failed"


def test_seed_shard_failure_remains_visible_and_blocks_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    expected = ["tests/test_seed.py::test_failed"]
    prepared = _prepare_testmon_seed_shards(
        {
            "protocol_version": TESTMON_SEED_PROTOCOL_VERSION,
            "status": "running",
            "identity": {
                "git_head": "head",
                "worktree_fingerprint": "tree",
                "python": "python",
                "skip_slow": False,
                "lab": False,
                **_testmon_runtime_identity_fields(Path.cwd()),
            },
            "resume": False,
            "run_id": "sharded-failure",
            "artifact_dir": ".cache/verify/runs/sharded-failure",
        },
        selection={"selected_count": 1, "selected_nodeids": expected, "selected_nodeids_omitted": 0},
    )
    artifact_dir = tmp_path / "shard-failure"
    artifact_dir.mkdir()
    (artifact_dir / "selection.json").write_text(
        json.dumps({"selected_count": 1, "selected_nodeids": expected, "selected_nodeids_omitted": 0})
    )
    (artifact_dir / "events.jsonl").write_text(
        json.dumps({"event": "test_report", "nodeid": expected[0], "when": "call", "outcome": "failed"}) + "\n"
    )
    TESTMON_DATA.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(TESTMON_DATA) as connection:
        connection.execute("create table environment (id integer primary key, environment_name text)")
        connection.execute("create table file_fp (id integer primary key, filename text, fsha text)")
        connection.execute("create table test_execution (id integer primary key, test_name text, failed integer)")
        connection.execute("create table test_execution_file_fp (test_execution_id integer, fingerprint_id integer)")
        connection.execute("insert into test_execution values (1, ?, 1)", (expected[0],))
        connection.execute("insert into file_fp values (1, 'test_seed.py', 'sha')")
        connection.execute("insert into test_execution_file_fp values (1, 1)")
    _write_run_receipt(tmp_path, "sharded-failure")

    checkpointed = _checkpoint_testmon_seed_shard(
        prepared=prepared,
        shard_index=1,
        step={"name": "pytest seed-testmon shard 1/1", "exit": 1, "artifact_dir": str(artifact_dir)},
    )
    receipt = _finalize_testmon_seed_attempt(
        prepared=checkpointed,
        step_results=[{"name": "pytest seed-testmon shard 1/1", "exit": 1, "artifact_dir": str(artifact_dir)}],
        exit_code=1,
    )

    assert receipt["shards"][0]["status"] == "complete"
    assert receipt["unsuccessful_nodeids"] == expected
    assert receipt["release_baseline_allowed"] is False


def test_resumed_seed_uses_affected_selection_for_remaining_tests() -> None:
    steps = build_verify_steps(
        quick=False,
        lab=False,
        skip_slow=False,
        seed_testmon=True,
        resume_testmon_seed=True,
    )

    label, command = steps[-1]
    assert label == "pytest seed-testmon collect (resume)"
    assert "--collect-only" in command
    assert "--testmon" not in command


def test_full_verify_includes_full_pytest_without_testmon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYLOGUE_PYTEST_WORKERS", raising=False)
    monkeypatch.setattr("devtools.verify.adaptive_pytest_worker_count", lambda _env: 8)
    steps = build_verify_steps(quick=False, lab=False, skip_slow=False, full_pytest=True)

    # #1775: the full diagnostic runs as two lanes — a parallel bulk lane plus a
    # single-process isolated lane for load-sensitive/tui tests. Neither uses
    # testmon; the bulk lane keeps xdist parallelism, the isolated lane forces -n 0.
    labels = [label for label, _command in steps]
    assert labels[-2:] == ["pytest full (parallel)", "pytest load-sensitive (isolated)"]

    bulk_label, bulk_command = steps[-2]
    assert bulk_label == "pytest full (parallel)"
    assert "--testmon" not in bulk_command
    assert "-n" in bulk_command
    assert bulk_command[bulk_command.index("-n") + 1] == "8"
    assert "--dist=loadgroup" in bulk_command

    isolated_label, isolated_command = steps[-1]
    assert isolated_label == "pytest load-sensitive (isolated)"
    assert "--testmon" not in isolated_command
    assert isolated_command[isolated_command.index("-n") + 1] == "0"


def test_seed_collection_refuses_parallel_worker_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYLOGUE_PYTEST_WORKERS", "4")

    steps = build_verify_steps(quick=False, lab=False, skip_slow=False, seed_testmon=True)

    label, command = steps[-1]
    assert label == "pytest seed-testmon collect"
    assert command[command.index("-n") + 1] == "0"


def test_seed_defaults_to_managed_scratch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shm, scratch = _patch_basetemp_roots(monkeypatch, tmp_path, realm_mounted=True)
    _patch_resource_capacity(monkeypatch, shm=shm, scratch=scratch, available_mb=8192)
    for name in (
        "POLYLOGUE_PYTEST_BASETEMP_ROOT",
        "POLYLOGUE_PYTEST_TMPFS",
        "POLYLOGUE_PYTEST_TMPFS_MAX_MB",
        "POLYLOGUE_PYTEST_BASETEMP_REQUIRED_MB",
        "POLYLOGUE_PYTEST_BASETEMP_MIN_FREE_MB",
    ):
        monkeypatch.delenv(name, raising=False)
    completed = subprocess.CompletedProcess(args=["pytest"], returncode=0, stdout="1 passed in 0.1s\n", stderr="")

    with (
        patch("devtools.verify._run_pytest_with_heartbeat", return_value=completed) as run,
        patch("devtools.verify._read_pytest_report", return_value=None),
    ):
        rc, _elapsed, metadata = _run("pytest seed-testmon", ["pytest", "--testmon", "--testmon-noselect"])

    assert rc == 0
    assert metadata["pytest_tmpfs"] is False
    assert run.call_args.kwargs["env"]["POLYLOGUE_PYTEST_TMPFS"] == "0"
    assert run.call_args.kwargs["env"]["POLYLOGUE_PYTEST_BASETEMP_ROOT"] == str(scratch)
    assert run.call_args.kwargs["env"]["POLYLOGUE_PYTEST_SELECTION_NODEID_LIMIT"] == "50000"


def test_default_testmon_worker_count_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYLOGUE_PYTEST_WORKERS", "3")

    steps = build_verify_steps(quick=False, lab=False, skip_slow=False)

    label, command = steps[-1]
    assert label == "pytest testmon"
    assert command[command.index("-n") + 1] == "3"


def test_marker_filters_keep_testmon_selection_forced() -> None:
    steps = build_verify_steps(quick=False, lab=False, skip_slow=False)

    label, command = steps[-1]
    assert label == "pytest testmon"
    marker_expr = _pytest_marker_expr(command)
    assert "not benchmark" in marker_expr
    assert "not scale_medium" in marker_expr
    assert "not scale_large" in marker_expr
    assert "--testmon-forceselect" in command


def test_skip_slow_composes_with_forced_testmon_selection() -> None:
    steps = build_verify_steps(quick=False, lab=False, skip_slow=True)

    label, command = steps[-1]
    assert label == "pytest testmon"
    # Scale-tier policy (#1183): the default verify gate filters out
    # ``scale_medium``/``scale_large``; ``--skip-slow`` composes with that
    # filter via ``and`` rather than replacing it.
    marker_expr = _pytest_marker_expr(command)
    assert "not benchmark" in marker_expr
    assert "not slow" in marker_expr
    assert "not scale_medium" in marker_expr
    assert "not scale_large" in marker_expr
    assert "--testmon-forceselect" in command


def test_default_verify_excludes_medium_and_large_scale_markers() -> None:
    """Default verify pytest step deselects the medium/large scale tiers (#1183)."""
    steps = build_verify_steps(quick=False, lab=False, skip_slow=False)

    label, command = steps[-1]
    assert label == "pytest testmon"
    marker_expr = _pytest_marker_expr(command)
    assert "not benchmark" in marker_expr
    assert "not scale_medium" in marker_expr
    assert "not scale_large" in marker_expr
    # ``scale_small`` is *not* excluded — it runs in the default gate.
    assert "scale_small" not in marker_expr


def test_lab_verify_includes_medium_scale_marker() -> None:
    """``--lab`` lets ``scale_medium`` into the pytest step but still gates ``scale_large`` (#1183)."""
    steps = build_verify_steps(quick=False, lab=True, skip_slow=False)

    pytest_step = next((label, command) for label, command in steps if label.startswith("pytest"))
    label, command = pytest_step
    marker_expr = _pytest_marker_expr(command)
    assert "not benchmark" in marker_expr
    assert "not scale_large" in marker_expr
    assert "not scale_medium" not in marker_expr
    assert "scale_small" not in marker_expr


def test_lab_verify_delegates_to_lab_smoke() -> None:
    steps = build_verify_steps(quick=True, lab=True, skip_slow=False)

    labels = [label for label, _command in steps]
    assert "lab smoke" in labels
    assert "bench slo" in labels
    lab_step = next(step for step in steps if step[0] == "lab smoke")
    assert lab_step == (
        "lab smoke",
        [sys.executable, "-m", "devtools", "lab", "smoke", "run", "archive-smoke", "--tier", "0"],
    )


def test_lab_verify_runs_every_registered_lab_policy_command() -> None:
    """Every `lab policy <name>` CommandSpec must appear as a `--lab` verify
    step, or it is registered/documented but never actually runs as part of
    any standing gate -- reachable only by a human remembering the exact
    standalone command (the demo-tour-freshness gap CodeRabbit flagged:
    registered in the catalog + docs but absent from build_verify_steps'
    `if lab:` block, so `devtools verify --lab`/`--all` never exercised it)."""
    from devtools.command_catalog import COMMAND_SPECS

    registered_lab_policies = {spec.name for spec in COMMAND_SPECS if spec.name.startswith("lab policy ")}
    assert registered_lab_policies, "expected at least one registered `lab policy *` command"

    steps = build_verify_steps(quick=False, lab=True, skip_slow=False)
    step_labels = {label for label, _command in steps}

    missing = registered_lab_policies - step_labels
    assert not missing, f"lab policy commands registered but never run by `devtools verify --lab`: {sorted(missing)}"


def test_testmon_preflight_requires_seed_when_database_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    message = _testmon_preflight(seed_testmon=False, full_pytest=False, quick=False, commit=False)

    assert message is not None
    assert "devtools verify --seed-testmon" in message


def test_testmon_preflight_requires_seed_stamp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    TESTMON_DATA.parent.mkdir(parents=True)
    TESTMON_DATA.write_text("partial")

    message = _testmon_preflight(seed_testmon=False, full_pytest=False, quick=False, commit=False)

    assert message is not None
    assert ".cache/testmon/seed.json" in message


def test_testmon_preflight_accepts_seeded_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_real_testmon_state()

    assert _testmon_preflight(seed_testmon=False, full_pytest=False, quick=False, commit=False) is None


def test_testmon_preflight_rejects_stale_database_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_real_testmon_state()
    TESTMON_DATA.write_bytes(TESTMON_DATA.read_bytes() + b"stale")

    message = _testmon_preflight(seed_testmon=False, full_pytest=False, quick=False, commit=False)

    assert message is not None
    assert "stale" in message
    assert capsys.readouterr().err == ""


def test_testmon_preflight_rejects_malformed_sqlite_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    TESTMON_DATA.parent.mkdir(parents=True)
    TESTMON_DATA.write_text("mutated")
    seed_stamp = tmp_path / ".cache" / "testmon" / "seed.json"
    seed_stamp.parent.mkdir(parents=True, exist_ok=True)
    seed_stamp.write_text(
        json.dumps(
            {
                "protocol_version": TESTMON_SEED_PROTOCOL_VERSION,
                "status": "usable",
            }
        )
    )

    message = _testmon_preflight(seed_testmon=False, full_pytest=False, quick=False, commit=False)

    assert message is not None
    assert "stale" in message or "malformed" in message
    assert capsys.readouterr().err == ""


def test_testmon_preflight_rejects_incomplete_seed_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    TESTMON_DATA.parent.mkdir(parents=True)
    TESTMON_DATA.write_text("partial")
    seed_stamp = tmp_path / ".cache" / "testmon" / "seed.json"
    seed_stamp.write_text(
        json.dumps(
            {
                "protocol_version": TESTMON_SEED_PROTOCOL_VERSION,
                "status": "incomplete",
                "git_head": "current-head",
                "testmon_data": hashlib.sha256(b"partial").hexdigest(),
            }
        )
    )

    message = _testmon_preflight(seed_testmon=False, full_pytest=False, quick=False, commit=False)

    assert message is not None
    assert "stale" in message or "malformed" in message


def test_matching_incomplete_seed_is_resumable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    TESTMON_DATA.parent.mkdir(parents=True)
    TESTMON_DATA.write_text("partial")
    identity = {
        "git_head": "head",
        "git_tree": "tree-hash",
        "worktree_fingerprint": "tree",
        "python": "3.13",
        "skip_slow": True,
        "lab": False,
        "terminal_authorization": None,
    }
    TESTMON_SEED_ATTEMPT.write_text(
        json.dumps(
            {
                "protocol_version": TESTMON_SEED_PROTOCOL_VERSION,
                "status": "incomplete",
                "identity": identity,
                "expected_nodeids": ["tests/unit/test_example.py::test_one"],
                "expected_count": 1,
                "expected_digest": hashlib.sha256(b"tests/unit/test_example.py::test_one").hexdigest(),
                "run_id": "interrupted",
                "started_at": "2026-08-05T12:00:00+00:00",
                "testmon_data_before": "partial",
            }
        )
    )

    assert _testmon_seed_can_resume(identity) is True
    assert _testmon_seed_can_resume({**identity, "git_head": "other", "git_tree": "tree-hash"}) is True
    assert _testmon_seed_can_resume({**identity, "git_tree": "different-tree"}) is False
    assert _testmon_seed_can_resume({**identity, "worktree_fingerprint": "changed"}) is False
    assert _testmon_seed_can_resume({**identity, "skip_slow": False}) is False


def test_two_interrupted_resumes_flatten_all_carried_outcomes(tmp_path: Path) -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    try:
        expected = ["tests/test_a.py::test_one", "tests/test_b.py::test_two"]
        TESTMON_DATA.parent.mkdir(parents=True)
        TESTMON_DATA.write_text("partial")
        identity = {
            "git_head": "head",
            "git_tree": "tree-hash",
            "worktree_fingerprint": "tree",
            "python": "3.13",
            "skip_slow": False,
            "lab": False,
            "terminal_authorization": None,
        }
        TESTMON_SEED_ATTEMPT.write_text(
            json.dumps(
                {
                    "protocol_version": TESTMON_SEED_PROTOCOL_VERSION,
                    "status": "incomplete",
                    "identity": identity,
                    "expected_nodeids": expected,
                    "expected_count": len(expected),
                    "expected_digest": hashlib.sha256("\n".join(sorted(expected)).encode()).hexdigest(),
                    "node_outcomes": [{"nodeid": expected[0], "outcome": "passed"}],
                }
            )
        )
        first = VerifyRun(tier="seed-testmon", argv=["--seed-testmon"], git_head="head", root=tmp_path)
        _prepare_testmon_seed_attempt(identity=identity, run=first, resume=True)
        first_payload = json.loads(TESTMON_SEED_ATTEMPT.read_text())
        first_payload["status"] = "incomplete"
        first_payload["node_outcomes"] = [{"nodeid": expected[1], "outcome": "passed"}]
        TESTMON_SEED_ATTEMPT.write_text(json.dumps(first_payload))

        second = VerifyRun(tier="seed-testmon", argv=["--seed-testmon"], git_head="head", root=tmp_path)
        prepared = _prepare_testmon_seed_attempt(identity=identity, run=second, resume=True)

        assert {item["nodeid"] for item in prepared["prior_node_outcomes"]} == set(expected)
        assert {item["outcome"] for item in prepared["prior_node_outcomes"]} == {"passed"}
        assert _flatten_seed_outcomes(prepared) == prepared["prior_node_outcomes"]
    finally:
        monkeypatch.undo()


def test_focused_run_can_record_typed_affected_scope(tmp_path: Path) -> None:
    run = VerifyRun(tier="focused-test", argv=["tests/unit/example.py"], git_head="head", root=tmp_path)

    payload = run.finish(
        exit_code=0,
        duration_s=0.1,
        verification_scope="affected",
        release_baseline_allowed=False,
    )

    assert payload["verification_scope"] == "affected"
    assert payload["release_baseline_allowed"] is False
    assert json.loads((tmp_path / ".cache" / "verify" / "current-run.json").read_text()) == payload


def test_verify_run_writes_invocation_receipt_without_leaking_token_to_pytest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "invocation" / "run.json"
    monkeypatch.setenv(verify_runs.VERIFICATION_INVOCATION_ID_ENV, "invocation-1")
    monkeypatch.setenv(verify_runs.VERIFICATION_RECEIPT_PATH_ENV, str(receipt))
    run = VerifyRun(tier="focused-test", argv=["tests/unit/example.py"], git_head="head", root=tmp_path)
    artifacts = run.start_step(label="pytest focused", cmd=["pytest", "tests/unit/example.py"])

    payload = run.finish(
        exit_code=0,
        duration_s=0.1,
        verification_scope="affected",
        release_baseline_allowed=False,
    )
    child_env = verify_runs.env_for_pytest_step(dict(os.environ), run=run, artifacts=artifacts)

    assert json.loads(receipt.read_text()) == payload
    assert payload["invocation_id"] == "invocation-1"
    assert verify_runs.VERIFICATION_INVOCATION_ID_ENV not in child_env
    assert verify_runs.VERIFICATION_RECEIPT_PATH_ENV not in child_env


def test_aggregate_pytest_statistics_reduces_phases_fixtures_and_resources(tmp_path: Path) -> None:
    step = tmp_path / "step"
    step.mkdir()
    (step / "events.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "event": "test_report",
                    "nodeid": "a",
                    "when": "setup",
                    "duration_s": 1.0,
                    "outcome": "passed",
                    "worker_id": "controller",
                },
                {
                    "event": "test_report",
                    "nodeid": "a",
                    "when": "call",
                    "duration_s": 2.0,
                    "outcome": "passed",
                    "worker_id": "gw0",
                },
                {
                    "event": "test_report",
                    "nodeid": "a",
                    "when": "teardown",
                    "duration_s": 0.5,
                    "outcome": "passed",
                    "worker_id": "gw0",
                },
            )
        )
        + "\n"
    )
    (step / "resources.jsonl").write_text(
        json.dumps(
            {
                "basetemp": "/dev/shm/run",
                "basetemp_size_kb": 12,
                "tree_rss_kb": 100,
                "tree_pss_kb": 80,
                "cgroup_memory_peak_bytes": 200,
                "xdist_worker_count": 1,
            }
        )
        + "\n"
    )
    (step / "containment.json").write_text(json.dumps({"tmpfs_cleanup_complete": False, "exit_code": 0}))

    result = aggregate_pytest_statistics(
        step,
        command=["pytest"],
        step_result={"exit": 0, "basetemp_cleanup": "/realm/tmp/polylogue-pytest/pytest-polylogue-run"},
    )

    assert result["node_count"] == 1
    assert result["phases"]["call"]["p50_s"] == 2.0
    assert result["phases"]["setup"]["count"] == 1
    assert result["storage"]["basetemp_logical_bytes_max"] == 12 * 1024
    assert result["resources"]["peak_tree_pss_kb"] == 80
    assert result["cleanup"]["complete"] is True


def test_aggregate_pytest_statistics_deduplicates_xdist_reports_and_terminal_failures(tmp_path: Path) -> None:
    step = tmp_path / "step"
    step.mkdir()
    rows = [
        {
            "event": "test_report",
            "nodeid": "test_setup",
            "when": "setup",
            "outcome": "failed",
            "duration_s": 1.0,
            "worker_id": "gw0",
        },
        {
            "event": "test_report",
            "nodeid": "test_setup",
            "when": "setup",
            "outcome": "failed",
            "duration_s": 1.0,
            "worker_id": "controller",
        },
        {
            "event": "test_report",
            "nodeid": "test_teardown",
            "when": "call",
            "outcome": "passed",
            "duration_s": 0.2,
            "worker_id": "gw1",
        },
        {
            "event": "test_report",
            "nodeid": "test_teardown",
            "when": "teardown",
            "outcome": "failed",
            "duration_s": 0.3,
            "worker_id": "gw1",
        },
    ]
    (step / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = aggregate_pytest_statistics(step, command=["pytest", "-n", "2"])

    assert result["phases"]["setup"]["count"] == 1
    assert result["xdist"]["worker_count"] == 2
    assert result["outcomes"] == {"error": 2}

    compact_result = aggregate_pytest_statistics(step, command=["pytest", "--numprocesses=3"])
    assert compact_result["xdist"]["worker_count"] == 3


def test_aggregate_pytest_statistics_accounts_for_started_node_without_a_phase(tmp_path: Path) -> None:
    step = tmp_path / "step"
    step.mkdir()
    rows = [
        {"event": "test_started", "nodeid": "tests/a.py::test_completed", "worker_id": "gw0"},
        {
            "event": "test_report",
            "nodeid": "tests/a.py::test_completed",
            "when": "call",
            "outcome": "passed",
            "duration_s": 0.1,
            "worker_id": "gw0",
        },
        {"event": "test_started", "nodeid": "tests/a.py::test_interrupted", "worker_id": "gw1"},
    ]
    (step / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = aggregate_pytest_statistics(step)

    assert result["node_count"] == 2
    assert result["outcomes"] == {"passed": 1, "interrupted": 1}
    assert sum(result["outcomes"].values()) == result["node_count"]


def test_aggregate_pytest_statistics_uses_completed_report_to_fill_event_gaps(tmp_path: Path) -> None:
    step = tmp_path / "step"
    step.mkdir()
    (step / "events.jsonl").write_text(
        json.dumps(
            {
                "event": "test_report",
                "nodeid": "tests/a.py::test_event",
                "when": "call",
                "outcome": "passed",
                "duration_s": 0.1,
                "worker_id": "gw0",
            }
        )
        + "\n"
    )
    (step / "pytest-report.json").write_text(
        json.dumps(
            {
                "tests": [
                    {
                        "nodeid": "tests/a.py::test_event",
                        "outcome": "passed",
                        "call": {"outcome": "passed", "duration": 0.1},
                    },
                    {
                        "nodeid": "tests/a.py::test_redirected",
                        "outcome": "xfailed",
                        "setup": {"outcome": "passed", "duration": 0.2},
                        "call": {"outcome": "skipped", "duration": 0.3},
                        "teardown": {"outcome": "passed", "duration": 0.1},
                    },
                ]
            }
        )
    )

    result = aggregate_pytest_statistics(step)

    assert result["canonical_report_status"] == "present"
    assert result["node_count"] == 2
    assert result["outcomes"] == {"passed": 1, "xfailed": 1}
    assert result["phases"]["setup"]["count"] == 1
    assert result["phases"]["call"]["count"] == 2


def test_aggregate_pytest_statistics_recognizes_completed_empty_report(tmp_path: Path) -> None:
    step = tmp_path / "step"
    step.mkdir()
    (step / "pytest-report.json").write_text(json.dumps({"tests": []}))

    result = aggregate_pytest_statistics(step)

    assert result["canonical_report_status"] == "present"
    assert result["node_count"] == 0
    assert result["outcomes"] == {}


def test_verify_run_statistics_only_cover_pytest_steps(tmp_path: Path) -> None:
    run = VerifyRun(tier="quick", argv=["--quick"], git_head="head", root=tmp_path)
    artifacts = run.start_step(label="ruff check", cmd=["ruff", "check"])

    run.finish_step(step_id=artifacts.step_id, result={"exit": 0, "duration_s": 0.1})

    step = run._payload["steps"][0]
    assert "statistics" not in step
    assert not artifacts.statistics_path.exists()


def test_verify_run_embeds_compact_statistics_before_worktree_cleanup(tmp_path: Path) -> None:
    run = VerifyRun(tier="focused-test", argv=["tests/unit/example.py"], git_head="head", root=tmp_path)
    artifacts = run.start_step(label="pytest focused", cmd=["pytest", "tests/unit/example.py"])
    artifacts.events_merged_path.write_text(
        json.dumps(
            {
                "event": "test_report",
                "nodeid": "tests/unit/example.py::test_one",
                "when": "call",
                "duration_s": 0.25,
                "outcome": "passed",
                "worker_id": "controller",
            }
        )
        + "\n"
    )

    run.finish_step(step_id=artifacts.step_id, result={"exit": 0, "duration_s": 0.25})
    payload = run.finish(exit_code=0, duration_s=0.25)
    shutil.rmtree(run.run_dir)

    statistics = payload["steps"][0]["statistics"]
    assert statistics["node_count"] == 1
    assert statistics["phases"]["call"]["p50_s"] == 0.25


def test_interrupted_run_merges_worker_events_before_statistics(tmp_path: Path) -> None:
    run = VerifyRun(tier="focused-test", argv=["tests/unit/example.py"], git_head="head", root=tmp_path)
    artifacts = run.start_step(label="pytest focused", cmd=["pytest", "tests/unit/example.py"])
    artifacts.events_dir.mkdir()
    (artifacts.events_dir / "gw0-1.jsonl").write_text(
        json.dumps(
            {
                "event": "test_report",
                "nodeid": "tests/unit/example.py::test_one",
                "when": "call",
                "outcome": "passed",
                "duration_s": 0.2,
                "worker_id": "gw0",
            }
        )
        + "\n"
    )

    run.finish_interrupted_steps(exit_code=130, diagnosis="pytest_interrupted")

    assert artifacts.events_merged_path.exists()
    assert run._payload["steps"][0]["statistics"]["node_count"] == 1


def test_run_returns_finalized_statistics_for_verify_history(tmp_path: Path) -> None:
    completed = subprocess.CompletedProcess(args=["pytest"], returncode=0, stdout="1 passed in 0.1s\n", stderr="")
    run = VerifyRun(tier="quick", argv=["--quick"], git_head="head", root=tmp_path)
    history_path = tmp_path / "state" / "verify-history.jsonl"

    def _complete_with_evidence(
        *_args: object, artifacts: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert isinstance(artifacts, verify_runs.PytestStepArtifacts)
        artifacts.events_merged_path.write_text(
            json.dumps(
                {
                    "event": "test_report",
                    "nodeid": "tests/unit/example.py::test_one",
                    "when": "call",
                    "duration_s": 0.25,
                    "outcome": "passed",
                    "worker_id": "controller",
                }
            )
            + "\n"
        )
        artifacts.resources_path.write_text(json.dumps({"tree_rss_kb": 512}) + "\n")
        return completed

    with (
        patch("devtools.verify._run_pytest_with_heartbeat", side_effect=_complete_with_evidence),
        patch("devtools.verify._read_pytest_report", return_value=None),
        patch("devtools.verify.copy_current_pytest_artifacts"),
    ):
        rc, _elapsed, metadata = _run("pytest testmon", ["pytest", "-n", "0"], run=run)

    assert rc == 0
    append_verify_history(
        {"tier": "quick", "steps": [{"name": "pytest testmon", "exit": rc, **metadata}]}, path=history_path
    )
    shutil.rmtree(run.run_dir)

    durable_row = json.loads(history_path.read_text(encoding="utf-8"))
    assert durable_row["steps"][0]["statistics"]["node_count"] == 1
    assert durable_row["steps"][0]["statistics"]["resources"]["peak_tree_rss_kb"] == 512
    assert metadata["statistics_path"].endswith("statistics.json")


def test_interrupted_pytest_waits_for_forced_containment_quiescence() -> None:
    process = MagicMock()
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired(cmd="pytest", timeout=2.0), None]
    launch = MagicMock()

    with (
        patch("devtools.verify._request_supervisor_termination") as request_termination,
        patch("devtools.verify._force_kill_owned_run") as force_kill,
        patch("devtools.verify.reap_exited_children") as reap,
        patch(
            "devtools.verify.read_receipt",
            return_value={"status": "terminated", "controller_group_alive": False},
        ),
        patch("devtools.verify.descendant_process_identities", return_value=()),
    ):
        verify._await_interrupted_pytest_containment(
            process,
            launch,
            term_grace_s=1.0,
            preserved_runner_descendants=(),
        )

    request_termination.assert_called_once()
    force_kill.assert_called_once_with(process, launch, preserved_runner_descendants=())
    assert process.wait.call_count == 2
    reap.assert_called_once()


def test_interrupted_pytest_refuses_cleanup_without_containment_quiescence() -> None:
    process = MagicMock()
    process.poll.return_value = None
    process.wait.side_effect = [
        subprocess.TimeoutExpired(cmd="pytest", timeout=2.0),
        subprocess.TimeoutExpired(cmd="pytest", timeout=1.0),
    ]
    launch = MagicMock()

    with (
        patch("devtools.verify._request_supervisor_termination"),
        patch("devtools.verify._force_kill_owned_run") as force_kill,
        patch("devtools.verify.reap_exited_children") as reap,
        pytest.raises(verify.PytestContainmentError, match="did not quiesce"),
    ):
        verify._await_interrupted_pytest_containment(
            process,
            launch,
            term_grace_s=1.0,
            preserved_runner_descendants=(),
        )

    force_kill.assert_called_once_with(process, launch, preserved_runner_descendants=())
    reap.assert_not_called()


def test_interrupted_pytest_refuses_cleanup_when_controller_group_survives() -> None:
    process = MagicMock()
    process.poll.return_value = None
    process.wait.return_value = None
    launch = MagicMock()

    with (
        patch("devtools.verify._request_supervisor_termination"),
        patch("devtools.verify.reap_exited_children"),
        patch(
            "devtools.verify.read_receipt",
            return_value={"status": "terminated", "controller_group_alive": True},
        ),
        patch("devtools.verify.descendant_process_identities", return_value=()),
        pytest.raises(verify.PytestContainmentError, match="owned process tree"),
    ):
        verify._await_interrupted_pytest_containment(
            process,
            launch,
            term_grace_s=1.0,
            preserved_runner_descendants=(),
        )


def test_run_cleans_and_finalizes_only_after_contained_interrupt(tmp_path: Path) -> None:
    run = VerifyRun(tier="focused-test", argv=["tests/unit/example.py"], git_head="head", root=tmp_path)
    order: list[str] = []
    original_finish_step = run.finish_step

    def _contained_interrupt(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        order.append("contained")
        raise KeyboardInterrupt

    def _cleanup(**_kwargs: object) -> None:
        order.append("cleanup")
        return None

    def _finish_step(*, step_id: str, result: dict[str, Any]) -> dict[str, Any] | None:
        order.append("finalize")
        return original_finish_step(step_id=step_id, result=result)

    with (
        patch("devtools.verify._run_pytest_with_heartbeat", side_effect=_contained_interrupt),
        patch("devtools.verify.cleanup_managed_pytest_basetemp", side_effect=_cleanup),
        patch.object(run, "finish_step", side_effect=_finish_step),
    ):
        rc, _elapsed, _metadata = _run("pytest focused", ["pytest", "-n", "0"], run=run)

    assert rc == 130
    assert order == ["contained", "cleanup", "finalize"]


def test_run_terminalizes_containment_failure_without_cleaning_basetemp(tmp_path: Path) -> None:
    run = VerifyRun(tier="focused-test", argv=["tests/unit/example.py"], git_head="head", root=tmp_path)

    with (
        patch(
            "devtools.verify._run_pytest_with_heartbeat",
            side_effect=verify.PytestContainmentError("still running"),
        ),
        patch("devtools.verify.cleanup_managed_pytest_basetemp") as cleanup,
    ):
        rc, _elapsed, metadata = _run("pytest focused", ["pytest", "-n", "0"], run=run)

    cleanup.assert_not_called()
    assert rc == 125
    assert metadata["diagnosis"] == "pytest_containment_unproven"
    assert run._payload["steps"][0]["status"] == "failed"
    assert run._payload["steps"][0]["termination_reason"].startswith("pytest containment did not quiesce")


def test_run_recovers_xdist_collection_facts_after_containment_failure(tmp_path: Path) -> None:
    run = VerifyRun(tier="focused-test", argv=["tests/unit/example.py"], git_head="head", root=tmp_path)

    def _write_worker_facts(*_args: object, artifacts: PytestStepArtifacts, **_kwargs: object) -> None:
        artifacts.events_dir.mkdir(parents=True, exist_ok=True)
        for worker_id, pid, duration in (("gw1", 11, 1.5), ("gw0", 10, 2.5)):
            (artifacts.events_dir / f"{worker_id}-{pid}.collection.json").write_text(
                json.dumps(
                    {
                        "worker_id": worker_id,
                        "pid": pid,
                        "selected_count": 3,
                        "deselected_count": 2,
                        "selected_nodeids": ["tests/unit/example.py::test_selected"],
                        "collection_duration_s": duration,
                    }
                ),
                encoding="utf-8",
            )
        raise verify.PytestContainmentError("controller interrupted")

    with patch("devtools.verify._run_pytest_with_heartbeat", side_effect=_write_worker_facts):
        rc, _elapsed, metadata = _run("pytest focused", ["pytest", "-n", "2"], run=run)

    assert rc == 125
    assert metadata["selected_count"] == 3
    assert metadata["deselected_count"] == 2
    assert metadata["collection_duration_s"] == 2.5
    selection = json.loads((run.run_dir / "steps" / "01-pytest-focused" / "selection.json").read_text())
    assert selection["recovered_after_interruption"] is True
    assert selection["worker_id"] == "runner"


def test_verify_main_records_containment_failure_as_terminal_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    history_path = tmp_path / "verify-history.jsonl"
    monkeypatch.setattr(verify, "HISTORY_PATH", history_path)

    with (
        patch("devtools.verify._anchor_verification_paths"),
        patch("devtools.verify._git_head", return_value="head"),
        patch("devtools.verify._git_commit", return_value="base"),
        patch("devtools.verify._default_testmon_is_broad_change", return_value=False),
        patch("devtools.verify._testmon_preflight", return_value=None),
        patch("devtools.verify.build_verify_steps", return_value=[("pytest containment", ["pytest", "-n", "0"])]),
        patch("devtools.verify.apply_managed_pytest_runtime_policy", return_value=({}, None)),
        patch(
            "devtools.verify._run_pytest_with_heartbeat",
            side_effect=verify.PytestContainmentError("owned child still running"),
        ),
        patch("devtools.verify.cleanup_managed_pytest_basetemp") as cleanup,
        patch("devtools.verify._notify"),
    ):
        rc = main(["--json"])

    history = json.loads(history_path.read_text(encoding="utf-8"))
    run_json = next((tmp_path / ".cache" / "verify" / "runs").glob("*/run.json"))
    run_payload = json.loads(run_json.read_text(encoding="utf-8"))
    payload = json.loads(capsys.readouterr().out)

    assert rc == 125
    cleanup.assert_not_called()
    assert payload["diagnosis"] == "pytest_containment_unproven"
    assert history["exit_code"] == 125
    assert history["diagnosis"] == "pytest_containment_unproven"
    assert run_payload["status"] == "failed"
    assert run_payload["steps"][0]["status"] == "failed"


def test_print_history_accepts_verify_and_focused_run_records(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        verify,
        "_load_history",
        lambda: [
            {
                "timestamp": "2026-08-12T20:00:00+00:00",
                "tier": "quick",
                "git_head": "a" * 40,
                "total_duration_s": 2.0,
                "exit_code": 0,
                "steps": [{"name": "ruff", "duration_s": 1.0, "exit": 0}],
            },
            {
                "finished_at": "2026-08-12T20:01:00+00:00",
                "tier": "focused-test",
                "git_head": "b" * 40,
                "duration_s": 3.0,
                "exit_code": 1,
                "steps": [{"name": "pytest focused", "duration_s": None, "exit": 1}],
            },
            {
                "finished_at": "2026-08-12T20:02:00+00:00",
                "tier": "focused-test",
                "git_head": "c" * 40,
                "duration_s": "invalid",
                "exit_code": None,
                "steps": [{"name": "pytest interrupted", "duration_s": "invalid", "exit": None}],
            },
        ],
    )

    verify._print_history()

    output = capsys.readouterr().out
    assert "quick" in output
    assert "focused-" in output
    assert "pytest focused(0s FAIL)" in output
    assert "pytest interrupted(0s FAIL)" in output


def test_verify_history_appends_concurrent_records_without_interleaving(tmp_path: Path) -> None:
    history = tmp_path / "state" / "verify-history.jsonl"

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda sequence: append_verify_history({"sequence": sequence}, path=history), range(64)))

    rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
    assert sorted(row["sequence"] for row in rows) == list(range(64))


def test_verify_history_repairs_or_frames_an_incomplete_trailing_record(tmp_path: Path) -> None:
    history = tmp_path / "state" / "verify-history.jsonl"
    history.parent.mkdir(parents=True)
    history.write_text('{"sequence": 0}', encoding="utf-8")

    append_verify_history({"sequence": 1}, path=history)

    history.write_text(history.read_text(encoding="utf-8") + '{"interrupted":', encoding="utf-8")
    append_verify_history({"sequence": 2}, path=history)

    rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"sequence": 0}, {"sequence": 1}, {"sequence": 2}]


def test_verify_history_append_reads_only_the_trailing_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = tmp_path / "state" / "verify-history.jsonl"
    history.parent.mkdir(parents=True)
    history.write_bytes(b'{"padding":"' + (b"x" * (5 * 1024 * 1024)) + b'"}\n{"interrupted":')
    bytes_read = 0
    real_read = os.read

    def measured_read(descriptor: int, count: int) -> bytes:
        nonlocal bytes_read
        payload = real_read(descriptor, count)
        bytes_read += len(payload)
        return payload

    monkeypatch.setattr(os, "read", measured_read)

    append_verify_history({"sequence": 1}, path=history)

    assert bytes_read < 128 * 1024
    assert json.loads(history.read_text(encoding="utf-8").splitlines()[-1]) == {"sequence": 1}


def test_compare_against_last_skips_intervening_focused_history(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        verify,
        "_load_history",
        lambda: [
            {"tier": "quick", "steps": [{"name": "ruff check", "duration_s": 1.0}]},
            {"tier": "focused-test", "steps": [{"name": "pytest focused", "duration_s": 999.0}]},
        ],
    )

    flags = verify._compare_against_last([{"name": "ruff check", "duration_s": 7.0}])

    assert flags and "ruff check" in flags[0]


def test_compare_against_last_selects_prior_run_independently_per_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        verify,
        "_load_history",
        lambda: [
            {
                "tier": "default",
                "steps": [
                    {"name": "ruff check", "duration_s": 1.0},
                    {"name": "pytest testmon", "duration_s": 2.0},
                ],
            },
            {"tier": "quick", "steps": [{"name": "ruff check", "duration_s": 1.0}]},
        ],
    )

    flags = verify._compare_against_last(
        [
            {"name": "ruff check", "duration_s": 1.1},
            {"name": "pytest testmon", "duration_s": 8.0},
        ]
    )

    assert len(flags) == 1
    assert "pytest testmon" in flags[0]


def test_running_seed_recovers_ledger_from_selection_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    TESTMON_DATA.parent.mkdir(parents=True)
    TESTMON_DATA.write_text("partial")
    artifact_dir = tmp_path / ".cache" / "verify" / "runs" / "interrupted"
    step_dir = artifact_dir / "steps" / "17-pytest-seed-testmon"
    step_dir.mkdir(parents=True)
    expected = ["tests/unit/test_example.py::test_one"]
    (step_dir / "selection.json").write_text(
        json.dumps({"selected_nodeids": expected, "selected_nodeids_omitted": 0, "selected_count": 1})
    )
    identity = {
        "git_head": "head",
        "git_tree": "tree-hash",
        "worktree_fingerprint": "tree",
        "python": "3.13",
        "skip_slow": True,
        "lab": False,
        "terminal_authorization": None,
    }
    TESTMON_SEED_ATTEMPT.write_text(
        json.dumps(
            {
                "protocol_version": TESTMON_SEED_PROTOCOL_VERSION,
                "status": "running",
                "identity": identity,
                "expected_nodeids": [],
                "artifact_dir": str(artifact_dir.relative_to(tmp_path)),
            }
        )
    )

    assert _testmon_seed_can_resume({**identity, "git_head": "fixed", "git_tree": "tree-hash"}) is True

    run = VerifyRun(tier="seed-testmon", argv=["--seed-testmon"], git_head="fixed")
    prepared = _prepare_testmon_seed_attempt(
        identity={**identity, "git_head": "fixed", "git_tree": "tree-hash"}, run=run, resume=True
    )

    assert prepared["expected_nodeids"] == expected
    assert prepared["expected_count"] == 1
    assert prepared["expected_digest"] == hashlib.sha256("\n".join(sorted(expected)).encode()).hexdigest()
    persisted = json.loads(TESTMON_SEED_ATTEMPT.read_text())
    assert persisted["expected_digest"] == prepared["expected_digest"]


def test_seed_resume_rejects_selection_artifact_outside_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    TESTMON_DATA.parent.mkdir(parents=True)
    TESTMON_DATA.write_text("partial")
    outside = tmp_path.parent / "outside-testmon-artifacts"
    step_dir = outside / "steps" / "17-pytest-seed-testmon"
    step_dir.mkdir(parents=True)
    (step_dir / "selection.json").write_text(
        json.dumps(
            {
                "selected_nodeids": ["tests/unit/test_example.py::test_one"],
                "selected_nodeids_omitted": 0,
                "selected_count": 1,
            }
        )
    )
    identity = {
        "git_head": "head",
        "worktree_fingerprint": "tree",
        "python": "3.13",
        "skip_slow": True,
        "lab": False,
    }
    TESTMON_SEED_ATTEMPT.write_text(
        json.dumps(
            {
                "protocol_version": TESTMON_SEED_PROTOCOL_VERSION,
                "status": "running",
                "identity": identity,
                "expected_nodeids": [],
                "artifact_dir": str(outside),
            }
        )
    )

    assert _testmon_seed_can_resume(identity) is False


def test_resumed_seed_does_not_reuse_an_unexecuted_database_row(tmp_path: Path) -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    try:
        expected = ["tests/test_a.py::test_one", "tests/test_b.py::test_two"]
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        (artifact_dir / "selection.json").write_text(
            json.dumps({"selected_count": 1, "selected_nodeids": [expected[0]], "selected_nodeids_omitted": 0})
        )
        (artifact_dir / "events.jsonl").write_text(
            json.dumps({"event": "test_report", "nodeid": expected[0], "when": "call", "outcome": "passed"}) + "\n"
        )
        TESTMON_DATA.parent.mkdir(parents=True)
        with sqlite3.connect(TESTMON_DATA) as connection:
            connection.execute("create table environment (id integer primary key, environment_name text)")
            connection.execute("create table file_fp (id integer primary key, filename text, fsha text)")
            connection.execute("create table test_execution (id integer primary key, test_name text, failed integer)")
            connection.execute(
                "create table test_execution_file_fp (test_execution_id integer, fingerprint_id integer)"
            )
            connection.executemany("insert into test_execution values (?, ?, 0)", [(1, expected[0]), (2, expected[1])])
            connection.executemany("insert into file_fp values (?, ?, ?)", [(1, "a.py", "a"), (2, "b.py", "b")])
            connection.executemany("insert into test_execution_file_fp values (?, ?)", [(1, 1), (2, 2)])
        prepared = {
            "protocol_version": TESTMON_SEED_PROTOCOL_VERSION,
            "status": "running",
            "identity": {
                "git_head": "head",
                "worktree_fingerprint": "tree",
                "python": "python",
                "skip_slow": False,
                "lab": False,
                **_testmon_runtime_identity_fields(Path.cwd()),
            },
            "resume": True,
            "expected_nodeids": expected,
            "run_id": "resume",
            "artifact_dir": ".cache/verify/runs/resume",
        }
        _write_run_receipt(tmp_path, "resume")

        receipt = _finalize_testmon_seed_attempt(
            prepared=prepared,
            step_results=[{"name": "pytest seed-testmon (resume)", "artifact_dir": str(artifact_dir)}],
            exit_code=0,
        )

        assert receipt["status"] == "incomplete"
        assert {item["nodeid"]: item["outcome"] for item in receipt["node_outcomes"]} == {
            expected[0]: "passed",
            expected[1]: "missing",
        }

        (artifact_dir / "selection.json").write_text(json.dumps({}))
        (artifact_dir / "events.jsonl").write_text(
            "\n".join(
                json.dumps({"event": "test_report", "nodeid": nodeid, "when": "call", "outcome": "passed"})
                for nodeid in expected
            )
            + "\n"
        )
        missing_selection = _finalize_testmon_seed_attempt(
            prepared=prepared,
            step_results=[{"name": "pytest seed-testmon (resume)", "artifact_dir": str(artifact_dir)}],
            exit_code=0,
        )
        assert missing_selection["status"] == "incomplete"
    finally:
        monkeypatch.undo()


def test_testmon_database_state_reports_missing_and_failed_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    TESTMON_DATA.parent.mkdir(parents=True)
    with sqlite3.connect(TESTMON_DATA) as conn:
        conn.execute("CREATE TABLE environment (id INTEGER PRIMARY KEY, environment_name TEXT)")
        conn.execute("CREATE TABLE file_fp (id INTEGER PRIMARY KEY, filename TEXT, fsha TEXT)")
        conn.execute(
            "CREATE TABLE test_execution (id INTEGER PRIMARY KEY, test_name TEXT NOT NULL, failed INTEGER NOT NULL)"
        )
        conn.execute("CREATE TABLE test_execution_file_fp (test_execution_id INTEGER, fingerprint_id INTEGER)")
        conn.executemany(
            "INSERT INTO test_execution(test_name, failed) VALUES (?, ?)",
            [("tests/test_a.py::test_ok", 0), ("tests/test_b.py::test_failed", 1)],
        )
        conn.executemany(
            "INSERT INTO file_fp(id, filename, fsha) VALUES (?, ?, ?)",
            [(1, "a.py", "a"), (2, "b.py", "b")],
        )
        conn.executemany("INSERT INTO test_execution_file_fp VALUES (?, ?)", [(1, 1), (2, 2)])

    state = _testmon_database_state(
        ["tests/test_a.py::test_ok", "tests/test_b.py::test_failed", "tests/test_c.py::test_missing"]
    )

    assert state["recorded_count"] == 2
    assert state["failed_nodeids"] == ["tests/test_b.py::test_failed"]
    assert state["missing_nodeids"] == ["tests/test_c.py::test_missing"]
    assert state["node_outcomes"] == {
        "tests/test_a.py::test_ok": "passed",
        "tests/test_b.py::test_failed": "failed",
        "tests/test_c.py::test_missing": "missing",
    }


def test_worktree_fingerprint_hashes_untracked_file_contents(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], check=True)
    subprocess.run(["git", "config", "user.email", "tests@example.invalid"], check=True)
    subprocess.run(["git", "config", "user.name", "Polylogue Tests"], check=True)
    tracked = tmp_path / "tracked.py"
    tracked.write_text("VALUE = 1\n")
    subprocess.run(["git", "add", "tracked.py"], check=True)
    subprocess.run(["git", "commit", "-qm", "seed"], check=True)
    untracked = tmp_path / "candidate.py"
    untracked.write_text("VALUE = 1\n")

    before = _worktree_fingerprint()
    untracked.write_text("VALUE = 2\n")
    after = _worktree_fingerprint()

    assert before != after


def test_worktree_fingerprint_rejects_partial_git_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    real_run = subprocess.run

    def warning_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        result = cast(subprocess.CompletedProcess[bytes], real_run(*args, **kwargs))
        command = args[0]
        if isinstance(command, list) and command[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(command, 0, result.stdout, b"warning: partial enumeration\n")
        return result

    monkeypatch.setattr(subprocess, "run", warning_run)

    assert _worktree_fingerprint(tmp_path) == "unavailable"


def test_changed_paths_keep_start_time_base_when_remote_ref_advances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "tests@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Polylogue Tests"], cwd=tmp_path, check=True)
    source = tmp_path / "polylogue" / "example.py"
    source.parent.mkdir()
    source.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "polylogue/example.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(["git", "switch", "-qc", "feature"], cwd=tmp_path, check=True)
    source.write_text("value = 2\n", encoding="utf-8")
    subprocess.run(["git", "commit", "-qam", "feature"], cwd=tmp_path, check=True)
    feature_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(["git", "update-ref", "refs/remotes/origin/master", base], cwd=tmp_path, check=True)
    monkeypatch.setattr(verify, "ROOT", tmp_path)

    pinned_base = verify._git_commit("origin/master")
    assert pinned_base == base
    subprocess.run(["git", "update-ref", "refs/remotes/origin/master", "HEAD"], cwd=tmp_path, check=True)

    assert verify._changed_executable_paths(pinned_base, feature_head) == ("polylogue/example.py",)


def test_changed_paths_include_untracked_executable_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "tests@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Polylogue Tests"], cwd=tmp_path, check=True)
    tracked = tmp_path / "README.md"
    tracked.write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True
    ).stdout.strip()
    untracked = tmp_path / "devtools" / "new_command.py"
    untracked.parent.mkdir()
    untracked.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(verify, "ROOT", tmp_path)

    assert verify._changed_executable_paths(head, head) == ("devtools/new_command.py",)


def test_checkout_mutation_monitor_detects_a_change_that_reverts_before_the_final_fingerprint(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q"], check=True)
    subprocess.run(["git", "config", "user.email", "tests@example.invalid"], check=True)
    subprocess.run(["git", "config", "user.name", "Polylogue Tests"], check=True)
    tracked = tmp_path / "tracked.py"
    original = "VALUE = 1\n"
    tracked.write_text(original, encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], check=True)
    subprocess.run(["git", "commit", "-qm", "seed"], check=True)

    monitor = CheckoutMutationMonitor(tmp_path)
    monitor.start()
    tracked.write_text("VALUE = 2\n", encoding="utf-8")
    tracked.write_text(original, encoding="utf-8")
    observation = monitor.finish()

    assert observation.changed is True
    assert observation.unavailable is False
    assert observation.observed_path == "tracked.py"


def test_checkout_mutation_monitor_ignores_nested_disposable_cache_writes(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    package = tmp_path / "package"
    package.mkdir()
    monitor = CheckoutMutationMonitor(tmp_path)
    monitor.start()
    cache_file = package / "__pycache__" / "module.pyc"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"cache")
    observation = monitor.finish()

    assert observation == CheckoutMutationObservation(changed=False, unavailable=False)


def test_checkout_mutation_monitor_uses_gitignore_for_verifier_task_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(["git", "init", "-q"], check=True)
    (tmp_path / ".gitignore").write_text(".agent/*\n", encoding="utf-8")
    history = tmp_path / ".agent" / "task-history" / "tasks.jsonl"
    history.parent.mkdir(parents=True)

    def portable_watch(*_paths: Path, **kwargs: object) -> object:
        yield set()
        yield {(watchfiles.Change.modified, str(history))}
        stop_event = kwargs["stop_event"]
        assert isinstance(stop_event, threading.Event)
        stop_event.wait()

    monkeypatch.setattr(watchfiles, "watch", portable_watch)
    monitor = CheckoutMutationMonitor(tmp_path)
    monitor.start()
    observation = monitor.finish()

    assert observation == CheckoutMutationObservation(changed=False, unavailable=False)


def test_checkout_mutation_monitor_observes_tracked_file_that_matches_gitignore(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text(".agent/*\n", encoding="utf-8")
    tracked = tmp_path / ".agent" / "script.py"
    tracked.parent.mkdir()
    tracked.write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "add", "-f", ".agent/script.py"], cwd=tmp_path, check=True)

    monitor = CheckoutMutationMonitor(tmp_path)
    monitor.start()
    tracked.write_text("during\n", encoding="utf-8")
    tracked.write_text("before\n", encoding="utf-8")
    observation = monitor.finish()

    assert observation == CheckoutMutationObservation(
        changed=True,
        unavailable=False,
        observed_path=".agent/script.py",
    )


def test_checkout_mutation_monitor_uses_portable_watchfiles_events_without_linux_kernel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(["git", "init", "-q"], check=True)
    tracked = tmp_path / "tracked.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    calls: dict[str, object] = {}
    event_emitted = threading.Event()

    def portable_watch(*paths: Path, **kwargs: object) -> object:
        calls["paths"] = paths
        calls["kwargs"] = kwargs
        yield set()
        event_emitted.set()
        yield {(watchfiles.Change.modified, str(tracked))}

    monkeypatch.setattr(watchfiles, "watch", portable_watch)
    monitor = CheckoutMutationMonitor(tmp_path)
    monitor.start()
    assert event_emitted.wait(timeout=1)
    observation = monitor.finish()

    assert calls["paths"] == (tmp_path.resolve(), (tmp_path / ".git").resolve())
    assert calls["kwargs"] == {
        "watch_filter": None,
        "debounce": 0,
        "step": 1,
        "stop_event": monitor._stop,
        "rust_timeout": monitor._WATCH_RUST_TIMEOUT_MS,
        "yield_on_timeout": True,
        "raise_interrupt": False,
        "force_polling": False,
        "recursive": False,
    }
    assert observation == CheckoutMutationObservation(changed=True, unavailable=False, observed_path="tracked.py")


def test_checkout_mutation_monitor_rejects_forced_polling_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WATCHFILES_FORCE_POLLING", "1")

    def unexpected_watch(*_paths: Path, **_kwargs: object) -> object:
        raise AssertionError("polling mode must fail before watchfiles starts")
        yield set()

    monkeypatch.setattr(watchfiles, "watch", unexpected_watch)
    monitor = CheckoutMutationMonitor(tmp_path)
    monitor.start()
    observation = monitor.finish()

    assert observation == CheckoutMutationObservation(changed=False, unavailable=True)


def test_checkout_mutation_monitor_rejects_wsl_auto_polling_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WATCHFILES_FORCE_POLLING", raising=False)
    monkeypatch.setattr(
        platform,
        "uname",
        lambda: SimpleNamespace(system="Linux", release="6.6.0-microsoft-standard-WSL2"),
    )

    monitor = CheckoutMutationMonitor(tmp_path)
    monitor.start()
    observation = monitor.finish()

    assert observation == CheckoutMutationObservation(changed=False, unavailable=True)


def test_checkout_mutation_monitor_prunes_disposable_trees_and_observes_new_source_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(["git", "init", "-q"], check=True)
    (tmp_path / ".gitignore").write_text(
        "browser-extension/node_modules/\ncustom/generated-output/\n",
        encoding="utf-8",
    )
    source = tmp_path / "src" / "package"
    source.mkdir(parents=True)
    for disposable in (".venv", ".git", ".cache"):
        (tmp_path / disposable / "nested").mkdir(parents=True, exist_ok=True)
    ignored_dependency = tmp_path / "browser-extension" / "node_modules" / "dependency"
    ignored_dependency.mkdir(parents=True)
    ignored_build = tmp_path / "custom" / "generated-output" / "deep" / "tree"
    ignored_build.mkdir(parents=True)
    calls: dict[str, object] = {}
    allow_event = threading.Event()
    new_source = tmp_path / "new_source"

    def portable_watch(*paths: Path, **kwargs: object) -> object:
        calls["paths"] = paths
        calls["kwargs"] = kwargs
        yield set()
        assert allow_event.wait(timeout=1)
        yield {(watchfiles.Change.added, str(new_source))}

    monkeypatch.setattr(watchfiles, "watch", portable_watch)
    monitor = CheckoutMutationMonitor(tmp_path)
    monitor.start()
    new_source.mkdir()
    allow_event.set()
    observation = monitor.finish()

    raw_paths = calls["paths"]
    raw_kwargs = calls["kwargs"]
    assert isinstance(raw_paths, tuple)
    assert isinstance(raw_kwargs, dict)
    watched = {Path(path) for path in raw_paths}
    assert tmp_path.resolve() in watched
    assert source in watched
    assert all(
        path == (tmp_path / ".git").resolve() or not any(part in {".venv", ".git", ".cache"} for part in path.parts)
        for path in watched
    )
    assert all("node_modules" not in path.parts for path in watched)
    assert all("generated-output" not in path.parts for path in watched)
    assert tmp_path / "browser-extension" in watched
    assert tmp_path / "custom" in watched
    assert raw_kwargs["recursive"] is False
    assert observation == CheckoutMutationObservation(changed=True, unavailable=False, observed_path="new_source")


def test_checkout_mutation_monitor_remembers_deleted_ignored_root(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("browser-extension/node_modules/\n", encoding="utf-8")
    ignored_root = tmp_path / "browser-extension" / "node_modules"
    ignored_root.mkdir(parents=True)

    monitor = CheckoutMutationMonitor(tmp_path)
    monitor._watched_directories()
    shutil.rmtree(ignored_root)
    monitor._record_change(ignored_root)

    assert monitor.finish() == CheckoutMutationObservation(changed=False, unavailable=False)


@pytest.mark.uses_real_clock("waits for the real filesystem watcher to witness an index replacement")
def test_checkout_mutation_monitor_observes_transient_index_authority_change(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    baseline = tmp_path / "baseline.py"
    baseline.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore", "baseline.py"], cwd=tmp_path, check=True)
    monitor = CheckoutMutationMonitor(tmp_path)
    monitor.start()
    hidden = tmp_path / "ignored" / "hidden.py"
    hidden.parent.mkdir()
    hidden.write_text("secret authority\n", encoding="utf-8")
    subprocess.run(["git", "add", "-f", "ignored/hidden.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "reset", "-q", "--", "ignored/hidden.py"], cwd=tmp_path, check=True)
    observation = monitor.finish()

    assert observation == CheckoutMutationObservation(changed=True, unavailable=False, observed_path=".git/index")


def test_checkout_mutation_monitor_ignores_uncommitted_git_index_lock(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)

    monitor = CheckoutMutationMonitor(tmp_path)
    monitor._watched_directories()
    monitor._record_change(tmp_path / ".git" / "index.lock")
    observation = monitor.finish()

    assert observation == CheckoutMutationObservation(changed=False, unavailable=False)


def test_checkout_mutation_monitor_treats_every_ready_index_event_as_authority_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)
    index = tmp_path / ".git" / "index"

    def portable_watch(*_paths: Path, **_kwargs: object) -> object:
        yield set()
        yield {(watchfiles.Change.modified, str(index))}
        stop_event = _kwargs["stop_event"]
        assert isinstance(stop_event, threading.Event)
        stop_event.wait(timeout=1)

    monkeypatch.setattr(watchfiles, "watch", portable_watch)
    monitor = CheckoutMutationMonitor(tmp_path)
    monitor.start()
    observation = monitor.finish()

    assert observation == CheckoutMutationObservation(changed=True, unavailable=False, observed_path=".git/index")


def test_checkout_mutation_monitor_rejects_partial_git_enumeration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    real_run = subprocess.run

    def warning_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        result = cast(subprocess.CompletedProcess[bytes], real_run(*args, **kwargs))
        command = args[0]
        if isinstance(command, list) and command[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(command, 0, result.stdout, b"warning: partial enumeration\n")
        return result

    monkeypatch.setattr(subprocess, "run", warning_run)
    monitor = CheckoutMutationMonitor(tmp_path)
    monitor.start()
    observation = monitor.finish()

    assert observation == CheckoutMutationObservation(changed=False, unavailable=True)


def test_checkout_mutation_monitor_rejects_filesystem_walk_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    real_walk = os.walk

    def failing_walk(top: Path, *, onerror: object = None) -> object:
        assert callable(onerror)
        onerror(PermissionError("unreadable source directory"))
        yield from real_walk(top)

    monkeypatch.setattr(os, "walk", failing_walk)
    monitor = CheckoutMutationMonitor(tmp_path)
    monitor.start()
    observation = monitor.finish()

    assert observation == CheckoutMutationObservation(changed=False, unavailable=True)


def test_checkout_mutation_monitor_fails_closed_when_portable_watcher_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_watch(*_paths: Path, **_kwargs: object) -> object:
        raise OSError("watcher unavailable")
        yield set()

    monkeypatch.setattr(watchfiles, "watch", broken_watch)
    monitor = CheckoutMutationMonitor(tmp_path)
    monitor.start()
    observation = monitor.finish()

    assert observation == CheckoutMutationObservation(changed=False, unavailable=True)


def test_seed_receipt_classifies_every_node_terminal_outcome(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    expected = [
        "tests/test_seed.py::test_passed",
        "tests/test_seed.py::test_failed",
        "tests/test_seed.py::test_error",
        "tests/test_seed.py::test_timeout",
        "tests/test_seed.py::test_worker_crash",
        "tests/test_seed.py::test_missing",
    ]
    (artifact_dir / "selection.json").write_text(
        json.dumps(
            {
                "selected_count": len(expected),
                "deselected_count": 0,
                "selected_nodeids": expected,
                "selected_nodeids_omitted": 0,
            }
        )
    )
    events = [
        {"event": "test_report", "nodeid": expected[0], "when": "call", "outcome": "passed"},
        {
            "event": "test_report",
            "nodeid": expected[1],
            "when": "call",
            "outcome": "failed",
            "longrepr": "assert false",
        },
        {
            "event": "test_report",
            "nodeid": expected[2],
            "when": "setup",
            "outcome": "failed",
            "longrepr": "fixture exploded",
        },
        {
            "event": "test_report",
            "nodeid": expected[3],
            "when": "call",
            "outcome": "failed",
            "longrepr": "Failed: Timeout > 10s",
        },
        {"event": "test_started", "nodeid": expected[4]},
    ]
    (artifact_dir / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    TESTMON_DATA.parent.mkdir(parents=True)
    with sqlite3.connect(TESTMON_DATA) as conn:
        conn.execute("CREATE TABLE environment (id INTEGER PRIMARY KEY, environment_name TEXT)")
        conn.execute("CREATE TABLE file_fp (id INTEGER PRIMARY KEY, filename TEXT, fsha TEXT)")
        conn.execute(
            "CREATE TABLE test_execution (id INTEGER PRIMARY KEY, test_name TEXT NOT NULL, failed INTEGER NOT NULL)"
        )
        conn.execute("CREATE TABLE test_execution_file_fp (test_execution_id INTEGER, fingerprint_id INTEGER)")
        conn.executemany(
            "INSERT INTO test_execution(test_name, failed) VALUES (?, ?)",
            [(nodeid, int(nodeid != expected[0])) for nodeid in expected[:-1]],
        )
        conn.executemany(
            "INSERT INTO file_fp(id, filename, fsha) VALUES (?, ?, ?)",
            [(index, f"file-{index}.py", f"sha-{index}") for index, _nodeid in enumerate(expected[:-1], start=1)],
        )
        conn.executemany(
            "INSERT INTO test_execution_file_fp VALUES (?, ?)",
            [(index, index) for index, _nodeid in enumerate(expected[:-1], start=1)],
        )

    _write_run_receipt(tmp_path, "run-mixed")
    receipt = _finalize_testmon_seed_attempt(
        prepared={
            "protocol_version": TESTMON_SEED_PROTOCOL_VERSION,
            "status": "running",
            "identity": {
                "git_head": "head",
                "worktree_fingerprint": "tree",
                "python": "python",
                "skip_slow": False,
                "lab": False,
                **_testmon_runtime_identity_fields(Path.cwd()),
            },
            "resume": False,
            "expected_nodeids": [],
            "run_id": "run-mixed",
            "artifact_dir": ".cache/verify/runs/run-mixed",
        },
        step_results=[
            {
                "name": "pytest seed-testmon",
                "artifact_dir": str(artifact_dir),
                "exit": 1,
                "diagnosis": "xdist_worker_crash",
            }
        ],
        exit_code=1,
    )

    assert receipt["status"] == "incomplete"
    assert {item["nodeid"]: item["outcome"] for item in receipt["node_outcomes"]} == {
        expected[0]: "passed",
        expected[1]: "failed",
        expected[2]: "error",
        expected[3]: "timeout",
        expected[4]: "worker_crash",
        expected[5]: "missing",
    }
    assert receipt["node_outcome_counts"] == {
        "error": 1,
        "failed": 1,
        "missing": 1,
        "passed": 1,
        "timeout": 1,
        "worker_crash": 1,
    }


def test_seed_node_outcomes_preserve_interrupted_active_node(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(json.dumps({"event": "test_started", "nodeid": "tests/test_a.py::test_active"}) + "\n")

    outcomes = _seed_node_outcomes_from_events(
        events,
        expected_nodeids=["tests/test_a.py::test_active"],
        database={"node_outcomes": {"tests/test_a.py::test_active": "missing"}},
        pytest_step={"diagnosis": "terminated by signal"},
    )

    assert outcomes[0]["outcome"] == "interrupted"


def test_seed_node_outcomes_keep_unconfirmed_teardown_incomplete(tmp_path: Path) -> None:
    """A terminal teardown does not prove that the missing call phase passed."""
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            [
                json.dumps({"event": "test_started", "nodeid": "tests/test_a.py::test_finished"}),
                json.dumps(
                    {
                        "event": "test_report",
                        "nodeid": "tests/test_a.py::test_finished",
                        "when": "teardown",
                        "outcome": "passed",
                    }
                ),
                json.dumps({"event": "test_finished", "nodeid": "tests/test_a.py::test_finished"}),
            ]
        )
        + "\n"
    )

    outcomes = _seed_node_outcomes_from_events(
        events,
        expected_nodeids=["tests/test_a.py::test_finished"],
        database={"node_outcomes": {"tests/test_a.py::test_finished": "missing"}},
        pytest_step={"diagnosis": "pytest_failed"},
    )

    assert outcomes[0]["outcome"] == "missing"
    assert outcomes[0]["reason"] == "passing teardown without call report or testmon result"


def test_seed_resource_timeout_has_a_distinct_typed_terminal_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A supervisor resource stop must not collapse into generic incompleteness."""
    monkeypatch.chdir(tmp_path)
    expected = ["tests/test_seed.py::test_active"]
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "selection.json").write_text(
        json.dumps(
            {
                "selected_count": len(expected),
                "deselected_count": 0,
                "selected_nodeids": expected,
                "selected_nodeids_omitted": 0,
            }
        )
    )
    (artifact_dir / "events.jsonl").write_text(json.dumps({"event": "test_started", "nodeid": expected[0]}) + "\n")
    TESTMON_DATA.parent.mkdir(parents=True)
    with sqlite3.connect(TESTMON_DATA) as connection:
        connection.execute("create table environment (id integer primary key, environment_name text)")
        connection.execute("create table file_fp (id integer primary key, filename text, fsha text)")
        connection.execute("create table test_execution (id integer primary key, test_name text, failed integer)")
        connection.execute("create table test_execution_file_fp (test_execution_id integer, fingerprint_id integer)")
    _write_run_receipt(tmp_path, "run-resource-timeout")

    receipt = _finalize_testmon_seed_attempt(
        prepared={
            "protocol_version": TESTMON_SEED_PROTOCOL_VERSION,
            "status": "running",
            "identity": {
                "git_head": "head",
                "worktree_fingerprint": "tree",
                "python": "python",
                "skip_slow": False,
                "lab": False,
                **_testmon_runtime_identity_fields(Path.cwd()),
            },
            "resume": False,
            "expected_nodeids": [],
            "run_id": "run-resource-timeout",
            "artifact_dir": ".cache/verify/runs/run-resource-timeout",
        },
        step_results=[
            {
                "name": "pytest seed-testmon",
                "artifact_dir": str(artifact_dir),
                "exit": 124,
                "diagnosis": "pytest_terminated",
                "termination_reason": "pytest tmpfs budget exceeded: 512.0 MiB > 500 MiB",
            }
        ],
        exit_code=124,
    )

    assert receipt["status"] == "incomplete"
    assert receipt["outcome"] == "resource-timeout"
    assert receipt["release_baseline_allowed"] is False


def test_seed_node_outcomes_accept_setup_skip_as_terminal_skip(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps(
            {
                "event": "test_report",
                "nodeid": "tests/test_a.py::test_setup_skip",
                "when": "setup",
                "outcome": "skipped",
            }
        )
        + "\n"
    )

    outcomes = _seed_node_outcomes_from_events(
        events,
        expected_nodeids=["tests/test_a.py::test_setup_skip"],
        database={"node_outcomes": {"tests/test_a.py::test_setup_skip": "missing"}},
        pytest_step={},
        use_database_fallback=False,
    )

    assert outcomes == [
        {
            "nodeid": "tests/test_a.py::test_setup_skip",
            "outcome": "skipped",
            "reason": "test setup or teardown skipped",
            "started": False,
            "finished": False,
            "phases": [{"when": "setup", "outcome": "skipped", "duration_s": None}],
        }
    ]


def test_seed_node_outcomes_preserve_call_and_fixture_xfail_xpass(tmp_path: Path) -> None:
    """Durable pytest reports, including fixture ``pytest.xfail()``, finish seed nodes."""
    events = tmp_path / "events.jsonl"
    nodes = [
        "tests/test_a.py::test_call_xfailed",
        "tests/test_a.py::test_call_xpassed",
        "tests/test_a.py::test_setup_xfailed",
    ]
    events.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {"event": "test_report", "nodeid": nodes[0], "when": "call", "outcome": "xfailed"},
                {"event": "test_report", "nodeid": nodes[1], "when": "call", "outcome": "xpassed"},
                {"event": "test_report", "nodeid": nodes[2], "when": "setup", "outcome": "xfailed"},
            )
        )
        + "\n"
    )

    outcomes = _seed_node_outcomes_from_events(
        events,
        expected_nodeids=nodes,
        database={"node_outcomes": {}},
        pytest_step={},
        use_database_fallback=False,
    )

    assert {item["nodeid"]: item["outcome"] for item in outcomes} == dict(
        zip(nodes, ("xfailed", "xpassed", "xfailed"), strict=True)
    )


def test_resumed_seed_carries_forward_prior_terminal_outcome(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps(
            {"event": "test_report", "nodeid": "tests/test_a.py::test_repaired", "when": "call", "outcome": "passed"}
        )
        + "\n"
    )
    outcomes = _seed_node_outcomes_from_events(
        events,
        expected_nodeids=[
            "tests/test_a.py::test_repaired",
            "tests/test_b.py::test_prior",
            "tests/test_c.py::test_expected_failure",
        ],
        database={"node_outcomes": {"tests/test_b.py::test_prior": "passed"}},
        pytest_step={},
        use_database_fallback=False,
        prior_node_outcomes={
            "tests/test_b.py::test_prior": {"nodeid": "tests/test_b.py::test_prior", "outcome": "passed"},
            "tests/test_c.py::test_expected_failure": {
                "nodeid": "tests/test_c.py::test_expected_failure",
                "outcome": "xfailed",
            },
        },
    )

    assert {item["nodeid"]: item["outcome"] for item in outcomes} == {
        "tests/test_a.py::test_repaired": "passed",
        "tests/test_b.py::test_prior": "passed",
        "tests/test_c.py::test_expected_failure": "xfailed",
    }


def test_seed_completion_requires_full_failure_free_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    expected = ["tests/test_a.py::test_one", "tests/test_b.py::test_two"]
    (artifact_dir / "selection.json").write_text(
        json.dumps(
            {
                "selected_count": 2,
                "deselected_count": 0,
                "selected_nodeids": expected,
                "selected_nodeids_omitted": 0,
            }
        )
    )
    (artifact_dir / "events.jsonl").write_text(
        "".join(
            json.dumps({"event": "test_report", "nodeid": nodeid, "when": "call", "outcome": "passed"}) + "\n"
            for nodeid in expected
        )
    )
    TESTMON_DATA.parent.mkdir(parents=True)
    with sqlite3.connect(TESTMON_DATA) as conn:
        conn.execute("CREATE TABLE environment (id INTEGER PRIMARY KEY, environment_name TEXT)")
        conn.execute("CREATE TABLE file_fp (id INTEGER PRIMARY KEY, filename TEXT, fsha TEXT)")
        conn.execute(
            "CREATE TABLE test_execution (id INTEGER PRIMARY KEY, test_name TEXT NOT NULL, failed INTEGER NOT NULL)"
        )
        conn.execute("CREATE TABLE test_execution_file_fp (test_execution_id INTEGER, fingerprint_id INTEGER)")
        conn.executemany(
            "INSERT INTO test_execution(test_name, failed) VALUES (?, 0)",
            [(nodeid,) for nodeid in expected],
        )
        conn.executemany(
            "INSERT INTO file_fp(id, filename, fsha) VALUES (?, ?, ?)",
            [(index, f"file-{index}.py", f"sha-{index}") for index, _nodeid in enumerate(expected, start=1)],
        )
        conn.executemany(
            "INSERT INTO test_execution_file_fp VALUES (?, ?)",
            [(index, index) for index, _nodeid in enumerate(expected, start=1)],
        )

    _write_run_receipt(tmp_path, "run-1")
    _write_run_receipt(tmp_path, "run-stale-db")
    _write_run_receipt(tmp_path, "run-orphaned")
    receipt = _finalize_testmon_seed_attempt(
        prepared={
            "protocol_version": TESTMON_SEED_PROTOCOL_VERSION,
            "status": "running",
            "identity": {
                "git_head": "head",
                "worktree_fingerprint": "tree",
                "python": "python",
                "skip_slow": False,
                "lab": False,
                **_testmon_runtime_identity_fields(Path.cwd()),
            },
            "resume": False,
            "expected_nodeids": [],
            "run_id": "run-1",
            "artifact_dir": ".cache/verify/runs/run-1",
        },
        step_results=[{"name": "pytest seed-testmon", "artifact_dir": str(artifact_dir), "exit": 0}],
        exit_code=0,
    )

    assert receipt["status"] == "complete"
    assert receipt["expected_count"] == 2
    stamp = json.loads((tmp_path / ".cache" / "testmon" / "seed.json").read_text())
    assert stamp["status"] == "usable"
    assert stamp["collection"]["expected_count"] == 2

    _write_run_receipt(tmp_path, "run-authorized")
    authorized_receipt = _finalize_testmon_seed_attempt(
        prepared={
            "protocol_version": TESTMON_SEED_PROTOCOL_VERSION,
            "status": "running",
            "identity": {
                "git_head": "head",
                "git_tree": "tree-hash",
                "worktree_fingerprint": "tree",
                "python": "python",
                "skip_slow": True,
                "lab": False,
                "terminal_authorization": "narrow-terminal",
                **_testmon_runtime_identity_fields(Path.cwd()),
            },
            "resume": False,
            "expected_nodeids": [],
            "run_id": "run-authorized",
            "artifact_dir": ".cache/verify/runs/run-authorized",
        },
        step_results=[{"name": "pytest seed-testmon", "artifact_dir": str(artifact_dir), "exit": 0}],
        exit_code=0,
    )
    assert authorized_receipt["status"] == "complete"
    assert authorized_receipt["release_baseline_allowed"] is True

    _write_run_receipt(tmp_path, "run-red")
    with sqlite3.connect(TESTMON_DATA) as connection:
        connection.execute("update test_execution set failed = 1 where test_name = ?", (expected[0],))
    red_receipt = _finalize_testmon_seed_attempt(
        prepared={
            "protocol_version": TESTMON_SEED_PROTOCOL_VERSION,
            "status": "running",
            "identity": {
                "git_head": "head",
                "worktree_fingerprint": "tree",
                "python": "python",
                "skip_slow": False,
                "lab": False,
                **_testmon_runtime_identity_fields(Path.cwd()),
            },
            "resume": False,
            "expected_nodeids": [],
            "run_id": "run-red",
            "artifact_dir": ".cache/verify/runs/run-red",
        },
        step_results=[{"name": "pytest seed-testmon", "artifact_dir": str(artifact_dir), "exit": 1}],
        exit_code=1,
    )
    assert red_receipt["status"] == "reusable"
    assert red_receipt["release_baseline_allowed"] is False
    persisted_attempt = json.loads((tmp_path / ".cache" / "testmon" / "seed-attempt.json").read_text())
    assert persisted_attempt["release_baseline_allowed"] is False
    assert not (tmp_path / ".cache" / "testmon" / "seed.json").exists()

    (artifact_dir / "events.jsonl").write_text("")
    stale_database = _finalize_testmon_seed_attempt(
        prepared={
            "protocol_version": TESTMON_SEED_PROTOCOL_VERSION,
            "status": "running",
            "identity": {
                "git_head": "head",
                "worktree_fingerprint": "tree",
                "python": "python",
                "skip_slow": True,
                "lab": False,
            },
            "resume": False,
            "expected_nodeids": [],
            "run_id": "run-stale-db",
            "artifact_dir": ".cache/verify/runs/run-stale-db",
        },
        step_results=[{"name": "pytest seed-testmon", "artifact_dir": str(artifact_dir)}],
        exit_code=0,
    )
    assert stale_database["status"] == "incomplete"

    with sqlite3.connect(TESTMON_DATA) as connection:
        connection.execute("insert into test_execution_file_fp values (999, 1)")
    orphaned = _finalize_testmon_seed_attempt(
        prepared={
            "protocol_version": TESTMON_SEED_PROTOCOL_VERSION,
            "status": "running",
            "identity": {
                "git_head": "head",
                "worktree_fingerprint": "tree",
                "python": "python",
                "skip_slow": True,
                "lab": False,
            },
            "resume": False,
            "expected_nodeids": [],
            "run_id": "run-orphaned",
            "artifact_dir": ".cache/verify/runs/run-orphaned",
        },
        step_results=[{"name": "pytest seed-testmon", "artifact_dir": str(artifact_dir)}],
        exit_code=0,
    )
    assert orphaned["status"] == "incomplete"


def test_resumed_seed_persists_full_selection_before_stamp_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    expected = ["tests/test_a.py::test_one", "tests/test_b.py::test_two"]
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "selection.json").write_text(
        json.dumps({"selected_count": 1, "selected_nodeids": [expected[0]], "selected_nodeids_omitted": 0})
    )
    (artifact_dir / "events.jsonl").write_text(
        json.dumps({"event": "test_report", "nodeid": expected[0], "when": "call", "outcome": "passed"}) + "\n"
    )
    TESTMON_DATA.parent.mkdir(parents=True)
    with sqlite3.connect(TESTMON_DATA) as connection:
        connection.execute("create table environment (id integer primary key, environment_name text)")
        connection.execute("create table file_fp (id integer primary key, filename text, fsha text)")
        connection.execute("create table test_execution (id integer primary key, test_name text, failed integer)")
        connection.execute("create table test_execution_file_fp (test_execution_id integer, fingerprint_id integer)")
        connection.executemany("insert into test_execution values (?, ?, 0)", [(1, expected[0]), (2, expected[1])])
        connection.executemany("insert into file_fp values (?, ?, ?)", [(1, "a.py", "a"), (2, "b.py", "b")])
        connection.executemany("insert into test_execution_file_fp values (?, ?)", [(1, 1), (2, 2)])
    _write_run_receipt(tmp_path, "resumed")
    prepared = {
        "protocol_version": TESTMON_SEED_PROTOCOL_VERSION,
        "status": "running",
        "identity": {
            "git_head": "head",
            "git_tree": "tree-hash",
            "worktree_fingerprint": "tree",
            "python": "python",
            "skip_slow": False,
            "lab": False,
            "terminal_authorization": None,
            **_testmon_runtime_identity_fields(Path.cwd()),
        },
        "resume": True,
        "expected_nodeids": expected,
        "expected_count": len(expected),
        "expected_digest": hashlib.sha256("\n".join(sorted(expected)).encode()).hexdigest(),
        "prior_node_outcomes": [{"nodeid": expected[1], "outcome": "passed"}],
        "run_id": "resumed",
        "artifact_dir": ".cache/verify/runs/resumed",
    }
    original_write = verify._atomic_write_json

    def crash_before_stamp(path: Path, payload: object) -> None:
        if path == TESTMON_SEED_STAMP:
            raise RuntimeError("simulated crash before seed publication")
        assert isinstance(payload, dict)
        original_write(path, payload)

    with patch("devtools.verify._atomic_write_json", side_effect=crash_before_stamp):
        with pytest.raises(RuntimeError, match="before seed publication"):
            _finalize_testmon_seed_attempt(
                prepared=prepared,
                step_results=[{"name": "pytest seed-testmon (resume)", "artifact_dir": str(artifact_dir)}],
                exit_code=0,
            )

    persisted = json.loads(TESTMON_SEED_ATTEMPT.read_text())
    assert persisted["status"] == "complete"
    assert persisted["expected_count"] == len(expected)
    assert persisted["selection"]["selected_count"] == len(expected)
    assert persisted["selection"]["selected_nodeids_omitted"] == 0
    assert not TESTMON_SEED_STAMP.exists()


def test_classify_late_sigterm_after_pytest_success_summary() -> None:
    diagnosis = classify_pytest_result(
        returncode=-15,
        termination_reason=None,
        report_present=False,
        summary={"exitstatus": 0},
        progress_event="finished",
    )

    assert diagnosis == "report_missing_after_sessionfinish_success"


def test_resource_sampler_records_process_tree_sample(tmp_path: Path) -> None:
    path = tmp_path / "resources.jsonl"
    sampler = ResourceSampler(
        root_pid=os.getpid(),
        run_id="test-run",
        root=tmp_path,
        env={"POLYLOGUE_PYTEST_BASETEMP_ROOT": str(tmp_path)},
        output_path=path,
    )

    sample = sampler.sample(event="sample")
    summary = sampler.summary()

    assert sample["process_count"] >= 1
    assert sample["tree_rss_kb"] > 0
    assert summary["resource_sample_count"] == 1
    assert summary["peak_tree_rss_kb"] == sample["tree_rss_kb"]
    assert "peak_tree_anon_pss_kb" in summary
    assert "peak_tree_file_pss_kb" in summary
    assert "peak_tree_swap_pss_kb" in summary
    assert "tree_read_bytes_delta" in summary
    assert "tree_write_bytes_delta" in summary


def test_xdist_uninterruptible_stall_requires_a_complete_timeout() -> None:
    sample = {
        "xdist_worker_count": 6,
        "xdist_uninterruptible_count": 6,
        "all_xdist_workers_uninterruptible": True,
    }

    assert xdist_uninterruptible_stall_reason(sample, started_at=10.0, now=19.0, timeout_s=10.0) is None
    reason = xdist_uninterruptible_stall_reason(sample, started_at=10.0, now=20.1, timeout_s=10.0)
    assert reason is not None
    assert "6 workers" in reason
    assert "uninterruptible I/O sleep" in reason


def test_xdist_uninterruptible_stall_ignores_partial_or_moving_workers() -> None:
    assert (
        xdist_uninterruptible_stall_reason(
            {
                "xdist_worker_count": 6,
                "xdist_uninterruptible_count": 5,
                "all_xdist_workers_uninterruptible": False,
            },
            started_at=10.0,
            now=100.0,
            timeout_s=10.0,
        )
        is None
    )


def test_resource_sampler_resolves_worker_identity_from_in_process_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = tmp_path / "events"
    events.mkdir()
    (events / "worker.jsonl").write_text(
        json.dumps({"event": "session_started", "pid": 101, "worker_id": "gw0"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("devtools.verify_runs.process_tree", lambda _root_pid: [101])
    monkeypatch.setattr("devtools.verify_runs._status_values", lambda _pid: {"state": "D", "rss_kb": 30})
    monkeypatch.setattr("devtools.verify_runs._smaps_rollup_kb", lambda _pid: {})
    monkeypatch.setattr("devtools.verify_runs._process_io_bytes", lambda _pid: {})
    monkeypatch.setattr("devtools.verify_runs._process_identity", lambda _pid: "101:1")
    monkeypatch.setattr("devtools.verify_runs._cpu_seconds", lambda _pid: 1.0)
    monkeypatch.setattr("devtools.verify_runs._process_environ_value", lambda _pid, _key: None)

    sampler = ResourceSampler(
        root_pid=101,
        run_id="worker-events",
        root=tmp_path,
        env={
            "POLYLOGUE_PYTEST_BASETEMP_ROOT": str(tmp_path),
            "POLYLOGUE_PYTEST_EVENTS_DIR": str(events),
        },
        output_path=tmp_path / "resources.jsonl",
    )

    sample = sampler.sample(event="sample")

    assert sample["xdist_worker_count"] == 1
    assert sample["xdist_uninterruptible_count"] == 1


def test_six_worker_d_state_fixture_produces_typed_stall_diagnosis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = tmp_path / "events"
    events.mkdir()
    for index, pid in enumerate(range(201, 207)):
        (events / f"gw{index}.jsonl").write_text(
            json.dumps({"event": "session_started", "pid": pid, "worker_id": f"gw{index}"}) + "\n",
            encoding="utf-8",
        )
    monkeypatch.setattr("devtools.verify_runs.process_tree", lambda _root_pid: list(range(201, 207)))
    monkeypatch.setattr("devtools.verify_runs._status_values", lambda _pid: {"state": "D", "rss_kb": 30})
    monkeypatch.setattr("devtools.verify_runs._smaps_rollup_kb", lambda _pid: {})
    monkeypatch.setattr("devtools.verify_runs._process_io_bytes", lambda _pid: {})
    monkeypatch.setattr("devtools.verify_runs._process_identity", lambda pid: f"{pid}:1")
    monkeypatch.setattr("devtools.verify_runs._cpu_seconds", lambda _pid: 1.0)
    monkeypatch.setattr("devtools.verify_runs._process_environ_value", lambda _pid, _key: None)

    sampler = ResourceSampler(
        root_pid=201,
        run_id="six-worker-d-state",
        root=tmp_path,
        env={
            "POLYLOGUE_PYTEST_BASETEMP_ROOT": str(tmp_path),
            "POLYLOGUE_PYTEST_EVENTS_DIR": str(events),
        },
        output_path=tmp_path / "resources.jsonl",
    )
    sample = sampler.sample(event="sample")

    assert sample["xdist_worker_count"] == 6
    assert sample["xdist_uninterruptible_count"] == 6
    reason = xdist_uninterruptible_stall_reason(sample, started_at=10.0, now=40.1, timeout_s=30.0)
    assert (
        reason
        == "pytest xdist workers remained in uninterruptible I/O sleep for 30s (6 workers; likely SQLite/filesystem stall)"
    )


def test_resource_sampler_accounts_memory_swap_and_io_deltas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "smaps": {"Pss": 24, "Pss_Anon": 11, "Pss_File": 13, "SwapPss": 5},
        "io": {"read_bytes": 100, "write_bytes": 200, "cancelled_write_bytes": 0},
    }
    monkeypatch.setattr("devtools.verify_runs.process_tree", lambda _root_pid: [101])
    monkeypatch.setattr("devtools.verify_runs._status_values", lambda _pid: {"state": "S", "rss_kb": 30})
    monkeypatch.setattr("devtools.verify_runs._smaps_rollup_kb", lambda _pid: dict(state["smaps"]))
    monkeypatch.setattr("devtools.verify_runs._process_io_bytes", lambda _pid: dict(state["io"]))
    monkeypatch.setattr("devtools.verify_runs._process_identity", lambda _pid: "101:1")
    monkeypatch.setattr("devtools.verify_runs._cpu_seconds", lambda _pid: 1.0)
    monkeypatch.setattr("devtools.verify_runs._cgroup_path", lambda _pid: "/test.scope")
    monkeypatch.setattr(
        "devtools.verify_runs._cgroup_int",
        lambda _path, name: {"memory.current": 100, "memory.peak": 120, "memory.swap.current": 8}[name],
    )
    monkeypatch.setattr("devtools.verify_runs._cgroup_io_bytes", lambda _path: {"rbytes": 300, "wbytes": 400})
    sampler = ResourceSampler(
        root_pid=101,
        run_id="resource-deltas",
        root=tmp_path,
        env={"POLYLOGUE_PYTEST_BASETEMP_ROOT": str(tmp_path)},
        output_path=tmp_path / "resources.jsonl",
    )

    sampler.sample(event="started")
    state["smaps"] = {"Pss": 26, "Pss_Anon": 12, "Pss_File": 14, "SwapPss": 7}
    state["io"] = {"read_bytes": 160, "write_bytes": 260, "cancelled_write_bytes": 10}
    sampler.sample(event="finished")
    summary = sampler.summary()

    assert summary["peak_tree_anon_pss_kb"] == 12
    assert summary["peak_tree_file_pss_kb"] == 14
    assert summary["peak_tree_swap_pss_kb"] == 7
    assert summary["tree_swap_pss_delta_kb"] == 2
    assert summary["tree_read_bytes_delta"] == 60
    assert summary["tree_write_bytes_delta"] == 60
    assert summary["tree_cancelled_write_bytes_delta"] == 10
    assert summary["cgroup_path"] == "/test.scope"
    assert summary["peak_cgroup_memory_bytes"] == 120
    assert summary["final_cgroup_memory_swap_current_bytes"] == 8


def test_resource_sampler_throttles_basetemp_size_walk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "resources.jsonl"
    env = {
        "POLYLOGUE_PYTEST_BASETEMP_ROOT": str(tmp_path),
        "POLYLOGUE_VERIFY_BASETEMP_SIZE_INTERVAL_S": "60",
    }
    basetemp = pytest_basetemp_path(root=tmp_path, run_id="test-run", env=env)
    basetemp.mkdir(parents=True)
    (basetemp / "artifact.txt").write_text("payload")
    calls = 0

    def counted_usage(_path: Path) -> tuple[int, int]:
        nonlocal calls
        calls += 1
        return calls, calls + 1

    monkeypatch.setattr("devtools.verify_runs._dir_usage_kb", counted_usage)
    sampler = ResourceSampler(
        root_pid=os.getpid(),
        run_id="test-run",
        root=tmp_path,
        env=env,
        output_path=path,
    )

    first = sampler.sample(event="sample")
    second = sampler.sample(event="sample")

    assert first["basetemp_size_kb"] == 1
    assert first["basetemp_allocated_kb"] == 2
    assert second["basetemp_size_kb"] == 1
    assert calls == 1


def test_sparse_basetemp_enforces_allocated_tmpfs_bytes_and_retains_logical_evidence(tmp_path: Path) -> None:
    env = {
        "POLYLOGUE_PYTEST_BASETEMP_ROOT": str(tmp_path),
        "POLYLOGUE_PYTEST_TMPFS": "1",
        "POLYLOGUE_VERIFY_BASETEMP_SIZE_INTERVAL_S": "1",
    }
    run_id = "sparse-physical-accounting"
    basetemp = pytest_basetemp_path(root=tmp_path, run_id=run_id, env=env)
    basetemp.mkdir(parents=True)
    with (basetemp / "sparse.bin").open("wb") as handle:
        handle.seek(64 * 1024 * 1024)
        handle.write(b"x")
    sampler = ResourceSampler(
        root_pid=os.getpid(), run_id=run_id, root=tmp_path, env=env, output_path=tmp_path / "resources.jsonl"
    )

    sample = sampler.sample(event="sample")
    logical_kb = sample["basetemp_size_kb"]
    allocated_kb = sample["basetemp_allocated_kb"]

    assert isinstance(logical_kb, int)
    assert isinstance(allocated_kb, int)
    assert logical_kb > allocated_kb
    assert not pytest_tmpfs_budget_exceeded(sample, budget_kb=allocated_kb + 1)
    assert pytest_tmpfs_budget_exceeded(sample, budget_kb=allocated_kb - 1)


def test_resource_sampler_does_not_charge_symlink_targets_to_managed_basetemp(tmp_path: Path) -> None:
    env = {"POLYLOGUE_PYTEST_BASETEMP_ROOT": str(tmp_path)}
    run_id = "symlink-accounting"
    basetemp = pytest_basetemp_path(root=tmp_path, run_id=run_id, env=env)
    basetemp.mkdir(parents=True)
    outside_target = tmp_path / "outside-target.bin"
    outside_target.write_bytes(b"x" * (4 * 1024 * 1024))
    (basetemp / "external-link").symlink_to(outside_target)
    sampler = ResourceSampler(
        root_pid=os.getpid(), run_id=run_id, root=tmp_path, env=env, output_path=tmp_path / "resources.jsonl"
    )

    sample = sampler.sample(event="sample")

    assert sample["basetemp_size_kb"] < 512
    assert sample["basetemp_allocated_kb"] < 512


def test_pytest_basetemp_path_tracks_tmpfs_opt_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shm, _scratch = _patch_basetemp_roots(monkeypatch, tmp_path, realm_mounted=True)
    path = pytest_basetemp_path(root=tmp_path, run_id="run-1", env={"POLYLOGUE_PYTEST_TMPFS": "1"})

    assert path.parent == shm


def test_pytest_tmpfs_budget_is_shared_and_bounded() -> None:
    assert pytest_tmpfs_budget_kb({"POLYLOGUE_PYTEST_TMPFS": "1"}) == 512 * 1024
    assert (
        pytest_tmpfs_budget_kb(
            {
                "POLYLOGUE_PYTEST_TMPFS": "1",
                "POLYLOGUE_PYTEST_TMPFS_MAX_MB": "4096",
            }
        )
        == 2048 * 1024
    )
    assert (
        pytest_tmpfs_budget_kb(
            {
                "POLYLOGUE_PYTEST_TMPFS": "1",
                "POLYLOGUE_PYTEST_BASETEMP_ROOT": "/custom",
            }
        )
        is None
    )
    assert (
        pytest_tmpfs_budget_kb(
            {
                "POLYLOGUE_PYTEST_TMPFS": "1",
                "POLYLOGUE_PYTEST_BASETEMP_ROOT": "/dev/shm/polylogue-explicit",
            }
        )
        == 512 * 1024
    )


def test_adaptive_pytest_policy_uses_host_capacity_not_ten_percent_cap() -> None:
    """Reproduce ce4dd629's 15,190 MiB host headroom without under-capping tmpfs.

    The full parallel suite used four workers and reached 1,521.6 MiB in its
    basetemp.  Before this regression, the production policy returned 1,519
    MiB solely because it used ten percent of ``MemAvailable`` as the cap.
    """
    policy = adaptive_pytest_runtime_policy(
        available_kb=15_190 * 1024,
        memory_full_avg10=0.0,
        cpu_count=24,
        shm_free_kb=15_190 * 1024,
        worker_count=4,
    )

    assert policy.workers == 4
    assert policy.tmpfs_budget_mb == 2048
    assert policy.tmpfs_predicted_mb == 1522


def test_adaptive_pytest_policy_refuses_four_workers_at_four_gib_from_measured_envelope() -> None:
    with pytest.raises(PytestResourceError, match="cannot reserve measured pytest cgroup memory"):
        adaptive_pytest_runtime_policy(
            available_kb=4 * 1024 * 1024,
            memory_full_avg10=0.0,
            cpu_count=24,
            shm_free_kb=16 * 1024 * 1024,
            worker_count=4,
        )


def test_adaptive_pytest_policy_caps_near_threshold_from_full_cgroup_peak() -> None:
    policy = adaptive_pytest_runtime_policy(
        available_kb=6400 * 1024,
        memory_full_avg10=0.0,
        cpu_count=24,
        shm_free_kb=16 * 1024 * 1024,
        worker_count=4,
    )

    assert policy.tmpfs_budget_mb == 1338
    assert policy.tmpfs_predicted_mb is not None
    assert policy.tmpfs_budget_mb < policy.tmpfs_predicted_mb


def test_adaptive_pytest_policy_preserves_controller_memory_at_one_worker() -> None:
    with pytest.raises(PytestResourceError, match="cannot reserve measured pytest cgroup memory"):
        adaptive_pytest_runtime_policy(
            available_kb=2800 * 1024,
            memory_full_avg10=0.0,
            cpu_count=24,
            shm_free_kb=16 * 1024 * 1024,
            worker_count=1,
        )


def test_adaptive_pytest_policy_does_not_charge_a_serial_run_for_an_xdist_worker() -> None:
    policy = adaptive_pytest_runtime_policy(
        available_kb=2500 * 1024,
        memory_full_avg10=0.0,
        cpu_count=24,
        shm_free_kb=16 * 1024 * 1024,
        worker_count=0,
    )

    assert policy.workers == 0


def test_focused_serial_policy_does_not_inherit_full_suite_memory_residuals() -> None:
    policy = adaptive_pytest_runtime_policy(
        available_kb=1500 * 1024,
        memory_full_avg10=0.0,
        cpu_count=24,
        shm_free_kb=0,
        worker_count=0,
        full_suite=False,
    )

    assert policy.workers == 0
    assert policy.tmpfs_predicted_mb is None

    with pytest.raises(PytestResourceError, match="cannot reserve measured pytest cgroup memory"):
        adaptive_pytest_runtime_policy(
            available_kb=1500 * 1024,
            memory_full_avg10=0.0,
            cpu_count=24,
            shm_free_kb=0,
            worker_count=0,
            full_suite=True,
        )


def test_adaptive_pytest_policy_treats_full_run_basetemp_as_aggregate_demand() -> None:
    predictions = {
        adaptive_pytest_runtime_policy(
            available_kb=16 * 1024 * 1024,
            memory_full_avg10=0.0,
            cpu_count=24,
            shm_free_kb=16 * 1024 * 1024,
            worker_count=workers,
        ).tmpfs_predicted_mb
        for workers in (1, 4, 8)
    }

    assert predictions == {1522}


@pytest.mark.parametrize(
    ("worker_args", "expected"),
    [
        (["-n", "4"], 4),
        (["-n", "0"], 0),
        (["-n4"], 4),
        (["-n=4"], 4),
        (["--numprocesses", "8"], 8),
        (["--numprocesses=8"], 8),
        (["-n", "auto"], max(1, os.cpu_count() or 1)),
        (["-nauto"], max(1, os.cpu_count() or 1)),
        (["--numprocesses=auto"], max(1, os.cpu_count() or 1)),
    ],
)
def test_production_pytest_commands_reserve_every_xdist_spelling(
    monkeypatch: pytest.MonkeyPatch, worker_args: list[str], expected: int
) -> None:
    monkeypatch.delenv("PYTEST_XDIST_AUTO_NUM_WORKERS", raising=False)
    command = run_tests.build_pytest_cmd(["tests/unit/devtools", *worker_args])

    assert verify._pytest_command_concurrency(command) == expected


def test_pytest_auto_workers_reserve_environment_override() -> None:
    command = run_tests.build_pytest_cmd(["tests/unit/devtools", "-n", "auto"])

    assert verify._pytest_command_concurrency(command, env={"PYTEST_XDIST_AUTO_NUM_WORKERS": "32"}) == 32


def test_adaptive_pytest_policy_reduces_workers_under_pressure() -> None:
    policy = adaptive_pytest_runtime_policy(
        available_kb=16 * 1024 * 1024,
        memory_full_avg10=2.0,
        cpu_count=24,
        shm_free_kb=16 * 1024 * 1024,
    )

    assert policy.workers == 3


def test_managed_pytest_policy_refuses_low_memory() -> None:
    with pytest.raises(RuntimeError, match="requires at least 1.00 GiB"):
        adaptive_pytest_runtime_policy(
            available_kb=1024 * 1024 - 1,
            memory_full_avg10=0.0,
            cpu_count=24,
            shm_free_kb=16 * 1024 * 1024,
        )


def test_adaptive_pytest_policy_caps_host_capacity_to_cgroup_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verify_runs, "_meminfo", lambda: {"MemAvailable": 16 * 1024 * 1024})
    monkeypatch.setattr(verify_runs, "read_cgroup_memory_headroom_bytes", lambda: 4 * 1024 * 1024 * 1024)

    with pytest.raises(PytestResourceError, match="cannot reserve measured pytest cgroup memory"):
        adaptive_pytest_runtime_policy(
            memory_full_avg10=0.0,
            cpu_count=24,
            shm_free_kb=16 * 1024 * 1024,
            worker_count=4,
        )


def test_managed_pytest_policy_preserves_explicit_custom_root_and_memory_admission(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(verify_runs, "_meminfo", lambda: {"MemAvailable": 8 * 1024 * 1024})
    monkeypatch.setattr(verify_runs, "read_cgroup_memory_headroom_bytes", lambda: None)
    monkeypatch.setattr(verify_runs, "_pressure", lambda _kind: {"full_avg10": 0.0})
    monkeypatch.setattr(verify_runs, "_fs_usage", lambda _path: {"used_kb": 0, "free_kb": 16 * 1024 * 1024})
    env, policy = apply_managed_pytest_runtime_policy({"POLYLOGUE_PYTEST_BASETEMP_ROOT": str(tmp_path)}, worker_count=4)

    assert env["POLYLOGUE_PYTEST_BASETEMP_ROOT"] == str(tmp_path)
    assert policy is not None
    assert policy.workers == 4
    assert policy.basetemp_label == "configured"


def test_managed_pytest_policy_bounds_explicit_tmpfs_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verify_runs, "_meminfo", lambda: {"MemAvailable": 8 * 1024 * 1024})
    monkeypatch.setattr(verify_runs, "read_cgroup_memory_headroom_bytes", lambda: None)
    monkeypatch.setattr(verify_runs, "_pressure", lambda _kind: {"full_avg10": 0.0})
    monkeypatch.setattr(verify_runs, "_fs_usage", lambda _path: {"used_kb": 0, "free_kb": 16 * 1024 * 1024})

    env, policy = apply_managed_pytest_runtime_policy(
        {
            "POLYLOGUE_PYTEST_BASETEMP_ROOT": "/dev/shm/polylogue-explicit",
            "POLYLOGUE_PYTEST_TMPFS": "0",
        },
        worker_count=4,
        full_suite=False,
    )

    assert policy is not None
    assert env["POLYLOGUE_PYTEST_BASETEMP_ROOT"] == "/dev/shm/polylogue-explicit"
    assert env["POLYLOGUE_PYTEST_TMPFS"] == "1"
    assert pytest_tmpfs_budget_kb(env) == policy.tmpfs_budget_mb * 1024
    assert policy.basetemp_label == "configured"


def test_managed_pytest_policy_preserves_headroom_for_explicit_tmpfs_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(verify_runs, "_meminfo", lambda: {"MemAvailable": 8 * 1024 * 1024})
    monkeypatch.setattr(verify_runs, "read_cgroup_memory_headroom_bytes", lambda: None)
    monkeypatch.setattr(verify_runs, "_pressure", lambda _kind: {"full_avg10": 0.0})
    monkeypatch.setattr(verify_runs, "_fs_usage", lambda _path: {"used_kb": 0, "free_kb": 2500 * 1024})

    with pytest.raises(PytestResourceError, match="need >= 3024 MiB"):
        apply_managed_pytest_runtime_policy(
            {
                "POLYLOGUE_PYTEST_BASETEMP_ROOT": "/dev/shm/polylogue-explicit",
                "POLYLOGUE_PYTEST_BASETEMP_REQUIRED_MB": "1522",
                "POLYLOGUE_PYTEST_TMPFS_MAX_MB": "2048",
            },
            worker_count=4,
            full_suite=True,
        )


def test_full_suite_explicit_root_requires_measured_basetemp_space(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The managed test harness itself uses /dev/shm, so keep this custom-root
    # regression on a distinct disk route rather than accidentally admitting
    # it as a configured tmpfs path.
    monkeypatch.setattr(verify_runs, "PYTEST_TMPFS_ROOT", tmp_path / "other-shm")
    monkeypatch.setattr(verify_runs, "_meminfo", lambda: {"MemAvailable": 8 * 1024 * 1024})
    monkeypatch.setattr(verify_runs, "read_cgroup_memory_headroom_bytes", lambda: None)
    monkeypatch.setattr(verify_runs, "_pressure", lambda _kind: {"full_avg10": 0.0})
    monkeypatch.setattr(verify_runs, "_fs_usage", lambda _path: {"used_kb": 0, "free_kb": 1200 * 1024})

    with pytest.raises(PytestResourceError, match="no pytest basetemp location has enough free space"):
        apply_managed_pytest_runtime_policy(
            {"POLYLOGUE_PYTEST_BASETEMP_ROOT": str(tmp_path)},
            worker_count=4,
            full_suite=True,
        )


def test_managed_pytest_policy_rejects_explicit_root_when_workers_exceed_memory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(verify_runs, "_meminfo", lambda: {"MemAvailable": 4 * 1024 * 1024})
    monkeypatch.setattr(verify_runs, "read_cgroup_memory_headroom_bytes", lambda: None)
    monkeypatch.setattr(verify_runs, "_pressure", lambda _kind: {"full_avg10": 0.0})

    with pytest.raises(PytestResourceError, match="cannot reserve measured pytest cgroup memory"):
        apply_managed_pytest_runtime_policy({"POLYLOGUE_PYTEST_BASETEMP_ROOT": str(tmp_path)}, worker_count=12)


# ── basetemp placement: one resolution order, disk-headroom preflight ──────
#
# Root cause of the 2026-07-30 incident: `.claude/settings.json` sets
# POLYLOGUE_PYTEST_BASETEMP_ROOT=/tmp/polylogue-pytest for cloud sandboxes,
# but that env leaks into workstation agent shells too, where /tmp is a
# small 6 GiB tmpfs shared by ~8 concurrent agent lanes. Nothing checked free
# space before committing to a basetemp location, so it silently filled and
# an unrelated command (the docs renderer) crashed with a bare ENOSPC.
# `resolve_pytest_basetemp_root` is the single placement policy used by both
# `tests/conftest.py` (direct pytest) and the devtools supervisor
# (`apply_managed_pytest_runtime_policy`, tested above) so there is exactly
# one order instead of two that can silently disagree.


def _patch_basetemp_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, realm_mounted: bool) -> tuple[Path, Path]:
    """Point every placement-policy constant at real, tmp_path-backed dirs.

    Repoints ``PYTEST_TMPFS_ROOT``/``DEFAULT_PYTEST_BASETEMP_ROOT``/
    ``_CLOUD_PYTEST_BASETEMP_ROOT`` — the full candidate set — so no test in
    this module can accidentally probe or sweep a real ``/dev/shm``,
    ``/realm/tmp``, or ``/tmp`` path on the host running the suite (that host
    may have other agents' live basetemps on it).
    """
    shm = tmp_path / "dev-shm"
    shm.mkdir()
    scratch_parent = tmp_path / "realm-tmp"
    scratch = scratch_parent / "polylogue-pytest"
    cloud_fallback = tmp_path / "tmp" / "polylogue-pytest"
    cloud_fallback.parent.mkdir(parents=True)
    if realm_mounted:
        scratch_parent.mkdir()
    monkeypatch.setattr(verify_runs, "PYTEST_TMPFS_ROOT", shm)
    monkeypatch.setattr(verify_runs, "DEFAULT_PYTEST_BASETEMP_ROOT", scratch)
    monkeypatch.setattr(verify_runs, "_CLOUD_PYTEST_BASETEMP_ROOT", cloud_fallback)
    monkeypatch.setattr(verify_runs, "_is_tmpfs_dir", lambda path: path == shm)
    return shm, scratch


def _patch_resource_capacity(
    monkeypatch: pytest.MonkeyPatch,
    *,
    shm: Path,
    scratch: Path,
    available_mb: int,
) -> None:
    monkeypatch.setattr(verify_runs, "_meminfo", lambda: {"MemAvailable": available_mb * 1024})
    monkeypatch.setattr(verify_runs, "read_cgroup_memory_headroom_bytes", lambda: None)
    monkeypatch.setattr(verify_runs, "_pressure", lambda _kind: {"full_avg10": 0.0})
    monkeypatch.setattr(os, "cpu_count", lambda: 24)

    def fake_fs_usage(path: Path) -> dict[str, int] | None:
        if path in {shm, scratch.parent}:
            return {"used_kb": 0, "free_kb": 16 * 1024 * 1024}
        return None

    monkeypatch.setattr(verify_runs, "_fs_usage", fake_fs_usage)


def test_default_full_suite_workers_use_scratch_through_placement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shm, scratch = _patch_basetemp_roots(monkeypatch, tmp_path, realm_mounted=True)
    _patch_resource_capacity(monkeypatch, shm=shm, scratch=scratch, available_mb=8192)

    workers = adaptive_pytest_worker_count({})
    env, policy = apply_managed_pytest_runtime_policy({}, worker_count=workers)

    assert workers == 5
    assert policy is not None
    assert policy.workers == workers
    assert policy.basetemp_label == "scratch"
    assert env["POLYLOGUE_PYTEST_TMPFS"] == "0"
    assert env["POLYLOGUE_PYTEST_BASETEMP_ROOT"] == str(scratch)


def test_focused_selection_keeps_bounded_tmpfs_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shm, scratch = _patch_basetemp_roots(monkeypatch, tmp_path, realm_mounted=True)
    _patch_resource_capacity(monkeypatch, shm=shm, scratch=scratch, available_mb=8192)

    env, policy = apply_managed_pytest_runtime_policy({}, worker_count=1, full_suite=False)

    assert policy is not None
    assert policy.basetemp_label == "tmpfs opt-in"
    assert env["POLYLOGUE_PYTEST_TMPFS"] == "1"


def test_explicit_full_suite_tmpfs_choice_remains_bounded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shm, scratch = _patch_basetemp_roots(monkeypatch, tmp_path, realm_mounted=True)
    _patch_resource_capacity(monkeypatch, shm=shm, scratch=scratch, available_mb=15_190)

    env, policy = apply_managed_pytest_runtime_policy({"POLYLOGUE_PYTEST_TMPFS": "1"}, worker_count=4, full_suite=True)

    assert policy is not None
    assert policy.basetemp_label == "tmpfs opt-in"
    assert env["POLYLOGUE_PYTEST_TMPFS"] == "1"
    assert env["POLYLOGUE_PYTEST_TMPFS_MAX_MB"] == "2048"


def test_inherited_512_mib_tmpfs_cap_reroutes_measured_demand_to_scratch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shm, scratch = _patch_basetemp_roots(monkeypatch, tmp_path, realm_mounted=True)
    _patch_resource_capacity(monkeypatch, shm=shm, scratch=scratch, available_mb=15_190)

    env, policy = apply_managed_pytest_runtime_policy(
        {"POLYLOGUE_PYTEST_TMPFS_MAX_MB": "512"},
        worker_count=4,
    )

    assert policy is not None
    assert policy.tmpfs_budget_mb == 2048
    assert policy.tmpfs_predicted_mb == 1522
    assert policy.basetemp_label == "scratch"
    assert env["POLYLOGUE_PYTEST_TMPFS_MAX_MB"] == "512"
    assert env["POLYLOGUE_PYTEST_TMPFS"] == "0"
    assert env["POLYLOGUE_PYTEST_BASETEMP_ROOT"] == str(scratch)


def test_configured_tmpfs_root_reroutes_to_scratch_when_its_cap_is_too_small(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shm, scratch = _patch_basetemp_roots(monkeypatch, tmp_path, realm_mounted=True)
    _patch_resource_capacity(monkeypatch, shm=shm, scratch=scratch, available_mb=15_190)
    explicit_root = shm / "explicit"
    explicit_root.mkdir()

    env, policy = apply_managed_pytest_runtime_policy(
        {
            "POLYLOGUE_PYTEST_BASETEMP_ROOT": str(explicit_root),
            "POLYLOGUE_PYTEST_TMPFS_MAX_MB": "512",
        },
        worker_count=4,
    )

    assert policy is not None
    assert policy.basetemp_label == "scratch"
    assert env["POLYLOGUE_PYTEST_TMPFS"] == "0"
    assert env["POLYLOGUE_PYTEST_BASETEMP_ROOT"] == str(scratch)


def test_configured_tmpfs_reroute_keeps_admission_evidence_when_scratch_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shm, scratch = _patch_basetemp_roots(monkeypatch, tmp_path, realm_mounted=True)
    _patch_resource_capacity(monkeypatch, shm=shm, scratch=scratch, available_mb=15_190)
    configured = shm / "configured"
    configured.mkdir()

    def constrained_headroom(path: Path) -> int | None:
        if path == scratch:
            return 1 * 1024
        return 8 * 1024 * 1024

    monkeypatch.setattr(verify_runs, "_headroom_kb", constrained_headroom)

    with pytest.raises(PytestResourceError) as excinfo:
        apply_managed_pytest_runtime_policy(
            {
                "POLYLOGUE_PYTEST_BASETEMP_ROOT": str(configured),
                "POLYLOGUE_PYTEST_TMPFS_MAX_MB": "512",
            },
            worker_count=4,
        )

    message = str(excinfo.value)
    assert f"configured pytest basetemp declared demand exceeds its safe adaptive tmpfs budget ({configured}" in message
    assert "safe tmpfs budget=512 MiB" in message
    assert f"{scratch} (scratch): 1 MiB free" in message


def test_focused_policy_keeps_full_suite_basetemp_demand_out_of_scratch_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shm, scratch = _patch_basetemp_roots(monkeypatch, tmp_path, realm_mounted=True)
    _patch_resource_capacity(monkeypatch, shm=shm, scratch=scratch, available_mb=8192)

    def fake_fs_usage(path: Path) -> dict[str, int] | None:
        if path == shm:
            return None
        if path == scratch.parent:
            return {"used_kb": 0, "free_kb": 1200 * 1024}
        return None

    monkeypatch.setattr(verify_runs, "_fs_usage", fake_fs_usage)
    env, policy = apply_managed_pytest_runtime_policy({}, worker_count=0, full_suite=False)

    assert policy is not None
    assert policy.basetemp_required_mb is None
    assert policy.basetemp_label == "scratch"
    assert env["POLYLOGUE_PYTEST_TMPFS"] == "0"
    assert env["POLYLOGUE_PYTEST_BASETEMP_ROOT"] == str(scratch)


def test_inherited_tmpfs_cap_is_clamped_to_the_measured_host_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shm, scratch = _patch_basetemp_roots(monkeypatch, tmp_path, realm_mounted=True)
    _patch_resource_capacity(monkeypatch, shm=shm, scratch=scratch, available_mb=6400)

    env, policy = apply_managed_pytest_runtime_policy(
        {"POLYLOGUE_PYTEST_TMPFS_MAX_MB": "2048"},
        worker_count=4,
    )

    assert policy is not None
    assert policy.tmpfs_budget_mb == 1338
    assert env["POLYLOGUE_PYTEST_TMPFS_MAX_MB"] == "1338"
    assert env["POLYLOGUE_PYTEST_TMPFS"] == "0"
    assert env["POLYLOGUE_PYTEST_BASETEMP_ROOT"] == str(scratch)


def test_managed_policy_uses_scratch_when_tmpfs_is_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shm, scratch = _patch_basetemp_roots(monkeypatch, tmp_path, realm_mounted=True)
    monkeypatch.setattr(verify_runs, "_meminfo", lambda: {"MemAvailable": 15_190 * 1024})
    monkeypatch.setattr(verify_runs, "read_cgroup_memory_headroom_bytes", lambda: None)
    monkeypatch.setattr(verify_runs, "_pressure", lambda _kind: {"full_avg10": 0.0})
    monkeypatch.setattr(os, "cpu_count", lambda: 24)

    def fake_fs_usage(path: Path) -> dict[str, int] | None:
        if path == shm:
            return None
        if path == scratch.parent:
            return {"used_kb": 0, "free_kb": 16 * 1024 * 1024}
        return None

    monkeypatch.setattr(verify_runs, "_fs_usage", fake_fs_usage)

    env, policy = apply_managed_pytest_runtime_policy({}, worker_count=4)

    assert policy is not None
    assert policy.tmpfs_budget_mb == 0
    assert policy.basetemp_label == "scratch"
    assert env["POLYLOGUE_PYTEST_TMPFS"] == "0"
    assert env["POLYLOGUE_PYTEST_BASETEMP_ROOT"] == str(scratch)


def test_declared_basetemp_demand_above_tmpfs_cap_uses_scratch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shm, scratch = _patch_basetemp_roots(monkeypatch, tmp_path, realm_mounted=True)
    _patch_resource_capacity(monkeypatch, shm=shm, scratch=scratch, available_mb=15_190)

    env, policy = apply_managed_pytest_runtime_policy(
        {"POLYLOGUE_PYTEST_BASETEMP_REQUIRED_MB": "4096"},
        worker_count=4,
    )

    assert policy is not None
    assert policy.tmpfs_budget_mb == 2048
    assert policy.basetemp_required_mb == 4096
    assert policy.basetemp_label == "scratch"
    assert env["POLYLOGUE_PYTEST_TMPFS"] == "0"
    assert env["POLYLOGUE_PYTEST_BASETEMP_ROOT"] == str(scratch)


def test_resolve_basetemp_prefers_tmpfs_when_it_has_headroom(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shm, _scratch = _patch_basetemp_roots(monkeypatch, tmp_path, realm_mounted=True)
    monkeypatch.setattr(
        verify_runs, "_fs_usage", lambda path: {"used_kb": 0, "free_kb": 32 * 1024 * 1024} if path == shm else None
    )

    root, label = resolve_pytest_basetemp_root({})

    assert root == shm
    assert label == "tmpfs opt-in"


def test_resolve_basetemp_falls_back_to_nvme_scratch_when_tmpfs_is_low(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shm, scratch = _patch_basetemp_roots(monkeypatch, tmp_path, realm_mounted=True)

    def fake_fs_usage(path: Path) -> dict[str, int] | None:
        if path == shm:
            return {"used_kb": 0, "free_kb": 10 * 1024}  # 10 MiB — under the 1 GiB requirement
        if path == scratch.parent:
            return {"used_kb": 0, "free_kb": 200 * 1024 * 1024}
        return None

    monkeypatch.setattr(verify_runs, "_fs_usage", fake_fs_usage)

    root, label = resolve_pytest_basetemp_root({})

    assert root == scratch
    assert label == "scratch"


def test_resolve_basetemp_reroutes_known_demand_before_tmpfs_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shm, scratch = _patch_basetemp_roots(monkeypatch, tmp_path, realm_mounted=True)

    def fake_fs_usage(path: Path) -> dict[str, int] | None:
        if path == shm:
            return {"used_kb": 0, "free_kb": 2500 * 1024}
        if path == scratch.parent:
            return {"used_kb": 0, "free_kb": 4096 * 1024}
        return None

    monkeypatch.setattr(verify_runs, "_fs_usage", fake_fs_usage)

    root, label = resolve_pytest_basetemp_root({"POLYLOGUE_PYTEST_BASETEMP_REQUIRED_MB": "2048"})

    assert root == scratch
    assert label == "scratch"


def test_resolve_basetemp_reserves_the_allowed_cap_not_only_the_prediction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shm, scratch = _patch_basetemp_roots(monkeypatch, tmp_path, realm_mounted=True)

    def fake_fs_usage(path: Path) -> dict[str, int] | None:
        if path == shm:
            return {"used_kb": 0, "free_kb": 2500 * 1024}
        if path == scratch.parent:
            return {"used_kb": 0, "free_kb": 4096 * 1024}
        return None

    monkeypatch.setattr(verify_runs, "_fs_usage", fake_fs_usage)

    root, label = resolve_pytest_basetemp_root(
        {
            "POLYLOGUE_PYTEST_TMPFS": "1",
            "POLYLOGUE_PYTEST_BASETEMP_REQUIRED_MB": "1522",
            "POLYLOGUE_PYTEST_TMPFS_MAX_MB": "2048",
        }
    )

    assert root == scratch
    assert label == "scratch"


def test_resolve_basetemp_refuses_loudly_when_every_candidate_is_full(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The exact scenario from the incident: every candidate is starved.

    This demonstrates the low-space path deliberately, by mocking the
    free-space probe rather than actually filling a filesystem: every
    candidate reports far less free space than the headroom requirement, and
    the run must refuse up front with a message naming each path, its free
    space, and the requirement — not crash three layers away in an unrelated
    command.
    """
    shm, scratch = _patch_basetemp_roots(monkeypatch, tmp_path, realm_mounted=True)
    monkeypatch.setattr(verify_runs, "_fs_usage", lambda path: {"used_kb": 0, "free_kb": 5 * 1024})  # 5 MiB anywhere

    with pytest.raises(PytestResourceError) as excinfo:
        resolve_pytest_basetemp_root({})

    message = str(excinfo.value)
    assert "no pytest basetemp location has enough free space" in message
    assert str(shm) in message
    assert str(scratch) in message
    assert "5 MiB free" in message
    assert "need >= 1024 MiB" in message
    assert "rm -rf" in message


def test_resolve_basetemp_headroom_requirement_is_overridable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shm, _scratch = _patch_basetemp_roots(monkeypatch, tmp_path, realm_mounted=True)
    monkeypatch.setattr(verify_runs, "_fs_usage", lambda path: {"used_kb": 0, "free_kb": 200 * 1024})  # 200 MiB

    root, label = resolve_pytest_basetemp_root({"POLYLOGUE_PYTEST_BASETEMP_MIN_FREE_MB": "64"})

    assert root == shm
    assert label == "tmpfs opt-in"


def test_resolve_basetemp_strips_leaked_cloud_default_on_workstation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The literal incident trigger: the cloud-sandbox env leaking onto a workstation."""
    shm, _scratch = _patch_basetemp_roots(monkeypatch, tmp_path, realm_mounted=True)
    monkeypatch.setattr(
        verify_runs, "_fs_usage", lambda path: {"used_kb": 0, "free_kb": 32 * 1024 * 1024} if path == shm else None
    )

    # The literal env value `.claude/settings.json` sets for cloud sandboxes;
    # normalize_pytest_basetemp_env only strips it when it matches the
    # (here, patched) known cloud-sentinel constant.
    root, label = resolve_pytest_basetemp_root(
        {"POLYLOGUE_PYTEST_BASETEMP_ROOT": str(verify_runs._CLOUD_PYTEST_BASETEMP_ROOT)}
    )

    assert root == shm
    assert label == "tmpfs opt-in"


def test_resolve_basetemp_honors_a_genuine_explicit_override_but_still_checks_space(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_basetemp_roots(monkeypatch, tmp_path, realm_mounted=True)
    monkeypatch.setattr(verify_runs, "_fs_usage", lambda path: {"used_kb": 0, "free_kb": 5 * 1024})
    custom = tmp_path / "operator-chosen-root"

    with pytest.raises(PytestResourceError, match="configured"):
        resolve_pytest_basetemp_root({"POLYLOGUE_PYTEST_BASETEMP_ROOT": str(custom)})


def test_resolve_basetemp_uses_disk_fallback_only_when_realm_is_unmounted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """/tmp is a legitimate placement only when /realm/tmp genuinely isn't mounted (cloud sandbox)."""
    _shm, _scratch = _patch_basetemp_roots(monkeypatch, tmp_path, realm_mounted=False)
    monkeypatch.setattr(verify_runs, "_is_tmpfs_dir", lambda path: False)
    monkeypatch.setattr(
        verify_runs,
        "_fs_usage",
        lambda path: (
            {"used_kb": 0, "free_kb": 32 * 1024 * 1024}
            if path == verify_runs._CLOUD_PYTEST_BASETEMP_ROOT.parent
            else None
        ),
    )

    root, label = resolve_pytest_basetemp_root({})

    assert root == verify_runs._CLOUD_PYTEST_BASETEMP_ROOT
    assert label == "disk fallback"


def test_pytest_basetemp_known_roots_only_lists_existing_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configured = tmp_path / "configured"
    configured.mkdir()
    monkeypatch.setattr(verify_runs, "DEFAULT_PYTEST_BASETEMP_ROOT", tmp_path / "does-not-exist")
    monkeypatch.setattr(verify_runs, "_CLOUD_PYTEST_BASETEMP_ROOT", tmp_path / "also-missing")

    roots = pytest_basetemp_known_roots({"POLYLOGUE_PYTEST_BASETEMP_ROOT": str(configured)})

    assert configured in roots
    assert tmp_path / "does-not-exist" not in roots


def test_cleanup_managed_pytest_basetemp_removes_run_root(tmp_path: Path) -> None:
    env = {"POLYLOGUE_PYTEST_BASETEMP_ROOT": str(tmp_path)}
    basetemp = pytest_basetemp_path(root=tmp_path, run_id="run-1", env=env)
    (basetemp / "worker-output").mkdir(parents=True)
    verify_runs.pytest_basetemp_claim_path(basetemp, kind="managed").write_text("999999:1", encoding="utf-8")

    cleaned = cleanup_managed_pytest_basetemp(root=tmp_path, run_id="run-1", env=env)

    assert cleaned == basetemp
    assert not basetemp.exists()


def test_pytest_basetemp_claim_path_canonicalizes_symlink_aliases(tmp_path: Path) -> None:
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)

    real_basetemp = real_root / "pytest-polylogue-run"
    linked_basetemp = linked_root / "pytest-polylogue-run"

    assert verify_runs.pytest_basetemp_claim_path(real_basetemp, kind="lock") == verify_runs.pytest_basetemp_claim_path(
        linked_basetemp, kind="lock"
    )


def test_cleanup_managed_pytest_basetemp_leaves_successor_claim_while_locked(tmp_path: Path) -> None:
    env = {"POLYLOGUE_PYTEST_BASETEMP_ROOT": str(tmp_path)}
    basetemp = pytest_basetemp_path(root=tmp_path, run_id="reused-run", env=env)
    basetemp.mkdir(parents=True)
    (basetemp / "successor-fixture").write_text("live", encoding="utf-8")
    claim_path = verify_runs.pytest_basetemp_claim_path(basetemp, kind="managed")
    claim_path.write_text("999999:1", encoding="utf-8")
    lock_path = verify_runs.pytest_basetemp_claim_path(basetemp, kind="lock")

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        assert cleanup_managed_pytest_basetemp(root=tmp_path, run_id="reused-run", env=env) is None

    assert basetemp.exists()
    assert claim_path.exists()


def test_pytest_workload_receipt_uses_allocated_basetemp_peak() -> None:
    receipt = verify._pytest_workload_receipt(
        label="pytest sparse",
        cmd=["pytest", "tests/unit/example.py"],
        elapsed_s=1.0,
        returncode=0,
        termination_reason=None,
        resource_summary={"peak_basetemp_size_kb": 64 * 1024, "peak_basetemp_allocated_kb": 8},
        last_resource_sample={"tree_rss_kb": 0, "tree_pss_kb": 0, "tree_cpu_s": 0.0},
        tmpfs_budget_mb=1,
        basetemp_cleanup=None,
        concurrency=1,
    )

    execute = next(phase for phase in receipt["phases"] if phase["name"] == "execute")
    assert execute["temp_storage_bytes"] == 8 * 1024
    assert "Logical basetemp peak retained as diagnostic evidence: 67108864 bytes." in receipt["notes"]


def test_cleanup_managed_pytest_basetemp_recognizes_child_cleanup(tmp_path: Path) -> None:
    env = {"POLYLOGUE_PYTEST_BASETEMP_ROOT": str(tmp_path)}
    basetemp = pytest_basetemp_path(root=tmp_path, run_id="run-cleaned-by-child", env=env)

    cleaned = cleanup_managed_pytest_basetemp(root=tmp_path, run_id="run-cleaned-by-child", env=env)

    assert cleaned == basetemp


def test_cleanup_managed_pytest_basetemp_does_not_receipt_residual_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = {"POLYLOGUE_PYTEST_BASETEMP_ROOT": str(tmp_path / "scratch")}
    basetemp = pytest_basetemp_path(root=tmp_path, run_id="run-residual", env=env)
    basetemp.mkdir(parents=True)
    (basetemp / "still-open").write_text("residual")
    monkeypatch.setattr("devtools.verify_runs.shutil.rmtree", lambda _path: None)

    cleaned = cleanup_managed_pytest_basetemp(root=tmp_path, run_id="run-residual", env=env)

    assert cleaned is None
    assert basetemp.exists()


def test_cleanup_managed_pytest_basetemp_keeps_seed_cache(tmp_path: Path) -> None:
    env = {"POLYLOGUE_PYTEST_BASETEMP_ROOT": str(tmp_path)}
    seeded = pytest_basetemp_path(root=tmp_path, run_id="seeded-cache", env=env)
    seeded.mkdir()

    cleaned = cleanup_managed_pytest_basetemp(root=tmp_path, run_id="seeded-cache", env=env)

    assert cleaned is None
    assert seeded.exists()


def test_testmon_preflight_allows_seed_and_full_without_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    assert _testmon_preflight(seed_testmon=True, full_pytest=False, quick=False, commit=False) is None
    assert _testmon_preflight(seed_testmon=False, full_pytest=True, quick=False, commit=False) is None


def test_parse_pytest_test_count_from_summary() -> None:
    output = "bringing up nodes...\n\n6 passed, 2 skipped, 1 xfailed in 8.49s\n"

    assert _parse_pytest_test_count(output) == 9


def test_parse_pytest_test_count_handles_no_tests() -> None:
    assert _parse_pytest_test_count("no tests ran in 0.02s\n") == 0


def test_run_records_pytest_count_metadata_from_terminal_fallback() -> None:
    """When the JSON report is missing, _run falls back to terminal scraping."""
    completed = subprocess.CompletedProcess(
        args=["pytest"],
        returncode=0,
        stdout="....\n4 passed in 1.23s\n",
        stderr="",
    )

    with (
        patch("devtools.verify._run_pytest_with_heartbeat", return_value=completed),
        patch("devtools.verify._read_pytest_report", return_value=None),
    ):
        rc, _elapsed, metadata = _run("pytest affected", ["pytest"])

    assert rc == 0
    assert metadata["count"] == 4
    assert metadata["report_path"] is None
    assert metadata["events_path"] == str(PYTEST_EVENTS_PATH)
    assert metadata["output_path"] == str(PYTEST_OUTPUT_PATH)
    assert metadata["pytest_workers"] == "unset"
    assert metadata["pytest_selection"] == "full"
    receipt = metadata["workload_receipt"]
    assert receipt["spec"]["semantic_result"] == "complete"
    assert [phase["name"] for phase in receipt["phases"]] == ["execute", "quiescent"]


def test_run_records_managed_basetemp_cleanup_metadata(tmp_path: Path) -> None:
    completed = subprocess.CompletedProcess(args=["pytest"], returncode=0, stdout="1 passed in 0.1s\n", stderr="")
    cleaned = tmp_path / "pytest-polylogue-run-1"

    with (
        patch("devtools.verify._run_pytest_with_heartbeat", return_value=completed),
        patch("devtools.verify._read_pytest_report", return_value=None),
        patch("devtools.verify.cleanup_managed_pytest_basetemp", return_value=cleaned) as cleanup,
    ):
        rc, _elapsed, metadata = _run("pytest testmon", ["pytest", "--testmon", "-n", "4"])

    assert rc == 0
    cleanup.assert_called_once()
    assert metadata["basetemp_cleanup"] == str(cleaned)


@pytest.mark.uses_real_clock("the heartbeat loop computes elapsed containment time")
def test_heartbeat_persists_drained_output_before_interrupting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = VerifyRun(tier="focused-test", argv=["tests/unit/example.py"], git_head="head", root=tmp_path)
    artifacts = run.start_step(label="pytest buffered", cmd=["pytest"])
    stdout_read, stdout_write = os.pipe()
    stderr_read, stderr_write = os.pipe()
    os.write(stdout_write, b"buffered stdout before interrupt\n")
    os.close(stdout_write)
    os.close(stderr_write)
    stdout_pipe = os.fdopen(stdout_read, "rb", closefd=True)
    stderr_pipe = os.fdopen(stderr_read, "rb", closefd=True)

    class _Process:
        pid = os.getpid()
        stdout = stdout_pipe
        stderr = stderr_pipe
        returncode = None

        def poll(self) -> None:
            return None

    class _InterruptingSelector:
        def __init__(self) -> None:
            self.calls = 0

        def register(self, _fileobj: object, _events: int, _data: str) -> None:
            return None

        def get_map(self) -> dict[int, object]:
            return {1: object()}

        def select(self, timeout: float | None = None) -> list[tuple[SimpleNamespace, int]]:
            del timeout
            if self.calls == 0:
                self.calls += 1
                return [(SimpleNamespace(fd=stdout_pipe.fileno(), data="stdout", fileobj=stdout_pipe), 1)]
            raise KeyboardInterrupt

        def close(self) -> None:
            return None

    launch = SimpleNamespace(
        argv=["pytest"],
        receipt_path=tmp_path / "containment.json",
        request_path=tmp_path / "request.json",
        mode="process-group",
        unit=None,
        cgroup_path=None,
        fallback_argv=None,
        runtime_cap_s=0.0,
    )
    try:
        with (
            patch("devtools.verify.enable_child_subreaper", return_value=True),
            patch("devtools.verify.descendant_process_identities", return_value=()),
            patch("devtools.verify.build_supervisor_launch", return_value=launch),
            patch("devtools.verify.subprocess.Popen", return_value=_Process()),
            patch("devtools.verify._wait_for_supervisor_start", return_value={"status": "started"}),
            patch("devtools.verify.selectors.DefaultSelector", _InterruptingSelector),
            patch("devtools.verify._await_interrupted_pytest_containment"),
            patch("devtools.verify._write_pytest_progress"),
            patch("devtools.verify.ResourceSampler", return_value=MagicMock()),
        ):
            with pytest.raises(KeyboardInterrupt):
                verify._run_pytest_with_heartbeat(
                    ["pytest"], cwd=str(tmp_path), env={}, t0=time.monotonic(), run=run, artifacts=artifacts
                )
    finally:
        stdout_pipe.close()
        stderr_pipe.close()

    assert artifacts.stdout_path.read_text(encoding="utf-8") == "buffered stdout before interrupt\n"
    assert artifacts.stderr_path.read_text(encoding="utf-8") == ""
    assert artifacts.output_path.read_text(encoding="utf-8") == "buffered stdout before interrupt\n"
    assert (tmp_path / PYTEST_OUTPUT_PATH).read_text(encoding="utf-8") == "buffered stdout before interrupt\n"


def test_explicit_basetemp_root_retains_managed_resource_monitoring(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(verify_runs, "PYTEST_TMPFS_ROOT", tmp_path / "unselected-tmpfs")
    nvme_root = tmp_path / "realm-tmp" / "polylogue-pytest"
    nvme_root.mkdir(parents=True)
    monkeypatch.setattr(verify_runs, "_meminfo", lambda: {"MemAvailable": 8 * 1024 * 1024})
    monkeypatch.setattr(verify_runs, "read_cgroup_memory_headroom_bytes", lambda: None)
    monkeypatch.setattr(verify_runs, "_pressure", lambda _kind: {"full_avg10": 0.0})
    monkeypatch.setattr(verify_runs, "_fs_usage", lambda _path: {"used_kb": 0, "free_kb": 16 * 1024 * 1024})
    monkeypatch.setenv("POLYLOGUE_PYTEST_BASETEMP_ROOT", str(nvme_root))
    monkeypatch.delenv("POLYLOGUE_PYTEST_BASETEMP_REQUIRED_MB", raising=False)
    monkeypatch.delenv("POLYLOGUE_PYTEST_BASETEMP_MIN_FREE_MB", raising=False)
    run = VerifyRun(tier="configured-nvme", argv=[], git_head=None, root=tmp_path)

    rc, _elapsed, metadata = _run(
        "pytest configured NVMe root",
        [sys.executable, "-c", "print('managed resource sampler remains active')"],
        run=run,
    )

    assert rc == 0
    assert metadata["pytest_tmpfs"] is False
    assert metadata["pytest_tmpfs_budget_mb"] is None
    assert metadata["resource_sample_count"] >= 1


def test_run_propagates_explicit_basetemp_to_resource_policy(tmp_path: Path) -> None:
    explicit = tmp_path / "diagnostic-basetemp"
    captured: dict[str, str] = {}
    completed = subprocess.CompletedProcess(args=["pytest"], returncode=0, stdout="1 passed in 0.1s\n", stderr="")

    def apply_policy(env: dict[str, str], **_kwargs: object) -> tuple[dict[str, str], None]:
        captured.update(env)
        return env, None

    with (
        patch("devtools.verify.apply_managed_pytest_runtime_policy", side_effect=apply_policy),
        patch("devtools.verify._run_pytest_with_heartbeat", return_value=completed),
        patch("devtools.verify._read_pytest_report", return_value=None),
    ):
        rc, _elapsed, _metadata = _run("pytest focused", ["pytest", "--basetemp", str(explicit)])

    assert rc == 0
    assert captured["POLYLOGUE_PYTEST_EXPLICIT_BASETEMP"] == str(explicit)


def test_run_propagates_pytest_addopts_basetemp_to_resource_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    explicit = tmp_path / "diagnostic-basetemp"
    captured: dict[str, str] = {}
    completed = subprocess.CompletedProcess(args=["pytest"], returncode=0, stdout="1 passed in 0.1s\n", stderr="")
    monkeypatch.setenv("PYTEST_ADDOPTS", f"--basetemp {explicit}")

    def apply_policy(env: dict[str, str], **_kwargs: object) -> tuple[dict[str, str], None]:
        captured.update(env)
        return env, None

    with (
        patch("devtools.verify.apply_managed_pytest_runtime_policy", side_effect=apply_policy),
        patch("devtools.verify._run_pytest_with_heartbeat", return_value=completed),
        patch("devtools.verify._read_pytest_report", return_value=None),
    ):
        rc, _elapsed, _metadata = _run("pytest focused", ["pytest"])

    assert rc == 0
    assert captured["POLYLOGUE_PYTEST_EXPLICIT_BASETEMP"] == str(explicit)


def test_run_clears_stale_current_statistics_before_an_interrupted_pytest_step(tmp_path: Path) -> None:
    stale_statistics = tmp_path / verify_runs.CURRENT_STATISTICS_PATH
    stale_statistics.parent.mkdir(parents=True)
    stale_statistics.write_text('{"node_count": 99}\n', encoding="utf-8")

    with patch("devtools.verify._run_pytest_with_heartbeat", side_effect=KeyboardInterrupt):
        rc, _elapsed, metadata = _run("pytest focused", ["pytest"])

    assert rc == 130
    assert metadata["diagnosis"] == "pytest_interrupted"
    assert not stale_statistics.exists()


def test_explicit_basetemp_policy_uses_actual_path_for_admission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    shm, scratch = _patch_basetemp_roots(monkeypatch, tmp_path, realm_mounted=True)
    _patch_resource_capacity(monkeypatch, shm=shm, scratch=scratch, available_mb=15_190)
    explicit = tmp_path / "diagnostic-basetemp"
    explicit.mkdir()
    monkeypatch.setattr("devtools.verify_runs._headroom_kb", lambda _path: 32 * 1024 * 1024)

    _env, policy = apply_managed_pytest_runtime_policy(
        {"POLYLOGUE_PYTEST_EXPLICIT_BASETEMP": str(explicit)}, worker_count=0, full_suite=False
    )

    assert policy is not None
    assert policy.basetemp_root == str(explicit)
    assert policy.basetemp_label == "explicit"


def test_explicit_tmpfs_basetemp_requires_declared_demand_and_headroom(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    shm, scratch = _patch_basetemp_roots(monkeypatch, tmp_path, realm_mounted=True)
    _patch_resource_capacity(monkeypatch, shm=shm, scratch=scratch, available_mb=15_190)
    explicit = shm / "pytest-polylogue-diagnostic"
    monkeypatch.setattr("devtools.verify_runs._headroom_kb", lambda _path: 2500 * 1024)

    with pytest.raises(PytestResourceError, match="need >= 3072 MiB"):
        apply_managed_pytest_runtime_policy(
            {verify_runs.PYTEST_EXPLICIT_BASETEMP_ENV: str(explicit)}, worker_count=4, full_suite=True
        )


def test_explicit_tmpfs_basetemp_reports_adaptive_and_filesystem_refusals_together(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    shm, scratch = _patch_basetemp_roots(monkeypatch, tmp_path, realm_mounted=True)
    _patch_resource_capacity(monkeypatch, shm=shm, scratch=scratch, available_mb=3072)
    explicit = shm / "pytest-polylogue-diagnostic"
    monkeypatch.setattr("devtools.verify_runs._headroom_kb", lambda _path: 2 * 1024 * 1024)

    with pytest.raises(PytestResourceError) as excinfo:
        apply_managed_pytest_runtime_policy(
            {verify_runs.PYTEST_EXPLICIT_BASETEMP_ENV: str(explicit)}, worker_count=0, full_suite=True
        )
    message = str(excinfo.value)

    assert "declared demand=1522 MiB" in message
    assert "safe tmpfs budget=1082 MiB" in message
    assert "available filesystem space=2048 MiB" in message
    assert "required filesystem headroom=2546 MiB" in message


def test_supervisor_never_cleans_an_explicit_tmpfs_basetemp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verify_runs, "PYTEST_TMPFS_ROOT", tmp_path / "dev-shm")
    explicit = verify_runs.PYTEST_TMPFS_ROOT / "pytest-polylogue-diagnostic"

    cleanup_path = verify._supervised_tmpfs_cleanup_path(
        root=tmp_path,
        run_id="run-1",
        env={verify_runs.PYTEST_EXPLICIT_BASETEMP_ENV: str(explicit), "POLYLOGUE_PYTEST_TMPFS": "1"},
    )

    assert cleanup_path is None


def test_run_resource_refusal_returns_finalized_compact_statistics(tmp_path: Path) -> None:
    run = VerifyRun(tier="quick", argv=["--quick"], git_head="head", root=tmp_path)

    with patch(
        "devtools.verify.apply_managed_pytest_runtime_policy",
        side_effect=PytestResourceError("starved basetemp"),
    ):
        rc, _elapsed, metadata = _run("pytest testmon", ["pytest", "-n", "0"], run=run)

    assert rc == 125
    assert metadata["statistics"]["node_count"] == 0
    assert metadata["statistics_path"].endswith("statistics.json")


def test_run_receipt_uses_capped_pytest_command_concurrency() -> None:
    completed = subprocess.CompletedProcess(args=["pytest"], returncode=0, stdout="1 passed in 0.1s\n", stderr="")

    class UncappedPolicy:
        workers = 12

        def to_dict(self) -> dict[str, int]:
            return {"workers": self.workers}

    with (
        patch(
            "devtools.verify.apply_managed_pytest_runtime_policy", return_value=({}, UncappedPolicy())
        ) as apply_policy,
        patch("devtools.verify._run_pytest_with_heartbeat", return_value=completed),
        patch("devtools.verify._read_pytest_report", return_value=None),
    ):
        rc, _elapsed, metadata = _run("pytest seed-testmon", ["pytest", "--testmon", "-n", "4"])

    assert rc == 0
    assert apply_policy.call_args.kwargs["worker_count"] == 4
    assert apply_policy.call_args.kwargs["full_suite"] is True
    assert metadata["pytest_runtime_policy"] == {"workers": 12}
    assert metadata["workload_receipt"]["spec"]["concurrency"] == 4


@pytest.mark.parametrize(
    ("label", "full_suite"),
    [
        ("pytest focused", False),
        ("pytest testmon", False),
        ("pytest testmon (broad)", True),
        ("pytest seed-testmon", True),
        ("pytest seed-testmon shard 1/4", True),
        ("pytest full (parallel)", True),
        ("pytest load-sensitive (isolated)", True),
    ],
)
def test_run_scopes_measured_full_suite_basetemp_demand(tmp_path: Path, label: str, full_suite: bool) -> None:
    completed = subprocess.CompletedProcess(args=["pytest"], returncode=0, stdout="1 passed in 0.1s\n", stderr="")

    class FocusedPolicy:
        workers = 0

        def to_dict(self) -> dict[str, int]:
            return {"workers": self.workers}

    run = VerifyRun(tier="focused-test", argv=["tests/unit/example.py"], git_head="head", root=tmp_path)
    with (
        patch(
            "devtools.verify.apply_managed_pytest_runtime_policy", return_value=({}, FocusedPolicy())
        ) as apply_policy,
        patch("devtools.verify._run_pytest_with_heartbeat", return_value=completed),
        patch("devtools.verify._read_pytest_report", return_value=None),
    ):
        rc, _elapsed, _metadata = _run(label, ["pytest", "tests/unit/example.py", "-n", "0"], run=run)

    assert rc == 0
    assert apply_policy.call_args.kwargs["worker_count"] == 0
    assert apply_policy.call_args.kwargs["full_suite"] is full_suite


def test_bench_slo_forces_nested_pytest_to_managed_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shm, scratch = _patch_basetemp_roots(monkeypatch, tmp_path, realm_mounted=True)
    inherited_tmpfs_root = shm / "inherited-benchmark"
    inherited_tmpfs_root.mkdir()
    scratch.mkdir()
    run = VerifyRun(tier="lab", argv=[], git_head=None, root=tmp_path)
    monkeypatch.setenv("POLYLOGUE_PYTEST_BASETEMP_ROOT", str(inherited_tmpfs_root))
    managed_env = {
        "POLYLOGUE_PYTEST_TMPFS": "0",
        "POLYLOGUE_PYTEST_BASETEMP_ROOT": str(scratch),
    }
    completed = subprocess.CompletedProcess(args=["devtools", "bench", "slo"], returncode=0, stdout="", stderr="")

    with (
        patch("devtools.verify.apply_managed_pytest_runtime_policy", return_value=(managed_env, None)) as apply_policy,
        patch("devtools.verify.subprocess.run", return_value=completed) as subprocess_run,
    ):
        rc, _elapsed, _metadata = _run("bench slo", ["devtools", "bench", "slo"], run=run)

    assert rc == 0
    assert apply_policy.call_args.kwargs == {"worker_count": 0, "full_suite": False}
    policy_input = apply_policy.call_args.args[0]
    assert policy_input["POLYLOGUE_PYTEST_TMPFS"] == "0"
    assert "POLYLOGUE_PYTEST_BASETEMP_ROOT" not in policy_input
    env = subprocess_run.call_args.kwargs["env"]
    assert env["POLYLOGUE_VERIFY_RUN_ID"] == run.run_id
    assert env["POLYLOGUE_PYTEST_RUN_ID"] == run.run_id
    assert env["POLYLOGUE_PYTEST_TMPFS"] == "0"
    assert env["POLYLOGUE_PYTEST_BASETEMP_ROOT"] == str(scratch)


def test_run_forces_subprocesses_to_current_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYLOGUE_ROOT", "/stale/main")
    monkeypatch.setenv("POLYLOGUE_REPO_ROOT", "/stale/main")
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", "/stale/main/.cache/pycache")
    monkeypatch.setenv("PYTHONPATH", "/stale/main")
    completed = subprocess.CompletedProcess(args=["devtools"], returncode=0, stdout="", stderr="")

    with patch("devtools.verify.subprocess.run", return_value=completed) as run:
        rc, _elapsed, _metadata = _run("render all", ["devtools", "render all", "--check"])

    assert rc == 0
    env = run.call_args.kwargs["env"]
    assert env["POLYLOGUE_ROOT"] == str(ROOT)
    assert env["POLYLOGUE_REPO_ROOT"] == str(ROOT)
    assert env["PYTHONPYCACHEPREFIX"] == str(ROOT / ".cache" / "pycache")
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(ROOT)
    assert env["POLYLOGUE_PYTEST_EVENTS_PATH"] == str(ROOT / PYTEST_EVENTS_PATH)


def test_verify_subprocess_env_removes_cloud_basetemp_in_local_worktree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYLOGUE_PYTEST_BASETEMP_ROOT", "/tmp/polylogue-pytest")
    completed = subprocess.CompletedProcess(args=["devtools"], returncode=0, stdout="", stderr="")

    with patch("devtools.verify.subprocess.run", return_value=completed) as run:
        _run("render all", ["devtools", "render all", "--check"])

    env = run.call_args.kwargs["env"]
    assert "POLYLOGUE_PYTEST_BASETEMP_ROOT" not in env


def test_run_reads_structured_pytest_report() -> None:
    """The structured pytest report is the primary metadata source (#1026, #998)."""
    completed = subprocess.CompletedProcess(
        args=["pytest"],
        returncode=0,
        stdout="",
        stderr="",
    )
    report = {
        "summary": {"passed": 10, "failed": 1, "skipped": 2, "total": 13},
        "duration": 4.56,
    }

    with (
        patch("devtools.verify._run_pytest_with_heartbeat", return_value=completed),
        patch("devtools.verify._read_pytest_report", return_value=report),
        patch("devtools.verify.apply_managed_pytest_runtime_policy", return_value=({}, None)),
    ):
        rc, _elapsed, metadata = _run("pytest testmon", ["pytest", "--testmon", "-n", "8"])

    assert rc == 0
    assert metadata["count"] == 13  # passed+failed+skipped
    assert metadata["passed"] == 10
    assert metadata["failed"] == 1
    assert metadata["skipped"] == 2
    assert metadata["total"] == 13
    assert metadata["pytest_duration_s"] == 4.56
    assert metadata["report_path"] == str(PYTEST_REPORT_PATH)
    assert metadata["events_path"] == str(PYTEST_EVENTS_PATH)
    assert metadata["output_path"] == str(PYTEST_OUTPUT_PATH)
    assert metadata["pytest_workers"] == "8"
    assert metadata["pytest_selection"] == "testmon"


def test_pytest_run_emits_heartbeat_for_long_silent_child(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("POLYLOGUE_VERIFY_HEARTBEAT_S", "0.1")

    rc, _elapsed, metadata = _run("pytest heartbeat", [sys.executable, "-c", "import time; time.sleep(0.25)"])

    captured = capsys.readouterr()
    assert rc == 0
    assert metadata["heartbeat_s"] == 0.1
    assert "command:" in captured.err
    assert "still running: supervisor=" in captured.err
    assert ", controller=" in captured.err
    assert "elapsed=" in captured.err


def test_pytest_run_heartbeat_reports_latest_test_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POLYLOGUE_VERIFY_HEARTBEAT_S", "0.05")
    nodeid = "tests/unit/example.py::test_slow_node"
    script = (
        "import json, os, pathlib, time; "
        "path = pathlib.Path(os.environ['POLYLOGUE_PYTEST_EVENTS_PATH']); "
        "path.parent.mkdir(parents=True, exist_ok=True); "
        f"path.write_text(json.dumps({{'event': 'test_started', 'nodeid': {nodeid!r}}}) + '\\n'); "
        "time.sleep(0.2)"
    )

    rc, _elapsed, _metadata = _run("pytest heartbeat", [sys.executable, "-c", script])

    captured = capsys.readouterr()
    progress = json.loads((tmp_path / PYTEST_PROGRESS_PATH).read_text())
    assert rc == 0
    assert f"latest=test_started:{nodeid}" in captured.err
    assert progress["latest_test_event"]["nodeid"] == nodeid


def test_pytest_run_streams_child_output_live(capsys: pytest.CaptureFixture[str]) -> None:
    rc, _elapsed, _metadata = _run("pytest output", [sys.executable, "-c", "print('pytest-progress')"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "pytest-progress" in captured.err


def test_pytest_run_writes_live_progress_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    rc, _elapsed, metadata = _run("pytest progress", [sys.executable, "-c", "print('pytest-progress')"])

    assert rc == 0
    progress_path = tmp_path / PYTEST_PROGRESS_PATH
    assert metadata["progress_path"] == str(PYTEST_PROGRESS_PATH)
    assert metadata["output_path"] == str(PYTEST_OUTPUT_PATH)
    progress = json.loads(progress_path.read_text())
    assert progress["event"] == "finished"
    assert progress["returncode"] == 0
    assert progress["output_bytes"]["stdout"] > 0
    assert "updated_at" in progress
    assert "pytest-progress" in (tmp_path / PYTEST_OUTPUT_PATH).read_text()


def test_pytest_run_removes_stale_reports_before_child_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    stale_json = tmp_path / PYTEST_REPORT_PATH
    stale_junit = tmp_path / PYTEST_JUNIT_REPORT_PATH
    stale_json.parent.mkdir(parents=True)
    stale_junit.parent.mkdir(parents=True)
    stale_json.write_text('{"summary": {"failed": 99, "total": 99}}')
    stale_junit.write_text("<testsuite failures='99'/>")

    rc, _elapsed, metadata = _run("pytest stale", [sys.executable, "-c", "print('ok')"])

    assert rc == 0
    assert metadata["report_path"] is None
    assert metadata["report_status"] == "missing"
    assert metadata["junitxml_path"] == str(PYTEST_JUNIT_REPORT_PATH)
    assert not stale_json.exists()
    assert not stale_junit.exists()


def test_pytest_run_preserves_other_lane_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    primary_json = tmp_path / PYTEST_REPORT_PATH
    primary_junit = tmp_path / PYTEST_JUNIT_REPORT_PATH
    isolated_json = tmp_path / ".cache/pytest/last-pytest-isolated.json"
    isolated_junit = tmp_path / ".cache/test-reports/verify-latest-isolated.xml"
    primary_json.parent.mkdir(parents=True)
    primary_junit.parent.mkdir(parents=True)
    isolated_json.parent.mkdir(parents=True)
    primary_json.write_text('{"summary": {"passed": 7, "total": 7}}')
    primary_junit.write_text("<testsuite tests='7'/>")
    isolated_json.write_text('{"summary": {"failed": 99, "total": 99}}')
    isolated_junit.write_text("<testsuite failures='99'/>")

    rc, _elapsed, metadata = _run(
        "pytest isolated",
        [
            sys.executable,
            "-c",
            "print('ok')",
            f"--json-report-file={isolated_json}",
            f"--junitxml={isolated_junit}",
        ],
    )

    assert rc == 0
    assert primary_json.exists()
    assert primary_junit.exists()
    assert not isolated_json.exists()
    assert not isolated_junit.exists()
    assert metadata["report_path"] is None
    assert metadata["report_status"] == "missing"
    assert metadata["junitxml_path"] == str(isolated_junit)


def test_managed_pytest_run_reads_only_its_invocation_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    run = VerifyRun(tier="focused-test", argv=[], git_head="head", root=tmp_path)
    seen_report: Path | None = None

    def fake_pytest(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal seen_report
        seen_report = verify._pytest_json_report_path(cmd)
        assert seen_report is not None
        assert seen_report.parent == run.run_dir
        seen_report.write_text('{"summary":{"passed":1,"total":1}}', encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "1 passed in 0.01s\n", "")

    with patch("devtools.verify._run_pytest_with_heartbeat", side_effect=fake_pytest):
        rc, _elapsed, metadata = _run(
            "pytest focused",
            [sys.executable, "-m", "pytest", f"--json-report-file={PYTEST_REPORT_PATH}"],
            run=run,
        )

    assert rc == 0
    assert seen_report is not None and not seen_report.exists()
    assert metadata["report_path"].endswith("/pytest-report.json")


def test_pytest_progress_is_durable_per_step_and_mirrored_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    artifact_dir = tmp_path / ".cache" / "verify" / "runs" / "run" / "steps" / "01-pytest"

    verify._write_pytest_progress(
        event="running",
        cmd=["pytest"],
        started_at=0.0,
        elapsed_s=0.0,
        artifact_dir=str(artifact_dir),
    )

    durable = json.loads((artifact_dir / "progress.json").read_text(encoding="utf-8"))
    current = json.loads((tmp_path / PYTEST_PROGRESS_PATH).read_text(encoding="utf-8"))
    assert durable == current
    assert durable["event"] == "running"


def test_pytest_run_terminates_after_runtime_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POLYLOGUE_VERIFY_HEARTBEAT_S", "0.05")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_TIMEOUT_S", "1")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_STALL_TIMEOUT_S", "0")
    run = VerifyRun(tier="timeout-artifact", argv=[], git_head=None, root=tmp_path)

    rc, _elapsed, metadata = _run(
        "pytest timeout",
        [sys.executable, "-c", "import time; time.sleep(5)"],
        run=run,
    )

    captured = capsys.readouterr()
    step_containment = json.loads(
        (run.run_dir / "steps" / "01-pytest-timeout" / "containment.json").read_text(encoding="utf-8")
    )
    current_containment = json.loads((tmp_path / PYTEST_CONTAINMENT_PATH).read_text(encoding="utf-8"))
    assert rc == 124
    assert metadata["timeout_s"] == 1.0
    assert metadata["stall_timeout_s"] == 0.0
    assert metadata["events_path"] == str(PYTEST_EVENTS_PATH)
    assert metadata["output_path"] == str(PYTEST_OUTPUT_PATH)
    assert metadata["report_path"] is None
    assert metadata["report_status"] == "missing"
    assert metadata["progress_event"] == "terminated"
    assert metadata["termination_reason"] == "pytest runtime exceeded 1s"
    assert metadata["containment_mode"] in {"process-group", "systemd-scope"}
    assert metadata["containment_signals_sent"] == ["SIGTERM"]
    assert step_containment == current_containment
    assert current_containment["status"] == "terminated"
    assert current_containment["termination_reason"] == "pytest runtime exceeded 1s"
    assert "pytest runtime exceeded 1s" in captured.err
    assert "terminated owned pytest process group" in captured.err


def test_pytest_run_terminates_with_heartbeat_disabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("POLYLOGUE_VERIFY_HEARTBEAT_S", "0")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_TIMEOUT_S", "0.15")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_STALL_TIMEOUT_S", "0")

    rc, _elapsed, metadata = _run("pytest timeout", [sys.executable, "-c", "import time; time.sleep(5)"])

    captured = capsys.readouterr()
    assert rc == 124
    assert metadata["heartbeat_s"] == 0.0
    assert "pytest runtime exceeded 0.15s" in captured.err


def test_pytest_run_terminates_after_output_stall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("POLYLOGUE_VERIFY_HEARTBEAT_S", "0.05")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_TIMEOUT_S", "0")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_STALL_TIMEOUT_S", "0.15")

    run = VerifyRun(tier="output-stall", argv=[], git_head=None, root=tmp_path)
    rc, _elapsed, metadata = _run(
        "pytest stall",
        [sys.executable, "-c", "import time; print('progress', flush=True); time.sleep(5)"],
        run=run,
    )

    captured = capsys.readouterr()
    assert rc == 124
    assert metadata["timeout_s"] == 0.0
    assert metadata["stall_timeout_s"] == 0.15
    assert "progress" in captured.err
    assert "pytest produced no output for 0.15s" in captured.err
    assert "terminated owned pytest process group" in captured.err


def test_pytest_run_terminates_on_progress_stall_despite_flowing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """polylogue-27rb: an xdist-master-keeps-emitting D-state deadlock.

    Reproduces the confirmed hang shape: the child process keeps writing
    output continuously (never silent), but no NEW test-progress event is
    ever appended to the events ledger -- the output-silence stall check
    alone would run forever. The progress-based check must fire instead.
    """
    monkeypatch.setenv("POLYLOGUE_VERIFY_HEARTBEAT_S", "0.05")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_TIMEOUT_S", "0")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_STALL_TIMEOUT_S", "0.15")

    # _run's _clear_pytest_report wipes the event artifacts before the child
    # starts, so the child itself must write the one-and-only progress
    # event (matching a real pytest worker reporting "test started" then
    # wedging mid-test). Real xdist workers write their own live JSONL files
    # under POLYLOGUE_PYTEST_EVENTS_DIR; the merged path appears only after
    # pytest exits and therefore cannot drive an in-process stall detector.
    child_script = (
        "import json, os, time\n"
        "directory = os.environ['POLYLOGUE_PYTEST_EVENTS_DIR']\n"
        "os.makedirs(directory, exist_ok=True)\n"
        "path = os.path.join(directory, 'gw0.jsonl')\n"
        "with open(path, 'w') as f:\n"
        "    f.write(json.dumps({'event': 'test_started', 'nodeid': 'wedged::test', "
        "'updated_at': '2026-01-01T00:00:00Z'}) + '\\n')\n"
        "for _ in range(200):\n"
        "    print('progress', flush=True)\n"
        "    time.sleep(0.02)\n"
    )
    run = VerifyRun(tier="progress-stall", argv=[], git_head=None, root=tmp_path)
    rc, _elapsed, metadata = _run(
        "pytest stall",
        [sys.executable, "-c", child_script],
        run=run,
    )

    captured = capsys.readouterr()
    assert rc == 124
    assert metadata["stall_timeout_s"] == 0.15
    # Output kept flowing the whole time -- the OLD output-silence check
    # never had a chance to fire; only the progress-based check should.
    assert "progress" in captured.err
    assert "pytest produced no output" not in captured.err
    assert "pytest reported no test progress for 0.15s" in captured.err
    assert "wedged::test" in captured.err
    assert "terminated owned pytest process group" in captured.err


def test_pytest_timeout_env_defaults_and_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYLOGUE_VERIFY_PYTEST_TIMEOUT_S", raising=False)
    monkeypatch.delenv("POLYLOGUE_VERIFY_PYTEST_STALL_TIMEOUT_S", raising=False)
    assert _pytest_timeout_s() > 0
    assert _pytest_stall_timeout_s() > 0

    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_TIMEOUT_S", "-1")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_STALL_TIMEOUT_S", "nope")
    assert _pytest_timeout_s() == 0.0
    assert _pytest_stall_timeout_s() > 0

    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_TIMEOUT_S", "nan")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_STALL_TIMEOUT_S", "inf")
    assert _pytest_timeout_s() > 0
    assert _pytest_stall_timeout_s() > 0


def test_pytest_command_metadata_reports_worker_and_selection_policy() -> None:
    assert _pytest_command_metadata(["pytest", "--testmon", "-n", "8"]) == {
        "pytest_workers": "8",
        "pytest_selection": "testmon",
    }
    assert _pytest_command_metadata(["pytest", "--testmon", "--testmon-noselect", "-n", "16"]) == {
        "pytest_workers": "16",
        "pytest_selection": "testmon-noselect",
    }
    assert _pytest_command_metadata(["pytest", "-n", "16"]) == {
        "pytest_workers": "16",
        "pytest_selection": "full",
    }


def test_pytest_metadata_handles_empty_summary() -> None:
    """Robustness: a malformed/empty report still yields a metadata dict."""
    assert _pytest_metadata_from_report({}, report_path=PYTEST_REPORT_PATH) == {"report_path": str(PYTEST_REPORT_PATH)}


def test_read_pytest_report_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert _read_pytest_report(tmp_path / "missing.json") is None


def test_read_pytest_report_returns_none_for_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert _read_pytest_report(bad) is None


def test_read_pytest_report_returns_none_for_non_object(tmp_path: Path) -> None:
    bad = tmp_path / "list.json"
    bad.write_text("[1, 2, 3]")
    assert _read_pytest_report(bad) is None


def test_read_pytest_report_parses_valid_payload(tmp_path: Path) -> None:
    path = tmp_path / "ok.json"
    path.write_text('{"summary": {"passed": 3}, "duration": 1.0}')
    parsed = _read_pytest_report(path)
    assert parsed == {"summary": {"passed": 3}, "duration": 1.0}


def test_verify_continues_after_failed_cheap_step(capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[str] = []

    def fake_run(label: str, command: list[str], **kwargs: object) -> tuple[int, float, dict[str, object]]:
        calls.append(label)
        return 1, 0.01, {}

    with (
        patch("devtools.verify._run", side_effect=fake_run),
        patch("devtools.verify._git_head", return_value="head"),
        patch("devtools.verify._save_history"),
        patch("devtools.verify._stamp_head"),
        patch("devtools.verify._notify"),
    ):
        rc = main(["--quick", "--json"])

    assert rc == 1
    assert calls == [label for label, _command in build_verify_steps(quick=True, lab=False, skip_slow=False)]
    payload = json.loads(capsys.readouterr().out)
    assert payload["exit_code"] == 1
    assert payload["verification_scope"] == "non-test"
    assert payload["release_baseline_allowed"] is False


@pytest.mark.parametrize("fingerprints", [("unavailable", "stable"), ("stable", "unavailable")])
def test_verify_withholds_success_when_checkout_fingerprint_is_unavailable(
    capsys: pytest.CaptureFixture[str],
    fingerprints: tuple[str, str],
) -> None:
    with (
        patch("devtools.verify._run", return_value=(0, 0.01, {})),
        patch("devtools.verify._git_head", return_value="head"),
        patch("devtools.verify._save_history"),
        patch("devtools.verify._stamp_head"),
        patch("devtools.verify._notify"),
        patch("devtools.verify.worktree_fingerprint", side_effect=fingerprints),
    ):
        rc = main(["--quick", "--json"])

    assert rc == 125
    payload = json.loads(capsys.readouterr().out)
    assert payload["exit_code"] == 125
    checkout_step = next(step for step in payload["steps"] if step["name"] == "checkout stability")
    assert checkout_step["diagnosis"] == "checkout_fingerprint_unavailable"
    assert checkout_step["initial_worktree_fingerprint"] == fingerprints[0]
    assert checkout_step["final_worktree_fingerprint"] == fingerprints[1]


def test_verify_rejects_git_head_change_with_matching_worktree_fingerprints(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _StableMonitor:
        def __init__(self, _root: Path) -> None:
            pass

        def start(self) -> None:
            pass

        def finish(self) -> CheckoutMutationObservation:
            return CheckoutMutationObservation(changed=False, unavailable=False)

    with (
        patch("devtools.verify._run", return_value=(0, 0.01, {})),
        patch("devtools.verify._git_head", side_effect=("start-head", "different-head")),
        patch("devtools.verify.CheckoutMutationMonitor", _StableMonitor),
        patch("devtools.verify._save_history"),
        patch("devtools.verify._stamp_head"),
        patch("devtools.verify._notify"),
        patch("devtools.verify.worktree_fingerprint", return_value="stable"),
    ):
        assert main(["--quick", "--json"]) == 125

    payload = json.loads(capsys.readouterr().out)
    checkout_step = next(step for step in payload["steps"] if step["name"] == "checkout stability")
    assert checkout_step["diagnosis"] == "checkout_changed_during_verification"
    assert checkout_step["initial_git_head"] == "start-head"
    assert checkout_step["final_git_head"] == "different-head"


@pytest.mark.parametrize(
    ("fingerprints", "expected_diagnosis"),
    [
        (("unavailable", "stable"), "checkout_fingerprint_unavailable"),
        (("stable", "changed"), "checkout_changed_during_verification"),
    ],
)
def test_checkout_stability_failure_controls_every_broad_run_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fingerprints: tuple[str, str],
    expected_diagnosis: str,
) -> None:
    class _StableMonitor:
        def __init__(self, _root: Path) -> None:
            pass

        def start(self) -> None:
            pass

        def finish(self) -> CheckoutMutationObservation:
            return CheckoutMutationObservation(changed=False, unavailable=False)

    history: dict[str, Any] = {}
    receipt = tmp_path / "invocation" / "run.json"
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.setattr(
        verify,
        "assert_polylogue_matches_checkout",
        lambda *_args, **_kwargs: SimpleNamespace(polylogue_import_path=tmp_path / "polylogue", as_dict=lambda: {}),
    )
    monkeypatch.setenv(verify_runs.VERIFICATION_INVOCATION_ID_ENV, "broad-invocation")
    monkeypatch.setenv(verify_runs.VERIFICATION_RECEIPT_PATH_ENV, str(receipt))

    with (
        patch("devtools.verify._run", return_value=(0, 0.01, {"diagnosis": "pytest_passed"})),
        patch("devtools.verify._git_head", return_value="head"),
        patch("devtools.verify._save_history", side_effect=lambda entry: history.update(entry)),
        patch("devtools.verify._stamp_head"),
        patch("devtools.verify._notify"),
        patch("devtools.verify.CheckoutMutationMonitor", _StableMonitor),
        patch("devtools.verify.worktree_fingerprint", side_effect=fingerprints),
    ):
        assert main(["--quick", "--json"]) == 125

    payload = json.loads(capsys.readouterr().out)
    run_payload = json.loads(next((tmp_path / ".cache" / "verify" / "runs").glob("*/run.json")).read_text())
    current_payload = json.loads((tmp_path / ".cache" / "verify" / "current-run.json").read_text())
    receipt_payload = json.loads(receipt.read_text())

    for durable_payload in (history, payload, run_payload, current_payload, receipt_payload):
        assert durable_payload["diagnosis"] == expected_diagnosis
        assert durable_payload["final_worktree_fingerprint"] == fingerprints[1]


def test_transient_checkout_mutation_controls_every_broad_run_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _ChangedMonitor:
        def __init__(self, _root: Path) -> None:
            pass

        def start(self) -> None:
            pass

        def finish(self) -> CheckoutMutationObservation:
            return CheckoutMutationObservation(changed=True, unavailable=False)

    history: dict[str, Any] = {}
    receipt = tmp_path / "invocation" / "run.json"
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.setattr(
        verify,
        "assert_polylogue_matches_checkout",
        lambda *_args, **_kwargs: SimpleNamespace(polylogue_import_path=tmp_path / "polylogue", as_dict=lambda: {}),
    )
    monkeypatch.setattr(verify, "CheckoutMutationMonitor", _ChangedMonitor)
    monkeypatch.setenv(verify_runs.VERIFICATION_INVOCATION_ID_ENV, "broad-invocation")
    monkeypatch.setenv(verify_runs.VERIFICATION_RECEIPT_PATH_ENV, str(receipt))

    with (
        patch("devtools.verify._run", return_value=(0, 0.01, {"diagnosis": "pytest_passed"})),
        patch("devtools.verify._git_head", return_value="head"),
        patch("devtools.verify._save_history", side_effect=lambda entry: history.update(entry)),
        patch("devtools.verify._stamp_head"),
        patch("devtools.verify._notify"),
        patch("devtools.verify.worktree_fingerprint", return_value="stable"),
    ):
        assert main(["--quick", "--json"]) == 125

    payload = json.loads(capsys.readouterr().out)
    run_payload = json.loads(next((tmp_path / ".cache" / "verify" / "runs").glob("*/run.json")).read_text())
    current_payload = json.loads((tmp_path / ".cache" / "verify" / "current-run.json").read_text())
    receipt_payload = json.loads(receipt.read_text())
    for durable_payload in (history, payload, run_payload, current_payload, receipt_payload):
        assert durable_payload["diagnosis"] == "checkout_changed_during_verification"
        assert durable_payload["final_worktree_fingerprint"] == "stable"


def test_transient_checkout_mutation_discards_testmon_graph_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _ChangedMonitor:
        def __init__(self, _root: Path) -> None:
            pass

        def start(self) -> None:
            pass

        def finish(self) -> CheckoutMutationObservation:
            return CheckoutMutationObservation(changed=True, unavailable=False, observed_path="polylogue/example.py")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.setattr(verify, "CheckoutMutationMonitor", _ChangedMonitor)
    monkeypatch.setattr(
        verify,
        "assert_polylogue_matches_checkout",
        lambda *_args, **_kwargs: SimpleNamespace(polylogue_import_path=tmp_path / "polylogue", as_dict=lambda: {}),
    )
    TESTMON_DATA.parent.mkdir(parents=True)
    TESTMON_DATA.write_bytes(b"transient dependency graph")
    TESTMON_SEED_STAMP.write_text("{}", encoding="utf-8")
    affected_publish = MagicMock()
    selection_publish = MagicMock()

    with (
        patch("devtools.verify._anchor_verification_paths"),
        patch("devtools.verify.maybe_bootstrap_testmon_seed", return_value=None),
        patch("devtools.verify._testmon_preflight", return_value=None),
        patch("devtools.verify.build_verify_steps", return_value=[("pytest testmon", ["pytest", "--testmon"])]),
        patch("devtools.verify._run", return_value=(0, 0.01, {"selected_count": 1})),
        patch("devtools.verify._changed_executable_paths", return_value=("polylogue/example.py",)),
        patch("devtools.verify._record_testmon_affected_coverage", affected_publish),
        patch("devtools.verify._refresh_testmon_selection_attempt", selection_publish),
        patch("devtools.verify._git_head", return_value="head"),
        patch("devtools.verify._git_commit", return_value="base"),
        patch("devtools.verify._default_testmon_is_broad_change", return_value=False),
        patch("devtools.verify._testmon_release_baseline_permission", return_value=False),
        patch("devtools.verify._warn_low_memory"),
        patch("devtools.verify._save_history"),
        patch("devtools.verify._stamp_head"),
        patch("devtools.verify._notify"),
        patch("devtools.verify.worktree_fingerprint", return_value="stable"),
    ):
        assert main(["--json"]) == 125

    assert not TESTMON_DATA.exists()
    assert not TESTMON_SEED_STAMP.exists()
    affected_publish.assert_not_called()
    selection_publish.assert_not_called()
    assert json.loads(capsys.readouterr().out)["diagnosis"] == "checkout_changed_during_verification"


def test_verify_stops_after_failed_heavy_step(capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[str] = []

    def fake_run(label: str, command: list[str], **kwargs: object) -> tuple[int, float, dict[str, object]]:
        calls.append(label)
        return (1 if label.startswith("pytest") else 0), 0.01, {}

    with (
        patch("devtools.verify._run", side_effect=fake_run),
        patch("devtools.verify.build_verify_steps", return_value=[("pytest testmon", ["pytest"])]),
        patch("devtools.verify._git_head", return_value="head"),
        patch("devtools.verify._git_commit", return_value="base"),
        patch("devtools.verify._default_testmon_is_broad_change", return_value=False),
        patch("devtools.verify._save_history"),
        patch("devtools.verify._stamp_head"),
        patch("devtools.verify._notify"),
        patch("devtools.verify._testmon_preflight", return_value=None),
    ):
        rc = main(["--json"])

    assert rc == 1
    assert calls[-1].startswith("pytest")
    payload = capsys.readouterr().out
    assert '"exit_code": 1' in payload


@pytest.mark.parametrize(
    ("shard_results", "expected_exit", "expected_diagnosis", "expected_statuses"),
    [
        (
            [(124, "pytest_timeout"), (0, "pytest_passed")],
            124,
            "pytest_timeout",
            ["incomplete", "pending"],
        ),
        (
            [(1, "pytest_failed"), (0, "pytest_passed")],
            1,
            "pytest_failed",
            ["complete", "complete"],
        ),
        (
            [(1, "pytest_failed"), (124, "pytest_timeout")],
            124,
            "pytest_timeout",
            ["complete", "incomplete"],
        ),
        (
            [(1, "pytest_failed"), (0, "pytest_passed")],
            1,
            "pytest_failed",
            ["incomplete", "pending"],
        ),
    ],
)
def test_seed_testmon_stops_only_after_infrastructure_failed_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    shard_results: list[tuple[int, str]],
    expected_exit: int,
    expected_diagnosis: str,
    expected_statuses: list[str],
) -> None:
    nodeids = ["tests/test_seed.py::test_one", "tests/test_seed.py::test_two"]
    collection_dir = tmp_path / "collection"
    collection_dir.mkdir()
    (collection_dir / "selection.json").write_text(
        json.dumps(
            {
                "selected_count": len(nodeids),
                "selected_nodeids": nodeids,
                "selected_nodeids_omitted": 0,
            }
        )
    )
    calls: list[str] = []
    checkpointed: list[int] = []
    finalized_shard_statuses: list[str] = []

    def fake_run(label: str, command: list[str], **kwargs: object) -> tuple[int, float, dict[str, object]]:
        del command, kwargs
        calls.append(label)
        if label == "pytest seed-testmon collect":
            return 0, 0.01, {"artifact_dir": str(collection_dir)}
        if label.startswith("pytest seed-testmon shard "):
            shard_index = int(label.rsplit(" ", 1)[1].split("/", 1)[0])
            shard_exit, diagnosis = shard_results[shard_index - 1]
            return shard_exit, 0.01, {"diagnosis": diagnosis}
        pytest.fail(f"unexpected seed step: {label}")

    def fake_checkpoint(*, prepared: dict[str, object], shard_index: int, step: dict[str, object]) -> dict[str, object]:
        del step
        checkpointed.append(shard_index)
        raw_shards = prepared["shards"]
        assert isinstance(raw_shards, list)
        assert all(isinstance(shard, dict) for shard in raw_shards)
        shards = [dict(shard) for shard in raw_shards]
        shards[shard_index - 1]["status"] = expected_statuses[shard_index - 1]
        return {**prepared, "shards": shards}

    def fake_finalize(
        *, prepared: dict[str, object], step_results: list[dict[str, object]], exit_code: int
    ) -> dict[str, object]:
        del step_results
        assert exit_code == expected_exit
        raw_shards = prepared["shards"]
        assert isinstance(raw_shards, list)
        assert all(isinstance(shard, dict) for shard in raw_shards)
        finalized_shard_statuses.extend(str(shard["status"]) for shard in raw_shards)
        return {
            "status": "incomplete" if "incomplete" in expected_statuses else "complete",
            "outcome": "resource_timeout" if expected_exit == 124 else "red-baseline",
            "resume": False,
            "expected_count": len(nodeids),
            "release_baseline_allowed": False,
        }

    monkeypatch.setattr(verify, "TESTMON_SEED_SHARD_SIZE", 1)
    with (
        patch("devtools.verify._anchor_verification_paths"),
        patch("devtools.verify.maybe_bootstrap_testmon_seed", return_value=None),
        patch("devtools.verify._run", side_effect=fake_run),
        patch(
            "devtools.verify.build_verify_steps",
            return_value=[("pytest seed-testmon collect", ["pytest", "--collect-only"])],
        ),
        patch("devtools.verify._git_head", return_value="head"),
        patch("devtools.verify._git_committed_tree", return_value="tree"),
        patch(
            "devtools.verify._testmon_seed_identity",
            return_value={"git_head": "head", "git_tree": "tree", "skip_slow": False, "lab": False},
        ),
        patch("devtools.verify._testmon_seed_can_resume", return_value=False),
        patch("devtools.verify._checkpoint_testmon_seed_shard", side_effect=fake_checkpoint),
        patch("devtools.verify._finalize_testmon_seed_attempt", side_effect=fake_finalize),
        patch("devtools.verify._testmon_release_baseline_permission", return_value=False),
        patch("devtools.verify._warn_low_memory"),
        patch("devtools.verify._save_history"),
        patch("devtools.verify._stamp_head"),
        patch("devtools.verify._notify"),
    ):
        rc = main(["--seed-testmon", "--json"])

    assert rc == expected_exit
    executed_shards = sum(status != "pending" for status in expected_statuses)
    assert calls == [
        "pytest seed-testmon collect",
        *(f"pytest seed-testmon shard {index}/2" for index in range(1, executed_shards + 1)),
    ]
    assert checkpointed == list(range(1, executed_shards + 1))
    assert finalized_shard_statuses == expected_statuses
    output = json.loads(capsys.readouterr().out)
    assert output["exit_code"] == expected_exit
    assert output["diagnosis"] == expected_diagnosis


@pytest.mark.parametrize(
    ("argv", "expected_scope", "expected_permission"),
    [
        (["--all", "--skip-slow"], "narrow-terminal", False),
        (["--all", "--skip-slow", "--terminal-authorization", "narrow-terminal"], "narrow-terminal", True),
    ],
)
def test_verify_main_types_skip_slow_terminal_authority(
    capsys: pytest.CaptureFixture[str], argv: list[str], expected_scope: str, expected_permission: bool
) -> None:
    def fake_run(label: str, command: list[str], **kwargs: object) -> tuple[int, float, dict[str, object]]:
        del label, command, kwargs
        return 0, 0.01, {}

    with (
        patch("devtools.verify._run", side_effect=fake_run),
        patch("devtools.verify.build_verify_steps", return_value=[("pytest full", ["pytest"])]),
        patch("devtools.verify._git_head", return_value="head"),
        patch("devtools.verify._save_history") as save_history,
        patch("devtools.verify._stamp_head"),
        patch("devtools.verify._notify"),
    ):
        assert main([*argv, "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["verification_scope"] == expected_scope
    assert payload["release_baseline_allowed"] is expected_permission
    assert payload["terminal_authorization"] == ("narrow-terminal" if expected_permission else None)
    assert save_history.call_args.args[0]["checkout_root"] == str(ROOT.resolve())


def test_verify_refuses_unbudgeted_pytest_before_running_steps(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch("devtools.verify.build_verify_steps", side_effect=PytestResourceError("only 0.50 GiB available")),
        patch("devtools.verify._git_head", return_value="head"),
        patch("devtools.verify._git_commit", return_value="base"),
        patch("devtools.verify._default_testmon_is_broad_change", return_value=False),
        patch("devtools.verify._testmon_preflight", return_value=None),
        patch("devtools.verify._run") as run,
        patch("devtools.verify._save_history") as save_history,
    ):
        rc = main(["--json"])

    assert rc == 125
    run.assert_not_called()
    assert save_history.call_args.args[0]["diagnosis"] == "pytest_resource_preflight_failed"
    assert "only 0.50 GiB available" in capsys.readouterr().err


def test_verify_anchors_relative_state_to_checkout_when_invoked_from_subdirectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(ROOT / "devtools")

    _anchor_verification_paths()

    assert Path.cwd() == ROOT.resolve()


def test_verify_rejects_zero_testmon_selection_for_executable_change(
    capsys: pytest.CaptureFixture[str],
) -> None:
    changed_executable_paths = MagicMock(return_value=("polylogue/example.py",))

    def fake_run(label: str, command: list[str], **kwargs: object) -> tuple[int, float, dict[str, object]]:
        del command, kwargs
        return 0, 0.01, ({"selected_count": 0} if label.startswith("pytest") else {})

    with (
        patch("devtools.verify._run", side_effect=fake_run),
        patch("devtools.verify._git_head", return_value="head"),
        patch("devtools.verify._git_commit", return_value="pinned-base"),
        patch("devtools.verify._default_testmon_is_broad_change", return_value=False),
        patch("devtools.verify._save_history"),
        patch("devtools.verify._stamp_head"),
        patch("devtools.verify._notify"),
        patch("devtools.verify._testmon_preflight", return_value=None),
        patch("devtools.verify._changed_executable_paths", changed_executable_paths),
        patch("devtools.verify._matching_testmon_coverage", return_value=None),
    ):
        rc = main(["--json"])

    assert rc == 5
    payload = json.loads(capsys.readouterr().out)
    pytest_step = next(step for step in payload["steps"] if step["name"].startswith("pytest"))
    assert pytest_step["diagnosis"] == "zero_testmon_selection_for_executable_change"
    assert pytest_step["zero_selection_changed_paths"] == ["polylogue/example.py"]
    changed_executable_paths.assert_called_once_with("pinned-base", "head")


def test_verify_accepts_zero_testmon_selection_after_matching_coverage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run(label: str, command: list[str], **kwargs: object) -> tuple[int, float, dict[str, object]]:
        del command, kwargs
        return 0, 0.01, ({"selected_count": 0} if label.startswith("pytest") else {})

    with (
        patch("devtools.verify._run", side_effect=fake_run),
        patch("devtools.verify.build_verify_steps", return_value=[("pytest testmon", ["pytest"])]),
        patch("devtools.verify._git_head", return_value="head"),
        patch("devtools.verify._git_commit", return_value="base"),
        patch("devtools.verify._default_testmon_is_broad_change", return_value=False),
        patch("devtools.verify._save_history"),
        patch("devtools.verify._stamp_head"),
        patch("devtools.verify._notify"),
        patch("devtools.verify._testmon_preflight", return_value=None),
        patch("devtools.verify._changed_executable_paths", return_value=("polylogue/example.py",)),
        patch("devtools.verify._matching_testmon_coverage", return_value="successful_affected_run"),
    ):
        rc = main(["--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    pytest_step = next(step for step in payload["steps"] if step["name"].startswith("pytest"))
    assert pytest_step["zero_selection_coverage"] == "successful_affected_run"


def test_testmon_coverage_receipts_are_content_exact() -> None:
    paths = ("polylogue/example.py",)
    _write_real_testmon_state()
    assert _matching_testmon_coverage(paths) is None

    TESTMON_SEED_STAMP.unlink()
    with patch("devtools.verify.worktree_fingerprint", return_value="affected"):
        _record_testmon_affected_coverage(
            executable_paths=paths,
            selected_count=3,
            run_id="run-1",
        )
        assert TESTMON_AFFECTED_STAMP.exists()
        assert _matching_testmon_coverage(paths) == "successful_affected_run"
        assert _matching_testmon_coverage(("polylogue/other.py",)) is None

    with patch("devtools.verify.worktree_fingerprint", return_value="changed"):
        assert _matching_testmon_coverage(paths) is None

    TESTMON_AFFECTED_STAMP.write_text(json.dumps({"identity": {"worktree_fingerprint": "affected"}}))
    with patch("devtools.verify.worktree_fingerprint", return_value="affected"):
        assert _matching_testmon_coverage(paths) is None


def test_failed_step_stop_policy_distinguishes_cheap_and_heavy_steps() -> None:
    assert _stop_after_failed_step("ruff check") is False
    assert _stop_after_failed_step("verify layering") is False
    assert _stop_after_failed_step("pytest testmon") is True
    assert _stop_after_failed_step("lab smoke") is True
    assert _stop_after_failed_step("bench slo") is True


def test_completion_notification_uses_pytest_count() -> None:
    summary = _format_completion_notification(
        exit_code=0,
        total_duration=118.2,
        step_results=[
            {"name": "ruff check", "duration_s": 0.1, "exit": 0},
            {"name": "pytest affected", "duration_s": 100.0, "exit": 0, "count": 12},
        ],
    )

    assert summary == "PASS (118s), 12 tests"


def test_completion_notification_omits_unknown_pytest_count() -> None:
    summary = _format_completion_notification(
        exit_code=0,
        total_duration=118.2,
        step_results=[{"name": "pytest", "duration_s": 100.0, "exit": 0}],
    )

    assert summary == "PASS (118s)"


def test_default_testmon_step_pairs_marker_filter_with_forceselect() -> None:
    """#1632: any pytest -m marker filter in the default lane MUST be paired with --testmon-forceselect.

    Without ``--testmon-forceselect``, a marker selector deactivates
    pytest-testmon's affected-test selection and the run silently
    expands to the whole suite — PR #1550 fixed exactly this regression
    after a full week of every default verify running 9.5K tests
    instead of the affected subset. This invariant is the regression
    guard so the footgun cannot re-land silently again.
    """
    steps = build_verify_steps(quick=False, lab=False, skip_slow=False)
    label, command = steps[-1]
    assert label == "pytest testmon"
    if "-m" in command:
        assert "--testmon-forceselect" in command, (
            f"marker filter without --testmon-forceselect re-introduces the #1550 silent-deselection footgun: {command}"
        )


def test_skip_slow_testmon_step_keeps_forceselect_with_compound_marker() -> None:
    """``--skip-slow`` composes the marker; the pairing invariant must still hold."""
    steps = build_verify_steps(quick=False, lab=False, skip_slow=True)
    label, command = steps[-1]
    assert label == "pytest testmon"
    assert "-m" in command
    assert "--testmon-forceselect" in command


def test_verify_does_not_notify_on_pass() -> None:
    """Passing verify runs stay silent — only failures send a desktop popup."""

    def fake_run(label: str, command: list[str], **kwargs: object) -> tuple[int, float, dict[str, object]]:
        return 0, 0.01, {}

    with (
        patch("devtools.verify._run", side_effect=fake_run),
        patch("devtools.verify._git_head", return_value="head"),
        patch("devtools.verify._save_history"),
        patch("devtools.verify._stamp_head"),
        patch("devtools.verify._notify") as notify,
    ):
        rc = main(["--quick", "--json"])

    assert rc == 0
    notify.assert_not_called()


def test_verify_notifies_on_failure() -> None:
    """Failing verify runs still send a desktop popup so the operator notices."""

    def fake_run(label: str, command: list[str], **kwargs: object) -> tuple[int, float, dict[str, object]]:
        return 1, 0.01, {}

    with (
        patch("devtools.verify._run", side_effect=fake_run),
        patch("devtools.verify._git_head", return_value="head"),
        patch("devtools.verify._save_history"),
        patch("devtools.verify._stamp_head"),
        patch("devtools.verify._notify") as notify,
    ):
        rc = main(["--quick", "--json"])

    assert rc == 1
    notify.assert_called_once()
