from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

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
    _worktree_fingerprint,
    build_verify_steps,
    main,
)
from devtools.verify_runs import (
    PytestResourceError,
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
    pytest_tmpfs_budget_kb,
    resolve_pytest_basetemp_root,
    xdist_uninterruptible_stall_reason,
)


@pytest.fixture(autouse=True)
def _isolate_verify_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep supervisor and testmon receipts private to each test.

    These tests exercise the real checkout guard, so leaving a synthetic
    ``.cache/testmon`` behind makes a later guard test observe a fixture
    artifact as if it were a developer's checkout state.
    """
    monkeypatch.chdir(tmp_path)
    checkout_cache = ROOT / ".cache" / "testmon"
    if checkout_cache.exists():
        shutil.move(str(checkout_cache), str(tmp_path / "checkout-testmon-generated"))
    for name in (
        "TESTMON_DATA",
        "TESTMON_SEED_STAMP",
        "TESTMON_SEED_ATTEMPT",
        "TESTMON_AFFECTED_STAMP",
    ):
        isolated = tmp_path / ".cache" / "testmon" / getattr(verify, name).name
        monkeypatch.setattr(verify, name, isolated)
        monkeypatch.setattr(sys.modules[__name__], name, isolated)


@pytest.fixture(scope="session", autouse=True)
def _quarantine_checkout_testmon(tmp_path_factory: pytest.TempPathFactory) -> object:
    """Prevent subprocess-backed verify tests from contaminating the checkout.

    A few tests intentionally re-anchor verification to ``ROOT``.  Their child
    pytest process therefore uses the real checkout's relative testmon path,
    even though the parent test has a private working directory.  Keep any
    pre-existing state safe for restoration and quarantine only state created
    during this test module.
    """
    checkout_cache = ROOT / ".cache" / "testmon"
    quarantine = tmp_path_factory.mktemp("checkout-testmon")
    original: Path | None = None
    if checkout_cache.exists():
        original = quarantine / "original"
        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(checkout_cache), str(original))
    try:
        yield
    finally:
        if checkout_cache.exists():
            shutil.move(str(checkout_cache), str(quarantine / "generated"))
        if original is not None and not checkout_cache.exists():
            checkout_cache.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(original), str(checkout_cache))


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
                "outcome": "passed",
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
    assert json.loads(TESTMON_SEED_ATTEMPT.read_text())["shards"][0]["node_outcomes"][0]["outcome"] == "passed"
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
    (step / "containment.json").write_text(json.dumps({"tmpfs_cleanup_complete": True, "exit_code": 0}))

    result = aggregate_pytest_statistics(step, command=["pytest"], step_result={"exit": 0})

    assert result["node_count"] == 1
    assert result["phases"]["call"]["p50_s"] == 2.0
    assert result["phases"]["setup"]["count"] == 1
    assert result["storage"]["basetemp_logical_bytes_max"] == 12 * 1024
    assert result["resources"]["peak_tree_pss_kb"] == 80
    assert result["cleanup"]["complete"] is True


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
        ],
    )

    verify._print_history()

    output = capsys.readouterr().out
    assert "quick" in output
    assert "focused-" in output
    assert "pytest focused(0s FAIL)" in output


def test_verify_history_appends_concurrent_records_without_interleaving(tmp_path: Path) -> None:
    history = tmp_path / "state" / "verify-history.jsonl"

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda sequence: append_verify_history({"sequence": sequence}, path=history), range(64)))

    rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
    assert sorted(row["sequence"] for row in rows) == list(range(64))


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
        expected_nodeids=["tests/test_a.py::test_repaired", "tests/test_b.py::test_prior"],
        database={"node_outcomes": {"tests/test_b.py::test_prior": "passed"}},
        pytest_step={},
        use_database_fallback=False,
        prior_node_outcomes={
            "tests/test_b.py::test_prior": {"nodeid": "tests/test_b.py::test_prior", "outcome": "passed"}
        },
    )

    assert {item["nodeid"]: item["outcome"] for item in outcomes} == {
        "tests/test_a.py::test_repaired": "passed",
        "tests/test_b.py::test_prior": "passed",
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

    def counted_size(_path: Path) -> int:
        nonlocal calls
        calls += 1
        return calls

    monkeypatch.setattr("devtools.verify_runs._dir_size_kb", counted_size)
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
    assert second["basetemp_size_kb"] == 1
    assert calls == 1


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


def test_explicit_tmpfs_root_reroutes_to_scratch_when_its_cap_is_too_small(
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

    cleaned = cleanup_managed_pytest_basetemp(root=tmp_path, run_id="run-1", env=env)

    assert cleaned == basetemp
    assert not basetemp.exists()


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


def test_verify_stops_after_failed_heavy_step(capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[str] = []

    def fake_run(label: str, command: list[str], **kwargs: object) -> tuple[int, float, dict[str, object]]:
        calls.append(label)
        return (1 if label.startswith("pytest") else 0), 0.01, {}

    with (
        patch("devtools.verify._run", side_effect=fake_run),
        patch("devtools.verify._git_head", return_value="head"),
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
        patch("devtools.verify._save_history"),
        patch("devtools.verify._stamp_head"),
        patch("devtools.verify._notify"),
    ):
        assert main([*argv, "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["verification_scope"] == expected_scope
    assert payload["release_baseline_allowed"] is expected_permission
    assert payload["terminal_authorization"] == ("narrow-terminal" if expected_permission else None)


def test_verify_refuses_unbudgeted_pytest_before_running_steps(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch("devtools.verify.build_verify_steps", side_effect=PytestResourceError("only 0.50 GiB available")),
        patch("devtools.verify._git_head", return_value="head"),
        patch("devtools.verify._testmon_preflight", return_value=None),
        patch("devtools.verify._run") as run,
    ):
        rc = main(["--json"])

    assert rc == 125
    run.assert_not_called()
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
    def fake_run(label: str, command: list[str], **kwargs: object) -> tuple[int, float, dict[str, object]]:
        del command, kwargs
        return 0, 0.01, ({"selected_count": 0} if label.startswith("pytest") else {})

    with (
        patch("devtools.verify._run", side_effect=fake_run),
        patch("devtools.verify._git_head", return_value="head"),
        patch("devtools.verify._save_history"),
        patch("devtools.verify._stamp_head"),
        patch("devtools.verify._notify"),
        patch("devtools.verify._testmon_preflight", return_value=None),
        patch("devtools.verify._changed_executable_paths", return_value=("polylogue/example.py",)),
        patch("devtools.verify._matching_testmon_coverage", return_value=None),
    ):
        rc = main(["--json"])

    assert rc == 5
    payload = json.loads(capsys.readouterr().out)
    pytest_step = next(step for step in payload["steps"] if step["name"].startswith("pytest"))
    assert pytest_step["diagnosis"] == "zero_testmon_selection_for_executable_change"
    assert pytest_step["zero_selection_changed_paths"] == ["polylogue/example.py"]


def test_verify_accepts_zero_testmon_selection_after_matching_coverage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run(label: str, command: list[str], **kwargs: object) -> tuple[int, float, dict[str, object]]:
        del command, kwargs
        return 0, 0.01, ({"selected_count": 0} if label.startswith("pytest") else {})

    with (
        patch("devtools.verify._run", side_effect=fake_run),
        patch("devtools.verify._git_head", return_value="head"),
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
    with patch("devtools.verify._worktree_fingerprint", return_value="affected"):
        _record_testmon_affected_coverage(
            executable_paths=paths,
            selected_count=3,
            run_id="run-1",
        )
        assert TESTMON_AFFECTED_STAMP.exists()
        assert _matching_testmon_coverage(paths) == "successful_affected_run"
        assert _matching_testmon_coverage(("polylogue/other.py",)) is None

    with patch("devtools.verify._worktree_fingerprint", return_value="changed"):
        assert _matching_testmon_coverage(paths) is None

    TESTMON_AFFECTED_STAMP.write_text(json.dumps({"identity": {"worktree_fingerprint": "affected"}}))
    with patch("devtools.verify._worktree_fingerprint", return_value="affected"):
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
