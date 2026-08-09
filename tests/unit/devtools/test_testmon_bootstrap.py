"""Tests for the worktree testmon-seed bootstrap (devtools/testmon_bootstrap.py).

Covers polylogue-mq4vx: a fresh agent worktree lane starts with no local
`.cache/testmon/testmondata`, either paying the full `--seed-testmon` cost
again or hitting the unseeded-refusal preflight in `devtools/verify.py`. The
main checkout's testmondata is copyable (file_fp entries are relative paths
with per-file checksums, so a stale copy self-invalidates changed files), so
`maybe_bootstrap_testmon_seed` copies it in before that preflight runs.

These tests target `decide_testmon_bootstrap` (the pure decision) and
`bootstrap_testmon_seed_files` (the copy action) directly with tmp dirs --
not the full `devtools verify` pipeline, per the bootstrap's own module
docstring contract.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

import devtools.checkout_guard as checkout_guard
import devtools.testmon_bootstrap as testmon_bootstrap
import devtools.verify as verify
from devtools.testmon_bootstrap import (
    BootstrapDecision,
    bootstrap_testmon_seed_files,
    decide_testmon_bootstrap,
)
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

PROTOCOL_VERSION = 4


def _write_valid_seed_stamp(path: Path, *, protocol_version: int = PROTOCOL_VERSION) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = path.parent / "testmondata"
    if not data.exists():
        _write_sqlite_db(data)
    with sqlite3.connect(data) as conn:
        nodeids = tuple(row[0] for row in conn.execute("select test_name from test_execution"))
    graph = GraphInspection(GraphStatus.COMPLETE, len(nodeids), len(nodeids), (), 0, 0, None, ())
    stamp = _TestmonSeedStamp(
        protocol_version,
        CollectionStatus.COMPLETE,
        nodeids,
        0,
        BaselineStatus.GREEN,
        True,
        0,
        graph,
        _TestmonIdentity("head", "tree", "python", True, False, None, "narrow-terminal"),
        _TestmonBinding(BindingMode.EXACT, str(path.parent.resolve())),
        file_fingerprint(data),
        "seed",
        ".cache/verify/runs/seed",
    )
    artifact_dir = path.parent / ".cache" / "verify" / "runs" / "seed"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": "seed",
                "checkout_root": str(path.parent.resolve()),
                "artifact_dir": ".cache/verify/runs/seed",
            }
        )
    )
    path.write_text(json.dumps(stamp.as_dict()))


def _write_sqlite_db(path: Path, *, rows: tuple[str, ...] = ("a", "b")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE environment (id INTEGER PRIMARY KEY, environment_name TEXT)")
        conn.execute("CREATE TABLE file_fp (id INTEGER PRIMARY KEY, filename TEXT, fsha TEXT)")
        conn.execute(
            "CREATE TABLE test_execution (id INTEGER PRIMARY KEY, environment_id INTEGER, test_name TEXT, failed INTEGER)"
        )
        conn.execute("CREATE TABLE test_execution_file_fp (test_execution_id INTEGER, fingerprint_id INTEGER)")
        conn.executemany("INSERT INTO file_fp(filename, fsha) VALUES (?, ?)", [(row, f"sha-{row}") for row in rows])
        conn.executemany("INSERT INTO test_execution(test_name, failed) VALUES (?, 0)", [(row,) for row in rows])
        conn.executemany(
            "INSERT INTO test_execution_file_fp VALUES (?, ?)",
            [(index, index) for index, _row in enumerate(rows, start=1)],
        )
        conn.commit()
    finally:
        conn.close()


def _red_attempt_decision(tmp_path: Path) -> tuple[BootstrapDecision, Path, Path, Path, Path]:
    main_root = tmp_path / "main"
    main_data = main_root / "testmondata"
    _write_sqlite_db(main_data, rows=("tests/test.py::test_passed", "tests/test.py::test_failed"))
    attempt = main_root / "seed-attempt.json"
    attempt.write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "status": "reusable",
                "identity": {
                    "git_head": "head",
                    "worktree_fingerprint": "tree",
                    "python": "python",
                    "skip_slow": True,
                    "lab": False,
                },
                "selection": {"selected_count": 2, "selected_nodeids_omitted": 0},
                "expected_nodeids": ["tests/test.py::test_passed", "tests/test.py::test_failed"],
                "expected_count": 2,
                "expected_digest": hashlib.sha256(
                    "\n".join(sorted(["tests/test.py::test_passed", "tests/test.py::test_failed"])).encode()
                ).hexdigest(),
                "testmon_data": file_fingerprint(main_data),
                "node_outcomes": [
                    {"nodeid": "tests/test.py::test_passed", "outcome": "passed"},
                    {"nodeid": "tests/test.py::test_failed", "outcome": "failed"},
                ],
                "exit_code": 1,
                "run_id": "red-run",
                "artifact_dir": ".cache/verify/runs/red-run",
            }
        )
    )
    artifact = main_root / ".cache" / "verify" / "runs" / "red-run"
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "run.json").write_text(
        json.dumps(
            {
                "run_id": "red-run",
                "checkout_root": str(main_root.resolve()),
                "artifact_dir": ".cache/verify/runs/red-run",
            }
        )
    )
    lane = tmp_path / "lane"
    decision = decide_testmon_bootstrap(
        is_linked_worktree=True,
        local_testmon_data=lane / "testmondata",
        local_seed_stamp=lane / "seed.json",
        local_seed_attempt=lane / "seed-attempt.json",
        main_testmon_data=main_data,
        main_seed_stamp=main_root / "seed.json",
        main_seed_attempt=attempt,
        protocol_version=PROTOCOL_VERSION,
        main_checkout_root=main_root,
        local_checkout_root=lane,
    )
    return decision, lane / "testmondata", lane / "seed.json", lane / "seed-attempt.json", lane


def test_not_a_linked_worktree_never_bootstraps(tmp_path: Path) -> None:
    """The main checkout itself must never "bootstrap from itself"."""
    decision = decide_testmon_bootstrap(
        is_linked_worktree=False,
        local_testmon_data=tmp_path / "local" / "testmondata",
        local_seed_stamp=tmp_path / "local" / "seed.json",
        main_testmon_data=tmp_path / "main" / "testmondata",
        main_seed_stamp=tmp_path / "main" / "seed.json",
        protocol_version=PROTOCOL_VERSION,
    )
    assert decision == BootstrapDecision(False, decision.reason)
    assert not decision.should_bootstrap


def test_local_seed_already_present_skips_bootstrap(tmp_path: Path) -> None:
    """A worktree that already seeded itself must not be clobbered by main's copy."""
    local_data = tmp_path / "local" / "testmondata"
    local_stamp = tmp_path / "local" / "seed.json"
    _write_sqlite_db(local_data)
    _write_valid_seed_stamp(local_stamp)
    main_data = tmp_path / "main" / "testmondata"
    main_stamp = tmp_path / "main" / "seed.json"
    _write_sqlite_db(main_data)
    _write_valid_seed_stamp(main_stamp)

    decision = decide_testmon_bootstrap(
        is_linked_worktree=True,
        local_testmon_data=local_data,
        local_seed_stamp=local_stamp,
        main_testmon_data=main_data,
        main_seed_stamp=main_stamp,
        protocol_version=PROTOCOL_VERSION,
    )
    assert not decision.should_bootstrap
    assert "already has" in decision.reason


def test_invalid_local_seed_does_not_block_valid_main_bootstrap(tmp_path: Path) -> None:
    local_data = tmp_path / "local" / "testmondata"
    local_stamp = tmp_path / "local" / "seed.json"
    _write_sqlite_db(local_data)
    _write_valid_seed_stamp(local_stamp)
    local_data.write_bytes(local_data.read_bytes() + b"stale")
    main_data = tmp_path / "main" / "testmondata"
    main_stamp = tmp_path / "main" / "seed.json"
    _write_sqlite_db(main_data)
    _write_valid_seed_stamp(main_stamp)

    decision = decide_testmon_bootstrap(
        is_linked_worktree=True,
        local_testmon_data=local_data,
        local_seed_stamp=local_stamp,
        main_testmon_data=main_data,
        main_seed_stamp=main_stamp,
        protocol_version=PROTOCOL_VERSION,
    )

    assert decision.should_bootstrap


def test_main_seed_absent_skips_bootstrap(tmp_path: Path) -> None:
    decision = decide_testmon_bootstrap(
        is_linked_worktree=True,
        local_testmon_data=tmp_path / "local" / "testmondata",
        local_seed_stamp=tmp_path / "local" / "seed.json",
        main_testmon_data=tmp_path / "main" / "testmondata",
        main_seed_stamp=tmp_path / "main" / "seed.json",
        protocol_version=PROTOCOL_VERSION,
    )
    assert not decision.should_bootstrap
    assert "testmondata file is missing" in decision.reason


def test_main_seed_stamp_wrong_protocol_version_skips_bootstrap(tmp_path: Path) -> None:
    main_stamp = tmp_path / "main" / "seed.json"
    _write_valid_seed_stamp(main_stamp, protocol_version=PROTOCOL_VERSION + 1)

    decision = decide_testmon_bootstrap(
        is_linked_worktree=True,
        local_testmon_data=tmp_path / "local" / "testmondata",
        local_seed_stamp=tmp_path / "local" / "seed.json",
        main_testmon_data=tmp_path / "main" / "testmondata",
        main_seed_stamp=main_stamp,
        protocol_version=PROTOCOL_VERSION,
    )
    assert not decision.should_bootstrap
    assert "stale" in decision.reason or "no validated" in decision.reason


def test_main_seed_stamp_incomplete_status_skips_bootstrap(tmp_path: Path) -> None:
    main_stamp = tmp_path / "main" / "seed.json"
    main_stamp.parent.mkdir(parents=True, exist_ok=True)
    main_stamp.write_text(json.dumps({"protocol_version": PROTOCOL_VERSION, "status": "incomplete"}))
    _write_sqlite_db(tmp_path / "main" / "testmondata")

    decision = decide_testmon_bootstrap(
        is_linked_worktree=True,
        local_testmon_data=tmp_path / "local" / "testmondata",
        local_seed_stamp=tmp_path / "local" / "seed.json",
        main_testmon_data=tmp_path / "main" / "testmondata",
        main_seed_stamp=main_stamp,
        protocol_version=PROTOCOL_VERSION,
    )
    assert not decision.should_bootstrap


def test_main_seed_stamp_unreadable_json_skips_bootstrap(tmp_path: Path) -> None:
    main_stamp = tmp_path / "main" / "seed.json"
    main_stamp.parent.mkdir(parents=True, exist_ok=True)
    main_stamp.write_text("{not valid json")
    _write_sqlite_db(tmp_path / "main" / "testmondata")

    decision = decide_testmon_bootstrap(
        is_linked_worktree=True,
        local_testmon_data=tmp_path / "local" / "testmondata",
        local_seed_stamp=tmp_path / "local" / "seed.json",
        main_testmon_data=tmp_path / "main" / "testmondata",
        main_seed_stamp=main_stamp,
        protocol_version=PROTOCOL_VERSION,
    )
    assert not decision.should_bootstrap


def test_valid_seed_stamp_but_missing_testmondata_skips_bootstrap(tmp_path: Path) -> None:
    """A seed stamp claims completeness but the db file itself vanished -- don't copy nothing."""
    main_stamp = tmp_path / "main" / "seed.json"
    main_stamp.parent.mkdir(parents=True, exist_ok=True)
    main_stamp.write_text(json.dumps({"protocol_version": PROTOCOL_VERSION, "status": "usable"}))

    decision = decide_testmon_bootstrap(
        is_linked_worktree=True,
        local_testmon_data=tmp_path / "local" / "testmondata",
        local_seed_stamp=tmp_path / "local" / "seed.json",
        main_testmon_data=tmp_path / "main" / "testmondata",
        main_seed_stamp=main_stamp,
        protocol_version=PROTOCOL_VERSION,
    )
    assert not decision.should_bootstrap
    assert "testmondata file is missing" in decision.reason


def test_valid_main_seed_and_empty_local_bootstraps(tmp_path: Path) -> None:
    main_data = tmp_path / "main" / "testmondata"
    main_stamp = tmp_path / "main" / "seed.json"
    _write_sqlite_db(main_data)
    _write_valid_seed_stamp(main_stamp)

    decision = decide_testmon_bootstrap(
        is_linked_worktree=True,
        local_testmon_data=tmp_path / "local" / "testmondata",
        local_seed_stamp=tmp_path / "local" / "seed.json",
        main_testmon_data=main_data,
        main_seed_stamp=main_stamp,
        protocol_version=PROTOCOL_VERSION,
    )
    assert decision.should_bootstrap
    assert decision.main_testmon_data == main_data
    assert decision.main_seed_stamp == main_stamp


def test_complete_red_attempt_bootstraps_as_selection_only_state(tmp_path: Path) -> None:
    main_data = tmp_path / "main" / "testmondata"
    _write_sqlite_db(main_data, rows=("tests/test.py::test_passed", "tests/test.py::test_failed"))
    attempt = tmp_path / "main" / "seed-attempt.json"
    attempt.write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "status": "reusable",
                "identity": {
                    "git_head": "head",
                    "worktree_fingerprint": "tree",
                    "python": "python",
                    "skip_slow": True,
                    "lab": False,
                },
                "selection": {"selected_count": 2, "selected_nodeids_omitted": 0},
                "expected_nodeids": ["tests/test.py::test_passed", "tests/test.py::test_failed"],
                "expected_count": 2,
                "expected_digest": hashlib.sha256(
                    "\n".join(sorted(["tests/test.py::test_passed", "tests/test.py::test_failed"])).encode()
                ).hexdigest(),
                "testmon_data": file_fingerprint(main_data),
                "node_outcomes": [
                    {"nodeid": "tests/test.py::test_passed", "outcome": "passed"},
                    {"nodeid": "tests/test.py::test_failed", "outcome": "failed"},
                ],
                "exit_code": 1,
                "run_id": "red-run",
                "artifact_dir": ".cache/verify/runs/red-run",
            }
        )
    )
    red_artifact = tmp_path / "main" / ".cache" / "verify" / "runs" / "red-run"
    red_artifact.mkdir(parents=True, exist_ok=True)
    (red_artifact / "run.json").write_text(
        json.dumps(
            {
                "run_id": "red-run",
                "checkout_root": str((tmp_path / "main").resolve()),
                "artifact_dir": ".cache/verify/runs/red-run",
            }
        )
    )
    decision = decide_testmon_bootstrap(
        is_linked_worktree=True,
        local_testmon_data=tmp_path / "lane" / "testmondata",
        local_seed_stamp=tmp_path / "lane" / "seed.json",
        main_testmon_data=main_data,
        main_seed_stamp=tmp_path / "main" / "seed.json",
        main_seed_attempt=attempt,
        protocol_version=PROTOCOL_VERSION,
    )

    assert decision.should_bootstrap
    assert decision.main_seed_attempt == attempt
    local_data = tmp_path / "lane" / "testmondata"
    local_stamp = tmp_path / "lane" / "seed.json"
    local_attempt = tmp_path / "lane" / "seed-attempt.json"
    assert bootstrap_testmon_seed_files(
        decision,
        local_testmon_data=local_data,
        local_seed_stamp=local_stamp,
        local_seed_attempt=local_attempt,
        checkout_root=tmp_path / "lane",
        inherited_from=tmp_path / "main",
    )
    assert not local_stamp.exists()
    rebound_attempt = json.loads(local_attempt.read_text())
    assert rebound_attempt["artifact_dir"] == ".cache/verify/runs/red-run"
    assert rebound_attempt["testmon_data"] == file_fingerprint(local_data)
    rebound_receipt = json.loads(
        (tmp_path / "lane" / ".cache" / "verify" / "runs" / "red-run" / "run.json").read_text()
    )
    assert rebound_receipt["run_id"] == "red-run"
    assert rebound_receipt["checkout_root"] == str((tmp_path / "lane").resolve())
    current_run = json.loads((tmp_path / "lane" / ".cache" / "verify" / "current-run.json").read_text())
    assert current_run["run_id"] == "red-run"
    assert current_run["checkout_root"] == str((tmp_path / "lane").resolve())

    rebound_decision = decide_testmon_bootstrap(
        is_linked_worktree=True,
        local_testmon_data=local_data,
        local_seed_stamp=local_stamp,
        local_seed_attempt=local_attempt,
        main_testmon_data=main_data,
        main_seed_stamp=tmp_path / "main" / "seed.json",
        main_seed_attempt=attempt,
        protocol_version=PROTOCOL_VERSION,
        main_checkout_root=tmp_path / "main",
        local_checkout_root=tmp_path / "lane",
    )
    assert not rebound_decision.should_bootstrap
    assert "checkout-bound selection attempt" in rebound_decision.reason


def test_complete_typed_markerless_green_attempt_bootstraps_only_as_selection_state(tmp_path: Path) -> None:
    main_root = tmp_path / "main"
    main_data = main_root / "testmondata"
    _write_sqlite_db(main_data, rows=("tests/test.py::test_passed",))
    attempt = main_root / "seed-attempt.json"
    attempt.write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "status": "complete",
                "identity": {
                    "git_head": "head",
                    "worktree_fingerprint": "tree",
                    "python": "python",
                    "skip_slow": False,
                    "lab": False,
                    "terminal_authorization": None,
                },
                "selection": {"selected_count": 1, "selected_nodeids_omitted": 0},
                "expected_nodeids": ["tests/test.py::test_passed"],
                "expected_count": 1,
                "expected_digest": hashlib.sha256(b"tests/test.py::test_passed").hexdigest(),
                "node_outcomes": [{"nodeid": "tests/test.py::test_passed", "outcome": "passed"}],
                "exit_code": 0,
                "verification_scope": "release-baseline",
                "release_baseline_allowed": True,
                "run_id": "green-run",
                "artifact_dir": ".cache/verify/runs/green-run",
                "testmon_data": file_fingerprint(main_data),
            }
        )
    )
    artifact = main_root / ".cache" / "verify" / "runs" / "green-run"
    artifact.mkdir(parents=True)
    (artifact / "run.json").write_text(
        json.dumps(
            {
                "run_id": "green-run",
                "checkout_root": str(main_root.resolve()),
                "artifact_dir": ".cache/verify/runs/green-run",
            }
        )
    )

    decision = decide_testmon_bootstrap(
        is_linked_worktree=True,
        local_testmon_data=tmp_path / "lane" / "testmondata",
        local_seed_stamp=tmp_path / "lane" / "seed.json",
        main_testmon_data=main_data,
        main_seed_stamp=main_root / "seed.json",
        main_seed_attempt=attempt,
        protocol_version=PROTOCOL_VERSION,
    )

    assert decision.should_bootstrap
    assert decision.selection_only
    assert bootstrap_testmon_seed_files(
        decision,
        local_testmon_data=tmp_path / "lane" / "testmondata",
        local_seed_stamp=tmp_path / "lane" / "seed.json",
        local_seed_attempt=tmp_path / "lane" / "seed-attempt.json",
        checkout_root=tmp_path / "lane",
        inherited_from=main_root,
    )
    assert not (tmp_path / "lane" / "seed.json").exists()


def test_markerless_complete_bootstrap_passes_guard_and_verify_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main_root = tmp_path / "main"
    main_data = main_root / "testmondata"
    nodeid = "tests/test.py::test_passed"
    _write_sqlite_db(main_data, rows=(nodeid,))
    attempt = main_root / "seed-attempt.json"
    attempt.write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "status": "complete",
                "identity": {
                    "git_head": "head",
                    "worktree_fingerprint": "tree",
                    "python": "python",
                    "skip_slow": False,
                    "lab": False,
                    "terminal_authorization": None,
                },
                "selection": {"selected_count": 1, "selected_nodeids_omitted": 0},
                "expected_nodeids": [nodeid],
                "expected_count": 1,
                "expected_digest": hashlib.sha256(nodeid.encode()).hexdigest(),
                "node_outcomes": [{"nodeid": nodeid, "outcome": "passed"}],
                "exit_code": 0,
                "verification_scope": "release-baseline",
                "release_baseline_allowed": True,
                "run_id": "green-run",
                "artifact_dir": ".cache/verify/runs/green-run",
                "testmon_data": file_fingerprint(main_data),
            }
        )
    )
    artifact = main_root / ".cache" / "verify" / "runs" / "green-run"
    artifact.mkdir(parents=True)
    (artifact / "run.json").write_text(
        json.dumps(
            {
                "run_id": "green-run",
                "checkout_root": str(main_root.resolve()),
                "artifact_dir": ".cache/verify/runs/green-run",
            }
        )
    )

    lane = tmp_path / "lane"
    lane.mkdir()
    (lane / ".git").write_text("gitdir: /main/.git/worktrees/lane\n")
    (lane / ".venv" / "bin").mkdir(parents=True)
    package = lane / "polylogue"
    package.mkdir()
    (package / "__init__.py").write_text("")
    local_data = lane / ".cache" / "testmon" / "testmondata"
    local_stamp = lane / ".cache" / "testmon" / "seed.json"
    local_attempt = lane / ".cache" / "testmon" / "seed-attempt.json"
    decision = decide_testmon_bootstrap(
        is_linked_worktree=True,
        local_testmon_data=local_data,
        local_seed_stamp=local_stamp,
        main_testmon_data=main_data,
        main_seed_stamp=main_root / "seed.json",
        main_seed_attempt=attempt,
        protocol_version=PROTOCOL_VERSION,
    )
    assert decision.selection_only
    assert bootstrap_testmon_seed_files(
        decision,
        local_testmon_data=local_data,
        local_seed_stamp=local_stamp,
        local_seed_attempt=local_attempt,
        checkout_root=lane,
        inherited_from=main_root,
    )

    monkeypatch.setattr(checkout_guard, "_is_linked_worktree", lambda _root: True)
    fingerprint = checkout_guard.checkout_environment_fingerprint(
        lane,
        polylogue_import_path=package / "__init__.py",
        python_executable=lane / ".venv" / "bin" / "python",
    )
    assert fingerprint.clean
    monkeypatch.setattr(verify, "ROOT", lane)
    monkeypatch.setattr(verify, "TESTMON_DATA", local_data)
    monkeypatch.setattr(verify, "TESTMON_SEED_STAMP", local_stamp)
    monkeypatch.setattr(verify, "TESTMON_SEED_ATTEMPT", local_attempt)
    assert verify._testmon_preflight(seed_testmon=False, full_pytest=False, quick=False, commit=False) is None
    assert json.loads(local_attempt.read_text())["release_baseline_allowed"] is False


def test_local_seed_missing_only_stamp_still_bootstraps(tmp_path: Path) -> None:
    """Partial local state (e.g. a stale stamp with no db, or vice versa) still needs a fresh copy."""
    local_stamp = tmp_path / "local" / "seed.json"
    local_stamp.parent.mkdir(parents=True, exist_ok=True)
    local_stamp.write_text(json.dumps({"protocol_version": PROTOCOL_VERSION, "status": "usable"}))
    main_data = tmp_path / "main" / "testmondata"
    main_stamp = tmp_path / "main" / "seed.json"
    _write_sqlite_db(main_data)
    _write_valid_seed_stamp(main_stamp)

    decision = decide_testmon_bootstrap(
        is_linked_worktree=True,
        local_testmon_data=tmp_path / "local" / "testmondata",
        local_seed_stamp=local_stamp,
        main_testmon_data=main_data,
        main_seed_stamp=main_stamp,
        protocol_version=PROTOCOL_VERSION,
    )
    assert decision.should_bootstrap


def test_bootstrap_seed_files_copies_db_and_stamp(tmp_path: Path) -> None:
    main_data = tmp_path / "main?fragment#1" / "testmondata"
    main_stamp = tmp_path / "main?fragment#1" / "seed.json"
    _write_sqlite_db(main_data, rows=("x", "y", "z"))
    _write_valid_seed_stamp(main_stamp)
    local_data = tmp_path / "local" / "testmondata"
    local_stamp = tmp_path / "local" / "seed.json"

    decision = BootstrapDecision(
        True,
        "test",
        main_testmon_data=main_data,
        main_seed_stamp=main_stamp,
    )
    assert bootstrap_testmon_seed_files(
        decision,
        local_testmon_data=local_data,
        local_seed_stamp=local_stamp,
        checkout_root=tmp_path / "local",
        inherited_from=tmp_path / "main?fragment#1",
    )

    local_payload = json.loads(local_stamp.read_text())
    source_payload = json.loads(main_stamp.read_text())
    comparable_keys = set(source_payload) - {"binding", "testmon_data"}
    assert {key: local_payload[key] for key in comparable_keys} == {key: source_payload[key] for key in comparable_keys}
    assert local_payload["binding"]["checkout_root"] == str(tmp_path / "local")
    assert local_payload["binding"]["source_checkout_root"] == str(tmp_path / "main?fragment#1")
    conn = sqlite3.connect(local_data)
    try:
        rows = conn.execute("SELECT filename, fsha FROM file_fp ORDER BY filename").fetchall()
    finally:
        conn.close()
    assert rows == [("x", "sha-x"), ("y", "sha-y"), ("z", "sha-z")]
    # No temp files left behind.
    assert sorted(p.name for p in local_data.parent.iterdir()) == [".cache", "seed.json", "testmondata"]


def test_bootstrap_seed_files_marks_destination_and_source_checkout(tmp_path: Path) -> None:
    main_data = tmp_path / "main" / "testmondata"
    main_stamp = tmp_path / "main" / "seed.json"
    _write_sqlite_db(main_data)
    _write_valid_seed_stamp(main_stamp)
    local_data = tmp_path / "lane" / "testmondata"
    local_stamp = tmp_path / "lane" / "seed.json"

    bootstrap_testmon_seed_files(
        BootstrapDecision(True, "test", main_testmon_data=main_data, main_seed_stamp=main_stamp),
        local_testmon_data=local_data,
        local_seed_stamp=local_stamp,
        checkout_root=tmp_path / "lane",
        inherited_from=tmp_path / "main",
    )

    payload = json.loads(local_stamp.read_text())
    assert payload["binding"]["checkout_root"] == str((tmp_path / "lane").resolve())
    assert payload["binding"]["source_checkout_root"] == str((tmp_path / "main").resolve())
    source = json.loads(main_stamp.read_text())
    assert {key: payload[key] for key in source if key not in {"binding", "testmon_data"}} == {
        key: source[key] for key in source if key not in {"binding", "testmon_data"}
    }


def test_bootstrap_seed_files_rejects_paths_outside_or_colliding_with_destination(tmp_path: Path) -> None:
    main_data = tmp_path / "main" / "testmondata"
    main_stamp = tmp_path / "main" / "seed.json"
    _write_sqlite_db(main_data)
    _write_valid_seed_stamp(main_stamp)
    decision = BootstrapDecision(True, "test", main_testmon_data=main_data, main_seed_stamp=main_stamp)
    local_data = tmp_path / "lane" / "testmondata"

    assert not bootstrap_testmon_seed_files(
        decision,
        local_testmon_data=local_data,
        local_seed_stamp=tmp_path / "outside" / "seed.json",
        checkout_root=tmp_path / "lane",
        inherited_from=tmp_path / "main",
    )
    assert not (tmp_path / "outside" / "seed.json").exists()
    assert not local_data.exists()

    assert not bootstrap_testmon_seed_files(
        decision,
        local_testmon_data=local_data,
        local_seed_stamp=local_data,
        checkout_root=tmp_path / "lane",
        inherited_from=tmp_path / "main",
    )
    assert not local_data.exists()


def test_bootstrap_seed_files_keeps_copied_state_when_stamp_turns_invalid(tmp_path: Path) -> None:
    main_data = tmp_path / "main" / "testmondata"
    main_stamp = tmp_path / "main" / "seed.json"
    _write_sqlite_db(main_data)
    main_stamp.parent.mkdir(parents=True, exist_ok=True)
    main_stamp.write_text("{concurrent rewrite")
    local_data = tmp_path / "lane" / "testmondata"
    local_stamp = tmp_path / "lane" / "seed.json"

    stamped = bootstrap_testmon_seed_files(
        BootstrapDecision(True, "test", main_testmon_data=main_data, main_seed_stamp=main_stamp),
        local_testmon_data=local_data,
        local_seed_stamp=local_stamp,
        checkout_root=tmp_path / "lane",
        inherited_from=tmp_path / "main",
    )

    assert stamped is False
    assert not local_data.exists()
    assert not local_stamp.exists()


def test_bootstrap_graph_mismatch_publishes_no_destination_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main_data = tmp_path / "main" / "testmondata"
    main_stamp = tmp_path / "main" / "seed.json"
    _write_sqlite_db(main_data)
    _write_valid_seed_stamp(main_stamp)
    local_data = tmp_path / "lane" / "testmondata"
    local_stamp = tmp_path / "lane" / "seed.json"
    monkeypatch.setattr(testmon_bootstrap, "refresh_stamp", lambda *_args, **_kwargs: None)

    assert not bootstrap_testmon_seed_files(
        BootstrapDecision(True, "test", main_testmon_data=main_data, main_seed_stamp=main_stamp),
        local_testmon_data=local_data,
        local_seed_stamp=local_stamp,
        checkout_root=tmp_path / "lane",
        inherited_from=tmp_path / "main",
    )
    assert not local_data.exists()
    assert not local_stamp.exists()
    assert not (tmp_path / "lane" / ".cache" / "verify" / "current-run.json").exists()


def test_bootstrap_receipt_rebind_failure_publishes_no_destination_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main_data = tmp_path / "main" / "testmondata"
    main_stamp = tmp_path / "main" / "seed.json"
    _write_sqlite_db(main_data)
    _write_valid_seed_stamp(main_stamp)
    local_data = tmp_path / "lane" / "testmondata"
    local_stamp = tmp_path / "lane" / "seed.json"
    monkeypatch.setattr(testmon_bootstrap, "_rebind_run_receipt", lambda **_kwargs: False)

    assert not bootstrap_testmon_seed_files(
        BootstrapDecision(True, "test", main_testmon_data=main_data, main_seed_stamp=main_stamp),
        local_testmon_data=local_data,
        local_seed_stamp=local_stamp,
        checkout_root=tmp_path / "lane",
        inherited_from=tmp_path / "main",
    )
    assert not local_data.exists()
    assert not local_stamp.exists()
    assert not (tmp_path / "lane" / ".cache" / "verify" / "current-run.json").exists()


def test_bootstrap_rebound_attempt_failure_publishes_no_destination_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision, local_data, local_stamp, local_attempt, lane = _red_attempt_decision(tmp_path)
    original_stamp_from_attempt = cast(Callable[..., object], testmon_bootstrap.__dict__["stamp_from_attempt"])
    calls = 0

    def fail_rebound_attempt(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            return None
        return original_stamp_from_attempt(*args, **kwargs)

    monkeypatch.setattr(testmon_bootstrap, "stamp_from_attempt", fail_rebound_attempt)

    assert not bootstrap_testmon_seed_files(
        decision,
        local_testmon_data=local_data,
        local_seed_stamp=local_stamp,
        local_seed_attempt=local_attempt,
        checkout_root=lane,
        inherited_from=tmp_path / "main",
    )
    assert not local_data.exists()
    assert not local_stamp.exists()
    assert not local_attempt.exists()
    assert not (lane / ".cache" / "verify" / "current-run.json").exists()


def test_maybe_bootstrap_does_not_migrate_an_untyped_legacy_local_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lane = tmp_path / "lane"
    main = tmp_path / "main"
    local_data = lane / "cache" / "testmondata"
    local_stamp = lane / "cache" / "seed.json"
    _write_sqlite_db(local_data)
    local_stamp.parent.mkdir(parents=True, exist_ok=True)
    local_stamp.write_text(json.dumps({"protocol_version": PROTOCOL_VERSION, "status": "complete"}))
    monkeypatch.setattr(testmon_bootstrap, "_git_worktree_info", lambda _root: (True, main))

    message = testmon_bootstrap.maybe_bootstrap_testmon_seed(
        lane,
        testmon_data_relpath="cache/testmondata",
        seed_stamp_relpath="cache/seed.json",
        protocol_version=PROTOCOL_VERSION,
    )

    assert message is None
    assert json.loads(local_stamp.read_text())["status"] == "complete"


def test_bootstrap_seed_files_noop_when_decision_says_no(tmp_path: Path) -> None:
    local_data = tmp_path / "local" / "testmondata"
    local_stamp = tmp_path / "local" / "seed.json"
    decision = BootstrapDecision(False, "not needed")

    bootstrap_testmon_seed_files(decision, local_testmon_data=local_data, local_seed_stamp=local_stamp)

    assert not local_data.exists()
    assert not local_stamp.exists()
