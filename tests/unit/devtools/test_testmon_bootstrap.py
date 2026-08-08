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

import json
import sqlite3
from pathlib import Path

import pytest

import devtools.testmon_bootstrap as testmon_bootstrap
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
        _TestmonIdentity("head", "tree", "python", True, False),
        _TestmonBinding(BindingMode.EXACT, str(path.parent.parent.parent.resolve())),
        file_fingerprint(data),
        "seed",
        ".cache/verify/runs/seed",
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
    local_stamp.parent.mkdir(parents=True, exist_ok=True)
    local_stamp.write_text(json.dumps({"protocol_version": PROTOCOL_VERSION, "status": "usable"}))
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
                "status": "incomplete",
                "identity": {
                    "git_head": "head",
                    "worktree_fingerprint": "tree",
                    "python": "python",
                    "skip_slow": True,
                    "lab": False,
                },
                "selection": {"selected_count": 2, "selected_nodeids_omitted": 0},
                "expected_nodeids": ["tests/test.py::test_passed", "tests/test.py::test_failed"],
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
    assert bootstrap_testmon_seed_files(
        decision,
        local_testmon_data=local_data,
        local_seed_stamp=local_stamp,
        checkout_root=tmp_path / "lane",
        inherited_from=tmp_path / "main",
    )
    assert json.loads(local_stamp.read_text())["baseline"]["status"] == "red"


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
    main_data = tmp_path / "main" / "testmondata"
    main_stamp = tmp_path / "main" / "seed.json"
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
    bootstrap_testmon_seed_files(decision, local_testmon_data=local_data, local_seed_stamp=local_stamp)

    local_payload = json.loads(local_stamp.read_text())
    source_payload = json.loads(main_stamp.read_text())
    assert {key: local_payload[key] for key in source_payload if key != "testmon_data"} == {
        key: source_payload[key] for key in source_payload if key != "testmon_data"
    }
    conn = sqlite3.connect(local_data)
    try:
        rows = conn.execute("SELECT filename, fsha FROM file_fp ORDER BY filename").fetchall()
    finally:
        conn.close()
    assert rows == [("x", "sha-x"), ("y", "sha-y"), ("z", "sha-z")]
    # No temp files left behind.
    assert sorted(p.name for p in local_data.parent.iterdir()) == ["seed.json", "testmondata"]


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
