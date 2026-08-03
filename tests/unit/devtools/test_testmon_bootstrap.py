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

from devtools.testmon_bootstrap import (
    BootstrapDecision,
    bootstrap_testmon_seed_files,
    decide_testmon_bootstrap,
)

PROTOCOL_VERSION = 3


def _write_valid_seed_stamp(path: Path, *, protocol_version: int = PROTOCOL_VERSION) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"protocol_version": protocol_version, "status": "complete"}))


def _write_sqlite_db(path: Path, *, rows: tuple[str, ...] = ("a", "b")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE file_fp (path TEXT, fsha TEXT)")
        conn.executemany("INSERT INTO file_fp VALUES (?, ?)", [(row, f"sha-{row}") for row in rows])
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
    assert "no valid complete testmon seed stamp" in decision.reason


def test_main_seed_stamp_wrong_protocol_version_skips_bootstrap(tmp_path: Path) -> None:
    main_stamp = tmp_path / "main" / "seed.json"
    _write_valid_seed_stamp(main_stamp, protocol_version=PROTOCOL_VERSION + 1)
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
    assert "no valid complete testmon seed stamp" in decision.reason


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
    _write_valid_seed_stamp(main_stamp)

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


def test_local_seed_missing_only_stamp_still_bootstraps(tmp_path: Path) -> None:
    """Partial local state (e.g. a stale stamp with no db, or vice versa) still needs a fresh copy."""
    local_stamp = tmp_path / "local" / "seed.json"
    _write_valid_seed_stamp(local_stamp)
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

    assert local_stamp.read_text() == main_stamp.read_text()
    conn = sqlite3.connect(local_data)
    try:
        rows = conn.execute("SELECT path, fsha FROM file_fp ORDER BY path").fetchall()
    finally:
        conn.close()
    assert rows == [("x", "sha-x"), ("y", "sha-y"), ("z", "sha-z")]
    # No temp files left behind.
    assert sorted(p.name for p in local_data.parent.iterdir()) == ["seed.json", "testmondata"]


def test_bootstrap_seed_files_noop_when_decision_says_no(tmp_path: Path) -> None:
    local_data = tmp_path / "local" / "testmondata"
    local_stamp = tmp_path / "local" / "seed.json"
    decision = BootstrapDecision(False, "not needed")

    bootstrap_testmon_seed_files(decision, local_testmon_data=local_data, local_seed_stamp=local_stamp)

    assert not local_data.exists()
    assert not local_stamp.exists()
