"""Regression coverage for polylogue-61jg: interrupted ingest requeue.

Before this fix, ``CursorStore._mark_interrupted_ops_attempts`` stamped a
dangling ``running`` ``ingest_attempts`` row ``interrupted`` on the next
daemon start and stopped there -- nothing registered the affected source
path as retryable ``convergence_debt``, so a raw row acquired but never
validated/parsed waited for an accidental future touch instead of being
requeued (2026-07-31 acquisition-completeness audit, F-07/F-08).

These tests pin:
1. An interrupted ingest attempt registers ``raw_parse_recovery``
   convergence debt for every source path it covered (including the legacy
   single-``source_path`` fallback for rows without ``source_paths_json``).
2. ``make_raw_parse_recovery_stage``'s ``check``/``execute`` correctly
   detects and drains a stuck raw row (acquired, never parsed, no
   materialized session) for a given source path.
3. End to end: a simulated daemon kill mid-batch (SIGKILL -> reopen
   ``CursorStore``) demonstrably resumes parsing on next start once the
   registered debt is drained by the daemon's convergence loop -- the
   bead's AC3.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.daemon.convergence import DaemonConverger, StageState
from polylogue.daemon.convergence_stages import make_raw_parse_recovery_stage
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.archive_identity import archive_file_set_root
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

_CHATGPT_CONVERSATION = {
    "id": "conv-stuck",
    "title": "stuck conversation",
    "create_time": 1,
    "current_node": "message-1",
    "mapping": {
        "message-1": {
            "id": "message-1",
            "parent": None,
            "children": [],
            "message": {
                "id": "message-1",
                "author": {"role": "user"},
                "create_time": 1,
                "content": {"content_type": "text", "parts": ["stuck raw content"]},
            },
        }
    },
}


def _write_stuck_raw(archive_root: Path, *, source_path: str) -> str:
    """Write a raw row with real conversational content that is never parsed.

    Mirrors an ingest attempt that acquired bytes but was interrupted before
    validation/parse ever ran: ``parsed_at_ms``/``validated_at_ms`` stay NULL
    and no index session exists for it.
    """
    import json

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        return archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=json.dumps([_CHATGPT_CONVERSATION]).encode(),
            source_path=source_path,
            acquired_at_ms=1,
        )


def _sessions_for_raw(archive_root: Path, raw_id: str) -> list[tuple[object, ...]]:
    conn = sqlite3.connect(archive_root / "index.db")
    try:
        return list(conn.execute("SELECT native_id, raw_id FROM sessions WHERE raw_id = ?", (raw_id,)))
    finally:
        conn.close()


def test_interrupted_ingest_attempt_registers_raw_parse_recovery_debt(tmp_path: Path) -> None:
    db = tmp_path / "live.sqlite"
    store = CursorStore(db)
    src = tmp_path / "conv.json"
    src.write_text("x")
    store.begin_ingest_attempt(paths=[src], input_bytes=1, queued_file_count=1)

    # Simulate SIGKILL: the attempt never finished. Reopening CursorStore is
    # the daemon-restart recovery path.
    reopened = CursorStore(db)

    debts = reopened.list_convergence_debt(limit=50)
    matching = [d for d in debts if d.stage == "raw_parse_recovery" and d.subject_id == str(src)]
    assert len(matching) == 1, f"expected exactly one raw_parse_recovery debt row for {src}, got {debts}"
    assert matching[0].subject_type == "source_path"


def test_interrupted_attempt_without_source_paths_json_falls_back_to_source_path(tmp_path: Path) -> None:
    """Legacy rows populate only ``source_path``, not the JSON list -- still requeued."""
    db = tmp_path / "live.sqlite"
    store = CursorStore(db)
    src = tmp_path / "legacy.jsonl"
    src.write_text("x")
    attempt_id = store.begin_ingest_attempt(paths=[src], input_bytes=1, queued_file_count=1)

    ops_db = db.with_name("ops.db")
    conn = sqlite3.connect(ops_db)
    try:
        conn.execute(
            "UPDATE ingest_attempts SET source_paths_json = '[]', source_path = ? WHERE attempt_id = ?",
            (str(src), attempt_id),
        )
        conn.commit()
    finally:
        conn.close()

    reopened = CursorStore(db)
    debts = reopened.list_convergence_debt(limit=50)
    matching = [d for d in debts if d.stage == "raw_parse_recovery" and d.subject_id == str(src)]
    assert len(matching) == 1


def test_raw_parse_recovery_stage_drains_a_stuck_raw_row(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    path = tmp_path / "stuck.json"
    raw_id = _write_stuck_raw(tmp_path, source_path=str(path))

    stage = make_raw_parse_recovery_stage(tmp_path / "index.db")

    assert stage.check(path) is True, "stuck raw row was not detected as pending backlog"

    result = stage.execute(path)
    assert bool(result) is True

    assert stage.check(path) is False, "raw row should be fully materialized after execute"

    rows = _sessions_for_raw(tmp_path, raw_id)
    assert len(rows) == 1
    assert rows[0][0] == "conv-stuck"


@pytest.mark.parametrize("authority_state", ["violated", "unknown"])
def test_raw_parse_recovery_stage_blocks_unproven_cursor_authority(
    authority_state: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_active_archive_root(tmp_path)
    path = tmp_path / "stuck.json"
    raw_id = _write_stuck_raw(tmp_path, source_path=str(path))
    monkeypatch.setattr(
        "polylogue.readiness.capability.raw_frontier_source_selection_block_reason",
        lambda _root: f"{authority_state} cursor authority",
    )

    stage = make_raw_parse_recovery_stage(tmp_path / "index.db")

    assert stage.execute(path) is False
    assert stage.check(path) is True
    assert _sessions_for_raw(tmp_path, raw_id) == []


def test_raw_parse_recovery_stage_is_false_means_pending() -> None:
    assert make_raw_parse_recovery_stage(Path("/nonexistent/index.db")).false_means_pending is True


def test_raw_parse_recovery_missing_source_is_no_backlog(tmp_path: Path) -> None:
    stage = make_raw_parse_recovery_stage(tmp_path / "index.db")
    converger = DaemonConverger(stages=(stage,))

    states, _timings = converger.converge_batch([tmp_path / "missing.json"])

    state = states[tmp_path / "missing.json"]
    assert state.stages["raw_parse_recovery"] is StageState.DONE
    assert state.converged is True
    assert state.error_count == 0


def test_raw_parse_recovery_missing_source_tier_is_retryable(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    (tmp_path / "source.db").rename(tmp_path / "source.db.unavailable")
    stage = make_raw_parse_recovery_stage(tmp_path / "index.db")
    converger = DaemonConverger(stages=(stage,))

    states, _timings = converger.converge_batch([tmp_path / "missing-source.json"])

    state = states[tmp_path / "missing-source.json"]
    assert state.stages["raw_parse_recovery"] is StageState.FAILED
    assert state.converged is False
    assert state.error_count == 1


def test_raw_parse_recovery_no_qualifying_rows_is_done(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    stage = make_raw_parse_recovery_stage(tmp_path / "index.db")
    converger = DaemonConverger(stages=(stage,))

    states, _timings = converger.converge_batch([tmp_path / "untracked.json"])

    state = states[tmp_path / "untracked.json"]
    assert state.stages["raw_parse_recovery"] is StageState.DONE
    assert state.converged is True
    assert state.error_count == 0


def test_raw_parse_recovery_uses_active_index_pointer(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    path = tmp_path / "pointer.json"
    raw_id = _write_stuck_raw(tmp_path, source_path=str(path))

    active_index = tmp_path / "generations" / "active" / "index.db"
    active_index.parent.mkdir(parents=True)
    shutil.copy2(tmp_path / "index.db", active_index)
    with sqlite3.connect(active_index) as conn:
        origin = conn.execute("SELECT origin FROM main.sessions LIMIT 1").fetchone()
        if origin is None:
            with sqlite3.connect(tmp_path / "source.db") as source_conn:
                origin = source_conn.execute("SELECT origin FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()
        assert origin is not None
        conn.execute(
            "INSERT INTO sessions (native_id, origin, raw_id, content_hash) VALUES (?, ?, ?, zeroblob(32))",
            ("conv-stuck", origin[0], "stale-index-only-raw"),
        )
        conn.commit()

    (tmp_path / ".index-active-pointer").write_text(f"{active_index}\n", encoding="utf-8")
    stage = make_raw_parse_recovery_stage(tmp_path / "index.db")

    assert archive_file_set_root(archive_root=tmp_path, db_path=active_index) == tmp_path
    assert stage.check(path) is True


def test_raw_parse_recovery_retries_authorized_parse_failure(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    path = tmp_path / "retryable.json"
    raw_id = _write_stuck_raw(tmp_path, source_path=str(path))
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_sessions SET parse_error = ?, parsed_at_ms = NULL WHERE raw_id = ?",
            ("OperationalError: database is locked", raw_id),
        )
        conn.commit()

    stage = make_raw_parse_recovery_stage(tmp_path / "index.db")

    assert stage.check(path) is True


def test_raw_parse_recovery_source_open_failure_is_failed_and_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_active_archive_root(tmp_path)
    source_path = tmp_path / "unavailable.json"
    calls = 0

    def fail_connect(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise sqlite3.OperationalError("source tier unavailable")

    monkeypatch.setattr("polylogue.daemon.convergence_stages.sqlite3.connect", fail_connect)

    converger = DaemonConverger(stages=(make_raw_parse_recovery_stage(tmp_path / "index.db"),))
    states, _timings = converger.converge_batch([source_path])

    state = states[source_path]
    assert state.stages["raw_parse_recovery"] is StageState.FAILED
    assert state.converged is False
    assert state.error_count == 1

    states, _timings = converger.converge_batch([source_path])
    assert states[source_path].stages["raw_parse_recovery"] is StageState.FAILED
    assert states[source_path].error_count == 2
    assert calls == 2


@pytest.mark.parametrize("failure", ["attach", "query"])
def test_raw_parse_recovery_sqlite_probe_failure_is_failed_and_closes_connection(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_active_archive_root(tmp_path)
    source_path = tmp_path / f"{failure}-failed.json"

    class FailingConnection:
        closed = False

        def execute(self, sql: str, *_args: object, **_kwargs: object) -> object:
            if failure == "attach" or sql.lstrip().startswith("SELECT"):
                raise sqlite3.OperationalError(f"{failure} failed")
            return self

        def fetchone(self) -> tuple[int]:
            raise AssertionError("failed probe should not fetch a row")

        def close(self) -> None:
            self.closed = True

    connection = FailingConnection()
    monkeypatch.setattr("polylogue.daemon.convergence_stages.sqlite3.connect", lambda *_args, **_kwargs: connection)

    converger = DaemonConverger(stages=(make_raw_parse_recovery_stage(tmp_path / "index.db"),))
    states, _timings = converger.converge_batch([source_path])

    state = states[source_path]
    assert state.stages["raw_parse_recovery"] is StageState.FAILED
    assert state.converged is False
    assert state.error_count == 1
    assert connection.closed is True


def test_daemon_restart_resumes_parsing_of_an_interrupted_batch(tmp_path: Path) -> None:
    """End-to-end AC3: a daemon kill mid-batch demonstrably resumes on restart.

    Models: attempt begins for a source path -> raw bytes for that path are
    durably acquired -> the daemon dies before validation/parse ever runs ->
    a fresh ``CursorStore`` open (the restart) registers retryable debt ->
    the daemon's own convergence loop (``DaemonConverger`` with the
    ``raw_parse_recovery`` stage) drains that debt and the session is
    materialized, without any operator-triggered manual reprocess.
    """
    initialize_active_archive_root(tmp_path)
    source_path = tmp_path / "batch.json"
    source_path.write_text("placeholder")
    raw_id = _write_stuck_raw(tmp_path, source_path=str(source_path))

    live_db = tmp_path / "live.sqlite"
    store = CursorStore(live_db, ops_db_path=tmp_path / "ops.db")
    store.begin_ingest_attempt(paths=[source_path], input_bytes=1, queued_file_count=1)

    # Simulate the crash + restart.
    restarted_store = CursorStore(live_db, ops_db_path=tmp_path / "ops.db")
    debts = restarted_store.list_convergence_debt(limit=50)
    pending_paths = [Path(d.subject_id) for d in debts if d.stage == "raw_parse_recovery"]
    assert source_path in pending_paths

    converger = DaemonConverger(stages=(make_raw_parse_recovery_stage(tmp_path / "index.db"),))
    states, _timings = converger.converge_batch(pending_paths)
    assert states[source_path].converged is True

    rows = _sessions_for_raw(tmp_path, raw_id)
    assert len(rows) == 1
    assert rows[0][0] == "conv-stuck"


__all__: list[str] = []
