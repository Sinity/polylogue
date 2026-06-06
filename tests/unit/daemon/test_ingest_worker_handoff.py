"""Ingest worker handoff invariants under shutdown (#1182).

Pins the documented contract for archive ``ops.db.ingest_attempts`` rows:

- ``begin_ingest_attempt`` persists a durable row with status
  ``running`` before any work starts; the row survives process death.
- ``_mark_interrupted_attempts`` (run at ``CursorStore`` construction)
  closes orphan ``running`` rows from a previous daemon instance —
  this is the recovery path for a SIGKILL'd worker.
- A successful ``finish_ingest_attempt`` transitions the row to a
  terminal status; a subsequent ``CursorStore`` open does NOT mutate
  it.
- Concurrent attempts produce distinct ``attempt_id`` rows; the
  uniqueness contract (primary key) is preserved under racing
  inserts.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from polylogue.sources.live.cursor import CursorStore


def _attempt_rows(db_path: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(db_path.with_name("ops.db")))
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute("SELECT attempt_id, status, phase, error_message FROM ingest_attempts"))
    finally:
        conn.close()


def test_begin_attempt_persists_running_row(tmp_path: Path) -> None:
    db = tmp_path / "live.sqlite"
    store = CursorStore(db)
    src = tmp_path / "session.jsonl"
    src.write_text("x")
    attempt_id = store.begin_ingest_attempt(paths=[src], input_bytes=1, queued_file_count=1)
    rows = _attempt_rows(db)
    assert len(rows) == 1
    assert rows[0]["attempt_id"] == attempt_id
    assert rows[0]["status"] == "running"


def test_simulated_sigkill_recovered_by_next_open(tmp_path: Path) -> None:
    """A ``running`` row left by a killed daemon is marked ``interrupted`` on next open.

    This is the only mechanism that prevents the daemon from leaking
    ``running`` rows across restarts.
    """
    db = tmp_path / "live.sqlite"
    store = CursorStore(db)
    src = tmp_path / "session.jsonl"
    src.write_text("x")
    attempt_id = store.begin_ingest_attempt(paths=[src], input_bytes=1, queued_file_count=1)
    rows = _attempt_rows(db)
    assert rows[0]["status"] == "running"

    # Simulate SIGKILL: just open a new CursorStore on the same DB.
    CursorStore(db)
    rows = _attempt_rows(db)
    assert len(rows) == 1
    assert rows[0]["attempt_id"] == attempt_id
    assert rows[0]["status"] == "interrupted", (
        "ingest_attempts row left in 'running' after restart — the SIGKILL recovery path no longer runs"
    )
    assert rows[0]["phase"] == "interrupted"
    assert rows[0]["error_message"] is not None


def test_finished_attempt_is_not_remarked_on_reopen(tmp_path: Path) -> None:
    """Reopening the store after a clean finish must leave the row alone."""
    db = tmp_path / "live.sqlite"
    store = CursorStore(db)
    src = tmp_path / "session.jsonl"
    src.write_text("x")
    attempt_id = store.begin_ingest_attempt(paths=[src], input_bytes=1, queued_file_count=1)
    store.finish_ingest_attempt(attempt_id, status="completed", phase="committed")
    rows_before = _attempt_rows(db)
    assert rows_before[0]["status"] == "completed"

    # New daemon process: open the store again.
    CursorStore(db)
    rows_after = _attempt_rows(db)
    assert len(rows_after) == 1
    assert rows_after[0]["status"] == "completed", (
        "completed attempt was clobbered by the SIGKILL recovery pass — the recovery path is too aggressive"
    )


def test_concurrent_begin_attempts_produce_distinct_rows(tmp_path: Path) -> None:
    """Two workers calling begin_ingest_attempt in parallel both get distinct rows."""
    db = tmp_path / "live.sqlite"
    store = CursorStore(db)
    src = tmp_path / "session.jsonl"
    src.write_text("x")

    ids: list[str] = []
    lock = threading.Lock()
    error: list[BaseException] = []

    def caller() -> None:
        try:
            for _ in range(10):
                attempt_id = store.begin_ingest_attempt(paths=[src], input_bytes=1, queued_file_count=1)
                with lock:
                    ids.append(attempt_id)
        except BaseException as exc:  # pragma: no cover - defensive thread error capture
            error.append(exc)

    threads = [threading.Thread(target=caller) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not error, f"thread raised: {error}"

    assert len(ids) == len(set(ids)), "duplicate attempt_id produced under concurrent begin"
    rows = _attempt_rows(db)
    assert {row["attempt_id"] for row in rows} == set(ids), "some begin_ingest_attempt calls did not persist their row"


def test_handoff_under_shutdown_preserves_in_flight_row(tmp_path: Path) -> None:
    """In-flight attempts are visible from a fresh connection mid-flight.

    The handoff contract: a worker that begins an attempt but is then
    interrupted leaves a row another process can read and reconcile.
    """
    db = tmp_path / "live.sqlite"
    store = CursorStore(db)
    src = tmp_path / "session.jsonl"
    src.write_text("x")

    attempt_id = store.begin_ingest_attempt(paths=[src], input_bytes=1, queued_file_count=1)
    # Worker updates phase mid-flight, then "dies".
    store.update_ingest_attempt(attempt_id, phase="parsing")

    # Independent reader (e.g. status endpoint) sees the in-flight row.
    conn = sqlite3.connect(str(db.with_name("ops.db")))
    try:
        row = conn.execute(
            "SELECT status, phase FROM ingest_attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row[0] == "running"
    assert row[1] == "parsing"
