"""``CursorStore.ops_write_scope`` shares one ops.db connection without losing telemetry.

Anti-vacuity, per test:

* ``test_scope_reuses_one_ops_connection`` counts real ``sqlite3`` opens against
  ``ops.db``. Delete the held-connection branch of ``_connect_ops`` (or the
  scope itself) and the count returns to one open per operation, which the
  assertion rejects.
* ``test_buffered_stage_events_all_land`` asserts every buffered event is
  readable after the scope. Drop the scope-exit flush and the trailing events
  are missing.
* ``test_buffered_stage_events_survive_an_exception`` leaves the scope by
  raising. Guard the flush behind the success path and it goes red.
* ``test_buffer_is_bounded`` asserts the in-memory buffer never exceeds its cap
  while far more events are recorded, and that all of them still land. Remove
  the cap check and the observed high-water mark grows past it.
* ``test_stage_event_commit_false_defers_the_row`` holds the storage-level
  contract: ``commit=False`` must still write the row into the caller's
  transaction. Make it a no-op and the row is absent after the caller commits.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.ops_write import record_daemon_stage_event
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.write_lease import (
    UnleasedWriteError,
    arm_write_lease_enforcement,
    write_lease,
)
from tests.infra.cursor_authority import fixture_cursor_authority


@pytest.fixture
def store(tmp_path: Path) -> CursorStore:
    return CursorStore(tmp_path / "index.db")


def _events(store: CursorStore) -> list[tuple[str, str]]:
    with sqlite3.connect(store._ops_db_path) as conn:
        return [
            (str(row[0]), str(row[1]))
            for row in conn.execute("SELECT stage, status FROM daemon_stage_events ORDER BY rowid")
        ]


def _count_ops_opens(monkeypatch: pytest.MonkeyPatch, store: CursorStore) -> list[int]:
    """Count ``sqlite3.connect`` calls that target this store's ops.db."""
    opens = [0]
    real_connect = sqlite3.connect
    ops_name = store._ops_db_path.name

    def counting_connect(database: Any, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        if ops_name in str(database):
            opens[0] += 1
        return cast(sqlite3.Connection, real_connect(database, *args, **kwargs))

    monkeypatch.setattr(sqlite3, "connect", counting_connect)
    return opens


def test_scope_reuses_one_ops_connection(store: CursorStore, monkeypatch: pytest.MonkeyPatch) -> None:
    attempt_id = store.begin_ingest_attempt(paths=[Path("/tmp/a.jsonl")], input_bytes=1, queued_file_count=1)
    opens = _count_ops_opens(monkeypatch, store)
    for index in range(6):
        store.record_ingest_stage_event(attempt_id, phase=f"unscoped-{index}", status="running")
    unscoped_opens = opens[0]

    opens[0] = 0
    with store.ops_write_scope():
        for index in range(6):
            store.record_ingest_stage_event(attempt_id, phase=f"scoped-{index}", status="running")
        store.update_ingest_attempt(attempt_id, phase="scoped-update")
    scoped_opens = opens[0]

    assert unscoped_opens >= 6
    assert scoped_opens == 1


def test_buffered_stage_events_all_land(store: CursorStore) -> None:
    attempt_id = store.begin_ingest_attempt(paths=[Path("/tmp/a.jsonl")], input_bytes=1, queued_file_count=1)
    with store.ops_write_scope():
        for index in range(5):
            store.record_ingest_stage_event(attempt_id, phase=f"stage-{index}", status="running")
        # An intervening writer commits; the events buffered before it must be
        # visible no later than that commit, never after the attempt it finishes.
        store.finish_ingest_attempt(attempt_id, status="succeeded", phase="done")
        for index in range(5, 8):
            store.record_ingest_stage_event(attempt_id, phase=f"stage-{index}", status="running")

    stages = [stage for stage, _ in _events(store)]
    for index in range(8):
        assert f"stage-{index}" in stages


def test_buffered_stage_events_survive_an_exception(store: CursorStore) -> None:
    attempt_id = store.begin_ingest_attempt(paths=[Path("/tmp/a.jsonl")], input_bytes=1, queued_file_count=1)
    with pytest.raises(RuntimeError, match="batch failed"):
        with store.ops_write_scope():
            store.record_ingest_stage_event(attempt_id, phase="before-failure", status="running")
            raise RuntimeError("batch failed")

    assert "before-failure" in [stage for stage, _ in _events(store)]


def test_buffer_is_bounded(store: CursorStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("polylogue.sources.live.cursor._MAX_BUFFERED_OPS_STAGE_EVENTS", 4)
    attempt_id = store.begin_ingest_attempt(paths=[Path("/tmp/a.jsonl")], input_bytes=1, queued_file_count=1)
    high_water = 0
    with store.ops_write_scope():
        for index in range(20):
            store.record_ingest_stage_event(attempt_id, phase=f"bounded-{index}", status="running")
            high_water = max(high_water, len(store._ops_scope.pending))

    assert high_water <= 4
    stages = [stage for stage, _ in _events(store)]
    for index in range(20):
        assert f"bounded-{index}" in stages


def test_stage_event_commit_false_defers_the_row(tmp_path: Path) -> None:
    ops_path = tmp_path / "ops.db"
    initialize_archive_database(ops_path, ArchiveTier.OPS)
    conn = sqlite3.connect(ops_path)
    try:
        event_id = record_daemon_stage_event(
            conn,
            stage="deferred",
            status="running",
            observed_at_ms=1,
            commit=False,
        )
        assert conn.in_transaction
        conn.commit()
    finally:
        conn.close()

    with sqlite3.connect(ops_path) as reader:
        rows = reader.execute("SELECT stage FROM daemon_stage_events WHERE event_id = ?", (event_id,)).fetchall()
    assert [row[0] for row in rows] == ["deferred"]


def test_scope_refuses_an_unleased_entrant(store: CursorStore) -> None:
    """polylogue-vgxkk: the shared ops.db scope asserts the writer lease at entry.

    Anti-vacuity: delete the ``require_write_lease`` call at the top of
    ``ops_write_scope`` and an unserialized entrant enters the shared-connection
    scope silently again, so this raises nothing and the test goes red. The
    concurrent-ingest shape this refuses is two ``ingest_files`` calls on one
    event loop landing in the same thread-local scope, where one caller's commit
    would commit the other's in-flight statement.
    """
    with arm_write_lease_enforcement():
        with pytest.raises(UnleasedWriteError, match="live ingest ops write scope"):
            with store.ops_write_scope():
                pass

        # The scope is entered normally by the serialized writer that holds it.
        with write_lease("test-writer", archive_root=store._archive_root):
            with store.ops_write_scope():
                pass


def _cursor_size(store: CursorStore, path: Path) -> int | None:
    with sqlite3.connect(store._ops_db_path) as conn:
        row = conn.execute("SELECT stat_size FROM ingest_cursor WHERE source_path = ?", (str(path),)).fetchone()
    return None if row is None else int(row[0])


def test_ops_batch_commits_its_writes_once_at_the_end(store: CursorStore) -> None:
    """Anti-vacuity: a per-write commit makes the first cursor visible to
    another connection before the batch ends."""
    first, second = Path("/tmp/a.jsonl"), Path("/tmp/b.jsonl")
    with store.ops_write_scope():
        with store.ops_batch():
            store.set(first, 10, authority=fixture_cursor_authority(first))
            assert _cursor_size(store, first) is None
            store.set(second, 20, authority=fixture_cursor_authority(second))
        assert (_cursor_size(store, first), _cursor_size(store, second)) == (10, 20)


def test_ops_batch_rolls_back_as_a_unit(store: CursorStore) -> None:
    """Anti-vacuity: committing inside the batch leaves the first cursor behind."""
    path = Path("/tmp/a.jsonl")
    with pytest.raises(RuntimeError, match="cursor write failed"):
        with store.ops_write_scope(), store.ops_batch():
            store.set(path, 10, authority=fixture_cursor_authority(path))
            raise RuntimeError("cursor write failed")
    assert _cursor_size(store, path) is None


def test_ops_batch_holds_read_modify_writes_after_an_upsert(store: CursorStore) -> None:
    """A group's cursor writes mix upserts and locked read-modify-writes.

    Anti-vacuity: a read-modify-write that always issues ``BEGIN IMMEDIATE``
    raises "cannot start a transaction within a transaction" once the batch's
    first upsert has opened it (the published group then fails admission),
    and one that commits its own upsert publishes the cursor mid-batch.
    """
    path = Path("/tmp/a.jsonl")
    with store.ops_write_scope():
        with store.ops_batch():
            store.set(path, 10, authority=fixture_cursor_authority(path))
            store.mark_failed(path, authority=fixture_cursor_authority(path))
            assert _cursor_size(store, path) is None
        assert _cursor_size(store, path) == 10
    record = store.get_record(path)
    assert record is not None and record.failure_count == 1


def test_ops_batch_requires_a_scope(store: CursorStore) -> None:
    with pytest.raises(RuntimeError, match="ops_write_scope"):
        with store.ops_batch():
            pass
