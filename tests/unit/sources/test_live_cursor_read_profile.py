from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.sqlite.connection_profile import ReadFrame
from tests.infra.cursor_authority import fixture_cursor_authority
from tests.infra.durable_tier_fixtures import initialize_runtime_source_fixture


def _seed_unparsed_raw(source_db: Path, source_path: str, raw_id: str) -> None:
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, 'codex-session', ?, ?, 1, 1)
            """,
            (raw_id, source_path, bytes.fromhex("11" * 32)),
        )


def _store(archive_root: Path) -> CursorStore:
    return CursorStore(
        archive_root / "index.db",
        ops_db_path=archive_root / "ops.db",
    )


def test_interrupted_cursor_recovery_reads_committed_wal_rows(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    source_db = archive_root / "source.db"
    initialize_runtime_source_fixture(source_db)
    source_path = archive_root / "capture.jsonl"
    writer = sqlite3.connect(source_db)
    try:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"
        writer.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms
            ) VALUES ('raw-wal', 'codex-session', ?, ?, 1, 1)
            """,
            (str(source_path), bytes.fromhex("22" * 32)),
        )
        writer.commit()
        assert Path(f"{source_db}-wal").exists()

        store = _store(archive_root)
        store.set(
            source_path, 64, byte_offset=64, last_complete_newline=64, authority=fixture_cursor_authority(source_path)
        )

        store._rewind_interrupted_unparsed_cursors((str(source_path),))

        record = store.get_record(source_path)
        assert record is not None and record.byte_offset == 0
    finally:
        writer.close()


def test_interrupted_cursor_recovery_rebinds_after_midstream_expiry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_clock: Any,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    source_db = archive_root / "source.db"
    initialize_runtime_source_fixture(source_db)
    paths = tuple(str(archive_root / name) for name in ("first.jsonl", "second.jsonl"))
    for index, source_path in enumerate(paths):
        _seed_unparsed_raw(source_db, source_path, f"raw-{index}")

    store = _store(archive_root)
    for source_path in paths:
        store.set(
            Path(source_path),
            64,
            byte_offset=64,
            last_complete_newline=64,
            authority=fixture_cursor_authority(Path(source_path)),
        )

    frames: list[ReadFrame] = []
    real_stream = ReadFrame.stream
    expired = False

    def expire_inside_stream(frame: ReadFrame, sql: str, parameters: Any = ()) -> Iterator[sqlite3.Row]:
        nonlocal expired
        if frame not in frames:
            frames.append(frame)
        for row in real_stream(frame, sql, parameters):
            yield row
            if not expired and "FROM raw_sessions AS r" in sql:
                expired = True
                frozen_clock.advance(301)

    monkeypatch.setattr(ReadFrame, "stream", expire_inside_stream)

    store._rewind_interrupted_unparsed_cursors(paths)

    assert expired
    assert frames and frames[0].epoch >= 1
    for source_path in paths:
        record = store.get_record(Path(source_path))
        assert record is not None
        assert record.byte_offset == 0


def test_interrupted_cursor_recovery_keeps_cursor_when_source_tier_is_missing(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    store = _store(archive_root)
    source_path = archive_root / "capture.jsonl"
    store.set(
        source_path, 64, byte_offset=64, last_complete_newline=64, authority=fixture_cursor_authority(source_path)
    )

    store._rewind_interrupted_unparsed_cursors((str(source_path),))

    record = store.get_record(source_path)
    assert record is not None
    assert record.byte_offset == 64


def test_interrupted_attempts_stay_running_when_the_recovery_read_expires(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recovery read that keeps expiring fails startup and keeps the obligation.

    Anti-vacuity: mark attempts interrupted before the rewind read (the old
    order in ``_mark_interrupted_ops_attempts``) and the attempt leaves
    ``running``, so the next startup no longer knows the path owes a rewind.
    """
    from polylogue.storage.sqlite.connection_profile import ReadFrameExpiredError

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    source_db = archive_root / "source.db"
    initialize_runtime_source_fixture(source_db)
    source_path = str(archive_root / "capture.jsonl")
    _seed_unparsed_raw(source_db, source_path, "raw-0")
    store = _store(archive_root)
    store.set(
        Path(source_path),
        64,
        byte_offset=64,
        last_complete_newline=64,
        authority=fixture_cursor_authority(Path(source_path)),
    )
    with store._connect_ops() as conn:
        conn.execute(
            "INSERT INTO ingest_attempts (attempt_id, source_path, origin, status, phase, started_at_ms, "
            "heartbeat_at_ms, parsed_raw_count, materialized_count) "
            "VALUES ('attempt-1', ?, 'codex-session', 'running', 'parse', 1, 1, 0, 0)",
            (source_path,),
        )
        conn.commit()

    def always_expired(_frame: ReadFrame, _sql: str, _parameters: Any = ()) -> Iterator[sqlite3.Row]:
        raise ReadFrameExpiredError("synthetic expiry")
        yield  # pragma: no cover

    monkeypatch.setattr(ReadFrame, "stream", always_expired)
    monkeypatch.setattr(ReadFrame, "rebind", lambda _frame: None)

    with pytest.raises(ReadFrameExpiredError):
        store._mark_interrupted_ops_attempts()

    with store._connect_ops() as conn:
        assert conn.execute("SELECT status FROM ingest_attempts WHERE attempt_id = 'attempt-1'").fetchone() == (
            "running",
        )
    record = store.get_record(Path(source_path))
    assert record is not None and record.byte_offset == 64
