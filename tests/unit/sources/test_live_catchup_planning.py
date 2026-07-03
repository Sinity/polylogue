from __future__ import annotations

import asyncio
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.batch_support import _AppendPlan
from polylogue.sources.live.cursor import CursorStore
from tests.infra.frozen_clock import FrozenClock


def _write_archive_blob(archive_root: Path, blob_hash: bytes | str, payload: bytes) -> None:
    blob_hash_hex = blob_hash.hex() if isinstance(blob_hash, bytes) else blob_hash.lower()
    blob_path = archive_root / "blob" / blob_hash_hex[:2] / blob_hash_hex[2:]
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_bytes(payload)


def test_catch_up_plan_carries_statted_candidates_without_payload_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "src"
    root.mkdir()
    changed = root / "changed.jsonl"
    unchanged = root / "unchanged.jsonl"
    changed.write_text('{"role":"user","content":"new"}\n')
    unchanged.write_text('{"role":"user","content":"old"}\n')
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    cursor = CursorStore(tmp_path / "cursor.sqlite")
    watcher = LiveWatcher(cast(Any, polylogue), (WatchSource(name="test", root=root),), cursor=cursor)
    stat = unchanged.stat()
    cursor.set(
        unchanged,
        stat.st_size,
        byte_offset=stat.st_size,
        last_complete_newline=stat.st_size,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="already-known",
        source_name="test",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )

    def fail_fingerprint_file(path: Path) -> tuple[str, int]:
        raise AssertionError(f"unchanged catch-up planning should not fingerprint payloads: {path}")

    monkeypatch.setattr(live_watcher, "fingerprint_file", fail_fingerprint_file)
    candidates = watcher._scan_catch_up_candidates([root])
    plan = watcher._plan_catch_up(candidates)

    assert [candidate.path for candidate in candidates] == [changed, unchanged]
    assert plan.needed == (changed,)
    assert plan.skipped_file_count == 1
    assert plan.needed_bytes == changed.stat().st_size


def test_catch_up_repairs_missing_cursor_from_archive_source_row(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session_blob_ref
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "src"
    root.mkdir()
    archived = root / "archived.jsonl"
    archived.write_text('{"type":"session_meta","payload":{"id":"archived"}}\n', encoding="utf-8")
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    cursor = CursorStore(tmp_path / "ops.db")
    source_db = tmp_path / "source.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    with sqlite3.connect(source_db) as conn:
        blob_hash = b"a" * 32
        write_source_raw_session_blob_ref(
            conn,
            origin="codex-session",
            source_path=str(archived),
            source_index=0,
            blob_hash=blob_hash,
            blob_size=archived.stat().st_size,
            acquired_at_ms=1,
            native_id="archived",
        )
    _write_archive_blob(tmp_path, blob_hash, archived.read_bytes())
    watcher = LiveWatcher(cast(Any, polylogue), (WatchSource(name="codex", root=root),), cursor=cursor)

    plan = watcher._plan_catch_up(watcher._scan_catch_up_candidates([root]))
    record = cursor.get_record(archived)

    assert plan.needed == ()
    assert plan.skipped_file_count == 1
    assert record is not None
    assert record.byte_size == archived.stat().st_size
    assert record.content_fingerprint == ("61" * 32)
    assert record.parser_fingerprint == live_watcher._PARSER_FINGERPRINT


def test_catch_up_does_not_repair_cursor_from_archive_row_with_missing_blob(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session_blob_ref
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "src"
    root.mkdir()
    archived = root / "archived.jsonl"
    archived.write_text('{"type":"session_meta","payload":{"id":"archived"}}\n', encoding="utf-8")
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    cursor = CursorStore(tmp_path / "ops.db")
    source_db = tmp_path / "source.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    with sqlite3.connect(source_db) as conn:
        write_source_raw_session_blob_ref(
            conn,
            origin="codex-session",
            source_path=str(archived),
            source_index=0,
            blob_hash=b"a" * 32,
            blob_size=archived.stat().st_size,
            acquired_at_ms=1,
            native_id="archived",
        )
    watcher = LiveWatcher(cast(Any, polylogue), (WatchSource(name="codex", root=root),), cursor=cursor)

    plan = watcher._plan_catch_up(watcher._scan_catch_up_candidates([root]))

    assert plan.needed == (archived,)
    assert plan.skipped_file_count == 0
    assert cursor.get_record(archived) is None


def test_catch_up_reconciles_browser_capture_cursor_from_archive_origin(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session_blob_ref
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "browser-capture"
    root.mkdir()
    archived = root / "capture.json"
    archived.write_text('{"polylogue_capture_kind":"browser_llm_session"}', encoding="utf-8")
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    cursor = CursorStore(tmp_path / "ops.db")
    source_db = tmp_path / "source.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    with sqlite3.connect(source_db) as conn:
        blob_hash = b"b" * 32
        write_source_raw_session_blob_ref(
            conn,
            origin="chatgpt-export",
            source_path=str(archived),
            source_index=0,
            blob_hash=blob_hash,
            blob_size=archived.stat().st_size,
            acquired_at_ms=1,
            native_id="capture",
        )
    _write_archive_blob(tmp_path, blob_hash, archived.read_bytes())
    watcher = LiveWatcher(
        cast(Any, polylogue),
        (WatchSource(name="browser-capture", root=root, suffixes=(".json",)),),
        cursor=cursor,
    )

    plan = watcher._plan_catch_up(watcher._scan_catch_up_candidates([root]))
    record = cursor.get_record(archived)

    assert plan.needed == ()
    assert plan.skipped_file_count == 1
    assert record is not None
    assert record.source_name == "chatgpt"
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        assert (
            conn.execute("SELECT origin FROM ingest_cursor WHERE source_path = ?", (str(archived),)).fetchone()[0]
            == "chatgpt-export"
        )


def test_codex_append_plan_recovers_identity_from_session_meta_when_source_row_missing(
    tmp_path: Path,
) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "src"
    root.mkdir()
    source = root / "rollout-2026-06-18T02-59-46-conv-hot.jsonl"
    prefix = b'{"timestamp":"2026-06-18T01:05:23.888Z","type":"session_meta","payload":{"id":"conv-hot"}}\n'
    source.write_bytes(prefix + b'{"type":"message","payload":{"role":"user","content":"old"}}\n')
    old_offset = source.stat().st_size
    with source.open("ab") as handle:
        handle.write(b'{"type":"message","payload":{"role":"assistant","content":"new"}}\n')
    stat = source.stat()

    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, message_count, content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("conv-hot", "codex-session", "missing-source-raw", 1, b"b" * 32, 1, 1),
        )

    cursor = CursorStore(tmp_path / "ops.db")
    cursor.set(
        source,
        old_offset,
        byte_offset=old_offset,
        last_complete_newline=old_offset,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="already-known",
        source_name="codex",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db"))),
        (WatchSource(name="codex", root=root),),
        cursor=cursor,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )

    plan = processor._append_plan(source)

    assert isinstance(plan, _AppendPlan)
    assert plan.start_offset == old_offset
    assert b'"type":"session_meta","payload":{"id":"conv-hot"}' in plan.payload
    assert b'"content":"new"' in plan.payload


def test_catch_up_ingests_needed_files_in_bounded_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "src"
    root.mkdir()
    files = [root / f"session-{index}.jsonl" for index in range(5)]
    for index, path in enumerate(files):
        path.write_text("x" * (index + 1))
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    watcher = LiveWatcher(cast(Any, polylogue), (WatchSource(name="test", root=root),))
    monkeypatch.setattr(live_watcher, "_CATCH_UP_MAX_BATCH_FILES", 2)
    monkeypatch.setattr(live_watcher, "_CATCH_UP_MAX_BATCH_BYTES", 100)

    calls: list[tuple[list[Path], int | None, int]] = []
    retry_scan_calls: list[int] = []

    async def fake_ingest_files(
        paths: list[Path],
        *,
        queued_file_count: int | None = None,
        skipped_file_count: int = 0,
    ) -> None:
        calls.append((paths, queued_file_count, skipped_file_count))

    watcher._ingest_files = fake_ingest_files  # type: ignore[assignment,method-assign]
    watcher._schedule_failed_retry_scan = lambda: retry_scan_calls.append(len(calls))  # type: ignore[method-assign]

    asyncio.run(watcher._catch_up([root]))

    assert [paths for paths, _queued, _skipped in calls] == [files[:2], files[2:4], files[4:]]
    assert calls[0][1:] == (5, 0)
    assert calls[1][1:] == (2, 0)
    assert calls[2][1:] == (1, 0)
    assert retry_scan_calls == [3]


def test_catch_up_does_not_immediately_requeue_failed_paths(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    files = [root / f"session-{index}.jsonl" for index in range(3)]
    for path in files:
        path.write_text('{"role":"user","content":"x"}\n')
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    watcher = LiveWatcher(cast(Any, polylogue), (WatchSource(name="test", root=root),))

    async def fake_ingest_files(
        paths: list[Path],
        *,
        queued_file_count: int | None = None,
        skipped_file_count: int = 0,
    ) -> SimpleNamespace:
        return SimpleNamespace(failed_paths=[str(paths[1])])

    watcher._ingest_files = fake_ingest_files  # type: ignore[assignment,method-assign]

    asyncio.run(watcher._catch_up([root]))

    assert watcher._pending_paths == set()


@pytest.mark.frozen_clock_modules("polylogue.sources.live.watcher", "polylogue.sources.live.cursor")
def test_catch_up_noop_failed_retry_batch_advances_backoff(tmp_path: Path, frozen_clock: FrozenClock) -> None:
    root = tmp_path / "src"
    root.mkdir()
    failed = root / "failed.jsonl"
    failed.write_text('{"not":"a session"}\n')
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    watcher = LiveWatcher(cast(Any, polylogue), (WatchSource(name="test", root=root),))
    watcher._cursor.mark_failed(failed)
    past = (frozen_clock.now() - timedelta(seconds=1)).isoformat()
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute("UPDATE ingest_cursor SET next_retry_at = ? WHERE source_path = ?", (past, str(failed)))
        conn.commit()

    async def fake_ingest_files(
        paths: list[Path],
        *,
        queued_file_count: int | None = None,
        skipped_file_count: int = 0,
    ) -> SimpleNamespace:
        del paths, queued_file_count, skipped_file_count
        return SimpleNamespace(succeeded_file_count=0, failed_file_count=0)

    watcher._ingest_files = fake_ingest_files  # type: ignore[assignment,method-assign]

    asyncio.run(watcher._catch_up([root]))

    record = watcher._cursor.get_record(failed)
    assert record is not None
    assert record.failure_count == 2
    assert record.next_retry_at is not None
    assert datetime.fromisoformat(record.next_retry_at) > frozen_clock.now()
    assert watcher._pending_paths == set()


def test_flush_pending_does_not_hot_requeue_backed_off_failure(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    f = root / "session.jsonl"
    f.write_text('{"role":"user","content":"a"}\n')
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    watcher = LiveWatcher(cast(Any, polylogue), (WatchSource(name="test", root=root),))

    async def fake_ingest_files(
        paths: list[Path],
        *,
        queued_file_count: int | None = None,
        skipped_file_count: int = 0,
    ) -> SimpleNamespace:
        del queued_file_count, skipped_file_count
        watcher._cursor.mark_failed(paths[0])
        return SimpleNamespace(failed_paths=[str(paths[0])])

    watcher._ingest_files = fake_ingest_files  # type: ignore[assignment,method-assign]
    watcher._pending_paths.add(f)

    asyncio.run(watcher._flush_pending())

    assert watcher._pending_paths == set()
    record = watcher._cursor.get_record(f)
    assert record is not None
    assert record.failure_count == 1
    assert record.next_retry_at is not None


@pytest.mark.frozen_clock_modules("polylogue.sources.live.watcher", "polylogue.sources.live.cursor")
def test_failed_retry_scan_requeues_only_due_failures(tmp_path: Path, frozen_clock: FrozenClock) -> None:
    root = tmp_path / "src"
    root.mkdir()
    due = root / "due.jsonl"
    waiting = root / "waiting.jsonl"
    due.write_text('{"role":"user","content":"due"}\n')
    waiting.write_text('{"role":"user","content":"waiting"}\n')
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    watcher = LiveWatcher(cast(Any, polylogue), (WatchSource(name="test", root=root),))
    watcher._cursor.mark_failed(due)
    watcher._cursor.mark_failed(waiting)
    past = (frozen_clock.now() - timedelta(seconds=1)).isoformat()
    future = (frozen_clock.now() + timedelta(seconds=60)).isoformat()
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute("UPDATE ingest_cursor SET next_retry_at = ? WHERE source_path = ?", (past, str(due)))
        conn.execute("UPDATE ingest_cursor SET next_retry_at = ? WHERE source_path = ?", (future, str(waiting)))
        conn.commit()

    async def run_scan() -> None:
        watcher._schedule_failed_retry_scan()
        watcher.cancel_pending()

    asyncio.run(run_scan())

    assert watcher._pending_paths == {due}


@pytest.mark.frozen_clock_modules("polylogue.sources.live.watcher", "polylogue.sources.live.cursor")
def test_noop_failed_retry_batch_advances_backoff(tmp_path: Path, frozen_clock: FrozenClock) -> None:
    root = tmp_path / "src"
    root.mkdir()
    failed = root / "failed.json"
    failed.write_text('{"not":"a session"}\n')
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    watcher = LiveWatcher(cast(Any, polylogue), (WatchSource(name="test", root=root),))
    watcher._cursor.mark_failed(failed)
    past = (frozen_clock.now() - timedelta(seconds=1)).isoformat()
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute("UPDATE ingest_cursor SET next_retry_at = ? WHERE source_path = ?", (past, str(failed)))
        conn.commit()

    async def fake_ingest_files(
        paths: list[Path],
        *,
        queued_file_count: int | None = None,
        skipped_file_count: int = 0,
    ) -> SimpleNamespace:
        del paths, queued_file_count, skipped_file_count
        return SimpleNamespace(succeeded_file_count=0, failed_file_count=0)

    watcher._ingest_files = fake_ingest_files  # type: ignore[assignment,method-assign]
    watcher._pending_paths.add(failed)

    asyncio.run(watcher._flush_pending())

    record = watcher._cursor.get_record(failed)
    assert record is not None
    assert record.failure_count == 2
    assert record.next_retry_at is not None
    assert datetime.fromisoformat(record.next_retry_at) > frozen_clock.now()
    assert watcher._pending_paths == set()


@pytest.mark.frozen_clock_modules("polylogue.sources.live.watcher", "polylogue.sources.live.cursor")
def test_pending_failed_retry_without_needed_work_advances_backoff(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    root = tmp_path / "src"
    root.mkdir()
    failed = root / "failed.json"
    failed.write_text('{"not":"a session"}\n')
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    watcher = LiveWatcher(cast(Any, polylogue), (WatchSource(name="test", root=root),))
    watcher._cursor.mark_failed(failed)
    past = (frozen_clock.now() - timedelta(seconds=1)).isoformat()
    with sqlite3.connect(tmp_path / "ops.db") as conn:
        conn.execute("UPDATE ingest_cursor SET next_retry_at = ? WHERE source_path = ?", (past, str(failed)))
        conn.commit()

    async def fail_ingest_files(
        paths: list[Path],
        *,
        queued_file_count: int | None = None,
        skipped_file_count: int = 0,
    ) -> SimpleNamespace:
        del paths, queued_file_count, skipped_file_count
        raise AssertionError("no-needed-work retry should not ingest")

    watcher._ingest_files = fail_ingest_files  # type: ignore[assignment,method-assign]
    watcher._needs_work_from_state = lambda *args, **kwargs: False  # type: ignore[method-assign]
    watcher._pending_paths.add(failed)

    asyncio.run(watcher._flush_pending())

    record = watcher._cursor.get_record(failed)
    assert record is not None
    assert record.failure_count == 2
    assert record.next_retry_at is not None
    assert datetime.fromisoformat(record.next_retry_at) > frozen_clock.now()
    assert watcher._pending_paths == set()


@pytest.mark.frozen_clock_modules("polylogue.sources.live.cursor", "polylogue.sources.live.convergence_debt_retry")
def test_hot_insight_convergence_debt_uses_quiet_window_retry(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    cursor = CursorStore(tmp_path / "cursor.sqlite")
    cursor.record_convergence_debt(
        stage="insights",
        subject_type="session_id",
        subject_id="conv-hot",
        error="insights deferred until source quiet",
    )
    frozen_clock.advance(1)
    cursor.record_convergence_debt(
        stage="insights",
        subject_type="session_id",
        subject_id="conv-hot",
        error="insights deferred until source quiet",
    )

    debt = cursor.list_convergence_debt(limit=1)[0]
    retry_at = datetime.fromisoformat(debt.next_retry_at or "")
    failed_at = datetime.fromisoformat(debt.last_failed_at)
    assert debt.status == "deferred"
    assert debt.failure_count == 1
    assert retry_at - failed_at == timedelta(seconds=60)


@pytest.mark.frozen_clock_modules("polylogue.sources.live.cursor", "polylogue.sources.live.convergence_debt_retry")
def test_hot_insight_convergence_debt_advances_after_retry_is_due(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    cursor = CursorStore(tmp_path / "cursor.sqlite")
    cursor.record_convergence_debt(
        stage="insights",
        subject_type="session_id",
        subject_id="conv-hot",
        error="insights deferred until source quiet",
    )
    frozen_clock.advance(61)
    cursor.record_convergence_debt(
        stage="insights",
        subject_type="session_id",
        subject_id="conv-hot",
        error="insights deferred until source quiet",
    )

    debt = cursor.list_convergence_debt(limit=1)[0]
    retry_at = datetime.fromisoformat(debt.next_retry_at or "")
    failed_at = datetime.fromisoformat(debt.last_failed_at)
    assert debt.failure_count == 2
    assert retry_at - failed_at == timedelta(seconds=60)


@pytest.mark.frozen_clock_modules("polylogue.sources.live.cursor", "polylogue.sources.live.convergence_debt_retry")
def test_hot_insight_convergence_debt_uses_source_quiet_deadline(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    source = tmp_path / "active.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    source_mtime = frozen_clock.now().timestamp() - 10
    os.utime(source, (source_mtime, source_mtime))
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    with sqlite3.connect(source_db) as conn:
        raw_id = write_source_raw_session(
            conn,
            origin="codex-session",
            source_path=str(source),
            source_index=0,
            payload=b"{}\n",
            acquired_at_ms=1,
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, message_count, content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("conv-hot", "codex-session", raw_id, 0, b"b" * 32, 1, 1),
        )
        conn.commit()

    cursor = CursorStore(index_db)
    cursor.record_convergence_debt(
        stage="insights",
        subject_type="session_id",
        subject_id="codex-session:conv-hot",
        error="insights deferred until source quiet",
    )

    debt = cursor.list_convergence_debt(limit=1)[0]
    retry_at = datetime.fromisoformat(debt.next_retry_at or "")
    failed_at = datetime.fromisoformat(debt.last_failed_at)
    assert retry_at - failed_at == timedelta(seconds=50)


@pytest.mark.frozen_clock_modules("polylogue.sources.live.cursor", "polylogue.sources.live.convergence_debt_retry")
def test_hot_insight_convergence_debt_uses_archive_source_quiet_deadline(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    source = tmp_path / "active-v1.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    source_mtime = frozen_clock.now().timestamp() - 10
    os.utime(source, (source_mtime, source_mtime))
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index,
                blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("raw-hot-v1", "codex-session", "conv-hot-v1", str(source), 0, b"a" * 32, 2, 1),
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, message_count, content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("conv-hot-v1", "codex-session", "raw-hot-v1", 0, b"b" * 32, 1, 1),
        )

    cursor = CursorStore(index_db)
    cursor.record_convergence_debt(
        stage="insights",
        subject_type="session_id",
        subject_id="codex-session:conv-hot-v1",
        error="insights deferred until source quiet",
    )

    debt = cursor.list_convergence_debt(limit=1)[0]
    retry_at = datetime.fromisoformat(debt.next_retry_at or "")
    failed_at = datetime.fromisoformat(debt.last_failed_at)
    assert retry_at - failed_at == timedelta(seconds=50)
