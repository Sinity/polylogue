from __future__ import annotations

import asyncio
import hashlib
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
from polylogue.sources.live.batch_support import _AppendPlan, encode_cursor_hash_authority
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
    unchanged_digest = hashlib.sha256(unchanged.read_bytes()).hexdigest()
    cursor.set(
        unchanged,
        stat.st_size,
        byte_offset=stat.st_size,
        last_complete_newline=stat.st_size,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint="already-known",
        # Modern cursor authority (#2710): the hot stat-match skip only trusts
        # cursors carrying an encoded prefix/tail digest bound to this file's
        # ctime. A legacy cursor without one deliberately takes one full
        # route to (re-)establish that authority instead of hot-skipping.
        tail_hash=encode_cursor_hash_authority(
            unchanged_digest,
            unchanged_digest,
            ctime_ns=stat.st_ctime_ns,
        ),
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
    index_db = tmp_path / "index.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    with sqlite3.connect(source_db) as conn:
        # Reconciliation (#2710) now re-verifies the archived blob hash
        # against the live file's real bytes, so this must be the file's
        # actual digest rather than an arbitrary placeholder.
        blob_hash = hashlib.sha256(archived.read_bytes()).digest()
        raw_id = write_source_raw_session_blob_ref(
            conn,
            origin="codex-session",
            source_path=str(archived),
            source_index=0,
            blob_hash=blob_hash,
            blob_size=archived.stat().st_size,
            acquired_at_ms=1,
            native_id="archived",
        )
        # Archive reconciliation (#2676) only trusts a raw row that is both
        # parsed and materialized into a session, so mark it parsed here.
        conn.execute("UPDATE raw_sessions SET parsed_at_ms = ? WHERE raw_id = ?", (1, raw_id))
        conn.commit()
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, message_count, content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("archived", "codex-session", raw_id, 1, b"c" * 32, 1, 1),
        )
        conn.commit()
    _write_archive_blob(tmp_path, blob_hash, archived.read_bytes())
    watcher = LiveWatcher(cast(Any, polylogue), (WatchSource(name="codex", root=root),), cursor=cursor)

    plan = watcher._plan_catch_up(watcher._scan_catch_up_candidates([root]))
    record = cursor.get_record(archived)

    assert plan.needed == ()
    assert plan.skipped_file_count == 1
    assert record is not None
    assert record.byte_size == archived.stat().st_size
    assert record.content_fingerprint == blob_hash.hex()
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
    index_db = tmp_path / "index.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    with sqlite3.connect(source_db) as conn:
        # Reconciliation (#2710) now re-verifies the archived blob hash
        # against the live file's real bytes, so this must be the file's
        # actual digest rather than an arbitrary placeholder.
        blob_hash = hashlib.sha256(archived.read_bytes()).digest()
        raw_id = write_source_raw_session_blob_ref(
            conn,
            origin="chatgpt-export",
            source_path=str(archived),
            source_index=0,
            blob_hash=blob_hash,
            blob_size=archived.stat().st_size,
            acquired_at_ms=1,
            native_id="capture",
        )
        # Archive reconciliation (#2676) only trusts a raw row that is both
        # parsed and materialized into a session, so mark it parsed here.
        conn.execute("UPDATE raw_sessions SET parsed_at_ms = ? WHERE raw_id = ?", (1, raw_id))
        conn.commit()
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, message_count, content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("capture", "chatgpt-export", raw_id, 1, b"d" * 32, 1, 1),
        )
        conn.commit()
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
    old_content = prefix + b'{"type":"message","payload":{"role":"user","content":"old"}}\n'
    source.write_bytes(old_content)
    old_offset = source.stat().st_size
    with source.open("ab") as handle:
        handle.write(b'{"type":"message","payload":{"role":"assistant","content":"new"}}\n')
    stat = source.stat()
    old_content_digest = hashlib.sha256(old_content).hexdigest()

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
        # Modern cursor authority (#2710): append planning only trusts a
        # cursor carrying an encoded accepted-prefix digest. A legacy cursor
        # without one is correctly refused the append route.
        tail_hash=encode_cursor_hash_authority(
            old_content_digest,
            old_content_digest,
            ctime_ns=stat.st_ctime_ns,
        ),
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
    # #3539 (polylogue-u19l) retired splicing a synthetic session_meta header
    # into a Codex append payload before hashing/storing it -- the stored
    # blob must stay a literal byte-slice of the live file so live-source
    # byte-identity re-verification stays possible. Recovered identity now
    # flows as a sidecar hint (native_id_hint) instead, applied as the
    # parser's fallback_id at replay time.
    assert plan.native_id_hint == "conv-hot"
    assert plan.payload == source.read_bytes()[old_offset:]


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
    whole_archive_flags: list[bool] = []
    retry_scan_calls: list[int] = []

    async def fake_ingest_files(
        paths: list[Path],
        *,
        queued_file_count: int | None = None,
        skipped_file_count: int = 0,
        whole_archive_convergence: bool = True,
    ) -> None:
        calls.append((paths, queued_file_count, skipped_file_count))
        whole_archive_flags.append(whole_archive_convergence)

    watcher._ingest_files = fake_ingest_files  # type: ignore[assignment,method-assign]
    watcher._schedule_failed_retry_scan = lambda: retry_scan_calls.append(len(calls))  # type: ignore[method-assign]

    asyncio.run(watcher._catch_up([root]))

    assert [paths for paths, _queued, _skipped in calls] == [files[:2], files[2:4], files[4:]]
    assert calls[0][1:] == (5, 0)
    assert calls[1][1:] == (2, 0)
    assert calls[2][1:] == (1, 0)
    # Whole-archive convergence stages run once, on the last chunk.
    assert whole_archive_flags == [False, False, True]
    assert retry_scan_calls == [3]


def test_catch_up_ingests_a_cold_backlog_without_a_recent_source(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    """Anti-vacuity: breaking out of the priority loop on an empty hot group skips the whole backlog."""
    root = tmp_path / "src"
    root.mkdir()
    historical = [root / f"historical-{index}.jsonl" for index in range(2)]
    now = 1_800_000_000.0
    stale = now - live_watcher._CATCH_UP_HOT_FILE_AGE_S - 1
    for path in historical:
        path.write_text('{"role":"user","content":"old"}\n')
        os.utime(path, (stale, stale))
    watcher = LiveWatcher(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=None)),
        (WatchSource(name="test", root=root),),
    )
    frozen_clock.set_time(now)

    calls: list[list[Path]] = []

    async def fake_ingest_files(paths: list[Path], **_kwargs: object) -> None:
        calls.append(paths)

    watcher._ingest_files = fake_ingest_files  # type: ignore[assignment,method-assign]

    asyncio.run(watcher._catch_up([root]))

    assert calls == [historical]


def test_catch_up_ingests_recent_source_before_historical_backlog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_clock: FrozenClock,
) -> None:
    """A live session does not wait behind a full historical catch-up plan."""
    root = tmp_path / "src"
    root.mkdir()
    historical = root / "historical.jsonl"
    current = root / "current.jsonl"
    historical.write_text('{"role":"user","content":"old"}\n')
    current.write_text('{"role":"user","content":"live"}\n')
    now = 1_800_000_000.0
    os.utime(
        historical, (now - live_watcher._CATCH_UP_HOT_FILE_AGE_S - 1, now - live_watcher._CATCH_UP_HOT_FILE_AGE_S - 1)
    )
    os.utime(current, (now, now))
    watcher = LiveWatcher(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=None)),
        (WatchSource(name="test", root=root),),
    )
    frozen_clock.set_time(now)
    monkeypatch.setattr(live_watcher, "_CATCH_UP_MAX_BATCH_FILES", 1)

    calls: list[list[Path]] = []

    async def fake_ingest_files(paths: list[Path], **_kwargs: object) -> None:
        calls.append(paths)

    watcher._ingest_files = fake_ingest_files  # type: ignore[assignment,method-assign]

    asyncio.run(watcher._catch_up([root]))

    assert calls == [[current], [historical]]


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
def test_derived_convergence_debt_uses_exponential_retry(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    cursor = CursorStore(tmp_path / "cursor.sqlite")
    cursor.record_convergence_debt(
        stage="derived",
        subject_type="session_id",
        subject_id="conv-hot",
        error="derived stage returned False",
        deferred=True,
    )
    frozen_clock.advance(1)
    cursor.record_convergence_debt(
        stage="derived",
        subject_type="session_id",
        subject_id="conv-hot",
        error="derived stage returned False",
        deferred=True,
    )

    debt = cursor.list_convergence_debt(limit=1)[0]
    retry_at = datetime.fromisoformat(debt.next_retry_at or "")
    failed_at = datetime.fromisoformat(debt.last_failed_at)
    assert debt.status == "deferred"
    assert debt.failure_count == 1
    assert retry_at - failed_at == timedelta(seconds=60)


@pytest.mark.frozen_clock_modules("polylogue.sources.live.cursor", "polylogue.sources.live.convergence_debt_retry")
def test_derived_convergence_debt_advances_after_retry_is_due(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    cursor = CursorStore(tmp_path / "cursor.sqlite")
    cursor.record_convergence_debt(
        stage="derived",
        subject_type="session_id",
        subject_id="conv-hot",
        error="derived stage returned False",
        deferred=True,
    )
    frozen_clock.advance(61)
    cursor.record_convergence_debt(
        stage="derived",
        subject_type="session_id",
        subject_id="conv-hot",
        error="derived stage returned False",
        deferred=True,
    )

    debt = cursor.list_convergence_debt(limit=1)[0]
    retry_at = datetime.fromisoformat(debt.next_retry_at or "")
    failed_at = datetime.fromisoformat(debt.last_failed_at)
    assert debt.failure_count == 2
    assert retry_at - failed_at == timedelta(seconds=120)


def _seed_healthy_hot_skip_cursor(cursor: CursorStore, path: Path, *, source_name: str) -> None:
    """Write a cursor that the byte/fingerprint hot-skip path would trust on its own."""
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cursor.set(
        path,
        stat.st_size,
        byte_offset=stat.st_size,
        last_complete_newline=stat.st_size,
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
        content_fingerprint=digest,
        tail_hash=encode_cursor_hash_authority(digest, digest, ctime_ns=stat.st_ctime_ns),
        source_name=source_name,
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )


def _seed_parsed_source_raw(conn: sqlite3.Connection, *, path: Path, native_id: str, raw_id: str) -> None:
    conn.execute(
        """
        INSERT INTO raw_sessions (
            raw_id, origin, native_id, source_path, source_index,
            blob_hash, blob_size, acquired_at_ms, parsed_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (raw_id, "codex-session", native_id, str(path), 0, b"a" * 32, path.stat().st_size, 1, 1),
    )


def test_catch_up_demotes_hot_skip_cursor_when_index_holds_no_sessions(tmp_path: Path) -> None:
    """polylogue-emx2: a cursor claiming acquisition must not skip when the
    index tier that would corroborate materialization is empty -- the
    signature left by a full index reset/rebuild (Finding 8: 14,879 cursors
    skipped 100% of files against an empty post-reset index)."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "src"
    root.mkdir()
    healthy = root / "healthy.jsonl"
    healthy.write_text('{"role":"user","content":"hello"}\n', encoding="utf-8")

    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    cursor = CursorStore(tmp_path / "cursor.sqlite")
    _seed_healthy_hot_skip_cursor(cursor, healthy, source_name="test")

    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    with sqlite3.connect(source_db) as conn:
        _seed_parsed_source_raw(conn, path=healthy, native_id="healthy", raw_id="raw-healthy")
        conn.commit()
    # index.db intentionally left with zero rows in `sessions` -- the
    # post-reset state this bead targets.

    watcher = LiveWatcher(cast(Any, polylogue), (WatchSource(name="test", root=root),), cursor=cursor)
    candidates = watcher._scan_catch_up_candidates([root])
    plan = watcher._plan_catch_up(candidates)

    assert plan.needed == (healthy,), "a cursor the index cannot corroborate must be demoted to needed, not hot-skipped"
    assert plan.skipped_file_count == 0


def test_catch_up_trusts_hot_skip_cursor_when_index_corroborates_it(tmp_path: Path) -> None:
    """The corroborated companion to the demotion test above: once the raw
    this cursor claims is actually materialized as a session in the index,
    the hot-skip path is trusted again and no re-ingest is scheduled."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    root = tmp_path / "src"
    root.mkdir()
    healthy = root / "healthy.jsonl"
    healthy.write_text('{"role":"user","content":"hello"}\n', encoding="utf-8")

    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    cursor = CursorStore(tmp_path / "cursor.sqlite")
    _seed_healthy_hot_skip_cursor(cursor, healthy, source_name="test")

    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    with sqlite3.connect(source_db) as conn:
        _seed_parsed_source_raw(conn, path=healthy, native_id="healthy", raw_id="raw-healthy")
        conn.commit()
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, message_count, content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("healthy", "codex-session", "raw-healthy", 1, b"c" * 32, 1, 1),
        )
        conn.commit()

    watcher = LiveWatcher(cast(Any, polylogue), (WatchSource(name="test", root=root),), cursor=cursor)
    candidates = watcher._scan_catch_up_candidates([root])
    plan = watcher._plan_catch_up(candidates)

    assert plan.needed == ()
    assert plan.skipped_file_count == 1


def test_catch_up_index_lacks_all_corroboration_detects_empty_index_with_parsed_raw(tmp_path: Path) -> None:
    """Unit-level pin on the cheap global gate itself, independent of the
    higher-level planning flow exercised above."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    with sqlite3.connect(source_db) as source_conn, sqlite3.connect(index_db) as index_conn:
        assert (
            live_watcher.LiveWatcher._index_lacks_all_corroboration(source_conn=source_conn, index_conn=index_conn)
            is False
        ), "no parsed raw material yet -- nothing for the index to corroborate"

        source_conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index,
                blob_hash, blob_size, acquired_at_ms, parsed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("raw-1", "codex-session", "conv-1", "/tmp/does-not-matter.jsonl", 0, b"a" * 32, 2, 1, 1),
        )
        source_conn.commit()
        assert (
            live_watcher.LiveWatcher._index_lacks_all_corroboration(source_conn=source_conn, index_conn=index_conn)
            is True
        ), "parsed raw exists but index.db has zero sessions -- the post-reset signature"

        index_conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, message_count, content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("conv-1", "codex-session", "raw-1", 1, b"b" * 32, 1, 1),
        )
        index_conn.commit()
        assert (
            live_watcher.LiveWatcher._index_lacks_all_corroboration(source_conn=source_conn, index_conn=index_conn)
            is False
        ), "index now shows at least one materialized session -- corroborated again"


# ---------------------------------------------------------------------------
# Halted sources: excluded where work is SELECTED, and published as state
# ---------------------------------------------------------------------------


class _RecordingCoordinator:
    """Write coordinator double that records every lease it hands out."""

    def __init__(self) -> None:
        self.actors: list[str] = []

    async def run(self, actor: str, operation: Any) -> None:
        self.actors.append(actor)
        await operation()

    async def run_sync(self, actor: str, function: Any, /, *args: Any, **kwargs: Any) -> Any:
        self.actors.append(actor)
        return function(*args, **kwargs)


def _stub_metrics(succeeded: int, paths: list[Path]) -> Any:
    from polylogue.sources.live.metrics import LiveBatchMetrics

    offered = sum(path.stat().st_size for path in paths)
    return LiveBatchMetrics(
        queued_file_count=len(paths),
        needed_file_count=len(paths),
        skipped_file_count=0,
        succeeded_file_count=succeeded,
        failed_file_count=0,
        source_group_count=1,
        input_bytes=offered,
        ingested_bytes=offered if succeeded else 0,
        source_payload_read_bytes=offered if succeeded else 0,
        cursor_fingerprint_read_bytes=0,
        ingest_worker_count_max=1,
        append_file_count=0,
        full_file_count=len(paths),
        archive_bytes_before=0,
        archive_bytes_after=0,
        archive_write_bytes_delta=0,
        parse_time_s=0.0,
        convergence_time_s=0.0,
        total_time_s=0.01,
    )


def _two_source_watcher(
    tmp_path: Path,
    *,
    files_per_source: int = 3,
    write_coordinator: object | None = None,
    event_emitter: Any | None = None,
) -> tuple[Any, list[Path], list[Path]]:
    root = tmp_path / "sources"
    (root / "alpha").mkdir(parents=True)
    (root / "beta").mkdir(parents=True)
    alpha_files = []
    beta_files = []
    for index in range(files_per_source):
        alpha = root / "alpha" / f"a{index}.jsonl"
        beta = root / "beta" / f"b{index}.jsonl"
        alpha.write_text('{"type":"user"}\n', encoding="utf-8")
        beta.write_text('{"type":"user"}\n', encoding="utf-8")
        alpha_files.append(alpha)
        beta_files.append(beta)
    polylogue = SimpleNamespace(archive_root=tmp_path, backend=None)
    watcher = LiveWatcher(
        cast(Any, polylogue),
        (
            WatchSource(name="alpha", root=root / "alpha"),
            WatchSource(name="beta", root=root / "beta"),
        ),
        cursor=CursorStore(tmp_path / "cursor.sqlite"),
        write_coordinator=cast(Any, write_coordinator),
        event_emitter=event_emitter,
    )
    return watcher, alpha_files, beta_files


def test_halted_source_is_excluded_where_catch_up_work_is_selected(tmp_path: Path) -> None:
    """A halted source contributes no planned work; its siblings still do.

    Anti-vacuity: delete the ``source_halt`` check in ``_plan_catch_up`` and
    the halted files reappear in ``plan.needed``, which is the rehearsal-11
    shape -- chunks planned for a source that could not ingest anything.
    """
    from polylogue.core.degraded import DegradedReason
    from polylogue.core.source_halts import clear_all_source_halts, set_source_halt

    watcher, alpha_files, beta_files = _two_source_watcher(tmp_path, files_per_source=2)
    roots = [alpha_files[0].parent, beta_files[0].parent]
    clear_all_source_halts()
    try:
        before = watcher._plan_catch_up(watcher._scan_catch_up_candidates(roots))
        assert set(before.needed) == set(alpha_files) | set(beta_files)
        assert before.halted_file_count == 0

        set_source_halt(
            "alpha",
            DegradedReason(code="schema_version_mismatch", message="stale derived tier", derived_only=True),
        )
        after = watcher._plan_catch_up(watcher._scan_catch_up_candidates(roots))
    finally:
        clear_all_source_halts()
        watcher.stop()

    assert set(after.needed) == set(beta_files)
    assert after.halted_file_count == len(alpha_files)
    assert after.halted_sources == ("alpha",)


def test_halted_file_count_is_not_folded_into_the_ordinary_skip_count(tmp_path: Path) -> None:
    """A stopped source must not hide inside "nothing to do" bookkeeping.

    Anti-vacuity: count halted files as ``skipped_file_count`` instead and
    this goes red. That collapse is what let a dead source read as an idle
    one across the whole run.
    """
    from polylogue.core.degraded import DegradedReason
    from polylogue.core.source_halts import clear_all_source_halts, set_source_halt

    watcher, alpha_files, beta_files = _two_source_watcher(tmp_path, files_per_source=2)
    clear_all_source_halts()
    try:
        set_source_halt("alpha", DegradedReason(code="database_layout_mismatch", message="unreadable"))
        plan = watcher._plan_catch_up(watcher._scan_catch_up_candidates([alpha_files[0].parent, beta_files[0].parent]))
    finally:
        clear_all_source_halts()
        watcher.stop()

    assert plan.skipped_file_count == 0
    assert plan.halted_file_count == len(alpha_files)


def test_structural_database_error_halts_only_its_own_source() -> None:
    """The production handler records a per-source halt, not just a global flag.

    Anti-vacuity: drop the ``set_source_halt`` call in
    ``handle_structural_database_error`` and this goes red while the
    process-wide flag still passes -- the exact state that existed before,
    which no planner could act on.
    """
    from polylogue.core.degraded import clear_degraded, is_degraded
    from polylogue.core.errors import SchemaVersionMismatchError
    from polylogue.core.source_halts import clear_all_source_halts, source_halt
    from polylogue.sources.live.dedup import handle_structural_database_error, schema_warning_limiter

    clear_all_source_halts()
    clear_degraded()
    schema_warning_limiter.reset()
    try:
        handle_structural_database_error(
            "alpha",
            SchemaVersionMismatchError("index derived schema identity mismatch", current_version=1, expected_version=2),
        )
        halted = source_halt("alpha")
        healthy = source_halt("beta")
        process_wide = is_degraded()
    finally:
        clear_all_source_halts()
        clear_degraded()
        schema_warning_limiter.reset()

    assert halted is not None
    assert halted.code == "schema_version_mismatch"
    assert healthy is None
    assert process_wide is True


@pytest.mark.asyncio
async def test_source_halted_mid_run_takes_no_further_writer_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once a source halts, no later chunk of it acquires the writer lease.

    The plan is built before the halt exists, so the planning exclusion alone
    cannot help the run in flight -- rehearsal-11 halted at chunk 571 of
    12,150 and the remaining 926 claude-code chunks each still took the lease
    for 1.0-3.1 s to ingest nothing.

    Anti-vacuity: delete the ``source_halt`` filter in the chunk loop and the
    halted source's remaining chunks reappear as ``watcher.catch_up.chunk``
    leases, which is what this counts.
    """
    from polylogue.core.degraded import DegradedReason
    from polylogue.core.source_halts import clear_all_source_halts, set_source_halt

    monkeypatch.setattr(live_watcher, "_CATCH_UP_MAX_BATCH_FILES", 1)
    coordinator = _RecordingCoordinator()
    events: list[tuple[str, dict[str, object]]] = []
    watcher, alpha_files, beta_files = _two_source_watcher(
        tmp_path,
        files_per_source=4,
        write_coordinator=coordinator,
        event_emitter=lambda kind, payload: events.append((kind, payload)),
    )
    ingested_paths: list[Path] = []

    async def fake_ingest(paths: list[Path], **_kwargs: Any) -> Any:
        ingested_paths.extend(paths)
        if any(path in alpha_files for path in paths):
            set_source_halt(
                "alpha",
                DegradedReason(code="schema_version_mismatch", message="stale derived tier", derived_only=True),
            )
        return _stub_metrics(len(paths), paths)

    monkeypatch.setattr(watcher, "_ingest_files", fake_ingest)
    clear_all_source_halts()
    try:
        candidates = watcher._scan_catch_up_candidates([alpha_files[0].parent, beta_files[0].parent])
        await watcher._catch_up_candidates(candidates)
    finally:
        clear_all_source_halts()
        watcher.stop()

    chunk_leases = [actor for actor in coordinator.actors if actor == "watcher.catch_up.chunk"]
    alpha_ingested = [path for path in ingested_paths if path in alpha_files]
    beta_ingested = [path for path in ingested_paths if path in beta_files]
    # Exactly one alpha chunk ran -- the one that discovered the halt.
    assert len(alpha_ingested) == 1
    # The healthy source is untouched by its sibling's halt.
    assert sorted(beta_ingested) == sorted(beta_files)
    assert len(chunk_leases) == 1 + len(beta_files)
    # The halt is durable state, not one log line.
    halt_events = [payload for kind, payload in events if kind == "source_ingest_halted"]
    assert [payload["source_name"] for payload in halt_events] == ["alpha"]
    assert halt_events[0]["code"] == "schema_version_mismatch"


@pytest.mark.asyncio
async def test_process_wide_degrade_mid_run_ends_the_catch_up_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fully degraded daemon stops chunking instead of leasing per chunk.

    Anti-vacuity: remove the ``is_fully_degraded`` check from the chunk loop
    and every remaining chunk acquires the lease again, each one reaching the
    degraded short-circuit that already refuses it.
    """
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    monkeypatch.setattr(live_watcher, "_CATCH_UP_MAX_BATCH_FILES", 1)
    coordinator = _RecordingCoordinator()
    watcher, alpha_files, beta_files = _two_source_watcher(
        tmp_path,
        files_per_source=4,
        write_coordinator=coordinator,
    )

    async def fake_ingest(paths: list[Path], **_kwargs: Any) -> Any:
        set_degraded(DegradedReason(code="database_layout_mismatch", message="index tier unreadable"))
        return _stub_metrics(len(paths), paths)

    monkeypatch.setattr(watcher, "_ingest_files", fake_ingest)
    clear_degraded()
    try:
        candidates = watcher._scan_catch_up_candidates([alpha_files[0].parent, beta_files[0].parent])
        await watcher._catch_up_candidates(candidates)
    finally:
        clear_degraded()
        watcher.stop()

    chunk_leases = [actor for actor in coordinator.actors if actor == "watcher.catch_up.chunk"]
    assert len(chunk_leases) == 1


def _inbox_watcher(root: Path, archive_root: Path) -> LiveWatcher:
    polylogue = SimpleNamespace(archive_root=archive_root, backend=None)
    return LiveWatcher(
        cast(Any, polylogue),
        (WatchSource(name="inbox", root=root),),
        cursor=CursorStore(archive_root / "cursor.sqlite"),
    )


def test_catch_up_scan_reaches_a_symlinked_export_corpus(tmp_path: Path) -> None:
    """A directory symlink under a watch root is a deliberate placement -- the
    archive inbox exposes whole export corpora that way -- so the files behind
    it are catch-up candidates.

    Anti-vacuity: walking with ``followlinks=False``, or resolving ownership
    from the resolved path alone, leaves only ``staged`` and the linked corpus
    is acquired by nothing.
    """
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    staged = inbox / "staged.jsonl"
    staged.write_text('{"role":"user","content":"staged"}\n', encoding="utf-8")
    (corpus / "exported.jsonl").write_text('{"role":"user","content":"exported"}\n', encoding="utf-8")
    (inbox / "chatgpt").symlink_to(corpus, target_is_directory=True)

    watcher = _inbox_watcher(inbox, tmp_path)
    candidates = watcher._scan_catch_up_candidates([inbox])

    assert {candidate.path for candidate in candidates} == {staged, inbox / "chatgpt" / "exported.jsonl"}


def test_catch_up_scan_walks_a_symlink_cycle_once(tmp_path: Path) -> None:
    """Following directory symlinks stays bounded: a link pointing back at its
    own ancestor is not descended a second time.

    Anti-vacuity: drop the walked-identity check and the same file returns
    once per ``loop/nested`` repetition until the walk dies on path length.
    """
    inbox = tmp_path / "inbox"
    nested = inbox / "nested"
    nested.mkdir(parents=True)
    session = nested / "session.jsonl"
    session.write_text('{"role":"user","content":"hello"}\n', encoding="utf-8")
    (nested / "loop").symlink_to(inbox, target_is_directory=True)

    watcher = _inbox_watcher(inbox, tmp_path)
    candidates = watcher._scan_catch_up_candidates([inbox])

    assert [candidate.path for candidate in candidates] == [session]


def test_catch_up_scan_reaches_one_corpus_linked_twice_once(tmp_path: Path) -> None:
    """Two links to one corpus name one acquisition, not two.

    Anti-vacuity: without the shared walked-identity set the file is a
    candidate under both link names, and the daemon leases the writer twice
    to ingest identical bytes.
    """
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "exported.jsonl").write_text('{"role":"user","content":"exported"}\n', encoding="utf-8")
    (inbox / "chatgpt").symlink_to(corpus, target_is_directory=True)
    (inbox / "chatgpt-alias").symlink_to(corpus, target_is_directory=True)

    watcher = _inbox_watcher(inbox, tmp_path)
    candidates = watcher._scan_catch_up_candidates([inbox])

    assert len(candidates) == 1
    assert candidates[0].path.name == "exported.jsonl"
