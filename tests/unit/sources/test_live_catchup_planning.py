from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.cursor import CursorStore
from tests.infra.frozen_clock import FrozenClock


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

    async def fake_ingest_files(
        paths: list[Path],
        *,
        queued_file_count: int | None = None,
        skipped_file_count: int = 0,
    ) -> None:
        calls.append((paths, queued_file_count, skipped_file_count))

    watcher._ingest_files = fake_ingest_files  # type: ignore[assignment,method-assign]

    asyncio.run(watcher._catch_up([root]))

    assert [paths for paths, _queued, _skipped in calls] == [files[:2], files[2:4], files[4:]]
    assert calls[0][1:] == (5, 0)
    assert calls[1][1:] == (2, 0)
    assert calls[2][1:] == (1, 0)


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
    with sqlite3.connect(tmp_path / "polylogue.db") as conn:
        conn.execute("UPDATE live_cursor SET next_retry_at = ? WHERE source_path = ?", (past, str(due)))
        conn.execute("UPDATE live_cursor SET next_retry_at = ? WHERE source_path = ?", (future, str(waiting)))
        conn.commit()

    async def run_scan() -> None:
        watcher._schedule_failed_retry_scan()
        watcher.cancel_pending()

    asyncio.run(run_scan())

    assert watcher._pending_paths == {due}


@pytest.mark.frozen_clock_modules("polylogue.sources.live.cursor")
def test_hot_insight_convergence_debt_uses_quiet_window_retry(
    tmp_path: Path,
    frozen_clock: FrozenClock,
) -> None:
    cursor = CursorStore(tmp_path / "cursor.sqlite")
    cursor.record_convergence_debt(
        stage="insights",
        subject_type="conversation_id",
        subject_id="conv-hot",
        error="insights deferred until source quiet",
    )
    frozen_clock.advance(1)
    cursor.record_convergence_debt(
        stage="insights",
        subject_type="conversation_id",
        subject_id="conv-hot",
        error="insights deferred until source quiet",
    )

    debt = cursor.list_convergence_debt(limit=1)[0]
    retry_at = datetime.fromisoformat(debt.next_retry_at or "")
    failed_at = datetime.fromisoformat(debt.last_failed_at)
    assert debt.failure_count == 2
    assert retry_at - failed_at == timedelta(seconds=60)
