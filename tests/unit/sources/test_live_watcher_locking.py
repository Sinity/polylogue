"""Lock-contention behavior for the live filesystem watcher."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteEvent
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.cursor import CursorStore


def _make_watcher(tmp_path: Path, root: Path, *, debounce_s: float = 0.01) -> LiveWatcher:
    polylogue = cast(
        Any,
        SimpleNamespace(
            archive_root=tmp_path,
            backend=SimpleNamespace(db_path=tmp_path / "archive.sqlite"),
        ),
    )
    cursor = CursorStore(tmp_path / "archive.sqlite")
    return LiveWatcher(polylogue, (WatchSource(name="test", root=root),), debounce_s=debounce_s, cursor=cursor)


@pytest.mark.asyncio
async def test_flush_pending_requeues_when_archive_is_busy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "src"
    root.mkdir()
    source = root / "session.jsonl"
    source.write_text('{"role":"user","content":"a"}\n')
    watcher = _make_watcher(tmp_path, root, debounce_s=0)
    watcher._pending_paths.add(source)
    calls: list[tuple[list[Path], int | None, int]] = []

    async def locked_ingest(
        paths: list[Path],
        *,
        queued_file_count: int | None = None,
        skipped_file_count: int = 0,
    ) -> None:
        calls.append((paths, queued_file_count, skipped_file_count))
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(watcher, "_ingest_files", locked_ingest)

    flushed = await watcher._flush_pending()

    assert flushed is True
    assert calls == [([source], 1, 0)]
    assert watcher._pending_paths == {source}


@pytest.mark.asyncio
async def test_flush_pending_reraises_unexpected_sqlite_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "src"
    root.mkdir()
    source = root / "session.jsonl"
    source.write_text('{"role":"user","content":"a"}\n')
    watcher = _make_watcher(tmp_path, root, debounce_s=0)
    watcher._pending_paths.add(source)

    async def failing_ingest(
        paths: list[Path],
        *,
        queued_file_count: int | None = None,
        skipped_file_count: int = 0,
    ) -> None:
        del paths, queued_file_count, skipped_file_count
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(watcher, "_ingest_files", failing_ingest)

    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        await watcher._flush_pending()


@pytest.mark.asyncio
async def test_ingest_files_serializes_batch_processor_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "src"
    root.mkdir()
    first = root / "first.jsonl"
    second = root / "second.jsonl"
    watcher = _make_watcher(tmp_path, root)
    first_started = asyncio.Event()
    allow_first_finish = asyncio.Event()
    second_entered = asyncio.Event()
    active = 0
    max_active = 0
    calls: list[Path] = []

    async def ingest_files(
        paths: list[Path],
        *,
        queued_file_count: int | None = None,
        skipped_file_count: int = 0,
    ) -> None:
        del queued_file_count, skipped_file_count
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        current = paths[0]
        calls.append(current)
        if current == first:
            first_started.set()
            await allow_first_finish.wait()
        else:
            second_entered.set()
        active -= 1

    monkeypatch.setattr(watcher._batch_processor, "ingest_files", ingest_files)

    first_task = asyncio.create_task(watcher._ingest_files([first]))
    await first_started.wait()
    second_task = asyncio.create_task(watcher._ingest_files([second]))
    await asyncio.sleep(0.02)

    assert not second_entered.is_set()
    allow_first_finish.set()
    await asyncio.gather(first_task, second_task)

    assert calls == [first, second]
    assert max_active == 1


@pytest.mark.asyncio
async def test_watcher_queues_behind_daemon_maintenance_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "src"
    root.mkdir()
    source = root / "session.jsonl"
    source.write_text('{"role":"user","content":"a"}\n')
    watcher_queued = asyncio.Event()

    def observe(event: DaemonWriteEvent) -> None:
        if event.phase == "queued" and event.actor == "watcher.live_ingest":
            watcher_queued.set()

    coordinator = DaemonWriteCoordinator(observer=observe)
    watcher = _make_watcher(tmp_path, root)
    watcher._write_coordinator = coordinator
    maintenance_entered = asyncio.Event()
    release_maintenance = asyncio.Event()
    ingest_entered = asyncio.Event()

    async def maintenance() -> None:
        maintenance_entered.set()
        await release_maintenance.wait()

    async def ingest_files(*_args: object, **_kwargs: object) -> None:
        ingest_entered.set()

    monkeypatch.setattr(watcher._batch_processor, "ingest_files", ingest_files)
    maintenance_task = asyncio.create_task(coordinator.run("maintenance.raw_materialization", maintenance))
    await maintenance_entered.wait()
    watcher_task = asyncio.create_task(watcher._ingest_files([source]))
    await watcher_queued.wait()

    assert not ingest_entered.is_set()
    release_maintenance.set()
    await asyncio.gather(maintenance_task, watcher_task)
    assert ingest_entered.is_set()
