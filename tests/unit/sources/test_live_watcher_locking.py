"""Lock-contention behavior for the live filesystem watcher."""

from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import sys
import textwrap
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


@pytest.mark.parametrize("route", ["append", "full"])
def test_real_watcher_writer_routes_cannot_pin_process_exit(route: str) -> None:
    script = textwrap.dedent(
        f"""
        import asyncio
        import contextlib
        import tempfile
        import threading
        from pathlib import Path
        from types import SimpleNamespace

        from polylogue.daemon.write_coordinator import DaemonWriteCoordinator
        from polylogue.sources.live import LiveWatcher, WatchSource
        from polylogue.sources.live.batch_support import _AppendPlan
        from polylogue.sources.live.cursor import CursorStore

        async def main() -> None:
            root = Path(tempfile.mkdtemp(prefix="polylogue-watcher-exit-"))
            source_root = root / "sessions"
            source_root.mkdir()
            path = source_root / "session.jsonl"
            path.write_text('{{"type":"session_meta","payload":{{"id":"exit-proof"}}}}\\n')
            coordinator = DaemonWriteCoordinator()
            cursor = CursorStore(root / "index.db")
            polylogue = SimpleNamespace(archive_root=root, backend=SimpleNamespace(db_path=root / "index.db"))
            watcher = LiveWatcher(
                polylogue,
                (WatchSource(name="codex", root=source_root),),
                cursor=cursor,
                write_coordinator=coordinator,
            )
            started = threading.Event()

            def stuck(*args, **kwargs):
                started.set()
                threading.Event().wait()

            if {route!r} == "append":
                stat = path.stat()
                watcher._batch_processor._append_plan = lambda *args, **kwargs: _AppendPlan(
                    path=path,
                    source_name="codex",
                    start_offset=0,
                    last_complete_newline=stat.st_size,
                    stat_size=stat.st_size,
                    st_dev=stat.st_dev,
                    st_ino=stat.st_ino,
                    mtime_ns=stat.st_mtime_ns,
                    payload=path.read_bytes(),
                    payload_hash="exit-proof",
                    cursor_fingerprint="base",
                    bytes_read=stat.st_size,
                )
                watcher._batch_processor._ingest_append_plans = stuck
            else:
                watcher._batch_processor._append_plan = lambda *args, **kwargs: None
                watcher._batch_processor._ingest_full_paths_sync = stuck

            caller = asyncio.create_task(watcher._ingest_files([path]))
            while not started.is_set():
                await asyncio.sleep(0.001)
            caller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await caller
            assert await coordinator.shutdown(timeout=0.01) is False

        asyncio.run(main())
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[3],
        capture_output=True,
        text=True,
        timeout=6.0,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


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


@pytest.mark.asyncio
async def test_default_cursor_initialization_waits_for_batch_writer_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "src"
    root.mkdir()
    source = root / "session.jsonl"
    source.write_text('{"role":"user","content":"a"}\n')
    watcher_queued = asyncio.Event()

    def observe(event: DaemonWriteEvent) -> None:
        if event.phase == "queued" and event.actor == "watcher.live_batch":
            watcher_queued.set()

    coordinator = DaemonWriteCoordinator(observer=observe)
    polylogue = cast(
        Any,
        SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db")),
    )
    watcher = LiveWatcher(
        polylogue,
        (WatchSource(name="test", root=root),),
        debounce_s=0,
        write_coordinator=coordinator,
    )
    assert not (tmp_path / "ops.db").exists()
    watcher._pending_paths.add(source)
    monkeypatch.setattr(watcher, "_needs_work_from_state", lambda *_args, **_kwargs: False)
    maintenance_entered = asyncio.Event()
    release_maintenance = asyncio.Event()

    async def maintenance() -> None:
        maintenance_entered.set()
        await release_maintenance.wait()

    maintenance_task = asyncio.create_task(coordinator.run("maintenance.raw_materialization", maintenance))
    await maintenance_entered.wait()
    flush_task = asyncio.create_task(watcher._flush_pending())
    await watcher_queued.wait()

    assert not (tmp_path / "ops.db").exists()
    release_maintenance.set()
    assert await flush_task is True
    await maintenance_task
    assert (tmp_path / "ops.db").exists()


@pytest.mark.asyncio
async def test_incomplete_append_deferral_cannot_write_before_batch_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "src"
    root.mkdir()
    source = root / "session.jsonl"
    complete = b'{"role":"user","content":"a"}\n'
    source.write_bytes(complete)
    cursor = CursorStore(tmp_path / "index.db")
    stat = source.stat()
    cursor.set(
        source,
        len(complete),
        byte_offset=len(complete),
        last_complete_newline=len(complete),
        parser_fingerprint="live-batched-v2",
        content_fingerprint="base",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )
    source.write_bytes(complete + b'{"role":"assistant"')
    watcher_queued = asyncio.Event()
    deferral_attempted = asyncio.Event()

    def observe(event: DaemonWriteEvent) -> None:
        if event.phase == "queued" and event.actor == "watcher.live_batch":
            watcher_queued.set()

    original_set = cursor.set

    def observed_set(*args: Any, **kwargs: Any) -> None:
        deferral_attempted.set()
        original_set(*args, **kwargs)

    monkeypatch.setattr(cursor, "set", observed_set)
    coordinator = DaemonWriteCoordinator(observer=observe)
    watcher = _make_watcher(tmp_path, root)
    watcher._cursor = cursor
    watcher._batch_processor._cursor = cursor
    watcher._write_coordinator = coordinator
    watcher._pending_paths.add(source)
    maintenance_entered = asyncio.Event()
    release_maintenance = asyncio.Event()

    async def maintenance() -> None:
        maintenance_entered.set()
        await release_maintenance.wait()

    maintenance_task = asyncio.create_task(coordinator.run("maintenance.raw_materialization", maintenance))
    await maintenance_entered.wait()
    flush_task = asyncio.create_task(watcher._flush_pending())
    queued_wait = asyncio.create_task(watcher_queued.wait())
    deferral_wait = asyncio.create_task(deferral_attempted.wait())
    done, pending = await asyncio.wait((queued_wait, deferral_wait), return_when=asyncio.FIRST_COMPLETED)

    assert queued_wait in done
    assert deferral_wait in pending
    before_release = cursor.get_record(source)
    assert before_release is not None
    assert before_release.byte_size == len(complete)
    release_maintenance.set()
    assert await flush_task is True
    await maintenance_task
    await deferral_wait
    record = cursor.get_record(source)
    assert record is not None
    assert record.byte_size == source.stat().st_size
    assert record.byte_offset == len(complete)
