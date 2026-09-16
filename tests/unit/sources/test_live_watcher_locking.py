"""Lock-contention behavior for the live filesystem watcher."""

from __future__ import annotations

import asyncio
import contextlib
import selectors
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from polylogue.daemon.intake import AdmissionOutcome
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteEvent
from polylogue.operations.intake_adapters import DaemonIntakeContext, FileIntakeAdapter
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.live.watcher import _PARSER_FINGERPRINT


def _make_watcher(tmp_path: Path, root: Path) -> LiveWatcher:
    polylogue = cast(
        Any,
        SimpleNamespace(
            archive_root=tmp_path,
            backend=SimpleNamespace(db_path=tmp_path / "archive.sqlite"),
        ),
    )
    cursor = CursorStore(tmp_path / "archive.sqlite")
    return LiveWatcher(polylogue, (WatchSource(name="test", root=root),), cursor=cursor)


@pytest.mark.parametrize("route", ["append", "full"])
@pytest.mark.uses_real_clock("requires a bounded subprocess exit deadline while the injected writer remains blocked")
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
            # This proof targets the writer bridge's process-exit semantics.
            # Disable the independent prefetch lane so an executor worker
            # cannot determine the subprocess lifetime instead.
            watcher._parse_stage.shutdown()
            watcher._parse_stage = None
            watcher._batch_processor._parse_stage = None
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
            # The injected writer remains blocked through interpreter
            # termination. A non-daemon bridge thread would pin this
            # subprocess after the loop closes.
            watcher.stop()
            print("ready-for-interpreter-exit", flush=True)

        asyncio.run(main())
        """
    )

    process = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[3],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        assert selector.select(timeout=30.0), "writer subprocess did not reach its exit boundary"
        assert process.stdout.readline().strip() == "ready-for-interpreter-exit"
        stdout, stderr = process.communicate(timeout=2.0)
    except BaseException:
        process.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.communicate(timeout=2.0)
        raise
    finally:
        selector.close()

    assert process.returncode == 0, stdout + stderr


@pytest.mark.asyncio
async def test_a_locked_archive_leaves_the_whole_page_retryable_and_unacknowledged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lock race is the page's failure, and the dispatcher owns the retry.

    Anti-vacuity: acknowledge a failed item (report DUPLICATE or ADMITTED on
    an ingest that raised) and the resume cursor advances past a file that was
    never admitted -- the assertions on the outcomes and on ``_after`` below
    both go red.
    """

    root = tmp_path / "src"
    root.mkdir()
    first = root / "a-session.jsonl"
    second = root / "b-session.jsonl"
    first.write_text('{"role":"user","content":"a"}\n')
    second.write_text('{"role":"user","content":"b"}\n')
    watcher = _make_watcher(tmp_path, root)
    calls: list[list[Path]] = []

    async def locked_ingest(paths: list[Path], **_kwargs: object) -> None:
        calls.append(list(paths))
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(watcher, "_ingest_files", locked_ingest)
    source = watcher._sources[0]
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=watcher._sources),
        source,
    )
    page = await adapter.discover(limit=8)
    assert {Path(cast(Any, item.payload)) for item in page} == {first, second}

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        await adapter.admit_page(page)

    # One page, one batch attempt -- not one per file.
    assert calls == [[first, second]]
    assert adapter._after is None


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
        max_pass_seconds: float | None = None,
        whole_archive_convergence: bool = True,
        **_kwargs: object,
    ) -> None:
        del queued_file_count, skipped_file_count, max_pass_seconds, whole_archive_convergence
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
        if event.phase == "queued" and event.actor.startswith("watcher."):
            watcher_queued.set()

    coordinator = DaemonWriteCoordinator(observer=observe)
    polylogue = cast(
        Any,
        SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=tmp_path / "index.db")),
    )
    watcher = LiveWatcher(
        polylogue,
        (WatchSource(name="test", root=root),),
        write_coordinator=coordinator,
    )
    assert not (tmp_path / "ops.db").exists()
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=watcher._sources),
        watcher._sources[0],
    )
    page = await adapter.discover(limit=8)
    monkeypatch.setattr(watcher, "_needs_work_from_state", lambda *_args, **_kwargs: False)
    maintenance_entered = asyncio.Event()
    release_maintenance = asyncio.Event()

    async def maintenance() -> None:
        maintenance_entered.set()
        await release_maintenance.wait()

    maintenance_task = asyncio.create_task(coordinator.run("maintenance.raw_materialization", maintenance))
    await maintenance_entered.wait()
    flush_task = asyncio.create_task(adapter.admit_page(page))
    await watcher_queued.wait()

    assert not (tmp_path / "ops.db").exists()
    release_maintenance.set()
    outcomes = await flush_task
    assert [result.outcome for result in outcomes.values()] == [AdmissionOutcome.DUPLICATE]
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
        parser_fingerprint=_PARSER_FINGERPRINT,
        content_fingerprint="base",
        st_dev=stat.st_dev,
        st_ino=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )
    source.write_bytes(complete + b'{"role":"assistant"')
    watcher_queued = asyncio.Event()
    deferral_attempted = asyncio.Event()

    def observe(event: DaemonWriteEvent) -> None:
        if event.phase == "queued" and event.actor.startswith("watcher."):
            watcher_queued.set()

    original_set = cursor.set

    def observed_set(*args: Any, **kwargs: Any) -> Any:
        deferral_attempted.set()
        return original_set(*args, **kwargs)

    monkeypatch.setattr(cursor, "set", observed_set)
    coordinator = DaemonWriteCoordinator(observer=observe)
    watcher = _make_watcher(tmp_path, root)
    watcher._cursor = cursor
    watcher._batch_processor._cursor = cursor
    watcher._write_coordinator = coordinator
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(archive_root=tmp_path, watcher=watcher, sources=watcher._sources),
        watcher._sources[0],
    )
    page = await adapter.discover(limit=8)
    maintenance_entered = asyncio.Event()
    release_maintenance = asyncio.Event()

    async def maintenance() -> None:
        maintenance_entered.set()
        await release_maintenance.wait()

    maintenance_task = asyncio.create_task(coordinator.run("maintenance.raw_materialization", maintenance))
    await maintenance_entered.wait()
    flush_task = asyncio.create_task(adapter.admit_page(page))
    queued_wait = asyncio.create_task(watcher_queued.wait())
    deferral_wait = asyncio.create_task(deferral_attempted.wait())
    done, pending = await asyncio.wait((queued_wait, deferral_wait), return_when=asyncio.FIRST_COMPLETED)

    assert queued_wait in done
    assert deferral_wait in pending
    before_release = cursor.get_record(source)
    assert before_release is not None
    assert before_release.byte_size == len(complete)
    release_maintenance.set()
    assert await flush_task
    await maintenance_task
    await deferral_wait
    record = cursor.get_record(source)
    assert record is not None
    assert record.byte_size == source.stat().st_size
    assert record.byte_offset == len(complete)


def test_a_wrong_shaped_coordinator_cannot_silently_ungate_writes(tmp_path: Path) -> None:
    """An injected coordinator that lacks ``run`` fails loudly, not silently.

    Anti-vacuity: restore the ``getattr(self._write_coordinator, "run", None)``
    dispatch with a ``callable()`` fallback and this goes green while the write
    proceeds ungated -- which is how a test double's shape could defeat the
    single-writer invariant in production (polylogue-8qm4k).
    """

    class NotACoordinator:
        """Has neither ``run`` nor ``run_sync``."""

    ran: list[str] = []

    async def operation() -> None:
        ran.append("wrote")

    (tmp_path / "src").mkdir()
    watcher = LiveWatcher(
        cast(Any, SimpleNamespace(archive_root=tmp_path)),
        (WatchSource(name="src", root=tmp_path / "src"),),
        write_coordinator=cast(Any, NotACoordinator()),
    )
    with pytest.raises(AttributeError):
        asyncio.run(watcher._run_coordinated("test.actor", operation))

    assert ran == []


def test_an_absent_coordinator_remains_an_explicit_standalone_opt_out(tmp_path: Path) -> None:
    """``None`` still means standalone, so the guard above is not overreach."""
    ran: list[str] = []

    async def operation() -> None:
        ran.append("wrote")

    (tmp_path / "src").mkdir()
    watcher = LiveWatcher(
        cast(Any, SimpleNamespace(archive_root=tmp_path)),
        (WatchSource(name="src", root=tmp_path / "src"),),
        write_coordinator=None,
    )
    asyncio.run(watcher._run_coordinated("test.actor", operation))

    assert ran == ["wrote"]
