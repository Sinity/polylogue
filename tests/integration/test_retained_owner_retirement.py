"""Retained owner physical retirement after a reached body failure."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from contextlib import closing, suppress
from pathlib import Path

import pytest

from polylogue.core.compute_cancel import check_compute_cancelled, compute_cancel
from tests.infra.archive_templates import run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner
from tests.infra.revision_backfill_benchmark import build_independent_raw_corpus


@pytest.mark.asyncio
async def test_streaming_resolver_drains_owned_work_when_the_body_raises(tmp_path: Path) -> None:
    def acquire() -> tuple[str, ...]:
        build_independent_raw_corpus(tmp_path, raw_count=4, avg_payload_bytes=1000)
        with closing(sqlite3.connect(f"file:{tmp_path / 'source.db'}?mode=ro", uri=True)) as source:
            return tuple(str(row[0]) for row in source.execute("SELECT raw_id FROM raw_sessions ORDER BY rowid"))

    raw_ids = await run_archive_fixture_write(tmp_path, acquire)
    primary = RuntimeError("census apply failed")
    workers: tuple[threading.Thread, ...] = ()
    reached = False
    with pytest.raises(RuntimeError) as failure:
        async with prepared_live_convergence_owner(tmp_path) as owner:
            adapter = owner._compute_adapter

            def fail_body() -> None:
                nonlocal reached, workers
                reached = True
                workers = tuple(adapter.executor._threads)
                assert workers and any(worker.is_alive() for worker in workers)
                raise primary

            (await owner.replay_retained_raw_ids(raw_ids, before_publication=fail_body)).require_complete()
    assert failure.value is primary
    assert reached
    assert workers and not any(worker.is_alive() for worker in workers)
    snapshot = adapter.snapshot()
    assert snapshot.used_units == 0
    assert not snapshot.retained_sql_settlements


@pytest.mark.asyncio
async def test_cancelled_retained_publication_retires_original_creator(tmp_path: Path) -> None:
    def acquire() -> tuple[str, ...]:
        build_independent_raw_corpus(tmp_path, raw_count=1, avg_payload_bytes=1000)
        with closing(sqlite3.connect(f"file:{tmp_path / 'source.db'}?mode=ro", uri=True)) as source:
            return tuple(str(row[0]) for row in source.execute("SELECT raw_id FROM raw_sessions ORDER BY rowid"))

    raw_ids = await run_archive_fixture_write(tmp_path, acquire)
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    workers: tuple[threading.Thread, ...] = ()
    cancelled: threading.Event | None = None
    async with prepared_live_convergence_owner(tmp_path) as owner:
        adapter = owner._compute_adapter

        def original_boundary() -> None:
            nonlocal workers, cancelled
            cancelled = compute_cancel.get()
            assert cancelled is not None
            workers = tuple(adapter.executor._threads)
            loop.call_soon_threadsafe(entered.set)
            release.wait()
            check_compute_cancelled()

        operation = asyncio.create_task(owner.replay_retained_raw_ids(raw_ids, before_publication=original_boundary))
        boundary_wait = asyncio.create_task(entered.wait())
        try:
            completed, _ = await asyncio.wait((operation, boundary_wait), return_when=asyncio.FIRST_COMPLETED)
            if operation in completed:
                await operation
                raise AssertionError("retained operation ended before its original cancellation boundary")
            assert workers and any(worker.is_alive() for worker in workers)
            operation.cancel()
            assert cancelled is not None
            with pytest.raises(asyncio.CancelledError):
                while not cancelled.is_set():
                    if operation.done():
                        await operation
                        raise AssertionError("retained operation ended without original cancellation")
                    await asyncio.sleep(0)
                release.set()
                await operation
            assert cancelled.is_set()
        finally:
            release.set()
            boundary_wait.cancel()
            with suppress(asyncio.CancelledError):
                await boundary_wait
            if not operation.done():
                operation.cancel()
                with suppress(asyncio.CancelledError):
                    await operation
    assert workers and not any(worker.is_alive() for worker in workers)
    snapshot = adapter.snapshot()
    assert snapshot.used_units == 0
    assert not snapshot.retained_sql_settlements
