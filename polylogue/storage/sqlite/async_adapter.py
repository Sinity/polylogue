"""Storage-owned async adapter for synchronous archive read operations."""

from __future__ import annotations

import asyncio
import atexit
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import Context, copy_context
from dataclasses import dataclass
from threading import Lock, current_thread
from typing import TypeVar

from polylogue.core.sql_settlement import NativeSQLSettlementEvidence, SQLSettlementRetry, settle_native_sql

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetainedReadSQLSettlement:
    thread_name: str
    owner_count: int
    failure_types: tuple[str, ...]


@dataclass(slots=True)
class _ReadSQLSettlement:
    retry: SQLSettlementRetry
    evidence: RetainedReadSQLSettlement


class ArchiveReadAsyncAdapter:
    """Run synchronous archive reads on storage-owned worker threads.

    Admission, cancellation, snapshots, and SQLite interruption stay in the
    caller's production read controller. This adapter owns only worker
    selection and context propagation, keeping the async boundary out of the
    default executor shared with unrelated application work.
    """

    def __init__(self, *, max_workers: int = 4) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="polylogue-archive-read")
        self._closed = False
        self._lock = Lock()
        self._sql_settlements: dict[int, _ReadSQLSettlement] = {}

    async def run(
        self,
        operation: Callable[[], T],
        *,
        on_submitted: Callable[[], None] | None = None,
        on_completed: Callable[[], None] | None = None,
    ) -> T:
        """Await one already-admitted bounded submission with context vars.

        The read controller performs workload-class admission before this
        adapter receives work. This executor therefore contains only active
        reads, never scans blocked before admission.

        ``on_submitted`` runs synchronously after the executor accepts the
        operation. ``on_completed`` runs from the executor future's done
        callback, including when a queued future is canceled by
        ``shutdown(cancel_futures=True)``. Resource owners can therefore tie
        release to the real executor future rather than to the cancellable
        asyncio wrapper.
        """
        context = copy_context()
        loop = asyncio.get_running_loop()
        # Keep the adapter lock across the closed check and submission.  Without
        # this critical section, ``close()`` can pass ``shutdown()`` between
        # the check and ``submit()``, turning a valid admitted read into a
        # rejected operation after its admission lease has been transferred.
        with self._lock:
            if self._closed:
                raise RuntimeError("archive read adapter is closed")
            concurrent_future = self._executor.submit(self._run_in_context, context, operation)
        if on_submitted is not None:
            on_submitted()
        if on_completed is not None:
            concurrent_future.add_done_callback(lambda _future: on_completed())
        return await asyncio.wrap_future(concurrent_future, loop=loop)

    def _run_in_context(self, context: Context, operation: Callable[[], T]) -> T:
        def run() -> T:
            failure: BaseException | None = None
            result: T
            try:
                result = operation()
            except BaseException as error:
                failure = error
            settlement_failure = self._settle_native_sql()
            if failure is not None:
                if settlement_failure is not None:
                    failure.add_note(f"native SQL cleanup also failed: {type(settlement_failure).__name__}")
                raise failure
            if settlement_failure is not None:
                raise settlement_failure
            return result

        return context.run(run)

    def _settle_native_sql(self) -> BaseException | None:
        worker = current_thread()
        retry = SQLSettlementRetry()
        retained: _ReadSQLSettlement | None = None

        def on_pending(pending: NativeSQLSettlementEvidence) -> None:
            nonlocal retained
            evidence = RetainedReadSQLSettlement(worker.name, pending.owner_count, pending.failure_types)
            with self._lock:
                if retained is None:
                    retained = _ReadSQLSettlement(retry, evidence)
                    self._sql_settlements[id(worker)] = retained
                    closing = self._closed
                else:
                    retained.evidence = evidence
                    closing = False
            if closing:
                retry.request()

        def on_settled() -> None:
            with self._lock:
                self._sql_settlements.pop(id(worker), None)

        return settle_native_sql(retry=retry, on_pending=on_pending, on_settled=on_settled)

    def retained_sql_settlements(self) -> tuple[RetainedReadSQLSettlement, ...]:
        with self._lock:
            return tuple(entry.evidence for entry in self._sql_settlements.values())

    def retry_sql_settlement(self) -> None:
        """Wake retained creator workers without releasing read admission."""
        with self._lock:
            retained = tuple(self._sql_settlements.values())
        for entry in retained:
            entry.retry.request()

    def close(self) -> None:
        """Drain actual workers; failed SQL cleanup retains physical ownership."""
        with self._lock:
            self._closed = True
        self.retry_sql_settlement()
        self._executor.shutdown(wait=True, cancel_futures=True)


_default_adapter = ArchiveReadAsyncAdapter()
atexit.register(_default_adapter.close)


def default_archive_read_async_adapter() -> ArchiveReadAsyncAdapter:
    """Return the process-owned adapter shared by archive read transactions."""
    return _default_adapter


__all__ = ["ArchiveReadAsyncAdapter", "default_archive_read_async_adapter"]
