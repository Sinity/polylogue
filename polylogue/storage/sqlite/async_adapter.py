"""Storage-owned async adapter for synchronous archive read operations."""

from __future__ import annotations

import asyncio
import atexit
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import Context, copy_context
from threading import Lock
from typing import TypeVar

T = TypeVar("T")


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

    async def run(self, operation: Callable[[], T], *, on_submitted: Callable[[], None] | None = None) -> T:
        """Await one already-admitted bounded submission with context vars.

        The read controller performs workload-class admission before this
        adapter receives work. This executor therefore contains only active
        reads, never scans blocked before admission.

        ``on_submitted`` runs synchronously after the executor accepts the
        operation.  Callers that own resources from admission can therefore
        transfer release ownership to the operation itself without a task
        cancellation releasing them while the executor future is still
        running.
        """
        context = copy_context()
        with self._lock:
            if self._closed:
                raise RuntimeError("archive read adapter is closed")
            executor = self._executor
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(executor, _run_in_context, context, operation)
        if on_submitted is not None:
            on_submitted()
        return await future

    def close(self) -> None:
        """Drain worker threads and reject subsequent submissions."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)


def _run_in_context(context: Context, operation: Callable[[], T]) -> T:
    return context.run(operation)


_default_adapter = ArchiveReadAsyncAdapter()
atexit.register(_default_adapter.close)


def default_archive_read_async_adapter() -> ArchiveReadAsyncAdapter:
    """Return the process-owned adapter shared by archive read transactions."""
    return _default_adapter


__all__ = ["ArchiveReadAsyncAdapter", "default_archive_read_async_adapter"]
