"""Shared process-pool helpers for pipeline services."""

from __future__ import annotations

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor


def _initialize_worker_logging() -> None:
    """Apply the normal CLI log filter inside pool workers.

    Without this, subprocess workers keep structlog's default "notset"
    filtering and can leak debug parser messages into ordinary operator runs.
    """
    from polylogue.logging import configure_logging

    configure_logging(verbose=False)


def process_pool_context() -> multiprocessing.context.BaseContext:
    """Return a start method safe for multi-threaded callers."""
    start_methods = multiprocessing.get_all_start_methods()
    if "forkserver" in start_methods:
        return multiprocessing.get_context("forkserver")
    return multiprocessing.get_context("spawn")


def process_pool_executor(*, max_workers: int) -> ProcessPoolExecutor:
    """Create a process pool that avoids bare fork() in multi-threaded parents."""
    return ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_initialize_worker_logging,
        mp_context=process_pool_context(),
    )


def terminate_process_pool(executor: ProcessPoolExecutor, *, timeout: float = 1.0) -> None:
    """Cancel pending work and bound shutdown of already-running workers."""
    if timeout < 0:
        raise ValueError("process pool termination timeout must be non-negative")
    processes = tuple((getattr(executor, "_processes", None) or {}).values())
    executor.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(timeout=0.1)


__all__ = [
    "_initialize_worker_logging",
    "process_pool_context",
    "process_pool_executor",
    "terminate_process_pool",
]
