"""Shared process-pool helpers for pipeline services."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor


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
        mp_context=process_pool_context(),
    )


__all__ = [
    "process_pool_context",
    "process_pool_executor",
]
