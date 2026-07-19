"""Shared process-pool helpers for pipeline services."""

from __future__ import annotations

import multiprocessing
import os
import sys
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
    """Return a start method safe for multi-threaded callers.

    ``forkserver`` preloads ``__main__`` (via ``runpy.run_path`` on
    ``sys.argv[0]``) once, at forkserver-process boot, then forks each worker
    from that single preloaded process. For the real CLI entry point, that
    preload executes polylogue's entire import graph inside the forkserver
    process. Any thread or lock created as a side effect of that import is
    inherited — already running or held — by every subsequently forked
    worker, which can deadlock workers before they ever service a task
    (observed in production: forkserver alive in its serve loop, zero workers
    ever spawned, parent parked forever in ``as_completed``). ``spawn`` reruns
    the same ``__main__`` import once per worker instead of forking a shared
    preloaded process, so no inherited thread/lock state crosses into a
    worker. That per-worker import cost (~1-2s) is acceptable here because
    pool workers are long-lived and reused across many parse tasks.
    """
    return multiprocessing.get_context("spawn")


def resolve_parse_worker_count(*, env_var: str = "POLYLOGUE_INGEST_PARSE_WORKERS") -> int:
    """Resolve a CPU-bound parse worker count from ``env_var`` or CPU count.

    Default is ``min(8, cpus-1)`` clamped to at least 1. A value of ``1``
    (including an invalid override) disables pooling entirely and preserves
    exact sequential parse behavior as an escape hatch. Shared by every
    read-only blob->parsed-session decode stage (direct ingest, raw-authority
    census) so one operator knob bounds all of them consistently.
    """
    default = max(1, min(8, (os.cpu_count() or 2) - 1))
    raw = os.environ.get(env_var)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def parallel_threads_effective() -> bool:
    """True iff this interpreter is a genuinely free-threaded (no-GIL) build.

    Gates every ``ThreadPoolExecutor``-based CPU-bound parse dispatch (see
    ``sources/revision_backfill.py::_parse_unique_retained_raws``). The
    polylogue-7mtf control-run measurement is the entire reason this check
    exists: the SAME ``ThreadPoolExecutor`` parse code measured 3.9x-9.6x
    speedup (w=4..16) on a real free-threaded 3.14t build, but 0.93x-0.96x
    (i.e. no speedup, pure lock overhead) on a standard GIL build -- and,
    worse, a concurrent SQLite writer thread's commit latency inflated
    ~5000x (208ms vs an ~0.04ms/5ms cadence) when CPU-bound parse threads
    ran alongside it under the GIL. Threads must never take the
    parse-parallel path unless free-threading is provably active; a mistaken
    "yes" here would silently reintroduce that writer-starvation hazard in
    the daemon.

    ``sys._is_gil_enabled`` only exists on interpreters built with PEP 703/779
    support (CPython 3.13+); its absence means there is no free-threaded
    build to speak of, so that case is treated as "GIL enabled" (the safe
    default), not as an error. Resolved via ``getattr`` (not a literal
    ``sys._is_gil_enabled`` attribute expression) so mypy --strict does not
    require a per-Python-version type: ignore -- the underlying interpreter
    either has the attribute or it doesn't, independent of which stub set
    mypy resolves against.
    """
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    if is_gil_enabled is None:
        return False
    return not is_gil_enabled()


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
    "parallel_threads_effective",
    "process_pool_context",
    "process_pool_executor",
    "resolve_parse_worker_count",
    "terminate_process_pool",
]
