"""Shared process-pool helpers for pipeline services."""

from __future__ import annotations

import multiprocessing
import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from polylogue.runtime import available_cpus


class _BlobSized(Protocol):
    blob_size: int


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

    The ceiling is interpreter-dependent, because the two builds have opposite
    scaling. Under the GIL, CPU-bound parse workers were a spawn-based process
    pool and ``min(8, cpus-1)`` bounded fork cost and memory; going wider bought
    nothing (``parallel_threads_effective`` records 0.93x-0.96x for threads
    there). On a free-threaded build the same parse code measured **3.9x at
    w=4 rising to 9.6x at w=16** (polylogue-7mtf control run), so the historical
    cap of 8 discards most of the available speedup -- on a 24-thread host it
    left 16 threads idle during a 9.2-hour rebuild.

    So: ``min(16, cpus-2)`` free-threaded, ``min(8, cpus-1)`` otherwise. 16 is
    where the measurement ends, not a guess about what lies beyond it, and the
    -2 leaves room for the single SQLite writer plus the daemon's own work.
    Raising it further is an empirical question the rebuild's RebuildPassCost
    instrumentation (mib_per_s against parse_workers) can answer directly.

    A value of ``1`` (including an invalid override) disables pooling entirely
    and preserves exact sequential parse behavior as an escape hatch. Shared by
    every read-only blob->parsed-session decode stage (direct ingest,
    raw-authority census) so one operator knob bounds all of them consistently.
    """
    cpus = available_cpus() or 2
    default = max(1, min(16, cpus - 2)) if parallel_threads_effective() else max(1, min(8, cpus - 1))
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
    from polylogue.runtime import runtime_identity

    return runtime_identity().free_threaded


class PoolKind(StrEnum):
    """Which executor (if any) a resolved parse-dispatch plan should use."""

    THREAD = "thread"
    PROCESS = "process"
    SEQUENTIAL = "sequential"


@dataclass(frozen=True)
class ParseDispatchPlan:
    """One call site's resolved (pool kind, worker count) decision.

    polylogue-xecca: before this, four CPU-bound parse-dispatch call sites
    each computed their own worker-count formula inline (and one, the
    thread-vs-sequential choice under ``parallel_threads_effective``, also
    decided pool KIND inline) -- ``pipeline/services/validation_flow.py``,
    ``pipeline/services/archive_ingest.py``, ``pipeline/services/
    ingest_batch/_core.py``, and ``sources/revision_backfill.py``'s census
    parse family. None of the three worker-count formulas were wrong, but
    scattering them meant "how many workers, and thread or process, for a
    parse dispatch" had no single place to read or change. The
    ``resolve_*_dispatch`` functions below are that single place: one per
    site, since each site's formula reflects a genuinely different,
    independently measured input (workload shape, GIL-vs-free-threaded
    build, or a deliberate ignore of build capability) -- see each
    function's docstring for its own measurement citation. None of the four
    sites' *effective* behavior changes; only where the arithmetic lives
    does.
    """

    pool_kind: PoolKind
    worker_count: int


def resolve_archive_ingest_dispatch(*, path_count: int, total_bytes: int, worker_ceiling: int) -> ParseDispatchPlan:
    """Pool-kind + worker-count decision for ``archive_ingest.py``'s file-walk parse.

    Sized by the work the walk actually found, on the same byte tiers as
    :func:`resolve_ingest_batch_dispatch`: ``<= 8 MiB`` sequential, ``<= 64
    MiB`` capped at 4 workers, above that ``min(path_count, cpus, ceiling)``.
    A spawn pool costs a fresh interpreter and a full ``polylogue`` import per
    worker; below the first tier that setup exceeds the parse it replaces, and
    a spawn failure under host pressure is absorbed by the driver's per-file
    ``except`` as a silently dropped file rather than surfacing as an error.

    ``worker_ceiling`` is the caller's already-resolved
    :func:`resolve_parse_worker_count` value, so the operator knob keeps one
    home. A ceiling of 1 never reaches here: it selects the caller's
    source-iterator escape hatch, which is a different route from the walk.
    """
    if path_count <= 1 or total_bytes <= 8 * 1024 * 1024:
        return ParseDispatchPlan(PoolKind.SEQUENTIAL, 1)
    cpus = available_cpus() or 4
    if total_bytes <= 64 * 1024 * 1024:
        return ParseDispatchPlan(PoolKind.PROCESS, max(1, min(path_count, cpus, worker_ceiling, 4)))
    return ParseDispatchPlan(PoolKind.PROCESS, max(1, min(path_count, cpus, worker_ceiling)))


def resolve_validation_dispatch(*, record_count: int) -> ParseDispatchPlan:
    """Worker-count decision for ``validation_flow.py``'s raw-record validation batch.

    Unchanged formula: ``min(record_count, cpus, 8)``, always a process pool.
    Deliberately ignores :func:`parallel_threads_effective` -- unlike every
    other site here, this one's process-pool preference is *itself* the
    measured result, independent of GIL/free-threading: JSON decode's native
    C extension accelerator releases the GIL, so ``ProcessPoolExecutor``
    measured 605 MB/s at 8 workers against 160 MB/s for 24 threads (3.7x),
    a GIL-build result threads cannot approach regardless of build. Do not
    gate this on ``parallel_threads_effective()``.
    """
    cpus = available_cpus() or 4
    return ParseDispatchPlan(PoolKind.PROCESS, max(1, min(record_count, cpus, 8)))


def resolve_ingest_batch_dispatch(*, total_blob_bytes: int, record_count: int, worker_limit: int) -> ParseDispatchPlan:
    """Worker-count decision for ``ingest_batch/_core.py``'s size-tiered process-pool dispatch.

    Unchanged formula, size-tiered by aggregate blob bytes in the batch:
    ``<= 8 MiB`` stays sequential (pool setup cost isn't worth it for a tiny
    batch); ``<= 64 MiB`` caps at 4 workers (a mid-size batch doesn't benefit
    from wider fan-out); above that, ``min(record_count, cpus, worker_limit)``
    with no additional cap. Always a process pool when not sequential.
    """
    if total_blob_bytes <= 8 * 1024 * 1024:
        return ParseDispatchPlan(PoolKind.SEQUENTIAL, 1)
    if total_blob_bytes <= 64 * 1024 * 1024:
        cpus = available_cpus() or 4
        return ParseDispatchPlan(
            PoolKind.PROCESS,
            min(max(record_count, 1), cpus, worker_limit, 4),
        )
    cpus = available_cpus() or 4
    return ParseDispatchPlan(
        PoolKind.PROCESS,
        min(max(record_count, 1), cpus, worker_limit),
    )


def select_ingest_worker_count(
    raw_artifacts: Sequence[_BlobSized],
    ingest_workers: int | None,
    *,
    default_worker_limit: int = 16,
) -> int:
    """Resolve the size-tiered ingest worker count without importing ingest code."""
    total_blob_size = sum(record.blob_size for record in raw_artifacts)
    worker_limit = default_worker_limit if ingest_workers is None else ingest_workers
    return resolve_ingest_batch_dispatch(
        total_blob_bytes=total_blob_size,
        record_count=len(raw_artifacts),
        worker_limit=worker_limit,
    ).worker_count


def resolve_revision_backfill_census_dispatch(
    *, ingest_workers: int, record_count: int, free_threaded: bool
) -> ParseDispatchPlan:
    """Pool-kind + worker-count decision for the raw-authority census parse family.

    Unchanged formula (``sources/revision_backfill.py``'s
    ``_parse_unique_retained_raws``): sequential unless ``ingest_workers > 1``,
    ``record_count > 1``, AND ``free_threaded`` -- CPU-bound parse threads
    must never engage under the GIL (polylogue-7mtf measured 0.93x-0.96x
    speedup, i.e. none, while inflating a concurrent SQLite writer thread's
    commit latency ~5000x). This is the one site here whose pool KIND, not
    just worker count, is a build-capability decision: a process pool is
    never used for this family (the historical process-pool fallback was
    retired -- see that module's docstring).

    ``free_threaded`` is an explicit input (the caller's own
    :func:`parallel_threads_effective` read), not resolved internally here --
    matching the mission's "build capability via parallel_threads_effective"
    framing as one of this function's INPUTS rather than an ambient global it
    reaches for itself. This also keeps ``revision_backfill.py``'s own tests
    able to monkeypatch their local ``parallel_threads_effective`` import and
    have it actually reach this decision, rather than silently patching a
    different module's binding of the same name.
    """
    if ingest_workers <= 1 or record_count <= 1 or not free_threaded:
        return ParseDispatchPlan(PoolKind.SEQUENTIAL, 1)
    return ParseDispatchPlan(PoolKind.THREAD, min(ingest_workers, record_count))


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
    "ParseDispatchPlan",
    "PoolKind",
    "_initialize_worker_logging",
    "parallel_threads_effective",
    "process_pool_context",
    "process_pool_executor",
    "resolve_archive_ingest_dispatch",
    "resolve_ingest_batch_dispatch",
    "resolve_parse_worker_count",
    "resolve_revision_backfill_census_dispatch",
    "resolve_validation_dispatch",
    "select_ingest_worker_count",
    "terminate_process_pool",
]
