from __future__ import annotations

import sys
import threading
from concurrent.futures import as_completed

import pytest

from polylogue.pipeline.services.process_pool import (
    PoolKind,
    parallel_threads_effective,
    process_pool_context,
    process_pool_executor,
    resolve_archive_ingest_dispatch,
    resolve_ingest_batch_dispatch,
    resolve_revision_backfill_census_dispatch,
    resolve_validation_dispatch,
)


def _worker_wrapper_class_name() -> str:
    import structlog

    wrapper_class = structlog.get_config()["wrapper_class"]
    name = getattr(wrapper_class, "__name__", str(wrapper_class))
    return name if isinstance(name, str) else str(name)


def _square(x: int) -> int:
    return x * x


def test_process_pool_context_avoids_fork() -> None:
    assert process_pool_context().get_start_method() != "fork"


def test_process_pool_context_is_spawn() -> None:
    # Pins the start method explicitly rather than only excluding "fork":
    # forkserver forks every worker from a single preloaded process, so any
    # thread/lock created as a side effect of preloading __main__ (the whole
    # CLI import graph, in production) is inherited by every worker — this
    # caused a production deadlock where the forkserver stayed alive but
    # never spawned a single worker (polylogue-p0pw). spawn reruns __main__
    # fresh per worker instead of forking a shared preloaded process, so no
    # inherited thread/lock state can cross into a worker.
    assert process_pool_context().get_start_method() == "spawn"


def test_process_pool_workers_initialize_info_logging() -> None:
    with process_pool_executor(max_workers=1) as executor:
        wrapper_name = executor.submit(_worker_wrapper_class_name).result(timeout=10)

    assert wrapper_name == "BoundLoggerFilteringAtInfo"


@pytest.mark.timeout(45)
def test_process_pool_dispatch_from_worker_thread_completes() -> None:
    """Regression guard for polylogue-p0pw: production dispatches the pool
    from a thread-pool executor thread under an asyncio event loop, not the
    main thread. A pool start method that forks from a preloaded process
    (forkserver) is vulnerable to inherited thread/lock state hanging every
    worker forever; this must complete within a bounded timeout regardless
    of which thread creates the pool.
    """
    outcome: list[list[int]] = []
    error: list[BaseException] = []

    def dispatch() -> None:
        try:
            with process_pool_executor(max_workers=4) as executor:
                futures = {executor.submit(_square, i): i for i in range(8)}
                results = [future.result(timeout=30) for future in as_completed(futures, timeout=30)]
                outcome.append(sorted(results))
        except BaseException as exc:
            error.append(exc)

    worker_thread = threading.Thread(target=dispatch, daemon=True)
    worker_thread.start()
    worker_thread.join(timeout=40)

    assert not worker_thread.is_alive(), "pool dispatch from a worker thread hung past the bounded timeout"
    if error:
        raise error[0]
    assert outcome == [[i * i for i in range(8)]]


def test_parallel_threads_effective_true_when_gil_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """``sys._is_gil_enabled() -> False`` (real free-threading) must gate the
    thread-parallel parse path OPEN. This is the only state under which
    revision_backfill's ThreadPoolExecutor parse path may engage. Patching
    the real ``sys`` module (imported here directly, not via the production
    module's reference to it) affects ``process_pool.py``'s own
    ``sys._is_gil_enabled`` lookup identically, since both names resolve the
    same singleton module object at call time."""
    monkeypatch.setattr(sys, "_is_gil_enabled", lambda: False, raising=False)
    assert parallel_threads_effective() is True


def test_parallel_threads_effective_false_when_gil_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """``sys._is_gil_enabled() -> True`` (GIL build, or a free-threaded build
    that re-enabled the GIL at runtime) must gate the thread-parallel parse
    path CLOSED -- the polylogue-7mtf control run measured GIL-build threads
    give zero parse speedup and starve a concurrent writer thread ~5000x."""
    monkeypatch.setattr(sys, "_is_gil_enabled", lambda: True, raising=False)
    assert parallel_threads_effective() is False


def test_parallel_threads_effective_treats_missing_probe_as_gil_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Interpreters without PEP 703/779 support (no ``sys._is_gil_enabled``
    attribute at all, e.g. every CPython before 3.13) have no free-threaded
    build to speak of -- the safe default is "GIL enabled", not an error."""
    monkeypatch.delattr(sys, "_is_gil_enabled", raising=False)
    assert parallel_threads_effective() is False


# ---------------------------------------------------------------------------
# polylogue-xecca: unified parse-dispatch decision matrix
# ---------------------------------------------------------------------------
#
# Each of the four sites' formulas moved here verbatim from
# archive_ingest.py / validation_flow.py / ingest_batch/_core.py /
# revision_backfill.py -- these matrices pin the exact numbers those sites
# relied on before the extraction so a future edit to the shared function
# cannot silently drift any one site's effective behavior without this file
# failing first.


def test_resolve_archive_ingest_dispatch_defaults_to_resolve_parse_worker_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("polylogue.pipeline.services.process_pool.os.cpu_count", lambda: 9)
    monkeypatch.setattr(sys, "_is_gil_enabled", lambda: True, raising=False)
    plan = resolve_archive_ingest_dispatch()
    assert plan.pool_kind is PoolKind.PROCESS
    # GIL build: min(8, cpus-1) = min(8, 8) = 8.
    assert plan.worker_count == 8


def test_resolve_archive_ingest_dispatch_honors_explicit_override() -> None:
    """``parse_workers`` (the demo seeder's force-sequential knob) wins over
    the ambient CPU-based default, clamped to at least 1."""
    plan = resolve_archive_ingest_dispatch(parse_workers=1)
    assert plan.pool_kind is PoolKind.PROCESS
    assert plan.worker_count == 1

    plan_negative = resolve_archive_ingest_dispatch(parse_workers=-5)
    assert plan_negative.worker_count == 1


@pytest.mark.parametrize(
    ("record_count", "cpu_count", "expected_workers"),
    [
        (8, 24, 8),  # capped at 8 regardless of a wider CPU count
        (3, 24, 3),  # fewer records than the cap: use the record count
        (12, 4, 4),  # fewer CPUs than records or the cap: use CPU count
    ],
)
def test_resolve_validation_dispatch_matrix(
    monkeypatch: pytest.MonkeyPatch, record_count: int, cpu_count: int, expected_workers: int
) -> None:
    """validation_flow.py's own measured process-pool preference: always a
    process pool, ``min(record_count, cpus, 8)``, independent of
    ``parallel_threads_effective`` -- confirmed by never patching the GIL
    probe in this test at all."""
    monkeypatch.setattr("polylogue.pipeline.services.process_pool.os.cpu_count", lambda: cpu_count)
    plan = resolve_validation_dispatch(record_count=record_count)
    assert plan.pool_kind is PoolKind.PROCESS
    assert plan.worker_count == expected_workers


@pytest.mark.parametrize(
    ("total_blob_bytes", "record_count", "worker_limit", "cpu_count", "expected_kind", "expected_workers"),
    [
        (4 * 1024 * 1024, 10, 16, 16, PoolKind.SEQUENTIAL, 1),  # <= 8 MiB: sequential
        (16 * 1024 * 1024, 10, 16, 16, PoolKind.PROCESS, 4),  # <= 64 MiB: capped at 4
        (16 * 1024 * 1024, 2, 16, 16, PoolKind.PROCESS, 2),  # <= 64 MiB: record count under the 4-cap
        (100 * 1024 * 1024, 60, 16, 16, PoolKind.PROCESS, 16),  # > 64 MiB: no 4-cap, limited by worker_limit
        (100 * 1024 * 1024, 60, 4, 16, PoolKind.PROCESS, 4),  # > 64 MiB: explicit narrower limit wins
        (100 * 1024 * 1024, 2, 16, 16, PoolKind.PROCESS, 2),  # > 64 MiB: record count under every ceiling
    ],
)
def test_resolve_ingest_batch_dispatch_matrix(
    monkeypatch: pytest.MonkeyPatch,
    total_blob_bytes: int,
    record_count: int,
    worker_limit: int,
    cpu_count: int,
    expected_kind: PoolKind,
    expected_workers: int,
) -> None:
    monkeypatch.setattr("polylogue.pipeline.services.process_pool.os.cpu_count", lambda: cpu_count)
    plan = resolve_ingest_batch_dispatch(
        total_blob_bytes=total_blob_bytes, record_count=record_count, worker_limit=worker_limit
    )
    assert plan.pool_kind is expected_kind
    assert plan.worker_count == expected_workers


@pytest.mark.parametrize(
    ("ingest_workers", "record_count", "free_threaded", "expected_kind", "expected_workers"),
    [
        (1, 10, True, PoolKind.SEQUENTIAL, 1),  # ingest_workers<=1: sequential regardless of build
        (4, 1, True, PoolKind.SEQUENTIAL, 1),  # record_count<=1: sequential regardless of build
        (4, 10, False, PoolKind.SEQUENTIAL, 1),  # GIL build: sequential regardless of workers/records
        (4, 10, True, PoolKind.THREAD, 4),  # free-threaded + eligible: threads, worker_count = min(workers, records)
        (10, 4, True, PoolKind.THREAD, 4),  # worker_count capped by record_count, not ingest_workers
    ],
)
def test_resolve_revision_backfill_census_dispatch_matrix(
    ingest_workers: int, record_count: int, free_threaded: bool, expected_kind: PoolKind, expected_workers: int
) -> None:
    """``free_threaded`` is an explicit input here (the caller's own
    ``parallel_threads_effective()`` read), not resolved internally --
    exercised directly with both booleans rather than monkeypatching the
    GIL probe, proving the decision is a pure function of its three inputs."""
    plan = resolve_revision_backfill_census_dispatch(
        ingest_workers=ingest_workers, record_count=record_count, free_threaded=free_threaded
    )
    assert plan.pool_kind is expected_kind
    assert plan.worker_count == expected_workers
