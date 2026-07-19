from __future__ import annotations

import threading
from concurrent.futures import as_completed

import pytest

from polylogue.pipeline.services.process_pool import process_pool_context, process_pool_executor


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
