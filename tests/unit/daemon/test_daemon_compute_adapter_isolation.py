"""The shared daemon compute adapter must not survive its test.

``polylogue.core.compute._SHARED_COMPUTE_ADAPTER`` is a process-global
published once by whichever daemon owns an API server
(``publish_compute_adapter(api_server.execution_kernel)``) and read by
every lease-free background derivation through ``compute_adapter()``.

Process-lifetime publication is correct for a real daemon, whose adapter
outlives every request.  It is wrong for a test process that starts and
discards many daemons: a test that patches the API server with a ``MagicMock``
publishes ``mock.execution_kernel`` into the global, and a test that owns a
real adapter leaves a *shut down* one behind.  Both survive into later tests,
where ``convergence._converge_serialized`` fails on
``asyncio.wrap_future(submitted.future)`` with "concurrent.futures.Future is
expected, got <MagicMock ...>", or raises "daemon compute adapter is shutting
down".
"""

from __future__ import annotations

import pytest

import polylogue.core.compute as execution


def test_shared_compute_adapter_is_unpublished_before_every_test() -> None:
    """Anti-vacuity: red if the ``reset_compute_adapter`` call is
    removed from the autouse singleton-reset fixture in ``tests/conftest.py``
    *and* this file runs after any test that publishes an adapter — which is
    what ``tests/unit/daemon/test_daemon_cli.py`` does, and why that file
    sorts before this one.
    """
    assert execution._SHARED_COMPUTE_ADAPTER is None


def test_reset_drops_the_published_adapter_and_allows_republication() -> None:
    """``reset_compute_adapter`` must leave the global republishable."""
    first = execution.compute_adapter()
    assert execution._SHARED_COMPUTE_ADAPTER is first

    execution.reset_compute_adapter()
    assert execution._SHARED_COMPUTE_ADAPTER is None

    second = execution.compute_adapter()
    assert second is not first
    execution.reset_compute_adapter()


@pytest.mark.uses_real_clock("joins real worker threads")
def test_reset_joins_the_workers_it_shut_down() -> None:
    """A reset owner that asks for a join is left with no live worker.

    Anti-vacuity: drop the join in ``BoundedComputeAdapter.close`` and the
    worker that just finished a job can still be alive when this reads it.
    """
    adapter = execution.compute_adapter()
    assert adapter.submit(lambda: "done").future.result(timeout=5) == "done"
    workers = tuple(adapter.executor._threads)
    assert workers

    assert execution.reset_compute_adapter(join_timeout_s=5.0) == ()
    assert not any(worker.is_alive() for worker in workers)


@pytest.mark.uses_real_clock("waits on a real worker thread past a join deadline")
def test_close_names_a_worker_that_outlives_the_join_deadline() -> None:
    """A running job cannot be interrupted, so ``close`` names its worker."""
    import threading

    release = threading.Event()
    started = threading.Event()

    def blocked() -> None:
        started.set()
        release.wait(5)

    adapter = execution.BoundedComputeAdapter(max_workers=1, queue_units=1, thread_name_prefix="close-probe")
    adapter.submit(blocked)
    assert started.wait(5)
    try:
        assert adapter.close(join_timeout_s=0.05) == ("close-probe_0",)
    finally:
        release.set()
    assert adapter.close(join_timeout_s=5.0) == ()
