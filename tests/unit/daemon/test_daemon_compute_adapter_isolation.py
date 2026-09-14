"""The shared daemon compute adapter must not survive its test.

``polylogue.daemon.execution._SHARED_COMPUTE_ADAPTER`` is a process-global
published once by whichever daemon owns an API server
(``publish_daemon_compute_adapter(api_server.execution_kernel)``) and read by
every lease-free background derivation through ``daemon_compute_adapter()``.

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

import polylogue.daemon.execution as execution


def test_shared_compute_adapter_is_unpublished_before_every_test() -> None:
    """Anti-vacuity: red if the ``reset_daemon_compute_adapter`` call is
    removed from the autouse singleton-reset fixture in ``tests/conftest.py``
    *and* this file runs after any test that publishes an adapter — which is
    what ``tests/unit/daemon/test_daemon_cli.py`` does, and why that file
    sorts before this one.
    """
    assert execution._SHARED_COMPUTE_ADAPTER is None


def test_reset_drops_the_published_adapter_and_allows_republication() -> None:
    """``reset_daemon_compute_adapter`` must leave the global republishable."""
    first = execution.daemon_compute_adapter()
    assert execution._SHARED_COMPUTE_ADAPTER is first

    execution.reset_daemon_compute_adapter()
    assert execution._SHARED_COMPUTE_ADAPTER is None

    second = execution.daemon_compute_adapter()
    assert second is not first
    execution.reset_daemon_compute_adapter()
