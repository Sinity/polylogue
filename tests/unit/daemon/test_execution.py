"""Laws for the daemon's single bounded compute adapter."""

from __future__ import annotations

import threading

import pytest

from polylogue.daemon.execution import (
    BoundedComputeAdapter,
    CancellationHandle,
    DaemonBackpressureError,
)


def test_admission_is_bounded_by_units_and_bytes() -> None:
    """Anti-vacuity: removing either bound would accept this saturated mutant."""

    adapter = BoundedComputeAdapter(max_workers=1, queue_units=0, queue_bytes=4)
    started = threading.Event()
    release = threading.Event()

    def wait_for_release() -> bool:
        started.set()
        return release.wait(2)

    try:
        first = adapter.submit(wait_for_release, estimated_bytes=4)
        assert started.wait(1)
        assert adapter.snapshot().queued_bytes == 0
        with pytest.raises(DaemonBackpressureError, match="saturated"):
            adapter.submit(lambda: None)
        assert adapter.snapshot().rejected == 1
        release.set()
        first.future.result(timeout=2)
        assert adapter.snapshot().used_units == 0
        assert adapter.snapshot().used_bytes == 0
    finally:
        release.set()
        adapter.shutdown(wait=True)


def test_cancellation_interrupts_registered_connection_and_releases_capacity() -> None:
    """Anti-vacuity: deleting interrupt or the done callback leaks capacity."""

    class Connection:
        interrupted = 0

        def interrupt(self) -> None:
            self.interrupted += 1

    adapter = BoundedComputeAdapter(max_workers=1, queue_units=0, queue_bytes=8)
    handle = CancellationHandle()
    connection = Connection()
    registered = threading.Event()
    finished = threading.Event()

    def work() -> None:
        handle.register_connection(connection)
        registered.set()
        while not handle.cancelled:
            finished.wait(0.01)
        handle.unregister_connection(connection)

    try:
        submitted = adapter.submit(work, cancellation=handle, estimated_bytes=8)
        assert registered.wait(1)
        handle.cancel()
        submitted.future.result(timeout=2)
        assert connection.interrupted == 1
        assert adapter.snapshot().used_units == 0
        assert adapter.snapshot().used_bytes == 0
    finally:
        finished.set()
        adapter.shutdown(wait=True)
