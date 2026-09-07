"""Laws for the daemon's single bounded, class-aware compute scheduler."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from polylogue.daemon.execution import (
    ADMISSION_CLASSES,
    BACKGROUND_CLASSES,
    MAX_BACKGROUND_STARVATION_S,
    MIN_BACKGROUND_THROUGHPUT_FRACTION,
    BoundedComputeAdapter,
    CancellationHandle,
    DaemonBackpressureError,
    DaemonOperationCancelled,
    SubmittedOperation,
)


@contextmanager
def _adapter(**kwargs: int) -> Iterator[BoundedComputeAdapter]:
    adapter = BoundedComputeAdapter(**kwargs)  # type: ignore[arg-type]
    try:
        yield adapter
    finally:
        adapter.shutdown(wait=False)


class _Blocker:
    """A submittable body that reports when it starts and holds until released."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self._started = threading.Semaphore(0)
        self.starts = 0
        self._lock = threading.Lock()

    def __call__(self) -> bool:
        with self._lock:
            self.starts += 1
        self._started.release()
        return self.release.wait(5)

    def wait_started(self, count: int, timeout: float = 5.0) -> bool:
        return all(self._started.acquire(timeout=timeout) for _ in range(count))


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
    """Anti-vacuity: deleting interrupt or the release path leaks capacity."""

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


def test_reserved_shares_meet_the_declared_fairness_bounds() -> None:
    """The reserve table is the mechanism behind both declared fairness bounds.

    Anti-vacuity: a scheduler that reserves nothing for background work, or
    lets one class's ceiling reach total capacity, fails here.
    """

    with _adapter(max_workers=8, queue_units=16) as adapter:
        snapshot = adapter.snapshot()
        background = [entry for entry in snapshot.classes if entry.admission_class in BACKGROUND_CLASSES]
        assert background
        reserved_units = background[0].reserved_units
        assert reserved_units >= int(snapshot.capacity_units * MIN_BACKGROUND_THROUGHPUT_FRACTION)
        assert all(entry.reserved_slots >= 1 for entry in background)
        for entry in snapshot.classes:
            assert entry.ceiling_units < snapshot.capacity_units
            assert entry.ceiling_slots < snapshot.capacity_slots
        assert {entry.admission_class for entry in snapshot.classes} == set(ADMISSION_CLASSES)


def test_bulk_saturation_cannot_consume_interactive_or_control_capacity() -> None:
    """Bulk work stops at its class ceiling, so interactive admission survives.

    Anti-vacuity: with the per-class ceiling removed, bulk fills all 24 units
    and the interactive submit below is rejected.
    """

    blocker = _Blocker()
    with _adapter(max_workers=8, queue_units=16) as adapter:
        try:
            bulk: list[SubmittedOperation] = []
            with pytest.raises(DaemonBackpressureError) as rejection:
                for _index in range(adapter.capacity_units):
                    bulk.append(adapter.submit(blocker, admission_class="bulk-candidate"))
            bulk_ceiling = adapter.snapshot().by_class("bulk-candidate").ceiling_units
            assert len(bulk) == bulk_ceiling
            assert rejection.value.admission_class == "bulk-candidate"
            assert rejection.value.evidence["class_used_units"] == bulk_ceiling
            assert rejection.value.evidence["capacity_units"] == adapter.capacity_units

            adapter.submit(blocker, admission_class="interactive-read")
            adapter.submit(blocker, admission_class="control")
            snapshot = adapter.snapshot()
            assert snapshot.by_class("interactive-read").dispatched == 1
            assert snapshot.by_class("control").dispatched == 1
            assert snapshot.by_class("bulk-candidate").active_units <= snapshot.by_class("bulk-candidate").ceiling_slots
        finally:
            blocker.release.set()


def test_background_work_keeps_a_slot_while_interactive_load_saturates() -> None:
    """Mixed-load progress: background dispatch is bounded, not eventual.

    Anti-vacuity: raising the interactive slot ceiling to the worker count
    lets the interactive flood hold every slot and the background task below
    never starts.
    """

    interactive_body = _Blocker()
    background_body = _Blocker()
    with _adapter(max_workers=8, queue_units=16) as adapter:
        try:
            interactive_ceiling = adapter.snapshot().by_class("interactive-read").ceiling_units
            for _index in range(interactive_ceiling):
                adapter.submit(interactive_body, admission_class="interactive-read")
            slots = adapter.snapshot().by_class("interactive-read").ceiling_slots
            assert interactive_body.wait_started(slots)
            assert adapter.snapshot().by_class("interactive-read").queued_units > 0

            adapter.submit(background_body, admission_class="incremental-background")
            assert background_body.wait_started(1, timeout=MAX_BACKGROUND_STARVATION_S)
            snapshot = adapter.snapshot()
            assert snapshot.background_dispatched == 1
            assert snapshot.background_max_wait_s < MAX_BACKGROUND_STARVATION_S
        finally:
            interactive_body.release.set()
            background_body.release.set()


def test_cancelled_queued_work_never_runs_and_returns_its_reservation() -> None:
    """A queued read cancelled before dispatch leaves the queue, not the pool.

    Anti-vacuity: dropping the cancellation listener leaves the unit reserved
    until the body runs, so the final ``used_units`` assertion fails.
    """

    holder = _Blocker()
    ran = threading.Event()
    with _adapter(max_workers=1, queue_units=4) as adapter:
        try:
            adapter.submit(holder, admission_class="interactive-read")
            assert holder.wait_started(1)
            handle = CancellationHandle()
            queued = adapter.submit(ran.set, admission_class="interactive-read", cancellation=handle)
            assert adapter.snapshot().queued_units == 1
            handle.cancel()
            with pytest.raises(DaemonOperationCancelled):
                queued.future.result(timeout=2)
            assert adapter.snapshot().queued_units == 0
            assert adapter.snapshot().by_class("interactive-read").used_units == 1
        finally:
            holder.release.set()
        assert not ran.is_set()


def test_sequential_cancelled_reads_leave_capacity_unchanged() -> None:
    """N cancelled long reads must not erode capacity by a single unit.

    Anti-vacuity: removing the exactly-once release, or releasing only on the
    non-cancelled path, drifts ``used_units`` upward across the loop.
    """

    with _adapter(max_workers=2, queue_units=4) as adapter:
        baseline = adapter.snapshot()
        for _index in range(8):
            handle = CancellationHandle()
            entered = threading.Event()

            def body(handle: CancellationHandle = handle, entered: threading.Event = entered) -> None:
                entered.set()
                while not handle.cancelled:
                    threading.Event().wait(0.005)

            submitted = adapter.submit(body, cancellation=handle)
            assert entered.wait(5)
            handle.cancel()
            submitted.future.result(timeout=5)
        final = adapter.snapshot()
        assert final.used_units == baseline.used_units == 0
        assert final.active_units == 0
        assert final.by_class("interactive-read").used_units == 0


def test_unknown_admission_class_is_refused() -> None:
    """The class vocabulary is closed; a typo must not silently become a default."""

    with _adapter(max_workers=1, queue_units=1) as adapter:
        with pytest.raises(ValueError, match="unknown daemon admission class"):
            adapter.submit(lambda: None, admission_class="interactive")  # type: ignore[arg-type]
