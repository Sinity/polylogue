"""Laws for the daemon's single bounded, class-aware compute scheduler."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from time import monotonic

import pytest

from polylogue.core.compute import (
    ADMISSION_CLASSES,
    BACKGROUND_CLASSES,
    MAX_BACKGROUND_STARVATION_S,
    MIN_BACKGROUND_RESERVED_SLOTS,
    AdmissionClass,
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
        # Long enough that a loaded host cannot let the body finish on its own
        # and turn a saturation law into a vacuous pass.
        return self.release.wait(60)

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


def test_failed_completion_observes_released_reservation() -> None:
    """An errored public future is not done while its admission is retained.

    Anti-vacuity: completing the future before ``_release`` makes the snapshot
    after ``result`` nondeterministically retain this task's unit.
    """

    adapter = BoundedComputeAdapter(max_workers=1, queue_units=0, queue_bytes=4)

    def fail() -> None:
        raise RuntimeError("compute failed")

    try:
        submitted = adapter.submit(fail, estimated_bytes=4)
        with pytest.raises(RuntimeError, match="compute failed"):
            submitted.future.result(timeout=2)
        assert adapter.snapshot().used_units == 0
        assert adapter.snapshot().used_bytes == 0
    finally:
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


def test_reserves_keep_both_groups_runnable() -> None:
    """The reserve table keeps both groups eligible under the other group's load.

    Anti-vacuity: a scheduler that reserves nothing for background work, or
    lets one class's ceiling reach total capacity, fails here.
    """

    with _adapter(max_workers=8, queue_units=16) as adapter:
        snapshot = adapter.snapshot()
        background = [entry for entry in snapshot.classes if entry.admission_class in BACKGROUND_CLASSES]
        assert background
        reserved_units = background[0].reserved_units
        assert reserved_units == MIN_BACKGROUND_RESERVED_SLOTS
        assert all(entry.reserved_slots >= 1 for entry in background)
        for entry in snapshot.classes:
            assert entry.ceiling_units < snapshot.capacity_units
            assert entry.ceiling_slots < snapshot.capacity_slots
        assert {entry.admission_class for entry in snapshot.classes} == set(ADMISSION_CLASSES)


@pytest.mark.parametrize("queue_units", [0, 1, 8])
@pytest.mark.parametrize("workers", [1, 2, 8, 12, 24])
def test_one_shared_foreground_and_background_reserve(workers: int, queue_units: int) -> None:
    """Worker width increases useful capacity rather than idle percentage shares."""
    with _adapter(max_workers=workers, queue_units=queue_units) as adapter:
        snapshot = adapter.snapshot()
        reserve = int(workers > 1)
        for entry in snapshot.classes:
            assert entry.reserved_slots == entry.reserved_units == reserve
            assert entry.ceiling_slots == workers - reserve
            assert entry.ceiling_units == workers + queue_units - reserve
            group = "background" if entry.admission_class in BACKGROUND_CLASSES else "foreground"
            assert entry.to_dict()["reservation_group"] == group
        assert {entry.admission_class for entry in snapshot.classes} == set(ADMISSION_CLASSES)


def test_bulk_saturation_cannot_consume_interactive_or_control_capacity() -> None:
    """Bulk work stops at its ceiling, leaving one shared foreground unit.

    Anti-vacuity: with the group ceiling removed, bulk fills all 24 units
    and the interactive submit below is rejected.
    """

    blocker = _Blocker()
    with _adapter(max_workers=8, queue_units=16) as adapter:
        try:
            bulk: list[SubmittedOperation[object]] = []
            with pytest.raises(DaemonBackpressureError) as rejection:
                for _index in range(adapter.capacity_units):
                    bulk.append(adapter.submit(blocker, admission_class="bulk-candidate"))
            bulk_ceiling = adapter.snapshot().by_class("bulk-candidate").ceiling_units
            assert len(bulk) == bulk_ceiling
            assert rejection.value.admission_class == "bulk-candidate"
            assert rejection.value.evidence["class_used_units"] == bulk_ceiling
            assert rejection.value.evidence["capacity_units"] == adapter.capacity_units

            read = adapter.submit(blocker, admission_class="interactive-read")
            with pytest.raises(DaemonBackpressureError):
                adapter.submit(blocker, admission_class="control")
            snapshot = adapter.snapshot()
            assert snapshot.by_class("interactive-read").dispatched == 1
            assert snapshot.by_class("control").dispatched == 0
            assert read.queue_delay_s < 0.1
            assert snapshot.by_class("bulk-candidate").active_units <= snapshot.by_class("bulk-candidate").ceiling_slots
        finally:
            blocker.release.set()


def test_background_classes_share_one_reserve() -> None:
    """Bulk and incremental work cannot duplicate the aggregate background reserve.

    Anti-vacuity: allowing each background class its own ceiling lets the two
    queues occupy every worker and makes the interactive submission below
    wait behind background work instead of receiving reserved capacity.
    """

    blocker = _Blocker()
    with _adapter(max_workers=8, queue_units=0) as adapter:
        try:
            bulk_ceiling = adapter.snapshot().by_class("bulk-candidate").ceiling_units
            bulk = [adapter.submit(blocker, admission_class="bulk-candidate") for _index in range(bulk_ceiling)]
            assert len(bulk) == bulk_ceiling
            with pytest.raises(DaemonBackpressureError) as rejection:
                adapter.submit(blocker, admission_class="incremental-background")
            assert rejection.value.admission_class == "incremental-background"
            assert rejection.value.evidence["class_used_units"] == bulk_ceiling

            interactive = adapter.submit(blocker, admission_class="interactive-read")
            assert adapter.snapshot().by_class("interactive-read").dispatched == 1
            assert interactive.queue_delay_s < 0.1
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


@pytest.mark.uses_real_clock("queued starvation age is measured by the production scheduler clock")
def test_snapshot_includes_the_age_of_queued_background_work() -> None:
    """A queued background task contributes to the observable starvation bound.

    Anti-vacuity: computing ``max_wait_s`` only when dispatch starts leaves
    this admitted task at zero for its entire queue lifetime.
    """

    holder = _Blocker()
    with _adapter(max_workers=1, queue_units=2) as adapter:
        try:
            adapter.submit(holder, admission_class="interactive-read")
            assert holder.wait_started(1)
            queued = adapter.submit(lambda: None, admission_class="incremental-background")
            deadline = monotonic() + 1
            snapshot = adapter.snapshot()
            while snapshot.background_max_wait_s <= 0 and monotonic() < deadline:
                threading.Event().wait(0.01)
                snapshot = adapter.snapshot()
            assert snapshot.by_class("incremental-background").queued_units == 1
            assert snapshot.background_max_wait_s > 0
        finally:
            holder.release.set()
            queued.future.result(timeout=5)


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


def test_future_cancel_removes_queued_task_and_releases_reservation() -> None:
    """Anti-vacuity: Future.cancel alone must free queue capacity behind a held worker."""
    holder = _Blocker()
    with _adapter(max_workers=1, queue_units=2, queue_bytes=100) as adapter:
        try:
            adapter.submit(holder)
            assert holder.wait_started(1)
            queued = adapter.submit(lambda: "must not run", estimated_bytes=20)
            assert queued.future.cancel()
            snapshot = adapter.snapshot()
            assert snapshot.used_units == 1
            assert snapshot.used_bytes == 0
        finally:
            holder.release.set()


def test_admission_rejects_slot_demand_over_class_ceiling() -> None:
    """Anti-vacuity: a task above its class ceiling otherwise never dispatches."""
    with _adapter(max_workers=8) as adapter:
        with pytest.raises(DaemonBackpressureError, match="slot ceiling"):
            adapter.submit(lambda: None, admission_class="control", units=8)


def test_cancelled_queued_work_is_not_counted_as_completed_dispatch() -> None:
    """Scheduler counters distinguish a released queue reservation from executed work.

    Anti-vacuity: incrementing ``completed`` for every reservation release
    reports two completions after this queued cancellation and one dispatched
    body, which corrupts the background-throughput denominator.
    """

    holder = _Blocker()
    with _adapter(max_workers=1, queue_units=1) as adapter:
        try:
            running = adapter.submit(holder, admission_class="interactive-read")
            assert holder.wait_started(1)
            cancellation = CancellationHandle()
            queued = adapter.submit(lambda: None, admission_class="interactive-read", cancellation=cancellation)
            cancellation.cancel()
            with pytest.raises(DaemonOperationCancelled):
                queued.future.result(timeout=2)
            holder.release.set()
            running.future.result(timeout=2)
            snapshot = adapter.snapshot().by_class("interactive-read")
            assert snapshot.dispatched == 1
            assert snapshot.completed == 1
        finally:
            holder.release.set()


def test_background_dispatch_rotates_under_mixed_load() -> None:
    """Both background lanes progress while interactive and control stay active.

    Anti-vacuity: fixed incremental-first dispatch cannot start the first bulk
    task while the incremental backlog remains. A 1:1 rotation fails the
    declared two-incremental-per-bulk dispatch order.
    """

    holders = [_Blocker() for _ in range(4)]
    foreground = _Blocker()
    incremental = [_Blocker() for _ in range(4)]
    bulk = [_Blocker() for _ in range(2)]
    bodies = [*holders, foreground, *incremental, *bulk]
    adapter = BoundedComputeAdapter(max_workers=8, queue_units=16)
    try:
        running = [adapter.submit(body, admission_class="incremental-background") for body in holders]
        assert all(body.wait_started(1) for body in holders)
        for _index in range(3):
            adapter.submit(foreground, admission_class="interactive-read")
        adapter.submit(foreground, admission_class="control")
        assert foreground.wait_started(4)

        queued_incremental = [adapter.submit(body, admission_class="incremental-background") for body in incremental]
        queued_bulk = [adapter.submit(body, admission_class="bulk-candidate") for body in bulk]
        assert adapter.snapshot().queued_units == 6
        # Free one slot; every other worker remains occupied throughout the
        # sequence, so each completion determines exactly one new dispatch.
        holders[0].release.set()
        running[0].future.result(timeout=5)
        order = [
            (bulk[0], queued_bulk[0]),
            (incremental[0], queued_incremental[0]),
            (incremental[1], queued_incremental[1]),
            (bulk[1], queued_bulk[1]),
            (incremental[2], queued_incremental[2]),
            (incremental[3], queued_incremental[3]),
        ]
        for body, operation in order:
            assert body.wait_started(1)
            snapshot = adapter.snapshot()
            assert snapshot.by_class("interactive-read").active_units == 3
            assert snapshot.by_class("control").active_units == 1
            assert snapshot.used_units <= snapshot.capacity_units
            body.release.set()
            operation.future.result(timeout=5)
        assert adapter.snapshot().queued_units == 0
    finally:
        for body in bodies:
            body.release.set()
        adapter.shutdown(wait=True)
    assert adapter.snapshot().used_units == 0


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


@pytest.mark.uses_real_clock("real UDS listener and coordinator bound mixed-load admission waits")
def test_uds_status_uses_reserved_capacity_under_background_saturation(tmp_path: Path) -> None:
    """The installed operation route shares the saturated background scheduler.

    Anti-vacuity: dropping aggregate background reservations prevents status
    dispatch before these explicitly blocked bodies release their workers.
    """
    from tests.infra.daemon_operations import running_daemon_operations

    body = _Blocker()
    with running_daemon_operations(tmp_path / "archive", compute_workers=8) as stack:
        kernel = stack.execution_kernel
        operations: list[SubmittedOperation[bool]] = []
        try:
            with pytest.raises(DaemonBackpressureError) as rejection:
                for index in range(kernel.capacity_units):
                    operations.append(
                        kernel.submit(
                            body,
                            admission_class="incremental-background" if index % 2 else "bulk-candidate",
                        )
                    )
            assert rejection.value.code == "compute_backpressure"
            assert rejection.value.evidence["class_used_units"] == len(operations)
            assert body.wait_started(4)
            assert kernel.snapshot().queued_units > 0

            for _index in range(3):
                envelope = stack.client.operation("status", {}, archive_root=str(stack.archive_root))
                assert envelope is not None
                assert envelope["outcome"] == "completed"
                assert envelope["authority"]["writes"] == "daemon-owned"
                assert envelope["result"]["total_sessions"] == 0
            snapshot = kernel.snapshot()
            assert snapshot.by_class("interactive-read").dispatched == 3
            assert snapshot.by_class("incremental-background").dispatched > 0
            assert snapshot.by_class("bulk-candidate").dispatched > 0
            assert snapshot.used_units == len(operations)
        finally:
            body.release.set()
            for operation in operations:
                operation.future.result(timeout=5)
        assert kernel.snapshot().used_units == 0


def test_combined_foreground_classes_retain_background_dispatch() -> None:
    """Control and reads together cannot take the background reserve."""
    foreground = _Blocker()
    background = _Blocker()
    with _adapter(max_workers=8, queue_units=16) as adapter:
        try:
            for index in range(7):
                name: AdmissionClass = "control" if index % 2 else "interactive-read"
                adapter.submit(foreground, admission_class=name)
            assert foreground.wait_started(7)
            operation = adapter.submit(background, admission_class="incremental-background")
            assert background.wait_started(1)
            assert operation.queue_delay_s < 0.1
        finally:
            foreground.release.set()
            background.release.set()


def test_combined_foreground_classes_retain_background_queue() -> None:
    """The queue complement uses control and read occupancy together."""
    blocker = _Blocker()
    with _adapter(max_workers=8, queue_units=16) as adapter:
        try:
            for index in range(adapter.capacity_units - 1):
                name: AdmissionClass = "control" if index % 2 else "interactive-read"
                adapter.submit(blocker, admission_class=name)
            with pytest.raises(DaemonBackpressureError):
                adapter.submit(blocker, admission_class="interactive-read")
            adapter.submit(blocker, admission_class="incremental-background")
            assert adapter.snapshot().used_units == adapter.capacity_units
        finally:
            blocker.release.set()


def test_shared_foreground_dispatch_alternates_eligible_control_and_reads() -> None:
    """A control queue cannot repeatedly overtake already admitted reads."""
    holder = _Blocker()
    observed: list[str] = []
    with _adapter(max_workers=2, queue_units=8) as adapter:
        try:
            first = adapter.submit(holder, admission_class="control")
            assert holder.wait_started(1)
            operations = []
            for name in ("control", "control", "interactive-read", "interactive-read"):
                operations.append(adapter.submit(partial(observed.append, name), admission_class=name))
            holder.release.set()
            first.future.result(timeout=5)
            for operation in operations:
                operation.future.result(timeout=5)
            assert observed == ["interactive-read", "control", "interactive-read", "control"]
        finally:
            holder.release.set()


def test_multi_slot_background_head_is_not_overtaken_by_one_slot_bulk() -> None:
    """A seven-slot incremental head runs before later one-slot bulk work.

    With eight workers the shared foreground reserve leaves seven slots
    for background work. One bulk task holds a slot, so the incremental head
    needs every remaining one. Anti-vacuity: letting the complement check
    ``continue`` to later turns dispatches each new bulk task ahead of it, and
    the head never starts while bulk work keeps arriving.
    """

    first_bulk, incremental, later_bulk = _Blocker(), _Blocker(), _Blocker()
    adapter = BoundedComputeAdapter(max_workers=8, queue_units=16)
    try:
        running = adapter.submit(first_bulk, admission_class="bulk-candidate")
        assert first_bulk.wait_started(1)
        queued_incremental = adapter.submit(incremental, admission_class="incremental-background", units=7)
        queued_bulk = adapter.submit(later_bulk, admission_class="bulk-candidate")

        assert not later_bulk.wait_started(1, timeout=0.3)
        first_bulk.release.set()
        running.future.result(timeout=5)
        assert incremental.wait_started(1)
        assert later_bulk.starts == 0
        incremental.release.set()
        queued_incremental.future.result(timeout=5)
        assert later_bulk.wait_started(1)
        later_bulk.release.set()
        queued_bulk.future.result(timeout=5)
    finally:
        for body in (first_bulk, incremental, later_bulk):
            body.release.set()
        adapter.shutdown(wait=True)


def test_a_unit_larger_than_the_byte_envelope_runs_alone_instead_of_being_refused() -> None:
    """An input bigger than the byte pool reserves all of it and still runs.

    Anti-vacuity: refusing ``estimated_bytes > capacity_bytes`` outright makes
    the daemon unable to ever process that input.
    """

    adapter = BoundedComputeAdapter(max_workers=2, queue_units=2, queue_bytes=10)
    try:
        submitted = adapter.submit(lambda: "parsed", estimated_bytes=25)
        assert submitted.future.result(timeout=5) == "parsed"
        assert adapter.snapshot().used_bytes == 0
    finally:
        adapter.shutdown(wait=True)


def test_waiting_multi_slot_foreground_head_drains_background_backfill() -> None:
    """A three-slot read acquires capacity despite replenished background work."""
    bodies = [_Blocker() for _ in range(3)]
    later = _Blocker()
    foreground = _Blocker()
    with _adapter(max_workers=4, queue_units=8) as adapter:
        initial: list[SubmittedOperation[bool]] = []
        queued: list[SubmittedOperation[bool]] = []
        try:
            initial = [adapter.submit(body, admission_class="incremental-background") for body in bodies]
            assert all(body.wait_started(1) for body in bodies)
            read = adapter.submit(foreground, admission_class="interactive-read", units=3)
            queued = [adapter.submit(later, admission_class="bulk-candidate") for _ in range(2)]
            bodies[0].release.set()
            initial[0].future.result(timeout=5)
            assert not later.wait_started(1, timeout=0.05)
            bodies[1].release.set()
            initial[1].future.result(timeout=5)
            assert foreground.wait_started(1)
            assert adapter.snapshot().by_class("incremental-background").active_units == 1
            foreground.release.set()
            read.future.result(timeout=5)
            assert later.wait_started(2)
        finally:
            foreground.release.set()
            later.release.set()
            for body in bodies:
                body.release.set()
            for operation in (*initial, *queued):
                operation.future.result(timeout=5)
