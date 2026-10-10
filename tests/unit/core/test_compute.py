"""Physical ownership laws for the shared pure compute adapter."""

from __future__ import annotations

import contextvars
import sqlite3
import threading
from builtins import BaseExceptionGroup
from collections.abc import Callable
from concurrent.futures import Future

import pytest

from polylogue.core.compute import (
    AdmissionClass,
    BoundedComputeAdapter,
    CancellationHandle,
    DaemonBackpressureError,
    DaemonOperationCancelled,
    SubmittedOperation,
    current_cancellation,
)
from polylogue.core.compute_cancel import check_compute_cancelled

pytestmark = pytest.mark.uses_real_clock("compute worker synchronization uses OS waits")


def test_shared_compute_factory_uses_the_declared_default_width(monkeypatch: pytest.MonkeyPatch) -> None:
    import polylogue.core.compute as module

    monkeypatch.setattr(module, "_SHARED_COMPUTE_ADAPTER", None)
    adapter = module.compute_adapter()
    try:
        assert adapter.max_workers == module.DEFAULT_COMPUTE_WORKERS
        assert module.compute_adapter() is adapter
        assert adapter.snapshot().by_class("incremental-background").ceiling_slots > 1
        assert adapter.submit(lambda: "completed").future.result(timeout=5) == "completed"
    finally:
        adapter.shutdown(wait=True)


@pytest.mark.parametrize("reject_submission", [True, False])
def test_shutdown_during_queued_dispatch_settles_both_operation_futures(
    monkeypatch: pytest.MonkeyPatch, reject_submission: bool
) -> None:
    adapter = BoundedComputeAdapter(max_workers=1, queue_units=2)
    started = threading.Event()
    release = threading.Event()
    queued_invoked = threading.Event()

    def active() -> str:
        started.set()
        assert release.wait(5)
        return "active completed"

    first = adapter.submit(active)
    try:
        assert started.wait(5)
        second = adapter.submit(queued_invoked.set)
        original_submit = adapter.executor.submit

        def shutdown_at_submission(function: Callable[..., object], *args: object, **kwargs: object) -> Future[object]:
            if reject_submission:
                adapter.shutdown(wait=False)
                return original_submit(function, *args, **kwargs)
            execution = original_submit(function, *args, **kwargs)
            # The first task still owns the only physical worker, so this
            # accepted executor future cannot start before shutdown cancels it.
            adapter.shutdown(wait=False)
            assert execution.cancelled()
            return execution

        monkeypatch.setattr(adapter.executor, "submit", shutdown_at_submission)
        release.set()
        assert first.future.result(timeout=5) == "active completed"
        with pytest.raises(DaemonOperationCancelled):
            second.future.result(timeout=5)
        assert not queued_invoked.is_set()
        snapshot = adapter.snapshot()
        assert snapshot.used_units == snapshot.active_units == snapshot.queued_units == 0
    finally:
        release.set()
        adapter.shutdown(wait=True)


def test_graceful_shutdown_drains_already_admitted_work_without_accepting_more() -> None:
    adapter = BoundedComputeAdapter(max_workers=1, queue_units=1)
    started = threading.Event()
    release = threading.Event()

    def active() -> str:
        started.set()
        assert release.wait(5)
        return "active completed"

    first = adapter.submit(active)
    try:
        assert started.wait(5)
        second = adapter.submit(lambda: "queued completed")
        adapter.shutdown(wait=False, cancel_futures=False)
        with pytest.raises(DaemonBackpressureError):
            adapter.submit(lambda: "after shutdown")
        release.set()
        assert first.future.result(timeout=5) == "active completed"
        assert second.future.result(timeout=5) == "queued completed"
        snapshot = adapter.snapshot()
        assert snapshot.used_units == snapshot.active_units == snapshot.queued_units == 0
    finally:
        release.set()
        adapter.shutdown(wait=True)


@pytest.mark.parametrize("wrong_context", ["outside", "adapter", "bridge_thread"])
def test_preparation_creator_guard_refuses_an_unadmitted_or_transferred_reservation(wrong_context: str) -> None:
    from polylogue.core.compute import capture_compute_bridge

    adapter = BoundedComputeAdapter(max_workers=1, queue_units=0)
    other = BoundedComputeAdapter(max_workers=1, queue_units=0)

    def parent() -> None:
        adapter.require_current_creator()
        if wrong_context == "adapter":
            with pytest.raises(RuntimeError, match="admitted compute creator"):
                other.require_current_creator()
        else:
            borrow = capture_compute_bridge()
            failures: list[BaseException] = []

            def joined_bridge() -> None:
                with borrow():
                    try:
                        adapter.require_current_creator()
                    except BaseException as failure:
                        failures.append(failure)

            bridge = threading.Thread(target=joined_bridge)
            bridge.start()
            bridge.join()
            assert len(failures) == 1
            assert isinstance(failures[0], RuntimeError)
        adapter.require_current_creator()

    try:
        if wrong_context == "outside":
            with pytest.raises(RuntimeError, match="admitted compute creator"):
                adapter.require_current_creator()
        else:
            adapter.submit(parent).future.result(timeout=5)
        assert adapter.snapshot().used_units == other.snapshot().used_units == 0
    finally:
        adapter.shutdown(wait=True)
        other.shutdown(wait=True)


def test_preparation_creator_guard_observes_actual_parent_cancellation() -> None:
    adapter = BoundedComputeAdapter(max_workers=1, queue_units=0)
    cleaned: list[int] = []

    def parent() -> None:
        adapter.require_current_creator()
        cancellation = current_cancellation()
        assert cancellation is not None
        cancellation.cancel()
        try:
            adapter.require_current_creator()
        finally:
            assert adapter.snapshot().active_units == 1
            cleaned.append(threading.get_ident())

    try:
        with pytest.raises(DaemonOperationCancelled):
            adapter.submit(parent).future.result(timeout=5)
        assert len(cleaned) == 1 and cleaned[0] != threading.get_ident()
        assert adapter.snapshot().active_units == 0
    finally:
        adapter.shutdown(wait=True)


@pytest.mark.parametrize("capacity_bytes", [0, 100])
@pytest.mark.parametrize("initial_input_bytes", [0, 7])
def test_exclusive_byte_preparation_records_actual_growth_and_retries_after_release(
    capacity_bytes: int,
    initial_input_bytes: int,
) -> None:
    adapter = BoundedComputeAdapter(max_workers=2, queue_units=2, queue_bytes=capacity_bytes)
    hydrated = threading.Event()
    release = threading.Event()

    def first_preparation() -> int:
        adapter.require_current_creator()
        before_discovery = adapter.snapshot()
        assert before_discovery.used_bytes == capacity_bytes
        assert before_discovery.exclusive_byte_units == 1
        assert before_discovery.active_input_bytes == initial_input_bytes
        adapter.amend_current_input_demand(200)
        hydrated.set()
        assert release.wait(5)
        return adapter.snapshot().active_input_bytes

    first = adapter.submit(first_preparation, estimated_bytes=initial_input_bytes, exclusive_bytes=True)
    try:
        assert hydrated.wait(5)
        snapshot = adapter.snapshot()
        assert snapshot.used_bytes == capacity_bytes
        assert snapshot.active_input_bytes == initial_input_bytes + 200
        assert snapshot.exclusive_byte_units == 1
        with pytest.raises(DaemonBackpressureError):
            adapter.submit(lambda: "second", estimated_bytes=11, exclusive_bytes=True)
        with pytest.raises(DaemonBackpressureError):
            adapter.submit(lambda: "ordinary", estimated_bytes=1)
        release.set()
        assert first.future.result(timeout=5) == initial_input_bytes + 200
        second = adapter.submit(
            lambda: ("second", adapter.snapshot().active_input_bytes), estimated_bytes=11, exclusive_bytes=True
        )
        assert second.future.result(timeout=5) == ("second", 11)
        snapshot = adapter.snapshot()
        assert snapshot.used_units == snapshot.used_bytes == snapshot.active_input_bytes == 0
        assert snapshot.queued_input_bytes == snapshot.exclusive_byte_units == 0
    finally:
        release.set()
        adapter.shutdown(wait=True)


def test_cancelled_queued_exclusive_preparation_releases_its_entire_reservation_once() -> None:
    adapter = BoundedComputeAdapter(max_workers=1, queue_units=1, queue_bytes=100)
    started = threading.Event()
    release = threading.Event()
    invoked = threading.Event()

    def blocker() -> None:
        started.set()
        assert release.wait(5)

    first = adapter.submit(blocker, estimated_bytes=0)
    try:
        assert started.wait(5)
        queued = adapter.submit(invoked.set, estimated_bytes=7, exclusive_bytes=True)
        before = adapter.snapshot()
        assert before.used_units == 2
        assert before.used_bytes == 100
        assert before.exclusive_byte_units == 1
        assert before.queued_input_bytes == 7
        assert before.active_input_bytes == 0
        queued.cancellation.cancel()
        with pytest.raises(DaemonOperationCancelled):
            queued.future.result(timeout=5)
        assert not invoked.is_set()
        after = adapter.snapshot()
        assert after.used_units == 1
        assert after.used_bytes == after.exclusive_byte_units == 0
        assert after.queued_input_bytes == after.active_input_bytes == 0
        queued.cancellation.cancel()
        assert adapter.snapshot() == after
        release.set()
        first.future.result(timeout=5)
        later = adapter.submit(lambda: adapter.snapshot().active_input_bytes, estimated_bytes=11, exclusive_bytes=True)
        assert later.future.result(timeout=5) == 11
        final = adapter.snapshot()
        assert final.used_units == final.used_bytes == final.exclusive_byte_units == 0
        assert final.queued_input_bytes == final.active_input_bytes == 0
    finally:
        release.set()
        adapter.shutdown(wait=True)


def test_ordinary_byte_admission_cannot_amend_its_held_demand() -> None:
    adapter = BoundedComputeAdapter(max_workers=1, queue_units=0, queue_bytes=100)

    def ordinary() -> None:
        with pytest.raises(RuntimeError):
            adapter.amend_current_input_demand(200)
        assert adapter.snapshot().active_input_bytes == 7
        assert adapter.snapshot().used_bytes == 7

    try:
        adapter.submit(ordinary, estimated_bytes=7).future.result(timeout=5)
        assert adapter.snapshot().active_input_bytes == 0
    finally:
        adapter.shutdown(wait=True)


def test_exclusive_nested_preparation_requires_its_parent_exclusive_reservation() -> None:
    adapter = BoundedComputeAdapter(max_workers=1, queue_units=0, queue_bytes=100)
    invoked = threading.Event()

    def ordinary() -> None:
        with pytest.raises(RuntimeError):
            adapter.submit(invoked.set, exclusive_bytes=True)
        assert not invoked.is_set()
        assert adapter.snapshot().active_input_bytes == 7

    try:
        adapter.submit(ordinary, estimated_bytes=7).future.result(timeout=5)
        assert adapter.snapshot().used_units == 0
    finally:
        adapter.shutdown(wait=True)


def test_nested_compute_uses_one_worker_and_the_parent_reservation() -> None:
    adapter = BoundedComputeAdapter(max_workers=1, queue_units=0, queue_bytes=100)
    correlation = contextvars.ContextVar("compute_test_correlation", default="unset")
    token = correlation.set("synthetic")

    def parent() -> tuple[int, str, int, int]:
        before = adapter.snapshot()
        nested = adapter.submit(lambda: (threading.get_ident(), correlation.get()), estimated_bytes=50)
        child_thread, value = nested.future.result()
        after = adapter.snapshot()
        return child_thread, value, before.used_units, after.used_units

    try:
        result = adapter.submit(parent, estimated_bytes=100).future.result(timeout=5)
        assert result[1:] == ("synthetic", 1, 1)
        assert result[0] != threading.get_ident()
        assert adapter.snapshot().used_units == 0
    finally:
        correlation.reset(token)
        adapter.shutdown(wait=True)


def test_nested_failure_keeps_parent_owned_and_allows_cleanup() -> None:
    adapter = BoundedComputeAdapter(max_workers=1, queue_units=0, queue_bytes=100)
    cleaned: list[str] = []

    def child() -> None:
        raise ValueError("synthetic failure")

    def parent() -> None:
        try:
            adapter.submit(child).future.result()
        finally:
            assert adapter.snapshot().used_units == 1
            cleaned.append("parent cleanup")

    try:
        with pytest.raises(ValueError):
            adapter.submit(parent).future.result(timeout=5)
        assert cleaned == ["parent cleanup"]
        assert adapter.snapshot().used_units == 0
    finally:
        adapter.shutdown(wait=True)


def test_cancelled_nested_unit_drains_parent_cleanup_before_completion() -> None:
    adapter = BoundedComputeAdapter(max_workers=1, queue_units=0, queue_bytes=100)
    entered = threading.Event()
    proceed = threading.Event()
    cleaned = threading.Event()

    def parent() -> None:
        try:
            entered.set()
            assert proceed.wait(5)
            adapter.submit(check_compute_cancelled).future.result()
        finally:
            assert current_cancellation() is not None
            cleaned.set()

    try:
        operation = adapter.submit(parent)
        assert entered.wait(5)
        operation.cancellation.cancel()
        assert not operation.future.done()
        proceed.set()
        with pytest.raises(DaemonOperationCancelled):
            operation.future.result(timeout=5)
        assert cleaned.is_set()
        assert adapter.snapshot().used_units == 0
    finally:
        proceed.set()
        adapter.shutdown(wait=True)


def test_nested_map_failure_keeps_the_parent_reservation_for_cleanup() -> None:
    adapter = BoundedComputeAdapter(max_workers=1, queue_units=0, queue_bytes=100)
    visited: list[int] = []

    def child(marker: int) -> int:
        assert adapter.snapshot().used_units == 1
        visited.append(marker)
        if marker == 1:
            raise ValueError("synthetic nested map failure")
        return marker

    def parent() -> None:
        try:
            list(adapter.map(child, range(3), estimated_bytes=lambda _: 50))
        finally:
            assert adapter.snapshot().used_units == 1
            check_compute_cancelled()

    try:
        with pytest.raises(ValueError):
            adapter.submit(parent, estimated_bytes=100).future.result(timeout=5)
        assert visited == [0, 1]
        assert adapter.snapshot().used_units == 0
        assert adapter.submit(lambda: "reused").future.result(timeout=5) == "reused"
    finally:
        adapter.shutdown(wait=True)


def test_ordered_map_waits_for_its_own_byte_reservation_without_dropping_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = BoundedComputeAdapter(max_workers=3, queue_units=4, queue_bytes=10)
    release_first = threading.Event()
    refusal_count = 0
    original_submit = adapter.submit

    def submit(
        function: Callable[[], int],
        *,
        admission_class: AdmissionClass = "interactive-read",
        units: int = 1,
        estimated_bytes: int = 0,
        cancellation: CancellationHandle | None = None,
        exclusive_bytes: bool = False,
    ) -> SubmittedOperation[int]:
        nonlocal refusal_count
        try:
            return original_submit(
                function,
                admission_class=admission_class,
                units=units,
                estimated_bytes=estimated_bytes,
                cancellation=cancellation,
                exclusive_bytes=exclusive_bytes,
            )
        except DaemonBackpressureError:
            refusal_count += 1
            release_first.set()
            raise

    def job(marker: int) -> int:
        if marker == 0:
            assert release_first.wait(5)
        return marker

    monkeypatch.setattr(adapter, "submit", submit)
    try:
        assert list(adapter.map(job, range(3), estimated_bytes=lambda _: 10)) == [0, 1, 2]
        assert refusal_count >= 1
        assert adapter.snapshot().used_units == 0
        assert adapter.snapshot().used_bytes == 0
    finally:
        release_first.set()
        adapter.shutdown(wait=True)


def test_nested_context_changes_do_not_escape_to_the_parent_or_next_job() -> None:
    adapter = BoundedComputeAdapter(max_workers=1, queue_units=0, queue_bytes=100)
    correlation = contextvars.ContextVar("nested_context_correlation", default="outside")

    def child() -> str:
        assert correlation.get() == "parent"
        correlation.set("child")
        return correlation.get()

    def parent() -> tuple[str, str]:
        correlation.set("parent")
        child_value = adapter.submit(child).future.result()
        return child_value, correlation.get()

    try:
        assert adapter.submit(parent).future.result(timeout=5) == ("child", "parent")
        assert adapter.submit(correlation.get).future.result(timeout=5) == "outside"
        assert correlation.get() == "outside"
        assert adapter.snapshot().used_units == 0
    finally:
        adapter.shutdown(wait=True)


@pytest.mark.parametrize("terminal", ["success", "failure", "cancelled"])
def test_reused_worker_binds_current_logging_correlation_and_restores_parent(terminal: str) -> None:
    from polylogue.logging import bind, current_context

    adapter = BoundedComputeAdapter(max_workers=1, queue_units=0, queue_bytes=100)
    seen: list[tuple[int, dict[str, object]]] = []

    def work() -> None:
        seen.append((threading.get_ident(), dict(current_context())))
        with bind(run_id="worker-local"):
            if terminal == "failure":
                raise ValueError("synthetic failure")
            if terminal == "cancelled":
                raise DaemonOperationCancelled("synthetic cancellation")

    try:
        for run_id in ("first-request", "second-request"):
            with bind(run_id=run_id, trace_id=run_id, span_id=run_id):
                expected = dict(current_context())
                submitted = adapter.submit(work)
                if terminal == "success":
                    submitted.future.result(timeout=5)
                else:
                    with pytest.raises(ValueError if terminal == "failure" else DaemonOperationCancelled):
                        submitted.future.result(timeout=5)
                assert dict(current_context()) == expected
                assert seen[-1][1] == expected
        assert seen[0][0] == seen[1][0]
        assert adapter.submit(lambda: dict(current_context())).future.result(timeout=5) == dict(current_context())
    finally:
        adapter.shutdown(wait=True)


def test_joined_async_bridge_reuses_one_worker_reservation_and_context() -> None:
    import asyncio

    from polylogue.core.async_bridge import run_coroutine_sync

    adapter = BoundedComputeAdapter(max_workers=1, queue_units=0, queue_bytes=100)
    correlation = contextvars.ContextVar("bridge_correlation", default="outside")
    token = correlation.set("submitted")

    async def bridge_unit() -> tuple[str, int, int]:
        child = adapter.submit(lambda: (correlation.get(), threading.get_ident()), estimated_bytes=100)
        value, creator = child.future.result()
        return value, creator, adapter.snapshot().used_units

    async def parent_loop() -> tuple[str, int, int]:
        return run_coroutine_sync(bridge_unit())

    try:
        result = adapter.submit(lambda: asyncio.run(parent_loop()), estimated_bytes=100).future.result(timeout=5)
        assert result[0] == "submitted"
        assert result[1] != threading.get_ident()
        assert result[2] == 1
        assert adapter.snapshot().used_units == 0
    finally:
        correlation.reset(token)
        adapter.shutdown(wait=True)


@pytest.mark.parametrize("exclusive_bytes", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
def test_joined_bridge_failed_native_close_retains_parent_until_creator_retry(
    monkeypatch: pytest.MonkeyPatch,
    exclusive_bytes: bool,
    cancelled: bool,
) -> None:
    import asyncio

    from polylogue.core.async_bridge import run_coroutine_sync
    from polylogue.storage.io_phase_metrics import connect_measured
    from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError, NativeSQLCustodyOwner

    adapter = BoundedComputeAdapter(max_workers=1, queue_units=0, queue_bytes=100)
    opened = threading.Event()
    release_work = threading.Event()
    pending = threading.Event()
    handles: list[object] = []
    close_threads: list[threading.Thread] = []
    allow_close = threading.Event()

    async def bridge_unit() -> None:
        connection = connect_measured(":memory:")
        handles.append(connection)
        connection_type = type(connection)
        actual_close = connection_type.close

        def close(current: sqlite3.Connection) -> None:
            if current is connection:
                close_threads.append(threading.current_thread())
                if not allow_close.is_set():
                    pending.set()
                    raise OSError("synthetic bridge close refusal")
            actual_close(current)

        monkeypatch.setattr(connection_type, "close", close)
        NativeSQLCustodyOwner(connection)
        assert connection.execute("SELECT 1").fetchone() == (1,)
        opened.set()
        assert release_work.wait(5)

    async def parent_loop() -> None:
        if exclusive_bytes:
            adapter.amend_current_input_demand(200)
        run_coroutine_sync(bridge_unit())

    operation = adapter.submit(lambda: asyncio.run(parent_loop()), estimated_bytes=7, exclusive_bytes=exclusive_bytes)
    try:
        assert opened.wait(5)
        release_work.set()
        assert pending.wait(5)
        if cancelled:
            operation.cancellation.cancel()
        assert not operation.future.done()
        assert adapter.snapshot().used_units == 1
        assert adapter.snapshot().used_bytes == (100 if exclusive_bytes else 7)
        assert adapter.snapshot().active_input_bytes == (207 if exclusive_bytes else 7)
        allow_close.set()
        operation.retry_sql_settlement()
        with pytest.raises(NativeConnectionSettlementError):
            operation.future.result(timeout=5)
        assert len(close_threads) >= 2
        assert all(thread is close_threads[0] for thread in close_threads)
        assert adapter.snapshot().used_units == 0
        assert adapter.snapshot().active_input_bytes == 0
        assert adapter.snapshot().exclusive_byte_units == 0
    finally:
        release_work.set()
        allow_close.set()
        operation.retry_sql_settlement()
        adapter.shutdown(wait=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_failure", ["success", "failure", "cancelled"])
async def test_async_read_cancellation_keeps_original_worker_until_physical_result(worker_failure: str) -> None:
    import asyncio

    adapter = BoundedComputeAdapter(max_workers=1, queue_units=0)
    entered = threading.Event()
    release = threading.Event()
    settled = threading.Event()

    def read() -> int:
        entered.set()
        try:
            assert release.wait(5)
            if worker_failure == "failure":
                raise ValueError("original read failure")
            if worker_failure == "cancelled":
                raise asyncio.CancelledError("native SQLite work was cancelled")
            return 1
        finally:
            settled.set()

    submitted = adapter.submit(read, admission_class="interactive-read", estimated_bytes=7)
    waiter = asyncio.create_task(submitted.wait())
    try:
        assert entered.wait(5)
        await asyncio.sleep(0)
        waiter.cancel("original parent cancellation")
        await asyncio.sleep(0)
        waiter.cancel("repeated parent cancellation")
        await asyncio.sleep(0)
        assert not waiter.done()
        assert adapter.snapshot().active_input_bytes == 7
        release.set()
        if worker_failure == "failure":
            with pytest.raises(BaseExceptionGroup) as caught:
                await waiter
            assert any(isinstance(item, asyncio.CancelledError) for item in caught.value.exceptions)
            assert any(isinstance(item, ValueError) for item in caught.value.exceptions)
        else:
            with pytest.raises(asyncio.CancelledError) as caught_cancellation:
                await waiter
            assert caught_cancellation.value.args == ("original parent cancellation",)
        assert settled.is_set()
        assert submitted.future.done()
        assert adapter.snapshot().used_units == 0
    finally:
        release.set()
        assert adapter.close(join_timeout_s=5) == ()


def test_nested_owned_read_uses_actual_admitted_compute_creator() -> None:
    from polylogue.core.compute import compute_adapter

    adapter = BoundedComputeAdapter(max_workers=1, queue_units=0)
    try:

        def read() -> tuple[bool, bool]:
            creator = threading.get_ident()
            assert compute_adapter() is adapter
            nested = compute_adapter().submit(
                threading.get_ident, admission_class="interactive-read", estimated_bytes=3
            )
            return nested.future.done(), nested.future.result() == creator

        assert adapter.submit(read, admission_class="interactive-read", estimated_bytes=3).future.result(5) == (
            True,
            True,
        )
    finally:
        assert adapter.close(join_timeout_s=5) == ()


@pytest.mark.parametrize("capacity_bytes", [0, 100])
def test_ordered_map_retains_unknown_input_exclusivity_until_each_worker_settles(capacity_bytes: int) -> None:
    adapter = BoundedComputeAdapter(max_workers=2, queue_units=2, queue_bytes=capacity_bytes)

    def capture(size: int) -> tuple[int, int]:
        adapter.require_current_creator()
        snapshot = adapter.snapshot()
        assert snapshot.exclusive_byte_units == 1
        assert snapshot.used_bytes == capacity_bytes
        failures: list[BaseException] = []

        def unrelated_read() -> None:
            try:
                with pytest.raises(DaemonBackpressureError):
                    adapter.submit(lambda: None, estimated_bytes=1)
            except BaseException as failure:
                failures.append(failure)

        reader = threading.Thread(target=unrelated_read)
        reader.start()
        reader.join(timeout=5)
        assert not reader.is_alive()
        assert not failures
        return size, snapshot.active_input_bytes

    try:
        assert list(adapter.map(capture, (7, 11), estimated_bytes=lambda size: size, exclusive_bytes=True)) == [
            (7, 7),
            (11, 11),
        ]
        snapshot = adapter.snapshot()
        assert snapshot.used_units == snapshot.used_bytes == snapshot.exclusive_byte_units == 0
        assert adapter.submit(lambda: "settled", estimated_bytes=1).future.result(timeout=5) == "settled"
    finally:
        adapter.shutdown(wait=True)


async def test_complete_without_suspension_runs_on_a_thread_driving_a_loop() -> None:
    """A never-suspending coroutine completes where ``asyncio.run`` cannot.

    Admitted reads may run nested on a compute worker that is already driving
    an event loop. Anti-vacuity: implement the helper with ``asyncio.run`` and
    this call raises "cannot be called from a running event loop"; drop the
    suspension refusal and the suspending coroutine is reported as a
    completed ``None`` instead of being refused.
    """
    import asyncio

    from polylogue.core.async_bridge import complete_without_suspension

    async def synchronous_reader() -> str:
        return "pinned"

    async def answer() -> str:
        return await synchronous_reader()

    assert complete_without_suspension(answer()) == "pinned"

    async def suspends() -> None:
        await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="suspended"):
        complete_without_suspension(suspends())


@pytest.mark.asyncio
@pytest.mark.parametrize("completion", ["worker_cancelled_error", "cancelled_future", "success"])
async def test_wait_consumes_each_terminal_physical_result_once(
    monkeypatch: pytest.MonkeyPatch, completion: str
) -> None:
    import asyncio

    adapter = BoundedComputeAdapter(max_workers=1, queue_units=0)
    failure = asyncio.CancelledError("native SQLite work was cancelled")

    def worker() -> int:
        if completion == "worker_cancelled_error":
            raise failure
        return 7

    if completion == "cancelled_future":
        future: Future[int] = Future()
        assert future.cancel()
        operation = SubmittedOperation(future, CancellationHandle())
    else:
        operation = adapter.submit(worker)
        if completion == "worker_cancelled_error":
            with pytest.raises(asyncio.CancelledError) as caught:
                operation.future.result(timeout=5)
            assert caught.value is failure
            assert operation.future.done() and not operation.future.cancelled()
        else:
            assert operation.future.result(timeout=5) == 7
    original_shield = asyncio.shield
    shield_calls = 0

    def observe_terminal_wait(awaitable: asyncio.Future[int]) -> asyncio.Future[int]:
        nonlocal shield_calls
        shield_calls += 1
        # Fail synchronously on re-entry instead of letting the defective
        # waiter spin forever; the real shield still owns the first await.
        assert shield_calls == 1, "terminal physical result was shielded again"
        return original_shield(awaitable)

    monkeypatch.setattr(asyncio, "shield", observe_terminal_wait)
    try:
        if completion == "success":
            assert await operation.wait() == 7
        else:
            with pytest.raises(asyncio.CancelledError) as caught:
                await operation.wait()
            if completion == "worker_cancelled_error":
                assert caught.value is failure
        assert not operation.cancellation.cancelled
        assert shield_calls == 1
        assert adapter.snapshot().used_units == 0
    finally:
        adapter.shutdown(wait=True)
