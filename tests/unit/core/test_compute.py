"""Physical ownership laws for the shared pure compute adapter."""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Callable

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
    adapter = BoundedComputeAdapter(max_workers=2, queue_units=4, queue_bytes=10)
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
    ) -> SubmittedOperation[int]:
        nonlocal refusal_count
        try:
            return original_submit(
                function,
                admission_class=admission_class,
                units=units,
                estimated_bytes=estimated_bytes,
                cancellation=cancellation,
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


def test_joined_bridge_failed_native_close_retains_parent_until_creator_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from polylogue.core.async_bridge import run_coroutine_sync
    from polylogue.storage.io_phase_metrics import connect_measured
    from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError, NativeSQLCustodyOwner

    adapter = BoundedComputeAdapter(max_workers=1, queue_units=0)
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

        def close(current: object) -> None:
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
        run_coroutine_sync(bridge_unit())

    operation = adapter.submit(lambda: asyncio.run(parent_loop()))
    try:
        assert opened.wait(5)
        release_work.set()
        assert pending.wait(5)
        assert not operation.future.done()
        assert adapter.snapshot().used_units == 1
        allow_close.set()
        operation.retry_sql_settlement()
        with pytest.raises(NativeConnectionSettlementError):
            operation.future.result(timeout=5)
        assert len(close_threads) >= 2
        assert all(thread is close_threads[0] for thread in close_threads)
        assert adapter.snapshot().used_units == 0
    finally:
        release_work.set()
        allow_close.set()
        operation.retry_sql_settlement()
        adapter.shutdown(wait=True)
