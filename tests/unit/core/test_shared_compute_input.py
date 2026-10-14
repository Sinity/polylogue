"""Retained page input shares one charge, with actual creator settlement."""

from __future__ import annotations

import sqlite3
import threading

import pytest

from polylogue.core.compute import (
    FOREGROUND_BYTE_ALLOWANCE,
    BoundedComputeAdapter,
    DaemonBackpressureError,
    DaemonOperationCancelled,
    SharedInputReservation,
)

pytestmark = pytest.mark.uses_real_clock("physical worker and cancellation rendezvous")


def _capture(adapter: BoundedComputeAdapter, size: int) -> SharedInputReservation:
    return adapter.submit(
        lambda: adapter.retain_current_shared_input(size, owner=object()),
        admission_class="incremental-background",
        exclusive_bytes=True,
    ).future.result(timeout=5)


def test_shared_charge_survives_capture_and_counts_two_private_borrowers_once() -> None:
    adapter = BoundedComputeAdapter(max_workers=4, queue_units=4, queue_bytes=100)
    release = threading.Event()
    first_started, second_started = threading.Event(), threading.Event()
    shared = _capture(adapter, 60)

    def worker(started: threading.Event) -> None:
        started.set()
        assert release.wait(5)

    try:
        before = adapter.snapshot()
        assert (before.used_units, before.used_bytes, before.queued_bytes) == (0, 60, 0)
        assert (before.retained_input_count, before.retained_input_bytes) == (1, 60)
        first = adapter.submit(
            lambda: worker(first_started),
            admission_class="incremental-background",
            estimated_bytes=20,
            shared_input=shared,
        )
        second = adapter.submit(
            lambda: worker(second_started),
            admission_class="incremental-background",
            estimated_bytes=20,
            shared_input=shared,
        )
        assert first_started.wait(5) and second_started.wait(5)
        snapshot = adapter.snapshot()
        assert (snapshot.used_bytes, snapshot.active_input_bytes, snapshot.retained_input_bytes) == (100, 40, 60)
        with pytest.raises(DaemonBackpressureError):
            adapter.submit(lambda: None, admission_class="incremental-background", estimated_bytes=1)
        with pytest.raises(RuntimeError, match="physical compute borrowers"):
            shared.require_settled()
        shared.close()
        assert adapter.snapshot().retained_input_bytes == 60
        with pytest.raises(ValueError, match="closed"):
            adapter.submit(lambda: None, admission_class="incremental-background", shared_input=shared)
        release.set()
        first.future.result(timeout=5)
        second.future.result(timeout=5)
        assert adapter.snapshot().used_bytes == adapter.snapshot().retained_input_bytes == 0
        shared.close()
        assert adapter.snapshot().used_bytes == 0
    finally:
        release.set()
        shared.close()
        adapter.shutdown(wait=True)


@pytest.mark.parametrize("wrong_owner", ["adapter", "class", "nested"])
def test_shared_input_cannot_cross_adapter_class_or_current_parent(wrong_owner: str) -> None:
    adapter = BoundedComputeAdapter(max_workers=2, queue_bytes=100)
    other = BoundedComputeAdapter(max_workers=2, queue_bytes=100)
    shared = _capture(adapter, 60)
    try:
        if wrong_owner == "adapter":
            with pytest.raises(ValueError, match="adapter or admission class"):
                other.submit(lambda: None, admission_class="incremental-background", shared_input=shared)
        elif wrong_owner == "class":
            with pytest.raises(ValueError, match="adapter or admission class"):
                adapter.submit(lambda: None, admission_class="bulk-candidate", shared_input=shared)
        else:

            def parent() -> None:
                adapter.submit(lambda: None, admission_class="incremental-background", shared_input=shared)

            with pytest.raises(RuntimeError, match="different shared reservation"):
                adapter.submit(parent, admission_class="incremental-background").future.result(timeout=5)
        assert adapter.snapshot().used_bytes == 60
    finally:
        shared.close()
        adapter.shutdown(wait=True)
        other.shutdown(wait=True)


@pytest.mark.parametrize("shared_bytes", [60, 200])
def test_oversized_and_exclusive_borrowers_use_own_remaining_envelope(shared_bytes: int) -> None:
    adapter = BoundedComputeAdapter(max_workers=3, queue_bytes=100)
    shared = _capture(adapter, shared_bytes)
    started, release = threading.Event(), threading.Event()
    try:

        def exclusive() -> str:
            snapshot = adapter.snapshot()
            assert snapshot.used_bytes == 100
            assert snapshot.retained_input_bytes == shared_bytes
            adapter.amend_current_input_demand(500)
            started.set()
            assert release.wait(5)
            return "published"

        operation = adapter.submit(
            exclusive,
            admission_class="incremental-background",
            exclusive_bytes=True,
            estimated_bytes=11,
            shared_input=shared,
        )
        assert started.wait(5)
        with pytest.raises(DaemonBackpressureError):
            adapter.submit(
                lambda: None, admission_class="incremental-background", estimated_bytes=1, shared_input=shared
            )
        release.set()
        assert operation.future.result(timeout=5) == "published"
        assert (
            adapter.submit(
                lambda: "parsed", admission_class="incremental-background", estimated_bytes=101, shared_input=shared
            ).future.result(timeout=5)
            == "parsed"
        )
        assert adapter.snapshot().used_bytes == min(shared_bytes, 100)
    finally:
        release.set()
        shared.close()
        adapter.shutdown(wait=True)


def test_queued_cancel_retains_shared_input_until_its_running_borrower_settles() -> None:
    adapter = BoundedComputeAdapter(max_workers=1, queue_units=2, queue_bytes=100)
    shared = _capture(adapter, 30)
    started, release = threading.Event(), threading.Event()

    def worker() -> None:
        started.set()
        assert release.wait(5)

    first = adapter.submit(worker, admission_class="incremental-background", estimated_bytes=20, shared_input=shared)
    try:
        assert started.wait(5)
        second = adapter.submit(
            lambda: pytest.fail("cancelled queued borrower ran"),
            admission_class="incremental-background",
            estimated_bytes=20,
            shared_input=shared,
        )
        second.cancellation.cancel()
        with pytest.raises(DaemonOperationCancelled):
            second.future.result(timeout=5)
        shared.close()
        assert adapter.snapshot().used_bytes == 50
        release.set()
        first.future.result(timeout=5)
        assert adapter.snapshot().retained_input_count == adapter.snapshot().used_bytes == 0
    finally:
        release.set()
        shared.close()
        adapter.shutdown(wait=True)


def test_native_failed_close_keeps_shared_input_on_original_borrower_until_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.io_phase_metrics import connect_measured
    from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError, NativeSQLCustodyOwner

    adapter = BoundedComputeAdapter(max_workers=1, queue_bytes=100)
    shared = _capture(adapter, 60)
    pending, allow_close = threading.Event(), threading.Event()
    close_threads: list[threading.Thread] = []

    def worker() -> None:
        connection = connect_measured(":memory:")
        actual_close = type(connection).close

        def close(current: sqlite3.Connection) -> None:
            if current is connection:
                close_threads.append(threading.current_thread())
                if not allow_close.is_set():
                    pending.set()
                    raise OSError("synthetic borrowed close refusal")
            actual_close(current)

        monkeypatch.setattr(type(connection), "close", close)
        NativeSQLCustodyOwner(connection)
        assert connection.execute("SELECT 1").fetchone() == (1,)

    operation = adapter.submit(
        worker, admission_class="incremental-background", estimated_bytes=20, shared_input=shared
    )
    try:
        assert pending.wait(5)
        shared.close()
        operation.cancellation.cancel()
        assert not operation.future.done()
        assert adapter.snapshot().used_bytes == 80
        assert adapter.snapshot().retained_input_bytes == 60
        allow_close.set()
        operation.retry_sql_settlement()
        with pytest.raises(NativeConnectionSettlementError):
            operation.future.result(timeout=5)
        assert len(close_threads) >= 2 and all(thread is close_threads[0] for thread in close_threads)
        assert adapter.snapshot().used_bytes == adapter.snapshot().retained_input_count == 0
    finally:
        allow_close.set()
        operation.retry_sql_settlement()
        shared.close()
        adapter.shutdown(wait=True)


def test_page_retirement_fences_new_borrowing_before_payload_credit_releases() -> None:
    adapter = BoundedComputeAdapter(max_workers=2, queue_bytes=100)
    shared = _capture(adapter, 60)
    try:
        shared.require_settled()
        assert adapter.snapshot().used_bytes == 60
        with pytest.raises(ValueError, match="closed"):
            adapter.submit(lambda: None, admission_class="incremental-background", shared_input=shared)
        # A failed payload cleanup retries while retaining this same credit.
        shared.require_settled()
        assert adapter.snapshot().retained_input_count == 1
        shared.close()
        assert adapter.snapshot().used_bytes == 0
    finally:
        shared.close()
        adapter.shutdown(wait=True)


def test_reset_retains_shared_owner_after_workers_finish_until_page_retires(monkeypatch: pytest.MonkeyPatch) -> None:
    import polylogue.core.compute as module

    adapter = BoundedComputeAdapter(max_workers=1, queue_bytes=100)
    other = BoundedComputeAdapter(max_workers=1, queue_bytes=100)
    monkeypatch.setattr(module, "_SHARED_COMPUTE_ADAPTER", adapter)
    shared = _capture(adapter, 60)
    try:
        with pytest.raises(RuntimeError, match="still retains page input"):
            module.reset_compute_adapter(join_timeout_s=5)
        assert module.compute_adapter() is adapter
        assert adapter.snapshot().used_units == 0
        assert adapter.snapshot().retained_input_bytes == 60
        with pytest.raises(RuntimeError, match="physically settle"):
            module.publish_compute_adapter(other)
        shared.close()
        assert module.reset_compute_adapter(join_timeout_s=5) == ()
        module.publish_compute_adapter(other)
        assert module.compute_adapter() is other
    finally:
        shared.close()
        adapter.shutdown(wait=True)
        other.shutdown(wait=True)


@pytest.mark.parametrize("shared_bytes", [60, 200])
def test_waiting_large_read_does_not_block_its_page_publication(shared_bytes: int) -> None:
    adapter = BoundedComputeAdapter(max_workers=3, queue_bytes=100)
    shared = _capture(adapter, shared_bytes)
    started, release = threading.Event(), threading.Event()
    read_ran = threading.Event()

    def parser() -> None:
        started.set()
        assert release.wait(5)

    parsing = adapter.submit(parser, admission_class="incremental-background", estimated_bytes=101, shared_input=shared)
    try:
        assert started.wait(5)
        reading = adapter.submit(read_ran.set, estimated_bytes=FOREGROUND_BYTE_ALLOWANCE + 1)
        assert not read_ran.is_set()
        assert adapter.snapshot().foreground_pending_bytes == 100
        release.set()
        parsing.future.result(timeout=5)
        assert not read_ran.is_set()
        publishing = adapter.submit(
            lambda: "published",
            admission_class="incremental-background",
            exclusive_bytes=True,
            estimated_bytes=1,
            shared_input=shared,
        )
        assert publishing.future.result(timeout=5) == "published"
        assert not read_ran.is_set()
        shared.close()
        reading.future.result(timeout=5)
        assert read_ran.is_set()
        assert adapter.snapshot().retained_input_count == adapter.snapshot().used_bytes == 0
    finally:
        release.set()
        shared.close()
        adapter.shutdown(wait=True)
