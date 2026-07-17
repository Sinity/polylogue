"""Execution-control tests for archive reads (polylogue-z9gh.1).

These exercise the production primitives directly and through the real
``Polylogue.query_units`` route. The SLO thresholds are deliberately loose
for CI noise while still failing hard if the mechanism is removed:

- removing the progress guard → cancellation/deadline tests hang past SLO;
- removing worker offload (``asyncio.to_thread``) → the heartbeat test sees
  event-loop gaps the size of the whole query;
- removing admission release → the fairness/exactly-once tests fail.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from polylogue.archive.query.execution_control import (
    InterruptibleSQLiteRead,
    QueryAdmissionController,
    QueryCancelledError,
    QueryExecutionContext,
    QueryTimeoutError,
    execute_archive_read,
    execute_archive_read_sync,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

if TYPE_CHECKING:
    pass

#: An effectively unbounded statement: only the progress guard can stop it
#: inside the test SLO.
_EXPENSIVE_SQL = (
    "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM cnt WHERE x < 2000000000) SELECT count(*) FROM cnt"
)

#: Generous wall-clock bound for an aborted expensive statement. The real
#: abort latency is the progress-guard cadence (sub-millisecond scale).
_ABORT_SLO_S = 5.0


def _bootstrap_archive(tmp_path: Path) -> Path:
    with ArchiveStore(tmp_path):
        pass
    return tmp_path


def _expensive_work(store: ArchiveStore) -> object:
    return store._conn.execute(_EXPENSIVE_SQL).fetchone()


def _cheap_work(store: ArchiveStore) -> int:
    row = store._conn.execute("SELECT 1").fetchone()
    return int(row[0])


async def test_cancel_interrupts_expensive_statement_within_slo(tmp_path: Path) -> None:
    root = _bootstrap_archive(tmp_path)
    ctx = QueryExecutionContext.create(query_text="expensive", timeout_s=None)
    controller = QueryAdmissionController()

    started = time.monotonic()
    task = asyncio.create_task(execute_archive_read(root, _expensive_work, ctx=ctx, controller=controller))
    await asyncio.sleep(0.2)
    ctx.cancel()
    with pytest.raises(QueryCancelledError):
        await task
    elapsed = time.monotonic() - started

    assert elapsed < _ABORT_SLO_S
    assert ctx.receipt.state == "cancelled"
    assert ctx.receipt.interrupted is True
    assert controller.in_flight_weight == 0


async def test_deadline_aborts_expensive_statement(tmp_path: Path) -> None:
    root = _bootstrap_archive(tmp_path)
    ctx = QueryExecutionContext.create(query_text="expensive", timeout_s=0.3)
    controller = QueryAdmissionController()

    started = time.monotonic()
    with pytest.raises(QueryTimeoutError):
        await execute_archive_read(root, _expensive_work, ctx=ctx, controller=controller)
    elapsed = time.monotonic() - started

    assert elapsed < _ABORT_SLO_S
    assert ctx.receipt.state == "timed_out"
    assert controller.in_flight_weight == 0


async def test_client_disconnect_cancels_and_releases(tmp_path: Path) -> None:
    """asyncio cancellation (the MCP disconnect shape) interrupts the exact
    connection, drains the worker, and releases admission exactly once."""
    root = _bootstrap_archive(tmp_path)
    ctx = QueryExecutionContext.create(query_text="expensive", timeout_s=None)
    controller = QueryAdmissionController()

    task = asyncio.create_task(execute_archive_read(root, _expensive_work, ctx=ctx, controller=controller))
    await asyncio.sleep(0.2)
    started = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    elapsed = time.monotonic() - started

    assert elapsed < _ABORT_SLO_S
    assert ctx.cancelled is True
    assert ctx.receipt.state == "disconnected"
    assert controller.in_flight_weight == 0


async def test_event_loop_stays_responsive_during_expensive_read(tmp_path: Path) -> None:
    """A blocking-scale statement must not freeze the loop: cheap awaitables
    keep their interactive latency while the worker thread grinds."""
    root = _bootstrap_archive(tmp_path)
    ctx = QueryExecutionContext.create(query_text="expensive", timeout_s=1.5)
    controller = QueryAdmissionController()

    gaps: list[float] = []
    done = asyncio.Event()

    async def heartbeat() -> None:
        prev = time.monotonic()
        while not done.is_set():
            await asyncio.sleep(0.02)
            now = time.monotonic()
            gaps.append(now - prev)
            prev = now

    beat = asyncio.create_task(heartbeat())
    started = time.monotonic()
    with pytest.raises(QueryTimeoutError):
        await execute_archive_read(root, _expensive_work, ctx=ctx, controller=controller)
    blocking_span = time.monotonic() - started
    done.set()
    await beat

    # The read genuinely ran at blocking scale...
    assert blocking_span > 1.0
    # ...yet the loop kept its interactive heartbeat. If synchronous SQLite
    # returns to the caller thread, the largest gap becomes ~blocking_span.
    assert gaps, "heartbeat never ran"
    assert max(gaps) < 0.75


async def test_cheap_read_completes_while_expensive_read_runs(tmp_path: Path) -> None:
    root = _bootstrap_archive(tmp_path)
    controller = QueryAdmissionController()
    slow_ctx = QueryExecutionContext.create(query_text="expensive", timeout_s=2.0)
    slow = asyncio.create_task(execute_archive_read(root, _expensive_work, ctx=slow_ctx, controller=controller))
    await asyncio.sleep(0.1)

    cheap_ctx = QueryExecutionContext.create(query_text="cheap", timeout_s=5.0)
    started = time.monotonic()
    value = await execute_archive_read(root, _cheap_work, ctx=cheap_ctx, controller=controller)
    cheap_latency = time.monotonic() - started

    assert value == 1
    assert cheap_latency < 1.0
    slow_ctx.cancel()
    with pytest.raises(QueryCancelledError):
        await slow


def test_admission_fifo_within_class(tmp_path: Path) -> None:
    controller = QueryAdmissionController(capacity=1, reserved_interactive=0)
    holder = QueryExecutionContext.create(query_text="hold", timeout_s=None)
    order: list[str] = []
    admitted = threading.Event()

    def _hold() -> None:
        with controller.admit_blocking(holder):
            admitted.set()
            release.wait(timeout=10)

    release = threading.Event()
    hold_thread = threading.Thread(target=_hold)
    hold_thread.start()
    assert admitted.wait(timeout=5)

    contexts = [QueryExecutionContext.create(query_text=f"queued-{i}", timeout_s=None) for i in range(3)]

    def _worker(ctx: QueryExecutionContext) -> None:
        with controller.admit_blocking(ctx):
            order.append(ctx.query_ref)

    threads = []
    for ctx in contexts:
        thread = threading.Thread(target=_worker, args=(ctx,))
        thread.start()
        threads.append(thread)
        # Ensure deterministic enqueue order before starting the next waiter.
        deadline = time.monotonic() + 5
        while controller.queue_position(ctx) is None and time.monotonic() < deadline:
            time.sleep(0.005)

    assert [controller.queue_position(ctx) for ctx in contexts] == [0, 1, 2]
    release.set()
    hold_thread.join(timeout=10)
    for thread in threads:
        thread.join(timeout=10)

    assert order == [ctx.query_ref for ctx in contexts]
    assert controller.in_flight_weight == 0


def test_reserved_interactive_capacity_blocks_scans_not_interactive() -> None:
    controller = QueryAdmissionController(capacity=2, reserved_interactive=1)
    scan_a = QueryExecutionContext.create(query_text="scan-a", workload_class="scan", timeout_s=None)
    scan_b = QueryExecutionContext.create(query_text="scan-b", workload_class="scan", timeout_s=None)
    interactive = QueryExecutionContext.create(query_text="cheap", workload_class="interactive", timeout_s=None)

    scan_b_admitted = threading.Event()
    release_a = threading.Event()
    release_b = threading.Event()

    def _run_scan_a() -> None:
        with controller.admit_blocking(scan_a):
            release_a.wait(timeout=10)

    def _run_scan_b() -> None:
        with controller.admit_blocking(scan_b):
            scan_b_admitted.set()
            release_b.wait(timeout=10)

    thread_a = threading.Thread(target=_run_scan_a)
    thread_a.start()
    deadline = time.monotonic() + 5
    while controller.in_flight_weight != 1 and time.monotonic() < deadline:
        time.sleep(0.005)

    thread_b = threading.Thread(target=_run_scan_b)
    thread_b.start()
    time.sleep(0.15)
    # Scan class ceiling is capacity - reserved = 1, so scan B stays queued...
    assert not scan_b_admitted.is_set()
    assert controller.queue_position(scan_b) == 0

    # ...while an interactive read admits into the reserved headroom.
    with controller.admit_blocking(interactive):
        assert controller.in_flight_weight == 2

    release_a.set()
    thread_a.join(timeout=10)
    assert scan_b_admitted.wait(timeout=5)
    release_b.set()
    thread_b.join(timeout=10)
    assert controller.in_flight_weight == 0


def test_oversized_weight_is_clamped_never_refused() -> None:
    controller = QueryAdmissionController(capacity=4, reserved_interactive=1)
    huge_scan = QueryExecutionContext.create(
        query_text="huge", workload_class="scan", admission_weight=999, timeout_s=None
    )

    assert controller.clamped_weight(huge_scan) == 3
    with controller.admit_blocking(huge_scan):
        assert controller.in_flight_weight == 3
    assert controller.in_flight_weight == 0


def test_cancel_while_queued_releases_slot_and_raises() -> None:
    controller = QueryAdmissionController(capacity=1, reserved_interactive=0)
    holder = QueryExecutionContext.create(query_text="hold", timeout_s=None)
    queued = QueryExecutionContext.create(query_text="queued", timeout_s=None)
    release = threading.Event()
    admitted = threading.Event()
    outcome: list[BaseException | str] = []

    def _hold() -> None:
        with controller.admit_blocking(holder):
            admitted.set()
            release.wait(timeout=10)

    def _queued() -> None:
        try:
            with controller.admit_blocking(queued):
                outcome.append("admitted")
        except QueryCancelledError as exc:
            outcome.append(exc)

    hold_thread = threading.Thread(target=_hold)
    hold_thread.start()
    assert admitted.wait(timeout=5)
    queued_thread = threading.Thread(target=_queued)
    queued_thread.start()
    deadline = time.monotonic() + 5
    while controller.queue_position(queued) is None and time.monotonic() < deadline:
        time.sleep(0.005)

    queued.cancel()
    queued_thread.join(timeout=10)
    assert len(outcome) == 1
    assert isinstance(outcome[0], QueryCancelledError)
    assert controller.queue_position(queued) is None

    release.set()
    hold_thread.join(timeout=10)
    assert controller.in_flight_weight == 0


def test_admission_release_is_exactly_once() -> None:
    controller = QueryAdmissionController(capacity=2, reserved_interactive=0)
    ctx = QueryExecutionContext.create(query_text="once", timeout_s=None)
    with controller.admit_blocking(ctx):
        assert controller.in_flight_weight == 1
    assert controller.in_flight_weight == 0
    # A duplicate release (e.g. a double cleanup path) must be a no-op, not
    # an in-flight underflow that would over-admit later work.
    controller._release(ctx, 1)
    assert controller.in_flight_weight == 0


def test_receipts_carry_safe_refs_and_terminal_states(tmp_path: Path) -> None:
    root = _bootstrap_archive(tmp_path)
    secret_expression = "sessions where text:extremely-private-token"
    ctx = QueryExecutionContext.create(query_text=secret_expression, timeout_s=5.0)
    controller = QueryAdmissionController()

    value = execute_archive_read_sync(root, _cheap_work, ctx=ctx, controller=controller)

    assert value == 1
    receipt = ctx.receipt.to_dict()
    assert receipt["state"] == "completed"
    assert str(receipt["query_ref"]).startswith("expr:")
    assert "extremely-private-token" not in str(receipt)

    failing_ctx = QueryExecutionContext.create(query_text="boom", timeout_s=5.0)

    def _boom(store: ArchiveStore) -> None:
        raise RuntimeError("worker exploded")

    with pytest.raises(RuntimeError):
        execute_archive_read_sync(root, _boom, ctx=failing_ctx, controller=controller)
    assert failing_ctx.receipt.state == "failed"
    assert failing_ctx.receipt.error == "RuntimeError"
    assert controller.in_flight_weight == 0


def test_sync_deadline_aborts_for_http_route(tmp_path: Path) -> None:
    root = _bootstrap_archive(tmp_path)
    ctx = QueryExecutionContext.create(query_text="expensive", timeout_s=0.3)
    controller = QueryAdmissionController()

    started = time.monotonic()
    with pytest.raises(QueryTimeoutError):
        execute_archive_read_sync(root, _expensive_work, ctx=ctx, controller=controller)
    assert time.monotonic() - started < _ABORT_SLO_S
    assert ctx.receipt.state == "timed_out"


def test_interrupt_before_first_opcode_never_starts_statement(tmp_path: Path) -> None:
    root = _bootstrap_archive(tmp_path)
    ctx = QueryExecutionContext.create(query_text="pre-cancelled", timeout_s=None)
    ctx.cancel()
    reader = InterruptibleSQLiteRead(ctx)

    executed: list[bool] = []

    def _work(store: ArchiveStore) -> None:
        executed.append(True)

    with pytest.raises(QueryCancelledError):
        reader.run(root, _work)
    assert executed == []
    assert ctx.receipt.state == "cancelled"


async def test_api_query_units_routes_through_execution_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production ``Polylogue.query_units`` path consumes the execution
    primitives — removing that wiring (calling the envelope builder directly
    on the loop again) fails here."""
    from polylogue import Polylogue
    from polylogue.surfaces.payloads import QueryUnitEnvelope

    _bootstrap_archive(tmp_path)
    seen: list[str] = []
    original_run = InterruptibleSQLiteRead.run

    def _recording_run(self: InterruptibleSQLiteRead, *args: object, **kwargs: object) -> object:
        seen.append(self._ctx.owner_ref or "?")
        return original_run(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(InterruptibleSQLiteRead, "run", _recording_run)

    archive = Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")
    try:
        envelope = await archive.query_units("messages where text:missing")
    finally:
        await archive.close()

    assert isinstance(envelope, QueryUnitEnvelope)
    assert seen == ["api.query_units"]


def test_queued_deadline_expiry_reports_timeout_not_cancellation() -> None:
    """A deadline that expires while waiting for capacity is a timeout —
    surfaces map QueryTimeoutError to 503, not a generic failure."""
    controller = QueryAdmissionController(capacity=1, reserved_interactive=0)
    holder = QueryExecutionContext.create(query_text="hold", timeout_s=None)
    release = threading.Event()
    admitted = threading.Event()

    def _hold() -> None:
        with controller.admit_blocking(holder):
            admitted.set()
            release.wait(timeout=10)

    hold_thread = threading.Thread(target=_hold)
    hold_thread.start()
    assert admitted.wait(timeout=5)

    queued = QueryExecutionContext.create(query_text="queued", timeout_s=0.2)
    with pytest.raises(QueryTimeoutError):
        with controller.admit_blocking(queued):
            pass
    assert queued.receipt.state == "timed_out"

    release.set()
    hold_thread.join(timeout=10)
    assert controller.in_flight_weight == 0


def test_abort_during_python_post_processing_discards_result(tmp_path: Path) -> None:
    """The progress guard only sees SQLite; a deadline that lands during
    Python-side post-processing must still abort instead of completing."""
    root = _bootstrap_archive(tmp_path)
    ctx = QueryExecutionContext.create(query_text="python-phase", timeout_s=0.2)
    controller = QueryAdmissionController()

    def _python_heavy(store: ArchiveStore) -> str:
        store._conn.execute("SELECT 1").fetchone()
        time.sleep(0.5)  # deadline expires during non-SQL work
        return "stale result"

    with pytest.raises(QueryTimeoutError):
        execute_archive_read_sync(root, _python_heavy, ctx=ctx, controller=controller)
    assert ctx.receipt.state == "timed_out"
    assert controller.in_flight_weight == 0


async def test_disconnect_drain_is_bounded_for_uninterruptible_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker stuck in Python-side work cannot hold the disconnect path
    hostage: the drain wait is bounded and the worker cleans up on its own."""
    import polylogue.archive.query.execution_control as ec

    monkeypatch.setattr(ec, "DISCONNECT_DRAIN_TIMEOUT_S", 0.3)
    root = _bootstrap_archive(tmp_path)
    ctx = QueryExecutionContext.create(query_text="stuck", timeout_s=None)
    controller = QueryAdmissionController()
    worker_done = threading.Event()

    def _stuck(store: ArchiveStore) -> None:
        try:
            time.sleep(1.2)  # ignores cancellation: pure Python phase
        finally:
            worker_done.set()

    task = asyncio.create_task(execute_archive_read(root, _stuck, ctx=ctx, controller=controller))
    await asyncio.sleep(0.15)
    started = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    disconnect_latency = time.monotonic() - started

    assert disconnect_latency < 1.0  # bounded: did not wait out the full sleep
    assert ctx.receipt.state == "disconnected"
    # The orphaned worker still finishes and releases its admission slot.
    assert worker_done.wait(timeout=5)
    deadline = time.monotonic() + 5
    while controller.in_flight_weight != 0 and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    assert controller.in_flight_weight == 0


def test_workload_classification_routes_aggregates_to_scan() -> None:
    from polylogue.archive.query.execution_control import classify_unit_expression_workload

    assert classify_unit_expression_workload("messages where text:hello") == "interactive"
    assert classify_unit_expression_workload("actions where tool:bash | count") == "scan"
    assert classify_unit_expression_workload("messages where text:x | group by origin") == "scan"
    # Invalid/non-terminal expressions classify interactive; route-level
    # validation owns rejection.
    assert classify_unit_expression_workload("sessions where repo:x | group by origin | count") == "interactive"
    assert classify_unit_expression_workload("not a valid expression (") == "interactive"


async def test_api_query_units_classifies_aggregate_as_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The production route derives workload class from the expression —
    hard-coding interactive again fails here."""
    from polylogue import Polylogue

    _bootstrap_archive(tmp_path)
    seen: list[str] = []
    original_run = InterruptibleSQLiteRead.run

    def _recording_run(self: InterruptibleSQLiteRead, *args: object, **kwargs: object) -> object:
        seen.append(self._ctx.workload_class)
        return original_run(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(InterruptibleSQLiteRead, "run", _recording_run)

    archive = Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")
    try:
        await archive.query_units("messages where text:missing | count")
    finally:
        await archive.close()

    assert seen == ["scan"]
