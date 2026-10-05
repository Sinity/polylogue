"""Process-wide serialization for daemon archive writers.

SQLite remains the archive's physical single-writer boundary. This module
makes that boundary explicit inside ``polylogued`` so independent async loops
queue before opening write connections instead of contending through SQLite's
busy timeout.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import heapq
import os
import queue
import threading
import time
import weakref
from builtins import BaseExceptionGroup
from collections.abc import Awaitable, Callable, Coroutine, Iterator, Mapping
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import InvalidStateError
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, ParamSpec, TypeVar

from polylogue.core.compute import retain_current_creator_settlement
from polylogue.core.write_admission import WriteAdmission, active_write_admission
from polylogue.core.write_hold import enter_write_hold, exit_write_hold
from polylogue.core.write_lease import (
    WriteLeaseDelegation,
    adopt_write_lease,
    async_write_lease,
    bind_write_lease_thread,
    current_write_lease,
    delegate_write_lease,
    grant_write_lease_thread,
)
from polylogue.logging import ERROR, INFO, WARNING, emit

if TYPE_CHECKING:
    from polylogue.core.sql_settlement import AsyncSQLCustodyOwner, SQLCustodyOwner

P = ParamSpec("P")
T = TypeVar("T")
WritePhase = Literal["queued", "acquired", "released"]
WriteOutcome = Literal["success", "error", "cancelled"]

# Bulk-ingest actors (live watcher catch-up/append batches) re-queue almost
# immediately after each release, so under sustained backlog a strict-FIFO
# gate lets them monopolize admission indefinitely: a periodic maintenance
# actor (FTS merge, WAL checkpoint, convergence) that only wakes every N
# seconds could be stuck behind an ever-refilling line of ingest waiters
# (polylogue-de2a, live incident 2026-07-26: a 14-minute single hold plus
# continuous re-queueing starved FTS merge long enough for messages_fts_data
# to balloon to 705K rows). Everything except "watcher.*" actors is treated
# as maintenance/interactive and admitted ahead of any queued watcher actor,
# bounding worst-case maintenance wait to "current hold + at most one more
# already-queued ingest hold" instead of "current hold + unbounded backlog".
_BULK_INGEST_PRIORITY = 1
_DEFAULT_PRIORITY = 0
_MAX_DETACHED_WRITER_FAILURE_ACTORS = 32
_MAX_DETACHED_WRITER_FAILURE_ACTOR_LENGTH = 128
_DETACHED_WRITER_FAILURE_OVERFLOW_ACTOR = "<other>"
_DETACHED_WRITER_FAILURE_RESERVED_ACTOR_PREFIX = "<other>"
#: Cadence for the two ownership waits that have no clock to wait against: a
#: delegated route body that is still inside the archive, and an admitted
#: operation whose receipt can only arrive through the owner loop. Both are
#: bounded by a real event (settlement, loop death), not by these numbers.
_DELEGATION_SETTLEMENT_POLL_S = 0.01
_DELEGATION_SETTLEMENT_WARN_S = 5.0
_OWNER_LIVENESS_POLL_S = 0.25


def _report_handed_off_hold(actor: str, future: ConcurrentFuture[None]) -> None:
    """Drain a handed-off hold's terminal state so no failure is swallowed."""
    if future.cancelled():
        return
    error = future.exception()
    if error is None:
        return
    emit(
        "daemon.writer.handed_off_hold_failed",
        level=ERROR,
        outcome="error",
        reason="handed_off_hold_failed",
        actor=actor,
        error_type=type(error).__name__,
        error_detail=str(error),
    )


class DaemonWriterOwnerLoopStopped(RuntimeError):  # noqa: N818 - typed outcome name
    """An admitted write's owner loop stopped before it reported settlement.

    Not a cancellation and not a rollback: nothing was withdrawn, so the
    caller must re-read the archive rather than assume no effect landed.
    """

    code = "writer_owner_loop_stopped"


#: Declared hold budgets, longest matching actor prefix wins.
#:
#: These thresholds report long holds without withdrawing admitted work.
#: Explicit cancellation and terminal SQL settlement determine writer lifetime.
WRITE_HOLD_BUDGETS_S: Mapping[str, float] = {
    "watcher.catch_up.chunk": 30.0,
    "watcher.live_ingest": 30.0,
    "watcher.": 30.0,
    # Checkpointing gets its own ceiling rather than the general maintenance
    # one: it is the recurring hold most likely to grow with archive size, and
    # a budget it shares with publication cannot show that it did
    # (CHECKPOINT_HOLD_BUDGET_S in storage/sqlite/connection_profile.py, not
    # imported here because the daemon ring may not reach into storage).
    "maintenance.wal_checkpoint": 20.0,
    "maintenance.": 120.0,
}
_DEFAULT_WRITE_HOLD_BUDGET_S = 60.0


def write_hold_budget_s(actor: str) -> float:
    """The declared maximum hold for this actor: longest matching prefix."""
    best = ""
    for prefix in WRITE_HOLD_BUDGETS_S:
        if actor.startswith(prefix) and len(prefix) > len(best):
            best = prefix
    return WRITE_HOLD_BUDGETS_S[best] if best else _DEFAULT_WRITE_HOLD_BUDGET_S


def _actor_priority(actor: str) -> int:
    """Classify a writer actor for queue-admission ordering, not execution order.

    This only changes which *queued* request is admitted next when several are
    waiting; it does not preempt an actor that already holds the gate.
    """
    if actor.startswith("watcher."):
        return _BULK_INGEST_PRIORITY
    return _DEFAULT_PRIORITY


class _PriorityGate:
    """Single-holder async gate admitting the lowest-priority queued waiter first.

    Mirrors ``asyncio.Lock``'s cancellation-safety shape (a waiter removes
    itself from the wait set in all cases; a cancelled-after-woken waiter
    re-triggers the wake so the grant is never silently dropped) but orders
    waiters by ``(priority, sequence)`` instead of pure FIFO.
    """

    def __init__(self) -> None:
        self._locked = False
        self._waiters: list[tuple[int, int, asyncio.Future[None]]] = []
        self._sequence = 0

    @property
    def locked(self) -> bool:
        return self._locked

    async def acquire(self, priority: int) -> None:
        if not self._locked and not self._waiters:
            self._locked = True
            return
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        self._sequence += 1
        entry = (priority, self._sequence, fut)
        heapq.heappush(self._waiters, entry)
        try:
            await fut
        except asyncio.CancelledError:
            self._discard(entry)
            if not self._locked:
                self._wake_next()
            raise
        self._discard(entry)
        self._locked = True

    def release(self) -> None:
        if not self._locked:
            raise RuntimeError("_PriorityGate.release() called while not held")
        self._locked = False
        self._wake_next()

    def _discard(self, entry: tuple[int, int, asyncio.Future[None]]) -> None:
        try:
            self._waiters.remove(entry)
        except ValueError:
            return
        heapq.heapify(self._waiters)

    def _wake_next(self) -> None:
        while self._waiters:
            _, _, fut = self._waiters[0]
            if fut.done():
                if fut.cancelled():
                    heapq.heappop(self._waiters)
                    continue
                # Already granted and not yet resumed: the grant is in flight.
                # Skipping past it would hand the gate to a second waiter
                # (asyncio.Lock inspects only the head for the same reason).
                return
            fut.set_result(None)
            return


@dataclass(frozen=True, slots=True)
class DaemonWriteEvent:
    """One attributable queue/hold transition for a daemon writer."""

    phase: WritePhase
    actor: str
    sequence: int
    queue_depth: int
    wait_seconds: float | None = None
    hold_seconds: float | None = None
    outcome: WriteOutcome | None = None
    #: The declared budget for this actor and whether the hold exceeded it.
    #: Set on ``released`` events only.
    hold_budget_s: float | None = None
    hold_over_budget: bool = False


@dataclass(frozen=True, slots=True)
class DaemonWriteSnapshot:
    """Request-safe in-memory view of coordinator state."""

    active_actor: str | None
    queued_actors: tuple[str, ...]
    last_event: DaemonWriteEvent | None
    accepting: bool = True
    # polylogue-es7b: daemon-lifetime count of detached background-writer
    # tasks that raised -- previously surfaced only via a log line, with no
    # counter across the daemon's lifetime.
    detached_writer_failures: int = 0
    # Retain actor/session attribution alongside the scalar counter.
    detached_writer_failures_by_actor: tuple[tuple[str, int], ...] = ()
    #: Daemon-lifetime count of holds that ran past their declared budget.
    #: Nonzero means some writer held the single gate long enough to starve a
    #: non-gated one (polylogue-8qm4k).
    over_budget_holds: int = 0
    unsettled_writer_workers: int = 0
    unsettled_async_backends: int = 0
    sql_settlement_state: str = "idle"


@dataclass(slots=True)
class _WriteRequest:
    actor: str
    sequence: int
    queued_at: float
    acquired: bool = False
    caller_cancelled: bool = False


WriteEventObserver = Callable[[DaemonWriteEvent], None]
_TELEMETRY_LOCK = threading.Lock()
_LATEST_TELEMETRY: dict[str, object] = {
    "active_actor": None,
    "queued_actors": [],
    "queue_depth": 0,
    "accepting": True,
    "last_event": None,
    "detached_writer_failures": 0,
    "detached_writer_failures_by_actor": {},
    "over_budget_holds": 0,
    "unsettled_writer_workers": 0,
    "unsettled_async_backends": 0,
    "sql_settlement_state": "idle",
}


class DaemonWriterSettlementError(RuntimeError):
    """An original writer still owns SQL; a later admission may retry cleanup."""

    code = "writer_sql_unsettled"
    retryable = True
    is_transient = True


def _report_terminal_settlement(attempt: ConcurrentFuture[None]) -> None:
    """Preserve the actual cleanup result even when its waiter cancelled."""
    if attempt.cancelled():
        return
    error = attempt.exception()
    if error is not None:
        emit(
            "daemon.writer.terminal_settlement_failed",
            level=WARNING,
            outcome="error",
            reason="sql_cleanup_failed",
            error_type=type(error).__name__,
        )


class _TerminalWriter:
    """A cleanup-only mailbox for one existing admitted worker."""

    def __init__(
        self,
        cleanup: Callable[[], None],
        pending: Callable[[], bool],
        retire: Callable[[], None],
        settled: Callable[[], None],
    ) -> None:
        self._pid = os.getpid()
        self._cleanup = cleanup
        self._pending = pending
        self._retire = retire
        self._settled = settled
        self._guard = threading.Lock()
        self._requests: queue.SimpleQueue[ConcurrentFuture[None]] = queue.SimpleQueue()
        self._attempt: ConcurrentFuture[None] | None = None
        self._retired = False

    def _require_process(self) -> None:
        if self._pid != os.getpid():
            raise DaemonWriterSettlementError("cannot settle an inherited writer thread")

    @property
    def retired(self) -> bool:
        self._require_process()
        with self._guard:
            return self._retired

    @property
    def settling(self) -> bool:
        self._require_process()
        with self._guard:
            return self._attempt is not None and not self._attempt.done()

    def request_settlement(self) -> ConcurrentFuture[None]:
        self._require_process()
        with self._guard:
            if self._attempt is not None and not self._attempt.done():
                return self._attempt
            attempt: ConcurrentFuture[None] = ConcurrentFuture()
            if self._retired:
                attempt.set_result(None)
            else:
                self._requests.put(attempt)
            attempt.add_done_callback(_report_terminal_settlement)
            self._attempt = attempt
            return attempt

    def serve(self) -> None:
        """Stay on the original Thread until its actual SQL owners settle."""
        self._require_process()
        while True:
            attempt = self._requests.get()
            error: BaseException | None = None
            try:
                self._cleanup()
            except BaseException as exc:
                error = exc
            pending = self._pending()
            if not pending:
                try:
                    self._retire()
                except BaseException as exc:
                    error = (
                        BaseExceptionGroup("SQL cleanup and grant retirement failed", [error, exc])
                        if error is not None
                        else exc
                    )
                pending = self._pending()
                if not pending:
                    with self._guard:
                        self._retired = True
            if error is not None:
                refusal = DaemonWriterSettlementError("original writer SQL cleanup failed; retry settlement")
                refusal.__cause__ = error
                attempt.set_exception(refusal)
            elif pending:
                attempt.set_exception(DaemonWriterSettlementError("original writer SQL remains unsettled"))
            else:
                attempt.set_result(None)
            self._settled()
            if not pending:
                return


class DaemonWriteCoordinator:
    """Fair async gate around every archive write actor in one daemon.

    Cancellation is ownership-aware. A queued request is removed immediately;
    an admitted request continues in its coordinator-owned task until the
    underlying coroutine or thread really finishes. Thus callers can observe
    bounded cancellation without allowing the next SQLite writer to overlap.
    """

    def __init__(
        self,
        *,
        archive_root: str | Path,
        observer: WriteEventObserver | None = None,
    ) -> None:
        self._owner_pid = os.getpid()
        self._terminal_guard = threading.Lock()
        self._terminal_workers: set[_TerminalWriter] = set()
        self._terminal_async_backends: dict[int, AsyncSQLCustodyOwner] = {}
        self._terminal_async_attempt: asyncio.Task[None] | None = None
        self._lock = _PriorityGate()
        self._observer = observer
        self._archive_root = Path(archive_root).resolve()
        self._sequence = 0
        self._active_actor: str | None = None
        self._queued: list[tuple[int, str]] = []
        self._last_event: DaemonWriteEvent | None = None
        self._over_budget_holds = 0
        self._accepting = True
        self._executions: set[asyncio.Task[object]] = set()
        self._terminal_changed = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._detached_writer_failures = 0
        self._detached_writer_failures_by_actor: dict[str, int] = {}
        self._publish_telemetry()

    def _require_process(self) -> None:
        if self._owner_pid != os.getpid():
            raise DaemonWriterSettlementError("cannot use a coordinator inherited across fork")

    def _retained_workers(self) -> tuple[_TerminalWriter, ...]:
        self._require_process()
        with self._terminal_guard:
            return tuple(self._terminal_workers)

    def _retain_terminal_worker(self, worker: _TerminalWriter, loop: asyncio.AbstractEventLoop) -> None:
        self._require_process()
        with self._terminal_guard:
            self._terminal_workers.add(worker)
        if not loop.is_closed():
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._terminal_changed.set)

    def _terminal_worker_completed(self, worker: _TerminalWriter) -> None:
        if worker.retired:
            with self._terminal_guard:
                self._terminal_workers.discard(worker)
        self._terminal_changed.set()
        if not self._executions and not self._has_unsettled_sql():
            self._idle.set()
        self._publish_telemetry()

    def _retained_async_backends(self) -> tuple[AsyncSQLCustodyOwner, ...]:
        self._require_process()
        with self._terminal_guard:
            return tuple(self._terminal_async_backends.values())

    def _has_unsettled_sql(self) -> bool:
        return bool(self._retained_workers() or self._retained_async_backends())

    async def _settle_async_backends(self) -> None:
        from polylogue.operations.sql_settlement import retained_async_sql_owners

        errors: list[BaseException] = []
        for backend in self._retained_async_backends():
            try:
                await backend.close()
            except BaseException as exc:
                errors.append(exc)
            if not any(owner is backend for owner in retained_async_sql_owners()):
                with self._terminal_guard:
                    self._terminal_async_backends.pop(id(backend), None)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("Original async writer cleanup failed", errors)

    def _async_settlement_completed(self, attempt: asyncio.Task[None]) -> None:
        self._terminal_changed.set()
        if not attempt.cancelled():
            error = attempt.exception()
            if error is not None:
                emit(
                    "daemon.writer.terminal_settlement_failed",
                    level=WARNING,
                    outcome="error",
                    reason="sql_cleanup_failed",
                    error_type=type(error).__name__,
                )
        if not self._executions and not self._has_unsettled_sql():
            self._idle.set()
        self._publish_telemetry()

    async def _settle_terminal_workers(self) -> None:
        errors: list[BaseException] = []
        for worker in self._retained_workers():
            attempt = worker.request_settlement()
            try:
                wrapped = asyncio.wrap_future(attempt)
                wrapped.add_done_callback(lambda done: None if done.cancelled() else done.exception())
                await asyncio.wait((wrapped,))
                wrapped.result()
            except BaseException as exc:
                # Accepted cleanup retains its creator even if this waiter
                # cancels. Request every other owned terminal child too.
                errors.append(exc)
            if worker.retired:
                with self._terminal_guard:
                    self._terminal_workers.discard(worker)
        if self._retained_async_backends():
            for backend in self._retained_async_backends():
                backend.request_sql_settlement()
            async_attempt = self._terminal_async_attempt
            if async_attempt is None or async_attempt.done():
                async_attempt = asyncio.create_task(self._settle_async_backends())
                self._terminal_async_attempt = async_attempt
                async_attempt.add_done_callback(self._async_settlement_completed)
            try:
                await asyncio.wait((async_attempt,))
                async_attempt.result()
            except BaseException as exc:
                errors.append(exc)
        if not self._executions and not self._has_unsettled_sql():
            self._idle.set()
        self._publish_telemetry()
        if errors:
            failure = errors[0] if len(errors) == 1 else BaseExceptionGroup("Original writer cleanup failed", errors)
            cancellation = next((error for error in errors if isinstance(error, asyncio.CancelledError)), None)
            if cancellation is not None:
                other_failures = [error for error in errors if error is not cancellation]
                if not other_failures:
                    raise cancellation
                cause = (
                    other_failures[0]
                    if len(other_failures) == 1
                    else BaseExceptionGroup("Original writer cleanup failed", other_failures)
                )
                raise cancellation from cause
            if self._has_unsettled_sql():
                raise DaemonWriterSettlementError("original SQL cleanup remains unsettled") from failure
            raise failure

    def _sql_settlement_state(self) -> str:
        if any(worker.settling for worker in self._retained_workers()) or (
            self._terminal_async_attempt is not None and not self._terminal_async_attempt.done()
        ):
            return "settling"
        return "required" if self._has_unsettled_sql() else "idle"

    def snapshot(self) -> DaemonWriteSnapshot:
        return DaemonWriteSnapshot(
            active_actor=self._active_actor,
            queued_actors=tuple(actor for _sequence, actor in self._queued),
            last_event=self._last_event,
            accepting=self._accepting,
            detached_writer_failures=self._detached_writer_failures,
            detached_writer_failures_by_actor=tuple(sorted(self._detached_writer_failures_by_actor.items())),
            over_budget_holds=self._over_budget_holds,
            unsettled_writer_workers=len(self._retained_workers()),
            unsettled_async_backends=len(self._retained_async_backends()),
            sql_settlement_state=self._sql_settlement_state(),
        )

    async def run(
        self,
        actor: str,
        operation: Callable[[], Awaitable[T]],
        *,
        on_complete: Callable[[asyncio.Task[object]], None] | None = None,
        on_admit: Callable[[], None] | None = None,
    ) -> T:
        """Run one async write operation under the process-wide gate."""
        actor_label = actor.strip()
        if not actor_label:
            raise ValueError("daemon write actor must be non-empty")
        if actor_label.startswith(_DETACHED_WRITER_FAILURE_RESERVED_ACTOR_PREFIX):
            raise ValueError("daemon write actor uses a reserved telemetry label")

        self._require_process()
        current_task = asyncio.current_task()
        if current_task is None:
            raise RuntimeError("daemon write coordination requires an asyncio task")
        active_lease = active_write_admission.get()
        if active_lease is not None and active_lease.coordinator is self:
            if active_lease.admits(self, self._archive_root):
                return await operation()
            raise RuntimeError(
                "daemon write lease was inherited by a child task; nested writes must run in the owning task"
            )
        if not self._accepting:
            raise RuntimeError("daemon write coordinator is shutting down")

        self._sequence += 1
        request = _WriteRequest(actor=actor, sequence=self._sequence, queued_at=time.perf_counter())
        self._queued.append((request.sequence, actor))
        self._emit(
            DaemonWriteEvent(
                phase="queued",
                actor=actor,
                sequence=request.sequence,
                queue_depth=len(self._queued),
            )
        )

        execution = asyncio.create_task(
            self._execute(request, operation, on_admit),
            name=f"polylogue-writer:{actor}:{request.sequence}",
        )
        self._track_execution(execution, actor=actor, on_complete=on_complete, request=request)
        try:
            await asyncio.wait((execution,))
            return execution.result()
        except asyncio.CancelledError:
            request.caller_cancelled = True
            if not request.acquired:
                execution.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.wait((execution,))
                    execution.result()
            raise

    async def _execute(
        self,
        request: _WriteRequest,
        operation: Callable[[], Awaitable[T]],
        on_admit: Callable[[], None] | None = None,
    ) -> T:
        self._require_process()
        try:
            await self._lock.acquire(_actor_priority(request.actor))
        except BaseException:
            self._remove_queued(request.sequence)
            raise

        request.acquired = True
        self._remove_queued(request.sequence)
        acquired_at = time.perf_counter()
        wait_seconds = acquired_at - request.queued_at
        self._active_actor = request.actor
        self._emit(
            DaemonWriteEvent(
                phase="acquired",
                actor=request.actor,
                sequence=request.sequence,
                queue_depth=len(self._queued),
                wait_seconds=wait_seconds,
            )
        )
        owner = asyncio.current_task()
        if owner is None:  # pragma: no cover - asyncio always owns created tasks
            self._lock.release()
            raise RuntimeError("coordinator execution has no owning task")
        token = active_write_admission.set(WriteAdmission(self, owner, self._archive_root))
        budget_s = write_hold_budget_s(request.actor)
        hold_token = enter_write_hold(request.actor, budget_s)
        outcome: WriteOutcome = "success"
        admission_failed = False
        settlement_complete = False
        try:
            await self._settle_terminal_workers()
            settlement_complete = True
            if on_admit is not None:
                try:
                    on_admit()
                except BaseException:
                    admission_failed = True
                    raise
            # Establish the storage-side authorization in the coordinator-owned
            # task.  ``_run_writer_worker`` copies this context into the
            # actual writer thread, so every writable open remains behind the
            # same gate even when the callable is synchronous.
            async with async_write_lease(request.actor, archive_root=self._archive_root, coordinator=self) as lease:
                from polylogue.operations.sql_settlement import retained_async_sql_owners

                try:
                    value = await operation()
                finally:
                    retained = retained_async_sql_owners(lease=lease)
                    if retained:
                        with self._terminal_guard:
                            self._terminal_async_backends.update((id(backend), backend) for backend in retained)
                if retained:
                    raise DaemonWriterSettlementError("async writer returned with unsettled SQL")
                return value
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except BaseException:
            outcome = "error"
            raise
        finally:
            exit_write_hold(hold_token)
            active_write_admission.reset(token)
            hold_seconds = time.perf_counter() - acquired_at
            over_budget = hold_seconds > budget_s
            if over_budget:
                self._over_budget_holds += 1
            self._active_actor = None
            self._lock.release()
            self._emit(
                DaemonWriteEvent(
                    phase="released",
                    actor=request.actor,
                    sequence=request.sequence,
                    queue_depth=len(self._queued),
                    wait_seconds=wait_seconds,
                    hold_seconds=hold_seconds,
                    outcome=outcome,
                    hold_budget_s=budget_s,
                    hold_over_budget=over_budget,
                )
            )
            # One event per release, its level decided by the budget: a hold
            # that can starve an off-gate writer must not read like every
            # other release in a long log.
            emit(
                "daemon.writer.released",
                level=WARNING if over_budget else INFO,
                outcome="degraded" if over_budget else ("ok" if outcome == "success" else outcome),
                reason=(
                    "hold_over_budget"
                    if over_budget
                    else "terminal_settlement_failed"
                    if not settlement_complete
                    else "admission_hook_failed"
                    if admission_failed
                    else "within_budget"
                ),
                actor=request.actor,
                status=outcome,
                wait_ms=round(wait_seconds * 1000, 3),
                hold_ms=round(hold_seconds * 1000, 3),
                budget_ms=round(budget_s * 1000, 3),
                queued=len(self._queued),
            )

    async def run_sync(self, actor: str, function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
        """Run blocking writer work without making process exit unbounded."""
        return await self._run_sync(actor, function, None, None, *args, **kwargs)

    async def run_prepared_sync(
        self,
        actor: str,
        operation: Callable[[], T],
        *,
        submit_worker: Callable[[Callable[[], None]], ConcurrentFuture[None]],
        settlement_owners: Callable[[], tuple[SQLCustodyOwner, ...]],
    ) -> T:
        """Own an off-gate preparation worker through its final SQL settlement.

        The callable acquires ordinary bridge admission after preparing on its
        worker. Its result is delivered independently from the managed compute
        slot, which remains occupied while that same thread settles failed SQL.
        """
        self._require_process()
        if current_write_lease() is not None:
            raise RuntimeError("prepared writer work must begin outside admission")
        if not self._accepting:
            raise RuntimeError("daemon writer is shutting down")
        task = asyncio.create_task(
            _run_writer_worker(self, _WorkerDispatch(submit_worker, settlement_owners), operation, actor),
            name=f"polylogue-prepared-writer:{actor}",
        )
        self._track_execution(task, actor=actor)
        await asyncio.wait((task,))
        return task.result()

    async def run_sync_with_completion(
        self,
        actor: str,
        function: Callable[P, T],
        on_complete: Callable[[asyncio.Task[object]], None],
        on_admit: Callable[[], None] | None = None,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Run sync work and observe its coordinator-owned completion task."""
        return await self._run_sync(actor, function, on_complete, on_admit, *args, **kwargs)

    async def _run_sync(
        self,
        actor: str,
        function: Callable[P, T],
        on_complete: Callable[[asyncio.Task[object]], None] | None,
        on_admit: Callable[[], None] | None,
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        async def operation() -> T:
            return await _run_writer_worker(
                self,
                None,
                function,
                f"polylogue-writer:{actor}",
                *args,
                **kwargs,
            )

        return await self.run(actor, operation, on_complete=on_complete, on_admit=on_admit)

    async def shutdown(self, *, timeout: float) -> bool:
        """Stop admission and wait at most ``timeout`` seconds for real idle.

        ``False`` means an admitted writer still owns the gate. The coordinator
        deliberately leaves it held; releasing an uncooperative sync writer is
        not a safe shutdown operation.
        """
        self._require_process()
        if timeout < 0:
            raise ValueError("shutdown timeout must be non-negative")
        self._accepting = False
        self._publish_telemetry()
        try:
            async with asyncio.timeout(timeout):
                while True:
                    self._terminal_changed.clear()
                    if self._has_unsettled_sql():
                        try:
                            await self._settle_terminal_workers()
                        except DaemonWriterSettlementError:
                            return False
                    if not self._executions and not self._has_unsettled_sql():
                        return True
                    await self._terminal_changed.wait()
        except TimeoutError:
            return False
        return True

    def _track_execution(
        self,
        execution: asyncio.Task[T],
        *,
        actor: str,
        on_complete: Callable[[asyncio.Task[object]], None] | None = None,
        request: _WriteRequest | None = None,
    ) -> None:
        task = execution  # preserve the concrete result type for ``run``
        self._executions.add(task)
        self._idle.clear()

        def record_failure() -> None:
            self._detached_writer_failures += 1
            actor_label = actor.strip()
            if (
                not actor_label
                or len(actor_label) > _MAX_DETACHED_WRITER_FAILURE_ACTOR_LENGTH
                or any(ord(character) < 0x20 for character in actor_label)
            ):
                actor_label = _DETACHED_WRITER_FAILURE_OVERFLOW_ACTOR
            if (
                actor_label not in self._detached_writer_failures_by_actor
                and len(self._detached_writer_failures_by_actor) >= _MAX_DETACHED_WRITER_FAILURE_ACTORS - 1
            ):
                actor_label = _DETACHED_WRITER_FAILURE_OVERFLOW_ACTOR
            self._detached_writer_failures_by_actor[actor_label] = (
                self._detached_writer_failures_by_actor.get(actor_label, 0) + 1
            )
            self._publish_telemetry()

        def completed(done: asyncio.Task[object]) -> None:
            self._terminal_changed.set()
            self._executions.discard(done)
            if done.cancelled():
                if on_complete is not None and (request is None or request.acquired):
                    try:
                        on_complete(done)
                    except BaseException:
                        emit(
                            "daemon.writer.completion_callback_failed",
                            level=ERROR,
                            outcome="error",
                            reason="completion_callback_raised",
                            actor=actor,
                        )
                if not self._executions and not self._has_unsettled_sql():
                    self._idle.set()
                return
            try:
                exception = done.exception()
            except Exception:
                # polylogue-es7b: a detached writer's own exception-retrieval
                # failed (e.g. the task itself never actually completed
                # cleanly) -- still count it as a lost forensic detail so the
                # daemon-lifetime counter reflects every such event, not only
                # the ordinary "task.exception() returned non-None" path.
                record_failure()
                emit(
                    "daemon.writer.detached_failed",
                    level=WARNING,
                    outcome="error",
                    reason="exception_retrieval_failed",
                    actor=actor,
                )
            else:
                if exception is not None:
                    record_failure()
                    emit(
                        "daemon.writer.detached_failed",
                        level=WARNING,
                        outcome="error",
                        reason="writer_raised",
                        actor=actor,
                        error_type=type(exception).__name__,
                        error_detail=str(exception),
                    )
            if on_complete is not None and (request is None or request.acquired):
                try:
                    on_complete(done)
                except BaseException:
                    # Completion publication must never interfere with writer
                    # lifecycle accounting. Callers that need durable retry
                    # should enqueue a coordinator-managed task.
                    emit(
                        "daemon.writer.completion_callback_failed",
                        level=ERROR,
                        outcome="error",
                        reason="completion_callback_raised",
                        actor=actor,
                    )
            if not self._executions and not self._has_unsettled_sql():
                self._idle.set()

        task.add_done_callback(completed)

    def _remove_queued(self, sequence: int) -> None:
        self._queued = [item for item in self._queued if item[0] != sequence]
        self._publish_telemetry()

    def _emit(self, event: DaemonWriteEvent) -> None:
        self._last_event = event
        self._publish_telemetry()
        if self._observer is None:
            return
        try:
            self._observer(event)
        except Exception as exc:
            emit(
                "daemon.writer.telemetry_observer_failed",
                level=WARNING,
                outcome="degraded",
                reason="observer_raised",
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )

    def _publish_telemetry(self) -> None:
        snapshot = self.snapshot()
        event = snapshot.last_event
        payload: dict[str, object] = {
            "active_actor": snapshot.active_actor,
            "queued_actors": list(snapshot.queued_actors),
            "queue_depth": len(snapshot.queued_actors),
            "accepting": snapshot.accepting,
            "last_event": None,
            "detached_writer_failures": snapshot.detached_writer_failures,
            "detached_writer_failures_by_actor": dict(snapshot.detached_writer_failures_by_actor),
            # Keep the daemon-lifetime hold counter in the process telemetry
            # envelope as well as the typed snapshot.  Checkpoint wait/hold
            # evidence is consumed through this route by status surfaces; if
            # the counter only lived on ``snapshot()`` it disappeared at the
            # production boundary.
            "over_budget_holds": snapshot.over_budget_holds,
            "unsettled_writer_workers": snapshot.unsettled_writer_workers,
            "unsettled_async_backends": snapshot.unsettled_async_backends,
            "sql_settlement_state": snapshot.sql_settlement_state,
        }
        if event is not None:
            payload["last_event"] = {
                "phase": event.phase,
                "actor": event.actor,
                "sequence": event.sequence,
                "queue_depth": event.queue_depth,
                "wait_seconds": event.wait_seconds,
                "hold_seconds": event.hold_seconds,
                "outcome": event.outcome,
                "hold_budget_s": event.hold_budget_s,
                "hold_over_budget": event.hold_over_budget,
            }
        with _TELEMETRY_LOCK:
            _LATEST_TELEMETRY.clear()
            _LATEST_TELEMETRY.update(payload)


@dataclass(frozen=True)
class _WorkerDispatch:
    submit: Callable[[Callable[[], None]], ConcurrentFuture[None]]
    settlement_owners: Callable[[], tuple[SQLCustodyOwner, ...]]


async def _run_writer_worker(
    coordinator: DaemonWriteCoordinator,
    dispatch: _WorkerDispatch | None,
    function: Callable[P, T],
    thread_name: str,
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Await one context-preserving worker that cannot pin interpreter exit."""
    loop = asyncio.get_running_loop()
    result: ConcurrentFuture[T] = ConcurrentFuture()
    context = contextvars.copy_context()
    # Mint the thread grant here, on the owning side, before the worker
    # exists. On a free-threading build every spawned thread inherits the
    # lease, so a worker that bound *itself* would be self-authorizing rather
    # than deliberately admitted (polylogue-1oa7o); the grant is single-use
    # and checked against this exact lease object.
    thread_grant = grant_write_lease_thread() if current_write_lease() is not None else None

    def worker() -> None:
        from polylogue.operations.sql_settlement import retained_async_sql_owners

        custody = thread_grant.lease.custody if thread_grant is not None else None

        def sql_owners() -> tuple[SQLCustodyOwner, ...]:
            from polylogue.operations.sql_settlement import retained_sync_sql_owners

            registered = retained_sync_sql_owners()
            prepared = dispatch.settlement_owners() if dispatch is not None else ()
            return tuple({id(owner): owner for owner in (*registered, *prepared)}.values())

        def reconcile_cached_handles() -> None:
            from polylogue.operations.sql_settlement import settle_cached_sql

            if custody is not None:
                settle_cached_sql(custody)

        retirement_requested = False

        def pending() -> bool:
            return bool(
                sql_owners()
                or retained_async_sql_owners()
                or (retirement_requested and thread_grant is not None and not thread_grant.custody_retired)
            )

        def cleanup() -> None:
            failures: list[BaseException] = []
            for owner in sql_owners():
                try:
                    owner.close()
                except BaseException as exc:
                    failures.append(exc)

            async def close_async_owners() -> None:
                for backend in retained_async_sql_owners():
                    try:
                        await backend.close()
                    except BaseException as exc:
                        failures.append(exc)

            if retained_async_sql_owners():
                asyncio.run(close_async_owners())
            if len(failures) == 1:
                raise failures[0]
            if failures:
                raise BaseExceptionGroup("Writer child cleanup failed", failures)

        def retire() -> None:
            nonlocal retirement_requested
            retirement_requested = True
            if thread_grant is not None and not thread_grant.custody_retired:
                thread_grant.complete()

        error: BaseException | None = None
        value: T
        try:

            def invoke() -> T:
                if thread_grant is not None:
                    bind_write_lease_thread(thread_grant)
                return function(*args, **kwargs)

            value = context.run(invoke)
        except BaseException as exc:
            error = exc

        try:
            context.run(reconcile_cached_handles)
        except BaseException as exc:
            error = BaseExceptionGroup("Writer and cache cleanup failed", [error, exc]) if error is not None else exc

        if not pending():
            try:
                context.run(retire)
            except BaseException as exc:
                error = (
                    BaseExceptionGroup("Writer and grant retirement failed", [error, exc]) if error is not None else exc
                )

        terminal: _TerminalWriter | None = None
        release_creator_settlement: Callable[[], None] | None = None
        if pending():

            def settled() -> None:
                assert terminal is not None
                if release_creator_settlement is not None:
                    release_creator_settlement()
                if not loop.is_closed():
                    with contextlib.suppress(RuntimeError):
                        loop.call_soon_threadsafe(coordinator._terminal_worker_completed, terminal)

            terminal = _TerminalWriter(lambda: context.run(cleanup), pending, lambda: context.run(retire), settled)
            coordinator._retain_terminal_worker(terminal, loop)
            # On a compute creator thread the adapter owns retry and shutdown
            # for that thread; let them reach this parked terminal as well.
            release_creator_settlement = retain_current_creator_settlement(
                terminal.request_settlement,
                owner_count=len(sql_owners()),
                failure_types=() if error is None else (type(error).__name__,),
            )
            refusal = DaemonWriterSettlementError("writer returned with unsettled SQL; retry terminal settlement")
            if error is not None:
                refusal.__cause__ = error
            error = refusal
        if error is not None:
            with contextlib.suppress(InvalidStateError):
                result.set_exception(error)
        else:
            with contextlib.suppress(InvalidStateError):
                result.set_result(value)

        if terminal is not None:
            terminal.serve()

        if loop.is_closed():
            emit(
                "daemon.writer.result_abandoned",
                level=WARNING,
                outcome="error",
                reason="event_loop_closed",
                thread=thread_name,
                error_type=type(error).__name__ if error is not None else None,
                error_detail=str(error) if error is not None else None,
            )

    try:
        if dispatch is None:
            threading.Thread(target=worker, name=thread_name, daemon=True).start()
        else:
            submission = dispatch.submit(worker)

            def submission_finished(done: ConcurrentFuture[None]) -> None:
                if result.done():
                    return
                try:
                    done.result()
                except BaseException as exc:
                    with contextlib.suppress(InvalidStateError):
                        result.set_exception(exc)

            submission.add_done_callback(submission_finished)
    except BaseException as primary:
        if thread_grant is not None:
            try:
                thread_grant.complete()
            except BaseException as cleanup:
                raise BaseExceptionGroup("Writer dispatch and grant cleanup failed", [primary, cleanup]) from primary
        raise
    return await asyncio.wrap_future(result, loop=loop)


class StagedTask(Generic[T]):
    """A coroutine on the owner loop whose future settles only with its task.

    ``run_coroutine_threadsafe`` marks its proxy future cancelled at once,
    while the task behind it may still be awaiting a running compute phase and
    its ``finally`` cleanup. Shutdown and the exchange's settled callback wait
    on this future, so cancellation is forwarded to the task and the future
    takes the task's terminal state only when the task has actually finished.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        start: Callable[[], Coroutine[Any, Any, T]],
        *,
        name: str | None = None,
    ) -> None:
        self.future: ConcurrentFuture[T] = ConcurrentFuture()
        self._loop = loop
        self._task: asyncio.Task[T] | None = None
        self._cancelled = False
        self._name = name
        loop.call_soon_threadsafe(self._start, start)

    def cancel(self) -> None:
        """Request cancellation from any thread; the future settles with the task."""
        self._loop.call_soon_threadsafe(self._cancel)

    def _start(self, start: Callable[[], Coroutine[Any, Any, T]]) -> None:
        # ``_start`` and ``_cancel`` both run on the owner loop in submission
        # order, so a cancellation either precedes the task or reaches it.
        if self._cancelled:
            return
        self._task = self._loop.create_task(start(), name=self._name)
        self._task.add_done_callback(self._settle)

    def _cancel(self) -> None:
        if self._task is not None:
            self._task.cancel()
        elif not self._cancelled:
            self._cancelled = True
            self.future.cancel()

    def _settle(self, task: asyncio.Task[T]) -> None:
        if self.future.done():
            return
        if task.cancelled():
            self.future.cancel()
        elif (exc := task.exception()) is not None:
            self.future.set_exception(exc)
        else:
            self.future.set_result(task.result())


class DaemonWriteThreadBridge:
    """Let synchronous daemon request threads hold the main-loop write gate."""

    def __init__(
        self,
        coordinator: DaemonWriteCoordinator,
        loop: asyncio.AbstractEventLoop,
        *,
        timeout: float = 30.0,
    ) -> None:
        self._coordinator = coordinator
        self._loop = loop
        self._timeout = timeout

    @property
    def coordinator(self) -> DaemonWriteCoordinator:
        """Borrow the exact coordinator whose delegations this bridge grants."""
        self._coordinator._require_process()
        return self._coordinator

    def run_admitted_async(
        self,
        delegation: WriteLeaseDelegation,
        operation: Callable[[], Awaitable[T]],
        *,
        timeout: float | None,
    ) -> T:
        """Run an already-admitted body on the existing writer worker route."""
        self._coordinator._require_process()
        if delegation.lease.coordinator is not self._coordinator:
            raise RuntimeError("the admitted body belongs to another coordinator")
        if self._loop.is_closed() or not self._loop.is_running():
            raise DaemonWriterOwnerLoopStopped("the admitted body's writer owner loop stopped")

        async def admitted() -> T:
            task = asyncio.current_task()
            assert task is not None
            self._coordinator._track_execution(task, actor=delegation.actor)
            token = active_write_admission.set(WriteAdmission(self._coordinator, task, self._coordinator._archive_root))
            try:
                with adopt_write_lease(delegation):
                    child_delegation = delegate_write_lease()

                    def work() -> T:
                        async def child() -> T:
                            with adopt_write_lease(child_delegation):
                                return await operation()

                        return asyncio.run(child())

                    return await _run_writer_worker(
                        self._coordinator, None, work, f"polylogue-writer:{delegation.actor}"
                    )
            finally:
                active_write_admission.reset(token)

        coroutine = admitted()
        try:
            future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        except RuntimeError as error:
            coroutine.close()
            raise DaemonWriterOwnerLoopStopped("the admitted body's writer owner loop stopped") from error
        if timeout is not None:
            return future.result(timeout=timeout)
        return self._await_owner_settlement(delegation.actor, future)

    @contextmanager
    def hold(self, actor: str) -> Iterator[WriteLeaseDelegation]:
        """Hold the loop-owned write gate and yield the caller's authorization.

        The lease itself is entered in a coroutine on the owner loop, so the
        calling thread's ambient context never sees it and no widening of the
        ambient rules could make it: the body runs on a third thread inside a
        freshly created event loop. The yielded delegation is the explicit
        authorization that unit of work presents through
        :func:`~polylogue.core.write_lease.adopt_write_lease`; holding the gate
        without presenting it still cannot open a write connection.

        The gate is released when the *delegated execution* settles, not when
        the calling thread stops waiting. A mutating route whose bounded wait
        expires (``DaemonMutationIndeterminate``) unwinds this context manager
        while its body is still inside the archive; releasing here would admit
        a second writer alongside it (polylogue-8r4zq AC2).
        """
        entered = threading.Event()
        settled = threading.Event()
        release = asyncio.Event()
        granted: list[WriteLeaseDelegation] = []

        async def hold_lease() -> None:
            async def wait_for_release() -> None:
                delegation = delegate_write_lease()
                granted.append(delegation)
                entered.set()
                settled.set()
                await release.wait()
                await self._retain_until_delegation_settles(actor, delegation)

            try:
                await self._coordinator.run(actor, wait_for_release)
            finally:
                settled.set()

        if self._loop.is_closed() or not self._loop.is_running():
            raise DaemonWriterOwnerLoopStopped(
                f"daemon writer owner loop closed before {actor} was admitted; the write did not start"
            )
        # A named task, so the daemon's task inventory attributes the held gate.
        future = StagedTask(self._loop, hold_lease, name=f"polylogue-writer-hold:{actor}").future
        future.add_done_callback(lambda _future: settled.set())
        while not settled.wait(_DELEGATION_SETTLEMENT_POLL_S):
            # This observes an actual stopped execution owner, not elapsed
            # work time. A running loop may queue valid writes indefinitely.
            if not self._loop.is_running():
                future.cancel()
                raise DaemonWriterOwnerLoopStopped(f"daemon writer owner loop stopped before {actor} was admitted")
        if not entered.is_set():
            future.result()
            raise RuntimeError(f"daemon write gate ended before acquisition actor={actor}")
        try:
            yield granted[0]
        finally:
            # Retire *before* signalling the release, so the answer cannot be
            # invalidated by a body adopting in the gap: ``False`` means no
            # execution can ever adopt this delegation again.
            handed_off = granted[0].retire()
            self._loop.call_soon_threadsafe(release.set)
            if handed_off:
                # The body outlived this caller -- an expired mutating-route
                # deadline is the live case. The owner loop keeps the gate
                # until that body settles; waiting for it *here* would make
                # the route's bounded client deadline "deadline plus however
                # long the write runs", which is the bound this bead exists
                # to give the client. No ``return`` here: returning out of a
                # ``finally`` in a @contextmanager generator suppresses the
                # very DaemonMutationIndeterminate the route is raising.
                future.add_done_callback(functools.partial(_report_handed_off_hold, actor))
                emit(
                    "daemon.writer.hold_handed_off",
                    level=WARNING,
                    outcome="unmeasured",
                    reason="delegated_body_outlived_caller",
                    actor=actor,
                )
            else:
                future.result()

    async def _retain_until_delegation_settles(self, actor: str, delegation: WriteLeaseDelegation) -> None:
        """Keep the admitted hold until the delegated execution really leaves.

        ``retire`` is atomic against adoption: once it answers ``False`` the
        delegation can never be adopted again, so the caller thread unwinding
        is a real end of ownership. ``True`` means a body is inside the
        archive right now, and the only safe answer is to keep holding --
        cancelling it is precisely what this bead forbids, and releasing the
        gate would hand SQLite to a second writer.
        """
        if not delegation.retire():
            return
        waited = 0.0
        warned = False
        while not delegation.settled:
            # A caller-side release timeout cancels this hold task. That is a
            # statement about the *client's* patience, not about the writer,
            # so it must not become an early release.
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.sleep(_DELEGATION_SETTLEMENT_POLL_S)
            waited += _DELEGATION_SETTLEMENT_POLL_S
            if not warned and waited >= _DELEGATION_SETTLEMENT_WARN_S:
                warned = True
                emit(
                    "daemon.writer.delegation_unsettled",
                    level=WARNING,
                    outcome="degraded",
                    reason="delegated_body_still_running",
                    actor=actor,
                    wait_ms=round(waited * 1000, 3),
                )

    @property
    def owner_loop(self) -> asyncio.AbstractEventLoop:
        """The explicitly composed loop that owns this bridge's writer."""
        return self._loop

    async def run_async(self, actor: str, function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
        """Borrow the already-owned loop for one staged operation publication."""
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("staged publication must run on the bridge's owner loop")
        pending = asyncio.create_task(
            self._coordinator.run_sync(actor, function, *args, **kwargs),
            name=f"polylogue-writer-staged:{actor}",
        )
        try:
            await asyncio.wait((pending,))
            return pending.result()
        except asyncio.CancelledError:
            # A lifecycle cancellation is not permission to release the
            # writer or abandon the receipt of an admitted callable.
            while not pending.done():
                try:
                    await asyncio.wait((pending,))
                    pending.result()
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not pending.cancelled():
                pending.exception()
            raise

    def run_sync(self, actor: str, function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
        """Run a blocking request operation through the daemon's sole writer.

        Unlike :meth:`hold`, this is for a complete bounded request operation:
        the coordinator owns the worker thread until the function has really
        returned, so a timed-out HTTP caller never admits a second writer.

        Waits at most this bridge's constructor ``timeout`` (default 30s) for
        completion. Use :meth:`run_sync_with_timeout` for an operation whose
        own contract needs a longer bound (see polylogue-ogn1).
        """
        return self.run_sync_with_timeout(actor, self._timeout, function, *args, **kwargs)

    def run_sync_with_timeout(
        self,
        actor: str,
        timeout: float | None,
        function: Callable[P, T],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Like :meth:`run_sync`, waiting up to ``timeout`` seconds instead of the bridge default.

        polylogue-ogn1: the bridge's constructor ``timeout`` (30s) is sized for
        ordinary request-scoped writes (reset, ingest, maintenance run). A
        bounded index rebuild pass can legitimately run far longer -- the
        CLI/HTTP contract already allows up to 600s (``_run_daemon_rebuild``'s
        ``urlopen(..., timeout=600)``) -- so that call site needs its own,
        longer wait here rather than being silently killed by the bridge's
        default gate at 30s while the rebuild is still replaying.
        A None timeout preserves daemon ownership until the operation returns
        its receipt. This is necessary for no-promote canaries: abandoning a
        completed inactive candidate after a caller-side timeout would lose
        the only authority capable of discarding it safely. It is still bound
        -- on the owner loop's liveness rather than on a clock: if that loop
        stops without closing, the admitted operation can never deliver its
        receipt here and an unbounded wait strands the caller forever
        (polylogue-8r4zq AC5).
        """
        if self._loop.is_closed() or not self._loop.is_running():
            raise DaemonWriterOwnerLoopStopped(
                f"daemon writer owner loop closed before {actor} was admitted; the write did not start"
            )
        operation = self._coordinator.run_sync(actor, function, *args, **kwargs)
        try:
            future = asyncio.run_coroutine_threadsafe(operation, self._loop)
        except RuntimeError as exc:
            # The loop can stop in the small gap after the liveness check. Do
            # not expose the implementation detail or leave ``operation``
            # unawaited; callers need the same typed outcome as the polling
            # path below.
            operation.close()
            raise DaemonWriterOwnerLoopStopped(
                f"daemon writer owner loop stopped before {actor} was admitted; the write did not start"
            ) from exc
        if timeout is not None:
            return future.result(timeout=timeout)
        return self._await_owner_settlement(actor, future)

    def _await_owner_settlement(self, actor: str, future: ConcurrentFuture[T]) -> T:
        """Wait for an admitted operation for as long as its owner loop lives.

        The writer is never cancelled here. It may already be inside a SQLite
        transaction on its own thread, so withdrawing it on a wait failure
        would be the same lie in the other direction: this raises a typed
        *indeterminate* outcome and leaves the operation alone.
        """
        while True:
            try:
                return future.result(timeout=_OWNER_LIVENESS_POLL_S)
            except FutureTimeoutError:
                pass
            if self._loop.is_closed() or not self._loop.is_running():
                if future.done():
                    # It settled in the same breath the loop stopped.
                    return future.result()
                emit(
                    "daemon.writer.owner_loop_stopped",
                    level=ERROR,
                    outcome="unmeasured",
                    reason="owner_loop_stopped",
                    actor=actor,
                )
                raise DaemonWriterOwnerLoopStopped(
                    f"daemon writer owner loop stopped before {actor} reported settlement; "
                    "the write may still be in flight -- re-read before retrying"
                )


def daemon_write_telemetry_payload() -> dict[str, object]:
    """Return the bounded process-global writer state for status surfaces."""
    with _TELEMETRY_LOCK:
        payload = dict(_LATEST_TELEMETRY)
        queued = payload.get("queued_actors")
        if isinstance(queued, list):
            payload["queued_actors"] = list(queued)
        event = payload.get("last_event")
        if isinstance(event, dict):
            payload["last_event"] = dict(event)
        actor_failures = payload.get("detached_writer_failures_by_actor")
        if isinstance(actor_failures, dict):
            payload["detached_writer_failures_by_actor"] = dict(actor_failures)
        return payload


_COORDINATORS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, DaemonWriteCoordinator] = (
    weakref.WeakKeyDictionary()
)


def daemon_write_coordinator() -> DaemonWriteCoordinator:
    """Return the sole coordinator for the current process event loop."""
    loop = asyncio.get_running_loop()
    coordinator = _COORDINATORS.get(loop)
    if coordinator is None:
        from polylogue.paths import archive_root

        coordinator = DaemonWriteCoordinator(archive_root=archive_root())
        _COORDINATORS[loop] = coordinator
    return coordinator


def register_write_coordinator(loop: asyncio.AbstractEventLoop, coordinator: DaemonWriteCoordinator) -> None:
    """Bind ``coordinator`` as *the* coordinator for ``loop``.

    A caller that constructs its own coordinator -- the standalone HTTP write
    runtime is the only one -- must register it, or a later
    :func:`daemon_write_coordinator` call on that loop mints a second
    coordinator and the process has two serialization queues instead of one.
    """
    existing = _COORDINATORS.get(loop)
    if existing is not None and existing is not coordinator:
        raise RuntimeError("a different write coordinator is already registered for this event loop")
    _COORDINATORS[loop] = coordinator


__all__ = [
    "DaemonWriteCoordinator",
    "DaemonWriteEvent",
    "DaemonWriteSnapshot",
    "DaemonWriteThreadBridge",
    "DaemonWriterOwnerLoopStopped",
    "daemon_write_coordinator",
    "daemon_write_telemetry_payload",
    "register_write_coordinator",
]
