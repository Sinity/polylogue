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
import threading
import time
import weakref
from collections.abc import Awaitable, Callable, Iterator, Mapping
from concurrent.futures import CancelledError, InvalidStateError
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, ParamSpec, TypeVar

from polylogue.core.write_hold import enter_write_hold, exit_write_hold
from polylogue.core.write_lease import (
    WriteLeaseDelegation,
    bind_write_lease_thread,
    current_write_lease,
    delegate_write_lease,
    grant_write_lease_thread,
    write_lease,
)
from polylogue.logging import ERROR, INFO, WARNING, emit

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
#: The coordinator cannot abort an operation that is already inside a SQLite
#: transaction, so a budget does not preempt. It is published to the admitted
#: unit of work through :mod:`polylogue.core.write_hold`, and every checkpoint
#: that unit offers -- between files, between records -- ends the unit with a
#: typed ``WriteHoldBudgetError`` once the bound is spent, so overshoot is
#: one work item.
#:
#: The numbers come from measurement, not preference. A non-gated writer times
#: out after the storage layer's busy timeout -- 30 s, DB_TIMEOUT in
#: storage/sqlite/connection_profile.py, not imported here because the daemon
#: ring may not reach into storage -- so any hold longer than that can starve
#: one. A live catch-up chunk held 1.0-3.1 s in rehearsal-11,
#: and maintenance.drive_catchup was measured at hold_max 18,623 s
#: (daemon/cli.py), which is the hold this budget exists to surface
#: (polylogue-8qm4k).
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


@dataclass(slots=True)
class _WriteRequest:
    actor: str
    sequence: int
    queued_at: float
    acquired: bool = False
    caller_cancelled: bool = False


WriteEventObserver = Callable[[DaemonWriteEvent], None]
_ACTIVE_LEASE: contextvars.ContextVar[tuple[DaemonWriteCoordinator, asyncio.Task[object]] | None] = (
    contextvars.ContextVar("polylogue_active_daemon_write_lease", default=None)
)
_TELEMETRY_LOCK = threading.Lock()
_LATEST_TELEMETRY: dict[str, object] = {
    "active_actor": None,
    "queued_actors": [],
    "queue_depth": 0,
    "accepting": True,
    "last_event": None,
    "detached_writer_failures": 0,
    "detached_writer_failures_by_actor": {},
}


def daemon_write_lease_active() -> bool:
    """Return whether the current context owns the daemon's writer gate.

    ``run_sync`` deliberately propagates context into its worker thread, so
    storage code can distinguish a coordinator-authorized online write from an
    unrelated maintenance process racing the daemon.  This is an authority
    check, not merely a daemon-process check.
    """
    return _ACTIVE_LEASE.get() is not None


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
        observer: WriteEventObserver | None = None,
        archive_root: str | Path | None = None,
    ) -> None:
        self._lock = _PriorityGate()
        self._observer = observer
        self._archive_root = archive_root
        self._sequence = 0
        self._active_actor: str | None = None
        self._queued: list[tuple[int, str]] = []
        self._last_event: DaemonWriteEvent | None = None
        self._over_budget_holds = 0
        self._accepting = True
        self._executions: set[asyncio.Task[object]] = set()
        self._managed: set[asyncio.Task[object]] = set()
        self._idle = asyncio.Event()
        self._idle.set()
        self._detached_writer_failures = 0
        self._detached_writer_failures_by_actor: dict[str, int] = {}
        self._publish_telemetry()

    def snapshot(self) -> DaemonWriteSnapshot:
        return DaemonWriteSnapshot(
            active_actor=self._active_actor,
            queued_actors=tuple(actor for _sequence, actor in self._queued),
            last_event=self._last_event,
            accepting=self._accepting,
            detached_writer_failures=self._detached_writer_failures,
            detached_writer_failures_by_actor=tuple(sorted(self._detached_writer_failures_by_actor.items())),
            over_budget_holds=self._over_budget_holds,
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

        current_task = asyncio.current_task()
        if current_task is None:
            raise RuntimeError("daemon write coordination requires an asyncio task")
        active_lease = _ACTIVE_LEASE.get()
        if active_lease is not None and active_lease[0] is self:
            if active_lease[1] is current_task:
                return await operation()
            raise RuntimeError(
                "daemon write lease was inherited by a child task; nested writes must run in the owning task"
            )
        if not self._accepting and current_task not in self._managed:
            # Shutdown drains managed post-write tasks, so their writes stay
            # admissible: refusing them would strand the terminal receipt the
            # drain exists to deliver. New external callers are still refused.
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
            return await asyncio.shield(execution)
        except asyncio.CancelledError:
            request.caller_cancelled = True
            if not request.acquired:
                execution.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.shield(execution)
            raise

    async def _execute(
        self,
        request: _WriteRequest,
        operation: Callable[[], Awaitable[T]],
        on_admit: Callable[[], None] | None = None,
    ) -> T:
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
        if on_admit is not None:
            try:
                on_admit()
            except BaseException:
                self._active_actor = None
                self._lock.release()
                raise
        owner = asyncio.current_task()
        if owner is None:  # pragma: no cover - asyncio always owns created tasks
            self._lock.release()
            raise RuntimeError("coordinator execution has no owning task")
        token = _ACTIVE_LEASE.set((self, owner))
        budget_s = write_hold_budget_s(request.actor)
        hold_token = enter_write_hold(request.actor, budget_s)
        outcome: WriteOutcome = "success"
        try:
            # Establish the storage-side authorization in the coordinator-owned
            # task.  ``_run_in_daemon_thread`` copies this context into the
            # actual writer thread, so every writable open remains behind the
            # same gate even when the callable is synchronous.
            with write_lease(request.actor, archive_root=self._archive_root, coordinator=self):
                return await operation()
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except BaseException:
            outcome = "error"
            raise
        finally:
            exit_write_hold(hold_token)
            _ACTIVE_LEASE.reset(token)
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
                outcome="degraded" if over_budget else "ok",
                reason="hold_over_budget" if over_budget else "within_budget",
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
            return await _run_in_daemon_thread(
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
        if timeout < 0:
            raise ValueError("shutdown timeout must be non-negative")
        self._accepting = False
        self._publish_telemetry()
        try:
            async with asyncio.timeout(timeout):
                await self._idle.wait()
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
                if not self._executions:
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
            if not self._executions:
                self._idle.set()

        task.add_done_callback(completed)

    def create_managed_task(self, operation: Awaitable[object], *, actor: str) -> asyncio.Task[object]:
        """Track post-write lifecycle work so shutdown drains it before loop close."""

        async def managed_operation() -> object:
            return await operation

        task: asyncio.Task[object] = asyncio.create_task(managed_operation(), name=f"polylogue-managed:{actor}")
        self._managed.add(task)
        task.add_done_callback(self._managed.discard)
        self._track_execution(task, actor=actor)
        return task

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
            }
        with _TELEMETRY_LOCK:
            _LATEST_TELEMETRY.clear()
            _LATEST_TELEMETRY.update(payload)


async def _run_in_daemon_thread(
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
        error: BaseException | None = None
        try:
            # The context copied from the coordinator task carries the lease;
            # bind this concrete worker thread before any SQLite factory runs.
            def invoke() -> T:
                if thread_grant is not None:
                    bind_write_lease_thread(thread_grant)
                return function(*args, **kwargs)

            value = context.run(invoke)
        except BaseException as exc:
            error = exc
            with contextlib.suppress(InvalidStateError):
                result.set_exception(exc)
        else:
            with contextlib.suppress(InvalidStateError):
                result.set_result(value)

        if loop.is_closed():
            # The result future is abandoned either way; the error lane keeps
            # the original failure attached instead of dropping it.
            emit(
                "daemon.writer.result_abandoned",
                level=WARNING,
                outcome="error",
                reason="event_loop_closed",
                thread=thread_name,
                error_type=type(error).__name__ if error is not None else None,
                error_detail=str(error) if error is not None else None,
            )

    threading.Thread(target=worker, name=thread_name, daemon=True).start()
    return await asyncio.wrap_future(result, loop=loop)


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

        future = asyncio.run_coroutine_threadsafe(hold_lease(), self._loop)
        if not settled.wait(self._timeout):
            self._loop.call_soon_threadsafe(release.set)
            future.cancel()
            with contextlib.suppress(TimeoutError, CancelledError):
                future.result(timeout=self._timeout)
            raise TimeoutError(f"timed out waiting for daemon write gate actor={actor}")
        if not entered.is_set():
            future.result(timeout=self._timeout)
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
                try:
                    future.result(timeout=self._timeout)
                except TimeoutError:
                    future.cancel()
                    emit(
                        "daemon.writer.release_timed_out",
                        level=WARNING,
                        outcome="degraded",
                        reason="release_timeout",
                        actor=actor,
                        timeout_ms=round(self._timeout * 1000, 3),
                    )

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
                    waited_ms=round(waited * 1000, 3),
                )

    @property
    def owner_loop(self) -> asyncio.AbstractEventLoop:
        """The explicitly composed loop that owns this bridge's writer."""
        return self._loop

    async def run_async(self, actor: str, function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
        """Borrow the already-owned loop for one staged operation publication."""
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("staged publication must run on the bridge's owner loop")
        pending = asyncio.create_task(self._coordinator.run_sync(actor, function, *args, **kwargs))
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            # A lifecycle cancellation is not permission to release the
            # writer or abandon the receipt of an admitted callable.
            while not pending.done():
                try:
                    await asyncio.shield(pending)
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
        future = asyncio.run_coroutine_threadsafe(
            self._coordinator.run_sync(actor, function, *args, **kwargs), self._loop
        )
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
