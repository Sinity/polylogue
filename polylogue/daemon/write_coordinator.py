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
import threading
import time
import weakref
from collections.abc import Awaitable, Callable, Iterator
from concurrent.futures import CancelledError
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, ParamSpec, TypeVar

from polylogue.logging import get_logger

logger = get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")
WritePhase = Literal["queued", "acquired", "released"]
WriteOutcome = Literal["success", "error", "cancelled"]


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


@dataclass(frozen=True, slots=True)
class DaemonWriteSnapshot:
    """Request-safe in-memory view of coordinator state."""

    active_actor: str | None
    queued_actors: tuple[str, ...]
    last_event: DaemonWriteEvent | None
    accepting: bool = True


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
}


class DaemonWriteCoordinator:
    """Fair async gate around every archive write actor in one daemon.

    Cancellation is ownership-aware. A queued request is removed immediately;
    an admitted request continues in its coordinator-owned task until the
    underlying coroutine or thread really finishes. Thus callers can observe
    bounded cancellation without allowing the next SQLite writer to overlap.
    """

    def __init__(self, *, observer: WriteEventObserver | None = None) -> None:
        self._lock = asyncio.Lock()
        self._observer = observer
        self._sequence = 0
        self._active_actor: str | None = None
        self._queued: list[tuple[int, str]] = []
        self._last_event: DaemonWriteEvent | None = None
        self._accepting = True
        self._executions: set[asyncio.Task[object]] = set()
        self._idle = asyncio.Event()
        self._idle.set()
        self._publish_telemetry()

    def snapshot(self) -> DaemonWriteSnapshot:
        return DaemonWriteSnapshot(
            active_actor=self._active_actor,
            queued_actors=tuple(actor for _sequence, actor in self._queued),
            last_event=self._last_event,
            accepting=self._accepting,
        )

    async def run(self, actor: str, operation: Callable[[], Awaitable[T]]) -> T:
        """Run one async write operation under the process-wide gate."""
        if not actor:
            raise ValueError("daemon write actor must be non-empty")

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
            self._execute(request, operation),
            name=f"polylogue-writer:{actor}:{request.sequence}",
        )
        self._track_execution(execution)
        try:
            return await asyncio.shield(execution)
        except asyncio.CancelledError:
            request.caller_cancelled = True
            if not request.acquired:
                execution.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.shield(execution)
            raise

    async def _execute(self, request: _WriteRequest, operation: Callable[[], Awaitable[T]]) -> T:
        try:
            await self._lock.acquire()
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
        token = _ACTIVE_LEASE.set((self, owner))
        outcome: WriteOutcome = "success"
        try:
            return await operation()
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except BaseException:
            outcome = "error"
            raise
        finally:
            _ACTIVE_LEASE.reset(token)
            if request.caller_cancelled and outcome == "success":
                outcome = "cancelled"
            hold_seconds = time.perf_counter() - acquired_at
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
                )
            )
            logger.info(
                "daemon writer released actor=%s wait_s=%.6f hold_s=%.6f outcome=%s queued=%d",
                request.actor,
                wait_seconds,
                hold_seconds,
                outcome,
                len(self._queued),
            )

    async def run_sync(self, actor: str, function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
        """Run blocking writer work without releasing the gate on cancellation."""

        async def operation() -> T:
            return await asyncio.to_thread(function, *args, **kwargs)

        return await self.run(actor, operation)

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

    def _track_execution(self, execution: asyncio.Task[T]) -> None:
        task = execution  # preserve the concrete result type for ``run``
        self._executions.add(task)
        self._idle.clear()

        def completed(done: asyncio.Task[object]) -> None:
            self._executions.discard(done)
            if not self._executions:
                self._idle.set()
            if done.cancelled():
                return
            try:
                exception = done.exception()
            except Exception:
                logger.warning("detached daemon writer failed", exc_info=True)
            else:
                if exception is not None:
                    logger.warning("detached daemon writer failed: %s", exception)

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
        except Exception:
            logger.warning("daemon writer telemetry observer failed", exc_info=True)

    def _publish_telemetry(self) -> None:
        snapshot = self.snapshot()
        event = snapshot.last_event
        payload: dict[str, object] = {
            "active_actor": snapshot.active_actor,
            "queued_actors": list(snapshot.queued_actors),
            "queue_depth": len(snapshot.queued_actors),
            "accepting": snapshot.accepting,
            "last_event": None,
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
    def hold(self, actor: str) -> Iterator[None]:
        entered = threading.Event()
        settled = threading.Event()
        release = asyncio.Event()

        async def hold_lease() -> None:
            async def wait_for_release() -> None:
                entered.set()
                settled.set()
                await release.wait()

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
            yield
        finally:
            self._loop.call_soon_threadsafe(release.set)
            try:
                future.result(timeout=self._timeout)
            except TimeoutError:
                future.cancel()
                logger.warning("timed out releasing daemon write gate actor=%s", actor)


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
        return payload


_COORDINATORS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, DaemonWriteCoordinator] = (
    weakref.WeakKeyDictionary()
)


def daemon_write_coordinator() -> DaemonWriteCoordinator:
    """Return the sole coordinator for the current process event loop."""
    loop = asyncio.get_running_loop()
    coordinator = _COORDINATORS.get(loop)
    if coordinator is None:
        coordinator = DaemonWriteCoordinator()
        _COORDINATORS[loop] = coordinator
    return coordinator


__all__ = [
    "DaemonWriteCoordinator",
    "DaemonWriteEvent",
    "DaemonWriteSnapshot",
    "DaemonWriteThreadBridge",
    "daemon_write_coordinator",
    "daemon_write_telemetry_payload",
]
