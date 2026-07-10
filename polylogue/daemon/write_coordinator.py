"""Process-wide serialization for daemon archive writers.

SQLite remains the archive's physical single-writer boundary.  This module
makes that boundary explicit inside ``polylogued`` so independent async loops
queue before opening write connections instead of contending through SQLite's
busy timeout.
"""

from __future__ import annotations

import asyncio
import contextvars
import time
import weakref
from collections.abc import Awaitable, Callable
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


WriteEventObserver = Callable[[DaemonWriteEvent], None]
_ACTIVE_COORDINATOR: contextvars.ContextVar[DaemonWriteCoordinator | None] = contextvars.ContextVar(
    "polylogue_active_daemon_write_coordinator",
    default=None,
)


class DaemonWriteCoordinator:
    """Fair async gate around every archive write actor in one daemon.

    ``asyncio.Lock`` documents FIFO acquisition for waiting tasks.  Admitted
    work is shielded and drained: cancelling the awaiting daemon task cannot
    release the gate while its coroutine or nested worker still owns SQLite.
    """

    def __init__(self, *, observer: WriteEventObserver | None = None) -> None:
        self._lock = asyncio.Lock()
        self._observer = observer
        self._sequence = 0
        self._active_actor: str | None = None
        self._queued: list[tuple[int, str]] = []
        self._last_event: DaemonWriteEvent | None = None

    def snapshot(self) -> DaemonWriteSnapshot:
        return DaemonWriteSnapshot(
            active_actor=self._active_actor,
            queued_actors=tuple(actor for _sequence, actor in self._queued),
            last_event=self._last_event,
        )

    async def run(self, actor: str, operation: Callable[[], Awaitable[T]]) -> T:
        """Run one async write operation under the process-wide gate."""
        if not actor:
            raise ValueError("daemon write actor must be non-empty")
        if _ACTIVE_COORDINATOR.get() is self:
            return await operation()

        self._sequence += 1
        sequence = self._sequence
        queued_at = time.perf_counter()
        self._queued.append((sequence, actor))
        self._emit(
            DaemonWriteEvent(
                phase="queued",
                actor=actor,
                sequence=sequence,
                queue_depth=len(self._queued),
            )
        )
        try:
            await self._lock.acquire()
        except BaseException:
            self._remove_queued(sequence)
            raise

        self._remove_queued(sequence)
        acquired_at = time.perf_counter()
        wait_seconds = acquired_at - queued_at
        self._active_actor = actor
        self._emit(
            DaemonWriteEvent(
                phase="acquired",
                actor=actor,
                sequence=sequence,
                queue_depth=len(self._queued),
                wait_seconds=wait_seconds,
            )
        )
        token = _ACTIVE_COORDINATOR.set(self)
        outcome: WriteOutcome = "success"
        operation_task: asyncio.Future[T] | None = None
        try:
            operation_task = asyncio.ensure_future(operation())
            return await asyncio.shield(operation_task)
        except asyncio.CancelledError:
            outcome = "cancelled"
            if operation_task is not None:
                await _drain_cancelled_operation(operation_task, actor=actor)
            raise
        except BaseException:
            outcome = "error"
            raise
        finally:
            _ACTIVE_COORDINATOR.reset(token)
            hold_seconds = time.perf_counter() - acquired_at
            self._active_actor = None
            self._lock.release()
            event = DaemonWriteEvent(
                phase="released",
                actor=actor,
                sequence=sequence,
                queue_depth=len(self._queued),
                wait_seconds=wait_seconds,
                hold_seconds=hold_seconds,
                outcome=outcome,
            )
            self._emit(event)
            logger.info(
                "daemon writer released actor=%s wait_s=%.6f hold_s=%.6f outcome=%s queued=%d",
                actor,
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

    def _remove_queued(self, sequence: int) -> None:
        self._queued = [item for item in self._queued if item[0] != sequence]

    def _emit(self, event: DaemonWriteEvent) -> None:
        self._last_event = event
        if self._observer is None:
            return
        try:
            self._observer(event)
        except Exception:
            logger.warning("daemon writer telemetry observer failed", exc_info=True)


async def _drain_cancelled_operation(operation: asyncio.Future[T], *, actor: str) -> None:
    """Wait until an admitted writer releases every nested worker resource."""
    while not operation.done():
        try:
            await asyncio.shield(operation)
        except asyncio.CancelledError:
            # Repeated shutdown signals must not let another writer overlap the
            # still-running thread.  The caller remains cancelled and exits as
            # soon as the thread reaches its own bounded completion point.
            continue
        except BaseException:
            break
    if operation.cancelled():
        return
    exception = operation.exception()
    if exception is not None:
        logger.warning("cancelled daemon writer failed before release actor=%s: %s", actor, exception)


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
    "daemon_write_coordinator",
]
