"""Observable progress of one long unit of work, as throttled structured events.

Preparing a large source can run for minutes without any durable archive
change until it publishes. A watcher that judges liveness by archive changes
alone cannot tell that from a stall. Work that declares itself with
:func:`reports_work_progress` instead emits ``daemon.work.progress`` with
monotonically growing counters (messages and bytes processed) at most every
:data:`PROGRESS_INTERVAL_S`, plus one final event, so liveness is judged by
work done rather than by elapsed time.

Counting is cheap and context-local: :func:`advance_work_progress` is a no-op
outside a declared unit, and the unit belongs to the context that entered it.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Final, ParamSpec, TypeVar

from polylogue.logging import emit

P = ParamSpec("P")
T = TypeVar("T")

#: Seconds between progress events of one unit of work.
PROGRESS_INTERVAL_S: Final = 10.0

WORK_PROGRESS_EVENT: Final = "daemon.work.progress"


class WorkProgress:
    """Cumulative counters of one unit of work and its event throttle."""

    __slots__ = ("phase", "messages", "bytes", "_started", "_last_emitted")

    def __init__(self, phase: str) -> None:
        self.phase = phase
        self.messages = 0
        self.bytes = 0
        self._started = time.monotonic()
        self._last_emitted = self._started

    def advance(self, *, messages: int = 0, bytes: int = 0) -> None:
        self.messages += messages
        self.bytes += bytes
        now = time.monotonic()
        if now - self._last_emitted >= PROGRESS_INTERVAL_S:
            self._emit(now, outcome="ok")

    def _emit(self, now: float, *, outcome: str) -> None:
        self._last_emitted = now
        emit(
            WORK_PROGRESS_EVENT,
            outcome=outcome,
            phase=self.phase,
            messages=self.messages,
            bytes=self.bytes,
            duration_ms=round((now - self._started) * 1000, 3),
        )


_CURRENT: ContextVar[WorkProgress | None] = ContextVar("polylogue_work_progress", default=None)


@contextmanager
def work_progress(phase: str) -> Iterator[WorkProgress]:
    """Declare one unit of work whose progress is reported while it runs.

    A unit nested in another reports through the outer one, so one piece of
    work never emits two interleaved counter streams.
    """
    current = _CURRENT.get()
    if current is not None:
        yield current
        return
    progress = WorkProgress(phase)
    token = _CURRENT.set(progress)
    try:
        yield progress
    finally:
        _CURRENT.reset(token)
        progress._emit(time.monotonic(), outcome="ok")


def reports_work_progress(phase: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Run the decorated function as one declared unit of work."""

    def decorate(function: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with work_progress(phase):
                return function(*args, **kwargs)

        return wrapper

    return decorate


def advance_work_progress(*, messages: int = 0, bytes: int = 0) -> None:
    """Count work done by the current unit; outside one this does nothing."""
    progress = _CURRENT.get()
    if progress is not None:
        progress.advance(messages=messages, bytes=bytes)


__all__ = [
    "PROGRESS_INTERVAL_S",
    "WORK_PROGRESS_EVENT",
    "WorkProgress",
    "advance_work_progress",
    "reports_work_progress",
    "work_progress",
]
