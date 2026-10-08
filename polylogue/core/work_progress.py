"""Observable progress of one long unit of work, as throttled structured events.

Preparing a large source can run for minutes without any durable archive
change until it publishes. A watcher that judges liveness by archive changes
alone cannot tell that from a stall. Work that declares itself with
:func:`reports_work_progress` instead emits ``daemon.work.progress`` with
monotonically growing counters (messages and bytes processed) at most every
:data:`PROGRESS_INTERVAL_S`, plus one final event, so liveness is judged by
work done rather than by elapsed time.

Each event has a unique ``unit_id`` for one invocation and an optional
``productive_id`` for the stable source recipe across retries. Consumers can
ignore counter resets and count only advances beyond that recipe's high-water.

Counting is cheap and context-local: :func:`advance_work_progress` is a no-op
outside a declared unit, and the unit belongs to the context that entered it.
"""

from __future__ import annotations

import functools
import hashlib
import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Final, ParamSpec, TypeVar
from uuid import uuid4

from polylogue.logging import emit

P = ParamSpec("P")
T = TypeVar("T")

#: Seconds between progress events of one unit of work.
PROGRESS_INTERVAL_S: Final = 10.0

WORK_PROGRESS_EVENT: Final = "daemon.work.progress"


class WorkProgress:
    """Cumulative counters of one invocation and its productive source recipe."""

    __slots__ = ("phase", "unit_id", "productive_id", "messages", "bytes", "_started", "_last_emitted")

    def __init__(self, phase: str, productive_id: str | None) -> None:
        self.phase = phase
        self.unit_id = uuid4().hex
        self.productive_id = productive_id
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
            unit_id=self.unit_id,
            productive_id=self.productive_id,
            messages=self.messages,
            bytes=self.bytes,
            duration_ms=round((now - self._started) * 1000, 3),
        )


_CURRENT: ContextVar[WorkProgress | None] = ContextVar("polylogue_work_progress", default=None)


@contextmanager
def work_progress(phase: str, *, productive_id: str | None = None) -> Iterator[WorkProgress]:
    """Declare one unit of work whose progress is reported while it runs.

    A unit nested in another reports through the outer one, so one piece of
    work never emits two interleaved counter streams.
    """
    current = _CURRENT.get()
    if current is not None:
        yield current
        return
    progress = WorkProgress(phase, productive_id)
    token = _CURRENT.set(progress)
    try:
        yield progress
    finally:
        _CURRENT.reset(token)
        progress._emit(time.monotonic(), outcome="ok")


def reports_work_progress(
    phase: str,
    *,
    productive_identity: Callable[..., str | None] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Run a function as one declared unit with an optional retry-stable recipe ID."""

    def decorate(function: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            identity = productive_identity(*args, **kwargs) if productive_identity is not None else None
            with work_progress(phase, productive_id=identity):
                return function(*args, **kwargs)

        return wrapper

    return decorate


def advance_work_progress(*, messages: int = 0, bytes: int = 0) -> None:
    """Count work done by the current unit; outside one this does nothing."""
    progress = _CURRENT.get()
    if progress is not None:
        progress.advance(messages=messages, bytes=bytes)


def stable_productive_identity(recipe: tuple[object, ...]) -> str:
    """Hash an explicit, source-derived recipe for retry-comparable work."""
    encoded = json.dumps(recipe, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def utf8_byte_length(value: str) -> int:
    """Count UTF-8 bytes without allocating a second copy of a large string."""
    chunk_size = 64 * 1024
    return sum(
        len(value[offset : offset + chunk_size].encode("utf-8", "surrogatepass"))
        for offset in range(0, len(value), chunk_size)
    )


__all__ = [
    "PROGRESS_INTERVAL_S",
    "WORK_PROGRESS_EVENT",
    "WorkProgress",
    "advance_work_progress",
    "reports_work_progress",
    "stable_productive_identity",
    "utf8_byte_length",
    "work_progress",
]
