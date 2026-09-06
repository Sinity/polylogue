"""Bounded domain observations: the one shape every status surface reads.

A status number is only meaningful together with how it was obtained. An
observation therefore carries its collection state beside its value, and a
value exists only in the ``MEASURED`` state. There is no path from a
failed, timed-out, or unavailable read to the integer ``0``, the empty
list, or an ``ok`` severity.

Producers are the domain services the supervisor owns; each publishes its
own last observation onto a board. Compact status composes the board and
performs no archive walk of its own, so one slow domain cannot dominate
every answer.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

__all__ = [
    "Observation",
    "ObservationBoard",
    "ObservationState",
    "UnmeasuredObservationError",
    "observe_bounded",
    "observe_bounded_async",
    "worst_state",
]

T = TypeVar("T")


class ObservationState(str, Enum):
    """How a component's last collection attempt ended.

    Ordered worst-last by :data:`_SEVERITY_ORDER`; ``MEASURED`` is the only
    state that carries a value.
    """

    MEASURED = "measured"
    """The value is a real reading, including a genuine zero or empty set."""

    SKIPPED = "skipped"
    """Not collected: the component is not part of this configuration."""

    UNAVAILABLE = "unavailable"
    """The source of the value does not exist here (absent tier, no daemon)."""

    STALE = "stale"
    """A previous reading survives but its frame no longer matches."""

    TIMED_OUT = "timed_out"
    """Collection exceeded its declared budget and was abandoned."""

    UNREADABLE = "unreadable"
    """The source exists but could not be read (permission, corruption, lock)."""

    DEGRADED = "degraded"
    """Collected partially; the value covers less than it claims to."""

    FAILED = "failed"
    """Collection raised."""


_SEVERITY_ORDER: tuple[ObservationState, ...] = (
    ObservationState.MEASURED,
    ObservationState.SKIPPED,
    ObservationState.STALE,
    ObservationState.UNAVAILABLE,
    ObservationState.DEGRADED,
    ObservationState.TIMED_OUT,
    ObservationState.UNREADABLE,
    ObservationState.FAILED,
)


class UnmeasuredObservationError(RuntimeError):
    """Raised when a caller demands the value of an unmeasured observation."""


@dataclass(frozen=True, slots=True)
class Observation(Generic[T]):
    """One component's last reading, with the authority that produced it."""

    component: str
    state: ObservationState
    value: T | None = None
    reason: str | None = None
    frame: str | None = None
    """Identity of what was observed (generation, schema identity, source id)."""

    observed_at: float | None = None
    duration_s: float | None = None
    detail_operation: str | None = None
    """Name of the bounded operation that produces the expensive exact form."""

    @property
    def is_measured(self) -> bool:
        return self.state is ObservationState.MEASURED

    def require(self) -> T:
        """Return the measured value, or raise.

        Callers that cannot represent absence use this so an unmeasured
        component becomes a loud error rather than a plausible number.
        """
        if self.state is not ObservationState.MEASURED:
            raise UnmeasuredObservationError(
                f"{self.component}: {self.state.value}" + (f" ({self.reason})" if self.reason else "")
            )
        return self.value  # type: ignore[return-value]

    def as_dict(self) -> dict[str, object]:
        """Return the surface-neutral payload every adapter renders."""
        payload: dict[str, object] = {
            "component": self.component,
            "state": self.state.value,
            "value": self.value if self.state is ObservationState.MEASURED else None,
        }
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.frame is not None:
            payload["frame"] = self.frame
        if self.observed_at is not None:
            payload["observed_at"] = self.observed_at
        if self.duration_s is not None:
            payload["duration_s"] = self.duration_s
        if self.detail_operation is not None:
            payload["detail_operation"] = self.detail_operation
        return payload

    @classmethod
    def measured(
        cls,
        component: str,
        value: T,
        *,
        frame: str | None = None,
        observed_at: float | None = None,
        duration_s: float | None = None,
        detail_operation: str | None = None,
    ) -> Observation[T]:
        return cls(
            component=component,
            state=ObservationState.MEASURED,
            value=value,
            frame=frame,
            observed_at=observed_at if observed_at is not None else time.time(),
            duration_s=duration_s,
            detail_operation=detail_operation,
        )

    @classmethod
    def unmeasured(
        cls,
        component: str,
        state: ObservationState,
        *,
        reason: str,
        frame: str | None = None,
        observed_at: float | None = None,
        duration_s: float | None = None,
        detail_operation: str | None = None,
    ) -> Observation[T]:
        if state is ObservationState.MEASURED:
            raise ValueError("unmeasured() cannot construct a MEASURED observation")
        return cls(
            component=component,
            state=state,
            value=None,
            reason=reason,
            frame=frame,
            observed_at=observed_at if observed_at is not None else time.time(),
            duration_s=duration_s,
            detail_operation=detail_operation,
        )

    def staled(self, *, reason: str) -> Observation[T]:
        """Return this observation demoted to ``STALE``, dropping its value.

        A reading bound to a frame that has since changed is evidence that
        something was once true, never evidence about now.
        """
        return Observation(
            component=self.component,
            state=ObservationState.STALE,
            value=None,
            reason=reason,
            frame=self.frame,
            observed_at=self.observed_at,
            duration_s=self.duration_s,
            detail_operation=self.detail_operation,
        )


def worst_state(states: Iterable[ObservationState]) -> ObservationState:
    """Return the most severe state in *states*, or ``MEASURED`` if empty."""
    worst = ObservationState.MEASURED
    for state in states:
        if _SEVERITY_ORDER.index(state) > _SEVERITY_ORDER.index(worst):
            worst = state
    return worst


def observe_bounded(
    component: str,
    collect: Callable[[], T],
    *,
    budget_s: float,
    frame: str | None = None,
    detail_operation: str | None = None,
) -> Observation[T]:
    """Collect one observation under a wall-clock budget.

    A raise becomes ``FAILED``/``UNREADABLE``; exceeding *budget_s* becomes
    ``TIMED_OUT``. The collector keeps running in its thread after a
    timeout -- it is abandoned, not killed -- but its result is discarded,
    so a stalled collector delays nothing.
    """
    started = time.monotonic()
    result: list[T] = []
    failure: list[BaseException] = []

    def _run() -> None:
        try:
            result.append(collect())
        except BaseException as exc:
            failure.append(exc)

    worker = threading.Thread(target=_run, name=f"observe-{component}", daemon=True)
    worker.start()
    worker.join(timeout=budget_s)
    duration = time.monotonic() - started

    if worker.is_alive():
        return Observation.unmeasured(
            component,
            ObservationState.TIMED_OUT,
            reason=f"exceeded {budget_s:g}s budget",
            frame=frame,
            duration_s=duration,
            detail_operation=detail_operation,
        )
    if failure:
        return _failure_observation(
            component, failure[0], frame=frame, duration_s=duration, detail_operation=detail_operation
        )
    return Observation.measured(
        component,
        result[0],
        frame=frame,
        duration_s=duration,
        detail_operation=detail_operation,
    )


async def observe_bounded_async(
    component: str,
    collect: Callable[[], object],
    *,
    budget_s: float,
    frame: str | None = None,
    detail_operation: str | None = None,
) -> Observation[object]:
    """Await :func:`observe_bounded` without blocking the event loop."""
    return await asyncio.to_thread(
        observe_bounded,
        component,
        collect,
        budget_s=budget_s,
        frame=frame,
        detail_operation=detail_operation,
    )


def _failure_observation(
    component: str,
    exc: BaseException,
    *,
    frame: str | None,
    duration_s: float,
    detail_operation: str | None,
) -> Observation[T]:
    state = ObservationState.FAILED
    if isinstance(exc, (OSError, PermissionError)):
        state = ObservationState.UNREADABLE
    else:
        # sqlite3 is imported lazily: status must stay importable in a
        # process that never opens a database.
        import sqlite3

        if isinstance(exc, sqlite3.Error):
            state = ObservationState.UNREADABLE
    return Observation.unmeasured(
        component,
        state,
        reason=f"{type(exc).__name__}: {exc}",
        frame=frame,
        duration_s=duration_s,
        detail_operation=detail_operation,
    )


class ObservationBoard:
    """The last observation published by each component.

    Producers publish; status reads. Reading never collects, so a compact
    status answer costs a dictionary copy.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._observations: dict[str, Observation[object]] = {}

    def publish(self, observation: Observation[object]) -> None:
        with self._lock:
            self._observations[observation.component] = observation

    def get(self, component: str) -> Observation[object] | None:
        with self._lock:
            return self._observations.get(component)

    def get_or_unavailable(self, component: str, *, reason: str = "never published") -> Observation[object]:
        """Return *component*'s observation, or an explicit ``UNAVAILABLE`` one.

        A component that never reported is unavailable, never zero.
        """
        observation = self.get(component)
        if observation is None:
            return Observation.unmeasured(component, ObservationState.UNAVAILABLE, reason=reason)
        return observation

    def snapshot(self) -> Mapping[str, Observation[object]]:
        with self._lock:
            return dict(self._observations)

    def clear(self) -> None:
        with self._lock:
            self._observations.clear()
