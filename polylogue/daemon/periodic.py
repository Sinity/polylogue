"""One observable runner for every daemon cadence loop.

Sixteen declared ``PERIODIC`` services (``polylogue/daemon/services.py``) each
carried their own ``while True`` in ``polylogue/daemon/cli.py``. Each one
re-implemented the same five decisions -- sleep before or after the work, how a
missing archive file is skipped, how an exception is classified, how a startup
gate is awaited, what the cadence is -- and each got them slightly differently.
The visible cost was that a frozen loop was indistinguishable from an idle one
from outside the process: nothing recorded when a loop last ran, when it is due,
or what its last failure was (polylogue-74wvj).

:class:`PeriodicRunner` owns those five decisions once:

* the cadence comes from the service registry, not a constant next to the body;
* each tick is jittered so sixteen loops do not converge on one instant;
* a ``precondition`` returning ``False`` is a recorded *skip*, not a silent
  ``continue``;
* an exception is recorded against the loop and the cadence continues, unless
  the loop declares ``on_error="propagate"`` (the supervisor's ``DEGRADE`` and
  restart-on-recovery policies need the failure to escape);
* :class:`PeriodicGate` records *who* is waiting on a startup gate and since
  when, so "waiting for catch-up" reads differently from "stalled".

:meth:`PeriodicRunner.snapshot` is what the status and metrics surfaces render.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from polylogue.logging import WARNING, emit

#: Fraction of the interval added as random delay before each tick. Sixteen
#: loops started in one composition tick would otherwise stay phase-locked for
#: the life of the process, and their writer admissions would collide forever.
DEFAULT_JITTER_RATIO = 0.1

OnError = Literal["record", "propagate"]


class PassOutcome(str, Enum):
    """What one cadence pass found, for loops that drain a backlog.

    A loop that returns one of these from its work callable gets drain-cycle
    accounting; a loop that returns anything else (every maintenance sweep
    with no backlog of its own) is unaffected and reports nothing here.
    """

    PROGRESSED = "progressed"
    """The pass did work. The backlog is not known to be empty."""

    DRAINED = "drained"
    """The pass completed and found nothing left to do."""


#: How long a gated loop waits for watcher registration before
#: proceeding without having observed it. One value, because every gated loop
#: waits on the same gate.
WATCHER_REGISTRATION_TIMEOUT_SECONDS = 1800.0


@dataclass(slots=True)
class PeriodicLoopState:
    """Externally readable state of one cadence loop."""

    name: str
    interval_s: float
    #: Wall-clock seconds; these are read by operators, not used for scheduling.
    last_run_started_at: float | None = None
    last_run_completed_at: float | None = None
    next_run_at: float | None = None
    last_error: str | None = None
    last_error_type: str | None = None
    last_error_at: float | None = None
    runs: int = 0
    failures: int = 0
    skips: int = 0
    #: Passes this loop started because an event woke it rather than because
    #: its cadence elapsed. Zero here on a loop wired to the bus means the
    #: producer is not reaching it, which polling alone could never show.
    wakeups: int = 0
    #: Name of the gate this loop is waiting on right now, and since when.
    #: Present only while blocked: an idle loop between ticks has neither,
    #: which is exactly the idle/stalled distinction the old loops lost.
    blocked_on: str | None = None
    blocked_since: float | None = None
    #: Whether the last reported pass found the backlog empty. ``None`` on a
    #: loop that does not report a pass outcome at all.
    drained: bool | None = None
    #: Completed drain cycles: the number of times this loop crossed from
    #: "there was work" into "there is none". It counts *edges*, so a backlog
    #: that is already empty and gets woken a hundred times still shows one.
    #: An empty loop that kept re-announcing completion is how polylogue-09rn
    #: turned an idle backlog into a 300-second spin.
    drain_transitions: int = 0
    last_drained_at: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "interval_s": self.interval_s,
            "last_run_started_at": self.last_run_started_at,
            "last_run_completed_at": self.last_run_completed_at,
            "next_run_at": self.next_run_at,
            "last_error": self.last_error,
            "last_error_type": self.last_error_type,
            "last_error_at": self.last_error_at,
            "runs": self.runs,
            "failures": self.failures,
            "skips": self.skips,
            "wakeups": self.wakeups,
            "blocked_on": self.blocked_on,
            "blocked_since": self.blocked_since,
            "drained": self.drained,
            "drain_transitions": self.drain_transitions,
            "last_drained_at": self.last_drained_at,
        }


@dataclass(slots=True)
class PeriodicGate:
    """A startup gate that reports who it is blocking.

    The old call sites awaited a bare :class:`asyncio.Event`, so a daemon whose
    registration never completed showed nothing at all: no loop named itself as
    waiting, and the absence of ticks looked like an idle archive.
    """

    name: str
    event: asyncio.Event | None
    timeout_s: float
    waiting: set[str] = field(default_factory=set)

    @property
    def released(self) -> bool:
        return self.event is None or self.event.is_set()

    async def wait(self, waiter: str) -> bool:
        """Await release for ``waiter``. Returns whether the gate was observed."""
        if self.released:
            return True
        self.waiting.add(waiter)
        try:
            await asyncio.wait_for(self.event.wait(), timeout=self.timeout_s)  # type: ignore[union-attr]
            return True
        except TimeoutError:
            emit(
                "daemon.watcher_registered.timeout",
                level=WARNING,
                outcome="unmeasured",
                reason="watcher_registration_not_observed",
                loop=waiter,
                timeout_ms=round(self.timeout_s * 1000, 3),
            )
            return False
        finally:
            self.waiting.discard(waiter)


class PeriodicRunner:
    """Owns every daemon cadence loop's scheduling, gating and error policy."""

    def __init__(
        self,
        *,
        jitter_ratio: float = DEFAULT_JITTER_RATIO,
        rng: random.Random | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._states: dict[str, PeriodicLoopState] = {}
        self._jitter_ratio = jitter_ratio
        self._rng = rng if rng is not None else random.Random()
        # Left unbound when not injected so the lookup is late: a test that
        # patches ``asyncio.sleep`` to end a loop still reaches this runner,
        # the way it reached each loop's own ``await asyncio.sleep`` before.
        self._sleep = sleep
        self._clock = clock

    def snapshot(self) -> tuple[PeriodicLoopState, ...]:
        """Current per-loop state, in registration order."""
        return tuple(self._states.values())

    def state(self, name: str) -> PeriodicLoopState | None:
        return self._states.get(name)

    def payload(self) -> dict[str, object]:
        """Bounded, secret-free payload for the status and metrics surfaces."""
        return {"loops": [state.as_dict() for state in self._states.values()]}

    async def run(
        self,
        name: str,
        work: Callable[[], Awaitable[object]],
        *,
        interval_s: float | Callable[[], float],
        gate: PeriodicGate | None = None,
        wakeup: asyncio.Event | None = None,
        precondition: Callable[[], bool] | None = None,
        run_first: bool = False,
        on_error: OnError = "record",
        error_event: str | None = None,
    ) -> None:
        """Drive one cadence loop until cancelled.

        ``run_first`` distinguishes the loops that must publish something before
        their first sleep (status snapshot, convergence retry) from those whose
        first tick is deliberately one interval away (WAL checkpoint, daily
        optimize). Preserving that split per loop is the point: collapsing it
        would either delay first readiness or add a startup IO burst.

        ``wakeup`` shortens -- never replaces -- the wait. A loop given one
        reacts as soon as an event fires and still takes its declared
        reconciliation tick, because in-process delivery can be missed (no
        subscriber yet, a handler that raised) and is therefore an
        optimization, not authority (``polylogue/daemon/event_bus.py``).
        """
        state = self._states.setdefault(name, PeriodicLoopState(name=name, interval_s=_resolve(interval_s)))
        if gate is not None and not gate.released:
            state.blocked_on = gate.name
            state.blocked_since = self._clock()
            try:
                await gate.wait(name)
            finally:
                state.blocked_on = None
                state.blocked_since = None

        first = True
        while True:
            interval = _resolve(interval_s)
            state.interval_s = interval
            if not (first and run_first):
                delay = interval + self._rng.uniform(0.0, interval * self._jitter_ratio)
                state.next_run_at = self._clock() + delay
                if wakeup is None:
                    await self._delay(delay)
                else:
                    await self._sleep_until_woken(wakeup, delay, state)
            first = False
            if precondition is not None and not precondition():
                state.skips += 1
                state.next_run_at = self._clock() + interval
                continue
            state.last_run_started_at = self._clock()
            try:
                outcome = await work()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                state.failures += 1
                state.last_error = str(exc)
                state.last_error_type = type(exc).__name__
                state.last_error_at = self._clock()
                emit(
                    error_event or "daemon.periodic.failed",
                    level=WARNING,
                    outcome="error",
                    loop=name,
                    error_type=type(exc).__name__,
                    error_detail=str(exc),
                )
                if on_error == "propagate":
                    raise
            else:
                state.runs += 1
                state.last_run_completed_at = self._clock()
                if isinstance(outcome, PassOutcome):
                    self._record_pass_outcome(state, outcome)
            state.next_run_at = self._clock() + interval

    def _record_pass_outcome(self, state: PeriodicLoopState, outcome: PassOutcome) -> None:
        """Count the progressed -> drained edge, never the flat stretch after it.

        The terminal transition belongs to the *cycle*, not to the pass: a
        drained backlog woken again by an ingest that changed nothing has not
        completed a second drain, and publishing one per wake is exactly the
        idle resubmission this exists to make impossible. Re-arming happens
        on the next pass that actually progresses.
        """
        drained = outcome is PassOutcome.DRAINED
        if drained and state.drained is not True:
            state.drain_transitions += 1
            state.last_drained_at = self._clock()
        state.drained = drained

    async def _delay(self, seconds: float) -> None:
        if self._sleep is None:
            await asyncio.sleep(seconds)
            return
        await self._sleep(seconds)

    async def _sleep_until_woken(self, wakeup: asyncio.Event, delay: float, state: PeriodicLoopState) -> None:
        """Wait out the cadence unless the event fires first."""
        sleeper = asyncio.ensure_future(self._delay(delay))
        waiter = asyncio.ensure_future(wakeup.wait())
        try:
            done, _pending = await asyncio.wait({sleeper, waiter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (sleeper, waiter):
                if not task.done():
                    task.cancel()
        if waiter in done:
            wakeup.clear()
            state.wakeups += 1


def watcher_registered_gate(event: asyncio.Event | None) -> PeriodicGate:
    """Wrap the watcher's registration event so waiting loops name themselves."""
    return PeriodicGate(name="watcher_registered", event=event, timeout_s=WATCHER_REGISTRATION_TIMEOUT_SECONDS)


def _resolve(interval_s: float | Callable[[], float]) -> float:
    return float(interval_s() if callable(interval_s) else interval_s)


_RUNNER: PeriodicRunner | None = None


def daemon_periodic_runner() -> PeriodicRunner:
    """The process-wide runner every daemon cadence loop registers with."""
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = PeriodicRunner()
    return _RUNNER


def reset_daemon_periodic_runner() -> None:
    """Drop the process-wide runner. For daemon composition and tests only."""
    global _RUNNER
    _RUNNER = None


def periodic_loop_payload() -> dict[str, object]:
    """Per-loop cadence evidence for status and metrics surfaces."""
    return daemon_periodic_runner().payload()


__all__ = [
    "WATCHER_REGISTRATION_TIMEOUT_SECONDS",
    "DEFAULT_JITTER_RATIO",
    "PassOutcome",
    "PeriodicGate",
    "PeriodicLoopState",
    "PeriodicRunner",
    "watcher_registered_gate",
    "daemon_periodic_runner",
    "periodic_loop_payload",
    "reset_daemon_periodic_runner",
]
