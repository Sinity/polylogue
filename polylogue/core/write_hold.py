"""The declared bound on one archive-writer hold, readable from any ring.

The daemon's writer gate admits one unit of work at a time and declares how
long it may hold the sole archive writer. The work that has to respect the
bound runs in ``polylogue.sources`` and ``polylogue.storage``, which must not
import ``polylogue.daemon``, so the active hold is published here -- the same
placement argument as ``polylogue.core.degraded``.

A hold cannot be preempted: work already inside a SQLite transaction has to
finish it. What the bound buys is that every checkpoint a unit of work does
offer -- between files, between records -- refuses to continue once the bound
is spent, so overshoot is one work item rather than however long the rest of
the unit takes.
"""

from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WriteHold:
    """One admitted writer hold and the bound it was admitted under."""

    actor: str
    budget_s: float
    started: float

    def elapsed_s(self) -> float:
        return time.monotonic() - self.started

    def remaining_s(self) -> float:
        return self.budget_s - self.elapsed_s()


class WriteHoldBudgetError(RuntimeError):
    """A unit of work reached a checkpoint past its declared hold bound."""

    def __init__(self, *, actor: str, checkpoint: str, hold_seconds: float, budget_s: float) -> None:
        super().__init__(
            f"writer hold past its declared budget actor={actor} checkpoint={checkpoint} "
            f"hold_s={hold_seconds:.3f} budget_s={budget_s:.3f}"
        )
        self.actor = actor
        self.checkpoint = checkpoint
        self.hold_seconds = hold_seconds
        self.budget_s = budget_s


_ACTIVE_WRITE_HOLD: contextvars.ContextVar[WriteHold | None] = contextvars.ContextVar(
    "polylogue_active_write_hold", default=None
)


def enter_write_hold(actor: str, budget_s: float) -> contextvars.Token[WriteHold | None]:
    """Publish the bound this context's writer hold was admitted under."""
    return _ACTIVE_WRITE_HOLD.set(WriteHold(actor=actor, budget_s=budget_s, started=time.monotonic()))


def exit_write_hold(token: contextvars.Token[WriteHold | None]) -> None:
    """Retire the hold published by :func:`enter_write_hold`."""
    _ACTIVE_WRITE_HOLD.reset(token)


def active_write_hold() -> WriteHold | None:
    """The hold this context runs under, or ``None`` off a coordinated write."""
    return _ACTIVE_WRITE_HOLD.get()


def write_hold_remaining_s() -> float | None:
    """Seconds left in this context's declared bound, ``None`` off a hold."""
    hold = _ACTIVE_WRITE_HOLD.get()
    return None if hold is None else hold.remaining_s()


def check_write_hold_budget(checkpoint: str) -> None:
    """Raise when this context's hold has already outrun its declared bound.

    Call it wherever the unit of work could stop; ``checkpoint`` names that
    point so the failure says which item overran. Off a coordinated write
    there is no bound to enforce and no clock is read.
    """
    hold = _ACTIVE_WRITE_HOLD.get()
    if hold is None:
        return
    elapsed = hold.elapsed_s()
    if elapsed > hold.budget_s:
        raise WriteHoldBudgetError(
            actor=hold.actor,
            checkpoint=checkpoint,
            hold_seconds=elapsed,
            budget_s=hold.budget_s,
        )


__all__ = [
    "WriteHold",
    "WriteHoldBudgetError",
    "active_write_hold",
    "check_write_hold_budget",
    "enter_write_hold",
    "exit_write_hold",
    "write_hold_remaining_s",
]
