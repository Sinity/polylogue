"""Writer hold timing shared by daemon telemetry and admitted work.

Elapsed thresholds describe hold duration. They never withdraw admitted work
or prevent an acquired input from publishing its cursor.
"""

from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WriteHold:
    """One admitted writer hold and its diagnostic threshold."""

    actor: str
    budget_s: float
    started: float

    def elapsed_s(self) -> float:
        return time.monotonic() - self.started


_ACTIVE_WRITE_HOLD: contextvars.ContextVar[WriteHold | None] = contextvars.ContextVar(
    "polylogue_active_write_hold", default=None
)


def enter_write_hold(actor: str, budget_s: float) -> contextvars.Token[WriteHold | None]:
    """Publish timing metadata for the admitted hold."""
    return _ACTIVE_WRITE_HOLD.set(WriteHold(actor=actor, budget_s=budget_s, started=time.monotonic()))


def exit_write_hold(token: contextvars.Token[WriteHold | None]) -> None:
    """Retire the current hold's timing metadata."""
    _ACTIVE_WRITE_HOLD.reset(token)


def active_write_hold() -> WriteHold | None:
    """Return the current hold's timing metadata, or None outside admission."""
    return _ACTIVE_WRITE_HOLD.get()


__all__ = ["WriteHold", "active_write_hold", "enter_write_hold", "exit_write_hold"]
