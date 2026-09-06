"""The process's sole authority for opening a write-mode archive connection.

SQLite is the physical single-writer boundary; this module makes the in-process
boundary structural rather than conventional. Where enforcement is armed, a
write-mode connection is obtainable only from a held :class:`WriteLease`, so an
unserialized writer raises instead of contending through the busy timeout.

Enforcement is armed by the owner of the process's writer discipline (the
daemon) and is off elsewhere: one-shot CLI and API writers are their own single
writer and have no gate to be outside of.
"""

from __future__ import annotations

import contextvars
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

__all__ = [
    "UnleasedWriteError",
    "WriteHoldExceededError",
    "WriteLease",
    "arm_write_lease_enforcement",
    "current_write_lease",
    "require_write_lease",
    "write_lease",
    "write_lease_enforced",
]


class UnleasedWriteError(RuntimeError):
    """A write-mode connection was requested without holding the write lease."""

    code = "unleased_write"


class WriteHoldExceededError(RuntimeError):
    """A lease was released after its declared maximum hold.

    Raised at release, not during the hold: the lease cannot abort an operation
    already inside a SQLite transaction, and interrupting one would leave the
    hold's own writer in an undefined state. Its purpose is to make an
    over-budget hold a typed failure the caller must handle rather than a
    longer wait every other writer silently absorbs.
    """

    code = "write_hold_exceeded"


@dataclass(frozen=True, slots=True)
class WriteLease:
    """A held authorization to open write-mode connections in this context."""

    actor: str
    acquired_at: float
    max_hold_seconds: float | None

    @property
    def held_seconds(self) -> float:
        return time.perf_counter() - self.acquired_at

    @property
    def over_budget(self) -> bool:
        return self.max_hold_seconds is not None and self.held_seconds > self.max_hold_seconds


_ACTIVE: contextvars.ContextVar[WriteLease | None] = contextvars.ContextVar(
    "polylogue_active_write_lease", default=None
)
_ENFORCEMENT = threading.local()
_ENFORCEMENT_DEFAULT = False


def write_lease_enforced() -> bool:
    """Whether an unleased write-mode open is an error in this thread."""
    return bool(getattr(_ENFORCEMENT, "armed", _ENFORCEMENT_DEFAULT))


@contextmanager
def arm_write_lease_enforcement(*, armed: bool = True) -> Iterator[None]:
    """Make unleased write-mode opens raise for the duration of the block.

    Thread-local rather than process-global because the daemon runs its writer
    on dedicated threads while pytest and embedded callers share the process:
    arming globally would refuse writes the daemon never claimed to own.
    """
    previous = getattr(_ENFORCEMENT, "armed", _ENFORCEMENT_DEFAULT)
    _ENFORCEMENT.armed = armed
    try:
        yield
    finally:
        _ENFORCEMENT.armed = previous


def current_write_lease() -> WriteLease | None:
    """Return the lease held by this context, if any."""
    return _ACTIVE.get()


def require_write_lease(purpose: str) -> WriteLease | None:
    """Assert the caller may open a write-mode connection for ``purpose``.

    Returns the held lease, or ``None`` where enforcement is not armed. Raising
    here is what makes "the daemon is the sole writer" an exception rather than
    a review rule: every write-mode factory calls this before connecting.
    """
    lease = _ACTIVE.get()
    if lease is not None:
        return lease
    if not write_lease_enforced():
        return None
    raise UnleasedWriteError(
        f"{purpose} requires the daemon write lease; open it inside write_lease(...) "
        "so the single-writer boundary is serialized in-process"
    )


@contextmanager
def write_lease(actor: str, *, max_hold_seconds: float | None = None) -> Iterator[WriteLease]:
    """Hold the write lease for ``actor``, authorizing write-mode opens.

    Re-entrant within one context: nested acquisitions return the outer lease
    rather than a second authority, so a publish nested inside a batch does not
    reset the outer hold's budget.
    """
    held = _ACTIVE.get()
    if held is not None:
        yield held
        return
    lease = WriteLease(actor=actor, acquired_at=time.perf_counter(), max_hold_seconds=max_hold_seconds)
    token = _ACTIVE.set(lease)
    try:
        yield lease
    finally:
        _ACTIVE.reset(token)
        if lease.over_budget:
            raise WriteHoldExceededError(
                f"writer {actor} held the lease {lease.held_seconds:.3f}s against a declared "
                f"{lease.max_hold_seconds:.3f}s budget; every other writer waited behind it"
            )
