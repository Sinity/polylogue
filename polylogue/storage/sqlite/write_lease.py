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

import asyncio
import contextvars
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "UnleasedWriteError",
    "WriteHoldExceededError",
    "WriteLease",
    "arm_write_lease_enforcement",
    "bind_write_lease_thread",
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


@dataclass(slots=True)
class WriteLease:
    """A held authorization to open write-mode connections in this context."""

    actor: str
    acquired_at: float
    max_hold_seconds: float | None
    archive_root: Path | None = None
    coordinator: object | None = None
    owner_task_id: int | None = None
    owner_thread_id: int = 0
    bound_thread_ids: set[int] | None = None

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
_PROCESS_ENFORCEMENT = False


def _current_task_id() -> int | None:
    """Return the running task identity without requiring an event loop."""
    try:
        task = asyncio.current_task()
    except RuntimeError:
        return None
    return id(task) if task is not None else None


def write_lease_enforced() -> bool:
    """Whether an unleased write-mode open is an error in this thread."""
    return _PROCESS_ENFORCEMENT or bool(getattr(_ENFORCEMENT, "armed", _ENFORCEMENT_DEFAULT))


@contextmanager
def arm_write_lease_enforcement(*, armed: bool = True, process_wide: bool = False) -> Iterator[None]:
    """Make unleased write-mode opens raise for the duration of the block.

    Thread-local rather than process-global because the daemon runs its writer
    on dedicated threads while pytest and embedded callers share the process.
    The daemon opts into ``process_wide=True`` for its process-lifetime writer
    boundary; tests and one-shot callers keep the default local scope.
    """
    global _PROCESS_ENFORCEMENT
    previous = getattr(_ENFORCEMENT, "armed", _ENFORCEMENT_DEFAULT)
    previous_process = _PROCESS_ENFORCEMENT
    _ENFORCEMENT.armed = armed
    if process_wide:
        _PROCESS_ENFORCEMENT = armed
    try:
        yield
    finally:
        _ENFORCEMENT.armed = previous
        _PROCESS_ENFORCEMENT = previous_process


def current_write_lease() -> WriteLease | None:
    """Return the lease held by this context, if any."""
    return _ACTIVE.get()


def require_write_lease(purpose: str, *, archive_root: str | Path | None = None) -> WriteLease | None:
    """Assert the caller may open a write-mode connection for ``purpose``.

    Returns the held lease, or ``None`` where enforcement is not armed. Raising
    here is what makes "the daemon is the sole writer" an exception rather than
    a review rule: every write-mode factory calls this before connecting.
    """
    lease = _ACTIVE.get()
    if lease is not None:
        task_id = _current_task_id()
        thread_id = threading.get_ident()
        allowed_threads = lease.bound_thread_ids or {lease.owner_thread_id}
        if task_id is not None:
            if task_id != lease.owner_task_id:
                raise UnleasedWriteError(f"{purpose} uses a write lease inherited by a child task")
        elif thread_id not in allowed_threads:
            raise UnleasedWriteError(f"{purpose} uses a write lease from an unauthorized thread")
        if archive_root is not None and lease.archive_root is not None:
            expected = Path(archive_root).resolve()
            actual = lease.archive_root.resolve()
            if expected != actual:
                raise UnleasedWriteError(
                    f"{purpose} is outside the archive bound to writer {lease.actor}: {expected} != {actual}"
                )
        return lease
    if not write_lease_enforced():
        return None
    raise UnleasedWriteError(
        f"{purpose} requires the daemon write lease; open it inside write_lease(...) "
        "so the single-writer boundary is serialized in-process"
    )


def bind_write_lease_thread() -> None:
    """Authorize the current thread for a coordinator-owned sync operation."""
    lease = _ACTIVE.get()
    if lease is None:
        raise UnleasedWriteError("cannot bind a thread without an active write lease")
    if lease.bound_thread_ids is None:
        lease.bound_thread_ids = {lease.owner_thread_id}
    lease.bound_thread_ids.add(threading.get_ident())


@contextmanager
def write_lease(
    actor: str,
    *,
    max_hold_seconds: float | None = None,
    archive_root: str | Path | None = None,
    coordinator: object | None = None,
) -> Iterator[WriteLease]:
    """Hold the write lease for ``actor``, authorizing write-mode opens.

    Re-entrant within one context: nested acquisitions return the outer lease
    rather than a second authority, so a publish nested inside a batch does not
    reset the outer hold's budget.
    """
    held = _ACTIVE.get()
    if held is not None:
        if (
            archive_root is not None
            and held.archive_root is not None
            and Path(archive_root).resolve() != held.archive_root.resolve()
        ):
            raise UnleasedWriteError("nested write lease requested for a different archive root")
        if coordinator is not None and held.coordinator is not None and coordinator is not held.coordinator:
            raise UnleasedWriteError("nested write lease requested by a different coordinator")
        yield held
        return
    lease = WriteLease(
        actor=actor,
        acquired_at=time.perf_counter(),
        max_hold_seconds=max_hold_seconds,
        archive_root=Path(archive_root).resolve() if archive_root is not None else None,
        coordinator=coordinator,
        owner_task_id=_current_task_id(),
        owner_thread_id=threading.get_ident(),
        bound_thread_ids={threading.get_ident()},
    )
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
