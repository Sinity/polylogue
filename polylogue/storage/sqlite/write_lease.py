"""The process's sole authority for opening a write-mode archive connection.

SQLite is the physical single-writer boundary; this module makes the in-process
boundary structural rather than conventional. Where enforcement is armed, a
write-mode connection is obtainable only from a held :class:`WriteLease`, so an
unserialized writer raises instead of contending through the busy timeout.

Enforcement is armed by the owner of the process's writer discipline (the
daemon) and is off elsewhere: one-shot CLI and API writers are their own single
writer and have no gate to be outside of.

**Contexts do not isolate threads on this interpreter** (polylogue-1oa7o).
This checkout runs a free-threading CPython build, and a new
``threading.Thread`` or ``ThreadPoolExecutor.submit`` reads the *creating*
thread's ``ContextVar`` values rather than the declared defaults. So every
thread spawned while a lease is held inherits ``_ACTIVE`` -- the lease object
itself, not a copy. Authority therefore rests entirely on explicit thread
identity: ``WriteLease.bound_thread_ids`` and the ``owner_task_id`` check in
:func:`require_write_lease`, never on "the context did not carry it". Every
place that widens ``bound_thread_ids`` or hands back an existing lease
re-checks that identity, because an inheriting thread would otherwise pass by
default.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from polylogue.logging import get_logger

__all__ = [
    "UnleasedWriteError",
    "declared_unguarded_write",
    "install_archive_write_guard",
    "WriteHoldExceededError",
    "WriteLease",
    "WriteLeaseDelegation",
    "WriteLeaseThreadGrant",
    "adopt_write_lease",
    "arm_write_lease_enforcement",
    "bind_write_lease_thread",
    "current_write_lease",
    "delegate_write_lease",
    "grant_write_lease_thread",
    "require_write_lease",
    "write_lease",
    "write_lease_enforced",
]

logger = get_logger(__name__)


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

    Raised only when the hold exits cleanly. A hold that is already failing
    reports its own exception, which is the stronger signal and may carry
    typed partial-write facts; the budget breach is logged instead.
    """

    code = "write_hold_exceeded"


@dataclass(eq=False, slots=True)
class WriteLeaseDelegation:
    """One explicit, revocable authorization to execute work under a lease.

    A lease's ambient identity -- the ``ContextVar`` plus the bound thread set
    -- cannot survive a hand-off to an arbitrary worker thread running a
    freshly created event loop, which is exactly the shape the daemon's HTTP
    write gate uses. Widening the ambient rules until it did survive would
    authorize every thread that happens to inherit the context. A delegation
    instead carries ownership as a *value*: the holder mints one and hands it
    to the unit of work that will actually write, and only a holder of that
    object can adopt the lease.

    The sole-writer guarantee is preserved by three properties:

    * it can only be minted by a context that already passes
      :func:`require_write_lease`, so it is never an escalation;
    * at most one execution may adopt it at a time, so it cannot fan a single
      admission out into concurrent writers;
    * it is revoked when its lease is released, so a stashed delegation
      authorizes nothing afterwards.

    Revocation refuses *future* adoptions; it cannot withdraw an execution
    that already adopted the lease and may be inside a SQLite transaction.
    :meth:`retire` therefore reports whether such an execution is still in
    flight, and :attr:`settled` tells the admitting holder when it has really
    finished, so the holder can keep the single-writer gate until then
    instead of releasing it because a *caller* stopped waiting
    (polylogue-8r4zq).
    """

    actor: str
    lease: WriteLease
    _guard: threading.Lock = field(default_factory=threading.Lock)
    _adopted_by: int | None = None
    _revoked: bool = False
    _settled: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        # Never adopted is trivially settled; adoption clears it.
        self._settled.set()

    @property
    def live(self) -> bool:
        """Whether this delegation still authorizes an adoption."""
        with self._guard:
            return not self._revoked

    @property
    def adopted(self) -> bool:
        """Whether an execution currently holds this authorization."""
        with self._guard:
            return self._adopted_by is not None

    @property
    def settled(self) -> bool:
        """Whether no execution is currently running under this authorization."""
        return self._settled.is_set()

    def wait_for_settlement(self, timeout: float | None = None) -> bool:
        """Block until the adopted execution leaves, or ``timeout`` elapses."""
        return self._settled.wait(timeout)

    def revoke(self) -> None:
        """Retire this delegation; further adoptions are refused."""
        with self._guard:
            self._revoked = True

    def retire(self) -> bool:
        """Revoke, and report whether an adopted execution is still running.

        Atomic against :func:`adopt_write_lease`: a ``False`` answer means no
        execution can ever adopt this delegation again, so the holder may
        release its gate. ``True`` means one is in flight and the holder must
        wait for :attr:`settled` before another writer is admitted.
        """
        with self._guard:
            self._revoked = True
            return self._adopted_by is not None


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
    delegations: list[WriteLeaseDelegation] = field(default_factory=list)
    #: Guards ``bound_thread_ids``. On a free-threading build several
    #: inheriting threads can reach ``bind_write_lease_thread`` concurrently,
    #: and the previous check-then-assign on a bare ``set`` was unsynchronized
    #: (polylogue-1oa7o residual 4).
    _bind_guard: threading.Lock = field(default_factory=threading.Lock)

    def authorize_thread(self, thread_id: int) -> None:
        """Add ``thread_id`` to the authorized set under the lease's guard."""
        with self._bind_guard:
            if self.bound_thread_ids is None:
                self.bound_thread_ids = {self.owner_thread_id}
            self.bound_thread_ids.add(thread_id)

    def authorized_threads(self) -> frozenset[int]:
        with self._bind_guard:
            return frozenset(self.bound_thread_ids or {self.owner_thread_id})

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
        allowed_threads = lease.authorized_threads()
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


@dataclass(eq=False, slots=True)
class WriteLeaseThreadGrant:
    """One single-use authorization for *one* thread to join a held lease.

    polylogue-1oa7o residual 1: ``bind_write_lease_thread()`` used to read the
    ambient lease and add ``threading.get_ident()`` to it. On a free-threading
    build the ambient lease is inherited by *every* thread spawned during the
    hold, so that call was self-authorization -- any inheriting thread could
    bind itself into the daemon's live lease. Binding is now delegated by the
    owner rather than self-served: only a context that already passes
    :func:`require_write_lease` can mint a grant, and the spawned thread must
    present it.
    """

    lease: WriteLease
    _guard: threading.Lock = field(default_factory=threading.Lock)
    _used: bool = False

    def _claim(self) -> None:
        with self._guard:
            if self._used:
                raise UnleasedWriteError(
                    f"write lease thread grant for {self.lease.actor} was already used; one grant authorizes one thread"
                )
            self._used = True


def grant_write_lease_thread() -> WriteLeaseThreadGrant:
    """Mint a single-use authorization for one spawned thread to join this lease.

    Callable only from a context that already holds the lease -- the check runs
    through :func:`require_write_lease` -- so it can never manufacture
    authority the caller does not have. The owner mints this *before* starting
    the worker thread; the worker calls :func:`bind_write_lease_thread` with it.
    """
    lease = require_write_lease("granting a write lease thread binding")
    if lease is None:
        raise UnleasedWriteError(
            "cannot grant a write lease thread binding without holding the lease; "
            "mint the grant inside write_lease(...)"
        )
    return WriteLeaseThreadGrant(lease=lease)


def bind_write_lease_thread(grant: WriteLeaseThreadGrant) -> None:
    """Authorize the current thread for the lease ``grant`` was minted from.

    Refuses a grant whose lease is not the one this thread inherited: an
    inheriting thread must not be able to present a stale grant and join a
    different, later hold.
    """
    lease = _ACTIVE.get()
    if lease is None:
        raise UnleasedWriteError("cannot bind a thread without an active write lease")
    if grant.lease is not lease:
        raise UnleasedWriteError(
            f"write lease thread grant for {grant.lease.actor} does not authorize the lease "
            f"held by {lease.actor} in this thread"
        )
    grant._claim()
    lease.authorize_thread(threading.get_ident())


def delegate_write_lease() -> WriteLeaseDelegation:
    """Mint an explicit authorization for another execution unit to write.

    Callable only from a context that itself holds the lease: the ownership
    check runs through :func:`require_write_lease`, so delegation can never
    manufacture authority that the caller does not already have.
    """
    lease = require_write_lease("delegating the daemon write lease")
    if lease is None:
        raise UnleasedWriteError(
            "cannot delegate the write lease without holding it; mint the delegation "
            "inside write_lease(...) so the delegated work stays behind one admission"
        )
    delegation = WriteLeaseDelegation(actor=lease.actor, lease=lease)
    lease.delegations.append(delegation)
    return delegation


@contextmanager
def adopt_write_lease(delegation: WriteLeaseDelegation) -> Iterator[WriteLease]:
    """Execute this block under the lease the ``delegation`` authorizes.

    Binds the adopting task *and* thread, so the adopted view is authorized
    exactly where it is presented and nowhere else. The hold budget stays with
    the minting lease, which is the hold that is actually being measured; an
    adopted view never raises :class:`WriteHoldExceededError` of its own.
    """
    with delegation._guard:
        if delegation._revoked:
            raise UnleasedWriteError(
                f"write lease delegation for {delegation.actor} was revoked when its lease was released"
            )
        if delegation._adopted_by is not None:
            raise UnleasedWriteError(
                f"write lease delegation for {delegation.actor} is already executing on thread "
                f"{delegation._adopted_by}; one admission authorizes one writer at a time"
            )
        delegation._adopted_by = threading.get_ident()
        delegation._settled.clear()
    source = delegation.lease
    adopted = WriteLease(
        actor=source.actor,
        acquired_at=source.acquired_at,
        max_hold_seconds=None,
        archive_root=source.archive_root,
        coordinator=source.coordinator,
        owner_task_id=_current_task_id(),
        owner_thread_id=threading.get_ident(),
        bound_thread_ids={threading.get_ident()},
    )
    token = _ACTIVE.set(adopted)
    try:
        yield adopted
    finally:
        _ACTIVE.reset(token)
        with delegation._guard:
            delegation._adopted_by = None
            delegation._settled.set()


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
        # polylogue-1oa7o residual 2: the re-entrant branch used to hand back
        # the inherited lease with no ownership check, so a thread that
        # inherited it through free-threading contextvar propagation got the
        # parent's authority simply by asking for a nested lease. Re-run the
        # same identity check every write-mode open runs.
        require_write_lease(f"nested write lease for {actor}")
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
    except BaseException:
        # The hold budget never displaces the hold's own failure. Raising from
        # the ``finally`` below would replace an in-flight exception with this
        # timing complaint, demoting it to ``__context__`` where no ``except``
        # clause matches it. That is worst precisely under contention -- the
        # only condition that puts a hold over budget -- and it erases typed
        # partial-write facts: ``SessionProfileMarkerLoweringError`` carries
        # ``index_family_committed``, which ``_publication_commit_known``
        # recovers by ``isinstance``, so masking it reports a committed index
        # replacement with an unlowered marker as an ordinary failure and
        # drops the committed fact the operator needs. An over-budget hold is
        # still reported: the caller's own error is the stronger signal, and
        # the budget breach is logged rather than raised.
        if lease.over_budget:
            logger.warning(
                "writer %s held the lease %.3fs against a declared %.3fs budget "
                "while failing; reporting the hold's own error",
                actor,
                lease.held_seconds,
                lease.max_hold_seconds,
            )
        _ACTIVE.reset(token)
        # Release revokes, whether or not the hold succeeded: the delegation
        # contract is that a stashed delegation authorizes nothing once its
        # lease is gone, and a failing hold releases the lease just the same.
        for delegation in lease.delegations:
            delegation.revoke()
        raise
    else:
        _ACTIVE.reset(token)
        for delegation in lease.delegations:
            delegation.revoke()
        if lease.over_budget:
            raise WriteHoldExceededError(
                f"writer {actor} held the lease {lease.held_seconds:.3f}s against a declared "
                f"{lease.max_hold_seconds:.3f}s budget; every other writer waited behind it"
            )


# The connection-level half of this boundary. ``write_guard`` intercepts
# ``sqlite3.connect`` so a writable archive-tier open that never touched a
# declared factory is refused too; it is re-exported here because the lease and
# the guard are one authority and callers should not have to know which module
# holds which half.
from polylogue.storage.sqlite.write_guard import (  # noqa: E402
    declared_unguarded_write,
    install_archive_write_guard,
)
