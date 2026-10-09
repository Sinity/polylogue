"""Bounded daemon compute admission, class fairness, and cancellation.

The daemon has one SQLite publication owner, but reads and pure preparation are
allowed to run concurrently.  This module is the one process-local seam for
that work.

Admission is finite in work units and byte concurrency charge, and every unit
carries an admission class. Fixed-input work declares its actual input bytes;
dependency-discovering preparation explicitly reserves exclusive byte admission
and amends its accounted input bytes before hydration.  A class never takes the units or worker slots another
class has reserved, so interactive reads keep headroom while bulk work runs and
background work keeps a slot while interactive load saturates the rest.  A
cancelled read interrupts its SQLite connection, leaves its queue before it can
start, and returns its reservation exactly once.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import threading
from builtins import BaseExceptionGroup
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from functools import partial
from time import monotonic
from typing import Generic, Literal, TypeVar, cast

from polylogue.core.sql_settlement import (
    NativeSQLSettlementEvidence,
    SQLCustodyOwner,
    SQLSettlementRetry,
    capture_native_sql_owners,
    settle_native_sql,
)

AdmissionClass = Literal["interactive-read", "control", "incremental-background", "bulk-candidate"]
T = TypeVar("T")
InputT = TypeVar("InputT")

#: Dispatch order between classes that are both eligible.  Control publishes,
#: so it precedes reads; bulk candidate construction is the most deferrable.
ADMISSION_CLASSES: tuple[AdmissionClass, ...] = (
    "control",
    "interactive-read",
    "incremental-background",
    "bulk-candidate",
)

BACKGROUND_CLASSES: frozenset[str] = frozenset({"incremental-background", "bulk-candidate"})

# Incremental catch-up receives two turns per candidate turn. Advance only on
# dispatch so a continuously replenished incremental queue cannot starve bulk.
_BACKGROUND_TURNS = ("incremental-background", "incremental-background", "bulk-candidate")

#: Background work keeps at least this fraction of capacity reserved against
#: interactive and control pressure.  Both fairness bounds below are falsifiable
#: claims about the shipped scheduler, not aspirations.
MIN_BACKGROUND_THROUGHPUT_FRACTION = 0.2

#: A background unit that is admitted and runnable may not wait longer than
#: this before it starts.  The reserved slot bounds the wait by the remaining
#: runtime of the one task holding it, which every read route caps well below
#: this window.
MAX_BACKGROUND_STARVATION_S = 60.0

#: Reserved share per class.  The background entry is the aggregate reserve for
#: :data:`BACKGROUND_CLASSES`; the two background classes share it.
_CLASS_RESERVED_SHARE: Mapping[str, float] = {
    "control": 0.15,
    "interactive-read": 0.40,
    "background": MIN_BACKGROUND_THROUGHPUT_FRACTION,
}


class DaemonBackpressureError(RuntimeError):
    """The bounded compute queue cannot accept another unit of work."""

    code = "compute_backpressure"

    def __init__(
        self,
        message: str,
        *,
        admission_class: str | None = None,
        evidence: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.admission_class = admission_class
        self.evidence: dict[str, int] = dict(evidence or {})


class DaemonOperationCancelled(RuntimeError):  # noqa: N818 - public typed outcome name
    """A queued or running operation was cancelled before publication."""

    code = "operation_cancelled"


def _reserved_units(total: int, share: float) -> int:
    """Units a class holds against every other class.

    Reserves vanish on capacities too small to divide, which keeps a
    single-slot adapter usable at the cost of the fairness guarantee.
    """

    return max(0, min(total - 1, int(total * share)))


@dataclass(frozen=True, slots=True)
class ClassAdmissionSnapshot:
    """Per-class counters for status, benchmarks, and fairness assertions."""

    admission_class: str
    reserved_units: int
    ceiling_units: int
    reserved_slots: int
    ceiling_slots: int
    used_units: int
    queued_units: int
    active_units: int
    admitted: int
    dispatched: int
    completed: int
    rejected: int
    max_wait_s: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "admission_class": self.admission_class,
            "reserved_units": self.reserved_units,
            "ceiling_units": self.ceiling_units,
            "reserved_slots": self.reserved_slots,
            "ceiling_slots": self.ceiling_slots,
            "used_units": self.used_units,
            "queued_units": self.queued_units,
            "active_units": self.active_units,
            "admitted": self.admitted,
            "dispatched": self.dispatched,
            "completed": self.completed,
            "rejected": self.rejected,
            "max_wait_s": self.max_wait_s,
        }


@dataclass(frozen=True, slots=True)
class AdmissionSnapshot:
    """Operational concurrency charge and original input bytes accounted so far.

    ``active_input_bytes`` and ``queued_input_bytes`` are monotone input
    measurements while their reservations live, not forecasts of final input
    or parsed memory. An exclusive unit can begin before discovering any
    input, already holding its full ``used_bytes`` concurrency charge.
    """

    capacity_units: int
    used_units: int
    capacity_bytes: int
    used_bytes: int
    queued_units: int
    queued_bytes: int
    active_units: int
    rejected: int
    capacity_slots: int = 0
    classes: tuple[ClassAdmissionSnapshot, ...] = ()
    retained_sql_settlements: tuple[RetainedSQLSettlement, ...] = ()
    active_input_bytes: int = 0
    queued_input_bytes: int = 0
    exclusive_byte_units: int = 0

    @property
    def queued(self) -> int:
        return self.queued_units

    def by_class(self, admission_class: str) -> ClassAdmissionSnapshot:
        for entry in self.classes:
            if entry.admission_class == admission_class:
                return entry
        raise KeyError(admission_class)

    @property
    def background_max_wait_s(self) -> float:
        """Longest observed wait between admission and dispatch for background work."""

        waits = [entry.max_wait_s for entry in self.classes if entry.admission_class in BACKGROUND_CLASSES]
        return max(waits, default=0.0)

    @property
    def background_dispatched(self) -> int:
        return sum(entry.dispatched for entry in self.classes if entry.admission_class in BACKGROUND_CLASSES)

    def to_dict(self) -> dict[str, object]:
        return {
            "capacity_units": self.capacity_units,
            "used_units": self.used_units,
            "capacity_bytes": self.capacity_bytes,
            "used_bytes": self.used_bytes,
            "queued_units": self.queued_units,
            "queued_bytes": self.queued_bytes,
            "active_units": self.active_units,
            "active_input_bytes": self.active_input_bytes,
            "queued_input_bytes": self.queued_input_bytes,
            "exclusive_byte_units": self.exclusive_byte_units,
            "rejected": self.rejected,
            "capacity_slots": self.capacity_slots,
            "background_max_wait_s": self.background_max_wait_s,
            "classes": [entry.to_dict() for entry in self.classes],
            "retained_sql_settlements": [
                {
                    "thread_name": entry.thread_name,
                    "admission_class": entry.admission_class,
                    "owner_count": entry.owner_count,
                    "failure_types": list(entry.failure_types),
                }
                for entry in self.retained_sql_settlements
            ],
        }


class CancellationHandle:
    """One-shot cancellation handle shared by a request and its SQLite read."""

    def __init__(self) -> None:
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        self._connections: dict[int, object] = {}
        self._listeners: list[Callable[[], None]] = []

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def register_connection(self, connection: object) -> None:
        with self._lock:
            if self.cancelled:
                _interrupt(connection)
                return
            self._connections[id(connection)] = connection

    def unregister_connection(self, connection: object) -> None:
        with self._lock:
            self._connections.pop(id(connection), None)

    def add_listener(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Run ``listener`` when cancellation fires, or immediately if it already has."""

        with self._lock:
            already_cancelled = self.cancelled
            if not already_cancelled:
                self._listeners.append(listener)
        if already_cancelled:
            listener()

        def remove() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return remove

    def cancel(self) -> None:
        self._cancelled.set()
        with self._lock:
            connections = tuple(self._connections.values())
            listeners = tuple(self._listeners)
            self._listeners.clear()
        for connection in connections:
            _interrupt(connection)
        for listener in listeners:
            with contextlib.suppress(Exception):
                listener()


def _interrupt(connection: object) -> None:
    interrupt = getattr(connection, "interrupt", None)
    if callable(interrupt):
        with contextlib.suppress(Exception):
            interrupt()


_CURRENT_CANCELLATION: contextvars.ContextVar[CancellationHandle | None] = contextvars.ContextVar(
    "polylogue_current_daemon_cancellation", default=None
)

_CURRENT_COMPUTE = threading.local()


def current_cancellation() -> CancellationHandle | None:
    """Return the request cancellation handle in daemon compute code."""

    return getattr(_CURRENT_COMPUTE, "cancellation", _CURRENT_CANCELLATION.get())


@dataclass(frozen=True, slots=True)
class SubmittedOperation(Generic[T]):
    """One admitted operation with its result type retained for awaiters."""

    future: Future[T]
    cancellation: CancellationHandle
    _task: _Task | None = None

    async def wait(self) -> T:
        """Await the physical result, retaining cancellation through SQL settlement."""
        wrapped = asyncio.wrap_future(self.future)
        cancelled: asyncio.CancelledError | None = None
        while True:
            try:
                result = await asyncio.shield(wrapped)
            except asyncio.CancelledError as failure:
                if wrapped.cancelled():
                    raise cancelled or failure from None
                if cancelled is None:
                    cancelled = failure
                self.cancellation.cancel()
                self.retry_sql_settlement()
                continue
            except BaseException as failure:
                if cancelled is not None:
                    if isinstance(failure, DaemonOperationCancelled):
                        raise cancelled from failure
                    raise BaseExceptionGroup(
                        "read cancellation and physical settlement failed", [cancelled, failure]
                    ) from None
                raise
            if cancelled is not None:
                raise cancelled
            return result

    def retry_sql_settlement(self) -> None:
        """Request cleanup on this operation's physical creator worker."""
        if self._task is None:
            if not self.future.done():
                raise ValueError("pending operation has no physical cleanup owner")
            return
        self._task.sql_retry.request()

    @property
    def queue_delay_s(self) -> float:
        """Admission-to-dispatch wait for this unit; zero while still queued."""

        return 0.0 if self._task is None else self._task.queue_delay_s


class _ClassState:
    __slots__ = (
        "admission_class",
        "ceiling_slots",
        "ceiling_units",
        "reserved_slots",
        "reserved_units",
        "used_units",
        "active_units",
        "active_slots",
        "admitted",
        "dispatched",
        "completed",
        "rejected",
        "max_wait_s",
    )

    def __init__(self, admission_class: str) -> None:
        self.admission_class = admission_class
        self.reserved_units = 0
        self.ceiling_units = 0
        self.reserved_slots = 0
        self.ceiling_slots = 0
        self.used_units = 0
        self.active_units = 0
        self.active_slots = 0
        self.admitted = 0
        self.dispatched = 0
        self.completed = 0
        self.rejected = 0
        self.max_wait_s = 0.0


@dataclass(frozen=True, slots=True)
class RetainedSQLSettlement:
    """Physical compute work retained for original-worker SQL cleanup."""

    thread_name: str
    admission_class: str
    owner_count: int
    failure_types: tuple[str, ...]


class _ForwardingSettlementRetry(SQLSettlementRetry):
    """Deliver an adapter retry to a cleanup owner parked on its creator thread."""

    def __init__(self, forward: Callable[[], object]) -> None:
        super().__init__()
        self._forward = forward

    def request(self) -> None:
        super().request()
        self._forward()


def retain_current_creator_settlement(
    request: Callable[[], object], *, owner_count: int, failure_types: tuple[str, ...]
) -> Callable[[], None] | None:
    """Expose a cleanup owner parked on the current compute task to its adapter.

    A writer that keeps its admitted compute thread to settle failed SQL
    waits for settlement requests of its own. Without this registration the
    adapter's retry and shutdown cannot reach it, and joining the adapter waits
    on a thread that nothing will ever wake. Returns the release callback, or
    ``None`` outside a compute task.
    """
    adapter = getattr(_CURRENT_COMPUTE, "adapter", None)
    task = getattr(_CURRENT_COMPUTE, "task", None)
    if adapter is None or task is None:
        return None
    return cast("BoundedComputeAdapter", adapter)._retain_creator_settlement(
        task, request, owner_count=owner_count, failure_types=failure_types
    )


@dataclass(slots=True)
class _RetainedSQLSettlement:
    retry: SQLSettlementRetry
    evidence: RetainedSQLSettlement


def capture_compute_bridge() -> Callable[[], contextlib.AbstractContextManager[None]]:
    """Borrow a joined bridge's exact running reservation, never new capacity.

    The parent remains physically blocked on the bridge. Creator-thread SQL
    cleanup must finish inside the bridge Task before its loop or thread ends.
    """
    adapter = getattr(_CURRENT_COMPUTE, "adapter", None)
    task = getattr(_CURRENT_COMPUTE, "task", None)
    cancellation = current_cancellation()

    @contextlib.contextmanager
    def borrow() -> Iterator[None]:
        if adapter is None or task is None:
            yield
            return
        prior = {name: getattr(_CURRENT_COMPUTE, name, None) for name in ("adapter", "task", "cancellation")}
        token = _CURRENT_CANCELLATION.set(cancellation)
        _CURRENT_COMPUTE.adapter = adapter
        _CURRENT_COMPUTE.task = task
        _CURRENT_COMPUTE.cancellation = cancellation
        preserved = capture_native_sql_owners()
        primary: BaseException | None = None
        try:
            yield
        except BaseException as failure:
            primary = failure
            raise
        finally:
            try:
                cleanup_failure = adapter._settle_native_sql(task, preserved_native_owners=preserved)
                if cleanup_failure is not None:
                    if primary is not None:
                        primary.add_note(f"bridge native cleanup failed: {cleanup_failure!r}")
                    else:
                        raise cleanup_failure
            finally:
                for name, value in prior.items():
                    if value is None:
                        delattr(_CURRENT_COMPUTE, name)
                    else:
                        setattr(_CURRENT_COMPUTE, name, value)
                _CURRENT_CANCELLATION.reset(token)

    return borrow


class _Task:
    __slots__ = (
        "admission_class",
        "bytes",
        "input_demand_bytes",
        "exclusive_bytes",
        "cancellation",
        "context",
        "creator_thread",
        "function",
        "future",
        "queue_delay_s",
        "queued_at",
        "slots",
        "state",
        "sql_retry",
        "sql_observed_generation",
        "units",
    )

    def __init__(
        self,
        *,
        function: Callable[[], object],
        context: contextvars.Context,
        future: Future[object],
        cancellation: CancellationHandle,
        admission_class: str,
        units: int,
        estimated_bytes: int,
        input_demand_bytes: int,
        exclusive_bytes: bool,
        slots: int,
    ) -> None:
        self.function = function
        self.context = context
        self.creator_thread: threading.Thread | None = None
        self.future = future
        self.cancellation = cancellation
        self.admission_class = admission_class
        self.units = units
        self.bytes = estimated_bytes
        self.input_demand_bytes = input_demand_bytes
        self.exclusive_bytes = exclusive_bytes
        self.slots = slots
        self.queued_at = monotonic()
        self.queue_delay_s = 0.0
        self.state = "admitted"
        self.sql_retry = SQLSettlementRetry()
        self.sql_observed_generation = 0


class BoundedComputeAdapter:
    """The sole bounded in-process compute adapter used by daemon work.

    One queue reservation bounds accepted work; one dispatcher decides which
    accepted unit runs next.  Both are class-aware: a class may consume its own
    reserve plus the shared remainder, never another class's reserve.

    That invariant is about *every* other reserve at once, not the submitting
    group's own ceiling.  A per-group ceiling alone is true of each group taken
    separately and false of any two together: with eight workers the control
    ceiling is four slots and the background ceiling is four slots, so four
    blocking control tasks beside four blocking background tasks occupy the
    whole pool while each group stays inside its own bound -- and the three
    slots the snapshot reports as reserved for ``interactive-read`` are gone.
    Admission and dispatch therefore both check the *complement*: a unit may
    start only if what remains afterwards still covers every other group's
    unmet reserve.

    The reserve is hard, so a saturated mix leaves the reserved capacity idle
    when the reserved class has nothing queued. That is the same cost the
    per-group ceilings already pay for a single class (background alone has
    never been able to occupy more than its ceiling), extended to the combined
    case; a soft reserve would hand the slot away exactly when it is needed,
    which is at the moment an interactive request *arrives*.
    """

    def __init__(
        self,
        *,
        max_workers: int = 8,
        queue_units: int = 16,
        queue_bytes: int = 64 * 1024 * 1024,
        thread_name_prefix: str = "polylogue-compute",
    ) -> None:
        if max_workers < 1 or queue_units < 0 or queue_bytes < 0:
            raise ValueError("compute admission capacities are invalid")
        self.max_workers = max_workers
        self.capacity_units = max_workers + queue_units
        self.capacity_bytes = queue_bytes
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._used_units = 0
        self._used_bytes = 0
        self._active_units = 0
        self._active_bytes = 0
        self._active_input_bytes = 0
        self._used_input_bytes = 0
        self._exclusive_byte_units = 0
        self._active_slots = 0
        self._rejected = 0
        self._shutdown = False
        self._cancel_on_shutdown = False
        self._sql_settlements: dict[int, _RetainedSQLSettlement] = {}
        self._classes: dict[str, _ClassState] = {name: _ClassState(name) for name in ADMISSION_CLASSES}
        self._queues: dict[str, deque[_Task]] = {name: deque() for name in ADMISSION_CLASSES}
        self._background_turn = 0
        self._apply_reservations()
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=thread_name_prefix)

    # -- reservations ---------------------------------------------------

    def _apply_reservations(self) -> None:
        unit_reserve = self._reserve_table(self.capacity_units)
        slot_reserve = self._reserve_table(self.max_workers)
        self._reserved_units_by_group = dict(unit_reserve)
        self._reserved_slots_by_group = dict(slot_reserve)
        self._classes_by_group = {
            key: tuple(name for name in ADMISSION_CLASSES if self._class_group(name) == key) for key in unit_reserve
        }
        for name, state in self._classes.items():
            key = "background" if name in BACKGROUND_CLASSES else name
            state.reserved_units = unit_reserve[key]
            state.reserved_slots = slot_reserve[key]
            state.ceiling_units = self.capacity_units - self._other_reserves(unit_reserve, key)
            state.ceiling_slots = self.max_workers - self._other_reserves(slot_reserve, key)

    @staticmethod
    def _reserve_table(total: int) -> dict[str, int]:
        table = {key: _reserved_units(total, share) for key, share in _CLASS_RESERVED_SHARE.items()}
        while sum(table.values()) >= total and any(table.values()):
            largest = max(table, key=lambda key: table[key])
            table[largest] -= 1
        return table

    @staticmethod
    def _other_reserves(table: Mapping[str, int], key: str) -> int:
        return sum(value for name, value in table.items() if name != key)

    # -- admission ------------------------------------------------------

    def _evidence(self, state: _ClassState) -> dict[str, int]:
        group_used_units = self._group_used_units(state.admission_class)
        return {
            "capacity_units": self.capacity_units,
            "used_units": self._used_units,
            "queued_units": max(0, self._used_units - self._active_units),
            "capacity_bytes": self.capacity_bytes,
            "used_bytes": self._used_bytes,
            # Background classes share one reserve.  Report the aggregate
            # usage so a rejected sibling cannot make the bounded admission
            # envelope look larger than it is.
            "class_used_units": group_used_units,
            "class_ceiling_units": state.ceiling_units,
            # A refusal can come from the complement rather than from this
            # class's own ceiling: another group's reserve is still unmet, so
            # the shared remainder is smaller than the ceiling suggests. Name
            # that term instead of leaving the evidence looking contradictory.
            "other_reserved_units": self._unmet_other_unit_reserves(state.admission_class),
        }

    @staticmethod
    def _class_group(admission_class: str) -> str:
        return "background" if admission_class in BACKGROUND_CLASSES else admission_class

    def _group_used_units(self, admission_class: str) -> int:
        return sum(
            self._classes[name].used_units for name in self._classes_by_group[self._class_group(admission_class)]
        )

    def _group_active_slots(self, admission_class: str) -> int:
        return sum(
            self._classes[name].active_slots for name in self._classes_by_group[self._class_group(admission_class)]
        )

    def _unmet_other_unit_reserves(self, admission_class: str) -> int:
        """Queue units every *other* group's reserve still entitles it to.

        Zero once a group has already taken at least its reserve, and zero
        throughout on a capacity too small to divide (``_reserved_units``
        gives up the guarantee there rather than making a single-slot adapter
        unusable).
        """
        group = self._class_group(admission_class)
        return sum(
            max(0, reserved - sum(self._classes[name].used_units for name in self._classes_by_group[other]))
            for other, reserved in self._reserved_units_by_group.items()
            if other != group
        )

    def _unmet_other_slot_reserves(self, admission_class: str) -> int:
        """Worker slots every *other* group's reserve still entitles it to."""
        group = self._class_group(admission_class)
        return sum(
            max(0, reserved - sum(self._classes[name].active_slots for name in self._classes_by_group[other]))
            for other, reserved in self._reserved_slots_by_group.items()
            if other != group
        )

    def _acquire_locked(
        self, state: _ClassState, units: int, estimated_bytes: int, *, input_demand_bytes: int, exclusive_bytes: bool
    ) -> None:
        group_used_units = self._group_used_units(state.admission_class)
        other_reserved_units = self._unmet_other_unit_reserves(state.admission_class)
        if (
            self._used_units + units > self.capacity_units - other_reserved_units
            or self._used_bytes + estimated_bytes > self.capacity_bytes
            or group_used_units + units > state.ceiling_units
            or (exclusive_bytes and (self._used_input_bytes > 0 or self._exclusive_byte_units > 0))
            or (input_demand_bytes > 0 and self._exclusive_byte_units > 0)
        ):
            state.rejected += 1
            self._rejected += 1
            raise DaemonBackpressureError(
                "daemon compute admission is saturated; retry shortly",
                admission_class=state.admission_class,
                evidence=self._evidence(state),
            )
        self._used_units += units
        self._used_bytes += estimated_bytes
        self._exclusive_byte_units += int(exclusive_bytes)
        self._used_input_bytes += input_demand_bytes
        state.used_units += units
        state.admitted += 1

    def require_current_creator(self) -> None:
        """Require this reservation's actual creator for SQL-backed preparation."""
        task: _Task | None = getattr(_CURRENT_COMPUTE, "task", None)
        if (
            getattr(_CURRENT_COMPUTE, "adapter", None) is not self
            or task is None
            or task.state != "running"
            or task.creator_thread is not threading.current_thread()
        ):
            raise RuntimeError("preparation requires its admitted compute creator")
        from polylogue.core.compute_cancel import check_compute_cancelled

        check_compute_cancelled()

    def amend_current_input_demand(self, additional_bytes: int) -> None:
        """Record original inputs before hydration inside exclusive byte admission.

        The canonical input owner deduplicates its exact pinned coordinates.
        This changes measured demand, never the already-held concurrency charge.
        """
        if type(additional_bytes) is not int or additional_bytes < 0:
            raise ValueError("input demand amendment requires a nonnegative byte length")
        self.require_current_creator()
        task: _Task = _CURRENT_COMPUTE.task
        if not task.exclusive_bytes:
            raise RuntimeError("dependency-discovering preparation requires exclusive byte admission")
        with self._lock:
            task.input_demand_bytes += additional_bytes
            self._active_input_bytes += additional_bytes
            self._used_input_bytes += additional_bytes

    def submit(
        self,
        function: Callable[[], T],
        *,
        admission_class: AdmissionClass = "interactive-read",
        units: int = 1,
        estimated_bytes: int = 0,
        exclusive_bytes: bool = False,
        cancellation: CancellationHandle | None = None,
    ) -> SubmittedOperation[T]:
        if admission_class not in self._classes:
            raise ValueError(f"unknown daemon admission class: {admission_class!r}")
        if units < 1 or estimated_bytes < 0:
            raise DaemonBackpressureError(
                "operation declares an invalid compute admission demand",
                admission_class=admission_class,
                evidence={"units": units, "estimated_bytes": estimated_bytes},
            )
        input_demand_bytes = estimated_bytes
        # A unit larger than the whole byte envelope reserves all of it, and
        # so runs with no other byte-holding work, instead of being refused:
        # the envelope bounds concurrency, not the size of an input the daemon
        # must eventually process.
        estimated_bytes = self.capacity_bytes if exclusive_bytes else min(estimated_bytes, self.capacity_bytes)
        handle = cancellation or CancellationHandle()
        future: Future[T] = Future()
        if getattr(_CURRENT_COMPUTE, "adapter", None) is self:
            # This pure subunit belongs to the running parent's reservation.
            # Queueing and waiting here would deadlock a one-worker adapter.
            # The caller includes its subunits in the parent's byte estimate;
            # no extra worker or reservation is created. Scratch opened by
            # the subunit settles before its synchronous completion.
            parent = _CURRENT_COMPUTE.cancellation
            parent_task = _CURRENT_COMPUTE.task
            if exclusive_bytes and not parent_task.exclusive_bytes:
                raise RuntimeError("nested preparation cannot acquire exclusive bytes inside ordinary admission")

            def run_nested() -> None:
                preserved = capture_native_sql_owners()
                result: T | None = None
                failure: BaseException | None = None
                try:
                    if handle.cancelled or (parent is not None and parent.cancelled):
                        raise DaemonOperationCancelled("operation cancelled before nested compute started")
                    result = function()
                except BaseException as exc:
                    failure = exc
                finally:
                    settlement_failure = self._settle_native_sql(parent_task, preserved_native_owners=preserved)
                    if settlement_failure is not None:
                        if failure is None:
                            failure = settlement_failure
                        else:
                            failure.add_note(f"native SQL cleanup also failed: {type(settlement_failure).__name__}")
                if failure is not None:
                    future.set_exception(failure)
                else:
                    future.set_result(cast("T", result))

            # Preserve submit's context isolation while reusing the physical
            # worker and reservation. Native cleanup runs in that same context.
            contextvars.copy_context().run(run_nested)
            return SubmittedOperation(future=future, cancellation=parent or handle, _task=parent_task)
        future.add_done_callback(lambda done: handle.cancel() if done.cancelled() else None)
        context = contextvars.copy_context()
        task = _Task(
            function=function,
            context=context,
            # The scheduler queue is heterogeneous, while this public handle
            # retains the concrete result type supplied by its caller.
            future=cast("Future[object]", future),
            cancellation=handle,
            admission_class=admission_class,
            units=units,
            estimated_bytes=estimated_bytes,
            input_demand_bytes=input_demand_bytes,
            exclusive_bytes=exclusive_bytes,
            slots=min(units, self.max_workers),
        )

        with self._lock:
            if self._shutdown:
                raise DaemonBackpressureError(
                    "daemon compute adapter is shutting down",
                    admission_class=admission_class,
                    evidence={"capacity_units": self.capacity_units},
                )
            state = self._classes[admission_class]
            if min(units, self.max_workers) > state.ceiling_slots:
                raise DaemonBackpressureError(
                    "operation exceeds its admission class slot ceiling",
                    admission_class=admission_class,
                    evidence={
                        "class_ceiling_slots": state.ceiling_slots,
                        "requested_slots": min(units, self.max_workers),
                    },
                )
            self._acquire_locked(
                state, units, estimated_bytes, input_demand_bytes=input_demand_bytes, exclusive_bytes=exclusive_bytes
            )
            self._queues[admission_class].append(task)
            runnable = self._drain_locked()

        def cancel_task() -> None:
            task.sql_retry.request()
            self._cancel_before_start(task)

        handle.add_listener(cancel_task)
        self._run_all(runnable)
        return SubmittedOperation(future=future, cancellation=handle, _task=task)

    def map(
        self,
        function: Callable[[InputT], T],
        items: Iterable[InputT],
        *,
        admission_class: AdmissionClass = "incremental-background",
        estimated_bytes: Callable[[InputT], int] = lambda _item: 0,
        discard_unconsumed: Callable[[T], None] | None = None,
        exclusive_bytes: bool = False,
        checkpoint: Callable[[], None] | None = None,
    ) -> Generator[T, None, None]:
        """Run pure units in input order through this adapter's admission.

        The window owns only its submitted operations, never the shared pool.
        A failure or consumer cancellation drains those operations before the
        caller can discard their scratch. Nested maps execute synchronously
        under the parent's exact reservation. ``checkpoint`` observes caller
        cancellation while waiting; its failure cancels and physically drains
        this window before the caller can release its owned state.
        """
        if admission_class not in self._classes:
            raise ValueError(f"unknown compute admission class: {admission_class!r}")
        pending: deque[SubmittedOperation[T]] = deque()
        iterator = iter(items)
        waiting_item: tuple[InputT] | None = None
        window = (
            1 if getattr(_CURRENT_COMPUTE, "adapter", None) is self else self._classes[admission_class].ceiling_slots
        )
        try:
            exhausted = False
            while pending or not exhausted:
                if checkpoint is not None:
                    checkpoint()
                while not exhausted and len(pending) < window:
                    if waiting_item is None:
                        try:
                            item = next(iterator)
                        except StopIteration:
                            exhausted = True
                            break
                    else:
                        item = waiting_item[0]
                    try:
                        operation = self.submit(
                            partial(function, item),
                            admission_class=admission_class,
                            estimated_bytes=estimated_bytes(item),
                            exclusive_bytes=exclusive_bytes,
                        )
                    except DaemonBackpressureError:
                        if not pending:
                            raise
                        # Our accepted work can release this capacity. Do not
                        # drop the input or abandon and retry the same prefix.
                        waiting_item = (item,)
                        break
                    waiting_item = None
                    pending.append(operation)
                if pending:
                    if checkpoint is not None:
                        while not pending[0].future.done():
                            # A scheduling checkpoint, never a work deadline.
                            wait((pending[0].future,), timeout=0.05)
                            checkpoint()
                        checkpoint()
                    result = pending[0].future.result()
                    pending.popleft()
                    yield result
        finally:
            for operation in pending:
                if not operation.future.done():
                    operation.cancellation.cancel()
            cleanup_failures: list[BaseException] = []
            for operation in pending:
                try:
                    unconsumed_result = operation.future.result()
                except BaseException:
                    continue
                else:
                    if discard_unconsumed is not None:
                        try:
                            discard_unconsumed(unconsumed_result)
                        except BaseException as exc:
                            cleanup_failures.append(exc)
            if cleanup_failures:
                raise BaseExceptionGroup("compute result cleanup failed", cleanup_failures)

    # -- dispatch -------------------------------------------------------

    def _select_locked(self) -> _Task | None:
        """Return the next runnable task, or None while every class is capped.

        A class is eligible only for slots outside every other class's reserve,
        which is what makes the background starvation window finite under
        sustained interactive load. "Every other class" is the complement over
        all groups at once: the per-group ceiling alone lets two saturated
        groups jointly occupy the pool while each stays inside its own bound.
        """

        background_order = [
            (self._background_turn + offset) % len(_BACKGROUND_TURNS) for offset in range(len(_BACKGROUND_TURNS))
        ]
        candidates: list[tuple[str, int | None]] = [
            (name, None) for name in ADMISSION_CLASSES if name not in BACKGROUND_CLASSES
        ]
        candidates.extend((_BACKGROUND_TURNS[turn], turn) for turn in background_order)
        for name, background_turn in candidates:
            queue = self._queues[name]
            if not queue:
                continue
            task = queue[0]
            state = self._classes[name]
            if self._active_slots + task.slots > self.max_workers - self._unmet_other_slot_reserves(name):
                # A background head waits for capacity rather than being
                # overtaken: a later one-slot background turn fits the same
                # remainder every time and would starve a multi-slot head.
                if background_turn is not None:
                    return None
                continue
            if self._group_active_slots(name) + task.slots > state.ceiling_slots:
                # Do not backfill a background slot with a later turn while
                # the selected class head is runnable except for its own
                # multi-slot footprint. Its current occupants will release
                # together, allowing this head to make progress.
                if background_turn is not None:
                    return None
                continue
            queue.popleft()
            if background_turn is not None:
                self._background_turn = (background_turn + 1) % len(_BACKGROUND_TURNS)
            task.state = "running"
            task.queue_delay_s = monotonic() - task.queued_at
            self._active_units += task.units
            self._active_bytes += task.bytes
            self._active_input_bytes += task.input_demand_bytes
            self._active_slots += task.slots
            state.active_units += task.units
            state.active_slots += task.slots
            state.dispatched += 1
            state.max_wait_s = max(state.max_wait_s, task.queue_delay_s)
            return task
        return None

    def _drain_locked(self) -> list[_Task]:
        """Dispatch every task the current capacity admits, not only the first.

        One completion can free several slots when a unit occupies more than
        one; stopping after one dispatch would leave the rest idle until the
        next unrelated event.
        """

        if self._shutdown and self._cancel_on_shutdown:
            return []
        runnable: list[_Task] = []
        while (task := self._select_locked()) is not None:
            runnable.append(task)
        return runnable

    def _run_all(self, tasks: list[_Task]) -> None:
        for task in tasks:
            self._run_task(task)

    def _run_task(self, task: _Task | None) -> None:
        if task is None:
            return
        if not task.future.set_running_or_notify_cancel():
            self._release(task, active=True)
            return

        def run() -> None:
            task.creator_thread = threading.current_thread()
            task.sql_observed_generation = task.sql_retry.generation()
            token = _CURRENT_CANCELLATION.set(task.cancellation)
            _CURRENT_COMPUTE.adapter = self
            _CURRENT_COMPUTE.cancellation = task.cancellation
            _CURRENT_COMPUTE.task = task
            result: object | None = None
            failure: BaseException | None = None
            try:
                if task.cancellation.cancelled:
                    failure = DaemonOperationCancelled("operation cancelled before compute started")
                else:
                    result = task.function()
            except BaseException as exc:
                failure = exc
            finally:
                settlement_failure = self._settle_native_sql(task)
                if settlement_failure is not None:
                    if failure is None:
                        failure = settlement_failure
                    else:
                        failure.add_note(f"native SQL cleanup also failed: {type(settlement_failure).__name__}")
                del _CURRENT_COMPUTE.adapter
                del _CURRENT_COMPUTE.cancellation
                del _CURRENT_COMPUTE.task
                _CURRENT_CANCELLATION.reset(token)
                task.creator_thread = None
                self._release(task, active=True)
            if failure is not None:
                task.future.set_exception(failure)
            else:
                task.future.set_result(result)

        def settle_executor_cancellation(execution: Future[None]) -> None:
            if execution.cancelled():
                self._release(task, active=True)
                task.future.set_exception(DaemonOperationCancelled("daemon compute adapter shut down"))

        try:
            execution = self.executor.submit(task.context.run, run)
        except BaseException as submission_failure:
            self._release(task, active=True)
            task.future.set_exception(
                DaemonOperationCancelled("daemon compute adapter shut down") if self._shutdown else submission_failure
            )
        else:
            execution.add_done_callback(settle_executor_cancellation)

    def _settle_native_sql(
        self,
        task: _Task,
        *,
        preserved_native_owners: tuple[SQLCustodyOwner, ...] = (),
    ) -> BaseException | None:
        """Keep the physical future and worker owned until native SQL settles.

        Only an explicit retry or shutdown wakes a failed cleanup. Cancellation
        forbids new work but cannot surrender its creator-thread cleanup owner.
        """
        retry = task.sql_retry
        retained: _RetainedSQLSettlement | None = None

        def on_pending(pending: NativeSQLSettlementEvidence) -> None:
            nonlocal retained
            evidence = RetainedSQLSettlement(
                thread_name=threading.current_thread().name,
                admission_class=task.admission_class,
                owner_count=pending.owner_count,
                failure_types=pending.failure_types,
            )
            with self._lock:
                if retained is None:
                    retained = _RetainedSQLSettlement(retry, evidence)
                    self._sql_settlements[id(task)] = retained
                    shutting_down = self._shutdown
                else:
                    retained.evidence = evidence
                    shutting_down = False
            if shutting_down:
                retry.request()

        def on_settled() -> None:
            task.sql_observed_generation = retry.generation()
            with self._lock:
                self._sql_settlements.pop(id(task), None)

        return settle_native_sql(
            retry=retry,
            on_pending=on_pending,
            on_settled=on_settled,
            preserved_native_owners=preserved_native_owners,
            initial_observed_generation=task.sql_observed_generation,
        )

    def _retain_creator_settlement(
        self,
        task: _Task,
        request: Callable[[], object],
        *,
        owner_count: int,
        failure_types: tuple[str, ...],
    ) -> Callable[[], None]:
        """Register a cleanup owner the running task keeps on its creator thread."""
        entry = _RetainedSQLSettlement(
            _ForwardingSettlementRetry(request),
            RetainedSQLSettlement(
                thread_name=threading.current_thread().name,
                admission_class=task.admission_class,
                owner_count=owner_count,
                failure_types=failure_types,
            ),
        )
        with self._lock:
            self._sql_settlements[id(entry)] = entry
            shutting_down = self._shutdown
        if shutting_down:
            entry.retry.request()

        def release() -> None:
            with self._lock:
                self._sql_settlements.pop(id(entry), None)

        return release

    def retained_sql_settlements(self) -> tuple[RetainedSQLSettlement, ...]:
        """Return unresolved physical ownership without touching native handles."""
        with self._lock:
            return tuple(entry.evidence for entry in self._sql_settlements.values())

    def retry_sql_settlement(self) -> None:
        """Ask each retained creator worker to retry its own cleanup once."""
        with self._lock:
            retained = tuple(self._sql_settlements.values())
        for entry in retained:
            entry.retry.request()

    def _release(self, task: _Task, *, active: bool) -> None:
        """Return one task's reservation exactly once and pump the queues."""

        with self._lock:
            if task.state == "done":
                return
            task.state = "done"
            state = self._classes[task.admission_class]
            # ``completed`` is the denominator for work that actually reached
            # dispatch.  A queued cancellation releases an admission
            # reservation, but it never consumed a worker and therefore must
            # not make completion exceed dispatch in the scheduler evidence.
            if active:
                state.completed += 1
            self._used_units -= task.units
            self._used_bytes -= task.bytes
            self._exclusive_byte_units -= int(task.exclusive_bytes)
            self._used_input_bytes -= task.input_demand_bytes
            state.used_units -= task.units
            if active:
                self._active_units -= task.units
                self._active_bytes -= task.bytes
                self._active_input_bytes -= task.input_demand_bytes
                self._active_slots -= task.slots
                state.active_units -= task.units
                state.active_slots -= task.slots
            runnable = self._drain_locked()
            idle = self._used_units == 0
            if idle:
                self._idle.notify_all()
            finish_graceful_shutdown = idle and self._shutdown and not self._cancel_on_shutdown
        self._run_all(runnable)
        if finish_graceful_shutdown:
            self.executor.shutdown(wait=False, cancel_futures=False)

    def _cancel_before_start(self, task: _Task) -> None:
        """Drop an admitted-but-unstarted task and return its reservation."""

        with self._lock:
            if task.state != "admitted":
                return
            with contextlib.suppress(ValueError):
                self._queues[task.admission_class].remove(task)
        self._release(task, active=False)
        with contextlib.suppress(Exception):
            task.future.set_exception(DaemonOperationCancelled("operation cancelled before compute started"))

    # -- observation ----------------------------------------------------

    def snapshot(self) -> AdmissionSnapshot:
        with self._lock:
            # ``max_wait_s`` is a current starvation signal, not only a
            # historical dispatch metric.  A queued background task has not
            # reached ``_select_locked`` yet, so its delay must be accounted
            # for here or status can report zero while that task is already
            # beyond the declared starvation bound.
            observed_at = monotonic()
            classes = tuple(
                ClassAdmissionSnapshot(
                    admission_class=state.admission_class,
                    reserved_units=state.reserved_units,
                    ceiling_units=state.ceiling_units,
                    reserved_slots=state.reserved_slots,
                    ceiling_slots=state.ceiling_slots,
                    used_units=state.used_units,
                    queued_units=max(0, state.used_units - state.active_units),
                    active_units=state.active_units,
                    admitted=state.admitted,
                    dispatched=state.dispatched,
                    completed=state.completed,
                    rejected=state.rejected,
                    max_wait_s=max(
                        state.max_wait_s,
                        max((observed_at - task.queued_at for task in self._queues[name]), default=0.0),
                    ),
                )
                for name, state in ((name, self._classes[name]) for name in ADMISSION_CLASSES)
            )
            return AdmissionSnapshot(
                capacity_units=self.capacity_units,
                used_units=self._used_units,
                capacity_bytes=self.capacity_bytes,
                used_bytes=self._used_bytes,
                queued_units=max(0, self._used_units - self._active_units),
                queued_bytes=max(0, self._used_bytes - self._active_bytes),
                active_units=self._active_units,
                rejected=self._rejected,
                capacity_slots=self.max_workers,
                classes=classes,
                retained_sql_settlements=tuple(entry.evidence for entry in self._sql_settlements.values()),
                active_input_bytes=self._active_input_bytes,
                queued_input_bytes=sum(task.input_demand_bytes for queue in self._queues.values() for task in queue),
                exclusive_byte_units=self._exclusive_byte_units,
            )

    def shutdown(self, *, wait: bool = False, cancel_futures: bool = True) -> None:
        """Close admission and settle every accepted operation.

        Cancelling shutdown stops dispatch and settles both scheduler and
        executor queues. Graceful shutdown keeps the executor open until the
        already-admitted work drains; the last release closes it even when
        the caller does not wait.
        """
        with self._lock:
            self._shutdown = True
            self._cancel_on_shutdown = self._cancel_on_shutdown or cancel_futures
            queued: list[_Task] = []
            if self._cancel_on_shutdown:
                for queue in self._queues.values():
                    while queue:
                        queued.append(queue.popleft())
        for task in queued:
            self._release(task, active=False)
            with contextlib.suppress(Exception):
                task.future.set_exception(DaemonOperationCancelled("daemon compute adapter shut down"))
        self.retry_sql_settlement()
        with self._idle:
            if not self._cancel_on_shutdown:
                if wait:
                    self._idle.wait_for(lambda: self._used_units == 0 or self._cancel_on_shutdown)
                elif self._used_units:
                    return
            cancel_executor_futures = self._cancel_on_shutdown
        self.executor.shutdown(wait=wait, cancel_futures=cancel_executor_futures)

    def close(self, *, join_timeout_s: float) -> tuple[str, ...]:
        """Shut down and join the worker threads within one shared deadline.

        Cancelling the future that awaited a worker does not stop the worker,
        so an owner that needs its threads gone has to join them. Returns the
        names of workers still alive when the deadline expires: a running
        job cannot be interrupted, only named.
        """
        self.shutdown(wait=False, cancel_futures=True)
        deadline = monotonic() + join_timeout_s
        workers = tuple(getattr(self.executor, "_threads", ()))
        for worker in workers:
            worker.join(max(0.0, deadline - monotonic()))
        return tuple(worker.name for worker in workers if worker.is_alive())


#: The one compute capacity daemon-internal lease-free work is admitted
#: through. The HTTP/UDS servers each publish the adapter they already own, so
#: background derivation shares a published pool instead of standing up a
#: second one; a process that publishes none (a test, a daemon started without
#: the API) gets one small shared adapter rather than a pool per caller.
_SHARED_COMPUTE_ADAPTER: BoundedComputeAdapter | None = None
_SHARED_COMPUTE_LOCK = threading.Lock()


def publish_compute_adapter(adapter: BoundedComputeAdapter) -> None:
    """Declare the already-owned adapter as this process's shared capacity."""
    global _SHARED_COMPUTE_ADAPTER
    with _SHARED_COMPUTE_LOCK:
        if _SHARED_COMPUTE_ADAPTER is not None and _SHARED_COMPUTE_ADAPTER is not adapter:
            raise RuntimeError("reset and physically settle the current compute owner before replacement")
        _SHARED_COMPUTE_ADAPTER = adapter


def compute_adapter() -> BoundedComputeAdapter:
    """Return the shared compute capacity, creating the fallback exactly once."""
    global _SHARED_COMPUTE_ADAPTER
    current = getattr(_CURRENT_COMPUTE, "adapter", None)
    if current is not None:
        if not isinstance(current, BoundedComputeAdapter):
            raise TypeError("current compute owner is not a bounded adapter")
        return current
    with _SHARED_COMPUTE_LOCK:
        if _SHARED_COMPUTE_ADAPTER is None:
            _SHARED_COMPUTE_ADAPTER = BoundedComputeAdapter(
                max_workers=2,
                queue_units=8,
                thread_name_prefix="polylogue-derive",
            )
        return _SHARED_COMPUTE_ADAPTER


def reset_compute_adapter(*, join_timeout_s: float = 0.0) -> tuple[str, ...]:
    """Retire the shared owner only after its physical workers settle.

    Surviving workers remain owned by the published adapter; another reset
    can retry settlement. Returns their names after *join_timeout_s*.
    """
    global _SHARED_COMPUTE_ADAPTER
    with _SHARED_COMPUTE_LOCK:
        adapter = _SHARED_COMPUTE_ADAPTER
    if adapter is None:
        return ()
    if not isinstance(adapter, BoundedComputeAdapter):
        with _SHARED_COMPUTE_LOCK:
            if _SHARED_COMPUTE_ADAPTER is adapter:
                _SHARED_COMPUTE_ADAPTER = None
        return ()
    surviving = adapter.close(join_timeout_s=join_timeout_s)
    if not surviving:
        with _SHARED_COMPUTE_LOCK:
            if _SHARED_COMPUTE_ADAPTER is adapter:
                _SHARED_COMPUTE_ADAPTER = None
    return surviving


__all__ = [
    "ADMISSION_CLASSES",
    "BACKGROUND_CLASSES",
    "MAX_BACKGROUND_STARVATION_S",
    "MIN_BACKGROUND_THROUGHPUT_FRACTION",
    "AdmissionClass",
    "AdmissionSnapshot",
    "BoundedComputeAdapter",
    "CancellationHandle",
    "ClassAdmissionSnapshot",
    "DaemonBackpressureError",
    "DaemonOperationCancelled",
    "SubmittedOperation",
    "compute_adapter",
    "publish_compute_adapter",
    "reset_compute_adapter",
]


def compute_window_length(record_count: int, requested: int | None = None) -> int:
    """Bound a caller's outstanding units by the shared background admission.

    This is a submission window, not another executor or reservation. Actual
    weighted admission remains the shared adapter's decision for every unit.
    """
    capacity = compute_adapter().snapshot().by_class("incremental-background").ceiling_units
    return max(1, min(max(1, record_count), capacity, capacity if requested is None else max(1, requested)))
