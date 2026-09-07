"""Bounded daemon compute admission, class fairness, and cancellation.

The daemon has one SQLite publication owner, but reads and pure preparation are
allowed to run concurrently.  This module is the one process-local seam for
that work.

Admission is finite in work units and estimated bytes, and every unit carries
an admission class.  A class never takes the units or worker slots another
class has reserved, so interactive reads keep headroom while bulk work runs and
background work keeps a slot while interactive load saturates the rest.  A
cancelled read interrupts its SQLite connection, leaves its queue before it can
start, and returns its reservation exactly once.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import monotonic
from typing import Literal, TypeVar

AdmissionClass = Literal["interactive-read", "control", "incremental-background", "bulk-candidate"]
T = TypeVar("T")

#: Dispatch order between classes that are both eligible.  Control publishes,
#: so it precedes reads; bulk candidate construction is the most deferrable.
ADMISSION_CLASSES: tuple[AdmissionClass, ...] = (
    "control",
    "interactive-read",
    "incremental-background",
    "bulk-candidate",
)

BACKGROUND_CLASSES: frozenset[str] = frozenset({"incremental-background", "bulk-candidate"})

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
    """Safe operational counters for status and benchmark attribution."""

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
            "rejected": self.rejected,
            "capacity_slots": self.capacity_slots,
            "background_max_wait_s": self.background_max_wait_s,
            "classes": [entry.to_dict() for entry in self.classes],
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

    def add_listener(self, listener: Callable[[], None]) -> None:
        """Run ``listener`` when cancellation fires, or immediately if it already has."""

        with self._lock:
            if not self.cancelled:
                self._listeners.append(listener)
                return
        listener()

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


def current_cancellation() -> CancellationHandle | None:
    """Return the request cancellation handle in daemon compute code."""

    return _CURRENT_CANCELLATION.get()


@dataclass(frozen=True, slots=True)
class SubmittedOperation:
    future: Future[object]
    cancellation: CancellationHandle
    _task: _Task | None = None

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


class _Task:
    __slots__ = (
        "admission_class",
        "bytes",
        "cancellation",
        "function",
        "future",
        "queue_delay_s",
        "queued_at",
        "slots",
        "state",
        "units",
    )

    def __init__(
        self,
        *,
        function: Callable[[], object],
        future: Future[object],
        cancellation: CancellationHandle,
        admission_class: str,
        units: int,
        estimated_bytes: int,
        slots: int,
    ) -> None:
        self.function = function
        self.future = future
        self.cancellation = cancellation
        self.admission_class = admission_class
        self.units = units
        self.bytes = estimated_bytes
        self.slots = slots
        self.queued_at = monotonic()
        self.queue_delay_s = 0.0
        self.state = "admitted"


class BoundedComputeAdapter:
    """The sole bounded in-process compute adapter used by daemon work.

    One queue reservation bounds accepted work; one dispatcher decides which
    accepted unit runs next.  Both are class-aware: a class may consume its own
    reserve plus the shared remainder, never another class's reserve.
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
        self._used_units = 0
        self._used_bytes = 0
        self._active_units = 0
        self._active_bytes = 0
        self._active_slots = 0
        self._rejected = 0
        self._shutdown = False
        self._classes: dict[str, _ClassState] = {name: _ClassState(name) for name in ADMISSION_CLASSES}
        self._queues: dict[str, deque[_Task]] = {name: deque() for name in ADMISSION_CLASSES}
        self._apply_reservations()
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=thread_name_prefix)

    # -- reservations ---------------------------------------------------

    def _apply_reservations(self) -> None:
        unit_reserve = self._reserve_table(self.capacity_units)
        slot_reserve = self._reserve_table(self.max_workers)
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
        return {
            "capacity_units": self.capacity_units,
            "used_units": self._used_units,
            "queued_units": max(0, self._used_units - self._active_units),
            "capacity_bytes": self.capacity_bytes,
            "used_bytes": self._used_bytes,
            "class_used_units": state.used_units,
            "class_ceiling_units": state.ceiling_units,
        }

    def _acquire_locked(self, state: _ClassState, units: int, estimated_bytes: int) -> None:
        if (
            self._used_units + units > self.capacity_units
            or self._used_bytes + estimated_bytes > self.capacity_bytes
            or state.used_units + units > state.ceiling_units
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
        state.used_units += units
        state.admitted += 1

    def submit(
        self,
        function: Callable[[], T],
        *,
        admission_class: AdmissionClass = "interactive-read",
        units: int = 1,
        estimated_bytes: int = 0,
        cancellation: CancellationHandle | None = None,
    ) -> SubmittedOperation:
        if admission_class not in self._classes:
            raise ValueError(f"unknown daemon admission class: {admission_class!r}")
        if units < 1 or estimated_bytes < 0 or estimated_bytes > self.capacity_bytes:
            raise DaemonBackpressureError(
                "operation exceeds the daemon compute admission envelope",
                admission_class=admission_class,
                evidence={"capacity_bytes": self.capacity_bytes, "estimated_bytes": max(0, estimated_bytes)},
            )
        handle = cancellation or CancellationHandle()
        future: Future[object] = Future()
        task = _Task(
            function=function,
            future=future,
            cancellation=handle,
            admission_class=admission_class,
            units=units,
            estimated_bytes=estimated_bytes,
            slots=min(units, self.max_workers),
        )

        with self._lock:
            if self._shutdown:
                raise DaemonBackpressureError(
                    "daemon compute adapter is shutting down",
                    admission_class=admission_class,
                    evidence={"capacity_units": self.capacity_units},
                )
            self._acquire_locked(self._classes[admission_class], units, estimated_bytes)
            self._queues[admission_class].append(task)
            runnable = self._drain_locked()

        handle.add_listener(lambda: self._cancel_before_start(task))
        self._run_all(runnable)
        return SubmittedOperation(future=future, cancellation=handle, _task=task)

    # -- dispatch -------------------------------------------------------

    def _select_locked(self) -> _Task | None:
        """Return the next runnable task, or None while every class is capped.

        A class is eligible only for slots outside every other class's reserve,
        which is what makes the background starvation window finite under
        sustained interactive load.
        """

        for name in ADMISSION_CLASSES:
            queue = self._queues[name]
            if not queue:
                continue
            task = queue[0]
            state = self._classes[name]
            if self._active_slots + task.slots > self.max_workers:
                continue
            if state.active_slots + task.slots > state.ceiling_slots:
                continue
            queue.popleft()
            task.state = "running"
            task.queue_delay_s = monotonic() - task.queued_at
            self._active_units += task.units
            self._active_bytes += task.bytes
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
            token = _CURRENT_CANCELLATION.set(task.cancellation)
            try:
                if task.cancellation.cancelled:
                    task.future.set_exception(DaemonOperationCancelled("operation cancelled before compute started"))
                else:
                    task.future.set_result(task.function())
            except BaseException as exc:
                task.future.set_exception(exc)
            finally:
                _CURRENT_CANCELLATION.reset(token)
                self._release(task, active=True)

        try:
            self.executor.submit(run)
        except BaseException:
            self._release(task, active=True)
            raise

    def _release(self, task: _Task, *, active: bool) -> None:
        """Return one task's reservation exactly once and pump the queues."""

        with self._lock:
            if task.state == "done":
                return
            task.state = "done"
            state = self._classes[task.admission_class]
            state.completed += 1
            self._used_units -= task.units
            self._used_bytes -= task.bytes
            state.used_units -= task.units
            if active:
                self._active_units -= task.units
                self._active_bytes -= task.bytes
                self._active_slots -= task.slots
                state.active_units -= task.units
                state.active_slots -= task.slots
            runnable = self._drain_locked()
        self._run_all(runnable)

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
                    max_wait_s=state.max_wait_s,
                )
                for state in (self._classes[name] for name in ADMISSION_CLASSES)
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
            )

    def shutdown(self, *, wait: bool = False, cancel_futures: bool = True) -> None:
        with self._lock:
            self._shutdown = True
            queued: list[_Task] = []
            if cancel_futures:
                for queue in self._queues.values():
                    while queue:
                        queued.append(queue.popleft())
        for task in queued:
            self._release(task, active=False)
            with contextlib.suppress(Exception):
                task.future.set_exception(DaemonOperationCancelled("daemon compute adapter shut down"))
        self.executor.shutdown(wait=wait, cancel_futures=cancel_futures)


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
]
