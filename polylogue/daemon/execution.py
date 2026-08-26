"""Bounded daemon compute admission and cancellation primitives.

The daemon has one SQLite publication owner, but reads and pure preparation are
allowed to run concurrently.  This module is the one process-local seam for
that work: admission is finite in both work units and estimated bytes, and a
cancelled read interrupts its SQLite connection before its reservation is
returned.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal, TypeVar

AdmissionClass = Literal["interactive-read", "control", "incremental-background", "bulk-candidate"]
T = TypeVar("T")


class DaemonBackpressureError(RuntimeError):
    """The bounded compute queue cannot accept another unit of work."""

    code = "compute_backpressure"


class DaemonOperationCancelled(RuntimeError):  # noqa: N818 - public typed outcome name
    """A queued or running operation was cancelled before publication."""

    code = "operation_cancelled"


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

    @property
    def queued(self) -> int:
        return self.queued_units

    def to_dict(self) -> dict[str, int]:
        return {
            "capacity_units": self.capacity_units,
            "used_units": self.used_units,
            "capacity_bytes": self.capacity_bytes,
            "used_bytes": self.used_bytes,
            "queued_units": self.queued_units,
            "queued_bytes": self.queued_bytes,
            "active_units": self.active_units,
            "rejected": self.rejected,
        }


class CancellationHandle:
    """One-shot cancellation handle shared by a request and its SQLite read."""

    def __init__(self) -> None:
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        self._connections: dict[int, object] = {}

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

    def cancel(self) -> None:
        self._cancelled.set()
        with self._lock:
            connections = tuple(self._connections.values())
        for connection in connections:
            _interrupt(connection)


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


class _Admission:
    def __init__(self, capacity_units: int, capacity_bytes: int) -> None:
        if capacity_units < 1 or capacity_bytes < 0:
            raise ValueError("compute admission capacities are invalid")
        self.capacity_units = capacity_units
        self.capacity_bytes = capacity_bytes
        self.used_units = 0
        self.used_bytes = 0
        self.active_units = 0
        self.active_bytes = 0
        self.rejected = 0
        self.lock = threading.Lock()

    def acquire(self, units: int, estimated_bytes: int) -> None:
        if units < 1 or estimated_bytes < 0 or estimated_bytes > self.capacity_bytes:
            raise DaemonBackpressureError("operation exceeds the daemon compute admission envelope")
        with self.lock:
            if self.used_units + units > self.capacity_units or self.used_bytes + estimated_bytes > self.capacity_bytes:
                self.rejected += 1
                raise DaemonBackpressureError("daemon compute admission is saturated; retry shortly")
            self.used_units += units
            self.used_bytes += estimated_bytes

    def mark_active(self, units: int, estimated_bytes: int) -> None:
        with self.lock:
            self.active_units += units
            self.active_bytes += estimated_bytes

    def release(self, units: int, estimated_bytes: int, *, active: bool) -> None:
        with self.lock:
            self.used_units -= units
            self.used_bytes -= estimated_bytes
            if active:
                self.active_units -= units
                self.active_bytes -= estimated_bytes

    def snapshot(self) -> AdmissionSnapshot:
        with self.lock:
            return AdmissionSnapshot(
                capacity_units=self.capacity_units,
                used_units=self.used_units,
                capacity_bytes=self.capacity_bytes,
                used_bytes=self.used_bytes,
                queued_units=max(0, self.used_units - self.active_units),
                queued_bytes=max(0, self.used_bytes - self.active_bytes),
                active_units=self.active_units,
                rejected=self.rejected,
            )


class BoundedComputeAdapter:
    """The sole bounded in-process compute adapter used by daemon reads."""

    def __init__(
        self,
        *,
        max_workers: int = 8,
        queue_units: int = 16,
        queue_bytes: int = 64 * 1024 * 1024,
        thread_name_prefix: str = "polylogue-compute",
    ) -> None:
        self.max_workers = max_workers
        self._admission = _Admission(max_workers + queue_units, queue_bytes)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=thread_name_prefix)

    def submit(
        self,
        function: Callable[[], T],
        *,
        admission_class: AdmissionClass = "interactive-read",
        units: int = 1,
        estimated_bytes: int = 0,
        cancellation: CancellationHandle | None = None,
    ) -> SubmittedOperation:
        del admission_class  # class is carried for future quota selection and telemetry.
        handle = cancellation or CancellationHandle()
        self._admission.acquire(units, estimated_bytes)

        released = False
        release_lock = threading.Lock()
        started = False

        def release_once() -> None:
            nonlocal released
            with release_lock:
                if released:
                    return
                released = True
                self._admission.release(units, estimated_bytes, active=started)

        def run() -> T:
            nonlocal started
            self._admission.mark_active(units, estimated_bytes)
            started = True
            token = _CURRENT_CANCELLATION.set(handle)
            try:
                if handle.cancelled:
                    raise DaemonOperationCancelled("operation cancelled before compute started")
                return function()
            finally:
                _CURRENT_CANCELLATION.reset(token)
                release_once()

        try:
            future: Future[object] = self.executor.submit(run)
        except BaseException:
            release_once()
            raise

        def release(_future: Future[object]) -> None:
            release_once()

        future.add_done_callback(release)
        return SubmittedOperation(future=future, cancellation=handle)

    def snapshot(self) -> AdmissionSnapshot:
        return self._admission.snapshot()

    def shutdown(self, *, wait: bool = False, cancel_futures: bool = True) -> None:
        self.executor.shutdown(wait=wait, cancel_futures=cancel_futures)


__all__ = [
    "AdmissionClass",
    "AdmissionSnapshot",
    "BoundedComputeAdapter",
    "CancellationHandle",
    "DaemonBackpressureError",
    "DaemonOperationCancelled",
    "SubmittedOperation",
    "current_cancellation",
]
