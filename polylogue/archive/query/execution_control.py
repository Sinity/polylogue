"""Execution-control primitives for archive reads (polylogue-z9gh.1).

Archive read surfaces (API/MCP/HTTP) are async or threaded in signature but
historically executed synchronous SQLite work on the caller's event loop with
no deadline, cancellation, or admission control — one pathological query could
make the whole server unavailable (the 8.5 GiB MCP incident).

This module supplies the reusable layer the shared query transaction
(polylogue-z9gh.9.1) consumes:

- :class:`QueryExecutionContext` — immutable identity/deadline/cancellation
  state shared between the async caller and the SQLite worker thread.
- :class:`QueryAdmissionController` — FIFO-within-class weighted admission
  with reserved interactive capacity so scans cannot starve cheap reads.
  Admission is backpressure, never a semantic refusal: valid large work is
  eventually admitted.
- :class:`InterruptibleSQLiteRead` — runs one unit of archive read work on a
  dedicated read-only connection in a worker thread, with a SQLite progress
  handler enforcing cancellation/deadline and a cross-thread interrupt path.
- :class:`QueryExecutionReceipt` — the observable outcome record. Receipts
  carry safe query refs (hashes), never raw expressions.

Paging/spool formats and request-level budget contracts remain z9gh.9.1
scope; this slice owns execution and ownership primitives only.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

from polylogue.logging import get_logger

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

logger = get_logger(__name__)

T = TypeVar("T")

WorkloadClass = Literal["interactive", "scan"]

#: Default wall-clock deadline for one archive read. This is host protection,
#: not a semantic query limit — z9gh.9.1's resumable delivery is the answer
#: for legitimately long work.
DEFAULT_READ_DEADLINE_S = 120.0

#: SQLite VM opcodes between progress-guard checks (~sub-millisecond cadence
#: on cheap statements, bounded staleness on expensive ones).
PROGRESS_GUARD_OPCODES = 2000

#: Total concurrent admission weight per process.
DEFAULT_CAPACITY = 4

#: Capacity units the scan class may never consume, so an interactive read
#: always has headroom while scans are running.
DEFAULT_RESERVED_INTERACTIVE = 1

QueryExecutionState = Literal[
    "queued",
    "admitted",
    "running",
    "completed",
    "cancelled",
    "timed_out",
    "disconnected",
    "failed",
]


class QueryCancelledError(Exception):
    """The read was cancelled (explicit cancel or client disconnect)."""


class QueryTimeoutError(Exception):
    """The read exceeded its execution deadline."""


@dataclass
class QueryExecutionReceipt:
    """Observable outcome of one controlled archive read.

    ``query_ref`` is a content hash of the source expression — receipts never
    carry the raw expression text.
    """

    call_id: str
    query_ref: str
    workload_class: WorkloadClass
    state: QueryExecutionState = "queued"
    queue_position: int | None = None
    queued_s: float = 0.0
    run_s: float = 0.0
    interrupted: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "call_id": self.call_id,
            "query_ref": self.query_ref,
            "workload_class": self.workload_class,
            "state": self.state,
            "queue_position": self.queue_position,
            "queued_s": round(self.queued_s, 6),
            "run_s": round(self.run_s, 6),
            "interrupted": self.interrupted,
            "error": self.error,
        }


@dataclass(frozen=True)
class QueryExecutionContext:
    """Immutable identity + cancellation/deadline state for one read.

    The cancellation event is a ``threading.Event`` so the SQLite progress
    handler (worker thread) and the async caller share one flag without an
    event loop dependency. The receipt slot is mutable by design: the runner
    records the outcome on the context that owns the work.
    """

    call_id: str
    query_ref: str
    workload_class: WorkloadClass = "interactive"
    admission_weight: int = 1
    deadline_monotonic: float | None = None
    owner_ref: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, compare=False, repr=False)
    receipt: QueryExecutionReceipt = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "receipt",
            QueryExecutionReceipt(call_id=self.call_id, query_ref=self.query_ref, workload_class=self.workload_class),
        )

    @classmethod
    def create(
        cls,
        *,
        query_text: str | None = None,
        workload_class: WorkloadClass = "interactive",
        admission_weight: int = 1,
        timeout_s: float | None = DEFAULT_READ_DEADLINE_S,
        owner_ref: str | None = None,
    ) -> QueryExecutionContext:
        query_ref = hashlib.sha256((query_text or "").encode("utf-8")).hexdigest()[:16]
        deadline = time.monotonic() + timeout_s if timeout_s is not None else None
        return cls(
            call_id=uuid.uuid4().hex,
            query_ref=f"expr:{query_ref}",
            workload_class=workload_class,
            admission_weight=max(1, admission_weight),
            deadline_monotonic=deadline,
            owner_ref=owner_ref,
        )

    def cancel(self) -> None:
        self.cancel_event.set()

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def deadline_exceeded(self) -> bool:
        return self.deadline_monotonic is not None and time.monotonic() > self.deadline_monotonic

    def should_abort(self) -> bool:
        return self.cancelled or self.deadline_exceeded()


class QueryAdmissionController:
    """Weighted admission with FIFO order inside each workload class.

    - Total in-flight weight never exceeds ``capacity``.
    - ``reserved_interactive`` units are never granted to the scan class, so
      an interactive read always has capacity headroom (anti-starvation for
      cheap reads under scan pressure).
    - Weights are clamped to the maximum their class could ever be granted,
      so an over-declared estimate delays admission until the queue drains
      but can never become a permanent refusal.
    - The core is thread-safe (``threading.Condition``): async callers admit
      via ``asyncio.to_thread``; HTTP handler threads admit directly.
    """

    def __init__(
        self,
        *,
        capacity: int = DEFAULT_CAPACITY,
        reserved_interactive: int = DEFAULT_RESERVED_INTERACTIVE,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if not 0 <= reserved_interactive < capacity:
            raise ValueError("reserved_interactive must be in [0, capacity)")
        self._capacity = capacity
        self._reserved_interactive = reserved_interactive
        self._cond = threading.Condition()
        self._in_flight = 0
        self._queues: dict[WorkloadClass, deque[str]] = {"interactive": deque(), "scan": deque()}
        self._released: set[str] = set()

    def _class_ceiling(self, workload_class: WorkloadClass) -> int:
        if workload_class == "scan":
            return self._capacity - self._reserved_interactive
        return self._capacity

    def clamped_weight(self, ctx: QueryExecutionContext) -> int:
        return min(max(1, ctx.admission_weight), self._class_ceiling(ctx.workload_class))

    def queue_position(self, ctx: QueryExecutionContext) -> int | None:
        """0-based position of a queued context inside its class, else None."""
        with self._cond:
            queue = self._queues[ctx.workload_class]
            try:
                return queue.index(ctx.call_id)
            except ValueError:
                return None

    def _may_admit_locked(self, ctx: QueryExecutionContext, weight: int) -> bool:
        queue = self._queues[ctx.workload_class]
        if not queue or queue[0] != ctx.call_id:
            return False  # FIFO within class
        return self._in_flight + weight <= self._class_ceiling(ctx.workload_class)

    def _admit_blocking(self, ctx: QueryExecutionContext) -> int:
        weight = self.clamped_weight(ctx)
        with self._cond:
            self._queues[ctx.workload_class].append(ctx.call_id)
            ctx.receipt.queue_position = len(self._queues[ctx.workload_class]) - 1
            started = time.monotonic()
            try:
                while not self._may_admit_locked(ctx, weight):
                    if ctx.should_abort():
                        raise QueryCancelledError(f"cancelled while queued (call {ctx.call_id})")
                    self._cond.wait(timeout=0.05)
            except BaseException:
                self._remove_queued_locked(ctx)
                self._cond.notify_all()
                raise
            self._queues[ctx.workload_class].popleft()
            self._in_flight += weight
            ctx.receipt.queued_s = time.monotonic() - started
            ctx.receipt.state = "admitted"
            self._cond.notify_all()
        return weight

    def _remove_queued_locked(self, ctx: QueryExecutionContext) -> None:
        queue = self._queues[ctx.workload_class]
        with suppress(ValueError):
            queue.remove(ctx.call_id)

    def _release(self, ctx: QueryExecutionContext, weight: int) -> None:
        with self._cond:
            if ctx.call_id in self._released:
                return  # exactly-once release
            self._released.add(ctx.call_id)
            if len(self._released) > 4096:
                self._released.clear()
                self._released.add(ctx.call_id)
            self._in_flight -= weight
            self._cond.notify_all()

    @contextmanager
    def admit_blocking(self, ctx: QueryExecutionContext) -> Iterator[None]:
        """Synchronous admission for threaded callers (daemon HTTP handlers)."""
        weight = self._admit_blocking(ctx)
        try:
            yield
        finally:
            self._release(ctx, weight)

    @property
    def in_flight_weight(self) -> int:
        with self._cond:
            return self._in_flight


class InterruptibleSQLiteRead:
    """One archive read on a dedicated read-only connection in this thread.

    The runner opens its own :class:`ArchiveStore` (never a shared or writer
    connection), installs a progress guard that aborts on cancellation or
    deadline, and publishes the store so :meth:`interrupt` can abort the
    active statement from another thread. Cleanup (progress handler, store,
    published handle) happens exactly once in ``finally``.
    """

    def __init__(self, ctx: QueryExecutionContext) -> None:
        self._ctx = ctx
        self._store_lock = threading.Lock()
        self._store: ArchiveStore | None = None

    def interrupt(self) -> None:
        """Interrupt the active statement, if any. Safe from any thread."""
        with self._store_lock:
            store = self._store
        if store is not None:
            # Benign race with worker cleanup: interrupting a just-closed
            # connection has no one left to notify.
            with suppress(Exception):  # pragma: no cover
                store.interrupt_reads()

    def run(self, archive_root: Path, work: Callable[[ArchiveStore], T], *, read_timeout: float = 5.0) -> T:
        """Execute ``work`` against a dedicated read-only store (worker thread)."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        ctx = self._ctx
        ctx.receipt.state = "running"
        started = time.monotonic()

        def _guard() -> int:
            return 1 if ctx.should_abort() else 0

        store = ArchiveStore.open_existing(archive_root, read_timeout=read_timeout)
        with self._store_lock:
            self._store = store
        try:
            store.set_read_progress_guard(_guard, n_opcodes=PROGRESS_GUARD_OPCODES)
            # A cancel/deadline that landed before the first opcode must not
            # start the statement at all.
            if ctx.should_abort():
                raise _abort_error(ctx)
            try:
                result = work(store)
            except Exception as exc:
                if ctx.should_abort() and _is_interrupt_error(exc):
                    ctx.receipt.interrupted = True
                    raise _abort_error(ctx) from exc
                raise
            ctx.receipt.state = "completed"
            return result
        finally:
            ctx.receipt.run_s = time.monotonic() - started
            with self._store_lock:
                self._store = None
            store.close()


def _is_interrupt_error(exc: Exception) -> bool:
    import sqlite3

    return isinstance(exc, sqlite3.OperationalError) and "interrupt" in str(exc).lower()


def _abort_error(ctx: QueryExecutionContext) -> QueryCancelledError | QueryTimeoutError:
    if ctx.cancelled:
        ctx.receipt.state = "cancelled"
        return QueryCancelledError(f"archive read cancelled (call {ctx.call_id})")
    ctx.receipt.state = "timed_out"
    return QueryTimeoutError(f"archive read exceeded deadline (call {ctx.call_id})")


_default_controller: QueryAdmissionController | None = None
_default_controller_lock = threading.Lock()


def default_admission_controller() -> QueryAdmissionController:
    """Process-wide admission controller shared by all read surfaces."""
    global _default_controller
    with _default_controller_lock:
        if _default_controller is None:
            _default_controller = QueryAdmissionController()
        return _default_controller


def reset_default_admission_controller_for_tests() -> None:
    global _default_controller
    with _default_controller_lock:
        _default_controller = None


async def execute_archive_read(
    archive_root: Path,
    work: Callable[[ArchiveStore], T],
    *,
    ctx: QueryExecutionContext,
    controller: QueryAdmissionController | None = None,
    read_timeout: float = 5.0,
) -> T:
    """Run one archive read off the event loop under admission control.

    Caller cancellation (``asyncio.CancelledError``, which is how MCP client
    disconnects surface) sets the shared cancellation state, interrupts the
    exact connection, and waits for the worker to release its resources
    before re-raising — the loop is never left with an orphaned reader.
    """
    admission = controller or default_admission_controller()
    reader = InterruptibleSQLiteRead(ctx)

    def _admitted_run() -> T:
        with admission.admit_blocking(ctx):
            return reader.run(archive_root, work, read_timeout=read_timeout)

    worker = asyncio.create_task(asyncio.to_thread(_admitted_run))
    try:
        result = await asyncio.shield(worker)
    except asyncio.CancelledError:
        ctx.cancel()
        reader.interrupt()
        try:
            await worker
        except (QueryCancelledError, QueryTimeoutError, asyncio.CancelledError):
            pass
        except Exception:
            logger.warning("archive read worker failed during disconnect drain (call %s)", ctx.call_id)
        # Stamp after the drain: the worker's own abort path records
        # "cancelled", but the caller-side truth here is a disconnect.
        ctx.receipt.state = "disconnected"
        raise
    except (QueryCancelledError, QueryTimeoutError):
        raise
    except Exception as exc:
        ctx.receipt.state = "failed"
        ctx.receipt.error = f"{type(exc).__name__}"
        raise
    else:
        return result
    finally:
        logger.debug("archive read receipt: %s", ctx.receipt.to_dict())


def execute_archive_read_sync(
    archive_root: Path,
    work: Callable[[ArchiveStore], T],
    *,
    ctx: QueryExecutionContext,
    controller: QueryAdmissionController | None = None,
    read_timeout: float = 5.0,
) -> T:
    """Synchronous variant for threaded callers (daemon HTTP handlers).

    The handler thread is already dedicated, so the read runs in place; the
    progress guard still enforces deadline/cancellation, and admission shares
    the same process-wide controller as the async surfaces.
    """
    admission = controller or default_admission_controller()
    reader = InterruptibleSQLiteRead(ctx)
    try:
        with admission.admit_blocking(ctx):
            return reader.run(archive_root, work, read_timeout=read_timeout)
    except (QueryCancelledError, QueryTimeoutError):
        raise
    except Exception as exc:
        ctx.receipt.state = "failed"
        ctx.receipt.error = f"{type(exc).__name__}"
        raise
    finally:
        logger.debug("archive read receipt: %s", ctx.receipt.to_dict())


__all__ = [
    "DEFAULT_READ_DEADLINE_S",
    "InterruptibleSQLiteRead",
    "QueryAdmissionController",
    "QueryCancelledError",
    "QueryExecutionContext",
    "QueryExecutionReceipt",
    "QueryTimeoutError",
    "WorkloadClass",
    "default_admission_controller",
    "execute_archive_read",
    "execute_archive_read_sync",
    "reset_default_admission_controller_for_tests",
]
