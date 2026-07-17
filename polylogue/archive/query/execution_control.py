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
  dedicated read-only connection and snapshot in a worker thread, with a SQLite
  progress handler enforcing cancellation/deadline/work budgets and a
  cross-thread interrupt path.
- :class:`QueryExecutionReceipt` — the observable outcome record. Receipts
  carry safe query refs (hashes), never raw expressions.

Paging/spool formats and cross-surface budget policy remain z9gh.9.1 scope.
This layer owns execution, snapshot/connection lifetime, optional SQLite VM-work
containment, and production work/cleanup receipts.
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

#: How long a disconnect waits for the worker thread to drain before leaving
#: it to finish in the background. The thread's own cleanup still runs when
#: the (uninterruptible Python-side) work eventually returns.
DISCONNECT_DRAIN_TIMEOUT_S = 10.0

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
    "work_budget_exceeded",
    "disconnected",
    "failed",
]


class QueryCancelledError(Exception):
    """The read was cancelled (explicit cancel or client disconnect)."""


class QueryTimeoutError(Exception):
    """The read exceeded its execution deadline."""


class QueryWorkBudgetExceededError(Exception):
    """The read exceeded its declared SQLite VM-work budget."""


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
    sqlite_progress_callbacks: int = 0
    sqlite_vm_steps_lower_bound: int = 0
    sqlite_vm_step_budget: int | None = None
    selected_rows_exact: int | None = None
    rows_emitted: int = 0
    result_pages_emitted: int = 0
    cleanup_complete: bool = False
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
            "sqlite_progress_callbacks": self.sqlite_progress_callbacks,
            "sqlite_vm_steps_lower_bound": self.sqlite_vm_steps_lower_bound,
            "sqlite_vm_step_budget": self.sqlite_vm_step_budget,
            "selected_rows_exact": self.selected_rows_exact,
            "rows_emitted": self.rows_emitted,
            "result_pages_emitted": self.result_pages_emitted,
            "cleanup_complete": self.cleanup_complete,
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
    sqlite_vm_step_budget: int | None = None
    owner_ref: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, compare=False, repr=False)
    _receipt_lock: threading.Lock = field(default_factory=threading.Lock, compare=False, repr=False)
    receipt: QueryExecutionReceipt = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.sqlite_vm_step_budget is not None and self.sqlite_vm_step_budget < 0:
            raise ValueError("sqlite_vm_step_budget must be non-negative")
        object.__setattr__(
            self,
            "receipt",
            QueryExecutionReceipt(
                call_id=self.call_id,
                query_ref=self.query_ref,
                workload_class=self.workload_class,
                sqlite_vm_step_budget=self.sqlite_vm_step_budget,
            ),
        )

    @classmethod
    def create(
        cls,
        *,
        query_text: str | None = None,
        workload_class: WorkloadClass = "interactive",
        admission_weight: int = 1,
        timeout_s: float | None = DEFAULT_READ_DEADLINE_S,
        sqlite_vm_step_budget: int | None = None,
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
            sqlite_vm_step_budget=sqlite_vm_step_budget,
            owner_ref=owner_ref,
        )

    def cancel(self) -> None:
        self.cancel_event.set()

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def deadline_exceeded(self) -> bool:
        return self.deadline_monotonic is not None and time.monotonic() > self.deadline_monotonic

    def record_sqlite_progress(self, opcodes: int) -> None:
        """Record one real SQLite progress callback without estimating rows."""

        with self._receipt_lock:
            self.receipt.sqlite_progress_callbacks += 1
            self.receipt.sqlite_vm_steps_lower_bound += max(1, int(opcodes))

    def record_result_page(self, *, emitted_rows: int, selected_rows_exact: int | None = None) -> None:
        """Record a delivered logical page and exact selection evidence when known."""

        with self._receipt_lock:
            self.receipt.result_pages_emitted += 1
            self.receipt.rows_emitted += max(0, int(emitted_rows))
            if selected_rows_exact is not None:
                normalized = max(0, int(selected_rows_exact))
                existing = self.receipt.selected_rows_exact
                if existing is not None and existing != normalized:
                    raise RuntimeError("query execution reported contradictory exact selection counts")
                self.receipt.selected_rows_exact = normalized

    def mark_cleanup_complete(self) -> None:
        with self._receipt_lock:
            self.receipt.cleanup_complete = True

    def work_budget_exceeded(self) -> bool:
        budget = self.sqlite_vm_step_budget
        if budget is None:
            return False
        with self._receipt_lock:
            return self.receipt.sqlite_vm_steps_lower_bound >= budget

    def abort_reason(self) -> Literal["cancelled", "timed_out", "work_budget_exceeded"] | None:
        if self.cancelled:
            return "cancelled"
        if self.deadline_exceeded():
            return "timed_out"
        if self.work_budget_exceeded():
            return "work_budget_exceeded"
        return None

    def should_abort(self) -> bool:
        return self.abort_reason() is not None


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
                        # A deadline that expires while queued is a timeout,
                        # not a cancellation — surfaces map the two to
                        # different responses (503 vs disconnect).
                        raise _abort_error(ctx)
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
    connection), begins one read snapshot, installs a progress guard that aborts
    on cancellation, deadline, or a declared VM-work budget, and publishes the
    store so :meth:`interrupt` can abort the active statement from another
    thread. Cleanup (progress handler, transaction, store, published handle)
    happens exactly once in ``finally``.
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
        progress_opcodes = PROGRESS_GUARD_OPCODES
        if ctx.sqlite_vm_step_budget is not None:
            progress_opcodes = min(progress_opcodes, max(1, ctx.sqlite_vm_step_budget))

        def _guard() -> int:
            ctx.record_sqlite_progress(progress_opcodes)
            return 1 if ctx.should_abort() else 0

        store = ArchiveStore.open_existing(archive_root, read_timeout=read_timeout)
        with self._store_lock:
            self._store = store
        try:
            _set_progress_guard(store, _guard, n_opcodes=progress_opcodes)
            # An abort that landed before ownership begins must not start the
            # caller's statement or acquire a read snapshot.
            if ctx.should_abort():
                raise _abort_error(ctx)
            try:
                store.begin_read_snapshot()
                result = work(store)
            except Exception as exc:
                if ctx.should_abort() and _is_interrupt_error(exc):
                    ctx.receipt.interrupted = True
                    raise _abort_error(ctx) from exc
                raise
            else:
                # The progress guard only observes SQLite execution. An abort
                # that lands during Python-side post-processing (row grouping,
                # envelope assembly) must still abort here rather than return a
                # "completed" result nobody is waiting for.
                if ctx.should_abort():
                    raise _abort_error(ctx)
                ctx.receipt.state = "completed"
                return result
        finally:
            ctx.receipt.run_s = time.monotonic() - started
            with self._store_lock:
                self._store = None
            _close_store(store)
            ctx.mark_cleanup_complete()

    @contextmanager
    def open_context(self, archive_root: Path, *, read_timeout: float = 5.0) -> Iterator[ArchiveStore]:
        """Yield the controlled store to synchronous read-surface adapters."""
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        ctx = self._ctx
        with default_admission_controller().admit_blocking(ctx):
            # Keep lightweight archive doubles compatible with the positional
            # open contract; the real store supplies its bounded busy timeout.
            store = ArchiveStore.open_existing(archive_root)
            with self._store_lock:
                self._store = store
            try:
                _set_progress_guard(store, lambda: 1 if ctx.should_abort() else 0)
                if ctx.should_abort():
                    raise _abort_error(ctx)
                begin_snapshot = getattr(store, "begin_read_snapshot", None)
                if callable(begin_snapshot):
                    begin_snapshot()
                ctx.receipt.state = "running"
                yield store
                if ctx.should_abort():
                    raise _abort_error(ctx)
                ctx.receipt.state = "completed"
            except Exception as exc:
                ctx.receipt.state = "failed"
                ctx.receipt.error = f"{type(exc).__name__}"
                raise
            finally:
                with self._store_lock:
                    self._store = None
                _close_store(store)
                ctx.mark_cleanup_complete()


def _is_interrupt_error(exc: Exception) -> bool:
    import sqlite3

    return isinstance(exc, sqlite3.OperationalError) and "interrupt" in str(exc).lower()


def _set_progress_guard(
    store: ArchiveStore, guard: Callable[[], int], *, n_opcodes: int = PROGRESS_GUARD_OPCODES
) -> None:
    setter = getattr(store, "set_read_progress_guard", None)
    if callable(setter):
        setter(guard, n_opcodes=n_opcodes)


def _close_store(store: ArchiveStore) -> None:
    """Clear read state and close full stores while tolerating test doubles."""
    clear_guard = getattr(store, "clear_read_progress_guard", None)
    if callable(clear_guard):
        clear_guard()
    end_snapshot = getattr(store, "end_read_snapshot", None)
    if callable(end_snapshot):
        end_snapshot()
    closer = getattr(store, "close", None)
    if callable(closer):
        closer()


def _abort_error(
    ctx: QueryExecutionContext,
) -> QueryCancelledError | QueryTimeoutError | QueryWorkBudgetExceededError:
    reason = ctx.abort_reason()
    if reason == "cancelled":
        ctx.receipt.state = "cancelled"
        return QueryCancelledError(f"archive read cancelled (call {ctx.call_id})")
    if reason == "work_budget_exceeded":
        ctx.receipt.state = "work_budget_exceeded"
        return QueryWorkBudgetExceededError(f"archive read exceeded SQLite work budget (call {ctx.call_id})")
    ctx.receipt.state = "timed_out"
    return QueryTimeoutError(f"archive read exceeded deadline (call {ctx.call_id})")


def classify_unit_expression_workload(expression: str) -> WorkloadClass:
    """Classify a terminal unit expression for admission control.

    Aggregation, grouping, and pipeline stages can trigger archive-wide
    materialization (the z9gh incident shape), so they admit as ``scan`` and
    never consume the reserved interactive headroom. Anything unparsable is
    classified interactive — the route's own validation owns rejection.
    """
    from polylogue.archive.query.expression import ExpressionCompileError, parse_unit_source_expression

    try:
        source = parse_unit_source_expression(expression)
    except ExpressionCompileError:
        return "interactive"
    if source is None:
        return "interactive"
    if source.aggregate is not None or source.group_by is not None or source.pipeline_stages:
        return "scan"
    return "interactive"


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
            # Bounded drain: SQLite work aborts at progress-guard cadence,
            # but Python-side post-processing is uninterruptible — do not
            # wait on it forever. An undrained worker keeps running in the
            # background and still performs its own cleanup on exit.
            await asyncio.wait_for(worker, timeout=DISCONNECT_DRAIN_TIMEOUT_S)
        except (QueryCancelledError, QueryTimeoutError, QueryWorkBudgetExceededError, asyncio.CancelledError):
            pass
        except TimeoutError:
            logger.warning(
                "archive read worker did not drain within %.0fs after disconnect (call %s); leaving it to finish in the background",
                DISCONNECT_DRAIN_TIMEOUT_S,
                ctx.call_id,
            )
        except Exception:
            logger.warning("archive read worker failed during disconnect drain (call %s)", ctx.call_id)
        # Stamp after the drain: the worker's own abort path records
        # "cancelled", but the caller-side truth here is a disconnect.
        ctx.receipt.state = "disconnected"
        raise
    except (QueryCancelledError, QueryTimeoutError, QueryWorkBudgetExceededError):
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
    except (QueryCancelledError, QueryTimeoutError, QueryWorkBudgetExceededError):
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
    "QueryWorkBudgetExceededError",
    "WorkloadClass",
    "classify_unit_expression_workload",
    "default_admission_controller",
    "execute_archive_read",
    "execute_archive_read_sync",
    "reset_default_admission_controller_for_tests",
]
