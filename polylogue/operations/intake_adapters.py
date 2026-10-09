"""Domain adapters for the fair daemon intake scheduler.

This module owns the source/storage-facing part of intake composition.  The
daemon scheduler stays storage- and source-independent; adapters retain only
disposable discovery hints and delegate writes to the daemon's existing writer
runner.
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import os
import sqlite3
import stat
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

from polylogue.core.compute_cancel import raise_if_operation_cancelled
from polylogue.core.durable_fs import DurableFilesystemError
from polylogue.core.storage_faults import ArchiveStorageFaultError
from polylogue.daemon.intake import (
    DEFAULT_INTAKE_BYTE_BUDGET,
    UNMEASURABLE_INTAKE_COST_BYTES,
    AdmissionOutcome,
    AdmissionResult,
    FairIntakeDispatcher,
    IntakeAdapter,
    IntakeItem,
    IntakePass,
    is_transient_admission_error,
)
from polylogue.logging import ERROR, WARNING, emit
from polylogue.maintenance.candidate_capacity import ArchiveCapacityError, InsufficientCapacityError
from polylogue.maintenance.receipt_fs import MaintenanceReceiptPathError
from polylogue.operations.cold_build_coverage import ColdBuildCoverageError, promote_cold_build_covering_active_index
from polylogue.operations.drive_readiness import DriveCatchupReport, DriveCatchupState
from polylogue.sources.live.batch import CursorAuthorityBlockedError
from polylogue.sources.live.cold_build import (
    ColdBuildGeneration,
    active_cold_build_generation,
    active_index_generation_is_empty,
    clear_cold_build_generation,
    is_transient_cold_storage_errno,
    register_cold_build_generation,
)
from polylogue.sources.live.discovery import _bounded_source_paths as _bounded_source_paths
from polylogue.sources.live.discovery import _source_path_steps
from polylogue.sources.live.metrics import (
    REFUSED_UNATTEMPTED,
    REFUSED_UNATTEMPTED_TIME_BUDGET,
)
from polylogue.sources.live.production_baseline import ProductionBaselineError, ProductionBaselineReadUnavailableError
from polylogue.sources.live.source_selection import deepest_source_for_path
from polylogue.sources.live.watcher import LiveWatcher, WatchSource, _log_ingest_metrics
from polylogue.sources.walk_faults import WalkFault, WalkRefusedError
from polylogue.storage.archive_identity import ArchiveLocationError
from polylogue.storage.sqlite.reference_seal import ReferenceSealError, ReferenceSealStaleError
from polylogue.storage.sqlite.write_lease import UnleasedWriteError

_T = TypeVar("_T")

__all__ = [
    "ColdBuildCoverageError",
    "ColdBuildGeneration",
    "DaemonIntakeContext",
    "DaemonIntakeService",
    "FileIntakeAdapter",
    "MultiplexIntakeAdapter",
    "CallbackIntakeAdapter",
    "RawMaterializationIntakeAdapter",
    "RawMaterializationDiscovery",
    "SubUnitHaltPolicy",
    "active_index_generation_is_empty",
    "build_intake_adapters",
    "clear_cold_build_generation",
    "active_cold_build_generation",
    "discover_pending_raw_ids",
    "promote_cold_build_covering_active_index",
    "register_cold_build_generation",
]


_RAW_DISCOVERY_INSPECTION_LIMIT = 32
#: A raw sweep that follows a completed one pages at the service's idle
#: cadence until this multiple of the previous sweep's duration has passed,
#: then at the prompt one. Prompt paging therefore holds at most about a tenth
#: of wall time however large the archive grows, while a raw that becomes owed
#: without an arrival is still reached within about ten sweep durations.
_RAW_RESWEEP_REST_FACTOR = 9.0
_FILE_DISCOVERY_STEP_LIMIT = 256
#: Fresh paths discovered beyond the offered page, so the next page's parsing
#: can be prefetched while the current page publishes.
_FRESH_LOOKAHEAD_PAGES = 1
_FILE_DISCOVERY_RESCAN_S = 600.0
_FILE_RETRY_DELAY_S = 5.0


@dataclass(frozen=True, slots=True)
class ColdBuildSettlement:
    state: Literal["complete", "retryable", "blocked"]
    reason: str | None = None
    attempts: int = 0
    next_retry_at: float | None = None


def classify_cold_build_settlement_failure(exc: Exception) -> tuple[str, bool] | None:
    """Classify only the named settlement faults; let other failures reach supervision."""
    while isinstance(exc, (DurableFilesystemError, MaintenanceReceiptPathError)) and isinstance(
        exc.__cause__, Exception
    ):
        exc = exc.__cause__
    if isinstance(exc, ColdBuildCoverageError):
        return "active_coverage_incomplete", False
    if isinstance(exc, ReferenceSealStaleError):
        return "promotion_evidence_changed", True
    if isinstance(exc, ReferenceSealError):
        return "durable_reference_preservation", False
    if isinstance(exc, ProductionBaselineReadUnavailableError):
        return "source_integrity", True
    if isinstance(exc, ProductionBaselineError):
        return "source_integrity", False
    if isinstance(exc, InsufficientCapacityError):
        return "capacity_unavailable", False
    if isinstance(exc, ArchiveCapacityError):
        capacity_cause = exc.__cause__
        if isinstance(capacity_cause, ArchiveLocationError):
            capacity_cause = capacity_cause.__cause__
        if isinstance(capacity_cause, OSError) and (
            is_transient_cold_storage_errno(capacity_cause.errno)
            or capacity_cause.errno in {errno.EACCES, errno.EPERM, errno.ENOENT, errno.ENOSPC, errno.EDQUOT}
        ):
            return "capacity_inventory_unavailable", True
    if isinstance(exc, sqlite3.Error):
        code = getattr(exc, "sqlite_errorcode", None)
        primary = code & 0xFF if isinstance(code, int) else None
        if primary in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
            return "sqlite_busy", True
        if primary in (sqlite3.SQLITE_CANTOPEN, sqlite3.SQLITE_PERM):
            return "sqlite_open_unavailable", True
        if primary == sqlite3.SQLITE_FULL:
            return "capacity_unavailable", False
        if primary == sqlite3.SQLITE_IOERR:
            return "candidate_storage_unavailable", True
        return None
    if isinstance(exc, OSError):
        if is_transient_cold_storage_errno(exc.errno):
            return ("storage_busy" if exc.errno in (errno.EAGAIN, errno.EBUSY) else "storage_io_unavailable"), True
        if exc.errno in (errno.ENOSPC, errno.EDQUOT):
            return "capacity_unavailable", False
        if exc.errno in (errno.ENOENT, errno.EACCES):
            return "source_evidence_unavailable", False
    return None


@dataclass(frozen=True, slots=True)
class DaemonIntakeContext:
    archive_root: Path
    watcher: LiveWatcher
    sources: tuple[WatchSource, ...]
    write_runner: Callable[..., Awaitable[Any]] | None = None

    async def run_write(self, actor: str, function: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
        if self.write_runner is None:
            return await asyncio.to_thread(function, *args, **kwargs)
        return cast(_T, await self.write_runner(actor, function, *args, **kwargs))


class FileIntakeAdapter(IntakeAdapter):
    """Browser/configured-local adapter; durable cursors are its ack state."""

    def __init__(
        self,
        context: DaemonIntakeContext,
        source: WatchSource,
        *,
        class_name: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.context = context
        self.source = source
        self.class_name = class_name or source.name
        self._clock = clock
        self._after: str | None = None
        self._retry_after: str | None = None
        self._retry_skip_after: str | None = None
        self._retry_through: str | None = None
        self._retry_turn = False
        self._retry_page = False
        self._retry_page_paths: tuple[Path, ...] = ()
        self._retry_page_pending = False
        self._ops_ledger_generation = self._ledger_generation()
        self._last_root_mtime_ns: int | None = None
        self._last_source_entries: tuple[int, int] | None = None
        self._last_hint_revision = context.watcher.intake_revision(source)
        self._fresh_walk: Iterator[Path | None] | None = None
        self._fresh_pending: list[Path] = []
        self._fresh_page_paths: tuple[Path, ...] = ()
        self._fresh_page_pending = False
        self._fresh_attempted_paths: set[Path] = set()
        # One admission page's deferrals that have no durable retry cursor.
        # Acknowledgement must leave these in the existing fresh/local retry walk.
        self._deferred_without_cursor: set[Path] = set()
        self._root_refused_pending = False
        self._retry_state_lock = threading.Lock()
        self._fresh_retry_debt: dict[Path, float] = {}
        self._overflow_rescan_due_at: float | None = None
        self._local_retry_page = False
        self._prefer_local_retry = True
        self._fresh_exhausted = False
        self._fresh_exhausted_at: float | None = None
        self._rescan_after_walk = False
        self._discovery_lock = threading.Lock()
        self._discovery_thread = threading.local()

    async def discover(self, *, limit: int) -> Sequence[IntakeItem]:
        from polylogue.core.degraded import is_fully_degraded

        if is_fully_degraded():
            # Discovery reads the cursor tier (due retries, pending fresh
            # pages); a structurally degraded daemon must not touch it on every
            # event, and has nothing to admit anyway (#1003).
            return ()
        # Filesystem enumeration and path probes can be slow on mounted
        # sources. Keep them off the daemon's event loop.
        cancelled = threading.Event()
        try:
            return await asyncio.to_thread(self._observed_discover_sync, limit, cancelled)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    def _observed_discover_sync(self, limit: int, cancelled: threading.Event) -> Sequence[IntakeItem]:
        from polylogue.daemon.discovery_progress import begin_discovery, end_discovery

        with self._discovery_lock:
            if cancelled.is_set():
                return ()
            token = begin_discovery(self.source.name, owner=self)
            self._discovery_thread.token = token
            failed = True
            try:
                result = self._discover_sync(limit)
                failed = False
                return result
            finally:
                del self._discovery_thread.token
                end_discovery(token, failed=failed, pending=self.discovery_pending)

    def _reset_fresh_walk(self) -> None:
        self._fresh_walk = None
        self._fresh_pending.clear()
        self._fresh_page_paths = ()
        self._fresh_page_pending = False
        self._fresh_attempted_paths.clear()
        self._root_refused_pending = False
        self._fresh_exhausted = False
        self._fresh_exhausted_at = None
        self._rescan_after_walk = False

    def _request_fresh_rescan(self) -> None:
        if self._fresh_walk is not None or self._fresh_pending:
            # Finish the bounded continuation and its unacknowledged page.
            # Repeated hints coalesce into one complete follow-up walk.
            self._rescan_after_walk = True
        else:
            self._after = None
            self._reset_fresh_walk()

    @property
    def discovery_pending(self) -> bool:
        return not self._root_refused_pending and (
            self._fresh_walk is not None or bool(self._fresh_pending) or self._rescan_after_walk
        )

    @property
    def retry_due_in_s(self) -> float | None:
        # The walk lock can be held across filesystem I/O. This short-held
        # state lock keeps the event loop responsive while snapshotting debt.
        with self._retry_state_lock:
            deadlines = (*self._fresh_retry_debt.values(),)
            if self._overflow_rescan_due_at is not None:
                deadlines += (self._overflow_rescan_due_at,)
        return max(0.0, min(deadlines) - self._clock()) if deadlines else None

    @staticmethod
    def _pending_path_is_live(path: Path) -> bool:
        try:
            return stat.S_ISREG(path.lstat().st_mode)
        except (FileNotFoundError, NotADirectoryError):
            return False
        except OSError:
            # An inaccessible live file still needs its retryable admission.
            return True

    @staticmethod
    def _has_durable_retry_record(record: Any) -> bool:
        return (
            record is not None
            and not getattr(record, "excluded", False)
            and (
                record.failure_count > 0
                or (getattr(record, "content_fingerprint", None) is None and record.next_retry_at is not None)
            )
        )

    def _offer_fresh_page(self, limit: int) -> list[Path]:
        page = self._fresh_pending[:limit]
        self._fresh_page_paths = tuple(page)
        self._fresh_page_pending = bool(page)
        return page

    def _discover_fresh_paths(self, limit: int) -> list[Path]:
        if limit <= 0:
            return []
        if not self.source.root.is_dir():
            # A missing mount is a refusal, not the disappearance of every
            # pending carrier. Keep the page for a later retry.
            self._root_refused_pending = True
            raise WalkRefusedError(
                "intake discovery could not read a source root",
                [WalkFault(self.source.root, "source root is unavailable")],
            )
        if self._root_refused_pending:
            self._root_refused_pending = False
            if self._fresh_page_pending:
                return list(self._fresh_page_paths[:limit])
        if self._fresh_page_pending:
            # Only attempted, unacknowledged paths owe a cooldown. A page
            # tail left outside the dispatcher or source budget is ordinary
            # backlog and stays in the immediate fresh continuation.
            offered = set(self._fresh_attempted_paths)
            due_at = self._clock() + _FILE_RETRY_DELAY_S
            cursor = getattr(self.context.watcher, "_cursor", None)
            get_records = getattr(cursor, "get_records", None)
            records = get_records(self._fresh_page_paths) if callable(get_records) else {}
            live_paths = [path for path in self._fresh_page_paths if self._pending_path_is_live(path)]
            with self._retry_state_lock:
                if len(live_paths) != len(self._fresh_page_paths) and self._overflow_rescan_due_at is None:
                    # A vanished unacknowledged carrier may reappear without a
                    # recursive watcher hint; revisit it after the cooldown.
                    self._overflow_rescan_due_at = due_at
                for retry_path in live_paths:
                    if retry_path not in offered:
                        continue
                    record = records.get(retry_path)
                    if self._has_durable_retry_record(record):
                        # The durable cursor owns this retry and its backoff.
                        self._fresh_retry_debt.pop(retry_path, None)
                        continue
                    if (
                        retry_path not in self._fresh_retry_debt
                        and len(self._fresh_retry_debt) >= _FILE_DISCOVERY_STEP_LIMIT
                    ):
                        # A due rescan recovers evicted debt without growing memory.
                        self._fresh_retry_debt.pop(next(iter(self._fresh_retry_debt)))
                        if self._overflow_rescan_due_at is None:
                            self._overflow_rescan_due_at = due_at
                    self._fresh_retry_debt[retry_path] = due_at
            self._fresh_pending = [path for path in self._fresh_pending if path not in offered]
        # Page items that were never attempted (the pass ran out of its time
        # budget before reaching them) stay ordinary backlog even when a
        # later-sorting sibling was acknowledged and advanced ``_after`` past
        # them. Filtering them by position dropped them from the continuation;
        # the resumed walk starts after ``_after`` too, so nothing offered them
        # again until the exhausted-walk rescan ten minutes later.
        unattempted = frozenset(self._fresh_page_paths) - self._fresh_attempted_paths
        self._fresh_page_pending = False
        self._fresh_page_paths = ()
        self._fresh_attempted_paths.clear()
        if self._after is not None:
            after = self._after
            self._fresh_pending = [path for path in self._fresh_pending if str(path) > after or path in unattempted]
        # A vanished file is retryable when it disappears after discovery, but
        # retaining that stale page forever prevents both later paths and a
        # queued rescan from running. Recreated files return in a later scan.
        self._fresh_pending = [path for path in self._fresh_pending if self._pending_path_is_live(path)]
        lookahead = limit * (1 + _FRESH_LOOKAHEAD_PAGES)
        if self._fresh_pending:
            # A live walk refills the lookahead behind the carried-over page;
            # without it every other page would have nothing to prefetch.
            if self._fresh_walk is not None:
                self._extend_fresh_pending(lookahead)
            return self._offer_fresh_page(limit)
        if self._fresh_exhausted:
            if self._rescan_after_walk:
                self._after = None
                self._reset_fresh_walk()
            elif (
                self._fresh_exhausted_at is None or self._clock() - self._fresh_exhausted_at < _FILE_DISCOVERY_RESCAN_S
            ):
                return []
            else:
                # A missed recursive watcher event may have inserted a file
                # before the acknowledged cursor. Reconcile from the beginning;
                # durable ingest identity makes previously admitted files cheap
                # duplicates rather than skipping the new file forever.
                self._after = None
                self._reset_fresh_walk()
        if self._fresh_walk is None:
            from polylogue.daemon.discovery_progress import advance_discovery

            self._fresh_walk = _source_path_steps(
                self.source,
                self.context.sources,
                after=self._after,
                on_inspected=lambda: advance_discovery(getattr(self._discovery_thread, "token", None), inspected=1),
                on_disposition=lambda _path, disposition, _reason: advance_discovery(
                    getattr(self._discovery_thread, "token", None), disposition=disposition
                ),
            )
        self._extend_fresh_pending(lookahead)
        return self._offer_fresh_page(limit)

    def _extend_fresh_pending(self, target: int) -> None:
        """Walk until ``target`` fresh paths are pending or the step budget is spent."""
        walk = self._fresh_walk
        if walk is None:
            return
        steps = max(_FILE_DISCOVERY_STEP_LIMIT, target)
        for _ in range(steps):
            if len(self._fresh_pending) >= target:
                break
            try:
                path = next(walk)
            except StopIteration:
                self._fresh_walk = None
                self._fresh_exhausted = True
                self._fresh_exhausted_at = self._clock()
                break
            except Exception:
                # The continuation is gone. If a hint arrived during this
                # walk, retry from the beginning before clearing its request.
                if self._rescan_after_walk:
                    self._after = None
                self._reset_fresh_walk()
                raise
            if path is not None:
                self._fresh_pending.append(path)

    def _discover_sync(self, limit: int) -> Sequence[IntakeItem]:
        generation = self._ledger_generation()
        if generation != self._ops_ledger_generation:
            # ops.db is disposable. Losing or replacing its inode invalidates
            # the scheduling hints even when no source directory changed.
            self._after = None
            self._retry_after = None
            self._retry_skip_after = None
            self._retry_through = None
            self._ops_ledger_generation = generation
            self._reset_fresh_walk()
        if self._retry_page_pending and self._retry_page_paths and not self._local_retry_page:
            # No item from the previous page reached admission or ack (for
            # example every item was in the dispatcher's cooldown). Rotate
            # past it within a finite sweep; the next sweep revisits it.
            self._retry_skip_after = str(self._retry_page_paths[-1])
        self._retry_page_pending = False
        self._retry_page_paths = ()
        self._local_retry_page = False
        hint_revision = self.context.watcher.intake_revision(self.source)
        if hint_revision != self._last_hint_revision:
            self._last_hint_revision = hint_revision
            self._request_fresh_rescan()
        # A producer may add a file before the walk's position. Root mtime
        # catches direct additions without a watcher hint, but SQLite sidecars
        # in a coincident archive/source root change that mtime too. Compare
        # source-visible entries before restarting the walk.
        try:
            root_mtime_ns = self.source.root.stat().st_mtime_ns
        except OSError:
            root_mtime_ns = None
        if root_mtime_ns is not None and root_mtime_ns != self._last_root_mtime_ns:
            entries = self._source_entries()
            if self._last_source_entries is not None and entries != self._last_source_entries:
                self._request_fresh_rescan()
            self._last_source_entries = entries
            self._last_root_mtime_ns = root_mtime_ns
        if not self.source.root.is_dir():
            # A missing mount cannot service local retry debt. Move expired
            # deadlines forward before the root refusal reaches the daemon,
            # so it uses the retry cooldown instead of a 50 ms wake loop.
            now = self._clock()
            due_at = now + _FILE_RETRY_DELAY_S
            with self._retry_state_lock:
                for path, deadline in self._fresh_retry_debt.items():
                    if deadline <= now:
                        self._fresh_retry_debt[path] = due_at
                if self._overflow_rescan_due_at is not None and self._overflow_rescan_due_at <= now:
                    self._overflow_rescan_due_at = due_at
        with self._retry_state_lock:
            overflow_due = self._overflow_rescan_due_at is not None and self._clock() >= self._overflow_rescan_due_at
            if overflow_due:
                self._overflow_rescan_due_at = None
        if overflow_due:
            self._request_fresh_rescan()
        # The cursor advances in ``acknowledge``, over items the dispatcher
        # actually consumed -- never here, over everything merely discovered.
        # A page is routinely truncated by the class deficit, so advancing on
        # discovery skipped every file past the first admitted one and, because
        # the walk only ever moves forward, those files were never revisited:
        # a static root silently retained just its lexicographically-first
        # session. Revisiting an already-admitted file is explicitly harmless
        # (durable cursor/raw identity, see above), so the conservative
        # direction here is to re-discover, never to skip.
        retry_turn = self._retry_turn
        self._retry_turn = not retry_turn
        paths = self._due_retry_paths(limit) if retry_turn else []
        self._retry_page = bool(paths)
        if not paths:
            paths = self._discover_fresh_paths(limit)
        if not paths and not retry_turn:
            paths = self._due_retry_paths(limit)
            self._retry_page = bool(paths)
        if self._retry_page:
            self._retry_page_paths = tuple(paths)
            self._retry_page_pending = True
        items: list[IntakeItem] = []
        for path in paths:
            revision: str | None
            try:
                observed = path.lstat()
                size = observed.st_size if stat.S_ISREG(observed.st_mode) else 1
                revision = f"{observed.st_size}:{observed.st_mtime_ns}:{observed.st_ino}"
            except OSError:
                size = 1
                revision = None
            items.append(
                IntakeItem(
                    item_id=f"file:{path.absolute()}",
                    class_name=self.class_name,
                    payload=path,
                    estimated_cost=max(1, size),
                    revision=revision,
                )
            )
        return tuple(items)

    def _ledger_generation(self) -> tuple[int, int] | None:
        cursor = getattr(self.context.watcher, "_cursor", None)
        generation = getattr(cursor, "ops_ledger_generation", None)
        return generation() if callable(generation) else None

    def _source_entries(self) -> tuple[int, int] | None:
        """Bounded-memory signature of direct source entries."""
        internal_paths = self._internal_ledger_paths()
        try:
            with os.scandir(self.source.root) as entries:
                count = 0
                signature = 0
                for entry in entries:
                    path = Path(entry.path)
                    if path in internal_paths:
                        continue
                    if not (
                        (entry.is_dir() and self.source.admits_directory(path))
                        or (entry.is_file(follow_symlinks=False) and self.source.accepts(path))
                    ):
                        continue
                    # XOR makes the result independent of scandir order;
                    # count prevents an unchanged value after paired changes.
                    token = entry.name.encode("utf-8", "surrogateescape") + b"\0" + str(entry.inode()).encode()
                    signature ^= int.from_bytes(hashlib.blake2b(token, digest_size=16).digest())
                    count += 1
                return count, signature
        except OSError:
            return None

    def _internal_ledger_paths(self) -> frozenset[Path]:
        cursor = getattr(self.context.watcher, "_cursor", None)
        path = getattr(cursor, "_ops_db_path", None)
        if not isinstance(path, Path):
            return frozenset()
        return frozenset((path, Path(f"{path}-wal"), Path(f"{path}-shm")))

    def _owns_retry_path(self, path: Path) -> bool:
        try:
            mode = path.lstat().st_mode
        except OSError:
            return False
        if stat.S_ISLNK(mode):
            # A retired carrier's symlink target may escape the source. Page
            # the old cursor by its lexical owner so admission can exclude it.
            lexical = path.absolute()
            owners = (
                source
                for source in self.context.sources
                if lexical.is_relative_to(source.root.absolute()) and source.accepts(path)
            )
            return max(owners, key=lambda source: len(source.root.parts), default=None) is self.source
        return (
            stat.S_ISREG(mode)
            and deepest_source_for_path(path, self.context.sources) is self.source
            and self.source.accepts(path)
        )

    def _due_retry_paths(self, limit: int) -> list[Path]:
        now = self._clock()
        cursor = getattr(self.context.watcher, "_cursor", None)
        due_local: list[Path] = []
        with self._retry_state_lock:
            local_snapshot = tuple(self._fresh_retry_debt.items())
        stale_local: list[Path] = []
        vanished_local: list[Path] = []
        for path, due_at in local_snapshot:
            if len(due_local) >= limit:
                break
            if due_at > now:
                continue
            if not self.source.root.is_dir():
                break
            try:
                mode = path.lstat().st_mode
            except (FileNotFoundError, NotADirectoryError):
                stale_local.append(path)
                vanished_local.append(path)
                continue
            except OSError:
                # Nested permissions can obscure a retained carrier without
                # proving that it vanished or changed ownership.
                due_local.append(path)
                continue
            if not stat.S_ISREG(mode) or not self._owns_retry_path(path):
                stale_local.append(path)
                continue
            due_local.append(path)
        get_records = getattr(cursor, "get_records", None)
        records = get_records(due_local) if due_local and callable(get_records) else {}
        local_without_durable_row: list[Path] = []
        with self._retry_state_lock:
            for path in stale_local:
                self._fresh_retry_debt.pop(path, None)
            if vanished_local and self._overflow_rescan_due_at is None:
                self._overflow_rescan_due_at = now + _FILE_RETRY_DELAY_S
            for path in due_local:
                record = records.get(path)
                if self._has_durable_retry_record(record):
                    self._fresh_retry_debt.pop(path, None)
                else:
                    local_without_durable_row.append(path)
        due_local = local_without_durable_row
        if due_local and self._prefer_local_retry:
            with self._retry_state_lock:
                for path in due_local:
                    if path in self._fresh_retry_debt:
                        self._fresh_retry_debt.pop(path)
                        self._fresh_retry_debt[path] = now + _FILE_RETRY_DELAY_S
            self._local_retry_page = True
            self._prefer_local_retry = False
            return due_local
        due_retries = getattr(cursor, "list_due_retry_paths", None)
        if not callable(due_retries):
            candidates = ()
        else:
            # Acknowledged deferrals can sit before the filesystem walk's
            # position forever. Retry and fresh discovery alternate so a growing
            # source cannot starve either side; both walks stay bounded.
            if self._retry_through is None:
                high_water = getattr(cursor, "due_retry_high_water", None)
                self._retry_through = high_water(self.source.root) if callable(high_water) else None
            resume_positions = tuple(
                value for value in (self._retry_after, self._retry_skip_after) if value is not None
            )
            candidates = due_retries(
                self.source.root,
                after=max(resume_positions) if resume_positions else None,
                limit=limit,
                through=self._retry_through,
                owns=self._owns_retry_path,
            )
        if candidates:
            self._prefer_local_retry = True
            return list(candidates)
        self._retry_after = None
        self._retry_skip_after = None
        self._retry_through = None
        if due_local:
            with self._retry_state_lock:
                for path in due_local:
                    if path in self._fresh_retry_debt:
                        self._fresh_retry_debt.pop(path)
                        self._fresh_retry_debt[path] = now + _FILE_RETRY_DELAY_S
            self._local_retry_page = True
            self._prefer_local_retry = False
            return due_local
        return []

    def _consume_retry_item(self, item: IntakeItem, *, acknowledged: bool = False) -> None:
        if not self._retry_page or not isinstance(item.payload, (str, Path)):
            return
        path = Path(item.payload)
        if path not in self._retry_page_paths:
            return
        if self._local_retry_page:
            with self._retry_state_lock:
                if acknowledged:
                    self._fresh_retry_debt.pop(path, None)
                elif path in self._fresh_retry_debt:
                    # A partially planned page must move its attempted head
                    # behind healthy siblings left outside the byte budget.
                    due_at = self._fresh_retry_debt.pop(path)
                    self._fresh_retry_debt[path] = due_at
            self._retry_page_pending = False
            return
        position = str(path)
        if self._retry_after is None or position > self._retry_after:
            self._retry_after = position
        self._retry_page_pending = False

    async def admit(self, item: IntakeItem) -> AdmissionResult:
        outcomes = await self.admit_page((item,))
        return outcomes[item.item_id]

    async def admit_page(self, items: Sequence[IntakeItem]) -> dict[str, AdmissionResult]:
        """Admit a whole discovery page as one ingest batch, one hold.

        Every fixed cost of a live batch -- the writer hold, the six-tier
        bootstrap, the retention scan, the archive-wide convergence pass and
        the parse stage's own warm -- is paid once per call. Admitting one
        file per call made each of those a per-file cost over a corpus of
        tens of thousands of files, and gave preparation a single path per
        batch, which is no parallelism at all.

        The batch is one call; the *outcomes* stay per item, read back from
        ``LiveBatchMetrics`` by path, so the dispatcher's deficit,
        ``retry_after`` and isolation accounting are exactly what they were
        under per-file admission.
        """
        from polylogue.core.degraded import degraded_reason, is_fully_degraded

        if is_fully_degraded():
            # A structurally degraded daemon admits nothing and touches nothing:
            # no authority gate (it reads the archive's existence journals), no
            # cursor initialization, no selection. Every item stays retryable,
            # so nothing is lost once the degradation is cleared (#1003), and
            # costs nothing, so the class deficit is not charged for work that
            # did not happen, and before any page bookkeeping, so the page stays
            # unattempted. Discovery normally returns no page while degraded;
            # this covers a page discovered just before.
            degradation = degraded_reason()
            detail = f"archive ingest is degraded: {degradation.code if degradation is not None else 'unknown'}"
            # Forget the offered durable-retry page without rotating past it:
            # the retry position did not advance, so the next discovery after
            # recovery re-offers the same due items, tail included.
            if self._local_retry_page:
                # Offering a local retry page pushed its deadlines forward;
                # nothing was attempted, so the items are due again at once.
                now = self._clock()
                with self._retry_state_lock:
                    for offered in self._retry_page_paths:
                        if offered in self._fresh_retry_debt:
                            self._fresh_retry_debt[offered] = now
            self._retry_page_pending = False
            self._retry_page_paths = ()
            self._local_retry_page = False
            return {
                item.item_id: AdmissionResult(
                    AdmissionOutcome.RETRYABLE, reason=detail, actual_cost=0, unattempted=True
                )
                for item in items
            }
        # Bookkeeping below assumes the page is attempted; a late degradation
        # (after this entry check) must restore it, since nothing was.
        retry_after_before = self._retry_after
        self._deferred_without_cursor.clear()
        offered_local_retries = self._retry_page_paths if self._local_retry_page else ()
        for item in items:
            self._consume_retry_item(item)
            if not self._retry_page and isinstance(item.payload, (str, Path)):
                fresh_path = Path(item.payload)
                if fresh_path in self._fresh_page_paths:
                    self._fresh_attempted_paths.add(fresh_path)
        outcomes: dict[str, AdmissionResult] = {}
        batch: list[IntakeItem] = []
        nonregular_paths: list[Path] = []
        for item in items:
            path = Path(item.payload) if isinstance(item.payload, (str, Path)) else None
            if path is None:
                outcomes[item.item_id] = AdmissionResult(
                    AdmissionOutcome.TERMINAL, reason="file intake item has no path"
                )
                continue
            try:
                regular_file = stat.S_ISREG(path.lstat().st_mode)
            except OSError:
                regular_file = False
            else:
                if not regular_file:
                    nonregular_paths.append(path)
            if (
                not regular_file
                or deepest_source_for_path(path, self.context.sources) is not self.source
                or not self.source.accepts(path)
            ):
                outcomes[item.item_id] = AdmissionResult(
                    AdmissionOutcome.RETRYABLE, reason=f"source carrier unavailable or no longer owned: {path}"
                )
                continue
            batch.append(item)
        cursor = getattr(self.context.watcher, "_cursor", None)
        mark_excluded = getattr(cursor, "mark_excluded", None)
        try:
            # The source-selection/cursor authority gate runs before anything
            # in this page can mutate cursor state -- before initialization,
            # before the needs-work selection reads a row. A page whose
            # authority is refused must leave the archive exactly as it was.
            processor = getattr(self.context.watcher, "_batch_processor", None)
            if processor is not None:
                processor.require_cursor_authority(
                    [*nonregular_paths, *(Path(cast(Any, item.payload)) for item in batch)]
                )
                refused_paths: frozenset[Path] = getattr(processor, "_refused_paths", frozenset())
                if refused_paths:
                    # The precheck has handled these paths. A page with no
                    # ingest call must not leak the refusal into a later page.
                    processor._refused_paths = frozenset()
                nonregular_paths = [path for path in nonregular_paths if path not in refused_paths]
                for item in batch:
                    if Path(cast(Any, item.payload)) in refused_paths:
                        outcomes[item.item_id] = AdmissionResult(
                            AdmissionOutcome.RETRYABLE, reason="source carrier refused by cursor authority"
                        )
                batch = [item for item in batch if Path(cast(Any, item.payload)) not in refused_paths]
            if nonregular_paths and callable(mark_excluded):

                def retire_nonregular() -> None:
                    for path in nonregular_paths:
                        try:
                            observed = path.lstat()
                        except OSError:
                            continue
                        if not stat.S_ISREG(observed.st_mode):
                            mark_excluded(path, observed_stat=observed)

                await self.context.run_write("daemon.intake.retire_nonregular_cursor", retire_nonregular)
            if not batch:
                return outcomes
            # Cursor initialization precedes every read of cursor state, and
            # takes the writer admission to do it: the selection below reads
            # the cursor rows, so doing it first would touch (and create) the
            # store outside the writer lease.
            cursor = getattr(self.context.watcher, "_cursor", None)
            if cursor is not None:
                if not self.context.watcher.has_write_coordinator:
                    raise UnleasedWriteError(
                        "watcher.intake.cursor_initialize writes the cursor store and requires the daemon write coordinator"
                    )
                await self.context.watcher._run_writer_sync("watcher.intake.cursor_initialize", cursor.initialize)
            # Narrow the page before it costs anything more: a bounded walk
            # re-offers files whose cursor already accounts for them, and
            # handing those to the batch buys a planning pass per file per
            # pass for no admission.
            paths = [Path(cast(Any, item.payload)) for item in batch]
            classify = getattr(self.context.watcher, "classify_ingest_candidates", None)
            pending_retry: set[Path] = set()
            if not callable(classify):
                needed = set(paths)
            else:
                # The selection is a read that can decide to write: an
                # incomplete-append deferral, an archived-cursor
                # reconciliation and a device-drift rebase all correct cursor
                # rows in place. It runs with the writer released (its
                # reconciliation hashes whole files); each of those writes is
                # admitted onto the writer on its own, after re-checking the
                # observation it was decided from.
                if self.context.watcher.has_write_coordinator:
                    selected, pending = await self.context.watcher.classify_ingest_candidates_off_writer(paths)
                else:
                    selected, pending = classify(paths)
                needed = set(selected)
                pending_retry = set(pending)
            skipped = [item for item in batch if Path(cast(Any, item.payload)) not in needed]
            for item in skipped:
                if Path(cast(Any, item.payload)) in pending_retry:
                    # A scheduled retry is owed work, not an admission: the
                    # item stays unacknowledged until the retry runs.
                    outcomes[item.item_id] = AdmissionResult(
                        AdmissionOutcome.DEFERRED,
                        reason=f"source retry pending: {item.payload}",
                        actual_cost=0,
                    )
                    continue
                outcomes[item.item_id] = AdmissionResult(
                    AdmissionOutcome.DUPLICATE, actual_cost=max(1, int(item.estimated_cost))
                )
            batch = [item for item in batch if Path(cast(Any, item.payload)) in needed]
            if not batch:
                return outcomes
            paths = [Path(cast(Any, item.payload)) for item in batch]
            metrics = await self.context.watcher._ingest_files(
                paths,
                queued_file_count=len(paths) + len(skipped),
                skipped_file_count=len(skipped),
                whole_archive_convergence=False,
            )
        except ArchiveStorageFaultError as exc:
            # Archive storage refused the page's writes (full disk, I/O error,
            # corrupt page, read-only mount). No item is at fault and none was
            # marked failed or quarantined; every one stays retryable. This is
            # an operator condition, so it is reported at ERROR with the fault
            # kind as its reason rather than as ordinary admission churn.
            emit(
                "daemon.intake.page_refused",
                level=ERROR,
                outcome="error",
                reason=exc.reason,
                component=self.class_name,
                files=len(batch),
                error_type=type(exc.__cause__ or exc).__name__,
                error_detail=str(exc),
            )
            for item in batch:
                outcomes[item.item_id] = AdmissionResult(AdmissionOutcome.RETRYABLE, reason=str(exc))
            return outcomes
        except (OSError, ValueError, RuntimeError) as exc:
            raise_if_operation_cancelled(exc)
            # A cursor-authority refusal lands here too: it is retryable for
            # every item in the page, and nothing in the archive changed. Say
            # so once per page: a class that reports only ``retried`` counts
            # is otherwise a silent refusal with no reason anywhere. Anything
            # else escapes to the dispatcher, which reports it the same way.
            reason = f"{type(exc).__name__}: {exc}"
            emit(
                "daemon.intake.page_refused",
                level=WARNING,
                outcome="degraded",
                reason="page_admission_failed",
                component=self.class_name,
                files=len(batch),
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            # A cursor-authority refusal clears once the frontier proof does;
            # any other error here is classified like one that escaped.
            transient = isinstance(exc, CursorAuthorityBlockedError) or is_transient_admission_error(exc)
            for item in batch:
                outcomes[item.item_id] = AdmissionResult(AdmissionOutcome.RETRYABLE, reason=reason, transient=transient)
            return outcomes

        stale_cursor_writes = int(getattr(metrics, "stale_cursor_write_count", 0) or 0)
        stale_cursor_paths = {str(path) for path in (getattr(metrics, "stale_cursor_paths", ()) or ())}
        if stale_cursor_writes and (not stale_cursor_paths or len(stale_cursor_paths) > stale_cursor_writes):
            # An incomplete per-path report cannot establish which rows lost
            # freshness, so retain the old fail-closed page result.
            for item in batch:
                outcomes[item.item_id] = AdmissionResult(
                    AdmissionOutcome.RETRYABLE, reason="source cursor write was stale"
                )
            return outcomes

        succeeded = {str(path) for path in (getattr(metrics, "succeeded_paths", ()) or ())}
        failed = set(getattr(metrics, "failed_paths", ()) or ())
        deferred = set(getattr(metrics, "deferred_paths", ()) or ())
        get_records = getattr(cursor, "get_records", None)
        deferred_records = (
            await asyncio.to_thread(get_records, tuple(Path(path) for path in deferred))
            if deferred and callable(get_records)
            else {}
        )
        self._deferred_without_cursor.update(
            Path(path) for path in deferred if not self._has_durable_retry_record(deferred_records.get(Path(path)))
        )
        # ``failed_paths`` carries the retry projection, deferrals included.
        # A deferral is its own outcome, so it must not be reported as a
        # failure here.
        failed -= deferred
        excluded_by_path = dict(getattr(metrics, "excluded_paths", {}) or {})
        settled = dict(getattr(metrics, "settled_exclusion_paths", {}) or {})
        partial_by_path = dict(getattr(metrics, "partial_admission_paths", {}) or {})
        if not succeeded:
            # This route calls ``_ingest_files`` directly, so the watcher's
            # own ``_log_ingest_metrics`` never runs for it and the
            # "admitted nothing" line was invisible on the intake path.
            _log_ingest_metrics(f"live.intake: {self.class_name}", metrics)

        # Reconcile the batch's measured read against the items that actually
        # produced it. An append-mode file reads far less than its size, so
        # charging every item its full estimate would overstate the class's
        # spend; distributing the measured total over the admitted items in
        # proportion to their estimates keeps the class budget denominated in
        # bytes actually read.
        # The degraded skip is a flag, not a refused-byte bucket: a batch of
        # empty files refuses zero bytes and still attempted nothing.
        degraded_skip = bool(getattr(metrics, "daemon_degraded_skip", False))
        unattempted_is_retryable = bool(getattr(metrics, "time_budget_exceeded", False)) or degraded_skip
        read_bytes = int(getattr(metrics, "source_payload_read_bytes", 0) or 0)
        estimated_total = sum(max(1, int(item.estimated_cost)) for item in batch)
        for item in batch:
            key = str(Path(cast(Any, item.payload)))
            item_estimate = max(1, int(item.estimated_cost))
            actual_cost = max(1, round(read_bytes * item_estimate / estimated_total)) if read_bytes else item_estimate
            if key in stale_cursor_paths:
                outcomes[item.item_id] = AdmissionResult(
                    AdmissionOutcome.RETRYABLE, reason="source cursor write was stale"
                )
            elif key in settled:
                # Acquired, but nothing admissible came of it (no session, or
                # corrupt input): the raw carries the typed terminal outcome,
                # and reporting ADMITTED counted a file that produced nothing
                # (polylogue-xf8qp).
                outcomes[item.item_id] = AdmissionResult(
                    AdmissionOutcome.EXCLUDED,
                    reason=f"{settled[key]}: {key}",
                    actual_cost=actual_cost,
                )
            elif key in succeeded:
                # A stable capture whose final record is truncated admits its
                # complete records; the result carries what it left out, so it
                # is never reported as a plain success (polylogue-xf8qp).
                partial = partial_by_path.get(key)
                outcomes[item.item_id] = AdmissionResult(
                    AdmissionOutcome.ADMITTED,
                    reason=(
                        f"{partial.reason}: {partial.complete_record_count} complete record(s) admitted, "
                        f"tail from byte {partial.complete_prefix_bytes} of {partial.source_bytes} left out"
                        if partial is not None
                        else None
                    ),
                    actual_cost=actual_cost,
                    partial=partial,
                )
            elif excluded_by_path.get(key) in {REFUSED_UNATTEMPTED, REFUSED_UNATTEMPTED_TIME_BUDGET}:
                self._fresh_attempted_paths.discard(Path(cast(Any, item.payload)))
                outcomes[item.item_id] = AdmissionResult(
                    AdmissionOutcome.RETRYABLE,
                    reason=f"source admission left {key} unattempted: {excluded_by_path[key]}",
                    actual_cost=0,
                    unattempted=True,
                )
            elif key in excluded_by_path:
                # polylogue-onbz3: a durable refusal is not "already admitted
                # under this identity". Reporting DUPLICATE here advanced the
                # cursor and counted the pass as progress.
                outcomes[item.item_id] = AdmissionResult(
                    AdmissionOutcome.EXCLUDED,
                    reason=f"source admission excluded {key}: {excluded_by_path[key]}",
                    actual_cost=item_estimate,
                )
            elif key in deferred:
                outcomes[item.item_id] = AdmissionResult(
                    AdmissionOutcome.DEFERRED,
                    reason=f"source admission deferred {key}",
                    actual_cost=item_estimate,
                )
            elif key in failed:
                outcomes[item.item_id] = AdmissionResult(
                    AdmissionOutcome.RETRYABLE, reason=f"source admission failed: {key}"
                )
            elif unattempted_is_retryable:
                # The pass ran out of its declared time budget, or refused
                # the whole batch while degraded: this item was never
                # attempted, so it is ordinary backlog, not a re-seen one.
                self._fresh_attempted_paths.discard(Path(cast(Any, item.payload)))
                outcomes[item.item_id] = AdmissionResult(
                    AdmissionOutcome.RETRYABLE,
                    reason=f"source admission left {key} unattempted",
                    actual_cost=0,
                    unattempted=True,
                )
            elif not succeeded:
                # A zero-success, zero-failure batch supplied no per-item
                # verdict. The watcher retries this same shape; classifying
                # it as DUPLICATE here acknowledged the fair-intake cursor
                # and silently dropped the item.
                outcomes[item.item_id] = AdmissionResult(
                    AdmissionOutcome.RETRYABLE,
                    reason=f"source admission produced no outcome: {key}",
                )
            else:
                # Offered and attempted, with nothing new to admit under this
                # identity: the ordinary re-discovery of an already-ingested
                # file. Acknowledgeable, never progress.
                outcomes[item.item_id] = AdmissionResult(AdmissionOutcome.DUPLICATE, actual_cost=item_estimate)

        if degraded_skip:
            self._retry_after = retry_after_before
            if offered_local_retries:
                now = self._clock()
                with self._retry_state_lock:
                    for offered in offered_local_retries:
                        if offered in self._fresh_retry_debt:
                            self._fresh_retry_debt[offered] = now
        admitted_paths = [path for path in paths if str(path) in succeeded and str(path) not in settled]
        if admitted_paths and not is_fully_degraded():
            # A later file in this batch may have degraded the daemon; derived
            # follow-up converges from durable evidence once it recovers.
            converge_embeddings = getattr(self.context.watcher, "_converge_embeddings_off_writer", None)
            if callable(converge_embeddings):
                await converge_embeddings(admitted_paths)
            converge_profiles = getattr(self.context.watcher, "_converge_session_profiles_off_writer", None)
            if callable(converge_profiles):
                await converge_profiles(tuple(getattr(metrics, "changed_session_ids", ()) or ()))
        return outcomes

    async def acknowledge(self, item: IntakeItem) -> None:
        # Files remain retained source carriers.  The live batch's durable
        # cursor/raw commit is the acknowledgement projection.  The scheduling
        # cursor advances here, monotonically, so an item dropped by the class
        # deficit is rediscovered on the next pass instead of being skipped.
        # Retry rows may be ahead of ordinary discovery. They cannot advance
        # that walk past files it has not offered yet.
        if isinstance(item.payload, (str, Path)) and Path(item.payload) in self._deferred_without_cursor:
            # The batch admitted the deferral, not unread source bytes. Keep
            # its unacknowledged fresh page/local retry until a coordinate can
            # be captured; a durable cursor does not yet own this obligation.
            self._deferred_without_cursor.discard(Path(item.payload))
            self._consume_retry_item(item)
            return
        if self._retry_page:
            self._consume_retry_item(item, acknowledged=True)
            return
        payload = item.payload
        if isinstance(payload, (str, Path)):
            with self._retry_state_lock:
                self._fresh_retry_debt.pop(Path(payload), None)
            self._fresh_attempted_paths.discard(Path(payload))
            if Path(payload) in self._fresh_page_paths:
                self._fresh_page_paths = tuple(path for path in self._fresh_page_paths if path != Path(payload))
                self._fresh_page_pending = bool(self._fresh_page_paths)
            position = str(payload)
            if self._after is None or position > self._after:
                self._after = position


@dataclass(frozen=True, slots=True)
class SubUnitHaltPolicy:
    """Halt authority for the units *inside* one multiplexed intake class.

    ``configured_local`` is one dispatcher class over every configured
    source. Halting the class for one source's terminal refusal would stop
    every sibling; not halting anything leaves the planner selecting a dead
    source's files forever, which is polylogue-kqrbw: a source logged
    "refusing further ingest until restart" once, and for the rest of the run
    the planner kept selecting its files and 926 empty chunks each took the
    writer lease.

    The two callables are supplied by the daemon composition root, which owns
    the durable halt registry. Keeping them as callables is deliberate: this
    module is product-layer and must not acquire an import edge onto the
    daemon's halt store to ask a yes/no question.
    """

    is_halted: Callable[[str], bool]
    """Whether *unit name* is durably halted, asked at planning time."""

    halt: Callable[[str, str], None]
    """Record *unit name* as terminally refusing, with its reason."""


class MultiplexIntakeAdapter(IntakeAdapter):
    """Bounded round-robin over several configured local file roots.

    The dispatcher has one logical ``configured_local`` class.  Keeping one
    adapter per root would duplicate that class identity and make construction
    order accidentally authoritative, so this adapter owns only a disposable
    root position and delegates each item to its source adapter.
    """

    def __init__(self, adapters: Sequence[IntakeAdapter], *, halts: SubUnitHaltPolicy | None = None) -> None:
        if not adapters:
            raise ValueError("at least one file adapter is required")
        self.adapters = tuple(adapters)
        self._halts = halts
        self._next = 0
        self._by_item: dict[str, IntakeAdapter] = {}

    def schedulable_adapters(self) -> tuple[IntakeAdapter, ...]:
        """Sub-adapters this pass may plan work for.

        A halted source is excluded *here*, where work is selected. Its files
        are never discovered, so nothing downstream forms a batch from them
        or takes the writer lease on their behalf.

        Explicit terminal refusals use the durable policy. Structural ingest
        faults use the process-local source registry: they stop selection for
        this process, but are rechecked after restart. Copying them into a
        durable terminal refusal would prevent a repaired source from ever
        being tried by the next daemon.
        """
        from polylogue.core.source_halts import source_halt

        schedulable: list[IntakeAdapter] = []
        for adapter in self.adapters:
            unit = _sub_unit_name(adapter)
            if unit is None:
                schedulable.append(adapter)
                continue
            if self._halts is not None and self._halts.is_halted(unit):
                from polylogue.daemon.discovery_progress import abandon_discovery

                abandon_discovery(adapter)
                continue
            if source_halt(unit) is not None:
                from polylogue.daemon.discovery_progress import abandon_discovery

                abandon_discovery(adapter)
                continue
            schedulable.append(adapter)
        return tuple(schedulable)

    @property
    def discovery_pending(self) -> bool:
        return any(bool(getattr(adapter, "discovery_pending", False)) for adapter in self.schedulable_adapters())

    @property
    def retry_due_in_s(self) -> float | None:
        due = (getattr(adapter, "retry_due_in_s", None) for adapter in self.schedulable_adapters())
        deadlines = tuple(value for value in due if value is not None)
        return min(deadlines) if deadlines else None

    async def discover(self, *, limit: int) -> Sequence[IntakeItem]:
        if limit <= 0:
            return ()
        # Ownership is needed only between this page's discovery and its
        # admission/acknowledgement; prior unacknowledged pages are rediscovered.
        self._by_item.clear()
        adapters = self.schedulable_adapters()
        if not adapters:
            return ()
        result: list[IntakeItem] = []
        owners: list[IntakeAdapter] = []
        start = self._next
        for offset in range(len(adapters)):
            adapter = adapters[(start + offset) % len(adapters)]
            remaining = limit - len(result)
            if remaining <= 0:
                break
            try:
                page = await adapter.discover(limit=remaining)
            except WalkRefusedError as exc:
                # One root's absence (an uninstalled or relocated tool) is
                # that sub-unit's own retryable gap, not this class's: a
                # sibling root that IS present must still offer its own
                # pending files this pass instead of the round-robin
                # aborting on the first unavailable root it visits.
                unit = _sub_unit_name(adapter)
                emit(
                    "daemon.intake.sub_unit_walk_refused",
                    level=WARNING,
                    outcome="degraded",
                    reason="walk_refused",
                    component=unit or "unknown",
                    error_detail=str(exc),
                )
                continue
            result.extend(page)
            owners.extend([adapter] * len(page))
        self._next = (start + 1) % len(adapters)
        self._by_item.update({item.item_id: adapter for item, adapter in zip(result, owners, strict=True)})
        return tuple(result[:limit])

    def _isolate_sub_unit(self, adapter: IntakeAdapter, result: AdmissionResult) -> AdmissionResult:
        """Convert one sub-unit's class-terminal refusal into that unit's halt.

        The refusal is real but its blast radius is not the class. Recording
        it against the source and returning a per-item ``TERMINAL`` keeps the
        siblings in this class draining, which is the whole reason they share
        a class identity rather than an outcome.
        """
        if result.outcome is not AdmissionOutcome.CLASS_TERMINAL:
            return result
        unit = _sub_unit_name(adapter)
        if self._halts is None or unit is None:
            return result
        reason = result.reason or "source reported terminal failure"
        self._halts.halt(unit, reason)
        from polylogue.daemon.discovery_progress import abandon_discovery

        abandon_discovery(adapter)
        emit(
            "daemon.intake.source_halted",
            level=WARNING,
            outcome="refused",
            reason="terminal_refusal",
            component=unit,
            error_detail=reason,
        )
        return AdmissionResult(AdmissionOutcome.TERMINAL, reason=f"{unit}: {reason}")

    async def admit(self, item: IntakeItem) -> AdmissionResult:
        adapter = self._by_item.get(item.item_id)
        if adapter is None:
            return AdmissionResult(AdmissionOutcome.RETRYABLE, reason="configured source item lost its adapter")
        return self._isolate_sub_unit(adapter, await adapter.admit(item))

    async def admit_page(self, items: Sequence[IntakeItem]) -> dict[str, AdmissionResult]:
        """Split one page along source ownership and admit each part as a page.

        One root's files are one ingest batch. Splitting here rather than
        admitting item by item is what keeps the per-batch fixed cost paid
        once per root per pass, and the parts are disjoint, so the per-item
        outcomes recombine without ambiguity.
        """
        outcomes: dict[str, AdmissionResult] = {}
        groups: dict[int, tuple[IntakeAdapter, list[IntakeItem]]] = {}
        for item in items:
            adapter = self._by_item.get(item.item_id)
            if adapter is None:
                outcomes[item.item_id] = AdmissionResult(
                    AdmissionOutcome.RETRYABLE, reason="configured source item lost its adapter"
                )
                continue
            groups.setdefault(id(adapter), (adapter, []))[1].append(item)
        for adapter, group in groups.values():
            admit_page = getattr(adapter, "admit_page", None)
            if admit_page is None:
                for item in group:
                    outcomes[item.item_id] = self._isolate_sub_unit(adapter, await adapter.admit(item))
                continue
            for item_id, result in (await admit_page(tuple(group))).items():
                outcomes[item_id] = self._isolate_sub_unit(adapter, result)
        return outcomes

    async def acknowledge(self, item: IntakeItem) -> None:
        adapter = self._by_item.pop(item.item_id, None)
        if adapter is not None:
            await adapter.acknowledge(item)


def _sub_unit_name(adapter: IntakeAdapter) -> str | None:
    """The halt identity of one sub-adapter, or ``None`` if it has none.

    File adapters carry the configured source they read; anything else in a
    multiplexed class has no unit of its own and is never halted separately.
    """
    source = getattr(adapter, "source", None)
    name = getattr(source, "name", None)
    return name if isinstance(name, str) and name else None


class CallbackIntakeAdapter(IntakeAdapter):
    """One-shot bounded remote/raw adapter around an existing domain route.

    The wrapped route reports a changed-row count, not payload bytes, so it
    has no honest estimate in the dispatcher's unit. It therefore charges
    ``UNMEASURABLE_INTAKE_COST_BYTES`` -- a full budget share -- rather than
    the literal one byte that let a large remote sync starve its siblings
    (polylogue-swicx).
    """

    def __init__(
        self,
        class_name: str,
        callback: Callable[[], Awaitable[AdmissionResult | int] | AdmissionResult | int],
        *,
        estimated_cost: int = UNMEASURABLE_INTAKE_COST_BYTES,
        persistent: bool = True,
        poll_interval_s: float = 0.0,
    ) -> None:
        self.class_name = class_name
        self.callback = callback
        self.estimated_cost = max(1, estimated_cost)
        self.persistent = persistent
        self._poll_interval_s = max(0.0, poll_interval_s)
        self._pending = True
        self._next_poll_at = 0.0

    async def discover(self, *, limit: int) -> Sequence[IntakeItem]:
        if limit <= 0 or not self._pending or time.monotonic() < self._next_poll_at:
            return ()
        return (IntakeItem(self.class_name, self.class_name, estimated_cost=self.estimated_cost),)

    async def admit(self, item: IntakeItem) -> AdmissionResult:
        try:
            changed = self.callback()
            if isinstance(changed, Awaitable):
                changed = await changed
            if isinstance(changed, AdmissionResult):
                return changed
            self._pending = self.persistent
            return AdmissionResult(
                AdmissionOutcome.ADMITTED if int(changed) else AdmissionOutcome.DUPLICATE,
                actual_cost=self.estimated_cost,
            )
        except Exception as exc:
            return AdmissionResult(
                AdmissionOutcome.RETRYABLE,
                # The exception type leads so `_failure_signature`, which
                # takes the text before the first colon, keys on it rather
                # than on the adapter's class name shared by every failure.
                reason=f"{type(exc).__name__}: {self.class_name}: {exc}",
                transient=is_transient_admission_error(exc),
            )

    async def acknowledge(self, item: IntakeItem) -> None:
        if not self.persistent:
            self._pending = False
        else:
            # Completion, not discovery, starts the next polling interval.
            # Failed attempts remain governed by the dispatcher's retry policy.
            self._next_poll_at = time.monotonic() + self._poll_interval_s


class DriveIntakeAdapter(CallbackIntakeAdapter):
    """A listing checkpoint yields fairly; only completion starts hourly polling."""

    def __init__(self, callback: Callable[[], Awaitable[DriveCatchupReport] | DriveCatchupReport]) -> None:
        super().__init__("configured_remote", lambda: 0, poll_interval_s=3600.0)
        self._drive_callback = callback
        self._drive_state = DriveCatchupState.UNKNOWN

    @property
    def discovery_pending(self) -> bool:
        return self._drive_state is DriveCatchupState.PENDING

    async def admit(self, item: IntakeItem) -> AdmissionResult:
        try:
            report = self._drive_callback()
            if isinstance(report, Awaitable):
                report = await report
            self._drive_state = report.state
            if report.state is DriveCatchupState.RETRYABLE:
                return AdmissionResult(
                    AdmissionOutcome.RETRYABLE,
                    reason="drive_retryable:" + ",".join(report.gaps),
                    actual_cost=self.estimated_cost,
                )
            if report.state is DriveCatchupState.COMPLETE:
                return AdmissionResult(
                    AdmissionOutcome.ADMITTED if report.changed_count else AdmissionOutcome.DUPLICATE,
                    actual_cost=self.estimated_cost,
                )
            return AdmissionResult(
                AdmissionOutcome.DEFERRED, reason="drive_" + report.state.value, actual_cost=self.estimated_cost
            )
        except Exception as exc:
            self._drive_state = DriveCatchupState.RETRYABLE
            return AdmissionResult(
                AdmissionOutcome.RETRYABLE,
                reason=f"{type(exc).__name__}: configured_remote: {exc}",
                transient=is_transient_admission_error(exc),
            )

    async def acknowledge(self, item: IntakeItem) -> None:
        # Pending is a cooperative continuation, not a failed attempt or a
        # settled poll. Blocked/unknown reobserve evidence on a short cadence.
        delay = (
            3600.0
            if self._drive_state is DriveCatchupState.COMPLETE
            else 60.0
            if self._drive_state in {DriveCatchupState.BLOCKED, DriveCatchupState.UNKNOWN}
            else 0.0
        )
        self._next_poll_at = time.monotonic() + delay


class RawMaterializationIntakeAdapter(IntakeAdapter):
    """Bounded raw-id discovery delegated to the canonical derivation route."""

    def __init__(
        self,
        discover_ids: Callable[[int], Awaitable[Sequence[tuple[str, int]]] | Sequence[tuple[str, int]]],
        admit_id: Callable[[str], Awaitable[AdmissionResult | int] | AdmissionResult | int],
        *,
        suspended: Callable[[], bool] | None = None,
        discovery_pending: Callable[[], bool] | None = None,
    ) -> None:
        self._discover_ids = discover_ids
        self._admit_id = admit_id
        self._suspended = suspended
        self._discovery_pending = discovery_pending

    @property
    def discovery_pending(self) -> bool:
        if self._suspended is not None and self._suspended():
            return False
        return self._discovery_pending is not None and self._discovery_pending()

    async def discover(self, *, limit: int) -> Sequence[IntakeItem]:
        if self._suspended is not None and self._suspended():
            return ()
        raw_ids = self._discover_ids(limit)
        if isinstance(raw_ids, Awaitable):
            raw_ids = await raw_ids
        return tuple(
            IntakeItem(raw_id, "raw_materialization", estimated_cost=max(1, int(cost))) for raw_id, cost in raw_ids
        )

    async def admit(self, item: IntakeItem) -> AdmissionResult:
        try:
            changed = self._admit_id(item.item_id)
            if isinstance(changed, Awaitable):
                changed = await changed
            if isinstance(changed, AdmissionResult):
                return changed
            return AdmissionResult(
                AdmissionOutcome.ADMITTED if int(changed) else AdmissionOutcome.DUPLICATE,
                actual_cost=item.estimated_cost,
            )
        except Exception as exc:
            return AdmissionResult(
                AdmissionOutcome.RETRYABLE,
                # The exception type leads, as elsewhere, so
                # `_failure_signature` (text before the first colon) keys on
                # it rather than on the constant "raw materialization" label
                # every failure of this class would otherwise share.
                reason=f"{type(exc).__name__}: raw materialization: {exc}",
                transient=is_transient_admission_error(exc),
            )

    async def acknowledge(self, item: IntakeItem) -> None:
        return None


@dataclass(frozen=True, slots=True)
class _RawDiscoveryBinding:
    """The canonical frame identity that makes a continuation reusable."""

    archive_root: str
    source_revision: str
    recipe_version: str


class RawMaterializationDiscovery:
    """One process-local, bounded traversal of canonical raw obligations.

    The cursor is an intake scheduling hint. It never records validity or an
    admission result, so losing it merely starts a new bounded traversal.

    Two lanes feed it. The *sweep* lane pages through the canonical required-key
    space and retains its continuation. The *arrival* lane serves raws admitted
    since this process last looked, read straight off the append-only
    ``raw_sessions`` rowid frontier. A new raw can sort lexically before the
    sweep cursor, so without the arrival lane it would wait for a whole
    traversal to wrap; but restarting the sweep for it -- what this class used
    to do -- meant that under a sustained arrival rate the cursor was reset
    before it ever advanced past its first page, and every obligation behind
    that page was never reached. Serving the new raws directly reaches them
    without discarding the sweep's position, and the lanes alternate so an
    unbroken stream of arrivals cannot stall the sweep either.
    """

    def __init__(self, archive_root: Path) -> None:
        self._archive_root = archive_root
        self._binding: _RawDiscoveryBinding | None = None
        self._cursor: str | None = None
        #: The continuation start of the page currently being offered and the
        #: keys it still owed when it was last inspected. A page at the same
        #: cursor that now owes a key it did not owe before is new work, not a
        #: stalled head.
        self._held_page: tuple[str | None, tuple[str, ...]] | None = None
        self._frontier: int = 0
        #: Which lane leads the next discovery call. Arrivals, the sweep and
        #: any queued project dependents rotate, so no lane -- not even a
        #: project scan whose every page owes work -- starves the others.
        self._lane_turn = 0
        #: Sessions whose enrichment evidence just arrived: a project's
        #: ``sessions-index.json`` names the transcripts beside it. Inspected
        #: on later calls, after the evidence itself was admitted, so a
        #: title curated after its transcript was written converges promptly.
        #: A scheduling hint only: the sweep re-inspects every raw anyway, and
        #: inspection compares each output's evidence binding with the
        #: evidence the archive holds (install-wide history and thread state
        #: converge through that sweep).
        #: Pending project scans: (project directory, last served path,
        #: last served rowid). Paged by ``_dependents_selected``.
        self._evidence_projects: deque[tuple[str, str, int]] = deque()
        #: Monotonic start of the sweep in progress, and the end and duration
        #: of the last completed one, which pace the next (see
        #: ``_RAW_RESWEEP_REST_FACTOR``).
        self._sweep_started_at: float | None = None
        self._last_sweep_finished_at: float | None = None
        self._last_sweep_duration_s = 0.0

    @property
    def discovery_pending(self) -> bool:
        """An unfinished bounded traversal due for prompt paging, not a claim that rows owe work."""
        if self._evidence_projects:
            return True
        if self._cursor is None:
            return False
        if self._last_sweep_finished_at is None:
            return True
        rest_s = _RAW_RESWEEP_REST_FACTOR * self._last_sweep_duration_s
        return time.monotonic() >= self._last_sweep_finished_at + rest_s

    def _raw_frontier(self) -> int:
        """Return the durable high-water mark for admitted raw observations."""
        from polylogue.storage.sqlite.connection_profile import open_readonly_connection

        source_db = self._archive_root / "source.db"
        with closing(open_readonly_connection(source_db, timeout=5.0)) as conn:
            row = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM raw_sessions").fetchone()
        return int(row[0]) if row is not None else 0

    def _arrived_since_frontier(self, limit: int) -> tuple[tuple[str, ...], int]:
        """Raw ids admitted after the recorded frontier, oldest first.

        ``raw_sessions`` is append-only for admitted observations, so a bounded
        ``rowid >`` page is the whole arrival set and its last rowid is the new
        frontier. Publishing changes to an existing raw does not advance the
        rowid and therefore produces no arrival.
        """
        from polylogue.storage.sqlite.connection_profile import open_readonly_connection

        source_db = self._archive_root / "source.db"
        with closing(open_readonly_connection(source_db, timeout=5.0)) as conn:
            rows = conn.execute(
                "SELECT rowid, raw_id FROM raw_sessions WHERE rowid > ? ORDER BY rowid LIMIT ?",
                (self._frontier, limit),
            ).fetchall()
        if not rows:
            return (), self._frontier
        return tuple(str(row[1]) for row in rows), int(rows[-1][0])

    def discover_pending_raw_ids(self, limit: int) -> tuple[tuple[str, int], ...]:
        """Inspect one bounded canonical page and retain its continuation.

        Raw discovery is deliberately a read-only derivation traversal.  The old
        census preview was a second backlog authority and could miss observations
        whose parser census was complete but whose derived output had been lost.
        A discovery call inspects one bounded page before admitting only non-valid
        observations. The next call resumes after that page, even if it contained
        only valid or already-isolated observations. The
        payload size is the scheduler's cost estimate, read from the durable raw
        rows rather than guessed by the adapter.
        """
        if limit <= 0:
            return ()
        if not (self._archive_root / "source.db").exists():
            self._binding = None
            self._cursor = None
            self._held_page = None
            self._evidence_projects.clear()
            # polylogue-f7pdm: a fresh archive root simply has no raw tier
            # yet. That is an empty page, not a failure: the tier appears as
            # soon as the first acquisition commits and the next pass
            # discovers it. Reporting it as an error here is what made the
            # daemon latch raw materialization off for its whole lifetime and
            # log a whale-schedule warning every 30 s on an empty root.
            return ()
        from polylogue.operations.raw_observation_derivation import (
            raw_observation_inspection_frame,
        )
        from polylogue.storage.derived.raw import RAW_OBSERVATION_DOMAIN, RawObservationInspection

        frame = raw_observation_inspection_frame(self._archive_root)
        binding = _RawDiscoveryBinding(
            archive_root=frame.archive_root,
            source_revision=frame.source_revision,
            recipe_version=frame.recipe_version(RAW_OBSERVATION_DOMAIN),
        )
        if binding != self._binding:
            # A replacement generation invalidates the traversal itself, not
            # merely its position: start a fresh sweep and treat everything
            # already durable as swept rather than as an arrival.
            self._binding = binding
            self._cursor = None
            self._held_page = None
            self._frontier = self._raw_frontier()
            self._lane_turn = 0
            # A new generation owes a fresh sweep now, not after a rest.
            self._sweep_started_at = None
            self._last_sweep_finished_at = None

        inspected_limit = min(limit, _RAW_DISCOVERY_INSPECTION_LIMIT)
        adapter = RawObservationInspection(self._archive_root)
        # Queued project dependents join the rotation only while a scan is
        # pending, so arrivals and the sweep otherwise keep alternating.
        rotation: tuple[Callable[[Any, Any, int], tuple[str, ...]], ...] = (
            self._arrival_selected,
            self._sweep_selected,
            *((self._dependents_selected,) if self._evidence_projects else ()),
        )
        turn = self._lane_turn % len(rotation)
        self._lane_turn += 1
        for lane in (*rotation[turn:], *rotation[:turn]):
            selected = lane(frame, adapter, inspected_limit)
            if selected:
                return self._with_costs(selected)
        return ()

    def _arrival_selected(self, frame: Any, adapter: Any, limit: int) -> tuple[str, ...]:
        page, frontier = self._arrived_since_frontier(limit)
        self._frontier = frontier
        if not page:
            return ()
        self._queue_evidence_dependents(page)
        statuses = adapter.inspect(frame, page)
        refusals = adapter.terminal_decode_refusals(page)
        return tuple(raw_id for raw_id in page if statuses.get(raw_id) != "valid" and raw_id not in refusals)

    def _dependents_selected(self, frame: Any, adapter: Any, limit: int) -> tuple[str, ...]:
        """Serve at most ``limit`` transcripts of the queued project scans.

        Each queued scan is a continuation over one project's transcripts, so
        a project with a long retained history is paged like the sweep rather
        than listed whole when its index arrives.
        """
        if not self._evidence_projects:
            return ()
        from polylogue.storage.sqlite.connection_profile import open_readonly_connection

        page: list[str] = []
        with closing(open_readonly_connection(self._archive_root / "source.db", timeout=5.0)) as conn:
            while self._evidence_projects and len(page) < limit:
                project, after_path, after_rowid = self._evidence_projects[0]
                wanted = limit - len(page)
                rows = conn.execute(
                    "SELECT raw_id, source_path, rowid FROM raw_sessions "
                    "WHERE source_path >= ? AND source_path < ? AND source_path LIKE '%.jsonl' "
                    "AND (source_path > ? OR (source_path = ? AND rowid > ?)) "
                    "ORDER BY source_path, rowid LIMIT ?",
                    (project + "/", project + "0", after_path, after_path, after_rowid, wanted),
                ).fetchall()
                page.extend(str(row[0]) for row in rows)
                if len(rows) < wanted:
                    self._evidence_projects.popleft()
                else:
                    self._evidence_projects[0] = (project, str(rows[-1][1]), int(rows[-1][2]))
        if not page:
            return ()
        statuses = adapter.inspect(frame, tuple(page))
        refusals = adapter.terminal_decode_refusals(tuple(page))
        return tuple(raw_id for raw_id in page if statuses.get(raw_id) != "valid" and raw_id not in refusals)

    def _queue_evidence_dependents(self, arrived: Sequence[str]) -> None:
        """Queue a scan of the project each newly admitted session index describes."""
        from polylogue.storage.sqlite.connection_profile import open_readonly_connection

        source_db = self._archive_root / "source.db"
        with closing(open_readonly_connection(source_db, timeout=5.0)) as conn:
            placeholders = ",".join("?" for _ in arrived)
            indexes = conn.execute(
                f"SELECT source_path FROM raw_sessions WHERE raw_id IN ({placeholders}) "
                "AND source_path LIKE '%/sessions-index.json'",
                tuple(arrived),
            ).fetchall()
        arrived_projects = dict.fromkeys(str(index_path).rsplit("/", 1)[0] for (index_path,) in indexes)
        if not arrived_projects:
            return
        # A project already queued restarts from its first transcript: the
        # ones its scan already served were inspected against the older
        # index. One scan per project keeps the queue bounded by projects.
        pending = [entry for entry in self._evidence_projects if entry[0] not in arrived_projects]
        self._evidence_projects.clear()
        self._evidence_projects.extend(pending)
        self._evidence_projects.extend((project, "", -1) for project in arrived_projects)

    def _sweep_selected(self, frame: Any, adapter: Any, limit: int) -> tuple[str, ...]:
        # At most one released page is skipped per call, so a stalled head
        # costs one extra bounded page read rather than the whole cycle.
        for _attempt in range(2):
            page_cursor = self._cursor
            if page_cursor is None:
                self._sweep_started_at = time.monotonic()
            page, next_cursor = adapter.required_page(frame, cursor=page_cursor, limit=limit)
            # An empty or fully valid page is progress through the required-key
            # space. ``None`` is the completed-traversal marker; the following
            # pass starts a fresh sweep so new work before this cursor is
            # eventually revisited.
            selected: tuple[str, ...] = ()
            if page:
                statuses = adapter.inspect(frame, page)
                refusals = adapter.terminal_decode_refusals(page)
                selected = tuple(
                    raw_id for raw_id in page if statuses.get(raw_id) != "valid" and raw_id not in refusals
                )
            if not selected:
                self._advance_sweep(next_cursor)
                return ()
            held = self._held_page
            if held is not None and held[0] == page_cursor and set(held[1]) == set(selected):
                # Re-inspected the same page and nothing moved: the blockage is
                # not budget pressure, so stop pinning the traversal behind it
                # and go on to the next page in this same call.
                self._advance_sweep(next_cursor)
                continue
            # Hold the continuation over keys this page still owes. The
            # dispatcher admits the offered items only until its class budget
            # is spent, so advancing over the whole inspected page moves the
            # cursor past obligations nobody took -- the same
            # producer-advances-past-the-consumer shape
            # ``DerivationRunner.run_domain`` avoids with its ``stopped_at``
            # offset. The authority for "processed" is the output relation
            # re-inspected on the next pass.
            self._held_page = (page_cursor, selected)
            self._cursor = page_cursor
            return selected
        return ()

    def _advance_sweep(self, next_cursor: str | None) -> None:
        self._cursor = next_cursor
        self._held_page = None
        if next_cursor is None and self._sweep_started_at is not None:
            now = time.monotonic()
            self._last_sweep_duration_s = max(0.0, now - self._sweep_started_at)
            self._last_sweep_finished_at = now
            self._sweep_started_at = None

    def _with_costs(self, selected: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
        from polylogue.operations.operation_context import open_operation_read

        with open_operation_read(self._archive_root) as pinned:
            sizes = pinned.archive.raw_payload_sizes(selected)
        return tuple((raw_id, max(1, int(sizes.get(raw_id, 1)))) for raw_id in selected)


def discover_pending_raw_ids(archive_root: Path, limit: int) -> tuple[tuple[str, int], ...]:
    """Discover one bounded raw page for callers without an intake lifetime.

    The daemon owns a ``RawMaterializationDiscovery`` instance for its whole
    fair-intake lifetime. This compatibility helper deliberately has no
    continuation because it cannot retain one across independent callers.
    """
    return RawMaterializationDiscovery(archive_root).discover_pending_raw_ids(limit)


class DaemonIntakeService:
    """Supervisor-owned bounded loop; scheduler owns all policy."""

    def __init__(
        self,
        dispatcher: FairIntakeDispatcher,
        *,
        budget: int = DEFAULT_INTAKE_BYTE_BUDGET,
        idle_delay_s: float = 5.0,
        wakeup: asyncio.Event | None = None,
        on_backlog_drained: Callable[[], Awaitable[ColdBuildSettlement | None] | ColdBuildSettlement | None]
        | None = None,
        has_pending_backlog: Callable[[], Awaitable[bool] | bool] | None = None,
        on_pass_complete: Callable[[IntakePass], Awaitable[None] | None] | None = None,
        settlement_revision: Callable[[], tuple[int, ...]] | None = None,
        settlement_external_revision: Callable[[], tuple[int, ...]] | None = None,
    ) -> None:
        self.dispatcher = dispatcher
        # A count-scale budget (the previous literal 64) left a class's
        # per-pass share three to four orders of magnitude below one ordinary
        # session file, so a single admission drove the deficit deeply
        # negative and the class did no work for hundreds of passes.
        # ``DEFAULT_INTAKE_BYTE_BUDGET`` declares the unit.
        self.budget = max(1, budget)
        self.idle_delay_s = max(0.05, idle_delay_s)
        self._wakeup = wakeup if wakeup is not None else asyncio.Event()
        # Settlement follows a fully observed quiescent pass. Candidate
        # writes can arrive through the operation route without any successful
        # watched-file admission. The callback owns promotion versus discard.
        self._on_backlog_drained = on_backlog_drained
        self._has_pending_backlog = has_pending_backlog
        self._on_pass_complete = on_pass_complete
        self._settlement: ColdBuildSettlement | None = None
        self._settlement_revision = settlement_revision
        self._settlement_external_revision = settlement_external_revision
        self._blocked_revision: tuple[int, ...] | None = None
        self._progress_since_blocked = False

    async def run(self) -> None:
        from polylogue.core.degraded import is_fully_degraded

        while True:
            self._wakeup.clear()
            if is_fully_degraded():
                # Parked, not drained: a structurally degraded daemon runs no
                # pass (every adapter's discovery reads its own tier), so no
                # retry deadline or retained page can spin the loop, and an
                # empty pass can never read as a drained backlog that settles
                # a cold build. Resume at the idle cadence or on a wakeup.
                try:
                    async with asyncio.timeout(self.idle_delay_s):
                        await self._wakeup.wait()
                except TimeoutError:
                    pass
                continue
            result = await self.dispatcher.run_once(budget=self.budget)
            if is_fully_degraded():
                # The pass itself degraded the daemon; park before any
                # post-pass callback touches the archive (next iteration),
                # but keep the progress it did commit, which later cold-build
                # settlement depends on.
                if result.progressed:
                    self._progress_since_blocked = True
                continue
            schedulable = self.dispatcher.schedulable_classes()
            discovery_pending = any(bool(getattr(spec.adapter, "discovery_pending", False)) for spec in schedulable)
            retry_delays = tuple(
                delay for spec in schedulable if (delay := getattr(spec.adapter, "retry_due_in_s", None)) is not None
            )
            # Cold promotion settles the local baseline. Remote completeness
            # has its own configured-source witness and public claim gate.
            settlement_result = IntakePass(
                tuple(report for report in result.classes if report.name != "configured_remote"),
                tuple(name for name in result.skipped_halted if name != "configured_remote"),
            )
            settlement_specs = tuple(spec for spec in schedulable if spec.name != "configured_remote")
            settlement_discovery_pending = any(
                bool(getattr(spec.adapter, "discovery_pending", False)) for spec in settlement_specs
            )
            settlement_retry_pending = any(
                getattr(spec.adapter, "retry_due_in_s", None) is not None for spec in settlement_specs
            )
            if result.progressed and self._on_pass_complete is not None:
                outcome = self._on_pass_complete(result)
                if isinstance(outcome, Awaitable):
                    await outcome
            if settlement_result.progressed:
                self._progress_since_blocked = True
            elif (
                settlement_result.quiescent
                and not settlement_discovery_pending
                and not settlement_retry_pending
                and self._on_backlog_drained is not None
            ):
                pending = self._has_pending_backlog() if self._has_pending_backlog is not None else False
                if isinstance(pending, Awaitable):
                    pending = await pending
                if not pending:
                    settlement = self._settlement
                    now = time.monotonic()
                    revision = self._settlement_revision() if self._settlement_revision is not None else ()
                    due = (
                        settlement is None
                        or (
                            settlement.state == "retryable"
                            and settlement.next_retry_at is not None
                            and now >= settlement.next_retry_at
                        )
                        or (
                            settlement.state == "blocked"
                            and (self._progress_since_blocked or revision != self._blocked_revision)
                        )
                    )
                    if due:
                        external_before = (
                            self._settlement_external_revision()
                            if self._settlement_external_revision is not None
                            else None
                        )
                        settlement_outcome = self._on_backlog_drained()
                        if isinstance(settlement_outcome, Awaitable):
                            settlement_outcome = await settlement_outcome
                        if settlement_outcome is None or settlement_outcome.state == "complete":
                            self._on_backlog_drained = None
                            self._settlement = None
                        else:
                            self._settlement = settlement_outcome
                            if settlement_outcome.state == "blocked":
                                # The attempt itself may update candidate
                                # metadata or WAL files. Only a later change
                                # to the evidence may wake a blocked verdict.
                                self._blocked_revision = (
                                    self._settlement_revision() if self._settlement_revision is not None else ()
                                )
                                self._progress_since_blocked = (
                                    external_before != self._settlement_external_revision()
                                    if external_before is not None and self._settlement_external_revision is not None
                                    else False
                                )
            try:
                idle_delay = 0.05 if result.progressed or discovery_pending else self.idle_delay_s
                if retry_delays:
                    idle_delay = min(idle_delay, max(0.05, min(retry_delays)))
                if self._settlement is not None and self._settlement.next_retry_at is not None:
                    idle_delay = min(idle_delay, max(0.05, self._settlement.next_retry_at - time.monotonic()))
                async with asyncio.timeout(idle_delay):
                    await self._wakeup.wait()
            except TimeoutError:
                pass


def build_intake_adapters(
    context: DaemonIntakeContext,
    *,
    remote_callback: Callable[[], Awaitable[DriveCatchupReport] | DriveCatchupReport] | None = None,
    raw_callback: Callable[..., Awaitable[AdmissionResult | int] | AdmissionResult | int] | None = None,
    raw_discover: Callable[[int], Awaitable[Sequence[tuple[str, int]]] | Sequence[tuple[str, int]]] | None = None,
    raw_suspended: Callable[[], bool] | None = None,
    raw_discovery_pending: Callable[[], bool] | None = None,
    hook_events_callback: Callable[..., Awaitable[AdmissionResult | int] | AdmissionResult | int] | None = None,
    hook_events_discover: Callable[[int], Awaitable[Sequence[tuple[str, int]]] | Sequence[tuple[str, int]]]
    | None = None,
    source_halts: SubUnitHaltPolicy | None = None,
) -> tuple[tuple[str, IntakeAdapter], ...]:
    """Compose browser, hook, local, remote, admitted-raw and hook-event classes."""

    result: list[tuple[str, IntakeAdapter]] = []
    local: list[FileIntakeAdapter] = []
    hook_carriers: list[FileIntakeAdapter] = []
    for source in context.sources:
        if source.role in {"primary-writable", "legacy-read-only"}:
            # Hook carriers are acquired by the ordinary file route -- they are
            # append-only JSONL like every other live source. They keep their
            # own intake class so a burst of hook traffic cannot crowd out
            # session files under the dispatcher's per-class fair share.
            hook_carriers.append(FileIntakeAdapter(context, source, class_name="hook_carrier"))
            continue
        if source.name == "browser-capture":
            result.append(("browser_capture", FileIntakeAdapter(context, source, class_name="browser_capture")))
        else:
            local.append(FileIntakeAdapter(context, source, class_name="configured_local"))
    if local:
        result.append(("configured_local", MultiplexIntakeAdapter(local, halts=source_halts)))
    if hook_carriers:
        result.append(("hook_carrier", MultiplexIntakeAdapter(hook_carriers, halts=source_halts)))
    if remote_callback is not None:
        result.append(("configured_remote", DriveIntakeAdapter(remote_callback)))
    if raw_callback is not None:
        if raw_discover is None:
            result.append(("raw_materialization", CallbackIntakeAdapter("raw_materialization", raw_callback)))
        else:

            async def admit_raw(raw_id: str) -> AdmissionResult | int:
                value = raw_callback(raw_id)
                if isinstance(value, Awaitable):
                    value = await value
                return value

            result.append(
                (
                    "raw_materialization",
                    RawMaterializationIntakeAdapter(
                        raw_discover, admit_raw, suspended=raw_suspended, discovery_pending=raw_discovery_pending
                    ),
                )
            )
    if hook_events_callback is not None and hook_events_discover is not None:
        # Hook-event materialization is keyed by carrier raw id exactly as raw
        # materialization is keyed by session raw id, so it reuses the same
        # adapter rather than a second one that would have to re-derive the
        # same retry, isolation and cost accounting.

        async def admit_hook_carrier(raw_id: str) -> AdmissionResult | int:
            value = hook_events_callback(raw_id)
            if isinstance(value, Awaitable):
                value = await value
            return value

        result.append(("hook_events", RawMaterializationIntakeAdapter(hook_events_discover, admit_hook_carrier)))
    return tuple(result)
