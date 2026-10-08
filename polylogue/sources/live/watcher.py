"""Live source watch and the single live-ingest entry point.

Watches one or more roots via ``watchfiles`` and turns each observation into
a disposable intake hint for ``FairIntakeDispatcher``, which owns discovery,
planning and admission. The dispatcher's file adapter calls
``LiveWatcher._ingest_files`` with a whole page, and ``LiveBatchProcessor``
does the work. Ingestion is idempotent via content-hash dedup; the cursor
table suppresses re-work when the stored content fingerprint and parser
fingerprint still match the file.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
import time
from collections.abc import Awaitable, Callable, Iterable, Iterator, Sequence
from contextlib import closing, contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, cast

from polylogue.archive.revision_authority import decided_unresolved_membership_sql, raw_receipt_order_sql
from polylogue.core.enums import Origin, Provider
from polylogue.core.evidence import Measured
from polylogue.core.protocols import ArchiveRootOwner
from polylogue.core.raw_failure_evidence import RawFailureEvidenceKind
from polylogue.core.sources import provider_from_origin
from polylogue.core.stage_admission import admit_stage_write, stage_write_admission
from polylogue.logging import get_logger
from polylogue.sources.hooks import (
    HookSpoolSourceSpec,
    hook_carrier_provider_dir,
    hook_spool_sources,
)
from polylogue.sources.live.archive_open import _source_tier_acquisition_required
from polylogue.sources.live.batch import (
    LiveBatchEventEmitter,
    LiveBatchProcessor,
    fingerprint_file,
)
from polylogue.sources.live.batch_support import (
    LiveRetainedRunner,
    _AppendPlan,
    _AppendResult,
    _archive_blob_exists,
    claude_semantic_frontier_for_prefix,
    cursor_ctime_ns,
    cursor_prefix_hash,
    cursor_tail_hash,
    encode_cursor_hash_authority,
    sha256_range_from_path,
    tail_hash_and_last_complete_newline_from_path,
    tail_hash_from_path,
)
from polylogue.sources.live.cursor import (
    CursorObservationRebase,
    CursorPathAuthority,
    CursorRecord,
    CursorStore,
)
from polylogue.sources.live.deferred_cursor import record_deferred_append_cursor
from polylogue.sources.live.metrics import LiveBatchMetrics
from polylogue.sources.live.source_selection import deepest_source_for_path
from polylogue.sources.live.sqlite_capture import LiveSQLiteCaptureStage
from polylogue.sources.parsers.hermes_identity import declares_profile_identity
from polylogue.sources.source_layout import (
    SourceLayout,
    declared_source_layout,
    hook_carrier_layout,
    source_layout_for,
)
from polylogue.sources.source_staging import SourceInputBinding, bind_source_input
from polylogue.sources.sqlite_snapshot import (
    is_sqlite_path,
    sqlite_database_for_sidecar,
    sqlite_member_revision,
    sqlite_source_revision,
)
from polylogue.storage.archive_identity import ArchiveLocationError, resolve_active_index_path
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from polylogue.storage.sqlite.write_lease import UnleasedWriteError
from polylogue.storage.tier_access import capture_sqlite_read

logger = get_logger(__name__)
# Bump whenever parser semantics change the values derived from already-
# observed bytes. A cursor stamped with a superseded fingerprint is treated
# as needing work, which routes its source back through parse on the next
# watcher pass -- the production convergence route, not a manual rebuild.
# v3: tool-result outcomes now derive `is_error` from an explicit exit code
# (#4539), so records parsed under v2 retain a stale unknown outcome.
# v4: relocated Claude Code cwds lead working_directories, argument-less
# ChatGPT commands and ID-less Antigravity tool steps become paired tool
# calls, OTel session ids escape their components, and a complete JSONL
# record that does not decode is terminal for every provider.
# v5: Hermes ``.jsonl.txt`` traces are recognized as JSONL (rs02d 10.F010),
# so a cursor excluded as an unsupported source class under v4 must get a
# fresh attempt.
# v6: recursive reserved-value identity and complete branch witnesses change
# parsed identities, including for sources whose observed bytes are unchanged.
_PARSER_FINGERPRINT = "live-batched-v6"
# polylogue-11cg9: the dispatcher's byte budget bounds an admitted page's
# *size* but not the *time* a single full-ingest pass can hold the sole
# archive writer -- a handful of files, or one slow-to-parse file, can still
# exceed it by any margin (the original de2a incident was an in-size-bounds
# 7 MB batch that held the writer for 860s). Mirrors de2a's
# ``_RAW_MATERIALIZATION_MAX_PASS_SECONDS`` / qlae's
# ``_DRIVE_CATCHUP_MAX_PASS_SECONDS`` constant and value -- checked between
# acquired files, full-ingest progress groups and archive-write records (a
# single session write cannot be split mid-transaction), never mid-record, so
# overshoot is one work item. Past the writer gate's own declared hold bound
# elapsed writer thresholds are diagnostic and never refuse an admitted item.
_LIVE_INGEST_MAX_PASS_SECONDS = 20.0
_RAW_RETENTION_RETRY_BUDGET_SECONDS = 30.0
_INCOMPLETE_APPEND_PROBE_BYTES = 64 * 1024 * 1024
# polylogue-dhkuu: the probe's own working set, independent of how much tail
# it is allowed to scan. The scan looks only for the first b"\n", so it never
# needs the tail resident: a single ``handle.read(remaining_bytes)`` sized the
# allocation by the *input* instead, and an unterminated multi-GB tail then
# raised ``MemoryError`` -- which ``except OSError`` does not catch -- before
# any deferral was recorded, so every catch-up pass re-attempted it forever.
_INCOMPLETE_APPEND_PROBE_CHUNK_BYTES = 1024 * 1024
# polylogue-2qrx: minimum age a deferred incomplete-tail observation must
# reach (``cursor.updated_at`` unchanged, i.e. the stat-match fast path kept
# firing) before escalating to an unbounded full-tail probe. An ordinary
# in-progress writer routinely leaves a genuinely-incomplete trailing record
# (no closing newline yet) between two polls milliseconds apart -- that must
# NOT be treated as "the writer is done" just because the stat happened to
# match on a second, immediate check. Only a source that has been sitting at
# the exact same byte state for a long time is plausibly finished rather than
# merely paused; an hour is far longer than the dispatcher's own
# re-discovery cadence, so the escalation cannot fire before an ordinary
# pass would have re-observed the file anyway.
_STUCK_DEFERRED_APPEND_AGE_S = 60.0 * 60.0
INBOX_SOURCE_SUFFIXES = (".jsonl", ".zip", ".json", ".ndjson", ".db", ".sqlite", ".sqlite3")


# Lifecycle evidence kinds are carried by failed raws; only a classification
# artifact can declare a raw non-session.
_FAILURE_EVIDENCE_KINDS_SQL = ", ".join(f"'{kind.value}'" for kind in RawFailureEvidenceKind)


def _stat_identity(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class _ArchivedCursorReconciliation(str, Enum):
    """Whether archive evidence can safely restore a live cursor."""

    RECONCILED = "reconciled"
    UNAVAILABLE = "unavailable"
    INCOMPATIBLE = "incompatible"


def _stage_timing_summary(stage_timings_s: dict[str, float], *, limit: int = 8) -> str:
    if not stage_timings_s:
        return "none"
    ordered = sorted(stage_timings_s.items(), key=lambda item: item[1], reverse=True)
    shown = ",".join(f"{name}:{elapsed:.3f}" for name, elapsed in ordered[:limit])
    omitted = len(ordered) - limit
    if omitted <= 0:
        return shown
    return f"{shown},+{omitted} more"


def _log_ingest_metrics(prefix: str, metrics: LiveBatchMetrics) -> None:
    """Log actual live-ingest read work separately from candidate file size."""
    input_bytes = getattr(metrics, "input_bytes", 0)
    source_payload_read_bytes = getattr(metrics, "source_payload_read_bytes", 0)
    read_amp = source_payload_read_bytes / input_bytes if input_bytes > 0 else 0.0
    stage_timings_s = getattr(metrics, "stage_timings_s", {})
    stage_summary = _stage_timing_summary(stage_timings_s if isinstance(stage_timings_s, dict) else {})
    partial_paths = getattr(metrics, "partial_admission_paths", {}) or {}
    partial_reasons = getattr(metrics, "partial_reasons", {}) or {}
    partial_left_out_bytes = sum(
        int(partial.source_bytes) - int(partial.complete_prefix_bytes) for partial in partial_paths.values()
    )
    logger.info(
        "%s complete: read=%.1f MB input=%.1f MB read_amp=%.6fx append_files=%d full_files=%d "
        "succeeded=%d partial=%d partial_reasons=%s partial_left_out_bytes=%d "
        "failed=%d excluded=%d parse_s=%.3f convergence_s=%.3f stages=%s "
        "time_budget_exceeded=%s",
        prefix,
        source_payload_read_bytes / 1e6,
        input_bytes / 1e6,
        read_amp,
        getattr(metrics, "append_file_count", 0),
        getattr(metrics, "full_file_count", 0),
        getattr(metrics, "succeeded_file_count", 0),
        len(partial_paths),
        ", ".join(f"{reason} x{count}" for reason, count in sorted(partial_reasons.items())) or "none",
        partial_left_out_bytes,
        getattr(metrics, "failed_file_count", 0),
        getattr(metrics, "excluded_file_count", 0),
        getattr(metrics, "parse_time_s", 0.0),
        getattr(metrics, "convergence_time_s", 0.0),
        stage_summary,
        getattr(metrics, "time_budget_exceeded", False),
    )
    excluded_reasons = getattr(metrics, "excluded_reasons", {})
    if excluded_reasons:
        logger.info(
            "%s: admitted nothing for %d planned path(s): %s",
            prefix,
            getattr(metrics, "excluded_file_count", 0),
            ", ".join(f"{reason} x{count}" for reason, count in sorted(excluded_reasons.items())),
        )
    if getattr(metrics, "time_budget_exceeded", False):
        logger.info(
            "%s: max_pass_seconds budget exceeded -- remaining files deferred to the next tick (polylogue-11cg9)",
            prefix,
        )


def _directory_identity(path: Path) -> tuple[int, int] | None:
    """Return the device/inode a directory really occupies.

    ``None`` for anything that cannot be statted -- a broken symlink or a
    directory that vanished mid-walk -- neither of which can be descended.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_dev, stat.st_ino)


def _relative_parts(path: Path, root: Path) -> tuple[str, ...] | None:
    """``path`` relative to ``root``: lexically first, then by resolved location."""

    try:
        return Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(root))).parts
    except ValueError:
        pass
    try:
        return path.resolve().relative_to(root.resolve()).parts
    except (OSError, ValueError):
        return None


#: Pause before re-listing the watched directories after one vanished mid-arm.
_WATCH_REARM_RETRY_S = 0.5


class _RearmSignal:
    """The ``stop_event`` handed to one ``awatch`` arming.

    It reports set when the watcher stops or when the watched directory set
    has to change; ``watchfiles`` polls ``is_set`` between steps.
    """

    def __init__(self, stop: asyncio.Event) -> None:
        self._stop = stop
        self._requested = False

    def request(self) -> None:
        self._requested = True

    def is_set(self) -> bool:
        return self._requested or self._stop.is_set()


@dataclass(frozen=True, slots=True)
class WatchSource:
    """A source root and the declared layout of the material below it.

    Every route that enumerates a source -- daemon discovery, the live event
    filter, the cold-build baseline, one-shot ingest and schema inference --
    admits only what ``layout`` places (``polylogue.sources.source_layout``).
    A canonical watch-source name resolves its declared layout; a caller with
    a synthetic root passes one explicitly. ``exact_paths`` names explicitly
    declared input files, which are admitted as given.
    """

    name: str
    root: Path
    # Resolved from ``name`` in ``__post_init__`` when not given explicitly.
    layout: SourceLayout = cast(SourceLayout, None)
    # Hook sources carry durable topology identity.  Ordinary sources retain
    # their historical name-only contract.
    source_id: str | None = None
    role: str | None = None
    required: bool = False
    exact_paths: frozenset[Path] | None = None

    def __post_init__(self) -> None:
        if self.layout is None:
            object.__setattr__(self, "layout", source_layout_for(self.name))

    def exists(self) -> bool:
        return self.root.exists()

    def artifact_kind(self, path: Path) -> str | None:
        """The declared artifact kind at ``path``, or ``None`` outside the layout."""

        parts = _relative_parts(path, self.root)
        if not parts:
            return None
        return self.layout.artifact_kind(parts)

    def accepts(self, path: Path) -> bool:
        if self.exact_paths is not None:
            try:
                return path.resolve() in self.exact_paths
            except OSError:
                return False
        return self.artifact_kind(path) is not None

    def admits_directory(self, path: Path) -> bool:
        """Whether a directory under the root can hold an artifact of this source."""

        parts = _relative_parts(path, self.root)
        if parts is None:
            return False
        return not parts or self.layout.admits_directory(parts)


#: The harnesses that write hook carriers. Each gets its own watched
#: directory so acquisition is provider-scoped: the origin-spec artifact rule
#: and the materialized ``origin`` both follow the directory, and a carrier
#: never has to be opened to learn which harness wrote it.
HOOK_CARRIER_PROVIDERS: tuple[str, ...] = ("claude-code", "codex", "hermes")


def hook_carrier_watch_sources(specs: Iterable[HookSpoolSourceSpec]) -> tuple[WatchSource, ...]:
    """Watch every declared hook root's carriers as ordinary append-only sources.

    A carrier is acquired, retained and revision-bound exactly like any other
    growing JSONL source; nothing here is hook-specific beyond the directory.
    The events inside are materialized later by the ``hook_events`` derivation
    out of the retained bytes -- the watcher never parses a hook envelope and
    never writes ``raw_hook_events``.
    """

    return tuple(
        WatchSource(
            # Distinct from the provider's own session source of the same
            # harness name: two sources may not share a name, or every
            # by-name lookup silently sees only the last one.
            name=f"{provider}-hooks",
            root=hook_carrier_provider_dir(provider, spec.root),
            source_id=f"{spec.source_id}:{provider}",
            role=spec.role,
            layout=hook_carrier_layout(Provider.from_string(provider)),
        )
        for spec in specs
        for provider in HOOK_CARRIER_PROVIDERS
    )


class WriteCoordinator(Protocol):
    """The daemon's in-process write gate, as this module needs it.

    Declared as a Protocol so a coordinator that does not satisfy it fails
    loudly at the call. The previous ``getattr(coordinator, "run", None)``
    dispatch silently downgraded to an ungated archive write whenever the
    injected object had the wrong shape, which made the single-writer
    invariant defeasible by a test double (polylogue-8qm4k). ``None`` remains
    the explicit standalone opt-out; a wrong shape is now an error.
    """

    async def run(self, actor: str, operation: Callable[[], Awaitable[Any]], /) -> Any: ...

    async def run_sync(self, actor: str, function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any: ...


class EmbeddingConvergenceOwner(Protocol):
    """Converge one batch's embeddings with the writer gate released.

    The embedding stage inside a coordinated ingest defers rather than calling
    the provider under the writer gate, so this owner is what actually performs
    the pass: it runs on the daemon's compute capacity and admits each short
    write back through the coordinator. ``None`` is the standalone opt-out --
    without a coordinator there is no gate to be outside of, and the stage
    embeds inline.
    """

    async def __call__(self, index_db_path: Path, paths: Sequence[Path], /) -> bool: ...


class SessionProfileConvergenceCallback(Protocol):
    """Converge post-ingest changes or an archive-wide periodic sweep."""

    def __call__(self, session_ids: Sequence[str], /) -> Awaitable[object]: ...


def _is_retryable_lock_error(exc: sqlite3.OperationalError) -> bool:
    """SQLite lock contention, as opposed to a broken database."""
    message = str(exc).lower()
    return "database is locked" in message or "database table is locked" in message or "busy" in message


class WatcherRootsUnavailableError(RuntimeError):
    """Raised when the watcher is started with no existing source root."""


def _published_index_path(archive_root: Path) -> Path:
    """The index this watcher's writes publish into.

    Cursor corroboration asks whether a file's raw is materialized in the
    index. During a cold build that is the inactive candidate generation the
    writer is filling, not the still-active (empty or old) index: checking
    the active one demoted every cursor the build had just written and
    re-ingested the file from scratch.
    """
    from polylogue.sources.live.cold_build import active_cold_build_generation

    generation = active_cold_build_generation(archive_root)
    if generation is not None:
        return Path(generation.generation.index_path)
    return resolve_active_index_path(archive_root)


class LiveWatcher:
    """Filesystem watch that wakes the fair-intake dispatcher.

    Acquisition has one route: ``FairIntakeDispatcher`` discovers, plans and
    admits pages through ``FileIntakeAdapter``, which calls
    :meth:`_ingest_files`. This watcher owns no queue, no catch-up scan and
    no schedule of its own -- it turns a filesystem event into a bumped
    intake revision plus a wakeup, so the dispatcher's next pass is prompt
    rather than waiting out its idle delay. It also owns the batch
    processor, the cursor store and the supplied capture stage the adapter's ingest
    runs through.
    """

    def __init__(
        self,
        polylogue: ArchiveRootOwner,
        sources: Iterable[WatchSource],
        *,
        cursor: CursorStore | None = None,
        converger: object | None = None,  # DaemonConverger | None — avoids circular import
        event_emitter: LiveBatchEventEmitter | None = None,
        write_coordinator: WriteCoordinator | None = None,
        sqlite_capture_stage: LiveSQLiteCaptureStage | None = None,
        embedding_owner: EmbeddingConvergenceOwner | None = None,
        session_profile_callback: SessionProfileConvergenceCallback | None = None,
        append_runner: Callable[[Any, list[_AppendPlan]], Awaitable[_AppendResult]] | None = None,
        convergence_runner: Callable[..., Awaitable[Any]] | None = None,
        retained_runner: LiveRetainedRunner | None = None,
        intake_wakeup: asyncio.Event | None = None,
    ) -> None:
        self._polylogue = polylogue
        self._sources = tuple(sources)
        self._cursor = cursor or CursorStore(
            _cursor_db_path(polylogue),
            initialize=write_coordinator is None,
            ops_db_path=Path(polylogue.archive_root) / "ops.db",
        )
        self._converger = converger
        self._write_coordinator = write_coordinator
        # Injected rather than imported: the provider call this owner performs
        # must run with the writer gate released, and the owner that can admit
        # its short writes lives in the daemon ring, which this one may not
        # import (polylogue-c0l7n).
        self._embedding_owner = embedding_owner
        self._session_profile_callback = session_profile_callback
        self._intake_wakeup = intake_wakeup
        self._intake_revisions = dict.fromkeys((source.root for source in self._sources), 0)
        self._event_emitter = event_emitter
        self._published_source_halts: dict[str, str] = {}
        # The watcher owns the supplied capture stage's lifecycle. The stage
        # borrows the resident kernel and never shuts that kernel down.
        self._sqlite_capture_stage = sqlite_capture_stage
        self._ingest_lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._watcher_ready = asyncio.Event()
        # Per thread: selection runs on a worker thread off the writer, and a
        # connection never crosses threads.
        self._archived_cursor_local = threading.local()
        self._batch_processor = LiveBatchProcessor(
            polylogue,
            self._sources,
            cursor=self._cursor,
            parser_fingerprint=lambda: _PARSER_FINGERPRINT,
            converger=converger,
            stop_requested=self._stop.is_set,
            event_emitter=event_emitter,
            # Without a write coordinator there is no writer to run Source
            # bodies on; the processor then refuses them instead of running
            # them unleased.
            sync_runner=self._run_writer_sync if write_coordinator is not None else None,
            append_runner=append_runner,
            convergence_runner=convergence_runner,
            retained_runner=retained_runner,
            sqlite_capture_stage=self._sqlite_capture_stage,
        )

    async def _run_writer_sync(
        self,
        actor: str,
        function: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run blocking watcher writes on the coordinator's writer."""
        if self._write_coordinator is None:
            raise UnleasedWriteError(f"{actor} writes the archive and requires the daemon write coordinator")
        return await self._write_coordinator.run_sync(actor, function, *args, **kwargs)

    @property
    def _archived_cursor_conns(self) -> tuple[sqlite3.Connection, sqlite3.Connection] | None:
        conns: tuple[sqlite3.Connection, sqlite3.Connection] | None = getattr(
            self._archived_cursor_local, "conns", None
        )
        return conns

    @_archived_cursor_conns.setter
    def _archived_cursor_conns(self, conns: tuple[sqlite3.Connection, sqlite3.Connection] | None) -> None:
        self._archived_cursor_local.conns = conns

    async def classify_ingest_candidates_off_writer(
        self, paths: Sequence[Path]
    ) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
        """Run :meth:`classify_ingest_candidates` with the writer released.

        Selection reads cursors and archive rows and hashes source bytes; a
        whole-file reconciliation hash under the writer queued every other
        archive writer behind it. Each cursor correction it decides is a short
        section admitted onto the writer, which first re-checks that the file
        observation and cursor row it was decided from still hold.
        """
        coordinator = self._write_coordinator
        if coordinator is None:
            raise UnleasedWriteError("watcher.intake.select writes cursors and requires the daemon write coordinator")
        loop = asyncio.get_running_loop()

        def admission(actor: str, work: Callable[[], Any]) -> Any:
            return asyncio.run_coroutine_threadsafe(coordinator.run_sync(actor, work), loop).result()

        def select() -> tuple[tuple[Path, ...], tuple[Path, ...]]:
            with stage_write_admission(admission):
                return self.classify_ingest_candidates(paths)

        selection = asyncio.ensure_future(asyncio.to_thread(select))
        try:
            return await asyncio.shield(selection)
        except asyncio.CancelledError:
            # The worker may be inside an admitted write; let it settle while
            # the loop still serves it, then propagate the cancellation.
            with suppress(BaseException):
                await selection
            raise

    def _admit_observed_cursor_write(
        self,
        actor: str,
        path: Path,
        *,
        stat: os.stat_result,
        expected: CursorRecord | None,
        write: Callable[[], object],
    ) -> bool:
        """Apply one selection cursor write if its observation still holds.

        The decision was made, and its bytes hashed, without the writer. Under
        the writer the file's identity and the cursor row are compared with
        what the decision read; a changed file or a cursor another writer
        moved refuses the write, and the caller treats the file as needing
        ingest.
        """

        def guarded() -> bool:
            try:
                current = path.stat()
            except OSError:
                return False
            if _stat_identity(current) != _stat_identity(stat):
                return False
            if self._cursor.get_record(path) != expected:
                return False
            write()
            return True

        return bool(admit_stage_write(actor, guarded))

    @property
    def has_write_coordinator(self) -> bool:
        """Whether archive writes run on a daemon write coordinator."""
        return self._write_coordinator is not None

    @property
    def watcher_ready(self) -> asyncio.Event:
        """Set when this watcher has registered its source roots."""
        return self._watcher_ready

    def intake_revision(self, source: WatchSource) -> int:
        """Disposable invalidation of one source's file-discovery position."""
        return self._intake_revisions[source.root]

    async def retry_raw_retention_backlog(self) -> None:
        """Drain retry-due retention debt even when no source changed.

        Bounded passes repeat while the due backlog keeps changing, inside a
        budget shorter than the convergence tick. One pass per tick made the
        drain cadence-bound: after a cold build every admitted file owes
        retention, and a fixed page per minute is a day of backlog for a
        full archive. A pass that retains nothing re-records its paths with
        backoff, so the due set moves on or empties; an unchanged set ends
        the drain. The ingest lock is released between passes.
        """
        deadline = time.monotonic() + _RAW_RETENTION_RETRY_BUDGET_SECONDS
        previous: list[Path] | None = None
        while True:
            async with self._ingest_lock:
                backlog = self._batch_processor._raw_retention_backlog_paths(exclude=set())
                if not backlog or backlog == previous:
                    return
                await self._batch_processor._run_source_writer(
                    "watcher.live_ingest.raw_compaction_retry",
                    self._batch_processor._compact_superseded_raw_snapshots,
                    [],
                )
            previous = backlog
            if time.monotonic() >= deadline:
                return

    def _existing_source_roots(self) -> list[Path]:
        """Return configured roots that exist at the instant of a scan."""
        return [source.root for source in self._sources if source.exists()]

    def prepare_watch_roots(self) -> list[Path]:
        """Create the writable hook carrier roots and return what can be watched.

        Hook commands create their first carrier lazily.  The nested root must
        exist before ``awatch`` snapshots its roots, otherwise a daemon that
        starts before the first hook event never sees that file.  An empty
        result means there is nothing to watch; the daemon resolves that as an
        unavailable watcher before starting :meth:`run`.
        """
        for source in self._hook_sources():
            # Untagged single-root callers predate the topology contract and
            # are necessarily the primary.  Tagged legacy roots remain
            # strictly read-only and are never created by the watcher.
            if source.role in {None, "primary-writable"}:
                source.root.mkdir(parents=True, exist_ok=True)
        return self._existing_source_roots()

    async def run(self) -> None:
        roots = self.prepare_watch_roots()
        if not roots:
            # Reachable only when every root vanished after the daemon's own
            # check. Returning would read as a completed watch, so raise and
            # let the declared failure policy decide.
            self._watcher_ready.set()
            raise WatcherRootsUnavailableError("no configured source root exists")

        watch_task = asyncio.create_task(self._watch_changes())
        await asyncio.sleep(0)
        try:
            # Discovery is owned by FairIntakeDispatcher, so nothing here
            # gates on an acquisition sweep of its own. The event stays for
            # the maintenance loops that still wait on it: it is ready as
            # soon as the watch is registered.
            self._watcher_ready.set()
            logger.info("live.watcher: watching %s", ", ".join(str(r) for r in roots))
            await watch_task
        finally:
            if not watch_task.done():
                watch_task.cancel()
            with suppress(asyncio.CancelledError):
                await watch_task

    def watched_directories(self) -> list[Path]:
        """Every existing directory some source's declared layout reaches.

        The live watch is installed on exactly these directories, each
        non-recursively, so a subtree outside every layout (a nested copy of
        a provider tree, ``.git``, an install's caches) costs no inotify watch
        and raises no event. A directory symlink is followed only while its
        target stays inside the source root, as discovery follows it; a
        target already entered is not entered again.
        """

        directories: list[Path] = []
        listed: set[Path] = set()
        for source in self._sources:
            root = source.root
            if not root.is_dir():
                continue
            try:
                root_real = os.path.realpath(root)
            except OSError:
                continue
            entered = {root_real}
            stack = [root]
            while stack:
                directory = stack.pop()
                if directory not in listed:
                    listed.add(directory)
                    directories.append(directory)
                try:
                    with os.scandir(directory) as entries:
                        children = [Path(entry.path) for entry in entries if entry.is_dir()]
                except OSError:
                    continue
                for child in sorted(children):
                    if not source.admits_directory(child):
                        continue
                    try:
                        real = os.path.realpath(child)
                    except OSError:
                        continue
                    if real in entered or (real != root_real and not real.startswith(root_real + os.sep)):
                        continue
                    entered.add(real)
                    stack.append(child)
        return directories

    async def _watch_changes(self) -> None:
        """Watch the layout-reachable directories, re-arming as the set changes.

        A new directory a layout reaches (a fresh session's ``subagents/``)
        re-arms the watch with it included, and a watched directory that
        disappears re-arms it without. Files written into a new directory
        before its watch exists are covered by the intake hint its creation
        event raises: discovery rescans the source.
        """

        from watchfiles import Change, awatch

        while not self._stop.is_set():
            directories = await asyncio.to_thread(self.watched_directories)
            if not directories:
                return
            watched = set(directories)
            rearm = _RearmSignal(self._stop)
            try:
                async for changes in awatch(
                    *directories,
                    watch_filter=self._watch_filter,
                    stop_event=rearm,
                    recursive=False,
                ):
                    for change, raw_path in changes:
                        path = Path(raw_path)
                        if change is Change.deleted:
                            if path in watched:
                                rearm.request()
                            continue
                        if path not in watched and path.is_dir() and self._source_for_directory(path) is not None:
                            rearm.request()
                        self._note_intake_hint(path)
            except FileNotFoundError:
                # A directory vanished between listing and watching: list
                # again. Its disappearance is ordinary producer churn.
                self._note_intake_hint(directories[0])
                await asyncio.sleep(_WATCH_REARM_RETRY_S)

    def stop(self) -> None:
        self._stop.set()
        if self._sqlite_capture_stage is not None:
            self._sqlite_capture_stage.shutdown()

    def _hook_sources(self) -> tuple[WatchSource, ...]:
        """Return the declared hook-carrier sources, preserving configured order.

        These are ordinary watched sources in every respect that matters to
        acquisition; the topology role survives only so the watcher knows
        which carrier roots it may create and which are strictly read-only.
        """
        return tuple(
            source
            for source in self._sources
            if source.source_id is not None and source.role in {"primary-writable", "legacy-read-only"}
        ) or tuple(source for source in self._sources if source.name == "hooks")

    def _note_intake_hint(self, path: Path) -> None:
        """Invalidate the dispatcher's walk position for the observed root.

        The hint is disposable in both directions: losing it costs latency
        (the dispatcher's idle pass still finds the file), and a spurious one
        costs one bounded re-walk. Nothing here decides what is ingested.
        """
        observed = path.resolve()
        for root in self._intake_revisions:
            resolved_root = root.resolve()
            if observed.is_relative_to(resolved_root) or resolved_root.is_relative_to(observed):
                self._intake_revisions[root] += 1
        if self._intake_wakeup is not None:
            self._intake_wakeup.set()

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def classify_ingest_candidates(self, paths: Sequence[Path]) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
        """Split one page into (needed now, pending a scheduled retry).

        A path whose cursor carries a retry that is not yet due is owed work,
        not accounted for: reporting it as already admitted acknowledged the
        page and dropped the obligation (polylogue-b8of0). Everything else
        not needed is covered by its cursor.
        """
        needed = self._select_ingest_candidates(paths)
        if len(needed) == len(paths):
            return needed, ()
        chosen = set(needed)
        remaining = [path for path in paths if path not in chosen]
        records = self._cursor.get_records(remaining)
        pending = tuple(
            path
            for path in remaining
            if (record := records.get(path)) is not None
            and not record.excluded
            and record.next_retry_at is not None
            and not _retry_due(record.next_retry_at)
        )
        return needed, pending

    def _select_ingest_candidates(self, paths: Sequence[Path]) -> tuple[Path, ...]:
        """Narrow one admitted page to the files that actually need ingesting.

        The dispatcher's discovery is a bounded walk with a disposable
        position, so it re-offers files whose cursor already accounts for
        them. This is the one place that decides, in bulk, which of those are
        real work: the cursor comparison, the archived-cursor reconciliation
        scope (one read-only connection pair for the whole page rather than
        one per file) and the device-drift rebase all run here, before the
        batch takes any writer admission.

        A file this returns nothing for is not refused -- the caller reports
        it as already admitted under its own identity, which is what the
        cursor says.
        """
        if not paths:
            return ()
        cursor_records = self._cursor.get_records(paths)
        rebases: list[CursorObservationRebase] = []
        needed: list[Path] = []
        with self._archived_cursor_reconciliation_scope():
            for path in paths:
                if self._stop.is_set():
                    break
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                if self._needs_work_from_state(
                    path,
                    stat=stat,
                    cursor=cursor_records.get(path),
                    rebase_queue=rebases,
                ):
                    needed.append(path)
        if rebases:
            needed.extend(self._admit_rebases(rebases))
        return tuple(needed)

    def _admit_rebases(self, rebases: Sequence[CursorObservationRebase]) -> list[Path]:
        """Persist proved rebases under the writer; return paths whose file moved since."""
        moved: list[Path] = []

        def write() -> None:
            current: list[CursorObservationRebase] = []
            for rebase in rebases:
                try:
                    observed = rebase.path.stat()
                except OSError:
                    moved.append(rebase.path)
                    continue
                if (observed.st_dev, observed.st_ino, observed.st_size, observed.st_mtime_ns) != (
                    rebase.st_dev,
                    rebase.st_ino,
                    rebase.expected.byte_size,
                    rebase.mtime_ns,
                ):
                    moved.append(rebase.path)
                    continue
                current.append(rebase)
            # The store compares each row with its expected record itself.
            self._cursor.rebase_authoritative_observations(current)

        admit_stage_write("watcher.intake.cursor_rebase", write)
        return moved

    def _needs_work(self, path: Path) -> bool:
        """Return True if the file is new, grown, or fingerprint-changed."""
        try:
            stat = path.stat()
        except FileNotFoundError:
            return False
        cursor = self._cursor.get_record(path)
        return self._needs_work_from_state(path, stat=stat, cursor=cursor)

    def _needs_work_from_state(
        self,
        path: Path,
        *,
        stat: os.stat_result,
        cursor: CursorRecord | None,
        rebase_queue: list[CursorObservationRebase] | None = None,
    ) -> bool:
        """Decide whether ``path`` needs (re-)ingestion, corroborated against the index.

        Delegates the byte/fingerprint-level decision to
        :meth:`_needs_work_from_state_uncorroborated`. When that would skip
        the file (trusting a cursor claim), check that file's parsed raw
        against the active index. A single materialized session does not
        corroborate other files after a partial index rebuild. The scope
        shares one connection pair across the bounded intake page.
        """
        needs_work = self._needs_work_from_state_uncorroborated(
            path, stat=stat, cursor=cursor, rebase_queue=rebase_queue
        )
        if needs_work:
            return True
        if self._cursor_skip_corroborated_by_index(path):
            return False
        logger.warning(
            "live.watcher: demoting uncorroborated cursor skip to needed for %s (index cannot show materialized raw)",
            path,
        )
        return True

    def _needs_work_from_state_uncorroborated(
        self,
        path: Path,
        *,
        stat: os.stat_result,
        cursor: CursorRecord | None,
        rebase_queue: list[CursorObservationRebase] | None = None,
    ) -> bool:
        size = stat.st_size
        if cursor is not None and (
            cursor.source_name == Provider.HERMES.value or self._source_name_for(path) == Provider.HERMES.value
        ):
            # Equal bytes and inode do not prove an equal declared profile.
            # Check before every exclusion, deferral, or content-based skip.
            from polylogue.sources.parsers.hermes_identity import observe_profile_namespace

            try:
                observed_profile = observe_profile_namespace(path, stat)
            except OSError:
                return True
            if cursor.captured_profile_key is None or cursor.captured_profile_key != observed_profile.key:
                return True
        if cursor is None:
            if not self._reconcile_archived_cursor(path, stat=stat, expected=None):
                return True
            cursor = self._cursor.get_record(path)
            return cursor is not None and size > cursor.byte_offset
        if cursor.excluded:
            identity_unchanged = (
                cursor.byte_size,
                cursor.st_dev,
                cursor.st_ino,
                cursor.mtime_ns,
            ) == (size, stat.st_dev, stat.st_ino, stat.st_mtime_ns)
            # polylogue-ix5r: exclusion revival was previously bound only to
            # file identity, so a parser fix could never revive a cursor
            # excluded before that fix shipped -- the file itself never
            # changes, so ``identity_unchanged`` stays True forever and the
            # cursor stays dark until someone manually re-ingests it. A
            # parser fingerprint change is exactly the other legitimate
            # reason a previously-poisoned observation deserves a fresh
            # attempt: the code that failed to parse it no longer exists.
            # Report the fresh attempt without clearing ``excluded``. The
            # quarantine is lifted by the cursor write of an ingest that
            # actually retained something, so a path that fails again stays
            # quarantined. Clearing it here instead left the row
            # ``excluded = 0`` carrying its old byte offset for the whole
            # window before acquisition ran, and the raw-frontier cursor map
            # reads exactly that shape as committed ingest authority with no
            # accepted head -- a source whose stat changes on every poll, a
            # live database, re-entered that window on every poll.
            return not (identity_unchanged and cursor.parser_fingerprint == _PARSER_FINGERPRINT)
        if cursor.parser_fingerprint != _PARSER_FINGERPRINT:
            # Retry and deferred-reconciliation state belongs to the parser
            # that produced it. Do not restamp its old outcome from archive
            # corroboration or postpone the new parser's first attempt.
            return True
        if cursor.failure_count == 0 and cursor.content_fingerprint is None and cursor.next_retry_at is not None:
            if not _retry_due(cursor.next_retry_at):
                return False
            reconciliation = self._reconcile_archived_cursor_outcome(path, stat=stat, expected=cursor)
            if reconciliation is _ArchivedCursorReconciliation.RECONCILED:
                reconciled = self._cursor.get_record(path)
                return reconciled is not None and size > reconciled.byte_offset
            if reconciliation is _ArchivedCursorReconciliation.UNAVAILABLE:
                return not self._admit_observed_cursor_write(
                    "watcher.intake.cursor_defer",
                    path,
                    stat=stat,
                    expected=cursor,
                    write=lambda: self._cursor.defer_full_cursor_reconciliation(path),
                )
            self._admit_observed_cursor_write(
                "watcher.intake.cursor_invalidate",
                path,
                stat=stat,
                expected=cursor,
                write=lambda: self._invalidate_deferred_full_cursor(path, stat=stat),
            )
            return True
        if cursor.failure_count > 0:
            if self._reconcile_archived_cursor(path, stat=stat, expected=cursor):
                cursor = self._cursor.get_record(path)
                return cursor is not None and size > cursor.byte_offset
            return _retry_due(cursor.next_retry_at)
        if self._is_hermes_database(path) or self._is_declared_codex_database(path):
            database_cursor = cursor

            def database_changed() -> bool:
                with bind_source_input(path) as binding:
                    if self._is_hermes_database(path) and (
                        database_cursor.captured_profile_key is None
                        or database_cursor.captured_profile_key != binding.captured_profile_key
                    ):
                        return True
                    if database_cursor.tail_hash == sqlite_source_revision(path, source_binding=binding):
                        return False
                    return self._database_content_changed(path, database_cursor, source_binding=binding)

            try:
                changed = capture_sqlite_read(database_changed)
            except (OSError, UnicodeDecodeError):
                # An unreadable or undecodable source is not proof of freshness.
                return True
            # A failed SQLite read is not proof of freshness either.
            return changed.value if isinstance(changed, Measured) else True
        if size == cursor.byte_size and cursor.content_fingerprint is not None:
            # Only an exact recorded observation authorizes the hot skip.
            # A bounded tail cannot prove that an earlier same-size prefix was
            # not rewritten, so any changed observation with modern tail
            # authority must return to the full route.
            if _cursor_stat_matches(cursor, stat):
                if cursor.byte_offset >= cursor.byte_size:
                    return False
                # polylogue-2qrx: ``byte_size`` matches the current file size
                # but ``byte_offset`` lags behind it. This is exactly the
                # state ``record_deferred_append_cursor`` leaves after a
                # bounded incomplete-tail probe (``_defer_incomplete_jsonl_
                # append``): it advances ``byte_size`` to the observed file
                # size but deliberately leaves ``byte_offset`` where it was,
                # pending a complete trailing record. Once the file then
                # stops changing (the writer finished), the stat-match check
                # above returned ``False`` unconditionally here forever --
                # measured live as 211 files / 414MB of stalled append
                # backlog, up to 329h stale on one 94.8MB lag, with zero
                # durable signal.
                #
                # An ordinary in-progress writer also leaves this exact state
                # between two close-together polls (a trailing record simply
                # not terminated *yet*), so escalate only once the deferred
                # observation is old enough to be implausible as "still
                # being written" -- ``cursor.updated_at`` is the timestamp of
                # that original bounded-probe deferral (nothing else touches
                # this row while the stat keeps matching).
                if not _cursor_age_exceeds(cursor, _STUCK_DEFERRED_APPEND_AGE_S):
                    return False
                # A real complete record past the original bounded window
                # reopens the normal append path; if truly nothing is there,
                # record a durable failure instead of parking silently again.
                if not self._defer_incomplete_jsonl_append(path, stat=stat, cursor=cursor, probe_bytes=None):
                    return True
                logger.warning(
                    "live.watcher: %s has no complete trailing record across its entire "
                    "outstanding tail (%d bytes past offset %d) and stopped changing; "
                    "marking failed for durable visibility instead of parking silently",
                    path,
                    cursor.byte_size - cursor.byte_offset,
                    cursor.byte_offset,
                )
                # The deferral above rewrote the row; compare against what it left.
                stuck = self._cursor.get_record(path) or cursor
                return not self._admit_observed_cursor_write(
                    "watcher.intake.cursor_stuck_append",
                    path,
                    stat=stat,
                    expected=stuck,
                    write=lambda: self._cursor.mark_failed(
                        path, authority=CursorPathAuthority.of_record(stuck), failed_stat=stat
                    ),
                )
            prefix_hash = cursor_prefix_hash(cursor.tail_hash)
            if prefix_hash is None:
                if self._reconcile_archived_cursor(path, stat=stat, expected=cursor):
                    reconciled = self._cursor.get_record(path)
                    return reconciled is None or size > reconciled.byte_offset
                return True
            try:
                current_prefix_hash, _bytes_read = sha256_range_from_path(
                    path,
                    start_offset=0,
                    end_offset=cursor.byte_offset,
                )
                final_stat = path.stat()
            except (EOFError, OSError):
                return True
            observation_changed = (
                final_stat.st_dev,
                final_stat.st_ino,
                final_stat.st_size,
                final_stat.st_mtime_ns,
                final_stat.st_ctime_ns,
            ) != (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if current_prefix_hash != prefix_hash or observation_changed:
                return True
            tail_hash = cursor_tail_hash(cursor.tail_hash)
            if tail_hash is None:
                return True
            rebase = CursorObservationRebase(
                path=path,
                expected=cursor,
                st_dev=final_stat.st_dev,
                st_ino=final_stat.st_ino,
                mtime_ns=final_stat.st_mtime_ns,
                tail_hash=encode_cursor_hash_authority(prefix_hash, tail_hash, ctime_ns=final_stat.st_ctime_ns),
            )
            if rebase_queue is None:
                self._cursor.rebase_authoritative_observations((rebase,))
            else:
                rebase_queue.append(rebase)
            return False
        if size > cursor.byte_offset:
            # A previous incomplete-tail probe recorded this exact filesystem
            # state.  The next useful observation is a write notification (or
            # a periodic scan that sees different stat evidence), not another
            # probe of the same unfinished record.  In particular, a single
            # oversized JSONL record must not make the 15-second safety scan
            # reread its first 64 MiB forever.
            if _cursor_stat_matches(cursor, stat):
                # polylogue-3r36h: the sibling branch above (size ==
                # cursor.byte_size) escalates a deferral that has sat at the
                # same byte state for _STUCK_DEFERRED_APPEND_AGE_S; this one
                # returned False forever instead, so a file whose recorded
                # byte_size disagrees with its size stalled with no durable
                # signal and nothing in list_retry_records. Same escalation,
                # same reason: an ordinary in-progress writer leaves this
                # state between two close polls, a finished one leaves it for
                # hours.
                if not _cursor_age_exceeds(cursor, _STUCK_DEFERRED_APPEND_AGE_S):
                    return False
                if not self._defer_incomplete_jsonl_append(path, stat=stat, cursor=cursor, probe_bytes=None):
                    return True
                logger.warning(
                    "live.watcher: %s has no complete trailing record across its entire "
                    "outstanding tail (%d bytes past offset %d) and stopped changing; "
                    "marking failed for durable visibility instead of parking silently",
                    path,
                    stat.st_size - cursor.byte_offset,
                    cursor.byte_offset,
                )
                # The deferral above rewrote the row; compare against what it left.
                stuck = self._cursor.get_record(path) or cursor
                return not self._admit_observed_cursor_write(
                    "watcher.intake.cursor_stuck_append",
                    path,
                    stat=stat,
                    expected=stuck,
                    write=lambda: self._cursor.mark_failed(
                        path, authority=CursorPathAuthority.of_record(stuck), failed_stat=stat
                    ),
                )
            return not self._defer_incomplete_jsonl_append(path, stat=stat, cursor=cursor)
        if cursor.content_fingerprint is None:
            return True
        try:
            fingerprint, _last_nl = fingerprint_file(path)
        except FileNotFoundError:
            return False
        return not (size == cursor.byte_size and fingerprint == cursor.content_fingerprint)

    def _defer_incomplete_jsonl_append(
        self,
        path: Path,
        *,
        stat: os.stat_result,
        cursor: CursorRecord,
        probe_bytes: int | None = _INCOMPLETE_APPEND_PROBE_BYTES,
    ) -> bool:
        """Record a grown-but-incomplete JSONL tail without scheduling ingest.

        ``probe_bytes=None`` (polylogue-2qrx escalation) reads the *entire*
        outstanding tail instead of the bounded default. The bounded probe
        exists so a routine watch/periodic tick never rereads a large
        unfinished record on every pass; but that same bound means a
        complete trailing record sitting just past the probe window (e.g. a
        multi-hundred-MB rollout whose byte range past ``cursor.byte_offset``
        happens to open with one oversized blob) is never found, and once
        the source file stops changing (the writing session ends), the
        stat-matches fast path below skips this cursor forever with zero
        durable signal -- see the 211-file, 414MB stalled-cursor backlog
        this bead measured. ``probe_bytes=None`` is only used for that one
        already-deferred-with-unchanged-stat case, so the full-tail read
        happens at most once per genuine state (immediately superseded by
        either a real append or a durable failure record, never repeated
        against the same stat).
        """
        if path.suffix.lower() not in {".jsonl", ".ndjson"}:
            return False
        if cursor.content_fingerprint is None:
            return False
        if cursor.st_dev is not None and cursor.st_dev != stat.st_dev:
            return False
        if cursor.st_ino is not None and cursor.st_ino != stat.st_ino:
            return False
        start_offset = max(cursor.byte_offset, 0)
        if stat.st_size <= start_offset:
            return False
        remaining_bytes = stat.st_size - start_offset
        bytes_to_probe = remaining_bytes if probe_bytes is None else min(remaining_bytes, probe_bytes)
        try:
            found_newline = _tail_begins_a_complete_record(path, start_offset=start_offset, scan_bytes=bytes_to_probe)
        except (OSError, MemoryError):
            # The probe could not be performed at all. That proves nothing
            # about the tail, but it is still a deferral, and it must be
            # RECORDED: returning True without recording left the cursor in
            # its previous state, so the next catch-up pass re-attempted the
            # identical probe against the identical stat, forever
            # (polylogue-dhkuu Finding A). Fall through to the deferral
            # record below.
            found_newline = False
        if found_newline:
            return False
        # The bounded probe can prove that no complete record begins at the
        # cursor, even when the unfinished record exceeds the probe budget.
        # Record this observed state so unchanged periodic catch-up scans skip
        # it; a subsequent append changes stat evidence and reopens the probe.
        # polylogue-hat0: this probe found no complete trailing record, not a
        # resolved authority state -- preserve any existing pending-authority
        # marker unchanged rather than clearing it.
        # A refused write (the file or cursor moved since the probe) is not a
        # deferral: the caller routes the file to ingest.
        return self._admit_observed_cursor_write(
            "watcher.intake.cursor_defer_append",
            path,
            stat=stat,
            expected=cursor,
            write=lambda: record_deferred_append_cursor(
                self._cursor,
                path,
                cursor=cursor,
                parser_fingerprint=_PARSER_FINGERPRINT,
                source_name=self._source_name_for(path),
                deferred_end_offset=cursor.deferred_end_offset,
            ),
        )

    def _invalidate_deferred_full_cursor(self, path: Path, *, stat: os.stat_result) -> None:
        """Clear a busy-handoff defer when current bytes reject archive authority."""

        existing = self._cursor.get_record(path)
        authority = CursorPathAuthority.of_record(existing) if existing is not None else None
        if authority is None:
            try:
                authority = CursorPathAuthority.observe(path)
            except FileNotFoundError:
                self._cursor.mark_failed(path, authority=None)
                return
        updated = self._cursor.set(
            path,
            stat.st_size,
            authority=authority,
            byte_offset=0,
            last_complete_newline=0,
            parser_fingerprint=_PARSER_FINGERPRINT,
            content_fingerprint=None,
            tail_hash=None,
            source_name=self._source_name_for(path),
            st_dev=stat.st_dev,
            st_ino=stat.st_ino,
            mtime_ns=stat.st_mtime_ns,
            failure_count=0,
            next_retry_at=None,
            excluded=False,
            allow_backward=True,
        )
        if not updated:
            raise sqlite3.OperationalError(f"failed to invalidate deferred cursor for {path}")

    def _reconcile_archived_cursor(self, path: Path, *, stat: os.stat_result, expected: CursorRecord | None) -> bool:
        """Restore a missing/stale cursor from proven archive raw state."""

        outcome = self._reconcile_archived_cursor_outcome(path, stat=stat, expected=expected)
        return outcome is _ArchivedCursorReconciliation.RECONCILED

    @contextmanager
    def _archived_cursor_reconciliation_scope(self) -> Iterator[None]:
        """Share one read-only connection pair across a bulk planning pass.

        Cursor reconciliation runs per cursor-less file; during a 20k-file
        catch-up plan, opening source.db+index.db fresh for every file cost
        ~10 minutes of silent startup CPU (observed live 2026-07-18). The
        scope is deliberately bounded to ONE planning pass — a long-lived
        cached connection would keep reading a replaced index.db inode
        across a blue-green generation swap.
        """
        if _source_tier_acquisition_required():
            self._archived_cursor_conns = None
            yield
            return
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        source_db = archive_root / "source.db"
        try:
            index_db = _published_index_path(archive_root)
        except (ArchiveLocationError, OSError, UnicodeError):
            self._archived_cursor_conns = None
            yield
            return
        conns: tuple[sqlite3.Connection, sqlite3.Connection] | None = None
        if source_db.exists() and index_db.exists():
            try:
                source_conn = open_readonly_connection(source_db, timeout=1.0)
                try:
                    index_conn = open_readonly_connection(index_db, timeout=1.0)
                except sqlite3.Error:
                    source_conn.close()
                    raise
                conns = (source_conn, index_conn)
            except sqlite3.Error:
                conns = None
        self._archived_cursor_conns = conns
        try:
            yield
        finally:
            self._archived_cursor_conns = None
            if conns is not None:
                for conn in conns:
                    with suppress(sqlite3.Error):
                        conn.close()

    @staticmethod
    def _archived_cursor_row(
        path: Path,
        *,
        source_conn: sqlite3.Connection,
        index_conn: sqlite3.Connection,
    ) -> tuple[object, ...] | None:
        """Newest session-bearing raw for ``path`` that the index contains.

        The row is ``(raw_id, origin, blob_hash, blob_size, acquired_at_ms)``.
        """
        rows = source_conn.execute(
            f"""
            SELECT raw_id, origin, blob_hash, blob_size, acquired_at_ms,
                   (SELECT profile_key FROM raw_profile_identity_receipts AS p
                    WHERE p.raw_id = raw_sessions.raw_id)
            FROM raw_sessions
            WHERE source_path = ?
              AND COALESCE(source_index, 0) >= 0
              AND (parsed_at_ms IS NOT NULL OR revision_authority IN ('asserted', 'byte_proven'))
              AND parse_error IS NULL
            ORDER BY {raw_receipt_order_sql("raw_sessions")} DESC, raw_id DESC
            """,
            (str(path),),
        ).fetchall()
        return next(
            (
                candidate
                for candidate in rows
                if index_conn.execute(
                    "SELECT 1 FROM sessions WHERE raw_id = ? LIMIT 1",
                    (candidate[0],),
                ).fetchone()
                is not None
            ),
            None,
        )

    @staticmethod
    def _decided_unresolved_cursor_row(
        path: Path,
        *,
        source_conn: sqlite3.Connection,
    ) -> tuple[object, ...] | None:
        """Newest raw for ``path`` whose membership arbitration decided unresolved.

        Such a raw is never parsed and never reaches the index, so
        :meth:`_archived_cursor_row` cannot see it; without this the cursor
        can never be restored from it and every start re-reads the whole
        file to reach the same decided verdict. Its retained bytes are still
        proof of what was consumed, and the caller re-verifies them against
        the archived blob hash before advancing, so a changed observation
        still returns through full ingest -- the only route that can carry
        the new evidence the verdict needs. The row has the shape of
        :meth:`_archived_cursor_row`'s.
        """
        return cast(
            "tuple[object, ...] | None",
            source_conn.execute(
                f"""
                SELECT r.raw_id, r.origin, r.blob_hash, r.blob_size, r.acquired_at_ms,
                       (SELECT profile_key FROM raw_profile_identity_receipts AS p WHERE p.raw_id = r.raw_id)
                FROM raw_sessions AS r
                WHERE r.source_path = ?
                  AND COALESCE(r.source_index, 0) >= 0
                  AND r.parse_error IS NULL
                  AND ({decided_unresolved_membership_sql("r")})
                ORDER BY {raw_receipt_order_sql("r")} DESC, r.raw_id DESC
                LIMIT 1
                """,
                (str(path),),
            ).fetchone(),
        )

    @classmethod
    def _newest_archived_outcome_row(
        cls,
        path: Path,
        *,
        source_conn: sqlite3.Connection,
        index_conn: sqlite3.Connection,
    ) -> tuple[object, ...] | None:
        """Newest raw for ``path`` with a settled outcome: materialized or decided unresolved.

        Both classes are proof of consumed bytes, so the newest acquisition
        across them is what the cursor restores to, ordered exactly as each
        class orders itself (acquisition time, then ``raw_id``). Preferring
        any materialized raw would restore an older, shorter prefix behind a
        newer decided raw, and every restart would re-read the newer bytes
        only to reach the same verdict.
        """
        candidates = [
            row
            for row in (
                cls._archived_cursor_row(path, source_conn=source_conn, index_conn=index_conn),
                cls._decided_unresolved_cursor_row(path, source_conn=source_conn),
            )
            if row is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda row: (int(cast("int", row[4])), str(row[0])))

    @classmethod
    def _path_corroborated_by_index(
        cls,
        path: Path,
        *,
        source_conn: sqlite3.Connection,
        index_conn: sqlite3.Connection,
    ) -> bool:
        """True unless ``path`` has session authority the index cannot show.

        A raw its artifact classification declares non-session (a Claude Code
        tool-result sidecar, a workflow fact) is parsed but never yields a
        session, so the index can never show it; counting it as session
        authority demoted every settled sidecar's cursor and re-ingested it on
        each periodic scan.
        """
        has_session_raw = source_conn.execute(
            f"""
            SELECT 1 FROM raw_sessions
            WHERE source_path = ?
              AND COALESCE(source_index, 0) >= 0
              AND (parsed_at_ms IS NOT NULL OR revision_authority IN ('asserted', 'byte_proven'))
              AND parse_error IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM raw_artifacts AS a
                  WHERE a.raw_id = raw_sessions.raw_id
                    AND a.parse_as_session = 0
                    AND a.artifact_kind NOT IN ({_FAILURE_EVIDENCE_KINDS_SQL})
              )
            LIMIT 1
            """,
            (str(path),),
        ).fetchone()
        if has_session_raw is None:
            # A killed candidate can leave a proven revision before its
            # source parse marker commits. Raw-only paths have no session
            # authority to corroborate.
            return True
        return cls._archived_cursor_row(path, source_conn=source_conn, index_conn=index_conn) is not None

    def _cursor_skip_corroborated_by_index(self, path: Path) -> bool:
        """Whether a cursor-trust skip for ``path`` is backed by a materialized session.

        Called for each cursor-based skip, including when the index holds
        sessions for other paths. A path with no parsed or proven session raw
        is treated as corroborated so unrelated cursor kinds are unaffected.
        """
        shared = self._archived_cursor_conns
        try:
            if shared is not None:
                return self._path_corroborated_by_index(path, source_conn=shared[0], index_conn=shared[1])
            archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
            source_db = archive_root / "source.db"
            index_db = _published_index_path(archive_root)
            if not source_db.exists() or not index_db.exists():
                return True
            with (
                closing(open_readonly_connection(source_db, timeout=1.0)) as source_conn,
                closing(open_readonly_connection(index_db, timeout=1.0)) as index_conn,
            ):
                return self._path_corroborated_by_index(path, source_conn=source_conn, index_conn=index_conn)
        except (ArchiveLocationError, OSError, UnicodeError, sqlite3.Error):
            # Cannot prove absence on a transient DB error -- don't force a
            # spurious re-ingest of an otherwise-healthy cursor.
            return True

    def _reconcile_archived_cursor_outcome(
        self,
        path: Path,
        *,
        stat: os.stat_result,
        expected: CursorRecord | None,
    ) -> _ArchivedCursorReconciliation:
        """Restore a missing/stale cursor from proven archive raw state.

        A daemon interruption can leave the archive source tier populated but
        the live cursor absent. Without this repair, startup catch-up replays
        the whole source file through the archive writer again. The archive row
        proves the stored prefix for the exact source path; if the live file
        has grown since that row was written, the cursor is restored to the
        archived prefix so catch-up can take the append path instead of
        parsing the whole active JSONL again.
        """
        if _source_tier_acquisition_required():
            # Derived corroboration is inapplicable in acquire-only mode.
            # Force a fresh source observation instead of deferring on an
            # index that this mode is explicitly forbidden to read.
            return _ArchivedCursorReconciliation.INCOMPATIBLE
        # ``expected`` is the cursor row the caller decided from (bulk-read
        # for a page); the restore re-checks it under the writer.
        shared = self._archived_cursor_conns
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        try:
            if shared is not None:
                row = self._newest_archived_outcome_row(path, source_conn=shared[0], index_conn=shared[1])
            else:
                source_db = archive_root / "source.db"
                index_db = _published_index_path(archive_root)
                if not source_db.exists() or not index_db.exists():
                    return _ArchivedCursorReconciliation.UNAVAILABLE
                with (
                    closing(open_readonly_connection(source_db, timeout=1.0)) as source_conn,
                    closing(open_readonly_connection(index_db, timeout=1.0)) as index_conn,
                ):
                    row = self._newest_archived_outcome_row(path, source_conn=source_conn, index_conn=index_conn)
        except (ArchiveLocationError, OSError, UnicodeError, sqlite3.Error):
            return _ArchivedCursorReconciliation.UNAVAILABLE
        if row is None:
            return _ArchivedCursorReconciliation.INCOMPATIBLE
        _raw_id, origin, blob_hash, blob_size, _acquired_at_ms = row[:5]
        captured_profile_key = cast("str | None", row[5]) if len(row) > 5 else None
        if origin is not None and provider_from_origin(Origin.from_string(str(origin))) is Provider.HERMES:
            from polylogue.sources.parsers.hermes_identity import observe_profile_namespace

            if captured_profile_key is None:
                return _ArchivedCursorReconciliation.INCOMPATIBLE
            try:
                observed_profile = observe_profile_namespace(path, stat)
            except OSError:
                return _ArchivedCursorReconciliation.UNAVAILABLE
            if observed_profile.key != captured_profile_key:
                return _ArchivedCursorReconciliation.INCOMPATIBLE
        archived_size = int(cast("int | None", blob_size) or 0)
        current_size = int(stat.st_size)
        if archived_size <= 0 or archived_size > current_size:
            return _ArchivedCursorReconciliation.INCOMPATIBLE
        if isinstance(blob_hash, bytes):
            content_fingerprint = blob_hash.hex()
        elif isinstance(blob_hash, str):
            content_fingerprint = blob_hash.lower()
        else:
            return _ArchivedCursorReconciliation.INCOMPATIBLE
        if not _archive_blob_exists(archive_root, content_fingerprint):
            return _ArchivedCursorReconciliation.INCOMPATIBLE
        try:
            current_fingerprint, _fingerprint_bytes = sha256_range_from_path(
                path,
                start_offset=0,
                end_offset=archived_size,
            )
            if current_fingerprint != content_fingerprint:
                return _ArchivedCursorReconciliation.INCOMPATIBLE
            if archived_size == current_size:
                tail_hash, last_complete_newline, _bytes_read = tail_hash_and_last_complete_newline_from_path(
                    path, current_size
                )
                if path.suffix.lower() not in {".jsonl", ".ndjson"}:
                    last_complete_newline = archived_size
            else:
                tail_hash, _bytes_read = tail_hash_from_path(path, archived_size)
                last_complete_newline = archived_size
            post_read_stat = path.stat()
        except OSError:
            return _ArchivedCursorReconciliation.UNAVAILABLE
        if (
            post_read_stat.st_dev,
            post_read_stat.st_ino,
            post_read_stat.st_size,
            post_read_stat.st_mtime_ns,
            post_read_stat.st_ctime_ns,
        ) != (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns):
            return _ArchivedCursorReconciliation.UNAVAILABLE
        source_provider = provider_from_origin(Origin.from_string(str(origin))) if origin is not None else None
        if source_provider is Provider.CLAUDE_CODE:
            semantic_tail_hash = claude_semantic_frontier_for_prefix(path, archived_size)
            if semantic_tail_hash is None:
                return _ArchivedCursorReconciliation.INCOMPATIBLE
            tail_hash = semantic_tail_hash
        else:
            tail_hash = encode_cursor_hash_authority(
                content_fingerprint,
                tail_hash,
                ctime_ns=stat.st_ctime_ns,
            )
        try:
            authority = CursorPathAuthority.observe(path)
        except FileNotFoundError:
            return _ArchivedCursorReconciliation.UNAVAILABLE
        if not declares_profile_identity(source_provider):
            # Acquisition captures a profile namespace only for Hermes (and
            # not-yet-detected) inputs; every other origin's raw carries none,
            # so its cursor carries none either.
            authority = CursorPathAuthority(authority.canonical_source_path, None)
        if authority.captured_profile_key != captured_profile_key:
            # The archived raw was captured under another profile namespace.
            return _ArchivedCursorReconciliation.INCOMPATIBLE
        source_name = (
            provider_from_origin(Origin.from_string(str(origin))).value
            if origin is not None
            else self._source_name_for(path)
        )

        def restore() -> None:
            self._cursor.set(
                path,
                archived_size,
                authority=authority,
                byte_offset=last_complete_newline,
                last_complete_newline=last_complete_newline,
                parser_fingerprint=_PARSER_FINGERPRINT,
                content_fingerprint=content_fingerprint,
                tail_hash=tail_hash,
                source_name=source_name,
                st_dev=stat.st_dev,
                st_ino=stat.st_ino,
                mtime_ns=stat.st_mtime_ns,
            )
            self._cursor.reset_failures(path)

        # The hashes above ran without the writer; the restore re-checks the
        # file and cursor it was proved against before it writes.
        if not self._admit_observed_cursor_write(
            "watcher.intake.cursor_reconcile", path, stat=stat, expected=expected, write=restore
        ):
            return _ArchivedCursorReconciliation.UNAVAILABLE
        logger.info("live.watcher: reconciled cursor from archive source row for %s", path)
        return _ArchivedCursorReconciliation.RECONCILED

    async def _ingest_files(
        self,
        paths: list[Path],
        *,
        queued_file_count: int | None = None,
        skipped_file_count: int = 0,
        whole_archive_convergence: bool = False,
    ) -> LiveBatchMetrics:
        """Ingest one admitted page through the daemon live batch processor.

        ``LiveBatchProcessor`` self-admits its ops-tier publications and its
        archive publication.  Do not wrap the whole page in the daemon writer:
        planning, parsing and convergence must leave maintenance able to run.

        The lock is process-local ordering on top of that: one live ingest at
        a time in this process.
        """
        from polylogue.core.degraded import is_fully_degraded

        if not is_fully_degraded():
            # A degraded batch returns its skip metrics without the gate.
            self._batch_processor.require_cursor_authority(paths)
        async with self._ingest_lock:
            return await self._batch_processor.ingest_files(
                paths,
                queued_file_count=queued_file_count,
                skipped_file_count=skipped_file_count,
                max_pass_seconds=_LIVE_INGEST_MAX_PASS_SECONDS,
                whole_archive_convergence=whole_archive_convergence,
            )

    async def _converge_embeddings_off_writer(self, paths: Sequence[Path]) -> None:
        """Converge this batch's embeddings after the ingest lease is released."""
        if self._embedding_owner is None or not paths:
            return
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        try:
            await self._embedding_owner(archive_root / "index.db", tuple(paths))
        except Exception:
            # The deferred obligation is already recorded as convergence debt,
            # so a refused or failed pass retries there rather than failing an
            # ingest batch whose source records are already durable.
            logger.warning("live.watcher: lease-free embedding convergence did not complete", exc_info=True)

    async def _converge_session_profiles_off_writer(self, session_ids: Sequence[str]) -> None:
        """Submit the bounded changed-session scope after ingest releases the gate."""
        if self._session_profile_callback is None or not session_ids:
            return
        try:
            await self._session_profile_callback(tuple(dict.fromkeys(session_ids)))
        except Exception:
            # Source admission is already durable.  The owner reconstructs
            # missed work from output inspection in its periodic no-hint pass.
            logger.warning("live.watcher: lease-free session profile convergence did not complete", exc_info=True)

    async def _run_coordinated(self, actor: str, operation: Callable[[], Awaitable[None]]) -> None:
        """Run a complete watcher write batch under the injected coordinator."""
        if self._write_coordinator is None:
            await operation()
            return
        await self._write_coordinator.run(actor, operation)

    def _source_name_for(self, path: Path) -> str:
        source = deepest_source_for_path(path, self._sources)
        if source is not None:
            return source.name
        return path.parent.name

    def _file_symlink_escapes_source(self, path: Path) -> bool:
        """Whether ``path`` is a file symlink whose target leaves its source root.

        Discovery refuses such a link as ``escaping_symlink``; a live event for
        the same link must not admit material the source was never configured
        to read. An explicitly declared file is its own containment.
        """
        if not path.is_symlink():
            return False
        source = deepest_source_for_path(path, self._sources)
        if source is None:
            return True
        try:
            target = path.resolve()
            if source.exact_paths is not None and target in source.exact_paths:
                return False
            return not target.is_relative_to(source.root.resolve())
        except OSError:
            return True

    def _source_accepts(self, path: Path) -> bool:
        source = deepest_source_for_path(path, self._sources)
        return source.accepts(path) if source is not None else False

    def _is_hermes_database(self, path: Path) -> bool:
        source = deepest_source_for_path(path, self._sources)
        return (
            source is not None
            and source.name == Provider.HERMES.value
            and source.accepts(path)
            and is_sqlite_path(path)
        )

    def _is_declared_codex_database(self, path: Path) -> bool:
        """Return whether *path* is a Codex database this watcher acquires.

        Codex state lives beside the rollout files under a suffix-filtered
        watch source, so the declaration in ``origin_specs`` -- not the
        filesystem -- decides which members are acquired at all.
        """
        from polylogue.sources.origin_specs import database_capability_for_provider

        if not is_sqlite_path(path):
            return False
        source = deepest_source_for_path(path, self._sources)
        if source is None or not str(source.name).startswith("codex"):
            return False
        capability = database_capability_for_provider(Provider.CODEX)
        if capability is None:
            return False
        member = capability.member(path.name)
        return member is not None and member.disposition != "out-of-scope"

    def _database_content_changed(
        self, path: Path, cursor: CursorRecord, *, source_binding: SourceInputBinding
    ) -> bool:
        """Return whether a database's logical content moved past the cursor.

        A database's page image differs after every commit, checkpoint and
        vacuum, so filesystem state can only prove that nothing happened. Once
        it has changed, the acquired logical revision is what decides whether
        there is anything to acquire: re-snapshotting an unchanged database
        writes a whole second page image for content the archive already holds.

        Only a recorded fingerprint that is itself a logical revision can
        answer; every other cursor shape falls through to work, which is what
        the filesystem observation already claimed. The revision is scoped to
        the member's declared logical tables, exactly as acquisition records
        it -- a whole-database digest would report work for a commit in a
        table nothing reads.
        """
        recorded = cursor.content_fingerprint
        if recorded is None:
            return True
        try:
            return sqlite_member_revision(path, source_binding=source_binding) != recorded
        except (sqlite3.Error, OSError, UnicodeDecodeError):
            # Acquisition owns the consistent read and reports its own typed
            # failure; a locked or damaged database is not silently fresh.
            return True

    def _canonical_watch_path(self, path: Path) -> Path | None:
        if self._source_accepts(path):
            return None if self._file_symlink_escapes_source(path) else path
        database = sqlite_database_for_sidecar(path)
        if database is not None and self._is_hermes_database(database):
            return database
        return None

    def _source_for_directory(self, path: Path) -> WatchSource | None:
        """Return the watched source whose layout reaches this directory."""

        source = deepest_source_for_path(path, self._sources)
        if source is None:
            return None
        return source if source.admits_directory(path) else None

    def _directory_is_watch_relevant(self, path: Path) -> bool:
        """Return whether a directory is owned or leads to a configured source root."""

        if self._source_for_directory(path) is not None:
            return True
        try:
            resolved = path.resolve()
        except OSError:
            return False
        for source in self._sources:
            try:
                if source.root.resolve().is_relative_to(resolved):
                    return True
            except (OSError, ValueError):
                continue
        return False

    def _enqueue_added_directory(self, directory: Path) -> None:
        """Cover files created before a recursive watcher installs its new sub-watch."""

        if not self._directory_is_watch_relevant(directory):
            return
        self._note_intake_hint(directory)

    def _watch_filter(self, _change: object, path: str) -> bool:
        """Accept configured source files under hidden canonical roots.

        watchfiles' default filter ignores hidden directories. Polylogue's
        normal roots live under paths such as ~/.claude, ~/.codex, ~/.local,
        and repo-local .cache, so the generic filter silently drops real source
        writes. This filter keeps the project's own source/suffix predicate as
        the gate instead.
        """
        observed_path = Path(path)
        return self._canonical_watch_path(observed_path) is not None or (
            observed_path.is_dir() and self._directory_is_watch_relevant(observed_path)
        )


def default_sources(*, hermes_root: Path | None = None) -> tuple[WatchSource, ...]:
    """Discover the default live-source roots from XDG/home conventions.

    Includes the archive inbox so that ``polylogue ingest PATH``
    (which stages to ``archive_root()/inbox``) is observed by the
    daemon-owned watcher.

    """
    from polylogue.paths import (
        antigravity_cli_path,
        antigravity_path,
        archive_root,
        browser_capture_spool_root,
        claude_code_path,
        claude_code_todos_path,
        codex_memories_path,
        codex_path,
        gemini_cli_path,
        hermes_sessions_path,
    )

    def declared(name: str, root: Path) -> WatchSource:
        return WatchSource(name=name, root=root, layout=declared_source_layout(name))

    # Each source is admitted only at the positions its declared layout names
    # (``polylogue.sources.source_layout``); overlapping roots never descend
    # into each other's trees because no layout reaches them.
    return (
        declared("claude-code", claude_code_path()),
        # polylogue-t0p: Claude Code's live plan-snapshot directory
        # (~/.claude/todos/) is a sibling of claude_code_path(), not nested
        # under it.
        declared("claude-code-todos", claude_code_todos_path()),
        # polylogue-ximhz: ``~/.claude/history.jsonl`` is the prompt-submission
        # log whose rows carry the paste evidence no transcript records, and it
        # sits beside the sessions root rather than under it. Its layout is
        # that one file; the install directory is never descended.
        declared("claude-code-history", claude_code_path().parent),
        declared("codex", codex_path()),
        # polylogue-0jf4: Codex keeps live SQLite state (thread titles, spawn
        # topology, goals, memories) and the install-level ``session_index``
        # and ``history`` JSONL sidecars as siblings of sessions/. The layout
        # names each declared database member and sidecar at its position;
        # the acquisition path (sources/live/batch.py) still verifies table
        # shape before treating a database as in-scope evidence.
        declared("codex-state", codex_path().parent),
        # polylogue-rovf5: harness-authored memory documents in
        # ~/.codex/memories/.
        declared("codex-memories", codex_memories_path()),
        declared("gemini-cli", gemini_cli_path()),
        # Hermes keeps state.db, verification_evidence.db, session snapshots,
        # NeMo Relay ATIF documents and the ATOF stream under its home, and a
        # complete home per profile under profiles/<name>/.
        declared("hermes", hermes_root if hermes_root is not None else hermes_sessions_path()),
        # Antigravity conversations are opaque protobufs. The ordinary live
        # batch route hands those files to the vendor language-server adapter;
        # brain documents and metadata remain source artifacts.
        declared("antigravity", antigravity_path()),
        # Antigravity's CLI writes one trajectory SQLite store per
        # conversation under ~/.gemini/antigravity-cli/conversations/.
        declared("antigravity-cli", antigravity_cli_path()),
        declared("browser-capture", browser_capture_spool_root()),
        # #1683: the inbox admits archive, zip, and json-line formats at any
        # depth so that GDPR exports (typically .zip) and raw .json dumps are
        # observed.
        declared("inbox", archive_root() / "inbox"),
        *hook_carrier_watch_sources(hook_spool_sources()),
    )


#: Watch sources whose directory Polylogue itself creates and writes; their
#: existence proves nothing about any tool's material.
POLYLOGUE_OWNED_SOURCE_NAMES = frozenset({"browser-capture", "inbox"})


def daemon_watch_sources(*, hermes_root: Path | None = None) -> tuple[WatchSource, ...]:
    """The daemon's watch set: every origin at its canonical location.

    There are no custom source roots. Each origin is acquired from the place
    its tool writes it, the archive inbox is a watched drop directory, and a
    relocated tool directory is followed by a symlink at the canonical path
    rather than by configuration. ``polylogue import`` stages outside this set
    (``operations/import_staging.py``): its ``ingest`` operation is the only
    route that acquires an import. Hermes's root is its own ``HERMES_HOME``,
    and the Polylogue-owned browser-capture spool lives under the archive root.
    """
    return default_sources(hermes_root=hermes_root)


def _cursor_db_path(polylogue: ArchiveRootOwner) -> Path:
    """Use the archive ops tier without opening a lazy archive backend.

    ``ops.db`` owns live cursor state: ``CursorStore`` writes every cursor row
    through ``upsert_archive_ingest_cursor(..., ArchiveTier.OPS)``, so this
    path is the canonical one rather than a fallback.

    A watcher only requires the ``ArchiveRootOwner`` contract.  Reading the
    richer ``Polylogue.backend`` property here forces SQLite bootstrap during
    daemon startup, after its admission has been released.  Cursor creation
    already receives the ops path explicitly and initialization is admitted by
    the watcher before its first mutable cursor operation.
    """
    return Path(polylogue.archive_root) / "ops.db"


def _tail_begins_a_complete_record(path: Path, *, start_offset: int, scan_bytes: int) -> bool:
    """Return whether a b"\n" occurs within ``scan_bytes`` past ``start_offset``.

    The question is the position of the first newline, so the scan reads
    fixed-size chunks and stops at the first one that contains it. Sizing a
    single ``read()`` by the outstanding tail instead made the reader's
    working set a function of its untrusted input: an unterminated multi-GB
    record raised ``MemoryError`` (not an ``OSError``) and the caller's
    handler never recorded the deferral, so the attempt repeated on every
    catch-up pass (polylogue-dhkuu Finding A).
    """

    remaining = scan_bytes
    with path.open("rb") as handle:
        handle.seek(start_offset)
        while remaining > 0:
            chunk = handle.read(min(remaining, _INCOMPLETE_APPEND_PROBE_CHUNK_BYTES))
            if not chunk:
                return False
            if b"\n" in chunk:
                return True
            remaining -= len(chunk)
    return False


def _cursor_age_exceeds(cursor: CursorRecord, min_age_s: float) -> bool:
    """Return True when ``cursor.updated_at`` is older than ``min_age_s``.

    A malformed/missing timestamp is treated as old (fail toward escalating
    rather than toward parking silently forever, matching this check's own
    purpose).
    """
    try:
        updated_at = datetime.fromisoformat(cursor.updated_at)
    except ValueError:
        return True
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)
    return (datetime.now(UTC) - updated_at).total_seconds() >= min_age_s


def _retry_due(next_retry_at: str | None) -> bool:
    if not next_retry_at:
        return True
    retry_at = _parse_retry_at(next_retry_at)
    if retry_at is None:
        return True
    return retry_at <= datetime.now(UTC)


def _parse_retry_at(next_retry_at: str | None) -> datetime | None:
    if not next_retry_at:
        return None
    try:
        retry_at = datetime.fromisoformat(next_retry_at)
    except ValueError:
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    return retry_at


def _cursor_stat_matches(cursor: CursorRecord, stat: os.stat_result) -> bool:
    """Return True when the cursor was written for this exact file state."""

    return (
        cursor.st_dev == stat.st_dev
        and cursor.st_ino == stat.st_ino
        and cursor.mtime_ns == stat.st_mtime_ns
        and cursor_ctime_ns(cursor.tail_hash) == stat.st_ctime_ns
    )


__all__ = ["LiveWatcher", "WatchSource", "WatcherRootsUnavailableError", "default_sources"]
