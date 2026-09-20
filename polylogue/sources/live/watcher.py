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
from collections.abc import Awaitable, Callable, Iterable, Iterator, Sequence
from contextlib import closing, contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, cast

from polylogue.archive.revision_authority import decided_unresolved_membership_sql
from polylogue.core.enums import Origin, Provider
from polylogue.core.protocols import ArchiveRootOwner
from polylogue.core.source_halts import halted_sources
from polylogue.core.sources import provider_from_origin
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
    CursorRecord,
    CursorStore,
)
from polylogue.sources.live.deferred_cursor import record_deferred_append_cursor
from polylogue.sources.live.metrics import LiveBatchMetrics
from polylogue.sources.live.parse_prefetch import LiveParseStage
from polylogue.sources.live.source_selection import deepest_source_for_path
from polylogue.sources.sqlite_snapshot import (
    is_sqlite_path,
    sqlite_database_for_sidecar,
    sqlite_member_revision,
    sqlite_source_revision,
)
from polylogue.storage.archive_identity import ArchiveLocationError, resolve_active_index_path

logger = get_logger(__name__)
# Bump whenever parser semantics change the values derived from already-
# observed bytes. A cursor stamped with a superseded fingerprint is treated
# as needing work, which routes its source back through parse on the next
# watcher pass -- the production convergence route, not a manual rebuild.
# v3: tool-result outcomes now derive `is_error` from an explicit exit code
# (#4539), so records parsed under v2 retain a stale unknown outcome.
_PARSER_FINGERPRINT = "live-batched-v3"
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
# the same checkpoints end the pass with ``WriteHoldBudgetError``.
_LIVE_INGEST_MAX_PASS_SECONDS = 20.0
_INCOMPLETE_APPEND_PROBE_BYTES = 64 * 1024 * 1024
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
    logger.info(
        "%s complete: read=%.1f MB input=%.1f MB read_amp=%.6fx append_files=%d full_files=%d "
        "succeeded=%d failed=%d excluded=%d parse_s=%.3f convergence_s=%.3f stages=%s "
        "time_budget_exceeded=%s",
        prefix,
        source_payload_read_bytes / 1e6,
        input_bytes / 1e6,
        read_amp,
        getattr(metrics, "append_file_count", 0),
        getattr(metrics, "full_file_count", 0),
        getattr(metrics, "succeeded_file_count", 0),
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


class _SourceTreeWalk:
    """One source's catch-up walk, following deliberate directory symlinks.

    A directory symlink under a watch root is a deliberate placement -- the
    inbox exposes whole export corpora that way -- so the walk enters it.
    Two things bound what that admits:

    ``_walked`` holds every directory identity already entered, so a link to
    an ancestor or to an already-walked tree is not followed a second time; a
    cycle terminates and one corpus reachable under two names is one candidate.

    ``_containment`` holds, per walked directory, the real root of the tree
    the walk is inside: the source root, or the target of the last symlink it
    followed. A file whose resolved path leaves that tree is a symlink
    escaping the watch root and is never a candidate.
    """

    def __init__(self, source: WatchSource) -> None:
        self._source = source
        root = source.root.resolve()
        self._walked = {identity for identity in (_directory_identity(source.root),) if identity is not None}
        self._containment: dict[str, Path] = {str(source.root): root}

    def descendable(self, directory: Path, dirnames: list[str]) -> list[str]:
        """Return the child directory names this walk may descend into."""
        inherited = self._containment.get(str(directory), self._source.root.resolve())
        descendable: list[str] = []
        for dirname in dirnames:
            child = directory / dirname
            if self._source.ignores_directory(child):
                continue
            identity = _directory_identity(child)
            if identity is None or identity in self._walked:
                continue
            self._walked.add(identity)
            self._containment[str(child)] = child.resolve() if child.is_symlink() else inherited
            descendable.append(dirname)
        return descendable

    def contains(self, directory: Path, path: Path) -> bool:
        """Whether ``path`` stays inside the real tree the walk is in."""
        root = self._containment.get(str(directory), self._source.root.resolve())
        try:
            return path.resolve().is_relative_to(root)
        except OSError:
            return False


#: Directory names no watched root ever descends into.
_DEFAULT_IGNORED_DIR_NAMES: frozenset[str] = frozenset({".git", "__pycache__", "node_modules", "venv", ".venv"})


@dataclass(frozen=True, slots=True)
class WatchSource:
    """A directory to watch for live session files."""

    name: str
    root: Path
    suffixes: tuple[str, ...] = (".jsonl",)
    ignored_dir_names: frozenset[str] = _DEFAULT_IGNORED_DIR_NAMES
    # Hook sources carry durable topology identity.  Ordinary sources retain
    # their historical name-only contract.
    source_id: str | None = None
    role: str | None = None
    # Most provider sources use OriginSpec path rules as an admission
    # escape-hatch for extensionless or otherwise path-scoped artifacts. A
    # source may disable that routing when its suffix set is deliberately a
    # hard boundary (for example, the default Codex state database source).
    allow_path_scoped_artifacts: bool = True

    def exists(self) -> bool:
        return self.root.exists()

    def accepts(self, path: Path) -> bool:
        name = path.name.lower()
        # A declared artifact rule is the source-owned escape hatch for
        # extensionless and path-scoped artifacts. Check it before suffixes,
        # then keep suffixes as the ordinary source filter.
        from polylogue.sources.origin_specs import artifact_rule_for_path

        try:
            provider = Provider.from_string(self.name)
        except ValueError:
            return any(name.endswith(suffix) for suffix in self.suffixes)
        if self.allow_path_scoped_artifacts and artifact_rule_for_path(provider, str(path)) is not None:
            return True
        return any(name.endswith(suffix) for suffix in self.suffixes)

    def ignores_directory(self, path: Path) -> bool:
        """Return whether a subtree cannot contain a live source artifact."""
        return path.name in self.ignored_dir_names


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
            suffixes=(".ndjson",),
            source_id=f"{spec.source_id}:{provider}",
            role=spec.role,
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


class LiveWatcher:
    """Filesystem watch that wakes the fair-intake dispatcher.

    Acquisition has one route: ``FairIntakeDispatcher`` discovers, plans and
    admits pages through ``FileIntakeAdapter``, which calls
    :meth:`_ingest_files`. This watcher owns no queue, no catch-up scan and
    no schedule of its own -- it turns a filesystem event into a bumped
    intake revision plus a wakeup, so the dispatcher's next pass is prompt
    rather than waiting out its idle delay. It also owns the batch
    processor, the cursor store and the parse stage the adapter's ingest
    runs through.
    """

    def __init__(
        self,
        polylogue: ArchiveRootOwner,
        sources: Iterable[WatchSource],
        *,
        cursor: CursorStore | None = None,
        max_workers: int | None = None,
        converger: object | None = None,  # DaemonConverger | None — avoids circular import
        event_emitter: LiveBatchEventEmitter | None = None,
        write_coordinator: WriteCoordinator | None = None,
        parse_stage: LiveParseStage | None = None,
        embedding_owner: EmbeddingConvergenceOwner | None = None,
        session_profile_callback: SessionProfileConvergenceCallback | None = None,
        intake_wakeup: asyncio.Event | None = None,
    ) -> None:
        self._polylogue = polylogue
        self._sources = tuple(sources)
        self._cursor = cursor or CursorStore(
            _cursor_db_path(polylogue),
            initialize=write_coordinator is None,
            ops_db_path=Path(polylogue.archive_root) / "ops.db",
        )
        self._max_workers = max_workers
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
        # polylogue-wf8a: always on -- pre-parsing runs entirely BEFORE the
        # write coordinator is ever asked for the writer hold
        # (``LiveBatchProcessor._ingest_full_paths``), so it never contends
        # with an active writer thread for the GIL regardless of interpreter
        # build (see ``polylogue.sources.live.parse_prefetch`` for the full
        # safety argument, identical in shape to ``DaemonParseStage``). An
        # explicit ``parse_stage`` always wins (tests / callers that want to
        # own the stage's lifecycle themselves); otherwise one is created
        # here, owned by this watcher, and shut down in ``stop()``.
        self._owns_parse_stage = parse_stage is None
        # polylogue-bp12n.6: a stage the watcher owns also writes each parsed
        # file's rows into a shard the writer copies. The directory is
        # disposable scratch beside the tiers it feeds; nothing in it
        # survives ``stop()``.
        self._parse_stage: LiveParseStage | None = (
            parse_stage
            if parse_stage is not None
            else LiveParseStage(shard_directory=Path(polylogue.archive_root) / "parse-shards")
        )
        self._ingest_lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._watcher_ready = asyncio.Event()
        self._archived_cursor_conns: tuple[sqlite3.Connection, sqlite3.Connection] | None = None
        # Set once per reconciliation scope: True when the index tier has no
        # materialized sessions at all despite source.db holding successfully
        # parsed raw material -- the post-index-reset signature (polylogue-emx2).
        # While true, cursor-trust skip decisions must be corroborated
        # per-candidate against index presence instead of being taken on faith.
        self._archived_cursor_index_untrusted = False
        self._batch_processor = LiveBatchProcessor(
            polylogue,
            self._sources,
            cursor=self._cursor,
            parser_fingerprint=lambda: _PARSER_FINGERPRINT,
            converger=converger,
            stop_requested=self._stop.is_set,
            event_emitter=event_emitter,
            sync_runner=self._run_writer_sync,
            parse_stage=self._parse_stage,
        )

    async def _run_writer_sync(
        self,
        actor: str,
        function: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run blocking watcher writes without joining the loop executor at exit."""
        if self._write_coordinator is None:
            return await asyncio.to_thread(function, *args, **kwargs)
        return await self._write_coordinator.run_sync(actor, function, *args, **kwargs)

    @property
    def watcher_ready(self) -> asyncio.Event:
        """Set when this watcher has registered its source roots."""
        return self._watcher_ready

    def intake_revision(self, source: WatchSource) -> int:
        """Disposable invalidation of one source's file-discovery position."""
        return self._intake_revisions[source.root]

    def _existing_source_roots(self) -> list[Path]:
        """Return configured roots that exist at the instant of a scan."""
        return [source.root for source in self._sources if source.exists()]

    async def run(self) -> None:
        # Hook commands create their first carrier lazily.  Ensure the nested
        # root exists before ``awatch`` snapshots its roots, otherwise a daemon
        # that starts before the first hook event never sees that file.
        for source in self._hook_sources():
            # Untagged single-root callers predate the topology contract and
            # are necessarily the primary.  Tagged legacy roots remain
            # strictly read-only and are never created by the watcher.
            if source.role in {None, "primary-writable"}:
                source.root.mkdir(parents=True, exist_ok=True)
        roots = self._existing_source_roots()
        if not roots:
            logger.warning("live.watcher: no source roots exist; nothing to watch")
            self._watcher_ready.set()
            return

        watch_task = asyncio.create_task(self._watch_changes(roots))
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

    async def _watch_changes(self, roots: list[Path]) -> None:
        from watchfiles import Change, awatch

        async for changes in awatch(
            *roots,
            watch_filter=self._watch_filter,
            stop_event=self._stop,
            recursive=True,
        ):
            for change, raw_path in changes:
                if change is Change.deleted:
                    continue
                self._note_intake_hint(Path(raw_path))

    def stop(self) -> None:
        self._stop.set()
        if self._parse_stage is not None and self._owns_parse_stage:
            self._parse_stage.shutdown()

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

    def select_ingest_candidates(self, paths: Sequence[Path]) -> tuple[Path, ...]:
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
            self._cursor.rebase_authoritative_observations(rebases)
        return tuple(needed)

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
        the file (trusting a cursor claim) but the index tier globally shows
        no corroborating material (:attr:`_archived_cursor_index_untrusted`,
        set once per catch-up scan), the skip is demoted to "needed" unless a
        per-file existence check on ``path`` specifically finds its raw
        material already materialized -- see polylogue-emx2.
        """
        needs_work = self._needs_work_from_state_uncorroborated(
            path, stat=stat, cursor=cursor, rebase_queue=rebase_queue
        )
        if needs_work or not self._archived_cursor_index_untrusted:
            return needs_work
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
        if cursor is None:
            if not self._reconcile_archived_cursor(path, stat=stat):
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
        if cursor.failure_count == 0 and cursor.content_fingerprint is None and cursor.next_retry_at is not None:
            if not _retry_due(cursor.next_retry_at):
                return False
            reconciliation = self._reconcile_archived_cursor_outcome(path, stat=stat)
            if reconciliation is _ArchivedCursorReconciliation.RECONCILED:
                reconciled = self._cursor.get_record(path)
                return reconciled is not None and size > reconciled.byte_offset
            if reconciliation is _ArchivedCursorReconciliation.UNAVAILABLE:
                self._cursor.defer_full_cursor_reconciliation(path)
                return False
            self._invalidate_deferred_full_cursor(path, stat=stat)
            return True
        if cursor.failure_count > 0:
            if self._reconcile_archived_cursor(path, stat=stat):
                cursor = self._cursor.get_record(path)
                return cursor is not None and size > cursor.byte_offset
            return _retry_due(cursor.next_retry_at)
        parser_matches = cursor.parser_fingerprint == _PARSER_FINGERPRINT
        if not parser_matches:
            return True
        if self._is_hermes_database(path) or self._is_declared_codex_database(path):
            if cursor.tail_hash == sqlite_source_revision(path):
                return False
            return self._database_content_changed(path, cursor)
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
                self._cursor.mark_failed(path, failed_stat=stat)
                return False
            prefix_hash = cursor_prefix_hash(cursor.tail_hash)
            if prefix_hash is None:
                if self._reconcile_archived_cursor(path, stat=stat):
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
                self._cursor.mark_failed(path, failed_stat=stat)
                return False
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
            with path.open("rb") as handle:
                handle.seek(start_offset)
                payload = handle.read(bytes_to_probe)
        except OSError:
            return True
        if b"\n" in payload:
            return False
        # The bounded probe can prove that no complete record begins at the
        # cursor, even when the unfinished record exceeds the probe budget.
        # Record this observed state so unchanged periodic catch-up scans skip
        # it; a subsequent append changes stat evidence and reopens the probe.
        # polylogue-hat0: this probe found no complete trailing record, not a
        # resolved authority state -- preserve any existing pending-authority
        # marker unchanged rather than clearing it.
        record_deferred_append_cursor(
            self._cursor,
            path,
            cursor=cursor,
            parser_fingerprint=_PARSER_FINGERPRINT,
            source_name=self._source_name_for(path),
            deferred_end_offset=cursor.deferred_end_offset,
        )
        return True

    def _invalidate_deferred_full_cursor(self, path: Path, *, stat: os.stat_result) -> None:
        """Clear a busy-handoff defer when current bytes reject archive authority."""

        updated = self._cursor.set(
            path,
            stat.st_size,
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

    def _reconcile_archived_cursor(self, path: Path, *, stat: os.stat_result) -> bool:
        """Restore a missing/stale cursor from proven archive raw state."""

        return self._reconcile_archived_cursor_outcome(path, stat=stat) is _ArchivedCursorReconciliation.RECONCILED

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
            self._archived_cursor_index_untrusted = False
            yield
            return
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        source_db = archive_root / "source.db"
        try:
            index_db = resolve_active_index_path(archive_root)
        except (ArchiveLocationError, OSError, UnicodeError):
            self._archived_cursor_conns = None
            self._archived_cursor_index_untrusted = False
            yield
            return
        conns: tuple[sqlite3.Connection, sqlite3.Connection] | None = None
        if source_db.exists() and index_db.exists():
            try:
                source_conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=1.0)
                try:
                    index_conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True, timeout=1.0)
                except sqlite3.Error:
                    source_conn.close()
                    raise
                conns = (source_conn, index_conn)
            except sqlite3.Error:
                conns = None
        self._archived_cursor_conns = conns
        self._archived_cursor_index_untrusted = (
            self._index_lacks_all_corroboration(source_conn=conns[0], index_conn=conns[1])
            if conns is not None
            else False
        )
        try:
            yield
        finally:
            self._archived_cursor_conns = None
            self._archived_cursor_index_untrusted = False
            if conns is not None:
                for conn in conns:
                    with suppress(sqlite3.Error):
                        conn.close()

    @staticmethod
    def _index_lacks_all_corroboration(
        *,
        source_conn: sqlite3.Connection,
        index_conn: sqlite3.Connection,
    ) -> bool:
        """True when the index tier holds zero sessions despite acquired raw material.

        A full ``index.db`` reset/rebuild leaves ``ops.db`` ingest cursors
        pointing at file offsets the daemon already acquired and parsed --
        cursor state that catch-up's cursor-trust fast paths would otherwise
        take on faith and skip re-materializing (polylogue-emx2, Finding 8:
        14,879 cursors skipped 100% of files against an empty post-reset
        index). One cheap pair of existence probes per catch-up scan detects
        this and forces every candidate through a per-file corroboration
        check instead.
        """
        has_parsed_raw = source_conn.execute(
            "SELECT 1 FROM raw_sessions WHERE parsed_at_ms IS NOT NULL AND parse_error IS NULL LIMIT 1"
        ).fetchone()
        if has_parsed_raw is None:
            return False
        has_any_session = index_conn.execute("SELECT 1 FROM sessions LIMIT 1").fetchone()
        return has_any_session is None

    @staticmethod
    def _archived_cursor_row(
        path: Path,
        *,
        source_conn: sqlite3.Connection,
        index_conn: sqlite3.Connection,
    ) -> tuple[object, ...] | None:
        """Newest parsed raw row for ``path`` that the index actually contains."""
        rows = source_conn.execute(
            """
            SELECT raw_id, origin, blob_hash, blob_size
            FROM raw_sessions
            WHERE source_path = ?
              AND COALESCE(source_index, 0) >= 0
              AND parsed_at_ms IS NOT NULL
              AND parse_error IS NULL
            ORDER BY acquired_at_ms DESC, raw_id DESC
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
        the new evidence the verdict needs.
        """
        return cast(
            "tuple[object, ...] | None",
            source_conn.execute(
                f"""
                SELECT r.raw_id, r.origin, r.blob_hash, r.blob_size
                FROM raw_sessions AS r
                WHERE r.source_path = ?
                  AND COALESCE(r.source_index, 0) >= 0
                  AND r.parse_error IS NULL
                  AND ({decided_unresolved_membership_sql("r")})
                ORDER BY r.acquired_at_ms DESC, r.raw_id DESC
                LIMIT 1
                """,
                (str(path),),
            ).fetchone(),
        )

    @classmethod
    def _path_corroborated_by_index(
        cls,
        path: Path,
        *,
        source_conn: sqlite3.Connection,
        index_conn: sqlite3.Connection,
    ) -> bool:
        """True unless ``path`` has parsed raw material the index cannot show."""
        has_parsed_raw = source_conn.execute(
            """
            SELECT 1 FROM raw_sessions
            WHERE source_path = ?
              AND COALESCE(source_index, 0) >= 0
              AND parsed_at_ms IS NOT NULL
              AND parse_error IS NULL
            LIMIT 1
            """,
            (str(path),),
        ).fetchone()
        if has_parsed_raw is None:
            # Nothing parsed for this path yet -- not this check's concern.
            return True
        return cls._archived_cursor_row(path, source_conn=source_conn, index_conn=index_conn) is not None

    def _cursor_skip_corroborated_by_index(self, path: Path) -> bool:
        """Whether a cursor-trust skip for ``path`` is backed by a materialized session.

        Only consulted while :attr:`_archived_cursor_index_untrusted` is set
        (index tier globally empty relative to acquired source material). A
        path with no successfully parsed raw row yet is not this bead's
        concern (nothing for the index to have lost) and is treated as
        corroborated so unrelated cursor kinds are unaffected.
        """
        shared = self._archived_cursor_conns
        try:
            if shared is not None:
                return self._path_corroborated_by_index(path, source_conn=shared[0], index_conn=shared[1])
            archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
            source_db = archive_root / "source.db"
            index_db = resolve_active_index_path(archive_root)
            if not source_db.exists() or not index_db.exists():
                return True
            with (
                closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=1.0)) as source_conn,
                closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True, timeout=1.0)) as index_conn,
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
        shared = self._archived_cursor_conns
        archive_root = Path(getattr(self._polylogue, "archive_root", self._cursor._db_path.parent))
        try:
            if shared is not None:
                row = self._archived_cursor_row(
                    path, source_conn=shared[0], index_conn=shared[1]
                ) or self._decided_unresolved_cursor_row(path, source_conn=shared[0])
            else:
                source_db = archive_root / "source.db"
                index_db = resolve_active_index_path(archive_root)
                if not source_db.exists() or not index_db.exists():
                    return _ArchivedCursorReconciliation.UNAVAILABLE
                with (
                    closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=1.0)) as source_conn,
                    closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True, timeout=1.0)) as index_conn,
                ):
                    row = self._archived_cursor_row(
                        path, source_conn=source_conn, index_conn=index_conn
                    ) or self._decided_unresolved_cursor_row(path, source_conn=source_conn)
        except (ArchiveLocationError, OSError, UnicodeError, sqlite3.Error):
            return _ArchivedCursorReconciliation.UNAVAILABLE
        if row is None:
            return _ArchivedCursorReconciliation.INCOMPATIBLE
        _raw_id, origin, blob_hash, blob_size = row
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
        self._cursor.set(
            path,
            archived_size,
            byte_offset=last_complete_newline,
            last_complete_newline=last_complete_newline,
            parser_fingerprint=_PARSER_FINGERPRINT,
            content_fingerprint=content_fingerprint,
            tail_hash=tail_hash,
            source_name=provider_from_origin(Origin.from_string(str(origin))).value
            if origin is not None
            else self._source_name_for(path),
            st_dev=stat.st_dev,
            st_ino=stat.st_ino,
            mtime_ns=stat.st_mtime_ns,
        )
        self._cursor.reset_failures(path)
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

    async def _publish_source_halts(self) -> None:
        """Record every newly halted source as a durable event, once each.

        Without this the halt exists only as one rate-limited log line, which
        scrolls away while the source stays stopped for the rest of the run.
        """
        if self._event_emitter is None:
            return
        emitter = self._event_emitter
        for source_name, reason in sorted(halted_sources().items()):
            if self._published_source_halts.get(source_name) == reason.code:
                continue
            self._published_source_halts[source_name] = reason.code
            payload: dict[str, object] = {
                "source_name": source_name,
                "code": reason.code,
                "message": reason.message,
                "derived_only": reason.derived_only,
                "detail": dict(reason.detail) if reason.detail is not None else None,
            }
            await self._run_writer_sync(
                "watcher.source_halt.event",
                emitter,
                "source_ingest_halted",
                payload,
            )

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

    def _source_accepts(self, path: Path) -> bool:
        source = deepest_source_for_path(path, self._sources)
        return source.accepts(path) if source is not None else False

    def _is_hermes_database(self, path: Path) -> bool:
        resolved = path.resolve()
        for source in self._sources:
            if source.name != "hermes":
                continue
            try:
                if resolved.is_relative_to(source.root.resolve()) and source.accepts(path):
                    return is_sqlite_path(path)
            except OSError:
                continue
        return False

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

    def _database_content_changed(self, path: Path, cursor: CursorRecord) -> bool:
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
            return sqlite_member_revision(path) != recorded
        except (sqlite3.Error, OSError, UnicodeDecodeError):
            # Acquisition owns the consistent read and reports its own typed
            # failure; a locked or damaged database is not silently fresh.
            return True

    def _canonical_watch_path(self, path: Path) -> Path | None:
        if self._source_accepts(path):
            return path
        database = sqlite_database_for_sidecar(path)
        if database is not None and self._is_hermes_database(database):
            return database
        return None

    def _source_for_directory(self, path: Path) -> WatchSource | None:
        """Return the watched source owning a non-ignored directory."""

        source = deepest_source_for_path(path, self._sources)
        if source is None:
            return None
        try:
            relative = path.resolve().relative_to(source.root.resolve())
        except (OSError, ValueError):
            return None
        return None if any(source.ignores_directory(Path(part)) for part in relative.parts) else source

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


def _legacy_data_home_inbox_sources() -> tuple[WatchSource, ...]:
    """Return the XDG data-home inbox when the archive root has moved away.

    ``archive_root()`` defaults to ``data_home()``, so an archive whose root
    was later pointed elsewhere leaves its inbox behind under no watch root at
    all: exports staged there before the move are acquired by nothing, and a
    wipe-and-reconverge never reads them. Same finite legacy-root topology the
    hook spools already carry. Inert where the two inboxes coincide.
    """
    from polylogue.paths import archive_root, data_home

    legacy_root = data_home() / "inbox"
    if legacy_root.resolve() == (archive_root() / "inbox").resolve():
        return ()
    # Named apart from the archive inbox: two watch sources may not share a
    # name, or every by-name lookup silently sees only the last one.
    return (WatchSource(name="inbox-legacy", root=legacy_root, suffixes=INBOX_SOURCE_SUFFIXES),)


def default_sources(*, hermes_root: Path | None = None) -> tuple[WatchSource, ...]:
    """Discover the default live-source roots from XDG/home conventions.

    Includes the archive inbox so that ``polylogue ingest PATH``
    (which stages to ``archive_root()/inbox``) is observed by the
    daemon-owned watcher.

    """
    from polylogue.core.enums import Provider
    from polylogue.paths import (
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
    from polylogue.sources.origin_specs import artifact_suffixes_for_provider

    return (
        WatchSource(
            name="claude-code",
            root=claude_code_path(),
            suffixes=artifact_suffixes_for_provider(Provider.CLAUDE_CODE, defaults=(".jsonl",)),
        ),
        # polylogue-t0p: Claude Code's live plan-snapshot directory
        # (~/.claude/todos/) is a sibling of claude_code_path(), not nested
        # under it -- a second, narrower WatchSource rooted there, same
        # precedent as "codex-state" below, so the main claude-code root
        # doesn't have to widen its own suffix/path assumptions to reach a
        # completely different directory tree.
        WatchSource(
            name="claude-code-todos",
            root=claude_code_todos_path(),
            suffixes=(".json",),
        ),
        # polylogue-ximhz: ``~/.claude/history.jsonl`` is the prompt-submission
        # log whose rows carry the paste evidence no transcript records, and it
        # sits beside the sessions root rather than under it. Rooted at the
        # install directory with no suffixes at all, so only the declared
        # ``prompt_history_log`` path rule admits a file; the two large
        # sibling trees have their own sources and are not descended twice.
        WatchSource(
            name="claude-code-history",
            root=claude_code_path().parent,
            suffixes=(),
            ignored_dir_names=_DEFAULT_IGNORED_DIR_NAMES | frozenset({"projects", "todos"}),
        ),
        WatchSource(name="codex", root=codex_path()),
        # polylogue-0jf4: Codex also keeps live SQLite state (thread titles,
        # spawn topology, goals, memories) as siblings of the sessions/
        # directory, not under it -- a second, narrower WatchSource rooted at
        # ~/.codex (codex_path().parent) rather than widening the "codex"
        # source's own root, so a broadened suffix set never has to reason
        # about history.jsonl/config.toml/log/ under the shared root. Suffix
        # filtering alone (".sqlite"/".db") keeps this cheap; the acquisition
        # path (sources/live/batch.py) re-verifies table shape by name and
        # structure before treating anything as in-scope evidence.
        WatchSource(
            name="codex-state",
            root=codex_path().parent,
            suffixes=(".sqlite", ".db"),
            allow_path_scoped_artifacts=False,
        ),
        # polylogue-rovf5: Codex keeps harness-authored memory documents in
        # ~/.codex/memories/, a sibling of sessions/. Rooted there rather
        # than widening "codex-state" so no Codex root admits ``.md``
        # globally: the source carries no suffixes at all and only the
        # declared ``agent_memory_document`` path rule admits a file.
        # Overlap with the shallower "codex-state" root is resolved by
        # ``deepest_source_for_path``, which prefers this one.
        WatchSource(
            name="codex-memories",
            root=codex_memories_path(),
            suffixes=(),
        ),
        WatchSource(name="gemini-cli", root=gemini_cli_path(), suffixes=(".json", ".jsonl")),
        # Hermes emits four independently durable source classes under its
        # runtime root: state.db, optional session snapshots, NeMo Relay ATIF
        # documents, and append-only ATOF JSONL.  The ledger database is
        # admitted as a live SQLite source too; parsing it remains a separate
        # fidelity/normalization contract rather than an implicit filename
        # fallback.
        WatchSource(
            name="hermes",
            root=hermes_root if hermes_root is not None else hermes_sessions_path(),
            suffixes=(".json", ".jsonl", ".db", ".sqlite", ".sqlite3"),
        ),
        # Antigravity conversations are opaque protobufs. The ordinary live
        # batch route hands those files to the vendor language-server adapter;
        # brain documents and metadata remain source artifacts.
        WatchSource(
            name="antigravity",
            root=antigravity_path(),
            suffixes=artifact_suffixes_for_provider(Provider.ANTIGRAVITY),
        ),
        WatchSource(name="browser-capture", root=browser_capture_spool_root(), suffixes=(".json",)),
        # #1683: inbox accepts archive, zip, and json-line formats so that
        # GDPR exports (typically .zip) and raw .json dumps are observed.
        WatchSource(name="inbox", root=archive_root() / "inbox", suffixes=INBOX_SOURCE_SUFFIXES),
        *_legacy_data_home_inbox_sources(),
        *hook_carrier_watch_sources(hook_spool_sources()),
    )


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


__all__ = ["LiveWatcher", "WatchSource", "default_sources"]
