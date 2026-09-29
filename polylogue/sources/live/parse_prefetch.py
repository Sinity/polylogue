"""Off-writer preparation for the live watcher's full-ingest route.

Path workers leave sealed SQLite carriers and row shards on disk. Admission
caps concurrent task count and captured source bytes; a lone oversized file
may run, while additional files remain retryable. The publisher checks the
captured blob hash before consuming a carrier.

Read-ahead (``prefetch_paths``) is speculative work with a bounded lifecycle.
The parent only stats a path; every source content read, including candidate
sampling, runs inside the executor. Speculation is charged to the slot and
byte budget from submission until it finishes or is reaped, and never takes
the last worker. It is preemptible: a warm whose required path does not fit
gives running unclaimed read-ahead a short grace, then reaps it (a process
pool is terminated and restarted; the warm resubmits its own collateral
work). Unclaimed read-ahead that outlives its lifetime is reaped the same way.
A finished read-ahead releases its charge at once but pays its full artifact
digest only when a warm claims it.

A worker process lost under a preparation is charged to the one file whose
task it held. Each task marks its parent-assigned attempt directory when a
worker takes it and when the worker lets go, so the tasks held at a pool break
are exactly those started and not finished. One held task is charged; several
are all deferred uncharged, and each is then prepared alone until a worker
comes back from it, so every later loss names a single file. Required work
has no wall-clock deadline, but a process worker whose attempt directory has
not grown for its hang bound (a floor far beyond any healthy parse, scaled by
source size) is stopped by the same pool restart that reaps read-ahead, and
charged. Losses on one unchanged observation of a file escalate to a terminal
failure the writer records, instead of a deferral retried for the whole build.

Publication order is enforced once, in reconciliation: a warm reconciles its
paths in intake order against a snapshot taken after every earlier
publication, admits at most one path per canonical session, and returns the
later same-session paths as held for the next warm. A reconciliation is valid
only for the warm that made it; any older one is discarded and redone.
"""

from __future__ import annotations

import os
import shutil
import stat
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from concurrent.futures.process import BrokenProcessPool
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from polylogue.core.enums import Provider
from polylogue.logging import WARNING, carry_context, emit, get_logger
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import parse_payload, parse_stream_payload
from polylogue.sources.live.retained_prefetch import PreparedLiveRetainedRaw, prepare_live_retained_raws
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.prepared_jsonl import PreparedJsonl as LivePathPreparation
from polylogue.sources.prepared_jsonl import VerificationCancelledError, prepare_jsonl_blob
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_shard
from polylogue.storage.sqlite.archive_tiers.write_shard import discard_session_shard

logger = get_logger(__name__)

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionWrite


class PreparedReadSnapshot(Protocol):
    @property
    def archive(self) -> ArchiveStore: ...


ReadSnapshot = Callable[[Path], AbstractContextManager[PreparedReadSnapshot]]

_DEFAULT_WORKER_COUNT_FLOOR = 1
_DEFAULT_PROCESS_WORKER_CAP = 8
#: Stage calls (warms and prefetches) a prefetched result may wait to be
#: claimed before it is dropped. Prefetch looks ahead about two pages and each
#: page costs one prefetch and one warm, so a claimed guess is warmed within
#: four or six; anything older is a path selection skipped. Counting
#: prefetches too keeps a walk that only prefetches (every file skipped after
#: cursor reconciliation) from accumulating results.
_SPECULATIVE_LIFETIME_CALLS = 8
_DEFAULT_STALL_REPORT_SECONDS = 60.0
#: How long a warm blocked on capacity lets running unclaimed read-ahead
#: finish before reaping it.
_PREEMPT_GRACE_SECONDS = 10.0
#: How often a waiting warm re-reads its preparations' progress.
_PROGRESS_POLL_SECONDS = 5.0
#: Consecutive worker losses on one unchanged observation of a file before
#: its preparation is a terminal failure rather than another deferral. A
#: preparation stopped at its hang bound counts as a loss.
_MAX_WORKER_LOSSES_PER_OBSERVATION = 3
#: A process worker whose attempt directory has not grown for this long,
#: plus the source size at the floor throughput below, is presumed hung. Both
#: sit far outside any healthy parse, so slow work that advances is never
#: stopped; only work that has stopped advancing is.
_PREPARATION_HANG_BASE_SECONDS = 600.0
_PREPARATION_HANG_FLOOR_BYTES_PER_SECOND = 256 * 1024

#: A source file's (size, mtime_ns, inode) when its preparation was submitted.
SourceObservation = tuple[int, int, int]

# The dispatcher's per-pass byte budget already caps one admitted page at
# 64 MiB, so the adaptive budget below only needs to cover one page's worth
# of small JSONL files, not a whole-archive whale pass like the census
# prefetch cache. Floor/ceiling still scale with the machine rather than a
# fixed constant, mirroring ``daemon_parse_stage_max_inflight_bytes``.
_MIN_MAX_INFLIGHT_BYTES = 64 * 1024 * 1024  # 64 MiB
_MAX_MAX_INFLIGHT_BYTES = 512 * 1024 * 1024  # 512 MiB


def live_watcher_parse_stage_worker_count() -> int:
    """Bounded worker cap for the watcher's off-writer-hold pre-parse pool.

    Mirrors ``daemon_parse_stage_worker_count``'s cpu-1 convention. Override
    with ``POLYLOGUE_LIVE_WATCHER_PARSE_STAGE_WORKERS``.
    """
    from polylogue.config import load_polylogue_config

    configured = load_polylogue_config().live_watcher_parse_stage_workers
    if configured is not None and configured > 0:
        return configured
    from polylogue.runtime import available_cpus

    return max(_DEFAULT_WORKER_COUNT_FLOOR, (available_cpus() or 2) - 1)


def _parse_stage_workers_configured() -> bool:
    from polylogue.config import load_polylogue_config

    configured = load_polylogue_config().live_watcher_parse_stage_workers
    return configured is not None and configured > 0


def live_watcher_parse_stage_max_inflight_bytes() -> int:
    """Whale-memory budget for the watcher's pre-parse payload cache.

    Adaptive: 1/32 of physical RAM clamped to [64 MiB, 512 MiB]. Override with
    ``POLYLOGUE_LIVE_WATCHER_PARSE_STAGE_MAX_INFLIGHT_BYTES``.
    """
    from polylogue.config import load_polylogue_config

    configured = load_polylogue_config().live_watcher_parse_stage_max_inflight_bytes
    if configured is not None and configured > 0:
        return configured
    from polylogue.pipeline.parsed_tree_size import effective_physical_memory_bytes

    physical = effective_physical_memory_bytes()
    if physical is None:
        return _MIN_MAX_INFLIGHT_BYTES
    return max(_MIN_MAX_INFLIGHT_BYTES, min(_MAX_MAX_INFLIGHT_BYTES, physical // 32))


def live_watcher_parse_stage_stall_report_seconds() -> float:
    """Seconds without forward progress before a preparation is reported stalled.

    Not a deadline: a warm waits for its preparations to finish, however long
    a large file takes, and this window only decides when a worker that has
    stopped advancing is reported (``live.parse_prefetch.preparation_stalled``).

    Override with ``POLYLOGUE_LIVE_WATCHER_PARSE_STAGE_STALL_REPORT_SECONDS``.
    """
    from polylogue.config import load_polylogue_config

    configured = load_polylogue_config().live_watcher_parse_stage_stall_report_seconds
    if configured is not None and configured > 0:
        return configured
    return _DEFAULT_STALL_REPORT_SECONDS


def _completed_reporting_stalls(futures: Iterable[Future[Any]], *, stall_window: float) -> Iterator[Future[Any]]:
    """Yield futures as they complete, never giving up on the rest.

    A window with no completion is reported as a stall and waiting continues:
    abandoning finished-in-a-moment work and reparsing it under the writer
    lease is the livelock this replaces.
    """
    pending = set(futures)
    while pending:
        done, pending = wait(pending, timeout=stall_window, return_when=FIRST_COMPLETED)
        if not done:
            emit(
                "live.parse_prefetch.preparation_stalled",
                level=WARNING,
                outcome="degraded",
                reason="no_completion_in_window",
                paths=len(pending),
                wait_ms=round(stall_window * 1000),
            )
        yield from done


@dataclass(frozen=True, slots=True)
class LiveParseCandidate:
    """One file eligible for off-writer-hold pre-parse."""

    cache_key: str
    provider: Provider
    payload: bytes
    source_path: str
    fallback_id: str
    is_stream: bool


def live_parse_worker(
    cache_key: str,
    provider_value: str,
    payload: bytes,
    source_path: str,
    fallback_id: str,
    *,
    is_stream: bool,
) -> tuple[str, list[ParsedSession] | None, BaseException | None]:
    """Parse one candidate's payload -- a pure function of its bytes, no archive access.

    Identical call shape to ``_ingest_full_records_archive``'s in-hold parse
    branch (``polylogue.sources.live.batch``): the same ``_iter_json_stream``
    decode, the same ``parse_stream_payload``/``parse_payload`` dispatch. Any
    exception is returned (not raised) so a worker failure never surfaces
    anywhere the writer-held pass would not have -- it simply leaves this
    file uncached, and the writer-held pass reparses (and correctly records)
    it exactly as it would with no prewarm at all.
    """
    try:
        provider = Provider.from_string(provider_value)
        source_name = source_path.rsplit("/", 1)[-1]
        if is_stream:
            sessions = parse_stream_payload(
                provider,
                _iter_json_stream(
                    BytesIO(payload),
                    source_name,
                    fail_on_decode_error=provider is Provider.UNKNOWN,
                ),
                fallback_id,
                source_path=source_path,
            )
        else:
            payloads = list(
                _iter_json_stream(
                    BytesIO(payload),
                    source_name,
                    fail_on_decode_error=provider is Provider.UNKNOWN,
                )
            )
            sessions = parse_payload(provider, payloads, fallback_id, source_path=source_path)
        return cache_key, sessions, None
    except Exception as exc:
        return cache_key, None, exc


@dataclass(frozen=True, slots=True)
class LiveParsedEntry:
    """What one prewarm pass produced for one file.

    ``shard_path`` is a sealed shard holding the same sessions' message and
    block rows (polylogue-bp12n.6), or ``None`` when the stage was not
    building shards. The writer attaches it and copies; the sessions are
    still required either way, for everything a shard does not carry.
    """

    sessions: list[ParsedSession]
    shard_path: Path | None


#: Written by a worker as its first act on a task, so a parent whose pool
#: just broke can tell a task a worker held from one still queued.
_WORKER_STARTED_MARKER = ".worker-started"
#: Written when the task returns or raises. A worker that wrote it was alive
#: at the end of its task, so it cannot have died holding it.
_WORKER_FINISHED_MARKER = ".worker-finished"


def _run_holding_attempt(
    worker: Callable[..., LivePathPreparation],
    /,
    *args: Any,
    attempt_directory: str,
    **kwargs: Any,
) -> LivePathPreparation:
    """Run one path task inside the executor, marking when it is held."""
    attempt = Path(attempt_directory)
    (attempt / _WORKER_STARTED_MARKER).touch()
    try:
        return worker(*args, attempt_directory=attempt_directory, **kwargs)
    finally:
        (attempt / _WORKER_FINISHED_MARKER).touch()


def _worker_held_task(attempt_directory: Path | None) -> bool:
    if attempt_directory is None:
        return False
    return (attempt_directory / _WORKER_STARTED_MARKER).exists() and not (
        attempt_directory / _WORKER_FINISHED_MARKER
    ).exists()


def _observe(source_path: str) -> SourceObservation | None:
    try:
        observed = Path(source_path).stat()
    except OSError:
        return None
    return (observed.st_size, observed.st_mtime_ns, observed.st_ino)


@dataclass(frozen=True, slots=True)
class LiveEnrichmentEvidence:
    """Picklable archive coordinates for worker-side retained enrichment."""

    source_db_path: str
    index_db_path: str
    blob_root: str


def live_parse_path_worker(
    provider_value: str,
    source_path: str,
    fallback_id: str,
    *,
    is_stream: bool,
    shard_directory: str,
    attempt_directory: str | None = None,
    evidence: LiveEnrichmentEvidence | None = None,
) -> LivePathPreparation:
    """Seal one live source with the same interpretation retained replay uses.

    ``evidence`` names the archive's source/index databases and blob root.
    With it, every admitted session is enriched from retained archive evidence
    exactly as retained replay would enrich the same bytes. ``None`` is only
    for callers with no archive (the stage then publishes parsed content).
    """
    from polylogue.sources.dispatch import is_jsonl_source_path
    from polylogue.sources.live.batch_support import _detect_provider_from_path_sample, jsonl_complete_prefix_path
    from polylogue.sources.live.sidecar_resolution import FilesystemSidecarResolver

    source = Path(source_path)
    provider = _detect_provider_from_path_sample(source, Provider.from_string(provider_value))
    boundary = jsonl_complete_prefix_path(source) if is_jsonl_source_path(source_path) else None
    source_size = source.stat().st_size
    parse_prefix_size = (
        boundary.prefix_size
        if boundary is not None and 0 < boundary.prefix_size < source_size and not boundary.malformed_record
        else None
    )
    # Apply evidence filtering before sealing so publication can use the indexed sequence.
    if evidence is None:
        return prepare_jsonl_blob(
            source_path,
            source_path,
            provider.value,
            fallback_id,
            is_stream=is_stream,
            shard_directory=shard_directory,
            attempt_directory=None if attempt_directory is None else Path(attempt_directory),
            parse_prefix_size=parse_prefix_size,
            prepare_session=lambda session: session,
            sidecar_resolver=FilesystemSidecarResolver(),
        )
    from polylogue.sources.revision_backfill import open_retained_session_enricher

    with open_retained_session_enricher(
        provider,
        source_path=source_path,
        source_db_path=evidence.source_db_path,
        index_db_path=evidence.index_db_path,
        blob_root=evidence.blob_root,
    ) as enrich:
        # The evidence read here predates the writer's admission of this
        # pass. The sealed digest lets the writer detect evidence that moved
        # in between (a sidecar admitted in the same pass) and re-enrich.
        return prepare_jsonl_blob(
            source_path,
            source_path,
            provider.value,
            fallback_id,
            is_stream=is_stream,
            shard_directory=shard_directory,
            attempt_directory=None if attempt_directory is None else Path(attempt_directory),
            parse_prefix_size=parse_prefix_size,
            prepare_session=enrich,
            # The live parse joins tool-output sidecars from the source tree
            # (as ``parse_payload`` does by default); a sealed carrier without
            # them would keep masked excerpts the live route replaces.
            sidecar_resolver=FilesystemSidecarResolver(),
            preparation_dependency=lambda: (
                enrich.dependency_digest(),
                str(Path(evidence.index_db_path).resolve()),
            ),
        )


def _publication_index_path(archive_root: Path) -> Path:
    from polylogue.sources.live.cold_build import active_cold_build_generation
    from polylogue.storage.archive_identity import resolve_active_index_path

    generation = active_cold_build_generation(archive_root)
    if generation is not None:
        return Path(generation.generation.index_path)
    return resolve_active_index_path(archive_root)


def _enrichment_evidence(archive_root: Path | None) -> LiveEnrichmentEvidence | None:
    """Worker coordinates for the index the writer will publish into.

    A cold build's candidate, otherwise the active generation. The writer
    rejects a carrier enriched elsewhere.
    """
    if archive_root is None:
        return None
    return LiveEnrichmentEvidence(
        source_db_path=str(archive_root / "source.db"),
        index_db_path=str(_publication_index_path(archive_root)),
        blob_root=str(archive_root / "blob"),
    )


def live_lookahead_path_worker(
    fallback_provider_value: str,
    source_path: str,
    fallback_id: str,
    *,
    shard_directory: str,
    attempt_directory: str | None = None,
    evidence: LiveEnrichmentEvidence | None = None,
) -> LivePathPreparation:
    """Select and prepare one read-ahead path inside the executor.

    Candidate sampling reads the source, so it runs here, where a stuck read
    can be reaped with its worker, rather than on a parent thread nothing can
    stop. A path that is not a regular file (discovery saw one, but it may
    since have become a symlink out of the source) is refused before any
    read. A refused path, or one that is not a preparation candidate, yields
    a retryable result, which a claiming warm discards and prepares itself.
    """
    from polylogue.sources.live.batch import _live_parse_stage_path_candidates

    try:
        regular = stat.S_ISREG(os.lstat(source_path).st_mode)
    except OSError:
        regular = False
    if not regular:
        return LivePathPreparation(None, None, None, "read-ahead path is not a regular file", deferred=True)

    selected = _live_parse_stage_path_candidates(
        [Path(source_path)], fallback_provider=Provider.from_string(fallback_provider_value)
    )
    if not selected:
        return LivePathPreparation(None, None, None, "read-ahead path is not a preparation candidate", deferred=True)
    _selected_path, provider, is_stream = selected[0]
    return live_parse_path_worker(
        provider.value,
        source_path,
        fallback_id,
        is_stream=is_stream,
        shard_directory=shard_directory,
        attempt_directory=attempt_directory,
        evidence=evidence,
    )


def live_parse_and_shard_worker(
    cache_key: str,
    provider_value: str,
    payload: bytes,
    source_path: str,
    fallback_id: str,
    *,
    is_stream: bool,
    shard_directory: str | None,
) -> tuple[str, list[ParsedSession] | None, BaseException | None, str | None]:
    """``live_parse_worker`` plus the shard its sessions' rows go into.

    Building the shard here is the point of polylogue-bp12n.6: row
    construction AND the per-row parameter binding both leave the writer
    thread, and what the writer gets is a file it copies with one statement
    per table. A shard build that fails leaves the parse result intact and
    the shard absent -- the writer then binds rows itself, exactly as it does
    for any other prefetch miss. ``shard_directory=None`` is that same
    outcome by configuration rather than by failure.
    """
    key, sessions, error = live_parse_worker(
        cache_key, provider_value, payload, source_path, fallback_id, is_stream=is_stream
    )
    if shard_directory is None or error is not None or not sessions:
        return key, sessions, error, None
    try:
        shard = prepare_session_shard(Path(shard_directory), sessions)
    except Exception:
        logger.warning("live watcher parse-stage prefetch: shard build failed for %s", source_path, exc_info=True)
        return key, sessions, None, None
    return key, sessions, None, str(shard.path)


class LiveParsePrefetchCache:
    """Thread-safe path-keyed cache of pre-parsed sessions, budgeted by payload bytes.

    Keyed on the watcher's own path string (see module docstring for why,
    unlike ``RawParsePrefetchCache``, this cannot be keyed on a content-hash
    raw_id ahead of time). Because the key is a path rather than a content
    hash, a file that changed on disk between the prewarm read and the
    writer-held read (e.g. a still-appending live file) must never be
    confused for the version that was actually parsed -- ``pop`` therefore
    requires the caller to pass the payload bytes it is ABOUT to write, and
    only returns the cached sessions when they byte-for-byte match what was
    parsed. A mismatch is treated exactly like any other cache miss: the
    caller falls back to parsing the current bytes inline.
    """

    def __init__(self, *, max_inflight_bytes: int) -> None:
        self._max_inflight_bytes = max_inflight_bytes
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[list[ParsedSession], bytes, Path | None]] = {}
        self._inflight_bytes = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def contains(self, cache_key: str) -> bool:
        with self._lock:
            return cache_key in self._entries

    def try_admit(
        self,
        cache_key: str,
        sessions: list[ParsedSession],
        *,
        payload: bytes,
        shard_path: Path | None = None,
    ) -> bool:
        """Admit ``sessions`` unless already present or the budget is exceeded.

        A single entry is always admitted even alone over budget (mirrors
        ``RawParsePrefetchCache``'s ``try_admit``) -- the budget bounds how
        much accumulates across MANY entries, not the size of any one file;
        rejecting a lone oversized entry would just mean parsing it inline
        anyway, an unhelpful distinction for a one-file cache.

        ``shard_path`` is the sealed shard the worker built for these same
        sessions (polylogue-bp12n.6). A rejected admission deletes it: the
        rows it carries are about to be rebuilt inline, and an orphan
        scratch file that nothing will ever attach is pure residue.
        """
        payload_bytes = len(payload)
        with self._lock:
            already_present = cache_key in self._entries
            over_budget = bool(self._entries) and self._inflight_bytes + payload_bytes > self._max_inflight_bytes
            admitted = not (already_present or over_budget)
            if admitted:
                self._entries[cache_key] = (sessions, payload, shard_path)
                self._inflight_bytes += payload_bytes
        if not admitted and shard_path is not None:
            discard_session_shard(shard_path)
        return admitted

    def pop(self, cache_key: str, *, payload: bytes) -> LiveParsedEntry | None:
        """Consume a cached entry, but only if it was parsed from ``payload`` exactly.

        Always removes the entry when present (win or mismatch) -- a stale
        entry for a path that has since changed on disk is never useful to a
        later lookup either, since the writer-held pass advances that path's
        cursor past the version prewarm saw. A mismatch also deletes the
        shard, whose rows describe bytes this archive is no longer writing.
        """
        with self._lock:
            entry = self._entries.pop(cache_key, None)
            if entry is None:
                return None
            sessions, cached_payload, shard_path = entry
            self._inflight_bytes -= len(cached_payload)
        if cached_payload != payload:
            logger.warning(
                "live watcher parse-stage prefetch: cached parse for %s no longer matches "
                "on-disk bytes (file changed between prewarm and writer hold); reparsing inline",
                cache_key,
            )
            if shard_path is not None:
                discard_session_shard(shard_path)
            return None
        return LiveParsedEntry(sessions=sessions, shard_path=shard_path)

    def discard_all(self) -> None:
        """Drop every entry and delete the shards they hold."""
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
            self._inflight_bytes = 0
        for _sessions, _payload, shard_path in entries:
            if shard_path is not None:
                discard_session_shard(shard_path)


class LiveParseStage:
    """Owns the watcher's bounded off-writer-hold parse executor + cache.

    One instance lives for the ``LiveWatcher``'s lifetime, created on
    construction. ``warm`` is synchronous/blocking -- callers run it off the
    event loop (``asyncio.to_thread``) and NEVER under the write
    coordinator's hold, for the same reason ``DaemonParseStage.warm`` never
    does.
    """

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        max_inflight_bytes: int | None = None,
        stall_report_seconds: float | None = None,
        shard_directory: Path | None = None,
        use_processes: bool = False,
        preempt_grace_seconds: float = _PREEMPT_GRACE_SECONDS,
        hang_base_seconds: float = _PREPARATION_HANG_BASE_SECONDS,
        hang_floor_bytes_per_second: float = _PREPARATION_HANG_FLOOR_BYTES_PER_SECOND,
    ) -> None:
        # polylogue-bp12n.6. Where a worker's sealed shard goes, or ``None``
        # to keep row binding on the writer thread. Path workers receive a
        # parent-created attempt child so their files remain attributable
        # until publication or discard.
        self._shard_directory = shard_directory
        #: Workers that parsed successfully but handed back no shard
        #: (polylogue-3r36h). A systematic shard-build failure otherwise
        #: removes the writer-side benefit with nothing but a per-file
        #: warning to show for it.
        self.shard_build_failure_count = 0
        self._path_results: dict[str, LivePathPreparation] = {}
        self._retained_by_path: dict[str, dict[str, PreparedLiveRetainedRaw]] = {}
        self._path_futures: dict[str, Future[LivePathPreparation]] = {}
        #: Paths submitted by ``prefetch_paths`` that no warm has claimed yet,
        #: with the warm count at submission. A prefetch is a guess about what
        #: a later batch will ingest; selection may skip the path (an
        #: unchanged file whose cursor is restored from the archive), so an
        #: unclaimed guess is dropped after ``_SPECULATIVE_LIFETIME_CALLS``
        #: stage calls rather than held, with its scratch, until shutdown.
        self._speculative: dict[str, int] = {}
        #: Running unclaimed read-ahead a blocked warm has preempted, with the
        #: monotonic time after which it is reaped.
        self._preempt_at: dict[str, float] = {}
        self._preempt_grace_seconds = preempt_grace_seconds
        #: Finished unclaimed read-ahead whose full artifact digest is still
        #: owed; the claiming warm pays it.
        self._unverified: set[str] = set()
        #: Reaped read-ahead on a thread executor, which cannot stop a
        #: thread: still running, counted against read-ahead slots until it
        #: ends, and discarded unused.
        self._orphans: dict[Future[LivePathPreparation], Path | None] = {}
        self._stage_calls = 0
        #: Held by a warm or a prefetch for its whole run. ``shutdown`` sets
        #: the active warm's cancellation and takes this lock before it stops
        #: the pool, so no stage call mutates bookkeeping shutdown is clearing.
        self._stage_lock = threading.Lock()
        #: Guards ``_closing`` against ``_active_cancel``: a warm publishes its
        #: event only while the stage is open, and shutdown closes the stage
        #: and reads the event in one step, so it can never miss an entering
        #: warm.
        self._publish_lock = threading.Lock()
        self._active_cancel: threading.Event | None = None
        #: Each running preparation's source observation at submission; its
        #: size is the preparation's charge against the byte budget.
        self._path_observations: dict[str, SourceObservation] = {}
        #: Each running preparation's attempt-directory size and the
        #: monotonic time it last grew: the hang signal.
        self._path_progress: dict[str, tuple[int, float]] = {}
        self._hang_base_seconds = hang_base_seconds
        self._hang_floor_bytes_per_second = max(1.0, hang_floor_bytes_per_second)
        #: source path -> (observation, consecutive worker losses on it).
        #: Rewritten content starts a new streak; a worker that comes back
        #: from the file, with any result, ends it.
        self._path_worker_losses: dict[str, tuple[SourceObservation, int]] = {}
        #: Paths a worker held when the pool broke. Each is prepared alone
        #: until a worker comes back from it, so its next loss is attributable.
        self._path_solo_suspects: set[str] = set()
        self._path_attempt_dirs: dict[str, Path] = {}
        self._path_inflight_bytes = 0
        self._closing = False
        self.cleanup_failure_count = 0
        self._cleanup_blocked = False
        if shard_directory is not None:
            shard_directory.mkdir(parents=True, exist_ok=True)
            self._attempt_root: Path | None = shard_directory / ".live-parse-attempts"
            self._attempt_root.mkdir(parents=True, exist_ok=True)
            # This private namespace contains only parent-assigned attempt
            # directories. Never sweep filenames in the shared shard root.
            for residue in tuple(self._attempt_root.iterdir()):
                if residue.is_dir() and residue.name.startswith("attempt-"):
                    self._remove_attempt_directory(residue)
        else:
            self._attempt_root = None
        worker_count = max_workers if max_workers is not None else live_watcher_parse_stage_worker_count()
        if use_processes and max_workers is None and not _parse_stage_workers_configured():
            # Each worker process is its own interpreter: about 120-150 MiB
            # resident once the parsers are imported. The default pool is
            # sized to what keeps the single writer fed, not to every core --
            # an unconfigured 24-core host otherwise spent ~3 GiB on idle
            # parser processes.
            worker_count = min(worker_count, _DEFAULT_PROCESS_WORKER_CAP)
        self._worker_count = worker_count
        # Every worker may hold a path. Memory is bounded by the in-flight
        # source-byte budget below (a whale still runs alone once it fills
        # it); a fixed two-path cap left the rest of the pool idle and put
        # parsing on the fresh build's critical path.
        self._max_path_pending = max(1, worker_count)
        if use_processes:
            # The ordinary watcher route runs on the supported GIL build too.
            # A process pool is the only way for its CPU-bound parser to make
            # genuine progress in parallel there; free-threaded callers and
            # test-owned stages retain the lighter thread executor.
            from polylogue.pipeline.services.process_pool import process_pool_executor

            self._executor: Executor = process_pool_executor(max_workers=worker_count)
        else:
            self._executor = ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="polylogue-live-parse-stage",
            )
        self.cache = LiveParsePrefetchCache(
            max_inflight_bytes=(
                max_inflight_bytes if max_inflight_bytes is not None else live_watcher_parse_stage_max_inflight_bytes()
            )
        )
        self._max_path_bytes = max_inflight_bytes or live_watcher_parse_stage_max_inflight_bytes()
        self._stall_report_seconds = (
            stall_report_seconds
            if stall_report_seconds is not None
            else live_watcher_parse_stage_stall_report_seconds()
        )

    def warm_paths(
        self,
        candidates: Sequence[tuple[str, Provider, bool]],
        *,
        archive_root: Path | None = None,
        read_snapshot: ReadSnapshot | None = None,
        capture_mode: Provider | None = None,
        source_index: int = 0,
        cancelled: threading.Event | None = None,
    ) -> frozenset[str]:
        """Prepare path-backed JSON/JSONL outside the writer lease.

        Every selected path gets a result, including worker death. Setting
        ``cancelled`` ends the wait at the next progress poll: running
        preparations stay owned by the stage for a later warm to collect, no
        result is recorded for unsubmitted paths, and no prepared write is
        installed from this warm's snapshot.
        The publisher can therefore retain raw bytes and retry without an
        accidental inline parse when preparation failed.

        Returns the paths held back for publication order: each shares a
        canonical session with an earlier candidate, so it must be warmed
        again after that candidate publishes.
        """
        if self._shard_directory is None or self._closing:
            return frozenset()
        with self._stage_lock:
            active = cancelled if cancelled is not None else threading.Event()
            with self._publish_lock:
                if self._closing:
                    return frozenset()
                self._active_cancel = active
            try:
                return self._warm_paths_locked(
                    candidates,
                    archive_root=archive_root,
                    read_snapshot=read_snapshot,
                    capture_mode=capture_mode,
                    source_index=source_index,
                    cancelled=active,
                )
            finally:
                with self._publish_lock:
                    self._active_cancel = None

    def _warm_paths_locked(
        self,
        candidates: Sequence[tuple[str, Provider, bool]],
        *,
        archive_root: Path | None,
        read_snapshot: ReadSnapshot | None,
        capture_mode: Provider | None,
        source_index: int,
        cancelled: threading.Event,
    ) -> frozenset[str]:
        self._stage_calls += 1
        self._collect_finished()
        # Read-ahead this warm now claims is required work from here on: it
        # is no longer preemptible, and it pays the full digest it deferred.
        # A retryable failure it produced (the source moved while it was
        # read ahead) is not this warm's answer: it is prepared again,
        # whether it finished before the warm or while the warm waited.
        claimed = {source_path for source_path, _p, _s in candidates if source_path in self._speculative}
        for source_path in claimed:
            self._speculative.pop(source_path, None)
            self._preempt_at.pop(source_path, None)
            if source_path in self._unverified:
                self._unverified.discard(source_path)
                self._verify_claimed(source_path, cancelled=cancelled)
        self._discard_retryable(claimed)
        if self._cleanup_blocked:
            for source_path, _provider, _is_stream in candidates:
                if source_path not in self._path_results and source_path not in self._path_futures:
                    self._path_results[source_path] = LivePathPreparation(
                        None, None, None, "worker stop could not be verified; scratch cleanup is blocked", deferred=True
                    )
            return frozenset()
        evidence = _enrichment_evidence(archive_root)
        self._warm_until(list(candidates), cancelled=cancelled, evidence=evidence)
        if cancelled.is_set():
            self._drop_stale_speculation()
            return frozenset()
        retry = self._discard_retryable(claimed)
        if retry:
            self._warm_until(
                [candidate for candidate in candidates if candidate[0] in retry], cancelled=cancelled, evidence=evidence
            )
        if cancelled.is_set():
            self._drop_stale_speculation()
            return frozenset()
        held: frozenset[str] = frozenset()
        if archive_root is not None:
            held = self._prepare_existing_session_writes(
                archive_root,
                read_snapshot=read_snapshot,
                capture_mode=capture_mode,
                source_index=source_index,
                paths=[source_path for source_path, _provider, _is_stream in candidates],
                cancelled=cancelled,
            )
        self._drop_stale_speculation()
        return held

    def _cancel_predicate(self) -> Callable[[], bool] | None:
        active = self._active_cancel
        return None if active is None else active.is_set

    def _verify_claimed(self, source_path: str, *, cancelled: threading.Event | None = None) -> None:
        result = self._path_results.get(source_path)
        if result is None or result.error is not None:
            return
        try:
            result.verify_files(full=True, stop=None if cancelled is None else cancelled.is_set)
        except VerificationCancelledError:
            # A cancelled warm records nothing; the next claim prepares again.
            self._path_results.pop(source_path).discard()
        except (OSError, ValueError) as exc:
            result.discard()
            self._path_results[source_path] = LivePathPreparation(
                None, None, None, f"worker artifact changed: {type(exc).__name__}"[:500], deferred=True
            )

    def _discard_retryable(self, paths: set[str]) -> set[str]:
        """Drop retryable failures recorded for ``paths``; return which."""
        dropped: set[str] = set()
        for source_path in paths:
            result = self._path_results.get(source_path)
            if result is not None and result.error is not None and result.deferred:
                self._path_results.pop(source_path).discard()
                dropped.add(source_path)
        return dropped

    def _warm_until(
        self,
        candidates: list[tuple[str, Provider, bool]],
        *,
        cancelled: threading.Event | None = None,
        evidence: LiveEnrichmentEvidence | None = None,
    ) -> None:
        """Submit ``candidates`` as capacity allows and wait until each is prepared.

        There is no deadline on required work. A large file's preparation is
        real progress toward the only ingest that file will get; giving up on
        it at a fixed wall time deferred the file, re-acquired it on the next
        pass and never finished it. What a clock may decide is only whether to
        *report* a worker that has stopped advancing: preparation writes its
        sealed carrier as it goes, so the attempt directory's size is the
        forward progress signal. Worker death still ends a wait
        (``BrokenProcessPool`` at collection), shutdown terminates stragglers,
        and a set ``cancelled`` ends the wait at the next poll with nothing
        recorded for what remains. The same signal, per path, bounds a hang:
        a process worker that has not advanced for its hang bound is stopped
        and charged (``_reap_hung_preparations``).

        Capacity held by unclaimed read-ahead is the one thing a warm does
        not wait out: when a required path does not fit, that read-ahead is
        preempted and, after a short grace, reaped.
        """
        wanted = {source_path for source_path, _provider, _is_stream in candidates}
        progress = -1
        last_progress = time.monotonic()
        reported_at = last_progress
        while not self._cleanup_blocked:
            if cancelled is not None and cancelled.is_set():
                return
            # Resubmit everything without a result or a future: a reap may
            # have cancelled this warm's own work as collateral.
            remaining = self._submit_path_candidates(list(candidates), evidence=evidence)
            selected = [future for path, future in self._path_futures.items() if path in wanted]
            if not remaining and not selected:
                break
            if remaining:
                self._preempt_speculation()
            # Capacity may be held by other work; wait on it too, since its
            # completion is what frees a slot for what remains.
            waiting = tuple(self._path_futures.values()) if remaining else tuple(selected)
            if not waiting:
                break
            done, _pending = wait(waiting, timeout=self._next_wait_seconds(), return_when=FIRST_COMPLETED)
            self._collect_finished()
            self._reap_due_preemptions()
            self._reap_hung_preparations()
            now = time.monotonic()
            advanced = self._attempt_bytes(wanted)
            if done or advanced > progress:
                progress = advanced
                last_progress = reported_at = now
            elif now - reported_at >= self._stall_report_seconds:
                reported_at = now
                emit(
                    "live.parse_prefetch.preparation_stalled",
                    level=WARNING,
                    outcome="degraded",
                    reason="no_forward_progress",
                    paths=len([path for path in wanted if path in self._path_futures]),
                    wait_ms=round((now - last_progress) * 1000),
                    attempt_bytes=advanced,
                )
        for source_path, _provider, _is_stream in candidates:
            if source_path in self._path_results or source_path in self._path_futures:
                continue
            self._path_results[source_path] = LivePathPreparation(
                None, None, None, "worker preparation capacity is unavailable", deferred=True
            )
        self._collect_finished()

    def _collect_finished(self) -> None:
        """Collect every finished preparation and settle finished orphans.

        Collection releases a future's slot and bytes. Unclaimed read-ahead
        is collected without its full digest, so this is cheap for paths the
        caller did not ask for.
        """
        for source_path, future in tuple(self._path_futures.items()):
            if future.done():
                self._collect_path_future(source_path, future)
        for future, attempt_directory in tuple(self._orphans.items()):
            if not future.done():
                continue
            del self._orphans[future]
            with suppress(Exception):
                future.result().discard()
            if attempt_directory is not None:
                self._remove_attempt_directory(attempt_directory)

    def _running_speculation(self) -> list[str]:
        return [
            source_path
            for source_path in self._speculative
            if (future := self._path_futures.get(source_path)) is not None and not future.done()
        ]

    def _preempt_speculation(self) -> None:
        """Give running unclaimed read-ahead a grace period before reaping it."""
        fresh = [source_path for source_path in self._running_speculation() if source_path not in self._preempt_at]
        if not fresh:
            return
        deadline = time.monotonic() + self._preempt_grace_seconds
        for source_path in fresh:
            self._preempt_at[source_path] = deadline
        emit(
            "live.parse_prefetch.speculation_preempted",
            outcome="degraded",
            reason="required_preparation_blocked",
            paths=len(fresh),
            budget_ms=round(self._preempt_grace_seconds * 1000),
        )

    def _next_wait_seconds(self) -> float:
        running = set(self._running_speculation())
        deadlines = [deadline for path, deadline in self._preempt_at.items() if path in running]
        if not deadlines:
            return _PROGRESS_POLL_SECONDS
        return max(0.0, min(_PROGRESS_POLL_SECONDS, min(deadlines) - time.monotonic()))

    def _reap_due_preemptions(self) -> None:
        now = time.monotonic()
        running = set(self._running_speculation())
        due = [path for path, deadline in self._preempt_at.items() if path in running and now >= deadline]
        if due:
            self._reap_speculation(due, reason="preempted_by_required_preparation")

    def _reap_speculation(self, paths: Sequence[str], *, reason: str) -> None:
        """Stop running unclaimed read-ahead and release its charge.

        A process pool is terminated and restarted: that is the only way to
        stop a worker stuck in a source read. Other unclaimed read-ahead it
        cancels is dropped; a warm's own collateral work loses only its
        restart deferral and is resubmitted by that warm. A thread cannot be
        stopped, so on a thread executor the read-ahead is orphaned: it keeps
        a read-ahead slot until it ends and its result is discarded.
        """
        emit(
            "live.parse_prefetch.speculation_reaped",
            level=WARNING,
            outcome="degraded",
            reason=reason,
            paths=len(paths),
        )
        reaped = frozenset(paths)
        if isinstance(self._executor, ProcessPoolExecutor):
            self._stop_process_workers(reason="worker pool restarted to reap read-ahead", reaped=reaped)
            return
        for source_path in reaped:
            future = self._path_futures.get(source_path)
            if future is None:
                continue
            self._orphans[future] = self._release_slot(source_path)[0]
            self._speculative.pop(source_path, None)
            self._preempt_at.pop(source_path, None)

    def _stop_process_workers(self, *, reason: str, reaped: frozenset[str] = frozenset()) -> None:
        """Restart the process pool on purpose and settle its collateral.

        ``reaped`` read-ahead is dropped without a digest. Other unclaimed
        read-ahead that did not finish is dropped too. Required work the
        restart interrupted loses its restart deferral, so a waiting warm
        resubmits it. A success that sealed just before the stop is kept.
        """
        affected = tuple(self._path_futures)
        self._restart_broken_process_pool(reason=reason, discard=reaped)
        if self._cleanup_blocked:
            return
        for source_path in affected:
            self._preempt_at.pop(source_path, None)
            result = self._path_results.get(source_path)
            if source_path in reaped or (
                source_path in self._speculative and (result is None or result.error is not None)
            ):
                self._speculative.pop(source_path, None)
                self._unverified.discard(source_path)
                if result is not None:
                    self._path_results.pop(source_path).discard()
            elif result is not None and result.error is not None and result.deferred:
                self._path_results.pop(source_path).discard()

    def _release_slot(self, source_path: str) -> tuple[Path | None, SourceObservation | None]:
        """Forget a running preparation's bookkeeping and release its charge.

        Returns its attempt directory and its submission observation.
        """
        self._path_futures.pop(source_path, None)
        self._path_progress.pop(source_path, None)
        observation = self._path_observations.pop(source_path, None)
        if observation is not None:
            self._path_inflight_bytes -= observation[0]
        return self._path_attempt_dirs.pop(source_path, None), observation

    def _attempt_bytes(self, paths: set[str]) -> int:
        """Bytes the running preparations of ``paths`` have written so far."""
        total = 0
        for source_path in paths:
            directory = self._path_attempt_dirs.get(source_path)
            if directory is None:
                continue
            try:
                for entry in os.scandir(directory):
                    with suppress(OSError):
                        total += entry.stat(follow_symlinks=False).st_size
            except OSError:
                continue
        return total

    def _drop_stale_speculation(self) -> None:
        """Discard read-ahead no warm claimed within its lifetime, reaping it if running."""
        expired = [
            source_path
            for source_path, submitted_at in self._speculative.items()
            if self._stage_calls - submitted_at >= _SPECULATIVE_LIFETIME_CALLS
        ]
        if not expired:
            return
        running_now = set(self._running_speculation())
        running = [path for path in expired if path in running_now]
        if running:
            self._reap_speculation(running, reason="read_ahead_expired_unclaimed")
        for source_path in expired:
            future = self._path_futures.get(source_path)
            if future is not None and future.done():
                # Finished but unclaimed: dropped without the full digest.
                self._collect_path_future(source_path, future)
            self._speculative.pop(source_path, None)
            self._unverified.discard(source_path)
            self._preempt_at.pop(source_path, None)
            result = self._path_results.pop(source_path, None)
            if result is not None:
                result.discard()

    def prefetch_paths(
        self, paths: Sequence[str], *, fallback_provider: Provider, archive_root: Path | None = None
    ) -> int:
        """Start preparing paths a later ``warm_paths`` will ask for, without waiting.

        The writer publishes one source group (and one page) while the next
        is still unparsed; submitting the upcoming paths now lets their
        parsing overlap that publication instead of following it. Only a
        stat happens here: candidate selection reads the source, so it runs
        in the worker (``live_lookahead_path_worker``). Read-ahead needs an
        executor slot beyond the one kept for required work and fits the
        same byte budget, which charges it from submission; a path that does
        not fit is left for the warm that needs it. Results are claimed and
        verified by the ordinary ``warm_paths`` and ``pop_path`` route.
        ``archive_root`` enriches read-ahead from the same retained evidence
        a required warm would use, so a claimed carrier is not re-prepared.
        Returns the number of paths newly submitted.
        """
        if self._shard_directory is None or self._cleanup_blocked or self._closing:
            return 0
        with self._stage_lock:
            if self._closing:
                return 0
            return self._prefetch_paths_locked(paths, fallback_provider=fallback_provider, archive_root=archive_root)

    def _prefetch_paths_locked(
        self, paths: Sequence[str], *, fallback_provider: Provider, archive_root: Path | None
    ) -> int:
        self._stage_calls += 1
        self._collect_finished()
        submitted: list[str] = []
        self._submit_path_candidates(
            [(source_path, fallback_provider, False) for source_path in paths],
            speculative=True,
            submitted=submitted,
            evidence=_enrichment_evidence(archive_root),
        )
        for source_path in submitted:
            self._speculative[source_path] = self._stage_calls
        self._drop_stale_speculation()
        return len(submitted)

    def _submit_path_candidates(
        self,
        candidates: list[tuple[str, Provider, bool]],
        *,
        speculative: bool = False,
        submitted: list[str] | None = None,
        evidence: LiveEnrichmentEvidence | None = None,
    ) -> list[tuple[str, Provider, bool]]:
        """Submit every candidate that fits the worker and byte budget; return the rest.

        Every running preparation, speculative or required, is charged. A
        speculative (prefetch) submission records no result when the path
        cannot be stat'ed or submitted: the warm that needs the path retries
        it then, instead of inheriting a transient failure.
        """
        next_wave: list[tuple[str, Provider, bool]] = []
        suspects = self._path_solo_suspects
        if suspects and not speculative:
            # A suspect is offered before anything else, so a suspect that
            # cannot run yet is known to be waiting before others are admitted.
            candidates = sorted(candidates, key=lambda candidate: candidate[0] not in suspects)
        suspect_waiting: bool | None = None
        for source_path, provider, is_stream in candidates:
            if source_path in self._path_results or source_path in self._path_futures:
                continue
            if suspects:
                if suspect_waiting is None and source_path not in suspects:
                    suspect_waiting = not speculative and any(
                        path in suspects and path not in self._path_results and path not in self._path_futures
                        for path, _provider, _is_stream in candidates
                    )
                if self._solo_preparation_blocks(
                    source_path, speculative=speculative, suspect_waiting=bool(suspect_waiting)
                ):
                    next_wave.append((source_path, provider, is_stream))
                    continue
            try:
                observed = Path(source_path).stat()
            except OSError as exc:
                if not speculative:
                    self._path_results[source_path] = LivePathPreparation(
                        None, None, None, f"source stat failed: {type(exc).__name__}"[:500], deferred=True
                    )
                continue
            source_bytes = observed.st_size
            pending_count = len(self._path_futures)
            # Read-ahead never takes the last worker: required work always has
            # an executor slot, even beside an orphaned thread. With a single
            # worker there is no spare slot, so no read-ahead.
            limit = self._max_path_pending - 1 if speculative else self._max_path_pending
            occupied = pending_count + len(self._orphans) if speculative else pending_count
            if occupied >= limit or (pending_count and self._path_inflight_bytes + source_bytes > self._max_path_bytes):
                next_wave.append((source_path, provider, is_stream))
                continue
            attempt_directory: Path | None = None
            try:
                attempt_directory = self._new_attempt_directory()
                if speculative:
                    future = self._executor.submit(
                        _run_holding_attempt,
                        live_lookahead_path_worker,
                        provider.value,
                        source_path,
                        Path(source_path).stem,
                        shard_directory=str(self._attempt_root),
                        attempt_directory=str(attempt_directory),
                        **({} if evidence is None else {"evidence": evidence}),
                    )
                else:
                    future = self._executor.submit(
                        carry_context(_run_holding_attempt),
                        live_parse_path_worker,
                        provider.value,
                        source_path,
                        Path(source_path).stem,
                        is_stream=is_stream,
                        shard_directory=str(self._attempt_root),
                        attempt_directory=str(attempt_directory),
                        **({} if evidence is None else {"evidence": evidence}),
                    )
            except Exception as exc:
                if attempt_directory is not None:
                    self._remove_attempt_directory(attempt_directory)
                if not speculative:
                    self._path_results[source_path] = LivePathPreparation(
                        None, None, None, f"worker submission failed: {type(exc).__name__}"[:500], deferred=True
                    )
                continue
            self._path_futures[source_path] = future
            self._preempt_at.pop(source_path, None)
            if submitted is not None:
                submitted.append(source_path)
            self._path_attempt_dirs[source_path] = attempt_directory
            self._path_observations[source_path] = (observed.st_size, observed.st_mtime_ns, observed.st_ino)
            self._path_progress[source_path] = (0, time.monotonic())
            self._path_inflight_bytes += source_bytes
        return next_wave

    def _prepare_existing_session_writes(
        self,
        archive_root: Path,
        *,
        read_snapshot: ReadSnapshot | None,
        capture_mode: Provider | None,
        source_index: int,
        paths: Sequence[str],
        cancelled: threading.Event | None = None,
    ) -> frozenset[str]:
        """Reconcile prior acquisitions on a read-only index before admission.

        This is the one place publication order is enforced. ``paths`` is
        this warm's intake order and the snapshot is taken after every
        earlier publication, so a reconciliation is valid only for the
        publication that follows this warm: an older one (a cancelled warm's,
        or a held path's) is closed and redone. Within the warm, the first
        path of each canonical session is installed and every later path of
        that session is held, unreconciled, and returned: its prepared write
        would pin predecessor state the earlier path is about to replace.

        A reconciliation finished after ``cancelled`` is set is closed rather
        than installed.
        """
        from polylogue.core.identity_law import session_id as archive_session_id
        from polylogue.core.sources import origin_from_provider
        from polylogue.storage.sqlite.archive_tiers.source_write import deterministic_raw_session_id
        from polylogue.storage.sqlite.archive_tiers.write import (
            prepare_session_write,
            prepared_session_rows_from_shard,
        )

        pending: dict[str, LivePathPreparation] = {}
        for path in dict.fromkeys(paths):
            result = self._path_results.get(path)
            if result is None or result.error is not None:
                continue
            if result.prepared_writes:
                for stale in result.prepared_writes:
                    stale.close()
                result = replace(result, prepared_writes=())
                self._path_results[path] = result
            pending[path] = result
        if not pending:
            return frozenset()
        if read_snapshot is None:
            for path, result in pending.items():
                result.discard()
                self._path_results[path] = LivePathPreparation(
                    None, None, None, "read-only preparation snapshot unavailable: RuntimeError", deferred=True
                )
            return frozenset()

        def prepare_one(
            path: str, result: LivePathPreparation
        ) -> tuple[LivePathPreparation, frozenset[str], dict[str, PreparedLiveRetainedRaw]]:
            # Each task owns its read transaction. Sharing one SQLite
            # connection across threads would also share its snapshot state.
            writes: list[PreparedSessionWrite] = []
            session_ids: set[str] = set()
            retained: dict[str, PreparedLiveRetainedRaw] = {}
            if cancelled is not None and cancelled.is_set():
                return result, frozenset(), {}
            opened_snapshot = False
            try:
                with read_snapshot(archive_root) as pinned:
                    opened_snapshot = True
                    archive = pinned.archive
                    index_conn = archive.index_connection
                    if index_conn is None:
                        raise RuntimeError("prepared live write has no readable index snapshot")
                    source_conn = archive.source_connection
                    assert result.blob_hash is not None
                    acquisition_provider = (
                        capture_mode
                        if capture_mode is not None and capture_mode is not Provider.UNKNOWN
                        else result.resolved_provider
                    )
                    if acquisition_provider is None:
                        raise ValueError("prepared live write has no acquisition provider")
                    expected_raw_id = deterministic_raw_session_id(
                        origin_from_provider(acquisition_provider),
                        str(path),
                        source_index,
                        bytes.fromhex(result.blob_hash),
                    )
                    for session in result.iter_sessions():
                        if cancelled is not None and cancelled.is_set():
                            # Cancelled mid-carrier: nothing from this
                            # snapshot is installed, so stop here.
                            for partial in writes:
                                partial.close()
                            return result, frozenset(), {}
                        session_id = archive_session_id(
                            origin_from_provider(session.source_name).value,
                            session.provider_session_id,
                        )
                        session_ids.add(session_id)
                        row = index_conn.execute(
                            "SELECT raw_id FROM sessions WHERE session_id = ?", (session_id,)
                        ).fetchone()
                        if row is None or row[0] is None or row[0] == expected_raw_id:
                            continue
                        writes.append(
                            prepare_session_write(
                                index_conn,
                                session,
                                merge_append=False,
                                source_conn=source_conn,
                                raw_id=expected_raw_id,
                                prepared_rows=prepared_session_rows_from_shard(result.shard_path, session_id)
                                if result.shard_path is not None
                                else None,
                            )
                        )
                    if result.attempt_directory is not None:
                        retained = prepare_live_retained_raws(
                            archive,
                            logical_keys=session_ids,
                            current_raw_id=expected_raw_id,
                            directory=result.attempt_directory / "retained",
                            worker_executor=self._executor,
                            index_db_path=_publication_index_path(Path(archive.archive_root)),
                            stop=None if cancelled is None else cancelled.is_set,
                        )
                return replace(result, prepared_writes=tuple(writes)), frozenset(session_ids), retained
            except Exception as exc:
                for prepared in writes:
                    prepared.close()
                for member in retained.values():
                    member.discard()
                result.discard()
                return (
                    LivePathPreparation(
                        None,
                        None,
                        None,
                        (
                            f"existing-session preparation failed: {type(exc).__name__}"
                            if opened_snapshot
                            else f"read-only preparation snapshot unavailable: {type(exc).__name__}"
                        )[:500],
                        deferred=True,
                    ),
                    frozenset(),
                    {},
                )

        # Bound reconciliation to the same path admission width as parsing.
        # Results are installed on the caller thread in intake order, so
        # read tasks finishing out of order cannot reorder publication.
        held: set[str] = set()
        claimed_sessions: set[str] = set()
        with ThreadPoolExecutor(max_workers=min(self._max_path_pending, len(pending))) as executor:
            futures = {
                path: executor.submit(carry_context(prepare_one), path, result) for path, result in pending.items()
            }
            broken_pool = False
            for path, future in futures.items():
                prepared, session_ids, retained = future.result()
                broken_pool |= prepared.error is not None and "BrokenProcessPool" in prepared.error
                overlaps = bool(session_ids & claimed_sessions)
                # A held path still claims its sessions: a later path sharing
                # any of them waits behind it, so overlap closes transitively.
                claimed_sessions |= session_ids
                if prepared.error is None and ((cancelled is not None and cancelled.is_set()) or overlaps):
                    for write in prepared.prepared_writes:
                        write.close()
                    for member in retained.values():
                        member.discard()
                    if overlaps:
                        held.add(path)
                    continue
                for member in self._retained_by_path.pop(path, {}).values():
                    member.discard()
                self._path_results[path] = prepared
                if retained:
                    self._retained_by_path[path] = retained
        if broken_pool:
            self._restart_broken_process_pool()
        return frozenset(held)

    def _collect_path_future(self, source_path: str, future: Future[LivePathPreparation]) -> None:
        if self._path_futures.get(source_path) is not future:
            return
        expired = (
            source_path in self._speculative
            and self._stage_calls - self._speculative[source_path] >= _SPECULATIVE_LIFETIME_CALLS
        )
        attempt_directory, observation = self._release_slot(source_path)
        try:
            result = future.result()
        except BrokenProcessPool:
            result = self._attribute_pool_death(source_path, attempt_directory, observation)
        except Exception as exc:
            # The worker came back from this file, whatever it raised.
            self._end_loss_streak(source_path)
            result = LivePathPreparation(None, None, None, f"worker failed: {type(exc).__name__}"[:500], deferred=True)
        else:
            self._end_loss_streak(source_path)
        if attempt_directory is not None:
            if result.error is None:
                try:
                    self._validate_attempt_result(result, attempt_directory)
                    result = replace(result, attempt_directory=attempt_directory)
                except (OSError, ValueError) as exc:
                    # A worker that wrote outside its attempt directory is a
                    # deterministic defect: fail the path visibly instead of
                    # deferring it into an endless retry. Only an OS error
                    # while checking may clear on a later pass.
                    result = LivePathPreparation(
                        None,
                        None,
                        None,
                        f"worker artifact ownership mismatch: {type(exc).__name__}: {exc}"[:500],
                        deferred=isinstance(exc, OSError),
                    )
                    self._remove_attempt_directory(attempt_directory)
            else:
                # A failed pool stop keeps this attempt in parent custody;
                # shutdown may reclaim it later after a verified reap.
                if not self._cleanup_blocked:
                    self._remove_attempt_directory(attempt_directory)
        self._preempt_at.pop(source_path, None)
        if expired:
            # No warm claimed this read-ahead within its lifetime, so nothing
            # will publish it: drop it without paying the full byte scan.
            self._speculative.pop(source_path, None)
            self._unverified.discard(source_path)
            old = self._path_results.pop(source_path, None)
            if old is not None:
                old.discard()
            result.discard()
            return
        if source_path in self._speculative and result.error is None:
            # Unclaimed read-ahead: its charge is released now, and the full
            # byte scan is paid by the warm that claims it, never by a warm
            # collecting it on the way to unrelated work.
            self._unverified.add(source_path)
        elif result.error is None:
            try:
                # A full byte scan belongs at the prefetch boundary, before
                # the caller enters the writer runner. Publication only needs
                # the cheap exact-inode check in pop_path/iter_sessions.
                result.verify_files(full=True, stop=self._cancel_predicate())
            except VerificationCancelledError:
                # The warm was cancelled mid-scan: record nothing.
                result.discard()
                old = self._path_results.pop(source_path, None)
                if old is not None:
                    old.discard()
                return
            except (OSError, ValueError) as exc:
                result.discard()
                result = LivePathPreparation(
                    None,
                    None,
                    None,
                    f"worker artifact changed: {type(exc).__name__}"[:500],
                    deferred=True,
                )
        old = self._path_results.pop(source_path, None)
        if old is not None:
            old.discard()
        self._path_results[source_path] = result

    def _end_loss_streak(self, source_path: str) -> None:
        self._path_worker_losses.pop(source_path, None)
        self._path_solo_suspects.discard(source_path)

    def _solo_preparation_running(self) -> bool:
        return any(source_path in self._path_solo_suspects for source_path in self._path_futures)

    def _solo_preparation_blocks(self, source_path: str, *, speculative: bool, suspect_waiting: bool) -> bool:
        """Whether submitting ``source_path`` now would share the pool with a suspect.

        A suspect runs only on an otherwise idle pool and nothing joins it.
        While a suspect waits for the pool to drain, no other required path is
        admitted, so the drain finishes instead of starving it; the waiting
        warm preempts read-ahead as it does for any blocked path. A suspect is
        never read ahead.
        """
        if self._solo_preparation_running():
            return True
        if source_path in self._path_solo_suspects:
            return speculative or bool(self._path_futures)
        return suspect_waiting

    def _attribute_pool_death(
        self,
        source_path: str,
        attempt_directory: Path | None,
        observation: SourceObservation | None,
    ) -> LivePathPreparation:
        """Charge a pool break to the one preparation a worker held, if unique.

        A broken pool completes every outstanding future with
        ``BrokenProcessPool``, not only the one whose worker died. The tasks
        held at the break are those whose attempt directory is marked started
        and not finished; queued and finished tasks are not the cause. One
        held task is charged. Several are deferred uncharged. Every held task
        becomes a solo suspect.
        """
        if self._closing:
            return LivePathPreparation(None, None, None, "worker pool stopped during shutdown", deferred=True)
        if self._cleanup_blocked:
            # An earlier stop was never verified, so this break may be the
            # same one, seen again through another future: charge nothing.
            return LivePathPreparation(
                None, None, None, "worker stop could not be verified; scratch cleanup is blocked", deferred=True
            )
        held: list[tuple[str, SourceObservation | None]] = []
        if _worker_held_task(attempt_directory):
            held.append((source_path, observation))
        for sibling_path, sibling in self._path_futures.items():
            if sibling.done() and not sibling.cancelled() and sibling.exception() is None:
                continue
            if _worker_held_task(self._path_attempt_dirs.get(sibling_path)):
                held.append((sibling_path, self._path_observations.get(sibling_path)))
        cause = "worker process died during preparation"
        self._path_solo_suspects.update(path for path, _observation in held)
        if len(held) > 1:
            emit(
                "live.parse_prefetch.worker_death_ambiguous",
                level=WARNING,
                outcome="degraded",
                reason="worker_death_ambiguous",
                path=source_path,
                attempts=len(held),
            )
            reason = f"{cause}; {len(held)} preparations were running, retrying each alone to attribute it"
        else:
            reason = "worker pool restarted after another preparation's worker died"
        culprit = held[0] if len(held) == 1 else None
        if culprit is not None and culprit[0] == source_path:
            result = self._charge_worker_loss(source_path, observation, cause)
        elif any(path == source_path for path, _observation in held):
            result = LivePathPreparation(None, None, None, reason, deferred=True)
        else:
            result = LivePathPreparation(
                None, None, None, "worker pool broke while this preparation was not running", deferred=True
            )
        # Every sibling the restart interrupts is deferred uncharged; a
        # sibling that is the one culprit is charged after it.
        self._restart_broken_process_pool(failed_attempt=attempt_directory, reason=reason)
        if culprit is not None and culprit[0] != source_path and not self._cleanup_blocked:
            culprit_path, culprit_observation = culprit
            old = self._path_results.pop(culprit_path, None)
            if old is not None:
                old.discard()
            self._path_results[culprit_path] = self._charge_worker_loss(culprit_path, culprit_observation, cause)
        return result

    def _charge_worker_loss(
        self, source_path: str, observation: SourceObservation | None, cause: str
    ) -> LivePathPreparation:
        """Count one lost worker against this exact observation of the file.

        Below the bound the loss is deferred, like any interrupted
        preparation. At the bound it is not: the writer records a parse
        failure, and the cursor's finite failure budget quarantines the file
        with a visible reason instead of retrying it every pass. A loss with
        no observation cannot be bound to a revision and is only deferred.
        """
        if observation is None:
            return LivePathPreparation(None, None, None, cause, deferred=True)
        previous = self._path_worker_losses.get(source_path)
        losses = previous[1] + 1 if previous is not None and previous[0] == observation else 1
        self._path_worker_losses[source_path] = (observation, losses)
        if losses < _MAX_WORKER_LOSSES_PER_OBSERVATION:
            return LivePathPreparation(None, None, None, cause, deferred=True)
        emit(
            "live.parse_prefetch.worker_death_escalated",
            level=WARNING,
            outcome="refused",
            reason="worker_died_repeatedly",
            path=source_path,
            attempts=losses,
            error_detail=cause,
        )
        return LivePathPreparation(
            None,
            None,
            None,
            f"{cause}; worker lost on {losses} consecutive preparations of this file",
            deferred=False,
            failed_observation=observation,
        )

    def _reap_hung_preparations(self) -> None:
        """Stop required preparations that stopped advancing; charge only them.

        Unclaimed read-ahead has its own lifetime and preemption. A thread
        cannot be stopped, so only a process pool is reaped. The restart
        frees every slot; interrupted siblings are resubmitted uncharged.
        """
        if self._closing or self._cleanup_blocked or not isinstance(self._executor, ProcessPoolExecutor):
            return
        now = time.monotonic()
        hung: list[tuple[str, SourceObservation | None, float]] = []
        for source_path, future in self._path_futures.items():
            if future.done() or source_path in self._speculative:
                continue
            written = self._attempt_bytes({source_path})
            last_written, advanced_at = self._path_progress.get(source_path, (written, now))
            if written > last_written or source_path not in self._path_progress:
                self._path_progress[source_path] = (written, now)
                continue
            observation = self._path_observations.get(source_path)
            size = 0 if observation is None else observation[0]
            idle = now - advanced_at
            if idle > self._hang_base_seconds + size / self._hang_floor_bytes_per_second:
                hung.append((source_path, observation, idle))
        if not hung:
            return
        for source_path, _observation, idle in hung:
            emit(
                "live.parse_prefetch.preparation_hung",
                level=WARNING,
                outcome="degraded",
                reason="no_forward_progress_within_hang_bound",
                path=source_path,
                wait_ms=round(idle * 1000),
            )
        self._stop_process_workers(reason="worker pool restarted to stop a hung preparation")
        if self._cleanup_blocked:
            return
        for source_path, observation, _idle in hung:
            if (result := self._path_results.get(source_path)) is not None and result.error is None:
                # It sealed as it was stopped: a success, kept by the restart.
                self._end_loss_streak(source_path)
                continue
            if result is not None:
                self._path_results.pop(source_path).discard()
            # A hang is attributed exactly; running it alone would only hold
            # the whole pool for its next hang bound.
            self._path_solo_suspects.discard(source_path)
            self._path_results[source_path] = self._charge_worker_loss(
                source_path, observation, "worker preparation made no progress within its hang bound"
            )

    def _new_attempt_directory(self) -> Path:
        if self._attempt_root is None:
            raise RuntimeError("path preparation has no owned scratch root")
        path = self._attempt_root / f"attempt-{uuid.uuid4().hex}"
        path.mkdir()
        return path

    def _validate_attempt_result(self, result: LivePathPreparation, attempt_directory: Path) -> None:
        if result.attempt_directory is not None and result.attempt_directory != attempt_directory:
            raise ValueError("worker returned another attempt directory")
        for path in (result.sessions_path, result.shard_path):
            if path is None or path.parent != attempt_directory:
                raise ValueError("worker artifact escaped its parent-assigned attempt directory")

    def _remove_attempt_directory(self, path: Path) -> bool:
        if self._attempt_root is None or path.parent != self._attempt_root or not path.name.startswith("attempt-"):
            self._cleanup_blocked = True
            self._record_cleanup_failure("refusing to remove a path outside the owned attempt namespace")
            return False
        try:
            shutil.rmtree(path)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            self._cleanup_blocked = True
            self._record_cleanup_failure("attempt scratch removal failed")
            return False

    def _record_cleanup_failure(self, reason: str) -> None:
        self.cleanup_failure_count += 1
        if self.cleanup_failure_count > 1:
            return
        emit(
            "live.parse_prefetch.cleanup_blocked",
            level=WARNING,
            outcome="degraded",
            reason=reason,
            cumulative_count=self.cleanup_failure_count,
        )

    def _restart_broken_process_pool(
        self,
        *,
        failed_attempt: Path | None = None,
        reason: str = "worker process died during preparation",
        discard: frozenset[str] = frozenset(),
    ) -> None:
        if not isinstance(self._executor, ProcessPoolExecutor):
            return
        from polylogue.pipeline.services.process_pool import process_pool_executor, terminate_process_pool

        pending = tuple(self._path_futures.items())
        for _pending_path, future in pending:
            future.cancel()
        stopped = terminate_process_pool(self._executor)
        if not stopped:
            self._cleanup_blocked = True
            self._record_cleanup_failure("worker process stop could not be verified; retaining attempt scratch")
            return
        if failed_attempt is not None:
            self._remove_attempt_directory(failed_attempt)
        for pending_path, future in pending:
            attempt_directory, _observation = self._release_slot(pending_path)
            # A sibling may have sealed successfully just before the pool
            # broke. Its result is still useful and retains its own carrier.
            if future.done() and not future.cancelled():
                try:
                    result = future.result()
                except Exception:
                    result = None
                if pending_path in discard:
                    # Reaped read-ahead: dropped without a digest.
                    if result is not None:
                        result.discard()
                    if attempt_directory is not None:
                        self._remove_attempt_directory(attempt_directory)
                    continue
                if result is not None and result.error is None and attempt_directory is not None:
                    try:
                        self._validate_attempt_result(result, attempt_directory)
                        if pending_path in self._speculative:
                            # Unclaimed read-ahead keeps its digest owed to
                            # the warm that claims it.
                            self._unverified.add(pending_path)
                        else:
                            result.verify_files(full=True, stop=self._cancel_predicate())
                        result = replace(result, attempt_directory=attempt_directory)
                        self._path_results[pending_path] = result
                        # A worker came back from this file: as a collected
                        # success does, that ends its loss streak.
                        self._end_loss_streak(pending_path)
                        continue
                    except (OSError, ValueError, VerificationCancelledError):
                        pass
            self._path_results[pending_path] = LivePathPreparation(None, None, None, reason, deferred=True)
            if attempt_directory is not None:
                self._remove_attempt_directory(attempt_directory)
        if self._cleanup_blocked or self._closing:
            # Shutdown owns the executor once it starts; a pool created now
            # would outlive the stage and could seal carriers after cleanup.
            return
        self._executor = process_pool_executor(max_workers=self._worker_count)

    def pop_path(self, source_path: str, *, blob_hash: str) -> LivePathPreparation | None:
        future = self._path_futures.get(source_path)
        if future is not None or source_path in self._unverified:
            # pop_path runs under writer admission. Even a finished future
            # needs a full artifact digest and prior-session reconciliation,
            # both of which belong to the next off-lease warm_paths pass.
            return LivePathPreparation(None, None, None, "worker preparation pending", deferred=True)
        result = self._path_results.pop(source_path, None)
        if result is None:
            return None
        if result.error is not None:
            # A stable parse error carries the hash of the bytes it failed to
            # parse, so only that error can be attributed to this capture.
            # Retryable worker failures may have no hash and must retain their
            # original reason.
            if result.blob_hash is not None and result.blob_hash != blob_hash:
                self._discard_retained_path(source_path)
                result.discard()
                return LivePathPreparation(None, None, None, "captured source changed after preparation", deferred=True)
            if result.failed_observation is not None and _observe(source_path) != result.failed_observation:
                # A lost worker never hashed the bytes it failed on. The file
                # unchanged since that preparation was submitted is the proof
                # the capture holds the same revision; anything else is new.
                result.discard()
                return LivePathPreparation(
                    None, None, None, "captured source changed after the failed preparation", deferred=True
                )
            return result
        if result.blob_hash != blob_hash:
            self._discard_retained_path(source_path)
            result.discard()
            return LivePathPreparation(None, None, None, "captured source changed after preparation", deferred=True)
        if result.error is None:
            try:
                result.verify_files(full=False)
            except (OSError, ValueError) as exc:
                self._discard_retained_path(source_path)
                result.discard()
                return LivePathPreparation(
                    None,
                    None,
                    None,
                    f"worker artifact identity changed: {type(exc).__name__}"[:500],
                    deferred=True,
                )
        return result

    def take_retained_path(self, source_path: str) -> dict[str, PreparedLiveRetainedRaw]:
        """Transfer sealed retained members alongside one accepted path carrier."""
        return self._retained_by_path.pop(source_path, {})

    def _discard_retained_path(self, source_path: str) -> None:
        for member in self._retained_by_path.pop(source_path, {}).values():
            member.discard()

    def resolved_path_provider(self, source_path: str) -> Provider | None:
        """Return a sealed worker's detection before durable source admission."""
        if source_path in self._path_futures:
            return None
        result = self._path_results.get(source_path)
        if result is None or result.error is not None:
            return None
        return result.resolved_provider

    def path_preparation_pending(self, source_path: str) -> bool:
        return source_path in self._path_futures

    def warm(self, candidates: Sequence[LiveParseCandidate]) -> int:
        """Pre-parse ``candidates`` outside any writer hold.

        Returns the number of files newly admitted to the cache. Read-only
        end to end: candidates are already-read payload bytes, so nothing
        here touches source.db, index.db, or the daemon's writer lease.
        """
        pending = [candidate for candidate in candidates if not self.cache.contains(candidate.cache_key)]
        if not pending:
            return 0
        shard_directory = None if self._shard_directory is None else str(self._shard_directory)
        futures = {
            self._executor.submit(
                carry_context(live_parse_and_shard_worker),
                candidate.cache_key,
                candidate.provider.value,
                candidate.payload,
                candidate.source_path,
                candidate.fallback_id,
                is_stream=candidate.is_stream,
                shard_directory=shard_directory,
            ): candidate
            for candidate in pending
        }
        warmed = 0
        completed = 0
        shard_build_failures = 0
        consumed: set[Future[tuple[str, list[ParsedSession] | None, BaseException | None, str | None]]] = set()
        for future in _completed_reporting_stalls(futures, stall_window=self._stall_report_seconds):
            completed += 1
            consumed.add(future)
            candidate = futures[future]
            try:
                result = future.result()
            except Exception:
                logger.warning(
                    "live watcher parse-stage prefetch: worker failed for %s",
                    candidate.source_path,
                    exc_info=True,
                )
                continue
            cache_key, sessions, error, shard_name = result
            shard_path = None if shard_name is None else Path(shard_name)
            if error is not None or sessions is None:
                # Parse failures are intentionally NOT cached: the
                # writer-held pass reparses (and correctly records) this
                # file exactly as it would with no prewarm at all.
                if shard_path is not None:
                    discard_session_shard(shard_path)
                continue
            if shard_directory is not None and sessions and shard_path is None:
                shard_build_failures += 1
            if self.cache.try_admit(cache_key, sessions, payload=candidate.payload, shard_path=shard_path):
                warmed += 1
        if shard_build_failures:
            self.shard_build_failure_count += shard_build_failures
            emit(
                "live.parse_prefetch.shard_build_failed",
                level=WARNING,
                outcome="degraded",
                reason="parsed files produced no shard; the writer binds those rows itself",
                count=shard_build_failures,
                total=len(futures),
                cumulative_count=self.shard_build_failure_count,
            )
        return warmed

    def shutdown(self) -> None:
        # A process worker may outlive a warm window indefinitely. Stop and
        # join it before removing scratch, so daemon stop stays bounded and
        # no worker can seal a carrier after cleanup.
        # A warm may still be running on another thread (daemon stop runs
        # before intake is cancelled). Cancel it and let it settle before
        # anything it owns is torn down.
        with self._publish_lock:
            self._closing = True
            active = self._active_cancel
        if active is not None:
            active.set()
        with self._stage_lock:
            self._shutdown_locked()

    def _shutdown_locked(self) -> None:
        stopped = True
        if isinstance(self._executor, ProcessPoolExecutor):
            from polylogue.pipeline.services.process_pool import terminate_process_pool

            stopped = terminate_process_pool(self._executor)
        else:
            self._executor.shutdown(wait=True, cancel_futures=True)
        if not stopped:
            self._cleanup_blocked = True
            self._record_cleanup_failure("shutdown could not verify worker stop; retaining attempt scratch")
            return
        for source_path, future in tuple(self._path_futures.items()):
            self._collect_path_future(source_path, future)
        self._collect_finished()
        self.cache.discard_all()
        for result in self._path_results.values():
            result.discard()
        self._path_results.clear()
        for source_path in tuple(self._retained_by_path):
            self._discard_retained_path(source_path)
        if self._attempt_root is not None:
            try:
                residues = tuple(self._attempt_root.iterdir())
            except FileNotFoundError:
                residues = ()
            for residue in residues:
                if residue.is_dir() and residue.name.startswith("attempt-"):
                    self._remove_attempt_directory(residue)
            # Preserve a non-empty namespace, including unrelated files.
            with suppress(OSError):
                self._attempt_root.rmdir()


__all__ = [
    "LiveParseCandidate",
    "LiveParsePrefetchCache",
    "LiveParseStage",
    "LiveParsedEntry",
    "live_lookahead_path_worker",
    "live_parse_and_shard_worker",
    "live_parse_worker",
    "live_watcher_parse_stage_max_inflight_bytes",
    "live_watcher_parse_stage_stall_report_seconds",
    "live_watcher_parse_stage_worker_count",
]
