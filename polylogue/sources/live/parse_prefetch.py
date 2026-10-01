"""Off-writer preparation for the live watcher's full-ingest route.

Path workers leave sealed SQLite carriers and row shards on disk. Admission
caps concurrent task count and captured source bytes; a lone oversized file
may run, while additional files remain retryable. The publisher checks the
captured blob hash before consuming a carrier.

Read-ahead (``prefetch_paths``) is speculative work with a bounded lifecycle.
Source content reads and preparation use the shared compute admission. When
required work needs its local window, unclaimed speculation receives a
cooperative cancellation request. Its physical future and scratch remain
owned until the original worker finishes and its result is discarded. No
worker is killed, restarted or charged a terminal source failure because of
elapsed time. Required work reports stalls while retaining its reservation.

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
from builtins import BaseExceptionGroup
from collections.abc import Callable, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    wait,
)
from contextlib import AbstractContextManager, closing, suppress
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from polylogue.core.compute import (
    DaemonBackpressureError,
    DaemonOperationCancelled,
    SubmittedOperation,
    compute_adapter,
    current_cancellation,
)
from polylogue.core.compute_cancel import check_compute_cancelled, compute_cancel
from polylogue.core.enums import Provider
from polylogue.core.prepared_file import VerificationCancelledError
from polylogue.core.sql_settlement import retain_native_sql_lifetimes
from polylogue.logging import WARNING, emit, get_logger
from polylogue.sources.live.retained_prefetch import PreparedLiveRetainedRaw, prepare_live_retained_raws
from polylogue.sources.prepared_jsonl import PreparedJsonl as LivePathPreparation
from polylogue.sources.prepared_jsonl import prepare_jsonl_blob, source_snapshot

logger = get_logger(__name__)

if TYPE_CHECKING:
    from polylogue.sources.live.batch_support import PreAcquisitionDecision
    from polylogue.sources.sqlite_snapshot import SQLiteBlobSnapshot
    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionWrite


class PreparedReadSnapshot(Protocol):
    @property
    def archive(self) -> ArchiveStore: ...


ReadSnapshot = Callable[[Path], AbstractContextManager[PreparedReadSnapshot]]

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
#: A source file's (size, mtime_ns, inode) when its preparation was submitted.
SourceObservation = tuple[int, int, int]

# The dispatcher's per-pass byte budget already caps one admitted page at
# 64 MiB, so the adaptive budget below only needs to cover one page's worth
# of source files. Floor/ceiling scale with the machine rather than a fixed
# constant.
_MIN_MAX_INFLIGHT_BYTES = 64 * 1024 * 1024  # 64 MiB
_MAX_MAX_INFLIGHT_BYTES = 512 * 1024 * 1024  # 512 MiB


def live_watcher_parse_stage_worker_count() -> int:
    """Bounded worker cap for the watcher's off-writer-hold pre-parse pool.

    Uses the cpu-1 convention. Override with
    ``POLYLOGUE_LIVE_WATCHER_PARSE_STAGE_WORKERS``.
    """
    from polylogue.config import load_polylogue_config

    configured = load_polylogue_config().live_watcher_parse_stage_workers
    if configured is not None and configured > 0:
        return configured
    return compute_adapter().snapshot().by_class("incremental-background").ceiling_units


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


def _observe(source_path: str) -> SourceObservation | None:
    try:
        observed = Path(source_path).stat()
    except OSError:
        return None
    return (observed.st_size, observed.st_mtime_ns, observed.st_ino)


@dataclass(frozen=True, slots=True)
class LiveEnrichmentEvidence:
    """Immutable archive coordinates for worker-side retained enrichment."""

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

    Provider sampling, the JSONL frontier and the parse all read one private
    snapshot of the source, and the carrier is sealed with that snapshot's
    digest, so the provider it names is the one those exact bytes select.
    """
    with source_snapshot(
        Path(source_path), Path(attempt_directory) if attempt_directory is not None else Path(shard_directory)
    ) as (snapshot, snapshot_sha256, profile):
        return _prepare_path_snapshot(
            provider_value,
            source_path,
            snapshot,
            snapshot_sha256,
            fallback_id,
            profile_identity=profile.key,
            is_stream=is_stream,
            shard_directory=shard_directory,
            attempt_directory=attempt_directory,
            evidence=evidence,
        )


def _prepare_path_snapshot(
    provider_value: str,
    source_path: str,
    snapshot: Path,
    snapshot_sha256: str,
    fallback_id: str,
    *,
    profile_identity: str,
    is_stream: bool,
    shard_directory: str,
    attempt_directory: str | None,
    evidence: LiveEnrichmentEvidence | None,
) -> LivePathPreparation:
    from polylogue.sources.dispatch import is_jsonl_source_path
    from polylogue.sources.live.batch_support import (
        _detect_provider_from_path,
        jsonl_complete_prefix_path,
        jsonl_parse_prefix_size,
    )
    from polylogue.sources.live.sidecar_resolution import FilesystemSidecarResolver

    provider = _detect_provider_from_path(snapshot, Provider.from_string(provider_value))
    boundary = jsonl_complete_prefix_path(snapshot) if is_jsonl_source_path(source_path) else None
    snapshot_size = snapshot.stat().st_size
    parse_prefix_size = jsonl_parse_prefix_size(boundary, snapshot_size) if boundary is not None else None
    # Apply evidence filtering before sealing so publication can use the indexed sequence.
    if evidence is None:
        return prepare_jsonl_blob(
            str(snapshot),
            source_path,
            provider.value,
            fallback_id,
            is_stream=is_stream,
            profile_identity=profile_identity,
            shard_directory=shard_directory,
            attempt_directory=None if attempt_directory is None else Path(attempt_directory),
            parse_prefix_size=parse_prefix_size,
            prepare_session=lambda session: session,
            sidecar_resolver=FilesystemSidecarResolver(),
            source_sha256=snapshot_sha256,
            strict_jsonl_records=True,
        )
    from polylogue.sources.revision_backfill import open_retained_session_enricher
    from polylogue.storage.blob_publication import ArchiveBlobPublisher

    with open_retained_session_enricher(
        provider,
        source_path=source_path,
        captured_zip_coordinate=None,
        source_db_path=evidence.source_db_path,
        index_db_path=evidence.index_db_path,
        blob_root=evidence.blob_root,
    ) as enrich:
        # The evidence read here predates the writer's admission of this
        # pass. The sealed digest lets the writer detect evidence that moved
        # in between (a sidecar admitted in the same pass) and re-enrich.
        return prepare_jsonl_blob(
            str(snapshot),
            source_path,
            provider.value,
            fallback_id,
            is_stream=is_stream,
            shard_directory=shard_directory,
            publication_publisher=ArchiveBlobPublisher(Path(evidence.source_db_path), Path(evidence.blob_root)),
            attempt_directory=None if attempt_directory is None else Path(attempt_directory),
            parse_prefix_size=parse_prefix_size,
            profile_identity=profile_identity,
            prepare_session=enrich,
            # The live parse joins tool-output sidecars from the source tree
            # (as ``parse_payload`` does by default); a sealed carrier without
            # them would keep masked excerpts the live route replaces.
            sidecar_resolver=FilesystemSidecarResolver(),
            preparation_dependency=lambda: (
                enrich.dependency_digest(),
                str(Path(evidence.index_db_path).resolve()),
            ),
            source_sha256=snapshot_sha256,
            strict_jsonl_records=True,
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


@dataclass(frozen=True, slots=True)
class PreparedLiveSQLiteCapture:
    """One accepted logical export and its existing publication owner."""

    snapshot: SQLiteBlobSnapshot | None
    publisher: ArchiveBlobPublisher
    preparation: LivePathPreparation | None
    admission: PreAcquisitionDecision
    source_stat: os.stat_result
    observed_at_ns: int

    def discard(self) -> None:
        failures: list[BaseException] = []
        if self.preparation is not None:
            try:
                self.preparation.discard()
            except BaseException as exc:
                failures.append(exc)
        try:
            self.publisher.discard_pending()
        except BaseException as exc:
            failures.append(exc)
        if failures:
            raise BaseExceptionGroup("SQLite capture cleanup failed", failures)


class LiveParseStage:
    """Owns the watcher's bounded off-writer-hold path preparation executor.

    One instance lives for the ``LiveWatcher``'s lifetime, created on
    construction. ``warm_paths`` and ``prefetch_paths`` are synchronous and
    blocking: callers run them off the event loop (``asyncio.to_thread``) and
    never under the write coordinator's hold, because a preparation parses a
    whole file.
    """

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        max_inflight_bytes: int | None = None,
        stall_report_seconds: float | None = None,
        shard_directory: Path | None = None,
        preempt_grace_seconds: float = _PREEMPT_GRACE_SECONDS,
    ) -> None:
        # polylogue-bp12n.6. Where a worker's sealed shard goes, or ``None``
        # to keep row binding on the writer thread. Path workers receive a
        # parent-created attempt child so their files remain attributable
        # until publication or discard.
        self._shard_directory = shard_directory
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
        self._path_attempt_dirs: dict[str, Path] = {}
        self._path_inflight_bytes = 0
        self._closing = False
        self.cleanup_failure_count = 0
        self._cleanup_blocked = False
        if shard_directory is not None:
            shard_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._attempt_root: Path | None = shard_directory / f".live-parse-attempts-{uuid.uuid4().hex}"
            self._attempt_root.mkdir(mode=0o700)
        else:
            self._attempt_root = None
        self._executor = compute_adapter()
        requested = max_workers if max_workers is not None else live_watcher_parse_stage_worker_count()
        self._max_path_pending = max(
            1,
            min(
                requested,
                self._executor.snapshot().by_class("incremental-background").ceiling_units,
            ),
        )
        self._operations: dict[Future[LivePathPreparation], SubmittedOperation[LivePathPreparation]] = {}
        self._max_path_bytes = max_inflight_bytes or live_watcher_parse_stage_max_inflight_bytes()
        self._stall_report_seconds = (
            stall_report_seconds
            if stall_report_seconds is not None
            else live_watcher_parse_stage_stall_report_seconds()
        )

    def prepare_sqlite_paths(
        self,
        paths: Sequence[Path],
        *,
        archive_root: Path,
        cancelled: threading.Event,
        fallback_provider: Provider,
        source_only: bool,
    ) -> dict[Path, PreparedLiveSQLiteCapture | Exception]:
        """Acquire and seal declared Codex state before writer admission."""
        from polylogue.sources.live.batch_support import classify_pre_acquisition
        from polylogue.sources.source_staging import bind_source_input
        from polylogue.sources.sqlite_snapshot import snapshot_sqlite_to_blob
        from polylogue.storage.blob_publication import ArchiveBlobPublisher

        captures: dict[Path, PreparedLiveSQLiteCapture | Exception] = {}
        with self._stage_lock:
            with self._publish_lock:
                if self._closing:
                    raise DaemonOperationCancelled("state preparation is closing")
                self._active_cancel = cancelled

            def prepare(path: Path) -> tuple[Path, PreparedLiveSQLiteCapture | Exception]:
                with retain_native_sql_lifetimes(self._attempt_root):
                    token = compute_cancel.set(cancelled)
                    publisher = ArchiveBlobPublisher(archive_root / "source.db", archive_root / "blob")
                    artifact: LivePathPreparation | None = None
                    try:
                        check_compute_cancelled()
                        with bind_source_input(path) as binding:
                            observed_at_ns = time.time_ns()
                            source_stat = os.stat(
                                binding.physical_path.name, dir_fd=binding.parent_anchor, follow_symlinks=False
                            )
                            admission = classify_pre_acquisition(
                                binding.source_path,
                                fallback_provider=fallback_provider,
                                source_only=source_only,
                                size_bytes=source_stat.st_size,
                                source_binding=binding,
                            )
                            if admission.excluded_reason is not None:
                                return path, PreparedLiveSQLiteCapture(
                                    None, publisher, None, admission, source_stat, observed_at_ns
                                )
                            snapshot = snapshot_sqlite_to_blob(path, publisher, source_binding=binding)
                        if not source_only:
                            retained_path = publisher.blob_path(snapshot.blob_hash)
                            attempt = self._new_attempt_directory()
                            with retain_native_sql_lifetimes(attempt):
                                artifact = prepare_jsonl_blob(
                                    str(retained_path),
                                    str(snapshot.source_path),
                                    Provider.CODEX.value,
                                    Path(snapshot.source_path).stem,
                                    is_stream=False,
                                    shard_directory=str(self._attempt_root),
                                    attempt_directory=attempt,
                                    source_sha256=snapshot.blob_hash,
                                    publication_publisher=publisher,
                                )
                                if artifact.error is None:
                                    artifact.verify_files(full=True, stop=cancelled.is_set)
                        check_compute_cancelled()
                        return path, PreparedLiveSQLiteCapture(
                            snapshot, publisher, artifact, admission, source_stat, observed_at_ns
                        )
                    except (DaemonOperationCancelled, VerificationCancelledError) as cancelled_failure:
                        if artifact is not None:
                            artifact.discard()
                        publisher.discard_pending()
                        if isinstance(cancelled_failure, DaemonOperationCancelled):
                            raise
                        raise DaemonOperationCancelled("state preparation cancelled") from cancelled_failure
                    except Exception as failure:
                        if artifact is not None:
                            try:
                                artifact.discard()
                            except BaseException as cleanup:
                                failure.add_note(f"state artifact cleanup failed: {cleanup!r}")
                        try:
                            publisher.discard_pending()
                        except BaseException as cleanup:
                            failure.add_note(f"state capture cleanup failed: {cleanup!r}")
                        return path, failure
                    finally:
                        compute_cancel.reset(token)

            try:
                with closing(
                    self._executor.map(
                        prepare,
                        paths,
                        estimated_bytes=lambda path: path.stat().st_size,
                        discard_unconsumed=lambda item: (
                            item[1].discard() if isinstance(item[1], PreparedLiveSQLiteCapture) else None
                        ),
                    )
                ) as prepared:
                    for path, capture in prepared:
                        captures[path] = capture
                return captures
            except BaseException:
                for capture in captures.values():
                    if isinstance(capture, PreparedLiveSQLiteCapture):
                        capture.discard()
                raise
            finally:
                with self._publish_lock:
                    self._active_cancel = None

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
        forward progress signal. Cancellation stops new admission and asks
        accepted pure units to stop cooperatively. Their physical completion
        remains the boundary for scratch cleanup. Unclaimed read-ahead may
        leave the local window, but keeps its shared reservation until drain.
        """
        wanted = {source_path for source_path, _provider, _is_stream in candidates}
        progress = -1
        last_progress = time.monotonic()
        reported_at = last_progress
        while not self._cleanup_blocked:
            if cancelled is not None and cancelled.is_set():
                return
            # Admit remaining required work as the shared capacity permits.
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
            self._operations.pop(future, None)
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

        Cancellation does not certify physical completion. An orphan keeps
        its shared reservation and scratch until its original worker settles;
        collection then discards the unused carrier.
        """
        emit(
            "live.parse_prefetch.speculation_reaped",
            level=WARNING,
            outcome="degraded",
            reason=reason,
            paths=len(paths),
        )
        reaped = frozenset(paths)
        for source_path in reaped:
            future = self._path_futures.get(source_path)
            if future is None:
                continue
            self._operations[future].cancellation.cancel()
            self._orphans[future] = self._release_slot(source_path)[0]
            self._speculative.pop(source_path, None)
            self._preempt_at.pop(source_path, None)

    def _release_slot(self, source_path: str) -> tuple[Path | None, SourceObservation | None]:
        """Forget a running preparation's bookkeeping and release its charge.

        Returns its attempt directory and its submission observation.
        """
        self._path_futures.pop(source_path, None)
        observation = self._path_observations.pop(source_path, None)
        if observation is not None:
            self._path_inflight_bytes -= observation[0]
        return self._path_attempt_dirs.pop(source_path, None), observation

    def _attempt_bytes(self, paths: set[str]) -> int:
        """Bytes the running preparations of ``paths`` have written so far.

        The source snapshot a worker copies first sits one directory down
        (``source_snapshot``); its growth is progress too.
        """
        total = 0
        for source_path in paths:
            directory = self._path_attempt_dirs.get(source_path)
            if directory is None:
                continue
            try:
                for entry in os.scandir(directory):
                    with suppress(OSError):
                        if entry.is_dir(follow_symlinks=False):
                            total += sum(
                                nested.stat(follow_symlinks=False).st_size for nested in os.scandir(entry.path)
                            )
                        else:
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
        for source_path, provider, is_stream in candidates:
            if source_path in self._path_results or source_path in self._path_futures:
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
                future = self._submit_path_task(
                    source_path,
                    provider,
                    is_stream,
                    attempt_directory,
                    speculative=speculative,
                    evidence=evidence,
                    estimated_bytes=source_bytes,
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
            self._path_inflight_bytes += source_bytes
        return next_wave

    def _submit_path_task(
        self,
        source_path: str,
        provider: Provider,
        is_stream: bool,
        attempt_directory: Path,
        *,
        speculative: bool,
        evidence: LiveEnrichmentEvidence | None,
        estimated_bytes: int,
    ) -> Future[LivePathPreparation]:
        worker = live_lookahead_path_worker if speculative else live_parse_path_worker
        arguments: dict[str, Any] = {
            "shard_directory": str(self._attempt_root),
            "attempt_directory": str(attempt_directory),
        }
        if not speculative:
            arguments["is_stream"] = is_stream
        if evidence is not None:
            arguments["evidence"] = evidence
        function = partial(worker, provider.value, source_path, Path(source_path).stem, **arguments)
        cancelled = self._active_cancel or compute_cancel.get()

        def run() -> LivePathPreparation:
            token = compute_cancel.set(cancelled)
            try:
                result = function()
                handle = current_cancellation()
                if (cancelled is not None and cancelled.is_set()) or (handle is not None and handle.cancelled):
                    result.discard()
                    raise DaemonOperationCancelled("path preparation cancelled")
                return result
            finally:
                compute_cancel.reset(token)

        with retain_native_sql_lifetimes(self._attempt_root, attempt_directory):
            operation = self._executor.submit(
                run,
                admission_class="incremental-background",
                estimated_bytes=estimated_bytes,
            )
        self._operations[operation.future] = operation
        return operation.future

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
                        writes.append(
                            prepare_session_write(
                                index_conn,
                                session,
                                merge_append=False,
                                source_conn=source_conn,
                                raw_id=expected_raw_id,
                                child_source_path=str(path),
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

        def prepare_item(
            item: tuple[str, LivePathPreparation],
        ) -> tuple[str, tuple[LivePathPreparation, frozenset[str], dict[str, PreparedLiveRetainedRaw]]]:
            path, result = item
            return path, prepare_one(path, result)

        def discard_unconsumed(
            item: tuple[str, tuple[LivePathPreparation, frozenset[str], dict[str, PreparedLiveRetainedRaw]]],
        ) -> None:
            _path, (prepared, _session_ids, retained) = item
            for write in prepared.prepared_writes:
                write.close()
            for member in retained.values():
                member.discard()

        prepared_results = self._executor.map(
            prepare_item,
            pending.items(),
            estimated_bytes=lambda item: item[1].sessions_seal.size if item[1].sessions_seal is not None else 0,
            discard_unconsumed=discard_unconsumed,
        )
        with closing(prepared_results):
            for path, (prepared, session_ids, retained) in prepared_results:
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
        return frozenset(held)

    def _collect_path_future(self, source_path: str, future: Future[LivePathPreparation]) -> None:
        if self._path_futures.get(source_path) is not future:
            return
        expired = (
            source_path in self._speculative
            and self._stage_calls - self._speculative[source_path] >= _SPECULATIVE_LIFETIME_CALLS
        )
        attempt_directory, observation = self._release_slot(source_path)
        self._operations.pop(future, None)
        try:
            result = future.result()
        except (DaemonBackpressureError, DaemonOperationCancelled) as exc:
            result = LivePathPreparation(None, None, None, exc.code, deferred=True)
        except Exception as exc:
            # The worker came back from this file, whatever it raised.
            result = LivePathPreparation(None, None, None, f"worker failed: {type(exc).__name__}"[:500], deferred=True)
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

    def _new_attempt_directory(self) -> Path:
        if self._attempt_root is None:
            raise RuntimeError("path preparation has no owned scratch root")
        path = self._attempt_root / f"attempt-{uuid.uuid4().hex}"
        path.mkdir(mode=0o700)
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
        from polylogue.storage.sqlite.connection_profile import retained_native_sql_owners_for_lifetime

        if retained_native_sql_owners_for_lifetime(path) or (
            self._closing and retained_native_sql_owners_for_lifetime(self._attempt_root)
        ):
            self._cleanup_blocked = True
            self._record_cleanup_failure("attempt scratch still has native custody")
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

    def pop_path(
        self, source_path: str, *, blob_hash: str, profile_identity: str | None = None
    ) -> LivePathPreparation | None:
        future = self._path_futures.get(source_path)
        if future is not None or source_path in self._unverified:
            # pop_path runs under writer admission. Even a finished future
            # needs a full artifact digest and prior-session reconciliation,
            # both of which belong to the next off-lease warm_paths pass.
            return LivePathPreparation(None, None, None, "worker preparation pending", deferred=True)
        result = self._path_results.pop(source_path, None)
        if result is None:
            return None
        if result.resolved_provider is Provider.HERMES and result.captured_profile_key != profile_identity:
            self._discard_retained_path(source_path)
            result.discard()
            return LivePathPreparation(None, None, None, "captured profile changed after preparation", deferred=True)
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
        members = self._retained_by_path.get(source_path, {})
        failures: list[BaseException] = []
        for raw_id, member in tuple(members.items()):
            try:
                member.discard()
            except BaseException as exc:
                failures.append(exc)
            else:
                del members[raw_id]
        if not members:
            self._retained_by_path.pop(source_path, None)
        if failures:
            raise BaseExceptionGroup("retained path cleanup failed", failures)

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

    def shutdown(self) -> None:
        # Join every admitted physical task before removing its scratch.
        # Failed native settlement keeps that task pending on its creator.
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
        for future, operation in self._operations.items():
            if not future.done():
                operation.cancellation.cancel()
                operation.retry_sql_settlement()
        for future in self._operations:
            with suppress(BaseException):
                future.result()
        for source_path, future in tuple(self._path_futures.items()):
            self._collect_path_future(source_path, future)
        self._collect_finished()
        failures: list[BaseException] = []
        for source_path, result in tuple(self._path_results.items()):
            try:
                result.discard()
            except BaseException as exc:
                failures.append(exc)
            else:
                del self._path_results[source_path]
        for source_path in tuple(self._retained_by_path):
            try:
                self._discard_retained_path(source_path)
            except BaseException as exc:
                failures.append(exc)
        # A failed carrier still owns its paths; do not sweep beneath it.
        if failures:
            raise BaseExceptionGroup("live stage cleanup failed", failures)
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
    "LiveParseStage",
    "live_lookahead_path_worker",
    "live_watcher_parse_stage_max_inflight_bytes",
    "live_watcher_parse_stage_stall_report_seconds",
    "live_watcher_parse_stage_worker_count",
]
