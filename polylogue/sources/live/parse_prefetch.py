"""Off-writer preparation for the live watcher's full-ingest route.

Path workers leave sealed SQLite carriers and row shards on disk. Admission
caps concurrent task count and captured source bytes; a lone oversized file
may run, while additional files remain retryable. The publisher checks the
captured blob hash before consuming a carrier.
"""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from concurrent.futures.process import BrokenProcessPool
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from polylogue.core.enums import Provider
from polylogue.logging import WARNING, emit, get_logger
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import parse_payload, parse_stream_payload
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.prepared_jsonl import PreparedJsonl as LivePathPreparation
from polylogue.sources.prepared_jsonl import prepare_jsonl_blob
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_shard
from polylogue.storage.sqlite.archive_tiers.write_shard import discard_session_shard

logger = get_logger(__name__)

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionWrite


#: Consecutive worker deaths on one unchanged file before its preparation is
#: reported as a failure rather than deferred again. A preparation that
#: outlives its deadline counts as a death.
_MAX_WORKER_DEATHS_PER_OBSERVATION = 3

#: A path preparation still running past its deadline is treated as a hung
#: worker: the pool is restarted and only that path is charged (7c5k6). The
#: deadline grows with the captured bytes at a floor throughput far below any
#: healthy parse, so a slow large file finishes and only a hang reaches it.
_PATH_PREPARATION_DEADLINE_BASE_S = 600.0
_PATH_PREPARATION_DEADLINE_FLOOR_BYTES_PER_S = 256 * 1024


class PreparedReadSnapshot(Protocol):
    @property
    def archive(self) -> ArchiveStore: ...


ReadSnapshot = Callable[[Path], AbstractContextManager[PreparedReadSnapshot]]

_DEFAULT_WORKER_COUNT_FLOOR = 1
_DEFAULT_WARM_TIMEOUT_SECONDS = 60.0

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


def live_watcher_parse_stage_warm_timeout_seconds() -> float:
    """Bound on how long a watcher prefetch warm() pass waits for its workers.

    Override with ``POLYLOGUE_LIVE_WATCHER_PARSE_STAGE_WARM_TIMEOUT_SECONDS``.
    """
    from polylogue.config import load_polylogue_config

    configured = load_polylogue_config().live_watcher_parse_stage_warm_timeout_seconds
    if configured is not None and configured > 0:
        return configured
    return _DEFAULT_WARM_TIMEOUT_SECONDS


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


#: Marker a worker writes as its first act, so a caller whose pool just
#: broke can tell an actively-running preparation from one still queued
#: when it died: only the former can plausibly be the death's cause.
_WORKER_STARTED_MARKER = ".worker-started"
#: Marker a worker writes when its preparation returns or raises. A worker
#: that wrote it was alive at the end of its task, so it cannot have died
#: while holding it. Started-without-finished names the tasks that were
#: held by a worker at the moment the pool broke.
_WORKER_FINISHED_MARKER = ".worker-finished"


def live_parse_path_worker(
    provider_value: str,
    source_path: str,
    fallback_id: str,
    *,
    is_stream: bool,
    shard_directory: str,
    attempt_directory: str | None = None,
) -> LivePathPreparation:
    if attempt_directory is not None:
        (Path(attempt_directory) / _WORKER_STARTED_MARKER).touch()
    try:
        return _prepare_live_path(
            provider_value,
            source_path,
            fallback_id,
            is_stream=is_stream,
            shard_directory=shard_directory,
            attempt_directory=attempt_directory,
        )
    finally:
        if attempt_directory is not None:
            (Path(attempt_directory) / _WORKER_FINISHED_MARKER).touch()


def _prepare_live_path(
    provider_value: str,
    source_path: str,
    fallback_id: str,
    *,
    is_stream: bool,
    shard_directory: str,
    attempt_directory: str | None,
) -> LivePathPreparation:
    from polylogue.sources.dispatch import is_jsonl_source_path
    from polylogue.sources.live.batch_support import _detect_provider_from_path_sample, jsonl_complete_prefix_path

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
    )


def _worker_held_task(attempt_directory: Path | None) -> bool:
    if attempt_directory is None:
        return False
    return (attempt_directory / _WORKER_STARTED_MARKER).exists() and not (
        attempt_directory / _WORKER_FINISHED_MARKER
    ).exists()


def _observe(source_path: str) -> tuple[int, int, int] | None:
    try:
        observed = Path(source_path).stat()
    except OSError:
        return None
    return (observed.st_size, observed.st_mtime_ns, observed.st_ino)


def _discard_orphaned_shard(
    future: Future[tuple[str, list[ParsedSession] | None, BaseException | None, str | None]],
) -> None:
    """Discard the shard sealed by a worker nobody is waiting on any more.

    Runs on the worker's own thread once it finishes. A worker abandoned by a
    ``warm()`` timeout still seals a shard, and its ``shard_name`` is never
    returned to a consumer, so without this the file survives until the stage
    shuts down. Failure to parse, or to remove, is not the caller's problem:
    the shard is already unreferenced either way.
    """
    if future.cancelled():
        return
    if future.exception() is not None:
        return
    shard_name = future.result()[3]
    if shard_name is not None:
        discard_session_shard(Path(shard_name))


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
        warm_timeout_seconds: float | None = None,
        shard_directory: Path | None = None,
        use_processes: bool = False,
        preparation_deadline_base_seconds: float = _PATH_PREPARATION_DEADLINE_BASE_S,
        preparation_deadline_floor_bytes_per_second: float = _PATH_PREPARATION_DEADLINE_FLOOR_BYTES_PER_S,
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
        self._path_futures: dict[str, Future[LivePathPreparation]] = {}
        self._path_sizes: dict[str, int] = {}
        #: source path -> (observation, consecutive worker losses on it). The
        #: observation is the file's (size, mtime_ns, inode) when submitted,
        #: so rewritten content starts a new streak, and any successful
        #: preparation clears the path. A file that kills or hangs its worker
        #: every time escalates to a terminal preparation failure instead of
        #: being deferred and re-submitted for the whole build (7c5k6).
        self._path_worker_deaths: dict[str, tuple[tuple[int, int, int], int]] = {}
        #: source path -> its observation when its running preparation began.
        self._path_observations: dict[str, tuple[int, int, int]] = {}
        #: Paths whose preparation a worker held when the pool broke. Each is
        #: prepared alone until it succeeds: with two held at one break the
        #: death is unattributable and neither is charged, and running each
        #: alone makes every later death name exactly one file.
        self._path_solo_suspects: set[str] = set()
        #: source path -> monotonic submission time of its running preparation.
        self._path_started: dict[str, float] = {}
        self._preparation_deadline_base_s = preparation_deadline_base_seconds
        self._preparation_deadline_floor_bytes_per_s = preparation_deadline_floor_bytes_per_second
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
        self._worker_count = worker_count
        self._max_path_pending = max(1, min(worker_count, 2))
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
        self._warm_timeout_seconds = (
            warm_timeout_seconds
            if warm_timeout_seconds is not None
            else live_watcher_parse_stage_warm_timeout_seconds()
        )

    def warm_paths(
        self,
        candidates: Sequence[tuple[str, Provider, bool]],
        *,
        archive_root: Path | None = None,
        read_snapshot: ReadSnapshot | None = None,
        capture_mode: Provider | None = None,
        source_index: int = 0,
    ) -> int:
        """Prepare path-backed JSON/JSONL outside the writer lease.

        Every selected path gets a result, including worker death and timeout.
        The publisher can therefore retain raw bytes and retry without an
        accidental inline parse when preparation failed.
        """
        if self._shard_directory is None:
            return 0
        if self._cleanup_blocked:
            for source_path, _provider, _is_stream in candidates:
                if source_path not in self._path_results and source_path not in self._path_futures:
                    self._path_results[source_path] = LivePathPreparation(
                        None, None, None, "worker stop could not be verified; scratch cleanup is blocked", deferred=True
                    )
            return len(candidates)
        self._reap_hung_preparations()
        deadline = time.monotonic() + self._warm_timeout_seconds
        remaining = list(candidates)
        while remaining:
            for source_path, future in tuple(self._path_futures.items()):
                if future.done():
                    self._collect_path_future(source_path, future)
            next_wave: list[tuple[str, Provider, bool]] = []
            suspect_waiting = any(
                source_path in self._path_solo_suspects
                and source_path not in self._path_results
                and source_path not in self._path_futures
                for source_path, _provider, _is_stream in remaining
            )
            for source_path, provider, is_stream in remaining:
                if source_path in self._path_results or source_path in self._path_futures:
                    continue
                if self._solo_preparation_blocks(source_path, suspect_waiting=suspect_waiting):
                    next_wave.append((source_path, provider, is_stream))
                    continue
                try:
                    observed = Path(source_path).stat()
                    source_bytes = observed.st_size
                except OSError as exc:
                    self._path_results[source_path] = LivePathPreparation(
                        None, None, None, f"source stat failed: {type(exc).__name__}"[:500], deferred=True
                    )
                    continue
                if len(self._path_futures) >= self._max_path_pending or (
                    self._path_futures and self._path_inflight_bytes + source_bytes > self._max_path_bytes
                ):
                    next_wave.append((source_path, provider, is_stream))
                    continue
                attempt_directory: Path | None = None
                try:
                    attempt_directory = self._new_attempt_directory()
                    future = self._executor.submit(
                        live_parse_path_worker,
                        provider.value,
                        source_path,
                        Path(source_path).stem,
                        is_stream=is_stream,
                        shard_directory=str(self._attempt_root),
                        attempt_directory=str(attempt_directory),
                    )
                except Exception as exc:
                    if attempt_directory is not None:
                        self._remove_attempt_directory(attempt_directory)
                    self._path_results[source_path] = LivePathPreparation(
                        None, None, None, f"worker submission failed: {type(exc).__name__}"[:500], deferred=True
                    )
                    continue
                self._path_futures[source_path] = future
                self._path_started[source_path] = time.monotonic()
                self._path_attempt_dirs[source_path] = attempt_directory
                self._path_sizes[source_path] = source_bytes
                self._path_observations[source_path] = (observed.st_size, observed.st_mtime_ns, observed.st_ino)
                self._path_inflight_bytes += source_bytes
            remaining = next_wave
            if not remaining:
                break
            available = max(0.0, deadline - time.monotonic())
            if not self._path_futures or available == 0:
                break
            done, _pending = wait(tuple(self._path_futures.values()), timeout=available, return_when=FIRST_COMPLETED)
            if not done:
                break
        for source_path, _provider, _is_stream in remaining:
            if source_path in self._path_results or source_path in self._path_futures:
                continue
            reason = (
                "worker preparation is held for a suspected worker-killing file running alone"
                if self._path_solo_suspects
                and (source_path in self._path_solo_suspects or self._solo_preparation_running())
                else "worker preparation capacity is busy"
                if len(self._path_futures) >= self._max_path_pending
                else "worker preparation byte capacity is busy"
            )
            self._path_results[source_path] = LivePathPreparation(None, None, None, reason, deferred=True)
        selected_futures = {
            self._path_futures[source_path]
            for source_path, _provider, _is_stream in candidates
            if source_path in self._path_futures
        }
        if selected_futures:
            _done, _pending = wait(selected_futures, timeout=max(0.0, deadline - time.monotonic()))
            for source_path, future in tuple(self._path_futures.items()):
                if future.done():
                    self._collect_path_future(source_path, future)
            # A slow worker is still the owner of its captured source. Keep
            # its future so a later pass can consume the sealed artifact;
            # restarting the pool here killed every whale at the same warm
            # deadline on each retry. BrokenProcessPool is handled when the
            # finished future is collected, and shutdown terminates stragglers.
        for source_path, future in tuple(self._path_futures.items()):
            if future.done():
                self._collect_path_future(source_path, future)
        self._reap_hung_preparations()
        if archive_root is not None:
            self._prepare_existing_session_writes(
                archive_root,
                read_snapshot=read_snapshot,
                capture_mode=capture_mode,
                source_index=source_index,
            )
        return len(candidates)

    def _prepare_existing_session_writes(
        self,
        archive_root: Path,
        *,
        read_snapshot: ReadSnapshot | None,
        capture_mode: Provider | None,
        source_index: int,
    ) -> None:
        """Reconcile prior acquisitions on a read-only index before admission."""
        from polylogue.core.identity_law import session_id as archive_session_id
        from polylogue.core.sources import origin_from_provider
        from polylogue.storage.sqlite.archive_tiers.source_write import deterministic_raw_session_id
        from polylogue.storage.sqlite.archive_tiers.write import (
            prepare_session_write,
            prepared_session_rows_from_shard,
        )

        pending = {
            path: result
            for path, result in self._path_results.items()
            if result.error is None and not result.prepared_writes
        }
        if not pending:
            return
        if read_snapshot is None:
            for path, result in pending.items():
                result.discard()
                self._path_results[path] = LivePathPreparation(
                    None, None, None, "read-only preparation snapshot unavailable: RuntimeError", deferred=True
                )
            return

        def prepare_one(path: str, result: LivePathPreparation) -> LivePathPreparation:
            # Each task owns its read transaction. Sharing one SQLite
            # connection across threads would also share its snapshot state.
            writes: list[PreparedSessionWrite] = []
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
                        session_id = archive_session_id(
                            origin_from_provider(session.source_name).value,
                            session.provider_session_id,
                        )
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
                return replace(result, prepared_writes=tuple(writes))
            except Exception as exc:
                for prepared in writes:
                    prepared.close()
                result.discard()
                return LivePathPreparation(
                    None,
                    None,
                    None,
                    (
                        f"existing-session preparation failed: {type(exc).__name__}"
                        if opened_snapshot
                        else f"read-only preparation snapshot unavailable: {type(exc).__name__}"
                    )[:500],
                    deferred=True,
                )

        # Bound reconciliation to the same path admission width as parsing.
        # Results are installed on the caller thread, so publication order is
        # still the intake order even when read tasks finish out of order.
        with ThreadPoolExecutor(max_workers=min(self._max_path_pending, len(pending))) as executor:
            futures = {path: executor.submit(prepare_one, path, result) for path, result in pending.items()}
            for path, future in futures.items():
                self._path_results[path] = future.result()

    def _collect_path_future(self, source_path: str, future: Future[LivePathPreparation]) -> None:
        if self._path_futures.get(source_path) is not future:
            return
        self._path_futures.pop(source_path, None)
        self._path_started.pop(source_path, None)
        observed_size = self._path_sizes.pop(source_path, 0)
        self._path_inflight_bytes -= observed_size
        attempt_directory = self._path_attempt_dirs.pop(source_path, None)
        observation = self._path_observations.pop(source_path, (observed_size, 0, 0))
        try:
            result = future.result()
        except BrokenProcessPool:
            result = self._attribute_pool_death(source_path, attempt_directory, observation)
        except Exception as exc:
            # The worker came back alive, whatever it raised: the loss
            # streak and the suspicion both end here.
            self._path_worker_deaths.pop(source_path, None)
            self._path_solo_suspects.discard(source_path)
            result = LivePathPreparation(None, None, None, f"worker failed: {type(exc).__name__}"[:500], deferred=True)
        else:
            self._path_worker_deaths.pop(source_path, None)
            self._path_solo_suspects.discard(source_path)
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
        if result.error is None:
            try:
                # A full byte scan belongs at the prefetch boundary, before
                # the caller enters the writer runner. Publication only needs
                # the cheap exact-inode check in pop_path/iter_sessions.
                result.verify_files(full=True)
            except (OSError, ValueError) as exc:
                result.discard()
                result = LivePathPreparation(
                    None,
                    None,
                    None,
                    f"worker artifact changed: {type(exc).__name__}"[:500],
                    deferred=True,
                )
        if result.error is None:
            self._path_worker_deaths.pop(source_path, None)
        old = self._path_results.pop(source_path, None)
        if old is not None:
            old.discard()
        self._path_results[source_path] = result

    def _solo_preparation_running(self) -> bool:
        return any(source_path in self._path_solo_suspects for source_path in self._path_futures)

    def _solo_preparation_blocks(self, source_path: str, *, suspect_waiting: bool) -> bool:
        """Whether submitting ``source_path`` now would share the pool with a suspect.

        A suspect runs only on an otherwise idle pool, and nothing joins it.
        While a suspect waits for the pool to drain, no other path is
        admitted, so the drain finishes instead of starving it.
        """
        if not self._path_solo_suspects:
            return False
        if self._solo_preparation_running():
            return True
        if source_path in self._path_solo_suspects:
            return bool(self._path_futures)
        return suspect_waiting

    def _attribute_pool_death(
        self,
        source_path: str,
        attempt_directory: Path | None,
        observation: tuple[int, int, int],
    ) -> LivePathPreparation:
        """Charge a pool break to the one preparation a worker held, if unique.

        A broken pool completes every outstanding future with
        ``BrokenProcessPool``, not only the one whose worker died. Each worker
        marks its attempt directory when it takes a task and when it lets go
        of it, so the tasks held at the break are exactly those started and
        not finished. Queued tasks and tasks that finished are not the cause.
        One held task is charged. Two or more are all deferred uncharged. Every
        held task becomes a solo suspect, prepared alone until it succeeds, so
        each later death has a single holder and the charge streak is exact.
        """
        if self._closing:
            return LivePathPreparation(None, None, None, "worker pool stopped during shutdown", deferred=True)
        held: list[tuple[str, tuple[int, int, int]]] = []
        if _worker_held_task(attempt_directory):
            held.append((source_path, observation))
        for sibling_path, sibling in self._path_futures.items():
            if sibling.done() and not sibling.cancelled() and sibling.exception() is None:
                continue
            if _worker_held_task(self._path_attempt_dirs.get(sibling_path)):
                held.append(
                    (
                        sibling_path,
                        self._path_observations.get(sibling_path, (self._path_sizes.get(sibling_path, 0), 0, 0)),
                    )
                )
        cause = "worker process died during preparation"
        ambiguous = f"{cause}; {len(held)} preparations were running, retrying each alone to attribute it"
        culprit = held[0] if len(held) == 1 else None
        if len(held) > 1:
            self._path_solo_suspects.update(path for path, _observation in held)
            emit(
                "live.parse_prefetch.worker_death_ambiguous",
                level=WARNING,
                outcome="degraded",
                reason="worker_death_ambiguous",
                path=source_path,
                attempts=len(held),
            )
        if culprit is not None and culprit[0] == source_path:
            self._path_solo_suspects.add(source_path)
            result = self._charge_worker_loss(source_path, observation, cause)
        elif len(held) > 1 and source_path in self._path_solo_suspects:
            result = LivePathPreparation(None, None, None, ambiguous, deferred=True)
        else:
            result = LivePathPreparation(
                None, None, None, "worker pool broke while this preparation was not running", deferred=True
            )
        # Siblings the restart cancels are deferred uncharged; the one
        # charged sibling is overwritten below.
        self._restart_broken_process_pool(
            failed_attempts=(attempt_directory,) if attempt_directory else (),
            reason=ambiguous if len(held) > 1 else "worker pool restarted after another preparation's worker died",
        )
        if culprit is not None and culprit[0] != source_path:
            culprit_path, culprit_observation = culprit
            self._path_solo_suspects.add(culprit_path)
            old = self._path_results.pop(culprit_path, None)
            if old is not None:
                old.discard()
            self._path_results[culprit_path] = self._charge_worker_loss(culprit_path, culprit_observation, cause)
        return result

    def _charge_worker_loss(
        self, source_path: str, observation: tuple[int, int, int], cause: str
    ) -> LivePathPreparation:
        """Count one lost worker against this exact observation of the file.

        Below the bound the loss is deferred, like any interrupted
        preparation. At the bound it is not deferred: the writer records a
        parse failure, and the cursor's finite failure budget quarantines the
        file with a visible reason instead of retrying it every pass.
        """
        previous = self._path_worker_deaths.get(source_path)
        deaths = previous[1] + 1 if previous is not None and previous[0] == observation else 1
        self._path_worker_deaths[source_path] = (observation, deaths)
        if deaths < _MAX_WORKER_DEATHS_PER_OBSERVATION:
            return LivePathPreparation(None, None, None, cause, deferred=True)
        # A terminal failure is never prepared again, so it stops being a
        # suspect that would hold the pool for a solo run.
        self._path_solo_suspects.discard(source_path)
        emit(
            "live.parse_prefetch.worker_death_escalated",
            level=WARNING,
            outcome="refused",
            reason="worker_died_repeatedly",
            path=source_path,
            attempts=deaths,
            error_detail=cause,
        )
        return LivePathPreparation(
            None,
            None,
            None,
            f"{cause}; worker lost on {deaths} consecutive preparations of this file",
            deferred=False,
            failed_observation=observation,
        )

    def _preparation_deadline_s(self, observed_size: int) -> float:
        return self._preparation_deadline_base_s + observed_size / max(
            1.0, self._preparation_deadline_floor_bytes_per_s
        )

    def _reap_hung_preparations(self) -> None:
        """Stop preparations running past their deadline; charge only them.

        A thread worker cannot be stopped, so only a process pool is reaped.
        Sibling preparations cancelled by the restart are deferred without a
        charge, and the fresh pool frees every slot the hung workers held.
        """
        if self._closing or self._cleanup_blocked or not isinstance(self._executor, ProcessPoolExecutor):
            return
        now = time.monotonic()
        hung = [
            source_path
            for source_path, future in self._path_futures.items()
            if not future.done()
            and now - self._path_started.get(source_path, now)
            > self._preparation_deadline_s(self._path_sizes.get(source_path, 0))
        ]
        if not hung:
            return
        failed_attempts: list[Path] = []
        for source_path in hung:
            self._path_futures.pop(source_path, None)
            started = self._path_started.pop(source_path, now)
            observed_size = self._path_sizes.pop(source_path, 0)
            self._path_inflight_bytes -= observed_size
            observation = self._path_observations.pop(source_path, (observed_size, 0, 0))
            attempt_directory = self._path_attempt_dirs.pop(source_path, None)
            if attempt_directory is not None:
                failed_attempts.append(attempt_directory)
            emit(
                "live.parse_prefetch.preparation_deadline_exceeded",
                level=WARNING,
                outcome="degraded",
                reason="preparation_deadline_exceeded",
                path=source_path,
                duration_ms=round((now - started) * 1000),
            )
            old = self._path_results.pop(source_path, None)
            if old is not None:
                old.discard()
            self._path_solo_suspects.discard(source_path)
            self._path_results[source_path] = self._charge_worker_loss(
                source_path, observation, "worker preparation exceeded its deadline"
            )
        self._restart_broken_process_pool(
            failed_attempts=tuple(failed_attempts),
            reason="worker pool restarted after a hung preparation",
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
        failed_attempts: Sequence[Path] = (),
        reason: str = "worker process died during preparation",
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
        for failed_attempt in failed_attempts:
            self._remove_attempt_directory(failed_attempt)
        for pending_path, future in pending:
            attempt_directory = self._path_attempt_dirs.pop(pending_path, None)
            self._path_futures.pop(pending_path, None)
            self._path_started.pop(pending_path, None)
            self._path_inflight_bytes -= self._path_sizes.pop(pending_path, 0)
            self._path_observations.pop(pending_path, None)
            # A sibling may have sealed successfully just before the pool
            # broke. Its result is still useful and retains its own carrier.
            if future.done() and not future.cancelled():
                try:
                    result = future.result()
                except Exception:
                    result = None
                if result is not None and result.error is None and attempt_directory is not None:
                    try:
                        self._validate_attempt_result(result, attempt_directory)
                        result.verify_files(full=True)
                        result = replace(result, attempt_directory=attempt_directory)
                        self._path_results[pending_path] = result
                        # A success ends its worker-loss streak and its
                        # suspicion here too, as a collected success does.
                        self._path_worker_deaths.pop(pending_path, None)
                        self._path_solo_suspects.discard(pending_path)
                        continue
                    except (OSError, ValueError):
                        pass
            self._path_results[pending_path] = LivePathPreparation(None, None, None, reason, deferred=True)
            if attempt_directory is not None:
                self._remove_attempt_directory(attempt_directory)
        if self._cleanup_blocked:
            return
        self._executor = process_pool_executor(max_workers=self._worker_count)

    def pop_path(self, source_path: str, *, blob_hash: str) -> LivePathPreparation | None:
        future = self._path_futures.get(source_path)
        if future is not None:
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
                result.discard()
                return LivePathPreparation(None, None, None, "captured source changed after preparation", deferred=True)
            if result.failed_observation is not None and _observe(source_path) != result.failed_observation:
                # A lost worker never hashed the bytes it failed on. The file
                # unchanged since that preparation began is the proof the
                # capture holds the same revision; anything else is new bytes.
                result.discard()
                return LivePathPreparation(
                    None, None, None, "captured source changed after the failed preparation", deferred=True
                )
            return result
        if result.blob_hash != blob_hash:
            result.discard()
            return LivePathPreparation(None, None, None, "captured source changed after preparation", deferred=True)
        if result.error is None:
            try:
                result.verify_files(full=False)
            except (OSError, ValueError) as exc:
                result.discard()
                return LivePathPreparation(
                    None,
                    None,
                    None,
                    f"worker artifact identity changed: {type(exc).__name__}"[:500],
                    deferred=True,
                )
        return result

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
                live_parse_and_shard_worker,
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
        try:
            for future in as_completed(futures, timeout=self._warm_timeout_seconds):
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
        except TimeoutError:
            pending_count = len(futures) - completed
            # Leaving the futures alone kept unstarted work queued behind the
            # next warm() and let a late worker's shard sit unreferenced on
            # disk (polylogue-3r36h). Cancel what has not started, discard the
            # shards of what finished unread, and attach the discard to what
            # is still running so the worker cleans up after itself as soon
            # as it finishes (polylogue-nfr2u): a thread cannot be preempted,
            # and nobody consumes its ``shard_name`` after the timeout.
            cancelled = 0
            drained = 0
            for future in futures:
                if future in consumed:
                    continue
                if future.cancel():
                    cancelled += 1
                    continue
                if not future.done():
                    future.add_done_callback(_discard_orphaned_shard)
                    continue
                if future.exception() is not None:
                    continue
                _cache_key, _sessions, _error, shard_name = future.result()
                if shard_name is not None:
                    discard_session_shard(Path(shard_name))
                    drained += 1
            logger.warning(
                "live watcher parse-stage prefetch: warm() timed out after %.0fs waiting on %d of %d file(s); "
                "cancelled %d unstarted worker(s), discarded %d unread shard(s); "
                "leaving unfinished file(s) uncached for the writer-held pass to reparse normally",
                self._warm_timeout_seconds,
                pending_count,
                len(futures),
                cancelled,
                drained,
            )
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
        self._closing = True
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
        self.cache.discard_all()
        for result in self._path_results.values():
            result.discard()
        self._path_results.clear()
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
    "live_parse_and_shard_worker",
    "live_parse_worker",
    "live_watcher_parse_stage_max_inflight_bytes",
    "live_watcher_parse_stage_warm_timeout_seconds",
    "live_watcher_parse_stage_worker_count",
]
