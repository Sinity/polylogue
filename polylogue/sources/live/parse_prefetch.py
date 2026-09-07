"""Off-writer-hold parallel pre-parse for the live watcher's full-ingest route.

polylogue-wf8a (xikl free-threading epic). Mirrors
``polylogue.daemon.parse_prefetch.DaemonParseStage`` (polylogue-m6tp phase
(a)) for the watcher's catch-up / live-batch full-ingest path, instead of the
raw-materialization census path: the CPU-bound ``parse_payload``/
``parse_stream_payload`` decode for small (below-streaming-threshold) JSONL
session files is dispatched to a bounded ``ThreadPoolExecutor`` BEFORE
``LiveBatchProcessor._ingest_full_paths_sync`` ever asks the write
coordinator for its writer hold -- exactly the same safety argument as
``DaemonParseStage`` (see its docstring): threads never run CONCURRENTLY with
an active writer thread, so there is no writer-starvation hazard even on a
standard GIL build, and no wall-clock speedup either until the free-threaded
3.14t deploy makes the same code path genuinely parallel.

Unlike the census prefetch cache (keyed on a content-hash ``raw_id`` already
known from ``source.db``), the live watcher does not know a file's raw_id
until ``ArchiveBlobPublisher.write_from_bytes`` runs INSIDE the writer hold
(the archive tier assigns content-hash identity at write time). This cache is
therefore keyed on the watcher's own path string and bridged to the real
raw_id the instant ``_ingest_full_paths_sync`` computes it -- see the
``LiveParsePrefetchCache.pop`` call site in ``polylogue.sources.live.batch``.

Graceful degradation, same contract as ``DaemonParseStage``: a cache miss
(no ``LiveParseStage`` instance, path not selected for prewarm, worker
exception, or a warm() timeout) always falls back to parsing inline exactly
as before. Prefetch only ever shortcuts the happy path; it never changes
what gets parsed or how parse failures are recorded.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.logging import get_logger
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import parse_payload, parse_stream_payload
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.sqlite.archive_tiers.write import prepare_session_shard
from polylogue.storage.sqlite.archive_tiers.write_shard import discard_session_shard

logger = get_logger(__name__)

_DEFAULT_WORKER_COUNT_FLOOR = 1
_DEFAULT_WARM_TIMEOUT_SECONDS = 60.0

# The watcher's own catch-up/live-batch chunking already caps a single batch
# at 64 MiB (``_CATCH_UP_MAX_BATCH_BYTES`` in ``sources/live/watcher.py``), so
# the adaptive budget below only needs to comfortably cover one batch's worth
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
    """Owns the watcher's bounded off-writer-hold pre-parse ``ThreadPoolExecutor`` + cache.

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
    ) -> None:
        # polylogue-bp12n.6. Where a worker's sealed shard goes, or ``None``
        # to keep row binding on the writer thread. The stage owns the
        # directory's contents: a shard lives from the worker that sealed it
        # to the writer that copied it, and nothing outlives ``shutdown``.
        self._shard_directory = shard_directory
        if shard_directory is not None:
            shard_directory.mkdir(parents=True, exist_ok=True)
            # Anything already here belongs to a process that died before it
            # could copy or delete it. The archive has one writer, so there
            # is no other owner to consult.
            for residue in shard_directory.glob("shard-*"):
                discard_session_shard(residue)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers if max_workers is not None else live_watcher_parse_stage_worker_count(),
            thread_name_prefix="polylogue-live-parse-stage",
        )
        self.cache = LiveParsePrefetchCache(
            max_inflight_bytes=(
                max_inflight_bytes if max_inflight_bytes is not None else live_watcher_parse_stage_max_inflight_bytes()
            )
        )
        self._warm_timeout_seconds = (
            warm_timeout_seconds
            if warm_timeout_seconds is not None
            else live_watcher_parse_stage_warm_timeout_seconds()
        )

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
        try:
            for future in as_completed(futures, timeout=self._warm_timeout_seconds):
                completed += 1
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
                if self.cache.try_admit(cache_key, sessions, payload=candidate.payload, shard_path=shard_path):
                    warmed += 1
        except TimeoutError:
            pending_count = len(futures) - completed
            logger.warning(
                "live watcher parse-stage prefetch: warm() timed out after %.0fs waiting on %d of %d file(s); "
                "leaving unfinished file(s) uncached for the writer-held pass to reparse normally",
                self._warm_timeout_seconds,
                pending_count,
                len(futures),
            )
        return warmed

    def shutdown(self) -> None:
        # Wait for workers that already entered shard construction before
        # removing the directory's residue; otherwise a late worker could
        # seal a shard after cleanup and leave an unattached file behind.
        self._executor.shutdown(wait=True, cancel_futures=True)
        self.cache.discard_all()
        if self._shard_directory is not None:
            for residue in self._shard_directory.glob("shard-*"):
                discard_session_shard(residue)


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
