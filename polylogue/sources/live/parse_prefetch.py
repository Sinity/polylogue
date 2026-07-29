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

import os
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO

from polylogue.core.enums import Provider
from polylogue.logging import get_logger
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import parse_payload, parse_stream_payload
from polylogue.sources.parsers.base import ParsedSession

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
    return max(_DEFAULT_WORKER_COUNT_FLOOR, (os.cpu_count() or 2) - 1)


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
                _iter_json_stream(BytesIO(payload), source_name),
                fallback_id,
                source_path=source_path,
            )
        else:
            payloads = list(_iter_json_stream(BytesIO(payload), source_name))
            sessions = parse_payload(provider, payloads, fallback_id, source_path=source_path)
        return cache_key, sessions, None
    except Exception as exc:
        return cache_key, None, exc


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
        self._entries: dict[str, tuple[list[ParsedSession], bytes]] = {}
        self._inflight_bytes = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def contains(self, cache_key: str) -> bool:
        with self._lock:
            return cache_key in self._entries

    def try_admit(self, cache_key: str, sessions: list[ParsedSession], *, payload: bytes) -> bool:
        """Admit ``sessions`` unless already present or the budget is exceeded.

        A single entry is always admitted even alone over budget (mirrors
        ``RawParsePrefetchCache``'s ``try_admit``) -- the budget bounds how
        much accumulates across MANY entries, not the size of any one file;
        rejecting a lone oversized entry would just mean parsing it inline
        anyway, an unhelpful distinction for a one-file cache.
        """
        payload_bytes = len(payload)
        with self._lock:
            if cache_key in self._entries:
                return False
            if self._entries and self._inflight_bytes + payload_bytes > self._max_inflight_bytes:
                return False
            self._entries[cache_key] = (sessions, payload)
            self._inflight_bytes += payload_bytes
            return True

    def pop(self, cache_key: str, *, payload: bytes) -> list[ParsedSession] | None:
        """Consume a cached entry, but only if it was parsed from ``payload`` exactly.

        Always removes the entry when present (win or mismatch) -- a stale
        entry for a path that has since changed on disk is never useful to a
        later lookup either, since the writer-held pass advances that path's
        cursor past the version prewarm saw.
        """
        with self._lock:
            entry = self._entries.pop(cache_key, None)
            if entry is None:
                return None
            sessions, cached_payload = entry
            self._inflight_bytes -= len(cached_payload)
            if cached_payload != payload:
                logger.warning(
                    "live watcher parse-stage prefetch: cached parse for %s no longer matches "
                    "on-disk bytes (file changed between prewarm and writer hold); reparsing inline",
                    cache_key,
                )
                return None
            return sessions


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
    ) -> None:
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
        futures = {
            self._executor.submit(
                live_parse_worker,
                candidate.cache_key,
                candidate.provider.value,
                candidate.payload,
                candidate.source_path,
                candidate.fallback_id,
                is_stream=candidate.is_stream,
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
                    cache_key, sessions, error = future.result()
                except Exception:
                    logger.warning(
                        "live watcher parse-stage prefetch: worker failed for %s",
                        candidate.source_path,
                        exc_info=True,
                    )
                    continue
                if error is not None or sessions is None:
                    # Parse failures are intentionally NOT cached: the
                    # writer-held pass reparses (and correctly records) this
                    # file exactly as it would with no prewarm at all.
                    continue
                if self.cache.try_admit(cache_key, sessions, payload=candidate.payload):
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
        self._executor.shutdown(wait=False, cancel_futures=True)


__all__ = [
    "LiveParseCandidate",
    "LiveParsePrefetchCache",
    "LiveParseStage",
    "live_parse_worker",
    "live_watcher_parse_stage_max_inflight_bytes",
    "live_watcher_parse_stage_warm_timeout_seconds",
    "live_watcher_parse_stage_worker_count",
]
