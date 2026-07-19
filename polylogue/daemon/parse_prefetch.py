"""Daemon-owned parse-stage extraction: parse census candidates off the writer hold.

polylogue-m6tp phase (a). The raw-materialization conveyor's writer hold
(``DaemonWriteCoordinator.run_sync``) used to cover BOTH the CPU-bound
blob->``ParsedSession`` decode (census parse) and the SQLite writes that
record it, so a large or slow parse extended the writer hold by exactly as
long as the parse took -- starving every other write actor (live ingest,
status snapshots, insight convergence) queued behind the same coordinator.

This module lets the daemon pre-parse the NEXT pass's candidate raws in a
bounded ``ThreadPoolExecutor`` BEFORE the writer hold is ever requested. The
writer-held pass then finds those results already warmed in a
``RawParsePrefetchCache`` (``polylogue.sources.revision_backfill``) and skips
reparsing them -- see that class's docstring for why a miss (empty cache,
budget-rejected entry, or the flag simply being off) always degrades to the
exact unmodified parse path rather than incorrect behavior.

Why threads are safe here even on a standard (GIL) build: the polylogue-7mtf
control-run measurement (``parallel_threads_effective`` in
``polylogue.pipeline.services.process_pool``) found threaded parse gives no
GIL-build speedup AND inflates a *concurrently write-holding* thread's commit
latency ~5000x. That hazard is specifically about a parse thread running
WHILE a writer thread is active. This module never does that: ``warm()`` is
called by the conveyor BEFORE it ever asks the write coordinator for the
writer hold, so there is no writer thread to contend with. On a GIL build
this still gives little or no wall-clock parse speedup (CPython serializes
the CPU-bound decode across threads) -- that is expected and is the point of
phase (a): prove the parse/apply seam is correct and equivalence-safe ahead
of the free-threaded 3.14t deploy (phase (b), polylogue-m6tp), which is what
turns the same code path into a real speedup.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from polylogue.config import Config
from polylogue.logging import get_logger
from polylogue.sources import revision_backfill
from polylogue.sources.dispatch import is_stream_record_provider
from polylogue.sources.revision_backfill import RawParsePrefetchCache
from polylogue.storage.repair import (
    raw_materialization_pending_census_raw_ids,
    raw_materialization_readonly_descriptors,
)

logger = get_logger(__name__)

_DEFAULT_MAX_INFLIGHT_BYTES = 64 * 1024 * 1024  # 64 MiB


def daemon_parse_stage_worker_count() -> int:
    """Bounded worker cap for the daemon-owned pre-parse thread pool.

    ``cpu_count - 1`` leaves one core free for the daemon's own event loop,
    mirroring ``resolve_parse_worker_count``'s cpu-1 convention (see
    ``polylogue.pipeline.services.process_pool``). Override with
    ``POLYLOGUE_DAEMON_PARSE_STAGE_WORKERS``.
    """
    raw = os.environ.get("POLYLOGUE_DAEMON_PARSE_STAGE_WORKERS")
    if raw is not None:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
    return max(1, (os.cpu_count() or 2) - 1)


def daemon_parse_stage_max_inflight_bytes() -> int:
    """Whale-memory budget for parsed sessions held in the prefetch cache.

    Override with ``POLYLOGUE_DAEMON_PARSE_STAGE_MAX_INFLIGHT_BYTES``.
    """
    raw = os.environ.get("POLYLOGUE_DAEMON_PARSE_STAGE_MAX_INFLIGHT_BYTES")
    if raw is not None:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
    return _DEFAULT_MAX_INFLIGHT_BYTES


class DaemonParseStage:
    """Owns the daemon's bounded pre-parse ``ThreadPoolExecutor`` and cache.

    One instance lives for the daemon process's lifetime (created lazily by
    the raw-materialization conveyor loop when ``daemon_parse_stage_split``
    is enabled). ``warm`` is synchronous/blocking -- callers run it off the
    event loop (``asyncio.to_thread``), exactly like every other conveyor
    pass, and NEVER under ``daemon_write_coordinator().run_sync``: doing so
    would defeat the entire point, since the pre-parse must run without the
    writer hold held.
    """

    def __init__(self, *, max_workers: int | None = None, max_inflight_bytes: int | None = None) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers if max_workers is not None else daemon_parse_stage_worker_count(),
            thread_name_prefix="polylogue-parse-stage",
        )
        self.cache = RawParsePrefetchCache(
            max_inflight_bytes=(
                max_inflight_bytes if max_inflight_bytes is not None else daemon_parse_stage_max_inflight_bytes()
            )
        )

    def warm(self, config: Config, *, limit: int, max_payload_bytes: int) -> int:
        """Pre-parse up to ``limit`` pending census candidates outside any writer hold.

        Returns the number of raws newly admitted to the cache. Read-only
        end to end: candidate discovery and descriptor lookup both open
        ``mode=ro`` SQLite connections (``polylogue.storage.repair``);
        parsing reads only already-published blob bytes via a stateless
        ``ArchiveBlobPublisher``, mirroring the production census parse
        worker exactly (``census_parse_worker``, the same function the
        writer-held path dispatches to a process/thread pool). Nothing here
        writes to source.db, index.db, or takes the daemon's writer lease.
        """
        candidate_raw_ids = raw_materialization_pending_census_raw_ids(
            config, limit=limit, max_payload_bytes=max_payload_bytes
        )
        raw_ids = [raw_id for raw_id in candidate_raw_ids if not self.cache.contains(raw_id)]
        if not raw_ids:
            return 0
        archive_root = config.archive_root
        descriptors = raw_materialization_readonly_descriptors(archive_root, raw_ids)
        blob_root_str = str(archive_root / "blob")
        source_db_path_str = str(archive_root / "source.db")

        futures = {}
        for raw_id in raw_ids:
            descriptor = descriptors.get(raw_id)
            if descriptor is None:
                continue
            provider, blob_hash, source_path, _kind, _size = descriptor
            future = self._executor.submit(
                revision_backfill.census_parse_worker,
                raw_id,
                provider.value,
                blob_hash,
                source_path,
                is_stream_record_provider(source_path, str(provider)),
                blob_root_str,
                source_db_path_str,
            )
            futures[future] = raw_id

        warmed = 0
        for future in as_completed(futures):
            raw_id = futures[future]
            try:
                _raw_id, sessions, error = future.result()
            except Exception:
                logger.warning("parse-stage prefetch: worker failed for raw_id=%s", raw_id, exc_info=True)
                continue
            if error is not None or sessions is None:
                # Parse failures are intentionally NOT cached: the writer-held
                # pass reparses (and correctly quarantines/records) this raw
                # exactly as it would with the flag off. Prefetch only ever
                # shortcuts the happy path.
                continue
            _provider, _blob_hash, _source_path, kind, payload_size = descriptors[raw_id]
            if self.cache.try_admit(raw_id, sessions, payload_bytes=payload_size, revision_kind=kind):
                warmed += 1
        return warmed

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


__all__ = [
    "DaemonParseStage",
    "daemon_parse_stage_max_inflight_bytes",
    "daemon_parse_stage_worker_count",
]
