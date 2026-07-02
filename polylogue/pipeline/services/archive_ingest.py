"""source parsing helpers."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from polylogue.config import Source
from polylogue.logging import get_logger
from polylogue.pipeline.services.parsing_models import ParseResult
from polylogue.sources.parsers.base import ParsedSession, RawSessionData
from polylogue.sources.source_parsing import (
    iter_antigravity_language_server_sessions,
    iter_source_sessions_with_raw,
    parse_one_source_path,
)
from polylogue.sources.source_walk import _setup_source_walk
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

logger = get_logger(__name__)

# Work-based commit batching (#dogfood ingest-commit-batching). Re-ingest is
# I/O-wait-bound: source.db and index.db each fsync per session under
# per-session commit. Committing once ~8000 accumulated messages have been
# written amortizes the fsync/WAL churn (validated ~1.37x throughput, ~4x fewer
# bytes, peak WAL ~14 MB << 40 MB autocheckpoint). Session-count batching was
# rejected (uneven transaction size -> larger WAL); one-shot was rejected
# (slower + large WAL). Override with POLYLOGUE_INGEST_COMMIT_BATCH_MESSAGES;
# a value <= 0 restores per-session commit (escape hatch).
COMMIT_BATCH_MESSAGE_THRESHOLD = 8000


def _commit_batch_message_threshold() -> int:
    raw = os.environ.get("POLYLOGUE_INGEST_COMMIT_BATCH_MESSAGES")
    if raw is None:
        return COMMIT_BATCH_MESSAGE_THRESHOLD
    try:
        return int(raw)
    except ValueError:
        return COMMIT_BATCH_MESSAGE_THRESHOLD


def _parse_worker_count() -> int:
    """Resolve the source-parse worker count.

    Parse is ~44% of re-ingest wall time and CPU-bound; the single SQLite
    writer is I/O-bound. Running file parsing across worker processes overlaps
    parse CPU with write I/O. Default = min(8, cpus-1), clamped to >=1. A value
    of 1 disables the pool entirely and preserves the exact sequential behavior
    as an escape hatch.
    """
    raw = os.environ.get("POLYLOGUE_INGEST_PARSE_WORKERS")
    if raw is None:
        return max(1, min(8, (os.cpu_count() or 2) - 1))
    try:
        return max(1, int(raw))
    except ValueError:
        return max(1, min(8, (os.cpu_count() or 2) - 1))


def _parse_source_path_worker(
    path_str: str,
    file_mtime: str | None,
    source_name: str,
    sidecar_data: Any,
    capture_raw: bool,
) -> list[tuple[RawSessionData | None, ParsedSession]]:
    """ProcessPool worker: parse one file and return materialized tuples.

    Materializing into a list is required so the worker can pickle results back
    to the main process. Errors propagate via the future and are caught by the
    driver. ``cursor_state`` is intentionally ``None``: re-ingest does not use
    cursor state, and it could not cross the process boundary anyway.
    """
    return list(
        parse_one_source_path(
            path_str,
            file_mtime=file_mtime,
            source_name=source_name,
            sidecar_data=sidecar_data,
            capture_raw=capture_raw,
            cursor_state=None,
        )
    )


async def parse_sources_archive(archive_root: Path, sources: list[Source]) -> ParseResult:
    """Parse configured sources directly into archive source/index tiers.

    Source files are parsed across a process pool (``POLYLOGUE_INGEST_PARSE_WORKERS``)
    while the main process remains the single SQLite writer. Write order does
    not matter: archive writes are idempotent by content hash and session links
    resolve out-of-order. Blob writes from workers are
    content-addressed and atomic, so concurrent worker writes are process-safe.
    """
    result = ParseResult()
    acquired_at_ms = int(datetime.now(UTC).timestamp() * 1000)
    threshold = _commit_batch_message_threshold()
    batched = threshold > 0
    workers = _parse_worker_count()
    counters = {"raw_rows": 0, "index_rows": 0, "pending_messages": 0}

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:

        async def write_pair(
            source: Source,
            raw_data: RawSessionData | None,
            session: ParsedSession,
        ) -> None:
            payload = _archive_raw_payload(raw_data, session)
            source_path = _archive_raw_source_path(raw_data, source)
            source_index = _archive_raw_source_index(raw_data)
            try:
                _raw_id, session_id = archive.write_raw_and_parsed(
                    session,
                    payload=payload,
                    source_path=source_path,
                    acquired_at_ms=acquired_at_ms,
                    source_index=source_index,
                    stage_timings_s=result.stage_timings_s,
                    manage_transaction=not batched,
                )
            except Exception:
                # Discard the in-flight uncommitted batch so a failed write
                # never leaves prior sessions in this batch half-applied.
                # Re-ingest is restartable from durable source evidence.
                if batched:
                    archive.rollback()
                raise
            counters["raw_rows"] += 1
            counters["index_rows"] += 1
            await result.merge_result(
                session_id,
                {
                    "sessions": 1,
                    "messages": len(session.messages),
                    "attachments": len(session.attachments),
                    "skipped_sessions": 0,
                    "skipped_messages": 0,
                    "skipped_attachments": 0,
                },
                content_changed=True,
            )
            if batched:
                counters["pending_messages"] += len(session.messages)
                if counters["pending_messages"] >= threshold:
                    archive.commit()
                    counters["pending_messages"] = 0

        if workers <= 1:
            # Escape hatch: exact sequential behavior, no pool.
            for source in sources:
                for raw_data, session in iter_source_sessions_with_raw(source, capture_raw=True):
                    await write_pair(source, raw_data, session)
        else:
            # Antigravity language-server export stays sequential (it drives a
            # local loopback subprocess); only the file-walk parallelizes.
            for source in sources:
                for raw_data, session in iter_antigravity_language_server_sessions(source):
                    await write_pair(source, raw_data, session)

            failed = 0
            total_paths = 0
            with ProcessPoolExecutor(max_workers=workers) as pool:
                future_to_source: dict[Any, tuple[Source, Path]] = {}
                for source in sources:
                    walk = _setup_source_walk(
                        source,
                        cursor_state=None,
                        include_mtime=True,
                        known_mtimes=None,
                        discover_sidecars=True,
                    )
                    if walk is None:
                        continue
                    for path, file_mtime in walk.paths_to_process:
                        future = pool.submit(
                            _parse_source_path_worker,
                            str(path),
                            file_mtime,
                            source.name,
                            walk.sidecar_data,
                            True,
                        )
                        future_to_source[future] = (source, path)
                        total_paths += 1

                for future in as_completed(future_to_source):
                    source, path = future_to_source[future]
                    try:
                        pairs = future.result()
                    except Exception as exc:
                        # Worker error isolation: one bad file must not kill the
                        # run. Mirror the sequential iterator's failure handling.
                        failed += 1
                        result.parse_failures += 1
                        logger.error("Failed to parse %s in worker: %s", path, exc)
                        continue
                    for raw_data, session in pairs:
                        await write_pair(source, raw_data, session)

            if failed > 0:
                logger.warning(
                    "Skipped %d of %d files due to parse/read errors during parallel ingest.",
                    failed,
                    total_paths,
                )

        if batched and counters["pending_messages"] > 0:
            archive.commit()

    raw_rows_written = counters["raw_rows"]
    index_rows_written = counters["index_rows"]
    result.batch_observations.append(
        {
            "primary_ingest_store": "archive_file_set",
            "archive_primary_write": True,
            "archive_write_mode": "archive",
            "archive_root": str(archive_root),
            "archive_write_targets": ["source.db", "index.db"],
            "archive_source_rows": raw_rows_written,
            "archive_index_rows": index_rows_written,
            "records": len(result.processed_ids),
            "sessions": result.counts["sessions"],
            "messages": result.counts["messages"],
            "changed_sessions": result.changed_counts["sessions"],
        }
    )
    return result


def _archive_raw_payload(raw_data: object, session: Any) -> bytes:
    from polylogue.storage.blob_store import get_blob_store

    raw_bytes = getattr(raw_data, "raw_bytes", None)
    if isinstance(raw_bytes, bytes) and raw_bytes:
        return raw_bytes
    blob_hash = getattr(raw_data, "blob_hash", None)
    if isinstance(blob_hash, str) and blob_hash:
        return get_blob_store().read_all(blob_hash)
    if callable(getattr(session, "model_dump_json", None)):
        return str(session.model_dump_json()).encode("utf-8")
    return json.dumps(str(session), sort_keys=True).encode("utf-8")


def _archive_raw_source_path(raw_data: object, source: Source) -> str:
    source_path = getattr(raw_data, "source_path", None)
    if source_path is not None:
        return str(source_path)
    return str(source.path)


def _archive_raw_source_index(raw_data: object) -> int:
    source_index = getattr(raw_data, "source_index", None)
    return int(source_index) if source_index is not None else 0


__all__ = ["parse_sources_archive"]
