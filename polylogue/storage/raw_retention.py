"""Retention cleanup for superseded live raw payload snapshots."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from polylogue.storage.blob_store import BlobStore, get_blob_store

_RAW_CANDIDATE_SQL = """
WITH ranked AS (
    SELECT
        raw_id,
        source_path,
        source_index,
        blob_size,
        acquired_at,
        ROW_NUMBER() OVER (
            PARTITION BY source_path, source_index
            ORDER BY acquired_at DESC, raw_id DESC
        ) AS recency
    FROM raw_sessions
    WHERE source_index IN (-1, 0)
      AND (? IS NULL OR source_path = ?)
      AND (? IS NULL OR acquired_at >= ?)
)
SELECT raw_id, source_path, source_index, blob_size
FROM ranked AS r
WHERE r.recency > CASE WHEN r.source_index = 0 THEN ? ELSE ? END
  AND NOT EXISTS (
      SELECT 1 FROM sessions AS c WHERE c.raw_id = r.raw_id
  )
  AND NOT EXISTS (
      SELECT 1 FROM artifact_observations AS ao WHERE ao.raw_id = r.raw_id
  )
  AND NOT EXISTS (
      SELECT 1 FROM pending_blob_refs AS p WHERE p.blob_hash = r.raw_id
  )
ORDER BY r.blob_size DESC, r.acquired_at ASC, r.raw_id ASC
LIMIT ?
"""

_V1_RAW_CANDIDATE_SQL = """
WITH ranked AS (
    SELECT
        raw_id,
        source_path,
        source_index,
        blob_hash,
        blob_size,
        acquired_at_ms,
        ROW_NUMBER() OVER (
            PARTITION BY source_path, source_index
            ORDER BY acquired_at_ms DESC, raw_id DESC
        ) AS recency
    FROM raw_sessions
    WHERE source_index IN (-1, 0)
      AND (? IS NULL OR source_path = ?)
      AND (? IS NULL OR acquired_at_ms >= ?)
)
SELECT raw_id, source_path, source_index, blob_hash, blob_size
FROM ranked AS r
WHERE r.recency > CASE WHEN r.source_index = 0 THEN ? ELSE ? END
ORDER BY r.blob_size DESC, r.acquired_at_ms ASC, r.raw_id ASC
LIMIT ?
"""


@dataclass(frozen=True)
class RawSnapshotCleanupCandidate:
    raw_id: str
    source_path: str
    source_index: int
    blob_size: int
    blob_hash: str | None = None

    @property
    def blob_store_hash(self) -> str:
        return self.blob_hash or self.raw_id


@dataclass(frozen=True)
class RawSnapshotCleanupResult:
    candidate_count: int
    deleted_raw_count: int
    deleted_blob_count: int
    deleted_raw_bytes: int
    deleted_blob_bytes: int
    skipped_missing_source_count: int
    errors: tuple[str, ...] = ()


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = ? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _blob_hash_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.hex() if len(value) == 32 else None
    text = str(value)
    return text if text else None


def _timestamp_ms(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    return int(datetime.fromisoformat(value).timestamp() * 1000)


def _superseded_archive_raw_session_candidates(
    conn: sqlite3.Connection,
    *,
    source_path: Path | None,
    keep_full_snapshots: int,
    keep_append_snapshots: int,
    min_acquired_at: str | None,
    limit: int,
) -> list[RawSnapshotCleanupCandidate]:
    source_path_str = str(source_path) if source_path is not None else None
    min_acquired_at_ms = _timestamp_ms(min_acquired_at)
    rows = conn.execute(
        _V1_RAW_CANDIDATE_SQL,
        (
            source_path_str,
            source_path_str,
            min_acquired_at_ms,
            min_acquired_at_ms,
            max(1, keep_full_snapshots),
            max(1, keep_append_snapshots),
            limit,
        ),
    ).fetchall()
    candidates: list[RawSnapshotCleanupCandidate] = []
    for row in rows:
        row_source_path = str(row[1])
        if not Path(row_source_path).exists():
            continue
        blob_hash = _blob_hash_text(row[3])
        if blob_hash is None:
            continue
        candidates.append(
            RawSnapshotCleanupCandidate(
                raw_id=str(row[0]),
                source_path=row_source_path,
                source_index=int(row[2] or 0),
                blob_size=int(row[4] or 0),
                blob_hash=blob_hash,
            )
        )
    return candidates


def superseded_raw_snapshot_candidates(
    conn: sqlite3.Connection,
    *,
    source_path: Path | None = None,
    keep_full_snapshots: int = 1,
    keep_append_snapshots: int = 1,
    min_acquired_at: str | None = None,
    limit: int = 1_000,
) -> list[RawSnapshotCleanupCandidate]:
    """Return redundant live raw snapshots that are safe to compact.

    The source file must still exist on disk before a row is returned. If
    the source disappeared, the raw blob may be the only remaining copy and
    is deliberately preserved.
    """
    if limit <= 0:
        return []

    if not _table_exists(conn, "raw_sessions"):
        return []
    return _superseded_archive_raw_session_candidates(
        conn,
        source_path=source_path,
        keep_full_snapshots=keep_full_snapshots,
        keep_append_snapshots=keep_append_snapshots,
        min_acquired_at=min_acquired_at,
        limit=limit,
    )


def cleanup_superseded_raw_snapshots(
    conn: sqlite3.Connection,
    *,
    source_path: Path | None = None,
    keep_full_snapshots: int = 1,
    keep_append_snapshots: int = 1,
    min_acquired_at: str | None = None,
    limit: int = 1_000,
    dry_run: bool = True,
    blob_store: BlobStore | None = None,
) -> RawSnapshotCleanupResult:
    candidates = superseded_raw_snapshot_candidates(
        conn,
        source_path=source_path,
        keep_full_snapshots=keep_full_snapshots,
        keep_append_snapshots=keep_append_snapshots,
        min_acquired_at=min_acquired_at,
        limit=limit,
    )
    if not candidates:
        return RawSnapshotCleanupResult(
            candidate_count=0,
            deleted_raw_count=0,
            deleted_blob_count=0,
            deleted_raw_bytes=0,
            deleted_blob_bytes=0,
            skipped_missing_source_count=0,
        )

    raw_ids = [candidate.raw_id for candidate in candidates]
    raw_bytes = sum(candidate.blob_size for candidate in candidates)
    if dry_run:
        return RawSnapshotCleanupResult(
            candidate_count=len(candidates),
            deleted_raw_count=0,
            deleted_blob_count=0,
            deleted_raw_bytes=raw_bytes,
            deleted_blob_bytes=0,
            skipped_missing_source_count=0,
        )

    placeholders = ", ".join("?" for _ in raw_ids)
    conn.execute(f"DELETE FROM raw_sessions WHERE raw_id IN ({placeholders})", raw_ids)
    conn.commit()

    store = blob_store if blob_store is not None else get_blob_store()
    deleted_blob_count = 0
    deleted_blob_bytes = 0
    errors: list[str] = []
    for candidate in candidates:
        try:
            path = store.blob_path(candidate.blob_store_hash)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not path.exists():
            continue
        with suppress(OSError):
            deleted_blob_bytes += path.stat().st_size
        try:
            path.unlink()
            deleted_blob_count += 1
        except OSError as exc:
            errors.append(f"{candidate.raw_id[:16]}: {exc}")

    return RawSnapshotCleanupResult(
        candidate_count=len(candidates),
        deleted_raw_count=len(candidates),
        deleted_blob_count=deleted_blob_count,
        deleted_raw_bytes=raw_bytes,
        deleted_blob_bytes=deleted_blob_bytes,
        skipped_missing_source_count=0,
        errors=tuple(errors),
    )


def compact_paths_superseded_raw_snapshots(
    conn: sqlite3.Connection,
    source_paths: Iterable[Path],
    *,
    limit_per_path: int = 25,
    min_acquired_at: str | None = None,
    dry_run: bool = False,
) -> RawSnapshotCleanupResult:
    totals = RawSnapshotCleanupResult(
        candidate_count=0,
        deleted_raw_count=0,
        deleted_blob_count=0,
        deleted_raw_bytes=0,
        deleted_blob_bytes=0,
        skipped_missing_source_count=0,
    )
    errors: list[str] = []
    for path in source_paths:
        result = cleanup_superseded_raw_snapshots(
            conn,
            source_path=path,
            keep_full_snapshots=1_000_000,
            min_acquired_at=min_acquired_at,
            limit=limit_per_path,
            dry_run=dry_run,
        )
        errors.extend(result.errors)
        totals = RawSnapshotCleanupResult(
            candidate_count=totals.candidate_count + result.candidate_count,
            deleted_raw_count=totals.deleted_raw_count + result.deleted_raw_count,
            deleted_blob_count=totals.deleted_blob_count + result.deleted_blob_count,
            deleted_raw_bytes=totals.deleted_raw_bytes + result.deleted_raw_bytes,
            deleted_blob_bytes=totals.deleted_blob_bytes + result.deleted_blob_bytes,
            skipped_missing_source_count=totals.skipped_missing_source_count + result.skipped_missing_source_count,
            errors=tuple(errors),
        )
    return totals


__all__ = [
    "RawSnapshotCleanupCandidate",
    "RawSnapshotCleanupResult",
    "cleanup_superseded_raw_snapshots",
    "compact_paths_superseded_raw_snapshots",
    "superseded_raw_snapshot_candidates",
]
