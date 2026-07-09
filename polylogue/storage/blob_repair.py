"""Blob cleanup helpers for archive repair surfaces."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from polylogue.config import Config


@dataclass(frozen=True)
class BlobRepairOutcome:
    repaired_count: int
    success: bool
    detail: str


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


def _archive_blob_hashes(conn: sqlite3.Connection, *, surface_prefix: str) -> tuple[set[str], list[str]]:
    hashes: set[str] = set()
    surfaces: list[str] = []
    for table in ("raw_sessions", "blob_refs"):
        if not _table_exists(conn, table):
            continue
        columns = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}
        if "blob_hash" not in columns:
            continue
        rows = conn.execute(f"SELECT blob_hash FROM {table}").fetchall()
        table_hashes = {hash_text for row in rows if (hash_text := _blob_hash_text(row[0])) is not None}
        if table_hashes:
            surfaces.append(f"{surface_prefix}.{table}")
            hashes.update(table_hashes)
    return hashes, surfaces


def _referenced_blob_hashes(
    conn: sqlite3.Connection,
    *,
    db_path: Path | None = None,
) -> tuple[set[str], list[str]]:
    hashes, surfaces = _archive_blob_hashes(conn, surface_prefix="current")

    source_db_path = db_path.with_name("source.db") if db_path is not None else None
    if source_db_path is not None and source_db_path != db_path and source_db_path.exists():
        try:
            source_conn = sqlite3.connect(f"file:{source_db_path}?mode=ro", uri=True)
            try:
                source_hashes, source_surfaces = _archive_blob_hashes(
                    source_conn,
                    surface_prefix=source_db_path.name,
                )
                hashes.update(source_hashes)
                surfaces.extend(source_surfaces)
            finally:
                source_conn.close()
        except sqlite3.Error:
            pass

    return hashes, surfaces


def _surface_detail(surfaces: list[str]) -> str:
    if not surfaces:
        return "references: none"
    return "references: " + ", ".join(sorted(dict.fromkeys(surfaces)))


def count_orphaned_blobs_sync(conn: sqlite3.Connection, *, db_path: Path | str | None = None) -> int:
    from polylogue.storage.blob_store import get_blob_store

    referenced_hashes, _surfaces = _referenced_blob_hashes(conn, db_path=Path(db_path) if db_path else None)
    return get_blob_store().detect_orphans(referenced_hashes).orphan_count


def repair_orphaned_blobs_data(config: Config, dry_run: bool = False) -> BlobRepairOutcome:
    """Delete orphaned blobs via the reference/generation-safe GC planner.

    Previously this called ``BlobStore.detect_orphans``/``cleanup_orphans``
    directly, comparing disk hashes only against committed ``raw_sessions``/
    ``blob_refs`` rows, with no generation-age floor. ``run_blob_gc_report``
    already applies both safety invariants that still apply (committed
    reference, generation-age floor); routing the doctor repair path
    through it avoids duplicating that safety logic. (A lease-based third
    invariant was removed in polylogue-v7e0 — it was never reachable from
    any production ingest path; see ``docs/internals.md`` "GC concurrency
    model".)
    """
    from polylogue.paths import blob_store_root
    from polylogue.storage.blob_gc import run_blob_gc_report

    report = run_blob_gc_report(
        config.db_path,
        blob_store_root(),
        max_batch=100_000,
        dry_run=dry_run,
    )
    protected_detail = f"; skipped {report.skipped_referenced} referenced" if report.skipped_referenced else ""
    if dry_run:
        return BlobRepairOutcome(
            report.would_delete_count,
            True,
            f"Would: delete {report.would_delete_count} orphaned blobs{protected_detail}",
        )
    errors = report.skipped_unlink_error
    return BlobRepairOutcome(
        report.deleted_count,
        errors == 0,
        (
            f"Deleted {report.deleted_count} orphaned blobs ({report.reclaimed_bytes:,} bytes)"
            f"{protected_detail}" + (f" with {errors} errors" if errors else "")
        ),
    )


__all__ = ["BlobRepairOutcome", "count_orphaned_blobs_sync", "repair_orphaned_blobs_data"]
