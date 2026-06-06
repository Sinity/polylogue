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
    from polylogue.storage.blob_store import get_blob_store
    from polylogue.storage.sqlite.connection import open_read_connection

    blob_store = get_blob_store()
    with open_read_connection(config.db_path) as conn:
        referenced_hashes, surfaces = _referenced_blob_hashes(conn, db_path=config.db_path)

    surface_detail = _surface_detail(surfaces)
    detect_result = blob_store.detect_orphans(referenced_hashes)
    if detect_result.orphan_count == 0:
        return BlobRepairOutcome(0, True, f"No orphaned blobs found ({surface_detail})")

    orphan_hashes = {h for h in blob_store.iter_all() if h not in referenced_hashes}
    cleanup_result = blob_store.cleanup_orphans(orphan_hashes, dry_run=dry_run)
    if dry_run:
        return BlobRepairOutcome(
            cleanup_result.would_delete_count,
            True,
            (
                f"Would: delete {cleanup_result.would_delete_count} orphaned blobs "
                f"({cleanup_result.would_delete_bytes:,} bytes; {surface_detail})"
            ),
        )
    return BlobRepairOutcome(
        cleanup_result.deleted_count,
        cleanup_result.errors == 0,
        (
            f"Deleted {cleanup_result.deleted_count} orphaned blobs ({cleanup_result.deleted_bytes:,} bytes)"
            f" ({surface_detail})" + (f" with {cleanup_result.errors} errors" if cleanup_result.errors else "")
        ),
    )


__all__ = ["BlobRepairOutcome", "count_orphaned_blobs_sync", "repair_orphaned_blobs_data"]
