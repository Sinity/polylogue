"""Durable reservations for content-addressed blob publication."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from polylogue.storage.blob_store import BlobStore


@dataclass(frozen=True, slots=True)
class BlobPublicationReservationStore:
    """Source-tier reservation writer injected into a substrate-neutral store."""

    source_db_path: Path
    publisher_id: str

    @classmethod
    def create(cls, source_db_path: Path) -> BlobPublicationReservationStore:
        return cls(source_db_path=source_db_path, publisher_id=str(uuid4()))

    def reserve(self, blob_hash: str, _size_bytes: int) -> None:
        """Commit protection before the blob is visible at its final path."""
        from polylogue.storage.sqlite.connection_profile import open_connection

        now_ms = int(time.time() * 1000)
        conn = open_connection(self.source_db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO blob_publication_reservations (
                    blob_hash, publisher_id, reserved_at_ms, refreshed_at_ms
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(blob_hash) DO UPDATE SET
                    publisher_id = excluded.publisher_id,
                    refreshed_at_ms = excluded.refreshed_at_ms
                """,
                (bytes.fromhex(blob_hash), self.publisher_id, now_ms, now_ms),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


@dataclass(frozen=True, slots=True)
class BlobPublicationReconciliation:
    """Crash-recovery classification for durable publication reservations."""

    cleared_referenced: int
    cleared_missing: int
    unresolved: int


def reconcile_blob_publication_reservations(
    source_db_path: Path,
    blob_root: Path,
    *,
    index_db_path: Path | None = None,
) -> BlobPublicationReconciliation:
    """Clear provably redundant reservations and retain ambiguous debt.

    There is deliberately no age-based expiry. A blob with no committed
    reference is retained regardless of age because a publisher may still be
    alive; source reacquisition or explicit operator adjudication resolves it.
    """
    from polylogue.storage.sqlite.connection_profile import open_connection

    conn = open_connection(source_db_path)
    resolved_index_db = index_db_path or source_db_path.with_name("index.db")
    index_conn: sqlite3.Connection | None = None
    cleared_referenced = 0
    cleared_missing = 0
    unresolved = 0
    try:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute("SELECT blob_hash FROM blob_publication_reservations ORDER BY blob_hash").fetchall()
        if resolved_index_db.exists():
            index_conn = sqlite3.connect(f"file:{resolved_index_db}?mode=ro", uri=True)
        store = BlobStore(blob_root)
        for row in rows:
            blob_hash_bytes = bytes(row[0])
            referenced = conn.execute(
                """
                SELECT 1 FROM raw_sessions WHERE blob_hash = ?
                UNION ALL
                SELECT 1 FROM blob_refs WHERE blob_hash = ?
                LIMIT 1
                """,
                (blob_hash_bytes, blob_hash_bytes),
            ).fetchone()
            if referenced is None and index_conn is not None:
                has_attachments = index_conn.execute(
                    "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'attachments'"
                ).fetchone()
                if has_attachments is not None:
                    referenced = index_conn.execute(
                        "SELECT 1 FROM attachments WHERE blob_hash = ? LIMIT 1",
                        (blob_hash_bytes,),
                    ).fetchone()
            if referenced is not None:
                conn.execute(
                    "DELETE FROM blob_publication_reservations WHERE blob_hash = ?",
                    (blob_hash_bytes,),
                )
                cleared_referenced += 1
                continue
            if not store.exists(blob_hash_bytes.hex()):
                conn.execute(
                    "DELETE FROM blob_publication_reservations WHERE blob_hash = ?",
                    (blob_hash_bytes,),
                )
                cleared_missing += 1
                continue
            unresolved += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        if index_conn is not None:
            index_conn.close()
        conn.close()
    return BlobPublicationReconciliation(
        cleared_referenced=cleared_referenced,
        cleared_missing=cleared_missing,
        unresolved=unresolved,
    )


def reserved_blob_store(blob_root: Path, *, source_db_path: Path | None = None) -> BlobStore:
    """Build an archive-owned BlobStore whose publications are reserved."""
    resolved_source_db = source_db_path or blob_root.parent / "source.db"
    reservations = BlobPublicationReservationStore.create(resolved_source_db)
    return BlobStore(blob_root, before_publish=reservations.reserve)


def consume_blob_publication_reservation(conn: sqlite3.Connection, blob_hash: bytes) -> None:
    """Consume a reservation inside the transaction adding its durable ref."""
    conn.execute(
        "DELETE FROM blob_publication_reservations WHERE blob_hash = ?",
        (blob_hash,),
    )


__all__ = [
    "BlobPublicationReservationStore",
    "BlobPublicationReconciliation",
    "consume_blob_publication_reservation",
    "reconcile_blob_publication_reservations",
    "reserved_blob_store",
]
