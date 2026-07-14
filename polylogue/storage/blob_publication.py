"""Archive-owned protection for content-addressed blob publication."""

from __future__ import annotations

import fcntl
import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, BinaryIO
from uuid import uuid4

from polylogue.storage.blob_store import BlobStore, Heartbeat, PreparedBlob


@dataclass(frozen=True, slots=True)
class BlobPublicationReceipt:
    """Identity of one publication attempt, independent of content identity."""

    publication_id: str
    blob_hash: str
    size_bytes: int
    publisher_id: str


@dataclass(frozen=True, slots=True)
class ArchiveWriterExclusion:
    """Proof that every archive-owned publisher is excluded for one archive."""

    source_db_path: Path
    _lock_file: IO[bytes]


@dataclass(frozen=True, slots=True)
class BlobPublicationInspection:
    publication_id: str
    blob_hash: str
    size_bytes: int
    publisher_id: str
    reserved_at_ms: int
    blob_present: bool
    referenced: bool


@dataclass(frozen=True, slots=True)
class BlobPublicationReconciliation:
    cleared_referenced: int = 0
    cleared_missing: int = 0
    retained_referenced: int = 0
    retained_missing: int = 0
    unresolved: int = 0


@dataclass(frozen=True, slots=True)
class BlobPublicationAbandonment:
    abandoned: int
    skipped_referenced: int
    missing_receipts: int


def _writer_lock_path(source_db_path: Path) -> Path:
    return source_db_path.with_name(".blob-publication-writers.lock")


@contextmanager
def exclude_archive_blob_publishers(source_db_path: Path) -> Iterator[ArchiveWriterExclusion]:
    """Acquire archive-wide exclusion against instrumented blob publishers."""
    lock_path = _writer_lock_path(source_db_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield ArchiveWriterExclusion(source_db_path.resolve(), lock_file)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _archive_blob_publisher_slot(source_db_path: Path) -> Iterator[None]:
    lock_path = _writer_lock_path(source_db_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True, slots=True)
class BlobPublicationReservationStore:
    """Persist receipt rows in one source-tier transaction per batch."""

    source_db_path: Path

    def reserve_many(self, receipts: Sequence[BlobPublicationReceipt]) -> None:
        if not receipts:
            return
        now_ms = int(time.time() * 1000)
        # Do not apply the general archive connection profile here. Its
        # journal-mode PRAGMA fails immediately when GC owns the source write
        # lock; the publication protocol must instead wait at BEGIN IMMEDIATE.
        conn = sqlite3.connect(self.source_db_path, timeout=30.0)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.executemany(
                """
                INSERT INTO blob_publication_reservations (
                    publication_id, blob_hash, size_bytes, publisher_id, reserved_at_ms
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (
                        receipt.publication_id,
                        bytes.fromhex(receipt.blob_hash),
                        receipt.size_bytes,
                        receipt.publisher_id,
                        now_ms,
                    )
                    for receipt in receipts
                ),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


class ArchiveBlobPublisher(BlobStore):
    """Batch prepare, reserve, then publish blobs for one archive."""

    def __init__(self, source_db_path: Path, blob_root: Path, *, store: BlobStore | None = None) -> None:
        super().__init__(blob_root)
        self.source_db_path = source_db_path
        self.publisher_id = str(uuid4())
        self._store = store or BlobStore(blob_root)
        if self._store.root != blob_root:
            raise ValueError("publisher store root must match blob_root")
        self._pending: list[tuple[BlobPublicationReceipt, PreparedBlob]] = []
        self._latest_receipt_by_hash: dict[str, str] = {}
        self._pending_by_hash: dict[str, PreparedBlob] = {}

    def _queue(self, prepared: PreparedBlob) -> tuple[str, int]:
        receipt = BlobPublicationReceipt(
            publication_id=str(uuid4()),
            blob_hash=prepared.hash_hex,
            size_bytes=prepared.size_bytes,
            publisher_id=self.publisher_id,
        )
        self._pending.append((receipt, prepared))
        self._latest_receipt_by_hash[prepared.hash_hex] = receipt.publication_id
        self._pending_by_hash[prepared.hash_hex] = prepared
        return prepared.hash_hex, prepared.size_bytes

    def write_from_path(self, source: Path, *, heartbeat: Heartbeat | None = None) -> tuple[str, int]:
        return self._queue(self._store.prepare_from_path(source, heartbeat=heartbeat))

    def write_from_fileobj(self, source: IO[bytes], *, heartbeat: Heartbeat | None = None) -> tuple[str, int]:
        return self._queue(self._store.prepare_from_fileobj(source, heartbeat=heartbeat))

    def write_from_bytes(self, data: bytes) -> tuple[str, int]:
        return self._queue(self._store.prepare_from_bytes(data))

    def receipt_id(self, blob_hash: str) -> str | None:
        """Return the receipt for the most recent write of *blob_hash*."""
        return self._latest_receipt_by_hash.get(blob_hash)

    def flush(self) -> tuple[BlobPublicationReceipt, ...]:
        """Commit all receipts once, then expose all corresponding final paths."""
        if not self._pending:
            return ()
        pending = tuple(self._pending)
        receipts = tuple(receipt for receipt, _prepared in pending)
        with _archive_blob_publisher_slot(self.source_db_path):
            BlobPublicationReservationStore(self.source_db_path).reserve_many(receipts)
            self._store.publish_many(prepared for _receipt, prepared in pending)
        self._pending.clear()
        self._pending_by_hash.clear()
        return receipts

    def discard_pending(self) -> None:
        for _receipt, prepared in self._pending:
            self._store.discard_prepared(prepared)
        self._pending.clear()
        self._pending_by_hash.clear()

    def blob_path(self, hash_hex: str) -> Path:
        final_path = self._store.blob_path(hash_hex)
        if final_path.exists():
            return final_path
        prepared = self._pending_by_hash.get(hash_hex)
        return prepared.temporary_path if prepared is not None else final_path

    def exists(self, hash_hex: str) -> bool:
        return self.blob_path(hash_hex).exists()

    def open(self, hash_hex: str) -> BinaryIO:
        return self.blob_path(hash_hex).open("rb")

    def read_prefix(self, hash_hex: str, n: int = 65536) -> bytes:
        with self.open(hash_hex) as handle:
            return handle.read(n)

    def read_all(self, hash_hex: str) -> bytes:
        return self.blob_path(hash_hex).read_bytes()


def publication_receipt_id(blob_store: BlobStore, blob_hash: str) -> str | None:
    """Read an optional receipt without coupling pure source APIs to archives."""
    receipt_getter = getattr(blob_store, "receipt_id", None)
    if not callable(receipt_getter):
        return None
    receipt_id = receipt_getter(blob_hash)
    return str(receipt_id) if receipt_id is not None else None


def flush_blob_publications(blob_store: BlobStore) -> tuple[BlobPublicationReceipt, ...]:
    """Flush an injected archive publisher; plain BlobStore is already final."""
    flush = getattr(blob_store, "flush", None)
    if not callable(flush):
        return ()
    result = flush()
    return tuple(result)


def consume_blob_publication_receipt(
    conn: sqlite3.Connection,
    publication_id: str | None,
    blob_hash: bytes,
) -> None:
    """Consume exactly one publication receipt in its durable-ref transaction."""
    if publication_id is None:
        return
    conn.execute(
        "DELETE FROM blob_publication_reservations WHERE publication_id = ? AND blob_hash = ?",
        (publication_id, blob_hash),
    )


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        is not None
    )


def _is_referenced(
    source_conn: sqlite3.Connection,
    index_conn: sqlite3.Connection | None,
    blob_hash: bytes,
) -> bool:
    for table in ("raw_sessions", "blob_refs"):
        if (
            _table_exists(source_conn, table)
            and source_conn.execute(f"SELECT 1 FROM {table} WHERE blob_hash = ? LIMIT 1", (blob_hash,)).fetchone()
            is not None
        ):
            return True
    return bool(
        index_conn is not None
        and _table_exists(index_conn, "attachments")
        and index_conn.execute("SELECT 1 FROM attachments WHERE blob_hash = ? LIMIT 1", (blob_hash,)).fetchone()
    )


def inspect_blob_publication_receipts(
    source_db_path: Path,
    blob_root: Path,
    *,
    index_db_path: Path | None = None,
) -> tuple[BlobPublicationInspection, ...]:
    """Return every receipt with its current path/reference evidence."""
    from polylogue.paths import sibling_index_db

    source_conn = sqlite3.connect(f"file:{source_db_path}?mode=ro", uri=True)
    source_conn.row_factory = sqlite3.Row
    if index_db_path is not None:
        resolved_index: Path = index_db_path
    else:
        sibling = sibling_index_db(source_db_path, require_exists=False)
        resolved_index = sibling if sibling is not None else source_db_path.with_name("index.db")
    index_conn = sqlite3.connect(f"file:{resolved_index}?mode=ro", uri=True) if resolved_index.exists() else None
    store = BlobStore(blob_root)
    try:
        if not _table_exists(source_conn, "blob_publication_reservations"):
            return ()
        rows = source_conn.execute(
            """
            SELECT publication_id, blob_hash, size_bytes, publisher_id, reserved_at_ms
            FROM blob_publication_reservations
            ORDER BY reserved_at_ms, publication_id
            """
        ).fetchall()
        return tuple(
            BlobPublicationInspection(
                publication_id=str(row["publication_id"]),
                blob_hash=bytes(row["blob_hash"]).hex(),
                size_bytes=int(row["size_bytes"]),
                publisher_id=str(row["publisher_id"]),
                reserved_at_ms=int(row["reserved_at_ms"]),
                blob_present=store.exists(bytes(row["blob_hash"]).hex()),
                referenced=_is_referenced(source_conn, index_conn, bytes(row["blob_hash"])),
            )
            for row in rows
        )
    finally:
        if index_conn is not None:
            index_conn.close()
        source_conn.close()


def reconcile_blob_publication_reservations(
    source_db_path: Path,
    blob_root: Path,
    *,
    index_db_path: Path | None = None,
    writer_exclusion: ArchiveWriterExclusion | None = None,
) -> BlobPublicationReconciliation:
    """Classify receipts; clear safe rows only with archive-wide exclusion."""
    inspections = inspect_blob_publication_receipts(
        source_db_path,
        blob_root,
        index_db_path=index_db_path,
    )
    may_clear = (
        writer_exclusion is not None
        and writer_exclusion.source_db_path == source_db_path.resolve()
        and not writer_exclusion._lock_file.closed
    )
    cleared_referenced = 0
    cleared_missing = 0
    retained_referenced = 0
    retained_missing = 0
    unresolved = 0
    clear_ids: list[str] = []
    for item in inspections:
        if item.referenced:
            if may_clear:
                clear_ids.append(item.publication_id)
                cleared_referenced += 1
            else:
                retained_referenced += 1
        elif not item.blob_present:
            if may_clear:
                clear_ids.append(item.publication_id)
                cleared_missing += 1
            else:
                retained_missing += 1
        else:
            unresolved += 1
    if clear_ids:
        conn = sqlite3.connect(source_db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.executemany(
                "DELETE FROM blob_publication_reservations WHERE publication_id = ?",
                ((publication_id,) for publication_id in clear_ids),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    return BlobPublicationReconciliation(
        cleared_referenced=cleared_referenced,
        cleared_missing=cleared_missing,
        retained_referenced=retained_referenced,
        retained_missing=retained_missing,
        unresolved=unresolved,
    )


def abandon_blob_publication_receipts(
    source_db_path: Path,
    blob_root: Path,
    publication_ids: Sequence[str],
    *,
    confirmed: bool,
    index_db_path: Path | None = None,
) -> BlobPublicationAbandonment:
    """Explicitly abandon selected unreferenced receipts under exclusion."""
    if not confirmed:
        raise ValueError("confirmed=True is required to abandon publication receipts")
    requested = tuple(dict.fromkeys(publication_ids))
    with exclude_archive_blob_publishers(source_db_path):
        by_id = {
            item.publication_id: item
            for item in inspect_blob_publication_receipts(
                source_db_path,
                blob_root,
                index_db_path=index_db_path,
            )
        }
        abandoned: list[str] = []
        skipped_referenced = 0
        for publication_id in requested:
            item = by_id.get(publication_id)
            if item is None:
                continue
            if item.referenced:
                skipped_referenced += 1
                continue
            abandoned.append(publication_id)
        if abandoned:
            conn = sqlite3.connect(source_db_path)
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.executemany(
                    "DELETE FROM blob_publication_reservations WHERE publication_id = ?",
                    ((publication_id,) for publication_id in abandoned),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
    return BlobPublicationAbandonment(
        abandoned=len(abandoned),
        skipped_referenced=skipped_referenced,
        missing_receipts=len(requested) - len(abandoned) - skipped_referenced,
    )


__all__ = [
    "ArchiveBlobPublisher",
    "ArchiveWriterExclusion",
    "BlobPublicationAbandonment",
    "BlobPublicationInspection",
    "BlobPublicationReceipt",
    "BlobPublicationReconciliation",
    "BlobPublicationReservationStore",
    "abandon_blob_publication_receipts",
    "consume_blob_publication_receipt",
    "exclude_archive_blob_publishers",
    "flush_blob_publications",
    "inspect_blob_publication_receipts",
    "publication_receipt_id",
    "reconcile_blob_publication_reservations",
]
