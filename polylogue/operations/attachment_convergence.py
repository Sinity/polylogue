"""Daemon-owned convergence for provider-hosted attachment bytes.

Attachment acquisition used to happen only while the Drive folder iterator
was reading a payload.  This owner works from the canonical attachment rows
instead, so a payload restored from a ZIP (or any other route) is still
eligible for the same live Drive fetch.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from polylogue.daemon.convergence import ConvergenceStage, StageExecuteReturn
from polylogue.logging import get_logger
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveSourceBlobRef, write_source_blob_refs

logger = get_logger(__name__)

DEFAULT_ATTACHMENT_CONVERGENCE_LIMIT = 25
DEFAULT_MAX_ATTACHMENT_BYTES = 50 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class AttachmentConvergenceResult:
    inspected: int = 0
    acquired: int = 0
    terminal: int = 0
    deferred: int = 0

    @property
    def complete(self) -> bool:
        return self.deferred == 0


def _candidate_rows(conn: sqlite3.Connection, *, limit: int) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return list(
        conn.execute(
            """
            SELECT a.attachment_id, r.ref_id, r.session_id, r.upload_origin,
                   r.source_url, s.raw_id,
                   COALESCE(
                       (SELECT native_id FROM attachment_native_ids
                        WHERE ref_id = r.ref_id AND id_kind = 'file'
                        ORDER BY native_id LIMIT 1),
                       (SELECT native_id FROM attachment_native_ids
                        WHERE ref_id = r.ref_id AND id_kind = 'drive'
                        ORDER BY native_id LIMIT 1),
                       (SELECT native_id FROM attachment_native_ids
                        WHERE ref_id = r.ref_id AND id_kind = 'attachment'
                        ORDER BY native_id LIMIT 1)
                   ) AS provider_file_id
            FROM attachments AS a
            JOIN attachment_refs AS r ON r.attachment_id = a.attachment_id
            JOIN sessions AS s ON s.session_id = r.session_id
            WHERE a.acquisition_status = 'unfetched'
              AND r.upload_origin = 'drive'
            ORDER BY a.attachment_id, r.ref_id
            LIMIT ?
            """,
            (max(0, int(limit)),),
        ).fetchall()
    )


def _permanent_failure(exc: BaseException) -> bool:
    """Classify only explicit absence/access failures as terminal.

    Transport/auth failures remain retryable.  The Drive client deliberately
    exposes ``DriveNotFoundError`` for the permanent 404 case; importing the
    type lazily keeps this module usable with small test doubles.
    """
    from polylogue.sources.drive.types import DriveNotFoundError

    if isinstance(exc, (DriveNotFoundError, PermissionError)):
        return True
    # The gateway preserves provider HTTP errors for callers that need the
    # response metadata.  Treat only explicit absence/access responses as
    # terminal; rate limits and transport failures must retain convergence
    # debt for a later pass.
    status = getattr(getattr(exc, "resp", None), "status", None)
    return status in {403, 404}


def converge_drive_attachments(
    index_conn: sqlite3.Connection,
    source_conn: sqlite3.Connection,
    *,
    archive_root: Path,
    download_bytes: Callable[[str], bytes],
    limit: int = DEFAULT_ATTACHMENT_CONVERGENCE_LIMIT,
    max_attachment_bytes: int = DEFAULT_MAX_ATTACHMENT_BYTES,
    now_ms: Callable[[], int] | None = None,
) -> AttachmentConvergenceResult:
    """Fetch one bounded window and publish durable attachment state.

    The query is intentionally route-neutral: it starts at indexed attachment
    references, not at ``iter_drive_raw_data`` or a source path.  Successful
    bytes are hashed from the bytes actually read and published before the
    index row is marked acquired.  Oversize and explicit not-found results are
    terminal ``unavailable`` rows; all other failures remain retryable.
    """
    rows = _candidate_rows(index_conn, limit=limit)
    if not rows:
        return AttachmentConvergenceResult()
    publisher = ArchiveBlobPublisher(archive_root / "source.db", archive_root / "blob")
    acquired_refs: list[ArchiveSourceBlobRef] = []
    acquired_rows: list[tuple[str, bytes, int]] = []
    terminal_ids: list[str] = []
    deferred = 0
    # One content-addressed attachment can have refs in several sessions.  A
    # bounded pass must not spend one Drive request per ref; retain only the
    # fetch outcome (never the payload) and still emit one source ref per raw.
    fetch_outcomes: dict[str, tuple[str, bytes | None, int]] = {}
    observed_at_ms = now_ms() if now_ms is not None else int(time.time() * 1000)

    try:
        for row in rows:
            attachment_id = str(row["attachment_id"])
            provider_file_id = row["provider_file_id"]
            if not isinstance(provider_file_id, str) or not provider_file_id:
                terminal_ids.append(attachment_id)
                continue
            cached = fetch_outcomes.get(provider_file_id)
            if cached is not None:
                outcome, cached_hash, cached_size = cached
                if outcome == "terminal":
                    terminal_ids.append(attachment_id)
                    continue
                if outcome == "deferred":
                    deferred += 1
                    continue
                assert outcome == "acquired" and cached_hash is not None
                blob_hash = cached_hash
                byte_count = cached_size
                acquired_rows.append((attachment_id, blob_hash, byte_count))
                acquired_refs.append(
                    ArchiveSourceBlobRef(
                        blob_hash=blob_hash,
                        raw_id=str(row["raw_id"]),
                        ref_type="attachment",
                        source_path=str(row["source_url"] or "attachment-convergence"),
                        size_bytes=byte_count,
                        acquired_at_ms=observed_at_ms,
                        publication_receipt_id=publisher.receipt_id(blob_hash.hex()),
                    )
                )
                continue
            try:
                payload = download_bytes(provider_file_id)
                if len(payload) > max_attachment_bytes:
                    fetch_outcomes[provider_file_id] = ("terminal", None, 0)
                    terminal_ids.append(attachment_id)
                    continue
                blob_hash_hex, byte_count = publisher.write_from_bytes(payload)
            except Exception as exc:
                if _permanent_failure(exc):
                    fetch_outcomes[provider_file_id] = ("terminal", None, 0)
                    terminal_ids.append(attachment_id)
                else:
                    fetch_outcomes[provider_file_id] = ("deferred", None, 0)
                    deferred += 1
                    logger.info("attachment convergence deferred %s: %s", attachment_id, exc)
                continue

            blob_hash = bytes.fromhex(blob_hash_hex)
            assert hashlib.sha256(payload).digest() == blob_hash
            fetch_outcomes[provider_file_id] = ("acquired", blob_hash, byte_count)
            acquired_rows.append((attachment_id, blob_hash, byte_count))
            acquired_refs.append(
                ArchiveSourceBlobRef(
                    blob_hash=blob_hash,
                    raw_id=str(row["raw_id"]),
                    ref_type="attachment",
                    source_path=str(row["source_url"] or "attachment-convergence"),
                    size_bytes=byte_count,
                    acquired_at_ms=observed_at_ms,
                    publication_receipt_id=publisher.receipt_id(blob_hash_hex),
                )
            )

        publisher.flush()
        if acquired_refs:
            by_raw_id: dict[str, list[ArchiveSourceBlobRef]] = {}
            for ref in acquired_refs:
                by_raw_id.setdefault(str(ref.raw_id), []).append(ref)
            for raw_id, refs in by_raw_id.items():
                write_source_blob_refs(source_conn, raw_id, tuple(refs))
            with index_conn:
                for attachment_id, blob_hash, byte_count in acquired_rows:
                    index_conn.execute(
                        """
                        UPDATE attachments
                        SET blob_hash = ?, byte_count = ?, acquisition_status = 'acquired'
                        WHERE attachment_id = ? AND acquisition_status = 'unfetched'
                        """,
                        (blob_hash, byte_count, attachment_id),
                    )
        if terminal_ids:
            with index_conn:
                index_conn.executemany(
                    "UPDATE attachments SET acquisition_status = 'unavailable' WHERE attachment_id = ? AND acquisition_status = 'unfetched'",
                    ((attachment_id,) for attachment_id in terminal_ids),
                )
        if _candidate_rows(index_conn, limit=1):
            # The scheduler records a false result as convergence debt.  Count
            # the remaining canonical rows as deferred even when this window
            # itself had no transport failure.
            deferred += 1
    finally:
        publisher.discard_pending()

    return AttachmentConvergenceResult(
        inspected=len(rows),
        acquired=len(acquired_rows),
        terminal=len(terminal_ids),
        deferred=deferred,
    )


def make_attachment_convergence_stage(
    db_path: Path,
    *,
    archive_root: Path,
    client_factory: Callable[[], object],
    limit: int = DEFAULT_ATTACHMENT_CONVERGENCE_LIMIT,
) -> ConvergenceStage:
    """Build the bounded whole-archive stage used by daemon convergence."""

    def _open() -> tuple[sqlite3.Connection, sqlite3.Connection]:
        from polylogue.storage.sqlite.connection_profile import open_daemon_connection

        index = open_daemon_connection(db_path, archive_root=archive_root)
        source = open_daemon_connection(archive_root / "source.db", archive_root=archive_root)
        return index, source

    def _has_work() -> bool:
        if not db_path.exists():
            return False
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            return bool(
                conn.execute(
                    """
                    SELECT 1 FROM attachments a JOIN attachment_refs r ON r.attachment_id = a.attachment_id
                    WHERE a.acquisition_status = 'unfetched' AND r.upload_origin = 'drive' LIMIT 1
                    """
                ).fetchone()
            )
        finally:
            conn.close()

    def check(_path: Path) -> bool:
        return _has_work()

    def execute(_path: Path) -> StageExecuteReturn:
        index, source = _open()
        try:
            client = client_factory()
            result = converge_drive_attachments(
                index,
                source,
                archive_root=archive_root,
                download_bytes=client.download_bytes,  # type: ignore[attr-defined]
                limit=limit,
            )
            return result.complete
        finally:
            source.close()
            index.close()

    return ConvergenceStage(
        name="attachment_bytes",
        description="Backfill provider-hosted attachment bytes from canonical references",
        check=check,
        execute=execute,
        false_means_pending=True,
        whole_archive=True,
    )


def make_configured_attachment_convergence_stage(db_path: Path, *, archive_root: Path) -> ConvergenceStage:
    """Construct the daemon stage with the runtime's configured Drive client."""
    from polylogue.config import resolve_runtime_config
    from polylogue.sources.drive.source_factory import build_drive_source_client

    def client_factory() -> object:
        return build_drive_source_client(config=resolve_runtime_config().drive_config)

    return make_attachment_convergence_stage(
        db_path,
        archive_root=archive_root,
        client_factory=client_factory,
    )


__all__ = [
    "AttachmentConvergenceResult",
    "DEFAULT_ATTACHMENT_CONVERGENCE_LIMIT",
    "DEFAULT_MAX_ATTACHMENT_BYTES",
    "converge_drive_attachments",
    "make_configured_attachment_convergence_stage",
    "make_attachment_convergence_stage",
]
