"""Generation-local accounting for current attachment references.

Writer module: source.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from typing import Literal

AttachmentDisposition = Literal[
    "pending",
    "acquired",
    "duplicate",
    "expired",
    "access_denied",
    "source_missing",
    "malformed",
    "policy_rejected",
    "partial",
    "interrupted",
]


@dataclass(frozen=True, slots=True)
class SourceAttachment:
    reference_id: str
    origin: str
    source_class: str
    reference_count: int = 1
    payload_identity: str | None = None
    payload_bytes: bytes | None = None
    blob_hash: bytes | None = None
    byte_count: int | None = None
    disposition: AttachmentDisposition = "pending"
    reason: str | None = None
    evidence_ref: str | None = None


def record_source_attachments(
    conn: sqlite3.Connection,
    *,
    source_generation_id: str,
    attachments: tuple[SourceAttachment, ...],
    observed_at_ms: int,
) -> None:
    """Record the complete current-source attachment denominator idempotently.

    A reference is keyed by its generation and source identity. Replaying the
    same request is a no-op; changing a terminal fact requires an explicit new
    generation, which prevents a late retry from rewriting a sealed census.
    """
    for attachment in attachments:
        if attachment.reference_count <= 0:
            raise ValueError("reference_count must be positive")
        if attachment.disposition == "acquired":
            if (
                attachment.blob_hash is None
                or attachment.byte_count is None
                or attachment.payload_identity is None
                or attachment.payload_bytes is None
            ):
                raise ValueError("acquired attachment requires hash, bytes, and payload identity")
            if attachment.reason is not None:
                raise ValueError("acquired attachment cannot have an unavailability reason")
            if len(attachment.blob_hash) != 32:
                raise ValueError("attachment blob hash must be SHA-256")
            if hashlib.sha256(attachment.payload_bytes).digest() != attachment.blob_hash:
                raise ValueError("acquired attachment hash does not match its bytes")
            if len(attachment.payload_bytes) != attachment.byte_count:
                raise ValueError("acquired attachment byte count does not match its bytes")
        elif not attachment.reason:
            raise ValueError("unavailable attachment requires an evidence-backed reason")
        conn.execute(
            """INSERT INTO source_attachments(
                source_generation_id, reference_id, origin, source_class,
                reachability, reference_count, payload_identity, blob_hash,
                byte_count, disposition, reason, evidence_ref,
                observed_at_ms, updated_at_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_generation_id, reference_id) DO NOTHING""",
            (
                source_generation_id,
                attachment.reference_id,
                attachment.origin,
                attachment.source_class,
                "current" if attachment.disposition == "acquired" else "unavailable",
                attachment.reference_count,
                attachment.payload_identity,
                attachment.blob_hash,
                attachment.byte_count,
                attachment.disposition,
                attachment.reason,
                attachment.evidence_ref,
                observed_at_ms,
                observed_at_ms,
            ),
        )
    conn.commit()


def source_attachment_census(conn: sqlite3.Connection, source_generation_id: str) -> dict[str, object]:
    """Return exact grouped counts and distinct acquired payload bytes."""
    rows = conn.execute(
        """SELECT origin, source_class, reachability, disposition,
                   COUNT(*) AS reference_rows, SUM(reference_count) AS reference_count,
                   COUNT(DISTINCT payload_identity) AS distinct_payloads,
                   COUNT(DISTINCT blob_hash) AS distinct_blobs,
                   COALESCE(SUM(byte_count), 0) AS bytes
            FROM source_attachments WHERE source_generation_id = ?
            GROUP BY origin, source_class, reachability, disposition
            ORDER BY origin, source_class, reachability, disposition""",
        (source_generation_id,),
    ).fetchall()
    groups = [dict(row) for row in rows]
    distinct_bytes = conn.execute(
        """SELECT COALESCE(SUM(byte_count), 0) FROM (
             SELECT blob_hash, MAX(byte_count) AS byte_count
             FROM source_attachments
             WHERE source_generation_id = ? AND disposition = 'acquired'
             GROUP BY blob_hash)""",
        (source_generation_id,),
    ).fetchone()[0]
    pending = conn.execute(
        "SELECT COUNT(*) FROM source_attachments WHERE source_generation_id = ? AND disposition = 'pending'",
        (source_generation_id,),
    ).fetchone()[0]
    return {
        "source_generation_id": source_generation_id,
        "groups": groups,
        "distinct_payload_bytes": int(distinct_bytes or 0),
        "pending": int(pending or 0),
        "sealable": pending == 0,
    }


__all__ = ["AttachmentDisposition", "SourceAttachment", "record_source_attachments", "source_attachment_census"]
