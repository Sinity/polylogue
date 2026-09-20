"""Generation-local accounting for current attachment references.

Writer module: source.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from typing import Literal

from polylogue.core.errors import PolylogueError

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


#: Dispositions that state a settled outcome for a reference. ``pending`` is
#: the only non-terminal one, so ``pending`` -> terminal is the single allowed
#: progress transition (polylogue-8v4rm).
TERMINAL_DISPOSITIONS: frozenset[str] = frozenset(
    {
        "acquired",
        "duplicate",
        "expired",
        "access_denied",
        "source_missing",
        "malformed",
        "policy_rejected",
        "partial",
        "interrupted",
    }
)

#: Declared facts compared when the same reference is recorded twice. A replay
#: that agrees on all of them is idempotent; one that disagrees is a conflict.
_COMPARED_FIELDS: tuple[str, ...] = (
    "origin",
    "source_class",
    "reachability",
    "reference_count",
    "payload_identity",
    "blob_hash",
    "byte_count",
    "disposition",
    "reason",
    "evidence_ref",
)


class SourceAttachmentConflictError(PolylogueError):
    """A second recording contradicts a settled attachment reference.

    ``ON CONFLICT DO NOTHING`` used to swallow this: two different terminal
    facts for one (generation, reference) silently kept whichever arrived
    first, with no record that the archive had been told two incompatible
    things (polylogue-8v4rm). Identical replay is still a no-op and
    ``pending`` -> terminal is still allowed progress; anything else raises.
    """

    def __init__(self, *, reference_id: str, field: str, stored: object, offered: object) -> None:
        super().__init__(
            f"source attachment {reference_id!r} already recorded a different {field}: "
            f"stored {stored!r}, offered {offered!r}"
        )
        self.reference_id = reference_id
        self.field = field
        self.stored = stored
        self.offered = offered


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
    commit: bool = True,
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
        offered: dict[str, object] = {
            "origin": attachment.origin,
            "source_class": attachment.source_class,
            "reachability": "current" if attachment.disposition == "acquired" else "unavailable",
            "reference_count": attachment.reference_count,
            "payload_identity": attachment.payload_identity,
            "blob_hash": attachment.blob_hash,
            "byte_count": attachment.byte_count,
            "disposition": attachment.disposition,
            "reason": attachment.reason,
            "evidence_ref": attachment.evidence_ref,
        }
        # Select positionally and zip: the caller's ``row_factory`` is not
        # this module's to assume, and a plain tuple row has no name lookup.
        stored_row = conn.execute(
            f"SELECT {', '.join(_COMPARED_FIELDS)} FROM source_attachments "
            "WHERE source_generation_id = ? AND reference_id = ?",
            (source_generation_id, attachment.reference_id),
        ).fetchone()
        if stored_row is not None:
            stored = {
                field: bytes(value) if isinstance(value, memoryview) else value
                for field, value in zip(_COMPARED_FIELDS, tuple(stored_row), strict=True)
            }
            _apply_replay(
                conn,
                source_generation_id=source_generation_id,
                reference_id=attachment.reference_id,
                stored=stored,
                offered=offered,
                observed_at_ms=observed_at_ms,
            )
            continue
        conn.execute(
            """INSERT INTO source_attachments(
                source_generation_id, reference_id, origin, source_class,
                reachability, reference_count, payload_identity, blob_hash,
                byte_count, disposition, reason, evidence_ref,
                observed_at_ms, updated_at_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                source_generation_id,
                attachment.reference_id,
                offered["origin"],
                offered["source_class"],
                offered["reachability"],
                offered["reference_count"],
                offered["payload_identity"],
                offered["blob_hash"],
                offered["byte_count"],
                offered["disposition"],
                offered["reason"],
                offered["evidence_ref"],
                observed_at_ms,
                observed_at_ms,
            ),
        )
    if commit:
        conn.commit()


def _apply_replay(
    conn: sqlite3.Connection,
    *,
    source_generation_id: str,
    reference_id: str,
    stored: dict[str, object],
    offered: dict[str, object],
    observed_at_ms: int,
) -> None:
    """Resolve a second recording of one reference: no-op, progress, or conflict."""

    differing = [field for field in _COMPARED_FIELDS if stored[field] != offered[field]]
    if not differing:
        return
    stored_disposition = str(stored["disposition"])
    offered_disposition = str(offered["disposition"])
    progressing = stored_disposition == "pending" and offered_disposition in TERMINAL_DISPOSITIONS
    if not progressing:
        field = differing[0]
        raise SourceAttachmentConflictError(
            reference_id=reference_id,
            field=field,
            stored=stored[field],
            offered=offered[field],
        )
    # A pending row carries no settled payload evidence, so the terminal fact
    # replaces every declared column atomically with the rest of the batch.
    conn.execute(
        """UPDATE source_attachments
           SET origin = ?, source_class = ?, reachability = ?, reference_count = ?,
               payload_identity = ?, blob_hash = ?, byte_count = ?, disposition = ?,
               reason = ?, evidence_ref = ?, updated_at_ms = ?
           WHERE source_generation_id = ? AND reference_id = ?""",
        (
            offered["origin"],
            offered["source_class"],
            offered["reachability"],
            offered["reference_count"],
            offered["payload_identity"],
            offered["blob_hash"],
            offered["byte_count"],
            offered["disposition"],
            offered["reason"],
            offered["evidence_ref"],
            observed_at_ms,
            source_generation_id,
            reference_id,
        ),
    )


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


__all__ = [
    "TERMINAL_DISPOSITIONS",
    "AttachmentDisposition",
    "SourceAttachment",
    "SourceAttachmentConflictError",
    "record_source_attachments",
    "source_attachment_census",
]
