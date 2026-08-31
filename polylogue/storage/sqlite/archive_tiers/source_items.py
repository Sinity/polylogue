"""Durable source-generation accounting and idempotent item transitions.

Writer module: source.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from polylogue.core.enums import IngestOutcome, Origin
from polylogue.pipeline.ingest_outcomes import bounded_diagnostic
from polylogue.security.excision_policy import ExcisionPolicySnapshot, read_excision_policy_projection

from .source_attachments import SourceAttachment, record_source_attachments, source_attachment_census


class AcquisitionDisposition(StrEnum):
    PENDING = "pending"
    ADMITTED = "admitted"
    NON_SESSION = "non_session"
    EMPTY = "empty"
    UNSUPPORTED = "unsupported"
    CORRUPT = "corrupt"
    UNKNOWN_BLOCKING = "unknown_blocking"


@dataclass(frozen=True, slots=True)
class SourceItem:
    source_generation_id: str
    source_item_id: str
    logical_coordinate: str
    addressing_mode: str
    disposition: AcquisitionDisposition
    outcome_code: IngestOutcome
    stage: str
    revision: int
    retryable: bool | None
    raw_id: str | None
    blob_hash: bytes | None


def source_item_id(*, source_generation_id: str, logical_coordinate: str, addressing_mode: str) -> str:
    """Derive identity only from generation-bound manifest coordinates."""
    payload = json.dumps(
        [source_generation_id, addressing_mode, logical_coordinate],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def publish_source_generation(
    conn: sqlite3.Connection,
    *,
    source_generation_id: str,
    manifest_digest: str,
    addressing_mode: str,
    coordinates: tuple[str, ...],
    observed_at_ms: int,
    origin: Origin | str | None = None,
    source_paths: Mapping[str, str] | None = None,
    attachments: tuple[SourceAttachment, ...] = (),
    policy_snapshot: ExcisionPolicySnapshot | None = None,
) -> tuple[str, ...]:
    """Publish every manifest coordinate before any read/decode/admission work."""
    if len(manifest_digest) != 64:
        raise ValueError("manifest_digest must be a SHA-256 hex digest")
    ids = tuple(
        source_item_id(source_generation_id=source_generation_id, logical_coordinate=c, addressing_mode=addressing_mode)
        for c in coordinates
    )
    existing = conn.execute(
        "SELECT manifest_digest, addressing_mode, item_count FROM source_generations WHERE source_generation_id=?",
        (source_generation_id,),
    ).fetchone()
    if existing is not None and tuple(existing) != (manifest_digest, addressing_mode, len(coordinates)):
        raise ValueError(f"source generation manifest changed: {source_generation_id}")
    conn.execute(
        """INSERT INTO source_generations(source_generation_id, manifest_digest, addressing_mode, item_count, created_at_ms)
           VALUES (?, ?, ?, ?, ?) ON CONFLICT(source_generation_id) DO NOTHING""",
        (source_generation_id, manifest_digest, addressing_mode, len(coordinates), observed_at_ms),
    )
    for coordinate, item_id in zip(coordinates, ids, strict=True):
        conn.execute(
            """INSERT INTO source_items(
                 source_generation_id, source_item_id, logical_coordinate, addressing_mode,
                 origin, source_path, disposition, outcome_code, stage, observed_at_ms, updated_at_ms)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', 'interrupted', 'manifest', ?, ?)
               ON CONFLICT(source_generation_id, source_item_id) DO NOTHING""",
            (
                source_generation_id,
                item_id,
                coordinate,
                addressing_mode,
                getattr(origin, "value", origin),
                (source_paths or {}).get(coordinate),
                observed_at_ms,
                observed_at_ms,
            ),
        )
    if policy_snapshot is not None:
        conn.execute("""CREATE TABLE IF NOT EXISTS excision_policy_projections (
            source_generation_id TEXT PRIMARY KEY REFERENCES source_generations(source_generation_id) ON DELETE CASCADE,
            policy_digest TEXT NOT NULL CHECK(length(policy_digest) = 64),
            user_generation INTEGER NOT NULL CHECK(user_generation >= 0),
            audit_generation INTEGER NOT NULL CHECK(audit_generation >= 0),
            audit_head TEXT NOT NULL CHECK(length(audit_head) = 64),
            assertion_refs_json TEXT NOT NULL DEFAULT '[]',
            generated_at_ms INTEGER NOT NULL CHECK(generated_at_ms >= 0)
        ) STRICT""")
        conn.execute(
            """INSERT INTO excision_policy_projections(
               source_generation_id, policy_digest, user_generation, audit_generation,
               audit_head, assertion_refs_json, generated_at_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(source_generation_id) DO UPDATE SET
                 policy_digest=excluded.policy_digest,
                 user_generation=excluded.user_generation,
                 audit_generation=excluded.audit_generation,
                 audit_head=excluded.audit_head,
                 assertion_refs_json=excluded.assertion_refs_json,
                 generated_at_ms=excluded.generated_at_ms""",
            (
                source_generation_id,
                policy_snapshot.digest,
                policy_snapshot.user_generation,
                policy_snapshot.audit_generation,
                policy_snapshot.audit_head,
                json.dumps(policy_snapshot.assertion_refs, separators=(",", ":")),
                observed_at_ms,
            ),
        )
        # Exercise the same production read path that validates a generation
        # binding after projection replacement.
        read_excision_policy_projection(conn, source_generation_id)
    record_source_attachments(
        conn,
        source_generation_id=source_generation_id,
        attachments=attachments,
        observed_at_ms=observed_at_ms,
    )
    return ids


def seal_source_generation(conn: sqlite3.Connection, *, source_generation_id: str, sealed_at_ms: int) -> None:
    """Seal only a complete, payload-backed manifest; otherwise fail closed."""
    census = source_generation_census(conn, source_generation_id)
    if not census["sealable"]:
        raise ValueError(f"source generation is not sealable: {source_generation_id}")
    attachment_table = conn.execute(
        "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='source_attachments'"
    ).fetchone()
    if attachment_table is not None:
        attachment_census = source_attachment_census(conn, source_generation_id)
        if not attachment_census["sealable"]:
            raise ValueError(f"source generation has pending attachments: {source_generation_id}")
    conn.execute(
        "UPDATE source_generations SET sealed_at_ms=? WHERE source_generation_id=?",
        (sealed_at_ms, source_generation_id),
    )
    conn.commit()


def transition_source_item(
    conn: sqlite3.Connection,
    *,
    source_generation_id: str,
    source_item_id: str,
    request_id: str,
    disposition: AcquisitionDisposition,
    outcome_code: IngestOutcome,
    stage: str,
    observed_at_ms: int,
    retryable: bool | None = None,
    diagnostic: str | None = None,
    evidence_ref: str | None = None,
    content_fingerprint: str | None = None,
    source_fingerprint: str | None = None,
    parser_fingerprint: str | None = None,
    policy_fingerprint: str | None = None,
    raw_id: str | None = None,
    blob_hash: bytes | None = None,
) -> int:
    """Advance one current fact exactly once; replaying a request is a no-op."""
    row = conn.execute(
        "SELECT revision, request_id FROM source_items WHERE source_generation_id=? AND source_item_id=?",
        (source_generation_id, source_item_id),
    ).fetchone()
    if row is None:
        raise KeyError(f"unmanifested source item: {source_generation_id}/{source_item_id}")
    if row[1] == request_id:
        return int(row[0])
    revision = int(row[0]) + 1
    conn.execute(
        """UPDATE source_items SET disposition=?, outcome_code=?, stage=?, retryable=?, diagnostic=?,
           evidence_ref=?, content_fingerprint=COALESCE(?,content_fingerprint),
           source_fingerprint=COALESCE(?,source_fingerprint), parser_fingerprint=COALESCE(?,parser_fingerprint),
           policy_fingerprint=COALESCE(?,policy_fingerprint), raw_id=COALESCE(?,raw_id),
           blob_hash=COALESCE(?,blob_hash), revision=?, request_id=?, observed_at_ms=?, updated_at_ms=?
           WHERE source_generation_id=? AND source_item_id=?""",
        (
            disposition.value,
            outcome_code.value,
            stage,
            None if retryable is None else int(retryable),
            bounded_diagnostic(diagnostic, max_len=4096),
            evidence_ref,
            content_fingerprint,
            source_fingerprint,
            parser_fingerprint,
            policy_fingerprint,
            raw_id,
            blob_hash,
            revision,
            request_id,
            observed_at_ms,
            observed_at_ms,
            source_generation_id,
            source_item_id,
        ),
    )
    conn.commit()
    return revision


def source_generation_census(conn: sqlite3.Connection, source_generation_id: str) -> dict[str, int | bool]:
    cursor = conn.execute(
        "SELECT * FROM source_item_reconciliation WHERE source_generation_id=?", (source_generation_id,)
    )
    row = cursor.fetchone()
    if row is None:
        raise KeyError(source_generation_id)
    names = [column[0] for column in cursor.description or ()]
    return {
        name: (bool(value) if name == "sealable" else (value if name == "source_generation_id" else int(value or 0)))
        for name, value in zip(names, row, strict=True)
    }


__all__ = [
    "AcquisitionDisposition",
    "SourceItem",
    "publish_source_generation",
    "seal_source_generation",
    "source_generation_census",
    "source_item_id",
    "transition_source_item",
]
