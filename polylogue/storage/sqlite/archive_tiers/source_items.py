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
from polylogue.security.excision_policy import ExcisionPolicySnapshot

from .common import require_vocabulary
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


@dataclass(frozen=True, slots=True)
class FrozenSourceInput:
    """One retained physical input, never an actuator payload or bearer secret."""

    coordinate: str
    source_path: str
    blob_hash: str
    publication_receipt_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "coordinate": self.coordinate,
            "source_path": self.source_path,
            "blob_hash": self.blob_hash,
            "publication_receipt_id": self.publication_receipt_id,
        }


@dataclass(frozen=True, slots=True)
class FrozenSourceManifest:
    """The complete typed source denominator fixed before machine acceptance."""

    source_generation_id: str
    enumeration_fingerprint: str
    inputs: tuple[FrozenSourceInput, ...]

    def __post_init__(self) -> None:
        _require_digest(self.enumeration_fingerprint, "enumeration_fingerprint")
        if not self.source_generation_id or not 1 <= len(self.inputs) <= 10_000:
            raise ValueError("frozen manifest requires a generation and bounded input set")
        if len({item.coordinate for item in self.inputs}) != len(self.inputs):
            raise ValueError("frozen manifest coordinates must be distinct")
        for item in self.inputs:
            _require_digest(item.blob_hash, "input blob_hash")
            if not item.coordinate.strip() or not item.source_path.strip() or not item.publication_receipt_id:
                raise ValueError("frozen input requires exact coordinates and publication receipt")

    @property
    def manifest_digest(self) -> str:
        # A publication receipt proves retention, not input identity. Exclude
        # its independently generated ID from this immutable content digest.
        content = [
            self.enumeration_fingerprint,
            [(item.coordinate, item.source_path, item.blob_hash) for item in self.inputs],
        ]
        return hashlib.sha256(json.dumps(content, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "source_generation_id": self.source_generation_id,
            "enumeration_fingerprint": self.enumeration_fingerprint,
            "inputs": [item.to_dict() for item in self.inputs],
        }

    @classmethod
    def from_dict(cls, value: object) -> FrozenSourceManifest:
        if not isinstance(value, dict) or set(value) != {"source_generation_id", "enumeration_fingerprint", "inputs"}:
            raise ValueError("invalid frozen source manifest fields")
        if not isinstance(value["source_generation_id"], str) or not isinstance(value["enumeration_fingerprint"], str):
            raise ValueError("invalid frozen source manifest identity")
        raw_inputs = value["inputs"]
        if not isinstance(raw_inputs, list):
            raise ValueError("frozen source inputs must be a list")
        inputs = []
        for item in raw_inputs:
            if (
                not isinstance(item, dict)
                or set(item) != {"coordinate", "source_path", "blob_hash", "publication_receipt_id"}
                or any(not isinstance(field, str) for field in item.values())
            ):
                raise ValueError("invalid frozen source input fields")
            inputs.append(FrozenSourceInput(**item))
        return cls(value["source_generation_id"], value["enumeration_fingerprint"], tuple(inputs))


def prepare_frozen_source_manifest(
    conn: sqlite3.Connection, manifest: FrozenSourceManifest, *, prepared_at_ms: int
) -> None:
    """Join source-WAL prepare; no independent commit or raw admission."""
    from polylogue.storage.blob_publication import consume_blob_publication_receipt

    if not conn.in_transaction:
        raise ValueError("frozen manifest requires the source-WAL prepare transaction")
    for item in manifest.inputs:
        if (
            conn.execute(
                "SELECT 1 FROM blob_publication_reservations WHERE publication_id=? AND blob_hash=?",
                (item.publication_receipt_id, bytes.fromhex(item.blob_hash)),
            ).fetchone()
            is None
        ):
            raise ValueError("frozen input publication reservation is missing or mismatched")
    publish_source_generation(
        conn,
        source_generation_id=manifest.source_generation_id,
        manifest_digest=manifest.manifest_digest,
        addressing_mode="physical-file-v1",
        coordinates=tuple(item.coordinate for item in manifest.inputs),
        source_paths={item.coordinate: item.source_path for item in manifest.inputs},
        input_blob_hashes={item.coordinate: bytes.fromhex(item.blob_hash) for item in manifest.inputs},
        enumeration_fingerprint=manifest.enumeration_fingerprint,
        observed_at_ms=prepared_at_ms,
        commit=False,
    )
    for item in manifest.inputs:
        consume_blob_publication_receipt(conn, item.publication_receipt_id, bytes.fromhex(item.blob_hash))


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
    input_blob_hashes: Mapping[str, bytes] | None = None,
    enumeration_fingerprint: str | None = None,
    commit: bool = True,
) -> tuple[str, ...]:
    """Publish every manifest coordinate before any read/decode/admission work."""
    if len(manifest_digest) != 64 or any(c not in "0123456789abcdef" for c in manifest_digest):
        raise ValueError("manifest_digest must be a SHA-256 hex digest")
    if len(set(coordinates)) != len(coordinates) or any(not coordinate.strip() for coordinate in coordinates):
        raise ValueError("manifest coordinates must be distinct and nonempty")
    if enumeration_fingerprint is not None:
        _require_digest(enumeration_fingerprint, "enumeration_fingerprint")
        if input_blob_hashes is None or set(input_blob_hashes) != set(coordinates):
            raise ValueError("enumerated manifest requires every frozen input blob")
    if input_blob_hashes is not None and (
        set(input_blob_hashes) != set(coordinates)
        or any(not isinstance(value, bytes) or len(value) != 32 for value in input_blob_hashes.values())
    ):
        raise ValueError("input_blob_hashes must bind every coordinate to a SHA-256 blob")
    origin_value = require_vocabulary(origin, Origin, field="origin") if origin is not None else None
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
    if existing is not None:
        expected = {
            (item_id, coordinate, (input_blob_hashes or {}).get(coordinate), enumeration_fingerprint)
            for item_id, coordinate in zip(ids, coordinates, strict=True)
        }
        actual = {
            tuple(row)
            for row in conn.execute(
                "SELECT source_item_id, logical_coordinate, blob_hash, enumeration_fingerprint "
                "FROM source_items WHERE source_generation_id=?",
                (source_generation_id,),
            )
        }
        if actual != expected:
            raise ValueError(f"source generation input binding changed: {source_generation_id}")
    conn.execute(
        """INSERT INTO source_generations(source_generation_id, manifest_digest, addressing_mode, item_count, created_at_ms)
           VALUES (?, ?, ?, ?, ?) ON CONFLICT(source_generation_id) DO NOTHING""",
        (source_generation_id, manifest_digest, addressing_mode, len(coordinates), observed_at_ms),
    )
    for coordinate, item_id in zip(coordinates, ids, strict=True):
        conn.execute(
            """INSERT INTO source_items(
                 source_generation_id, source_item_id, logical_coordinate, addressing_mode,
                 origin, source_path, disposition, outcome_code, stage, observed_at_ms, updated_at_ms,
                 blob_hash, enumeration_fingerprint)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', 'interrupted', 'manifest', ?, ?, ?, ?)
               ON CONFLICT(source_generation_id, source_item_id) DO NOTHING""",
            (
                source_generation_id,
                item_id,
                coordinate,
                addressing_mode,
                origin_value,
                (source_paths or {}).get(coordinate),
                observed_at_ms,
                observed_at_ms,
                (input_blob_hashes or {}).get(coordinate),
                enumeration_fingerprint,
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
    record_source_attachments(
        conn,
        source_generation_id=source_generation_id,
        attachments=attachments,
        observed_at_ms=observed_at_ms,
        commit=False,
    )
    if commit:
        conn.commit()
    return ids


def seal_source_generation(
    conn: sqlite3.Connection,
    *,
    source_generation_id: str,
    sealed_at_ms: int,
    commit: bool = True,
) -> None:
    """Seal only a complete, payload-backed manifest; otherwise fail closed."""
    existing = conn.execute(
        "SELECT sealed_at_ms FROM source_generations WHERE source_generation_id=?",
        (source_generation_id,),
    ).fetchone()
    if existing is None:
        raise KeyError(f"unknown source generation: {source_generation_id}")
    if existing[0] is not None:
        raise ValueError(f"source generation is already sealed: {source_generation_id}")
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
    if commit:
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
    commit: bool = True,
    expected_revision: int | None = None,
) -> int:
    """Advance a fact; a retry of the most recent transition is a no-op.

    The command journal owns historical request deduplication. Callers joining
    a domain publication transaction use ``commit=False`` and can reject stale
    competing item updates with ``expected_revision``.
    """
    row = conn.execute(
        "SELECT revision, request_id, blob_hash, enumeration_fingerprint FROM source_items "
        "WHERE source_generation_id=? AND source_item_id=?",
        (source_generation_id, source_item_id),
    ).fetchone()
    if row is None:
        raise KeyError(f"unmanifested source item: {source_generation_id}/{source_item_id}")
    if row[1] == request_id:
        return int(row[0])
    if expected_revision is not None and int(row[0]) != expected_revision:
        raise ValueError("source item revision changed")
    if row[3] is not None and blob_hash is not None and blob_hash != row[2]:
        raise ValueError("frozen source input blob cannot change")
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
    if commit:
        conn.commit()
    return revision


def _require_digest(value: str, field: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{field} must be a SHA-256 hex digest")


def record_source_item_raw_member(
    conn: sqlite3.Connection,
    *,
    source_generation_id: str,
    source_item_id: str,
    record_coordinate: str,
    raw_id: str,
    raw_blob_hash: bytes,
) -> None:
    """Join successful raw admission in the caller's source transaction.

    An existing edge whose raw was retired is immutable historical evidence,
    not permission to reacquire it. Call this for deduplicated admissions too.
    """
    if not conn.in_transaction:
        raise ValueError("raw membership requires the raw admission transaction")
    if not record_coordinate.strip() or not raw_id or not isinstance(raw_blob_hash, bytes) or len(raw_blob_hash) != 32:
        raise ValueError("raw membership requires a coordinate, raw id and SHA-256 blob")
    item = conn.execute(
        "SELECT enumeration_fingerprint, enumerated_at_ms FROM source_items "
        "WHERE source_generation_id=? AND source_item_id=?",
        (source_generation_id, source_item_id),
    ).fetchone()
    if item is None or item[0] is None:
        raise ValueError("raw membership requires a frozen enumerated input")
    existing = conn.execute(
        "SELECT raw_id, raw_blob_hash FROM source_item_raw_members "
        "WHERE source_generation_id=? AND source_item_id=? AND record_coordinate=?",
        (source_generation_id, source_item_id, record_coordinate),
    ).fetchone()
    if existing is not None:
        if existing[0] is None:
            raise ValueError("source member raw was retired; readmission is forbidden")
        if tuple(existing) != (raw_id, raw_blob_hash):
            raise ValueError("source record membership changed")
        return
    if item[1] is not None:
        raise ValueError("completed source enumeration cannot gain members")
    raw = conn.execute("SELECT blob_hash FROM raw_sessions WHERE raw_id=?", (raw_id,)).fetchone()
    if raw is None or raw[0] != raw_blob_hash:
        raise ValueError("source member requires matching admitted raw evidence")
    conn.execute(
        "INSERT INTO source_item_raw_members "
        "(source_generation_id, source_item_id, record_coordinate, raw_id, raw_blob_hash) VALUES (?, ?, ?, ?, ?)",
        (source_generation_id, source_item_id, record_coordinate, raw_id, raw_blob_hash),
    )


def complete_source_item_enumeration(
    conn: sqlite3.Connection,
    *,
    source_generation_id: str,
    source_item_id: str,
    enumeration_fingerprint: str,
    record_coordinates: tuple[str, ...],
    enumerated_at_ms: int,
) -> str:
    """Record exhausted decoder evidence, atomically with its final admission.

    The caller supplies the complete coordinate denominator only after normal
    iterator exhaustion. Interruption or a skipped malformed record must not
    call this function. An empty tuple therefore means proven empty input.
    """
    if not conn.in_transaction:
        raise ValueError("enumeration completion requires a source transaction")
    _require_digest(enumeration_fingerprint, "enumeration_fingerprint")
    if len(set(record_coordinates)) != len(record_coordinates) or any(not c.strip() for c in record_coordinates):
        raise ValueError("enumeration coordinates must be distinct and nonempty")
    item = conn.execute(
        "SELECT enumeration_fingerprint, enumerated_record_count, enumeration_digest, enumerated_at_ms "
        "FROM source_items WHERE source_generation_id=? AND source_item_id=?",
        (source_generation_id, source_item_id),
    ).fetchone()
    if item is None or item[0] != enumeration_fingerprint:
        raise ValueError("source enumeration decoder binding changed")
    members = list(
        conn.execute(
            "SELECT record_coordinate, raw_blob_hash, raw_id FROM source_item_raw_members "
            "WHERE source_generation_id=? AND source_item_id=? ORDER BY record_coordinate",
            (source_generation_id, source_item_id),
        )
    )
    if {row[0] for row in members} != set(record_coordinates):
        raise ValueError("source enumeration has missing or unexpected raw members")
    if any(row[2] is None for row in members):
        raise ValueError("source enumeration contains retired raw members")
    digest = hashlib.sha256(
        json.dumps([(row[0], row[1].hex()) for row in members], ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if item[3] is not None:
        if tuple(item[1:3]) != (len(members), digest):
            raise ValueError("completed source enumeration changed")
        return digest
    conn.execute(
        "UPDATE source_items SET enumerated_record_count=?, enumeration_digest=?, enumerated_at_ms=? "
        "WHERE source_generation_id=? AND source_item_id=?",
        (len(members), digest, enumerated_at_ms, source_generation_id, source_item_id),
    )
    return digest


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
    "FrozenSourceInput",
    "FrozenSourceManifest",
    "complete_source_item_enumeration",
    "publish_source_generation",
    "prepare_frozen_source_manifest",
    "record_source_item_raw_member",
    "seal_source_generation",
    "source_generation_census",
    "source_item_id",
    "transition_source_item",
]
