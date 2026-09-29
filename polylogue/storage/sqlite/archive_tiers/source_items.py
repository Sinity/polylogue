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

from polylogue.core.enums import INGEST_OUTCOME_RETRYABLE, IngestOutcome, Origin
from polylogue.pipeline.ingest_outcomes import bounded_diagnostic
from polylogue.security.excision_policy import ExcisionPolicySnapshot

from .common import require_vocabulary
from .source_attachments import (
    SourceAttachment,
    _preflight_source_attachments,
    record_source_attachments,
    source_attachment_census,
)


class AcquisitionDisposition(StrEnum):
    """Storage-owned source-item state machine, checked by its writer.

    Durable source DDL intentionally does not mirror this vocabulary. Add a
    state only with its transition, retryability, reconciliation, and reader
    behavior reviewed together.
    """

    PENDING = "pending"
    ADMITTED = "admitted"
    NON_SESSION = "non_session"
    EMPTY = "empty"
    UNSUPPORTED = "unsupported"
    CORRUPT = "corrupt"
    UNKNOWN_BLOCKING = "unknown_blocking"


class SourceItemMemberDisposition(StrEnum):
    """Why a central-directory member was not admitted as raw evidence.

    This storage-local audit vocabulary is validated at
    ``record_source_item_member_disposition``. Extend it only when the ZIP
    admission policy and enumeration/reconciliation reader understand the new
    outcome; durable source DDL remains vocabulary-free.
    """

    REFUSED = "refused"
    UNSELECTED = "unselected"


_MAX_MEMBER_DIAGNOSTIC_CHARS = 4080


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
class RetainedSourceInput:
    """An accepted source reference; publication authority was already consumed."""

    source_item_id: str
    coordinate: str
    source_path: str
    blob_hash: str
    enumeration_complete: bool
    stage: str
    revision: int


@dataclass(frozen=True, slots=True)
class RetainedSourceGeneration:
    source_generation_id: str
    enumeration_fingerprint: str
    inputs: tuple[RetainedSourceInput, ...]
    item_count: int = 0


@dataclass(frozen=True, slots=True)
class SourceItemMemberDispositionRecord:
    source_generation_id: str
    source_item_id: str
    entry_ordinal: int
    member_name: str
    disposition: SourceItemMemberDisposition
    diagnostic: str
    observed_at_ms: int


def retained_source_generation_header(conn: sqlite3.Connection, source_generation_id: str) -> RetainedSourceGeneration:
    """Read the accepted decoder and count without materializing input members."""
    row = conn.execute(
        "SELECT item_count FROM source_generations WHERE source_generation_id=?", (source_generation_id,)
    ).fetchone()
    if row is None or type(row[0]) is not int or row[0] < 1:
        raise ValueError("accepted source generation is absent or empty")
    first = page_retained_source_inputs(conn, source_generation_id, limit=1)
    if not first:
        raise ValueError("accepted source generation has no retained input")
    fingerprint = first[0][1]
    _require_digest(fingerprint, "enumeration_fingerprint")
    return RetainedSourceGeneration(source_generation_id, fingerprint, (), row[0])


def page_retained_source_inputs(
    conn: sqlite3.Connection, source_generation_id: str, *, after: tuple[str, str] | None = None, limit: int = 256
) -> tuple[tuple[RetainedSourceInput, str], ...]:
    """Read a bounded accepted input page with a stable two-column keyset."""
    if not 1 <= limit <= 256:
        raise ValueError("accepted input page limit must be between 1 and 256")
    rows = conn.execute(
        "SELECT source_item_id, logical_coordinate, source_path, blob_hash, enumeration_fingerprint, "
        "enumerated_at_ms, stage, revision FROM source_items WHERE source_generation_id=? "
        "AND (logical_coordinate, source_item_id) > (?, ?) ORDER BY logical_coordinate, source_item_id LIMIT ?",
        (source_generation_id, *(after or ("", "")), limit),
    ).fetchall()
    result: list[tuple[RetainedSourceInput, str]] = []
    for row in rows:
        if row[2] is None or not isinstance(row[3], bytes) or len(row[3]) != 32 or row[4] is None:
            raise ValueError("accepted source input lost its retained physical identity")
        result.append(
            (
                RetainedSourceInput(
                    str(row[0]), str(row[1]), str(row[2]), row[3].hex(), row[5] is not None, str(row[6]), int(row[7])
                ),
                str(row[4]),
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class FrozenSourceManifest:
    """The complete typed source denominator fixed before machine acceptance."""

    source_generation_id: str
    enumeration_fingerprint: str
    inputs: tuple[FrozenSourceInput, ...]
    source_name: str | None = None

    def __post_init__(self) -> None:
        _require_digest(self.enumeration_fingerprint, "enumeration_fingerprint")
        if not self.source_generation_id or not 1 <= len(self.inputs) <= 10_000:
            raise ValueError("frozen manifest requires a generation and bounded input set")
        if len({item.coordinate for item in self.inputs}) != len(self.inputs):
            raise ValueError("frozen manifest coordinates must be distinct")
        if self.source_name is not None and (not self.source_name.strip() or len(self.source_name) > 255):
            raise ValueError("frozen source name must be nonempty and bounded")
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
        if self.source_name is not None:
            content.append(["source_name", self.source_name])
        return hashlib.sha256(json.dumps(content, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "source_generation_id": self.source_generation_id,
            "enumeration_fingerprint": self.enumeration_fingerprint,
            "inputs": [item.to_dict() for item in self.inputs],
        }
        if self.source_name is not None:
            value["source_name"] = self.source_name
        return value

    @classmethod
    def from_dict(cls, value: object) -> FrozenSourceManifest:
        required = {"source_generation_id", "enumeration_fingerprint", "inputs"}
        if not isinstance(value, dict) or not required <= set(value) or set(value) - required - {"source_name"}:
            raise ValueError("invalid frozen source manifest fields")
        if not isinstance(value["source_generation_id"], str) or not isinstance(value["enumeration_fingerprint"], str):
            raise ValueError("invalid frozen source manifest identity")
        source_name = value.get("source_name")
        if source_name is not None and not isinstance(source_name, str):
            raise ValueError("invalid frozen source name")
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
        return cls(value["source_generation_id"], value["enumeration_fingerprint"], tuple(inputs), source_name)


@dataclass(frozen=True, slots=True)
class SealedSourceManifestRef:
    """Bounded authority for a complete prepared denominator in source.db."""

    source_generation_id: str
    enumeration_fingerprint: str
    manifest_digest: str
    custody_digest: str
    input_count: int
    source_name: str | None = None

    def __post_init__(self) -> None:
        if not self.source_generation_id or self.input_count < 1:
            raise ValueError("sealed manifest requires a nonempty generation")
        for name in ("enumeration_fingerprint", "manifest_digest", "custody_digest"):
            _require_digest(getattr(self, name), name)
        if self.source_name is not None and (not self.source_name.strip() or len(self.source_name) > 255):
            raise ValueError("sealed source name must be bounded and nonempty")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_generation_id": self.source_generation_id,
            "enumeration_fingerprint": self.enumeration_fingerprint,
            "manifest_digest": self.manifest_digest,
            "custody_digest": self.custody_digest,
            "input_count": self.input_count,
            "source_name": self.source_name,
        }

    @classmethod
    def from_dict(cls, value: object) -> SealedSourceManifestRef:
        fields = {
            "source_generation_id",
            "enumeration_fingerprint",
            "manifest_digest",
            "custody_digest",
            "input_count",
            "source_name",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("invalid sealed source manifest reference")
        if any(type(value[field]) is not str for field in fields - {"input_count", "source_name"}):
            raise ValueError("invalid sealed source manifest identity")
        if type(value["input_count"]) is not int or (
            value["source_name"] is not None and type(value["source_name"]) is not str
        ):
            raise ValueError("invalid sealed source manifest count or name")
        return cls(**value)


def source_manifest_from_dict(value: object) -> FrozenSourceManifest | SealedSourceManifestRef:
    """Decode old bounded audit commands and new sealed references exactly."""
    if isinstance(value, dict) and "inputs" in value:
        return FrozenSourceManifest.from_dict(value)
    return SealedSourceManifestRef.from_dict(value)


def begin_prepared_source_manifest(
    conn: sqlite3.Connection,
    *,
    source_generation_id: str,
    publisher_id: str,
    enumeration_fingerprint: str,
    source_name: str | None,
) -> None:
    if not conn.in_transaction or not source_generation_id or not publisher_id:
        raise ValueError("prepared source manifest needs a source transaction and generation")
    _require_digest(enumeration_fingerprint, "enumeration_fingerprint")
    if source_name is not None and (not source_name.strip() or len(source_name) > 255):
        raise ValueError("invalid source name")
    conn.execute(
        "INSERT INTO prepared_source_manifests(source_generation_id, publisher_id, enumeration_fingerprint, source_name) "
        "VALUES (?, ?, ?, ?)",
        (source_generation_id, publisher_id, enumeration_fingerprint, source_name),
    )


def abort_prepared_source_manifest(conn: sqlite3.Connection, *, source_generation_id: str, publisher_id: str) -> None:
    """Release this attempt's staged rows and reservations before acceptance.

    The publisher belongs to this ingest execution alone. A promoted generation
    is never abortable, including after an ambiguous audit interruption.
    """
    if not conn.in_transaction or not source_generation_id or not publisher_id:
        raise ValueError("prepared source abort requires a transaction and identities")
    header = conn.execute(
        "SELECT publisher_id FROM prepared_source_manifests WHERE source_generation_id=?", (source_generation_id,)
    ).fetchone()
    if header is not None and header[0] != publisher_id:
        raise ValueError("prepared source publisher identity changed")
    if (
        conn.execute(
            "SELECT 1 FROM source_generations WHERE source_generation_id=?", (source_generation_id,)
        ).fetchone()
        is not None
    ):
        raise ValueError("accepted source generation cannot be aborted")
    conn.execute("DELETE FROM prepared_source_manifest_members WHERE source_generation_id=?", (source_generation_id,))
    conn.execute("DELETE FROM prepared_source_manifests WHERE source_generation_id=?", (source_generation_id,))
    conn.execute("DELETE FROM blob_publication_reservations WHERE publisher_id=?", (publisher_id,))


def reconcile_unaccepted_prepared_source_manifests(conn: sqlite3.Connection) -> int:
    """Reclaim dead pre-accept staging after audit continuity has settled.

    The daemon startup owner calls this before admitting requests. A surviving
    accepted audit transition has already promoted its source generation; an
    unresolved audit transition prevents startup from reaching this call.
    """
    if not conn.in_transaction:
        raise ValueError("prepared source reconciliation requires a source transaction")
    orphan = (
        "SELECT source_generation_id, publisher_id FROM prepared_source_manifests p "
        "WHERE NOT EXISTS (SELECT 1 FROM source_generations g "
        "WHERE g.source_generation_id=p.source_generation_id)"
    )
    count = int(conn.execute(f"SELECT COUNT(*) FROM ({orphan})").fetchone()[0])
    conn.execute(
        f"DELETE FROM blob_publication_reservations WHERE publisher_id IN (SELECT publisher_id FROM ({orphan}))"
    )
    conn.execute(
        "DELETE FROM prepared_source_manifest_members WHERE source_generation_id IN "
        f"(SELECT source_generation_id FROM ({orphan}))"
    )
    conn.execute(
        "DELETE FROM prepared_source_manifests WHERE NOT EXISTS "
        "(SELECT 1 FROM source_generations g WHERE g.source_generation_id=prepared_source_manifests.source_generation_id)"
    )
    return count


def append_prepared_source_inputs(
    conn: sqlite3.Connection, source_generation_id: str, start_ordinal: int, inputs: tuple[FrozenSourceInput, ...]
) -> None:
    if not conn.in_transaction or not 1 <= len(inputs) <= 256:
        raise ValueError("prepared source input batch must be bounded by 256")
    row = conn.execute(
        "SELECT input_count, sealed_at_ms FROM prepared_source_manifests WHERE source_generation_id=?",
        (source_generation_id,),
    ).fetchone()
    if row is None or row[1] is not None or int(row[0]) != start_ordinal:
        raise ValueError("prepared source input batch is not the next unsealed page")
    for offset, item in enumerate(inputs):
        _require_digest(item.blob_hash, "input blob_hash")
        if not item.coordinate.strip() or not item.source_path.strip() or not item.publication_receipt_id:
            raise ValueError("prepared source input is incomplete")
        conn.execute(
            "INSERT INTO prepared_source_manifest_members VALUES (?, ?, ?, ?, ?, ?)",
            (
                source_generation_id,
                start_ordinal + offset,
                item.coordinate,
                item.source_path,
                bytes.fromhex(item.blob_hash),
                item.publication_receipt_id,
            ),
        )
    conn.execute(
        "UPDATE prepared_source_manifests SET input_count=? WHERE source_generation_id=?",
        (start_ordinal + len(inputs), source_generation_id),
    )


def _prepared_manifest_digests(
    conn: sqlite3.Connection, generation_id: str, fingerprint: str, source_name: str | None
) -> tuple[str, str, int]:
    semantic = hashlib.sha256()
    semantic.update(("[" + json.dumps(fingerprint, ensure_ascii=False, separators=(",", ":")) + ",[").encode())
    custody = hashlib.sha256()
    custody.update(
        json.dumps([generation_id, fingerprint, source_name], ensure_ascii=False, separators=(",", ":")).encode()
    )
    count = 0
    for ordinal, coordinate, source_path, blob_hash, receipt_id in conn.execute(
        "SELECT ordinal, coordinate, source_path, blob_hash, publication_receipt_id "
        "FROM prepared_source_manifest_members WHERE source_generation_id=? ORDER BY ordinal",
        (generation_id,),
    ):
        if type(ordinal) is not int or ordinal != count or not isinstance(blob_hash, bytes) or len(blob_hash) != 32:
            raise ValueError("prepared source manifest has a missing or malformed ordinal")
        item = [str(coordinate), str(source_path), blob_hash.hex()]
        semantic.update((("," if count else "") + json.dumps(item, ensure_ascii=False, separators=(",", ":"))).encode())
        custody.update(
            ("\n" + json.dumps([ordinal, *item, str(receipt_id)], ensure_ascii=False, separators=(",", ":"))).encode()
        )
        count += 1
    semantic.update(b"]")
    if source_name is not None:
        semantic.update(
            (',["source_name",' + json.dumps(source_name, ensure_ascii=False, separators=(",", ":")) + "]").encode()
        )
    semantic.update(b"]")
    return semantic.hexdigest(), custody.hexdigest(), count


def seal_prepared_source_manifest(
    conn: sqlite3.Connection, generation_id: str, *, sealed_at_ms: int
) -> SealedSourceManifestRef:
    if not conn.in_transaction:
        raise ValueError("prepared source seal requires a source transaction")
    row = conn.execute(
        "SELECT enumeration_fingerprint, source_name, input_count, sealed_at_ms FROM prepared_source_manifests WHERE source_generation_id=?",
        (generation_id,),
    ).fetchone()
    if row is None or row[3] is not None or int(row[2]) < 1:
        raise ValueError("prepared source manifest is absent, empty, or already sealed")
    semantic, custody, count = _prepared_manifest_digests(conn, generation_id, str(row[0]), row[1])
    if count != int(row[2]):
        raise ValueError("prepared source manifest count differs from its rows")
    ref = SealedSourceManifestRef(generation_id, str(row[0]), semantic, custody, count, row[1])
    _require_prepared_reservations(conn, ref)
    conn.execute(
        "UPDATE prepared_source_manifests SET semantic_digest=?, custody_digest=?, sealed_at_ms=? WHERE source_generation_id=?",
        (semantic, custody, sealed_at_ms, generation_id),
    )
    return ref


def _require_prepared_reservations(conn: sqlite3.Connection, ref: SealedSourceManifestRef) -> None:
    missing = conn.execute(
        "SELECT 1 FROM prepared_source_manifest_members m LEFT JOIN blob_publication_reservations r "
        "ON r.publication_id=m.publication_receipt_id AND r.blob_hash=m.blob_hash "
        "WHERE m.source_generation_id=? AND r.publication_id IS NULL LIMIT 1",
        (ref.source_generation_id,),
    ).fetchone()
    if missing is not None:
        raise ValueError("sealed input publication reservation is missing or mismatched")


def validate_sealed_source_manifest(conn: sqlite3.Connection, ref: SealedSourceManifestRef) -> None:
    if not conn.in_transaction:
        raise ValueError("sealed source manifest requires a source transaction")
    row = conn.execute(
        "SELECT enumeration_fingerprint, source_name, input_count, semantic_digest, custody_digest, sealed_at_ms "
        "FROM prepared_source_manifests WHERE source_generation_id=?",
        (ref.source_generation_id,),
    ).fetchone()
    if (
        row is None
        or tuple(row[:5])
        != (ref.enumeration_fingerprint, ref.source_name, ref.input_count, ref.manifest_digest, ref.custody_digest)
        or row[5] is None
    ):
        raise ValueError("sealed source manifest reference differs from durable header")
    semantic, custody, count = _prepared_manifest_digests(
        conn, ref.source_generation_id, ref.enumeration_fingerprint, ref.source_name
    )
    if (semantic, custody, count) != (ref.manifest_digest, ref.custody_digest, ref.input_count):
        raise ValueError("sealed source manifest rows differ from its digests")
    accepted = conn.execute(
        "SELECT manifest_digest, item_count FROM source_generations WHERE source_generation_id=?",
        (ref.source_generation_id,),
    ).fetchone()
    if accepted is None:
        _require_prepared_reservations(conn, ref)
    elif tuple(accepted) != (ref.manifest_digest, ref.input_count):
        raise ValueError("accepted source generation differs from sealed manifest")
    else:
        accepted_count = conn.execute(
            "SELECT COUNT(*) FROM source_items WHERE source_generation_id=?",
            (ref.source_generation_id,),
        ).fetchone()[0]
        mismatch = conn.execute(
            "SELECT 1 FROM prepared_source_manifest_members m LEFT JOIN source_items i "
            "ON i.source_generation_id=m.source_generation_id AND i.logical_coordinate=m.coordinate "
            "WHERE m.source_generation_id=? AND (i.source_item_id IS NULL OR i.source_path<>m.source_path "
            "OR i.blob_hash<>m.blob_hash OR i.enumeration_fingerprint<>?) LIMIT 1",
            (ref.source_generation_id, ref.enumeration_fingerprint),
        ).fetchone()
        if int(accepted_count) != ref.input_count or mismatch is not None:
            raise ValueError("accepted source inputs differ from sealed manifest")


def publish_sealed_source_manifest(
    conn: sqlite3.Connection, ref: SealedSourceManifestRef, *, prepared_at_ms: int
) -> None:
    """Publish all accepted members and consume custody in one source transaction."""
    from polylogue.storage.blob_publication import consume_blob_publication_receipt

    validate_sealed_source_manifest(conn, ref)
    existing = conn.execute(
        "SELECT 1 FROM source_generations WHERE source_generation_id=?",
        (ref.source_generation_id,),
    ).fetchone()
    if existing is not None:
        actual_count = conn.execute(
            "SELECT COUNT(*) FROM source_items WHERE source_generation_id=?",
            (ref.source_generation_id,),
        ).fetchone()[0]
        if int(actual_count) != ref.input_count:
            raise ValueError("accepted source generation is incomplete")
        return
    conn.execute(
        "INSERT INTO source_generations(source_generation_id, manifest_digest, addressing_mode, item_count, created_at_ms) "
        "VALUES (?, ?, 'physical-file-v1', ?, ?)",
        (ref.source_generation_id, ref.manifest_digest, ref.input_count, prepared_at_ms),
    )
    for _ordinal, coordinate, source_path, blob_hash, receipt_id in conn.execute(
        "SELECT ordinal, coordinate, source_path, blob_hash, publication_receipt_id "
        "FROM prepared_source_manifest_members WHERE source_generation_id=? ORDER BY ordinal",
        (ref.source_generation_id,),
    ):
        item_id = source_item_id(
            source_generation_id=ref.source_generation_id,
            logical_coordinate=str(coordinate),
            addressing_mode="physical-file-v1",
        )
        conn.execute(
            "INSERT INTO source_items(source_generation_id, source_item_id, logical_coordinate, addressing_mode, "
            "source_path, disposition, outcome_code, stage, observed_at_ms, updated_at_ms, blob_hash, enumeration_fingerprint) "
            "VALUES (?, ?, ?, 'physical-file-v1', ?, 'pending', 'interrupted', 'manifest', ?, ?, ?, ?)",
            (
                ref.source_generation_id,
                item_id,
                coordinate,
                source_path,
                prepared_at_ms,
                prepared_at_ms,
                blob_hash,
                ref.enumeration_fingerprint,
            ),
        )
        consume_blob_publication_receipt(conn, str(receipt_id), blob_hash)


def validate_frozen_source_manifest(conn: sqlite3.Connection, manifest: FrozenSourceManifest) -> None:
    """Prove every frozen input still holds its publication reservation.

    Read-only on purpose.  The source-WAL prepare transaction must be able to
    roll back to nothing, so the prepare phase only *checks* the manifest; the
    durable publication happens in :func:`publish_frozen_source_manifest`
    during promotion, once the audit commit that accepts it is durable.
    """
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


def publish_frozen_source_manifest(
    conn: sqlite3.Connection, manifest: FrozenSourceManifest, *, prepared_at_ms: int
) -> None:
    """Join the source-WAL promotion; no independent commit or raw admission."""
    from polylogue.storage.blob_publication import consume_blob_publication_receipt

    if not conn.in_transaction:
        raise ValueError("frozen manifest requires the source-WAL promotion transaction")
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
        or any(type(value) is not bytes or len(value) != 32 for value in input_blob_hashes.values())
    ):
        raise ValueError("input_blob_hashes must bind every coordinate to a SHA-256 blob")
    origin_value = require_vocabulary(origin, Origin, field="origin") if origin is not None else None
    _preflight_source_attachments(attachments)
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
        # ``excision_policy_projections`` is canonical source DDL. An ordinary
        # write must never create a durable shape: it leaves the table absent
        # from every archive this generation was not written into, and forces
        # readers to probe sqlite_schema instead of querying (polylogue-j264r).
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
    # These vocabularies are owned by the source-item state machine and
    # IngestOutcome. SQLite intentionally carries no enum registry; normalize
    # and reject at this public write boundary before even the idempotent path.
    disposition_value = require_vocabulary(disposition, AcquisitionDisposition, field="disposition")
    outcome_value = require_vocabulary(outcome_code, IngestOutcome, field="outcome_code")
    if retryable is None:
        retryable = INGEST_OUTCOME_RETRYABLE[IngestOutcome(outcome_value)]
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
            disposition_value,
            outcome_value,
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
    if not record_coordinate.strip() or not raw_id or type(raw_blob_hash) is not bytes or len(raw_blob_hash) != 32:
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
    member_ordinals: tuple[int, ...] | None = None,
    member_count: int | None = None,
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
        "SELECT enumeration_fingerprint, enumerated_record_count, enumeration_digest, enumerated_at_ms, "
        "enumerated_member_count, enumeration_member_digest "
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
    is_zip_member = any(str(row[0]).startswith('["zip-v2"') for row in members)
    if member_count is None and is_zip_member:
        raise ValueError("ZIP source enumeration requires its central-directory denominator")
    if member_count is not None:
        if member_count < 0:
            raise ValueError("source member denominator must be non-negative")
        if member_ordinals is None:
            raise ValueError("source member denominator requires central-directory ordinals")
        if len(set(member_ordinals)) != len(member_ordinals) or any(value < 0 for value in member_ordinals):
            raise ValueError("source member ordinals must be distinct and non-negative")
        if len(member_ordinals) != member_count or set(member_ordinals) != set(range(member_count)):
            raise ValueError("source enumeration has missing or unexpected central-directory members")
        disposition_rows = list(
            conn.execute(
                "SELECT entry_ordinal, member_name, disposition, diagnostic FROM source_item_member_dispositions "
                "WHERE source_generation_id=? AND source_item_id=? ORDER BY entry_ordinal",
                (source_generation_id, source_item_id),
            )
        )
        accepted_ordinals: set[int] = set()
        for row in conn.execute(
            "SELECT DISTINCT c.entry_ordinal FROM source_item_raw_members m "
            "JOIN raw_container_coordinates c ON c.raw_id=m.raw_id "
            "WHERE m.source_generation_id=? AND m.source_item_id=? AND m.raw_id IS NOT NULL",
            (source_generation_id, source_item_id),
        ):
            accepted_ordinals.add(int(row[0]))
        disposition_ordinals = {int(row[0]) for row in disposition_rows}
        if accepted_ordinals & disposition_ordinals or accepted_ordinals | disposition_ordinals != set(member_ordinals):
            raise ValueError("source enumeration has overlapping or missing central-directory dispositions")
        member_payload = [(ordinal, "accepted", "", "") for ordinal in sorted(accepted_ordinals)] + [
            (int(row[0]), str(row[1]), str(row[2]), str(row[3])) for row in disposition_rows
        ]
        member_digest = hashlib.sha256(
            json.dumps(member_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    else:
        # Non-container inputs predate the physical-member denominator. Their
        # one record is also their one physical member; preserve that route's
        # established source-43 digest while making the new columns complete.
        member_count = len(members)
        member_digest = hashlib.sha256(
            json.dumps(tuple(range(member_count)), separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    digest = hashlib.sha256(
        json.dumps([(row[0], row[1].hex()) for row in members], ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if item[3] is not None:
        if tuple(item[1:3]) != (len(members), digest) or item[4:] != (member_count, member_digest):
            raise ValueError("completed source enumeration changed")
        return digest
    conn.execute(
        "UPDATE source_items SET enumerated_record_count=?, enumeration_digest=?, enumerated_at_ms=?, "
        "enumerated_member_count=?, enumeration_member_digest=? "
        "WHERE source_generation_id=? AND source_item_id=?",
        (len(members), digest, enumerated_at_ms, member_count, member_digest, source_generation_id, source_item_id),
    )
    return digest


def record_source_item_member_disposition(
    conn: sqlite3.Connection,
    *,
    source_generation_id: str,
    source_item_id: str,
    entry_ordinal: int,
    member_name: str,
    disposition: SourceItemMemberDisposition,
    diagnostic: str,
    observed_at_ms: int,
) -> None:
    """Persist one refused/unselected central-directory member idempotently."""
    value = require_vocabulary(disposition, SourceItemMemberDisposition, field="member disposition")
    if entry_ordinal < 0:
        raise ValueError("source member ordinal must be non-negative")
    if not member_name:
        raise ValueError("source member name must be non-empty")
    item = conn.execute(
        "SELECT enumerated_at_ms FROM source_items WHERE source_generation_id=? AND source_item_id=?",
        (source_generation_id, source_item_id),
    ).fetchone()
    if item is None:
        raise KeyError(f"unmanifested source item: {source_generation_id}/{source_item_id}")
    if item[0] is not None:
        raise ValueError("completed source enumeration cannot gain member dispositions")
    admitted = conn.execute(
        "SELECT 1 FROM source_item_raw_members m "
        "JOIN raw_container_coordinates c ON c.raw_id=m.raw_id "
        "WHERE m.source_generation_id=? AND m.source_item_id=? AND c.entry_ordinal=?",
        (source_generation_id, source_item_id, entry_ordinal),
    ).fetchone()
    if admitted is not None:
        raise ValueError("source member already has an admitted raw record")
    # The ordinal identifies the entry; retain its exact source name as evidence.
    bounded = bounded_diagnostic(diagnostic, max_len=_MAX_MEMBER_DIAGNOSTIC_CHARS) or ""
    row = conn.execute(
        "SELECT member_name, disposition, diagnostic, observed_at_ms FROM source_item_member_dispositions "
        "WHERE source_generation_id=? AND source_item_id=? AND entry_ordinal=?",
        (source_generation_id, source_item_id, entry_ordinal),
    ).fetchone()
    if row is not None:
        if tuple(row[:3]) != (member_name, value, bounded):
            raise ValueError("source member disposition changed")
        return
    conn.execute(
        "INSERT INTO source_item_member_dispositions "
        "(source_generation_id, source_item_id, entry_ordinal, member_name, disposition, diagnostic, observed_at_ms) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (source_generation_id, source_item_id, entry_ordinal, member_name, value, bounded, observed_at_ms),
    )


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
    "SourceItemMemberDisposition",
    "SourceItemMemberDispositionRecord",
    "SourceItem",
    "FrozenSourceInput",
    "FrozenSourceManifest",
    "complete_source_item_enumeration",
    "publish_source_generation",
    "validate_frozen_source_manifest",
    "publish_frozen_source_manifest",
    "record_source_item_raw_member",
    "record_source_item_member_disposition",
    "seal_source_generation",
    "source_generation_census",
    "source_item_id",
    "transition_source_item",
]
