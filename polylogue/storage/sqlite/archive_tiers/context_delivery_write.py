"""Durable receipts for exact context images that crossed a delivery boundary.

Writer module: user.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Literal, TypeAlias

from polylogue.context.compiler import (
    ContextImage,
    ContextSnapshotRecord,
    canonical_context_image_json,
    context_image_sha256,
)
from polylogue.core.refs import ObjectRef, normalize_object_ref_text

ContextDeliveryWriteOutcome: TypeAlias = Literal["recorded", "idempotent"]


@dataclass(frozen=True, slots=True)
class ArchiveContextDeliveryEnvelope:
    snapshot_ref: str
    recipient_ref: str
    run_ref: str | None
    boundary: str
    inheritance_mode: str
    context_image_sha256: str
    context_image: ContextImage
    segment_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    assertion_refs: tuple[str, ...]
    omissions: tuple[dict[str, object], ...]
    caveats: tuple[str, ...]
    metadata: dict[str, str]
    delivered_by_ref: str
    delivered_at_ms: int
    outcome: ContextDeliveryWriteOutcome = "recorded"


def _now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def _json_list(value: object) -> list[object]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("stored context delivery list field is not a list")
    return parsed


def _json_dict(value: object) -> dict[str, object]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("stored context delivery metadata is not an object")
    return parsed


def _normalized_ref(value: str, *, field: str, kinds: frozenset[str] | None = None) -> str:
    normalized = normalize_object_ref_text(value)
    parsed = ObjectRef.parse(normalized)
    if kinds is not None and parsed.kind not in kinds:
        expected = ", ".join(sorted(kinds))
        raise ValueError(f"{field} must use one of these ref kinds: {expected}")
    return normalized


def write_context_delivery(
    conn: sqlite3.Connection,
    *,
    image: ContextImage,
    record: ContextSnapshotRecord,
    recipient_ref: str,
    delivered_by_ref: str,
    delivered_at_ms: int | None = None,
) -> ArchiveContextDeliveryEnvelope:
    """Persist one exact delivery; exact retries succeed and all drift fails."""

    snapshot_ref = _normalized_ref(record.snapshot_ref, field="snapshot_ref", kinds=frozenset({"context-snapshot"}))
    recipient = _normalized_ref(recipient_ref, field="recipient_ref")
    actor = _normalized_ref(delivered_by_ref, field="delivered_by_ref", kinds=frozenset({"agent", "user"}))
    run_ref = (
        None if record.run_ref is None else _normalized_ref(record.run_ref, field="run_ref", kinds=frozenset({"run"}))
    )
    boundary = record.boundary.strip()
    if not boundary:
        raise ValueError("context delivery boundary must not be empty")

    image_json = canonical_context_image_json(image)
    digest = context_image_sha256(image)
    if record.metadata.get("context_image_sha256") != digest:
        raise ValueError("context snapshot record digest does not match the delivered image")
    segment_refs = tuple(segment.segment_id for segment in image.segments)
    evidence_refs = tuple(ref.format() for ref in image.evidence_refs)
    if tuple(record.segment_refs) != segment_refs:
        raise ValueError("context snapshot record segments do not match the delivered image")
    if tuple(ref.format() for ref in record.evidence_refs) != evidence_refs:
        raise ValueError("context snapshot record evidence refs do not match the delivered image")

    omissions = tuple(item.model_dump(mode="json", exclude_none=False) for item in image.omitted)
    metadata = {str(key): str(value) for key, value in record.metadata.items()}
    requested = {
        "recipient_ref": recipient,
        "run_ref": run_ref,
        "boundary": boundary,
        "inheritance_mode": record.inheritance_mode,
        "context_image_sha256": digest,
        "segment_refs": segment_refs,
        "evidence_refs": evidence_refs,
        "assertion_refs": tuple(image.assertion_refs),
        "omissions": omissions,
        "caveats": tuple(image.caveats),
        "metadata": metadata,
        "delivered_by_ref": actor,
    }
    existing = read_context_delivery(conn, snapshot_ref)
    if existing is not None:
        drift = [name for name, value in requested.items() if getattr(existing, name) != value]
        if delivered_at_ms is not None and existing.delivered_at_ms != delivered_at_ms:
            drift.append("delivered_at_ms")
        if drift:
            raise ValueError(
                "context snapshot ref already exists with different delivery identity: " + ", ".join(drift)
            )
        return replace(existing, outcome="idempotent")

    timestamp = _now_ms() if delivered_at_ms is None else delivered_at_ms
    conn.execute(
        """
        INSERT INTO context_deliveries (
            snapshot_ref, recipient_ref, run_ref, boundary, inheritance_mode,
            context_image_json, context_image_sha256, segment_refs_json,
            evidence_refs_json, assertion_refs_json, omissions_json, caveats_json,
            metadata_json, delivered_by_ref, delivered_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            snapshot_ref,
            recipient,
            run_ref,
            boundary,
            record.inheritance_mode,
            image_json,
            digest,
            json.dumps(segment_refs),
            json.dumps(evidence_refs),
            json.dumps(image.assertion_refs),
            json.dumps(omissions, sort_keys=True),
            json.dumps(image.caveats),
            json.dumps(metadata, sort_keys=True),
            actor,
            timestamp,
        ),
    )
    envelope = read_context_delivery(conn, snapshot_ref)
    if envelope is None:
        raise RuntimeError("context delivery insert did not round-trip")
    return envelope


def read_context_delivery(conn: sqlite3.Connection, snapshot_ref: str) -> ArchiveContextDeliveryEnvelope | None:
    normalized = _normalized_ref(snapshot_ref, field="snapshot_ref", kinds=frozenset({"context-snapshot"}))
    row = conn.execute(
        """
        SELECT snapshot_ref, recipient_ref, run_ref, boundary, inheritance_mode,
               context_image_json, context_image_sha256, segment_refs_json,
               evidence_refs_json, assertion_refs_json, omissions_json, caveats_json,
               metadata_json, delivered_by_ref, delivered_at_ms
        FROM context_deliveries WHERE snapshot_ref = ?
        """,
        (normalized,),
    ).fetchone()
    if row is None:
        return None
    omissions = _json_list(row[10])
    return ArchiveContextDeliveryEnvelope(
        snapshot_ref=str(row[0]),
        recipient_ref=str(row[1]),
        run_ref=None if row[2] is None else str(row[2]),
        boundary=str(row[3]),
        inheritance_mode=str(row[4]),
        context_image=ContextImage.model_validate_json(str(row[5])),
        context_image_sha256=str(row[6]),
        segment_refs=tuple(map(str, _json_list(row[7]))),
        evidence_refs=tuple(map(str, _json_list(row[8]))),
        assertion_refs=tuple(map(str, _json_list(row[9]))),
        omissions=tuple(dict(item) for item in omissions if isinstance(item, dict)),
        caveats=tuple(map(str, _json_list(row[11]))),
        metadata={str(key): str(value) for key, value in _json_dict(row[12]).items()},
        delivered_by_ref=str(row[13]),
        delivered_at_ms=int(row[14]),
    )


def list_context_deliveries(
    conn: sqlite3.Connection, *, recipient_ref: str | None = None, assertion_ref: str | None = None, limit: int = 50
) -> list[ArchiveContextDeliveryEnvelope]:
    if limit < 1:
        raise ValueError("context delivery limit must be positive")
    where: list[str] = []
    params: list[object] = []
    if recipient_ref is not None:
        where.append("recipient_ref = ?")
        params.append(_normalized_ref(recipient_ref, field="recipient_ref"))
    if assertion_ref is not None:
        where.append("EXISTS (SELECT 1 FROM json_each(assertion_refs_json) WHERE value = ?)")
        params.append(_normalized_ref(assertion_ref, field="assertion_ref", kinds=frozenset({"assertion"})))
    clause = " WHERE " + " AND ".join(where) if where else ""
    rows = conn.execute(
        f"SELECT snapshot_ref FROM context_deliveries{clause} ORDER BY delivered_at_ms DESC, snapshot_ref LIMIT ?",
        (*params, limit),
    ).fetchall()
    return [item for row in rows if (item := read_context_delivery(conn, str(row[0]))) is not None]


__all__ = [
    "ArchiveContextDeliveryEnvelope",
    "ContextDeliveryWriteOutcome",
    "list_context_deliveries",
    "read_context_delivery",
    "write_context_delivery",
]
