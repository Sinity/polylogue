"""Reconstruct a session from manifest + segment bytes -- no archive DB involved.

``decode_session_revision`` trusts its input bytes structurally (strict
sequencing is still enforced because reconstruction order depends on it);
tamper/corruption detection against the manifest's declared digests/anchors
is ``verify.py``'s job and should run before decode on any untrusted input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from polylogue.core.json import JSONValue
from polylogue.material_protocol.v1.canonical import parse_json_value
from polylogue.material_protocol.v1.errors import SegmentMissingError, SequenceOrderError
from polylogue.material_protocol.v1.manifest import RevisionManifest


@dataclass
class DecodedMessage:
    message_id: str
    native_id: str | None
    position: int
    variant_index: int
    role: str
    message_type: str
    material_origin: str
    text: str | None
    occurred_at_ms: int | None
    model_name: str | None
    parent_message_id: str | None
    usage: dict[str, JSONValue]
    blocks: list[dict[str, JSONValue]] = field(default_factory=list)
    attachments: list[dict[str, JSONValue]] = field(default_factory=list)
    session_events: list[dict[str, JSONValue]] = field(default_factory=list)


@dataclass
class DecodedSession:
    session: dict[str, JSONValue]
    messages: list[DecodedMessage] = field(default_factory=list)
    lineage: list[dict[str, JSONValue]] = field(default_factory=list)
    usage: list[dict[str, JSONValue]] = field(default_factory=list)
    trailing_session_events: list[dict[str, JSONValue]] = field(default_factory=list)


def iter_records(manifest: RevisionManifest, segment_bytes: dict[int, bytes]) -> list[dict[str, JSONValue]]:
    """Return every record in the revision, in strict seq order, from raw segment bytes."""
    records: list[dict[str, JSONValue]] = []
    expected_seq = 0
    for descriptor in sorted(manifest.segments, key=lambda d: d.index):
        raw = segment_bytes.get(descriptor.index)
        if raw is None:
            raise SegmentMissingError(f"segment {descriptor.index} ({descriptor.filename}) not supplied")
        for raw_line in raw.split(b"\n"):
            if not raw_line:
                continue
            parsed = parse_json_value(raw_line)
            if not isinstance(parsed, dict):
                raise SequenceOrderError(f"segment {descriptor.index} contains a non-object record")
            seq = parsed.get("seq")
            if seq != expected_seq:
                raise SequenceOrderError(f"expected seq={expected_seq}, got seq={seq!r} in segment {descriptor.index}")
            records.append(parsed)
            expected_seq += 1
    return records


def decode_session_revision(manifest: RevisionManifest, segment_bytes: dict[int, bytes]) -> DecodedSession:
    records = iter_records(manifest, segment_bytes)

    session_records = [r for r in records if r["kind"] == "session"]
    if len(session_records) != 1:
        raise SequenceOrderError(f"expected exactly 1 session record, found {len(session_records)}")
    decoded = DecodedSession(session=session_records[0])

    messages_by_id: dict[str, DecodedMessage] = {}
    message_order: list[str] = []

    for record in records:
        kind = record["kind"]
        if kind == "session":
            continue
        if kind == "lineage":
            decoded.lineage.append(record)
        elif kind == "usage":
            decoded.usage.append(record)
        elif kind == "message":
            message_id = str(record["message_id"])
            message_order.append(message_id)
            usage_payload = record["usage"]
            assert isinstance(usage_payload, dict)
            native_id = record.get("native_id")
            parent_message_id = record.get("parent_message_id")
            messages_by_id[message_id] = DecodedMessage(
                message_id=message_id,
                native_id=str(native_id) if native_id is not None else None,
                position=int(record["position"]),  # type: ignore[arg-type]
                variant_index=int(record["variant_index"]),  # type: ignore[arg-type]
                role=str(record["role"]),
                message_type=str(record["message_type"]),
                material_origin=str(record["material_origin"]),
                text=str(record["text"]) if record.get("text") is not None else None,
                occurred_at_ms=int(record["occurred_at_ms"]) if record.get("occurred_at_ms") is not None else None,  # type: ignore[arg-type]
                model_name=str(record["model_name"]) if record.get("model_name") is not None else None,
                parent_message_id=str(parent_message_id) if parent_message_id is not None else None,
                usage=usage_payload,
            )
        elif kind == "block":
            message_id = str(record["message_id"])
            messages_by_id[message_id].blocks.append(record)
        elif kind == "attachment":
            message_id = str(record["message_id"])
            messages_by_id[message_id].attachments.append(record)
        elif kind == "session_event":
            source_message_id = record.get("source_message_id")
            if source_message_id is not None and str(source_message_id) in messages_by_id:
                messages_by_id[str(source_message_id)].session_events.append(record)
            else:
                decoded.trailing_session_events.append(record)
        else:
            raise SequenceOrderError(f"unknown record kind {kind!r}")

    decoded.messages = [messages_by_id[message_id] for message_id in message_order]
    return decoded


__all__ = ["DecodedMessage", "DecodedSession", "decode_session_revision", "iter_records"]
