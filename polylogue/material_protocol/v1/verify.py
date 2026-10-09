"""Compatibility verification + single-record anchor resolution for material protocol v1.

Two entry points:

- ``resolve_anchor``: fetch exactly one record by id, reading only the one
  segment its anchor names, verifying it hashes to what the manifest declared.
  This is the "reconstruct one citation without scanning everything" path.
- ``verify_revision``: a full compatibility pass over the head segment plus
  every transcript segment the manifest declares, run once before trusting a
  revision wholesale (e.g. before ``decode_session_revision`` on untrusted
  input). Beyond digests/counts/anchors it enforces cross-record SEMANTIC
  CLOSURE laws so a revision cannot verify while carrying contradictory
  facts:

  - exactly one session record, and its ``message_count`` equals the actual
    number of message records in the transcript;
  - every message record's ``block_count`` equals its actual block records;
  - head contains only session/lineage/usage kinds; transcript only
    message/block/attachment/session_event kinds.

  Fails closed: any mismatch raises a typed ``MaterialProtocolError``
  subclass rather than returning a boolean.
"""

from __future__ import annotations

import json

from polylogue.core.enums import Origin
from polylogue.core.hashing import hash_bytes
from polylogue.core.identity_law import block_id, message_local_id, split_message_local_id
from polylogue.core.json import JSONValue
from polylogue.material_protocol.v1.canonical import canonical_bytes, parse_json_value
from polylogue.material_protocol.v1.constants import HEAD_SEGMENT_INDEX
from polylogue.material_protocol.v1.encode import HEAD_KINDS, SEQUENCE_RULE, TRANSCRIPT_KINDS
from polylogue.material_protocol.v1.errors import (
    AnchorMismatchError,
    AnchorNotFoundError,
    DigestMismatchError,
    MaterialManifestError,
    RecordCountMismatchError,
    SegmentMissingError,
    SemanticClosureError,
    SequenceOrderError,
)
from polylogue.material_protocol.v1.manifest import (
    AnchorEntry,
    RevisionManifest,
    SegmentDescriptor,
    require_current_semantics,
)
from polylogue.material_protocol.v1.origin_vocab import check_origin_vocabulary


def resolve_anchor(manifest: RevisionManifest, segment_bytes: dict[int, bytes], record_id: str) -> dict[str, JSONValue]:
    """Resolve *record_id* to its full record, reading only its own segment.

    Single-record resolution is a read of domain bytes, so it is gated on the
    declared semantics version exactly as the full pass is: anchor digests
    prove the line is intact, never that this reader understands its shape.
    """
    require_current_semantics(manifest)
    check_origin_vocabulary(manifest.origin_vocabulary_version, manifest.origin_vocabulary_digest)
    anchor = manifest.anchors.get(record_id)
    if anchor is None:
        raise AnchorNotFoundError(f"no anchor for record_id={record_id!r}")

    if anchor.segment_index == HEAD_SEGMENT_INDEX:
        descriptor: SegmentDescriptor | None = manifest.head_segment
    else:
        descriptor = next((d for d in manifest.segments if d.index == anchor.segment_index), None)
    if descriptor is None:
        raise AnchorMismatchError(f"anchor for {record_id!r} names unknown segment {anchor.segment_index}")

    raw = segment_bytes.get(anchor.segment_index)
    if raw is None:
        raise SegmentMissingError(f"segment {anchor.segment_index} ({descriptor.filename}) not supplied")

    lines = _check_segment(descriptor, raw)
    if anchor.line_index < 0 or anchor.line_index >= len(lines):
        raise AnchorMismatchError(
            f"anchor for {record_id!r} names line_index={anchor.line_index}, segment has {len(lines)} lines"
        )

    line = lines[anchor.line_index]
    parsed = parse_json_value(line)
    if not isinstance(parsed, dict):
        raise AnchorMismatchError(f"line at anchor for {record_id!r} is not a JSON object")

    actual_sha = hash_bytes(canonical_bytes(parsed))
    if actual_sha != anchor.sha256:
        raise AnchorMismatchError(
            f"anchor sha256 mismatch for {record_id!r}: manifest={anchor.sha256!r}, actual={actual_sha!r}"
        )
    if str(parsed.get("record_id")) != record_id:
        raise AnchorMismatchError(
            f"anchor for {record_id!r} resolved to a record with record_id={parsed.get('record_id')!r}"
        )
    if type(parsed.get("seq")) is not int or parsed.get("seq") != anchor.seq or parsed.get("kind") != anchor.kind:
        raise AnchorMismatchError(
            f"anchor seq mismatch for {record_id!r}: manifest={anchor.seq}, actual={parsed.get('seq')!r}"
        )

    return parsed


def _check_segment(descriptor: SegmentDescriptor, raw: bytes) -> list[bytes]:
    descriptor.require_valid()
    actual_sha = hash_bytes(raw)
    if actual_sha != descriptor.sha256:
        raise DigestMismatchError(
            f"segment {descriptor.index} sha256 mismatch: manifest={descriptor.sha256!r}, actual={actual_sha!r}"
        )
    if len(raw) != descriptor.size_bytes:
        raise DigestMismatchError(
            f"segment {descriptor.index} size mismatch: manifest={descriptor.size_bytes}, actual={len(raw)}"
        )
    lines = [line for line in raw.split(b"\n") if line]
    if len(lines) != descriptor.record_count:
        raise RecordCountMismatchError(
            f"segment {descriptor.index} record_count mismatch: manifest={descriptor.record_count}, actual={len(lines)}"
        )
    return lines


def _walk_records(
    descriptor: SegmentDescriptor,
    lines: list[bytes],
    *,
    expected_seq: int,
    allowed_kinds: frozenset[str],
    space: str,
    kind_counts: dict[str, int],
    anchors: dict[str, AnchorEntry],
) -> tuple[int, list[dict[str, JSONValue]]]:
    parsed_records: list[dict[str, JSONValue]] = []
    for line_index, line in enumerate(lines):
        parsed = parse_json_value(line)
        if not isinstance(parsed, dict):
            raise SequenceOrderError(f"segment {descriptor.index} line {line_index} is not a JSON object")
        seq = parsed.get("seq")
        if type(seq) is not int or seq != expected_seq:
            raise SequenceOrderError(
                f"expected {space} seq={expected_seq} at segment {descriptor.index} line {line_index}, got {seq!r}"
            )
        record_id = parsed.get("record_id")
        kind = parsed.get("kind")
        if not isinstance(record_id, str) or not record_id or not isinstance(kind, str):
            raise SemanticClosureError("material record kind and identity must be nonempty text")
        if record_id in anchors:
            raise SemanticClosureError(f"duplicate material record_id {record_id!r}")
        if kind not in allowed_kinds:
            raise SemanticClosureError(f"record kind {kind!r} is not allowed in the {space} (record_id={record_id!r})")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        anchors[record_id] = AnchorEntry(
            segment_index=descriptor.index,
            line_index=line_index,
            seq=expected_seq,
            kind=kind,
            sha256=hash_bytes(canonical_bytes(parsed)),
        )
        parsed_records.append(parsed)
        expected_seq += 1
    if lines and (descriptor.first_seq != parsed_records[0]["seq"] or descriptor.last_seq != parsed_records[-1]["seq"]):
        raise SequenceOrderError("segment sequence declaration does not match its records")
    return expected_seq, parsed_records


def _check_semantic_closure(
    manifest: RevisionManifest,
    head_records: list[dict[str, JSONValue]],
    transcript_records: list[dict[str, JSONValue]],
) -> None:
    session_records = [record for record in head_records if record.get("kind") == "session"]
    if len(session_records) != 1:
        raise SemanticClosureError(f"expected exactly 1 session record in the head, found {len(session_records)}")
    session = session_records[0]
    if (
        session.get("record_id") != manifest.session_id
        or session.get("session_id") != manifest.session_id
        or session.get("origin") != manifest.origin
        or session.get("native_id") != manifest.native_id
        or manifest.session_id != f"{manifest.origin}:{manifest.native_id}"
        or manifest.revision_id != manifest.content_digest.polylogue_sha256
    ):
        raise SemanticClosureError(
            f"session record_id {session.get('record_id')!r} does not match manifest session_id {manifest.session_id!r}"
        )

    message_records = [record for record in transcript_records if record.get("kind") == "message"]
    messages_by_id: dict[str, dict[str, JSONValue]] = {}
    message_coordinates: set[tuple[int, int]] = set()
    for message in message_records:
        message_id = message.get("message_id")
        if not isinstance(message_id, str) or message.get("record_id") != message_id or message_id in messages_by_id:
            raise SemanticClosureError("message identity is inconsistent or duplicated")
        messages_by_id[message_id] = message
        position, variant = message.get("position"), message.get("variant_index")
        if type(position) is not int or type(variant) is not int or position < 0 or variant < 0:
            raise SemanticClosureError("invalid message ordinal")
        coordinate = (position, variant)
        if coordinate in message_coordinates:
            raise SemanticClosureError("duplicate message ordinal")
        message_coordinates.add(coordinate)
        source_name_json = message.get("source_native_id_json")
        if source_name_json is not None:
            try:
                if not isinstance(source_name_json, str) or not isinstance(json.loads(source_name_json), str):
                    raise ValueError("Source occurrence name is not JSON text")
            except (ValueError, TypeError) as exc:
                raise SemanticClosureError("invalid Source occurrence name") from exc
        native_id = message.get("native_id")
        try:
            if native_id is not None and not isinstance(native_id, str):
                raise ValueError("native message id is not text")
            if native_id:
                local_id = message_local_id(native_id)
            else:
                stored_native, identity, occurrence = split_message_local_id(
                    message_id, parent_session_id=manifest.session_id
                )
                if stored_native is not None:
                    raise ValueError("idless message claims native identity")
                local_id = message_local_id(None, content_identity=identity, content_occurrence=occurrence)
            if message_id != f"{manifest.session_id}:{local_id}":
                raise ValueError("message identity disagrees with its owner or native declaration")
        except ValueError as exc:
            raise SemanticClosureError("invalid message identity") from exc
    for record in (*head_records, *transcript_records):
        kind = record.get("kind")
        if kind == "lineage":
            if record.get("src_session_id") != manifest.session_id:
                raise SemanticClosureError("lineage source belongs to a different session")
        elif record.get("session_id") != manifest.session_id:
            raise SemanticClosureError("material record belongs to a different session")
        if kind in {"block", "attachment"}:
            owner = record.get("message_id")
        elif kind == "session_event":
            owner = record.get("source_message_id")
            if owner is None:
                continue
        else:
            continue
        if not isinstance(owner, str) or owner not in messages_by_id:
            raise SemanticClosureError(f"{kind} names an absent message owner")
        if record["seq"] <= messages_by_id[owner]["seq"]:  # type: ignore[operator]
            raise SemanticClosureError(f"{kind} precedes its message owner")
        if kind == "block":
            identity = record.get("content_identity")
            occurrence = record.get("content_occurrence")
            try:
                if not isinstance(identity, str) or type(occurrence) is not int:
                    raise ValueError("invalid block content coordinates")
                expected_id = block_id(owner, content_identity=identity, content_occurrence=occurrence)
                if record.get("record_id") != expected_id or record.get("block_id") != expected_id:
                    raise ValueError("block identity disagrees with its content coordinates")
            except ValueError as exc:
                raise SemanticClosureError("invalid block identity") from exc
    declared_message_count = session.get("message_count")
    if type(declared_message_count) is not int or declared_message_count != len(message_records):
        raise SemanticClosureError(
            f"session record declares message_count={declared_message_count!r} but the transcript "
            f"contains {len(message_records)} message records"
        )

    blocks_by_message: dict[str, int] = {}
    for record in transcript_records:
        if record.get("kind") == "block":
            blocks_by_message[str(record.get("message_id"))] = (
                blocks_by_message.get(str(record.get("message_id")), 0) + 1
            )
    for message in message_records:
        message_id = str(message.get("message_id"))
        declared_blocks = message.get("block_count")
        actual_blocks = blocks_by_message.get(message_id, 0)
        if type(declared_blocks) is not int or declared_blocks != actual_blocks:
            raise SemanticClosureError(
                f"message {message_id!r} declares block_count={declared_blocks!r} but the transcript "
                f"contains {actual_blocks} block records for it"
            )


def verify_revision(manifest: RevisionManifest, segment_bytes: dict[int, bytes]) -> None:
    """Full compatibility + semantic-closure pass. Raises a MaterialProtocolError subclass on any mismatch."""
    require_current_semantics(manifest)
    check_origin_vocabulary(manifest.origin_vocabulary_version, manifest.origin_vocabulary_digest)
    if manifest.completeness != "complete" or manifest.sequence_rule != SEQUENCE_RULE:
        raise MaterialManifestError("unsupported material completeness or sequence rule")
    try:
        Origin(manifest.origin)
    except ValueError as exc:
        raise MaterialManifestError("unknown material origin") from exc
    if any(type(count) is not int or count < 0 for count in manifest.expected_record_counts.values()):
        raise MaterialManifestError("record counts must be nonnegative exact integers")
    for anchor in manifest.anchors.values():
        if any(type(value) is not int for value in (anchor.segment_index, anchor.line_index, anchor.seq)):
            raise MaterialManifestError("anchor coordinates must be exact integers")
    if [descriptor.index for descriptor in sorted(manifest.segments, key=lambda d: d.index)] != list(
        range(len(manifest.segments))
    ):
        raise MaterialManifestError("transcript segment indexes must be unique and contiguous from zero")

    if manifest.head_segment.index != HEAD_SEGMENT_INDEX:
        raise SemanticClosureError(
            f"manifest head_segment.index must be {HEAD_SEGMENT_INDEX}, got {manifest.head_segment.index}"
        )
    head_raw = segment_bytes.get(HEAD_SEGMENT_INDEX)
    if head_raw is None:
        raise SegmentMissingError(f"head segment ({manifest.head_segment.filename}) not supplied")
    head_lines = _check_segment(manifest.head_segment, head_raw)

    ordered_transcript_raw: list[bytes] = []
    for descriptor in sorted(manifest.segments, key=lambda d: d.index):
        if descriptor.index == HEAD_SEGMENT_INDEX:
            raise SemanticClosureError("transcript segment list must not contain the head segment index")
        raw = segment_bytes.get(descriptor.index)
        if raw is None:
            raise SegmentMissingError(f"segment {descriptor.index} ({descriptor.filename}) not supplied")
        _check_segment(descriptor, raw)
        ordered_transcript_raw.append(raw)

    joined = head_raw + b"".join(ordered_transcript_raw)
    actual_content_sha = hash_bytes(joined)
    if actual_content_sha != manifest.content_digest.polylogue_sha256:
        raise DigestMismatchError(
            "revision content digest mismatch: "
            f"manifest={manifest.content_digest.polylogue_sha256!r}, actual={actual_content_sha!r}"
        )
    if len(joined) != manifest.content_digest.size_bytes:
        raise DigestMismatchError(
            f"revision content size mismatch: manifest={manifest.content_digest.size_bytes}, actual={len(joined)}"
        )

    actual_kind_counts: dict[str, int] = {}
    actual_anchors: dict[str, AnchorEntry] = {}
    _, head_records = _walk_records(
        manifest.head_segment,
        head_lines,
        expected_seq=0,
        allowed_kinds=HEAD_KINDS,
        space="head",
        kind_counts=actual_kind_counts,
        anchors=actual_anchors,
    )

    transcript_records: list[dict[str, JSONValue]] = []
    expected_seq = 0
    for descriptor in sorted(manifest.segments, key=lambda d: d.index):
        raw = segment_bytes[descriptor.index]
        lines = [line for line in raw.split(b"\n") if line]
        expected_seq, records = _walk_records(
            descriptor,
            lines,
            expected_seq=expected_seq,
            allowed_kinds=TRANSCRIPT_KINDS,
            space="transcript",
            kind_counts=actual_kind_counts,
            anchors=actual_anchors,
        )
        transcript_records.extend(records)

    if actual_kind_counts != manifest.expected_record_counts:
        raise RecordCountMismatchError(
            f"expected_record_counts mismatch: manifest={manifest.expected_record_counts!r}, actual={actual_kind_counts!r}"
        )

    if actual_anchors.keys() != manifest.anchors.keys():
        missing = manifest.anchors.keys() - actual_anchors.keys()
        extra = actual_anchors.keys() - manifest.anchors.keys()
        raise AnchorMismatchError(f"anchor key set mismatch: missing={sorted(missing)!r}, extra={sorted(extra)!r}")

    for record_id, expected_anchor in manifest.anchors.items():
        actual_anchor = actual_anchors[record_id]
        if actual_anchor != expected_anchor:
            raise AnchorMismatchError(
                f"anchor mismatch for {record_id!r}: manifest={expected_anchor!r}, actual={actual_anchor!r}"
            )

    _check_semantic_closure(manifest, head_records, transcript_records)


__all__ = ["resolve_anchor", "verify_revision"]
