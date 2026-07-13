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

from polylogue.core.hashing import hash_bytes
from polylogue.core.json import JSONValue
from polylogue.material_protocol.v1.canonical import canonical_bytes, parse_json_value
from polylogue.material_protocol.v1.constants import HEAD_SEGMENT_INDEX
from polylogue.material_protocol.v1.encode import HEAD_KINDS, TRANSCRIPT_KINDS
from polylogue.material_protocol.v1.errors import (
    AnchorMismatchError,
    AnchorNotFoundError,
    DigestMismatchError,
    RecordCountMismatchError,
    SegmentMissingError,
    SemanticClosureError,
    SequenceOrderError,
)
from polylogue.material_protocol.v1.manifest import AnchorEntry, RevisionManifest, SegmentDescriptor
from polylogue.material_protocol.v1.origin_vocab import check_origin_vocabulary


def resolve_anchor(manifest: RevisionManifest, segment_bytes: dict[int, bytes], record_id: str) -> dict[str, JSONValue]:
    """Resolve *record_id* to its full record, reading only its own segment."""
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

    lines = [line for line in raw.split(b"\n") if line]
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
    if int(parsed.get("seq", -1)) != anchor.seq:  # type: ignore[arg-type]
        raise AnchorMismatchError(
            f"anchor seq mismatch for {record_id!r}: manifest={anchor.seq}, actual={parsed.get('seq')!r}"
        )

    return parsed


def _check_segment(descriptor: SegmentDescriptor, raw: bytes) -> list[bytes]:
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
        if seq != expected_seq:
            raise SequenceOrderError(
                f"expected {space} seq={expected_seq} at segment {descriptor.index} line {line_index}, got {seq!r}"
            )
        record_id = str(parsed.get("record_id"))
        kind = str(parsed.get("kind"))
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
    if str(session.get("record_id")) != manifest.session_id:
        raise SemanticClosureError(
            f"session record_id {session.get('record_id')!r} does not match manifest session_id {manifest.session_id!r}"
        )

    message_records = [record for record in transcript_records if record.get("kind") == "message"]
    declared_message_count = session.get("message_count")
    if declared_message_count != len(message_records):
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
        if declared_blocks != actual_blocks:
            raise SemanticClosureError(
                f"message {message_id!r} declares block_count={declared_blocks!r} but the transcript "
                f"contains {actual_blocks} block records for it"
            )


def verify_revision(manifest: RevisionManifest, segment_bytes: dict[int, bytes]) -> None:
    """Full compatibility + semantic-closure pass. Raises a MaterialProtocolError subclass on any mismatch."""
    check_origin_vocabulary(manifest.origin_vocabulary_version, manifest.origin_vocabulary_digest)

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
