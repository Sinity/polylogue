"""Compatibility verification + single-record anchor resolution for material protocol v1.

Two entry points:

- ``resolve_anchor``: fetch exactly one record by id, reading only the one
  segment its anchor names, verifying it hashes to what the manifest declared.
  This is the "reconstruct one citation without scanning everything" path.
- ``verify_revision``: a full compatibility pass over every segment the
  manifest declares, run once before trusting a revision wholesale (e.g.
  before ``decode_session_revision`` on untrusted input). Fails closed: any
  mismatch raises a typed ``MaterialProtocolError`` subclass rather than
  returning a boolean.
"""

from __future__ import annotations

from polylogue.core.hashing import hash_bytes
from polylogue.core.json import JSONValue
from polylogue.material_protocol.v1.canonical import canonical_bytes, parse_json_value
from polylogue.material_protocol.v1.errors import (
    AnchorMismatchError,
    AnchorNotFoundError,
    DigestMismatchError,
    RecordCountMismatchError,
    SegmentMissingError,
    SequenceOrderError,
)
from polylogue.material_protocol.v1.manifest import AnchorEntry, RevisionManifest
from polylogue.material_protocol.v1.origin_vocab import check_origin_vocabulary


def resolve_anchor(manifest: RevisionManifest, segment_bytes: dict[int, bytes], record_id: str) -> dict[str, JSONValue]:
    """Resolve *record_id* to its full record, reading only its own segment."""
    anchor = manifest.anchors.get(record_id)
    if anchor is None:
        raise AnchorNotFoundError(f"no anchor for record_id={record_id!r}")

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


def _verify_segment_bytes(manifest: RevisionManifest, segment_bytes: dict[int, bytes]) -> list[bytes]:
    ordered_raw: list[bytes] = []
    for descriptor in sorted(manifest.segments, key=lambda d: d.index):
        raw = segment_bytes.get(descriptor.index)
        if raw is None:
            raise SegmentMissingError(f"segment {descriptor.index} ({descriptor.filename}) not supplied")
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
        ordered_raw.append(raw)
    return ordered_raw


def verify_revision(manifest: RevisionManifest, segment_bytes: dict[int, bytes]) -> None:
    """Full compatibility pass. Raises a MaterialProtocolError subclass on any mismatch."""
    check_origin_vocabulary(manifest.origin_vocabulary_version, manifest.origin_vocabulary_digest)

    ordered_raw = _verify_segment_bytes(manifest, segment_bytes)

    joined = b"".join(ordered_raw)
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

    expected_seq = 0
    actual_kind_counts: dict[str, int] = {}
    actual_anchors: dict[str, AnchorEntry] = {}
    for descriptor in sorted(manifest.segments, key=lambda d: d.index):
        raw = segment_bytes[descriptor.index]
        lines = [line for line in raw.split(b"\n") if line]
        for line_index, line in enumerate(lines):
            parsed = parse_json_value(line)
            if not isinstance(parsed, dict):
                raise SequenceOrderError(f"segment {descriptor.index} line {line_index} is not a JSON object")
            seq = parsed.get("seq")
            if seq != expected_seq:
                raise SequenceOrderError(
                    f"expected seq={expected_seq} at segment {descriptor.index} line {line_index}, got {seq!r}"
                )
            record_id = str(parsed.get("record_id"))
            kind = str(parsed.get("kind"))
            actual_kind_counts[kind] = actual_kind_counts.get(kind, 0) + 1
            actual_anchors[record_id] = AnchorEntry(
                segment_index=descriptor.index,
                line_index=line_index,
                seq=expected_seq,
                kind=kind,
                sha256=hash_bytes(canonical_bytes(parsed)),
            )
            expected_seq += 1

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


__all__ = ["resolve_anchor", "verify_revision"]
