"""Encode a SessionMaterial into bounded immutable NDJSON segments + a revision manifest.

Ordering contract (this *is* the "sequence rule" recorded in the manifest):
one global, strictly increasing ``seq`` counter across the whole revision,
assigned in this fixed walk order:

    0. the session record
    1..k. lineage records, sorted by (dst_origin, dst_native_id, link_type)
    k+1..m. usage records, sorted by model_name
    for each message in ``material.messages`` order (transcript order --
    NOT re-sorted by timestamp, because equal/missing timestamps can't
    total-order a transcript; position/variant_index is what's authoritative):
        the message record
        its block records, in block position order
        its attachment records, in attachment position order
        session_event records whose source_message matches this message,
            in event position order
    finally, any session_event records with no matching source message,
        in event position order

This walk order is deliberately append-friendly: growing a session by
appending trailing messages (the common live-watcher case) only ever adds
records after every previously-emitted record, which is what lets
``encode_appended_revision`` reuse prior segments byte-for-byte.
"""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.core.hashing import hash_bytes
from polylogue.core.json import JSONValue
from polylogue.material_protocol.v1.canonical import canonical_bytes, canonical_line
from polylogue.material_protocol.v1.constants import (
    CANONICALIZER_VERSION,
    DEFAULT_MAX_RECORDS_PER_SEGMENT,
    PROTOCOL_VERSION,
    SEGMENT_FILENAME_TEMPLATE,
    SEGMENT_MEDIA_TYPE,
    SEMANTICS_VERSION,
)
from polylogue.material_protocol.v1.errors import NotAnAppendError
from polylogue.material_protocol.v1.input_model import SessionEventInput, SessionMaterial
from polylogue.material_protocol.v1.manifest import (
    AnchorEntry,
    ContentDigest,
    FidelityGap,
    RevisionManifest,
    SegmentDescriptor,
)
from polylogue.material_protocol.v1.origin_vocab import resolve_current_origin_vocabulary
from polylogue.material_protocol.v1.records import (
    attachment_record,
    block_record,
    lineage_record,
    message_id_for,
    message_record,
    session_event_record,
    session_record,
    usage_record,
)

SEQUENCE_RULE = (
    "global strictly-increasing seq per revision; order = session, "
    "sorted lineage, sorted usage, then per-message(transcript order): "
    "message, blocks(position), attachments(position), "
    "owned session_events(position); trailing unowned session_events last"
)


@dataclass(frozen=True, slots=True)
class EncodedRevision:
    manifest: RevisionManifest
    segments: dict[int, bytes]  # segment_index -> full sealed segment bytes (includes ALL segments, prior+new)

    def segment_filenames(self) -> dict[int, str]:
        return {descriptor.index: descriptor.filename for descriptor in self.manifest.segments}


def _ordered_records(material: SessionMaterial) -> list[dict[str, JSONValue]]:
    session_id = material.session_id
    records: list[dict[str, JSONValue]] = [session_record(material)]

    for lineage in sorted(
        material.lineage, key=lambda entry: (entry.dst_origin.value, entry.dst_native_id, entry.link_type.value)
    ):
        records.append(lineage_record(session_id, lineage))

    for usage in sorted(material.usage, key=lambda entry: entry.model_name):
        records.append(usage_record(session_id, usage))

    owned_events: dict[str, list[SessionEventInput]] = {}
    unowned_events: list[SessionEventInput] = []
    for event in material.session_events:
        native_id = event.source_message_native_id
        if native_id is None:
            unowned_events.append(event)
        else:
            owned_events.setdefault(native_id, []).append(event)
    for bucket in owned_events.values():
        bucket.sort(key=lambda event: event.position)

    for message in material.messages:
        message_id = message_id_for(session_id, message)
        records.append(message_record(session_id, message))
        for block in sorted(message.blocks, key=lambda b: b.position):
            records.append(block_record(session_id, message_id, block))
        for attachment in sorted(message.attachments, key=lambda a: a.position):
            records.append(attachment_record(session_id, message_id, attachment))
        owned = owned_events.pop(message.native_id, []) if message.native_id is not None else []
        for event in owned:
            records.append(session_event_record(session_id, event, source_message_id=message_id))

    # Any remaining events either had no source message, or named a native_id
    # that isn't in this revision's message set -- both cases are "unowned"
    # for anchoring purposes and go last, in event position order.
    trailing_events = unowned_events + [event for bucket in owned_events.values() for event in bucket]
    for event in sorted(trailing_events, key=lambda event: event.position):
        records.append(session_event_record(session_id, event, source_message_id=None))

    return records


def _pack_segments(
    records: list[dict[str, JSONValue]],
    *,
    start_seq: int,
    start_segment_index: int,
    max_records_per_segment: int,
) -> tuple[list[SegmentDescriptor], dict[int, bytes], dict[str, AnchorEntry], dict[str, int]]:
    descriptors: list[SegmentDescriptor] = []
    segment_bytes: dict[int, bytes] = {}
    anchors: dict[str, AnchorEntry] = {}
    kind_counts: dict[str, int] = {}

    seq = start_seq
    segment_index = start_segment_index
    for chunk_start in range(0, len(records), max_records_per_segment):
        chunk = records[chunk_start : chunk_start + max_records_per_segment]
        buffer = bytearray()
        first_seq = seq
        for line_index, record in enumerate(chunk):
            record_with_seq = {**record, "seq": seq}
            line_bytes = canonical_line(record_with_seq)
            buffer.extend(line_bytes)
            record_id = str(record["record_id"])
            kind = str(record["kind"])
            anchors[record_id] = AnchorEntry(
                segment_index=segment_index,
                line_index=line_index,
                seq=seq,
                kind=kind,
                sha256=hash_bytes(canonical_bytes(record_with_seq)),
            )
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
            seq += 1
        sealed = bytes(buffer)
        segment_bytes[segment_index] = sealed
        descriptors.append(
            SegmentDescriptor(
                index=segment_index,
                filename=SEGMENT_FILENAME_TEMPLATE.format(index=segment_index),
                sha256=hash_bytes(sealed),
                size_bytes=len(sealed),
                record_count=len(chunk),
                first_seq=first_seq,
                last_seq=seq - 1,
            )
        )
        segment_index += 1

    return descriptors, segment_bytes, anchors, kind_counts


def _content_digest(segments_in_order: list[bytes]) -> ContentDigest:
    joined = b"".join(segments_in_order)
    return ContentDigest(
        polylogue_sha256=hash_bytes(joined),
        canonicalizer_version=CANONICALIZER_VERSION,
        size_bytes=len(joined),
        media_type=SEGMENT_MEDIA_TYPE,
    )


def encode_session_revision(
    material: SessionMaterial,
    *,
    revision_created_at: str,
    superseded_revision_id: str | None = None,
    max_records_per_segment: int = DEFAULT_MAX_RECORDS_PER_SEGMENT,
) -> EncodedRevision:
    """Encode a fresh (non-append) revision for *material*."""
    records = _ordered_records(material)
    descriptors, segment_bytes, anchors, kind_counts = _pack_segments(
        records, start_seq=0, start_segment_index=0, max_records_per_segment=max_records_per_segment
    )
    ordered_segment_bytes = [segment_bytes[d.index] for d in descriptors]
    content_digest = _content_digest(ordered_segment_bytes)
    vocab_version, vocab_digest = resolve_current_origin_vocabulary()

    manifest = RevisionManifest(
        protocol_version=PROTOCOL_VERSION,
        semantics_version=SEMANTICS_VERSION,
        origin_vocabulary_version=vocab_version,
        origin_vocabulary_digest=vocab_digest,
        session_id=material.session_id,
        origin=material.origin.value,
        native_id=material.native_id,
        revision_id=content_digest.polylogue_sha256,
        superseded_revision_id=superseded_revision_id,
        content_digest=content_digest,
        segments=tuple(descriptors),
        expected_record_counts=kind_counts,
        anchors=anchors,
        sequence_rule=SEQUENCE_RULE,
        completeness="complete",
        fidelity_gaps=tuple(
            FidelityGap(scope=gap.scope, record_id=gap.record_id, gap_kind=gap.gap_kind, detail=gap.detail)
            for gap in material.fidelity_gaps
        ),
        revision_created_at=revision_created_at,
    )
    return EncodedRevision(manifest=manifest, segments=segment_bytes)


def encode_appended_revision(
    prior_manifest: RevisionManifest,
    prior_segments: dict[int, bytes],
    full_material: SessionMaterial,
    *,
    revision_created_at: str,
    max_records_per_segment: int = DEFAULT_MAX_RECORDS_PER_SEGMENT,
) -> EncodedRevision:
    """Encode a new revision that extends *prior_manifest* with trailing records only.

    Requires that ``full_material``'s ordered record list reproduces the prior
    revision's record_id sequence exactly as a prefix (same records, same
    order, same seq assignment) -- otherwise this isn't an append, it's a
    resegmentation/edit, and callers must use ``encode_session_revision``
    (getting a fresh, unrelated segmentation linked only via
    ``superseded_revision_id``) instead of claiming append-anchor-stability.
    """
    prior_record_count = sum(segment.record_count for segment in prior_manifest.segments)
    prior_records_by_seq: dict[int, str] = {
        anchor.seq: record_id for record_id, anchor in prior_manifest.anchors.items()
    }

    full_records = _ordered_records(full_material)
    if len(full_records) < prior_record_count:
        raise NotAnAppendError(
            f"full_material has fewer records ({len(full_records)}) than the prior revision ({prior_record_count})"
        )
    for seq in range(prior_record_count):
        candidate_id = str(full_records[seq]["record_id"])
        expected_id = prior_records_by_seq.get(seq)
        if candidate_id != expected_id:
            raise NotAnAppendError(
                f"record at seq={seq} changed from {expected_id!r} to {candidate_id!r}; not a pure append"
            )

    new_records = full_records[prior_record_count:]
    start_segment_index = len(prior_manifest.segments)
    new_descriptors, new_segment_bytes, new_anchors, new_kind_counts = _pack_segments(
        new_records,
        start_seq=prior_record_count,
        start_segment_index=start_segment_index,
        max_records_per_segment=max_records_per_segment,
    )

    all_descriptors = list(prior_manifest.segments) + new_descriptors
    all_segments = dict(prior_segments)
    all_segments.update(new_segment_bytes)
    ordered_segment_bytes = [all_segments[d.index] for d in all_descriptors]
    content_digest = _content_digest(ordered_segment_bytes)

    all_anchors = dict(prior_manifest.anchors)
    all_anchors.update(new_anchors)

    all_kind_counts = dict(prior_manifest.expected_record_counts)
    for kind, count in new_kind_counts.items():
        all_kind_counts[kind] = all_kind_counts.get(kind, 0) + count

    vocab_version, vocab_digest = resolve_current_origin_vocabulary()
    manifest = RevisionManifest(
        protocol_version=PROTOCOL_VERSION,
        semantics_version=SEMANTICS_VERSION,
        origin_vocabulary_version=vocab_version,
        origin_vocabulary_digest=vocab_digest,
        session_id=full_material.session_id,
        origin=full_material.origin.value,
        native_id=full_material.native_id,
        revision_id=content_digest.polylogue_sha256,
        superseded_revision_id=prior_manifest.revision_id,
        content_digest=content_digest,
        segments=tuple(all_descriptors),
        expected_record_counts=all_kind_counts,
        anchors=all_anchors,
        sequence_rule=SEQUENCE_RULE,
        completeness="complete",
        fidelity_gaps=tuple(
            FidelityGap(scope=gap.scope, record_id=gap.record_id, gap_kind=gap.gap_kind, detail=gap.detail)
            for gap in full_material.fidelity_gaps
        ),
        revision_created_at=revision_created_at,
    )
    return EncodedRevision(manifest=manifest, segments=all_segments)


__all__ = ["EncodedRevision", "encode_appended_revision", "encode_session_revision"]
