"""Encode a SessionMaterial into a per-revision head segment plus bounded
immutable transcript NDJSON segments, tied together by a revision manifest.

The revision is split into two record spaces with different mutability
contracts (semantics v2):

**Head** (``head.ndjson``, reserved segment index ``-1``) — the
revision-mutable summary records, re-encoded fresh on EVERY revision:

    0. the session record (title, tags, metadata, updated_at, message_count —
       all legitimately change as a session grows)
    1..k. lineage records, sorted by (dst_origin, dst_native_id, link_type)
       (status/confidence/observed_at are revision-mutable)
    k+1..m. usage records, sorted by model_name (aggregates grow per append)

**Transcript** (``seg-NNNNN.ndjson``) — the immutable observed facts, with
their own strictly-increasing ``seq`` starting at 0:

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

Only transcript segments are ever byte-reused by ``encode_appended_revision``,
and reuse is gated on canonical-byte equality of the whole prior transcript
prefix (via the manifest's per-record anchor hashes), not merely record-id
equality. Mutable facts therefore cannot go stale inside reused bytes: they
live in the head, which is never reused.
"""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.core.hashing import hash_bytes
from polylogue.core.json import JSONValue
from polylogue.material_protocol.v1.canonical import canonical_bytes, canonical_line
from polylogue.material_protocol.v1.constants import (
    CANONICALIZER_VERSION,
    DEFAULT_MAX_RECORDS_PER_SEGMENT,
    HEAD_FILENAME,
    HEAD_SEGMENT_INDEX,
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
    "two seq spaces: head (session, sorted lineage, sorted usage; re-encoded "
    "every revision, never byte-reused) and transcript (strictly-increasing "
    "seq from 0; per-message(transcript order): message, blocks(position), "
    "attachments(position), owned session_events(position); trailing unowned "
    "session_events last; append-only, byte-reuse gated on canonical-byte "
    "prefix equality)"
)

HEAD_KINDS = frozenset({"session", "lineage", "usage"})
TRANSCRIPT_KINDS = frozenset({"message", "block", "attachment", "session_event"})


@dataclass(frozen=True, slots=True)
class EncodedRevision:
    manifest: RevisionManifest
    #: segment_index -> full sealed segment bytes. Includes the head segment
    #: under HEAD_SEGMENT_INDEX plus ALL transcript segments (prior + new).
    segments: dict[int, bytes]

    def segment_filenames(self) -> dict[int, str]:
        filenames = {descriptor.index: descriptor.filename for descriptor in self.manifest.segments}
        filenames[self.manifest.head_segment.index] = self.manifest.head_segment.filename
        return filenames


def _head_records(material: SessionMaterial) -> list[dict[str, JSONValue]]:
    session_id = material.session_id
    records: list[dict[str, JSONValue]] = [session_record(material)]

    for lineage in sorted(
        material.lineage, key=lambda entry: (entry.dst_origin.value, entry.dst_native_id, entry.link_type.value)
    ):
        records.append(lineage_record(session_id, lineage))

    for usage in sorted(material.usage, key=lambda entry: entry.model_name):
        records.append(usage_record(session_id, usage))

    return records


def _transcript_records(material: SessionMaterial) -> list[dict[str, JSONValue]]:
    session_id = material.session_id
    records: list[dict[str, JSONValue]] = []

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


def _pack_head(
    records: list[dict[str, JSONValue]],
) -> tuple[SegmentDescriptor, bytes, dict[str, AnchorEntry], dict[str, int]]:
    """Seal the head segment: its own seq space (0..n-1), one segment, always fresh."""
    buffer = bytearray()
    anchors: dict[str, AnchorEntry] = {}
    kind_counts: dict[str, int] = {}
    for line_index, record in enumerate(records):
        record_with_seq = {**record, "seq": line_index}
        buffer.extend(canonical_line(record_with_seq))
        record_id = str(record["record_id"])
        kind = str(record["kind"])
        anchors[record_id] = AnchorEntry(
            segment_index=HEAD_SEGMENT_INDEX,
            line_index=line_index,
            seq=line_index,
            kind=kind,
            sha256=hash_bytes(canonical_bytes(record_with_seq)),
        )
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    sealed = bytes(buffer)
    descriptor = SegmentDescriptor(
        index=HEAD_SEGMENT_INDEX,
        filename=HEAD_FILENAME,
        sha256=hash_bytes(sealed),
        size_bytes=len(sealed),
        record_count=len(records),
        first_seq=0,
        last_seq=len(records) - 1,
    )
    return descriptor, sealed, anchors, kind_counts


def _content_digest(head_bytes: bytes, transcript_segments_in_order: list[bytes]) -> ContentDigest:
    joined = head_bytes + b"".join(transcript_segments_in_order)
    return ContentDigest(
        polylogue_sha256=hash_bytes(joined),
        canonicalizer_version=CANONICALIZER_VERSION,
        size_bytes=len(joined),
        media_type=SEGMENT_MEDIA_TYPE,
    )


def _build_manifest(
    material: SessionMaterial,
    *,
    head_descriptor: SegmentDescriptor,
    head_bytes: bytes,
    transcript_descriptors: list[SegmentDescriptor],
    transcript_segments: dict[int, bytes],
    anchors: dict[str, AnchorEntry],
    kind_counts: dict[str, int],
    superseded_revision_id: str | None,
    revision_created_at: str,
) -> RevisionManifest:
    ordered_transcript = [transcript_segments[d.index] for d in sorted(transcript_descriptors, key=lambda d: d.index)]
    content_digest = _content_digest(head_bytes, ordered_transcript)
    vocab_version, vocab_digest = resolve_current_origin_vocabulary()
    return RevisionManifest(
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
        head_segment=head_descriptor,
        segments=tuple(sorted(transcript_descriptors, key=lambda d: d.index)),
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


def encode_session_revision(
    material: SessionMaterial,
    *,
    revision_created_at: str,
    superseded_revision_id: str | None = None,
    max_records_per_segment: int = DEFAULT_MAX_RECORDS_PER_SEGMENT,
) -> EncodedRevision:
    """Encode a fresh (non-append) revision for *material*."""
    head_descriptor, head_bytes, head_anchors, head_kind_counts = _pack_head(_head_records(material))
    transcript = _transcript_records(material)
    descriptors, segment_bytes, anchors, kind_counts = _pack_segments(
        transcript, start_seq=0, start_segment_index=0, max_records_per_segment=max_records_per_segment
    )

    all_anchors = dict(head_anchors)
    all_anchors.update(anchors)
    all_kind_counts = dict(head_kind_counts)
    for kind, count in kind_counts.items():
        all_kind_counts[kind] = all_kind_counts.get(kind, 0) + count

    manifest = _build_manifest(
        material,
        head_descriptor=head_descriptor,
        head_bytes=head_bytes,
        transcript_descriptors=descriptors,
        transcript_segments=segment_bytes,
        anchors=all_anchors,
        kind_counts=all_kind_counts,
        superseded_revision_id=superseded_revision_id,
        revision_created_at=revision_created_at,
    )
    all_segments = dict(segment_bytes)
    all_segments[HEAD_SEGMENT_INDEX] = head_bytes
    return EncodedRevision(manifest=manifest, segments=all_segments)


def encode_appended_revision(
    prior_manifest: RevisionManifest,
    prior_segments: dict[int, bytes],
    full_material: SessionMaterial,
    *,
    revision_created_at: str,
    max_records_per_segment: int = DEFAULT_MAX_RECORDS_PER_SEGMENT,
) -> EncodedRevision:
    """Encode a new revision that extends *prior_manifest* with trailing transcript records.

    The head (session/lineage/usage summary) is ALWAYS re-encoded fresh from
    ``full_material`` — mutable summary facts never survive by byte reuse.

    Transcript reuse requires that ``full_material``'s transcript record list
    reproduces the prior revision's transcript prefix by CANONICAL BYTES
    (checked against the manifest's per-record anchor hashes), not merely by
    record id — a changed record with a stable id is an edit, not an append.
    Otherwise this raises ``NotAnAppendError`` and callers must use
    ``encode_session_revision`` (getting a fresh, unrelated segmentation
    linked only via ``superseded_revision_id``) instead of claiming
    append-anchor-stability.
    """
    # Session identity must match the prior revision explicitly: with the
    # session record living in the (never-compared) head, an empty-transcript
    # prior would otherwise accept a completely different session as an
    # "append" of the first.
    if (
        full_material.session_id != prior_manifest.session_id
        or full_material.origin.value != prior_manifest.origin
        or full_material.native_id != prior_manifest.native_id
    ):
        raise NotAnAppendError(
            f"session identity changed: prior=({prior_manifest.session_id!r}, {prior_manifest.origin!r}, "
            f"{prior_manifest.native_id!r}), full_material=({full_material.session_id!r}, "
            f"{full_material.origin.value!r}, {full_material.native_id!r})"
        )
    prior_transcript_count = sum(segment.record_count for segment in prior_manifest.segments)
    prior_anchors_by_seq: dict[int, tuple[str, AnchorEntry]] = {
        anchor.seq: (record_id, anchor)
        for record_id, anchor in prior_manifest.anchors.items()
        if anchor.segment_index != HEAD_SEGMENT_INDEX
    }

    full_transcript = _transcript_records(full_material)
    if len(full_transcript) < prior_transcript_count:
        raise NotAnAppendError(
            f"full_material has fewer transcript records ({len(full_transcript)}) "
            f"than the prior revision ({prior_transcript_count})"
        )
    for seq in range(prior_transcript_count):
        expected = prior_anchors_by_seq.get(seq)
        if expected is None:
            raise NotAnAppendError(f"prior manifest has no transcript anchor for seq={seq}")
        expected_id, expected_anchor = expected
        candidate = full_transcript[seq]
        candidate_id = str(candidate["record_id"])
        if candidate_id != expected_id:
            raise NotAnAppendError(
                f"transcript record at seq={seq} changed from {expected_id!r} to {candidate_id!r}; not a pure append"
            )
        candidate_sha = hash_bytes(canonical_bytes({**candidate, "seq": seq}))
        if candidate_sha != expected_anchor.sha256:
            raise NotAnAppendError(
                f"transcript record {candidate_id!r} at seq={seq} changed canonical bytes "
                f"(prior sha256={expected_anchor.sha256!r}, now {candidate_sha!r}); an edit is not an append"
            )

    head_descriptor, head_bytes, head_anchors, head_kind_counts = _pack_head(_head_records(full_material))

    new_records = full_transcript[prior_transcript_count:]
    start_segment_index = len(prior_manifest.segments)
    new_descriptors, new_segment_bytes, new_anchors, new_kind_counts = _pack_segments(
        new_records,
        start_seq=prior_transcript_count,
        start_segment_index=start_segment_index,
        max_records_per_segment=max_records_per_segment,
    )

    all_descriptors = list(prior_manifest.segments) + new_descriptors
    transcript_segments = {descriptor.index: prior_segments[descriptor.index] for descriptor in prior_manifest.segments}
    transcript_segments.update(new_segment_bytes)

    all_anchors = dict(head_anchors)
    for record_id, anchor in prior_manifest.anchors.items():
        if anchor.segment_index != HEAD_SEGMENT_INDEX:
            all_anchors[record_id] = anchor
    all_anchors.update(new_anchors)

    all_kind_counts = dict(head_kind_counts)
    prior_transcript_counts = {
        kind: count for kind, count in prior_manifest.expected_record_counts.items() if kind in TRANSCRIPT_KINDS
    }
    for kind, count in prior_transcript_counts.items():
        all_kind_counts[kind] = all_kind_counts.get(kind, 0) + count
    for kind, count in new_kind_counts.items():
        all_kind_counts[kind] = all_kind_counts.get(kind, 0) + count

    manifest = _build_manifest(
        full_material,
        head_descriptor=head_descriptor,
        head_bytes=head_bytes,
        transcript_descriptors=all_descriptors,
        transcript_segments=transcript_segments,
        anchors=all_anchors,
        kind_counts=all_kind_counts,
        superseded_revision_id=prior_manifest.revision_id,
        revision_created_at=revision_created_at,
    )
    all_segments = dict(transcript_segments)
    all_segments[HEAD_SEGMENT_INDEX] = head_bytes
    return EncodedRevision(manifest=manifest, segments=all_segments)


__all__ = [
    "EncodedRevision",
    "HEAD_KINDS",
    "TRANSCRIPT_KINDS",
    "encode_appended_revision",
    "encode_session_revision",
]
