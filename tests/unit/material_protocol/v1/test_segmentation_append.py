"""Large-session segmentation + append-preserves-anchors proof (polylogue-303r.1 AC).

Two properties under test:

1. A session larger than one segment's record bound seals multiple bounded
   immutable segments.
2. Appending trailing messages (the common live-watcher growth pattern)
   produces a new revision whose manifest reuses every prior segment's exact
   descriptor and every prior record's exact anchor -- byte-for-byte, not
   just value-equal by coincidence -- while a *non-append* fresh encode of
   different content is an unrelated revision that never touches the first
   revision's bytes at all.
"""

from __future__ import annotations

from polylogue.material_protocol.v1 import (
    EncodedRevision,
    encode_appended_revision,
    encode_session_revision,
    verify_revision,
)
from tests.unit.material_protocol.v1.fixture import build_large_session_material


def _encode(message_count: int) -> EncodedRevision:
    material = build_large_session_material(message_count)
    return encode_session_revision(material, revision_created_at="2026-07-12T00:00:00Z", max_records_per_segment=10)


def test_large_session_seals_multiple_segments() -> None:
    encoded = _encode(45)
    assert len(encoded.manifest.segments) > 1
    for descriptor in encoded.manifest.segments:
        assert descriptor.record_count <= 10


def test_append_reuses_prior_segments_byte_for_byte() -> None:
    prior = _encode(45)
    full_material = build_large_session_material(90)

    appended = encode_appended_revision(
        prior.manifest,
        prior.segments,
        full_material,
        revision_created_at="2026-07-12T01:00:00Z",
        max_records_per_segment=10,
    )

    verify_revision(appended.manifest, appended.segments)
    assert appended.manifest.superseded_revision_id == prior.manifest.revision_id
    assert appended.manifest.revision_id != prior.manifest.revision_id
    assert len(appended.manifest.segments) > len(prior.manifest.segments)

    prior_segment_by_index = {d.index: d for d in prior.manifest.segments}
    appended_segment_by_index = {d.index: d for d in appended.manifest.segments}
    for index, prior_descriptor in prior_segment_by_index.items():
        assert appended_segment_by_index[index] == prior_descriptor
        assert appended.segments[index] == prior.segments[index]

    head_index = prior.manifest.head_segment.index
    for record_id, prior_anchor in prior.manifest.anchors.items():
        if prior_anchor.segment_index == head_index:
            continue  # head anchors are revision-local: the head is re-encoded fresh
        assert appended.manifest.anchors[record_id] == prior_anchor

    # The head is NEVER byte-reused: the appended revision's summary is current.
    assert appended.segments[head_index] != prior.segments[head_index]


def test_append_twice_keeps_stacking_stable_prefixes() -> None:
    revision_1 = _encode(12)
    revision_2 = encode_appended_revision(
        revision_1.manifest,
        revision_1.segments,
        build_large_session_material(24),
        revision_created_at="t2",
        max_records_per_segment=10,
    )
    revision_3 = encode_appended_revision(
        revision_2.manifest,
        revision_2.segments,
        build_large_session_material(36),
        revision_created_at="t3",
        max_records_per_segment=10,
    )

    verify_revision(revision_3.manifest, revision_3.segments)
    head_index = revision_1.manifest.head_segment.index
    for record_id, anchor in revision_1.manifest.anchors.items():
        if anchor.segment_index == head_index:
            continue
        assert revision_3.manifest.anchors[record_id] == anchor
    for record_id, anchor in revision_2.manifest.anchors.items():
        if anchor.segment_index == head_index:
            continue
        assert revision_3.manifest.anchors[record_id] == anchor
    assert revision_3.manifest.superseded_revision_id == revision_2.manifest.revision_id


def test_a_regenerated_non_append_revision_never_touches_prior_bytes() -> None:
    """Regenerated provider files get a new manifest; old material is untouched, not rewritten."""
    prior = _encode(20)
    prior_segments_snapshot = dict(prior.segments)

    regenerated_material = build_large_session_material(20, native_prefix="regenerated")
    regenerated = encode_session_revision(
        regenerated_material,
        revision_created_at="2026-07-12T02:00:00Z",
        superseded_revision_id=prior.manifest.revision_id,
        max_records_per_segment=10,
    )

    # The prior EncodedRevision's own segment bytes are unaffected (still the
    # exact dict we started with -- nothing here mutates in place).
    assert prior.segments == prior_segments_snapshot

    # The regenerated revision has its own independent segmentation/anchors.
    assert set(regenerated.manifest.anchors) != set(prior.manifest.anchors)
    assert regenerated.manifest.superseded_revision_id == prior.manifest.revision_id
    verify_revision(regenerated.manifest, regenerated.segments)
    verify_revision(prior.manifest, prior.segments)


def test_append_keeps_mutable_summary_current_in_the_head() -> None:
    """Regression for the append semantic-closure bug (2026-07-13 review).

    The old encoder reused the segment holding the session record byte-for-
    byte whenever record IDs matched, so an appended revision could verify
    while its session record still claimed the OLD message_count. The head/
    transcript split re-encodes the summary every revision; this pins the
    exact contradiction the report reproduced.

    Anti-vacuity: reverting encode_appended_revision to id-gated whole-prefix
    reuse (or moving session records back into transcript segments) makes the
    message_count assertion fail; removing the verifier's semantic-closure
    law makes the tampered-head fixture below pass.
    """
    import json

    prior = _encode(2)
    appended = encode_appended_revision(
        prior.manifest,
        prior.segments,
        build_large_session_material(4),
        revision_created_at="2026-07-12T03:00:00Z",
        max_records_per_segment=10,
    )
    verify_revision(appended.manifest, appended.segments)

    head_index = appended.manifest.head_segment.index
    head_records = [json.loads(line) for line in appended.segments[head_index].decode().splitlines()]
    session = next(record for record in head_records if record["kind"] == "session")
    assert session["message_count"] == 4
    assert appended.manifest.expected_record_counts["message"] == 4
    # transcript prefix stayed byte-identical
    for descriptor in prior.manifest.segments:
        assert appended.segments[descriptor.index] == prior.segments[descriptor.index]


def test_append_rejects_edited_prefix_even_with_stable_record_ids() -> None:
    """A changed record with a stable id is an edit, not an append.

    Anti-vacuity: weakening the append gate back to record-id comparison
    accepts this material and silently serves stale transcript bytes.
    """
    import dataclasses

    import pytest

    from polylogue.material_protocol.v1 import NotAnAppendError

    prior = _encode(2)
    full = build_large_session_material(4)
    edited_messages = list(full.messages)
    edited_messages[0] = dataclasses.replace(edited_messages[0], text="EDITED " + (edited_messages[0].text or ""))
    edited = dataclasses.replace(full, messages=tuple(edited_messages))

    with pytest.raises(NotAnAppendError, match="changed canonical bytes"):
        encode_appended_revision(
            prior.manifest,
            prior.segments,
            edited,
            revision_created_at="2026-07-12T04:00:00Z",
            max_records_per_segment=10,
        )


def test_verifier_rejects_contradictory_session_message_count() -> None:
    """Semantic-closure law: session.message_count must equal actual message records.

    Builds a self-consistent revision whose head was re-sealed with a wrong
    message_count (digests, anchors, and counts all recomputed so ONLY the
    cross-record law can catch it).
    """
    import dataclasses
    import json

    import pytest

    from polylogue.core.hashing import hash_bytes
    from polylogue.material_protocol.v1 import SemanticClosureError
    from polylogue.material_protocol.v1.canonical import canonical_bytes, canonical_line
    from polylogue.material_protocol.v1.manifest import AnchorEntry

    encoded = _encode(3)
    head_index = encoded.manifest.head_segment.index
    head_records = [json.loads(line) for line in encoded.segments[head_index].decode().splitlines()]

    tampered_lines = bytearray()
    tampered_anchors = dict(encoded.manifest.anchors)
    for record in head_records:
        if record["kind"] == "session":
            record = {**record, "message_count": 999}
        tampered_lines.extend(canonical_line(record))
        tampered_anchors[str(record["record_id"])] = AnchorEntry(
            segment_index=head_index,
            line_index=int(record["seq"]),
            seq=int(record["seq"]),
            kind=str(record["kind"]),
            sha256=hash_bytes(canonical_bytes(record)),
        )
    tampered_head = bytes(tampered_lines)

    tampered_head_descriptor = dataclasses.replace(
        encoded.manifest.head_segment,
        sha256=hash_bytes(tampered_head),
        size_bytes=len(tampered_head),
    )
    ordered_transcript = b"".join(
        encoded.segments[d.index] for d in sorted(encoded.manifest.segments, key=lambda d: d.index)
    )
    joined = tampered_head + ordered_transcript
    tampered_manifest = dataclasses.replace(
        encoded.manifest,
        head_segment=tampered_head_descriptor,
        anchors=tampered_anchors,
        content_digest=dataclasses.replace(
            encoded.manifest.content_digest,
            polylogue_sha256=hash_bytes(joined),
            size_bytes=len(joined),
        ),
    )
    tampered_segments = dict(encoded.segments)
    tampered_segments[head_index] = tampered_head

    with pytest.raises(SemanticClosureError, match="message_count"):
        verify_revision(tampered_manifest, tampered_segments)
