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

    for record_id, prior_anchor in prior.manifest.anchors.items():
        assert appended.manifest.anchors[record_id] == prior_anchor


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
    for record_id, anchor in revision_1.manifest.anchors.items():
        assert revision_3.manifest.anchors[record_id] == anchor
    for record_id, anchor in revision_2.manifest.anchors.items():
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
