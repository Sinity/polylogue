"""Every compatibility-failure mode named in polylogue-303r.1's acceptance
criteria must fail closed with a typed MaterialProtocolError subclass:
removing a required segment/record, changing a byte/count/digest, reordering
a record, shifting an anchor, or using an unknown Origin vocabulary version.

Each test starts from a known-good encoded revision and applies exactly one
mutation, so a failure to raise pinpoints exactly which guard regressed.
"""

from __future__ import annotations

import dataclasses

import pytest

from polylogue.material_protocol.v1 import (
    AnchorMismatchError,
    DigestMismatchError,
    EncodedRevision,
    RecordCountMismatchError,
    SegmentMissingError,
    SequenceOrderError,
    UnknownOriginVocabularyError,
    encode_session_revision,
    resolve_anchor,
    verify_revision,
)
from tests.unit.material_protocol.v1.fixture import SMALL_SESSION_REVISION_CREATED_AT, build_small_session_material


@pytest.fixture
def encoded() -> EncodedRevision:
    material = build_small_session_material()
    return encode_session_revision(
        material, revision_created_at=SMALL_SESSION_REVISION_CREATED_AT, max_records_per_segment=4
    )


def test_removing_a_required_segment_fails(encoded: EncodedRevision) -> None:
    tampered = dict(encoded.segments)
    del tampered[0]
    with pytest.raises(SegmentMissingError):
        verify_revision(encoded.manifest, tampered)


def test_removing_a_required_record_within_a_segment_fails(encoded: EncodedRevision) -> None:
    tampered = dict(encoded.segments)
    lines = [line for line in tampered[0].split(b"\n") if line]
    del lines[0]
    tampered[0] = b"\n".join(lines) + b"\n"
    with pytest.raises((DigestMismatchError, RecordCountMismatchError, SequenceOrderError)):
        verify_revision(encoded.manifest, tampered)


def test_changing_a_byte_fails(encoded: EncodedRevision) -> None:
    tampered = dict(encoded.segments)
    mutated = bytearray(tampered[0])
    mutated[5] ^= 0xFF
    tampered[0] = bytes(mutated)
    with pytest.raises(DigestMismatchError):
        verify_revision(encoded.manifest, tampered)


def test_changing_a_declared_record_count_fails(encoded: EncodedRevision) -> None:
    bad_manifest = dataclasses.replace(
        encoded.manifest,
        expected_record_counts={**encoded.manifest.expected_record_counts, "message": 999},
    )
    with pytest.raises(RecordCountMismatchError):
        verify_revision(bad_manifest, encoded.segments)


def test_changing_the_content_digest_fails(encoded: EncodedRevision) -> None:
    bad_digest = dataclasses.replace(encoded.manifest.content_digest, polylogue_sha256="0" * 64)
    bad_manifest = dataclasses.replace(encoded.manifest, content_digest=bad_digest)
    with pytest.raises(DigestMismatchError):
        verify_revision(bad_manifest, encoded.segments)


def test_reordering_a_record_fails(encoded: EncodedRevision) -> None:
    tampered = dict(encoded.segments)
    lines = [line for line in tampered[0].split(b"\n") if line]
    assert len(lines) >= 2
    lines[0], lines[1] = lines[1], lines[0]
    tampered[0] = b"\n".join(lines) + b"\n"
    with pytest.raises((DigestMismatchError, SequenceOrderError, AnchorMismatchError)):
        verify_revision(encoded.manifest, tampered)


def test_shifting_an_anchor_line_index_fails_resolve_anchor(encoded: EncodedRevision) -> None:
    record_id = next(iter(encoded.manifest.anchors))
    real_anchor = encoded.manifest.anchors[record_id]
    shifted = dataclasses.replace(real_anchor, line_index=real_anchor.line_index + 1)
    bad_manifest = dataclasses.replace(encoded.manifest, anchors={**encoded.manifest.anchors, record_id: shifted})
    with pytest.raises(AnchorMismatchError):
        resolve_anchor(bad_manifest, encoded.segments, record_id)


def test_shifting_an_anchor_line_index_fails_verify_revision(encoded: EncodedRevision) -> None:
    record_id = next(iter(encoded.manifest.anchors))
    real_anchor = encoded.manifest.anchors[record_id]
    shifted = dataclasses.replace(real_anchor, line_index=real_anchor.line_index + 1)
    bad_manifest = dataclasses.replace(encoded.manifest, anchors={**encoded.manifest.anchors, record_id: shifted})
    with pytest.raises(AnchorMismatchError):
        verify_revision(bad_manifest, encoded.segments)


def test_shifting_an_anchor_to_a_different_segment_fails(encoded: EncodedRevision) -> None:
    record_id = next(iter(encoded.manifest.anchors))
    real_anchor = encoded.manifest.anchors[record_id]
    other_segment = next(d.index for d in encoded.manifest.segments if d.index != real_anchor.segment_index)
    shifted = dataclasses.replace(real_anchor, segment_index=other_segment)
    bad_manifest = dataclasses.replace(encoded.manifest, anchors={**encoded.manifest.anchors, record_id: shifted})
    with pytest.raises(AnchorMismatchError):
        resolve_anchor(bad_manifest, encoded.segments, record_id)


def test_unknown_origin_vocabulary_version_fails(encoded: EncodedRevision) -> None:
    bad_manifest = dataclasses.replace(encoded.manifest, origin_vocabulary_version=9999)
    with pytest.raises(UnknownOriginVocabularyError):
        verify_revision(bad_manifest, encoded.segments)


def test_stale_origin_vocabulary_digest_fails(encoded: EncodedRevision) -> None:
    bad_manifest = dataclasses.replace(encoded.manifest, origin_vocabulary_digest="0" * 64)
    with pytest.raises(UnknownOriginVocabularyError):
        verify_revision(bad_manifest, encoded.segments)
