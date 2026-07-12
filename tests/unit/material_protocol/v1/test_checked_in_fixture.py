"""The checked-in fixture (tests/fixtures/material_protocol/v1/small-session/)
is the byte-for-byte cross-repo artifact polylogue-303r.1's design calls for
("Check the same synthetic fixture and digest into both repositories").

This test proves our own side never silently drifts: a fresh encode of the
exact same SessionMaterial must reproduce the checked-in manifest.json and
every segment file byte-for-byte, and every declared SHA-256 must match a
live recompute. A future encoder change that alters framing/ordering/hashing
must fail this test and get an explicit fixture regeneration + version bump,
not a silent bytes drift.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from polylogue.material_protocol.v1 import encode_session_revision, read_revision, verify_revision
from tests.unit.material_protocol.v1.fixture import SMALL_SESSION_REVISION_CREATED_AT, build_small_session_material

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "fixtures" / "material_protocol" / "v1" / "small-session"


def test_fixture_directory_exists() -> None:
    assert FIXTURE_DIR.is_dir(), f"missing checked-in fixture: {FIXTURE_DIR}"
    assert (FIXTURE_DIR / "manifest.json").is_file()


def test_checked_in_bytes_match_a_fresh_encode() -> None:
    material = build_small_session_material()
    fresh = encode_session_revision(
        material, revision_created_at=SMALL_SESSION_REVISION_CREATED_AT, max_records_per_segment=4
    )

    checked_in_manifest, checked_in_segments = read_revision(FIXTURE_DIR)

    assert fresh.manifest.to_dict() == checked_in_manifest.to_dict()
    assert fresh.segments == checked_in_segments


def test_checked_in_fixture_passes_verify_revision() -> None:
    manifest, segments = read_revision(FIXTURE_DIR)
    verify_revision(manifest, segments)  # must not raise


def test_checked_in_segment_sha256_matches_manifest() -> None:
    manifest, segments = read_revision(FIXTURE_DIR)
    for descriptor in manifest.segments:
        actual = hashlib.sha256(segments[descriptor.index]).hexdigest()
        assert actual == descriptor.sha256

    joined = b"".join(segments[d.index] for d in sorted(manifest.segments, key=lambda d: d.index))
    assert hashlib.sha256(joined).hexdigest() == manifest.content_digest.polylogue_sha256
    assert manifest.revision_id == manifest.content_digest.polylogue_sha256
