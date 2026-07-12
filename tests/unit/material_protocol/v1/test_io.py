"""Filesystem write/read round trip for a revision directory."""

from __future__ import annotations

from pathlib import Path

from polylogue.material_protocol.v1 import encode_session_revision, read_revision, verify_revision, write_revision
from tests.unit.material_protocol.v1.fixture import SMALL_SESSION_REVISION_CREATED_AT, build_small_session_material


def test_write_then_read_revision_round_trips(tmp_path: Path) -> None:
    material = build_small_session_material()
    encoded = encode_session_revision(
        material, revision_created_at=SMALL_SESSION_REVISION_CREATED_AT, max_records_per_segment=4
    )

    write_revision(encoded, tmp_path / "revision")
    manifest, segments = read_revision(tmp_path / "revision")

    assert manifest.to_dict() == encoded.manifest.to_dict()
    assert segments == encoded.segments
    verify_revision(manifest, segments)


def test_write_revision_produces_one_file_per_segment(tmp_path: Path) -> None:
    material = build_small_session_material()
    encoded = encode_session_revision(
        material, revision_created_at=SMALL_SESSION_REVISION_CREATED_AT, max_records_per_segment=4
    )
    write_revision(encoded, tmp_path / "revision")

    segment_files = sorted((tmp_path / "revision" / "segments").glob("*.ndjson"))
    assert len(segment_files) == len(encoded.manifest.segments)
