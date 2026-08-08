"""Contracts for the shared scale outlier fixture pack."""

from __future__ import annotations

import json
from pathlib import Path

from tests.infra.whale_fixtures import (
    WHALE_FIXTURE_DIMENSIONS,
    CodexRevisionChainFixture,
    WhaleFixtureDimensions,
    _write_padding_record,
    multi_million_codex_stream,
    write_codex_whale_fixture_pack,
)


def test_whale_fixture_manifest_pins_all_outlier_axes() -> None:
    dimensions = dict(WHALE_FIXTURE_DIMENSIONS.manifest_dimensions())

    assert dimensions == {
        "fixture_id": "codex-whale-bounds-v2",
        "revision_count": 804,
        "terminal_wire_bytes": 90_822_451,
        "near_terminal_predecessor_bytes": 32 * 1024 * 1024,
        "stream_event_count": 2_000_000,
        "giant_attachment_raw_bytes": 12 * 1024 * 1024,
        "ordinary_blob_limit_bytes": 64 * 1024 * 1024,
        "whale_blob_limit_bytes": 8 * 1024 * 1024 * 1024,
    }

    assert WHALE_FIXTURE_DIMENSIONS.giant_attachment_raw_bytes > 0
    assert (
        WHALE_FIXTURE_DIMENSIONS.giant_attachment_raw_bytes
        < WHALE_FIXTURE_DIMENSIONS.near_terminal_predecessor_bytes
        < WHALE_FIXTURE_DIMENSIONS.ordinary_blob_limit_bytes
        < WHALE_FIXTURE_DIMENSIONS.terminal_wire_bytes
        < WHALE_FIXTURE_DIMENSIONS.whale_blob_limit_bytes
    )
    assert WHALE_FIXTURE_DIMENSIONS.stream_event_count > WHALE_FIXTURE_DIMENSIONS.revision_count


def test_multi_million_stream_emits_realistic_distinct_records() -> None:
    stream = multi_million_codex_stream()
    first = next(stream)
    second = next(stream)
    assert first["type"] == "session_meta"
    assert second["type"] == "response_item"
    first_state = next(stream)
    second_state = next(stream)
    assert first_state is not second_state
    assert first_state == {"record_type": "state", "sequence": 0}
    assert second_state == {"record_type": "state", "sequence": 1}


def test_padding_generator_never_writes_an_outlier_sized_chunk() -> None:
    class RecordingSink:
        position = 0
        max_write = 0

        def write(self, value: bytes) -> int:
            self.position += len(value)
            self.max_write = max(self.max_write, len(value))
            return len(value)

    sink = RecordingSink()
    target_bytes = 4 * 1024 * 1024
    _write_padding_record(sink, revision=803, target_bytes=target_bytes, current_bytes=0)  # type: ignore[arg-type]

    assert sink.position == target_bytes
    assert sink.max_write <= 1024 * 1024


def test_fixture_pack_generator_writes_a_complete_manifest(tmp_path: Path) -> None:
    dimensions = WhaleFixtureDimensions(
        fixture_id="codex-whale-bounds-test",
        revision_count=5,
        terminal_wire_bytes=1024 * 1024,
        near_terminal_predecessor_bytes=256 * 1024,
        stream_event_count=100,
        giant_attachment_raw_bytes=32 * 1024,
        ordinary_blob_limit_bytes=512 * 1024,
        whale_blob_limit_bytes=2 * 1024 * 1024,
    )
    fixture = CodexRevisionChainFixture(dimensions=dimensions, session_native_id="codex-whale-test")

    source_path, manifest_path = write_codex_whale_fixture_pack(tmp_path, fixture)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert source_path.stat().st_size == dimensions.terminal_wire_bytes
    assert manifest["fixture_id"] == dimensions.fixture_id
    assert manifest["revision_sizes"] == [4_096, 64 * 1024, 128 * 1024, 256 * 1024, 1024 * 1024]
    assert len(manifest["revision_sha256"]) == dimensions.revision_count
    assert all(len(value) == 64 for value in manifest["revision_sha256"])
    assert manifest["terminal_features"] == ["compaction", "giant-base64-attachment", "codex-stream-dispatch"]

    rerun_source_path, rerun_manifest_path = write_codex_whale_fixture_pack(tmp_path, fixture)
    assert rerun_source_path == source_path
    assert rerun_source_path.stat().st_size == dimensions.terminal_wire_bytes
    assert json.loads(rerun_manifest_path.read_text(encoding="utf-8")) == manifest

    changed_fixture = CodexRevisionChainFixture(dimensions=dimensions, session_native_id="codex-whale-changed")
    _changed_source, changed_manifest_path = write_codex_whale_fixture_pack(tmp_path / "changed", changed_fixture)
    changed_manifest = json.loads(changed_manifest_path.read_text(encoding="utf-8"))
    assert changed_manifest["revision_sizes"] == manifest["revision_sizes"]
    assert changed_manifest["revision_sha256"] != manifest["revision_sha256"]
