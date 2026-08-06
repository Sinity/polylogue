"""Contracts for the shared scale outlier fixture pack."""

from __future__ import annotations

from tests.infra.whale_fixtures import WHALE_FIXTURE_DIMENSIONS, multi_million_codex_stream


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


def test_multi_million_stream_keeps_record_identity_bounded() -> None:
    stream = multi_million_codex_stream()
    first = next(stream)
    second = next(stream)
    assert first["type"] == "session_meta"
    assert second["type"] == "response_item"
    first_state = next(stream)
    second_state = next(stream)
    assert first_state is second_state
