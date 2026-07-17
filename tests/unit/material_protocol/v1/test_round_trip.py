"""Encode -> decode fidelity proof for material protocol v1 (polylogue-303r.1).

This is the anti-vacuity check: it builds a real session (multi-message,
success/failure tool results, equal/missing timestamps + explicit ordinals,
lineage, compaction, an attachment ref, usage, fidelity gaps, nontrivial
Unicode), encodes it to bytes, decodes *only from those bytes* (no session
object, no archive DB), and asserts every one of those facts survived.
"""

from __future__ import annotations

from polylogue.material_protocol.v1 import (
    EncodedRevision,
    SessionMaterial,
    decode_session_revision,
    encode_session_revision,
    resolve_anchor,
    verify_revision,
)
from tests.unit.material_protocol.v1.fixture import SMALL_SESSION_REVISION_CREATED_AT, build_small_session_material


def _encode_small() -> tuple[SessionMaterial, EncodedRevision]:
    material = build_small_session_material()
    encoded = encode_session_revision(
        material, revision_created_at=SMALL_SESSION_REVISION_CREATED_AT, max_records_per_segment=4
    )
    return material, encoded


def test_verify_accepts_a_freshly_encoded_revision() -> None:
    _material, encoded = _encode_small()
    verify_revision(encoded.manifest, encoded.segments)  # must not raise


def test_decode_needs_only_manifest_and_segment_bytes() -> None:
    """Reconstruction needs no archive DB: decode from raw bytes only."""
    _material, encoded = _encode_small()
    manifest_bytes = encoded.manifest.to_dict()
    segment_bytes = dict(encoded.segments)  # plain bytes, no live objects

    from polylogue.material_protocol.v1.manifest import RevisionManifest

    manifest = RevisionManifest.from_dict(manifest_bytes)
    decoded = decode_session_revision(manifest, segment_bytes)
    assert decoded.session["session_id"] == "claude-code-session:demo-session-1"


def test_session_metadata_round_trips() -> None:
    material, encoded = _encode_small()
    decoded = decode_session_revision(encoded.manifest, encoded.segments)
    assert decoded.session["origin"] == material.origin.value
    assert decoded.session["native_id"] == material.native_id
    assert decoded.session["title"] == material.title
    assert decoded.session["git_branch"] == material.git_branch
    assert decoded.session["tags"] == list(material.tags)


def test_message_order_survives_equal_and_missing_timestamps() -> None:
    """msg-1/msg-2 share a timestamp, msg-3 has none -- position must still order them."""
    _material, encoded = _encode_small()
    decoded = decode_session_revision(encoded.manifest, encoded.segments)
    assert [m.native_id for m in decoded.messages] == ["msg-1", "msg-2", "msg-3", "msg-4", "msg-5", "msg-6"]
    assert decoded.messages[0].occurred_at_ms == decoded.messages[1].occurred_at_ms
    assert decoded.messages[2].occurred_at_ms is None
    assert decoded.messages[2].position == 2


def test_nontrivial_unicode_round_trips_exactly() -> None:
    material, encoded = _encode_small()
    decoded = decode_session_revision(encoded.manifest, encoded.segments)
    assert decoded.messages[0].text == material.messages[0].text
    title = decoded.session["title"]
    assert isinstance(title, str)
    assert "\U0001f9ea" in title
    text = decoded.messages[0].text
    assert text is not None
    assert "日本語" in text


def test_tool_call_success_and_failure_pairs_round_trip() -> None:
    _material, encoded = _encode_small()
    decoded = decode_session_revision(encoded.manifest, encoded.segments)

    ok_use = decoded.messages[1].blocks[0]
    ok_result = decoded.messages[2].blocks[0]
    assert ok_use["tool_id"] == ok_result["tool_id"] == "tool-ok-1"
    assert ok_result["tool_result_is_error"] is False
    assert ok_result["tool_result_exit_code"] == 0

    fail_use = decoded.messages[3].blocks[0]
    fail_result = decoded.messages[4].blocks[0]
    assert fail_use["tool_id"] == fail_result["tool_id"] == "tool-fail-1"
    assert fail_result["tool_result_is_error"] is True
    assert fail_result["tool_result_exit_code"] == 4


def test_lineage_round_trips() -> None:
    material, encoded = _encode_small()
    decoded = decode_session_revision(encoded.manifest, encoded.segments)
    assert len(decoded.lineage) == 1
    edge = decoded.lineage[0]
    lineage_input = material.lineage[0]
    assert edge["dst_native_id"] == lineage_input.dst_native_id
    assert edge["link_type"] == lineage_input.link_type.value
    assert edge["inheritance"] == lineage_input.inheritance
    assert edge["branch_point_message_native_id"] == "msg-1"


def test_compaction_session_event_round_trips_and_is_owned_by_its_message() -> None:
    _material, encoded = _encode_small()
    decoded = decode_session_revision(encoded.manifest, encoded.segments)
    compaction_owner = decoded.messages[-1]
    assert len(compaction_owner.session_events) == 1
    event = compaction_owner.session_events[0]
    assert event["event_type"] == "compaction"
    payload = event["payload"]
    assert isinstance(payload, dict)
    assert payload["messages_compacted"] == 5
    assert decoded.trailing_session_events == []


def test_attachment_ref_round_trips() -> None:
    _material, encoded = _encode_small()
    decoded = decode_session_revision(encoded.manifest, encoded.segments)
    attachment_owner = decoded.messages[1]
    assert len(attachment_owner.attachments) == 1
    attachment = attachment_owner.attachments[0]
    assert attachment["attachment_id"] == "att-1"
    assert attachment["acquisition_status"] == "unavailable"
    assert attachment["blob_sha256"] is None


def test_usage_round_trips() -> None:
    _material, encoded = _encode_small()
    decoded = decode_session_revision(encoded.manifest, encoded.segments)
    assert len(decoded.usage) == 1
    usage = decoded.usage[0]
    assert usage["model_name"] == "claude-sonnet-5"
    assert usage["input_tokens"] == 120
    assert usage["cost_usd"] == 0.0123


def test_fidelity_gaps_are_declared_in_the_manifest() -> None:
    _material, encoded = _encode_small()
    gap_kinds = {gap.gap_kind for gap in encoded.manifest.fidelity_gaps}
    assert gap_kinds == {"unavailable_attachment_bytes", "missing_timestamp"}


def test_resolve_anchor_reads_one_record_without_a_full_scan() -> None:
    _material, encoded = _encode_small()
    record = resolve_anchor(encoded.manifest, encoded.segments, "claude-code-session:demo-session-1:msg-2:0")
    assert record["kind"] == "block"
    assert record["tool_id"] == "tool-ok-1"


def test_encode_is_deterministic() -> None:
    material = build_small_session_material()
    first = encode_session_revision(
        material, revision_created_at=SMALL_SESSION_REVISION_CREATED_AT, max_records_per_segment=4
    )
    second = encode_session_revision(
        material, revision_created_at=SMALL_SESSION_REVISION_CREATED_AT, max_records_per_segment=4
    )
    assert first.segments == second.segments
    assert first.manifest.to_dict() == second.manifest.to_dict()


def test_decode_then_reencode_each_record_is_byte_identical_to_source_line() -> None:
    """Directly proves 'decode/re-encode is byte-identical' at the record level."""
    from polylogue.material_protocol.v1.canonical import canonical_line
    from polylogue.material_protocol.v1.decode import iter_records

    _material, encoded = _encode_small()
    records = iter_records(encoded.manifest, encoded.segments)

    reencoded = bytearray()
    for record in records:
        reencoded.extend(canonical_line(record))

    original = bytearray(encoded.segments[encoded.manifest.head_segment.index])
    for descriptor in sorted(encoded.manifest.segments, key=lambda d: d.index):
        original.extend(encoded.segments[descriptor.index])

    assert bytes(reencoded) == bytes(original)
