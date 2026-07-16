"""ParsedSession-to-material-v1 coverage and exact-byte reconciliation."""

from __future__ import annotations

import json
from types import SimpleNamespace

from polylogue.sinex.material_adapter import (
    encode_parsed_session_publication,
    session_material_from_parsed_session,
)


def _parsed_session() -> SimpleNamespace:
    first = SimpleNamespace(
        position=0,
        provider_message_id="m1",
        role="user",
        text="hello",
        message_type="message",
        material_origin="native",
        occurred_at_ms=1_000,
        model_name="gpt-x",
        input_tokens=3,
        output_tokens=0,
        cache_read_tokens=1,
        cache_write_tokens=0,
        duration_ms=5,
        blocks=[{"type": "text", "text": "hello"}],
        delivery_status="sent",  # explicit v1 fidelity gap
    )
    second = SimpleNamespace(
        position=1,
        provider_message_id="m2",
        parent_message_provider_id="m1",
        role="assistant",
        text="world",
        message_type="message",
        material_origin="native",
        occurred_at_ms=2_000,
        model_name="gpt-x",
        input_tokens=0,
        output_tokens=4,
        cache_read_tokens=0,
        cache_write_tokens=0,
        blocks=[
            {"type": "tool_use", "tool_name": "Shell", "tool_id": "t1", "tool_input": {"cmd": "pwd"}},
            {"type": "not-a-real-block"},
        ],
    )
    attachment = SimpleNamespace(
        provider_attachment_id="a1",
        message_position=1,
        filename="out.txt",
        media_type="text/plain",
        size_bytes=3,
        sha256="a" * 64,
        text="abc",
        metadata={"provider_only": "declared-gap"},
    )
    event = SimpleNamespace(
        position=0,
        event_type="checkpoint",
        summary="checkpoint saved",
        payload={"n": 1},
        source_message_provider_id="m2",
        occurred_at_ms=2_500,
    )
    return SimpleNamespace(
        messages=[first, second],
        attachments=[attachment],
        title="Fixture",
        session_kind="chat",
        created_at=1_000,
        updated_at=3_000,
        git_branch="main",
        git_repository_url="https://example.invalid/repo",
        provider_project_ref="p",
        working_directories=["/repo"],
        metadata={"k": "v"},
        ingest_flags=["fixture"],
        parent_session_provider_id="parent",
        branch_type="branch",
        branch_point_message_provider_id="m1",
        lineage_status="resolved",
        lineage_confidence=0.9,
        reported_cost_usd=0.01,
        cost_provenance="reported",
        session_events=[event],
    )


def test_adapter_covers_every_available_material_unit_and_names_gaps() -> None:
    material = session_material_from_parsed_session(_parsed_session(), session_id="claude-code-session:s1")
    assert len(material.messages) == 2
    assert sum(len(message.blocks) for message in material.messages) == 2
    assert sum(len(message.attachments) for message in material.messages) == 1
    assert len(material.lineage) == 1
    assert len(material.usage) == 1
    assert len(material.session_events) == 1
    assert {gap.gap_kind for gap in material.fidelity_gaps} >= {
        "dropped_block",
        "unsupported_normalized_fields",
    }


def test_production_adapter_runs_real_encoder_verifier_decoder_and_preserves_wire_names() -> None:
    payload = encode_parsed_session_publication(_parsed_session(), session_id="claude-code-session:s1")
    manifest = json.loads(payload.manifest_bytes)
    assert payload.protocol_version == "polylogue.material-protocol/v1"
    assert payload.revision_id == manifest["revision_id"]
    assert payload.object_id == manifest["session_id"]
    assert payload.manifest_digest
    assert [name for name, _data in payload.segments][0] == "head.ndjson"
    assert manifest["expected_record_counts"]["message"] == 2
    assert manifest["expected_record_counts"]["attachment"] == 1
    assert manifest["expected_record_counts"]["lineage"] == 1
    assert manifest["expected_record_counts"]["usage"] == 1
    assert manifest["expected_record_counts"]["session_event"] == 1
