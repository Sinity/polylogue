"""ParsedSession-to-material-v1 coverage and exact-byte reconciliation."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from polylogue.core.enums import (
    BlockType,
    BranchType,
    MaterialOrigin,
    MessageType,
    Provider,
    Role,
    SessionKind,
)
from polylogue.sinex.material_adapter import (
    PublicationBackpressureError,
    encode_parsed_session_publication,
    session_material_from_parsed_session,
)
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
)


def _parsed_session() -> ParsedSession:
    first = ParsedMessage(
        position=0,
        provider_message_id="m1",
        role=Role.USER,
        text="hello",
        message_type=MessageType.MESSAGE,
        material_origin=MaterialOrigin.HUMAN_AUTHORED,
        occurred_at_ms=1_000,
        model_name="gpt-x",
        input_tokens=3,
        output_tokens=0,
        cache_read_tokens=1,
        cache_write_tokens=0,
        duration_ms=5,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hello", metadata={"provider_only": "gap"})],
        delivery_status="sent",  # explicit v1 fidelity gap
    )
    second = ParsedMessage(
        position=1,
        provider_message_id="m2",
        parent_message_provider_id="m1",
        role=Role.ASSISTANT,
        text="world",
        message_type=MessageType.MESSAGE,
        material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
        occurred_at_ms=2_000,
        model_name="gpt-x",
        input_tokens=0,
        output_tokens=4,
        cache_read_tokens=0,
        cache_write_tokens=0,
        blocks=[
            ParsedContentBlock(
                type=BlockType.TOOL_USE,
                tool_name="Shell",
                tool_id="t1",
                tool_input={"cmd": "pwd"},
            ),
        ],
    )
    attachment = ParsedAttachment(
        provider_attachment_id="a1",
        message_provider_id="m2",
        name="out.txt",
        mime_type="text/plain",
        size_bytes=3,
        path="provider-only.txt",
    )
    event = ParsedSessionEvent(
        event_type="checkpoint",
        payload={"n": 1, "summary": "checkpoint saved"},
        source_message_provider_id="m2",
        timestamp="1970-01-01T00:00:02.500Z",
    )
    return ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="s1",
        messages=[first, second],
        attachments=[attachment],
        title="Fixture",
        session_kind=SessionKind.STANDARD,
        created_at="1970-01-01T00:00:01Z",
        updated_at="1970-01-01T00:00:03Z",
        git_branch="main",
        git_repository_url="https://example.invalid/repo",
        provider_project_ref="p",
        working_directories=["/repo"],
        ingest_flags=["fixture"],
        parent_session_provider_id="parent",
        branch_type=BranchType.FORK,
        reported_cost_usd=0.01,
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
    assert {gap.gap_kind for gap in material.fidelity_gaps} == {"unsupported_normalized_fields"}


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


def test_naive_timestamps_are_utc_and_unordered_metadata_is_deterministic() -> None:
    parsed = _parsed_session()
    parsed.created_at = datetime(2026, 7, 16, 3, 0, 0).isoformat()
    parsed.messages[0].blocks[0].metadata = {"unordered": {"z", "a"}}
    first = encode_parsed_session_publication(parsed, session_id="claude-code-session:s1")
    second = encode_parsed_session_publication(parsed, session_id="claude-code-session:s1")
    assert first.manifest_bytes == second.manifest_bytes
    assert first.segments == second.segments


def test_payload_budget_rejects_before_protocol_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    parsed = _parsed_session()
    parsed.messages[0].text = "x" * 1_024
    called = False

    def unexpected_encoder(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("protocol encoder must not allocate an over-budget payload")

    monkeypatch.setattr("polylogue.sinex.material_adapter.encode_session_revision", unexpected_encoder)
    with pytest.raises(PublicationBackpressureError):
        encode_parsed_session_publication(
            parsed,
            session_id="claude-code-session:s1",
            max_payload_bytes=128,
        )
    assert not called
