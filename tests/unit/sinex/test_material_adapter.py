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
    ToolOutcome,
    ToolResultUnknownReason,
)
from polylogue.material_protocol.v1 import DecodedSession, RevisionManifest, decode_session_revision, verify_revision
from polylogue.pipeline.ids import session_content_hash
from polylogue.sinex.material_adapter import (
    PublicationBackpressureError,
    encode_parsed_session_publication,
    session_material_from_parsed_session,
)
from polylogue.sinex.models import PublicationPayload
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
)


def _decoded_publication(session: ParsedSession) -> tuple[PublicationPayload, DecodedSession]:
    payload = encode_parsed_session_publication(session, session_id="claude-code-session:s1")
    manifest = RevisionManifest.from_dict(json.loads(payload.manifest_bytes))
    names = dict(payload.segments)
    segments = {
        descriptor.index: names[descriptor.filename] for descriptor in (*manifest.segments, manifest.head_segment)
    }
    verify_revision(manifest, segments)
    return payload, decode_session_revision(manifest, segments)


@pytest.mark.parametrize("natives", [("m", "m"), (" m ", "m")])
def test_publication_preserves_exact_duplicate_and_distinct_opaque_native_ids(natives: tuple[str, str]) -> None:
    parsed = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="s1",
        messages=[
            ParsedMessage(provider_message_id=natives[0], position=0, role=Role.USER, text="first"),
            ParsedMessage(provider_message_id=natives[1], position=1, role=Role.USER, text="second"),
        ],
    )
    _payload, decoded = _decoded_publication(parsed)
    assert [message.text for message in decoded.messages] == ["first", "second"]
    assert len({message.message_id for message in decoded.messages}) == 2
    if natives[0] == natives[1]:
        assert all(":c:" in message.message_id for message in decoded.messages)
    else:
        assert [message.native_id for message in decoded.messages] == list(natives)
        assert [message.message_id for message in decoded.messages] == [
            f"claude-code-session:s1:n:{native}" for native in natives
        ]


def test_operational_unicode_changes_remain_distinct_publication_revisions() -> None:
    parsed = _parsed_session()
    parsed.messages[1].blocks[0].tool_input = {"path": "cafe\u0301", "é": "first", "e\u0301": "second"}
    first, decoded = _decoded_publication(parsed)
    first_hash = session_content_hash(parsed)
    assert decoded.messages[1].blocks[0]["tool_input"] == parsed.messages[1].blocks[0].tool_input
    parsed.messages[1].blocks[0].tool_input = {"path": "café", "é": "first", "e\u0301": "second"}
    second, _decoded = _decoded_publication(parsed)
    assert session_content_hash(parsed) != first_hash
    assert first.revision_id != second.revision_id


def test_parent_cut_and_available_session_context_survive_publication() -> None:
    parsed = _parsed_session()
    parsed.branch_point_provider_message_id = "cut"
    parsed.display_name = "neutral-agent"
    parsed.pending_drafts = [{"text": "unsent"}]
    parsed.messages[0].stop_reason = "max_tokens"
    parsed.session_events[0].boundary_start_position = 0
    material = session_material_from_parsed_session(parsed, session_id="claude-code-session:s1")
    assert material.lineage[0].branch_point_message_native_id == "cut"
    assert material.metadata["display_name"] == "neutral-agent"
    assert material.metadata["pending_drafts"] == [{"text": "unsent"}]
    gaps = " ".join(gap.detail for gap in material.fidelity_gaps)
    assert "stop_reason" in gaps
    assert "boundary_start_position" in gaps


def test_tool_result_association_does_not_rename_source_tool_use() -> None:
    parsed = _parsed_session()
    parsed.messages[1].blocks = [parsed.messages[1].blocks[0]]
    first = session_material_from_parsed_session(parsed, session_id="claude-code-session:s1").messages[1].blocks[0]
    parsed.messages[1].blocks.append(
        ParsedContentBlock(type=BlockType.TOOL_RESULT, tool_id=first.tool_id, text="done", is_error=False)
    )
    second = session_material_from_parsed_session(parsed, session_id="claude-code-session:s1").messages[1].blocks[0]
    assert first.content_identity == second.content_identity
    assert first.content_occurrence == second.content_occurrence


def test_attachment_and_event_owners_preserve_exact_native_names() -> None:
    parsed = _parsed_session()
    parsed.messages[1].provider_message_id = " m2 "
    parsed.attachments[0].message_provider_id = " m2 "
    parsed.session_events[0].source_message_provider_id = " m2 "
    _payload, decoded = _decoded_publication(parsed)
    assert len(decoded.messages[1].attachments) == 1
    assert len(decoded.messages[1].session_events) == 1


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
                tool_outcome=ToolOutcome.OK,
            ),
            ParsedContentBlock(
                type=BlockType.TOOL_RESULT,
                tool_id="t1",
                text="unknown result",
                tool_outcome=ToolOutcome.UNKNOWN,
                outcome_unknown_reason=ToolResultUnknownReason.NOT_REPORTED.value,
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
    assert sum(len(message.blocks) for message in material.messages) == 3
    assert material.messages[1].blocks[0].tool_outcome is ToolOutcome.UNKNOWN
    assert material.messages[1].blocks[1].tool_outcome is ToolOutcome.UNKNOWN
    assert material.messages[1].blocks[1].tool_result_outcome_unknown_reason == "not_reported"
    assert sum(len(message.attachments) for message in material.messages) == 1
    assert len(material.lineage) == 1
    assert len(material.usage) == 1
    assert len(material.session_events) == 1
    assert {gap.gap_kind for gap in material.fidelity_gaps} == {"unsupported_normalized_fields"}


def test_adapter_derives_tool_outcomes_before_publication() -> None:
    parsed = _parsed_session()
    tool_use, tool_result = parsed.messages[1].blocks
    parsed.messages[1].blocks = [
        tool_use.model_copy(update={"tool_outcome": None}),
        tool_result.model_copy(
            update={
                "tool_outcome": None,
                "is_error": False,
                "outcome_unknown_reason": None,
            }
        ),
    ]

    material = session_material_from_parsed_session(parsed, session_id="claude-code-session:s1")

    assert material.messages[1].blocks[0].tool_outcome is ToolOutcome.OK
    assert material.messages[1].blocks[1].tool_outcome is ToolOutcome.OK


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
    parsed.messages[0].occurred_at_ms = 0
    parsed.messages[0].blocks[0].metadata = {"unordered": {"z", "a"}}
    material = session_material_from_parsed_session(parsed, session_id="claude-code-session:s1")
    first = encode_parsed_session_publication(parsed, session_id="claude-code-session:s1")
    second = encode_parsed_session_publication(parsed, session_id="claude-code-session:s1")
    assert material.messages[0].occurred_at_ms == 0
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


def test_publication_parent_and_block_ids_preserve_opaque_native_whitespace() -> None:
    parsed = _parsed_session()
    parsed.messages[0].provider_message_id = " m1 "
    parsed.messages[1].provider_message_id = " m2 "
    parsed.messages[1].parent_message_provider_id = " m1 "
    parsed.attachments[0].message_provider_id = " m2 "
    parsed.session_events[0].source_message_provider_id = " m2 "
    _payload, decoded = _decoded_publication(parsed)
    assert decoded.messages[1].parent_message_id == decoded.messages[0].message_id
    assert decoded.messages[1].blocks[0]["block_id"].startswith(decoded.messages[1].message_id + ":b:")
