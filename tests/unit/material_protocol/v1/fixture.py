"""Synthetic session fixtures for material protocol v1.

``build_small_session_material()`` is the checked-in cross-repo fixture
(polylogue-303r.1 design: "Check the same synthetic fixture and digest into
both repositories"): it deliberately exercises every acceptance-criteria
requirement in one small session --

- multiple messages
- a successful tool call (block pair, tool_result_is_error=False) and a
  failed one (tool_result_is_error=True, non-zero exit code)
- equal timestamps on two messages, a missing timestamp on a third, all
  disambiguated only by explicit position/variant_index ordinals
- one lineage edge (resume) and one compaction session_event
- one attachment ref (with unavailable acquired bytes -> a fidelity gap)
- one usage row
- an explicit fidelity gap list entry
- nontrivial Unicode: combining diacritics, CJK, emoji, RTL Arabic

``build_large_session_material(n)`` / the "append" pair below back the
segmentation/stable-anchor test and are not checked in as golden bytes.
"""

from __future__ import annotations

from polylogue.core.enums import BlockType, LinkType, MaterialOrigin, MessageType, Origin, Role, SessionKind
from polylogue.material_protocol.v1 import (
    AttachmentInput,
    BlockInput,
    FidelityGapInput,
    LineageInput,
    MessageInput,
    SessionEventInput,
    SessionMaterial,
    UsageInput,
)

SMALL_SESSION_REVISION_CREATED_AT = "2026-07-12T00:00:00Z"


def build_small_session_material() -> SessionMaterial:
    m_user = MessageInput(
        native_id="msg-1",
        position=0,
        role=Role.USER,
        message_type=MessageType.MESSAGE,
        material_origin=MaterialOrigin.HUMAN_AUTHORED,
        text="Café — café (combining check), 日本語, emoji \U0001f9ea, RTL: مرحبا",
        occurred_at_ms=1_720_000_000_000,
    )
    m_assistant_ok_tool = MessageInput(
        native_id="msg-2",
        position=1,
        role=Role.ASSISTANT,
        message_type=MessageType.MESSAGE,
        material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
        text=None,
        # Equal timestamp to msg-1: ordering must come from position, not time.
        occurred_at_ms=1_720_000_000_000,
        model_name="claude-sonnet-5",
        parent_native_id="msg-1",
        input_tokens=120,
        output_tokens=40,
        blocks=(
            BlockInput(
                position=0,
                block_type=BlockType.TOOL_USE,
                tool_name="run_tests",
                tool_id="tool-ok-1",
                tool_input={"command": "pytest -k café", "cwd": "/repo"},
            ),
        ),
        attachments=(
            AttachmentInput(
                position=0,
                attachment_id="att-1",
                display_name="log-日本語.txt",
                media_type="text/plain",
                byte_count=0,
                blob_sha256=None,
                acquisition_status="unavailable",
                upload_origin="paste",
            ),
        ),
    )
    m_tool_result_ok = MessageInput(
        native_id="msg-3",
        position=2,
        role=Role.ASSISTANT,
        message_type=MessageType.MESSAGE,
        material_origin=MaterialOrigin.TOOL_RESULT,
        text=None,
        # Missing timestamp: ordinal (position=2) is the only order signal.
        occurred_at_ms=None,
        blocks=(
            BlockInput(
                position=0,
                block_type=BlockType.TOOL_RESULT,
                tool_id="tool-ok-1",
                text="3 passed in 0.42s",
                tool_result_is_error=False,
                tool_result_exit_code=0,
            ),
        ),
    )
    m_assistant_failing_tool = MessageInput(
        native_id="msg-4",
        position=3,
        role=Role.ASSISTANT,
        message_type=MessageType.MESSAGE,
        material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
        text=None,
        occurred_at_ms=1_720_000_060_000,
        model_name="claude-sonnet-5",
        parent_native_id="msg-3",
        blocks=(
            BlockInput(
                position=0,
                block_type=BlockType.TOOL_USE,
                tool_name="run_tests",
                tool_id="tool-fail-1",
                tool_input={"command": "pytest -k missing"},
            ),
        ),
    )
    m_tool_result_fail = MessageInput(
        native_id="msg-5",
        position=4,
        role=Role.ASSISTANT,
        message_type=MessageType.MESSAGE,
        material_origin=MaterialOrigin.TOOL_RESULT,
        text=None,
        occurred_at_ms=1_720_000_061_000,
        blocks=(
            BlockInput(
                position=0,
                block_type=BlockType.TOOL_RESULT,
                tool_id="tool-fail-1",
                text="ERROR: no tests matched 'missing'",
                tool_result_is_error=True,
                tool_result_exit_code=4,
            ),
        ),
    )
    m_compaction_marker = MessageInput(
        native_id="msg-6",
        position=5,
        role=Role.ASSISTANT,
        message_type=MessageType.MESSAGE,
        material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
        text="Summarized the run above after context compaction.",
        occurred_at_ms=1_720_000_120_000,
        model_name="claude-sonnet-5",
    )

    lineage = (
        LineageInput(
            dst_origin=Origin.CLAUDE_CODE_SESSION,
            dst_native_id="parent-session-0",
            link_type=LinkType.RESUME,
            branch_point_message_native_id="msg-1",
            inheritance="prefix-sharing",
            status="repaired",
            confidence=0.97,
            observed_at_ms=1_720_000_000_500,
        ),
    )
    usage = (
        UsageInput(
            model_name="claude-sonnet-5",
            input_tokens=120,
            output_tokens=40,
            cache_read_tokens=64,
            cache_write_tokens=0,
            cost_usd=0.0123,
            cost_provenance="priced",
        ),
    )
    session_events = (
        SessionEventInput(
            position=0,
            event_type="compaction",
            summary="Auto-compacted after 5 messages",
            payload={"messages_compacted": 5, "trigger": "context_window"},
            source_message_native_id="msg-6",
            occurred_at_ms=1_720_000_119_000,
        ),
    )
    fidelity_gaps = (
        FidelityGapInput(
            scope="attachment",
            record_id="claude-code-session:demo-session-1:msg-2:attachment:0",
            gap_kind="unavailable_attachment_bytes",
            detail="attachment referenced by the provider export but bytes were never fetched",
        ),
        FidelityGapInput(
            scope="message",
            record_id="claude-code-session:demo-session-1:msg-3",
            gap_kind="missing_timestamp",
            detail="provider omitted occurred_at; position ordinal is authoritative",
        ),
    )

    return SessionMaterial(
        origin=Origin.CLAUDE_CODE_SESSION,
        native_id="demo-session-1",
        title="Fix café test flake (\U0001f9ea demo)",
        session_kind=SessionKind.STANDARD,
        created_at_ms=1_720_000_000_000,
        updated_at_ms=1_720_000_120_000,
        git_branch="feature/fix-flake",
        git_repository_url="https://github.com/example/example",
        provider_project_ref="proj-42",
        working_directories=("/repo",),
        metadata={"demo": True},
        tags=("demo", "fixture"),
        messages=(
            m_user,
            m_assistant_ok_tool,
            m_tool_result_ok,
            m_assistant_failing_tool,
            m_tool_result_fail,
            m_compaction_marker,
        ),
        lineage=lineage,
        usage=usage,
        session_events=session_events,
        fidelity_gaps=fidelity_gaps,
    )


def build_large_session_material(message_count: int, *, native_prefix: str = "big") -> SessionMaterial:
    """A session with *message_count* plain messages, for segmentation tests."""
    messages = tuple(
        MessageInput(
            native_id=f"{native_prefix}-{i}",
            position=i,
            role=Role.USER if i % 2 == 0 else Role.ASSISTANT,
            material_origin=MaterialOrigin.HUMAN_AUTHORED if i % 2 == 0 else MaterialOrigin.ASSISTANT_AUTHORED,
            text=f"message body #{i}",
            occurred_at_ms=1_720_000_000_000 + i * 1000,
        )
        for i in range(message_count)
    )
    return SessionMaterial(
        origin=Origin.CODEX_SESSION,
        native_id="large-session-1",
        title="Large session segmentation fixture",
        messages=messages,
    )


__all__ = ["SMALL_SESSION_REVISION_CREATED_AT", "build_large_session_material", "build_small_session_material"]
