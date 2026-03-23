"""Tests for the canonical semantic facts layer."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from polylogue.lib.messages import MessageCollection
from polylogue.lib.models import Conversation, ConversationSummary, Message
from polylogue.lib.pricing import harmonize_session_cost
from polylogue.lib.semantic_facts import (
    build_conversation_semantic_facts,
    build_mcp_summary_semantic_facts,
    build_projection_semantic_facts,
)
from polylogue.lib.session_profile import build_session_profile
from polylogue.storage.store import AttachmentRecord, ConversationRecord, ConversationRenderProjection, MessageRecord


def _semantic_conversation() -> Conversation:
    return Conversation(
        id="conv-semantic-facts",
        provider="claude-code",
        title="Semantic Facts",
        created_at=datetime(2026, 3, 23, 9, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 23, 9, 5, tzinfo=timezone.utc),
        messages=MessageCollection(
            messages=[
                Message(
                    id="u1",
                    role="user",
                    provider="claude-code",
                    text="Please inspect README.md and summarize the result clearly.",
                    timestamp=datetime(2026, 3, 23, 9, 0, tzinfo=timezone.utc),
                ),
                Message(
                    id="a1",
                    role="assistant",
                    provider="claude-code",
                    text="I will inspect the file",
                    timestamp=datetime(2026, 3, 23, 9, 1, tzinfo=timezone.utc),
                    provider_meta={
                        "raw": {
                            "type": "assistant",
                            "uuid": "a1",
                            "message": {
                                "role": "assistant",
                                "content": [
                                    {"type": "text", "text": "I will inspect the file"},
                                    {
                                        "type": "tool_use",
                                        "id": "tool-1",
                                        "name": "Read",
                                        "input": {"file_path": "/realm/project/polylogue/README.md"},
                                    },
                                ],
                            },
                        }
                    },
                ),
                Message(
                    id="a2",
                    role="assistant",
                    provider="claude-code",
                    text="",
                    timestamp=datetime(2026, 3, 23, 9, 4, tzinfo=timezone.utc),
                    branch_index=1,
                    attachments=[],
                    provider_meta={
                        "raw": {
                            "type": "assistant",
                            "uuid": "a2",
                            "message": {
                                "role": "assistant",
                                "content": [{"type": "thinking", "thinking": "step by step"}],
                            },
                        }
                    },
                ),
            ]
        ),
    )


def test_build_projection_semantic_facts_counts_renderable_and_empty_messages() -> None:
    projection = ConversationRenderProjection(
        conversation=ConversationRecord(
            conversation_id="conv-projection",
            provider_name="chatgpt",
            provider_conversation_id="provider-conv-projection",
            title="Projection Facts",
            content_hash="hash-projection",
        ),
        messages=[
            MessageRecord(
                message_id="m1",
                conversation_id="conv-projection",
                role="user",
                text="hello",
                sort_key=1.0,
                content_hash="hash-m1",
            ),
            MessageRecord(
                message_id="m2",
                conversation_id="conv-projection",
                role="assistant",
                text=None,
                sort_key=2.0,
                content_hash="hash-m2",
                has_thinking=1,
            ),
            MessageRecord(
                message_id="m3",
                conversation_id="conv-projection",
                role="assistant",
                text=None,
                content_hash="hash-m3",
                has_tool_use=1,
            ),
        ],
        attachments=[
            AttachmentRecord(
                attachment_id="att-m2",
                conversation_id="conv-projection",
                message_id="m2",
                path="/tmp/att-m2.txt",
            )
        ],
    )

    facts = build_projection_semantic_facts(projection)

    assert facts.total_messages == 3
    assert facts.renderable_messages == 2
    assert facts.timestamped_renderable_messages == 2
    assert facts.attachment_count == 1
    assert facts.empty_messages == 1
    assert facts.thinking_messages == 1
    assert facts.tool_messages == 1
    assert facts.renderable_role_counts == {"assistant": 1, "user": 1}


def test_build_conversation_semantic_facts_collects_semantic_counts() -> None:
    conversation = _semantic_conversation()

    facts = build_conversation_semantic_facts(conversation)

    assert facts.conversation_id == "conv-semantic-facts"
    assert facts.provider == "claude-code"
    assert facts.total_messages == 3
    assert facts.substantive_messages == 1
    assert facts.text_messages == 2
    assert facts.message_ids == ("u1", "a1", "a2")
    assert facts.text_message_ids == ("u1", "a1")
    assert facts.text_role_counts == {"assistant": 1, "user": 1}
    assert facts.timestamped_text_messages == 2
    assert facts.thinking_messages == 1
    assert facts.tool_messages == 1
    assert facts.branch_messages == 1
    assert facts.word_count == 13
    assert facts.tool_category_counts == {"file_read": 1}
    assert facts.message_facts[1].tool_category_counts == {"file_read": 1}
    assert facts.message_facts[1].affected_paths == ("/realm/project/polylogue/README.md",)
    assert facts.first_message_at == datetime(2026, 3, 23, 9, 0, tzinfo=timezone.utc)
    assert facts.last_message_at == datetime(2026, 3, 23, 9, 4, tzinfo=timezone.utc)
    assert facts.wall_duration_ms == 240000


def test_build_session_profile_reuses_shared_semantic_facts() -> None:
    profile = build_session_profile(_semantic_conversation())

    assert profile.message_count == 3
    assert profile.substantive_count == 1
    assert profile.tool_use_count == 1
    assert profile.thinking_count == 1
    assert profile.tool_categories == {"file_read": 1}
    assert profile.canonical_projects == ("polylogue",)
    assert profile.first_message_at == datetime(2026, 3, 23, 9, 0, tzinfo=timezone.utc)
    assert profile.last_message_at == datetime(2026, 3, 23, 9, 4, tzinfo=timezone.utc)
    assert profile.wall_duration_ms == 240000


def test_build_mcp_summary_semantic_facts_uses_canonical_summary_shape() -> None:
    summary = ConversationSummary(
        id="conv-semantic-facts",
        provider="claude-code",
        title="Semantic Facts",
        created_at=datetime(2026, 3, 23, 9, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 23, 9, 5, tzinfo=timezone.utc),
    )

    facts = build_mcp_summary_semantic_facts(summary, message_count=3)

    assert facts.conversation_id == "conv-semantic-facts"
    assert facts.provider == "claude-code"
    assert facts.title == "Semantic Facts"
    assert facts.messages == 3
    assert facts.created_at == "2026-03-23T09:00:00+00:00"
    assert facts.updated_at == "2026-03-23T09:05:00+00:00"
    assert facts.tags == ()
    assert facts.summary is None


def test_harmonize_session_cost_uses_canonical_harmonized_model_and_tokens() -> None:
    conversation = Conversation(
        id="conv-cost",
        provider="chatgpt",
        messages=MessageCollection(
            messages=[
                Message(
                    id="m1",
                    role="assistant",
                    provider="chatgpt",
                    text="Estimated response",
                    provider_meta={
                        "model": "gpt-4o",
                        "usage": {
                            "input_tokens": 1000,
                            "output_tokens": 500,
                        },
                    },
                )
            ]
        ),
    )

    cost_usd, is_estimated = harmonize_session_cost(conversation)

    assert is_estimated is True
    assert cost_usd == pytest.approx(0.0075)
