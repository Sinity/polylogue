"""Tests for provider-specific model viewport methods.

Retained: ClaudeAI message and conversation structural tests (timestamp
parsing, to_meta, to_content_blocks, conversation properties, integration).
Role normalization example tables superseded by test_harmonization_contracts.py
property laws; Claude Code / Gemini / ChatGPT viewport example tables removed.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from polylogue.sources.providers.claude_ai import ClaudeAIChatMessage, ClaudeAIConversation


class TestClaudeAIChatMessageRoleNormalizedAndTimestamp:
    """Test ClaudeAIChatMessage role_normalized and parsed_timestamp."""

    CLAUDE_AI_ROLE_MAPPING = [
        ("human", "user"),
        ("assistant", "assistant"),
        ("system", "system"),
        ("", "unknown"),
    ]

    @pytest.mark.parametrize("sender,expected", CLAUDE_AI_ROLE_MAPPING, ids=[
        "human", "assistant", "system", "empty"
    ])
    def test_role_normalized(self, sender, expected):
        msg = ClaudeAIChatMessage(uuid="1", text="hi", sender=sender)
        assert msg.role_normalized == expected

    @pytest.mark.parametrize("created_at,expect_datetime,test_id", [
        ("2024-06-15T10:30:00Z", True, "iso_z"),
        ("2024-06-15T10:30:00+00:00", True, "iso_offset"),
        ("not-a-date", False, "invalid"),
        ("", False, "empty"),
        (None, False, "none"),
        ("2024-06-15 10:30:00", None, "malformed_iso"),
    ])
    def test_parsed_timestamp(self, created_at, expect_datetime, test_id):
        msg = ClaudeAIChatMessage(
            uuid="1", text="hi", sender="human",
            created_at=created_at,
        )
        ts = msg.parsed_timestamp
        if expect_datetime is True:
            assert ts is not None and isinstance(ts, datetime)
        elif expect_datetime is False:
            assert ts is None
        else:
            assert ts is None or isinstance(ts, datetime)


class TestClaudeAIChatMessageToMeta:
    """Test ClaudeAIChatMessage.to_meta conversion."""

    @pytest.mark.parametrize("uuid_,sender,created_at,expect_timestamp,test_id", [
        ("msg-1", "human", None, False, "basic"),
        ("msg-2", "assistant", "2024-06-15T10:30:00Z", True, "with_timestamp"),
        ("msg-3", "human", "bad-date", False, "invalid_timestamp"),
    ])
    def test_to_meta(self, uuid_, sender, created_at, expect_timestamp, test_id):
        msg = ClaudeAIChatMessage(uuid=uuid_, text="hi", sender=sender, created_at=created_at)
        meta = msg.to_meta()
        assert meta.id == uuid_
        assert meta.provider == "claude-ai"
        if sender == "human":
            assert meta.role == "user"
        else:
            assert meta.role == sender
        if expect_timestamp:
            assert meta.timestamp is not None
            assert meta.timestamp.year == 2024
        else:
            assert meta.timestamp is None


class TestClaudeAIChatMessageToContentBlocks:
    """Test ClaudeAIChatMessage.to_content_blocks."""

    @pytest.mark.parametrize("text,sender,test_id", [
        ("hello world", "human", "basic"),
        ("response", "assistant", "assistant"),
        ("", "human", "empty_text"),
    ])
    def test_to_content_blocks(self, text, sender, test_id):
        from polylogue.lib.viewports import ContentType
        msg = ClaudeAIChatMessage(uuid="1", text=text, sender=sender)
        blocks = msg.to_content_blocks()
        assert len(blocks) == 1
        assert blocks[0].type == ContentType.TEXT
        assert blocks[0].text == text


class TestClaudeAIConversationProperties:
    """Test ClaudeAIConversation properties (title, created_datetime, updated_datetime)."""

    @pytest.mark.parametrize("name,expected_title,test_id", [
        ("My Conversation", "My Conversation", "title_from_name"),
        ("", "", "title_empty_name"),
    ])
    def test_title(self, name, expected_title, test_id):
        conv = ClaudeAIConversation(
            uuid="c-1", name=name,
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )
        assert conv.title == expected_title

    @pytest.mark.parametrize("date_str,expect_valid,test_id", [
        ("2024-06-15T10:30:00Z", True, "iso_z"),
        ("2024-06-15T10:30:00+05:00", True, "iso_offset"),
        ("not-a-date", False, "invalid"),
    ])
    def test_created_datetime(self, date_str, expect_valid, test_id):
        conv = ClaudeAIConversation(
            uuid="c-1", name="Test",
            created_at=date_str,
            updated_at="2024-06-15T10:30:00Z",
        )
        dt = conv.created_datetime
        if expect_valid:
            assert dt is not None
            assert dt.year == 2024 and dt.month == 6 and dt.day == 15
        else:
            assert dt is None

    @pytest.mark.parametrize("updated_date,expect_valid,test_id", [
        ("2024-06-16T11:30:00Z", True, "valid"),
        ("bad-date", False, "invalid"),
    ])
    def test_updated_datetime(self, updated_date, expect_valid, test_id):
        conv = ClaudeAIConversation(
            uuid="c-1", name="Test",
            created_at="2024-01-01T00:00:00Z",
            updated_at=updated_date,
        )
        dt = conv.updated_datetime
        if expect_valid:
            assert dt is not None
            assert dt.year == 2024 and dt.month == 6 and dt.day == 16
        else:
            assert dt is None


class TestClaudeAIConversationMessages:
    """Test ClaudeAIConversation.messages property."""

    def test_messages_alias(self):
        """messages alias returns chat_messages."""
        conv = ClaudeAIConversation(
            uuid="c-1",
            name="Test",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            chat_messages=[
                ClaudeAIChatMessage(uuid="m1", text="hi", sender="human"),
                ClaudeAIChatMessage(uuid="m2", text="hello", sender="assistant"),
            ],
        )
        messages = conv.messages
        assert len(messages) == 2
        assert messages[0].text == "hi"
        assert messages[1].text == "hello"

    def test_messages_empty(self):
        conv = ClaudeAIConversation(
            uuid="c-1",
            name="Test",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )
        messages = conv.messages
        assert len(messages) == 0


class TestClaudeAIConversationIntegration:
    """Integration tests for full conversation workflow."""

    def test_full_conversation_workflow(self):
        """Test a complete conversation with all features."""
        conv = ClaudeAIConversation(
            uuid="conv-full",
            name="Full Test",
            created_at="2024-06-15T10:00:00Z",
            updated_at="2024-06-15T11:00:00Z",
            chat_messages=[
                ClaudeAIChatMessage(
                    uuid="m1",
                    text="Hello, how are you?",
                    sender="human",
                    created_at="2024-06-15T10:00:00Z",
                ),
                ClaudeAIChatMessage(
                    uuid="m2",
                    text="I'm doing well, thanks!",
                    sender="assistant",
                    created_at="2024-06-15T10:01:00Z",
                ),
            ],
        )

        assert conv.title == "Full Test"
        assert conv.created_datetime is not None
        assert conv.updated_datetime is not None
        assert len(conv.messages) == 2

        for msg in conv.messages:
            meta = msg.to_meta()
            assert meta is not None
            blocks = msg.to_content_blocks()
            assert len(blocks) > 0

    def test_conversation_with_invalid_timestamps(self):
        """Test conversation with unparseable timestamps."""
        conv = ClaudeAIConversation(
            uuid="conv-bad",
            name="Bad Times",
            created_at="bad-created",
            updated_at="bad-updated",
            chat_messages=[
                ClaudeAIChatMessage(
                    uuid="m1",
                    text="Hi",
                    sender="human",
                    created_at="bad-message-date",
                ),
            ],
        )

        assert conv.created_datetime is None
        assert conv.updated_datetime is None
        assert conv.messages[0].parsed_timestamp is None
