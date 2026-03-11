"""Pinned provider-model regressions that remain worth keeping as examples.

Broad viewport/meta/content laws now live in the unified semantic and viewport
contract suites. This file keeps only exact timestamp/conversation regressions
that benefit from explicit examples.
"""

from __future__ import annotations

import pytest

from polylogue.sources.providers.claude_ai import ClaudeAIChatMessage, ClaudeAIConversation


class TestClaudeAIChatMessagePinnedRegressions:
    """Small pinned cases for Claude AI message quirks."""

    def test_empty_sender_normalizes_to_unknown(self) -> None:
        message = ClaudeAIChatMessage(uuid="m1", text="hi", sender="")
        assert message.role_normalized == "unknown"

    @pytest.mark.parametrize("created_at", ["not-a-date", "", None])
    def test_invalid_or_malformed_timestamp_returns_none(self, created_at) -> None:
        message = ClaudeAIChatMessage(uuid="m1", text="hi", sender="human", created_at=created_at)
        assert message.parsed_timestamp is None

    def test_space_separated_timestamp_is_still_accepted(self) -> None:
        message = ClaudeAIChatMessage(
            uuid="m1",
            text="hi",
            sender="human",
            created_at="2024-06-15 10:30:00",
        )
        assert message.parsed_timestamp is not None


class TestClaudeAIConversationProperties:
    """Pinned conversation property regressions."""

    @pytest.mark.parametrize("name,expected_title", [("My Conversation", "My Conversation"), ("", "")])
    def test_title(self, name: str, expected_title: str) -> None:
        conversation = ClaudeAIConversation(
            uuid="c-1",
            name=name,
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )
        assert conversation.title == expected_title

    @pytest.mark.parametrize("date_str,expect_valid", [("2024-06-15T10:30:00Z", True), ("2024-06-15T10:30:00+05:00", True), ("not-a-date", False)])
    def test_created_datetime(self, date_str: str, expect_valid: bool) -> None:
        conversation = ClaudeAIConversation(
            uuid="c-1",
            name="Test",
            created_at=date_str,
            updated_at="2024-06-15T10:30:00Z",
        )
        created = conversation.created_datetime
        if expect_valid:
            assert created is not None
            assert created.year == 2024 and created.month == 6 and created.day == 15
        else:
            assert created is None

    @pytest.mark.parametrize("updated_date,expect_valid", [("2024-06-16T11:30:00Z", True), ("bad-date", False)])
    def test_updated_datetime(self, updated_date: str, expect_valid: bool) -> None:
        conversation = ClaudeAIConversation(
            uuid="c-1",
            name="Test",
            created_at="2024-01-01T00:00:00Z",
            updated_at=updated_date,
        )
        updated = conversation.updated_datetime
        if expect_valid:
            assert updated is not None
            assert updated.year == 2024 and updated.month == 6 and updated.day == 16
        else:
            assert updated is None


class TestClaudeAIConversationMessages:
    """Explicit message alias examples."""

    def test_messages_alias(self) -> None:
        conversation = ClaudeAIConversation(
            uuid="c-1",
            name="Test",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            chat_messages=[
                ClaudeAIChatMessage(uuid="m1", text="hi", sender="human"),
                ClaudeAIChatMessage(uuid="m2", text="hello", sender="assistant"),
            ],
        )
        messages = conversation.messages
        assert len(messages) == 2
        assert messages[0].text == "hi"
        assert messages[1].text == "hello"

    def test_messages_empty(self) -> None:
        conversation = ClaudeAIConversation(
            uuid="c-1",
            name="Test",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )
        assert conversation.messages == []


class TestClaudeAIConversationIntegration:
    """Pinned end-to-end examples that still add value."""

    def test_full_conversation_workflow(self) -> None:
        conversation = ClaudeAIConversation(
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

        assert conversation.title == "Full Test"
        assert conversation.created_datetime is not None
        assert conversation.updated_datetime is not None
        assert len(conversation.messages) == 2

        for message in conversation.messages:
            meta = message.to_meta()
            assert meta is not None
            blocks = message.to_content_blocks()
            assert len(blocks) > 0

    def test_conversation_with_invalid_timestamps(self) -> None:
        conversation = ClaudeAIConversation(
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

        assert conversation.created_datetime is None
        assert conversation.updated_datetime is None
        assert conversation.messages[0].parsed_timestamp is None
