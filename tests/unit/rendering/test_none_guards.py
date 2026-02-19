"""Tests for rendering None guards and edge cases.

Covers:
- 3914533: msg.text is None → Jinja2 TypeError in render
- 45c8578: None role in site builder → crash
- f9c88e2: updated_at is None → display shows nothing
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from polylogue.lib.models import (
    Conversation,
    ConversationSummary,
    Message,
)
from polylogue.lib.messages import MessageCollection
from polylogue.rendering.core import format_conversation_markdown


class TestFormatConversationMarkdownNoneGuards:
    """format_conversation_markdown must handle None text, None role, etc."""

    def _make_conv(self, messages: list[Message], title: str = "Test") -> Conversation:
        return Conversation(
            id="test-conv",
            provider="test",
            title=title,
            messages=MessageCollection(messages=messages),
        )

    def test_none_text_message_skipped(self):
        """Message with None text should be skipped, not crash."""
        conv = self._make_conv([
            Message(id="m1", role="user", text=None),
            Message(id="m2", role="assistant", text="Hello!"),
        ])
        md = format_conversation_markdown(conv)
        assert "Hello!" in md
        # None-text message should be skipped (empty after strip)
        assert md.count("## ") == 1  # Only assistant section

    def test_empty_text_message_skipped(self):
        """Message with empty text should be skipped."""
        conv = self._make_conv([
            Message(id="m1", role="user", text=""),
            Message(id="m2", role="assistant", text="Response"),
        ])
        md = format_conversation_markdown(conv)
        assert "Response" in md

    def test_whitespace_only_text_skipped(self):
        """Message with whitespace-only text should be skipped."""
        conv = self._make_conv([
            Message(id="m1", role="user", text="   \n\t  "),
            Message(id="m2", role="assistant", text="Answer"),
        ])
        md = format_conversation_markdown(conv)
        assert "Answer" in md

    def test_none_role_renders_as_unknown(self):
        """None role should render as 'unknown', not crash (45c8578)."""
        conv = self._make_conv([
            Message(id="m1", role="unknown", text="Message with unknown role"),
        ])
        md = format_conversation_markdown(conv)
        assert "unknown" in md
        assert "Message with unknown role" in md

    def test_none_title_renders_as_untitled(self):
        """None title should render as 'Untitled'."""
        conv = self._make_conv(
            [Message(id="m1", role="user", text="Hello")],
            title=None,
        )
        md = format_conversation_markdown(conv)
        assert "Untitled" in md

    def test_json_text_wrapped_in_code_block(self):
        """JSON text should be wrapped in code blocks."""
        json_text = json.dumps({"key": "value"})
        conv = self._make_conv([
            Message(id="m1", role="assistant", text=json_text),
        ])
        md = format_conversation_markdown(conv)
        assert "```json" in md

    def test_all_messages_none_text(self):
        """All messages with None text should produce header-only markdown."""
        conv = self._make_conv([
            Message(id="m1", role="user", text=None),
            Message(id="m2", role="assistant", text=None),
        ])
        md = format_conversation_markdown(conv)
        assert "# Test" in md
        # No message sections
        assert "## " not in md

    def test_empty_messages_list(self):
        """Conversation with no messages should not crash."""
        conv = self._make_conv([])
        md = format_conversation_markdown(conv)
        assert "# Test" in md


class TestConversationSummaryDisplayDate:
    """ConversationSummary.display_date must handle None timestamps (f9c88e2)."""

    def test_both_none_returns_none(self):
        summary = ConversationSummary(
            id="test", provider="test",
            created_at=None, updated_at=None,
        )
        assert summary.display_date is None

    def test_only_created_at(self):
        dt = datetime(2024, 6, 15, tzinfo=timezone.utc)
        summary = ConversationSummary(
            id="test", provider="test",
            created_at=dt, updated_at=None,
        )
        assert summary.display_date == dt

    def test_only_updated_at(self):
        dt = datetime(2024, 6, 15, tzinfo=timezone.utc)
        summary = ConversationSummary(
            id="test", provider="test",
            created_at=None, updated_at=dt,
        )
        assert summary.display_date == dt

    def test_both_present_prefers_updated(self):
        created = datetime(2024, 1, 1, tzinfo=timezone.utc)
        updated = datetime(2024, 6, 15, tzinfo=timezone.utc)
        summary = ConversationSummary(
            id="test", provider="test",
            created_at=created, updated_at=updated,
        )
        assert summary.display_date == updated

    def test_display_title_none_title_uses_id(self):
        summary = ConversationSummary(id="abcdef12", provider="test", title=None)
        assert summary.display_title == "abcdef12"

    def test_display_title_empty_title_uses_id(self):
        summary = ConversationSummary(id="abcdef12", provider="test", title="")
        # Empty string is falsy, should fall through to ID
        assert summary.display_title == "abcdef12"

    def test_display_title_from_metadata(self):
        summary = ConversationSummary(
            id="test", provider="test",
            title="Original",
            metadata={"title": "User Title"},
        )
        assert summary.display_title == "User Title"
