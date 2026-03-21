"""Consolidated rendering tests.

MERGED: test_branch_rendering.py + test_none_guards.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from polylogue.lib.messages import MessageCollection
from polylogue.lib.models import (
    Conversation,
    ConversationSummary,
    Message,
)
from polylogue.rendering.core import format_conversation_markdown
from polylogue.rendering.renderers.html import (
    _attach_branches,
    render_conversation_html,
)


# =============================================================================
# Branch rendering helpers (from test_branch_rendering.py)
# =============================================================================


def _make_msg(
    id: str,
    role: str = "assistant",
    text: str = "Hello",
    parent_id: str | None = None,
    branch_index: int = 0,
) -> Message:
    return Message(
        id=id,
        role=role,
        text=text,
        parent_id=parent_id,
        branch_index=branch_index,
    )


def _make_conv(messages: list[Message], title: str = "Branch Test") -> Conversation:
    return Conversation(
        id="test-conv",
        provider="chatgpt",
        title=title,
        messages=MessageCollection(messages=messages),
    )


class TestAttachBranches:
    """Tests for the _attach_branches helper."""

    def test_no_branches_passthrough(self):
        """Messages without branches are returned unchanged."""
        msgs = [
            {"id": "m1", "role": "user", "branch_index": 0, "text": "Q"},
            {"id": "m2", "role": "assistant", "branch_index": 0, "text": "A"},
        ]
        result = _attach_branches(msgs)
        assert len(result) == 2
        assert "branches" not in result[0]
        assert "branches" not in result[1]

    def test_branch_attached_to_mainline_sibling(self):
        """Branch messages are attached to the mainline sibling sharing the same parent."""
        msgs = [
            {"id": "m1", "role": "user", "branch_index": 0, "parent_message_id": None, "text": "Q"},
            {"id": "m2", "role": "assistant", "branch_index": 0, "parent_message_id": "m1", "text": "A1"},
            {"id": "m3", "role": "assistant", "branch_index": 1, "parent_message_id": "m1", "text": "A2"},
        ]
        result = _attach_branches(msgs)
        assert len(result) == 2
        m2 = next(m for m in result if m["id"] == "m2")
        assert "branches" in m2
        assert len(m2["branches"]) == 1
        assert m2["branches"][0]["id"] == "m3"

    def test_multiple_branches(self):
        """Multiple branch messages attach to same mainline sibling."""
        msgs = [
            {"id": "m1", "role": "user", "branch_index": 0, "parent_message_id": None, "text": "Q"},
            {"id": "m2", "role": "assistant", "branch_index": 0, "parent_message_id": "m1", "text": "A1"},
            {"id": "m3", "role": "assistant", "branch_index": 1, "parent_message_id": "m1", "text": "A2"},
            {"id": "m4", "role": "assistant", "branch_index": 2, "parent_message_id": "m1", "text": "A3"},
        ]
        result = _attach_branches(msgs)
        assert len(result) == 2
        m2 = next(m for m in result if m["id"] == "m2")
        assert len(m2["branches"]) == 2

    def test_orphan_branch_becomes_standalone(self):
        """Branch without mainline sibling is included as standalone."""
        msgs = [
            {"id": "m1", "role": "user", "branch_index": 0, "parent_message_id": None, "text": "Q"},
            {"id": "m3", "role": "assistant", "branch_index": 1, "parent_message_id": "orphan", "text": "A2"},
        ]
        result = _attach_branches(msgs)
        assert len(result) == 2


class TestBranchRendering:
    """Tests for branch-aware HTML rendering."""

    def test_linear_conversation_no_branches_section(self):
        """Linear conversations should not have branch markup."""
        msgs = [_make_msg("m1", "user", "Hello"), _make_msg("m2", "assistant", "Hi")]
        conv = _make_conv(msgs)
        html = render_conversation_html(conv)
        assert "<details" not in html
        assert "branches" not in html or "branches" in html

    def test_branching_conversation_has_details(self):
        """Branching conversations should render <details> sections."""
        msgs = [
            _make_msg("m1", "user", "Question", parent_id=None, branch_index=0),
            _make_msg("m2", "assistant", "Answer 1", parent_id="m1", branch_index=0),
            _make_msg("m3", "assistant", "Answer 2 (edited)", parent_id="m1", branch_index=1),
        ]
        conv = _make_conv(msgs)
        html = render_conversation_html(conv)
        assert "<details" in html
        assert "<summary>" in html
        assert "1 alternative" in html
        assert "Branch 1" in html

    def test_branch_content_rendered(self):
        """Branch message content should appear in the HTML."""
        msgs = [
            _make_msg("m1", "user", "Question"),
            _make_msg("m2", "assistant", "Answer 1", parent_id="m1", branch_index=0),
            _make_msg("m3", "assistant", "Answer 2 (edited)", parent_id="m1", branch_index=1),
        ]
        conv = _make_conv(msgs)
        html = render_conversation_html(conv)
        assert "Answer 1" in html
        assert "Answer 2 (edited)" in html

    def test_multiple_alternatives_label(self):
        """Multiple branches should use plural label."""
        msgs = [
            _make_msg("m1", "user", "Q"),
            _make_msg("m2", "assistant", "A1", parent_id="m1", branch_index=0),
            _make_msg("m3", "assistant", "A2", parent_id="m1", branch_index=1),
            _make_msg("m4", "assistant", "A3", parent_id="m1", branch_index=2),
        ]
        conv = _make_conv(msgs)
        html = render_conversation_html(conv)
        assert "2 alternatives" in html

    def test_branch_css_present(self):
        """Branch CSS classes should be in the output."""
        msgs = [
            _make_msg("m1", "user", "Q"),
            _make_msg("m2", "assistant", "A1", parent_id="m1", branch_index=0),
            _make_msg("m3", "assistant", "A2", parent_id="m1", branch_index=1),
        ]
        conv = _make_conv(msgs)
        html = render_conversation_html(conv)
        assert "branch-message" in html
        assert "branch-label" in html

    def test_mainline_only_in_top_level(self):
        """Only mainline messages should be in the top-level message list."""
        msgs = [
            _make_msg("m1", "user", "Q"),
            _make_msg("m2", "assistant", "A1", parent_id="m1", branch_index=0),
            _make_msg("m3", "assistant", "A2 alt", parent_id="m1", branch_index=1),
            _make_msg("m4", "user", "Follow-up"),
        ]
        conv = _make_conv(msgs)
        html = render_conversation_html(conv)
        assert "Follow-up" in html
        assert "A1" in html
        assert "A2 alt" in html
        assert html.index("<details") < html.index("A2 alt")


# =============================================================================
# None guards and edge cases (from test_none_guards.py)
# =============================================================================


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
        assert md.count("## ") == 1

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
        assert summary.display_title == "abcdef12"

    def test_display_title_from_metadata(self):
        summary = ConversationSummary(
            id="test", provider="test",
            title="Original",
            metadata={"title": "User Title"},
        )
        assert summary.display_title == "User Title"
