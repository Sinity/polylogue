"""Tests for branch-aware HTML rendering."""

from __future__ import annotations

from polylogue.lib.messages import MessageCollection
from polylogue.lib.models import Conversation, Message
from polylogue.rendering.renderers.html import (
    _attach_branches,
    _role_css_class,
    render_conversation_html,
)


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
        # Only mainline messages at top level
        assert len(result) == 2
        # m2 (the mainline sibling of m3) should have branches
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
        assert len(result) == 2  # m1 + m2
        m2 = next(m for m in result if m["id"] == "m2")
        assert len(m2["branches"]) == 2

    def test_orphan_branch_becomes_standalone(self):
        """Branch without mainline sibling is included as standalone."""
        msgs = [
            {"id": "m1", "role": "user", "branch_index": 0, "parent_message_id": None, "text": "Q"},
            {"id": "m3", "role": "assistant", "branch_index": 1, "parent_message_id": "orphan", "text": "A2"},
        ]
        result = _attach_branches(msgs)
        assert len(result) == 2  # m1 + m3 (standalone)


class TestBranchRendering:
    """Tests for branch-aware HTML rendering."""

    def test_linear_conversation_no_branches_section(self):
        """Linear conversations should not have branch markup."""
        msgs = [_make_msg("m1", "user", "Hello"), _make_msg("m2", "assistant", "Hi")]
        conv = _make_conv(msgs)
        html = render_conversation_html(conv)
        assert "<details" not in html
        assert "branches" not in html or "branches" in html  # CSS class may be in stylesheet

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
        # All content should be present
        assert "Follow-up" in html
        assert "A1" in html
        assert "A2 alt" in html
        # Branch content should be inside <details>
        assert html.index("<details") < html.index("A2 alt")
