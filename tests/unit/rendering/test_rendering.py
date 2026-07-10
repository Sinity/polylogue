"""Consolidated rendering tests.

MERGED: test_branch_rendering.py + test_none_guards.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from polylogue.archive.models import Message, Session, SessionSummary
from polylogue.core.enums import BlockType, Origin, Provider
from polylogue.rendering.block_models import RenderableBlock
from polylogue.rendering.blocks import (
    render_blocks_html,
    render_blocks_markdown,
    render_blocks_plaintext,
)
from polylogue.rendering.core import format_session_markdown
from polylogue.rendering.renderers.html import (
    _attach_branches,
    render_session_html,
)
from polylogue.types import SessionId
from polylogue.ui.facade_console import PlainConsole
from tests.infra.builders import make_conv, make_msg

# =============================================================================
# Branch rendering helpers (from test_branch_rendering.py)
# =============================================================================

MessagePayload = dict[str, object]


def _branches(payload: MessagePayload) -> list[MessagePayload]:
    value = payload["branches"]
    assert isinstance(value, list)
    branches: list[MessagePayload] = []
    for item in value:
        assert isinstance(item, dict)
        branches.append({str(key): field for key, field in item.items()})
    return branches


def _make_msg(
    id: str,
    role: str = "assistant",
    text: str = "Hello",
    parent_id: str | None = None,
    branch_index: int = 0,
) -> Message:
    return make_msg(
        id=id,
        role=role,
        text=text,
        parent_id=parent_id,
        branch_index=branch_index,
    )


def _make_conv(messages: list[Message], title: str | None = "Branch Test") -> Session:
    return make_conv(
        id="test-conv",
        provider=Provider.CHATGPT,
        title=title,
        messages=messages,
    )


def _make_media_block(
    block_type: str,
    *,
    name: str,
    url: str | None = None,
    mime_type: str | None = None,
) -> RenderableBlock:
    return RenderableBlock(type=block_type, name=name, url=url, mime_type=mime_type)


class TestAttachBranches:
    """Tests for the _attach_branches helper."""

    def test_no_branches_passthrough(self) -> None:
        """Messages without branches are returned unchanged."""
        msgs: list[MessagePayload] = [
            {"id": "m1", "role": "user", "branch_index": 0, "text": "Q"},
            {"id": "m2", "role": "assistant", "branch_index": 0, "text": "A"},
        ]
        result = _attach_branches(msgs)
        assert len(result) == 2
        assert "branches" not in result[0]
        assert "branches" not in result[1]

    def test_branch_attached_to_mainline_sibling(self) -> None:
        """Branch messages are attached to the mainline sibling sharing the same parent."""
        msgs: list[MessagePayload] = [
            {"id": "m1", "role": "user", "branch_index": 0, "parent_message_id": None, "text": "Q"},
            {"id": "m2", "role": "assistant", "branch_index": 0, "parent_message_id": "m1", "text": "A1"},
            {"id": "m3", "role": "assistant", "branch_index": 1, "parent_message_id": "m1", "text": "A2"},
        ]
        result = _attach_branches(msgs)
        assert len(result) == 2
        m2 = next(m for m in result if m["id"] == "m2")
        assert "branches" in m2
        branches = _branches(m2)
        assert len(branches) == 1
        assert branches[0]["id"] == "m3"

    def test_multiple_branches(self) -> None:
        """Multiple branch messages attach to same mainline sibling."""
        msgs: list[MessagePayload] = [
            {"id": "m1", "role": "user", "branch_index": 0, "parent_message_id": None, "text": "Q"},
            {"id": "m2", "role": "assistant", "branch_index": 0, "parent_message_id": "m1", "text": "A1"},
            {"id": "m3", "role": "assistant", "branch_index": 1, "parent_message_id": "m1", "text": "A2"},
            {"id": "m4", "role": "assistant", "branch_index": 2, "parent_message_id": "m1", "text": "A3"},
        ]
        result = _attach_branches(msgs)
        assert len(result) == 2
        m2 = next(m for m in result if m["id"] == "m2")
        assert len(_branches(m2)) == 2

    def test_orphan_branch_becomes_standalone(self) -> None:
        """Branch without mainline sibling is included as standalone."""
        msgs: list[MessagePayload] = [
            {"id": "m1", "role": "user", "branch_index": 0, "parent_message_id": None, "text": "Q"},
            {"id": "m3", "role": "assistant", "branch_index": 1, "parent_message_id": "orphan", "text": "A2"},
        ]
        result = _attach_branches(msgs)
        assert len(result) == 2


class TestBranchRendering:
    """Tests for branch-aware HTML rendering."""

    def test_linear_session_no_branches_section(self) -> None:
        """Linear sessions should not have branch markup."""
        msgs = [_make_msg("m1", "user", "Hello"), _make_msg("m2", "assistant", "Hi")]
        conv = _make_conv(msgs)
        html = render_session_html(conv)
        assert "<details" not in html
        assert "branches" not in html or "branches" in html

    def test_branching_session_has_details(self) -> None:
        """Branching sessions should render <details> sections."""
        msgs = [
            _make_msg("m1", "user", "Question", parent_id=None, branch_index=0),
            _make_msg("m2", "assistant", "Answer 1", parent_id="m1", branch_index=0),
            _make_msg("m3", "assistant", "Answer 2 (edited)", parent_id="m1", branch_index=1),
        ]
        conv = _make_conv(msgs)
        html = render_session_html(conv)
        assert "<details" in html
        assert "<summary>" in html
        assert "1 alternative" in html
        assert "Branch 1" in html

    def test_branch_content_rendered(self) -> None:
        """Branch message content should appear in the HTML."""
        msgs = [
            _make_msg("m1", "user", "Question"),
            _make_msg("m2", "assistant", "Answer 1", parent_id="m1", branch_index=0),
            _make_msg("m3", "assistant", "Answer 2 (edited)", parent_id="m1", branch_index=1),
        ]
        conv = _make_conv(msgs)
        html = render_session_html(conv)
        assert "Answer 1" in html
        assert "Answer 2 (edited)" in html

    def test_multiple_alternatives_label(self) -> None:
        """Multiple branches should use plural label."""
        msgs = [
            _make_msg("m1", "user", "Q"),
            _make_msg("m2", "assistant", "A1", parent_id="m1", branch_index=0),
            _make_msg("m3", "assistant", "A2", parent_id="m1", branch_index=1),
            _make_msg("m4", "assistant", "A3", parent_id="m1", branch_index=2),
        ]
        conv = _make_conv(msgs)
        html = render_session_html(conv)
        assert "2 alternatives" in html

    def test_branch_css_present(self) -> None:
        """Branch CSS classes should be in the output."""
        msgs = [
            _make_msg("m1", "user", "Q"),
            _make_msg("m2", "assistant", "A1", parent_id="m1", branch_index=0),
            _make_msg("m3", "assistant", "A2", parent_id="m1", branch_index=1),
        ]
        conv = _make_conv(msgs)
        html = render_session_html(conv)
        assert "branch-message" in html
        assert "branch-label" in html

    def test_mainline_only_in_top_level(self) -> None:
        """Only mainline messages should be in the top-level message list."""
        msgs = [
            _make_msg("m1", "user", "Q"),
            _make_msg("m2", "assistant", "A1", parent_id="m1", branch_index=0),
            _make_msg("m3", "assistant", "A2 alt", parent_id="m1", branch_index=1),
            _make_msg("m4", "user", "Follow-up"),
        ]
        conv = _make_conv(msgs)
        html = render_session_html(conv)
        assert "Follow-up" in html
        assert "A1" in html
        assert "A2 alt" in html
        assert html.index("<details") < html.index("A2 alt")


class TestMediaBlockRendering:
    """Tests for structured media/document/file rendering across surfaces."""

    def test_media_blocks_render_across_markdown_html_and_plaintext(self) -> None:
        blocks = [
            _make_media_block(
                BlockType.IMAGE.value,
                name="Preview image",
                url="https://example.com/image.png",
                mime_type="image/png",
            ),
            _make_media_block(
                BlockType.DOCUMENT.value,
                name="Spec",
                url="https://example.com/spec.pdf",
                mime_type="application/pdf",
            ),
            _make_media_block("file", name="Archive", mime_type="text/plain"),
        ]

        markdown = render_blocks_markdown(blocks)
        html = render_blocks_html(blocks)
        plaintext = render_blocks_plaintext(blocks)

        assert "[Preview image](https://example.com/image.png) (image/png)" in markdown
        assert "[Spec](https://example.com/spec.pdf) (application/pdf)" in markdown
        assert "[Archive] (text/plain)" in markdown

        assert '<div class="media-block" data-type="image">' in html
        assert '<a class="media-link" href="https://example.com/image.png">Preview image</a>' in html
        assert '<div class="media-block" data-type="document">' in html
        assert '<a class="media-link" href="https://example.com/spec.pdf">Spec</a>' in html
        assert '<div class="media-block" data-type="file">' in html
        assert '<span class="media-name">Archive</span>' in html

        assert "Preview image https://example.com/image.png (image/png)" in plaintext
        assert "Spec https://example.com/spec.pdf (application/pdf)" in plaintext
        assert "Archive (text/plain)" in plaintext


class TestToolUseInputSummary:
    """`_tool_input_summary`'s folded one-line summary for tool_use blocks.

    Regression for the #2629 review: a ChatGPT web-search tool_input shape
    (``{"search_query": [{"q": "..."}], "response_length": "medium"}``)
    previously fell through to the generic key=value fallback, which picks
    the first scalar-valued key -- "response_length=medium" -- silently
    dropping the actual search terms from the folded summary.
    """

    def test_chatgpt_search_query_list_of_dicts_shows_query_text(self) -> None:
        block = RenderableBlock(
            type=BlockType.TOOL_USE.value,
            tool_name="web",
            tool_id="call-1",
            tool_input={
                "search_query": [{"q": "Hetzner Cloud prices"}, {"q": "CCX53 pricing"}],
                "response_length": "medium",
            },
        )
        markdown = render_blocks_markdown([block])
        assert "Hetzner Cloud prices" in markdown
        assert "CCX53 pricing" in markdown
        assert "response_length=medium" not in markdown

    def test_search_query_list_of_plain_strings_shows_query_text(self) -> None:
        block = RenderableBlock(
            type=BlockType.TOOL_USE.value,
            tool_name="web",
            tool_id="call-2",
            tool_input={"search_query": ["plain string query"], "response_length": "short"},
        )
        markdown = render_blocks_markdown([block])
        assert "plain string query" in markdown

    def test_command_tool_input_unaffected(self) -> None:
        block = RenderableBlock(
            type=BlockType.TOOL_USE.value,
            tool_name="Bash",
            tool_id="call-3",
            tool_input={"command": "ls -la"},
        )
        markdown = render_blocks_markdown([block])
        assert "`ls -la`" in markdown


class TestPlainConsoleLiteralOutput:
    """PlainConsole must preserve non-markup literal string content."""

    def test_prints_strings_literally(self, capsys: pytest.CaptureFixture[str]) -> None:
        console = PlainConsole()
        console.print("[bold]literal[/bold]", "```python\nprint(1)\n```")
        output = capsys.readouterr().out
        assert "literal" in output
        assert "```python" in output
        assert "print(1)" in output


# =============================================================================
# None guards and edge cases (from test_none_guards.py)
# =============================================================================


class TestFormatSessionMarkdownNoneGuards:
    """format_session_markdown must handle None text, None role, etc."""

    def _make_conv(self, messages: list[Message], title: str | None = "Test") -> Session:
        return make_conv(
            id="test-conv",
            provider=Provider.UNKNOWN,
            title=title,
            messages=messages,
        )

    def test_none_text_message_skipped(self) -> None:
        """Message with None text should be skipped, not crash."""
        conv = self._make_conv(
            [
                make_msg(id="m1", role="user", text=None),
                make_msg(id="m2", role="assistant", text="Hello!"),
            ]
        )
        md = format_session_markdown(conv)
        assert "Hello!" in md
        assert md.count("## ") == 1

    def test_empty_text_message_skipped(self) -> None:
        """Message with empty text should be skipped."""
        conv = self._make_conv(
            [
                make_msg(id="m1", role="user", text=""),
                make_msg(id="m2", role="assistant", text="Response"),
            ]
        )
        md = format_session_markdown(conv)
        assert "Response" in md

    def test_whitespace_only_text_skipped(self) -> None:
        """Message with whitespace-only text should be skipped."""
        conv = self._make_conv(
            [
                make_msg(id="m1", role="user", text="   \n\t  "),
                make_msg(id="m2", role="assistant", text="Answer"),
            ]
        )
        md = format_session_markdown(conv)
        assert "Answer" in md

    def test_none_role_renders_as_unknown(self) -> None:
        """None role should render as 'unknown', not crash (45c8578)."""
        conv = self._make_conv(
            [
                make_msg(id="m1", role="unknown", text="Message with unknown role"),
            ]
        )
        md = format_session_markdown(conv)
        assert "unknown" in md
        assert "Message with unknown role" in md

    def test_none_title_renders_as_untitled(self) -> None:
        """None title should render as 'Untitled'."""
        conv = self._make_conv(
            [make_msg(id="m1", role="user", text="Hello")],
            title=None,
        )
        md = format_session_markdown(conv)
        assert "Untitled" in md

    def test_json_text_wrapped_in_code_block(self) -> None:
        """JSON text should be wrapped in code blocks."""
        json_text = json.dumps({"key": "value"})
        conv = self._make_conv(
            [
                make_msg(id="m1", role="assistant", text=json_text),
            ]
        )
        md = format_session_markdown(conv)
        assert "```json" in md

    def test_message_text_heading_marker_does_not_create_extra_section(self) -> None:
        """Body lines that look like renderer headers stay message content."""
        conv = self._make_conv(
            [
                make_msg(id="m1", role="user", text="## not a message header"),
                make_msg(id="m2", role="assistant", text="Response"),
            ]
        )
        md = format_session_markdown(conv)
        assert "\\## not a message header" in md
        assert sum(1 for line in md.splitlines() if line.startswith("## ")) == 2

    def test_all_messages_none_text(self) -> None:
        """All messages with None text should produce header-only markdown."""
        conv = self._make_conv(
            [
                make_msg(id="m1", role="user", text=None),
                make_msg(id="m2", role="assistant", text=None),
            ]
        )
        md = format_session_markdown(conv)
        assert "# Test" in md
        assert "## " not in md

    def test_empty_messages_list(self) -> None:
        """Session with no messages should not crash."""
        conv = self._make_conv([])
        md = format_session_markdown(conv)
        assert "# Test" in md


class TestSessionSummaryDisplayDate:
    """SessionSummary.display_date must handle None timestamps (f9c88e2)."""

    def test_both_none_returns_none(self) -> None:
        summary = SessionSummary(
            id=SessionId("test"),
            origin=Origin.UNKNOWN_EXPORT,
            created_at=None,
            updated_at=None,
        )
        assert summary.display_date is None

    def test_only_created_at(self) -> None:
        dt = datetime(2024, 6, 15, tzinfo=timezone.utc)
        summary = SessionSummary(
            id=SessionId("test"),
            origin=Origin.UNKNOWN_EXPORT,
            created_at=dt,
            updated_at=None,
        )
        assert summary.display_date == dt

    def test_only_updated_at(self) -> None:
        dt = datetime(2024, 6, 15, tzinfo=timezone.utc)
        summary = SessionSummary(
            id=SessionId("test"),
            origin=Origin.UNKNOWN_EXPORT,
            created_at=None,
            updated_at=dt,
        )
        assert summary.display_date == dt

    def test_both_present_prefers_updated(self) -> None:
        created = datetime(2024, 1, 1, tzinfo=timezone.utc)
        updated = datetime(2024, 6, 15, tzinfo=timezone.utc)
        summary = SessionSummary(
            id=SessionId("test"),
            origin=Origin.UNKNOWN_EXPORT,
            created_at=created,
            updated_at=updated,
        )
        assert summary.display_date == updated

    def test_display_title_none_title_uses_id(self) -> None:
        summary = SessionSummary(id=SessionId("abcdef12"), origin=Origin.UNKNOWN_EXPORT, title=None)
        assert summary.display_title == "abcdef12"

    def test_display_title_empty_title_uses_id(self) -> None:
        summary = SessionSummary(id=SessionId("abcdef12"), origin=Origin.UNKNOWN_EXPORT, title="")
        assert summary.display_title == "abcdef12"

    def test_display_title_from_metadata(self) -> None:
        summary = SessionSummary(
            id=SessionId("test"),
            origin=Origin.UNKNOWN_EXPORT,
            title="Original",
            metadata={"title": "User Title"},
        )
        assert summary.display_title == "User Title"

    def test_display_title_from_provider_display_label(self) -> None:
        # The provider-derived display label (e.g. a Gemini session with a weak
        # identifier-like title) now overrides the raw title through the typed
        # ``metadata["title"]`` channel rather than a ``provider_meta`` envelope.
        summary = SessionSummary(
            id=SessionId("test"),
            origin=Origin.AISTUDIO_DRIVE,
            title="gemini-20250422-1234",
            metadata={"title": "Review the project plan"},
        )
        assert summary.display_title == "Review the project plan"
