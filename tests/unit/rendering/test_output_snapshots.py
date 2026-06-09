"""Snapshot tests for rendered session output.

These tests replace fragile string-searching assertions with syrupy snapshot
regression detection. The first run creates .ambr baseline files; subsequent
runs detect any accidental output changes.

Run ``pytest --snapshot-update`` to regenerate snapshots after intentional
renderer changes.

Anti-pattern (rejected case): we intentionally do NOT snapshot
non-deterministic fields like wall-clock timestamps, internal hash prefixes,
or process timings. All sessions built here use stable IDs, fixed roles,
and fixed text so that the snapshot files pin output *shape* (column order,
heading structure, redaction markers, code-fence preservation, attachment
shape) rather than implementation noise. If a future renderer change wants
to add e.g. a per-render generated UUID, the test must be updated to redact
it before the snapshot comparison — not the snapshot baseline regenerated to
absorb the change.
"""

from __future__ import annotations

import pytest

syrupy = pytest.importorskip("syrupy")

from polylogue.archive.attachment.models import Attachment
from polylogue.archive.models import Message, Session
from polylogue.rendering.core_markdown import format_session_markdown
from polylogue.rendering.renderers.html import render_session_html
from polylogue.types import BlockType
from tests.infra.builders import make_conv as build_conv
from tests.infra.builders import make_msg as build_msg


def _make_msg(
    id: str,
    role: str = "assistant",
    text: str = "Hello",
    parent_id: str | None = None,
    branch_index: int = 0,
) -> Message:
    return build_msg(
        id=id,
        role=role,
        text=text,
        parent_id=parent_id,
        branch_index=branch_index,
    )


def _make_conv(messages: list[Message], title: str = "Snapshot Test") -> Session:
    return build_conv(
        id="snap-conv-01",
        provider="chatgpt",
        title=title,
        messages=messages,
    )


# ---------------------------------------------------------------------------
# HTML renderer snapshots
# ---------------------------------------------------------------------------


def test_linear_session_html_snapshot(snapshot: object) -> None:
    """Linear 2-message session renders stably."""
    msgs = [_make_msg("m1", "user", "Hello there"), _make_msg("m2", "assistant", "Hi!")]
    conv = _make_conv(msgs, title="Linear Session")
    html = render_session_html(conv)
    assert html == snapshot


def test_branching_session_html_snapshot(snapshot: object) -> None:
    """Branching session with 1 alternative renders stably."""
    msgs = [
        _make_msg("m1", "user", "Question", parent_id=None, branch_index=0),
        _make_msg("m2", "assistant", "Answer 1", parent_id="m1", branch_index=0),
        _make_msg("m3", "assistant", "Answer 2 (edited)", parent_id="m1", branch_index=1),
    ]
    conv = _make_conv(msgs, title="Branching Session")
    html = render_session_html(conv)
    assert html == snapshot


def test_multi_branch_session_html_snapshot(snapshot: object) -> None:
    """Session with 2 alternatives renders stably."""
    msgs = [
        _make_msg("m1", "user", "Question"),
        _make_msg("m2", "assistant", "A1", parent_id="m1", branch_index=0),
        _make_msg("m3", "assistant", "A2", parent_id="m1", branch_index=1),
        _make_msg("m4", "assistant", "A3", parent_id="m1", branch_index=2),
    ]
    conv = _make_conv(msgs, title="Multi-Branch Session")
    html = render_session_html(conv)
    assert html == snapshot


def test_session_with_followup_html_snapshot(snapshot: object) -> None:
    """Branching session with post-branch follow-up renders stably."""
    msgs = [
        _make_msg("m1", "user", "Q"),
        _make_msg("m2", "assistant", "A1", parent_id="m1", branch_index=0),
        _make_msg("m3", "assistant", "A2 alt", parent_id="m1", branch_index=1),
        _make_msg("m4", "user", "Follow-up"),
    ]
    conv = _make_conv(msgs, title="Followup After Branch")
    html = render_session_html(conv)
    assert html == snapshot


# ---------------------------------------------------------------------------
# Markdown renderer snapshots
#
# These pin user-visible Markdown contract shape: heading levels, fenced code
# preservation, structured-block expansion (tool_use/thinking/attachments),
# JSON pretty-printing. Sessions are built with stable IDs and fixed
# content so the snapshot files are deterministic.
# ---------------------------------------------------------------------------


def test_empty_session_markdown_snapshot(snapshot: object) -> None:
    """A session with no messages renders header-only Markdown."""
    conv = build_conv(
        id="snap-md-empty",
        provider="chatgpt",
        title="Empty Session",
        messages=[],
    )
    md = format_session_markdown(conv)
    assert md == snapshot


def test_single_user_turn_markdown_snapshot(snapshot: object) -> None:
    """Single user message renders one ``## user`` section."""
    msgs = [build_msg(id="m1", role="user", text="What is the meaning of life?")]
    conv = build_conv(
        id="snap-md-user",
        provider="chatgpt",
        title="Single User Turn",
        messages=msgs,
    )
    md = format_session_markdown(conv)
    assert md == snapshot


def test_single_assistant_turn_markdown_snapshot(snapshot: object) -> None:
    """Single assistant message renders one ``## assistant`` section."""
    msgs = [
        build_msg(
            id="m1",
            role="assistant",
            text="The answer is 42.",
        )
    ]
    conv = build_conv(
        id="snap-md-assistant",
        provider="chatgpt",
        title="Single Assistant Turn",
        messages=msgs,
    )
    md = format_session_markdown(conv)
    assert md == snapshot


def test_tool_use_roundtrip_markdown_snapshot(snapshot: object) -> None:
    """Tool-use + tool-result blocks render in markdown form."""
    msgs = [
        build_msg(id="m1", role="user", text="List files in cwd"),
        build_msg(
            id="m2",
            role="assistant",
            text="",
            content_blocks=[
                {
                    "type": BlockType.TOOL_USE.value,
                    "name": "bash",
                    "id": "call-1",
                    "input": {"command": "ls"},
                },
            ],
        ),
        build_msg(
            id="m3",
            role="tool",
            text="",
            content_blocks=[
                {
                    "type": BlockType.TOOL_RESULT.value,
                    "tool_use_id": "call-1",
                    "content": "README.md\npyproject.toml\n",
                }
            ],
        ),
    ]
    conv = build_conv(
        id="snap-md-tool",
        provider="claude-ai",
        title="Tool Use Roundtrip",
        messages=msgs,
    )
    md = format_session_markdown(conv)
    assert md == snapshot


def test_thinking_block_markdown_snapshot(snapshot: object) -> None:
    """Thinking block renders as a collapsible ``<details>`` section."""
    msgs = [
        build_msg(
            id="m1",
            role="assistant",
            text="",
            content_blocks=[
                {
                    "type": BlockType.THINKING.value,
                    "text": "Let me think step by step about this.",
                },
                {
                    "type": BlockType.TEXT.value,
                    "text": "Final answer.",
                },
            ],
        )
    ]
    conv = build_conv(
        id="snap-md-thinking",
        provider="claude-ai",
        title="Thinking Block",
        messages=msgs,
    )
    md = format_session_markdown(conv)
    assert md == snapshot


def test_attachment_markdown_snapshot(snapshot: object) -> None:
    """Messages with attachments render an ``Attachment:`` line per asset."""
    msgs = [
        build_msg(
            id="m1",
            role="user",
            text="Please review this document",
            attachments=[
                Attachment(
                    id="att-1",
                    name="spec.pdf",
                    mime_type="application/pdf",
                    path="assets/spec.pdf",
                ),
            ],
        ),
    ]
    conv = build_conv(
        id="snap-md-attachment",
        provider="chatgpt",
        title="Attachment Message",
        messages=msgs,
    )
    md = format_session_markdown(conv)
    assert md == snapshot


def test_code_fence_with_backticks_markdown_snapshot(snapshot: object) -> None:
    """Code blocks render via a code block; embedded backticks preserved verbatim."""
    msgs = [
        build_msg(
            id="m1",
            role="assistant",
            text="",
            content_blocks=[
                {
                    "type": BlockType.CODE.value,
                    "language": "python",
                    "text": "# example\nprint('inline `backticks` survive')\n",
                }
            ],
        )
    ]
    conv = build_conv(
        id="snap-md-code",
        provider="chatgpt",
        title="Code With Backticks",
        messages=msgs,
    )
    md = format_session_markdown(conv)
    assert md == snapshot


def test_json_payload_markdown_snapshot(snapshot: object) -> None:
    """Plain-text JSON payloads get wrapped in a fenced ``json`` block."""
    msgs = [
        build_msg(
            id="m1",
            role="assistant",
            text='{"answer": 42, "ok": true}',
        )
    ]
    conv = build_conv(
        id="snap-md-json",
        provider="chatgpt",
        title="JSON Payload",
        messages=msgs,
    )
    md = format_session_markdown(conv)
    assert md == snapshot


# ---------------------------------------------------------------------------
# HTML renderer matrix snapshots
#
# Mirror the markdown matrix to pin contract-shaped HTML output: empty
# session, single turns, structured blocks, attachments. Renderer-internal
# noise (Pygments CSS classes) is preserved deliberately — those classes ARE
# part of the rendered surface contract and any change to them affects shipped
# stylesheets. See the module docstring "anti-pattern" note.
# ---------------------------------------------------------------------------


def test_empty_session_html_snapshot(snapshot: object) -> None:
    """Empty session renders chrome (header/template) but no message rows."""
    conv = build_conv(
        id="snap-html-empty",
        provider="chatgpt",
        title="Empty Session",
        messages=[],
    )
    html = render_session_html(conv)
    assert html == snapshot


def test_single_user_turn_html_snapshot(snapshot: object) -> None:
    """Single user message renders one message row."""
    msgs = [build_msg(id="m1", role="user", text="What is the meaning of life?")]
    conv = build_conv(
        id="snap-html-user",
        provider="chatgpt",
        title="Single User Turn",
        messages=msgs,
    )
    html = render_session_html(conv)
    assert html == snapshot


def test_tool_use_roundtrip_html_snapshot(snapshot: object) -> None:
    """Tool-use round-trip renders structured tool/tool-result blocks in HTML."""
    msgs = [
        build_msg(id="m1", role="user", text="List files in cwd"),
        build_msg(
            id="m2",
            role="assistant",
            text="",
            content_blocks=[
                {
                    "type": BlockType.TOOL_USE.value,
                    "name": "bash",
                    "id": "call-1",
                    "input": {"command": "ls"},
                },
            ],
        ),
        build_msg(
            id="m3",
            role="tool",
            text="",
            content_blocks=[
                {
                    "type": BlockType.TOOL_RESULT.value,
                    "tool_use_id": "call-1",
                    "content": "README.md\npyproject.toml\n",
                }
            ],
        ),
    ]
    conv = build_conv(
        id="snap-html-tool",
        provider="claude-ai",
        title="Tool Use Roundtrip",
        messages=msgs,
    )
    html = render_session_html(conv)
    assert html == snapshot


def test_thinking_block_html_snapshot(snapshot: object) -> None:
    """Thinking block renders as a collapsible section in HTML."""
    msgs = [
        build_msg(
            id="m1",
            role="assistant",
            text="",
            content_blocks=[
                {
                    "type": BlockType.THINKING.value,
                    "text": "Let me think step by step about this.",
                },
                {
                    "type": BlockType.TEXT.value,
                    "text": "Final answer.",
                },
            ],
        )
    ]
    conv = build_conv(
        id="snap-html-thinking",
        provider="claude-ai",
        title="Thinking Block",
        messages=msgs,
    )
    html = render_session_html(conv)
    assert html == snapshot


def test_code_fence_with_backticks_html_snapshot(snapshot: object) -> None:
    """Code blocks render via Pygments; embedded backticks survive escaping."""
    msgs = [
        build_msg(
            id="m1",
            role="assistant",
            text="",
            content_blocks=[
                {
                    "type": BlockType.CODE.value,
                    "language": "python",
                    "text": "# example\nprint('inline `backticks` survive')\n",
                }
            ],
        )
    ]
    conv = build_conv(
        id="snap-html-code",
        provider="chatgpt",
        title="Code With Backticks",
        messages=msgs,
    )
    html = render_session_html(conv)
    assert html == snapshot


def test_media_blocks_render_in_session_html() -> None:
    """Structured media blocks should survive the HTML boundary."""
    msgs = [
        build_msg(
            id="m1",
            role="assistant",
            text="This fallback text should not be rendered",
            content_blocks=[
                {
                    "type": BlockType.DOCUMENT.value,
                    "name": "Spec",
                    "url": "https://example.com/spec.pdf",
                    "mime_type": "application/pdf",
                }
            ],
        )
    ]
    conv = _make_conv(msgs, title="Media Session")
    html = render_session_html(conv)
    assert "media-block" in html
    assert 'data-type="document"' in html
    assert 'href="https://example.com/spec.pdf"' in html
    assert "Spec" in html
    assert "This fallback text should not be rendered" not in html
