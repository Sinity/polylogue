"""Snapshot tests for rendered conversation output.

These tests replace fragile string-searching assertions with syrupy snapshot
regression detection. The first run creates .ambr baseline files; subsequent
runs detect any accidental output changes.

Run ``pytest --snapshot-update`` to regenerate snapshots after intentional
renderer changes.
"""

from __future__ import annotations

import pytest

syrupy = pytest.importorskip("syrupy")

from polylogue.lib.models import Conversation, Message
from polylogue.rendering.renderers.html import render_conversation_html
from polylogue.types import ContentBlockType
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


def _make_conv(messages: list[Message], title: str = "Snapshot Test") -> Conversation:
    return build_conv(
        id="snap-conv-01",
        provider="chatgpt",
        title=title,
        messages=messages,
    )


# ---------------------------------------------------------------------------
# HTML renderer snapshots
# ---------------------------------------------------------------------------


def test_linear_conversation_html_snapshot(snapshot: object) -> None:
    """Linear 2-message conversation renders stably."""
    msgs = [_make_msg("m1", "user", "Hello there"), _make_msg("m2", "assistant", "Hi!")]
    conv = _make_conv(msgs, title="Linear Conversation")
    html = render_conversation_html(conv)
    assert html == snapshot


def test_branching_conversation_html_snapshot(snapshot: object) -> None:
    """Branching conversation with 1 alternative renders stably."""
    msgs = [
        _make_msg("m1", "user", "Question", parent_id=None, branch_index=0),
        _make_msg("m2", "assistant", "Answer 1", parent_id="m1", branch_index=0),
        _make_msg("m3", "assistant", "Answer 2 (edited)", parent_id="m1", branch_index=1),
    ]
    conv = _make_conv(msgs, title="Branching Conversation")
    html = render_conversation_html(conv)
    assert html == snapshot


def test_multi_branch_conversation_html_snapshot(snapshot: object) -> None:
    """Conversation with 2 alternatives renders stably."""
    msgs = [
        _make_msg("m1", "user", "Question"),
        _make_msg("m2", "assistant", "A1", parent_id="m1", branch_index=0),
        _make_msg("m3", "assistant", "A2", parent_id="m1", branch_index=1),
        _make_msg("m4", "assistant", "A3", parent_id="m1", branch_index=2),
    ]
    conv = _make_conv(msgs, title="Multi-Branch Conversation")
    html = render_conversation_html(conv)
    assert html == snapshot


def test_conversation_with_followup_html_snapshot(snapshot: object) -> None:
    """Branching conversation with post-branch follow-up renders stably."""
    msgs = [
        _make_msg("m1", "user", "Q"),
        _make_msg("m2", "assistant", "A1", parent_id="m1", branch_index=0),
        _make_msg("m3", "assistant", "A2 alt", parent_id="m1", branch_index=1),
        _make_msg("m4", "user", "Follow-up"),
    ]
    conv = _make_conv(msgs, title="Followup After Branch")
    html = render_conversation_html(conv)
    assert html == snapshot


def test_media_blocks_render_in_conversation_html() -> None:
    """Structured media blocks should survive the HTML boundary."""
    msgs = [
        build_msg(
            id="m1",
            role="assistant",
            text="This fallback text should not be rendered",
            content_blocks=[
                {
                    "type": ContentBlockType.DOCUMENT.value,
                    "name": "Spec",
                    "url": "https://example.com/spec.pdf",
                    "mime_type": "application/pdf",
                }
            ],
        )
    ]
    conv = _make_conv(msgs, title="Media Conversation")
    html = render_conversation_html(conv)
    assert "media-block" in html
    assert 'data-type="document"' in html
    assert 'href="https://example.com/spec.pdf"' in html
    assert "Spec" in html
    assert "This fallback text should not be rendered" not in html
