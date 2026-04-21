"""Regression tests for HTML message shaping with structured-only content."""

from __future__ import annotations

from polylogue.rendering.renderers.html import render_conversation_html
from tests.infra.builders import make_conv, make_msg


def test_render_conversation_html_keeps_messages_with_structured_blocks_only() -> None:
    """Structured blocks should keep a message renderable even when text is empty."""
    conversation = make_conv(
        messages=[
            make_msg(
                id="m1",
                role="assistant",
                text="",
                content_blocks=[
                    {
                        "type": "code",
                        "text": "print('hello from blocks')",
                        "language": "python",
                    }
                ],
            )
        ]
    )

    html = render_conversation_html(conversation)

    assert "hello from blocks" in html
    assert "code-block" in html
