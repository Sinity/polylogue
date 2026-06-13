"""Regression tests for HTML message shaping with structured-only content."""

from __future__ import annotations

from polylogue.rendering.renderers.html import render_session_html
from tests.infra.builders import make_conv, make_msg


def test_render_session_html_keeps_messages_with_structured_blocks_only() -> None:
    """Structured blocks should keep a message renderable even when text is empty."""
    session = make_conv(
        messages=[
            make_msg(
                id="m1",
                role="assistant",
                text="",
                blocks=[
                    {
                        "type": "code",
                        "text": "print('hello from blocks')",
                        "language": "python",
                    }
                ],
            )
        ]
    )

    html = render_session_html(session)

    assert "hello from blocks" in html
    assert "code-block" in html
