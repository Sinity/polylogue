"""Contracts for the conversation HTML sanitizer (defense-in-depth filter)."""

from __future__ import annotations

import pytest

from polylogue.rendering.renderers.html_sanitizer import sanitize_html


@pytest.mark.parametrize(
    "html",
    [
        # Plain script tag.
        "<p>before</p><script>alert(1)</script><p>after</p>",
        # Closing tag with trailing whitespace before > (CodeQL-flagged bypass).
        "<p>before</p><script>alert(1)</script ><p>after</p>",
        "<p>before</p><script>alert(1)</script\n><p>after</p>",
        "<p>before</p><script>alert(1)</script\t>",
        # Mixed case.
        "<p>before</p><SCRIPT>alert(1)</SCRIPT >",
        # Attribute on opening tag.
        "<script type='text/javascript'>alert(1)</script>",
        # Multiline body.
        "<script>\nalert(1);\nfoo();\n</script  >",
    ],
)
def test_sanitize_strips_script_tags(html: str) -> None:
    sanitized = str(sanitize_html(html))
    assert "alert(1)" not in sanitized
    assert "<script" not in sanitized.lower()
    assert "</script" not in sanitized.lower()


def test_sanitize_strips_event_handlers() -> None:
    sanitized = str(sanitize_html('<a href="x" onclick="alert(1)">x</a>'))
    assert "onclick" not in sanitized
    assert "alert(1)" not in sanitized


def test_sanitize_strips_javascript_urls() -> None:
    sanitized = str(sanitize_html('<a href="javascript:alert(1)">x</a>'))
    assert "javascript:" not in sanitized


def test_sanitize_preserves_safe_markup() -> None:
    safe = "<p>hello <strong>world</strong></p>"
    assert str(sanitize_html(safe)) == safe


def test_sanitize_returns_markup() -> None:
    from markupsafe import Markup

    assert isinstance(sanitize_html("<p>x</p>"), Markup)


def test_sanitize_empty_input() -> None:
    assert str(sanitize_html("")) == ""
