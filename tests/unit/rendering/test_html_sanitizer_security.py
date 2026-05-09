"""Security regression tests for the HTML sanitizer (issue #813).

The previous regex-based implementation had documented bypass classes:

* unquoted event-handler attributes (``<a onclick=alert(1)>``) were not
  matched by ``_EVENT_ATTR_RE`` which required quoted values
* ``javascript:`` URI guard only applied to ``href``, not ``src``
* ``data:`` URIs were never stripped
* ``<iframe>``/``<object>``/``<embed>`` were left intact
* ``<svg>``/``<math>`` could carry event handlers

Each test below pins one of those bypass classes against the current
``nh3``-backed sanitizer.
"""

from __future__ import annotations

import pytest

from polylogue.rendering.renderers.html_sanitizer import sanitize_html


@pytest.mark.parametrize(
    "html",
    [
        "<a onclick=alert(1)>x</a>",
        "<a ONCLICK=alert(1)>x</a>",
        "<img onerror=alert(1) src=x>",
        "<button onfocus=alert(1) autofocus>x</button>",
        "<div onmouseover=alert(1)>x</div>",
    ],
)
def test_unquoted_event_handlers_stripped(html: str) -> None:
    sanitized = str(sanitize_html(html))
    lowered = sanitized.lower()
    assert "alert(1)" not in sanitized
    assert "onclick" not in lowered
    assert "onerror" not in lowered
    assert "onfocus" not in lowered
    assert "onmouseover" not in lowered


@pytest.mark.parametrize(
    "html",
    [
        '<a href="data:text/html,<script>alert(1)</script>">x</a>',
        "<a href=data:text/html,xss>x</a>",
        '<img src="data:text/html,<script>alert(1)</script>">',
        '<iframe src="data:text/html,<script>alert(1)</script>"></iframe>',
    ],
)
def test_data_uris_stripped(html: str) -> None:
    sanitized = str(sanitize_html(html))
    assert "data:" not in sanitized.lower()
    assert "alert(1)" not in sanitized


@pytest.mark.parametrize(
    "html",
    [
        '<img src="javascript:alert(1)">',
        "<img src=javascript:alert(1)>",
        '<a href="javascript:alert(1)">x</a>',
        "<a href=javascript:alert(1)>x</a>",
    ],
)
def test_javascript_urls_stripped_on_any_attribute(html: str) -> None:
    sanitized = str(sanitize_html(html))
    assert "javascript:" not in sanitized.lower()
    assert "alert(1)" not in sanitized


@pytest.mark.parametrize(
    "html",
    [
        '<iframe src="https://evil.example/xss"></iframe>',
        '<object data="https://evil.example/x.swf"></object>',
        '<embed src="https://evil.example/x.swf">',
        '<frame src="https://evil.example">',
        '<frameset><frame src="x"></frameset>',
    ],
)
def test_frame_and_object_elements_stripped(html: str) -> None:
    sanitized = str(sanitize_html(html))
    lowered = sanitized.lower()
    assert "<iframe" not in lowered
    assert "<object" not in lowered
    assert "<embed" not in lowered
    assert "<frame" not in lowered
    assert "<frameset" not in lowered
    assert "evil.example" not in sanitized


@pytest.mark.parametrize(
    "html",
    [
        "<svg onload=alert(1)></svg>",
        "<svg><script>alert(1)</script></svg>",
        "<math><mtext><script>alert(1)</script></mtext></math>",
        "<svg><a href=javascript:alert(1)>x</a></svg>",
    ],
)
def test_svg_and_math_neutralized(html: str) -> None:
    sanitized = str(sanitize_html(html))
    lowered = sanitized.lower()
    assert "alert(1)" not in sanitized
    assert "onload" not in lowered
    assert "<script" not in lowered
    assert "javascript:" not in lowered
    # SVG/MathML are not on the allowlist; their tags should be dropped.
    assert "<svg" not in lowered
    assert "<math" not in lowered


def test_safe_links_preserved() -> None:
    sanitized = str(sanitize_html('<a href="https://example.com">x</a>'))
    assert 'href="https://example.com"' in sanitized
    assert ">x</a>" in sanitized


def test_mailto_links_preserved() -> None:
    sanitized = str(sanitize_html('<a href="mailto:a@b.com">m</a>'))
    assert "mailto:a@b.com" in sanitized
