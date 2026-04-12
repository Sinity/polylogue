from __future__ import annotations

from polylogue.rendering.renderers.html_highlighting import (
    HTML_RENDER_CACHE_TEXT_MAX_CHARS,
    HTML_RENDER_MARKDOWN_MAX_CHARS,
    HTMLMessageRenderer,
)


def test_html_message_renderer_caches_bounded_repeated_messages() -> None:
    renderer = HTMLMessageRenderer()

    first = renderer.render("Hello **world**")
    second = renderer.render("Hello **world**")

    assert first == second
    cache_info = renderer._render_cached.cache_info()
    assert cache_info.hits == 1
    assert cache_info.currsize == 1


def test_html_message_renderer_bypasses_cache_for_large_messages() -> None:
    renderer = HTMLMessageRenderer()
    oversized = "x" * (HTML_RENDER_CACHE_TEXT_MAX_CHARS + 1)

    renderer.render(oversized)
    renderer.render(oversized)

    cache_info = renderer._render_cached.cache_info()
    assert cache_info.hits == 0
    assert cache_info.misses == 0


def test_html_message_renderer_plain_text_fast_path_matches_markdown_it_output() -> None:
    renderer = HTMLMessageRenderer()
    text = 'Hello "quoted" world.\n\nSecond paragraph.'

    rendered = renderer.render(text)
    expected = renderer._enhance_code_blocks(renderer.md.render(text))

    assert rendered == expected


def test_html_message_renderer_preserves_blockquotes_via_markdown_it() -> None:
    renderer = HTMLMessageRenderer()

    rendered = renderer.render("> quoted line")

    assert "<blockquote>" in rendered


def test_html_message_renderer_preserves_hard_breaks_via_markdown_it() -> None:
    renderer = HTMLMessageRenderer()

    rendered = renderer.render("first line  \nsecond line")

    assert "<br />" in rendered


def test_html_message_renderer_oversized_markdown_like_text_bypasses_markdown_it(monkeypatch) -> None:
    renderer = HTMLMessageRenderer()
    oversized = "# heading\n" + ("x" * HTML_RENDER_MARKDOWN_MAX_CHARS)

    def _fail(_text: str) -> str:
        raise AssertionError("oversized text should not hit markdown-it")

    monkeypatch.setattr(renderer.md, "render", _fail)

    rendered = renderer.render(oversized)

    assert rendered.startswith('<pre class="plain-text-block">')
    assert "# heading" in rendered
