"""Highlighting and rich-text rendering helpers for HTML output."""

from __future__ import annotations

import re
from functools import lru_cache
from html import escape

from markdown_it import MarkdownIt
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.util import ClassNotFound

HTML_RENDER_CACHE_MAX_ENTRIES = 8192
HTML_RENDER_CACHE_TEXT_MAX_CHARS = 4096
HTML_RENDER_MARKDOWN_MAX_CHARS = 200_000
_URL_RE = re.compile(r"(https?://|www\.)", re.IGNORECASE)
_MARKDOWN_BLOCK_RE = re.compile(r"(^|\n)(?: {0,3}(?:[-+*] |\d+[.)] |>)| {4}|\t)")
_PLAIN_TEXT_MARKDOWN_CHARS = frozenset("`*_#[]|~")


class PygmentsHighlighter:
    """Code highlighter using Pygments."""

    _lexer_cache: dict[str, object] = {}

    def __init__(self, style: str = "monokai") -> None:
        self.formatter = HtmlFormatter(
            style=style,
            cssclass="highlight",
            linenos=False,
            wrapcode=True,
        )

    def get_css(self) -> str:
        return str(self.formatter.get_style_defs(".highlight"))

    def highlight_code(self, code: str, language: str | None = None) -> str:
        try:
            if language:
                lexer = PygmentsHighlighter._lexer_cache.get(language)
                if lexer is None:
                    lexer = get_lexer_by_name(language, stripall=True)
                    PygmentsHighlighter._lexer_cache[language] = lexer
            elif len(code) < 50:
                lexer = get_lexer_by_name("text", stripall=True)
            else:
                lexer = guess_lexer(code)
        except ClassNotFound:
            return self._plain_code_block(code)

        return str(highlight(code, lexer, self.formatter))

    @staticmethod
    def _plain_code_block(code: str) -> str:
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f'<pre class="highlight"><code>{escaped}</code></pre>'


class HTMLMessageRenderer:
    """Markdown-to-HTML renderer used inside the HTML conversation renderer."""

    def __init__(self, highlighter: PygmentsHighlighter | None = None) -> None:
        self.highlighter = highlighter or PygmentsHighlighter()
        self.md = MarkdownIt("commonmark", {"html": False, "linkify": True})
        self.md.enable("table")
        self._render_cached = lru_cache(maxsize=HTML_RENDER_CACHE_MAX_ENTRIES)(self._render_uncached)

    def render(self, text: str) -> str:
        if not text:
            return ""
        if len(text) > HTML_RENDER_MARKDOWN_MAX_CHARS:
            return self._render_large_text_fast(text)
        if len(text) > HTML_RENDER_CACHE_TEXT_MAX_CHARS:
            return self._render_uncached(text)
        return self._render_cached(text)

    def _render_uncached(self, text: str) -> str:
        if self._can_render_plain_text_fast(text):
            return self._render_plain_text_fast(text)
        return self._enhance_code_blocks(self.md.render(text))

    @staticmethod
    def _can_render_plain_text_fast(text: str) -> bool:
        return bool(text) and (
            "  \n" not in text
            and "<" not in text
            and not _URL_RE.search(text)
            and not _MARKDOWN_BLOCK_RE.search(text)
            and not any(ch in text for ch in _PLAIN_TEXT_MARKDOWN_CHARS)
        )

    @staticmethod
    def _escape_like_markdown_it(text: str) -> str:
        return escape(text, quote=False).replace('"', "&quot;")

    @classmethod
    def _render_plain_text_fast(cls, text: str) -> str:
        parts = [paragraph.strip() for paragraph in re.split(r"\n\s*\n+", text.strip()) if paragraph.strip()]
        return "".join(f"<p>{cls._escape_like_markdown_it(paragraph)}</p>\n" for paragraph in parts)

    @classmethod
    def _render_large_text_fast(cls, text: str) -> str:
        return f'<pre class="plain-text-block">{cls._escape_like_markdown_it(text)}</pre>'

    def _enhance_code_blocks(self, html: str) -> str:
        pattern = r'<pre><code class="language-(\w+)">(.*?)</code></pre>'

        def replace_code(match: re.Match[str]) -> str:
            language = match.group(1)
            code = match.group(2)
            code = code.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            return self.highlighter.highlight_code(code, language)

        html = re.sub(pattern, replace_code, html, flags=re.DOTALL)

        pattern_plain = r"<pre><code>([^<]*)</code></pre>"

        def replace_plain_code(match: re.Match[str]) -> str:
            code = match.group(1)
            code = code.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            return self.highlighter.highlight_code(code, None)

        return re.sub(pattern_plain, replace_plain_code, html, flags=re.DOTALL)


__all__ = [
    "HTMLMessageRenderer",
    "HTML_RENDER_CACHE_MAX_ENTRIES",
    "HTML_RENDER_CACHE_TEXT_MAX_CHARS",
    "PygmentsHighlighter",
]
