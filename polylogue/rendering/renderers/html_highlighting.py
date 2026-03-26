"""Highlighting and rich-text rendering helpers for HTML output."""

from __future__ import annotations

import re

from markdown_it import MarkdownIt
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.util import ClassNotFound


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
        return self.formatter.get_style_defs(".highlight")

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
            escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            return f'<pre class="highlight"><code>{escaped}</code></pre>'

        return highlight(code, lexer, self.formatter)


class HTMLMessageRenderer:
    """Markdown-to-HTML renderer used inside the HTML conversation renderer."""

    def __init__(self, highlighter: PygmentsHighlighter | None = None) -> None:
        self.highlighter = highlighter or PygmentsHighlighter()
        self.md = MarkdownIt("commonmark", {"html": False, "linkify": True})
        self.md.enable("table")

    def render(self, text: str) -> str:
        if not text:
            return ""
        return self._enhance_code_blocks(self.md.render(text))

    def _enhance_code_blocks(self, html: str) -> str:
        pattern = r'<pre><code class="language-(\w+)">(.*?)</code></pre>'

        def replace_code(match: re.Match[str]) -> str:
            language = match.group(1)
            code = match.group(2)
            code = code.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            return self.highlighter.highlight_code(code, language)

        html = re.sub(pattern, replace_code, html, flags=re.DOTALL)

        pattern_plain = r'<pre><code>([^<]*)</code></pre>'

        def replace_plain_code(match: re.Match[str]) -> str:
            code = match.group(1)
            code = code.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            return self.highlighter.highlight_code(code, None)

        return re.sub(pattern_plain, replace_plain_code, html, flags=re.DOTALL)


__all__ = ["HTMLMessageRenderer", "PygmentsHighlighter"]
