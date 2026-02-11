"""Enhanced HTML renderer with Pygments syntax highlighting and modern styling."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import DictLoader, Environment, FileSystemLoader, select_autoescape
from markdown_it import MarkdownIt
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.util import ClassNotFound

from polylogue.render_paths import render_root
from polylogue.rendering.core import ConversationFormatter

if TYPE_CHECKING:
    pass


class PygmentsHighlighter:
    """Code highlighter using Pygments."""

    def __init__(self, style: str = "monokai") -> None:
        """Initialize highlighter with a Pygments style.

        Args:
            style: Pygments style name (default: monokai for dark theme)
        """
        self.formatter = HtmlFormatter(
            style=style,
            cssclass="highlight",
            linenos=False,
            wrapcode=True,
        )

    def get_css(self) -> str:
        """Get CSS for syntax highlighting.

        Returns:
            CSS string for Pygments highlighting
        """
        return self.formatter.get_style_defs(".highlight")

    def highlight_code(self, code: str, language: str | None = None) -> str:
        """Highlight code block.

        Args:
            code: Source code to highlight
            language: Language hint (optional)

        Returns:
            HTML with syntax highlighting
        """
        try:
            lexer = get_lexer_by_name(language, stripall=True) if language else guess_lexer(code)
        except ClassNotFound:
            # Fall back to plain text
            escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            return f'<pre class="highlight"><code>{escaped}</code></pre>'

        return highlight(code, lexer, self.formatter)


class MarkdownRenderer:
    """Enhanced markdown renderer with code highlighting."""

    def __init__(self, highlighter: PygmentsHighlighter | None = None) -> None:
        """Initialize markdown renderer.

        Args:
            highlighter: Optional Pygments highlighter instance
        """
        self.highlighter = highlighter or PygmentsHighlighter()
        self.md = MarkdownIt("commonmark", {"html": False, "linkify": True})
        self.md.enable("table")

    def render(self, text: str) -> str:
        """Render markdown to HTML with syntax highlighting.

        Args:
            text: Markdown text to render

        Returns:
            HTML string with syntax-highlighted code blocks
        """
        if not text:
            return ""

        # First pass: standard markdown rendering
        html = self.md.render(text)

        # Second pass: enhance code blocks with Pygments
        html = self._enhance_code_blocks(html)

        return html

    def _enhance_code_blocks(self, html: str) -> str:
        """Replace code blocks with Pygments-highlighted versions.

        Args:
            html: HTML string with code blocks

        Returns:
            HTML with enhanced code blocks
        """
        # Match ```language\ncode\n``` patterns that became <pre><code class="language-xxx">
        pattern = r'<pre><code class="language-(\w+)">(.*?)</code></pre>'

        def replace_code(match: re.Match[str]) -> str:
            language = match.group(1)
            code = match.group(2)
            # Unescape HTML entities
            code = code.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            return self.highlighter.highlight_code(code, language)

        html = re.sub(pattern, replace_code, html, flags=re.DOTALL)

        # Also handle code blocks without language (generic <pre><code>)
        pattern_plain = r'<pre><code>([^<]*)</code></pre>'

        def replace_plain_code(match: re.Match[str]) -> str:
            code = match.group(1)
            code = code.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            # Try to guess the language
            return self.highlighter.highlight_code(code, None)

        html = re.sub(pattern_plain, replace_plain_code, html, flags=re.DOTALL)

        return html


DEFAULT_HTML_TEMPLATE = (Path(__file__).parent.parent / "templates" / "conversation.html").read_text()


class HTMLRenderer:
    """Enhanced HTML renderer with syntax highlighting and polished styling.

    Features:
    - Pygments syntax highlighting for code blocks
    - Dark/light theme support with prefers-color-scheme
    - Message bubbles with role-based styling
    - Table of contents sidebar with message anchors
    - Navigation breadcrumbs
    """

    def __init__(
        self,
        archive_root: Path,
        template_path: Path | None = None,
        theme: str = "dark",
    ) -> None:
        """Initialize the HTML renderer.

        Args:
            archive_root: Root directory for archived conversations
            template_path: Optional path to custom Jinja2 HTML template
            theme: Theme name ("dark" or "light"), default "dark"
        """
        self.archive_root = archive_root
        self.template_path = template_path
        self.theme = theme
        self.formatter = ConversationFormatter(archive_root)

        # Initialize Pygments highlighter
        style = "monokai" if theme == "dark" else "default"
        self.highlighter = PygmentsHighlighter(style=style)
        self.md_renderer = MarkdownRenderer(self.highlighter)

    def supports_format(self) -> str:
        """Return the output format this renderer supports.

        Returns:
            'html'
        """
        return "html"

    def _prepare_messages(self, conversation_id: str) -> list[dict[str, str | None]]:
        """Prepare messages for template rendering with HTML content.

        Args:
            conversation_id: ID of the conversation

        Returns:
            List of message dicts with html_content field
        """
        from polylogue.storage.backends.sqlite import open_connection

        messages: list[dict[str, str | None]] = []
        with open_connection(None) as conn:
            rows = conn.execute(
                """
                SELECT message_id, role, text, timestamp
                FROM messages
                WHERE conversation_id = ?
                ORDER BY
                    (timestamp IS NULL),
                    CASE
                        WHEN timestamp IS NULL THEN NULL
                        WHEN timestamp GLOB '*[^0-9.]*' THEN CAST(strftime('%s', timestamp) AS INTEGER)
                        ELSE CAST(timestamp AS REAL)
                    END,
                    message_id
                """,
                (conversation_id,),
            ).fetchall()

            for row in rows:
                text = row["text"] or ""
                if not text:
                    continue
                html_content = self.md_renderer.render(text)
                messages.append({
                    "id": row["message_id"],
                    "role": row["role"] or "message",
                    "text": text,
                    "html_content": html_content,
                    "timestamp": row["timestamp"],
                })

        return messages

    def render(self, conversation_id: str, output_path: Path) -> Path:
        """Render a conversation to enhanced HTML format.

        Args:
            conversation_id: ID of the conversation to render
            output_path: Directory where the HTML file should be written

        Returns:
            Path to the generated HTML file

        Raises:
            ValueError: If conversation not found
            IOError: If output path is invalid or write fails
        """
        # Use shared formatter to get metadata
        formatted = self.formatter.format(conversation_id)

        # Determine output path
        render_root_path = render_root(output_path, formatted.provider, conversation_id)
        render_root_path.mkdir(parents=True, exist_ok=True)

        # Save markdown file (for backward compatibility)
        md_path = render_root_path / "conversation.md"
        md_path.write_text(formatted.markdown_text, encoding="utf-8")

        # Prepare messages with HTML content
        messages = self._prepare_messages(conversation_id)

        # Set up template
        loader: FileSystemLoader | DictLoader
        if self.template_path and self.template_path.exists():
            loader = FileSystemLoader(self.template_path.parent)
            template_name = self.template_path.name
        else:
            loader = DictLoader({"conversation.html": DEFAULT_HTML_TEMPLATE})
            template_name = "conversation.html"

        env = Environment(loader=loader, autoescape=select_autoescape(["html", "xml"]))
        template = env.get_template(template_name)

        # Render HTML
        html_content = template.render(
            title=formatted.title,
            provider=formatted.provider,
            conversation_id=conversation_id,
            messages=messages,
            message_count=len(messages),
            created_at=formatted.metadata.get("created_at"),
            highlight_css=self.highlighter.get_css(),
            theme=self.theme,
        )

        html_path = render_root_path / "conversation.html"
        html_path.write_text(html_content, encoding="utf-8")

        return html_path


__all__ = ["HTMLRenderer", "PygmentsHighlighter", "MarkdownRenderer"]
