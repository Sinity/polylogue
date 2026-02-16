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
    from polylogue.lib.models import Conversation


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

_ROLE_CLASS_RE = re.compile(r"[^a-z0-9-]")


def _role_css_class(role: str) -> str:
    """Convert a role name to a CSS-safe class like ``message-tool-use``."""
    return "message-" + _ROLE_CLASS_RE.sub("-", role.lower())


def _attach_branches(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    """Group branch messages under their mainline siblings.

    Mainline messages (``branch_index == 0``) are returned as the top-level
    list.  Messages with ``branch_index > 0`` are attached as a ``branches``
    list on the mainline message that shares the same ``parent_message_id``.

    If no branching is present, the input is returned unchanged (all
    messages have ``branch_index == 0`` and no ``branches`` key).
    """
    # Fast path: if no message has a non-zero branch_index, skip grouping
    if not any(m.get("branch_index", 0) for m in messages):
        return messages

    # Index mainline messages by their id for parent lookups
    mainline: list[dict[str, object]] = []
    mainline_by_id: dict[str, dict[str, object]] = {}

    for msg in messages:
        if not msg.get("branch_index"):
            mainline.append(msg)
            mainline_by_id[str(msg["id"])] = msg

    # Attach branch messages to the mainline sibling that shares the same parent
    for msg in messages:
        branch_idx = msg.get("branch_index", 0)
        parent_id = msg.get("parent_message_id")
        if not branch_idx or not parent_id:
            continue

        # Find the mainline sibling: the mainline message with the same parent_id
        # This is the branch_index=0 message that shares the same parent
        sibling = None
        for m in mainline:
            if m.get("parent_message_id") == parent_id and not m.get("branch_index"):
                sibling = m
                break

        if sibling is not None:
            branches = sibling.setdefault("branches", [])
            assert isinstance(branches, list)
            branches.append(msg)
        else:
            # No mainline sibling found — include as standalone
            mainline.append(msg)

    return mainline


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

    async def _prepare_messages(self, conversation_id: str) -> list[dict[str, object]]:
        """Prepare messages for template rendering with HTML content.

        Includes branch metadata so the template can render branch
        points with collapsible ``<details>`` sections.

        Args:
            conversation_id: ID of the conversation

        Returns:
            List of mainline message dicts.  Each dict may contain a
            ``branches`` key holding a list of alternative branch dicts.
        """
        from polylogue.storage.backends.async_sqlite import SQLiteBackend

        backend = self.formatter.backend or SQLiteBackend(db_path=self.formatter.db_path)
        raw_messages: list[dict[str, object]] = []

        async with backend._get_connection() as conn:
            cursor = await conn.execute(
                """
                SELECT message_id, role, text, timestamp,
                       parent_message_id, branch_index
                FROM messages
                WHERE conversation_id = ?
                ORDER BY
                    branch_index,
                    (timestamp IS NULL),
                    CASE
                        WHEN timestamp IS NULL THEN NULL
                        WHEN timestamp GLOB '*[^0-9.]*' THEN CAST(strftime('%s', timestamp) AS INTEGER)
                        ELSE CAST(timestamp AS REAL)
                    END,
                    message_id
                """,
                (conversation_id,),
            )
            rows = await cursor.fetchall()

            for row in rows:
                text = row["text"] or ""
                if not text:
                    continue
                role = row["role"] or "message"
                html_content = self.md_renderer.render(text)
                raw_messages.append({
                    "id": row["message_id"],
                    "role": role,
                    "role_class": _role_css_class(role),
                    "text": text,
                    "html_content": html_content,
                    "timestamp": row["timestamp"],
                    "parent_message_id": row["parent_message_id"],
                    "branch_index": row["branch_index"] or 0,
                })

        return _attach_branches(raw_messages)

    async def render(self, conversation_id: str, output_path: Path) -> Path:
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
        formatted = await self.formatter.format(conversation_id)

        # Determine output path
        render_root_path = render_root(output_path, formatted.provider, conversation_id)
        render_root_path.mkdir(parents=True, exist_ok=True)

        # Save markdown file (used by CLI search and query)
        md_path = render_root_path / "conversation.md"
        md_path.write_text(formatted.markdown_text, encoding="utf-8")

        # Prepare messages with HTML content
        messages = await self._prepare_messages(conversation_id)

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


def render_conversation_html(conv: Conversation, theme: str = "dark") -> str:
    """Render an in-memory Conversation to a standalone HTML string.

    Uses the same Jinja2 template and Pygments highlighting as
    :class:`HTMLRenderer`, but works directly from a ``Conversation``
    object rather than querying the database.

    Args:
        conv: Conversation with messages to render.
        theme: ``"dark"`` or ``"light"``.

    Returns:
        Complete HTML document as a string.
    """
    style = "monokai" if theme == "dark" else "default"
    highlighter = PygmentsHighlighter(style=style)
    md_renderer = MarkdownRenderer(highlighter)

    raw_messages: list[dict[str, object]] = []
    for msg in conv.messages:
        if not msg.text:
            continue
        role = msg.role or "message"
        html_content = md_renderer.render(msg.text)
        raw_messages.append({
            "id": msg.id,
            "role": role,
            "role_class": _role_css_class(role),
            "text": msg.text,
            "html_content": html_content,
            "timestamp": str(msg.timestamp) if msg.timestamp else None,
            "parent_message_id": msg.parent_id,
            "branch_index": msg.branch_index,
        })

    messages = _attach_branches(raw_messages)
    title = conv.display_title or str(conv.id)

    env = Environment(
        loader=DictLoader({"conversation.html": DEFAULT_HTML_TEMPLATE}),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("conversation.html")

    return template.render(
        title=title,
        provider=conv.provider,
        conversation_id=str(conv.id),
        messages=messages,
        message_count=len(messages),
        created_at=str(conv.created_at) if conv.created_at else None,
        highlight_css=highlighter.get_css(),
        theme=theme,
    )


__all__ = ["HTMLRenderer", "PygmentsHighlighter", "MarkdownRenderer", "render_conversation_html"]
