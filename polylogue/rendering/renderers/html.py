"""Enhanced HTML renderer with Pygments syntax highlighting and modern styling."""

from __future__ import annotations

import asyncio
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
from polylogue.rendering.core import ConversationFormatter, FormattedConversation

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.store import ConversationRenderProjection


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

    _lexer_cache: dict[str, object] = {}

    def highlight_code(self, code: str, language: str | None = None) -> str:
        """Highlight code block.

        Args:
            code: Source code to highlight
            language: Language hint (optional)

        Returns:
            HTML with syntax highlighting
        """
        try:
            if language:
                # Cache named lexers — they're stateless singletons
                lexer = PygmentsHighlighter._lexer_cache.get(language)
                if lexer is None:
                    lexer = get_lexer_by_name(language, stripall=True)
                    PygmentsHighlighter._lexer_cache[language] = lexer
            elif len(code) < 50:
                # Skip expensive guess_lexer() on tiny snippets — rarely accurate
                lexer = get_lexer_by_name("text", stripall=True)
            else:
                lexer = guess_lexer(code)
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

# Cached Jinja2 environment + compiled template — avoids re-parsing on every call
_CACHED_TEMPLATE_ENV: Environment | None = None


def _get_cached_template():
    """Return a module-level cached Jinja2 template for render_conversation_html()."""
    global _CACHED_TEMPLATE_ENV
    if _CACHED_TEMPLATE_ENV is None:
        _CACHED_TEMPLATE_ENV = Environment(
            loader=DictLoader({"conversation.html": DEFAULT_HTML_TEMPLATE}),
            autoescape=select_autoescape(["html", "xml"]),
        )
    return _CACHED_TEMPLATE_ENV.get_template("conversation.html")


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

    # Index mainline messages by parent_id for O(1) sibling lookup
    mainline_by_parent: dict[str, dict[str, object]] = {}
    for msg in mainline:
        pid = msg.get("parent_message_id")
        if pid:
            mainline_by_parent[str(pid)] = msg

    # Attach branch messages to the mainline sibling that shares the same parent
    for msg in messages:
        branch_idx = msg.get("branch_index", 0)
        parent_id = msg.get("parent_message_id")
        if not branch_idx or not parent_id:
            continue

        sibling = mainline_by_parent.get(str(parent_id))

        if sibling is not None:
            branches = sibling.setdefault("branches", [])
            assert isinstance(branches, list)
            branches.append(msg)
        else:
            # No mainline sibling found — include as standalone
            mainline.append(msg)

    return mainline


def _html_message_entry(
    *,
    message_id: object,
    role: object,
    text: str,
    timestamp: object,
    parent_message_id: object,
    branch_index: object,
    md_renderer: MarkdownRenderer,
) -> dict[str, object]:
    normalized_role = role or "message"
    if hasattr(normalized_role, "value"):
        normalized_role = normalized_role.value
    normalized_role = str(normalized_role)
    return {
        "id": message_id,
        "role": normalized_role,
        "role_class": _role_css_class(normalized_role),
        "text": text[:120],
        "html_content": md_renderer.render(text),
        "timestamp": timestamp,
        "parent_message_id": parent_message_id,
        "branch_index": branch_index,
    }


class HTMLRenderer:
    """Enhanced HTML renderer with syntax highlighting and polished styling.

    Performance architecture:
    - Single DB query per conversation (merged format + prepare)
    - CPU work (Pygments, MarkdownIt, Jinja2) offloaded to thread pool
    - Shared backend eliminates per-render schema checks / connection setup
    - File writes offloaded to thread pool
    """

    def __init__(
        self,
        archive_root: Path,
        template_path: Path | None = None,
        theme: str = "dark",
        backend: SQLiteBackend | None = None,
    ) -> None:
        """Initialize the HTML renderer.

        Args:
            archive_root: Root directory for archived conversations
            template_path: Optional path to custom Jinja2 HTML template
            theme: Theme name ("dark" or "light"), default "dark"
            backend: Shared async SQLite backend (avoids per-render connection overhead)
        """
        self.archive_root = archive_root
        self.template_path = template_path
        self.theme = theme
        self._formatter = ConversationFormatter(archive_root, backend=backend)

        # Initialize Pygments highlighter
        style = "monokai" if theme == "dark" else "default"
        self.highlighter = PygmentsHighlighter(style=style)
        self.md_renderer = MarkdownRenderer(self.highlighter)

        # Pre-compile Jinja2 template once (avoid per-render overhead)
        if self.template_path and self.template_path.exists():
            self._jinja_env = Environment(
                loader=FileSystemLoader(self.template_path.parent),
                autoescape=select_autoescape(["html", "xml"]),
            )
            self._template = self._jinja_env.get_template(self.template_path.name)
        else:
            self._jinja_env = None
            self._template = _get_cached_template()
        self._highlight_css = self.highlighter.get_css()

    def supports_format(self) -> str:
        """Return the output format this renderer supports.

        Returns:
            'html'
        """
        return "html"

    def _render_content_sync(
        self,
        formatted: FormattedConversation,
        projection: ConversationRenderProjection,
    ) -> tuple[str, str]:
        """All CPU-bound rendering work: markdown + Pygments + Jinja2.

        Designed to run in a thread pool via asyncio.to_thread() so the
        event loop stays responsive for DB queries and other workers.

        Returns:
            (markdown_text, html_content) tuple
        """
        title = formatted.title
        provider = formatted.provider
        conversation_id = formatted.conversation_id
        created_at = formatted.metadata["created_at"]
        md_text = formatted.markdown_text

        # --- Generate HTML messages (Pygments + MarkdownIt) ---
        raw_html_messages: list[dict[str, object]] = []
        for msg in projection.messages:
            text = msg.text or ""
            if not text:
                continue
            raw_html_messages.append(
                _html_message_entry(
                    message_id=msg.message_id,
                    role=msg.role,
                    text=text,
                    timestamp=msg.sort_key,
                    parent_message_id=msg.parent_message_id,
                    branch_index=msg.branch_index,
                    md_renderer=self.md_renderer,
                )
            )

        html_messages = _attach_branches(raw_html_messages)

        # --- Render Jinja2 template ---
        html_output = self._template.render(
            title=title,
            provider=provider,
            conversation_id=conversation_id,
            messages=html_messages,
            message_count=len(html_messages),
            created_at=created_at,
            highlight_css=self._highlight_css,
            theme=self.theme,
        )

        return md_text, html_output

    async def render(self, conversation_id: str, output_path: Path) -> Path:
        """Render a conversation to enhanced HTML format.

        Pipeline: repository projection (async) → CPU render (thread) → file write (thread).
        """
        projection = await self._formatter.load_projection(conversation_id)
        formatted = self._formatter.format_projection(projection)

        # Phase 2: CPU work in thread pool (Pygments, MarkdownIt, Jinja2)
        md_text, html_content = await asyncio.to_thread(
            self._render_content_sync, formatted, projection,
        )

        # Phase 3: File writes in thread pool
        render_root_path = render_root(output_path, formatted.provider, conversation_id)

        def _write_files() -> None:
            render_root_path.mkdir(parents=True, exist_ok=True)
            (render_root_path / "conversation.md").write_text(md_text, encoding="utf-8")
            (render_root_path / "conversation.html").write_text(html_content, encoding="utf-8")

        await asyncio.to_thread(_write_files)

        return render_root_path / "conversation.html"


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
        raw_messages.append(
            _html_message_entry(
                message_id=msg.id,
                role=msg.role,
                text=msg.text,
                timestamp=str(msg.timestamp) if msg.timestamp else None,
                parent_message_id=msg.parent_id,
                branch_index=msg.branch_index,
                md_renderer=md_renderer,
            )
        )

    messages = _attach_branches(raw_messages)
    title = conv.display_title or str(conv.id)

    template = _get_cached_template()

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
