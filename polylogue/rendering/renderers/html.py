"""Enhanced HTML renderer with Pygments syntax highlighting and modern styling."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import DictLoader, Environment, FileSystemLoader, select_autoescape
from markdown_it import MarkdownIt
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.util import ClassNotFound

from polylogue.assets import asset_path
from polylogue.render_paths import render_root

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation
    from polylogue.storage.backends.async_sqlite import SQLiteBackend


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
        self._backend = backend

        # Initialize Pygments highlighter
        style = "monokai" if theme == "dark" else "default"
        self.highlighter = PygmentsHighlighter(style=style)
        self.md_renderer = MarkdownRenderer(self.highlighter)

        # Pre-compile Jinja2 template once (avoid per-render overhead)
        loader: FileSystemLoader | DictLoader
        if self.template_path and self.template_path.exists():
            loader = FileSystemLoader(self.template_path.parent)
            template_name = self.template_path.name
        else:
            loader = DictLoader({"conversation.html": DEFAULT_HTML_TEMPLATE})
            template_name = "conversation.html"
        self._jinja_env = Environment(
            loader=loader, autoescape=select_autoescape(["html", "xml"])
        )
        self._template = self._jinja_env.get_template(template_name)
        self._highlight_css = self.highlighter.get_css()

    def supports_format(self) -> str:
        """Return the output format this renderer supports.

        Returns:
            'html'
        """
        return "html"

    def _get_backend(self) -> SQLiteBackend:
        """Get or create the backend instance."""
        if self._backend is not None:
            return self._backend
        from polylogue.storage.backends.async_sqlite import SQLiteBackend as _SQLiteBackend
        return _SQLiteBackend()

    async def _fetch_conversation_data(
        self, conversation_id: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        """Fetch all data for a conversation in a single DB connection.

        Returns (metadata, messages, attachments) — plain dicts extracted
        from sqlite3.Row objects so they're safely usable across threads.
        """
        backend = self._get_backend()

        async with backend._get_connection() as conn:
            # Conversation metadata
            cursor = await conn.execute(
                "SELECT conversation_id, title, provider_name, created_at, updated_at "
                "FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
            convo = await cursor.fetchone()
            if not convo:
                raise ValueError(f"Conversation not found: {conversation_id}")

            metadata = {
                "conversation_id": convo["conversation_id"],
                "title": convo["title"] or conversation_id,
                "provider": convo["provider_name"],
                "created_at": convo["created_at"],
                "updated_at": convo["updated_at"],
            }

            # Messages (all columns needed for both markdown and HTML)
            cursor = await conn.execute(
                """
                SELECT message_id, role, text, timestamp,
                       parent_message_id, branch_index
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
            )
            messages: list[dict[str, Any]] = []
            while True:
                rows = await cursor.fetchmany(200)
                if not rows:
                    break
                for row in rows:
                    messages.append({
                        "message_id": row["message_id"],
                        "role": row["role"],
                        "text": row["text"],
                        "timestamp": row["timestamp"],
                        "parent_message_id": row["parent_message_id"],
                        "branch_index": row["branch_index"] or 0,
                    })

            # Attachments
            cursor = await conn.execute(
                """
                SELECT
                    attachment_refs.message_id,
                    attachments.attachment_id,
                    attachments.mime_type,
                    attachments.size_bytes,
                    attachments.path,
                    attachments.provider_meta
                FROM attachment_refs
                JOIN attachments ON attachments.attachment_id = attachment_refs.attachment_id
                WHERE attachment_refs.conversation_id = ?
                """,
                (conversation_id,),
            )
            attachments: list[dict[str, Any]] = []
            att_rows = await cursor.fetchall()
            for row in att_rows:
                attachments.append({
                    "message_id": row["message_id"],
                    "attachment_id": row["attachment_id"],
                    "mime_type": row["mime_type"],
                    "size_bytes": row["size_bytes"],
                    "path": row["path"],
                    "provider_meta": row["provider_meta"],
                })

        return metadata, messages, attachments

    def _render_content_sync(
        self,
        metadata: dict[str, Any],
        messages: list[dict[str, Any]],
        attachments: list[dict[str, Any]],
    ) -> tuple[str, str]:
        """All CPU-bound rendering work: markdown + Pygments + Jinja2.

        Designed to run in a thread pool via asyncio.to_thread() so the
        event loop stays responsive for DB queries and other workers.

        Returns:
            (markdown_text, html_content) tuple
        """
        title = metadata["title"]
        provider = metadata["provider"]
        conversation_id = metadata["conversation_id"]
        created_at = metadata["created_at"]

        # Build attachments mapping for markdown
        attachments_by_message: dict[str | None, list[dict[str, Any]]] = {}
        for att in attachments:
            attachments_by_message.setdefault(att["message_id"], []).append(att)

        # --- Generate markdown text ---
        md_text = self._format_markdown(
            title, provider, conversation_id, messages, attachments_by_message,
        )

        # --- Generate HTML messages (Pygments + MarkdownIt) ---
        raw_html_messages: list[dict[str, object]] = []
        for msg in messages:
            text = msg["text"] or ""
            if not text:
                continue
            role = msg["role"] or "message"
            html_content = self.md_renderer.render(text)
            raw_html_messages.append({
                "id": msg["message_id"],
                "role": role,
                "role_class": _role_css_class(role),
                "text": text[:120],
                "html_content": html_content,
                "timestamp": msg["timestamp"],
                "parent_message_id": msg["parent_message_id"],
                "branch_index": msg["branch_index"],
            })

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

    def _format_markdown(
        self,
        title: str,
        provider: str,
        conversation_id: str,
        messages: list[dict[str, Any]],
        attachments_by_message: dict[str | None, list[dict[str, Any]]],
    ) -> str:
        """Format conversation data to markdown text.

        Inline implementation avoids depending on ConversationFormatter's
        DB-coupled format() method, enabling fully-offline rendering in
        thread pool workers.
        """
        def _format_text(text: str) -> str:
            if not text:
                return ""
            stripped = text.strip()
            if (stripped.startswith("{") and stripped.endswith("}")) or (
                stripped.startswith("[") and stripped.endswith("]")
            ):
                try:
                    parsed = json.loads(stripped)
                    return f"```json\n{json.dumps(parsed, indent=2)}\n```"
                except json.JSONDecodeError:
                    pass
            return text

        def _append_attachment(att: dict[str, Any], lines: list[str]) -> None:
            name = None
            meta = att.get("provider_meta")
            if meta:
                try:
                    meta_dict = json.loads(meta)
                    name = meta_dict.get("name") or meta_dict.get("provider_id") or meta_dict.get("drive_id")
                except (json.JSONDecodeError, TypeError):
                    name = None
            label = name or att["attachment_id"]
            path_value = att.get("path") or str(asset_path(self.archive_root, att["attachment_id"]))
            lines.append(f"- Attachment: {label} ({path_value})")

        lines = [f"# {title}", "", f"Provider: {provider}", f"Conversation ID: {conversation_id}", ""]
        message_ids = set()

        for msg in messages:
            message_ids.add(msg["message_id"])
            role = msg["role"] or "message"
            text = msg["text"] or ""
            timestamp = msg["timestamp"]
            msg_atts = attachments_by_message.get(msg["message_id"], [])

            if not text.strip() and not msg_atts:
                continue

            lines.append(f"## {role}")
            if timestamp:
                lines.append(f"_Timestamp: {timestamp}_")
            lines.append("")

            formatted_text = _format_text(text)
            if formatted_text:
                lines.append(formatted_text)
                lines.append("")

            for att in msg_atts:
                _append_attachment(att, lines)
            lines.append("")

        # Orphaned attachments
        orphan_keys = [key for key in attachments_by_message if key not in message_ids]
        if orphan_keys:
            lines.append("## attachments")
            lines.append("")
            for key in sorted(orphan_keys, key=lambda item: (item is None, str(item) if item else "")):
                for att in attachments_by_message.get(key, []):
                    _append_attachment(att, lines)
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    async def render(self, conversation_id: str, output_path: Path) -> Path:
        """Render a conversation to enhanced HTML format.

        Pipeline: DB query (async) → CPU render (thread) → file write (thread).
        The DB connection is released before CPU work begins, keeping pool
        connections available for other workers.
        """
        # Phase 1: Single DB query — borrows and returns connection quickly
        metadata, messages, attachments = await self._fetch_conversation_data(conversation_id)

        # Phase 2: CPU work in thread pool (Pygments, MarkdownIt, Jinja2)
        md_text, html_content = await asyncio.to_thread(
            self._render_content_sync, metadata, messages, attachments,
        )

        # Phase 3: File writes in thread pool
        render_root_path = render_root(output_path, metadata["provider"], conversation_id)

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
        role = msg.role or "message"
        html_content = md_renderer.render(msg.text)
        raw_messages.append({
            "id": msg.id,
            "role": role,
            "role_class": _role_css_class(role),
            "text": msg.text[:120],
            "html_content": html_content,
            "timestamp": str(msg.timestamp) if msg.timestamp else None,
            "parent_message_id": msg.parent_id,
            "branch_index": msg.branch_index,
        })

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
