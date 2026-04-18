"""Enhanced HTML renderer with Pygments syntax highlighting and modern styling."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, select_autoescape

from polylogue.paths import conversation_render_root
from polylogue.rendering.core import (
    ConversationFormatter,
    FormattedConversation,
)
from polylogue.rendering.renderers.html_highlighting import (
    HTMLMessageRenderer,
    PygmentsHighlighter,
)
from polylogue.rendering.renderers.html_messages import (
    _attach_branches,
    build_conversation_html_messages,
    build_projection_html_messages,
)
from polylogue.rendering.renderers.html_template import get_cached_template

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.state_views import ConversationRenderProjection


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
        self.message_renderer = HTMLMessageRenderer(self.highlighter)

        # Pre-compile Jinja2 template once (avoid per-render overhead)
        self._jinja_env: Environment | None
        if self.template_path and self.template_path.exists():
            self._jinja_env = Environment(
                loader=FileSystemLoader(self.template_path.parent),
                autoescape=select_autoescape(["html", "xml"]),
            )
            self._template = self._jinja_env.get_template(self.template_path.name)
        else:
            self._jinja_env = None
            self._template = get_cached_template()
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

        html_messages = build_projection_html_messages(
            projection,
            render_html=self.message_renderer.render,
            preview_limit=120,
        )

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
            self._render_content_sync,
            formatted,
            projection,
        )

        # Phase 3: File writes in thread pool
        render_root_path = conversation_render_root(output_path, formatted.provider, conversation_id)

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
    message_renderer = HTMLMessageRenderer(highlighter)

    messages = build_conversation_html_messages(
        conv,
        render_html=message_renderer.render,
        preview_limit=120,
    )
    title = conv.display_title or str(conv.id)

    template = get_cached_template()

    rendered = template.render(
        title=title,
        provider=conv.provider,
        conversation_id=str(conv.id),
        messages=messages,
        message_count=len(messages),
        created_at=str(conv.display_date) if conv.display_date else None,
        highlight_css=highlighter.get_css(),
        theme=theme,
    )
    return str(rendered)


__all__ = [
    "HTMLMessageRenderer",
    "HTMLRenderer",
    "PygmentsHighlighter",
    "_attach_branches",
    "render_conversation_html",
]
