"""Serialize a session as HTML using shared UI theme tokens."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from tempfile import TemporaryFile
from typing import TYPE_CHECKING

from polylogue.rendering.renderers.html_highlighting import (
    HTMLMessageRenderer,
    PygmentsHighlighter,
)
from polylogue.rendering.renderers.html_messages import (
    _attach_branches,
    iter_html_messages,
)
from polylogue.rendering.renderers.html_template import get_cached_template
from polylogue.ui.theme import ThemeMode, css_variable_declarations, syntax_theme

if TYPE_CHECKING:
    from polylogue.archive.models import Message, Session


def render_session_html(conv: Session, theme: str = "dark") -> str:
    """Render an in-memory Session to a standalone HTML string.

    Uses the same Jinja2 template and Pygments highlighting as
    :class:`HTMLRenderer`, but works directly from a ``Session``
    object rather than querying the database.

    Args:
        conv: Session with messages to render.
        theme: ``"dark"`` or ``"light"``.

    Returns:
        Complete HTML document as a string.
    """
    return "".join(session_html_chunks(conv, conv.messages, theme))


def session_html_chunks(conv: Session, messages: Iterable[Message], theme: str = "dark") -> Iterator[str]:
    from polylogue.rendering.renderers.html_records import HTMLRecords

    active_theme: ThemeMode = "light" if theme == "light" else "dark"
    highlighter = PygmentsHighlighter(style=syntax_theme("html", active_theme))
    message_renderer = HTMLMessageRenderer(highlighter)
    with TemporaryFile(mode="w+t", encoding="utf-8") as stream:
        records = HTMLRecords(
            stream, iter_html_messages(messages, render_html=message_renderer.render, preview_limit=120)
        )
        template = get_cached_template()
        yield from template.generate(
            title=conv.display_title or str(conv.id),
            origin=conv.origin.value,
            session_id=str(conv.id),
            messages=records,
            message_count=len(records),
            created_at=str(conv.display_date) if conv.display_date else None,
            highlight_css=highlighter.get_css(),
            theme=active_theme,
            theme_css=css_variable_declarations(active_theme),
        )


__all__ = ["HTMLMessageRenderer", "PygmentsHighlighter", "_attach_branches", "render_session_html"]
