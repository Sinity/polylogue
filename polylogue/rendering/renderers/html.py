"""Enhanced HTML renderer with Pygments syntax highlighting and modern styling."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.rendering.renderers.html_highlighting import (
    HTMLMessageRenderer,
    PygmentsHighlighter,
)
from polylogue.rendering.renderers.html_messages import (
    _attach_branches,
    build_session_html_messages,
)
from polylogue.rendering.renderers.html_template import get_cached_template
from polylogue.ui.theme import ThemeMode, css_variable_declarations, syntax_theme

if TYPE_CHECKING:
    from polylogue.archive.models import Session


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
    active_theme: ThemeMode = "light" if theme == "light" else "dark"
    style = syntax_theme("html", active_theme)
    highlighter = PygmentsHighlighter(style=style)
    message_renderer = HTMLMessageRenderer(highlighter)

    messages = build_session_html_messages(
        conv,
        render_html=message_renderer.render,
        preview_limit=120,
    )
    title = conv.display_title or str(conv.id)

    template = get_cached_template()

    rendered = template.render(
        title=title,
        origin=conv.origin.value,
        session_id=str(conv.id),
        messages=messages,
        message_count=len(messages),
        created_at=str(conv.display_date) if conv.display_date else None,
        highlight_css=highlighter.get_css(),
        theme=active_theme,
        theme_css=css_variable_declarations(active_theme),
    )
    return str(rendered)


__all__ = [
    "HTMLMessageRenderer",
    "PygmentsHighlighter",
    "_attach_branches",
    "render_session_html",
]
