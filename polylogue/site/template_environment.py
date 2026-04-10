"""Template environment construction for the static site."""

from __future__ import annotations

from jinja2 import DictLoader, Environment, select_autoescape

from polylogue.rendering.renderers.html import PygmentsHighlighter
from polylogue.site.templates import (
    CONVERSATION_TEMPLATE,
    DASHBOARD_TEMPLATE,
    INDEX_TEMPLATE,
)


def build_template_environments(highlighter: PygmentsHighlighter) -> tuple[Environment, Environment]:
    """Build index/dashboard and conversation template environments."""
    conversation_template = CONVERSATION_TEMPLATE.replace(
        "{{ highlight_css }}",
        highlighter.get_css(),
    )
    index_env = Environment(
        loader=DictLoader(
            {
                "index.html": INDEX_TEMPLATE,
                "dashboard.html": DASHBOARD_TEMPLATE,
            }
        ),
        autoescape=select_autoescape(["html", "xml"]),
        enable_async=True,
    )
    page_env = Environment(
        loader=DictLoader(
            {
                "conversation.html": conversation_template,
            }
        ),
        autoescape=select_autoescape(["html", "xml"]),
        enable_async=True,
    )
    return index_env, page_env


__all__ = ["build_template_environments"]
