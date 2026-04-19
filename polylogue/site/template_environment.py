"""Template environment construction for the static site."""

from __future__ import annotations

from jinja2 import DictLoader, Environment, select_autoescape

from polylogue.rendering.renderers.html import PygmentsHighlighter
from polylogue.site.templates import (
    CONVERSATION_TEMPLATE,
    DASHBOARD_TEMPLATE,
    INDEX_TEMPLATE,
)


def _build_environment(templates: dict[str, str]) -> Environment:
    return Environment(
        loader=DictLoader(templates),
        autoescape=select_autoescape(["html", "xml"]),
        enable_async=True,
    )


def build_template_environments(highlighter: PygmentsHighlighter) -> tuple[Environment, Environment]:
    """Build index/dashboard and conversation template environments."""
    conversation_template = CONVERSATION_TEMPLATE.replace(
        "{{ highlight_css }}",
        highlighter.get_css(),
    )
    index_env = _build_environment(
        {
            "index.html": INDEX_TEMPLATE,
            "dashboard.html": DASHBOARD_TEMPLATE,
        }
    )
    page_env = _build_environment({"conversation.html": conversation_template})
    return index_env, page_env


__all__ = ["build_template_environments"]
