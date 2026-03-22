"""Stable facade for static-site template and page rendering helpers."""

from polylogue.site.conversation_pages import (
    generate_conversation_page,
    iter_conversation_page_messages,
    write_template_stream,
)
from polylogue.site.index_pages import (
    generate_dashboard,
    generate_provider_indexes,
    generate_root_index,
)
from polylogue.site.templates import (
    CONVERSATION_TEMPLATE,
    DASHBOARD_TEMPLATE,
    INDEX_TEMPLATE,
    build_template_environments,
)

__all__ = [
    "CONVERSATION_TEMPLATE",
    "DASHBOARD_TEMPLATE",
    "INDEX_TEMPLATE",
    "build_template_environments",
    "generate_conversation_page",
    "generate_dashboard",
    "generate_provider_indexes",
    "generate_root_index",
    "iter_conversation_page_messages",
    "write_template_stream",
]
