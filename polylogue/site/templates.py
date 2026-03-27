"""Small public root for static-site template families."""

from __future__ import annotations

from polylogue.site.templates_conversation import CONVERSATION_TEMPLATE
from polylogue.site.templates_dashboard import DASHBOARD_TEMPLATE
from polylogue.site.templates_index import INDEX_TEMPLATE

__all__ = [
    "CONVERSATION_TEMPLATE",
    "DASHBOARD_TEMPLATE",
    "INDEX_TEMPLATE",
]
