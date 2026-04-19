"""Shared helpers and public roots for static-site template families."""

from __future__ import annotations

from polylogue.site.templates_conversation import CONVERSATION_TEMPLATE
from polylogue.site.templates_dashboard import DASHBOARD_STYLE, DASHBOARD_TEMPLATE
from polylogue.site.templates_index import INDEX_TEMPLATE
from polylogue.site.templates_index_style import INDEX_STYLE

SITE_FONT_STACK = "'Inter', -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, sans-serif"
SITE_CODE_FONT_STACK = "'JetBrains Mono', 'Fira Code', monospace"


def build_html_document(*, title: str, styles: str, body: str) -> str:
    """Wrap a body/style fragment in the shared site HTML shell."""
    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
{styles}
    </style>
</head>
<body>
{body}
</body>
</html>
"""


__all__ = [
    "CONVERSATION_TEMPLATE",
    "DASHBOARD_STYLE",
    "DASHBOARD_TEMPLATE",
    "INDEX_STYLE",
    "INDEX_TEMPLATE",
    "SITE_CODE_FONT_STACK",
    "SITE_FONT_STACK",
    "build_html_document",
]
