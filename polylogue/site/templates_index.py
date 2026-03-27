"""Index page templates for the static site."""

from __future__ import annotations

from polylogue.site.templates_index_body import INDEX_BODY
from polylogue.site.templates_index_style import INDEX_STYLE

INDEX_TEMPLATE = (
    """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
"""
    + INDEX_STYLE
    + """
    </style>
</head>
<body>
"""
    + INDEX_BODY
    + """
</body>
</html>
"""
)

__all__ = ["INDEX_TEMPLATE"]
