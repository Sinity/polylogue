"""HTML sanitization filter for Jinja2 templates.

Provides a ``sanitize_html`` filter that strips dangerous constructs
(``<script>``, event handler attributes, ``javascript:`` URLs) from
pre-rendered HTML while preserving safe markup. Intended as a
defense-in-depth replacement for ``| safe`` in conversation templates.
"""

from __future__ import annotations

import re

from markupsafe import Markup

_SCRIPT_RE = re.compile(r"<script[\s>].*?</script>", re.IGNORECASE | re.DOTALL)
_EVENT_ATTR_RE = re.compile(r"\s+on\w+\s*=\s*[\"'][^\"']*[\"']", re.IGNORECASE)
_JAVASCRIPT_URL_RE = re.compile(r"""href\s*=\s*["']javascript:[^"']*["']""", re.IGNORECASE)


def sanitize_html(html: str) -> Markup:
    """Strip dangerous constructs from pre-rendered HTML.

    Removes ``<script>`` tags, event handler attributes (``onclick``,
    ``onload``, etc.), and ``javascript:`` URLs. Returns a ``Markup``
    object so Jinja2 autoescaping treats the result as safe HTML.
    """
    if not html:
        return Markup("")
    sanitized = _SCRIPT_RE.sub("", html)
    sanitized = _EVENT_ATTR_RE.sub("", sanitized)
    sanitized = _JAVASCRIPT_URL_RE.sub("", sanitized)
    return Markup(sanitized)
