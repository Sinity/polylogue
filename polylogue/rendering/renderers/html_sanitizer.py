"""HTML sanitization filter for Jinja2 templates.

Provides a ``sanitize_html`` filter that strips dangerous constructs
(``<script>``, event handler attributes, ``javascript:`` URLs) from
pre-rendered HTML while preserving safe markup. Intended as a
defense-in-depth replacement for ``| safe`` in conversation templates.
"""

from __future__ import annotations

import re

from markupsafe import Markup

# Closing tag tolerates whitespace AND attribute-like junk before ``>``.
# Real browsers accept ``</script bar>``, ``</script\t\n>``, etc., even
# though strict HTML5 forbids them. Match ``</script\s*[^>]*>`` so end
# tags with whitespace, tabs, newlines, or trailing junk-attributes are
# all stripped — the previous strict ``</script>`` literal allowed each
# of those as a sanitizer bypass.
#
# Note: regex sanitization is defense-in-depth here; templates already
# Jinja-autoescape. A proper HTML parser (``bleach``) would be sturdier
# but is overkill for this filter's role.
_SCRIPT_RE = re.compile(r"<script[\s>].*?</script\s*[^>]*>", re.IGNORECASE | re.DOTALL)
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
