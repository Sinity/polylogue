"""HTML sanitization filter for Jinja2 templates.

Provides a ``sanitize_html`` filter that strips dangerous constructs
(``<script>``, event handler attributes, ``javascript:`` URLs, embedded
frames, raw SVG/MathML interactivity) from pre-rendered HTML while
preserving safe markup. Intended as a defense-in-depth replacement for
``| safe`` in session templates.

Backed by ``nh3`` (Rust ``ammonia`` bindings), an HTML5 parser-based
sanitizer that is robust against the regex bypass classes the prior
implementation suffered from:

* unquoted event-handler attributes (``<a onclick=alert(1)>``)
* ``javascript:`` and ``data:`` URIs on any URL-bearing attribute
* ``<iframe>``/``<object>``/``<embed>`` element injection
* ``<svg>``/``<math>`` interactivity (event handlers, foreign content)
"""

from __future__ import annotations

from typing import Final

import nh3
from markupsafe import Markup

# Conservative allowlist: structural and inline text formatting tags only.
# Excludes form, media, object, frame, svg, math, script, and style tags.
_ALLOWED_TAGS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "abbr",
        "b",
        "blockquote",
        "br",
        "cite",
        "code",
        "dd",
        "del",
        "details",
        "div",
        "dl",
        "dt",
        "em",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "i",
        "ins",
        "kbd",
        "li",
        "mark",
        "ol",
        "p",
        "pre",
        "q",
        "s",
        "samp",
        "small",
        "span",
        "strong",
        "sub",
        "summary",
        "sup",
        "table",
        "tbody",
        "td",
        "th",
        "thead",
        "tr",
        "u",
        "ul",
        "var",
    }
)

# Per-tag attribute allowlist. ``href`` only on ``<a>``; ``class``/``id``
# everywhere for renderer-driven styling. Event handlers (``on*``) are
# not in any allowlist and so are stripped by the parser.
_ALLOWED_ATTRIBUTES: Final[dict[str, set[str]]] = {
    "*": {"class", "id", "title", "lang", "dir"},
    "a": {"href"},
    "abbr": {"title"},
    "ol": {"start", "type"},
    "ul": {"type"},
    "td": {"colspan", "rowspan"},
    "th": {"colspan", "rowspan", "scope"},
}

# URL schemes permitted on URL-bearing attributes (currently only
# ``href`` on ``<a>``). ``javascript:`` and ``data:`` are intentionally
# absent so they are stripped.
_ALLOWED_URL_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https", "mailto"})


def sanitize_html(html: str) -> Markup:
    """Strip dangerous constructs from pre-rendered HTML.

    Removes ``<script>`` tags, event handler attributes (``onclick``,
    ``onload``, ``onerror`` — including unquoted variants), URL schemes
    other than ``http``/``https``/``mailto`` (so ``javascript:`` and
    ``data:`` URIs are dropped from any attribute), and frame/object
    embedding tags. Returns a ``Markup`` object so Jinja2 autoescaping
    treats the result as safe HTML.
    """
    if not html:
        return Markup("")
    sanitized = nh3.clean(
        html,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRIBUTES,
        generic_attribute_prefixes={"data-", "aria-"},
        url_schemes=_ALLOWED_URL_SCHEMES,
        link_rel="noopener noreferrer",
    )
    return Markup(sanitized)
