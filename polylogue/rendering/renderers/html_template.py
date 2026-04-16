"""Template support for HTML conversation rendering."""

from __future__ import annotations

from pathlib import Path

from jinja2 import DictLoader, Environment, select_autoescape

DEFAULT_HTML_TEMPLATE = (Path(__file__).parent.parent / "templates" / "conversation.html").read_text()

_CACHED_TEMPLATE_ENV: Environment | None = None


def get_cached_template():
    """Return a module-level cached Jinja2 template for HTML rendering."""
    global _CACHED_TEMPLATE_ENV
    if _CACHED_TEMPLATE_ENV is None:
        _CACHED_TEMPLATE_ENV = Environment(
            loader=DictLoader({"conversation.html": DEFAULT_HTML_TEMPLATE}),
            autoescape=select_autoescape(["html", "xml"]),
        )
    return _CACHED_TEMPLATE_ENV.get_template("conversation.html")


__all__ = ["DEFAULT_HTML_TEMPLATE", "get_cached_template"]
