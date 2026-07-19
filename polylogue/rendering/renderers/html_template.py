"""Template support for HTML session rendering."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import DictLoader, Environment, select_autoescape

from polylogue.rendering.renderers.html_sanitizer import sanitize_html

DEFAULT_HTML_TEMPLATE = (Path(__file__).parent.parent / "templates" / "session.html").read_text()

_CACHED_TEMPLATE_ENV: Environment | None = None
_CACHED_TEMPLATE_ENV_LOCK = threading.Lock()

if TYPE_CHECKING:
    from jinja2 import Template


def _build_template_environment() -> Environment:
    env = Environment(
        loader=DictLoader({"session.html": DEFAULT_HTML_TEMPLATE}),
        autoescape=select_autoescape(["html", "xml"]),
    )
    env.filters["sanitize_html"] = sanitize_html
    return env


def get_cached_template() -> Template:
    """Return a module-level cached Jinja2 template for HTML rendering."""
    global _CACHED_TEMPLATE_ENV
    with _CACHED_TEMPLATE_ENV_LOCK:
        if _CACHED_TEMPLATE_ENV is None:
            _CACHED_TEMPLATE_ENV = _build_template_environment()
        env = _CACHED_TEMPLATE_ENV
    # Jinja2 Environment/Template objects are documented thread-safe for
    # concurrent template retrieval/rendering once built, so get_template()
    # runs outside the lock -- only the singleton's first-build race needs
    # guarding.
    return env.get_template("session.html")


__all__ = ["DEFAULT_HTML_TEMPLATE", "get_cached_template"]
