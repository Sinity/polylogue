"""Renderer factory and implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.config import Config
from polylogue.protocols import OutputRenderer

from .html import HTMLRenderer
from .markdown import MarkdownRenderer

if TYPE_CHECKING:
    from polylogue.storage.backends.async_sqlite import SQLiteBackend


def _markdown_renderer(config: Config, backend: SQLiteBackend | None) -> OutputRenderer:
    del backend
    return MarkdownRenderer(archive_root=config.archive_root)


def _html_renderer(config: Config, backend: SQLiteBackend | None) -> OutputRenderer:
    template_path = config.html_template if hasattr(config, "html_template") else None
    return HTMLRenderer(
        archive_root=config.archive_root,
        template_path=template_path,
        backend=backend,
    )


_RENDERER_FACTORIES = {
    "markdown": _markdown_renderer,
    "html": _html_renderer,
}


def create_renderer(
    format: str,
    config: Config,
    backend: SQLiteBackend | None = None,
) -> OutputRenderer:
    """Create a renderer for the specified format.

    Args:
        format: Output format ('markdown' or 'html')
        config: Application configuration
        backend: Optional shared backend for connection reuse

    Returns:
        OutputRenderer implementation for the requested format

    Raises:
        ValueError: If format is not supported
    """
    format_lower = format.lower()
    factory = _RENDERER_FACTORIES.get(format_lower)
    if factory is None:
        raise ValueError(f"Unsupported format: {format}. Supported formats: markdown, html")
    return factory(config, backend)


def list_formats() -> list[str]:
    """List all supported output formats.

    Returns:
        List of format identifiers
    """
    return sorted(_RENDERER_FACTORIES)


__all__ = [
    "HTMLRenderer",
    "MarkdownRenderer",
    "create_renderer",
    "list_formats",
]
