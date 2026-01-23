"""Renderer factory and implementations."""

from __future__ import annotations

from pathlib import Path

from polylogue.config import Config
from polylogue.protocols import OutputRenderer

from .html import HTMLRenderer
from .markdown import MarkdownRenderer


def create_renderer(format: str, config: Config) -> OutputRenderer:
    """Create a renderer for the specified format.

    Args:
        format: Output format ('markdown' or 'html')
        config: Application configuration

    Returns:
        OutputRenderer implementation for the requested format

    Raises:
        ValueError: If format is not supported
    """
    format_lower = format.lower()

    if format_lower == "markdown":
        return MarkdownRenderer(archive_root=config.archive_root)
    elif format_lower == "html":
        template_path = config.html_template if hasattr(config, "html_template") else None
        return HTMLRenderer(archive_root=config.archive_root, template_path=template_path)
    else:
        raise ValueError(f"Unsupported format: {format}. Supported formats: markdown, html")


def list_formats() -> list[str]:
    """List all supported output formats.

    Returns:
        List of format identifiers
    """
    return ["markdown", "html"]


__all__ = [
    "HTMLRenderer",
    "MarkdownRenderer",
    "create_renderer",
    "list_formats",
]
