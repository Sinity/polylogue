"""Markdown renderer implementation."""

from __future__ import annotations

from pathlib import Path

from polylogue.render_paths import render_root
from polylogue.rendering.core import ConversationFormatter


class MarkdownRenderer:
    """Renders conversations to plain Markdown format."""

    def __init__(self, archive_root: Path):
        """Initialize the Markdown renderer.

        Args:
            archive_root: Root directory for archived conversations
        """
        self.archive_root = archive_root
        self.formatter = ConversationFormatter(archive_root)

    def supports_format(self) -> str:
        """Return the output format this renderer supports.

        Returns:
            'markdown'
        """
        return "markdown"

    async def render(self, conversation_id: str, output_path: Path) -> Path:
        """Render a conversation to Markdown format.

        Args:
            conversation_id: ID of the conversation to render
            output_path: Directory where the markdown file should be written

        Returns:
            Path to the generated markdown file

        Raises:
            ValueError: If conversation not found
            IOError: If output path is invalid or write fails
        """
        # Use shared formatter to get markdown
        formatted = await self.formatter.format(conversation_id)

        # Determine output path
        render_root_path = render_root(output_path, formatted.provider, conversation_id)
        render_root_path.mkdir(parents=True, exist_ok=True)
        md_path = render_root_path / "conversation.md"

        # Write markdown file
        md_path.write_text(formatted.markdown_text, encoding="utf-8")

        return md_path


__all__ = ["MarkdownRenderer"]
