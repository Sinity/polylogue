"""Plaintext renderer implementation."""

from __future__ import annotations

import re
from pathlib import Path

from polylogue.paths.sanitize import conversation_render_root
from polylogue.rendering.core import ConversationFormatter


class PlaintextRenderer:
    """Renders conversations to plain text format."""

    def __init__(self, archive_root: Path):
        """Initialize the plaintext renderer.

        Args:
            archive_root: Root directory for archived conversations
        """
        self.archive_root = archive_root
        self.formatter = ConversationFormatter(archive_root)

    def supports_format(self) -> str:
        """Return the output format this renderer supports.

        Returns:
            'plaintext'
        """
        return "plaintext"

    async def render(self, conversation_id: str, output_path: Path) -> Path:
        """Render a conversation to plain text format.

        Strips markdown formatting, returning just the raw message content.

        Args:
            conversation_id: ID of the conversation to render
            output_path: Directory where the plaintext file should be written

        Returns:
            Path to the generated plaintext file

        Raises:
            ValueError: If conversation not found
            IOError: If output path is invalid or write fails
        """
        formatted = await self.formatter.format(conversation_id)

        render_root_path = conversation_render_root(output_path, formatted.provider, conversation_id)
        render_root_path.mkdir(parents=True, exist_ok=True)
        txt_path = render_root_path / "conversation.txt"

        text = _strip_markdown(formatted.markdown_text)
        txt_path.write_text(text, encoding="utf-8")

        return txt_path


def _strip_markdown(markdown_text: str) -> str:
    """Strip common markdown formatting, returning plain text."""
    text = markdown_text

    # Remove headings markers (# through ######)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove bold/italic markers
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)

    # Remove inline code backticks
    text = re.sub(r"`(.+?)`", r"\1", text)

    # Remove image/link text (keep alt/label)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]+\)", r"\1", text)

    return text.strip()


__all__ = ["PlaintextRenderer"]
