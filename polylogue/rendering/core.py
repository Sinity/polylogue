"""Core rendering utilities shared across all renderers.

This module provides the ConversationFormatter class, which eliminates
duplication between MarkdownRenderer and HTMLRenderer by extracting the
common database query and formatting logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation

from polylogue.assets import asset_path
from polylogue.storage.backends.async_sqlite import SQLiteBackend


def format_message_text(text: str) -> str:
    """Format message text, wrapping JSON-looking payloads in fenced blocks."""
    if not text:
        return ""
    stripped = text.strip()
    if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
        try:
            parsed = json.loads(stripped)
            return f"```json\n{json.dumps(parsed, indent=2)}\n```"
        except json.JSONDecodeError:
            pass
    return text


def append_attachment_markdown(att: dict[str, Any], lines: list[str], archive_root: Path) -> None:
    """Append a single attachment line to a markdown output buffer."""
    name = None
    meta = att.get("provider_meta")
    if meta:
        try:
            meta_dict = json.loads(meta)
            name = meta_dict.get("name") or meta_dict.get("provider_id") or meta_dict.get("drive_id")
        except (json.JSONDecodeError, TypeError):
            name = None
    label = name or att["attachment_id"]
    path_value = att.get("path") or str(asset_path(archive_root, att["attachment_id"]))
    lines.append(f"- Attachment: {label} ({path_value})")


def render_markdown_document(
    *,
    title: str,
    provider: str,
    conversation_id: str,
    messages: list[dict[str, Any]],
    attachments_by_message: dict[str | None, list[dict[str, Any]]],
    archive_root: Path,
) -> str:
    """Render a conversation payload to canonical markdown."""
    lines = [f"# {title}", "", f"Provider: {provider}", f"Conversation ID: {conversation_id}", ""]
    message_ids: set[str] = set()

    for msg in messages:
        message_id = msg["message_id"]
        message_ids.add(message_id)
        role = msg["role"] or "message"
        text = msg["text"] or ""
        timestamp = msg.get("timestamp")
        msg_atts = attachments_by_message.get(message_id, [])

        if not text.strip() and not msg_atts:
            continue

        lines.append(f"## {role}")
        if timestamp:
            lines.append(f"_Timestamp: {timestamp}_")
        lines.append("")

        formatted_text = format_message_text(text)
        if formatted_text:
            lines.append(formatted_text)
            lines.append("")

        for att in msg_atts:
            append_attachment_markdown(att, lines, archive_root)
        lines.append("")

    orphan_keys = [key for key in attachments_by_message if key not in message_ids]
    if orphan_keys:
        lines.append("## attachments")
        lines.append("")
        for key in sorted(orphan_keys, key=lambda item: (item is None, str(item) if item else "")):
            for att in attachments_by_message.get(key, []):
                append_attachment_markdown(att, lines, archive_root)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


@dataclass
class FormattedConversation:
    """Structured representation of a rendered conversation.

    This intermediate format can be consumed by different renderers
    (Markdown, HTML, PDF, etc.) without duplicating the formatting logic.
    """

    title: str
    provider: str
    conversation_id: str
    markdown_text: str
    metadata: dict[str, Any]


class ConversationFormatter:
    """Formats conversations from database to structured output.

    This class eliminates duplication between renderers by providing
    a single source of truth for:
    - Database queries (conversation, messages, attachments)
    - Attachment metadata extraction
    - Text formatting (JSON detection, code blocks)
    - Markdown generation

    Usage:
        formatter = ConversationFormatter(archive_root)
        formatted = formatter.format(conversation_id)
        # Use formatted.markdown_text in MarkdownRenderer
        # Or convert to HTML in HTMLRenderer
    """

    def __init__(self, archive_root: Path, db_path: Path | None = None, backend: SQLiteBackend | None = None):
        """Initialize the formatter.

        Args:
            archive_root: Root directory for archived conversations
            db_path: Optional database path (defaults to standard location)
            backend: Optional async SQLite backend instance
        """
        self.archive_root = archive_root
        self.db_path = db_path
        self.backend = backend

    async def format(self, conversation_id: str) -> FormattedConversation:
        """Format a conversation to structured output.

        Args:
            conversation_id: ID of the conversation to format

        Returns:
            FormattedConversation with all formatted data

        Raises:
            ValueError: If conversation not found
        """
        # Use provided backend or create one
        backend = self.backend or SQLiteBackend(db_path=self.db_path)

        # Query database
        async with backend.connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
            convo = await cursor.fetchone()
            if not convo:
                raise ValueError(f"Conversation not found: {conversation_id}")

            cursor = await conn.execute(
                """
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY (sort_key IS NULL), sort_key, message_id
                """,
                (conversation_id,),
            )
            messages = await cursor.fetchall()

            cursor = await conn.execute(
                """
                SELECT
                    attachment_refs.message_id,
                    attachments.attachment_id,
                    attachments.mime_type,
                    attachments.size_bytes,
                    attachments.path,
                    attachments.provider_meta
                FROM attachment_refs
                JOIN attachments ON attachments.attachment_id = attachment_refs.attachment_id
                WHERE attachment_refs.conversation_id = ?
                """,
                (conversation_id,),
            )
            attachments = await cursor.fetchall()

        # Build attachments mapping
        attachments_by_message: dict[str, list[Any]] = {}
        for att in attachments:
            attachments_by_message.setdefault(att["message_id"], []).append(att)

        # Extract metadata
        title = convo["title"] or conversation_id
        provider = convo["provider_name"]

        # Format to markdown
        markdown_text = self._format_markdown(
            title=title,
            provider=provider,
            conversation_id=conversation_id,
            messages=messages,
            attachments_by_message=attachments_by_message,
        )

        return FormattedConversation(
            title=title,
            provider=provider,
            conversation_id=conversation_id,
            markdown_text=markdown_text,
            metadata={
                "message_count": len(messages),
                "attachment_count": len(attachments),
                "created_at": convo["created_at"],
                "updated_at": convo["updated_at"],
            },
        )

    def _format_markdown(
        self,
        title: str,
        provider: str,
        conversation_id: str,
        messages: list[Any],
        attachments_by_message: dict[str, list[Any]],
    ) -> str:
        """Format conversation data to markdown text."""
        normalized_messages = [dict(msg) for msg in messages]
        normalized_attachments = {
            key: [dict(att) for att in atts]
            for key, atts in attachments_by_message.items()
        }
        return render_markdown_document(
            title=title,
            provider=provider,
            conversation_id=conversation_id,
            messages=normalized_messages,
            attachments_by_message=normalized_attachments,
            archive_root=self.archive_root,
        )


def format_conversation_markdown(conv: Conversation) -> str:
    """Format a loaded Conversation domain object to markdown.

    Works with a Conversation that has messages already loaded (eager or lazy).
    This avoids duplicating markdown formatting logic in multiple places
    (TUI browser, CLI display, etc.)

    Args:
        conv: A Conversation domain object with .title, .provider, .messages

    Returns:
        Formatted markdown string
    """
    lines = [f"# {conv.title or 'Untitled'}", ""]

    if hasattr(conv, "provider") and conv.provider:
        lines.append(f"**Provider:** {conv.provider}")
    if hasattr(conv, "created_at") and conv.created_at:
        lines.append(f"**Date:** {conv.created_at}")
    lines.append("")

    for msg in conv.messages:
        role = (msg.role.value if hasattr(msg.role, "value") else str(msg.role)) if msg.role else "unknown"
        text = msg.text or ""

        if not text.strip():
            continue

        lines.append(f"## {role}")
        if hasattr(msg, "timestamp") and msg.timestamp:
            lines.append(f"_{msg.timestamp}_")
        lines.append("")

        # Wrap raw JSON in code blocks
        stripped = text.strip()
        if (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        ):
            try:
                parsed = json.loads(stripped)
                text = f"```json\n{json.dumps(parsed, indent=2)}\n```"
            except json.JSONDecodeError:
                pass

        lines.append(text)
        lines.append("")

        if hasattr(msg, "attachments") and msg.attachments:
            lines.append(f"**Attachments:** {len(msg.attachments)}")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


__all__ = ["ConversationFormatter", "FormattedConversation", "format_conversation_markdown"]
