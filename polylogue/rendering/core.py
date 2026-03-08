"""Core rendering utilities shared across all renderers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation
    from polylogue.storage.store import ConversationRenderProjection

from polylogue.assets import asset_path
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository


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
            meta_dict = meta if isinstance(meta, dict) else json.loads(meta)
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


def _normalize_markdown_timestamp(timestamp: Any) -> str | None:
    if timestamp is None:
        return None
    if isinstance(timestamp, datetime):
        return timestamp.isoformat()
    return str(timestamp)


def _normalize_markdown_attachment(
    *,
    attachment_id: str,
    path: str | Path | None,
    provider_meta: Any,
) -> dict[str, Any]:
    return {
        "attachment_id": attachment_id,
        "path": path,
        "provider_meta": provider_meta,
    }


def _normalize_markdown_message(
    *,
    message_id: str,
    role: Any,
    text: str | None,
    timestamp: Any,
    default_role: str,
) -> dict[str, Any]:
    normalized_role = (
        (role.value if hasattr(role, "value") else str(role))
        if role
        else default_role
    )
    return {
        "message_id": message_id,
        "role": normalized_role,
        "text": text,
        "timestamp": _normalize_markdown_timestamp(timestamp),
    }


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
    """Formats repository render projections to structured output."""

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

    async def load_projection(self, conversation_id: str) -> ConversationRenderProjection:
        """Load the canonical repository-owned render projection."""
        backend = self.backend or SQLiteBackend(db_path=self.db_path)
        repository = ConversationRepository(backend=backend)
        owns_backend = self.backend is None
        try:
            projection = await repository.get_render_projection(conversation_id)
        finally:
            if owns_backend:
                await repository.close()
        if projection is None:
            raise ValueError(f"Conversation not found: {conversation_id}")
        return projection

    def format_projection(self, projection: ConversationRenderProjection) -> FormattedConversation:
        """Format a repository projection to structured output."""
        conversation = projection.conversation
        conversation_id = str(conversation.conversation_id)
        title = conversation.title or conversation_id
        provider = conversation.provider_name
        normalized_messages = [
            _normalize_markdown_message(
                message_id=message.message_id,
                role=message.role,
                text=message.text,
                timestamp=message.timestamp,
                default_role="message",
            )
            for message in projection.messages
        ]
        normalized_attachments = {
            key: [
                _normalize_markdown_attachment(
                    attachment_id=attachment.attachment_id,
                    path=attachment.path,
                    provider_meta=attachment.provider_meta,
                )
                for attachment in attachments
            ]
            for key, attachments in _group_projection_attachments(projection).items()
        }
        markdown_text = render_markdown_document(
            title=title,
            provider=provider,
            conversation_id=conversation_id,
            messages=normalized_messages,
            attachments_by_message=normalized_attachments,
            archive_root=self.archive_root,
        )
        return FormattedConversation(
            title=title,
            provider=provider,
            conversation_id=conversation_id,
            markdown_text=markdown_text,
            metadata={
                "message_count": len(projection.messages),
                "attachment_count": len(projection.attachments),
                "created_at": conversation.created_at,
                "updated_at": conversation.updated_at,
            },
        )

    async def format(self, conversation_id: str) -> FormattedConversation:
        """Format a conversation to structured output."""
        return self.format_projection(await self.load_projection(conversation_id))


def _group_projection_attachments(
    projection: ConversationRenderProjection,
) -> dict[str | None, list[Any]]:
    attachments_by_message: dict[str | None, list[Any]] = {}
    for attachment in projection.attachments:
        attachments_by_message.setdefault(attachment.message_id, []).append(attachment)
    return attachments_by_message


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
    attachments_by_message: dict[str | None, list[dict[str, Any]]] = {}
    normalized_messages = []

    for msg in conv.messages:
        message_id = str(msg.id)
        normalized_messages.append(
            _normalize_markdown_message(
                message_id=message_id,
                role=msg.role,
                text=msg.text,
                timestamp=msg.timestamp,
                default_role="unknown",
            )
        )
        if getattr(msg, "attachments", None):
            attachments_by_message[message_id] = [
                _normalize_markdown_attachment(
                    attachment_id=str(att.id),
                    path=att.path,
                    provider_meta=att.provider_meta,
                )
                for att in msg.attachments
            ]

    archive_root = Path(".")
    return render_markdown_document(
        title=conv.title or "Untitled",
        provider=conv.provider,
        conversation_id=str(conv.id),
        messages=normalized_messages,
        attachments_by_message=attachments_by_message,
        archive_root=archive_root,
    )


__all__ = ["ConversationFormatter", "FormattedConversation", "format_conversation_markdown"]
