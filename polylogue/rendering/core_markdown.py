"""Markdown normalization and document rendering helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from polylogue.assets import asset_path
from polylogue.lib.roles import Role

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation
    from polylogue.storage.state_views import ConversationRenderProjection


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
    if isinstance(timestamp, (int, float)):
        try:
            return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat().replace("+00:00", "Z")
        except (ValueError, OSError):
            return None
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
    default_role: Role | str,
) -> dict[str, Any]:
    normalized_role = role if isinstance(role, Role) else (Role.normalize(str(role)) if role else default_role)
    return {
        "message_id": message_id,
        "role": str(normalized_role),
        "text": text,
        "timestamp": _normalize_markdown_timestamp(timestamp),
    }


def _group_projection_attachments(
    projection: ConversationRenderProjection,
) -> dict[str | None, list[Any]]:
    attachments_by_message: dict[str | None, list[Any]] = {}
    for attachment in projection.attachments:
        attachments_by_message.setdefault(attachment.message_id, []).append(attachment)
    return attachments_by_message


def format_conversation_markdown(conv: Conversation) -> str:
    """Format a loaded Conversation domain object to markdown."""
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
                default_role="message",
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

    return render_markdown_document(
        title=conv.title or "Untitled",
        provider=str(conv.provider),
        conversation_id=str(conv.id),
        messages=normalized_messages,
        attachments_by_message=attachments_by_message,
        archive_root=Path("."),
    )


__all__ = [
    "append_attachment_markdown",
    "format_conversation_markdown",
    "format_message_text",
    "render_markdown_document",
]
