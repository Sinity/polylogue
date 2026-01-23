"""ID generation and content hashing logic for pipeline items."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from polylogue.assets import asset_path
from polylogue.core.hashing import hash_file, hash_payload, hash_text
from polylogue.ingestion import ParsedAttachment, ParsedConversation, ParsedMessage
from polylogue.types import AttachmentId, ContentHash, ConversationId, MessageId

# Sentinel values to distinguish None from empty in hash computations
_NULL_SENTINEL = "__POLYLOGUE_NULL__"
_EMPTY_SENTINEL = "__POLYLOGUE_EMPTY__"


def move_attachment_to_archive(source: Path, dest: Path) -> None:
    """Move attachment file to archive location.

    Creates parent directories and moves the file atomically. Raises on failure.

    Args:
        source: Source file path.
        dest: Destination file path.

    Raises:
        FileNotFoundError: If source file doesn't exist.
        PermissionError: If move fails due to permissions.
        OSError: For other filesystem errors.
    """
    if not source.exists():
        raise FileNotFoundError(f"Source attachment not found: {source}")

    # Create parent directories
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Move file (will raise on failure)
    shutil.move(str(source), str(dest))


def _normalize_for_hash(value: Any) -> Any:
    """Normalize a value for hashing, distinguishing None from empty.

    Args:
        value: Any value to normalize.

    Returns:
        Normalized value with None → _NULL_SENTINEL and "" → _EMPTY_SENTINEL.
    """
    if value is None:
        return _NULL_SENTINEL
    if value == "":
        return _EMPTY_SENTINEL
    return value


def attachment_seed(provider_name: str, attachment: ParsedAttachment) -> str:
    return "|".join(
        str(value)
        for value in [
            provider_name,
            attachment.provider_attachment_id,
            attachment.message_provider_id,
            attachment.name,
            attachment.mime_type,
            attachment.size_bytes,
            attachment.path,
        ]
        if value is not None
    )


def attachment_content_id(
    provider_name: str,
    attachment: ParsedAttachment,
    *,
    archive_root: Path,
) -> tuple[str, dict | None, str | None]:
    """Compute attachment content ID and return updated metadata.

    Returns:
        Tuple of (attachment_id, updated_provider_meta, updated_path).
        The caller is responsible for applying any updates to the attachment.
        This function does NOT mutate the attachment object.

    Raises:
        OSError: If attachment file move fails (FileNotFoundError, PermissionError, etc).
    """
    meta = dict(attachment.provider_meta or {})
    updated_path = attachment.path
    for key in ("sha256", "digest", "hash"):
        value = meta.get(key)
        if isinstance(value, str) and value:
            return (value, meta, updated_path)
    raw_path = attachment.path
    if isinstance(raw_path, str) and raw_path:
        path = Path(raw_path)
        if path.exists() and path.is_file():
            digest = hash_file(path)
            meta.setdefault("sha256", digest)
            target = asset_path(archive_root, digest)
            if target != path:
                if target.exists():
                    path.unlink()
                else:
                    move_attachment_to_archive(path, target)
                updated_path = str(target)
            else:
                updated_path = str(path)
            return (digest, meta, updated_path)
    seed = attachment_seed(provider_name, attachment)
    return (hash_text(seed), meta, updated_path)


def conversation_id(provider_name: str, provider_conversation_id: str) -> ConversationId:
    """Generate deterministic conversation ID from provider info.

    Args:
        provider_name: Name of the provider (e.g., "chatgpt", "claude").
        provider_conversation_id: Provider's conversation identifier.

    Returns:
        Formatted conversation ID.

    Raises:
        ValueError: If provider_name or provider_conversation_id is empty.
    """
    if not provider_name or not provider_name.strip():
        raise ValueError("provider_name cannot be empty")
    if not provider_conversation_id or not provider_conversation_id.strip():
        raise ValueError("provider_conversation_id cannot be empty")
    return ConversationId(f"{provider_name}:{provider_conversation_id}")


def message_id(conversation_id: ConversationId, provider_message_id: str) -> MessageId:
    return MessageId(f"{conversation_id}:{provider_message_id}")


def message_content_hash(message: ParsedMessage, provider_message_id: str) -> ContentHash:
    """Generate content hash for a message.

    Uses sentinel values to distinguish None from empty string.

    Args:
        message: Parsed message object.
        provider_message_id: Provider's message identifier.

    Returns:
        Content hash string.
    """
    payload = {
        "id": provider_message_id,
        "role": message.role,
        "text": _normalize_for_hash(message.text),
        "timestamp": _normalize_for_hash(message.timestamp),
    }
    return ContentHash(hash_payload(payload))


def conversation_content_hash(convo: ParsedConversation) -> ContentHash:
    """Generate content hash for conversation.

    Uses sentinel values to distinguish None from empty/missing fields.

    Args:
        convo: Parsed conversation object.

    Returns:
        Content hash string.
    """
    messages_payload = []
    for idx, msg in enumerate(convo.messages, start=1):
        message_id = msg.provider_message_id or f"msg-{idx}"
        messages_payload.append(
            {
                "id": message_id,
                "role": msg.role,
                "text": _normalize_for_hash(msg.text),
                "timestamp": _normalize_for_hash(msg.timestamp),
            }
        )
    attachments_payload = sorted(
        [
            {
                "id": _normalize_for_hash(att.provider_attachment_id),
                "message_id": _normalize_for_hash(att.message_provider_id),
                "name": _normalize_for_hash(att.name),
                "mime_type": _normalize_for_hash(att.mime_type),
                "size_bytes": _normalize_for_hash(att.size_bytes),
            }
            for att in convo.attachments
        ],
        key=lambda item: (
            item.get("message_id") or "",
            item.get("id") or "",
            item.get("name") or "",
        ),
    )
    return ContentHash(hash_payload(
        {
            "title": _normalize_for_hash(convo.title),
            "created_at": _normalize_for_hash(convo.created_at),
            "updated_at": _normalize_for_hash(convo.updated_at),
            "messages": messages_payload,
            "attachments": attachments_payload,
        }
    ))
