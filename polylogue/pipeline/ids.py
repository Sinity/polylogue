"""ID generation and content hashing logic for pipeline items."""

from __future__ import annotations

from pathlib import Path

from polylogue.assets import asset_path
from polylogue.core.hashing import hash_file, hash_payload, hash_text
from polylogue.source_ingest import ParsedAttachment, ParsedConversation, ParsedMessage


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
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    path.unlink()
                else:
                    try:
                        path.replace(target)
                    except OSError:
                        target.write_bytes(path.read_bytes())
                        path.unlink()
                updated_path = str(target)
            else:
                updated_path = str(path)
            return (digest, meta, updated_path)
    seed = attachment_seed(provider_name, attachment)
    return (hash_text(seed), meta, updated_path)


def conversation_id(provider_name: str, provider_conversation_id: str) -> str:
    return f"{provider_name}:{provider_conversation_id}"


def message_id(conversation_id: str, provider_message_id: str) -> str:
    return f"{conversation_id}:{provider_message_id}"


def message_content_hash(message: ParsedMessage, provider_message_id: str) -> str:
    payload = {
        "id": provider_message_id,
        "role": message.role,
        "text": message.text,
        "timestamp": message.timestamp,
    }
    return hash_payload(payload)


def conversation_content_hash(convo: ParsedConversation) -> str:
    messages_payload = []
    for idx, msg in enumerate(convo.messages, start=1):
        message_id = msg.provider_message_id or f"msg-{idx}"
        messages_payload.append(
            {
                "id": message_id,
                "role": msg.role,
                "text": msg.text,
                "timestamp": msg.timestamp,
            }
        )
    attachments_payload = sorted(
        [
            {
                "id": att.provider_attachment_id,
                "message_id": att.message_provider_id,
                "name": att.name,
                "mime_type": att.mime_type,
                "size_bytes": att.size_bytes,
            }
            for att in convo.attachments
        ],
        key=lambda item: (
            item.get("message_id") or "",
            item.get("id") or "",
            item.get("name") or "",
        ),
    )
    return hash_payload(
        {
            "title": convo.title,
            "created_at": convo.created_at,
            "updated_at": convo.updated_at,
            "messages": messages_payload,
            "attachments": attachments_payload,
        }
    )
