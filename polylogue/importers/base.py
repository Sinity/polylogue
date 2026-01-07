from __future__ import annotations

from pydantic import BaseModel, Field

from polylogue.core.hashing import hash_text


class ParsedMessage(BaseModel):
    provider_message_id: str
    role: str
    text: str | None = None
    timestamp: str | None = None
    provider_meta: dict | None = None


class ParsedAttachment(BaseModel):
    provider_attachment_id: str
    message_provider_id: str | None = None
    name: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    path: str | None = None
    provider_meta: dict | None = None


class ParsedConversation(BaseModel):
    provider_name: str
    provider_conversation_id: str
    title: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    messages: list[ParsedMessage]
    attachments: list[ParsedAttachment] = Field(default_factory=list)


# hash_text is now imported from core.hashing


def normalize_role(role: str | None) -> str:
    if not role:
        return "message"
    lowered = str(role).strip().lower()
    if lowered in {"assistant", "model"}:
        return "assistant"
    if lowered in {"user", "human"}:
        return "user"
    if lowered in {"system"}:
        return "system"
    return lowered


def _make_attachment_id(seed: str) -> str:
    return f"att-{hash_text(seed)[:12]}"


def attachment_from_meta(meta: object, message_id: str | None, index: int) -> ParsedAttachment | None:
    if not isinstance(meta, dict):
        return None
    attachment_id = meta.get("id") or meta.get("file_id") or meta.get("fileId") or meta.get("uuid")
    name = meta.get("name") or meta.get("filename")
    if not attachment_id:
        if not name:
            return None
        seed = f"{message_id or 'msg'}:{name}:{index}"
        attachment_id = _make_attachment_id(seed)
    size_raw = meta.get("size") or meta.get("size_bytes") or meta.get("sizeBytes")
    size_bytes = None
    if isinstance(size_raw, (int, str)):
        try:
            size_bytes = int(size_raw)
        except ValueError:
            size_bytes = None
    mime_type = meta.get("mimeType") or meta.get("mime_type") or meta.get("content_type")
    return ParsedAttachment(
        provider_attachment_id=str(attachment_id),
        message_provider_id=message_id,
        name=name,
        mime_type=mime_type if isinstance(mime_type, str) else None,
        size_bytes=size_bytes,
        path=None,
        provider_meta=meta,
    )


def extract_messages_from_list(items: list) -> list[ParsedMessage]:
    messages: list[ParsedMessage] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        # Support various schema conventions
        payload = item.get("message") if isinstance(item.get("message"), dict) else item

        role = normalize_role(
            payload.get("role")
            or item.get("role")
            or payload.get("sender")
            or item.get("sender")
            or payload.get("author")
            or item.get("author")
        )

        timestamp = (
            item.get("timestamp")
            or payload.get("timestamp")
            or payload.get("created_at")
            or item.get("created_at")
            or payload.get("create_time")
            or item.get("create_time")
        )

        text = None
        if "text" in payload and isinstance(payload["text"], str):
            text = payload["text"]
        elif "content" in payload:
            content = payload["content"]
            if isinstance(content, str):
                text = content
            elif isinstance(content, dict):
                parts = content.get("parts")
                if isinstance(parts, list):
                    text = "\n".join(str(part) for part in parts)
                elif "text" in content and isinstance(content["text"], str):
                    text = content["text"]
            elif isinstance(content, list):
                # Simple concatenation for list of strings or dicts
                parts = []
                for part in content:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict):
                        parts.append(part.get("text", ""))
                text = "\n".join(parts)

        if text:
            msg_id = str(payload.get("id") or payload.get("uuid") or item.get("uuid") or item.get("id") or f"msg-{idx}")
            messages.append(
                ParsedMessage(
                    provider_message_id=msg_id,
                    role=role,
                    text=text,
                    timestamp=str(timestamp) if timestamp is not None else None,
                    provider_meta={"raw": item},
                )
            )
    return messages
