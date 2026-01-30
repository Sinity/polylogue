from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field, field_validator

from polylogue.core.hashing import hash_text


class ParsedMessage(BaseModel):
    provider_message_id: str
    role: str
    text: str | None = None
    timestamp: str | None = None
    provider_meta: dict[str, object] | None = None


class ParsedAttachment(BaseModel):
    provider_attachment_id: str
    message_provider_id: str | None = None
    name: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    path: str | None = None
    provider_meta: dict[str, object] | None = None

    @field_validator("path")
    @classmethod
    def sanitize_path(cls, v: str | None) -> str | None:
        """Sanitize path to prevent traversal attacks and other security issues."""
        if v is None:
            return v

        original_v = v

        # Remove null bytes
        v = v.replace("\x00", "")

        # Remove control characters (ASCII < 32 and 127)
        v = "".join(c for c in v if ord(c) >= 32 and ord(c) != 127)

        # Detect threats:
        # 1. Traversal attempts (..)
        # 2. Symlinks in path (potential traversal bypass)
        has_traversal = ".." in original_v

        # Check for symlinks in the path by checking path components
        has_symlink = False
        try:
            p = Path(v)
            # Check each parent in the path to see if it's a symlink
            # This prevents traversal via symlinks
            for parent in [p] + list(p.parents):
                if parent.is_symlink():
                    has_symlink = True
                    break
        except Exception:
            # If we can't check, assume it's safe
            pass

        # If traversal or symlinks were detected, hash to prevent re-assembly
        if has_traversal or has_symlink:
            from polylogue.core.hashing import hash_text
            # Hash the original to prevent reconstruction
            original_hash = hash_text(original_v)[:12]
            v = f"_blocked_{original_hash}"
        else:
            # Normal path: clean up path components
            is_absolute = v.startswith("/")

            # Safe directories that can use absolute paths (for testing/temp files)
            # Check before path cleaning
            safe_dirs = ("/tmp/", "/var/tmp/")
            is_safe_absolute = is_absolute and any(original_v.startswith(safe_dir) for safe_dir in safe_dirs)

            try:
                parts = []
                for component in v.split("/"):
                    component = component.strip()
                    # Skip empty or special dot components
                    if component and component not in (".", ".."):
                        parts.append(component)

                if parts:
                    v = "/".join(parts)

                    # For absolute paths:
                    # - If in safe directory (like /tmp/), preserve absolute path
                    # - Otherwise, convert to relative for sandboxing
                    if is_absolute:
                        if is_safe_absolute and not v.startswith("/"):
                            v = "/" + v
                        elif not is_safe_absolute and v.startswith("/"):
                            v = v.lstrip("/")
            except Exception:
                pass

        return v if v else None

    @field_validator("name")
    @classmethod
    def sanitize_name(cls, v: str | None) -> str | None:
        """Sanitize filename to prevent control chars and invalid names."""
        if v is None:
            return v

        # Remove null bytes
        v = v.replace("\x00", "")

        # Remove control characters (ASCII < 32 and 127)
        v = "".join(c for c in v if ord(c) >= 32 and ord(c) != 127)

        # Reject dots-only names
        if v and v.strip(".") == "":
            # Return a default name instead of empty
            v = "file"

        return v if v else None


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
    if not lowered:  # Handle whitespace-only strings
        return "message"
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
    name = meta.get("name") or meta.get("filename") or meta.get("file_name")
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


def extract_messages_from_list(items: list[object]) -> list[ParsedMessage]:
    messages: list[ParsedMessage] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        # Support various schema conventions
        message_val = item.get("message")
        payload = message_val if isinstance(message_val, dict) else item

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
        text_val = payload.get("text")
        if text_val is not None and isinstance(text_val, str):
            text = text_val
        else:
            content = payload.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, dict):
                parts = content.get("parts")
                if isinstance(parts, list):
                    text = "\n".join(str(part) for part in parts)
                else:
                    text_dict_val = content.get("text")
                    if text_dict_val is not None and isinstance(text_dict_val, str):
                        text = text_dict_val
            elif isinstance(content, list):
                # Simple concatenation for list of strings or dicts
                parts = []
                for part in content:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict):
                        part_text = part.get("text")
                        parts.append(part_text if isinstance(part_text, str) else "")
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
