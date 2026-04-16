"""Shared parser extraction helpers."""

from __future__ import annotations

from polylogue.lib.hashing import hash_text
from polylogue.lib.roles import Role

from .base_models import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
)


def content_blocks_from_segments(content: object) -> list[ParsedContentBlock]:
    """Convert raw API content (str, list, dict) to ParsedContentBlock list."""
    if isinstance(content, str):
        return [ParsedContentBlock(type="text", text=content)] if content else []
    if not isinstance(content, list):
        return []
    blocks: list[ParsedContentBlock] = []
    for seg in content:
        if isinstance(seg, str):
            if seg:
                blocks.append(ParsedContentBlock(type="text", text=seg))
            continue
        if not isinstance(seg, dict):
            continue
        seg_type = seg.get("type", "text")
        if seg_type == "thinking":
            text = seg.get("thinking") or seg.get("text") or ""
            if text:
                blocks.append(ParsedContentBlock(type="thinking", text=text))
        elif seg_type == "tool_use":
            tool_name = seg.get("name")
            tool_id = seg.get("id")
            tool_input = seg.get("input") if isinstance(seg.get("input"), dict) else None
            if tool_name or tool_id or tool_input:
                blocks.append(
                    ParsedContentBlock(
                        type="tool_use",
                        tool_name=tool_name,
                        tool_id=tool_id,
                        tool_input=tool_input,
                    )
                )
        elif seg_type == "tool_result":
            result_content = seg.get("content")
            result_text = None
            if isinstance(result_content, str):
                result_text = result_content
            elif isinstance(result_content, list):
                text_parts = [
                    block.get("text", "")
                    for block in result_content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                result_text = "\n".join(part for part in text_parts if part) or None
            blocks.append(
                ParsedContentBlock(
                    type="tool_result",
                    tool_id=seg.get("tool_use_id"),
                    text=result_text,
                )
            )
        elif seg_type in ("image", "document"):
            blocks.append(
                ParsedContentBlock(
                    type=seg_type,
                    media_type=seg.get("media_type"),
                    metadata={k: v for k, v in seg.items() if k not in ("type", "media_type")},
                )
            )
        elif seg_type == "code":
            text = seg.get("text") or seg.get("code") or ""
            if text:
                metadata = None
                language = seg.get("language")
                if isinstance(language, str) and language:
                    metadata = {"language": language}
                blocks.append(ParsedContentBlock(type="code", text=str(text), metadata=metadata))
        else:
            text = seg.get("text") or seg.get("content") or ""
            if text:
                blocks.append(ParsedContentBlock(type="text", text=str(text)))
    return blocks


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

        message_val = item.get("message")
        payload = message_val if isinstance(message_val, dict) else item

        role = Role.normalize(
            str(
                payload.get("role")
                or item.get("role")
                or payload.get("sender")
                or item.get("sender")
                or payload.get("author")
                or item.get("author")
                or "unknown"
            )
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
        content_blocks: list[ParsedContentBlock] = []
        text_val = payload.get("text")
        if text_val is not None and isinstance(text_val, str):
            text = text_val
            if text:
                content_blocks = [ParsedContentBlock(type="text", text=text)]
        else:
            content = payload.get("content")
            if isinstance(content, str):
                text = content
                if text:
                    content_blocks = [ParsedContentBlock(type="text", text=text)]
            elif isinstance(content, dict):
                parts = content.get("parts")
                if isinstance(parts, list):
                    texts: list[str] = []
                    for part in parts:
                        if isinstance(part, str) and part:
                            texts.append(part)
                            content_blocks.append(ParsedContentBlock(type="text", text=part))
                        elif isinstance(part, dict):
                            part_text = part.get("text")
                            if isinstance(part_text, str) and part_text:
                                texts.append(part_text)
                                content_blocks.append(ParsedContentBlock(type="text", text=part_text))
                    text = "\n".join(texts) or None
                else:
                    text_dict_val = content.get("text")
                    if text_dict_val is not None and isinstance(text_dict_val, str):
                        text = text_dict_val
                        if text:
                            content_blocks = [ParsedContentBlock(type="text", text=text)]
            elif isinstance(content, list):
                content_blocks = content_blocks_from_segments(content)
                text = "\n".join(block.text for block in content_blocks if block.text) or None

        if text:
            msg_id = str(payload.get("id") or payload.get("uuid") or item.get("uuid") or item.get("id") or f"msg-{idx}")
            messages.append(
                ParsedMessage(
                    provider_message_id=msg_id,
                    role=role,
                    text=text,
                    timestamp=str(timestamp) if timestamp is not None else None,
                    content_blocks=content_blocks,
                )
            )
    return messages
