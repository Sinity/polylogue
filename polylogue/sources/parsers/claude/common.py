"""Shared Claude parser helpers."""

from __future__ import annotations

import json

from polylogue.lib.roles import Role
from polylogue.types import ContentBlockType

from ..base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    attachment_from_meta,
    content_blocks_from_segments,
)


def reclassify_tool_result_envelope(role: Role, content_blocks: list[ParsedContentBlock]) -> Role:
    """Reclassify a ``role: user`` envelope whose content is all ``tool_result`` to ``Role.TOOL``.

    The Anthropic API protocol requires ``tool_result`` blocks to be carried by
    ``role: user`` messages — the assistant emits ``tool_use`` blocks and the
    runtime replies with corresponding ``tool_result`` blocks under the
    protocol-mandated ``user`` role. Polylogue's outer-envelope role
    normalization classifies these as ``Role.USER``, polluting
    ``--message-role user`` filters with non-typed content.

    See `#428 <https://github.com/Sinity/polylogue/issues/428>`_.
    """
    if role is not Role.USER:
        return role
    if not content_blocks:
        return role
    if all(block.type == ContentBlockType.TOOL_RESULT for block in content_blocks):
        return Role.TOOL
    return role


def extract_text_from_segments(segments: list[object]) -> str | None:
    lines: list[str] = []
    for segment in segments:
        if isinstance(segment, str):
            if segment:
                lines.append(segment)
            continue
        if not isinstance(segment, dict):
            continue
        seg_type = segment.get("type")
        if seg_type in {"tool_use", "tool_result"}:
            lines.append(json.dumps(segment, sort_keys=True))
            continue
        if seg_type == "thinking":
            seg_thinking = segment.get("thinking")
            if isinstance(seg_thinking, str):
                lines.append(f"<thinking>{seg_thinking}</thinking>")
                continue
        seg_text = segment.get("text")
        if isinstance(seg_text, str):
            lines.append(seg_text)
            continue
        seg_content = segment.get("content")
        if isinstance(seg_content, str):
            lines.append(seg_content)
            continue
    combined = "\n".join(line for line in lines if line)
    return combined or None


def normalize_timestamp(ts: int | float | str | None) -> str | None:
    if ts is None:
        return None
    try:
        val = float(ts)
        if val > 1e11:
            val = val / 1000.0
        return str(val)
    except (ValueError, TypeError):
        pass
    if isinstance(ts, str):
        from polylogue.core.timestamps import parse_timestamp

        dt = parse_timestamp(ts)
        if dt is not None:
            return str(dt.timestamp())
    return None


def extract_messages_from_chat_messages(
    chat_messages: list[object],
) -> tuple[list[ParsedMessage], list[ParsedAttachment]]:
    messages: list[ParsedMessage] = []
    attachments: list[ParsedAttachment] = []
    for idx, item in enumerate(chat_messages, start=1):
        if not isinstance(item, dict):
            continue
        message_id = str(item.get("uuid") or item.get("id") or item.get("message_id") or f"msg-{idx}")
        raw_role = item.get("sender") or item.get("role")
        if not raw_role or not isinstance(raw_role, str):
            continue
        role = Role.normalize(str(raw_role))

        raw_ts = item.get("created_at") or item.get("create_time") or item.get("timestamp")
        timestamp = normalize_timestamp(raw_ts)

        text = item.get("text") if isinstance(item.get("text"), str) else None
        if text is None:
            content = item.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = extract_text_from_segments(content)
            elif isinstance(content, dict):
                text = content.get("text") if isinstance(content.get("text"), str) else None
                if text is None and isinstance(content.get("parts"), list):
                    text = "\n".join(str(part) for part in content["parts"] if part)
        raw_content = item.get("content")
        content_blocks = content_blocks_from_segments(raw_content) if isinstance(raw_content, list) else []
        if not content_blocks and text:
            content_blocks = [ParsedContentBlock(type=ContentBlockType.TEXT, text=text)]

        role = reclassify_tool_result_envelope(role, content_blocks)

        if text:
            messages.append(
                ParsedMessage(
                    provider_message_id=message_id,
                    role=role,
                    text=text,
                    timestamp=timestamp,
                    content_blocks=content_blocks,
                )
            )
        for att_idx, meta in enumerate(item.get("attachments") or item.get("files") or [], start=1):
            attachment = attachment_from_meta(meta, message_id, att_idx)
            if attachment:
                attachments.append(attachment)
    return messages, attachments


def extract_message_text(message_content: object) -> str | None:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        return extract_text_from_segments(message_content)
    if isinstance(message_content, dict):
        text = message_content.get("text")
        if isinstance(text, str):
            return text
        parts = message_content.get("parts")
        if isinstance(parts, list):
            return "\n".join(str(p) for p in parts if p)
    return None


__all__ = [
    "extract_message_text",
    "extract_messages_from_chat_messages",
    "extract_text_from_segments",
    "normalize_timestamp",
]
