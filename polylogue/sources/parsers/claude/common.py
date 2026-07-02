"""Shared Claude parser helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType

from ..base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    attachment_from_meta,
    content_blocks_from_segments,
)


def _optional_non_negative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if value >= 0 else None
    if isinstance(value, str):
        try:
            parsed = int(float(value))
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


def _metadata_mapping(item: Mapping[str, object]) -> Mapping[str, object]:
    metadata = item.get("metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _first_string_field(item: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    metadata = _metadata_mapping(item)
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _message_model_name(item: Mapping[str, object]) -> str | None:
    return _first_string_field(item, "model", "model_name", "modelName", "model_slug")


def _message_model_effort(item: Mapping[str, object]) -> str | None:
    return _first_string_field(item, "effort", "model_effort", "modelEffort")


def _message_duration_ms(item: Mapping[str, object]) -> int | None:
    for key in ("durationMs", "duration_ms", "elapsed_ms"):
        value = item.get(key)
        if value is not None:
            return _optional_non_negative_int(value)
    metadata = _metadata_mapping(item)
    for key in ("durationMs", "duration_ms", "elapsed_ms"):
        value = metadata.get(key)
        if value is not None:
            return _optional_non_negative_int(value)
    return None


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
    if all(block.type == BlockType.TOOL_RESULT for block in content_blocks):
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
    from polylogue.core.timestamps import parse_timestamp

    try:
        val = float(ts)
        if val > 1e11:
            val = val / 1000.0
        dt = parse_timestamp(val)
        return dt.isoformat() if dt is not None else None
    except (ValueError, TypeError):
        pass
    if isinstance(ts, str):
        dt = parse_timestamp(ts)
        if dt is not None:
            return dt.isoformat()
    return None


def extract_messages_from_chat_messages(
    chat_messages: list[object],
) -> tuple[list[ParsedMessage], list[ParsedAttachment]]:
    messages: list[ParsedMessage] = []
    attachments: list[ParsedAttachment] = []
    message_position = 0
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
        if not text:
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
            content_blocks = [ParsedContentBlock(type=BlockType.TEXT, text=text)]

        role = reclassify_tool_result_envelope(role, content_blocks)

        if text:
            messages.append(
                ParsedMessage(
                    provider_message_id=message_id,
                    role=role,
                    text=text,
                    timestamp=timestamp,
                    blocks=content_blocks,
                    position=message_position,
                    variant_index=0,
                    is_active_path=True,
                    model_name=_message_model_name(item),
                    model_effort=_message_model_effort(item),
                    duration_ms=_message_duration_ms(item),
                )
            )
            message_position += 1
        raw_attachments: list[object] = []
        attachments_value = item.get("attachments")
        files_value = item.get("files")
        if isinstance(attachments_value, list):
            raw_attachments.extend(attachments_value)
        if isinstance(files_value, list):
            raw_attachments.extend(files_value)
        for att_idx, meta in enumerate(raw_attachments, start=1):
            attachment = attachment_from_meta(meta, message_id, att_idx)
            if attachment:
                attachments.append(attachment)
    if messages:
        active_leaf_message_provider_id = messages[-1].provider_message_id
        messages = [
            message.model_copy(
                update={"is_active_leaf": message.provider_message_id == active_leaf_message_provider_id}
            )
            for message in messages
        ]
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
