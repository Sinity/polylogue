"""Content-block helpers for Gemini/Drive parsing."""

from __future__ import annotations

from typing import Any

from polylogue.types import ContentBlockType

from .base import ParsedContentBlock


def viewport_block_payload(block: Any) -> dict[str, object] | None:
    raw_type = block.type.value if hasattr(block.type, "value") else str(block.type)
    block_type = {
        "file": "document",
        "audio": "document",
        "video": "document",
        "unknown": "text",
        "system": "text",
        "error": "text",
    }.get(raw_type, raw_type)
    if block_type not in {"text", "thinking", "tool_use", "tool_result", "image", "code", "document"}:
        return None
    payload: dict[str, object] = {"type": block_type}
    if block.text is not None:
        payload["text"] = block.text
    if block.language:
        payload["language"] = block.language
    if block.mime_type:
        payload["media_type"] = block.mime_type
    metadata: dict[str, object] = {}
    if isinstance(block.raw, dict) and block.raw:
        metadata.update(block.raw)
    if getattr(block, "url", None):
        metadata["url"] = block.url
    if metadata:
        payload["metadata"] = metadata
    return payload


def parsed_content_blocks_from_meta(blocks: object) -> list[ParsedContentBlock]:
    if not isinstance(blocks, list):
        return []
    parsed: list[ParsedContentBlock] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if not isinstance(block_type, str) or not block_type:
            continue
        meta_value = block.get("metadata")
        metadata: dict[str, object] | None = dict(meta_value) if isinstance(meta_value, dict) else None
        language = block.get("language")
        if isinstance(language, str) and language:
            metadata = dict(metadata or {})
            metadata.setdefault("language", language)
        parsed.append(
            ParsedContentBlock(
                type=ContentBlockType.from_string(block_type),
                text=block.get("text") if isinstance(block.get("text"), str) else None,
                media_type=block.get("media_type") if isinstance(block.get("media_type"), str) else None,
                metadata=metadata,
            )
        )
    return parsed


__all__ = ["parsed_content_blocks_from_meta", "viewport_block_payload"]
