"""Content-block helpers for Gemini/Drive parsing."""

from __future__ import annotations

from polylogue.archive.viewport.viewports import ContentBlock
from polylogue.core.json import JSONDocument, json_document, json_document_list
from polylogue.types import ContentBlockType

from .base import ParsedContentBlock


def viewport_block_payload(block: ContentBlock) -> JSONDocument | None:
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
    payload: JSONDocument = {"type": block_type}
    if block.text is not None:
        payload["text"] = block.text
    if block.language:
        payload["language"] = block.language
    if block.mime_type:
        payload["media_type"] = block.mime_type
    metadata: JSONDocument = {}
    if isinstance(block.raw, dict) and block.raw:
        metadata.update(json_document(block.raw))
    if getattr(block, "url", None):
        metadata["url"] = block.url
    if metadata:
        payload["metadata"] = metadata
    return payload


def parsed_content_blocks_from_meta(blocks: object) -> list[ParsedContentBlock]:
    parsed: list[ParsedContentBlock] = []
    for block in json_document_list(blocks):
        block_type = block.get("type")
        if not isinstance(block_type, str) or not block_type:
            continue
        metadata = json_document(block.get("metadata"))
        block_text = block.get("text")
        text = block_text if isinstance(block_text, str) else None
        raw_media_type = block.get("media_type")
        media_type = raw_media_type if isinstance(raw_media_type, str) else None
        language = block.get("language")
        metadata_out: dict[str, object] = {}
        for key, value in metadata.items():
            metadata_out[key] = value
        if isinstance(language, str) and language:
            metadata_out.setdefault("language", language)
        parsed.append(
            ParsedContentBlock(
                type=ContentBlockType.from_string(block_type),
                text=text,
                media_type=media_type,
                metadata=metadata_out or None,
            )
        )
    return parsed


__all__ = ["parsed_content_blocks_from_meta", "viewport_block_payload"]
