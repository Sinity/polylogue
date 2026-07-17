"""Content-block helpers for Gemini/Drive parsing."""

from __future__ import annotations

from polylogue.archive.viewport.viewports import ContentBlock
from polylogue.core.enums import BlockType
from polylogue.core.json import JSONDocument, json_document, json_document_list

from .base import ParsedContentBlock

_SUCCESS_OUTCOMES = frozenset({"ok", "success", "succeeded", "completed", "outcome_ok"})
_ERROR_OUTCOME_MARKERS = ("error", "fail", "timeout", "deadline", "cancel", "blocked")


def _optional_int(payload: JSONDocument, *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value))
            except ValueError:
                continue
    return None


def _tool_result_error(metadata: JSONDocument, exit_code: int | None) -> bool | None:
    for key in ("is_error", "isError", "error"):
        value = metadata.get(key)
        if isinstance(value, bool):
            return value
        if key == "error" and isinstance(value, str) and value:
            return True
    outcome = metadata.get("outcome") or metadata.get("status")
    if isinstance(outcome, str) and outcome:
        normalized = outcome.strip().lower()
        if normalized in _SUCCESS_OUTCOMES:
            return False
        if any(marker in normalized for marker in _ERROR_OUTCOME_MARKERS):
            return True
    if exit_code is not None:
        return exit_code != 0
    return None


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


def parsed_blocks_from_meta(blocks: object) -> list[ParsedContentBlock]:
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
        parsed_type = BlockType.from_string(block_type)
        exit_code = _optional_int(metadata, "exitCode", "exit_code") if parsed_type is BlockType.TOOL_RESULT else None
        parsed.append(
            ParsedContentBlock(
                type=parsed_type,
                text=text,
                media_type=media_type,
                metadata=metadata_out or None,
                is_error=(_tool_result_error(metadata, exit_code) if parsed_type is BlockType.TOOL_RESULT else None),
                exit_code=exit_code,
            )
        )
    return parsed


__all__ = ["parsed_blocks_from_meta", "viewport_block_payload"]
