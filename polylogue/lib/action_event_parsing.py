"""Tool-call parsing helpers for action events."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from polylogue.lib.json import JSONDocument, json_document
from polylogue.lib.viewports import ToolCall, ToolCategory, classify_tool
from polylogue.types import Provider


def _clean_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    return candidate or None


def _extract_first_string(mapping: Mapping[str, object], fields: tuple[str, ...]) -> str | None:
    for field in fields:
        value = _clean_str(mapping.get(field))
        if value is not None:
            return value
    return None


def _normalized_mapping(value: object) -> JSONDocument:
    if isinstance(value, Mapping):
        return json_document(dict(value))
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, Mapping):
            return json_document(dict(parsed))
    return {}


def _tool_category_from_semantic(value: object) -> ToolCategory | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return ToolCategory(value)
    except ValueError:
        return None


def build_tool_calls_from_content_blocks(
    *,
    provider: Provider | str | None,
    content_blocks: Sequence[Mapping[str, object]],
) -> tuple[ToolCall, ...]:
    """Normalize canonical ToolCall viewports from content blocks."""
    tool_result_outputs: dict[str, str] = {}
    tool_use_blocks: list[Mapping[str, object]] = []
    for block in content_blocks:
        block_type = str(block.get("type"))
        if block_type == "tool_result":
            tool_id = block.get("tool_id")
            text = block.get("text")
            if isinstance(tool_id, str) and tool_id and isinstance(text, str) and text:
                tool_result_outputs.setdefault(tool_id, text)
            continue
        if block_type != "tool_use":
            continue
        tool_use_blocks.append(block)

    if not tool_use_blocks:
        return ()

    normalized_provider = (
        provider if isinstance(provider, Provider) else Provider.from_string(provider) if provider is not None else None
    )
    calls: list[ToolCall] = []
    for block in tool_use_blocks:
        name = block.get("tool_name")
        if not isinstance(name, str) or not name:
            continue
        tool_id = block.get("tool_id")
        normalized_input = _normalized_mapping(block.get("tool_input"))
        semantic_category = _tool_category_from_semantic(block.get("semantic_type"))
        classified_category = classify_tool(name, normalized_input)
        if semantic_category is None or semantic_category is ToolCategory.OTHER:
            category = classified_category
        else:
            category = semantic_category
        raw = {
            "block_id": block.get("block_id"),
            "block_index": block.get("block_index"),
            "message_id": block.get("message_id"),
            "conversation_id": block.get("conversation_id"),
            "type": block.get("type"),
            "tool_name": name,
            "tool_id": tool_id,
            "tool_input": normalized_input,
            "media_type": block.get("media_type"),
            "metadata": _normalized_mapping(block.get("metadata")),
            "semantic_type": block.get("semantic_type"),
            "text": block.get("text"),
        }
        calls.append(
            ToolCall(
                name=name,
                id=tool_id if isinstance(tool_id, str) and tool_id else None,
                input=normalized_input,
                output=tool_result_outputs.get(tool_id) if isinstance(tool_id, str) else None,
                category=category,
                provider=normalized_provider,
                raw=raw,
            )
        )
    return tuple(calls)


__all__ = [
    "_clean_str",
    "_extract_first_string",
    "build_tool_calls_from_content_blocks",
]
