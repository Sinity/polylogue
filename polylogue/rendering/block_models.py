"""Typed content-block models used by rendering surfaces."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from polylogue.lib.json import JSONDocument, json_document
from polylogue.storage.runtime import ContentBlockRecord


@dataclass(frozen=True, slots=True)
class RenderableBlock:
    """Normalized block payload shared across markdown, HTML, and site output."""

    type: str
    text: str | None = None
    language: str | None = None
    url: str | None = None
    name: str | None = None
    mime_type: str | None = None
    tool_name: str | None = None
    tool_id: str | None = None
    tool_input: JSONDocument | None = None


def _mapping_with_string_keys(value: object) -> JSONDocument | None:
    if not isinstance(value, Mapping):
        return None
    normalized: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        normalized[key] = item
    return json_document(normalized)


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _first_string(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str):
            return value
    return None


def _json_mapping(value: str | None) -> JSONDocument | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return _mapping_with_string_keys(parsed)


def _coerce_tool_input(value: object) -> JSONDocument | None:
    mapping = _mapping_with_string_keys(value)
    if mapping is not None:
        return mapping
    if isinstance(value, str):
        return _json_mapping(value)
    return None


def _coerce_mapping_block(block: Mapping[str, object]) -> RenderableBlock:
    block_type = str(block.get("type", "text"))
    source = _mapping_with_string_keys(block.get("source")) or {}
    name = block.get("name") or block.get("title")
    return RenderableBlock(
        type=block_type,
        text=_first_string(block.get("thinking"), block.get("text"), block.get("code"), block.get("content")),
        language=_optional_string(block.get("language")),
        url=_first_string(block.get("url"), source.get("url")),
        name=str(name) if isinstance(name, str) else None,
        mime_type=_first_string(block.get("media_type"), block.get("mime_type")),
        tool_name=_optional_string(block.get("name")) if block_type == "tool_use" else None,
        tool_id=_first_string(block.get("id"), block.get("tool_use_id"), block.get("tool_id")),
        tool_input=_coerce_tool_input(block.get("input")),
    )


def _coerce_record_block(block: ContentBlockRecord) -> RenderableBlock:
    metadata = _json_mapping(block.metadata)
    return RenderableBlock(
        type=block.type.value,
        text=block.text,
        language=_optional_string(metadata.get("language")) if metadata is not None else None,
        url=_optional_string(metadata.get("url")) if metadata is not None else None,
        name=_optional_string(metadata.get("name")) if metadata is not None else None,
        mime_type=block.media_type,
        tool_name=block.tool_name,
        tool_id=block.tool_id,
        tool_input=_json_mapping(block.tool_input),
    )


def coerce_renderable_block(value: object) -> RenderableBlock | None:
    """Normalize arbitrary block payloads into the rendering contract."""
    if isinstance(value, RenderableBlock):
        return value
    if isinstance(value, ContentBlockRecord):
        return _coerce_record_block(value)
    mapping = _mapping_with_string_keys(value)
    if mapping is not None:
        return _coerce_mapping_block(mapping)
    return None


def coerce_renderable_blocks(values: Sequence[object] | None) -> tuple[RenderableBlock, ...]:
    """Normalize a sequence of heterogeneous block payloads."""
    if values is None:
        return ()
    blocks: list[RenderableBlock] = []
    for value in values:
        block = coerce_renderable_block(value)
        if block is not None:
            blocks.append(block)
    return tuple(blocks)


__all__ = [
    "coerce_renderable_block",
    "coerce_renderable_blocks",
    "RenderableBlock",
]
