"""Typed content-block models used by rendering surfaces."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from polylogue.storage.store import ContentBlockRecord


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
    tool_input: Mapping[str, object] | None = None


def _mapping_with_string_keys(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    if not all(isinstance(key, str) for key in value):
        return None
    return cast(Mapping[str, object], value)


def _json_mapping(value: str | None) -> Mapping[str, object] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return _mapping_with_string_keys(parsed)


def _coerce_tool_input(value: object) -> Mapping[str, object] | None:
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
        text=cast(str | None, block.get("thinking") or block.get("text") or block.get("code") or block.get("content")),
        language=cast(str | None, block.get("language")),
        url=cast(str | None, block.get("url") or source.get("url")),
        name=str(name) if isinstance(name, str) else None,
        mime_type=cast(str | None, block.get("media_type") or block.get("mime_type")),
        tool_name=cast(str | None, block.get("name")) if block_type == "tool_use" else None,
        tool_id=cast(str | None, block.get("id") or block.get("tool_use_id") or block.get("tool_id")),
        tool_input=_coerce_tool_input(block.get("input")),
    )


def _coerce_record_block(block: ContentBlockRecord) -> RenderableBlock:
    metadata = _json_mapping(block.metadata)
    return RenderableBlock(
        type=block.type.value,
        text=block.text,
        language=cast(str | None, metadata.get("language")) if metadata is not None else None,
        url=cast(str | None, metadata.get("url")) if metadata is not None else None,
        name=cast(str | None, metadata.get("name")) if metadata is not None else None,
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
