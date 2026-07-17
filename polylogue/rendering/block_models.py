"""Typed content-block models used by rendering surfaces."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from polylogue.core.json import JSONDocument, json_document


@dataclass(frozen=True, slots=True)
class RenderableBlock:
    """Normalized block payload shared across markdown, HTML, and site output.

    Rendering adapters retain structural outcome and raw-input evidence instead
    of reducing blocks to display text. ``tool_result_is_error=None`` and
    ``tool_result_exit_code=None`` mean that the source did not establish an
    outcome; consumers must not promote that state to success.
    """

    type: str
    text: str | None = None
    language: str | None = None
    url: str | None = None
    name: str | None = None
    mime_type: str | None = None
    block_id: str | None = None
    tool_name: str | None = None
    tool_id: str | None = None
    tool_input: JSONDocument | None = None
    tool_input_raw: str | None = None
    metadata: JSONDocument | None = None
    semantic_type: str | None = None
    tool_result_is_error: bool | None = None
    tool_result_exit_code: int | None = None
    text_encoding_replacements: int = 0


def _mapping_with_string_keys(value: object) -> JSONDocument | None:
    if not isinstance(value, Mapping):
        return None
    normalized: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        normalized[key] = item
    return json_document(normalized)


def _object_mapping_with_string_keys(value: object) -> dict[str, object] | None:
    """Retain non-JSON test/runtime values while validating mapping keys.

    Raw tool output can arrive as bytes before the storage boundary decodes it.
    Requiring the *entire* block mapping to satisfy ``JSONDocument`` would drop
    that block and its structural outcome rather than rendering a replacement-
    marked preview.
    """

    if not isinstance(value, Mapping):
        return None
    normalized: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        normalized[key] = item
    return normalized


def _block_attribute_mapping(value: object) -> dict[str, object] | None:
    """Project a block-shaped dataclass/model without importing its layer."""

    block_type = getattr(value, "type", getattr(value, "block_type", None))
    if block_type is None:
        return None
    result: dict[str, object] = {"type": getattr(block_type, "value", block_type)}
    attribute_names = (
        "text",
        "language",
        "url",
        "name",
        "mime_type",
        "block_id",
        "tool_name",
        "tool_id",
        "tool_input",
        "metadata",
        "semantic_type",
        "tool_result_is_error",
        "tool_result_exit_code",
    )
    for name in attribute_names:
        candidate = getattr(value, name, None)
        if candidate is not None:
            result[name] = getattr(candidate, "value", candidate)
    return result


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_text(value: object) -> tuple[str | None, int]:
    if isinstance(value, str):
        return value, value.count("\ufffd")
    if isinstance(value, (bytes, bytearray)):
        decoded = bytes(value).decode("utf-8", errors="replace")
        return decoded, decoded.count("\ufffd")
    return None, 0


def _first_text(*values: object) -> tuple[str | None, int]:
    for value in values:
        text, replacements = _optional_text(value)
        if text is not None:
            return text, replacements
    return None, 0


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


def _coerce_tool_input(value: object) -> tuple[JSONDocument | None, str | None]:
    mapping = _mapping_with_string_keys(value)
    if mapping is not None:
        return mapping, None
    if isinstance(value, str):
        return _json_mapping(value), value
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value).decode("utf-8", errors="replace")
        return _json_mapping(raw), raw
    return None, None


def _coerce_optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    return None


def _coerce_optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _coerce_mapping_block(block: Mapping[str, object]) -> RenderableBlock:
    block_type = str(block.get("type", "text"))
    source = _mapping_with_string_keys(block.get("source")) or {}
    raw_metadata = block.get("metadata")
    metadata = _mapping_with_string_keys(raw_metadata)
    if metadata is None and isinstance(raw_metadata, str):
        metadata = _json_mapping(raw_metadata)
    name = block.get("name") or block.get("title")
    if name is None and metadata is not None:
        name = metadata.get("name")
    text, replacements = _first_text(
        block.get("thinking"),
        block.get("text"),
        block.get("code"),
        block.get("content"),
    )
    raw_tool_input = block.get("tool_input", block.get("input"))
    tool_input, tool_input_raw = _coerce_tool_input(raw_tool_input)
    tool_name = _first_string(block.get("tool_name"), block.get("name")) if block_type == "tool_use" else None
    outcome_is_error = block.get("tool_result_is_error", block.get("is_error"))
    outcome_exit_code = block.get("tool_result_exit_code", block.get("exit_code"))
    if metadata is not None:
        if outcome_is_error is None:
            outcome_is_error = metadata.get("is_error")
        if outcome_exit_code is None:
            outcome_exit_code = metadata.get("exit_code")
    return RenderableBlock(
        type=block_type,
        text=text,
        language=_first_string(block.get("language"), metadata.get("language") if metadata is not None else None),
        url=_first_string(
            block.get("url"),
            source.get("url"),
            metadata.get("url") if metadata is not None else None,
        ),
        name=str(name) if isinstance(name, str) else None,
        mime_type=_first_string(
            block.get("media_type"),
            block.get("mime_type"),
            metadata.get("media_type") if metadata is not None else None,
            metadata.get("mime_type") if metadata is not None else None,
        ),
        block_id=_first_string(block.get("block_id"), block.get("id")),
        tool_name=tool_name,
        tool_id=_first_string(block.get("tool_use_id"), block.get("tool_id"), block.get("id")),
        tool_input=tool_input,
        tool_input_raw=tool_input_raw if tool_input is None else None,
        metadata=metadata,
        semantic_type=_optional_string(block.get("semantic_type")),
        tool_result_is_error=_coerce_optional_bool(outcome_is_error),
        tool_result_exit_code=_coerce_optional_int(outcome_exit_code),
        text_encoding_replacements=replacements,
    )


def coerce_renderable_block(value: object) -> RenderableBlock | None:
    """Normalize arbitrary block payloads into the rendering contract."""
    if isinstance(value, RenderableBlock):
        return value
    mapping = _object_mapping_with_string_keys(value)
    if mapping is None:
        mapping = _block_attribute_mapping(value)
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


__all__ = ["coerce_renderable_block", "coerce_renderable_blocks", "RenderableBlock"]
