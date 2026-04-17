"""Schema-driven message extraction using semantic role annotations.

This is the default extraction path for all providers. It resolves message
fields (role, body, timestamp, content blocks) using the schema's
``x-polylogue-semantic-role`` annotations, filtered through the pinning
system (only pinned annotations drive extraction; unpinned ones are
informational).

Provider-specific adapters in ``unified_adapters.py`` remain as override
hooks: if a provider has a registered adapter, it takes priority. Schema
extraction is the fallback when no adapter exists or the adapter raises.
"""

from __future__ import annotations

import logging
from typing import Any

from polylogue.lib.provider_semantics import (
    extract_content_blocks,
    extract_reasoning_traces,
    extract_tool_calls,
)
from polylogue.lib.roles import Role
from polylogue.lib.timestamps import parse_timestamp
from polylogue.lib.viewports import CostInfo
from polylogue.schemas.pinning import load_pins, resolve_pinned_paths
from polylogue.schemas.unified_models import HarmonizedMessage, extract_token_usage
from polylogue.types import Provider

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Path resolution helpers
# -------------------------------------------------------------------

_WELL_KNOWN_ROLE_NAMES = frozenset(
    {
        "role",
        "type",
        "sender",
        "author",
    }
)
_WELL_KNOWN_BODY_NAMES = frozenset(
    {
        "content",
        "text",
        "body",
        "message",
        "parts",
    }
)
_WELL_KNOWN_TIMESTAMP_NAMES = frozenset(
    {
        "timestamp",
        "created_at",
        "create_time",
        "time",
        "date",
    }
)
_WELL_KNOWN_ID_NAMES = frozenset(
    {
        "id",
        "uuid",
        "messageId",
        "message_id",
    }
)


def _resolve_json_path(raw: dict[str, Any], path: str) -> Any:
    """Resolve a dotted JSON path against a raw dict.

    Supports:
    - ``.field`` — dict key lookup
    - ``[]`` — iterate array
    - ``.*`` — iterate dict values (additionalProperties)
    - ``.anyOf[N]`` — ignored (schema structural, not data structural)

    Returns the first non-None value found, or None.
    """
    parts = _split_path(path)
    return _walk(raw, parts)


def _split_path(path: str) -> list[str]:
    """Split a dotted annotation path into segments."""
    # Remove leading dot/dollar
    path = path.lstrip("$").lstrip(".")
    if not path:
        return []

    segments: list[str] = []
    current = ""
    for char in path:
        if char == "." and not current.endswith("["):
            if current:
                segments.append(current)
            current = ""
        else:
            current += char
    if current:
        segments.append(current)
    return segments


def _walk(obj: Any, parts: list[str]) -> Any:
    """Walk a JSON object following path segments."""
    if not parts:
        return obj
    if obj is None:
        return None

    head, rest = parts[0], parts[1:]

    # Skip schema-only segments (anyOf[N], oneOf[N], allOf[N])
    if head.startswith("anyOf[") or head.startswith("oneOf[") or head.startswith("allOf["):
        return _walk(obj, rest)

    # Array iteration
    if head == "[]":
        if isinstance(obj, list):
            for item in obj:
                result = _walk(item, rest)
                if result is not None:
                    return result
        return None

    # AdditionalProperties wildcard
    if head == "*":
        if isinstance(obj, dict):
            for value in obj.values():
                result = _walk(value, rest)
                if result is not None:
                    return result
        return None

    # Regular dict key
    if isinstance(obj, dict):
        return _walk(obj.get(head), rest)

    return None


def _find_by_well_known_names(
    raw: dict[str, Any],
    names: frozenset[str],
) -> Any:
    """Fallback: search top-level and one level deep for well-known field names."""
    for key in raw:
        if key.lower() in names:
            return raw[key]
    # One level deep
    for _key, value in raw.items():
        if isinstance(value, dict):
            for subkey in value:
                if subkey.lower() in names:
                    return value[subkey]
    return None


# -------------------------------------------------------------------
# Content extraction from resolved body
# -------------------------------------------------------------------


def _extract_text_from_body(body: Any) -> str:
    """Extract displayable text from whatever the body field contains."""
    if isinstance(body, str):
        return body
    if isinstance(body, list):
        # List of content blocks or list of strings
        parts: list[str] = []
        for item in body:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str) and text:
                    parts.append(text)
            # else skip
        return "\n".join(parts) if parts else ""
    if isinstance(body, dict):
        # Nested content structure
        text = body.get("text") or body.get("content")
        if isinstance(text, str):
            return text
        parts_field = body.get("parts")
        if isinstance(parts_field, list):
            return _extract_text_from_body(parts_field)
    return str(body) if body else ""


def _extract_content_block_list(body: Any) -> list[dict[str, Any]]:
    """Extract raw content block dicts from the body field, if structured."""
    if isinstance(body, list):
        blocks = [item for item in body if isinstance(item, dict) and "type" in item]
        if blocks:
            return blocks
    if isinstance(body, dict):
        # The body itself might contain a list of blocks
        for key in ("content", "parts", "blocks"):
            nested = body.get(key)
            if isinstance(nested, list):
                blocks = [item for item in nested if isinstance(item, dict) and "type" in item]
                if blocks:
                    return blocks
    return []


# -------------------------------------------------------------------
# Main extraction entry point
# -------------------------------------------------------------------


def extract_message_from_schema(
    raw: dict[str, Any],
    *,
    schema: dict[str, Any],
    provider: Provider,
) -> HarmonizedMessage:
    """Extract a HarmonizedMessage using schema semantic annotations.

    Only pinned annotations are used for field resolution. If no pinned
    annotation exists for a role, falls back to well-known field name
    heuristics.

    Args:
        raw: The raw JSON record from the provider export.
        schema: The annotated JSON Schema for this provider/element.
        provider: The canonical provider identity.

    Returns:
        A HarmonizedMessage with as much structure preserved as possible.
    """
    pins = load_pins(provider)
    paths = resolve_pinned_paths(schema, pins)

    # Resolve role
    role_raw = None
    role_path = paths.get("message_role")
    if role_path:
        role_raw = _resolve_json_path(raw, role_path)
    if role_raw is None:
        role_raw = _find_by_well_known_names(raw, _WELL_KNOWN_ROLE_NAMES)
    role = Role.normalize(str(role_raw) if role_raw else "unknown")

    # Resolve body
    body_raw = None
    body_path = paths.get("message_body")
    if body_path:
        body_raw = _resolve_json_path(raw, body_path)
    if body_raw is None:
        body_raw = _find_by_well_known_names(raw, _WELL_KNOWN_BODY_NAMES)

    text = _extract_text_from_body(body_raw)

    # Extract structured content blocks from the body
    block_dicts = _extract_content_block_list(body_raw)
    content_blocks = extract_content_blocks(block_dicts) if block_dicts else []
    reasoning_traces = extract_reasoning_traces(block_dicts, provider) if block_dicts else []
    tool_calls = extract_tool_calls(block_dicts, provider) if block_dicts else []

    # Resolve timestamp
    ts_raw = None
    ts_path = paths.get("message_timestamp")
    if ts_path:
        ts_raw = _resolve_json_path(raw, ts_path)
    if ts_raw is None:
        ts_raw = _find_by_well_known_names(raw, _WELL_KNOWN_TIMESTAMP_NAMES)
    timestamp = parse_timestamp(ts_raw)

    # Resolve ID
    msg_id = None
    for id_field in ("uuid", "id", "messageId", "message_id"):
        if id_field in raw:
            msg_id = str(raw[id_field])
            break

    # Resolve optional metadata
    msg_data = raw.get("message", {}) if isinstance(raw.get("message"), dict) else raw
    usage_raw = msg_data.get("usage")
    tokens = extract_token_usage(usage_raw) if isinstance(usage_raw, dict) else None

    cost_raw = raw.get("costUSD") or raw.get("cost_usd")
    cost = CostInfo(total_usd=float(cost_raw)) if cost_raw else None

    duration_ms = raw.get("durationMs") or raw.get("duration_ms")
    model = msg_data.get("model")

    return HarmonizedMessage(
        id=msg_id,
        role=role,
        text=text,
        timestamp=timestamp,
        reasoning_traces=reasoning_traces,
        tool_calls=tool_calls,
        content_blocks=content_blocks,
        model=model if isinstance(model, str) else None,
        tokens=tokens,
        cost=cost,
        duration_ms=int(duration_ms) if duration_ms is not None else None,
        provider=provider,
        raw=raw,
    )


__all__ = ["extract_message_from_schema"]
