"""Schema-driven message extraction using semantic role annotations."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

from polylogue.lib.json import JSONDocument, json_document
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

SchemaPayload: TypeAlias = JSONDocument
SchemaBlockPayload: TypeAlias = list[SchemaPayload]

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


@dataclass(frozen=True)
class ResolvedSchemaBody:
    raw: object | None
    text: str
    block_payloads: SchemaBlockPayload


@dataclass(frozen=True)
class ResolvedSchemaMessageFields:
    message_id: str | None
    role_raw: object | None
    body: ResolvedSchemaBody
    timestamp_raw: str | int | float | None
    model: str | None
    usage: JSONDocument | None
    cost: CostInfo | None
    duration_ms: int | None


def _resolve_json_path(raw: SchemaPayload, path: str) -> object | None:
    """Resolve a dotted JSON path against a raw payload."""
    return _walk(raw, _split_path(path))


def _split_path(path: str) -> list[str]:
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
            continue
        current += char
    if current:
        segments.append(current)
    return segments


def _walk(obj: object | None, parts: list[str]) -> object | None:
    if not parts:
        return obj
    if obj is None:
        return None

    head, rest = parts[0], parts[1:]
    if head.startswith(("anyOf[", "oneOf[", "allOf[")):
        return _walk(obj, rest)
    if head == "[]":
        if isinstance(obj, list):
            for item in obj:
                result = _walk(item, rest)
                if result is not None:
                    return result
        return None
    if head == "*":
        if isinstance(obj, dict):
            for value in obj.values():
                result = _walk(value, rest)
                if result is not None:
                    return result
        return None
    if isinstance(obj, dict):
        return _walk(obj.get(head), rest)
    return None


def _schema_payload(value: object) -> SchemaPayload | None:
    if not isinstance(value, dict):
        return None
    return json_document(value)


def _object_record(value: SchemaPayload) -> dict[str, object]:
    return dict(value)


def _find_by_well_known_names(raw: SchemaPayload, names: frozenset[str]) -> object | None:
    for key, value in raw.items():
        if key.lower() in names:
            return value
    for value in raw.values():
        payload = _schema_payload(value)
        if payload is None:
            continue
        for subkey, subvalue in payload.items():
            if subkey.lower() in names:
                return subvalue
    return None


# -------------------------------------------------------------------
# Content extraction from resolved body
# -------------------------------------------------------------------


def _extract_text_from_body(body: object) -> str:
    if isinstance(body, str):
        return body
    if isinstance(body, list):
        parts: list[str] = []
        for item in body:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str) and text:
                    parts.append(text)
        return "\n".join(parts) if parts else ""
    if isinstance(body, dict):
        text = body.get("text") or body.get("content")
        if isinstance(text, str):
            return text
        nested = body.get("parts")
        if isinstance(nested, list):
            return _extract_text_from_body(nested)
    return str(body) if body else ""


def _extract_content_block_list(body: object) -> SchemaBlockPayload:
    if isinstance(body, list):
        blocks = [payload for item in body if (payload := _schema_payload(item)) is not None and "type" in payload]
        if blocks:
            return blocks
    payload = _schema_payload(body)
    if payload is not None:
        for key in ("content", "parts", "blocks"):
            nested = payload.get(key)
            if not isinstance(nested, list):
                continue
            blocks = [
                nested_payload
                for item in nested
                if (nested_payload := _schema_payload(item)) is not None and "type" in nested_payload
            ]
            if blocks:
                return blocks
    return []


def _extract_message_id(raw: SchemaPayload) -> str | None:
    for field_name in ("uuid", "id", "messageId", "message_id"):
        value = raw.get(field_name)
        if value is not None:
            return str(value)
    return None


def _float_from_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _int_from_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _timestamp_candidate(value: object) -> str | int | float | None:
    if isinstance(value, (str, int, float)):
        return value
    return None


def _extract_cost(raw: SchemaPayload) -> CostInfo | None:
    cost_raw = raw.get("costUSD") or raw.get("cost_usd")
    if cost_raw is None:
        return None
    total_usd = _float_from_value(cost_raw)
    if total_usd is None:
        return None
    return CostInfo(total_usd=total_usd)


def _extract_duration_ms(raw: SchemaPayload) -> int | None:
    duration_raw = raw.get("durationMs") or raw.get("duration_ms")
    if duration_raw is None:
        return None
    return _int_from_value(duration_raw)


# -------------------------------------------------------------------
# Main extraction entry point
# -------------------------------------------------------------------


def _resolve_role_raw(raw: SchemaPayload, paths: Mapping[str, str | None]) -> object | None:
    role_path = paths.get("message_role")
    role_raw = _resolve_json_path(raw, role_path) if role_path is not None else None
    return role_raw if role_raw is not None else _find_by_well_known_names(raw, _WELL_KNOWN_ROLE_NAMES)


def _resolve_body_raw(raw: SchemaPayload, paths: Mapping[str, str | None]) -> object | None:
    body_path = paths.get("message_body")
    body_raw = _resolve_json_path(raw, body_path) if body_path is not None else None
    if body_raw is not None:
        return body_raw
    message_payload = _resolve_message_meta_payload(raw)
    nested_body = _find_by_well_known_names(message_payload, _WELL_KNOWN_BODY_NAMES)
    return nested_body if nested_body is not None else _find_by_well_known_names(raw, _WELL_KNOWN_BODY_NAMES)


def _resolve_timestamp_raw(raw: SchemaPayload, paths: Mapping[str, str | None]) -> str | int | float | None:
    timestamp_path = paths.get("message_timestamp")
    ts_raw = _resolve_json_path(raw, timestamp_path) if timestamp_path is not None else None
    if ts_raw is None:
        ts_raw = _find_by_well_known_names(raw, _WELL_KNOWN_TIMESTAMP_NAMES)
    return _timestamp_candidate(ts_raw)


def _resolve_message_meta_payload(raw: SchemaPayload) -> SchemaPayload:
    message_payload = json_document(raw.get("message"))
    return message_payload if message_payload else raw


def _resolve_body(body_raw: object | None) -> ResolvedSchemaBody:
    block_payloads = _extract_content_block_list(body_raw)
    return ResolvedSchemaBody(
        raw=body_raw,
        text=_extract_text_from_body(body_raw),
        block_payloads=block_payloads,
    )


def _resolve_message_fields(
    raw: SchemaPayload,
    *,
    schema: SchemaPayload,
    provider: Provider,
) -> ResolvedSchemaMessageFields:
    pins = load_pins(provider)
    paths = resolve_pinned_paths(schema, pins)
    message_meta = _resolve_message_meta_payload(raw)
    usage = json_document(message_meta.get("usage")) or None
    model = message_meta.get("model")

    return ResolvedSchemaMessageFields(
        message_id=_extract_message_id(raw),
        role_raw=_resolve_role_raw(raw, paths),
        body=_resolve_body(_resolve_body_raw(raw, paths)),
        timestamp_raw=_resolve_timestamp_raw(raw, paths),
        model=model if isinstance(model, str) else None,
        usage=usage,
        cost=_extract_cost(raw),
        duration_ms=_extract_duration_ms(raw),
    )


def extract_message_from_schema(
    raw: SchemaPayload,
    *,
    schema: SchemaPayload,
    provider: Provider,
) -> HarmonizedMessage:
    """Extract a harmonized message using schema semantic annotations."""
    resolved = _resolve_message_fields(raw, schema=schema, provider=provider)
    block_dicts = resolved.body.block_payloads
    role = Role.normalize(str(resolved.role_raw) if resolved.role_raw else "unknown")
    timestamp = parse_timestamp(resolved.timestamp_raw)

    return HarmonizedMessage(
        id=resolved.message_id,
        role=role,
        text=resolved.body.text,
        timestamp=timestamp,
        reasoning_traces=extract_reasoning_traces(block_dicts, provider) if block_dicts else [],
        tool_calls=extract_tool_calls(block_dicts, provider) if block_dicts else [],
        content_blocks=extract_content_blocks(block_dicts) if block_dicts else [],
        model=resolved.model,
        tokens=extract_token_usage(resolved.usage),
        cost=resolved.cost,
        duration_ms=resolved.duration_ms,
        provider=provider,
        raw=_object_record(raw),
    )


__all__ = ["extract_message_from_schema"]
