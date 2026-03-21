"""Provider detection and payload dispatch for source parsing."""

from __future__ import annotations

import json
from io import BytesIO
from typing import TYPE_CHECKING, Any

from polylogue.logging import get_logger
from polylogue.types import Provider

from .decoders import _decode_json_bytes, _iter_json_stream
from .parsers import chatgpt, claude, codex, drive
from .parsers.base import ParsedConversation, extract_messages_from_list

if TYPE_CHECKING:
    from polylogue.schemas.packages import SchemaResolution

logger = get_logger(__name__)

GROUP_PROVIDERS = frozenset({Provider.CLAUDE_CODE, Provider.CODEX, Provider.GEMINI, Provider.DRIVE})
_MAX_PARSE_DEPTH = 10


def detect_provider(payload: Any, path: object | None = None) -> Provider | None:
    """Infer provider from payload shape. Path is accepted for surface compatibility."""
    del path

    if isinstance(payload, dict):
        if chatgpt.looks_like(payload):
            return Provider.CHATGPT
        if claude.looks_like_ai(payload):
            return Provider.CLAUDE_AI
        if claude.looks_like_code([payload]):
            return Provider.CLAUDE_CODE
        if codex.looks_like([payload]):
            return Provider.CODEX
        if "chunkedPrompt" in payload or ("chunks" in payload and isinstance(payload.get("chunks"), list)):
            return Provider.GEMINI
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            first = payload[0]
            if isinstance(first.get("mapping"), dict):
                return Provider.CHATGPT
            if isinstance(first.get("chat_messages"), list):
                return Provider.CLAUDE_AI
            if "chunkedPrompt" in first or ("chunks" in first and isinstance(first.get("chunks"), list)):
                return Provider.GEMINI
        if claude.looks_like_code(payload):
            return Provider.CLAUDE_CODE
        if codex.looks_like(payload):
            return Provider.CODEX
    return None


def _detect_provider_from_raw_bytes(
    raw_bytes: bytes,
    stream_name: str,
    fallback_provider: Provider,
) -> Provider:
    text = _decode_json_bytes(raw_bytes)
    if text:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        else:
            detected = detect_provider(payload)
            if detected is not None:
                return detected

    try:
        payloads = list(_iter_json_stream(BytesIO(raw_bytes), stream_name))
    except Exception:
        return fallback_provider

    return detect_provider(payloads) or fallback_provider


def _looks_like_chunked_conversation(payload: Any) -> bool:
    return isinstance(payload, dict) and (
        drive.looks_like(payload)
        or isinstance(payload.get("chunks"), list)
    )


def _looks_like_chunked_conversation_list(payload: list[Any]) -> bool:
    return bool(payload) and all(_looks_like_chunked_conversation(item) for item in payload)


def _parse_bundle_items(
    payload: list[Any],
    fallback_id: str,
    parser,
) -> list[ParsedConversation]:
    return [parser(item, f"{fallback_id}-{i}") for i, item in enumerate(payload) if isinstance(item, dict)]


def _schema_guided_payload(
    provider: Provider,
    payload: Any,
    schema_resolution: SchemaResolution | None,
) -> Any:
    """Apply schema-derived structural hints before provider-specific lowering."""
    if schema_resolution is None:
        return payload
    if schema_resolution.element_kind not in {"conversation_record_stream", "subagent_conversation_stream"}:
        return payload
    if provider in (Provider.CLAUDE_CODE, Provider.CODEX) and isinstance(payload, dict):
        messages = payload.get("messages")
        if isinstance(messages, list):
            return messages
        return [payload]
    return payload


def parse_payload(
    provider: str | Provider,
    payload: Any,
    fallback_id: str,
    _depth: int = 0,
    *,
    schema_resolution: SchemaResolution | None = None,
) -> list[ParsedConversation]:
    """Dispatch parsed payload to the appropriate provider parser."""
    runtime_provider = Provider.from_string(provider)
    if _depth > _MAX_PARSE_DEPTH:
        logger.warning("Recursion depth exceeded parsing %s (provider=%s)", fallback_id, provider)
        return []
    payload = _schema_guided_payload(runtime_provider, payload, schema_resolution)
    if isinstance(payload, dict) and isinstance(payload.get("conversations"), list):
        results: list[ParsedConversation] = []
        for i, item in enumerate(payload["conversations"]):
            if isinstance(item, dict):
                results.extend(
                    parse_payload(
                        runtime_provider,
                        item,
                        f"{fallback_id}-{i}",
                        _depth + 1,
                        schema_resolution=schema_resolution,
                    )
                )
        return results
    if runtime_provider is Provider.CHATGPT:
        if isinstance(payload, list):
            return _parse_bundle_items(payload, fallback_id, chatgpt.parse)
        return [chatgpt.parse(payload, fallback_id)]
    if runtime_provider is Provider.CLAUDE_AI:
        if isinstance(payload, list):
            return _parse_bundle_items(payload, fallback_id, claude.parse_ai)
        return [claude.parse_ai(payload, fallback_id)]
    if runtime_provider is Provider.CLAUDE_CODE:
        if isinstance(payload, list):
            return [claude.parse_code(payload, fallback_id)]
        if isinstance(payload, dict):
            if isinstance(payload.get("messages"), list):
                return [claude.parse_code(payload["messages"], fallback_id)]
            return [claude.parse_code([payload], fallback_id)]
    if runtime_provider is Provider.CODEX:
        if isinstance(payload, list):
            return [codex.parse(payload, fallback_id)]
        if isinstance(payload, dict):
            return [codex.parse([payload], fallback_id)]
    if runtime_provider in (Provider.GEMINI, Provider.DRIVE) and isinstance(payload, list):
        if _looks_like_chunked_conversation_list(payload):
            results = []
            for i, item in enumerate(payload):
                results.extend(
                    parse_payload(
                        runtime_provider,
                        item,
                        f"{fallback_id}-{i}",
                        _depth + 1,
                        schema_resolution=schema_resolution,
                    )
                )
            return results
        return [drive.parse_chunked_prompt(runtime_provider, {"chunks": payload}, fallback_id)]

    if isinstance(payload, dict):
        if "messages" in payload and isinstance(payload["messages"], list):
            messages = extract_messages_from_list(payload["messages"])
            title = payload.get("title") or payload.get("name") or fallback_id
            return [
                ParsedConversation(
                    provider_name=runtime_provider,
                    provider_conversation_id=str(payload.get("id") or fallback_id),
                    title=str(title),
                    created_at=None,
                    updated_at=None,
                    messages=messages,
                )
            ]

        if chatgpt.looks_like(payload):
            return [chatgpt.parse(payload, fallback_id)]
        if _looks_like_chunked_conversation(payload):
            return [drive.parse_chunked_prompt(runtime_provider, payload, fallback_id)]
        return []

    return []


def parse_drive_payload(provider: str | Provider, payload: Any, fallback_id: str, _depth: int = 0) -> list[ParsedConversation]:
    runtime_provider = Provider.from_string(provider)
    if _depth > _MAX_PARSE_DEPTH:
        logger.warning("Recursion depth exceeded parsing drive payload %s", fallback_id)
        return []
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict) and ("role" in payload[0] or "text" in payload[0]):
            return [drive.parse_chunked_prompt(runtime_provider, {"chunks": payload}, fallback_id)]

        results = []
        for i, item in enumerate(payload):
            results.extend(parse_drive_payload(runtime_provider, item, f"{fallback_id}-{i}", _depth + 1))
        return results
    if isinstance(payload, dict):
        if "chunkedPrompt" in payload or "chunks" in payload:
            return [drive.parse_chunked_prompt(runtime_provider, payload, fallback_id)]
        detected = detect_provider(payload) or runtime_provider
        return parse_payload(detected, payload, fallback_id)
    return []


__all__ = [
    "GROUP_PROVIDERS",
    "_detect_provider_from_raw_bytes",
    "detect_provider",
    "parse_drive_payload",
    "parse_payload",
]
