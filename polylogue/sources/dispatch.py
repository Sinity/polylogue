"""Provider detection and payload dispatch for source parsing."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from io import BytesIO
from typing import TYPE_CHECKING, TypeAlias

from polylogue.lib.payload_coercion import PayloadMapping, is_payload_mapping, optional_string
from polylogue.logging import get_logger
from polylogue.types import Provider

from .decoders import _decode_json_bytes, _iter_json_stream
from .parsers import chatgpt, claude, codex, drive
from .parsers.base import ParsedConversation, extract_messages_from_list

if TYPE_CHECKING:
    from polylogue.schemas.packages import SchemaResolution

logger = get_logger(__name__)

GROUP_PROVIDERS = frozenset({Provider.CLAUDE_CODE, Provider.CODEX, Provider.GEMINI, Provider.DRIVE})
STREAM_RECORD_PROVIDERS = frozenset({Provider.CLAUDE_CODE, Provider.CODEX})
_MAX_PARSE_DEPTH = 10

PayloadRecord: TypeAlias = dict[str, object]
ProviderParser: TypeAlias = Callable[[PayloadRecord, str], ParsedConversation]


def _payload_mapping(value: object) -> PayloadMapping | None:
    return value if is_payload_mapping(value) else None


def _payload_record(value: object) -> PayloadRecord | None:
    mapping = _payload_mapping(value)
    return dict(mapping) if mapping is not None else None


def _payload_list(value: object) -> list[object] | None:
    return value if isinstance(value, list) else None


def _payload_messages(record: PayloadMapping) -> list[object] | None:
    messages = record.get("messages")
    return messages if isinstance(messages, list) else None


def _payload_conversations(record: PayloadMapping) -> list[object] | None:
    conversations = record.get("conversations")
    return conversations if isinstance(conversations, list) else None


def _looks_like_gemini_mapping(record: PayloadMapping) -> bool:
    return "chunkedPrompt" in record or isinstance(record.get("chunks"), list)


def _detect_provider_from_mapping(record: PayloadMapping) -> Provider | None:
    if chatgpt.looks_like(record):
        return Provider.CHATGPT
    if claude.looks_like_ai(record):
        return Provider.CLAUDE_AI
    if claude.looks_like_code([dict(record)]):
        return Provider.CLAUDE_CODE
    if codex.looks_like([dict(record)]):
        return Provider.CODEX
    if _looks_like_gemini_mapping(record):
        return Provider.GEMINI
    return None


def _detect_provider_from_sequence(payloads: list[object]) -> Provider | None:
    if not payloads:
        return None

    first_record = _payload_mapping(payloads[0])
    if first_record is not None:
        if is_payload_mapping(first_record.get("mapping")):
            return Provider.CHATGPT
        if isinstance(first_record.get("chat_messages"), list):
            return Provider.CLAUDE_AI
        if _looks_like_gemini_mapping(first_record):
            return Provider.GEMINI
    if claude.looks_like_code(payloads):
        return Provider.CLAUDE_CODE
    if codex.looks_like(payloads):
        return Provider.CODEX
    return None


def _parse_bundle_items(
    payloads: list[object],
    fallback_id: str,
    parser: ProviderParser,
) -> list[ParsedConversation]:
    results: list[ParsedConversation] = []
    for index, item in enumerate(payloads):
        if record := _payload_record(item):
            results.append(parser(record, f"{fallback_id}-{index}"))
    return results


def _parse_grouped_records(
    payload: object,
    fallback_id: str,
    parser: Callable[[list[object], str], ParsedConversation],
) -> list[ParsedConversation]:
    payloads = _payload_list(payload)
    if payloads is not None:
        return [parser(payloads, fallback_id)]

    if record := _payload_record(payload):
        messages = _payload_messages(record)
        return [parser(messages if messages is not None else [record], fallback_id)]
    return []


def detect_provider(payload: object, path: object | None = None) -> Provider | None:
    """Infer provider from payload shape. Path is accepted for surface compatibility."""
    del path

    if record := _payload_mapping(payload):
        return _detect_provider_from_mapping(record)
    payloads = _payload_list(payload)
    return _detect_provider_from_sequence(payloads) if payloads is not None else None


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


def _looks_like_chunked_conversation(payload: object) -> bool:
    record = _payload_mapping(payload)
    return record is not None and (drive.looks_like(record) or isinstance(record.get("chunks"), list))


def _looks_like_chunked_conversation_list(payload: list[object]) -> bool:
    return bool(payload) and all(_looks_like_chunked_conversation(item) for item in payload)


def _schema_guided_payload(
    provider: Provider,
    payload: object,
    schema_resolution: SchemaResolution | None,
) -> object:
    """Apply schema-derived structural hints before provider-specific lowering."""
    if schema_resolution is None:
        return payload
    if schema_resolution.element_kind not in {"conversation_record_stream", "subagent_conversation_stream"}:
        return payload
    if provider in (Provider.CLAUDE_CODE, Provider.CODEX) and (record := _payload_record(payload)):
        messages = _payload_messages(record)
        if messages is not None:
            return messages
        return [record]
    return payload


def _generic_messages_conversation(
    provider: Provider,
    payload: PayloadMapping,
    fallback_id: str,
) -> ParsedConversation | None:
    messages_payload = _payload_messages(payload)
    if messages_payload is None:
        return None

    messages = extract_messages_from_list(messages_payload)
    title = optional_string(payload.get("title")) or optional_string(payload.get("name")) or fallback_id
    conversation_id = optional_string(payload.get("id")) or fallback_id
    return ParsedConversation(
        provider_name=provider,
        provider_conversation_id=conversation_id,
        title=title,
        created_at=None,
        updated_at=None,
        messages=messages,
    )


def parse_payload(
    provider: str | Provider,
    payload: object,
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

    shaped_payload = _schema_guided_payload(runtime_provider, payload, schema_resolution)

    if record := _payload_mapping(shaped_payload):
        if conversations := _payload_conversations(record):
            results: list[ParsedConversation] = []
            for index, item in enumerate(conversations):
                if item_record := _payload_mapping(item):
                    results.extend(
                        parse_payload(
                            runtime_provider,
                            item_record,
                            f"{fallback_id}-{index}",
                            _depth + 1,
                            schema_resolution=schema_resolution,
                        )
                    )
            return results
    else:
        record = None

    payloads = _payload_list(shaped_payload)

    if runtime_provider is Provider.CHATGPT:
        if payloads is not None:
            return _parse_bundle_items(payloads, fallback_id, chatgpt.parse)
        return [chatgpt.parse(dict(record), fallback_id)] if record is not None else []

    if runtime_provider is Provider.CLAUDE_AI:
        if payloads is not None:
            return _parse_bundle_items(payloads, fallback_id, claude.parse_ai)
        return [claude.parse_ai(dict(record), fallback_id)] if record is not None else []

    if runtime_provider is Provider.CLAUDE_CODE:
        return _parse_grouped_records(shaped_payload, fallback_id, claude.parse_code)

    if runtime_provider is Provider.CODEX:
        return _parse_grouped_records(shaped_payload, fallback_id, codex.parse)

    if runtime_provider in (Provider.GEMINI, Provider.DRIVE) and payloads is not None:
        if _looks_like_chunked_conversation_list(payloads):
            chunked_results: list[ParsedConversation] = []
            for index, item in enumerate(payloads):
                chunked_results.extend(
                    parse_payload(
                        runtime_provider,
                        item,
                        f"{fallback_id}-{index}",
                        _depth + 1,
                        schema_resolution=schema_resolution,
                    )
                )
            return chunked_results
        return [drive.parse_chunked_prompt(runtime_provider, {"chunks": payloads}, fallback_id)]

    if record is None:
        return []

    generic = _generic_messages_conversation(runtime_provider, record, fallback_id)
    if generic is not None:
        return [generic]

    if chatgpt.looks_like(record):
        return [chatgpt.parse(dict(record), fallback_id)]
    if _looks_like_chunked_conversation(record):
        return [drive.parse_chunked_prompt(runtime_provider, dict(record), fallback_id)]
    return []


def parse_stream_payload(
    provider: str | Provider,
    payloads: Iterable[object],
    fallback_id: str,
) -> list[ParsedConversation]:
    """Parse a grouped record stream without materializing the full payload list."""
    runtime_provider = Provider.from_string(provider)
    if runtime_provider is Provider.CLAUDE_CODE:
        return [claude.parse_stream(payloads, fallback_id)]
    if runtime_provider is Provider.CODEX:
        return [codex.parse_stream(payloads, fallback_id)]
    raise ValueError(f"provider {runtime_provider} does not support stream parsing")


def parse_drive_payload(
    provider: str | Provider,
    payload: object,
    fallback_id: str,
    _depth: int = 0,
) -> list[ParsedConversation]:
    runtime_provider = Provider.from_string(provider)
    if _depth > _MAX_PARSE_DEPTH:
        logger.warning("Recursion depth exceeded parsing drive payload %s", fallback_id)
        return []

    payloads = _payload_list(payload)
    if payloads is not None:
        if payloads and all(isinstance(item, str) or _payload_record(item) is not None for item in payloads):
            first_record = _payload_mapping(payloads[0]) if payloads else None
            if first_record is None or "role" in first_record or "text" in first_record:
                return [drive.parse_chunked_prompt(runtime_provider, {"chunks": payloads}, fallback_id)]

        nested_results: list[ParsedConversation] = []
        for index, item in enumerate(payloads):
            nested_results.extend(parse_drive_payload(runtime_provider, item, f"{fallback_id}-{index}", _depth + 1))
        return nested_results

    if record := _payload_mapping(payload):
        if "chunkedPrompt" in record or "chunks" in record:
            return [drive.parse_chunked_prompt(runtime_provider, dict(record), fallback_id)]
        detected = detect_provider(record) or runtime_provider
        return parse_payload(detected, record, fallback_id)
    return []


__all__ = [
    "GROUP_PROVIDERS",
    "STREAM_RECORD_PROVIDERS",
    "_detect_provider_from_raw_bytes",
    "detect_provider",
    "parse_drive_payload",
    "parse_payload",
    "parse_stream_payload",
]
