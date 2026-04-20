"""Provider detection and payload lowering for source parsing."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Literal, TypeAlias

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
LoweredPayloadMode: TypeAlias = Literal[
    "single_record",
    "bundle_record",
    "grouped_records",
    "chunked_prompt",
    "generic_messages",
]


@dataclass(frozen=True, slots=True)
class LoweredPayloadSpec:
    provider: Provider
    fallback_id: str
    mode: LoweredPayloadMode
    payload: PayloadRecord | list[object]


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


def _lower_bundle_specs(
    provider: Provider,
    payloads: list[object],
    fallback_id: str,
) -> list[LoweredPayloadSpec]:
    specs: list[LoweredPayloadSpec] = []
    for index, item in enumerate(payloads):
        if record := _payload_record(item):
            specs.append(
                LoweredPayloadSpec(
                    provider=provider,
                    fallback_id=f"{fallback_id}-{index}",
                    mode="bundle_record",
                    payload=record,
                )
            )
    return specs


def _lower_grouped_spec(
    provider: Provider,
    payload: object,
    fallback_id: str,
) -> list[LoweredPayloadSpec]:
    payloads = _payload_list(payload)
    if payloads is not None:
        return [
            LoweredPayloadSpec(provider=provider, fallback_id=fallback_id, mode="grouped_records", payload=payloads)
        ]

    record = _payload_record(payload)
    if record is None:
        return []

    messages = _payload_messages(record)
    grouped_payload = messages if messages is not None else [record]
    return [
        LoweredPayloadSpec(provider=provider, fallback_id=fallback_id, mode="grouped_records", payload=grouped_payload)
    ]


def _lower_drive_like_specs(
    runtime_provider: Provider,
    payload: object,
    fallback_id: str,
    *,
    depth: int,
    schema_resolution: SchemaResolution | None,
) -> list[LoweredPayloadSpec]:
    payloads = _payload_list(payload)
    if payloads is not None:
        if _looks_like_chunked_conversation_list(payloads):
            nested_specs: list[LoweredPayloadSpec] = []
            for index, item in enumerate(payloads):
                nested_specs.extend(
                    _lower_payload_specs(
                        runtime_provider,
                        item,
                        f"{fallback_id}-{index}",
                        depth=depth + 1,
                        schema_resolution=schema_resolution,
                    )
                )
            return nested_specs
        return [
            LoweredPayloadSpec(
                provider=runtime_provider,
                fallback_id=fallback_id,
                mode="chunked_prompt",
                payload=payloads,
            )
        ]

    record = _payload_record(payload)
    if record is None:
        return []

    if _payload_messages(record) is not None:
        return [
            LoweredPayloadSpec(
                provider=runtime_provider,
                fallback_id=fallback_id,
                mode="generic_messages",
                payload=record,
            )
        ]
    if chatgpt.looks_like(record):
        return [
            LoweredPayloadSpec(
                provider=Provider.CHATGPT,
                fallback_id=fallback_id,
                mode="single_record",
                payload=record,
            )
        ]
    if _looks_like_chunked_conversation(record):
        return [
            LoweredPayloadSpec(
                provider=runtime_provider,
                fallback_id=fallback_id,
                mode="chunked_prompt",
                payload=record,
            )
        ]
    return []


def _lower_payload_specs(
    provider: str | Provider,
    payload: object,
    fallback_id: str,
    *,
    depth: int = 0,
    schema_resolution: SchemaResolution | None = None,
) -> list[LoweredPayloadSpec]:
    runtime_provider = Provider.from_string(provider)
    if depth > _MAX_PARSE_DEPTH:
        logger.warning("Recursion depth exceeded parsing %s (provider=%s)", fallback_id, provider)
        return []

    shaped_payload = _schema_guided_payload(runtime_provider, payload, schema_resolution)
    record = _payload_record(shaped_payload)
    if record is not None and (conversations := _payload_conversations(record)):
        lowered_specs: list[LoweredPayloadSpec] = []
        for index, item in enumerate(conversations):
            if item_record := _payload_record(item):
                lowered_specs.extend(
                    _lower_payload_specs(
                        runtime_provider,
                        item_record,
                        f"{fallback_id}-{index}",
                        depth=depth + 1,
                        schema_resolution=schema_resolution,
                    )
                )
        return lowered_specs

    payloads = _payload_list(shaped_payload)

    if runtime_provider is Provider.CHATGPT:
        if payloads is not None:
            return _lower_bundle_specs(runtime_provider, payloads, fallback_id)
        return (
            [
                LoweredPayloadSpec(
                    provider=runtime_provider, fallback_id=fallback_id, mode="single_record", payload=record
                )
            ]
            if record is not None
            else []
        )

    if runtime_provider is Provider.CLAUDE_AI:
        if payloads is not None:
            return _lower_bundle_specs(runtime_provider, payloads, fallback_id)
        return (
            [
                LoweredPayloadSpec(
                    provider=runtime_provider, fallback_id=fallback_id, mode="single_record", payload=record
                )
            ]
            if record is not None
            else []
        )

    if runtime_provider is Provider.CLAUDE_CODE:
        return _lower_grouped_spec(runtime_provider, shaped_payload, fallback_id)

    if runtime_provider is Provider.CODEX:
        return _lower_grouped_spec(runtime_provider, shaped_payload, fallback_id)

    if runtime_provider in (Provider.GEMINI, Provider.DRIVE):
        return _lower_drive_like_specs(
            runtime_provider,
            shaped_payload,
            fallback_id,
            depth=depth,
            schema_resolution=schema_resolution,
        )

    if record is not None and _payload_messages(record) is not None:
        return [
            LoweredPayloadSpec(
                provider=runtime_provider,
                fallback_id=fallback_id,
                mode="generic_messages",
                payload=record,
            )
        ]
    if record is not None and chatgpt.looks_like(record):
        return [
            LoweredPayloadSpec(
                provider=Provider.CHATGPT,
                fallback_id=fallback_id,
                mode="single_record",
                payload=record,
            )
        ]
    if record is not None and _looks_like_chunked_conversation(record):
        return [
            LoweredPayloadSpec(
                provider=runtime_provider,
                fallback_id=fallback_id,
                mode="chunked_prompt",
                payload=record,
            )
        ]
    return []


def _generic_messages_conversation(
    provider: Provider,
    payload: PayloadRecord,
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


def _parse_lowered_spec(spec: LoweredPayloadSpec) -> list[ParsedConversation]:
    if spec.provider is Provider.CHATGPT:
        record = _payload_record(spec.payload)
        return [chatgpt.parse(record, spec.fallback_id)] if record is not None else []

    if spec.provider is Provider.CLAUDE_AI:
        record = _payload_record(spec.payload)
        return [claude.parse_ai(record, spec.fallback_id)] if record is not None else []

    if spec.provider is Provider.CLAUDE_CODE:
        payloads = _payload_list(spec.payload)
        return [claude.parse_code(payloads, spec.fallback_id)] if payloads is not None else []

    if spec.provider is Provider.CODEX:
        payloads = _payload_list(spec.payload)
        return [codex.parse(payloads, spec.fallback_id)] if payloads is not None else []

    if spec.mode == "chunked_prompt":
        record = _payload_record(spec.payload)
        payload = record if record is not None else {"chunks": _payload_list(spec.payload) or []}
        return [drive.parse_chunked_prompt(spec.provider, payload, spec.fallback_id)]

    if spec.mode == "generic_messages":
        record = _payload_record(spec.payload)
        generic = (
            _generic_messages_conversation(spec.provider, record, spec.fallback_id) if record is not None else None
        )
        return [generic] if generic is not None else []

    return []


def parse_payload(
    provider: str | Provider,
    payload: object,
    fallback_id: str,
    _depth: int = 0,
    *,
    schema_resolution: SchemaResolution | None = None,
) -> list[ParsedConversation]:
    """Dispatch parsed payload to the appropriate provider parser."""
    lowered_specs = _lower_payload_specs(
        provider,
        payload,
        fallback_id,
        depth=_depth,
        schema_resolution=schema_resolution,
    )
    conversations: list[ParsedConversation] = []
    for spec in lowered_specs:
        conversations.extend(_parse_lowered_spec(spec))
    return conversations


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
                spec = LoweredPayloadSpec(
                    provider=runtime_provider,
                    fallback_id=fallback_id,
                    mode="chunked_prompt",
                    payload=payloads,
                )
                return _parse_lowered_spec(spec)

        nested_conversations: list[ParsedConversation] = []
        for index, item in enumerate(payloads):
            nested_conversations.extend(
                parse_drive_payload(
                    runtime_provider,
                    item,
                    f"{fallback_id}-{index}",
                    _depth + 1,
                )
            )
        return nested_conversations

    if record := _payload_record(payload):
        if "chunkedPrompt" in record or "chunks" in record:
            spec = LoweredPayloadSpec(
                provider=runtime_provider,
                fallback_id=fallback_id,
                mode="chunked_prompt",
                payload=record,
            )
            return _parse_lowered_spec(spec)

        detected = detect_provider(record) or runtime_provider
        return parse_payload(detected, record, fallback_id, _depth + 1)

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
