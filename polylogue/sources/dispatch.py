"""Provider detection and payload lowering for source parsing."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from polylogue.core.json import JSONDocument, JSONValue, is_json_document, is_json_value, normalize_json_decimal
from polylogue.core.payload_coercion import optional_string
from polylogue.logging import get_logger
from polylogue.types import Provider

from .decoders import _decode_json_bytes, _iter_json_stream
from .parsers import antigravity, browser_capture, chatgpt, claude, codex, drive, local_agent
from .parsers.base import ParsedSession, extract_messages_from_list

if TYPE_CHECKING:
    from polylogue.schemas.packages import SchemaResolution

logger = get_logger(__name__)

BUNDLE_PROVIDERS = frozenset({Provider.CHATGPT, Provider.CLAUDE_AI})
GROUP_PROVIDERS = frozenset({Provider.CLAUDE_CODE, Provider.CODEX, Provider.GEMINI, Provider.DRIVE})
STREAM_RECORD_PROVIDERS = frozenset({Provider.CLAUDE_CODE, Provider.CODEX})
DRIVE_LIKE_PROVIDERS = frozenset({Provider.GEMINI, Provider.DRIVE})
_MAX_PARSE_DEPTH = 10

PayloadRecord: TypeAlias = JSONDocument
PayloadSequence: TypeAlias = list[JSONValue]
LoweredPayloadMode: TypeAlias = Literal[
    "bundle_record",
    "browser_capture",
    "chunked_prompt",
    "generic_messages",
    "grouped_records",
    "local_artifact_document",
    "local_agent_document",
    "single_record",
]


@dataclass(frozen=True, slots=True)
class LoweredPayloadSpec:
    provider: Provider
    fallback_id: str
    mode: LoweredPayloadMode
    payload: PayloadRecord | PayloadSequence
    source_path: str | None = None


def _payload_record(value: object) -> PayloadRecord | None:
    normalized = normalize_json_decimal(value)
    return normalized if is_json_document(normalized) else None


def _payload_sequence(value: object) -> PayloadSequence | None:
    if not isinstance(value, list):
        return None
    payloads: list[JSONValue] = []
    for item in value:
        normalized = normalize_json_decimal(item)
        if not is_json_value(normalized):
            return None
        payloads.append(normalized)
    return payloads


def _record_messages(record: PayloadRecord) -> list[JSONValue] | None:
    messages = record.get("messages")
    return messages if isinstance(messages, list) else None


def _record_sessions(record: PayloadRecord) -> list[JSONValue] | None:
    sessions = record.get("sessions")
    return sessions if isinstance(sessions, list) else None


def _looks_like_gemini_mapping(record: PayloadRecord) -> bool:
    return "chunkedPrompt" in record or isinstance(record.get("chunks"), list)


def _detect_provider_from_record(record: PayloadRecord) -> Provider | None:
    if browser_capture.looks_like(record):
        session = record.get("session")
        provider = session.get("provider") if isinstance(session, dict) else None
        return Provider.from_string(provider if isinstance(provider, str) else None)
    # Local-agent JSON session documents share enough generic message keys with
    # Claude Code that they must be recognized before broader validators.
    if local_agent.looks_like_gemini_cli(record):
        return Provider.GEMINI_CLI
    if local_agent.looks_like_hermes(record):
        return Provider.HERMES
    if antigravity.looks_like_markdown_export(record):
        return Provider.ANTIGRAVITY
    if antigravity.looks_like_brain_metadata(record, None):
        return Provider.ANTIGRAVITY
    # Specific type-level checks first (Codex, Claude Code use Pydantic
    # validation), then weaker dict-key checks (ChatGPT, Claude AI, Gemini).
    if codex.looks_like([dict(record)]):
        return Provider.CODEX
    if claude.looks_like_code([dict(record)]):
        return Provider.CLAUDE_CODE
    if chatgpt.looks_like(record):
        return Provider.CHATGPT
    if claude.looks_like_ai(record):
        return Provider.CLAUDE_AI
    if _looks_like_gemini_mapping(record):
        return Provider.GEMINI
    return None


def _detect_provider_from_sequence(payloads: PayloadSequence) -> Provider | None:
    if not payloads:
        return None

    first_record = _payload_record(payloads[0])
    if first_record is not None:
        if is_json_document(first_record.get("mapping")):
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

    if record := _payload_record(payload):
        return _detect_provider_from_record(record)
    payloads = _payload_sequence(payload)
    return _detect_provider_from_sequence(payloads) if payloads is not None else None


def _detect_provider_from_raw_bytes(
    raw_bytes: bytes,
    stream_name: str,
    fallback_provider: Provider,
    *,
    truncated_tail_ok: bool = False,
) -> Provider:
    jsonl_like = _is_jsonl_stream_name(stream_name)
    text = None if jsonl_like else _decode_json_bytes(raw_bytes)
    if text is not None:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        else:
            detected = detect_provider(payload)
            if detected is not None:
                return detected

    stream_bytes = _trim_jsonl_detection_prefix(raw_bytes, stream_name) if truncated_tail_ok else raw_bytes
    if not stream_bytes:
        return fallback_provider
    try:
        stream = _iter_json_stream(BytesIO(stream_bytes), stream_name)
        payloads = list(islice(stream, 32)) if jsonl_like else list(stream)
    except Exception as exc:
        # JSON-stream detection commonly fails on payloads that the
        # record-shape detectors above already handled (or that simply do
        # not parse as a record stream — e.g. ChatGPT bundles read via
        # `devtools pipeline-probe`). Emit a structured WARNING rather
        # than a Rich traceback so default invocations do not look
        # broken when the fallback is the intended path.
        logger.warning(
            "provider_detection_stream_fallback",
            stream_name=stream_name,
            fallback_provider=fallback_provider.value,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return fallback_provider

    return detect_provider(payloads) or fallback_provider


def _is_jsonl_stream_name(stream_name: str) -> bool:
    return stream_name.lower().endswith((".jsonl", ".jsonl.txt", ".ndjson"))


def _trim_jsonl_detection_prefix(raw_bytes: bytes, stream_name: str) -> bytes:
    if not _is_jsonl_stream_name(stream_name):
        return raw_bytes
    if raw_bytes.endswith((b"\n", b"\r")):
        return raw_bytes
    newline_at = raw_bytes.rfind(b"\n")
    return raw_bytes[: newline_at + 1] if newline_at >= 0 else b""


def _schema_guided_payload(
    provider: Provider,
    payload: object,
    schema_resolution: SchemaResolution | None,
) -> object:
    """Apply schema-derived structural hints before provider-specific lowering."""
    if schema_resolution is None:
        return payload
    if schema_resolution.element_kind not in {"session_record_stream", "subagent_session_stream"}:
        return payload
    if provider not in {Provider.CLAUDE_CODE, Provider.CODEX}:
        return payload

    record = _payload_record(payload)
    if record is None:
        return payload

    messages = _record_messages(record)
    if messages is not None:
        return messages
    return [record]


def _looks_like_chunked_session(payload: object) -> bool:
    record = _payload_record(payload)
    return record is not None and (drive.looks_like(record) or isinstance(record.get("chunks"), list))


def _looks_like_chunked_session_list(payloads: PayloadSequence) -> bool:
    return bool(payloads) and all(_looks_like_chunked_session(item) for item in payloads)


def _single_record_spec(provider: Provider, payload: PayloadRecord, fallback_id: str) -> LoweredPayloadSpec:
    return LoweredPayloadSpec(
        provider=provider,
        fallback_id=fallback_id,
        mode="single_record",
        payload=payload,
    )


def _chunked_prompt_spec(
    provider: Provider,
    payload: PayloadRecord | PayloadSequence,
    fallback_id: str,
) -> LoweredPayloadSpec:
    return LoweredPayloadSpec(
        provider=provider,
        fallback_id=fallback_id,
        mode="chunked_prompt",
        payload=payload,
    )


def _generic_messages_spec(
    provider: Provider,
    payload: PayloadRecord,
    fallback_id: str,
) -> LoweredPayloadSpec:
    return LoweredPayloadSpec(
        provider=provider,
        fallback_id=fallback_id,
        mode="generic_messages",
        payload=payload,
    )


def _local_agent_document_spec(
    provider: Provider,
    payload: PayloadRecord,
    fallback_id: str,
) -> LoweredPayloadSpec:
    return LoweredPayloadSpec(
        provider=provider,
        fallback_id=fallback_id,
        mode="local_agent_document",
        payload=payload,
    )


def _local_artifact_document_spec(
    provider: Provider,
    payload: PayloadRecord,
    fallback_id: str,
    *,
    source_path: str | None,
) -> LoweredPayloadSpec:
    return LoweredPayloadSpec(
        provider=provider,
        fallback_id=fallback_id,
        mode="local_artifact_document",
        payload=payload,
        source_path=source_path,
    )


def _grouped_records_spec(
    provider: Provider,
    payload: PayloadRecord | PayloadSequence,
    fallback_id: str,
) -> LoweredPayloadSpec:
    return LoweredPayloadSpec(
        provider=provider,
        fallback_id=fallback_id,
        mode="grouped_records",
        payload=payload,
    )


def _bundle_record_specs(
    provider: Provider,
    payloads: PayloadSequence,
    fallback_id: str,
) -> list[LoweredPayloadSpec]:
    return [
        LoweredPayloadSpec(
            provider=provider,
            fallback_id=f"{fallback_id}-{index}",
            mode="bundle_record",
            payload=record,
        )
        for index, item in enumerate(payloads)
        if (record := _payload_record(item)) is not None
    ]


def _lower_bundle_payload(
    provider: Provider,
    shaped_payload: object,
    fallback_id: str,
) -> list[LoweredPayloadSpec]:
    payloads = _payload_sequence(shaped_payload)
    if payloads is not None:
        return _bundle_record_specs(provider, payloads, fallback_id)
    record = _payload_record(shaped_payload)
    return [_single_record_spec(provider, record, fallback_id)] if record is not None else []


def _lower_grouped_payload(
    provider: Provider,
    shaped_payload: object,
    fallback_id: str,
) -> list[LoweredPayloadSpec]:
    payloads = _payload_sequence(shaped_payload)
    if payloads is not None:
        return [_grouped_records_spec(provider, payloads, fallback_id)]

    record = _payload_record(shaped_payload)
    if record is None:
        return []

    messages = _record_messages(record)
    grouped_payload = messages if messages is not None else [record]
    return [_grouped_records_spec(provider, grouped_payload, fallback_id)]


def _lower_drive_like_payload(
    provider: Provider,
    shaped_payload: object,
    fallback_id: str,
    *,
    depth: int,
    schema_resolution: SchemaResolution | None,
) -> list[LoweredPayloadSpec]:
    payloads = _payload_sequence(shaped_payload)
    if payloads is not None:
        if _looks_like_chunked_session_list(payloads):
            nested_specs: list[LoweredPayloadSpec] = []
            for index, item in enumerate(payloads):
                nested_specs.extend(
                    _lower_payload_specs(
                        provider,
                        item,
                        f"{fallback_id}-{index}",
                        depth=depth + 1,
                        schema_resolution=schema_resolution,
                    )
                )
            return nested_specs
        return [_chunked_prompt_spec(provider, payloads, fallback_id)]

    record = _payload_record(shaped_payload)
    if record is None:
        return []
    if local_agent.looks_like_gemini_cli(record):
        return [_local_agent_document_spec(Provider.GEMINI_CLI, record, fallback_id)]
    if _record_messages(record) is not None:
        return [_generic_messages_spec(provider, record, fallback_id)]
    if chatgpt.looks_like(record):
        return [_single_record_spec(Provider.CHATGPT, record, fallback_id)]
    if _looks_like_chunked_session(record):
        return [_chunked_prompt_spec(provider, record, fallback_id)]
    return []


def _lower_fallback_payload(
    provider: Provider,
    shaped_payload: object,
    fallback_id: str,
) -> list[LoweredPayloadSpec]:
    record = _payload_record(shaped_payload)
    if record is None:
        return []
    if _record_messages(record) is not None:
        return [_generic_messages_spec(provider, record, fallback_id)]
    if chatgpt.looks_like(record):
        return [_single_record_spec(Provider.CHATGPT, record, fallback_id)]
    if _looks_like_chunked_session(record):
        return [_chunked_prompt_spec(provider, record, fallback_id)]
    return []


def _lower_payload_specs(
    provider: str | Provider,
    payload: object,
    fallback_id: str,
    *,
    depth: int = 0,
    schema_resolution: SchemaResolution | None = None,
    source_path: str | None = None,
) -> list[LoweredPayloadSpec]:
    runtime_provider = Provider.from_string(provider)
    if depth > _MAX_PARSE_DEPTH:
        logger.warning("Recursion depth exceeded parsing %s (provider=%s)", fallback_id, provider)
        return []

    shaped_payload = _schema_guided_payload(runtime_provider, payload, schema_resolution)
    record = _payload_record(shaped_payload)
    if record is not None and browser_capture.looks_like(record):
        provider = _detect_provider_from_record(record) or runtime_provider
        return [
            LoweredPayloadSpec(
                provider=provider,
                fallback_id=fallback_id,
                mode="browser_capture",
                payload=record,
            )
        ]
    if record is not None and (sessions := _record_sessions(record)):
        lowered_specs: list[LoweredPayloadSpec] = []
        for index, item in enumerate(sessions):
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

    if runtime_provider in BUNDLE_PROVIDERS:
        return _lower_bundle_payload(runtime_provider, shaped_payload, fallback_id)
    if runtime_provider in {Provider.CLAUDE_CODE, Provider.CODEX}:
        return _lower_grouped_payload(runtime_provider, shaped_payload, fallback_id)
    if runtime_provider is Provider.GEMINI_CLI:
        record = _payload_record(shaped_payload)
        if record is not None and local_agent.looks_like_gemini_cli(record):
            return [_local_agent_document_spec(runtime_provider, record, fallback_id)]
        return []
    if runtime_provider in DRIVE_LIKE_PROVIDERS:
        return _lower_drive_like_payload(
            runtime_provider,
            shaped_payload,
            fallback_id,
            depth=depth,
            schema_resolution=schema_resolution,
        )
    if runtime_provider is Provider.HERMES:
        record = _payload_record(shaped_payload)
        if record is not None and local_agent.looks_like_hermes(record):
            return [_local_agent_document_spec(runtime_provider, record, fallback_id)]
        return []
    if runtime_provider is Provider.ANTIGRAVITY:
        record = _payload_record(shaped_payload)
        if record is not None and (
            antigravity.looks_like_markdown_export(record) or antigravity.looks_like_brain_metadata(record, source_path)
        ):
            return [
                _local_artifact_document_spec(
                    runtime_provider,
                    record,
                    fallback_id,
                    source_path=source_path,
                )
            ]
        return []
    return _lower_fallback_payload(runtime_provider, shaped_payload, fallback_id)


def _generic_messages_session(
    provider: Provider,
    payload: PayloadRecord,
    fallback_id: str,
) -> ParsedSession | None:
    messages_payload = _record_messages(payload)
    if messages_payload is None:
        return None

    messages = extract_messages_from_list(messages_payload)
    title = optional_string(payload.get("title")) or optional_string(payload.get("name")) or fallback_id
    session_id = optional_string(payload.get("id")) or fallback_id
    return ParsedSession(
        source_name=provider,
        provider_session_id=session_id,
        title=title,
        created_at=None,
        updated_at=None,
        messages=messages,
    )


def _parse_lowered_spec(spec: LoweredPayloadSpec) -> list[ParsedSession]:
    if spec.mode == "browser_capture":
        record = _payload_record(spec.payload)
        return [browser_capture.parse(record, spec.fallback_id)] if record is not None else []

    if spec.provider is Provider.CHATGPT:
        record = _payload_record(spec.payload)
        return [chatgpt.parse(record, spec.fallback_id)] if record is not None else []

    if spec.provider is Provider.CLAUDE_AI:
        record = _payload_record(spec.payload)
        return [claude.parse_ai(record, spec.fallback_id)] if record is not None else []

    if spec.provider is Provider.CLAUDE_CODE:
        payloads = _payload_sequence(spec.payload)
        return [claude.parse_code(payloads, spec.fallback_id)] if payloads is not None else []

    if spec.provider is Provider.CODEX:
        payloads = _payload_sequence(spec.payload)
        return [codex.parse(payloads, spec.fallback_id)] if payloads is not None else []

    if spec.mode == "local_agent_document":
        record = _payload_record(spec.payload)
        if record is None:
            return []
        if spec.provider is Provider.GEMINI_CLI:
            return [local_agent.parse_gemini_cli(record, spec.fallback_id)]
        if spec.provider is Provider.HERMES:
            return [local_agent.parse_hermes(record, spec.fallback_id)]
        return []

    if spec.mode == "local_artifact_document":
        record = _payload_record(spec.payload)
        if record is None:
            return []
        if spec.provider is Provider.ANTIGRAVITY:
            if antigravity.looks_like_markdown_export(record):
                return [antigravity.parse_markdown_export_payload(record, spec.fallback_id)]
            source_path = Path(spec.source_path) if spec.source_path is not None else Path(f"{spec.fallback_id}.md")
            return [antigravity.parse_brain_metadata(record, source_path, spec.fallback_id)]
        return []

    if spec.mode == "chunked_prompt":
        record = _payload_record(spec.payload)
        payload = record if record is not None else {"chunks": _payload_sequence(spec.payload) or []}
        return [drive.parse_chunked_prompt(spec.provider, payload, spec.fallback_id)]

    if spec.mode == "generic_messages":
        record = _payload_record(spec.payload)
        generic = _generic_messages_session(spec.provider, record, spec.fallback_id) if record is not None else None
        return [generic] if generic is not None else []

    return []


def parse_payload(
    provider: str | Provider,
    payload: object,
    fallback_id: str,
    _depth: int = 0,
    *,
    schema_resolution: SchemaResolution | None = None,
    source_path: str | None = None,
) -> list[ParsedSession]:
    """Dispatch parsed payload to the appropriate provider parser."""
    lowered_specs = _lower_payload_specs(
        provider,
        payload,
        fallback_id,
        depth=_depth,
        schema_resolution=schema_resolution,
        source_path=source_path,
    )
    sessions: list[ParsedSession] = []
    for spec in lowered_specs:
        sessions.extend(_parse_lowered_spec(spec))
    return sessions


def parse_stream_payload(
    provider: str | Provider,
    payloads: Iterable[object],
    fallback_id: str,
) -> list[ParsedSession]:
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
) -> list[ParsedSession]:
    """Adapter for Drive/Gemini payload parsing."""
    runtime_provider = Provider.from_string(provider)
    if _depth > _MAX_PARSE_DEPTH:
        logger.warning("Recursion depth exceeded parsing drive payload %s", fallback_id)
        return []

    payloads = _payload_sequence(payload)
    if payloads is not None:
        if payloads and all(isinstance(item, str) or _payload_record(item) is not None for item in payloads):
            first_record = _payload_record(payloads[0]) if payloads else None
            if first_record is None or "role" in first_record or "text" in first_record:
                spec = _chunked_prompt_spec(runtime_provider, payloads, fallback_id)
                return _parse_lowered_spec(spec)

        nested_sessions: list[ParsedSession] = []
        for index, item in enumerate(payloads):
            if _looks_like_chunked_session(item):
                nested_sessions.extend(
                    parse_drive_payload(
                        runtime_provider,
                        item,
                        f"{fallback_id}-{index}",
                        _depth + 1,
                    )
                )
                continue
            detected = detect_provider(item) or runtime_provider
            nested_sessions.extend(
                parse_payload(
                    detected,
                    item,
                    f"{fallback_id}-{index}",
                    _depth + 1,
                )
            )
        return nested_sessions

    record = _payload_record(payload)
    if record is None:
        return []
    if "chunkedPrompt" in record or "chunks" in record:
        spec = _chunked_prompt_spec(runtime_provider, record, fallback_id)
        return _parse_lowered_spec(spec)

    detected = detect_provider(record) or runtime_provider
    return parse_payload(detected, record, fallback_id, _depth + 1)


__all__ = [
    "GROUP_PROVIDERS",
    "STREAM_RECORD_PROVIDERS",
    "LoweredPayloadSpec",
    "_detect_provider_from_raw_bytes",
    "detect_provider",
    "parse_drive_payload",
    "parse_payload",
    "parse_stream_payload",
]
