"""Provider detection and payload lowering for source parsing."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument, JSONValue, is_json_document, is_json_value, normalize_json_decimal
from polylogue.core.payload_coercion import optional_string
from polylogue.logging import get_logger

from .decoders import _decode_json_bytes, _iter_json_stream
from .parsers import (
    antigravity,
    beads,
    browser_capture,
    chatgpt,
    claude,
    codex,
    drive,
    hermes_spans,
    hermes_state,
    hermes_verification,
    local_agent,
)
from .parsers.base import ParsedSession, extract_messages_from_list

if TYPE_CHECKING:
    from polylogue.schemas.packages import SchemaResolution

logger = get_logger(__name__)

BUNDLE_PROVIDERS = frozenset({Provider.CHATGPT, Provider.CLAUDE_AI})
GROUP_PROVIDERS = frozenset(
    {Provider.CLAUDE_CODE, Provider.CODEX, Provider.GEMINI, Provider.DRIVE, Provider.BEADS, Provider.HERMES}
)
STREAM_RECORD_PROVIDERS = frozenset({Provider.CLAUDE_CODE, Provider.CODEX, Provider.BEADS, Provider.HERMES})
DRIVE_LIKE_PROVIDERS = frozenset({Provider.GEMINI, Provider.DRIVE})
# The explicit record-shape branch order below remains the production
# implementation during the OriginSpec migration.  OriginSpec validates its
# declared detector tightness against this projection so a new declaration
# cannot silently contradict a stronger existing detector.
RECORD_DETECTOR_PROVIDER_ORDER = (
    Provider.GEMINI_CLI,
    Provider.HERMES,
    Provider.ANTIGRAVITY,
    Provider.BEADS,
    Provider.CODEX,
    Provider.CLAUDE_CODE,
    Provider.CHATGPT,
    Provider.CLAUDE_AI,
    Provider.GEMINI,
)
_MAX_PARSE_DEPTH = 10
_NO_LOOKAHEAD = object()

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


def _single_document_record(value: object) -> PayloadRecord | None:
    """Resolve a single JSON document, unwrapping a one-element sequence.

    Document-style providers (gemini-cli, hermes, antigravity) store one JSON
    object per file. The full-ingest path passes parsed payloads as a list
    (``list(_iter_json_stream(...))``), so a one-record file arrives here as a
    single-element list rather than a bare dict. ``_payload_record`` returns
    ``None`` for a list, which previously made these branches yield no sessions
    and marked the file as a permanent parse failure (perpetual retry).
    """
    record = _payload_record(value)
    if record is not None:
        return record
    sequence = _payload_sequence(value)
    if sequence is not None and len(sequence) == 1:
        return _payload_record(sequence[0])
    return None


def _record_messages(record: PayloadRecord) -> list[JSONValue] | None:
    messages = record.get("messages")
    return messages if isinstance(messages, list) else None


def _record_sessions(record: PayloadRecord) -> list[JSONValue] | None:
    sessions = record.get("sessions")
    return sessions if isinstance(sessions, list) else None


def is_jsonl_source_path(source_path: str | None) -> bool:
    """Return whether a path is a JSONL/NDJSON source path."""
    normalized_path = (source_path or "").lower()
    return normalized_path.endswith((".jsonl", ".jsonl.txt", ".ndjson")) or any(
        marker in normalized_path for marker in (".jsonl.", ".ndjson.")
    )


def is_stream_record_provider(source_path: str | None, provider: str | Provider | None) -> bool:
    """Return whether a source/provider pair should use stream-record parsing."""
    if provider is None:
        return False
    if not is_jsonl_source_path(source_path):
        return False
    return Provider.from_string(provider) in STREAM_RECORD_PROVIDERS


def _looks_like_gemini_mapping(record: PayloadRecord) -> bool:
    return drive.looks_like(record)


def _detect_provider_from_record(record: PayloadRecord) -> Provider | None:
    if browser_capture.looks_like(record):
        session = record.get("session")
        provider = session.get("provider") if isinstance(session, dict) else None
        return Provider.from_string(provider if isinstance(provider, str) else None)
    # Local-agent JSON session documents share enough generic message keys with
    # Claude Code that they must be recognized before broader validators.
    if local_agent.looks_like_gemini_cli(record):
        return Provider.GEMINI_CLI
    if hermes_state.looks_like_state_db_payload(record):
        return Provider.HERMES
    if hermes_verification.looks_like_verification_evidence_db_payload(record):
        return Provider.HERMES
    if hermes_spans.looks_like_atif_payload(record):
        return Provider.HERMES
    if hermes_spans.looks_like_atof_payload(record):
        return Provider.HERMES
    if local_agent.looks_like_hermes(record):
        return Provider.HERMES
    if antigravity.looks_like_markdown_export(record):
        return Provider.ANTIGRAVITY
    if antigravity.looks_like_brain_metadata(record, None):
        return Provider.ANTIGRAVITY
    if beads.looks_like(record):
        return Provider.BEADS
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
        # A one-document Gemini CLI JSON file reaches detection as a
        # one-element sequence on the stream path. Preserve the established
        # Claude-Code-before-Codex sequence ordering while restoring this
        # stronger local-session discriminator ahead of weaker family shapes.
        if len(payloads) == 1 and local_agent.looks_like_gemini_cli(first_record):
            return Provider.GEMINI_CLI
        if browser_capture.looks_like(first_record):
            return _detect_provider_from_record(first_record)
        if hermes_spans.looks_like_atof_payload(first_record):
            return Provider.HERMES
        if beads.looks_like(first_record):
            return Provider.BEADS
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
    return record is not None and drive.has_chunk_container(record)


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
    *,
    source_path: str | None = None,
) -> LoweredPayloadSpec:
    return LoweredPayloadSpec(
        provider=provider,
        fallback_id=fallback_id,
        mode="grouped_records",
        payload=payload,
        source_path=source_path,
    )


def _claude_code_grouped_record_specs(payloads: PayloadSequence, fallback_id: str) -> list[LoweredPayloadSpec]:
    """Split concatenated Claude Code JSONL aggregates into session streams."""
    current_session_id: str | None = None
    groups: dict[str, PayloadSequence] = {}
    pending_prefix: PayloadSequence = []

    for payload in payloads:
        record = _payload_record(payload)
        session_id = optional_string(record.get("sessionId")) if record is not None else None
        if session_id is None:
            if current_session_id is None:
                pending_prefix.append(payload)
            else:
                groups.setdefault(current_session_id, []).append(payload)
            continue

        if current_session_id is None:
            groups.setdefault(session_id, []).extend(pending_prefix)
            pending_prefix = []

        current_session_id = session_id
        groups.setdefault(session_id, []).append(payload)

    if len(groups) <= 1:
        return [_grouped_records_spec(Provider.CLAUDE_CODE, payloads, fallback_id)]
    return [
        _grouped_records_spec(
            Provider.CLAUDE_CODE,
            group_payloads,
            fallback_id if index == 0 else group_id,
        )
        for index, (group_id, group_payloads) in enumerate(groups.items())
    ]


def merge_parsed_session_chunks(sessions: Iterable[ParsedSession]) -> list[ParsedSession]:
    """Merge repeated provider-native sessions produced by streaming chunks."""

    merged: dict[str, ParsedSession] = {}
    for session in sessions:
        existing = merged.get(session.provider_session_id)
        if existing is None:
            merged[session.provider_session_id] = session
            continue

        messages = [*existing.messages, *session.messages]
        active_leaf_message_provider_id = messages[-1].provider_message_id if messages else None
        if active_leaf_message_provider_id is not None:
            messages = [
                message.model_copy(
                    update={
                        "position": position,
                        "is_active_leaf": message.provider_message_id == active_leaf_message_provider_id,
                    }
                )
                for position, message in enumerate(messages)
            ]

        reported_cost_usd: float | None
        if existing.reported_cost_usd is None and session.reported_cost_usd is None:
            reported_cost_usd = None
        else:
            reported_cost_usd = (existing.reported_cost_usd or 0.0) + (session.reported_cost_usd or 0.0)

        reported_duration_ms: int | None
        if existing.reported_duration_ms is None and session.reported_duration_ms is None:
            reported_duration_ms = None
        else:
            reported_duration_ms = (existing.reported_duration_ms or 0) + (session.reported_duration_ms or 0)

        created_values = [value for value in (existing.created_at, session.created_at) if value]
        updated_values = [value for value in (existing.updated_at, session.updated_at) if value]
        merged[session.provider_session_id] = existing.model_copy(
            update={
                "title": existing.title if existing.title != existing.provider_session_id else session.title,
                "created_at": min(created_values) if created_values else None,
                "parent_session_provider_id": (
                    existing.parent_session_provider_id or session.parent_session_provider_id
                ),
                "branch_type": existing.branch_type or session.branch_type,
                "updated_at": max(updated_values) if updated_values else None,
                "messages": messages,
                "active_leaf_message_provider_id": active_leaf_message_provider_id,
                "attachments": [*existing.attachments, *session.attachments],
                "session_events": [*existing.session_events, *session.session_events],
                "reported_cost_usd": reported_cost_usd,
                "reported_duration_ms": reported_duration_ms,
                "models_used": sorted({*existing.models_used, *session.models_used}),
                "working_directories": sorted({*existing.working_directories, *session.working_directories}),
                "ingest_flags": sorted({*existing.ingest_flags, *session.ingest_flags}),
            }
        )
    sessions = list(merged.values())
    return [
        claude.reconcile_code_session_chunks(session) if session.source_name is Provider.CLAUDE_CODE else session
        for session in sessions
    ]


def _claude_code_stream_sessions(payloads: Iterable[object], fallback_id: str) -> Iterator[ParsedSession]:
    """Parse Claude Code JSONL records without materializing the full stream.

    The eager ``parse_payload`` path preserves the strongest non-contiguous
    grouping semantics for already-materialized payloads. Raw JSONL ingest and
    repair, however, can be multi-GiB; for that path we split only on contiguous
    ``sessionId`` changes and feed each group to the provider parser as an
    iterator. Per-session record-index and UUID continuation state retains no
    raw payload bytes; its size is proportional to unique record identifiers and
    makes an interleaved stream semantically identical to eager grouping.
    """

    iterator = iter(payloads)
    lookahead: object = _NO_LOOKAHEAD
    pending_prefix: list[object] = []
    first_group = True
    record_counts_by_session: dict[str, int] = {}
    seen_record_uuids_by_session: dict[str, set[str]] = {}

    def next_item() -> object:
        nonlocal lookahead
        if lookahead is not _NO_LOOKAHEAD:
            item = lookahead
            lookahead = _NO_LOOKAHEAD
            return item
        return next(iterator)

    while True:
        try:
            first = next_item()
        except StopIteration:
            if pending_prefix:
                yield claude.parse_code_stream(iter(pending_prefix), fallback_id)
            return

        first_record = _payload_record(first)
        first_session_id = optional_string(first_record.get("sessionId")) if first_record is not None else None
        if first_session_id is None:
            pending_prefix.append(first)
            continue

        group_session_id = first_session_id
        group_fallback_id = fallback_id if first_group else group_session_id
        first_group = False
        prefix = pending_prefix
        pending_prefix = []

        group_record_count = 0

        def group_records(
            prefix: list[object] = prefix,
            first: object = first,
            group_session_id: str = group_session_id,
        ) -> Iterator[object]:
            nonlocal group_record_count, lookahead
            for prefix_item in prefix:
                group_record_count += 1
                yield prefix_item
            group_record_count += 1
            yield first
            for item in iterator:
                record = _payload_record(item)
                session_id = optional_string(record.get("sessionId")) if record is not None else None
                if session_id is not None and session_id != group_session_id:
                    lookahead = item
                    return
                group_record_count += 1
                yield item

        record_index_start = record_counts_by_session.get(group_session_id, 0)
        seen_record_uuids = seen_record_uuids_by_session.setdefault(group_session_id, set())
        session = claude.parse_code_stream(
            group_records(),
            group_fallback_id,
            record_index_start=record_index_start,
            seen_record_uuids=seen_record_uuids,
        )
        record_counts_by_session[group_session_id] = record_index_start + group_record_count
        yield session


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
        if provider is Provider.CLAUDE_CODE:
            return _claude_code_grouped_record_specs(payloads, fallback_id)
        return [_grouped_records_spec(provider, payloads, fallback_id)]

    record = _payload_record(shaped_payload)
    if record is None:
        return []

    messages = _record_messages(record)
    grouped_payload: PayloadSequence = messages if messages is not None else [record]
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
        if (
            payloads
            and any(drive.looks_like_chunk(item) for item in payloads)
            and not any(_looks_like_chunked_session(item) for item in payloads)
        ):
            return [_chunked_prompt_spec(provider, payloads, fallback_id)]
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
        # Drive exports and full-ingest streams can add one or more list/document
        # wrappers around session records. Recurse through those containers, but
        # never reinterpret arbitrary records as raw chunks: that would revive
        # the loose ``chunks`` detector this route is meant to replace.
        nested_specs = []
        for index, item in enumerate(payloads):
            nested_specs.extend(
                _lower_payload_specs(
                    provider,
                    item,
                    fallback_id if len(payloads) == 1 else f"{fallback_id}-{index}",
                    depth=depth + 1,
                    schema_resolution=schema_resolution,
                )
            )
        return nested_specs

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
    sequence = _payload_sequence(shaped_payload)
    if sequence:
        browser_capture_specs: list[LoweredPayloadSpec] = []
        for index, item in enumerate(sequence):
            item_record = _payload_record(item)
            if item_record is None or not browser_capture.looks_like(item_record):
                browser_capture_specs = []
                break
            provider = _detect_provider_from_record(item_record) or runtime_provider
            browser_capture_specs.append(
                LoweredPayloadSpec(
                    provider=provider,
                    fallback_id=fallback_id if len(sequence) == 1 else f"{fallback_id}-{index}",
                    mode="browser_capture",
                    payload=item_record,
                )
            )
        if browser_capture_specs:
            return browser_capture_specs
    if record is not None and (sessions := _record_sessions(record)):
        lowered_specs: list[LoweredPayloadSpec] = []
        for index, item in enumerate(sessions):
            lowered_specs.extend(
                _lower_payload_specs(
                    runtime_provider,
                    item,
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
    if runtime_provider is Provider.BEADS:
        payloads = _payload_sequence(shaped_payload)
        if payloads is not None and all(
            (record := _payload_record(item)) is not None and beads.looks_like(record) for item in payloads
        ):
            return [_grouped_records_spec(runtime_provider, payloads, fallback_id, source_path=source_path)]
        record = _single_document_record(shaped_payload)
        if record is not None and beads.looks_like(record):
            return [_grouped_records_spec(runtime_provider, [record], fallback_id, source_path=source_path)]
        return []
    if runtime_provider is Provider.GEMINI_CLI:
        record = _single_document_record(shaped_payload)
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
        payloads = _payload_sequence(shaped_payload)
        if (
            payloads is not None
            and payloads
            and all(
                (event := _payload_record(item)) is not None and hermes_spans.looks_like_atof_payload(event)
                for item in payloads
            )
        ):
            return [_grouped_records_spec(runtime_provider, payloads, fallback_id, source_path=source_path)]
        record = _single_document_record(shaped_payload)
        if record is not None and hermes_state.looks_like_state_db_payload(record):
            return [_local_artifact_document_spec(runtime_provider, record, fallback_id, source_path=source_path)]
        if record is not None and hermes_verification.looks_like_verification_evidence_db_payload(record):
            return [_local_artifact_document_spec(runtime_provider, record, fallback_id, source_path=source_path)]
        if record is not None and hermes_spans.looks_like_atif_payload(record):
            return [_local_artifact_document_spec(runtime_provider, record, fallback_id, source_path=source_path)]
        if record is not None and local_agent.looks_like_hermes(record):
            return [_local_agent_document_spec(runtime_provider, record, fallback_id)]
        return []
    if runtime_provider is Provider.ANTIGRAVITY:
        record = _single_document_record(shaped_payload)
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
    created_at = optional_string(
        payload.get("created_at") or payload.get("create_time") or payload.get("created") or payload.get("createdAt")
    )
    updated_at = optional_string(
        payload.get("updated_at")
        or payload.get("update_time")
        or payload.get("updated")
        or payload.get("updatedAt")
        or payload.get("modified")
    )
    return ParsedSession(
        source_name=provider,
        provider_session_id=session_id,
        title=title,
        created_at=created_at,
        updated_at=updated_at,
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

    if spec.provider is Provider.BEADS:
        payloads = _payload_sequence(spec.payload)
        return beads.parse(payloads, spec.fallback_id, source_path=spec.source_path) if payloads is not None else []

    if spec.provider is Provider.HERMES and spec.mode == "grouped_records":
        payloads = _payload_sequence(spec.payload)
        return hermes_spans.parse_atof_stream(payloads, spec.fallback_id) if payloads is not None else []

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
        if spec.provider is Provider.HERMES and hermes_state.looks_like_state_db_payload(record):
            return hermes_state.parse_state_db_payload(record, spec.fallback_id)
        if spec.provider is Provider.HERMES and hermes_verification.looks_like_verification_evidence_db_payload(record):
            return hermes_verification.parse_verification_evidence_db_payload(record, spec.fallback_id)
        if spec.provider is Provider.HERMES and hermes_spans.looks_like_atif_payload(record):
            return [hermes_spans.parse_atif_document(record, spec.fallback_id)]
        if spec.provider is Provider.ANTIGRAVITY:
            if antigravity.looks_like_markdown_export(record):
                return [antigravity.parse_markdown_export_payload(record, spec.fallback_id)]
            source_path = Path(spec.source_path) if spec.source_path is not None else Path(f"{spec.fallback_id}.md")
            return [antigravity.parse_brain_metadata(record, source_path, spec.fallback_id)]
        return []

    if spec.mode == "chunked_prompt":
        record = _payload_record(spec.payload)
        payload: JSONDocument = record if record is not None else {"chunks": _payload_sequence(spec.payload) or []}
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
    *,
    source_path: str | None = None,
) -> list[ParsedSession]:
    """Parse a grouped record stream."""
    runtime_provider = Provider.from_string(provider)
    if runtime_provider is Provider.CLAUDE_CODE:
        return merge_parsed_session_chunks(_claude_code_stream_sessions(payloads, fallback_id))
    if runtime_provider is Provider.CODEX:
        return [codex.parse_stream(payloads, fallback_id)]
    if runtime_provider is Provider.BEADS:
        return beads.parse(payloads, fallback_id, source_path=source_path)
    if runtime_provider is Provider.HERMES:
        return hermes_spans.parse_atof_stream(payloads, fallback_id)
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
    "is_jsonl_source_path",
    "is_stream_record_provider",
    "parse_drive_payload",
    "parse_payload",
    "parse_stream_payload",
]
