"""Unified subprocess worker: decode → validate → parse → transform in one pass.

Runs inside ProcessPoolExecutor. Returns plain tuples for direct SQL executemany,
avoiding Pydantic serialization overhead across the process boundary.

Performance: eliminates double blob decode (was: validate decodes, then parse decodes
the same blob again). Moves transform into subprocess for true parallelism.
"""

from __future__ import annotations

import pickle
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

from typing_extensions import TypedDict

from polylogue.archive.artifact_taxonomy import ArtifactClassification, ArtifactKind, classify_artifact
from polylogue.archive.artifact_taxonomy.support import is_subagent_path
from polylogue.archive.conversation.branch_type import BranchType
from polylogue.archive.message.roles import Role
from polylogue.archive.raw_payload.decode import RawPayloadEnvelope
from polylogue.core.common import format_malformed_jsonl_error as _format_malformed_jsonl_error
from polylogue.core.json import dumps as json_dumps
from polylogue.logging import get_logger
from polylogue.pipeline.materialization_runtime import (
    MaterializedContentBlock,
    MaterializedConversation,
    MaterializedMessage,
    materialize_conversation,
)
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import STREAM_RECORD_PROVIDERS
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.runtime import (
    ACTION_EVENT_MATERIALIZER_VERSION,
    PROVIDER_EVENT_MATERIALIZER_VERSION,
    RawConversationRecord,
    _json_or_none,
)
from polylogue.storage.sqlite.provider_event_model import project_provider_event_payload
from polylogue.types import (
    AttachmentId,
    ContentBlockType,
    ContentHash,
    ConversationId,
    MessageId,
    Provider,
    ProviderEventId,
    SemanticBlockType,
    ValidationMode,
    ValidationStatus,
)

if TYPE_CHECKING:
    from polylogue.archive.action_event.action_events import ActionEvent
    from polylogue.schemas.packages import SchemaResolution
    from polylogue.schemas.runtime_registry import SchemaRegistry
    from polylogue.sources.parsers.base import ParsedConversation


logger = get_logger(__name__)
_SOURCE_HASH_SUFFIX = re.compile(r"-(?:[0-9a-f]{16,64})$", re.IGNORECASE)
_SCHEMA_REGISTRY: SchemaRegistry | None = None


class _TimestampUpdates(TypedDict, total=False):
    created_at: str
    updated_at: str


ConversationTuple = tuple[
    ConversationId,
    str,
    str,
    str | None,
    str | None,
    str | None,
    float | None,
    ContentHash,
    str | None,
    str,
    int,
    ConversationId | None,
    BranchType | None,
    str | None,
    str,  # source_name
    str | None,  # working_directories_json
    str | None,  # git_branch
    str | None,  # git_repository_url
]
MessageTuple = tuple[
    MessageId,
    ConversationId,
    str,
    Role,
    str | None,
    float | None,
    ContentHash,
    int,
    MessageId | None,
    int,
    str | Provider,
    int,
    int,
    int,
    int,
    int,  # input_tokens
    int,  # output_tokens
    int,  # cache_read_tokens
    int,  # cache_write_tokens
    str | None,  # model_name
    str,  # message_type
]
ContentBlockTuple = tuple[
    str,
    MessageId,
    ConversationId,
    int,
    ContentBlockType,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    SemanticBlockType | str | None,
]
ActionEventTuple = tuple[
    str,
    str,
    str,
    int,
    str | None,
    str | None,
    float | None,
    int,
    str,
    str,
    str | None,
    str,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str,
]
ProviderEventTuple = tuple[
    ProviderEventId,
    ConversationId,
    Provider,
    int,
    str,
    str,
    str | None,
    float | None,
    dict[str, object],
    MessageId | None,
    str | None,
    int,
]
StatsTuple = tuple[ConversationId, str, int, int, int, int, int]
AttachmentTuple = tuple[
    AttachmentId,
    str | None,
    int | None,
    str | None,
    int,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
]
AttachmentRefTuple = tuple[
    str,
    AttachmentId,
    ConversationId,
    MessageId | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
]


# ---------------------------------------------------------------------------
# Result dataclasses — cheap to pickle (no Pydantic)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ConversationData:
    """All DB-ready data for one conversation, as plain tuples for executemany."""

    conversation_id: str
    content_hash: str

    # Tuple matching INSERT INTO conversations column order
    conversation_tuple: ConversationTuple

    # list[tuple] matching INSERT INTO messages column order
    message_tuples: list[MessageTuple] = field(default_factory=list)

    # list[tuple] matching INSERT INTO content_blocks column order
    block_tuples: list[ContentBlockTuple] = field(default_factory=list)

    # list[tuple] matching INSERT INTO action_events column order
    action_event_tuples: list[ActionEventTuple] = field(default_factory=list)

    # list[tuple] matching INSERT INTO provider_events column order
    provider_event_tuples: list[ProviderEventTuple] = field(default_factory=list)

    # (conversation_id, source_name, msg_count, word_count, tool_use_count, thinking_count)
    stats_tuple: StatsTuple | tuple[()] = ()

    # Attachments are rare; keep as list of simple tuples
    # Each: (attachment_id, conversation_id, message_id, mime_type, size_bytes, path, provider_meta_json)
    attachment_tuples: list[AttachmentTuple] = field(default_factory=list)
    attachment_ref_tuples: list[AttachmentRefTuple] = field(default_factory=list)

    # Source metadata
    source_name: str = ""
    raw_id: str | None = None
    append_only: bool = False


@dataclass(slots=True)
class IngestRecordResult:
    """Result from processing one raw record in a subprocess."""

    raw_id: str
    payload_provider: str | None = None
    validation_status: str = "skipped"  # ValidationStatus value
    validation_error: str | None = None
    parse_error: str | None = None
    error: str | None = None
    conversations: list[ConversationData] = field(default_factory=list)
    source_name: str | None = None
    serialized_size_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class _ActionEventMessage:
    id: str
    timestamp: datetime | None


ParsePlanMode = Literal["payload", "stream"]


@dataclass(frozen=True, slots=True)
class _IngestContext:
    raw_record: RawConversationRecord
    raw_source: Path
    archive_root: Path
    validation_mode: ValidationMode
    measure_serialized_size: bool
    source_name: str
    fallback_timestamp: str | None


@dataclass(frozen=True, slots=True)
class _ParsePlan:
    provider: Provider
    payload_provider: str
    artifact: ArtifactClassification
    mode: ParsePlanMode
    schema_payload: object | None
    schema_resolution: SchemaResolution | None = None
    payload: object | None = None
    stream_name: str | None = None
    malformed_jsonl_lines: int = 0
    malformed_jsonl_detail: str | None = None


@dataclass(frozen=True, slots=True)
class _PlanValidation:
    status: ValidationStatus
    validation_error: str | None = None
    parse_error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fallback_id(source_path: str | None, raw_id: str) -> str:
    if not source_path:
        return raw_id
    normalized = source_path.replace("\\", "/")
    entry_path = normalized.rsplit(":", 1)[-1]
    stem = Path(entry_path).stem
    if not stem:
        return raw_id
    cleaned = _SOURCE_HASH_SUFFIX.sub("", stem).strip("._- ")
    return cleaned or stem


def _make_ref_id(
    attachment_id: str,
    conversation_id: str,
    message_id: str | None,
) -> str:
    from hashlib import sha256

    key = f"{attachment_id}:{conversation_id}:{message_id or ''}"
    return sha256(key.encode()).hexdigest()[:32]


def _runtime_schema_registry() -> SchemaRegistry:
    global _SCHEMA_REGISTRY
    if _SCHEMA_REGISTRY is None:
        from polylogue.schemas.runtime_registry import SchemaRegistry

        _SCHEMA_REGISTRY = SchemaRegistry()
    assert _SCHEMA_REGISTRY is not None
    return _SCHEMA_REGISTRY


def _finalize_result(result: IngestRecordResult, *, measure_serialized_size: bool) -> IngestRecordResult:
    if not measure_serialized_size:
        return result
    result.serialized_size_bytes = len(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
    return result


def _normalized_conversation(
    convo: ParsedConversation,
    *,
    fallback_timestamp: str | None,
) -> ParsedConversation:
    updates: _TimestampUpdates = {}
    if convo.created_at is None and fallback_timestamp:
        updates["created_at"] = fallback_timestamp
    effective_created = updates.get("created_at", convo.created_at)
    if convo.updated_at is None and isinstance(effective_created, str) and effective_created:
        updates["updated_at"] = effective_created
    return convo.model_copy(update=updates) if updates else convo


def _is_stream_record_provider(source_path: str | None, provider: str | Provider | None) -> bool:
    if provider is None:
        return False
    if not _is_jsonl_source_path(source_path):
        return False
    return Provider.from_string(provider) in STREAM_RECORD_PROVIDERS


def _is_jsonl_source_path(source_path: str | None) -> bool:
    return (source_path or "").lower().endswith((".jsonl", ".jsonl.txt", ".ndjson"))


def _record_result(
    context: _IngestContext,
    payload_provider: str | None,
    *,
    validation_status: ValidationStatus,
    validation_error: str | None = None,
    parse_error: str | None = None,
    error: str | None = None,
    conversations: list[ConversationData] | None = None,
    include_source_name: bool = False,
) -> IngestRecordResult:
    return _finalize_result(
        IngestRecordResult(
            raw_id=context.raw_record.raw_id,
            payload_provider=payload_provider,
            validation_status=validation_status.value,
            validation_error=validation_error,
            parse_error=parse_error,
            error=error,
            conversations=conversations or [],
            source_name=context.source_name if include_source_name else None,
        ),
        measure_serialized_size=context.measure_serialized_size,
    )


def _schema_payload_for_artifact(
    *,
    artifact: ArtifactClassification,
    payload: object,
) -> object | None:
    return payload if artifact.schema_eligible else None


def _resolve_plan_schema(
    *,
    provider: Provider,
    schema_payload: object | None,
    source_path: str | None,
) -> SchemaResolution | None:
    if schema_payload is None:
        return None
    return _runtime_schema_registry().resolve_payload(
        provider,
        schema_payload,
        source_path=source_path,
    )


def _build_parse_plan(
    *,
    provider: Provider,
    payload_provider: str,
    artifact: ArtifactClassification,
    source_path: str | None,
    mode: ParsePlanMode,
    payload: object | None = None,
    schema_payload_source: object,
    stream_name: str | None = None,
    malformed_jsonl_lines: int = 0,
    malformed_jsonl_detail: str | None = None,
) -> _ParsePlan:
    schema_payload = _schema_payload_for_artifact(
        artifact=artifact,
        payload=schema_payload_source,
    )
    return _ParsePlan(
        provider=provider,
        payload_provider=payload_provider,
        artifact=artifact,
        mode=mode,
        schema_payload=schema_payload,
        schema_resolution=_resolve_plan_schema(
            provider=provider,
            schema_payload=schema_payload,
            source_path=source_path,
        ),
        payload=payload,
        stream_name=stream_name,
        malformed_jsonl_lines=malformed_jsonl_lines,
        malformed_jsonl_detail=malformed_jsonl_detail,
    )


def _build_stream_parse_plan(
    context: _IngestContext,
    *,
    payload_provider: str | None,
) -> _ParsePlan | None:
    from polylogue.archive.raw_payload.decode import _sample_jsonl_payload_with_detail
    from polylogue.sources.dispatch import detect_provider

    stream_name = context.raw_record.source_path or context.raw_record.raw_id

    try:
        sample_payloads, malformed_lines, malformed_detail = _sample_jsonl_payload_with_detail(
            context.raw_source,
            max_samples=64,
            jsonl_dict_only=True,
            scan_full=context.validation_mode is ValidationMode.STRICT,
        )
    except Exception:
        # Sampling helper failed entirely (file I/O, decode, or worse). Logging
        # this is critical because the caller falls back to a different parser
        # path on `None`, which can produce different content hashes for the
        # same input depending on whether the helper happened to succeed.
        logger.exception(
            "JSONL sample probe failed for %s; falling back to non-stream parsing",
            stream_name,
        )
        return None

    runtime_provider = Provider.from_string(payload_provider or context.raw_record.source_name)
    if runtime_provider not in STREAM_RECORD_PROVIDERS:
        detected_provider = detect_provider(sample_payloads)
        if detected_provider not in STREAM_RECORD_PROVIDERS:
            return None
        runtime_provider = detected_provider

    artifact = classify_artifact(
        sample_payloads,
        provider=runtime_provider,
        source_path=context.raw_record.source_path,
    )
    return _build_parse_plan(
        provider=runtime_provider,
        payload_provider=str(runtime_provider),
        artifact=artifact,
        source_path=context.raw_record.source_path,
        mode="stream",
        payload=sample_payloads,
        schema_payload_source=sample_payloads,
        stream_name=stream_name,
        malformed_jsonl_lines=malformed_lines,
        malformed_jsonl_detail=malformed_detail,
    )


def _build_fast_stream_parse_plan(
    context: _IngestContext,
    *,
    payload_provider: str | None,
) -> _ParsePlan | None:
    runtime_provider = Provider.from_string(payload_provider or context.raw_record.source_name)
    if runtime_provider not in STREAM_RECORD_PROVIDERS:
        return None

    kind = (
        ArtifactKind.SUBAGENT_CONVERSATION_STREAM
        if is_subagent_path(context.raw_record.source_path)
        else ArtifactKind.CONVERSATION_RECORD_STREAM
    )
    artifact = ArtifactClassification(
        provider=runtime_provider,
        kind=kind,
        parse_as_conversation=True,
        schema_eligible=False,
        default_priority=90 if kind is ArtifactKind.SUBAGENT_CONVERSATION_STREAM else 120,
        reason="known JSONL stream provider with validation off",
    )
    return _build_parse_plan(
        provider=runtime_provider,
        payload_provider=str(runtime_provider),
        artifact=artifact,
        source_path=context.raw_record.source_path,
        mode="stream",
        schema_payload_source=None,
        stream_name=context.raw_record.source_path or context.raw_record.raw_id,
    )


def _build_envelope_parse_plan(
    context: _IngestContext,
    envelope: RawPayloadEnvelope,
) -> _ParsePlan:
    return _build_parse_plan(
        provider=envelope.provider,
        payload_provider=str(envelope.provider),
        artifact=envelope.artifact,
        source_path=context.raw_record.source_path,
        mode="payload",
        payload=envelope.payload,
        schema_payload_source=envelope.payload,
        malformed_jsonl_lines=envelope.malformed_jsonl_lines,
        malformed_jsonl_detail=envelope.malformed_jsonl_detail,
    )


def _validate_parse_plan(
    context: _IngestContext,
    plan: _ParsePlan,
) -> _PlanValidation:
    from polylogue.schemas.validator import SchemaValidator

    if context.validation_mode is ValidationMode.OFF:
        return _PlanValidation(status=ValidationStatus.SKIPPED)
    if not plan.artifact.schema_eligible or plan.schema_payload is None:
        return _PlanValidation(status=ValidationStatus.PASSED)
    if plan.malformed_jsonl_lines and context.validation_mode is ValidationMode.STRICT:
        malformed_error = _format_malformed_jsonl_error(
            malformed_lines=plan.malformed_jsonl_lines,
            malformed_detail=plan.malformed_jsonl_detail,
        )
        return _PlanValidation(
            status=ValidationStatus.FAILED,
            validation_error=malformed_error,
            parse_error=malformed_error,
        )

    try:
        validator = SchemaValidator.for_payload(
            plan.provider,
            plan.schema_payload,
            source_path=context.raw_record.source_path,
            schema_resolution=plan.schema_resolution,
        )
    except (FileNotFoundError, ImportError):
        return _PlanValidation(
            status=ValidationStatus.SKIPPED,
        )

    validation_samples = validator.validation_samples(plan.schema_payload)
    if validation_samples:
        collected_errors: list[str] = []
        for sample in validation_samples:
            sample_result = validator.validate(sample, include_drift=False)
            if not sample_result.is_valid:
                collected_errors.extend(sample_result.errors[:2])
        if collected_errors and context.validation_mode is ValidationMode.STRICT:
            return _PlanValidation(
                status=ValidationStatus.FAILED,
                validation_error=f"Schema validation failed: {collected_errors[0]}",
            )

    return _PlanValidation(
        status=ValidationStatus.PASSED,
    )


def _parse_plan_conversations(
    context: _IngestContext,
    plan: _ParsePlan,
) -> list[ParsedConversation]:
    from polylogue.sources.dispatch import parse_payload, parse_stream_payload

    fallback_id = _fallback_id(context.raw_record.source_path, context.raw_record.raw_id)
    if plan.mode == "stream":
        assert plan.stream_name is not None
        stream_name = plan.stream_name
        with context.raw_source.open("rb") as handle:
            valid_record_count = 0

            def counted_stream() -> Iterable[object]:
                nonlocal valid_record_count
                for item in _iter_json_stream(handle, stream_name):
                    valid_record_count += 1
                    yield item

            conversations = parse_stream_payload(
                plan.provider,
                counted_stream(),
                fallback_id,
            )
            if valid_record_count == 0:
                raise ValueError(f"no valid JSON records in {stream_name}")
            return conversations

    return parse_payload(
        plan.provider,
        plan.payload,
        fallback_id,
        schema_resolution=plan.schema_resolution,
        source_path=context.raw_record.source_path,
    )


def _materialize_parsed_conversations(
    context: _IngestContext,
    plan: _ParsePlan,
    *,
    validation: _PlanValidation,
    parsed_conversations: list[ParsedConversation],
) -> IngestRecordResult:
    result_convos: list[ConversationData] = []
    for convo in parsed_conversations:
        normalized_convo = _normalized_conversation(
            convo,
            fallback_timestamp=context.fallback_timestamp,
        )
        try:
            cdata = _transform_to_tuples(
                normalized_convo,
                source_name=context.source_name,
                archive_root=context.archive_root,
                raw_id=context.raw_record.raw_id,
                append_only=context.raw_record.source_index == -1,
            )
            result_convos.append(cdata)
        except Exception as exc:
            return _record_result(
                context,
                plan.payload_provider,
                validation_status=validation.status,
                parse_error=f"transform: {exc}",
                error=f"transform: {exc}",
            )

    return _record_result(
        context,
        plan.payload_provider,
        validation_status=validation.status,
        validation_error=validation.validation_error,
        conversations=result_convos,
        include_source_name=True,
    )


def _run_parse_plan(
    context: _IngestContext,
    plan: _ParsePlan,
) -> IngestRecordResult:
    if not plan.artifact.parse_as_conversation:
        return _record_result(
            context,
            plan.payload_provider,
            validation_status=ValidationStatus.SKIPPED,
        )

    validation = _validate_parse_plan(context, plan)
    if validation.validation_error is not None:
        return _record_result(
            context,
            plan.payload_provider,
            validation_status=validation.status,
            validation_error=validation.validation_error,
            parse_error=validation.parse_error,
            error=validation.validation_error,
        )

    try:
        parsed_conversations = _parse_plan_conversations(
            context,
            plan,
        )
    except Exception as exc:
        return _record_result(
            context,
            plan.payload_provider,
            validation_status=validation.status,
            parse_error=f"parse: {exc}",
            error=f"parse: {exc}",
        )

    return _materialize_parsed_conversations(
        context,
        plan,
        validation=validation,
        parsed_conversations=parsed_conversations,
    )


# ---------------------------------------------------------------------------
# Main worker function — runs in subprocess
# ---------------------------------------------------------------------------


def ingest_record(
    raw_record: RawConversationRecord,
    archive_root_str: str,
    validation_mode_value: str = "advisory",
    measure_serialized_size: bool = False,
    *,
    blob_root_str: str | None = None,
) -> IngestRecordResult:
    """Decode + validate + parse + transform one raw record in a single pass.

    Returns DB-ready tuples, not Pydantic models. This function runs in a
    subprocess via ProcessPoolExecutor and must be self-contained (no shared
    state, no DB access).
    """
    from polylogue.archive.raw_payload import build_raw_payload_envelope
    from polylogue.paths import blob_store_root

    archive_root = Path(archive_root_str)
    validation_mode = ValidationMode.from_string(validation_mode_value)

    stored_payload_provider = raw_record.payload_provider
    if not isinstance(stored_payload_provider, str) or not stored_payload_provider.strip():
        stored_payload_provider = None

    resolved_blob_root = Path(blob_root_str) if blob_root_str is not None else blob_store_root()
    blob_store = BlobStore(resolved_blob_root)
    raw_source = blob_store.blob_path(raw_record.raw_id)
    context = _IngestContext(
        raw_record=raw_record,
        raw_source=raw_source,
        archive_root=archive_root,
        validation_mode=validation_mode,
        measure_serialized_size=measure_serialized_size,
        source_name=raw_record.source_name or raw_record.source_path or "",
        fallback_timestamp=raw_record.file_mtime,
    )

    if raw_record.blob_size == 0:
        error = "decode: Input is a zero-length, empty document"
        return _record_result(
            context,
            stored_payload_provider,
            validation_status=ValidationStatus.FAILED,
            validation_error=error,
            parse_error=error,
            error=error,
        )

    if _is_stream_record_provider(raw_record.source_path, stored_payload_provider or raw_record.source_name):
        if validation_mode is ValidationMode.OFF and (
            stream_plan := _build_fast_stream_parse_plan(context, payload_provider=stored_payload_provider)
        ):
            return _run_parse_plan(context, stream_plan)
        if stream_plan := _build_stream_parse_plan(context, payload_provider=stored_payload_provider):
            return _run_parse_plan(context, stream_plan)
    elif _is_jsonl_source_path(raw_record.source_path):
        if stream_plan := _build_stream_parse_plan(context, payload_provider=stored_payload_provider):
            return _run_parse_plan(context, stream_plan)

    # ── Phase 1: Decode blob (ONE decode, not two) ────────────────────
    try:
        envelope = build_raw_payload_envelope(
            context.raw_source,
            source_path=raw_record.source_path,
            fallback_provider=raw_record.source_name,
            payload_provider=stored_payload_provider,
        )
    except Exception as exc:
        return _record_result(
            context,
            stored_payload_provider,
            validation_status=ValidationStatus.FAILED,
            validation_error=f"decode: {exc}",
            parse_error=f"decode: {exc}",
            error=f"decode: {exc}",
        )

    return _run_parse_plan(context, _build_envelope_parse_plan(context, envelope))


# ---------------------------------------------------------------------------
# Transform — converts ParsedConversation to plain tuples
# ---------------------------------------------------------------------------


def _conversation_tuple(conversation: MaterializedConversation, *, raw_id: str | None) -> ConversationTuple:
    source_name = ""
    if conversation.provider_meta and isinstance(conversation.provider_meta, dict):
        raw = conversation.provider_meta.get("source")
        source_name = raw if isinstance(raw, str) else ""
    return (
        conversation.conversation_id,
        conversation.source_name,
        conversation.provider_conversation_id,
        conversation.title,
        conversation.created_at,
        conversation.updated_at,
        conversation.sort_key,
        conversation.content_hash,
        _json_or_none(conversation.provider_meta),
        "{}",
        1,
        conversation.parent_conversation_id,
        conversation.branch_type,
        raw_id,
        source_name,
        conversation.working_directories_json,
        conversation.git_branch,
        conversation.git_repository_url,
    )


def _message_tuple(conversation: MaterializedConversation, message: MaterializedMessage) -> MessageTuple:
    return (
        message.message_id,
        conversation.conversation_id,
        message.provider_message_id,
        message.role,
        None if message.blocks else message.text,
        message.sort_key,
        message.content_hash,
        1,
        message.parent_message_id,
        message.branch_index,
        conversation.source_name,
        message.word_count,
        message.has_tool_use,
        message.has_thinking,
        message.has_paste,
        message.input_tokens,
        message.output_tokens,
        message.cache_read_tokens,
        message.cache_write_tokens,
        message.model_name,
        message.message_type.value,
    )


def _message_tuples(conversation: MaterializedConversation) -> list[MessageTuple]:
    return [_message_tuple(conversation, message) for message in conversation.messages]


def _content_block_tuple(
    *,
    conversation_id: ConversationId,
    message_id: MessageId,
    block: MaterializedContentBlock,
) -> ContentBlockTuple:
    return (
        block.block_id,
        message_id,
        conversation_id,
        block.block_index,
        block.type,
        block.text,
        block.tool_name,
        block.tool_id,
        block.tool_input_json,
        block.metadata_json,
        block.semantic_type,
    )


def _content_block_tuples(conversation: MaterializedConversation) -> list[ContentBlockTuple]:
    return [
        _content_block_tuple(
            conversation_id=conversation.conversation_id,
            message_id=message.message_id,
            block=block,
        )
        for message in conversation.messages
        for block in message.blocks
    ]


_TupleT = TypeVar("_TupleT")


def _dedupe_by_key_preserve_last(items: list[_TupleT], key: Callable[[_TupleT], object]) -> list[_TupleT]:
    if len(items) < 2:
        return items
    seen: set[object] = set()
    deduped: list[_TupleT] = []
    for item in reversed(items):
        item_key = key(item)
        if item_key in seen:
            continue
        seen.add(item_key)
        deduped.append(item)
    if len(deduped) == len(items):
        return items
    deduped.reverse()
    return deduped


def _stats_tuple(conversation: MaterializedConversation, message_tuples: list[MessageTuple]) -> StatsTuple:
    return (
        conversation.conversation_id,
        conversation.source_name,
        len(message_tuples),
        sum(message[11] for message in message_tuples),
        sum(message[12] for message in message_tuples),
        sum(message[13] for message in message_tuples),
        sum(message[14] for message in message_tuples),
    )


def _provider_event_tuples(
    conversation: MaterializedConversation,
    *,
    raw_id: str | None,
) -> list[ProviderEventTuple]:
    tuples: list[ProviderEventTuple] = []
    for event in conversation.provider_events:
        projection = project_provider_event_payload(event.event_type, event.payload)
        tuples.append(
            (
                event.event_id,
                event.conversation_id,
                event.source_name,
                event.event_index,
                event.event_type,
                projection.normalized_kind,
                event.timestamp,
                event.sort_key,
                projection.payload,
                event.source_message_id,
                raw_id,
                PROVIDER_EVENT_MATERIALIZER_VERSION,
            )
        )
    return tuples


def _attachment_tuples(
    conversation: MaterializedConversation,
) -> tuple[list[AttachmentTuple], list[AttachmentRefTuple]]:
    attachment_tuples: list[AttachmentTuple] = []
    attachment_ref_tuples: list[AttachmentRefTuple] = []
    for attachment in conversation.attachments:
        meta_json = _json_or_none(attachment.provider_meta)
        provider_attachment_id = getattr(attachment, "provider_attachment_id", None) or None
        provider_file_id = getattr(attachment, "provider_file_id", None) or None
        provider_drive_id = getattr(attachment, "provider_drive_id", None) or None
        # #1252: upload_origin is the closed vocabulary the #1199 attachment
        # library uses to group attachments without scanning provider_meta.
        upload_origin = getattr(attachment, "upload_origin", None) or None
        attachment_tuples.append(
            (
                attachment.attachment_id,
                attachment.mime_type,
                attachment.size_bytes,
                attachment.path,
                0,
                meta_json,
                provider_attachment_id,
                provider_file_id,
                provider_drive_id,
                upload_origin,
            )
        )
        attachment_ref_tuples.append(
            (
                _make_ref_id(
                    attachment.attachment_id,
                    conversation.conversation_id,
                    attachment.message_id,
                ),
                attachment.attachment_id,
                conversation.conversation_id,
                attachment.message_id,
                meta_json,
                provider_attachment_id,
                provider_file_id,
                provider_drive_id,
                upload_origin,
            )
        )
    return attachment_tuples, attachment_ref_tuples


def _transform_to_tuples(
    convo: ParsedConversation,
    *,
    source_name: str,
    archive_root: Path,
    raw_id: str | None,
    append_only: bool = False,
) -> ConversationData:
    """Convert a ParsedConversation to DB-ready tuples."""
    materialized = materialize_conversation(
        convo,
        source_name=source_name,
        archive_root=archive_root,
    )
    message_tuples = _dedupe_by_key_preserve_last(_message_tuples(materialized), lambda item: item[0])
    block_tuples = _dedupe_by_key_preserve_last(_content_block_tuples(materialized), lambda item: item[0])
    action_event_tuples = _dedupe_by_key_preserve_last(_build_action_event_tuples(materialized), lambda item: item[0])
    provider_event_tuples = _dedupe_by_key_preserve_last(
        _provider_event_tuples(materialized, raw_id=raw_id), lambda item: item[0]
    )

    attachment_tuples, attachment_ref_tuples = _attachment_tuples(materialized)
    attachment_tuples = _dedupe_by_key_preserve_last(attachment_tuples, lambda item: item[0])
    attachment_ref_tuples = _dedupe_by_key_preserve_last(attachment_ref_tuples, lambda item: item[0])

    return ConversationData(
        conversation_id=materialized.conversation_id,
        content_hash=materialized.content_hash,
        conversation_tuple=_conversation_tuple(materialized, raw_id=raw_id),
        message_tuples=message_tuples,
        block_tuples=block_tuples,
        action_event_tuples=action_event_tuples,
        provider_event_tuples=provider_event_tuples,
        stats_tuple=_stats_tuple(materialized, message_tuples),
        attachment_tuples=attachment_tuples,
        attachment_ref_tuples=attachment_ref_tuples,
        source_name=source_name,
        raw_id=raw_id,
        append_only=append_only,
    )


def _timestamp_iso_for_action_event(event: ActionEvent) -> str | None:
    if event.timestamp is None:
        return None
    timestamp = event.timestamp
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.isoformat()


def _action_event_source_block_id(event: ActionEvent) -> str | None:
    if not isinstance(event.raw, dict):
        return None
    block_id = event.raw.get("block_id")
    return block_id if isinstance(block_id, str) else None


def _action_event_tuple(
    conversation: MaterializedConversation,
    message: MaterializedMessage,
    event: ActionEvent,
) -> ActionEventTuple:
    affected_paths_json = json_dumps(list(event.affected_paths)) if event.affected_paths else None
    branch_names_json = json_dumps(list(event.branch_names)) if event.branch_names else None
    return (
        event.event_id,
        conversation.conversation_id,
        message.message_id,
        ACTION_EVENT_MATERIALIZER_VERSION,
        _action_event_source_block_id(event),
        _timestamp_iso_for_action_event(event),
        message.sort_key,
        event.sequence_index,
        conversation.source_name,
        event.kind.value,
        event.tool_name,
        event.normalized_tool_name,
        event.tool_id,
        affected_paths_json,
        event.cwd_path,
        branch_names_json,
        event.command,
        event.query,
        event.url,
        event.output_text,
        event.search_text,
    )


def _action_event_message_timestamp(message: MaterializedMessage) -> datetime | None:
    if message.sort_key is None:
        return None
    try:
        return datetime.fromtimestamp(message.sort_key, tz=timezone.utc)
    except (OSError, ValueError):
        return None


def _content_block_mapping_for_action_event(
    *,
    conversation_id: ConversationId,
    message_id: MessageId,
    block: MaterializedContentBlock,
) -> dict[str, object]:
    return {
        "block_id": block.block_id,
        "message_id": message_id,
        "conversation_id": conversation_id,
        "block_index": block.block_index,
        "type": block.type,
        "text": block.text,
        "tool_name": block.tool_name,
        "tool_id": block.tool_id,
        "tool_input": block.tool_input_json,
        "metadata": block.metadata_json,
        "semantic_type": block.semantic_type,
    }


def _build_action_event_tuples(
    conversation: MaterializedConversation,
) -> list[ActionEventTuple]:
    """Build action event tuples for all messages in a conversation.

    Uses the lightweight action event builder directly from materialized content
    blocks, avoiding Pydantic storage-record/domain-message hydration on dense
    tool-call streams.
    """
    from polylogue.archive.action_event.action_events import build_action_events, build_tool_calls_from_content_blocks

    provider = Provider.from_string(conversation.source_name)
    action_tuples: list[ActionEventTuple] = []

    for message in conversation.messages:
        if not message.has_tool_use:
            continue

        content_blocks = tuple(
            _content_block_mapping_for_action_event(
                conversation_id=conversation.conversation_id,
                message_id=message.message_id,
                block=block,
            )
            for block in message.blocks
        )
        tool_calls = build_tool_calls_from_content_blocks(
            provider=provider,
            content_blocks=content_blocks,
        )
        action_message = _ActionEventMessage(
            id=message.message_id,
            timestamp=_action_event_message_timestamp(message),
        )
        for event in build_action_events(action_message, tool_calls):
            action_tuples.append(_action_event_tuple(conversation, message, event))

    return action_tuples


__all__ = ["ConversationData", "IngestRecordResult", "ingest_record"]

