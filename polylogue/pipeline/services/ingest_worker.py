"""Unified subprocess worker: decode → validate → parse → transform in one pass.

Runs inside ProcessPoolExecutor. Returns plain tuples for direct SQL executemany,
avoiding Pydantic serialization overhead across the process boundary.

Performance: eliminates double blob decode (was: validate decodes, then parse decodes
the same blob again). Moves transform into subprocess for true parallelism.
"""

from __future__ import annotations

import pickle
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from typing_extensions import TypedDict

from polylogue.archive.artifact_taxonomy import ArtifactClassification, ArtifactKind, classify_artifact
from polylogue.archive.artifact_taxonomy.support import is_subagent_path
from polylogue.archive.raw_payload.decode import RawPayloadEnvelope
from polylogue.core.common import format_malformed_jsonl_error as _format_malformed_jsonl_error
from polylogue.core.enums import Provider, ValidationMode, ValidationStatus
from polylogue.logging import get_logger
from polylogue.pipeline.ids import session_content_hash
from polylogue.pipeline.ids import session_id as make_session_id
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import STREAM_RECORD_PROVIDERS, is_jsonl_source_path, is_stream_record_provider
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.runtime import (
    RawSessionRecord,
)

if TYPE_CHECKING:
    from polylogue.schemas.packages import SchemaResolution
    from polylogue.schemas.runtime_registry import SchemaRegistry
    from polylogue.sources.parsers.base import ParsedSession


logger = get_logger(__name__)
_SOURCE_HASH_SUFFIX = re.compile(r"-(?:[0-9a-f]{16,64})$", re.IGNORECASE)
_SCHEMA_REGISTRY: SchemaRegistry | None = None


class _TimestampUpdates(TypedDict, total=False):
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SessionWritePayload:
    """Current archive write payload for one parsed session."""

    session_id: str
    content_hash: str
    parsed_session: ParsedSession
    message_count: int = 0
    attachment_count: int = 0
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
    sessions: list[SessionWritePayload] = field(default_factory=list)
    source_name: str | None = None
    serialized_size_bytes: int | None = None


ParsePlanMode = Literal["payload", "stream"]


@dataclass(frozen=True, slots=True)
class _IngestContext:
    raw_record: RawSessionRecord
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
    if stem.startswith("agent-"):
        return stem
    if "/drive-cache/" in normalized:
        return stem
    cleaned = _SOURCE_HASH_SUFFIX.sub("", stem).strip("._- ")
    return cleaned or stem


def _make_ref_id(
    attachment_id: str,
    session_id: str,
    message_id: str | None,
) -> str:
    from hashlib import sha256

    key = f"{attachment_id}:{session_id}:{message_id or ''}"
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


def _normalized_session(
    convo: ParsedSession,
    *,
    fallback_timestamp: str | None,
) -> ParsedSession:
    updates: _TimestampUpdates = {}
    if convo.created_at is None and fallback_timestamp:
        updates["created_at"] = fallback_timestamp
    effective_created = updates.get("created_at", convo.created_at)
    if convo.updated_at is None and isinstance(effective_created, str) and effective_created:
        updates["updated_at"] = effective_created
    return convo.model_copy(update=updates) if updates else convo


def _record_result(
    context: _IngestContext,
    payload_provider: str | None,
    *,
    validation_status: ValidationStatus,
    validation_error: str | None = None,
    parse_error: str | None = None,
    error: str | None = None,
    sessions: list[SessionWritePayload] | None = None,
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
            sessions=sessions or [],
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
            # The sample probe stays bounded (STRICT widens it). The accurate,
            # whole-file malformed-line count for the operator-facing surface is
            # produced by the durable artifact observation
            # (storage/artifacts/inspection.py full-scan, #1745); this probe
            # only needs to *detect* malformed presence for the advisory warning
            # and STRICT failure below, which the bounded sample already does.
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
        ArtifactKind.SUBAGENT_SESSION_STREAM
        if is_subagent_path(context.raw_record.source_path)
        else ArtifactKind.SESSION_RECORD_STREAM
    )
    artifact = ArtifactClassification(
        provider=runtime_provider,
        kind=kind,
        parse_as_session=True,
        schema_eligible=False,
        default_priority=90 if kind is ArtifactKind.SUBAGENT_SESSION_STREAM else 120,
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
    if plan.malformed_jsonl_lines:
        malformed_error = _format_malformed_jsonl_error(
            malformed_lines=plan.malformed_jsonl_lines,
            malformed_detail=plan.malformed_jsonl_detail,
        )
        if context.validation_mode is ValidationMode.STRICT:
            return _PlanValidation(
                status=ValidationStatus.FAILED,
                validation_error=malformed_error,
                parse_error=malformed_error,
            )
        # Advisory mode: do not fail the record, but surface the loss at
        # WARNING level so it is not silently dropped (#1745). The accurate
        # whole-file count is also persisted on the durable artifact
        # observation (storage/artifacts/inspection.py full-scan) and is
        # rolled into the artifact coverage's decode-error tally.
        logger.warning(
            "Malformed JSONL lines counted in advisory mode for %s: %s",
            context.raw_record.source_path or context.raw_record.raw_id,
            malformed_error,
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


def _parse_plan_sessions(
    context: _IngestContext,
    plan: _ParsePlan,
) -> list[ParsedSession]:
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

            sessions = parse_stream_payload(
                plan.provider,
                counted_stream(),
                fallback_id,
                source_path=context.raw_record.source_path,
            )
            if valid_record_count == 0:
                raise ValueError(f"no valid JSON records in {stream_name}")
            return sessions

    return parse_payload(
        plan.provider,
        plan.payload,
        fallback_id,
        schema_resolution=plan.schema_resolution,
        source_path=context.raw_record.source_path,
    )


def _enrich_parsed_sessions(
    context: _IngestContext,
    plan: _ParsePlan,
    parsed_sessions: list[ParsedSession],
) -> list[ParsedSession]:
    """Apply the same provider assembly enrichment direct ingest uses.

    Canonical raw-record ingest historically bypassed provider assembly, so
    daemon-ingested Codex sessions kept native-id titles (polylogue-ih67).
    ``context.raw_source`` is always the blob path here, so discovery keys
    off the recorded acquisition path (``raw_record.source_path``). When
    that path's runtime root is absent (foreign machine, deleted source),
    discovery yields nothing and only the parsed-content fallbacks apply.
    """
    from polylogue.sources.assembly import get_assembly_spec

    spec = get_assembly_spec(plan.provider)
    if spec is None:
        return parsed_sessions
    source_path = context.raw_record.source_path
    if not source_path:
        return parsed_sessions
    try:
        sidecar_data = spec.discover_sidecars([Path(source_path)])
        return [spec.enrich_session(convo, sidecar_data) for convo in parsed_sessions]
    except Exception:
        logger.exception(
            "assembly enrichment failed for %s; keeping unenriched sessions",
            context.raw_record.raw_id,
        )
        return parsed_sessions


def _materialize_parsed_sessions(
    context: _IngestContext,
    plan: _ParsePlan,
    *,
    validation: _PlanValidation,
    parsed_sessions: list[ParsedSession],
) -> IngestRecordResult:
    if not parsed_sessions:
        error = "parse: session artifact produced no materializable sessions"
        return _record_result(
            context,
            plan.payload_provider,
            validation_status=validation.status,
            validation_error=validation.validation_error,
            parse_error=error,
            error=error,
        )

    session_payloads: list[SessionWritePayload] = []
    for convo in parsed_sessions:
        normalized_convo = _normalized_session(
            convo,
            fallback_timestamp=context.fallback_timestamp,
        )
        try:
            session_payloads.append(
                SessionWritePayload(
                    session_id=str(make_session_id(normalized_convo.source_name, normalized_convo.provider_session_id)),
                    content_hash=str(session_content_hash(normalized_convo)),
                    parsed_session=normalized_convo,
                    message_count=len(normalized_convo.messages),
                    attachment_count=len(normalized_convo.attachments),
                    raw_id=context.raw_record.raw_id,
                    append_only=context.raw_record.source_index == -1,
                )
            )
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
        sessions=session_payloads,
        include_source_name=True,
    )


def _run_parse_plan(
    context: _IngestContext,
    plan: _ParsePlan,
) -> IngestRecordResult:
    if not plan.artifact.parse_as_session:
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
        parsed_sessions = _parse_plan_sessions(
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

    return _materialize_parsed_sessions(
        context,
        plan,
        validation=validation,
        parsed_sessions=_enrich_parsed_sessions(context, plan, parsed_sessions),
    )


# ---------------------------------------------------------------------------
# Main worker function — runs in subprocess
# ---------------------------------------------------------------------------


def ingest_record(
    raw_record: RawSessionRecord,
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
    raw_source = blob_store.blob_path(raw_record.blob_hash or raw_record.raw_id)
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

    if is_stream_record_provider(raw_record.source_path, stored_payload_provider or raw_record.source_name):
        if validation_mode is ValidationMode.OFF and (
            stream_plan := _build_fast_stream_parse_plan(context, payload_provider=stored_payload_provider)
        ):
            return _run_parse_plan(context, stream_plan)
        if stream_plan := _build_stream_parse_plan(context, payload_provider=stored_payload_provider):
            return _run_parse_plan(context, stream_plan)
    elif is_jsonl_source_path(raw_record.source_path):
        if stream_plan := _build_stream_parse_plan(context, payload_provider=stored_payload_provider):
            return _run_parse_plan(context, stream_plan)

    # ── Phase 1: Decode blob (ONE decode, not two) ────────────────────
    try:
        envelope = build_raw_payload_envelope(
            context.raw_source,
            source_path=raw_record.source_path,
            fallback_provider=raw_record.source_name or "",
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


__all__ = ["SessionWritePayload", "IngestRecordResult", "ingest_record"]
