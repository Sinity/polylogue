"""Inspection helpers for deriving durable artifact observations from raw rows."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from polylogue.archive.artifact_taxonomy import ArtifactKind, classify_artifact_path
from polylogue.archive.raw_payload import JSONValue, RawPayloadEnvelope, build_raw_payload_envelope
from polylogue.core.enums import ArtifactSupportStatus, Provider
from polylogue.schemas.observation import derive_bundle_scope, schema_cluster_id
from polylogue.schemas.packages import SchemaResolution
from polylogue.schemas.runtime_registry import SchemaRegistry
from polylogue.storage.blob_store import get_blob_store
from polylogue.storage.runtime import ArtifactObservationRecord, RawSessionRecord

_SCHEMA_REGISTRY = SchemaRegistry()


def artifact_observation_id(
    *,
    source_name: str | None,
    source_path: str,
    source_index: int | None,
) -> str:
    """Return a stable observation identifier for one source artifact."""
    seed = f"{source_name or ''}:{source_path}:{source_index if source_index is not None else ''}"
    return f"obs-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:24]}"


def _link_group_key(source_path: str | None) -> str | None:
    normalized = str(source_path or "").replace("\\", "/").lower()
    if not normalized:
        return None
    for suffix in (".meta.json", ".jsonl.txt", ".jsonl", ".ndjson"):
        if normalized.endswith(suffix):
            stem = normalized[: -len(suffix)]
            leaf = stem.rsplit("/", 1)[-1]
            if leaf.startswith("agent-"):
                return stem
    return None


def _build_payload_envelope(
    raw_content: bytes,
    record: RawSessionRecord,
) -> RawPayloadEnvelope:
    return build_raw_payload_envelope(
        raw_content,
        source_path=record.source_path,
        fallback_provider=record.source_name or "",
        payload_provider=record.payload_provider,
        jsonl_dict_only=False,
    )


def _normalize_payload_provider_hint(record: RawSessionRecord) -> str | None:
    hint = record.payload_provider or record.source_name
    if not isinstance(hint, str):
        return None
    candidate = hint.strip()
    return candidate or None


def _resolve_payload_support(
    registry: SchemaRegistry,
    payload_provider: Provider,
    payload: JSONValue,
    source_path: str | None,
) -> tuple[SchemaResolution | None, bool]:
    resolution = registry.resolve_payload(
        payload_provider,
        payload,
        source_path=source_path,
    )
    if resolution is None:
        return None, False

    package = registry.get_package(payload_provider, version=resolution.package_version)
    element = package.element(resolution.element_kind) if package is not None else None
    if package is None or element is None or not element.supported:
        return resolution, False

    return resolution, True


def _inspect_payload_envelope(record: RawSessionRecord) -> RawPayloadEnvelope:
    blob_store = get_blob_store()
    prefix = _inspection_prefix(record)
    try:
        envelope = _build_payload_envelope(prefix, record)
    except Exception:
        if not _full_json_inspection_allowed(record):
            raise
        return _build_payload_envelope(blob_store.read_all(record.raw_id), record)

    if _should_retry_full_json_inspection(record, wire_format=envelope.wire_format):
        return _build_payload_envelope(blob_store.read_all(record.raw_id), record)
    return envelope


def _sidecar_agent_type(payload: JSONValue) -> str | None:
    if isinstance(payload, dict):
        agent_type = payload.get("agentType")
        return agent_type if isinstance(agent_type, str) and agent_type else None
    if isinstance(payload, list):
        first_payload = payload[0] if payload else None
        if not isinstance(first_payload, dict):
            return None
        agent_type = first_payload.get("agentType")
        return agent_type if isinstance(agent_type, str) and agent_type else None
    return None


def _support_status(
    *,
    parse_as_session: bool,
    schema_eligible: bool,
    malformed_jsonl_lines: int,
    artifact_kind: str,
    has_supported_resolution: bool,
    had_decode_error: bool,
    partial_decode: bool = False,
) -> ArtifactSupportStatus:
    if had_decode_error:
        return ArtifactSupportStatus.DECODE_FAILED
    if malformed_jsonl_lines > 0:
        # A stream that still produced valid records lost only *some* lines —
        # that is partial loss, distinct from a wholesale decode failure. The
        # caller sets ``partial_decode`` once it has an accurate (full-scan,
        # not prefix-bounded) malformed-line count for a stream artifact.
        return ArtifactSupportStatus.PARTIAL_DECODE if partial_decode else ArtifactSupportStatus.DECODE_FAILED
    if artifact_kind == ArtifactKind.UNKNOWN.value:
        return ArtifactSupportStatus.UNKNOWN
    if not parse_as_session or not schema_eligible:
        return ArtifactSupportStatus.RECOGNIZED_UNPARSED
    if has_supported_resolution:
        return ArtifactSupportStatus.SUPPORTED_PARSEABLE
    return ArtifactSupportStatus.UNSUPPORTED_PARSEABLE


_INSPECTION_PREFIX_BYTES = 64 * 1024  # 64 KB — enough to classify any format
_FULL_JSON_INSPECTION_MAX_BYTES = 8 * 1024 * 1024  # 8 MB — bounded fallback for large JSON documents


def _inspection_prefix_from_bytes(raw_content: bytes, source_path: str | None) -> bytes:
    """Extract a small prefix of in-memory raw content for artifact classification."""
    if len(raw_content) <= _INSPECTION_PREFIX_BYTES:
        return raw_content
    normalized = (source_path or "").lower()
    is_jsonl = normalized.endswith((".jsonl", ".jsonl.txt", ".ndjson"))
    if is_jsonl:
        newline_pos = raw_content.find(b"\n")
        if newline_pos > 0:
            return raw_content[: newline_pos + 1]
    return raw_content[:_INSPECTION_PREFIX_BYTES]


def _inspection_prefix(record: RawSessionRecord) -> bytes:
    """Extract a small prefix of raw content sufficient for classification.

    Reads only the first 64 KB from the blob store — multi-GB files are
    never loaded into memory.
    """
    blob_store = get_blob_store()
    prefix = blob_store.read_prefix(record.raw_id, _INSPECTION_PREFIX_BYTES)
    normalized = (record.source_path or "").lower()
    is_jsonl = normalized.endswith((".jsonl", ".jsonl.txt", ".ndjson"))
    if is_jsonl and len(prefix) >= _INSPECTION_PREFIX_BYTES:
        newline_pos = prefix.find(b"\n")
        if newline_pos > 0:
            return prefix[: newline_pos + 1]
    return prefix


def _prefers_json_stream(source_path: str | None) -> bool:
    normalized = (source_path or "").lower()
    return normalized.endswith((".jsonl", ".jsonl.txt", ".ndjson"))


def _full_json_inspection_allowed(record: RawSessionRecord) -> bool:
    if _prefers_json_stream(record.source_path):
        return False
    return record.blob_size <= _FULL_JSON_INSPECTION_MAX_BYTES


def _should_retry_full_json_inspection(record: RawSessionRecord, *, wire_format: str | None) -> bool:
    return _full_json_inspection_allowed(record) and wire_format == "jsonl"


def _full_scan_malformed_jsonl(record: RawSessionRecord) -> tuple[int, bool]:
    """Stream the entire blob to count malformed JSONL lines.

    The prefix-based classification only inspects the first 64 KB, so malformed
    content past the prefix never marks the artifact failed (#1745). This scan
    streams the whole blob line-by-line (never materializing it) so the
    malformed-line count and decode status reflect the full artifact.

    Returns ``(malformed_lines, had_valid_records)``. ``had_valid_records`` is
    ``True`` when at least one line decoded successfully; the sampling helper
    raises ``ValueError`` only when no valid record exists, which is the
    no-valid-records signal.
    """
    from polylogue.archive.raw_payload.decode import _sample_jsonl_payload_with_detail

    blob_path = get_blob_store().blob_path(record.raw_id)
    try:
        _samples, malformed_lines, _detail = _sample_jsonl_payload_with_detail(
            blob_path,
            max_samples=1,
            jsonl_dict_only=False,
            scan_full=True,
        )
    except ValueError:
        # No valid JSONL records at all — leave the decision to the prefix-based
        # classification (which will have surfaced a decode error already).
        return 0, False
    return malformed_lines, True


def _stream_loss_accounting(
    record: RawSessionRecord,
    *,
    wire_format: str | None,
    prefix_malformed_lines: int,
) -> tuple[int, bool]:
    """Return ``(malformed_lines, partial_decode)`` for a stream artifact.

    For JSONL artifacts larger than the inspection prefix, runs a full-file scan
    so the malformed-line count is accurate rather than prefix-bounded. Marks
    ``partial_decode`` when malformed lines coexist with valid records (some
    records survived, some were lost).

    The decision keys off the *source path* being a JSONL/stream artifact, not
    the envelope ``wire_format``: a large JSONL file's inspection prefix is
    truncated to its first line, which decodes as a single JSON object
    (``wire_format == "json"``), so trusting ``wire_format`` here would skip the
    full scan exactly when loss past the prefix is most likely (#1745).
    """
    is_stream = _prefers_json_stream(record.source_path) or wire_format == "jsonl"
    if not is_stream or record.blob_size <= _INSPECTION_PREFIX_BYTES:
        # The prefix already covered the whole file (or this is not a stream).
        # Reaching the success branch with malformed lines means valid records
        # decoded too, so any loss here is partial.
        return prefix_malformed_lines, prefix_malformed_lines > 0
    malformed_lines, had_valid_records = _full_scan_malformed_jsonl(record)
    if malformed_lines == 0:
        return malformed_lines, False
    return malformed_lines, had_valid_records


def inspect_raw_artifact(record: RawSessionRecord) -> ArtifactObservationRecord:
    """Inspect one raw record into a durable artifact observation.

    Uses only a small prefix of raw_content for classification — never
    decodes the full payload. This keeps memory bounded regardless of
    file size (a 1.5 GB JSONL file is classified from its first line).
    """
    provider_hint = _normalize_payload_provider_hint(record)
    provider_token = provider_hint or record.source_name or ""
    bundle_scope = derive_bundle_scope(provider_token, record.source_path)
    observation_id = artifact_observation_id(
        source_name=record.source_name,
        source_path=record.source_path,
        source_index=record.source_index,
    )
    observed_at = record.acquired_at or datetime.now(tz=timezone.utc).isoformat()
    registry = _SCHEMA_REGISTRY

    try:
        envelope = _inspect_payload_envelope(record)
        payload_provider = envelope.provider
        resolution: SchemaResolution | None = None
        has_supported_resolution = False

        # The envelope's malformed count comes from the inspection prefix only.
        # For stream artifacts larger than the prefix, re-scan the whole blob so
        # malformed lines past the prefix are not silently invisible (#1745).
        malformed_jsonl_lines, partial_decode = _stream_loss_accounting(
            record,
            wire_format=envelope.wire_format,
            prefix_malformed_lines=envelope.malformed_jsonl_lines,
        )

        if envelope.artifact.parse_as_session and envelope.artifact.schema_eligible and malformed_jsonl_lines == 0:
            resolution, has_supported_resolution = _resolve_payload_support(
                registry=registry,
                payload_provider=payload_provider,
                payload=envelope.payload,
                source_path=record.source_path,
            )
        resolved_package_version = resolution.package_version if resolution is not None else None
        resolved_element_kind = resolution.element_kind if resolution is not None else None
        resolution_reason = resolution.reason if resolution is not None else None

        support_status = _support_status(
            parse_as_session=envelope.artifact.parse_as_session,
            schema_eligible=envelope.artifact.schema_eligible,
            malformed_jsonl_lines=malformed_jsonl_lines,
            artifact_kind=envelope.artifact.kind.value,
            has_supported_resolution=has_supported_resolution,
            had_decode_error=False,
            partial_decode=partial_decode,
        )

        return ArtifactObservationRecord(
            observation_id=observation_id,
            raw_id=record.raw_id,
            payload_provider=payload_provider,
            source_name=record.source_name,
            source_path=record.source_path,
            source_index=record.source_index,
            file_mtime=record.file_mtime,
            wire_format=envelope.wire_format,
            artifact_kind=envelope.artifact.kind.value,
            classification_reason=envelope.artifact.reason,
            parse_as_session=envelope.artifact.parse_as_session,
            schema_eligible=envelope.artifact.schema_eligible,
            support_status=support_status,
            malformed_jsonl_lines=malformed_jsonl_lines,
            decode_error=None,
            bundle_scope=bundle_scope,
            cohort_id=schema_cluster_id(envelope.payload, envelope.artifact.cohort),
            resolved_package_version=resolved_package_version,
            resolved_element_kind=resolved_element_kind,
            resolution_reason=resolution_reason,
            link_group_key=_link_group_key(record.source_path),
            sidecar_agent_type=(
                _sidecar_agent_type(envelope.payload)
                if envelope.artifact.kind is ArtifactKind.AGENT_SIDECAR_META
                else None
            ),
            first_observed_at=observed_at,
            last_observed_at=observed_at,
        )
    except Exception as exc:
        path_classification = classify_artifact_path(record.source_path, provider=provider_token or "")
        artifact_kind = (
            path_classification.kind.value if path_classification is not None else ArtifactKind.UNKNOWN.value
        )
        classification_reason = (
            path_classification.reason if path_classification is not None else f"decode failure: {type(exc).__name__}"
        )
        return ArtifactObservationRecord(
            observation_id=observation_id,
            raw_id=record.raw_id,
            payload_provider=Provider.from_string(provider_token),
            source_name=record.source_name,
            source_path=record.source_path,
            source_index=record.source_index,
            file_mtime=record.file_mtime,
            wire_format=None,
            artifact_kind=artifact_kind,
            classification_reason=classification_reason,
            parse_as_session=path_classification.parse_as_session if path_classification else False,
            schema_eligible=path_classification.schema_eligible if path_classification else False,
            support_status=_support_status(
                parse_as_session=path_classification.parse_as_session if path_classification else False,
                schema_eligible=path_classification.schema_eligible if path_classification else False,
                malformed_jsonl_lines=0,
                artifact_kind=artifact_kind,
                has_supported_resolution=False,
                had_decode_error=True,
            ),
            malformed_jsonl_lines=0,
            decode_error=f"{type(exc).__name__}: {exc}",
            bundle_scope=bundle_scope,
            cohort_id=None,
            resolved_package_version=None,
            resolved_element_kind=None,
            resolution_reason=None,
            link_group_key=_link_group_key(record.source_path),
            sidecar_agent_type=None,
            first_observed_at=observed_at,
            last_observed_at=observed_at,
        )


__all__ = [
    "SchemaRegistry",
    "artifact_observation_id",
    "inspect_raw_artifact",
]
