"""Row mappers for archive-core records."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import (
    ArtifactSupportStatus,
    BlockType,
    MaterialOrigin,
    Origin,
    Provider,
    SemanticBlockType,
    SessionKind,
    ValidationMode,
    ValidationStatus,
)
from polylogue.core.sources import provider_from_origin
from polylogue.storage.runtime import (
    ArtifactObservationRecord,
    BlockRecord,
    MessageRecord,
    RawSessionRecord,
    SessionRecord,
)
from polylogue.storage.sqlite.queries.mappers_support import (
    _json_object,
    _parse_json,
    _row_float,
    _row_get,
    _row_int,
    _row_text,
)
from polylogue.types import MessageId, SessionId


def _row_to_session(row: sqlite3.Row) -> SessionRecord:
    parent_session_id = _row_text(row, "parent_session_id")
    branch_type = _row_text(row, "branch_type")
    return SessionRecord(
        session_id=row["session_id"],
        native_id=row["native_id"],
        origin=row["origin"],
        title=row["title"],
        session_kind=SessionKind.normalize(_row_text(row, "session_kind")),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        sort_key=_row_float(row, "sort_key"),
        content_hash=row["content_hash"],
        metadata=_json_object(_parse_json(row["metadata"], field="metadata", record_id=row["session_id"])),
        version=row["version"],
        parent_session_id=SessionId(parent_session_id) if parent_session_id is not None else None,
        branch_type=BranchType(branch_type) if branch_type is not None else None,
        raw_id=_row_text(row, "raw_id"),
        working_directories_json=_row_text(row, "working_directories_json"),
        git_branch=_row_text(row, "git_branch"),
        git_repository_url=_row_text(row, "git_repository_url"),
        provider_project_ref=_row_text(row, "provider_project_ref"),
    )


def _row_to_message(row: sqlite3.Row) -> MessageRecord:
    role = _row_text(row, "role")
    normalized_role = Role.normalize(role) if role is not None and role.strip() else None
    parent_message_id = _row_text(row, "parent_message_id")
    return MessageRecord(
        message_id=row["message_id"],
        session_id=row["session_id"],
        provider_message_id=_row_text(row, "provider_message_id"),
        role=normalized_role,
        text=_row_text(row, "text"),
        sort_key=_row_float(row, "sort_key"),
        content_hash=row["content_hash"],
        version=row["version"],
        parent_message_id=MessageId(parent_message_id) if parent_message_id is not None else None,
        branch_index=_row_int(row, "branch_index", 0) or 0,
        source_name=_row_text(row, "source_name") or "",
        word_count=_row_int(row, "word_count", 0) or 0,
        has_tool_use=_row_int(row, "has_tool_use", 0) or 0,
        has_thinking=_row_int(row, "has_thinking", 0) or 0,
        has_paste=_row_int(row, "has_paste", 0) or 0,
        paste_boundary_state=_row_text(row, "paste_boundary_state"),
        input_tokens=_row_int(row, "input_tokens", 0) or 0,
        output_tokens=_row_int(row, "output_tokens", 0) or 0,
        cache_read_tokens=_row_int(row, "cache_read_tokens", 0) or 0,
        cache_write_tokens=_row_int(row, "cache_write_tokens", 0) or 0,
        model_name=_row_text(row, "model_name"),
        message_type=MessageType.normalize(_row_text(row, "message_type") or "message"),
        material_origin=MaterialOrigin.normalize(_row_text(row, "material_origin")),
    )


def _row_to_content_block(row: sqlite3.Row) -> BlockRecord:
    semantic_type = _row_text(row, "semantic_type")
    return BlockRecord(
        block_id=row["block_id"],
        message_id=MessageId(row["message_id"]),
        session_id=SessionId(row["session_id"]),
        block_index=row["block_index"],
        type=BlockType.from_string(row["type"]),
        text=_row_text(row, "text"),
        tool_name=_row_text(row, "tool_name"),
        tool_id=_row_text(row, "tool_id"),
        tool_input=_row_text(row, "tool_input"),
        metadata=_row_text(row, "metadata"),
        semantic_type=SemanticBlockType.from_string(semantic_type) if semantic_type is not None else None,
        tool_result_is_error=_row_int(row, "tool_result_is_error"),
        tool_result_exit_code=_row_int(row, "tool_result_exit_code"),
    )


def _ms_to_iso(value: object) -> str | None:
    """Convert an INTEGER epoch-ms column value back to a canonical ISO-8601 string."""
    if not isinstance(value, (int, float)):
        return None
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()


def _row_to_raw_session(row: sqlite3.Row) -> RawSessionRecord:
    validation_status = _row_text(row, "validation_status")
    validation_mode = _row_text(row, "validation_mode")
    blob_hash_value = _row_get(row, "blob_hash")
    blob_hash = bytes(blob_hash_value).hex() if isinstance(blob_hash_value, (bytes, bytearray)) else None
    # raw_sessions carries a single ``origin`` column (#1743). The in-memory
    # record still exposes provider-wire ``source_name``/``payload_provider``;
    # both project from the stored origin.
    capture_mode = _row_text(row, "capture_mode")
    provider = provider_from_origin(
        Origin.from_string(row["origin"]),
        family_hint=capture_mode,
    )
    logical_source_key = _row_text(row, "logical_source_key")
    source_revision = _row_text(row, "source_revision")
    generation = _row_int(row, "acquisition_generation")
    revision = None
    if logical_source_key is not None and source_revision is not None and generation is not None:
        revision = RawRevisionEnvelope(
            logical_source_key=logical_source_key,
            kind=RawRevisionKind(_row_text(row, "revision_kind") or "unknown"),
            source_revision=source_revision,
            acquisition_generation=generation,
            predecessor_source_revision=_row_text(row, "predecessor_source_revision"),
            predecessor_raw_id=_row_text(row, "predecessor_raw_id"),
            baseline_raw_id=_row_text(row, "baseline_raw_id"),
            append_start_offset=_row_int(row, "append_start_offset"),
            append_end_offset=_row_int(row, "append_end_offset"),
            authority=RawRevisionAuthority(_row_text(row, "revision_authority") or "quarantined"),
        )
    return RawSessionRecord(
        raw_id=row["raw_id"],
        blob_hash=blob_hash,
        payload_provider=provider,
        capture_mode=Provider.from_string(capture_mode) if capture_mode is not None else None,
        source_name=provider.value,
        source_path=row["source_path"],
        source_index=row["source_index"],
        blob_size=row["blob_size"],
        acquired_at=_ms_to_iso(row["acquired_at_ms"]) or "",
        file_mtime=_ms_to_iso(row["file_mtime_ms"]),
        parsed_at=_ms_to_iso(row["parsed_at_ms"]),
        parse_error=_row_text(row, "parse_error"),
        validated_at=_ms_to_iso(row["validated_at_ms"]),
        validation_status=(ValidationStatus.from_string(validation_status) if validation_status is not None else None),
        validation_error=_row_text(row, "validation_error"),
        validation_drift_count=_row_int(row, "validation_drift_count"),
        validation_provider=provider,
        validation_mode=(ValidationMode.from_string(validation_mode) if validation_mode is not None else None),
        revision=revision,
    )


def _row_to_artifact_observation(row: sqlite3.Row) -> ArtifactObservationRecord:
    # ``raw_artifacts`` carries a single ``origin`` column (#1743) and drops the
    # wire-format/bundle-scope/resolved-schema/file-mtime fields the record can
    # still hold; those project as ``None`` on read and are recomputed by the
    # inspection read model where fidelity is required.
    provider = provider_from_origin(Origin.from_string(row["origin"]))
    return ArtifactObservationRecord(
        observation_id=row["artifact_id"],
        raw_id=row["raw_id"],
        payload_provider=provider,
        source_name=provider.value,
        source_path=row["source_path"],
        source_index=_row_int(row, "source_index"),
        file_mtime=None,
        wire_format=None,
        artifact_kind=row["artifact_kind"],
        classification_reason=row["classification_reason"],
        parse_as_session=bool(_row_get(row, "parse_as_session", 0)),
        schema_eligible=bool(_row_get(row, "schema_eligible", 0)),
        support_status=ArtifactSupportStatus.from_string(row["support_status"]),
        malformed_jsonl_lines=int(_row_get(row, "malformed_jsonl_lines", 0) or 0),
        decode_error=_row_text(row, "decode_error"),
        bundle_scope=None,
        cohort_id=_row_text(row, "cohort_id"),
        resolved_package_version=None,
        resolved_element_kind=None,
        resolution_reason=None,
        link_group_key=_row_text(row, "link_group_key"),
        sidecar_agent_type=_row_text(row, "sidecar_agent_type"),
        first_observed_at=_ms_to_iso(row["first_observed_at_ms"]) or "",
        last_observed_at=_ms_to_iso(row["last_observed_at_ms"]) or "",
    )


__all__ = [
    "_ms_to_iso",
    "_row_to_artifact_observation",
    "_row_to_content_block",
    "_row_to_session",
    "_row_to_message",
    "_row_to_raw_session",
]
