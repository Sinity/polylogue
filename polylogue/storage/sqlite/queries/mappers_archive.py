"""Row mappers for archive-core records."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import (
    ArtifactSupportStatus,
    Origin,
    Provider,
    SessionKind,
    ValidationMode,
    ValidationStatus,
)
from polylogue.core.sources import provider_from_origin
from polylogue.core.types import MessageId, SessionId
from polylogue.storage.runtime import (
    ArtifactObservationRecord,
    BlockRecord,
    FileEditRecord,
    MessageRecord,
    RawSessionRecord,
    SessionCommitRecord,
    SessionRecord,
    SessionRefRecord,
    WebContentConstructRecord,
)
from polylogue.storage.sqlite.archive_tiers.archive_tiers_specs import BLOCKS_SPEC, MESSAGES_SPEC
from polylogue.storage.sqlite.queries.mappers_support import (
    _json_object,
    _json_object_list,
    _parse_json,
    _row_float,
    _row_get,
    _row_int,
    _row_optional_bool,
    _row_text,
)


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
        display_name=_row_text(row, "display_name"),
        run_settings=_json_object(
            _parse_json(_row_get(row, "run_settings_json"), field="run_settings_json", record_id=row["session_id"])
        ),
        pending_drafts=_json_object_list(
            _parse_json(_row_get(row, "pending_drafts_json"), field="pending_drafts_json", record_id=row["session_id"])
        ),
        reported_cost_usd=_row_float(row, "reported_cost_usd"),
    )


def _row_to_message(row: sqlite3.Row) -> MessageRecord:
    values = MESSAGES_SPEC.row_to_record_kwargs(row)
    if "parent_message_id" in values and values["parent_message_id"] is not None:
        values["parent_message_id"] = MessageId(str(values["parent_message_id"]))
    values.setdefault("message_id", row["message_id"])
    values.setdefault("session_id", row["session_id"])
    values.setdefault("content_hash", row["content_hash"])
    return MessageRecord(**values)


def _row_to_content_block(row: sqlite3.Row) -> BlockRecord:
    values = BLOCKS_SPEC.row_to_record_kwargs(row)
    values.setdefault("block_id", row["block_id"])
    values["message_id"] = MessageId(str(values["message_id"]))
    values["session_id"] = SessionId(str(values["session_id"]))
    return BlockRecord(**values)


def _row_to_file_edit(row: sqlite3.Row) -> FileEditRecord:
    structured_patch_raw = _parse_json(
        _row_get(row, "structured_patch_json"), field="structured_patch_json", record_id=row["tool_use_block_id"]
    )
    structured_patch: list[dict[str, object]] | None = None
    if isinstance(structured_patch_raw, list):
        structured_patch = [dict(item) for item in structured_patch_raw if isinstance(item, dict)]
    return FileEditRecord(
        tool_use_block_id=row["tool_use_block_id"],
        session_id=SessionId(row["session_id"]),
        message_id=MessageId(row["message_id"]),
        file_path=_row_text(row, "file_path"),
        structured_patch=structured_patch,
        original_file=_row_text(row, "original_file"),
        old_string=_row_text(row, "old_string"),
        new_string=_row_text(row, "new_string"),
        replace_all=_row_optional_bool(row, "replace_all"),
        user_modified=_row_optional_bool(row, "user_modified"),
        observed_at_ms=_row_int(row, "observed_at_ms"),
    )


def _row_to_session_ref(row: sqlite3.Row) -> SessionRefRecord:
    return SessionRefRecord(
        ref_id=row["ref_id"],
        session_id=SessionId(row["session_id"]),
        position=int(row["position"]),
        kind=row["kind"],
        repo=_row_text(row, "repo"),
        number=_row_int(row, "ref_number"),
        url=row["url"],
        observed_at_ms=_row_int(row, "observed_at_ms"),
    )


def _row_to_web_content_construct(row: sqlite3.Row) -> WebContentConstructRecord:
    return WebContentConstructRecord(
        construct_id=row["construct_id"],
        session_id=SessionId(row["session_id"]),
        message_id=MessageId(row["message_id"]),
        block_id=row["block_id"],
        position=int(row["position"]),
        provider=row["provider"],
        construct_type=row["construct_type"],
        provider_key=_row_text(row, "provider_key"),
        title=_row_text(row, "title"),
        url=_row_text(row, "url"),
        text=_row_text(row, "text"),
        source_id=_row_text(row, "source_id"),
        group_id=_row_text(row, "group_id"),
        group_title=_row_text(row, "group_title"),
        query=_row_text(row, "query"),
        asset_pointer=_row_text(row, "asset_pointer"),
        mime_type=_row_text(row, "mime_type"),
        status=_row_text(row, "status"),
        task_id=_row_text(row, "task_id"),
        task_type=_row_text(row, "task_type"),
        rank=_row_int(row, "rank"),
        start_index=_row_int(row, "start_index"),
        end_index=_row_int(row, "end_index"),
    )


def _row_to_session_commit(row: sqlite3.Row) -> SessionCommitRecord:
    return SessionCommitRecord(
        session_id=SessionId(row["session_id"]),
        commit_sha=row["commit_sha"],
        repo_id=_row_text(row, "repo_id"),
        detection_type=row["detection_type"],
        method=_row_text(row, "method"),
        confidence=float(row["confidence"]),
        evidence=row["evidence_json"],
        created_at_ms=int(row["created_at_ms"]),
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
    # Acquisition origin is immutable raw identity. A later parser may retain
    # its exact provider separately without rewriting that identity.
    capture_mode = _row_text(row, "capture_mode")
    detected_provider = _row_text(row, "detected_provider")
    acquisition_provider = provider_from_origin(
        Origin.from_string(row["origin"]),
        family_hint=capture_mode,
    )
    provider = Provider.from_string(detected_provider) if detected_provider is not None else acquisition_provider
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
        source_name=acquisition_provider.value,
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
    "_row_to_file_edit",
    "_row_to_session",
    "_row_to_session_ref",
    "_row_to_message",
    "_row_to_raw_session",
    "_row_to_web_content_construct",
]
