"""Row mappers for archive-core records."""

from __future__ import annotations

import sqlite3

from polylogue.archive.conversation.branch_type import BranchType
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.storage.backends.queries.mappers_support import (
    _json_object,
    _json_text_tuple,
    _parse_json,
    _row_float,
    _row_get,
    _row_int,
    _row_text,
)
from polylogue.storage.runtime import (
    ActionEventRecord,
    ArtifactObservationRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
    RawConversationRecord,
)
from polylogue.types import (
    ArtifactSupportStatus,
    ContentBlockType,
    ConversationId,
    MessageId,
    Provider,
    SemanticBlockType,
    ValidationMode,
    ValidationStatus,
)


def _row_to_conversation(row: sqlite3.Row) -> ConversationRecord:
    parent_conversation_id = _row_text(row, "parent_conversation_id")
    branch_type = _row_text(row, "branch_type")
    return ConversationRecord(
        conversation_id=row["conversation_id"],
        provider_name=row["provider_name"],
        provider_conversation_id=row["provider_conversation_id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        sort_key=_row_float(row, "sort_key"),
        content_hash=row["content_hash"],
        provider_meta=_json_object(
            _parse_json(row["provider_meta"], field="provider_meta", record_id=row["conversation_id"])
        ),
        metadata=_json_object(_parse_json(row["metadata"], field="metadata", record_id=row["conversation_id"])),
        version=row["version"],
        parent_conversation_id=ConversationId(parent_conversation_id) if parent_conversation_id is not None else None,
        branch_type=BranchType(branch_type) if branch_type is not None else None,
        raw_id=_row_text(row, "raw_id"),
    )


def _row_to_message(row: sqlite3.Row) -> MessageRecord:
    role = _row_text(row, "role")
    normalized_role = Role.normalize(role) if role is not None and role.strip() else None
    parent_message_id = _row_text(row, "parent_message_id")
    return MessageRecord(
        message_id=row["message_id"],
        conversation_id=row["conversation_id"],
        provider_message_id=_row_text(row, "provider_message_id"),
        role=normalized_role,
        text=_row_text(row, "text"),
        sort_key=_row_float(row, "sort_key"),
        content_hash=row["content_hash"],
        version=row["version"],
        parent_message_id=MessageId(parent_message_id) if parent_message_id is not None else None,
        branch_index=_row_int(row, "branch_index", 0) or 0,
        provider_name=_row_text(row, "provider_name") or "",
        word_count=_row_int(row, "word_count", 0) or 0,
        has_tool_use=_row_int(row, "has_tool_use", 0) or 0,
        has_thinking=_row_int(row, "has_thinking", 0) or 0,
        has_paste=_row_int(row, "has_paste", 0) or 0,
        message_type=MessageType.normalize(_row_text(row, "message_type") or "message"),
    )


def _row_to_content_block(row: sqlite3.Row) -> ContentBlockRecord:
    semantic_type = _row_text(row, "semantic_type")
    return ContentBlockRecord(
        block_id=row["block_id"],
        message_id=MessageId(row["message_id"]),
        conversation_id=ConversationId(row["conversation_id"]),
        block_index=row["block_index"],
        type=ContentBlockType.from_string(row["type"]),
        text=_row_text(row, "text"),
        tool_name=_row_text(row, "tool_name"),
        tool_id=_row_text(row, "tool_id"),
        tool_input=_row_text(row, "tool_input"),
        media_type=_row_text(row, "media_type"),
        metadata=_row_text(row, "metadata"),
        semantic_type=SemanticBlockType.from_string(semantic_type) if semantic_type is not None else None,
    )


def _row_to_raw_conversation(row: sqlite3.Row) -> RawConversationRecord:
    validation_status = _row_text(row, "validation_status")
    validation_provider = _row_text(row, "validation_provider")
    validation_mode = _row_text(row, "validation_mode")
    return RawConversationRecord(
        raw_id=row["raw_id"],
        provider_name=row["provider_name"],
        payload_provider=(
            Provider.from_string(_row_text(row, "payload_provider"))
            if _row_text(row, "payload_provider") is not None
            else None
        ),
        source_name=row["source_name"],
        source_path=row["source_path"],
        source_index=row["source_index"],
        blob_size=row["blob_size"],
        acquired_at=row["acquired_at"],
        file_mtime=row["file_mtime"],
        parsed_at=_row_text(row, "parsed_at"),
        parse_error=_row_text(row, "parse_error"),
        validated_at=_row_text(row, "validated_at"),
        validation_status=(ValidationStatus.from_string(validation_status) if validation_status is not None else None),
        validation_error=_row_text(row, "validation_error"),
        validation_drift_count=_row_int(row, "validation_drift_count"),
        validation_provider=(Provider.from_string(validation_provider) if validation_provider is not None else None),
        validation_mode=(ValidationMode.from_string(validation_mode) if validation_mode is not None else None),
    )


def _row_to_artifact_observation(row: sqlite3.Row) -> ArtifactObservationRecord:
    return ArtifactObservationRecord(
        observation_id=row["observation_id"],
        raw_id=row["raw_id"],
        provider_name=row["provider_name"],
        payload_provider=(
            Provider.from_string(_row_text(row, "payload_provider"))
            if _row_text(row, "payload_provider") is not None
            else None
        ),
        source_name=_row_text(row, "source_name"),
        source_path=row["source_path"],
        source_index=_row_int(row, "source_index"),
        file_mtime=_row_text(row, "file_mtime"),
        wire_format=_row_text(row, "wire_format"),
        artifact_kind=row["artifact_kind"],
        classification_reason=row["classification_reason"],
        parse_as_conversation=bool(_row_get(row, "parse_as_conversation", 0)),
        schema_eligible=bool(_row_get(row, "schema_eligible", 0)),
        support_status=ArtifactSupportStatus.from_string(row["support_status"]),
        malformed_jsonl_lines=int(_row_get(row, "malformed_jsonl_lines", 0) or 0),
        decode_error=_row_text(row, "decode_error"),
        bundle_scope=_row_text(row, "bundle_scope"),
        cohort_id=_row_text(row, "cohort_id"),
        resolved_package_version=_row_text(row, "resolved_package_version"),
        resolved_element_kind=_row_text(row, "resolved_element_kind"),
        resolution_reason=_row_text(row, "resolution_reason"),
        link_group_key=_row_text(row, "link_group_key"),
        sidecar_agent_type=_row_text(row, "sidecar_agent_type"),
        first_observed_at=row["first_observed_at"],
        last_observed_at=row["last_observed_at"],
    )


def _row_to_action_event(row: sqlite3.Row) -> ActionEventRecord:
    return ActionEventRecord(
        event_id=row["event_id"],
        conversation_id=ConversationId(row["conversation_id"]),
        message_id=MessageId(row["message_id"]),
        materializer_version=int(_row_int(row, "materializer_version", 1) or 1),
        source_block_id=_row_text(row, "source_block_id"),
        timestamp=_row_text(row, "timestamp"),
        sort_key=_row_float(row, "sort_key"),
        sequence_index=row["sequence_index"],
        provider_name=_row_text(row, "provider_name"),
        action_kind=row["action_kind"],
        tool_name=_row_text(row, "tool_name"),
        normalized_tool_name=row["normalized_tool_name"],
        tool_id=_row_text(row, "tool_id"),
        affected_paths=_json_text_tuple(_parse_json(_row_get(row, "affected_paths_json"))),
        cwd_path=_row_text(row, "cwd_path"),
        branch_names=_json_text_tuple(_parse_json(_row_get(row, "branch_names_json"))),
        command=_row_text(row, "command"),
        query_text=_row_text(row, "query_text"),
        url=_row_text(row, "url"),
        output_text=_row_text(row, "output_text"),
        search_text=row["search_text"],
    )


__all__ = [
    "_row_to_action_event",
    "_row_to_artifact_observation",
    "_row_to_content_block",
    "_row_to_conversation",
    "_row_to_message",
    "_row_to_raw_conversation",
]
