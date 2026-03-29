"""Row mappers for archive-core records."""

from __future__ import annotations

import sqlite3

from polylogue.storage.backends.queries.mappers import _parse_json, _row_get
from polylogue.storage.store import (
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
    return ConversationRecord(
        conversation_id=row["conversation_id"],
        provider_name=row["provider_name"],
        provider_conversation_id=row["provider_conversation_id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        sort_key=_row_get(row, "sort_key"),
        content_hash=row["content_hash"],
        provider_meta=_parse_json(row["provider_meta"], field="provider_meta", record_id=row["conversation_id"]),
        metadata=_parse_json(row["metadata"], field="metadata", record_id=row["conversation_id"]),
        version=row["version"],
        parent_conversation_id=_row_get(row, "parent_conversation_id"),
        branch_type=_row_get(row, "branch_type"),
        raw_id=_row_get(row, "raw_id"),
    )


def _row_to_message(row: sqlite3.Row) -> MessageRecord:
    return MessageRecord(
        message_id=row["message_id"],
        conversation_id=row["conversation_id"],
        provider_message_id=_row_get(row, "provider_message_id"),
        role=_row_get(row, "role"),
        text=_row_get(row, "text"),
        sort_key=_row_get(row, "sort_key"),
        content_hash=row["content_hash"],
        version=row["version"],
        parent_message_id=_row_get(row, "parent_message_id"),
        branch_index=_row_get(row, "branch_index", 0) or 0,
        provider_name=_row_get(row, "provider_name", "") or "",
        word_count=_row_get(row, "word_count", 0) or 0,
        has_tool_use=_row_get(row, "has_tool_use", 0) or 0,
        has_thinking=_row_get(row, "has_thinking", 0) or 0,
    )


def _row_to_content_block(row: sqlite3.Row) -> ContentBlockRecord:
    return ContentBlockRecord(
        block_id=row["block_id"],
        message_id=MessageId(row["message_id"]),
        conversation_id=ConversationId(row["conversation_id"]),
        block_index=row["block_index"],
        type=ContentBlockType.from_string(row["type"]),
        text=_row_get(row, "text"),
        tool_name=_row_get(row, "tool_name"),
        tool_id=_row_get(row, "tool_id"),
        tool_input=_row_get(row, "tool_input"),
        media_type=_row_get(row, "media_type"),
        metadata=_row_get(row, "metadata"),
        semantic_type=(
            SemanticBlockType.from_string(_row_get(row, "semantic_type"))
            if _row_get(row, "semantic_type") is not None
            else None
        ),
    )


def _row_to_raw_conversation(row: sqlite3.Row) -> RawConversationRecord:
    return RawConversationRecord(
        raw_id=row["raw_id"],
        provider_name=row["provider_name"],
        payload_provider=(
            Provider.from_string(_row_get(row, "payload_provider"))
            if _row_get(row, "payload_provider") is not None
            else None
        ),
        source_name=row["source_name"],
        source_path=row["source_path"],
        source_index=row["source_index"],
        blob_size=row["blob_size"],
        acquired_at=row["acquired_at"],
        file_mtime=row["file_mtime"],
        parsed_at=_row_get(row, "parsed_at"),
        parse_error=_row_get(row, "parse_error"),
        validated_at=_row_get(row, "validated_at"),
        validation_status=(
            ValidationStatus.from_string(_row_get(row, "validation_status"))
            if _row_get(row, "validation_status") is not None
            else None
        ),
        validation_error=_row_get(row, "validation_error"),
        validation_drift_count=_row_get(row, "validation_drift_count"),
        validation_provider=(
            Provider.from_string(_row_get(row, "validation_provider"))
            if _row_get(row, "validation_provider") is not None
            else None
        ),
        validation_mode=(
            ValidationMode.from_string(_row_get(row, "validation_mode"))
            if _row_get(row, "validation_mode") is not None
            else None
        ),
    )


def _row_to_artifact_observation(row: sqlite3.Row) -> ArtifactObservationRecord:
    return ArtifactObservationRecord(
        observation_id=row["observation_id"],
        raw_id=row["raw_id"],
        provider_name=row["provider_name"],
        payload_provider=(
            Provider.from_string(_row_get(row, "payload_provider"))
            if _row_get(row, "payload_provider") is not None
            else None
        ),
        source_name=_row_get(row, "source_name"),
        source_path=row["source_path"],
        source_index=_row_get(row, "source_index"),
        file_mtime=_row_get(row, "file_mtime"),
        wire_format=_row_get(row, "wire_format"),
        artifact_kind=row["artifact_kind"],
        classification_reason=row["classification_reason"],
        parse_as_conversation=bool(_row_get(row, "parse_as_conversation", 0)),
        schema_eligible=bool(_row_get(row, "schema_eligible", 0)),
        support_status=ArtifactSupportStatus.from_string(row["support_status"]),
        malformed_jsonl_lines=int(_row_get(row, "malformed_jsonl_lines", 0) or 0),
        decode_error=_row_get(row, "decode_error"),
        bundle_scope=_row_get(row, "bundle_scope"),
        cohort_id=_row_get(row, "cohort_id"),
        resolved_package_version=_row_get(row, "resolved_package_version"),
        resolved_element_kind=_row_get(row, "resolved_element_kind"),
        resolution_reason=_row_get(row, "resolution_reason"),
        link_group_key=_row_get(row, "link_group_key"),
        sidecar_agent_type=_row_get(row, "sidecar_agent_type"),
        first_observed_at=row["first_observed_at"],
        last_observed_at=row["last_observed_at"],
    )


def _row_to_action_event(row: sqlite3.Row) -> ActionEventRecord:
    return ActionEventRecord(
        event_id=row["event_id"],
        conversation_id=ConversationId(row["conversation_id"]),
        message_id=MessageId(row["message_id"]),
        materializer_version=int(_row_get(row, "materializer_version", 1) or 1),
        source_block_id=_row_get(row, "source_block_id"),
        timestamp=_row_get(row, "timestamp"),
        sort_key=_row_get(row, "sort_key"),
        sequence_index=row["sequence_index"],
        provider_name=_row_get(row, "provider_name"),
        action_kind=row["action_kind"],
        tool_name=_row_get(row, "tool_name"),
        normalized_tool_name=row["normalized_tool_name"],
        tool_id=_row_get(row, "tool_id"),
        affected_paths=tuple(_parse_json(_row_get(row, "affected_paths_json")) or []),
        cwd_path=_row_get(row, "cwd_path"),
        branch_names=tuple(_parse_json(_row_get(row, "branch_names_json")) or []),
        command=_row_get(row, "command"),
        query_text=_row_get(row, "query_text"),
        url=_row_get(row, "url"),
        output_text=_row_get(row, "output_text"),
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
