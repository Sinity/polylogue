"""Durable artifact observation writes for the async SQLite backend."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.runtime import ArtifactObservationRecord

__all__ = ["save_artifact_observation"]


async def save_artifact_observation(
    conn: aiosqlite.Connection,
    record: ArtifactObservationRecord,
    transaction_depth: int,
) -> bool:
    """Persist or refresh one artifact observation row."""
    exists_cursor = await conn.execute(
        "SELECT 1 FROM artifact_observations WHERE observation_id = ?",
        (record.observation_id,),
    )
    existed = await exists_cursor.fetchone() is not None

    await conn.execute(
        """
        INSERT INTO artifact_observations (
            observation_id,
            raw_id,
            provider_name,
            payload_provider,
            source_name,
            source_path,
            source_index,
            file_mtime,
            wire_format,
            artifact_kind,
            classification_reason,
            parse_as_conversation,
            schema_eligible,
            support_status,
            malformed_jsonl_lines,
            decode_error,
            bundle_scope,
            cohort_id,
            resolved_package_version,
            resolved_element_kind,
            resolution_reason,
            link_group_key,
            sidecar_agent_type,
            first_observed_at,
            last_observed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(observation_id) DO UPDATE SET
            raw_id = excluded.raw_id,
            provider_name = excluded.provider_name,
            payload_provider = excluded.payload_provider,
            source_name = excluded.source_name,
            source_path = excluded.source_path,
            source_index = excluded.source_index,
            file_mtime = excluded.file_mtime,
            wire_format = excluded.wire_format,
            artifact_kind = excluded.artifact_kind,
            classification_reason = excluded.classification_reason,
            parse_as_conversation = excluded.parse_as_conversation,
            schema_eligible = excluded.schema_eligible,
            support_status = excluded.support_status,
            malformed_jsonl_lines = excluded.malformed_jsonl_lines,
            decode_error = excluded.decode_error,
            bundle_scope = excluded.bundle_scope,
            cohort_id = excluded.cohort_id,
            resolved_package_version = excluded.resolved_package_version,
            resolved_element_kind = excluded.resolved_element_kind,
            resolution_reason = excluded.resolution_reason,
            link_group_key = excluded.link_group_key,
            sidecar_agent_type = excluded.sidecar_agent_type,
            first_observed_at = artifact_observations.first_observed_at,
            last_observed_at = excluded.last_observed_at
        """,
        (
            record.observation_id,
            record.raw_id,
            record.provider_name,
            record.payload_provider,
            record.source_name,
            record.source_path,
            record.source_index,
            record.file_mtime,
            record.wire_format,
            record.artifact_kind,
            record.classification_reason,
            int(record.parse_as_conversation),
            int(record.schema_eligible),
            str(record.support_status),
            record.malformed_jsonl_lines,
            record.decode_error,
            record.bundle_scope,
            record.cohort_id,
            record.resolved_package_version,
            record.resolved_element_kind,
            record.resolution_reason,
            record.link_group_key,
            record.sidecar_agent_type,
            record.first_observed_at,
            record.last_observed_at,
        ),
    )
    if transaction_depth == 0:
        await conn.commit()
    return not existed
