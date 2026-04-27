"""Persistence helpers for durable artifact observations."""

from __future__ import annotations

import sqlite3

from polylogue.storage.artifacts.inspection import inspect_raw_artifact
from polylogue.storage.backends.queries.mappers import _row_to_raw_conversation
from polylogue.storage.runtime import ArtifactObservationRecord


def _upsert_artifact_observation(
    conn: sqlite3.Connection,
    record: ArtifactObservationRecord,
) -> bool:
    exists = conn.execute(
        "SELECT 1 FROM artifact_observations WHERE observation_id = ?",
        (record.observation_id,),
    ).fetchone()
    conn.execute(
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
            str(record.payload_provider) if record.payload_provider is not None else None,
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
    return exists is None


def ensure_artifact_observations(
    conn: sqlite3.Connection,
    *,
    providers: list[str] | None = None,
    refresh_resolutions: bool = False,
) -> int:
    """Backfill or refresh durable artifact observations from raw records."""
    inserted = 0
    last_rowid = 0
    while True:
        state_clauses = ["o.observation_id IS NULL"]
        if refresh_resolutions:
            state_clauses.append(
                "("
                "o.observation_id IS NOT NULL AND "
                "o.parse_as_conversation = 1 AND "
                "o.schema_eligible = 1 AND "
                "o.malformed_jsonl_lines = 0 AND "
                "o.decode_error IS NULL"
                ")"
            )
        where_clauses = [f"({' OR '.join(state_clauses)})", "r.rowid > ?"]
        params: list[object] = [last_rowid]
        if providers:
            placeholders = ",".join("?" for _ in providers)
            where_clauses.append(f"COALESCE(r.payload_provider, r.provider_name) IN ({placeholders})")
            params.extend(providers)
        rows = conn.execute(
            f"""
            SELECT r.rowid AS raw_rowid, r.*
            FROM raw_conversations r
            LEFT JOIN artifact_observations o
              ON COALESCE(o.source_name, '') = COALESCE(r.source_name, '')
             AND o.source_path = r.source_path
             AND COALESCE(o.source_index, -1) = COALESCE(r.source_index, -1)
            WHERE {" AND ".join(where_clauses)}
            ORDER BY r.acquired_at DESC, r.raw_id ASC
            LIMIT 250
            """,
            params,
        ).fetchall()
        if not rows:
            break
        for row in rows:
            record = _row_to_raw_conversation(row)
            observation = inspect_raw_artifact(record)
            if _upsert_artifact_observation(conn, observation):
                inserted += 1
            last_rowid = max(last_rowid, int(row["raw_rowid"]))
        conn.commit()
    return inserted


__all__ = ["ensure_artifact_observations"]
