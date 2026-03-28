"""Durable artifact observation helpers and read models."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from typing import Any

from polylogue.lib.artifact_taxonomy import ArtifactKind, classify_artifact_path
from polylogue.lib.raw_payload import build_raw_payload_envelope
from polylogue.schemas.runtime_registry import SchemaRegistry
from polylogue.schemas.sampling import derive_bundle_scope, schema_cluster_id
from polylogue.storage.backends.queries.mappers import _row_to_artifact_observation, _row_to_raw_conversation
from polylogue.storage.store import (
    ArtifactCohortSummary,
    ArtifactObservationRecord,
    RawConversationRecord,
)
from polylogue.types import ArtifactSupportStatus, Provider

__all__ = [
    "artifact_observation_id",
    "ensure_artifact_observations",
    "inspect_raw_artifact",
    "list_artifact_cohorts",
    "list_artifact_observations",
]


_EFFECTIVE_PROVIDER_SQL = "COALESCE(payload_provider, provider_name)"
_MAX_SAMPLE_PATHS = 5


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


def _sidecar_agent_type(payload: Any) -> str | None:
    if isinstance(payload, dict):
        agent_type = payload.get("agentType")
        return agent_type if isinstance(agent_type, str) and agent_type else None
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        agent_type = payload[0].get("agentType")
        return agent_type if isinstance(agent_type, str) and agent_type else None
    return None


def _support_status(
    *,
    parse_as_conversation: bool,
    schema_eligible: bool,
    malformed_jsonl_lines: int,
    artifact_kind: str,
    has_supported_resolution: bool,
    had_decode_error: bool,
) -> ArtifactSupportStatus:
    if had_decode_error or malformed_jsonl_lines > 0:
        return ArtifactSupportStatus.DECODE_FAILED
    if artifact_kind == ArtifactKind.UNKNOWN.value:
        return ArtifactSupportStatus.UNKNOWN
    if not parse_as_conversation or not schema_eligible:
        return ArtifactSupportStatus.RECOGNIZED_UNPARSED
    if has_supported_resolution:
        return ArtifactSupportStatus.SUPPORTED_PARSEABLE
    return ArtifactSupportStatus.UNSUPPORTED_PARSEABLE


def _effective_provider(record: ArtifactObservationRecord) -> str:
    return str(record.payload_provider or Provider.from_string(record.provider_name))


def inspect_raw_artifact(record: RawConversationRecord) -> ArtifactObservationRecord:
    """Inspect one raw record into a durable artifact observation."""
    provider_hint = record.payload_provider or record.provider_name
    bundle_scope = derive_bundle_scope(provider_hint, record.source_path)
    observation_id = artifact_observation_id(
        source_name=record.source_name,
        source_path=record.source_path,
        source_index=record.source_index,
    )
    observed_at = record.acquired_at or datetime.now(tz=timezone.utc).isoformat()
    registry = SchemaRegistry()

    try:
        envelope = build_raw_payload_envelope(
            record.raw_content,
            source_path=record.source_path,
            fallback_provider=record.provider_name,
            payload_provider=record.payload_provider,
            jsonl_dict_only=False,
        )
        payload_provider = envelope.provider
        resolved_package_version: str | None = None
        resolved_element_kind: str | None = None
        resolution_reason: str | None = None
        has_supported_resolution = False

        if (
            envelope.artifact.parse_as_conversation
            and envelope.artifact.schema_eligible
            and envelope.malformed_jsonl_lines == 0
        ):
            resolution = registry.resolve_payload(
                payload_provider,
                envelope.payload,
                source_path=record.source_path,
            )
            if resolution is not None:
                package = registry.get_package(payload_provider, version=resolution.package_version)
                element = package.element(resolution.element_kind) if package is not None else None
                if package is not None and element is not None and element.supported:
                    has_supported_resolution = True
                    resolved_package_version = resolution.package_version
                    resolved_element_kind = resolution.element_kind
                    resolution_reason = resolution.reason

        support_status = _support_status(
            parse_as_conversation=envelope.artifact.parse_as_conversation,
            schema_eligible=envelope.artifact.schema_eligible,
            malformed_jsonl_lines=envelope.malformed_jsonl_lines,
            artifact_kind=envelope.artifact.kind.value,
            has_supported_resolution=has_supported_resolution,
            had_decode_error=False,
        )

        return ArtifactObservationRecord(
            observation_id=observation_id,
            raw_id=record.raw_id,
            provider_name=record.provider_name,
            payload_provider=payload_provider,
            source_name=record.source_name,
            source_path=record.source_path,
            source_index=record.source_index,
            file_mtime=record.file_mtime,
            wire_format=envelope.wire_format,
            artifact_kind=envelope.artifact.kind.value,
            classification_reason=envelope.artifact.reason,
            parse_as_conversation=envelope.artifact.parse_as_conversation,
            schema_eligible=envelope.artifact.schema_eligible,
            support_status=support_status,
            malformed_jsonl_lines=envelope.malformed_jsonl_lines,
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
        path_classification = classify_artifact_path(record.source_path, provider=provider_hint)
        artifact_kind = path_classification.kind.value if path_classification is not None else ArtifactKind.UNKNOWN.value
        classification_reason = (
            path_classification.reason
            if path_classification is not None
            else f"decode failure: {type(exc).__name__}"
        )
        return ArtifactObservationRecord(
            observation_id=observation_id,
            raw_id=record.raw_id,
            provider_name=record.provider_name,
            payload_provider=Provider.from_string(provider_hint) if provider_hint is not None else None,
            source_name=record.source_name,
            source_path=record.source_path,
            source_index=record.source_index,
            file_mtime=record.file_mtime,
            wire_format=None,
            artifact_kind=artifact_kind,
            classification_reason=classification_reason,
            parse_as_conversation=path_classification.parse_as_conversation if path_classification else False,
            schema_eligible=path_classification.schema_eligible if path_classification else False,
            support_status=_support_status(
                parse_as_conversation=path_classification.parse_as_conversation if path_classification else False,
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


def _append_observation_filters(
    where_clauses: list[str],
    params: list[Any],
    *,
    providers: list[str] | None = None,
    support_statuses: list[str] | None = None,
    artifact_kinds: list[str] | None = None,
) -> None:
    if providers:
        placeholders = ",".join("?" for _ in providers)
        where_clauses.append(f"{_EFFECTIVE_PROVIDER_SQL} IN ({placeholders})")
        params.extend(providers)
    if support_statuses:
        placeholders = ",".join("?" for _ in support_statuses)
        where_clauses.append(f"support_status IN ({placeholders})")
        params.extend(support_statuses)
    if artifact_kinds:
        placeholders = ",".join("?" for _ in artifact_kinds)
        where_clauses.append(f"artifact_kind IN ({placeholders})")
        params.extend(artifact_kinds)


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
        params: list[Any] = [last_rowid]
        if providers:
            placeholders = ",".join("?" for _ in providers)
            where_clauses.append(
                f"COALESCE(r.payload_provider, r.provider_name) IN ({placeholders})"
            )
            params.extend(providers)
        rows = conn.execute(
            f"""
            SELECT r.rowid AS raw_rowid, r.*
            FROM raw_conversations r
            LEFT JOIN artifact_observations o
              ON COALESCE(o.source_name, '') = COALESCE(r.source_name, '')
             AND o.source_path = r.source_path
             AND COALESCE(o.source_index, -1) = COALESCE(r.source_index, -1)
            WHERE {' AND '.join(where_clauses)}
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


def list_artifact_observations(
    conn: sqlite3.Connection,
    *,
    providers: list[str] | None = None,
    support_statuses: list[str] | None = None,
    artifact_kinds: list[str] | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[ArtifactObservationRecord]:
    """Return durable artifact observations with optional filters."""
    where_clauses: list[str] = []
    params: list[Any] = []
    _append_observation_filters(
        where_clauses,
        params,
        providers=providers,
        support_statuses=support_statuses,
        artifact_kinds=artifact_kinds,
    )

    sql = "SELECT * FROM artifact_observations"
    if where_clauses:
        sql += f" WHERE {' AND '.join(where_clauses)}"
    sql += " ORDER BY last_observed_at DESC, source_path ASC, COALESCE(source_index, -1) ASC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(max(0, int(limit)))
        if offset > 0:
            sql += " OFFSET ?"
            params.append(max(0, int(offset)))
    elif offset > 0:
        sql += " LIMIT -1 OFFSET ?"
        params.append(max(0, int(offset)))

    rows = conn.execute(sql, tuple(params)).fetchall()
    return [_row_to_artifact_observation(row) for row in rows]


def list_artifact_cohorts(
    conn: sqlite3.Connection,
    *,
    providers: list[str] | None = None,
    support_statuses: list[str] | None = None,
    artifact_kinds: list[str] | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[ArtifactCohortSummary]:
    """Summarize durable artifact cohorts over the observation ledger."""
    observations = list_artifact_observations(
        conn,
        providers=providers,
        support_statuses=support_statuses,
        artifact_kinds=artifact_kinds,
        limit=limit,
        offset=offset,
    )
    if not observations:
        return []

    stream_keys_by_provider: dict[str, set[str]] = {}
    for observation in observations:
        if (
            observation.artifact_kind == ArtifactKind.SUBAGENT_CONVERSATION_STREAM.value
            and observation.link_group_key is not None
        ):
            stream_keys_by_provider.setdefault(_effective_provider(observation), set()).add(
                observation.link_group_key
            )

    grouped: dict[
        tuple[str, str, ArtifactSupportStatus, str | None, str | None, str | None, str | None],
        dict[str, Any],
    ] = {}
    for observation in observations:
        provider = _effective_provider(observation)
        key = (
            provider,
            observation.artifact_kind,
            observation.support_status,
            observation.cohort_id,
            observation.resolved_package_version,
            observation.resolved_element_kind,
            observation.resolution_reason,
        )
        bucket = grouped.setdefault(
            key,
            {
                "provider_name": provider,
                "payload_provider": Provider.from_string(provider),
                "artifact_kind": observation.artifact_kind,
                "support_status": observation.support_status,
                "cohort_id": observation.cohort_id,
                "observation_count": 0,
                "raw_ids": set(),
                "first_observed_at": None,
                "last_observed_at": None,
                "bundle_scopes": set(),
                "sample_source_paths": set(),
                "resolved_package_version": observation.resolved_package_version,
                "resolved_element_kind": observation.resolved_element_kind,
                "resolution_reason": observation.resolution_reason,
                "link_groups": set(),
                "linked_sidecar_count": 0,
            },
        )
        bucket["observation_count"] += 1
        bucket["raw_ids"].add(observation.raw_id)
        if observation.first_observed_at is not None:
            bucket["first_observed_at"] = (
                observation.first_observed_at
                if bucket["first_observed_at"] is None
                else min(bucket["first_observed_at"], observation.first_observed_at)
            )
        if observation.last_observed_at is not None:
            bucket["last_observed_at"] = (
                observation.last_observed_at
                if bucket["last_observed_at"] is None
                else max(bucket["last_observed_at"], observation.last_observed_at)
            )
        if observation.bundle_scope is not None:
            bucket["bundle_scopes"].add(observation.bundle_scope)
        bucket["sample_source_paths"].add(observation.source_path)
        if observation.link_group_key is not None:
            bucket["link_groups"].add(observation.link_group_key)
        if (
            observation.artifact_kind == ArtifactKind.AGENT_SIDECAR_META.value
            and observation.link_group_key is not None
            and observation.link_group_key in stream_keys_by_provider.get(provider, set())
        ):
            bucket["linked_sidecar_count"] += 1

    summaries = [
        ArtifactCohortSummary(
            provider_name=bucket["provider_name"],
            payload_provider=bucket["payload_provider"],
            artifact_kind=bucket["artifact_kind"],
            support_status=bucket["support_status"],
            cohort_id=bucket["cohort_id"],
            observation_count=bucket["observation_count"],
            unique_raw_ids=len(bucket["raw_ids"]),
            first_observed_at=bucket["first_observed_at"],
            last_observed_at=bucket["last_observed_at"],
            bundle_scope_count=len(bucket["bundle_scopes"]),
            sample_source_paths=sorted(bucket["sample_source_paths"])[:_MAX_SAMPLE_PATHS],
            resolved_package_version=bucket["resolved_package_version"],
            resolved_element_kind=bucket["resolved_element_kind"],
            resolution_reason=bucket["resolution_reason"],
            link_group_count=len(bucket["link_groups"]),
            linked_sidecar_count=bucket["linked_sidecar_count"],
        )
        for bucket in grouped.values()
    ]
    return sorted(
        summaries,
        key=lambda item: (
            -item.observation_count,
            item.provider_name,
            item.artifact_kind,
            str(item.support_status),
            item.cohort_id or "",
        ),
    )
