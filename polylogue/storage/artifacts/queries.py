"""Read-model queries over durable artifact observations."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from polylogue.archive.artifact_taxonomy import ArtifactKind
from polylogue.storage.artifacts.views import ArtifactCohortSummary
from polylogue.storage.backends.queries.mappers import _row_to_artifact_observation
from polylogue.storage.query_models import ArtifactObservationListQuery
from polylogue.storage.runtime import ArtifactObservationRecord
from polylogue.types import ArtifactSupportStatus, Provider

_EFFECTIVE_PROVIDER_SQL = "COALESCE(payload_provider, provider_name)"
_MAX_SAMPLE_PATHS = 5


def _effective_provider(record: ArtifactObservationRecord) -> str:
    return str(record.payload_provider or Provider.from_string(record.provider_name))


@dataclass(slots=True)
class _ArtifactCohortBucket:
    provider_name: str
    payload_provider: Provider | None
    artifact_kind: str
    support_status: ArtifactSupportStatus
    cohort_id: str | None
    resolved_package_version: str | None
    resolved_element_kind: str | None
    resolution_reason: str | None
    observation_count: int = 0
    raw_ids: set[str] = field(default_factory=set)
    first_observed_at: str | None = None
    last_observed_at: str | None = None
    bundle_scopes: set[str] = field(default_factory=set)
    sample_source_paths: set[str] = field(default_factory=set)
    link_groups: set[str] = field(default_factory=set)
    linked_sidecar_count: int = 0


def _append_observation_filters(
    query: ArtifactObservationListQuery,
    where_clauses: list[str],
    params: list[object],
) -> None:
    if query.providers:
        placeholders = ",".join("?" for _ in query.providers)
        where_clauses.append(f"{_EFFECTIVE_PROVIDER_SQL} IN ({placeholders})")
        params.extend(query.providers)
    if query.support_statuses:
        placeholders = ",".join("?" for _ in query.support_statuses)
        where_clauses.append(f"support_status IN ({placeholders})")
        params.extend(query.support_statuses)
    if query.artifact_kinds:
        placeholders = ",".join("?" for _ in query.artifact_kinds)
        where_clauses.append(f"artifact_kind IN ({placeholders})")
        params.extend(query.artifact_kinds)


def list_artifact_observations(
    conn: sqlite3.Connection,
    query: ArtifactObservationListQuery,
) -> list[ArtifactObservationRecord]:
    """Return durable artifact observations with optional filters."""
    where_clauses: list[str] = []
    params: list[object] = []
    _append_observation_filters(query, where_clauses, params)

    sql = "SELECT * FROM artifact_observations"
    if where_clauses:
        sql += f" WHERE {' AND '.join(where_clauses)}"
    sql += " ORDER BY last_observed_at DESC, source_path ASC, COALESCE(source_index, -1) ASC"
    if query.limit is not None:
        sql += " LIMIT ?"
        params.append(max(0, int(query.limit)))
        if query.offset > 0:
            sql += " OFFSET ?"
            params.append(max(0, int(query.offset)))
    elif query.offset > 0:
        sql += " LIMIT -1 OFFSET ?"
        params.append(max(0, int(query.offset)))

    rows = conn.execute(sql, tuple(params)).fetchall()
    return [_row_to_artifact_observation(row) for row in rows]


def list_artifact_cohorts(
    conn: sqlite3.Connection,
    query: ArtifactObservationListQuery,
) -> list[ArtifactCohortSummary]:
    """Summarize durable artifact cohorts over the observation ledger."""
    observations = list_artifact_observations(conn, query)
    if not observations:
        return []

    stream_keys_by_provider: dict[str, set[str]] = {}
    for observation in observations:
        if (
            observation.artifact_kind == ArtifactKind.SUBAGENT_CONVERSATION_STREAM.value
            and observation.link_group_key is not None
        ):
            stream_keys_by_provider.setdefault(_effective_provider(observation), set()).add(observation.link_group_key)

    grouped: dict[
        tuple[str, str, ArtifactSupportStatus, str | None, str | None, str | None, str | None],
        _ArtifactCohortBucket,
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
            _ArtifactCohortBucket(
                provider_name=provider,
                payload_provider=Provider.from_string(provider),
                artifact_kind=observation.artifact_kind,
                support_status=observation.support_status,
                cohort_id=observation.cohort_id,
                resolved_package_version=observation.resolved_package_version,
                resolved_element_kind=observation.resolved_element_kind,
                resolution_reason=observation.resolution_reason,
            ),
        )
        bucket.observation_count += 1
        bucket.raw_ids.add(observation.raw_id)
        if observation.first_observed_at is not None:
            bucket.first_observed_at = (
                observation.first_observed_at
                if bucket.first_observed_at is None
                else min(bucket.first_observed_at, observation.first_observed_at)
            )
        if observation.last_observed_at is not None:
            bucket.last_observed_at = (
                observation.last_observed_at
                if bucket.last_observed_at is None
                else max(bucket.last_observed_at, observation.last_observed_at)
            )
        if observation.bundle_scope is not None:
            bucket.bundle_scopes.add(observation.bundle_scope)
        bucket.sample_source_paths.add(observation.source_path)
        if observation.link_group_key is not None:
            bucket.link_groups.add(observation.link_group_key)
        if (
            observation.artifact_kind == ArtifactKind.AGENT_SIDECAR_META.value
            and observation.link_group_key is not None
            and observation.link_group_key in stream_keys_by_provider.get(provider, set())
        ):
            bucket.linked_sidecar_count += 1

    summaries = [
        ArtifactCohortSummary(
            provider_name=bucket.provider_name,
            payload_provider=bucket.payload_provider,
            artifact_kind=bucket.artifact_kind,
            support_status=bucket.support_status,
            cohort_id=bucket.cohort_id,
            observation_count=bucket.observation_count,
            unique_raw_ids=len(bucket.raw_ids),
            first_observed_at=bucket.first_observed_at,
            last_observed_at=bucket.last_observed_at,
            bundle_scope_count=len(bucket.bundle_scopes),
            sample_source_paths=sorted(bucket.sample_source_paths)[:_MAX_SAMPLE_PATHS],
            resolved_package_version=bucket.resolved_package_version,
            resolved_element_kind=bucket.resolved_element_kind,
            resolution_reason=bucket.resolution_reason,
            link_group_count=len(bucket.link_groups),
            linked_sidecar_count=bucket.linked_sidecar_count,
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


__all__ = ["list_artifact_cohorts", "list_artifact_observations"]
