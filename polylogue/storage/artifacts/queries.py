"""Read-model transforms over inspected artifact observations.

These operate on in-memory ``ArtifactObservationRecord`` lists produced by
:func:`polylogue.storage.artifacts.persistence.materialize_artifact_observations`
so resolved schema-package and wire-format facts (not stored in
``raw_artifacts``) remain visible to listing and cohort surfaces (#1743).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from polylogue.archive.artifact_taxonomy import ArtifactKind
from polylogue.storage.artifacts.views import ArtifactCohortSummary
from polylogue.storage.query_models import ArtifactObservationListQuery
from polylogue.storage.runtime import ArtifactObservationRecord
from polylogue.types import ArtifactSupportStatus, Provider

_MAX_SAMPLE_PATHS = 5


def _effective_provider(record: ArtifactObservationRecord) -> str:
    return str(record.payload_provider or Provider.from_string(record.source_name or ""))


@dataclass(slots=True)
class _ArtifactCohortBucket:
    source_name: str
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


def _matches_filters(
    record: ArtifactObservationRecord,
    query: ArtifactObservationListQuery,
) -> bool:
    if query.providers and _effective_provider(record) not in set(query.providers):
        return False
    if query.support_statuses and record.support_status.value not in set(query.support_statuses):
        return False
    return not (query.artifact_kinds and record.artifact_kind not in set(query.artifact_kinds))


def filter_artifact_observations(
    records: Iterable[ArtifactObservationRecord],
    query: ArtifactObservationListQuery,
) -> list[ArtifactObservationRecord]:
    """Apply the durable-listing filters, ordering, and window in memory."""
    filtered = [record for record in records if _matches_filters(record, query)]
    filtered.sort(key=lambda r: r.source_index if r.source_index is not None else -1)
    filtered.sort(key=lambda r: r.source_path)
    filtered.sort(key=lambda r: r.last_observed_at or "", reverse=True)

    if query.limit is not None:
        start = max(0, query.offset)
        return filtered[start : start + max(0, int(query.limit))]
    if query.offset > 0:
        return filtered[max(0, query.offset) :]
    return filtered


def summarize_artifact_cohorts(
    records: Iterable[ArtifactObservationRecord],
    query: ArtifactObservationListQuery,
) -> list[ArtifactCohortSummary]:
    """Summarize artifact cohorts over the filtered observation set."""
    observations = filter_artifact_observations(records, query)
    if not observations:
        return []

    stream_keys_by_provider: dict[str, set[str]] = {}
    for observation in observations:
        if (
            observation.artifact_kind == ArtifactKind.SUBAGENT_SESSION_STREAM.value
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
                source_name=provider,
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
            source_name=bucket.source_name,
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
            item.source_name,
            item.artifact_kind,
            str(item.support_status),
            item.cohort_id or "",
        ),
    )


__all__ = ["filter_artifact_observations", "summarize_artifact_cohorts"]
