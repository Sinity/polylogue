"""Artifact-proof and durable-observation workflows."""

from __future__ import annotations

from pathlib import Path

from polylogue.lib.artifact_taxonomy import ArtifactKind
from polylogue.storage.artifact_observations import (
    ensure_artifact_observations,
)
from polylogue.storage.artifact_observations import (
    list_artifact_cohorts as list_durable_artifact_cohorts,
)
from polylogue.storage.artifact_observations import (
    list_artifact_observations as list_durable_artifact_observations,
)
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.state_views import ArtifactCohortSummary
from polylogue.storage.store import ArtifactObservationRecord
from polylogue.types import ArtifactSupportStatus, Provider

from .verification_models import ArtifactProofReport, ProviderArtifactProof
from .verification_requests import ArtifactObservationQuery, ArtifactProofRequest
from .verification_support import bounded_window


def _increment_count(counter: dict[str, int], key: str, amount: int = 1) -> None:
    counter[key] = counter.get(key, 0) + amount


def list_artifact_observation_rows(
    *,
    db_path: Path,
    request: ArtifactObservationQuery,
) -> list[ArtifactObservationRecord]:
    """Return durable artifact observations, hydrating historical rows as needed."""
    if not db_path.exists():
        return []

    bounded_limit, bounded_offset = bounded_window(request.record_limit, request.record_offset)
    with open_connection(db_path) as conn:
        ensure_artifact_observations(conn, providers=request.providers, refresh_resolutions=True)
        return list_durable_artifact_observations(
            conn,
            providers=request.providers,
            support_statuses=request.support_statuses,
            artifact_kinds=request.artifact_kinds,
            limit=bounded_limit,
            offset=bounded_offset,
        )


def list_artifact_cohort_rows(
    *,
    db_path: Path,
    request: ArtifactObservationQuery,
) -> list[ArtifactCohortSummary]:
    """Return durable artifact cohort summaries, hydrating historical rows as needed."""
    if not db_path.exists():
        return []

    bounded_limit, bounded_offset = bounded_window(request.record_limit, request.record_offset)
    with open_connection(db_path) as conn:
        ensure_artifact_observations(conn, providers=request.providers, refresh_resolutions=True)
        return list_durable_artifact_cohorts(
            conn,
            providers=request.providers,
            support_statuses=request.support_statuses,
            artifact_kinds=request.artifact_kinds,
            limit=bounded_limit,
            offset=bounded_offset,
        )


def prove_raw_artifact_coverage(
    *,
    db_path: Path,
    request: ArtifactProofRequest,
) -> ArtifactProofReport:
    """Report durable artifact support, unknowns, and Claude sidecar linkage."""
    bounded_limit, bounded_offset = bounded_window(request.record_limit, request.record_offset)
    if not db_path.exists():
        return ArtifactProofReport(
            providers={},
            total_records=0,
            record_limit=bounded_limit,
            record_offset=bounded_offset,
        )

    stats_by_provider: dict[str, ProviderArtifactProof] = {}
    linkage_state: dict[str, dict[str, set[str]]] = {}
    observations = list_artifact_observation_rows(
        db_path=db_path,
        request=ArtifactObservationQuery(
            providers=request.providers,
            record_limit=request.record_limit,
            record_offset=request.record_offset,
        ),
    )
    total_records = len(observations)

    for observation in observations:
        provider = str(observation.payload_provider or Provider.from_string(observation.provider_name))
        stats = stats_by_provider.setdefault(
            provider,
            ProviderArtifactProof(provider=provider),
        )
        stats.total_records += 1
        _increment_count(stats.artifact_counts, observation.artifact_kind)

        if observation.link_group_key is not None:
            state = linkage_state.setdefault(provider, {"sidecars": set(), "streams": set()})
            if observation.artifact_kind == ArtifactKind.AGENT_SIDECAR_META.value:
                state["sidecars"].add(observation.link_group_key)
                if observation.sidecar_agent_type is not None:
                    _increment_count(stats.sidecar_agent_types, observation.sidecar_agent_type)
            elif observation.artifact_kind == ArtifactKind.SUBAGENT_CONVERSATION_STREAM.value:
                state["streams"].add(observation.link_group_key)

        if observation.support_status is ArtifactSupportStatus.SUPPORTED_PARSEABLE:
            stats.contract_backed_records += 1
            if observation.resolved_package_version is not None:
                _increment_count(stats.package_versions, observation.resolved_package_version)
            if observation.resolved_element_kind is not None:
                _increment_count(stats.element_kinds, observation.resolved_element_kind)
            if observation.resolution_reason is not None:
                _increment_count(stats.resolution_reasons, observation.resolution_reason)
        elif observation.support_status is ArtifactSupportStatus.UNSUPPORTED_PARSEABLE:
            stats.unsupported_parseable_records += 1
        elif observation.support_status is ArtifactSupportStatus.RECOGNIZED_UNPARSED:
            stats.recognized_non_parseable_records += 1
        elif observation.support_status is ArtifactSupportStatus.UNKNOWN:
            stats.unknown_records += 1
        elif observation.support_status is ArtifactSupportStatus.DECODE_FAILED:
            stats.decode_errors += 1

    for provider, state in linkage_state.items():
        stats = stats_by_provider.setdefault(
            provider,
            ProviderArtifactProof(provider=provider),
        )
        linked = state["sidecars"] & state["streams"]
        stats.linked_sidecars = len(linked)
        stats.orphan_sidecars = len(state["sidecars"] - state["streams"])
        stats.subagent_streams = len(state["streams"])
        stats.streams_with_sidecars = len(linked)

    return ArtifactProofReport(
        providers=stats_by_provider,
        total_records=total_records,
        record_limit=bounded_limit,
        record_offset=bounded_offset,
    )


__all__ = [
    "list_artifact_cohort_rows",
    "list_artifact_observation_rows",
    "prove_raw_artifact_coverage",
]
