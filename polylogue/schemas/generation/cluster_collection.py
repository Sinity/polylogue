"""Cluster collection orchestration for schema generation."""

from __future__ import annotations

from pathlib import Path

from polylogue.schemas.generation_cluster_support import (
    _artifact_priority,
    _cluster_profile_tokens,
    _cluster_similarity,
    _dominant_keys_for_payload,
    _merge_cluster_accumulators,
    _merge_dominant_keys,
    _merge_profile_summary,
    _merge_representative_paths,
    _new_cluster_accumulator,
    _profile_similarity_threshold,
    _refine_coarse_clusters,
    _update_cluster_reservoir,
    _update_observed_window,
)
from polylogue.schemas.generation_models import (
    ClusterCollectionResult,
    _ClusterAccumulator,
    _ProfileSummary,
    _UnitMembership,
)
from polylogue.schemas.observation import profile_cluster_id


def _collect_cluster_accumulators(
    provider: str,
    *,
    db_path: Path,
    max_samples: int | None,
    reservoir_size: int,
    full_corpus: bool = False,
) -> tuple[dict[str, _ClusterAccumulator], list[_UnitMembership], int, dict[str, int]]:
    result = collect_cluster_analysis(
        provider,
        db_path=db_path,
        max_samples=max_samples,
        reservoir_size=reservoir_size,
        full_corpus=full_corpus,
    )
    return result.clusters, result.memberships, result.sample_count, result.artifact_counts


def collect_cluster_analysis(
    provider: str,
    *,
    db_path: Path,
    max_samples: int | None,
    reservoir_size: int,
    full_corpus: bool = False,
) -> ClusterCollectionResult:
    from polylogue.schemas.sampling import iter_schema_units

    units = list(iter_schema_units(provider, db_path=db_path, max_samples=max_samples, full_corpus=full_corpus))
    profile_summaries: dict[tuple[str, tuple[str, ...]], _ProfileSummary] = {}
    total_schema_samples = 0
    artifact_counts: dict[str, int] = {}

    for unit in units:
        summary_key = (unit.artifact_kind, unit.profile_tokens)
        summary = profile_summaries.get(summary_key)
        if summary is None:
            summary = _ProfileSummary(
                artifact_kind=unit.artifact_kind,
                profile_tokens=unit.profile_tokens,
                dominant_keys=_dominant_keys_for_payload(unit.cluster_payload)[:20],
            )
            profile_summaries[summary_key] = summary

        summary.sample_count += 1
        summary.schema_sample_count += len(unit.schema_samples)
        artifact_counts[unit.artifact_kind] = artifact_counts.get(unit.artifact_kind, 0) + 1

        if (
            unit.source_path
            and unit.source_path not in summary.representative_paths
            and len(summary.representative_paths) < 5
        ):
            summary.representative_paths.append(unit.source_path)

        total_schema_samples += len(unit.schema_samples)

    coarse_clusters: list[_ClusterAccumulator] = []
    ordered_summaries = sorted(
        profile_summaries.items(),
        key=lambda item: (
            _artifact_priority(item[1].artifact_kind),
            item[1].sample_count,
            item[1].schema_sample_count,
        ),
        reverse=True,
    )

    for _summary_key, summary in ordered_summaries:
        best_index: int | None = None
        best_score = 0.0
        for index, acc in enumerate(coarse_clusters):
            if acc.artifact_kind != summary.artifact_kind:
                continue
            score = _cluster_similarity(acc, summary.profile_tokens)
            if score > best_score:
                best_score = score
                best_index = index

        if best_index is not None and best_score >= _profile_similarity_threshold(summary.artifact_kind):
            _merge_profile_summary(coarse_clusters[best_index], summary)
            continue

        acc = _new_cluster_accumulator(
            artifact_kind=summary.artifact_kind,
            dominant_keys=summary.dominant_keys,
        )
        _merge_profile_summary(acc, summary)
        coarse_clusters.append(acc)

    coarse_clusters = _refine_coarse_clusters(coarse_clusters)

    clusters: dict[str, _ClusterAccumulator] = {}
    for acc in coarse_clusters:
        cluster_id = profile_cluster_id(acc.artifact_kind, _cluster_profile_tokens(acc))
        existing = clusters.get(cluster_id)
        if existing is None:
            clusters[cluster_id] = _new_cluster_accumulator(
                artifact_kind=acc.artifact_kind,
                dominant_keys=acc.dominant_keys,
            )
            clusters[cluster_id].representative_paths = list(acc.representative_paths)
            clusters[cluster_id].profile_token_counts.update(acc.profile_token_counts)
        else:
            existing.dominant_keys = _merge_dominant_keys(existing.dominant_keys, acc.dominant_keys)
            _merge_representative_paths(existing.representative_paths, acc.representative_paths)
            existing.profile_token_counts.update(acc.profile_token_counts)

    summary_cluster_ids: dict[tuple[str, tuple[str, ...]], str] = {}
    for acc in coarse_clusters:
        cluster_id = profile_cluster_id(acc.artifact_kind, _cluster_profile_tokens(acc))
        for profile_tokens in acc.member_profiles:
            summary_cluster_ids[(acc.artifact_kind, profile_tokens)] = cluster_id

    memberships: list[_UnitMembership] = []
    for unit in units:
        cluster_id = summary_cluster_ids[(unit.artifact_kind, unit.profile_tokens)]
        acc = clusters[cluster_id]
        acc.sample_count += 1
        acc.schema_sample_count += len(unit.schema_samples)
        acc.profile_token_counts.update(unit.profile_tokens)
        acc.member_profiles.add(unit.profile_tokens)
        acc.exact_structure_ids.add(unit.exact_structure_id)
        if unit.bundle_scope:
            acc.bundle_scopes.add(unit.bundle_scope)
        _update_observed_window(acc, unit.observed_at)
        if unit.source_path and unit.source_path not in acc.representative_paths and len(acc.representative_paths) < 5:
            acc.representative_paths.append(unit.source_path)
        memberships.append(_UnitMembership(unit=unit, profile_family_id=cluster_id))

    refined_clusters = _refine_coarse_clusters(
        list(clusters.values()),
        reservoir_size=reservoir_size,
    )
    final_clusters: dict[str, _ClusterAccumulator] = {}
    for acc in refined_clusters:
        cluster_id = profile_cluster_id(acc.artifact_kind, _cluster_profile_tokens(acc))
        existing = final_clusters.get(cluster_id)
        if existing is None:
            final_clusters[cluster_id] = acc
        else:
            _merge_cluster_accumulators(existing, acc, reservoir_size=reservoir_size)

    membership_cluster_map: dict[str, str] = {}
    for cluster_id, acc in final_clusters.items():
        for profile_tokens in acc.member_profiles:
            membership_cluster_map[profile_cluster_id(acc.artifact_kind, profile_tokens)] = cluster_id

    normalized_memberships: list[_UnitMembership] = []
    for membership in memberships:
        normalized_cluster_id = membership_cluster_map.get(membership.profile_family_id, membership.profile_family_id)
        normalized_memberships.append(_UnitMembership(unit=membership.unit, profile_family_id=normalized_cluster_id))
        final_acc = final_clusters[normalized_cluster_id]
        for sample in membership.unit.schema_samples:
            _update_cluster_reservoir(
                final_acc,
                sample,
                membership.unit.conversation_id,
                reservoir_size=reservoir_size,
            )

    return ClusterCollectionResult(
        clusters=final_clusters,
        memberships=normalized_memberships,
        sample_count=total_schema_samples,
        artifact_counts=artifact_counts,
    )


__all__ = [
    "_collect_cluster_accumulators",
    "collect_cluster_analysis",
]
