"""Cluster collection orchestration for schema generation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from polylogue.schemas.generation.cluster_support import (
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
    _update_observed_window,
)
from polylogue.schemas.generation.models import (
    ClusterCollectionResult,
    _ClusterAccumulator,
    _ProfileSummary,
    _UnitMembership,
)
from polylogue.schemas.generation.observation_journal import ObservationJournal
from polylogue.schemas.observation import SchemaUnit, profile_cluster_id


def _collect_cluster_accumulators(
    provider: str,
    *,
    db_path: Path,
    max_samples: int | None,
    full_corpus: bool = False,
    journal: ObservationJournal | None = None,
) -> tuple[dict[str, _ClusterAccumulator], Sequence[_UnitMembership], int, dict[str, int]]:
    result = collect_cluster_analysis(
        provider,
        db_path=db_path,
        max_samples=max_samples,
        full_corpus=full_corpus,
        journal=journal,
    )
    return result.clusters, result.memberships, result.sample_count, result.artifact_counts


def collect_cluster_analysis(
    provider: str,
    *,
    db_path: Path,
    max_samples: int | None,
    full_corpus: bool = False,
    journal: ObservationJournal | None = None,
) -> ClusterCollectionResult:
    from polylogue.schemas.sampling import iter_schema_units

    profile_summaries: dict[tuple[str, tuple[str, ...]], _ProfileSummary] = {}
    total_schema_samples = 0
    artifact_counts: dict[str, int] = {}
    retained_units: list[SchemaUnit] = []

    def observe_summary(unit: SchemaUnit) -> None:
        nonlocal total_schema_samples
        dominant_keys = _dominant_keys_for_payload(unit.cluster_payload)[:20]
        if journal is not None:
            journal.observe_profile_summary(unit, dominant_keys=dominant_keys)
            artifact_counts[unit.artifact_kind] = artifact_counts.get(unit.artifact_kind, 0) + 1
            total_schema_samples += len(unit.schema_samples)
            return
        summary_key = (unit.artifact_kind, unit.profile_tokens)
        summary = profile_summaries.get(summary_key)
        if summary is None:
            summary = _ProfileSummary(
                artifact_kind=unit.artifact_kind,
                profile_tokens=unit.profile_tokens,
                dominant_keys=dominant_keys,
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

    observed_units = iter_schema_units(
        provider,
        db_path=db_path,
        max_samples=max_samples,
        full_corpus=full_corpus,
        terminal_recorder=journal.record_terminal if journal is not None else None,
    )
    if journal is None:
        retained_units = list(observed_units)
        for unit in retained_units:
            observe_summary(unit)
    else:
        for unit in observed_units:
            observe_summary(unit)
            journal.append_unit(unit, retain_cluster_payload=False)

    coarse_clusters: list[_ClusterAccumulator] = []
    ordered_summaries: Iterable[tuple[object, _ProfileSummary]]
    if journal is None:
        ordered_summaries = sorted(
            profile_summaries.items(),
            key=lambda item: (
                _artifact_priority(item[1].artifact_kind),
                item[1].sample_count,
                item[1].schema_sample_count,
            ),
            reverse=True,
        )
    else:
        ordered_summaries = ((None, summary) for summary in journal.iter_profile_summaries())

    for _summary_key, summary in ordered_summaries:
        source_family_id = profile_cluster_id(summary.artifact_kind, summary.profile_tokens)
        if journal is not None:
            journal.assign_profile_summary_family(summary, source_family_id)
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
            _merge_profile_summary(
                coarse_clusters[best_index],
                summary,
                retain_member_profile=journal is None,
            )
            if journal is not None:
                coarse_clusters[best_index].source_family_ids.add(source_family_id)
            continue

        acc = _new_cluster_accumulator(
            artifact_kind=summary.artifact_kind,
            dominant_keys=summary.dominant_keys,
        )
        _merge_profile_summary(acc, summary, retain_member_profile=journal is None)
        if journal is not None:
            acc.source_family_ids.add(source_family_id)
        coarse_clusters.append(acc)

    coarse_clusters = _refine_coarse_clusters(coarse_clusters)
    if journal is not None:
        journal.normalize_profile_families(
            {
                source_family_id: profile_cluster_id(acc.artifact_kind, _cluster_profile_tokens(acc))
                for acc in coarse_clusters
                for source_family_id in acc.source_family_ids
            }
        )

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
            if journal is not None:
                clusters[cluster_id].source_family_ids.add(cluster_id)
        else:
            existing.dominant_keys = _merge_dominant_keys(existing.dominant_keys, acc.dominant_keys)
            _merge_representative_paths(existing.representative_paths, acc.representative_paths)
            existing.profile_token_counts.update(acc.profile_token_counts)

    summary_cluster_ids: dict[tuple[str, tuple[str, ...]], str] = {}
    if journal is None:
        for acc in coarse_clusters:
            cluster_id = profile_cluster_id(acc.artifact_kind, _cluster_profile_tokens(acc))
            for profile_tokens in acc.member_profiles:
                summary_cluster_ids[(acc.artifact_kind, profile_tokens)] = cluster_id
    else:
        journal.apply_profile_summary_families_to_units()

    memberships: list[_UnitMembership] = []
    identified_units = (
        (
            (unit_id, _UnitMembership(unit=unit, profile_family_id=""), len(unit.schema_samples))
            for unit_id, unit in enumerate(retained_units, start=1)
        )
        if journal is None
        else journal.iter_identified_membership_metadata()
    )
    for _unit_id, membership, unit_schema_sample_count in identified_units:
        unit = membership.unit
        cluster_id = (
            summary_cluster_ids[(unit.artifact_kind, unit.profile_tokens)]
            if journal is None
            else membership.profile_family_id
        )
        acc = clusters[cluster_id]
        acc.sample_count += 1
        acc.schema_sample_count += unit_schema_sample_count
        acc.profile_token_counts.update(unit.profile_tokens)
        if journal is None:
            acc.member_profiles.add(unit.profile_tokens)
        if journal is None:
            acc.exact_structure_ids.add(unit.exact_structure_id)
            if unit.bundle_scope:
                acc.bundle_scopes.add(unit.bundle_scope)
        _update_observed_window(acc, unit.observed_at)
        if unit.source_path and unit.source_path not in acc.representative_paths and len(acc.representative_paths) < 5:
            acc.representative_paths.append(unit.source_path)
        if journal is None:
            memberships.append(_UnitMembership(unit=unit, profile_family_id=cluster_id))

    refined_clusters = _refine_coarse_clusters(list(clusters.values()))
    initial_cluster_profiles = (
        {cluster_id: (acc.artifact_kind, frozenset(acc.member_profiles)) for cluster_id, acc in clusters.items()}
        if journal is None
        else {}
    )
    final_clusters: dict[str, _ClusterAccumulator] = {}
    for acc in refined_clusters:
        cluster_id = profile_cluster_id(acc.artifact_kind, _cluster_profile_tokens(acc))
        existing = final_clusters.get(cluster_id)
        if existing is None:
            final_clusters[cluster_id] = acc
        else:
            _merge_cluster_accumulators(existing, acc)

    membership_cluster_map: dict[str, str] = {}
    if journal is None:
        for cluster_id, acc in final_clusters.items():
            for profile_tokens in acc.member_profiles:
                membership_cluster_map[profile_cluster_id(acc.artifact_kind, profile_tokens)] = cluster_id
        for initial_cluster_id, (artifact_kind, member_profiles) in initial_cluster_profiles.items():
            matches = [
                final_cluster_id
                for final_cluster_id, acc in final_clusters.items()
                if acc.artifact_kind == artifact_kind and member_profiles.issubset(acc.member_profiles)
            ]
            if len(matches) != 1:
                raise RuntimeError(f"Initial profile family {initial_cluster_id} maps to {len(matches)} final families")
            membership_cluster_map[initial_cluster_id] = matches[0]
    else:
        membership_cluster_map = {
            source_family_id: final_cluster_id
            for final_cluster_id, acc in final_clusters.items()
            for source_family_id in acc.source_family_ids
        }

    if journal is None:
        normalized_memberships: Sequence[_UnitMembership] = [
            _UnitMembership(
                unit=membership.unit,
                profile_family_id=membership_cluster_map.get(
                    membership.profile_family_id,
                    membership.profile_family_id,
                ),
            )
            for membership in memberships
        ]
    else:
        journal.normalize_profile_families(membership_cluster_map)
        normalized_memberships = journal.memberships()

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
