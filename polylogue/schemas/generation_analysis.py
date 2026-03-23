"""Cluster analysis and package-candidate assembly for schema generation."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from polylogue.schemas.generation_models import (
    ClusterCollectionResult,
    PackageAssemblyResult,
    _ClusterAccumulator,
    _PackageAccumulator,
    _ProfileSummary,
    _UnitMembership,
)
from polylogue.schemas.observation import ProviderConfig, profile_cluster_id, profile_similarity
from polylogue.schemas.sampling import iter_schema_units

_PROFILE_CORE_MIN_RATIO = 0.5
_PROFILE_MAX_TOKENS = 128
_PROFILE_SIMILARITY_THRESHOLDS = {
    "conversation_document": 0.86,
    "conversation_record_stream": 0.8,
    "subagent_conversation_stream": 0.8,
}
_ANCHOR_ELEMENT_KINDS = {
    "conversation_document",
    "conversation_record_stream",
}


def _artifact_priority(artifact_kind: str) -> int:
    priorities = {
        "conversation_document": 120,
        "conversation_record_stream": 120,
        "subagent_conversation_stream": 90,
    }
    return priorities.get(artifact_kind, 0)


def _dominant_keys_for_payload(payload: object) -> list[str]:
    if isinstance(payload, dict):
        return sorted(payload.keys())
    if isinstance(payload, list):
        first_dict = next((item for item in payload if isinstance(item, dict)), None)
        if isinstance(first_dict, dict):
            return sorted(first_dict.keys())
    return []


def _cluster_sort_key(item: tuple[str, _ClusterAccumulator]) -> tuple[int, int, int]:
    _cluster_id, acc = item
    return (_artifact_priority(acc.artifact_kind), acc.sample_count, acc.schema_sample_count)


def _parse_observed_at(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _update_observed_window(acc: _ClusterAccumulator | _PackageAccumulator, observed_at: str | None) -> None:
    parsed = _parse_observed_at(observed_at)
    if parsed is None:
        return
    iso = parsed.astimezone(timezone.utc).isoformat()
    if acc.first_seen is None or _parse_observed_at(acc.first_seen) is None or parsed < _parse_observed_at(acc.first_seen):
        acc.first_seen = iso
    if acc.last_seen is None or _parse_observed_at(acc.last_seen) is None or parsed > _parse_observed_at(acc.last_seen):
        acc.last_seen = iso


def _cluster_reservoir_size(config: ProviderConfig, max_samples: int | None) -> int:
    if max_samples is not None:
        return max(64, min(max_samples, 500))
    if config.sample_granularity == "record":
        return 192
    return 500


def _profile_similarity_threshold(artifact_kind: str) -> float:
    return _PROFILE_SIMILARITY_THRESHOLDS.get(artifact_kind, 0.84)


def _cluster_profile_tokens(acc: _ClusterAccumulator) -> tuple[str, ...]:
    token_counts = getattr(acc, "profile_token_counts", None)
    if not token_counts:
        dominant_keys = getattr(acc, "dominant_keys", ())
        return tuple(f"field:{key}" for key in dominant_keys[:_PROFILE_MAX_TOKENS])

    min_count = 1 if acc.sample_count <= 2 else max(2, int(acc.sample_count * _PROFILE_CORE_MIN_RATIO))
    tokens = sorted(token for token, count in token_counts.items() if count >= min_count)
    if not tokens:
        tokens = [
            token
            for token, _count in sorted(
                token_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:_PROFILE_MAX_TOKENS]
        ]
    return tuple(tokens[:_PROFILE_MAX_TOKENS])


def _cluster_similarity(acc: _ClusterAccumulator, profile_tokens: tuple[str, ...]) -> float:
    if not profile_tokens:
        return 0.0
    return profile_similarity(set(_cluster_profile_tokens(acc)), set(profile_tokens))


def _merge_representative_paths(target: list[str], source: list[str]) -> None:
    for path in source:
        if path not in target and len(target) < 5:
            target.append(path)


def _membership_observed_window(memberships: list[_UnitMembership]) -> tuple[str, str] | tuple[None, None]:
    first_seen: str | None = None
    last_seen: str | None = None
    for membership in memberships:
        parsed = _parse_observed_at(membership.unit.observed_at)
        if parsed is None:
            continue
        iso = parsed.astimezone(timezone.utc).isoformat()
        if first_seen is None or parsed < _parse_observed_at(first_seen):
            first_seen = iso
        if last_seen is None or parsed > _parse_observed_at(last_seen):
            last_seen = iso
    return first_seen, last_seen


def _merge_dominant_keys(target: list[str], source: list[str]) -> list[str]:
    merged = list(dict.fromkeys([*target, *source]))
    return merged[:20]


def _new_cluster_accumulator(*, artifact_kind: str, dominant_keys: list[str]) -> _ClusterAccumulator:
    return _ClusterAccumulator(
        artifact_kind=artifact_kind,
        dominant_keys=dominant_keys[:20],
    )


def _merge_profile_summary(acc: _ClusterAccumulator, summary: _ProfileSummary) -> None:
    acc.sample_count += summary.sample_count
    acc.schema_sample_count += summary.schema_sample_count
    acc.profile_token_counts.update(dict.fromkeys(summary.profile_tokens, summary.sample_count))
    acc.member_profiles.add(summary.profile_tokens)
    acc.dominant_keys = _merge_dominant_keys(acc.dominant_keys, summary.dominant_keys)
    _merge_representative_paths(acc.representative_paths, summary.representative_paths)


def _merge_cluster_accumulators(
    target: _ClusterAccumulator,
    source: _ClusterAccumulator,
    *,
    reservoir_size: int | None = None,
) -> None:
    target.sample_count += source.sample_count
    target.schema_sample_count += source.schema_sample_count
    target.profile_token_counts.update(source.profile_token_counts)
    target.member_profiles.update(source.member_profiles)
    target.dominant_keys = _merge_dominant_keys(target.dominant_keys, source.dominant_keys)
    _merge_representative_paths(target.representative_paths, source.representative_paths)
    target.exact_structure_ids.update(source.exact_structure_ids)
    target.bundle_scopes.update(source.bundle_scopes)
    _update_observed_window(target, source.first_seen)
    _update_observed_window(target, source.last_seen)
    if reservoir_size is not None and source.reservoir_samples:
        combined = list(zip(target.reservoir_samples, target.reservoir_conv_ids, strict=False))
        combined.extend(zip(source.reservoir_samples, source.reservoir_conv_ids, strict=False))
        if len(combined) > reservoir_size:
            target.rng.shuffle(combined)
            combined = combined[:reservoir_size]
        target.reservoir_samples = [sample for sample, _conv_id in combined]
        target.reservoir_conv_ids = [conv_id for _sample, conv_id in combined]


def _refine_coarse_clusters(
    coarse_clusters: list[_ClusterAccumulator],
    *,
    reservoir_size: int | None = None,
) -> list[_ClusterAccumulator]:
    clusters = list(coarse_clusters)
    while True:
        best_pair: tuple[int, int, float] | None = None
        for left_index, left in enumerate(clusters):
            for right_index in range(left_index + 1, len(clusters)):
                right = clusters[right_index]
                if left.artifact_kind != right.artifact_kind:
                    continue
                score = profile_similarity(
                    set(_cluster_profile_tokens(left)),
                    set(_cluster_profile_tokens(right)),
                )
                if score < _profile_similarity_threshold(left.artifact_kind):
                    continue
                if best_pair is None or score > best_pair[2]:
                    best_pair = (left_index, right_index, score)

        if best_pair is None:
            return clusters

        left_index, right_index, _score = best_pair
        left = clusters[left_index]
        right = clusters[right_index]
        if right.sample_count > left.sample_count or (
            right.sample_count == left.sample_count and right.schema_sample_count > left.schema_sample_count
        ):
            left_index, right_index = right_index, left_index
            left, right = right, left

        _merge_cluster_accumulators(left, right, reservoir_size=reservoir_size)
        del clusters[right_index]


def _update_cluster_reservoir(
    acc: _ClusterAccumulator,
    sample: dict[str, object],
    conversation_id: str | None,
    *,
    reservoir_size: int,
) -> None:
    acc.schema_sample_count += 1
    if len(acc.reservoir_samples) < reservoir_size:
        acc.reservoir_samples.append(sample)
        acc.reservoir_conv_ids.append(conversation_id)
        return

    slot = acc.rng.randint(0, acc.schema_sample_count - 1)
    if slot < reservoir_size:
        acc.reservoir_samples[slot] = sample
        acc.reservoir_conv_ids[slot] = conversation_id


def _collect_cluster_accumulators(
    provider: str,
    *,
    db_path: Path,
    max_samples: int | None,
    reservoir_size: int,
) -> tuple[dict[str, _ClusterAccumulator], list[_UnitMembership], int, dict[str, int]]:
    """Cluster schema units into profile families and retain bundle memberships."""
    result = collect_cluster_analysis(
        provider,
        db_path=db_path,
        max_samples=max_samples,
        reservoir_size=reservoir_size,
    )
    return result.clusters, result.memberships, result.sample_count, result.artifact_counts


def collect_cluster_analysis(
    provider: str,
    *,
    db_path: Path,
    max_samples: int | None,
    reservoir_size: int,
) -> ClusterCollectionResult:
    units = list(iter_schema_units(provider, db_path=db_path, max_samples=max_samples))
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

        if unit.source_path and unit.source_path not in summary.representative_paths and len(summary.representative_paths) < 5:
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


def _membership_scope_key(membership: _UnitMembership) -> str:
    unit = membership.unit
    return (
        unit.bundle_scope
        or unit.raw_id
        or unit.source_path
        or f"{membership.profile_family_id}:{unit.artifact_kind}:{unit.exact_structure_id}"
    )


def _dedupe_bundle_memberships(memberships: list[_UnitMembership]) -> dict[str, list[_UnitMembership]]:
    scoped: dict[str, list[_UnitMembership]] = {}
    for membership in memberships:
        scoped.setdefault(_membership_scope_key(membership), []).append(membership)

    deduped: dict[str, list[_UnitMembership]] = {}
    for scope, items in scoped.items():
        items = sorted(
            items,
            key=lambda item: (
                item.unit.observed_at or "",
                item.unit.source_path or "",
                item.profile_family_id,
            ),
        )
        seen: set[tuple[str, str]] = set()
        retained: list[_UnitMembership] = []
        for membership in items:
            dedupe_key = (membership.unit.artifact_kind, membership.unit.exact_structure_id)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            retained.append(membership)
        deduped[scope] = retained
    return deduped


def _attach_package_membership(package: _PackageAccumulator, membership: _UnitMembership, *, scope: str) -> None:
    package.memberships.append(membership)
    package.bundle_scopes.add(scope)
    package.profile_family_ids.add(membership.profile_family_id)
    if membership.unit.source_path:
        _merge_representative_paths(package.representative_paths, [membership.unit.source_path])
    _update_observed_window(package, membership.unit.observed_at)


def _build_package_candidates(
    provider: str,
    *,
    memberships: list[_UnitMembership],
    clusters: dict[str, _ClusterAccumulator],
) -> tuple[list[_PackageAccumulator], dict[str, int]]:
    """Compatibility wrapper retained for tests while the workflow is split."""
    result = assemble_package_candidates(
        provider,
        memberships=memberships,
        clusters=clusters,
    )
    return result.packages, result.orphan_adjunct_counts


def assemble_package_candidates(
    provider: str,
    *,
    memberships: list[_UnitMembership],
    clusters: dict[str, _ClusterAccumulator],
) -> PackageAssemblyResult:
    scoped = _dedupe_bundle_memberships(memberships)
    packages: dict[str, _PackageAccumulator] = {}
    orphan_adjunct_counts: Counter[str] = Counter()

    for scope, items in scoped.items():
        anchor_families = sorted(
            {
                membership.profile_family_id
                for membership in items
                if membership.unit.artifact_kind in _ANCHOR_ELEMENT_KINDS
            }
        )

        if not anchor_families:
            for membership in items:
                orphan_adjunct_counts[membership.unit.artifact_kind] += 1
            continue

        for family_id in anchor_families:
            acc = packages.get(family_id)
            if acc is None:
                cluster = clusters[family_id]
                acc = _PackageAccumulator(
                    provider=provider,
                    anchor_family_id=family_id,
                    anchor_kind=cluster.artifact_kind,
                )
                packages[family_id] = acc
            for membership in items:
                if membership.profile_family_id == family_id and membership.unit.artifact_kind in _ANCHOR_ELEMENT_KINDS:
                    _attach_package_membership(acc, membership, scope=scope)

        if len(anchor_families) == 1:
            target = packages[anchor_families[0]]
            for membership in items:
                if membership.unit.artifact_kind not in _ANCHOR_ELEMENT_KINDS:
                    _attach_package_membership(target, membership, scope=scope)
        else:
            for membership in items:
                if membership.unit.artifact_kind not in _ANCHOR_ELEMENT_KINDS:
                    orphan_adjunct_counts[membership.unit.artifact_kind] += 1

    ordered = sorted(
        packages.values(),
        key=lambda item: (
            _parse_observed_at(item.first_seen) or datetime.max.replace(tzinfo=timezone.utc),
            -len(item.bundle_scopes),
            item.anchor_family_id,
        ),
    )
    return PackageAssemblyResult(
        packages=ordered,
        orphan_adjunct_counts=dict(orphan_adjunct_counts),
    )


def _element_profile_tokens(memberships: list[_UnitMembership]) -> list[str]:
    token_counts: Counter[str] = Counter()
    for membership in memberships:
        token_counts.update(membership.unit.profile_tokens)
    if not token_counts:
        return []
    min_count = max(1, len(memberships) // 2)
    tokens = sorted(token for token, count in token_counts.items() if count >= min_count)
    if not tokens:
        tokens = [
            token
            for token, _count in sorted(
                token_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:_PROFILE_MAX_TOKENS]
        ]
    return tokens[:_PROFILE_MAX_TOKENS]


__all__ = [
    "assemble_package_candidates",
    "collect_cluster_analysis",
    "_artifact_priority",
    "_build_package_candidates",
    "_cluster_profile_tokens",
    "_cluster_reservoir_size",
    "_cluster_sort_key",
    "_collect_cluster_accumulators",
    "_element_profile_tokens",
    "_membership_observed_window",
    "_merge_representative_paths",
    "_parse_observed_at",
]
