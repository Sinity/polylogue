"""Helper utilities for schema-generation cluster analysis."""

from __future__ import annotations

from datetime import datetime, timezone

from polylogue.schemas.generation_models import _ClusterAccumulator, _ProfileSummary
from polylogue.schemas.observation import ProviderConfig, profile_similarity

_PROFILE_CORE_MIN_RATIO = 0.5
_PROFILE_MAX_TOKENS = 128
_PROFILE_SIMILARITY_THRESHOLDS = {
    "conversation_document": 0.86,
    "conversation_record_stream": 0.8,
    "subagent_conversation_stream": 0.8,
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


def _update_observed_window(acc: _ClusterAccumulator, observed_at: str | None) -> None:
    parsed = _parse_observed_at(observed_at)
    if parsed is None:
        return
    iso = parsed.astimezone(timezone.utc).isoformat()
    first_seen = _parse_observed_at(acc.first_seen)
    if acc.first_seen is None or first_seen is None or parsed < first_seen:
        acc.first_seen = iso
    last_seen = _parse_observed_at(acc.last_seen)
    if acc.last_seen is None or last_seen is None or parsed > last_seen:
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


__all__ = [
    "_artifact_priority",
    "_cluster_profile_tokens",
    "_cluster_reservoir_size",
    "_cluster_similarity",
    "_cluster_sort_key",
    "_dominant_keys_for_payload",
    "_merge_cluster_accumulators",
    "_merge_dominant_keys",
    "_merge_profile_summary",
    "_merge_representative_paths",
    "_new_cluster_accumulator",
    "_parse_observed_at",
    "_profile_similarity_threshold",
    "_refine_coarse_clusters",
    "_update_cluster_reservoir",
    "_update_observed_window",
]
