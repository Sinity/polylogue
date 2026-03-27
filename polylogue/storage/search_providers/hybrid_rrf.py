"""Reciprocal Rank Fusion helpers for hybrid search."""

from __future__ import annotations


def reciprocal_rank_fusion(
    *result_lists: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Combine multiple ranked result lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}

    for result_list in result_lists:
        seen_in_list: set[str] = set()
        for rank, (item_id, _original_score) in enumerate(result_list, start=1):
            if item_id in seen_in_list:
                continue
            seen_in_list.add(item_id)
            rrf_score = 1.0 / (k + rank)
            scores[item_id] = scores.get(item_id, 0.0) + rrf_score

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


__all__ = ["reciprocal_rank_fusion"]
