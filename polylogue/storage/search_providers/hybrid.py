"""Reciprocal Rank Fusion — the shared ranking primitive for hybrid retrieval.

polylogue-a7xr.10 (kill-or-adopt the search-provider lane): ``HybridSearchProvider``
and ``FTS5Provider`` had zero production call sites — production hybrid
retrieval (``cli/archive_query.py``, ``archive/query/archive_execution.py``,
``archive/query/retrieval_search.py``) has always fused FTS and vector
results inline against live query-plan state rather than through those
classes, which existed only in their own tests. Killed the unproven classes
(this module used to also define ``HybridSearchProvider``); kept
:func:`reciprocal_rank_fusion` — the one piece of this module production code
actually imports (directly from here, and via
``polylogue.storage.search_providers.reciprocal_rank_fusion``).

See ``polylogue.storage.search_providers.hybrid_sessions`` for the companion
session-resolution SQL helpers, which are also kept — not because anything
calls them today, but because they are the target for the still-open dedup
fix (session resolution is currently reimplemented inline in
``archive_execution.py`` rather than reusing this module's helper).
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    *result_lists: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Combine multiple ranked result lists using Reciprocal Rank Fusion.

    Result lists are interpreted positionally: the i-th element is
    treated as rank ``i+1``. The fused output is sorted by descending
    fused score with a deterministic ascending ``item_id`` secondary key
    for ties — so two items that earn the same fused score always come
    out in the same order, regardless of which lane discovered them
    first or how Python's dict happens to iterate. This makes cursor and
    offset pagination over tied scores stable across runs.
    """
    scores: dict[str, float] = {}

    for result_list in result_lists:
        seen_in_list: set[str] = set()
        for rank, (item_id, _original_score) in enumerate(result_list, start=1):
            if item_id in seen_in_list:
                continue
            seen_in_list.add(item_id)
            rrf_score = 1.0 / (k + rank)
            scores[item_id] = scores.get(item_id, 0.0) + rrf_score

    # Primary: descending fused score. Secondary: ascending item_id so tied
    # scores produce a stable, permutation-invariant ordering.
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


__all__ = ["reciprocal_rank_fusion"]
