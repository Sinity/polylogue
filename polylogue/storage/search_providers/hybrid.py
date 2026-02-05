"""Hybrid search provider combining FTS5 and vector search with RRF fusion.

This module provides a SearchProvider implementation that combines full-text
search (FTS5) with semantic vector search (sqlite-vec) using Reciprocal Rank Fusion.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.protocols import VectorProvider
    from polylogue.storage.search_providers.fts5 import FTS5Provider


def reciprocal_rank_fusion(
    *result_lists: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Combine multiple ranked result lists using Reciprocal Rank Fusion.

    RRF is a simple but effective method for combining search results from
    different ranking sources. Each result receives a score based on its
    position: score = 1 / (k + rank), where k is a constant (default 60).

    The k parameter prevents over-weighting of top results. Higher k values
    reduce the influence of rank position differences.

    Reference: Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009).
    "Reciprocal rank fusion outperforms Condorcet and individual rank learning methods"

    Args:
        *result_lists: Variable number of ranked result lists, each containing
            (item_id, score) tuples. Scores from input lists are ignored;
            only the ranking matters.
        k: Rank fusion constant (default: 60). Higher values reduce the
            importance of rank differences.

    Returns:
        Fused results as (item_id, rrf_score) tuples, sorted by descending score.

    Example:
        >>> fts_results = [("msg1", 0.9), ("msg2", 0.8), ("msg3", 0.7)]
        >>> vec_results = [("msg2", 0.95), ("msg1", 0.85), ("msg4", 0.6)]
        >>> fused = reciprocal_rank_fusion(fts_results, vec_results)
        >>> # msg1 and msg2 score higher because they appear in both lists
    """
    scores: dict[str, float] = {}

    for result_list in result_lists:
        for rank, (item_id, _original_score) in enumerate(result_list, start=1):
            rrf_score = 1.0 / (k + rank)
            scores[item_id] = scores.get(item_id, 0.0) + rrf_score

    # Sort by fused score descending
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridSearchProvider:
    """SearchProvider combining FTS5 and vector search with RRF fusion.

    This provider executes both full-text search (FTS5) and semantic vector
    search (sqlite-vec) in parallel, then combines results using Reciprocal Rank
    Fusion to produce a unified ranking.

    Hybrid search leverages the complementary strengths of both approaches:
    - FTS5: Exact keyword matching, phrase search, boolean operators
    - Vector: Semantic similarity, synonym handling, concept matching

    Attributes:
        fts_provider: FTS5 provider for full-text search
        vector_provider: Vector provider for semantic search (SqliteVecProvider)
        rrf_k: RRF constant (default: 60)
    """

    def __init__(
        self,
        fts_provider: FTS5Provider,
        vector_provider: VectorProvider,
        *,
        rrf_k: int = 60,
    ) -> None:
        """Initialize hybrid search provider.

        Args:
            fts_provider: FTS5 provider for full-text search
            vector_provider: Vector provider for semantic search
            rrf_k: RRF fusion constant (default: 60)
        """
        self.fts_provider = fts_provider
        self.vector_provider = vector_provider
        self.rrf_k = rrf_k

    def search(self, query: str, limit: int = 20) -> list[tuple[str, float]]:
        """Execute hybrid search combining FTS5 and vector search.

        Runs both search methods and combines results using RRF fusion.
        Returns message IDs with their fused scores.

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of (message_id, rrf_score) tuples, sorted by descending score.
        """
        # Get FTS5 results (returns message IDs)
        # Fetch more than limit to ensure good fusion
        fts_limit = limit * 3
        fts_message_ids = self.fts_provider.search(query)[:fts_limit]

        # Convert to (id, rank_score) format - higher rank = higher score
        fts_results = [
            (msg_id, 1.0 / (i + 1)) for i, msg_id in enumerate(fts_message_ids)
        ]

        # Get vector search results
        vec_results = self.vector_provider.query(query, limit=fts_limit)

        # Combine using RRF
        fused = reciprocal_rank_fusion(
            fts_results,
            vec_results,
            k=self.rrf_k,
        )

        return fused[:limit]

    def search_conversations(self, query: str, limit: int = 20) -> list[str]:
        """Search and return unique conversation IDs.

        Executes hybrid search, then deduplicates by conversation.
        Returns conversation IDs that had matching messages.

        Args:
            query: Search query string
            limit: Maximum number of conversations to return

        Returns:
            List of conversation IDs, ordered by best-matching message score.
        """
        from polylogue.storage.backends.sqlite import open_connection

        # Get message-level results
        message_results = self.search(query, limit=limit * 3)

        if not message_results:
            return []

        message_ids = [msg_id for msg_id, _score in message_results]

        # Look up conversation IDs for these messages
        with open_connection(self.fts_provider.db_path) as conn:
            placeholders = ",".join("?" * len(message_ids))
            rows = conn.execute(
                f"SELECT message_id, conversation_id FROM messages WHERE message_id IN ({placeholders})",
                message_ids,
            ).fetchall()

        msg_to_conv = {row["message_id"]: row["conversation_id"] for row in rows}

        # Deduplicate while preserving order (first occurrence wins)
        seen_convs: set[str] = set()
        result_convs: list[str] = []

        for msg_id, _score in message_results:
            conv_id = msg_to_conv.get(msg_id)
            if conv_id and conv_id not in seen_convs:
                seen_convs.add(conv_id)
                result_convs.append(conv_id)
                if len(result_convs) >= limit:
                    break

        return result_convs


def create_hybrid_provider(
    db_path: Path | None = None,
    vector_provider: VectorProvider | None = None,
    rrf_k: int = 60,
) -> HybridSearchProvider | None:
    """Create a hybrid search provider if vector search is available.

    Factory function that creates a HybridSearchProvider if both FTS5
    and vector search are available. Returns None if vector search is
    not configured.

    Args:
        db_path: Optional database path for FTS5 provider
        vector_provider: Optional vector provider (uses default if None)
        rrf_k: RRF fusion constant

    Returns:
        HybridSearchProvider if vector search is available, None otherwise.
    """
    from polylogue.storage.search_providers import create_vector_provider
    from polylogue.storage.search_providers.fts5 import FTS5Provider

    # Get or create vector provider
    vec_provider = vector_provider or create_vector_provider()
    if vec_provider is None:
        return None

    # Create FTS5 provider
    fts_provider = FTS5Provider(db_path=db_path)

    return HybridSearchProvider(
        fts_provider=fts_provider,
        vector_provider=vec_provider,
        rrf_k=rrf_k,
    )


__all__ = [
    "HybridSearchProvider",
    "reciprocal_rank_fusion",
    "create_hybrid_provider",
]
