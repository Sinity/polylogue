"""Hybrid search provider combining FTS5 and vector search with RRF fusion.

This module provides a SearchProvider implementation that combines full-text
search (FTS5) with semantic vector search (sqlite-vec) using Reciprocal Rank Fusion.
"""

from __future__ import annotations

import sqlite3
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.storage.backends.connection import (
    open_connection as _open_connection,
)
from polylogue.storage.backends.connection import (
    open_read_connection,
)
from polylogue.storage.search_models import ConversationSearchResult
from polylogue.storage.search_providers.hybrid_conversations import (
    _resolve_ranked_conversation_hits,
    _resolve_ranked_conversation_ids,
)
from polylogue.storage.search_providers.hybrid_factory import create_hybrid_provider

if TYPE_CHECKING:
    from polylogue.protocols import VectorProvider
    from polylogue.storage.search_providers.fts5 import FTS5Provider
    from polylogue.storage.store import MessageRecord


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (formerly hybrid_rrf.py)
# ---------------------------------------------------------------------------


def open_connection(
    db_path: Path | str | sqlite3.Connection | None,
) -> AbstractContextManager[sqlite3.Connection]:
    """Return a readable connection for either DB paths or injected sqlite handles."""
    if isinstance(db_path, sqlite3.Connection):
        return _open_connection(db_path)
    return open_read_connection(db_path)


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


class HybridSearchProvider:
    """SearchProvider combining FTS5 and vector search with RRF fusion.

    This provider executes both full-text search (FTS5) and semantic vector
    search (sqlite-vec) in parallel, then combines results using Reciprocal Rank
    Fusion to produce a unified ranking.

    Hybrid search leverages the complementary strengths of both approaches:
    - FTS5: Exact keyword matching, phrase search, boolean operators
    - Vector: Semantic similarity, synonym handling, concept matching

    Conforms to the ``SearchProvider`` protocol (``search()`` returns ``list[str]``).
    Use ``search_scored()`` when RRF scores are needed.

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

    def index(self, messages: list[MessageRecord]) -> None:
        """Index messages via the underlying FTS5 provider.

        Delegates to the FTS5 provider's index method, conforming to
        the SearchProvider protocol.
        """
        self.fts_provider.index(messages)

    def search(self, query: str) -> list[str]:
        """Execute hybrid search conforming to SearchProvider protocol.

        Returns message IDs without scores. For scored results, use
        ``search_scored()``.

        Args:
            query: Search query string

        Returns:
            List of message IDs, ordered by descending RRF score.
        """
        return [msg_id for msg_id, _score in self.search_scored(query)]

    def search_scored(self, query: str, limit: int = 20) -> list[tuple[str, float]]:
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
        fts_message_ids = self.fts_provider.search(query, limit=fts_limit)

        # Convert to (id, rank_score) format - higher rank = higher score
        fts_results = [(msg_id, 1.0 / (i + 1)) for i, msg_id in enumerate(fts_message_ids)]

        # Get vector search results
        vec_results = self.vector_provider.query(query, limit=fts_limit)

        # Combine using RRF
        fused = reciprocal_rank_fusion(
            fts_results,
            vec_results,
            k=self.rrf_k,
        )

        return fused[:limit]

    def search_conversations(self, query: str, limit: int = 20, providers: list[str] | None = None) -> list[str]:
        """Search and return unique conversation IDs.

        Executes hybrid search, then deduplicates by conversation.
        Returns conversation IDs that had matching messages.

        Args:
            query: Search query string
            limit: Maximum number of conversations to return

        Returns:
            List of conversation IDs, ordered by best-matching message score.
        """
        return self.search_conversation_hits(query, limit=limit, providers=providers).conversation_ids()

    def search_conversation_hits(
        self,
        query: str,
        limit: int = 20,
        providers: list[str] | None = None,
    ) -> ConversationSearchResult:
        """Search and return ordered conversation hits."""
        if limit <= 0:
            return ConversationSearchResult(hits=[])

        # Get message-level results (scored for ranking)
        message_results = self.search_scored(query, limit=limit * 3)

        if not message_results:
            return ConversationSearchResult(hits=[])

        with open_connection(self.fts_provider.db_path) as conn:
            return _resolve_ranked_conversation_hits(
                conn,
                message_results=message_results,
                limit=limit,
                scope_names=providers,
            )


__all__ = [
    "HybridSearchProvider",
    "_resolve_ranked_conversation_hits",
    "_resolve_ranked_conversation_ids",
    "create_hybrid_provider",
    "reciprocal_rank_fusion",
]
