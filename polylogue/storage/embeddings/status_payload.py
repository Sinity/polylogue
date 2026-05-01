"""Embedding-status payload builder (substrate, click-free).

Counts come from a sync read connection over the embedding-status tables.
Surfaces (CLI, MCP, dashboards) consume :func:`embedding_status_payload`
and render in their own dialect.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Protocol

from typing_extensions import TypedDict

from polylogue.storage.embeddings.models import EmbeddingStatsSnapshot

if TYPE_CHECKING:
    from polylogue.config import Config


class _HasConfig(Protocol):
    @property
    def config(self) -> Config: ...


class RetrievalBandPayload(TypedDict, total=False):
    ready: bool
    status: str
    materialized_rows: int
    source_rows: int
    materialized_documents: int
    source_documents: int


class EmbeddingStatusPayload(TypedDict):
    status: str
    total_conversations: int
    embedded_conversations: int
    embedded_messages: int
    pending_conversations: int
    embedding_coverage_percent: float
    retrieval_ready: bool
    freshness_status: str
    stale_messages: int
    messages_missing_provenance: int
    oldest_embedded_at: str | None
    newest_embedded_at: str | None
    embedding_models: dict[str, int]
    embedding_dimensions: dict[int, int]
    retrieval_bands: dict[str, dict[str, object]]


def _payload_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _total_conversations(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()
    return _payload_int(row[0]) if row is not None else 0


def _coverage_percent(*, embedded_conversations: int, total_conversations: int) -> float:
    if total_conversations <= 0:
        return 0.0
    return embedded_conversations / total_conversations * 100


def _embedding_status(
    *,
    total_conversations: int,
    embedded_conversations: int,
    pending_conversations: int,
) -> str:
    if total_conversations <= 0:
        return "empty"
    if embedded_conversations <= 0:
        return "none"
    if pending_conversations > 0:
        return "partial"
    return "complete"


def _freshness_status(status: str, stats: EmbeddingStatsSnapshot) -> str:
    if stats.embedded_messages > 0 and (stats.stale_messages > 0 or stats.messages_missing_provenance > 0):
        return "stale"
    return status


def _retrieval_ready(stats: EmbeddingStatsSnapshot) -> bool:
    return stats.embedded_messages > stats.stale_messages


def _payload_from_stats(
    *,
    total_conversations: int,
    stats: EmbeddingStatsSnapshot,
) -> EmbeddingStatusPayload:
    embedded_conversations = stats.embedded_conversations
    pending_conversations = stats.pending_conversations or max(total_conversations - embedded_conversations, 0)
    status = _embedding_status(
        total_conversations=total_conversations,
        embedded_conversations=embedded_conversations,
        pending_conversations=pending_conversations,
    )
    return {
        "status": status,
        "total_conversations": total_conversations,
        "embedded_conversations": embedded_conversations,
        "embedded_messages": stats.embedded_messages,
        "pending_conversations": pending_conversations,
        "embedding_coverage_percent": round(
            _coverage_percent(
                embedded_conversations=embedded_conversations,
                total_conversations=total_conversations,
            ),
            1,
        ),
        "retrieval_ready": _retrieval_ready(stats),
        "freshness_status": _freshness_status(status, stats),
        "stale_messages": stats.stale_messages,
        "messages_missing_provenance": stats.messages_missing_provenance,
        "oldest_embedded_at": stats.oldest_embedded_at,
        "newest_embedded_at": stats.newest_embedded_at,
        "embedding_models": stats.model_counts,
        "embedding_dimensions": stats.dimension_counts,
        "retrieval_bands": stats.retrieval_bands,
    }


def embedding_status_payload(env: _HasConfig) -> EmbeddingStatusPayload:
    """Read canonical embedding-status statistics for operator surfaces."""
    from polylogue.storage.backends.connection import open_read_connection
    from polylogue.storage.embeddings.embedding_stats import read_embedding_stats_sync

    with open_read_connection(env.config.db_path) as conn:
        total_conversations = _total_conversations(conn)
        embedding_stats = read_embedding_stats_sync(conn)

    return _payload_from_stats(total_conversations=total_conversations, stats=embedding_stats)
