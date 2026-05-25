"""Embedding-status payload builder (substrate, click-free).

Counts come from a sync read connection over the embedding-status tables.
Surfaces (CLI, MCP, dashboards) consume :func:`embedding_status_payload`
and render in their own dialect.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from typing_extensions import TypedDict

from polylogue.storage.embeddings.models import EmbeddingStatsSnapshot
from polylogue.storage.embeddings.progress import EmbeddingCatchupRunPayload
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

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


class EmbeddingNextActionPayload(TypedDict):
    code: str
    command: str | None
    reason: str


class EmbeddingStatusPayload(TypedDict):
    config_enabled: bool
    has_voyage_api_key: bool
    daemon_stage_enabled: bool
    configured_model: str
    configured_dimension: int
    monthly_cost_cap_usd: float
    status: str
    total_conversations: int
    embedded_conversations: int
    embedded_messages: int
    pending_conversations: int
    pending_messages: int | None
    pending_messages_exact: bool
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
    failure_count: int
    total_estimated_cost_usd: float
    latest_catchup_run: EmbeddingCatchupRunPayload | None
    next_action: EmbeddingNextActionPayload


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
    from polylogue.storage.embeddings.support import optional_count_sync

    return optional_count_sync(conn, "SELECT COUNT(*) FROM conversations")


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


def _next_action(
    *,
    config_enabled: bool,
    has_voyage_api_key: bool,
    total_conversations: int,
    pending_conversations: int,
    retrieval_ready: bool,
    stale_messages: int,
    failure_count: int,
) -> EmbeddingNextActionPayload:
    if total_conversations <= 0:
        return {
            "code": "archive_empty",
            "command": None,
            "reason": "Archive contains no conversations to embed.",
        }
    if not has_voyage_api_key:
        return {
            "code": "set_voyage_key",
            "command": "polylogue embed enable --voyage-api-key ...",
            "reason": "Semantic retrieval needs a Voyage API key before embedding can run.",
        }
    if not config_enabled:
        return {
            "code": "enable_embeddings",
            "command": "polylogue embed enable --yes",
            "reason": "A Voyage key is available, but embedding convergence is disabled in config.",
        }
    if failure_count > 0 and pending_conversations > 0:
        return {
            "code": "inspect_failures",
            "command": "polylogue embed status --detail",
            "reason": "Recent embedding failures exist while backlog remains.",
        }
    if stale_messages > 0:
        return {
            "code": "refresh_stale",
            "command": "polylogue embed backfill --max-conversations 10",
            "reason": "Existing vectors are stale for at least one message.",
        }
    if pending_conversations > 0:
        return {
            "code": "drain_backlog",
            "command": "polylogue embed backfill --max-conversations 10",
            "reason": "Embedding convergence is enabled and pending conversations remain.",
        }
    if retrieval_ready:
        return {
            "code": "ready",
            "command": "polylogue --semantic <query>",
            "reason": "Embeddings are retrieval-ready.",
        }
    return {
        "code": "run_preflight",
        "command": "polylogue embed preflight --detail",
        "reason": "Embedding state is inconclusive; inspect exact pending-message and retrieval-band details.",
    }


def _payload_from_stats(
    *,
    config_enabled: bool,
    has_voyage_api_key: bool,
    configured_model: str,
    configured_dimension: int,
    monthly_cost_cap_usd: float,
    total_conversations: int,
    stats: EmbeddingStatsSnapshot,
    latest_catchup_run: EmbeddingCatchupRunPayload | None,
    pending_messages_exact: bool,
) -> EmbeddingStatusPayload:
    embedded_conversations = stats.embedded_conversations
    pending_conversations = stats.pending_conversations or max(total_conversations - embedded_conversations, 0)
    status = _embedding_status(
        total_conversations=total_conversations,
        embedded_conversations=embedded_conversations,
        pending_conversations=pending_conversations,
    )
    retrieval_ready = _retrieval_ready(stats)
    return {
        "config_enabled": config_enabled,
        "has_voyage_api_key": has_voyage_api_key,
        "daemon_stage_enabled": config_enabled and has_voyage_api_key,
        "configured_model": configured_model,
        "configured_dimension": configured_dimension,
        "monthly_cost_cap_usd": monthly_cost_cap_usd,
        "status": status,
        "total_conversations": total_conversations,
        "embedded_conversations": embedded_conversations,
        "embedded_messages": stats.embedded_messages,
        "pending_conversations": pending_conversations,
        "pending_messages": stats.pending_messages if pending_messages_exact else None,
        "pending_messages_exact": pending_messages_exact,
        "embedding_coverage_percent": round(
            _coverage_percent(
                embedded_conversations=embedded_conversations,
                total_conversations=total_conversations,
            ),
            1,
        ),
        "retrieval_ready": retrieval_ready,
        "freshness_status": _freshness_status(status, stats),
        "stale_messages": stats.stale_messages,
        "messages_missing_provenance": stats.messages_missing_provenance,
        "oldest_embedded_at": stats.oldest_embedded_at,
        "newest_embedded_at": stats.newest_embedded_at,
        "embedding_models": stats.model_counts,
        "embedding_dimensions": stats.dimension_counts,
        "retrieval_bands": stats.retrieval_bands,
        "failure_count": stats.failure_count,
        "total_estimated_cost_usd": stats.total_estimated_cost_usd,
        "latest_catchup_run": latest_catchup_run,
        "next_action": _next_action(
            config_enabled=config_enabled,
            has_voyage_api_key=has_voyage_api_key,
            total_conversations=total_conversations,
            pending_conversations=pending_conversations,
            retrieval_ready=retrieval_ready,
            stale_messages=stats.stale_messages,
            failure_count=stats.failure_count,
        ),
    }


def embedding_status_payload(
    env: _HasConfig,
    *,
    include_retrieval_bands: bool = False,
    include_detail: bool = False,
) -> EmbeddingStatusPayload:
    """Read canonical embedding-status statistics for operator surfaces."""
    from polylogue.config import load_polylogue_config
    from polylogue.storage.embeddings.embedding_stats import read_embedding_stats_sync
    from polylogue.storage.embeddings.progress import latest_embedding_catchup_run
    from polylogue.storage.embeddings.support import table_exists_sync

    cfg = load_polylogue_config()
    db_path = Path(env.config.db_path)
    if not db_path.exists():
        return _payload_from_stats(
            config_enabled=bool(cfg.embedding_enabled),
            has_voyage_api_key=bool(cfg.voyage_api_key),
            configured_model=cfg.embedding_model,
            configured_dimension=cfg.embedding_dimension,
            monthly_cost_cap_usd=cfg.embedding_max_cost_usd,
            total_conversations=0,
            stats=EmbeddingStatsSnapshot(),
            latest_catchup_run=None,
            pending_messages_exact=include_detail,
        )

    conn = open_readonly_connection(db_path)
    try:
        total_conversations = _total_conversations(conn)
        embedding_stats = read_embedding_stats_sync(
            conn,
            include_retrieval_bands=include_retrieval_bands,
            detail=include_detail,
        )
        latest_run = latest_embedding_catchup_run(conn) if table_exists_sync(conn, "embedding_catchup_runs") else None
    finally:
        conn.close()

    return _payload_from_stats(
        config_enabled=bool(cfg.embedding_enabled),
        has_voyage_api_key=bool(cfg.voyage_api_key),
        configured_model=cfg.embedding_model,
        configured_dimension=cfg.embedding_dimension,
        monthly_cost_cap_usd=cfg.embedding_max_cost_usd,
        total_conversations=total_conversations,
        stats=embedding_stats,
        latest_catchup_run=latest_run,
        pending_messages_exact=include_detail,
    )
