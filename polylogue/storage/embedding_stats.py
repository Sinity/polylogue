"""Shared helpers for optional embedding-related archive statistics."""

from __future__ import annotations

from polylogue.storage.embedding_stats_models import EmbeddingStatsSnapshot
from polylogue.storage.embedding_stats_runtime import (
    read_embedding_stats_async,
    read_embedding_stats_sync,
)

__all__ = [
    "EmbeddingStatsSnapshot",
    "read_embedding_stats_async",
    "read_embedding_stats_sync",
]
