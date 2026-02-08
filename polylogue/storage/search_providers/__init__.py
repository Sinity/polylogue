"""Search provider implementations and factory functions.

This package provides concrete implementations of the SearchProvider and
VectorProvider protocols defined in polylogue.protocols. It also includes
factory functions for creating provider instances from configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.lib.env import get_env
from polylogue.storage.search_providers.fts5 import FTS5Provider
from polylogue.storage.search_providers.hybrid import (
    HybridSearchProvider,
    create_hybrid_provider,
    reciprocal_rank_fusion,
)

if TYPE_CHECKING:
    from polylogue.config import Config
    from polylogue.protocols import SearchProvider, VectorProvider

logger = logging.getLogger(__name__)


def create_search_provider(
    config: Config | None = None,
    db_path: Path | None = None
) -> SearchProvider:
    """Create a search provider instance.

    Currently returns FTS5Provider as the default implementation.

    Args:
        config: Application configuration (unused currently)
        db_path: Optional database path override

    Returns:
        SearchProvider instance (currently always FTS5Provider)
    """
    return FTS5Provider(db_path=db_path)


def create_vector_provider(
    config: Config | None = None,
    *,
    voyage_api_key: str | None = None,
    db_path: Path | None = None,
) -> VectorProvider | None:
    """Create a vector provider instance if configured.

    Uses sqlite-vec for self-contained vector search with Voyage AI embeddings.
    Returns None if Voyage API key is not configured or sqlite-vec is unavailable.

    Environment variable precedence (checked in order):
    - POLYLOGUE_VOYAGE_API_KEY > VOYAGE_API_KEY

    Args:
        config: Application configuration with optional index_config
        voyage_api_key: Voyage AI API key (overrides config and env var)
        db_path: Optional database path override

    Returns:
        SqliteVecProvider if configured and available, None otherwise
    """
    # Resolve Voyage key with priority: explicit arg > config > env
    voyage_key = voyage_api_key
    if voyage_key is None and config and config.index_config:
        voyage_key = config.index_config.voyage_api_key
    if voyage_key is None:
        voyage_key = get_env("VOYAGE_API_KEY")

    if not voyage_key:
        return None

    # Check if sqlite-vec is available
    try:
        import sqlite_vec  # noqa: F401
    except ImportError:
        logger.warning("sqlite-vec not installed, vector search unavailable")
        return None

    # Import here to avoid circular imports and loading when not needed
    from polylogue.storage.search_providers.sqlite_vec import (
        SqliteVecError,
        SqliteVecProvider,
    )

    try:
        return SqliteVecProvider(
            voyage_key=voyage_key,
            db_path=db_path,
        )
    except SqliteVecError as exc:
        logger.warning("sqlite-vec initialization failed: %s", exc)
        return None


__all__ = [
    "FTS5Provider",
    "HybridSearchProvider",
    "reciprocal_rank_fusion",
    "create_search_provider",
    "create_vector_provider",
    "create_hybrid_provider",
]
