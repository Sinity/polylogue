"""Search provider implementations and factory functions.

This package provides concrete implementations of the SearchProvider and
VectorProvider protocols defined in polylogue.protocols. It also includes
factory functions for creating provider instances from configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.storage.search_providers.fts5 import FTS5Provider
from polylogue.storage.search_providers.qdrant import QdrantError, QdrantProvider

if TYPE_CHECKING:
    from polylogue.config import Config
    from polylogue.protocols import SearchProvider, VectorProvider


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
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    voyage_api_key: str | None = None,
) -> VectorProvider | None:
    """Create a vector provider instance if configured.

    Checks for Qdrant configuration in config, arguments, or environment variables.
    Priority: explicit arguments > config.index_config > environment variables.
    Returns None if Qdrant is not configured.

    Args:
        config: Application configuration with optional index_config
        qdrant_url: Qdrant server URL (overrides config and env var)
        qdrant_api_key: Qdrant API key (overrides config and env var)
        voyage_api_key: Voyage AI API key (overrides config and env var)

    Returns:
        QdrantProvider if configured, None otherwise

    Raises:
        ValueError: If Qdrant URL is provided but Voyage API key is missing
    """
    # Resolve Qdrant URL with priority: explicit arg > config > env
    url = qdrant_url
    if url is None and config and config.index_config:
        url = config.index_config.qdrant_url
    if url is None:
        url = os.environ.get("QDRANT_URL")
    if not url:
        return None

    # Resolve API keys with priority: explicit arg > config > env
    api_key = qdrant_api_key
    if api_key is None and config and config.index_config:
        api_key = config.index_config.qdrant_api_key
    if api_key is None:
        api_key = os.environ.get("QDRANT_API_KEY")

    voyage_key = voyage_api_key
    if voyage_key is None and config and config.index_config:
        voyage_key = config.index_config.voyage_api_key
    if voyage_key is None:
        voyage_key = os.environ.get("VOYAGE_API_KEY")

    if not voyage_key:
        raise ValueError(
            "Qdrant is configured but VOYAGE_API_KEY is not set. "
            "Vector search requires Voyage AI for embeddings."
        )

    return QdrantProvider(
        qdrant_url=url,
        api_key=api_key,
        voyage_key=voyage_key,
    )


__all__ = [
    "FTS5Provider",
    "QdrantProvider",
    "QdrantError",
    "create_search_provider",
    "create_vector_provider",
]
