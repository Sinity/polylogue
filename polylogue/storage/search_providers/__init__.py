"""Search provider implementations and factory functions.

This package provides the concrete ``VectorProvider`` implementation
(``SqliteVecProvider``) and its factory function, plus the shared
Reciprocal Rank Fusion primitive that production hybrid retrieval composes
directly. It no longer provides a ``SearchProvider`` implementation or
factory — see :mod:`polylogue.storage.search_providers.hybrid` for why
(polylogue-a7xr.10).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.logging import get_logger
from polylogue.storage.search_providers.hybrid import reciprocal_rank_fusion

if TYPE_CHECKING:
    from polylogue.config import Config
    from polylogue.core.protocols import VectorProvider

logger = get_logger(__name__)
_sqlite_vec_missing_warned = False


def _sqlite_vec_available() -> bool:
    return importlib.util.find_spec("sqlite_vec") is not None


def create_vector_provider(
    config: Config | None = None,
    *,
    voyage_api_key: str | None = None,
    db_path: Path | None = None,
    model: str | None = None,
    dimension: int | None = None,
) -> VectorProvider | None:
    """Create a vector provider instance if configured.

    Uses sqlite-vec for self-contained vector search with Voyage AI embeddings.
    Returns None if Voyage API key is not configured or sqlite-vec is unavailable.

    Args:
        config: Application configuration with optional index_config
        voyage_api_key: Voyage AI API key (overrides config and env var)
        db_path: Optional database path override
        model: Embedding model name (defaults to voyage-4 if None)
        dimension: Embedding dimension (defaults to 1024 if None)

    Returns:
        SqliteVecProvider if configured and available, None otherwise
    """
    global _sqlite_vec_missing_warned

    # Resolve Voyage key with priority: explicit arg > config > env
    voyage_key = voyage_api_key
    if voyage_key is None and config and config.index_config:
        voyage_key = config.index_config.voyage_api_key
    if voyage_key is None:
        voyage_key = os.environ.get("VOYAGE_API_KEY")

    if not voyage_key:
        return None

    if not _sqlite_vec_available():
        if not _sqlite_vec_missing_warned:
            logger.warning("sqlite-vec not installed, vector search unavailable")
            _sqlite_vec_missing_warned = True
        return None

    # Import here to avoid circular imports and loading when not needed
    from polylogue.storage.search_providers.sqlite_vec import (
        SqliteVecError,
        SqliteVecProvider,
    )

    kwargs: dict[str, object] = {"voyage_key": voyage_key, "db_path": db_path}
    if model is not None:
        kwargs["model"] = model
    if dimension is not None:
        kwargs["dimension"] = dimension

    try:
        return SqliteVecProvider(**kwargs)  # type: ignore[arg-type]
    except SqliteVecError as exc:
        logger.warning("sqlite-vec initialization failed: %s", exc)
        return None


__all__ = [
    "reciprocal_rank_fusion",
    "create_vector_provider",
]
