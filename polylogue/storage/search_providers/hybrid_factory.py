"""Factory helpers for hybrid search providers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylogue.protocols import VectorProvider
    from polylogue.storage.search_providers.hybrid import HybridSearchProvider


def create_hybrid_provider(
    db_path: Path | None = None,
    vector_provider: VectorProvider | None = None,
    rrf_k: int = 60,
) -> HybridSearchProvider | None:
    """Create a hybrid search provider if vector search is available."""
    from polylogue.storage.search_providers import create_vector_provider
    from polylogue.storage.search_providers.fts5 import FTS5Provider
    from polylogue.storage.search_providers.hybrid import HybridSearchProvider

    vec_provider = vector_provider or create_vector_provider(db_path=db_path)
    if vec_provider is None:
        return None

    return HybridSearchProvider(
        fts_provider=FTS5Provider(db_path=db_path),
        vector_provider=vec_provider,
        rrf_k=rrf_k,
    )


__all__ = ["create_hybrid_provider"]
