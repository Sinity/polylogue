"""Real :class:`PolylogueConfig` values for embedding-status tests.

``polylogue.storage.embeddings.status_payload.embedding_status_settings_from_config``
dispatches *nominally* over ``PolylogueConfig``/``Config`` and raises
``TypeError`` for anything else.  That is deliberate: the two config types
observe different facts (a legacy ``Config`` has no notion of
``embedding_enabled`` or ``embedding_max_cost_usd``, and reaches its Voyage key
through ``index_config``), so ``None`` in ``EmbeddingStatusSettings`` means
"this configuration type does not observe that fact".  Structural matching
cannot carry that distinction, which is why a duck-typed local double is not an
acceptable stand-in here: it could only ever prove that the double matches
itself, and would go on passing after the production type it imitates changed.

Tests that need a configuration for an embedding-status read build a real one
through this helper instead of re-declaring a private double per file.
"""

from __future__ import annotations

from polylogue.config import PolylogueConfig

__all__ = ["embedding_config"]


def embedding_config(
    *,
    embedding_enabled: bool = True,
    voyage_api_key: str | None = "test-key",
    embedding_model: str = "voyage-4",
    embedding_dimension: int = 1024,
    embedding_max_cost_usd: float = 0.0,
    **extra: object,
) -> PolylogueConfig:
    """Return a real ``PolylogueConfig`` carrying the embedding settings given."""

    return PolylogueConfig(
        {
            "embedding_enabled": embedding_enabled,
            "voyage_api_key": voyage_api_key,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
            "embedding_max_cost_usd": embedding_max_cost_usd,
            **extra,
        }
    )
