"""Search provider implementations and factory functions.

The package provides the concrete ``VectorProvider`` implementation
(``SqliteVecProvider``) and its factory. Canonical archive reads own lexical
retrieval and complete SQL lane-rank fusion.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.logging import get_logger

if TYPE_CHECKING:
    from polylogue.config import Config
    from polylogue.core.protocols import VectorProvider
    from polylogue.storage.embeddings.identity import EmbeddingRecipe

logger = get_logger(__name__)
_sqlite_vec_missing_warned = False


def _sqlite_vec_available() -> bool:
    return importlib.util.find_spec("sqlite_vec") is not None


def create_vector_provider(
    config: Config | None = None,
    *,
    voyage_api_key: str | None = None,
    db_path: Path | None = None,
    archive_root: Path | None = None,
    model: str | None = None,
    dimension: int | None = None,
    require_credentials: bool = True,
    query_recipe: EmbeddingRecipe | None = None,
) -> VectorProvider | None:
    """Create a vector provider instance if configured.

    Uses sqlite-vec for self-contained vector search with Voyage AI embeddings.
    Returns None if acquisition credentials are required but absent, or sqlite-vec is unavailable.
    Stored-vector consumers set ``require_credentials=False``; they never embed text.

    Args:
        config: Application configuration with optional index_config
        voyage_api_key: Voyage AI API key (overrides config and env var)
        db_path: Optional database path override
        model: Embedding model name (defaults to the configured model)
        dimension: Embedding dimension (defaults to the configured dimension)

    Returns:
        SqliteVecProvider if configured and available, None otherwise
    """
    global _sqlite_vec_missing_warned

    # Resolve Voyage key with priority: explicit arg > config > env
    voyage_key = voyage_api_key
    if voyage_key is None and config is not None and config.index_config is not None:
        voyage_key = config.index_config.voyage_api_key
    if voyage_key is None:
        from polylogue.config import load_polylogue_config

        voyage_key = load_polylogue_config().voyage_api_key

    if require_credentials and not voyage_key:
        return None

    if config is not None:
        if model is None:
            model = config.embedding_model
        if dimension is None:
            dimension = config.embedding_dimension
    elif model is None or dimension is None:
        # The ambient recipe is part of the archive contract, not a nicety the
        # caller may omit. ``resolve_optional_vector_provider`` (the repository
        # route behind ``SessionRepository.similarity_search``)
        # calls this factory with ``config=None``; loading only the ambient key
        # and skipping the recipe meant a declared ``model``/``dimension`` in
        # polylogue.toml was silently replaced by the library defaults, so the
        # repository wrote or queried vectors under a recipe the archive never
        # configured. Load the recipe from the same layered config the key
        # already comes from.
        from polylogue.config import load_polylogue_config

        settings = load_polylogue_config()
        if model is None:
            model = settings.embedding_model
        if dimension is None:
            dimension = settings.embedding_dimension

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

    if db_path is not None and db_path.name != "embeddings.db":
        raise SqliteVecError("public vector provider requires managed embeddings.db")
    if config is not None and archive_root is None:
        archive_root = config.archive_root
    if archive_root is None:
        from polylogue.paths import archive_root as configured_archive_root

        archive_root = configured_archive_root()
    if db_path is None and config is not None:
        db_path = archive_root / "embeddings.db"
    if db_path is not None:
        try:
            db_path.absolute().resolve(strict=False).relative_to(archive_root.absolute().resolve(strict=False))
        except ValueError as exc:
            raise SqliteVecError("public vector provider database is outside its trusted archive root") from exc

    kwargs: dict[str, object] = {
        "voyage_key": voyage_key,
        "db_path": db_path,
        "archive_root": archive_root,
        "query_recipe": query_recipe,
    }
    if model is not None:
        kwargs["model"] = model
    if dimension is not None:
        kwargs["dimension"] = dimension

    if query_recipe is not None:
        from polylogue.storage.embeddings.identity import EmbeddingRecipe

        document = EmbeddingRecipe.current(model=model, dimensions=dimension)
        if query_recipe.input_type != "query" or not document.retrieval_compatible(query_recipe):
            raise SqliteVecError("query and document recipes do not declare compatible retrieval contracts")

    try:
        return SqliteVecProvider(**kwargs)  # type: ignore[arg-type]
    except SqliteVecError as exc:
        logger.warning("sqlite-vec initialization failed: %s", exc)
        return None


__all__ = [
    "create_vector_provider",
]
