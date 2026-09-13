"""Operations seam for frames bound to the active embedding input generation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from polylogue.daemon.derivation import DerivationFrame
from polylogue.storage.archive_identity import resolve_active_index_path
from polylogue.storage.embeddings.derivation import EmbeddingDerivationAdapter
from polylogue.storage.source_sessions import session_ids_for_source_paths
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

__all__ = ["embedding_session_ids_for_paths", "make_embedding_frame"]


def make_embedding_frame(
    index_db_path: Path,
    *,
    archive_root: Path,
    adapter: EmbeddingDerivationAdapter,
    scope: Sequence[str] | None,
) -> DerivationFrame:
    """Bind one pass to an active index generation and complete recipe identity."""

    del index_db_path
    index_path = resolve_active_index_path(archive_root).resolve()
    return DerivationFrame(
        archive_root=str(archive_root),
        source_revision=f"index-generation:{index_path}",
        recipe_versions={adapter.domain: adapter.recipe_version},
        scope=None if scope is None else tuple(dict.fromkeys(str(session_id) for session_id in scope)),
    )


def embedding_session_ids_for_paths(
    index_db_path: Path,
    *,
    archive_root: Path,
    paths: Sequence[Path],
) -> tuple[str, ...]:
    """Resolve watcher paths against the archive's durable source tier.

    Active index generations can live below ``.index-generations`` while
    ``source.db`` remains at the archive root.  The shared source-session
    relation accepts that explicit tier path so path hints never silently
    become an empty embedding scope after an index-generation switch.
    """

    normalized = tuple(dict.fromkeys(Path(path) for path in paths))
    if not normalized:
        return ()
    with open_readonly_connection(index_db_path, timeout_class="background-read", validate_schema=False) as conn:
        by_path = session_ids_for_source_paths(conn, normalized, source_db=archive_root / "source.db")
    return tuple(dict.fromkeys(session_id for path in normalized for session_id in by_path.get(path, ())))
