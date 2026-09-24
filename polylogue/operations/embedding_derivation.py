"""Operations seam for frames bound to the active embedding input generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from polylogue.daemon.derivation import DerivationFrame
from polylogue.storage.archive_identity import resolve_active_index_path
from polylogue.storage.embeddings.derivation import EmbeddingDerivationAdapter, EmbeddingTextProvider
from polylogue.storage.embeddings.materialization import EmbeddingWriteAdmission
from polylogue.storage.source_sessions import session_ids_for_source_paths
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

__all__ = [
    "embedding_session_ids_for_paths",
    "select_embedding_session_window",
    "estimated_embedding_message_cost",
    "make_embedding_derivation",
    "make_embedding_frame",
    "mark_embedding_sessions_needs_reindex",
]

EmbeddingProgressCallback = Callable[[Mapping[str, object]], None]


def mark_embedding_sessions_needs_reindex(index_db_path: Path, *, embeddings_db_path: Path) -> None:
    """Mark rebuild work through the operations seam owned by the daemon."""
    from polylogue.storage.embeddings.materialization import mark_all_archive_sessions_needs_reindex

    mark_all_archive_sessions_needs_reindex(index_db_path, embeddings_db_path=embeddings_db_path)


def select_embedding_session_window(
    index_db_path: Path,
    *,
    archive_root: Path,
    rebuild: bool = False,
    max_sessions: int | None = None,
    max_messages: int | None = None,
    min_messages: int | None = None,
) -> tuple[str, ...]:
    """Resolve one bounded pending-session window for a daemon operation."""
    from polylogue.storage.embeddings.materialization import select_pending_session_window

    del archive_root
    with open_readonly_connection(index_db_path, timeout_class="background-read", validate_schema=False) as conn:
        rows = select_pending_session_window(
            conn,
            rebuild=rebuild,
            max_sessions=max_sessions,
            max_messages=max_messages,
        )
    if min_messages is not None:
        rows = [row for row in rows if row.message_count >= min_messages]
    return tuple(row.session_id for row in rows)


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


def estimated_embedding_message_cost() -> float:
    """Conservative provider cost used by daemon admission before computation."""
    from polylogue.storage.search_providers.sqlite_vec_support import (
        ESTIMATED_TOKENS_PER_MESSAGE,
        VOYAGE_4_COST_PER_1M_TOKENS,
    )

    return ESTIMATED_TOKENS_PER_MESSAGE * VOYAGE_4_COST_PER_1M_TOKENS / 1_000_000


def make_embedding_derivation(
    index_db_path: Path,
    *,
    archive_root: Path,
    voyage_api_key: str,
    model: str,
    dimension: int,
    reserve: EmbeddingWriteAdmission,
    quiet: Callable[[], bool] | None = None,
    progress_callback: EmbeddingProgressCallback | None = None,
) -> EmbeddingDerivationAdapter | None:
    """Bind the configured provider and archive to the storage-owned adapter."""
    from polylogue.storage.search_providers import create_vector_provider

    provider = create_vector_provider(
        voyage_api_key=voyage_api_key,
        db_path=archive_root / "embeddings.db",
        archive_root=archive_root,
        model=model,
        dimension=dimension,
    )
    if not isinstance(provider, EmbeddingTextProvider):
        return None

    # The common derivation kernel asks the adapter's quiet policy before each
    # key.  Use that observation point to publish an intermediate, bounded
    # progress frame while the pass is still running.  This deliberately
    # reports intent (before provider work) rather than claiming publication;
    # the terminal report remains the authority for committed output.
    def observe_progress(frame: object, key: str) -> bool:
        del frame
        is_quiet = bool(quiet and quiet())
        if is_quiet:
            return True
        if progress_callback is not None:
            message_id = key.removeprefix("message:").removeprefix("orphan:")
            session_id: str | None = None
            try:
                index_path = resolve_active_index_path(archive_root).resolve()
                with open_readonly_connection(
                    index_path,
                    timeout_class="background-read",
                    validate_schema=False,
                ) as conn:
                    row = conn.execute(
                        "SELECT session_id FROM messages WHERE message_id = ?",
                        (message_id,),
                    ).fetchone()
                session_id = None if row is None else str(row[0])
            except Exception:
                # Progress is observational; inability to resolve attribution
                # must never fail or alter the derivation itself.
                session_id = None
            progress_callback(
                {
                    "state": "started",
                    "message_id": message_id,
                    "session_id": session_id,
                }
            )
        return False

    return EmbeddingDerivationAdapter(
        index_db_path,
        provider,
        archive_root=archive_root,
        reserve=reserve,
        quiet=observe_progress if (quiet is not None or progress_callback is not None) else None,
    )
