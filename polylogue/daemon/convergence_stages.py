"""Convergence stage implementations for the daemon pipeline.

Each stage has a ``check`` that inspects current archive state and an
``execute`` that performs the missing work. The live watcher owns source
ingestion through daemon-side raw-record ingest; daemon convergence stages only
repair and refresh post-ingest archive state.

- fts: rebuild FTS if messages > indexed count
- embed: optional vectorization for changed conversations
- insights: refresh session profiles
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from polylogue.daemon.convergence import ConvergenceStage
from polylogue.logging import get_logger

logger = get_logger(__name__)

_DAEMON_INSIGHT_REBUILD_PAGE_SIZE = 10


@dataclass(frozen=True, slots=True)
class _FtsRepairNeeds:
    messages: bool = False
    actions: bool = False

    @property
    def any(self) -> bool:
        return self.messages or self.actions


# ── Stage: FTS ─────────────────────────────────────────────────────


def make_fts_stage(db_path: Path) -> ConvergenceStage:
    """Verify FTS coverage and repair gaps."""

    def check(path: Path) -> bool:  # noqa: ARG001
        if not db_path.exists():
            return False
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=5.0)
            try:
                conversation_ids = _conversation_ids_for_source_path(conn, path)
                if conversation_ids:
                    return _fts_needs_repair_for_conversations(conn, conversation_ids)
                total = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
                if total == 0:
                    return False
                fts_count = int(conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0])
                if fts_count < total:
                    return True
                if _table_exists(conn, "action_events") and _table_exists(conn, "action_events_fts"):
                    action_total = int(conn.execute("SELECT COUNT(*) FROM action_events").fetchone()[0])
                    action_fts_count = int(conn.execute("SELECT COUNT(*) FROM action_events_fts").fetchone()[0])
                    return action_fts_count < action_total
                return False
            finally:
                conn.close()
        except Exception:
            return False

    def execute(path: Path) -> bool:
        from polylogue.storage.fts.fts_lifecycle import rebuild_fts_index_sync
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=30.0)
            try:
                conversation_ids = _conversation_ids_for_source_path(conn, path)
                if conversation_ids:
                    needs = _fts_repair_needs_for_conversations(conn, conversation_ids)
                    _repair_changed_conversation_fts(conn, conversation_ids, needs=needs)
                    conn.commit()
                    logger.info("fts: repaired conversations=%d", len(conversation_ids))
                    return not _fts_needs_repair_for_conversations(conn, conversation_ids)
                total = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
                rebuild_fts_index_sync(conn)
                conn.commit()
                new_count = int(conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0])
                logger.info("fts: rebuilt — %d/%d indexed", new_count, total)
                return new_count >= total
            finally:
                conn.close()
        except Exception:
            logger.warning("fts: rebuild failed", exc_info=True)
            return False

    def check_many(paths: Sequence[Path]) -> set[Path]:
        if not paths:
            return set()
        if not db_path.exists():
            return set()
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=5.0)
            try:
                by_path = _conversation_ids_for_source_paths(conn, paths)
                return {
                    path
                    for path, conversation_ids in by_path.items()
                    if conversation_ids and _fts_repair_needs_for_conversations(conn, conversation_ids).any
                }
            finally:
                conn.close()
        except Exception:
            return set()

    def execute_many(paths: Sequence[Path]) -> bool:
        if not paths:
            return False
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=30.0)
            try:
                by_path = _conversation_ids_for_source_paths(conn, paths)
                conversation_ids = list(
                    dict.fromkeys(conversation_id for ids in by_path.values() for conversation_id in ids)
                )
                if not conversation_ids:
                    return execute(Path(paths[0]))
                needs = _fts_repair_needs_for_conversations(conn, conversation_ids)
                _repair_changed_conversation_fts(conn, conversation_ids, needs=needs)
                conn.commit()
                logger.info("fts: batch repaired paths=%d conversations=%d", len(paths), len(conversation_ids))
                return not _fts_needs_repair_for_conversations(conn, conversation_ids)
            finally:
                conn.close()
        except Exception:
            logger.warning("fts: batch repair failed", exc_info=True)
            return False

    return ConvergenceStage(
        name="fts",
        description="Verify FTS coverage and repair gaps",
        check=check,
        execute=execute,
        check_many=check_many,
        execute_many=execute_many,
        cpu_bound=False,
    )


# ── Stage: embed ───────────────────────────────────────────────────


def make_embed_stage(db_path: Path) -> ConvergenceStage:
    """Generate vector embeddings for changed conversations that need them."""

    def check(path: Path) -> bool:
        if not _embedding_env_enabled():
            return False
        if not db_path.exists():
            return False
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=5.0)
            try:
                conversation_ids = _conversation_ids_for_source_path(conn, path)
                return bool(_pending_embedding_conversation_ids(conn, conversation_ids))
            finally:
                conn.close()
        except Exception:
            return False

    def execute(path: Path) -> bool:
        if not _embedding_env_enabled():
            return True
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=5.0)
            try:
                conversation_ids = _pending_embedding_conversation_ids(
                    conn,
                    _conversation_ids_for_source_path(conn, path),
                )
            finally:
                conn.close()
            if not conversation_ids:
                return True
            return _embed_conversations_sync(db_path, conversation_ids)
        except Exception:
            logger.warning("embed: failed", exc_info=True)
            return False

    def check_many(paths: Sequence[Path]) -> set[Path]:
        if not paths or not _embedding_env_enabled() or not db_path.exists():
            return set()
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=5.0)
            try:
                by_path = _conversation_ids_for_source_paths(conn, paths)
                all_conversation_ids = list(
                    dict.fromkeys(conversation_id for ids in by_path.values() for conversation_id in ids)
                )
                pending_ids = set(_pending_embedding_conversation_ids(conn, all_conversation_ids))
                if not pending_ids:
                    return set()
                return {
                    path
                    for path, conversation_ids in by_path.items()
                    if any(conversation_id in pending_ids for conversation_id in conversation_ids)
                }
            finally:
                conn.close()
        except Exception:
            return set()

    def execute_many(paths: Sequence[Path]) -> bool:
        if not paths or not _embedding_env_enabled():
            return True
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=5.0)
            try:
                by_path = _conversation_ids_for_source_paths(conn, paths)
                conversation_ids = list(
                    dict.fromkeys(conversation_id for ids in by_path.values() for conversation_id in ids)
                )
                pending_ids = _pending_embedding_conversation_ids(conn, conversation_ids)
            finally:
                conn.close()
            if not pending_ids:
                return True
            return _embed_conversations_sync(db_path, pending_ids)
        except Exception:
            logger.warning("embed: batch failed", exc_info=True)
            return False

    return ConvergenceStage(
        name="embed",
        description="Generate vector embeddings for changed conversations",
        check=check,
        execute=execute,
        check_many=check_many,
        execute_many=execute_many,
        cpu_bound=False,
    )


# ── Stage: insights ────────────────────────────────────────────────


def make_insights_stage(db_path: Path) -> ConvergenceStage:
    """Refresh session insights for conversations missing them."""

    def check(path: Path) -> bool:  # noqa: ARG001
        if not db_path.exists():
            return False
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=5.0)
            try:
                if not _table_exists(conn, "session_profiles"):
                    return False
                if _conversation_ids_for_source_path(conn, path):
                    return True
                total_conv = int(conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0])
                if total_conv == 0:
                    return False
                profiled = int(conn.execute("SELECT COUNT(*) FROM session_profiles").fetchone()[0])
                return profiled < total_conv
            finally:
                conn.close()
        except Exception:
            return False

    def execute(path: Path) -> bool:  # noqa: ARG001
        from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync
        from polylogue.storage.sqlite.connection import open_connection

        try:
            with open_connection(db_path) as conn:
                conversation_ids = _conversation_ids_for_source_path(conn, path) or _conversation_ids_missing_profiles(
                    conn
                )
                counts = rebuild_session_insights_sync(
                    conn,
                    conversation_ids=conversation_ids,
                    page_size=_DAEMON_INSIGHT_REBUILD_PAGE_SIZE,
                )
                conn.commit()
                logger.info(
                    "insights: refreshed conversations=%d profiles=%d work_events=%d phases=%d threads=%d",
                    len(conversation_ids),
                    counts.profiles,
                    counts.work_events,
                    counts.phases,
                    counts.threads,
                )
            return True
        except Exception:
            logger.warning("insights: rebuild failed", exc_info=True)
            return False

    def check_many(paths: Sequence[Path]) -> set[Path]:
        if not db_path.exists() or not paths:
            return set()
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=5.0)
            try:
                if not _table_exists(conn, "session_profiles"):
                    return set()
                by_path = _conversation_ids_for_source_paths(conn, paths)
                paths_with_conversations = {path for path, conversation_ids in by_path.items() if conversation_ids}
                if paths_with_conversations:
                    return paths_with_conversations
                if _conversation_ids_missing_profiles(conn):
                    return {Path(paths[0])}
                return set()
            finally:
                conn.close()
        except Exception:
            return set()

    def execute_many(paths: Sequence[Path]) -> bool:
        from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync
        from polylogue.storage.sqlite.connection import open_connection

        try:
            with open_connection(db_path) as conn:
                by_path = _conversation_ids_for_source_paths(conn, paths)
                conversation_ids = list(
                    dict.fromkeys(conversation_id for ids in by_path.values() for conversation_id in ids)
                )
                if not conversation_ids:
                    conversation_ids = _conversation_ids_missing_profiles(conn)
                counts = rebuild_session_insights_sync(
                    conn,
                    conversation_ids=conversation_ids,
                    page_size=_DAEMON_INSIGHT_REBUILD_PAGE_SIZE,
                )
                conn.commit()
                logger.info(
                    "insights: batch refreshed paths=%d conversations=%d profiles=%d work_events=%d phases=%d threads=%d",
                    len(paths),
                    len(conversation_ids),
                    counts.profiles,
                    counts.work_events,
                    counts.phases,
                    counts.threads,
                )
            return True
        except Exception:
            logger.warning("insights: batch rebuild failed", exc_info=True)
            return False

    return ConvergenceStage(
        name="insights",
        description="Refresh session insights for new conversations",
        check=check,
        execute=execute,
        check_many=check_many,
        execute_many=execute_many,
        cpu_bound=False,
    )


def make_default_convergence_stages(db_path: Path) -> tuple[ConvergenceStage, ...]:
    """Build the daemon's default post-ingest convergence stage set."""
    stage_list = [make_fts_stage(db_path)]
    if _embedding_env_enabled():
        stage_list.append(make_embed_stage(db_path))
    stage_list.append(make_insights_stage(db_path))
    return tuple(stage_list)


# ── Helpers ────────────────────────────────────────────────────────


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _conversation_ids_for_source_path(conn: sqlite3.Connection, path: Path) -> list[str]:
    return _conversation_ids_for_source_paths(conn, [path]).get(path, [])


def _conversation_ids_for_source_paths(
    conn: sqlite3.Connection,
    paths: Sequence[Path],
) -> dict[Path, list[str]]:
    normalized_paths = tuple(dict.fromkeys(Path(path) for path in paths))
    if not normalized_paths or not _table_exists(conn, "raw_conversations") or not _table_exists(conn, "conversations"):
        return {}
    result: dict[Path, list[str]] = {path: [] for path in normalized_paths}
    paths_by_text = {str(path): path for path in normalized_paths}
    placeholders = ", ".join("?" for _ in normalized_paths)
    rows = conn.execute(
        f"""
        SELECT DISTINCT r.source_path, c.conversation_id
        FROM conversations AS c
        JOIN raw_conversations AS r ON r.raw_id = c.raw_id
        WHERE r.source_path IN ({placeholders})
        ORDER BY r.source_path, c.conversation_id
        """,
        tuple(paths_by_text),
    ).fetchall()
    for row in rows:
        path = paths_by_text.get(str(row[0]))
        if path is not None:
            result[path].append(str(row[1]))
    return result


def _fts_repair_needs_for_conversations(
    conn: sqlite3.Connection,
    conversation_ids: Sequence[str],
) -> _FtsRepairNeeds:
    if not conversation_ids:
        return _FtsRepairNeeds()
    placeholders = ", ".join("?" for _ in conversation_ids)
    params = tuple(conversation_ids)
    missing_messages = int(
        conn.execute(
            f"""
            SELECT COUNT(*)
            FROM messages AS m
            LEFT JOIN messages_fts AS f ON f.rowid = m.rowid
            WHERE m.text IS NOT NULL
              AND m.conversation_id IN ({placeholders})
              AND f.rowid IS NULL
            """,
            params,
        ).fetchone()[0]
    )
    messages_missing = missing_messages > 0
    if not _table_exists(conn, "action_events") or not _table_exists(conn, "action_events_fts"):
        return _FtsRepairNeeds(messages=messages_missing)
    missing_actions = int(
        conn.execute(
            f"""
            SELECT COUNT(*)
            FROM action_events AS ae
            LEFT JOIN action_events_fts AS f ON f.rowid = ae.rowid
            WHERE ae.conversation_id IN ({placeholders})
              AND f.rowid IS NULL
            """,
            params,
        ).fetchone()[0]
    )
    return _FtsRepairNeeds(messages=messages_missing, actions=missing_actions > 0)


def _fts_needs_repair_for_conversations(conn: sqlite3.Connection, conversation_ids: Sequence[str]) -> bool:
    return _fts_repair_needs_for_conversations(conn, conversation_ids).any


def _embedding_env_enabled() -> bool:
    import os

    enabled = os.environ.get("POLYLOGUE_DAEMON_ENABLE_EMBEDDINGS", "").lower() in {"1", "true", "yes"}
    return enabled and bool(os.environ.get("VOYAGE_API_KEY"))


def _pending_embedding_conversation_ids(
    conn: sqlite3.Connection,
    conversation_ids: Sequence[str],
) -> list[str]:
    if not conversation_ids or not _table_exists(conn, "embedding_status"):
        return []
    unique_ids = tuple(dict.fromkeys(conversation_ids))
    placeholders = ", ".join("?" for _ in unique_ids)
    rows = conn.execute(
        f"""
        SELECT c.conversation_id
        FROM conversations AS c
        LEFT JOIN embedding_status AS e ON e.conversation_id = c.conversation_id
        WHERE c.conversation_id IN ({placeholders})
          AND (e.conversation_id IS NULL OR e.needs_reindex = 1)
        ORDER BY COALESCE(c.updated_at, ''), c.conversation_id
        """,
        unique_ids,
    ).fetchall()
    return [str(row[0]) for row in rows]


def _embed_conversations_sync(db_path: Path, conversation_ids: Sequence[str]) -> bool:
    import os

    from polylogue.api.sync.bridge import run_coroutine_sync
    from polylogue.storage.embeddings.materialization import embed_conversation_sync
    from polylogue.storage.repository import ConversationRepository
    from polylogue.storage.search_providers import create_vector_provider
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

    voyage_key = os.environ.get("VOYAGE_API_KEY")
    if not voyage_key:
        return True

    vec_provider = create_vector_provider(voyage_api_key=voyage_key, db_path=db_path)
    if vec_provider is None:
        logger.warning("embed: vector provider unavailable")
        return False

    repo = ConversationRepository(backend=SQLiteBackend(db_path=db_path))
    errors = 0
    embedded = 0
    try:
        for conversation_id in dict.fromkeys(conversation_ids):
            outcome = embed_conversation_sync(repo, vec_provider, conversation_id)
            if outcome.status == "embedded":
                embedded += 1
            elif outcome.status == "error":
                errors += 1
                logger.warning("embed: %s failed: %s", conversation_id, outcome.error)
        logger.info("embed: %d done, %d errors", embedded, errors)
        return errors == 0
    finally:
        run_coroutine_sync(repo.close())


def _repair_changed_conversation_fts(
    conn: sqlite3.Connection,
    conversation_ids: Sequence[str],
    *,
    needs: _FtsRepairNeeds | None = None,
) -> None:
    from polylogue.storage.fts.fts_lifecycle import (
        insert_missing_action_fts_index_sync,
        repair_action_fts_index_sync,
        repair_message_fts_index_sync,
    )

    needs = needs or _fts_repair_needs_for_conversations(conn, conversation_ids)
    if needs.messages:
        repair_message_fts_index_sync(conn, conversation_ids)
    if needs.actions:
        if _action_fts_has_rows_for_conversations(conn, conversation_ids):
            repair_action_fts_index_sync(conn, conversation_ids)
        else:
            insert_missing_action_fts_index_sync(conn, conversation_ids)


def _action_fts_has_rows_for_conversations(conn: sqlite3.Connection, conversation_ids: Sequence[str]) -> bool:
    if not conversation_ids or not _table_exists(conn, "action_events_fts"):
        return False
    placeholders = ", ".join("?" for _ in conversation_ids)
    row = conn.execute(
        f"""
        SELECT 1
        FROM action_events_fts
        WHERE conversation_id IN ({placeholders})
        LIMIT 1
        """,
        tuple(conversation_ids),
    ).fetchone()
    return row is not None


def _conversation_ids_missing_profiles(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT c.conversation_id
        FROM conversations AS c
        LEFT JOIN session_profiles AS sp ON sp.conversation_id = c.conversation_id
        WHERE sp.conversation_id IS NULL
        ORDER BY COALESCE(c.sort_key, 0) DESC, c.conversation_id
        """,
    ).fetchall()
    return [str(row[0]) for row in rows]


__all__ = [
    "make_default_convergence_stages",
    "make_embed_stage",
    "make_fts_stage",
    "make_insights_stage",
]
