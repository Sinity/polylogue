"""Convergence stage implementations for the daemon pipeline.

Each stage has a ``check`` that inspects current archive state and an
``execute`` that performs the missing work. The live watcher owns
source ingestion through its batched ``parse_sources(...)`` path; daemon
convergence stages only repair and refresh post-ingest archive state.

- fts: rebuild FTS if messages > indexed count
- embed: vectorize un-embedded conversations via Voyage API
- insights: refresh session profiles
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

from polylogue.daemon.convergence import ConvergenceStage
from polylogue.logging import get_logger

logger = get_logger(__name__)


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
                total = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
                if total == 0:
                    return False
                fts_count = int(conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0])
                return fts_count < total
            finally:
                conn.close()
        except Exception:
            return False

    def execute(path: Path) -> bool:  # noqa: ARG001
        from polylogue.storage.fts.fts_lifecycle import rebuild_fts_index_sync
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=30.0)
            try:
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
        return {Path(paths[0])} if check(Path(paths[0])) else set()

    def execute_many(paths: Sequence[Path]) -> bool:
        return bool(paths) and execute(Path(paths[0]))

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
    """Generate vector embeddings for un-embedded conversations."""

    def check(path: Path) -> bool:  # noqa: ARG001
        import os

        if not os.environ.get("VOYAGE_API_KEY"):
            return False
        if not db_path.exists():
            return False
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=5.0)
            try:
                if not _table_exists(conn, "embedding_status"):
                    return False
                total = int(conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0])
                if total == 0:
                    return False
                embedded = int(conn.execute("SELECT COUNT(*) FROM embedding_status").fetchone()[0])
                return embedded < total
            finally:
                conn.close()
        except Exception:
            return False

    def execute(path: Path) -> bool:  # noqa: ARG001
        import asyncio

        from polylogue.api import Polylogue
        from polylogue.pipeline.run_stages import execute_embed_stage

        async def _embed() -> bool:
            async with Polylogue(archive_root=db_path.parent, db_path=db_path) as poly:
                try:
                    result = await execute_embed_stage(
                        config=poly._config,
                        backend=poly.backend,
                        model="voyage-4",
                    )
                    logger.info("embed: %d done, %d errors", result.embedded_count, result.error_count)
                    return result.error_count == 0
                except Exception:
                    logger.warning("embed: failed", exc_info=True)
                    return False

        return asyncio.run(_embed())

    def check_many(paths: Sequence[Path]) -> set[Path]:
        if not paths:
            return set()
        return {Path(paths[0])} if check(Path(paths[0])) else set()

    def execute_many(paths: Sequence[Path]) -> bool:
        return bool(paths) and execute(Path(paths[0]))

    return ConvergenceStage(
        name="embed",
        description="Generate vector embeddings for un-embedded conversations",
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
                counts = rebuild_session_insights_sync(conn, conversation_ids=conversation_ids)
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
                counts = rebuild_session_insights_sync(conn, conversation_ids=conversation_ids)
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
        path = paths_by_text.get(str(row["source_path"]))
        if path is not None:
            result[path].append(str(row["conversation_id"]))
    return result


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
    return [str(row["conversation_id"]) for row in rows]


__all__ = [
    "make_embed_stage",
    "make_fts_stage",
    "make_insights_stage",
]
