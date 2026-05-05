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

    return ConvergenceStage(
        name="fts",
        description="Verify FTS coverage and repair gaps",
        check=check,
        execute=execute,
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
            async with Polylogue() as poly:
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

    return ConvergenceStage(
        name="embed",
        description="Generate vector embeddings for un-embedded conversations",
        check=check,
        execute=execute,
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
        import asyncio

        from polylogue.api import Polylogue

        async def _refresh() -> None:
            async with Polylogue() as poly:
                await poly.rebuild_insights()

        try:
            asyncio.run(_refresh())
            return True
        except Exception:
            logger.warning("insights: rebuild failed", exc_info=True)
            return False

    return ConvergenceStage(
        name="insights",
        description="Refresh session insights for new conversations",
        check=check,
        execute=execute,
        cpu_bound=False,
    )


# ── Helpers ────────────────────────────────────────────────────────


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


__all__ = [
    "make_embed_stage",
    "make_fts_stage",
    "make_insights_stage",
]
