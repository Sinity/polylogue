"""Convergence stage implementations for the daemon pipeline.

Each stage has a ``check`` that inspects current archive state and an
``execute`` that performs the missing work. Stages run in order:
acquire → parse → materialize → index → embed → insights.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.daemon.convergence import ConvergenceStage
from polylogue.logging import get_logger

logger = get_logger(__name__)


def _fingerprint_file(path: Path) -> tuple[str, int]:
    """SHA-256 fingerprint + last complete newline offset."""
    import hashlib

    hasher = hashlib.sha256()
    last_nl = 0
    offset = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(128 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            nl = chunk.rfind(b"\n")
            if nl != -1:
                last_nl = offset + nl
            offset += len(chunk)
    return hasher.hexdigest(), last_nl


# ── Stage: acquire ────────────────────────────────────────────────


def make_acquire_stage(db_path: Path) -> ConvergenceStage:
    """Ensure the source file is stored as a blob and has a raw_conversations row."""

    def check(path: Path) -> bool:
        if not path.exists():
            return False
        from polylogue.sources.live.cursor import CursorStore

        cursor = CursorStore(db_path.parent / "polylogue.sqlite")
        record = cursor.get_record(path)
        if record is None:
            return True
        try:
            fp, _ = _fingerprint_file(path)
        except FileNotFoundError:
            return False
        stat = path.stat()
        return stat.st_size != record.byte_size or fp != record.content_fingerprint

    def execute(path: Path) -> bool:
        import asyncio

        from polylogue.api import Polylogue
        from polylogue.sources.live.cursor import CursorStore

        async def _acquire() -> bool:
            async with Polylogue() as poly:
                try:
                    result = await poly.parse_file(path)
                    return result is not None
                except Exception:
                    logger.warning("acquire failed for %s", path, exc_info=True)
                    return False

        try:
            ok = asyncio.run(_acquire())
        except Exception:
            logger.warning("acquire failed for %s", path, exc_info=True)
            return False

        if ok:
            cursor = CursorStore(db_path.parent / "polylogue.sqlite")
            try:
                stat = path.stat()
                fp, last_nl = _fingerprint_file(path)
                cursor.set(
                    path,
                    stat.st_size,
                    byte_offset=last_nl,
                    last_complete_newline=last_nl,
                    parser_fingerprint="convergence-v2",
                    content_fingerprint=fp,
                    source_name=path.parent.name,
                    st_dev=None,
                    st_ino=None,
                    mtime_ns=None,
                )
                cursor.reset_failures(path)
            except FileNotFoundError:
                pass
        return ok

    return ConvergenceStage(
        name="acquire",
        description="Hash and store source file as content-addressed blob",
        check=check,
        execute=execute,
        cpu_bound=False,
    )


# ── Stage: parse ──────────────────────────────────────────────────


def make_parse_stage(db_path: Path) -> ConvergenceStage:
    """Parse acquired blobs — decode JSONL, detect provider, extract records."""

    def check(path: Path) -> bool:
        """Check if raw record exists and hasn't been parsed yet."""
        if not db_path.exists():
            return True
        from polylogue.sources.live.cursor import CursorStore
        from polylogue.storage.sqlite.connection_profile import open_connection

        cursor = CursorStore(db_path.parent / "polylogue.sqlite")
        record = cursor.get_record(path)
        if record is None:
            return True  # Never seen — need to acquire first, but parse needs it too
        try:
            conn = open_connection(db_path, timeout=5.0)
            try:
                fp, _ = _fingerprint_file(path)
                row = conn.execute(
                    "SELECT 1 FROM raw_conversations r "
                    "JOIN live_cursor l ON l.source_path = r.source_path "
                    "WHERE r.source_path = ? AND r.parsed_at IS NOT NULL "
                    "AND l.content_fingerprint = ? "
                    "LIMIT 1",
                    (str(path), fp),
                ).fetchone()
                return row is None
            finally:
                conn.close()
        except Exception:
            return True

    def execute(path: Path) -> bool:
        import asyncio

        from polylogue.api import Polylogue

        async def _parse() -> bool:
            async with Polylogue() as poly:
                try:
                    await poly.parse_file(path)
                    return True
                except Exception:
                    logger.warning("parse failed for %s", path, exc_info=True)
                    return False

        return asyncio.run(_parse())

    return ConvergenceStage(
        name="parse",
        description="Decode JSONL, detect provider, extract message records",
        check=check,
        execute=execute,
        cpu_bound=True,
    )


# ── Stage: FTS converge ───────────────────────────────────────────


def make_fts_converge_stage(db_path: Path) -> ConvergenceStage:
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
                logger.info("convergence: FTS rebuilt — %d/%d indexed", new_count, total)
                return new_count >= total
            finally:
                conn.close()
        except Exception:
            logger.warning("convergence: FTS rebuild failed", exc_info=True)
            return False

    return ConvergenceStage(
        name="fts",
        description="Verify FTS coverage and repair gaps",
        check=check,
        execute=execute,
        cpu_bound=False,
    )


# ── Stage: embed ──────────────────────────────────────────────────


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
                    logger.info("embed: %d embedded, %d errors", result.embedded_count, result.error_count)
                    return result.error_count == 0
                except Exception:
                    logger.warning("embed failed", exc_info=True)
                    return False

        return asyncio.run(_embed())

    return ConvergenceStage(
        name="embed",
        description="Generate vector embeddings for un-embedded conversations",
        check=check,
        execute=execute,
        cpu_bound=False,
    )


# ── Stage: insights ───────────────────────────────────────────────


def make_insight_converge_stage(db_path: Path) -> ConvergenceStage:
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
            logger.warning("insight rebuild failed", exc_info=True)
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
    "make_acquire_stage",
    "make_embed_stage",
    "make_fts_converge_stage",
    "make_insight_converge_stage",
    "make_parse_stage",
]
