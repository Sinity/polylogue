"""Convergence stage implementations for the daemon pipeline.

Each stage has a ``check`` that inspects current archive state and an
``execute`` that performs the missing work. Stages run in order:
acquire-check → ingest → fts → embed → insights.

- acquire: fingerprint-based skip (no execute — ingest handles it)
- ingest: parse_file() for the full acquire+parse+materialize+write
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


def _fingerprint_file(path: Path) -> tuple[str, int]:
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


# ── Stage: acquire-check ───────────────────────────────────────────
# Checks fingerprint against cursor; execute is no-op because
# ingest handles the full pipeline. This separation lets callers
# batch-skip unchanged files before paying the parse cost.


def make_acquire_check_stage(db_path: Path) -> ConvergenceStage:
    """Fingerprint check — returns True if file needs (re)ingestion."""

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
        return True  # no-op — ingest stage handles the work

    return ConvergenceStage(
        name="acquire",
        description="Fingerprint check — skip unchanged files",
        check=check,
        execute=execute,
        cpu_bound=False,
    )


# ── Stage: ingest ──────────────────────────────────────────────────


def make_ingest_stage(db_path: Path) -> ConvergenceStage:
    """Ingest a file through the full pipeline: acquire + parse + materialize + write.

    This is the workhorse. It calls ``parse_file()`` which does the
    full pipeline for a single file. Subsequent stages (fts, embed,
    insights) handle post-ingest convergence.
    """

    def check(path: Path) -> bool:
        """Check if the file's raw record exists and is parsed."""
        if not path.exists():
            return False
        from polylogue.sources.live.cursor import CursorStore

        cursor = CursorStore(db_path.parent / "polylogue.sqlite")
        record = cursor.get_record(path)
        if record is None:
            return True
        # Check if the file fingerprint differs from what was parsed.
        try:
            fp, _ = _fingerprint_file(path)
        except FileNotFoundError:
            return False
        return fp != (record.content_fingerprint or "")

    def execute(path: Path) -> bool:
        import asyncio

        from polylogue.api import Polylogue
        from polylogue.sources.live.cursor import CursorStore

        async def _run() -> bool:
            source_name = path.parent.name
            try:
                async with Polylogue() as poly:
                    await poly.parse_file(path, source_name=source_name)
            except Exception:
                logger.warning("ingest failed for %s", path, exc_info=True)
                cursor = CursorStore(db_path.parent / "polylogue.sqlite")
                cursor.mark_failed(path)
                return False

            # Update cursor so subsequent checks skip this file.
            cursor = CursorStore(db_path.parent / "polylogue.sqlite")
            try:
                stat = path.stat()
                fp, last_nl = _fingerprint_file(path)
                cursor.set(
                    path,
                    stat.st_size,
                    byte_offset=last_nl,
                    last_complete_newline=last_nl,
                    parser_fingerprint="convergence-v3",
                    content_fingerprint=fp,
                    source_name=source_name,
                    st_dev=None,
                    st_ino=None,
                    mtime_ns=None,
                )
                cursor.reset_failures(path)
            except FileNotFoundError:
                pass
            return True

        return asyncio.run(_run())

    return ConvergenceStage(
        name="ingest",
        description="Full pipeline: acquire → parse → materialize → write",
        check=check,
        execute=execute,
        cpu_bound=True,
    )


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
    "make_acquire_check_stage",
    "make_embed_stage",
    "make_fts_stage",
    "make_ingest_stage",
    "make_insights_stage",
]
