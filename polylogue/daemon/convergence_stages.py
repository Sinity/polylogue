"""Convergence stage implementations for the daemon pipeline.

Each stage has a ``check`` that inspects current archive state and an
``execute`` that performs the missing work. Stages are designed to be
composed into a :class:`~polylogue.daemon.convergence.DaemonConverger`.

Stage order: acquire → ingest → converge (FTS + insights)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from polylogue.daemon.convergence import ConvergenceStage
from polylogue.logging import get_logger

logger = get_logger(__name__)


def _fingerprint_file(path: Path) -> tuple[str, int]:
    """SHA-256 fingerprint of file content, plus last complete newline offset.

    Returns (hex_digest, last_complete_newline_byte_offset).
    """
    hasher = hashlib.sha256()
    last_nl = 0
    offset = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(128 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            # Track last newline for JSONL files that may be mid-write.
            nl = chunk.rfind(b"\n")
            if nl != -1:
                last_nl = offset + nl
            offset += len(chunk)
    return hasher.hexdigest(), last_nl


def make_acquire_stage(db_path: Path, blob_root: Path) -> ConvergenceStage:
    """Stage: acquire — ensure the source file is stored as a content-addressed blob.

    Check returns True if the blob is missing or the file has changed.
    Execute hashes the file, writes to blob store, and inserts/updates
    the ``raw_conversations`` row.
    """

    def check(path: Path) -> bool:
        if not path.exists():
            return False
        from polylogue.sources.live.cursor import CursorStore

        # Use cursor-based fingerprint check — fast path for unchanged files.
        cursor = CursorStore(db_path.parent / "polylogue.sqlite")
        record = cursor.get_record(path)
        if record is None:
            return True
        try:
            fingerprint, _ = _fingerprint_file(path)
        except FileNotFoundError:
            return False
        stat = path.stat()
        return stat.st_size != record.byte_size or fingerprint != record.content_fingerprint

    def execute(path: Path) -> bool:
        from polylogue.api import Polylogue
        from polylogue.sources.live.cursor import CursorStore

        source_name = "ingest"
        try:
            resolved = path.resolve()
            # Derive source name from parent directory
            source_name = resolved.parent.name
        except OSError:
            pass

        cursor = CursorStore(db_path.parent / "polylogue.sqlite")
        import asyncio

        async def _run() -> None:
            async with Polylogue() as polylogue:
                await polylogue.parse_file(path, source_name=source_name)

        try:
            asyncio.run(_run())
        except Exception:
            logger.warning("acquire+ingest failed for %s", path, exc_info=True)
            cursor.mark_failed(path)
            return False

        # Update cursor so next check skips this file.
        try:
            stat = path.stat()
            fingerprint, last_nl = _fingerprint_file(path)
            cursor.set(
                path,
                stat.st_size,
                byte_offset=last_nl,
                last_complete_newline=last_nl,
                parser_fingerprint="convergence-v1",
                content_fingerprint=fingerprint,
                source_name=source_name,
                st_dev=getattr(stat, "st_dev", None),
                st_ino=getattr(stat, "st_ino", None),
                mtime_ns=getattr(stat, "st_mtime_ns", None),
            )
            cursor.reset_failures(path)
        except FileNotFoundError:
            pass
        return True

    return ConvergenceStage(
        name="acquire",
        description="Hash source file and store as content-addressed blob",
        check=check,
        execute=execute,
        cpu_bound=False,
    )


def make_fts_converge_stage(db_path: Path) -> ConvergenceStage:
    """Stage: converge — verify FTS coverage and repair gaps.

    Check returns True if the FTS index has gaps (fewer indexed messages
    than total messages). Execute rebuilds the FTS index.
    """

    def check(path: Path) -> bool:  # noqa: ARG001 — path unused, checks global state
        from polylogue.storage.sqlite.connection_profile import open_connection

        if not db_path.exists():
            return False
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
            logger.warning("convergence: FTS check failed", exc_info=True)
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
                logger.info("convergence: FTS rebuilt — %d/%d messages indexed", new_count, total)
                return new_count >= total
            finally:
                conn.close()
        except Exception:
            logger.warning("convergence: FTS rebuild failed", exc_info=True)
            return False

    return ConvergenceStage(
        name="converge",
        description="Verify FTS coverage and repair gaps",
        check=check,
        execute=execute,
        cpu_bound=False,
    )


def make_insight_converge_stage(db_path: Path) -> ConvergenceStage:
    """Stage: insights — verify session insight freshness.

    Check returns True if any conversations have stale or missing
    session insights. Execute rebuilds insights for affected conversations.
    """

    def check(path: Path) -> bool:  # noqa: ARG001
        from polylogue.storage.sqlite.connection_profile import open_connection

        if not db_path.exists():
            return False
        try:
            conn = open_connection(db_path, timeout=5.0)
            try:
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
            async with Polylogue() as polylogue:
                await polylogue.rebuild_insights()

        try:
            asyncio.run(_refresh())
            return True
        except Exception:
            logger.warning("convergence: insight rebuild failed", exc_info=True)
            return False

    return ConvergenceStage(
        name="insights",
        description="Refresh session insights for new conversations",
        check=check,
        execute=execute,
        cpu_bound=False,
    )


__all__ = [
    "make_acquire_stage",
    "make_fts_converge_stage",
    "make_insight_converge_stage",
]
