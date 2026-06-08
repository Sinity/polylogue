"""Convergence stage implementations for the daemon pipeline.

Each stage has a ``check`` that inspects current archive state and an
``execute`` that performs the missing work. The live watcher owns source
ingestion through daemon-side raw-record ingest; daemon convergence stages only
repair and refresh post-ingest archive state.

- fts: rebuild FTS if messages > indexed count
- embed: optional vectorization for changed sessions
- insights: refresh session profiles
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.config import load_polylogue_config
from polylogue.daemon.convergence import ConvergenceStage
from polylogue.logging import get_logger
from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION
from polylogue.storage.source_sessions import (
    session_ids_for_source_path,
    session_ids_for_source_paths,
)

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

_DAEMON_INSIGHT_REBUILD_PAGE_SIZE = 10
_HOT_INSIGHT_SOURCE_BYTES = 64 * 1024 * 1024
_HOT_INSIGHT_QUIET_SECONDS = 60.0
_DAEMON_EMBED_MAX_SESSIONS = 25
_DAEMON_EMBED_MAX_MESSAGES = 2_500
_DAEMON_EMBED_STOP_AFTER_SECONDS = 30
_DAEMON_EMBED_MAX_ERRORS = 3


@dataclass(frozen=True, slots=True)
class _FtsRepairNeeds:
    messages: bool = False

    @property
    def any(self) -> bool:
        return self.messages


# ── Stage: FTS ─────────────────────────────────────────────────────


def make_fts_stage(db_path: Path) -> ConvergenceStage:
    """Verify FTS coverage and repair gaps."""

    def check(path: Path) -> bool:
        archive_db = _active_archive_index_path(db_path)
        if archive_db is not None:
            return _archive_fts_check(archive_db, path)
        if not db_path.exists():
            return False
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=5.0)
            try:
                session_ids = _session_ids_for_source_path(conn, path)
                if session_ids:
                    return _fts_needs_repair_for_sessions(conn, session_ids)
                from polylogue.storage.fts.sql import FTS_INDEXABLE_MESSAGE_COUNT_SQL

                total = int(conn.execute(FTS_INDEXABLE_MESSAGE_COUNT_SQL).fetchone()[0])
                fts_count = _fts_doc_count(conn, "messages_fts_docsize")
                if fts_count != total:
                    return True
                if total == 0:
                    return False
                return False
            finally:
                conn.close()
        except Exception:
            return False

    def execute(path: Path) -> bool:
        archive_db = _active_archive_index_path(db_path)
        if archive_db is not None:
            return _archive_fts_execute(archive_db, path)
        from polylogue.storage.fts.fts_lifecycle import rebuild_fts_index_sync
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=30.0)
            try:
                session_ids = _session_ids_for_source_path(conn, path)
                if session_ids:
                    needs = _fts_repair_needs_for_sessions(conn, session_ids)
                    _repair_changed_session_fts(conn, session_ids, needs=needs)
                    _mark_message_fts_ready_after_targeted_repair(conn)
                    conn.commit()
                    logger.info("fts: repaired sessions=%d", len(session_ids))
                    return not _fts_needs_repair_for_sessions(conn, session_ids)
                from polylogue.storage.fts.sql import FTS_INDEXABLE_MESSAGE_COUNT_SQL

                total = int(conn.execute(FTS_INDEXABLE_MESSAGE_COUNT_SQL).fetchone()[0])
                rebuild_fts_index_sync(conn)
                conn.commit()
                new_count = _fts_doc_count(conn, "messages_fts_docsize")
                logger.info("fts: rebuilt — %d/%d indexed", new_count, total)
                return new_count == total
            finally:
                conn.close()
        except Exception:
            logger.warning("fts: rebuild failed", exc_info=True)
            return False

    def check_many(paths: Sequence[Path]) -> set[Path]:
        if not paths:
            return set()
        archive_db = _active_archive_index_path(db_path)
        if archive_db is not None:
            return _archive_fts_check_many(archive_db, paths)
        if not db_path.exists():
            return set()
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=5.0)
            try:
                by_path = _session_ids_for_source_paths(conn, paths)
                return {
                    path
                    for path, session_ids in by_path.items()
                    if session_ids and _fts_repair_needs_for_sessions(conn, session_ids).any
                }
            finally:
                conn.close()
        except Exception:
            return set()

    def execute_many(paths: Sequence[Path]) -> bool:
        if not paths:
            return False
        archive_db = _active_archive_index_path(db_path)
        if archive_db is not None:
            return _archive_fts_execute_many(archive_db, paths)
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=30.0)
            try:
                by_path = _session_ids_for_source_paths(conn, paths)
                session_ids = list(dict.fromkeys(session_id for ids in by_path.values() for session_id in ids))
                if not session_ids:
                    return execute(Path(paths[0]))
                needs = _fts_repair_needs_for_sessions(conn, session_ids)
                _repair_changed_session_fts(conn, session_ids, needs=needs)
                _mark_message_fts_ready_after_targeted_repair(conn)
                conn.commit()
                logger.info("fts: batch repaired paths=%d sessions=%d", len(paths), len(session_ids))
                return not _fts_needs_repair_for_sessions(conn, session_ids)
            finally:
                conn.close()
        except Exception:
            logger.warning("fts: batch repair failed", exc_info=True)
            return False

    def check_sessions(session_ids: Sequence[str]) -> set[str]:
        if not session_ids:
            return set()
        archive_db = _active_archive_index_path(db_path)
        if archive_db is not None:
            return _archive_fts_check_sessions(archive_db, session_ids)
        if not db_path.exists():
            return set()
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=5.0)
            try:
                return {
                    session_id
                    for session_id in dict.fromkeys(session_ids)
                    if _fts_repair_needs_for_sessions(conn, [session_id]).any
                }
            finally:
                conn.close()
        except Exception:
            return set()

    def execute_sessions(session_ids: Sequence[str]) -> bool:
        if not session_ids:
            return True
        archive_db = _active_archive_index_path(db_path)
        if archive_db is not None:
            return _archive_fts_execute_sessions(archive_db, session_ids)
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=30.0)
            try:
                ids = tuple(dict.fromkeys(session_ids))
                needs = _fts_repair_needs_for_sessions(conn, ids)
                _repair_changed_session_fts(conn, ids, needs=needs)
                _mark_message_fts_ready_after_targeted_repair(conn)
                conn.commit()
                logger.info("fts: repaired session debt sessions=%d", len(ids))
                return not _fts_needs_repair_for_sessions(conn, ids)
            finally:
                conn.close()
        except Exception:
            logger.warning("fts: session repair failed", exc_info=True)
            return False

    return ConvergenceStage(
        name="fts",
        description="Verify FTS coverage and repair gaps",
        check=check,
        execute=execute,
        check_many=check_many,
        execute_many=execute_many,
        check_sessions=check_sessions,
        execute_sessions=execute_sessions,
        cpu_bound=False,
        false_means_pending=True,
    )


# ── Stage: embed ───────────────────────────────────────────────────


def make_embed_stage(db_path: Path) -> ConvergenceStage:
    """Generate vector embeddings for changed sessions that need them.

    Before embedding, detects model/dimension config changes and marks
    affected rows for reindex. Enforces the configured cost cap during
    embedding.
    """

    def check(path: Path) -> bool:
        if not _embedding_config_enabled():
            return False
        archive_db = _active_archive_index_path(db_path)
        return _archive_embed_check(archive_db, path) if archive_db is not None else False

    def execute(path: Path) -> bool:
        if not _embedding_config_enabled():
            return True
        archive_db = _active_archive_index_path(db_path)
        return _archive_embed_execute(archive_db, path) if archive_db is not None else True

    def check_many(paths: Sequence[Path]) -> set[Path]:
        if not paths or not _embedding_config_enabled():
            return set()
        archive_db = _active_archive_index_path(db_path)
        return _archive_embed_check_many(archive_db, paths) if archive_db is not None else set()

    def execute_many(paths: Sequence[Path]) -> bool:
        if not paths or not _embedding_config_enabled():
            return True
        archive_db = _active_archive_index_path(db_path)
        return _archive_embed_execute_many(archive_db, paths) if archive_db is not None else True

    def check_sessions(session_ids: Sequence[str]) -> set[str]:
        if not session_ids or not _embedding_config_enabled():
            return set()
        archive_db = _active_archive_index_path(db_path)
        return _archive_embed_check_sessions(archive_db, session_ids) if archive_db is not None else set()

    def execute_sessions(session_ids: Sequence[str]) -> bool:
        if not session_ids or not _embedding_config_enabled():
            return True
        archive_db = _active_archive_index_path(db_path)
        return _archive_embed_execute_sessions(archive_db, session_ids) if archive_db is not None else True

    return ConvergenceStage(
        name="embed",
        description="Generate vector embeddings for changed sessions",
        check=check,
        execute=execute,
        check_many=check_many,
        execute_many=execute_many,
        check_sessions=check_sessions,
        execute_sessions=execute_sessions,
        cpu_bound=False,
        false_means_pending=True,
    )


# ── Stage: insights ────────────────────────────────────────────────


def make_insights_stage(db_path: Path) -> ConvergenceStage:
    """Refresh session insights for sessions missing them."""

    def check(path: Path) -> bool:
        archive_db = _active_archive_index_path(db_path)
        if archive_db is not None:
            return _archive_insights_check(archive_db, path)
        if not db_path.exists():
            return False
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=5.0)
            try:
                if not _table_exists(conn, "session_profiles"):
                    return False
                session_ids = _session_ids_for_source_path(conn, path)
                if session_ids:
                    return bool(_stale_session_profile_ids(conn, session_ids))
                total_conv = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
                if total_conv == 0:
                    return False
                profiled = int(conn.execute("SELECT COUNT(*) FROM session_profiles").fetchone()[0])
                return profiled < total_conv
            finally:
                conn.close()
        except Exception:
            return False

    def execute(path: Path) -> bool:
        archive_db = _active_archive_index_path(db_path)
        if archive_db is not None:
            return _archive_insights_execute(archive_db, path)
        from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync
        from polylogue.storage.sqlite.connection import open_connection

        try:
            with open_connection(db_path) as conn:
                session_ids = _session_ids_for_source_path(conn, path) or _session_ids_missing_profiles(conn)
                hot_ids = _hot_insight_session_ids(conn, session_ids)
                if hot_ids:
                    logger.info(
                        "insights: deferring hot source rebuild sessions=%d quiet_s=%.0f",
                        len(hot_ids),
                        _HOT_INSIGHT_QUIET_SECONDS,
                    )
                    return False
                counts = rebuild_session_insights_sync(
                    conn,
                    session_ids=session_ids,
                    page_size=_DAEMON_INSIGHT_REBUILD_PAGE_SIZE,
                )
                conn.commit()
                logger.info(
                    "insights: refreshed sessions=%d profiles=%d work_events=%d phases=%d threads=%d",
                    len(session_ids),
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
        archive_db = _active_archive_index_path(db_path)
        if archive_db is not None:
            return _archive_insights_check_many(archive_db, paths)
        if not db_path.exists() or not paths:
            return set()
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=5.0)
            try:
                if not _table_exists(conn, "session_profiles"):
                    return set()
                by_path = _session_ids_for_source_paths(conn, paths)
                paths_with_sessions = {
                    path
                    for path, session_ids in by_path.items()
                    if session_ids and _stale_session_profile_ids(conn, session_ids)
                }
                if paths_with_sessions:
                    return paths_with_sessions
                if _session_ids_missing_profiles(conn):
                    return {Path(paths[0])}
                return set()
            finally:
                conn.close()
        except Exception:
            return set()

    def execute_many(paths: Sequence[Path]) -> bool:
        archive_db = _active_archive_index_path(db_path)
        if archive_db is not None:
            return _archive_insights_execute_many(archive_db, paths)
        from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync
        from polylogue.storage.sqlite.connection import open_connection

        try:
            with open_connection(db_path) as conn:
                by_path = _session_ids_for_source_paths(conn, paths)
                session_ids = list(dict.fromkeys(session_id for ids in by_path.values() for session_id in ids))
                if not session_ids:
                    session_ids = _session_ids_missing_profiles(conn)
                hot_ids = _hot_insight_session_ids(conn, session_ids)
                if hot_ids:
                    logger.info(
                        "insights: deferring hot source batch rebuild sessions=%d quiet_s=%.0f",
                        len(hot_ids),
                        _HOT_INSIGHT_QUIET_SECONDS,
                    )
                    return False
                counts = rebuild_session_insights_sync(
                    conn,
                    session_ids=session_ids,
                    page_size=_DAEMON_INSIGHT_REBUILD_PAGE_SIZE,
                )
                conn.commit()
                logger.info(
                    "insights: batch refreshed paths=%d sessions=%d profiles=%d work_events=%d phases=%d threads=%d",
                    len(paths),
                    len(session_ids),
                    counts.profiles,
                    counts.work_events,
                    counts.phases,
                    counts.threads,
                )
            return True
        except Exception:
            logger.warning("insights: batch rebuild failed", exc_info=True)
            return False

    def check_sessions(session_ids: Sequence[str]) -> set[str]:
        if not session_ids:
            return set()
        archive_db = _active_archive_index_path(db_path)
        if archive_db is not None:
            return _archive_insights_check_sessions(archive_db, session_ids)
        if not db_path.exists():
            return set()
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=5.0)
            try:
                if not _table_exists(conn, "session_profiles"):
                    return set()
                return set(_stale_session_profile_ids(conn, tuple(dict.fromkeys(session_ids))))
            finally:
                conn.close()
        except Exception:
            return set()

    def execute_sessions(session_ids: Sequence[str]) -> bool:
        archive_db = _active_archive_index_path(db_path)
        if archive_db is not None:
            return _archive_insights_execute_sessions(archive_db, session_ids)
        from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync
        from polylogue.storage.sqlite.connection import open_connection

        try:
            with open_connection(db_path) as conn:
                ids = _existing_session_ids(conn, tuple(dict.fromkeys(session_ids)))
                if not ids:
                    return True
                hot_ids = _hot_insight_session_ids(conn, ids)
                if hot_ids:
                    logger.info(
                        "insights: deferring hot source session rebuild sessions=%d quiet_s=%.0f",
                        len(hot_ids),
                        _HOT_INSIGHT_QUIET_SECONDS,
                    )
                    return False
                counts = rebuild_session_insights_sync(
                    conn,
                    session_ids=ids,
                    page_size=_DAEMON_INSIGHT_REBUILD_PAGE_SIZE,
                )
                conn.commit()
                remaining = _stale_session_profile_ids(conn, ids)
                logger.info(
                    "insights: refreshed session debt sessions=%d profiles=%d work_events=%d phases=%d threads=%d remaining=%d",
                    len(ids),
                    counts.profiles,
                    counts.work_events,
                    counts.phases,
                    counts.threads,
                    len(remaining),
                )
                if remaining:
                    return False
            return True
        except Exception:
            logger.warning("insights: session rebuild failed", exc_info=True)
            return False

    return ConvergenceStage(
        name="insights",
        description="Refresh session insights for new sessions",
        check=check,
        execute=execute,
        check_many=check_many,
        execute_many=execute_many,
        check_sessions=check_sessions,
        execute_sessions=execute_sessions,
        cpu_bound=False,
    )


def make_default_convergence_stages(db_path: Path) -> tuple[ConvergenceStage, ...]:
    """Build the daemon's default post-ingest convergence stage set."""
    return (
        make_fts_stage(db_path),
        make_embed_stage(db_path),
        make_insights_stage(db_path),
    )


# ── Helpers ────────────────────────────────────────────────────────


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _fts_doc_count(conn: sqlite3.Connection, table: str) -> int:
    if not _table_exists(conn, table):
        return 0
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0] or 0) if row is not None else 0


def _session_ids_for_source_path(conn: sqlite3.Connection, path: Path) -> list[str]:
    return session_ids_for_source_path(conn, path)


def _session_ids_for_source_paths(
    conn: sqlite3.Connection,
    paths: Sequence[Path],
) -> dict[Path, list[str]]:
    return session_ids_for_source_paths(conn, paths)


def _fts_repair_needs_for_sessions(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
) -> _FtsRepairNeeds:
    if not session_ids:
        return _FtsRepairNeeds()
    if not _table_exists(conn, "messages_fts_docsize"):
        return _FtsRepairNeeds(messages=True)
    placeholders = ", ".join("?" for _ in session_ids)
    params = tuple(session_ids)
    missing_blocks = int(
        conn.execute(
            f"""
            SELECT COUNT(*)
            FROM blocks AS b
            LEFT JOIN messages_fts_docsize AS d ON d.id = b.rowid
            WHERE b.session_id IN ({placeholders})
              AND d.id IS NULL
              AND NULLIF(b.search_text, '') IS NOT NULL
            """,
            params,
        ).fetchone()[0]
    )
    return _FtsRepairNeeds(messages=missing_blocks > 0)


def _fts_needs_repair_for_sessions(conn: sqlite3.Connection, session_ids: Sequence[str]) -> bool:
    return _fts_repair_needs_for_sessions(conn, session_ids).any


def _embedding_config_enabled() -> bool:
    """Check whether embedding convergence is enabled via the shared config layer."""

    cfg = load_polylogue_config()
    return bool(cfg.embedding_enabled) and bool(cfg.voyage_api_key)


def _reconcile_embedding_config_change(conn: sqlite3.Connection) -> None:
    """Detect model/dimension config changes and mark rows for reindex.

    When the configured model differs from what is stored in embeddings_meta,
    all embedding_status rows are marked ``needs_reindex = 1``. When the
    configured dimension differs, the vec0 table is also dropped so it can
    be recreated with the new dimension.
    """
    from polylogue.storage.search_providers.sqlite_vec_runtime import _reconcile_vec0_dimension
    from polylogue.storage.search_providers.sqlite_vec_support import logger as vec_logger

    cfg = load_polylogue_config()
    configured_model = cfg.embedding_model
    configured_dimension = cfg.embedding_dimension

    if not _table_exists(conn, "embeddings_meta") or not _table_exists(conn, "embedding_status"):
        return

    # Check stored model
    stored_models = conn.execute(
        "SELECT DISTINCT model FROM embeddings_meta WHERE target_type='message' ORDER BY model"
    ).fetchall()
    stored_model = str(stored_models[0][0]) if stored_models else None

    model_changed = stored_model is not None and stored_model != configured_model
    dimension_changed = False

    if stored_model is not None:
        stored_dim_row = conn.execute(
            "SELECT dimension FROM embeddings_meta WHERE target_type='message' LIMIT 1"
        ).fetchone()
        if stored_dim_row:
            stored_dimension = int(stored_dim_row[0])
            dimension_changed = stored_dimension != configured_dimension

    if model_changed:
        vec_logger.info(
            "embedding model changed: stored=%s configured=%s — marking all for reindex",
            stored_model,
            configured_model,
        )
    if dimension_changed:
        vec_logger.info(
            "embedding dimension changed: stored=%d configured=%d — dropping vec0 + reindex",
            stored_model and _stored_dim_from_meta(conn) or 0,
            configured_dimension,
        )

    if model_changed or dimension_changed:
        conn.execute("UPDATE embedding_status SET needs_reindex = 1, error_message = NULL")
        if dimension_changed:
            _reconcile_vec0_dimension(conn, configured_dimension)


def _stored_dim_from_meta(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT dimension FROM embeddings_meta WHERE target_type='message' LIMIT 1").fetchone()
    return int(row[0]) if row else 0


def _repair_changed_session_fts(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
    *,
    needs: _FtsRepairNeeds | None = None,
) -> None:
    from polylogue.storage.fts.fts_lifecycle import repair_message_fts_index_sync

    needs = needs or _fts_repair_needs_for_sessions(conn, session_ids)
    if needs.messages:
        repair_message_fts_index_sync(conn, session_ids)


def _mark_message_fts_ready_after_targeted_repair(conn: sqlite3.Connection) -> None:
    from polylogue.storage.fts.freshness import READY, STALE, record_fts_surface_state_sync
    from polylogue.storage.fts.fts_lifecycle import message_fts_readiness_sync

    readiness = message_fts_readiness_sync(conn, verify_total_rows=False)
    freshness_columns = (
        {str(row[1]) for row in conn.execute("PRAGMA table_info(fts_freshness_state)").fetchall()}
        if _table_exists(conn, "fts_freshness_state")
        else set()
    )
    existing = (
        conn.execute(
            """
            SELECT source_rows, indexed_rows, missing_rows, excess_rows, duplicate_rows
            FROM fts_freshness_state
            WHERE surface = 'messages_fts'
            """,
        ).fetchone()
        if {"source_rows", "indexed_rows", "missing_rows", "excess_rows", "duplicate_rows"} <= freshness_columns
        else None
    )
    counts = existing if existing is not None else (0, 0, 0, 0, 0)
    record_fts_surface_state_sync(
        conn,
        surface="messages_fts",
        state=READY if bool(readiness["ready"]) else STALE,
        source_rows=int(counts[0] or 0),
        indexed_rows=int(counts[1] or 0),
        missing_rows=0 if bool(readiness["ready"]) else int(counts[2] or 0),
        excess_rows=0 if bool(readiness["ready"]) else int(counts[3] or 0),
        duplicate_rows=0 if bool(readiness["ready"]) else int(counts[4] or 0),
        detail=(
            "targeted changed-session repair complete"
            if bool(readiness["ready"])
            else "targeted changed-session repair left structural FTS readiness false"
        ),
    )


def _session_ids_missing_profiles(conn: sqlite3.Connection) -> list[str]:
    """Sessions whose session_profile is missing or stale (#1620)."""
    from polylogue.storage.insights.session.status import SESSION_PROFILE_REPAIR_CANDIDATES_SQL
    from polylogue.storage.runtime.store_constants import SESSION_INSIGHT_MATERIALIZER_VERSION

    rows = conn.execute(SESSION_PROFILE_REPAIR_CANDIDATES_SQL, (SESSION_INSIGHT_MATERIALIZER_VERSION,)).fetchall()
    return [str(row[0]) for row in rows]


def _existing_session_ids(conn: sqlite3.Connection, session_ids: Sequence[str]) -> list[str]:
    unique_ids = tuple(dict.fromkeys(session_ids))
    if not unique_ids or not _table_exists(conn, "sessions"):
        return []
    placeholders = ", ".join("?" for _ in unique_ids)
    rows = conn.execute(
        f"""
        SELECT session_id
        FROM sessions
        WHERE session_id IN ({placeholders})
        ORDER BY session_id
        """,
        unique_ids,
    ).fetchall()
    return [str(row[0]) for row in rows]


def _hot_insight_session_ids(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
    *,
    now: float | None = None,
) -> set[str]:
    """Return stale sessions whose source file is too hot for full insight rebuild.

    Live archive writes and targeted FTS repair must stay immediate. Session
    insight rebuilds can require rehydrating an entire session; for huge
    actively-appending agent sessions that turns every small append into a
    multi-GB read cycle. Returning False from the stage records durable
    convergence debt, so this is a quiet-window deferral, not a scope reduction.
    """

    unique_ids = tuple(dict.fromkeys(str(session_id) for session_id in session_ids if session_id))
    if not unique_ids or not _table_exists(conn, "sessions") or not _table_exists(conn, "raw_sessions"):
        return set()
    placeholders = ", ".join("?" for _ in unique_ids)
    rows = conn.execute(
        f"""
        SELECT DISTINCT c.session_id, r.source_path
        FROM sessions AS c
        JOIN raw_sessions AS r ON r.raw_id = c.raw_id
        WHERE c.session_id IN ({placeholders})
          AND r.source_path IS NOT NULL
          AND r.source_path != ''
        ORDER BY c.session_id
        """,
        unique_ids,
    ).fetchall()
    current = time.time() if now is None else now
    hot: set[str] = set()
    for session_id, source_path in rows:
        if _source_path_is_hot_for_insights(Path(str(source_path)), now=current):
            hot.add(str(session_id))
    return hot


def _source_path_is_hot_for_insights(path: Path, *, now: float | None = None) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    if stat.st_size < _HOT_INSIGHT_SOURCE_BYTES:
        return False
    current = time.time() if now is None else now
    return current - stat.st_mtime < _HOT_INSIGHT_QUIET_SECONDS


def _stale_session_profile_ids(conn: sqlite3.Connection, session_ids: Sequence[str]) -> list[str]:
    unique_ids = tuple(dict.fromkeys(str(session_id) for session_id in session_ids if session_id))
    if not unique_ids or not _table_exists(conn, "sessions") or not _table_exists(conn, "session_profiles"):
        return []
    placeholders = ", ".join("?" for _ in unique_ids)
    rows = conn.execute(
        f"""
        SELECT c.session_id
        FROM sessions AS c
        LEFT JOIN session_profiles AS sp ON sp.session_id = c.session_id
        WHERE c.session_id IN ({placeholders})
          AND (
              sp.session_id IS NULL
              OR sp.materializer_version != ?
              OR (
                  c.sort_key_ms IS NOT NULL
                  AND ABS(COALESCE(sp.source_sort_key, 0.0) - (CAST(c.sort_key_ms AS REAL) / 1000.0)) > 0.000001
              )
              OR (
                  c.sort_key_ms IS NULL
                  AND COALESCE(strftime('%s', sp.source_updated_at), sp.source_updated_at, '') !=
                      COALESCE(CAST(c.updated_at_ms / 1000 AS TEXT), '')
              )
          )
        ORDER BY c.session_id
        """,
        unique_ids + (SESSION_INSIGHT_MATERIALIZER_VERSION,),
    ).fetchall()
    return [str(row[0]) for row in rows]


# ── Archive file-set helpers ─────────────────────────────────────


def _attached_source_db_path(conn: sqlite3.Connection) -> Path:
    for _, name, path in conn.execute("PRAGMA database_list").fetchall():
        if str(name) == "main" and path:
            return Path(str(path)).with_name("source.db")
    return Path("source.db")


def _ensure_source_tier_attached(conn: sqlite3.Connection) -> bool:
    for _, name, _path in conn.execute("PRAGMA database_list").fetchall():
        if str(name) == "source_tier":
            return True
    source_db = _attached_source_db_path(conn)
    if not source_db.exists():
        return False
    conn.execute("ATTACH DATABASE ? AS source_tier", (str(source_db),))
    return True


def _active_archive_index_path(db_path: Path) -> Path | None:
    candidates: list[Path] = []
    if db_path.name == "index.db":
        candidates.append(db_path)
    candidates.append(db_path.with_name("index.db"))
    index_db = next((candidate for candidate in dict.fromkeys(candidates) if candidate.exists()), None)
    if index_db is None:
        return None
    try:
        conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True, timeout=5.0)
        try:
            return index_db if _table_exists(conn, "sessions") else None
        finally:
            conn.close()
    except Exception:
        logger.warning("archive convergence: failed to inspect archive", exc_info=True)
        return None


def _schema_archive_session_ids_for_source_path(conn: sqlite3.Connection, path: Path) -> list[str]:
    return _schema_archive_session_ids_for_source_paths(conn, [path]).get(path, [])


def _schema_archive_session_ids_for_source_paths(
    conn: sqlite3.Connection,
    paths: Sequence[Path],
) -> dict[Path, list[str]]:
    normalized_paths = tuple(dict.fromkeys(Path(path) for path in paths))
    if not normalized_paths or not _table_exists(conn, "sessions"):
        return {path: [] for path in normalized_paths}
    raw_table = "raw_sessions"
    if not _table_exists(conn, "raw_sessions"):
        raw_table = "source_tier.raw_sessions"
        try:
            if not _ensure_source_tier_attached(conn):
                return {path: [] for path in normalized_paths}
        except sqlite3.Error:
            return {path: [] for path in normalized_paths}
    result: dict[Path, list[str]] = {path: [] for path in normalized_paths}
    paths_by_text = {str(path): path for path in normalized_paths}
    placeholders = ", ".join("?" for _ in normalized_paths)
    rows = conn.execute(
        f"""
        SELECT DISTINCT r.source_path, s.session_id
        FROM {raw_table} AS r
        JOIN sessions AS s ON s.raw_id = r.raw_id
        WHERE r.source_path IN ({placeholders})
        ORDER BY r.source_path, s.session_id
        """,
        tuple(paths_by_text),
    ).fetchall()
    for source_path, session_id in rows:
        path = paths_by_text.get(str(source_path))
        if path is not None:
            result[path].append(str(session_id))
    return result


def _archive_existing_session_ids(conn: sqlite3.Connection, session_ids: Sequence[str]) -> list[str]:
    unique_ids = tuple(dict.fromkeys(str(session_id) for session_id in session_ids if session_id))
    if not unique_ids or not _table_exists(conn, "sessions"):
        return []
    placeholders = ", ".join("?" for _ in unique_ids)
    rows = conn.execute(
        f"""
        SELECT session_id
        FROM sessions
        WHERE session_id IN ({placeholders})
        ORDER BY session_id
        """,
        unique_ids,
    ).fetchall()
    return [str(row[0]) for row in rows]


def _archive_text_block_count(conn: sqlite3.Connection, session_ids: Sequence[str] | None = None) -> int:
    if not _table_exists(conn, "blocks"):
        return 0
    params: tuple[str, ...] = tuple(dict.fromkeys(str(session_id) for session_id in session_ids or () if session_id))
    filter_sql = ""
    if params:
        placeholders = ", ".join("?" for _ in params)
        filter_sql = f"AND session_id IN ({placeholders})"
    row = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM blocks
        WHERE search_text != ''
          {filter_sql}
        """,
        params,
    ).fetchone()
    return int(row[0] or 0) if row is not None else 0


def _archive_messages_fts_doc_count(conn: sqlite3.Connection) -> int:
    return _fts_doc_count(conn, "messages_fts_docsize")


def _archive_fts_needs_repair(conn: sqlite3.Connection, session_ids: Sequence[str] | None = None) -> bool:
    if not _table_exists(conn, "messages_fts") or not _table_exists(conn, "messages_fts_docsize"):
        return _archive_text_block_count(conn, session_ids) > 0
    ids = tuple(dict.fromkeys(str(session_id) for session_id in session_ids or () if session_id))
    if not ids:
        return _archive_messages_fts_doc_count(conn) != _archive_text_block_count(conn)
    placeholders = ", ".join("?" for _ in ids)
    missing = int(
        conn.execute(
            f"""
            SELECT COUNT(*)
            FROM blocks AS b
            LEFT JOIN messages_fts_docsize AS d ON d.id = b.rowid
            WHERE b.session_id IN ({placeholders})
              AND b.search_text != ''
              AND d.id IS NULL
            """,
            ids,
        ).fetchone()[0]
    )
    return missing > 0


def _archive_rebuild_messages_fts(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "messages_fts"):
        return
    from polylogue.storage.fts.fts_lifecycle import rebuild_fts_index_sync

    rebuild_fts_index_sync(conn)


def _archive_fts_check(db_path: Path, path: Path) -> bool:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        try:
            session_ids = _schema_archive_session_ids_for_source_path(conn, path)
            return _archive_fts_needs_repair(conn, session_ids or None)
        finally:
            conn.close()
    except Exception:
        return False


def _archive_fts_execute(db_path: Path, path: Path) -> bool:
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        try:
            session_ids = _schema_archive_session_ids_for_source_path(conn, path)
            _archive_rebuild_messages_fts(conn)
            conn.commit()
            logger.info("fts: archive rebuilt messages_fts sessions=%d", len(session_ids))
            return not _archive_fts_needs_repair(conn, session_ids or None)
        finally:
            conn.close()
    except Exception:
        logger.warning("fts: archive repair failed", exc_info=True)
        return False


def _archive_fts_check_many(db_path: Path, paths: Sequence[Path]) -> set[Path]:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        try:
            by_path = _schema_archive_session_ids_for_source_paths(conn, paths)
            return {
                path
                for path, session_ids in by_path.items()
                if session_ids and _archive_fts_needs_repair(conn, session_ids)
            }
        finally:
            conn.close()
    except Exception:
        return set()


def _archive_fts_execute_many(db_path: Path, paths: Sequence[Path]) -> bool:
    if not paths:
        return False
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        try:
            by_path = _schema_archive_session_ids_for_source_paths(conn, paths)
            session_ids = list(dict.fromkeys(session_id for ids in by_path.values() for session_id in ids))
            _archive_rebuild_messages_fts(conn)
            conn.commit()
            logger.info("fts: archive batch rebuilt messages_fts paths=%d sessions=%d", len(paths), len(session_ids))
            return not _archive_fts_needs_repair(conn, session_ids or None)
        finally:
            conn.close()
    except Exception:
        logger.warning("fts: archive batch repair failed", exc_info=True)
        return False


def _archive_fts_check_sessions(db_path: Path, session_ids: Sequence[str]) -> set[str]:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        try:
            ids = _archive_existing_session_ids(conn, session_ids)
            return {session_id for session_id in ids if _archive_fts_needs_repair(conn, [session_id])}
        finally:
            conn.close()
    except Exception:
        return set()


def _archive_fts_execute_sessions(db_path: Path, session_ids: Sequence[str]) -> bool:
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        try:
            ids = _archive_existing_session_ids(conn, session_ids)
            if not ids:
                return True
            _archive_rebuild_messages_fts(conn)
            conn.commit()
            logger.info("fts: archive rebuilt messages_fts session debt sessions=%d", len(ids))
            return not _archive_fts_needs_repair(conn, ids)
        finally:
            conn.close()
    except Exception:
        logger.warning("fts: archive session repair failed", exc_info=True)
        return False


def _archive_pending_embedding_session_ids(conn: sqlite3.Connection, session_ids: Sequence[str]) -> list[str]:
    from polylogue.storage.embeddings.materialization import select_pending_archive_session_window

    ids = tuple(dict.fromkeys(str(session_id) for session_id in session_ids if session_id))
    if not ids:
        return []
    return [
        item.session_id
        for item in select_pending_archive_session_window(
            conn,
            session_ids=ids,
            max_sessions=_DAEMON_EMBED_MAX_SESSIONS,
            max_messages=_DAEMON_EMBED_MAX_MESSAGES,
        )
    ]


def _archive_embed_check(db_path: Path, path: Path) -> bool:
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        try:
            _reconcile_embedding_config_change(conn)
            conn.commit()
            session_ids = _schema_archive_session_ids_for_source_path(conn, path)
            return bool(_archive_pending_embedding_session_ids(conn, session_ids))
        finally:
            conn.close()
    except Exception:
        return False


def _archive_embed_execute(db_path: Path, path: Path) -> bool:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        try:
            session_ids = _schema_archive_session_ids_for_source_path(conn, path)
            pending_ids = _archive_pending_embedding_session_ids(conn, session_ids)
        finally:
            conn.close()
        if not pending_ids:
            return True
        return _embed_archive_sessions_sync(db_path, pending_ids) and not _archive_embedding_debt_remaining(
            db_path, session_ids
        )
    except Exception:
        logger.warning("embed: archive failed", exc_info=True)
        return False


def _archive_embed_check_many(db_path: Path, paths: Sequence[Path]) -> set[Path]:
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        try:
            _reconcile_embedding_config_change(conn)
            conn.commit()
            by_path = _schema_archive_session_ids_for_source_paths(conn, paths)
            all_ids = list(dict.fromkeys(session_id for ids in by_path.values() for session_id in ids))
            pending = set(_archive_pending_embedding_session_ids(conn, all_ids))
            return {path for path, ids in by_path.items() if any(session_id in pending for session_id in ids)}
        finally:
            conn.close()
    except Exception:
        return set()


def _archive_embed_execute_many(db_path: Path, paths: Sequence[Path]) -> bool:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        try:
            by_path = _schema_archive_session_ids_for_source_paths(conn, paths)
            session_ids = list(dict.fromkeys(session_id for ids in by_path.values() for session_id in ids))
            pending_ids = _archive_pending_embedding_session_ids(conn, session_ids)
        finally:
            conn.close()
        if not pending_ids:
            return True
        return _embed_archive_sessions_sync(db_path, pending_ids) and not _archive_embedding_debt_remaining(
            db_path, session_ids
        )
    except Exception:
        logger.warning("embed: archive batch failed", exc_info=True)
        return False


def _archive_embed_check_sessions(db_path: Path, session_ids: Sequence[str]) -> set[str]:
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        try:
            _reconcile_embedding_config_change(conn)
            conn.commit()
            ids = _archive_existing_session_ids(conn, session_ids)
            return set(_archive_pending_embedding_session_ids(conn, ids))
        finally:
            conn.close()
    except Exception:
        return set()


def _archive_embed_execute_sessions(db_path: Path, session_ids: Sequence[str]) -> bool:
    ids = tuple(dict.fromkeys(str(session_id) for session_id in session_ids if session_id))
    if not ids:
        return True
    ok = _embed_archive_sessions_sync(db_path, ids)
    return ok and not _archive_embedding_debt_remaining(db_path, ids)


def _archive_embedding_debt_remaining(db_path: Path, session_ids: Sequence[str]) -> bool:
    if not session_ids:
        return False
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        try:
            return bool(_archive_pending_embedding_session_ids(conn, session_ids))
        finally:
            conn.close()
    except Exception:
        logger.warning("embed: archive failed to check remaining debt", exc_info=True)
        return True


def _embed_archive_sessions_sync(db_path: Path, session_ids: Sequence[str]) -> bool:
    from polylogue.storage.embeddings.materialization import embed_archive_session_sync
    from polylogue.storage.search_providers import create_vector_provider

    cfg = load_polylogue_config()
    voyage_key = cfg.get("voyage_api_key")
    if not voyage_key:
        return True
    embeddings_db = db_path.with_name("embeddings.db")
    vec_provider = create_vector_provider(
        voyage_api_key=str(voyage_key),
        db_path=embeddings_db,
        model=cfg.embedding_model,
        dimension=cfg.embedding_dimension,
    )
    if vec_provider is None:
        logger.warning("embed: archive vector provider unavailable")
        return False

    errors = 0
    embedded = 0
    started_at = time.monotonic()
    for session_id in tuple(dict.fromkeys(session_ids)):
        if time.monotonic() - started_at >= _DAEMON_EMBED_STOP_AFTER_SECONDS:
            break
        outcome = embed_archive_session_sync(db_path, vec_provider, session_id)
        if outcome.status == "embedded":
            embedded += 1
        elif outcome.status in {"no_messages", "no_embeddable_messages"}:
            logger.info("embed: archive %s has no embeddable messages", session_id)
        elif outcome.status == "error":
            errors += 1
            logger.warning("embed: archive %s failed: %s", outcome.session_id, outcome.error)
            if errors >= _DAEMON_EMBED_MAX_ERRORS:
                break
    logger.info("embed: archive %d done, %d errors", embedded, errors)
    return errors == 0


def _archive_hot_insight_session_ids(
    conn: sqlite3.Connection,
    session_ids: Sequence[str],
    *,
    now: float | None = None,
) -> set[str]:
    unique_ids = tuple(dict.fromkeys(str(session_id) for session_id in session_ids if session_id))
    if not unique_ids or not _table_exists(conn, "sessions"):
        return set()
    raw_table = "raw_sessions"
    if not _table_exists(conn, "raw_sessions"):
        raw_table = "source_tier.raw_sessions"
        try:
            if not _ensure_source_tier_attached(conn):
                return set()
        except sqlite3.Error:
            return set()
    placeholders = ", ".join("?" for _ in unique_ids)
    rows = conn.execute(
        f"""
        SELECT DISTINCT s.session_id, r.source_path
        FROM sessions AS s
        JOIN {raw_table} AS r ON r.raw_id = s.raw_id
        WHERE s.session_id IN ({placeholders})
          AND r.source_path IS NOT NULL
          AND r.source_path != ''
        ORDER BY s.session_id
        """,
        unique_ids,
    ).fetchall()
    current = time.time() if now is None else now
    return {
        str(session_id)
        for session_id, source_path in rows
        if _source_path_is_hot_for_insights(Path(str(source_path)), now=current)
    }


def _archive_stale_session_profile_ids(conn: sqlite3.Connection, session_ids: Sequence[str]) -> list[str]:
    unique_ids = tuple(dict.fromkeys(str(session_id) for session_id in session_ids if session_id))
    if not unique_ids or not _table_exists(conn, "sessions") or not _table_exists(conn, "session_profiles"):
        return []
    placeholders = ", ".join("?" for _ in unique_ids)
    rows = conn.execute(
        f"""
        SELECT s.session_id
        FROM sessions AS s
        LEFT JOIN session_profiles AS sp ON sp.session_id = s.session_id
        WHERE s.session_id IN ({placeholders})
          AND (
              sp.session_id IS NULL
              OR sp.materializer_version != ?
              OR (
                  s.sort_key_ms IS NOT NULL
                  AND ABS(COALESCE(sp.source_sort_key, 0.0) - (CAST(s.sort_key_ms AS REAL) / 1000.0)) > 0.000001
              )
              OR (
                  s.sort_key_ms IS NULL
                  AND COALESCE(strftime('%s', sp.source_updated_at), sp.source_updated_at, '') !=
                      COALESCE(CAST(s.updated_at_ms / 1000 AS TEXT), '')
              )
          )
        ORDER BY s.session_id
        """,
        unique_ids + (SESSION_INSIGHT_MATERIALIZER_VERSION,),
    ).fetchall()
    return [str(row[0]) for row in rows]


def _schema_archive_session_ids_missing_profiles(conn: sqlite3.Connection) -> list[str]:
    if not _table_exists(conn, "sessions"):
        return []
    rows = conn.execute("SELECT session_id FROM sessions ORDER BY session_id").fetchall()
    return _archive_stale_session_profile_ids(conn, [str(row[0]) for row in rows])


def _archive_insights_check(db_path: Path, path: Path) -> bool:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        try:
            session_ids = _schema_archive_session_ids_for_source_path(conn, path)
            if session_ids:
                return bool(_archive_stale_session_profile_ids(conn, session_ids))
            return bool(_schema_archive_session_ids_missing_profiles(conn))
        finally:
            conn.close()
    except Exception:
        return False


def _archive_insights_execute(db_path: Path, path: Path) -> bool:
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        try:
            session_ids = _schema_archive_session_ids_for_source_path(
                conn, path
            ) or _schema_archive_session_ids_missing_profiles(conn)
            return _archive_insights_execute_ids(conn, session_ids)
        finally:
            conn.close()
    except Exception:
        logger.warning("insights: archive refresh failed", exc_info=True)
        return False


def _archive_insights_check_many(db_path: Path, paths: Sequence[Path]) -> set[Path]:
    if not paths:
        return set()
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        try:
            by_path = _schema_archive_session_ids_for_source_paths(conn, paths)
            result = {
                path
                for path, session_ids in by_path.items()
                if session_ids and _archive_stale_session_profile_ids(conn, session_ids)
            }
            if result:
                return result
            return {Path(paths[0])} if _schema_archive_session_ids_missing_profiles(conn) else set()
        finally:
            conn.close()
    except Exception:
        return set()


def _archive_insights_execute_many(db_path: Path, paths: Sequence[Path]) -> bool:
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        try:
            by_path = _schema_archive_session_ids_for_source_paths(conn, paths)
            session_ids = list(dict.fromkeys(session_id for ids in by_path.values() for session_id in ids))
            if not session_ids:
                session_ids = _schema_archive_session_ids_missing_profiles(conn)
            return _archive_insights_execute_ids(conn, session_ids)
        finally:
            conn.close()
    except Exception:
        logger.warning("insights: archive batch refresh failed", exc_info=True)
        return False


def _archive_insights_check_sessions(db_path: Path, session_ids: Sequence[str]) -> set[str]:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        try:
            ids = _archive_existing_session_ids(conn, session_ids)
            return set(_archive_stale_session_profile_ids(conn, ids))
        finally:
            conn.close()
    except Exception:
        return set()


def _archive_insights_execute_sessions(db_path: Path, session_ids: Sequence[str]) -> bool:
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        try:
            ids = _archive_existing_session_ids(conn, session_ids)
            return _archive_insights_execute_ids(conn, ids)
        finally:
            conn.close()
    except Exception:
        logger.warning("insights: archive session refresh failed", exc_info=True)
        return False


def _archive_insights_execute_ids(conn: sqlite3.Connection, session_ids: Sequence[str]) -> bool:
    from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync

    if not session_ids:
        return True
    hot_ids = _archive_hot_insight_session_ids(conn, session_ids)
    if hot_ids:
        logger.info(
            "insights: deferring hot archive source rebuild sessions=%d quiet_s=%.0f",
            len(hot_ids),
            _HOT_INSIGHT_QUIET_SECONDS,
        )
        return False
    # The canonical rebuild function requires row-factory access on the
    # connection (name-based column reads throughout). The archive callers
    # use plain sqlite3.connect() without row_factory, so set it here.
    conn.row_factory = sqlite3.Row
    counts = rebuild_session_insights_sync(
        conn,
        session_ids=list(session_ids),
        page_size=_DAEMON_INSIGHT_REBUILD_PAGE_SIZE,
    )
    # rebuild_session_insights_sync commits internally when session_ids is
    # not None; no explicit conn.commit() needed here.
    remaining = _archive_stale_session_profile_ids(conn, list(session_ids))
    logger.info(
        "insights: archive refreshed sessions=%d profiles=%d work_events=%d phases=%d threads=%d remaining=%d",
        len(tuple(dict.fromkeys(session_ids))),
        counts.profiles,
        counts.work_events,
        counts.phases,
        counts.threads,
        len(remaining),
    )
    return not remaining


__all__ = [
    "make_default_convergence_stages",
    "make_embed_stage",
    "make_fts_stage",
    "make_insights_stage",
]
