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
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.config import load_polylogue_config
from polylogue.daemon.convergence import ConvergenceStage
from polylogue.logging import get_logger
from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION
from polylogue.storage.source_conversations import (
    conversation_ids_for_source_path,
    conversation_ids_for_source_paths,
)

if TYPE_CHECKING:
    from polylogue.storage.embeddings.materialization import PendingConversation

logger = get_logger(__name__)

_DAEMON_INSIGHT_REBUILD_PAGE_SIZE = 10
_HOT_INSIGHT_SOURCE_BYTES = 64 * 1024 * 1024
_HOT_INSIGHT_QUIET_SECONDS = 60.0
_DAEMON_EMBED_MAX_CONVERSATIONS = 25
_DAEMON_EMBED_MAX_MESSAGES = 2_500
_DAEMON_EMBED_STOP_AFTER_SECONDS = 30
_DAEMON_EMBED_MAX_ERRORS = 3


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

    def check(path: Path) -> bool:
        if not db_path.exists():
            return False
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=5.0)
            try:
                conversation_ids = _conversation_ids_for_source_path(conn, path)
                if conversation_ids:
                    return _fts_needs_repair_for_conversations(conn, conversation_ids)
                from polylogue.storage.fts.sql import FTS_INDEXABLE_MESSAGE_COUNT_SQL

                total = int(conn.execute(FTS_INDEXABLE_MESSAGE_COUNT_SQL).fetchone()[0])
                fts_count = _fts_doc_count(conn, "messages_fts_docsize")
                if fts_count != total:
                    return True
                if total == 0:
                    return False
                if _table_exists(conn, "action_events") and _table_exists(conn, "action_events_fts_docsize"):
                    action_total = int(conn.execute("SELECT COUNT(*) FROM action_events").fetchone()[0])
                    action_fts_count = _fts_doc_count(conn, "action_events_fts_docsize")
                    return action_fts_count != action_total
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
                    _mark_message_fts_ready_after_targeted_repair(conn)
                    conn.commit()
                    logger.info("fts: repaired conversations=%d", len(conversation_ids))
                    return not _fts_needs_repair_for_conversations(conn, conversation_ids)
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
                _mark_message_fts_ready_after_targeted_repair(conn)
                conn.commit()
                logger.info("fts: batch repaired paths=%d conversations=%d", len(paths), len(conversation_ids))
                return not _fts_needs_repair_for_conversations(conn, conversation_ids)
            finally:
                conn.close()
        except Exception:
            logger.warning("fts: batch repair failed", exc_info=True)
            return False

    def check_conversations(conversation_ids: Sequence[str]) -> set[str]:
        if not conversation_ids or not db_path.exists():
            return set()
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=5.0)
            try:
                return {
                    conversation_id
                    for conversation_id in dict.fromkeys(conversation_ids)
                    if _fts_repair_needs_for_conversations(conn, [conversation_id]).any
                }
            finally:
                conn.close()
        except Exception:
            return set()

    def execute_conversations(conversation_ids: Sequence[str]) -> bool:
        if not conversation_ids:
            return True
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=30.0)
            try:
                ids = tuple(dict.fromkeys(conversation_ids))
                needs = _fts_repair_needs_for_conversations(conn, ids)
                _repair_changed_conversation_fts(conn, ids, needs=needs)
                _mark_message_fts_ready_after_targeted_repair(conn)
                conn.commit()
                logger.info("fts: repaired conversation debt conversations=%d", len(ids))
                return not _fts_needs_repair_for_conversations(conn, ids)
            finally:
                conn.close()
        except Exception:
            logger.warning("fts: conversation repair failed", exc_info=True)
            return False

    return ConvergenceStage(
        name="fts",
        description="Verify FTS coverage and repair gaps",
        check=check,
        execute=execute,
        check_many=check_many,
        execute_many=execute_many,
        check_conversations=check_conversations,
        execute_conversations=execute_conversations,
        cpu_bound=False,
        false_means_pending=True,
    )


# ── Stage: embed ───────────────────────────────────────────────────


def make_embed_stage(db_path: Path) -> ConvergenceStage:
    """Generate vector embeddings for changed conversations that need them.

    Before embedding, detects model/dimension config changes and marks
    affected rows for reindex. Enforces the configured cost cap during
    embedding.
    """

    def check(path: Path) -> bool:
        if not _embedding_config_enabled():
            return False
        if not db_path.exists():
            return False
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=5.0)
            try:
                _reconcile_embedding_config_change(conn)
                conn.commit()
                conversation_ids = _conversation_ids_for_source_path(conn, path)
                return bool(_pending_embedding_conversation_ids(conn, conversation_ids))
            finally:
                conn.close()
        except Exception:
            return False

    def execute(path: Path) -> bool:
        if not _embedding_config_enabled():
            return True
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=5.0)
            try:
                source_conversation_ids = _conversation_ids_for_source_path(conn, path)
                conversation_ids = _pending_embedding_conversation_ids(conn, source_conversation_ids)
            finally:
                conn.close()
            if not conversation_ids:
                return True
            ok = _embed_conversations_sync(
                db_path,
                conversation_ids,
                max_errors=_DAEMON_EMBED_MAX_ERRORS,
                stop_after_seconds=_DAEMON_EMBED_STOP_AFTER_SECONDS,
            )
            return ok and not _embedding_debt_remaining_for_conversations(db_path, source_conversation_ids)
        except Exception:
            logger.warning("embed: failed", exc_info=True)
            return False

    def check_many(paths: Sequence[Path]) -> set[Path]:
        if not paths or not _embedding_config_enabled() or not db_path.exists():
            return set()
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=5.0)
            try:
                _reconcile_embedding_config_change(conn)
                conn.commit()
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
        if not paths or not _embedding_config_enabled():
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
            ok = _embed_conversations_sync(
                db_path,
                pending_ids,
                max_errors=_DAEMON_EMBED_MAX_ERRORS,
                stop_after_seconds=_DAEMON_EMBED_STOP_AFTER_SECONDS,
            )
            return ok and not _embedding_debt_remaining_for_conversations(db_path, conversation_ids)
        except Exception:
            logger.warning("embed: batch failed", exc_info=True)
            return False

    def check_conversations(conversation_ids: Sequence[str]) -> set[str]:
        if not conversation_ids or not _embedding_config_enabled() or not db_path.exists():
            return set()
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=5.0)
            try:
                _reconcile_embedding_config_change(conn)
                conn.commit()
                return set(_pending_embedding_conversation_ids(conn, tuple(dict.fromkeys(conversation_ids))))
            finally:
                conn.close()
        except Exception:
            return set()

    def execute_conversations(conversation_ids: Sequence[str]) -> bool:
        if not conversation_ids or not _embedding_config_enabled():
            return True
        ids = tuple(dict.fromkeys(conversation_ids))
        ok = _embed_conversations_sync(
            db_path,
            ids,
            max_errors=_DAEMON_EMBED_MAX_ERRORS,
            stop_after_seconds=_DAEMON_EMBED_STOP_AFTER_SECONDS,
        )
        return ok and not _embedding_debt_remaining_for_conversations(db_path, ids)

    return ConvergenceStage(
        name="embed",
        description="Generate vector embeddings for changed conversations",
        check=check,
        execute=execute,
        check_many=check_many,
        execute_many=execute_many,
        check_conversations=check_conversations,
        execute_conversations=execute_conversations,
        cpu_bound=False,
        false_means_pending=True,
    )


# ── Stage: insights ────────────────────────────────────────────────


def make_insights_stage(db_path: Path) -> ConvergenceStage:
    """Refresh session insights for conversations missing them."""

    def check(path: Path) -> bool:
        if not db_path.exists():
            return False
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=5.0)
            try:
                if not _table_exists(conn, "session_profiles"):
                    return False
                conversation_ids = _conversation_ids_for_source_path(conn, path)
                if conversation_ids:
                    return bool(_stale_session_profile_ids(conn, conversation_ids))
                total_conv = int(conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0])
                if total_conv == 0:
                    return False
                profiled = int(conn.execute("SELECT COUNT(*) FROM session_profiles").fetchone()[0])
                return profiled < total_conv
            finally:
                conn.close()
        except Exception:
            return False

    def execute(path: Path) -> bool:
        from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync
        from polylogue.storage.sqlite.connection import open_connection

        try:
            with open_connection(db_path) as conn:
                conversation_ids = _conversation_ids_for_source_path(conn, path) or _conversation_ids_missing_profiles(
                    conn
                )
                hot_ids = _hot_insight_conversation_ids(conn, conversation_ids)
                if hot_ids:
                    logger.info(
                        "insights: deferring hot source rebuild conversations=%d quiet_s=%.0f",
                        len(hot_ids),
                        _HOT_INSIGHT_QUIET_SECONDS,
                    )
                    return False
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
                paths_with_conversations = {
                    path
                    for path, conversation_ids in by_path.items()
                    if conversation_ids and _stale_session_profile_ids(conn, conversation_ids)
                }
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
                hot_ids = _hot_insight_conversation_ids(conn, conversation_ids)
                if hot_ids:
                    logger.info(
                        "insights: deferring hot source batch rebuild conversations=%d quiet_s=%.0f",
                        len(hot_ids),
                        _HOT_INSIGHT_QUIET_SECONDS,
                    )
                    return False
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

    def check_conversations(conversation_ids: Sequence[str]) -> set[str]:
        if not conversation_ids or not db_path.exists():
            return set()
        from polylogue.storage.sqlite.connection_profile import open_connection

        try:
            conn = open_connection(db_path, timeout=5.0)
            try:
                if not _table_exists(conn, "session_profiles"):
                    return set()
                return set(_stale_session_profile_ids(conn, tuple(dict.fromkeys(conversation_ids))))
            finally:
                conn.close()
        except Exception:
            return set()

    def execute_conversations(conversation_ids: Sequence[str]) -> bool:
        from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync
        from polylogue.storage.sqlite.connection import open_connection

        try:
            with open_connection(db_path) as conn:
                ids = _existing_conversation_ids(conn, tuple(dict.fromkeys(conversation_ids)))
                if not ids:
                    return True
                hot_ids = _hot_insight_conversation_ids(conn, ids)
                if hot_ids:
                    logger.info(
                        "insights: deferring hot source conversation rebuild conversations=%d quiet_s=%.0f",
                        len(hot_ids),
                        _HOT_INSIGHT_QUIET_SECONDS,
                    )
                    return False
                counts = rebuild_session_insights_sync(
                    conn,
                    conversation_ids=ids,
                    page_size=_DAEMON_INSIGHT_REBUILD_PAGE_SIZE,
                )
                conn.commit()
                remaining = _stale_session_profile_ids(conn, ids)
                logger.info(
                    "insights: refreshed conversation debt conversations=%d profiles=%d work_events=%d phases=%d threads=%d remaining=%d",
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
            logger.warning("insights: conversation rebuild failed", exc_info=True)
            return False

    return ConvergenceStage(
        name="insights",
        description="Refresh session insights for new conversations",
        check=check,
        execute=execute,
        check_many=check_many,
        execute_many=execute_many,
        check_conversations=check_conversations,
        execute_conversations=execute_conversations,
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


def _conversation_ids_for_source_path(conn: sqlite3.Connection, path: Path) -> list[str]:
    return conversation_ids_for_source_path(conn, path)


def _conversation_ids_for_source_paths(
    conn: sqlite3.Connection,
    paths: Sequence[Path],
) -> dict[Path, list[str]]:
    return conversation_ids_for_source_paths(conn, paths)


def _fts_repair_needs_for_conversations(
    conn: sqlite3.Connection,
    conversation_ids: Sequence[str],
) -> _FtsRepairNeeds:
    if not conversation_ids:
        return _FtsRepairNeeds()
    if not _table_exists(conn, "messages_fts_docsize"):
        return _FtsRepairNeeds(messages=True)
    messages_missing = False
    placeholders = ", ".join("?" for _ in conversation_ids)
    params = tuple(conversation_ids)
    missing_messages = int(
        conn.execute(
            f"""
            SELECT COUNT(*)
            FROM messages AS m
            LEFT JOIN messages_fts_docsize AS d ON d.id = m.rowid
            WHERE m.conversation_id IN ({placeholders})
              AND d.id IS NULL
              AND (
                  NULLIF(m.text, '') IS NOT NULL
                  OR EXISTS (
                      SELECT 1
                      FROM content_blocks AS cb
                      WHERE cb.message_id = m.message_id
                        AND (
                            NULLIF(cb.text, '') IS NOT NULL
                            OR NULLIF(cb.tool_input, '') IS NOT NULL
                            OR NULLIF(cb.metadata, '') IS NOT NULL
                        )
                  )
              )
            """,
            params,
        ).fetchone()[0]
    )
    messages_missing = missing_messages > 0
    if not _table_exists(conn, "action_events") or not _table_exists(conn, "action_events_fts"):
        return _FtsRepairNeeds(messages=messages_missing)
    if not _table_exists(conn, "action_events_fts_docsize"):
        return _FtsRepairNeeds(messages=messages_missing, actions=True)
    missing_actions = int(
        conn.execute(
            f"""
            SELECT COUNT(*)
            FROM action_events AS ae
            LEFT JOIN action_events_fts_docsize AS d ON d.id = ae.rowid
            WHERE ae.conversation_id IN ({placeholders})
              AND d.id IS NULL
            """,
            params,
        ).fetchone()[0]
    )
    return _FtsRepairNeeds(messages=messages_missing, actions=missing_actions > 0)


def _fts_needs_repair_for_conversations(conn: sqlite3.Connection, conversation_ids: Sequence[str]) -> bool:
    return _fts_repair_needs_for_conversations(conn, conversation_ids).any


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


def _pending_embedding_conversation_ids(conn: sqlite3.Connection, conversation_ids: Sequence[str]) -> list[str]:
    return [item.conversation_id for item in _pending_embedding_conversation_window(conn, conversation_ids)]


def _pending_embedding_conversation_window(
    conn: sqlite3.Connection,
    conversation_ids: Sequence[str],
) -> list[PendingConversation]:
    unique_ids = tuple(dict.fromkeys(conversation_ids))
    if not unique_ids:
        return []
    from polylogue.storage.embeddings.materialization import PendingConversation, select_pending_conversation_window

    if not _table_exists(conn, "conversations"):
        return [PendingConversation(conversation_id=conversation_id) for conversation_id in unique_ids]

    return select_pending_conversation_window(
        conn,
        conversation_ids=unique_ids,
        max_conversations=_DAEMON_EMBED_MAX_CONVERSATIONS,
        max_messages=_DAEMON_EMBED_MAX_MESSAGES,
    )


def _embedding_debt_remaining_for_conversations(db_path: Path, conversation_ids: Sequence[str]) -> bool:
    if not conversation_ids:
        return False
    from polylogue.storage.sqlite.connection_profile import open_connection

    try:
        conn = open_connection(db_path, timeout=5.0)
        try:
            return bool(_pending_embedding_conversation_ids(conn, tuple(dict.fromkeys(conversation_ids))))
        finally:
            conn.close()
    except Exception:
        logger.warning("embed: failed to check remaining debt", exc_info=True)
        return True


def _embed_conversations_sync(
    db_path: Path,
    conversation_ids: Sequence[str],
    *,
    max_errors: int | None = None,
    stop_after_seconds: int | None = None,
    max_cost_usd: float | None = None,
) -> bool:
    from contextlib import closing

    from polylogue.api.sync.bridge import run_coroutine_sync
    from polylogue.storage.embeddings.materialization import embed_conversation_sync
    from polylogue.storage.embeddings.progress import (
        CatchupRunDelta,
        CatchupRunStart,
        finish_embedding_catchup_run,
        record_embedding_catchup_progress,
        start_embedding_catchup_run,
    )
    from polylogue.storage.repository import ConversationRepository
    from polylogue.storage.search_providers import create_vector_provider
    from polylogue.storage.search_providers.sqlite_vec_support import (
        ESTIMATED_TOKENS_PER_MESSAGE,
        VOYAGE_4_COST_PER_1M_TOKENS,
    )
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    cfg = load_polylogue_config()
    voyage_key = cfg.get("voyage_api_key")
    if not voyage_key:
        return True

    with closing(open_readonly_connection(db_path, timeout=5.0)) as conn:
        pending = _pending_embedding_conversation_window(conn, conversation_ids)
    if not pending:
        return True

    configured_max_cost = float(str(cfg.get("embedding_max_cost_usd", 0.0)))
    max_cost = configured_max_cost if max_cost_usd is None else max_cost_usd
    model = cfg.embedding_model
    dimension = cfg.embedding_dimension
    planned_messages = sum(item.message_count for item in pending)
    run_id = start_embedding_catchup_run(
        db_path,
        CatchupRunStart(
            rebuild=False,
            max_conversations=_DAEMON_EMBED_MAX_CONVERSATIONS,
            max_messages=_DAEMON_EMBED_MAX_MESSAGES,
            stop_after_seconds=stop_after_seconds,
            max_errors=max_errors,
            planned_conversations=len(pending),
            planned_messages=planned_messages,
        ),
    )

    vec_provider = create_vector_provider(
        voyage_api_key=str(voyage_key), db_path=db_path, model=model, dimension=dimension
    )
    if vec_provider is None:
        logger.warning("embed: vector provider unavailable")
        finish_embedding_catchup_run(db_path, run_id, status="failed", stop_reason="vector provider unavailable")
        return False

    repo = ConversationRepository(backend=SQLiteBackend(db_path=db_path))
    errors = 0
    embedded = 0
    cumulative_cost = 0.0
    stop_reason: str | None = None
    started_at = time.monotonic()
    try:
        for item in pending:
            conversation_id = item.conversation_id
            if stop_after_seconds is not None and time.monotonic() - started_at >= stop_after_seconds:
                stop_reason = f"stop-after-seconds reached ({stop_after_seconds})"
                logger.info("embed: %s", stop_reason)
                break
            outcome = embed_conversation_sync(repo, vec_provider, conversation_id)
            if outcome.status == "embedded":
                embedded += 1
                batch_cost = (
                    outcome.embedded_message_count
                    * ESTIMATED_TOKENS_PER_MESSAGE
                    * VOYAGE_4_COST_PER_1M_TOKENS
                    / 1_000_000
                )
                cumulative_cost += batch_cost
                record_embedding_catchup_progress(
                    db_path,
                    run_id,
                    CatchupRunDelta(
                        conversation_id=outcome.conversation_id,
                        embedded=True,
                        embedded_messages=outcome.embedded_message_count,
                        estimated_cost_usd=batch_cost,
                    ),
                )
                if max_cost > 0.0 and cumulative_cost > max_cost:
                    stop_reason = f"cost cap reached ({cumulative_cost:.4f} > {max_cost:.2f})"
                    logger.info(
                        "embed: %s — stopping after %d conversations",
                        stop_reason,
                        embedded,
                    )
                    break
            elif outcome.status in {"no_messages", "no_embeddable_messages"}:
                logger.info("embed: %s has no embeddable messages", conversation_id)
                record_embedding_catchup_progress(
                    db_path,
                    run_id,
                    CatchupRunDelta(conversation_id=outcome.conversation_id, skipped=True),
                )
            elif outcome.status == "error":
                errors += 1
                logger.warning("embed: %s failed: %s", conversation_id, outcome.error)
                record_embedding_catchup_progress(
                    db_path,
                    run_id,
                    CatchupRunDelta(conversation_id=outcome.conversation_id, errored=True),
                )
                if max_errors is not None and errors >= max_errors:
                    stop_reason = f"max errors reached ({max_errors})"
                    logger.info("embed: %s", stop_reason)
                    break
        logger.info("embed: %d done, %d errors, est. cost $%.4f", embedded, errors, cumulative_cost)
        if stop_reason is not None:
            finish_embedding_catchup_run(db_path, run_id, status="stopped", stop_reason=stop_reason)
        else:
            finish_embedding_catchup_run(db_path, run_id, status="completed", stop_reason=None)
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
        if _action_events_exist_for_conversations(conn, conversation_ids):
            repair_action_fts_index_sync(conn, conversation_ids)
        else:
            insert_missing_action_fts_index_sync(conn, conversation_ids)


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
            "targeted changed-conversation repair complete"
            if bool(readiness["ready"])
            else "targeted changed-conversation repair left structural FTS readiness false"
        ),
    )


def _action_events_exist_for_conversations(conn: sqlite3.Connection, conversation_ids: Sequence[str]) -> bool:
    if not conversation_ids or not _table_exists(conn, "action_events"):
        return False
    placeholders = ", ".join("?" for _ in conversation_ids)
    row = conn.execute(
        f"""
        SELECT 1
        FROM action_events
        WHERE conversation_id IN ({placeholders})
        LIMIT 1
        """,
        tuple(conversation_ids),
    ).fetchone()
    return row is not None


def _conversation_ids_missing_profiles(conn: sqlite3.Connection) -> list[str]:
    """Conversations whose session_profile is missing or stale (#1620)."""
    from polylogue.storage.insights.session.status import SESSION_PROFILE_REPAIR_CANDIDATES_SQL
    from polylogue.storage.runtime.store_constants import SESSION_INSIGHT_MATERIALIZER_VERSION

    rows = conn.execute(SESSION_PROFILE_REPAIR_CANDIDATES_SQL, (SESSION_INSIGHT_MATERIALIZER_VERSION,)).fetchall()
    return [str(row[0]) for row in rows]


def _existing_conversation_ids(conn: sqlite3.Connection, conversation_ids: Sequence[str]) -> list[str]:
    unique_ids = tuple(dict.fromkeys(conversation_ids))
    if not unique_ids or not _table_exists(conn, "conversations"):
        return []
    placeholders = ", ".join("?" for _ in unique_ids)
    rows = conn.execute(
        f"""
        SELECT conversation_id
        FROM conversations
        WHERE conversation_id IN ({placeholders})
        ORDER BY conversation_id
        """,
        unique_ids,
    ).fetchall()
    return [str(row[0]) for row in rows]


def _hot_insight_conversation_ids(
    conn: sqlite3.Connection,
    conversation_ids: Sequence[str],
    *,
    now: float | None = None,
) -> set[str]:
    """Return stale conversations whose source file is too hot for full insight rebuild.

    Live archive writes and targeted FTS repair must stay immediate. Session
    insight rebuilds can require rehydrating an entire conversation; for huge
    actively-appending agent sessions that turns every small append into a
    multi-GB read cycle. Returning False from the stage records durable
    convergence debt, so this is a quiet-window deferral, not a scope reduction.
    """

    unique_ids = tuple(dict.fromkeys(str(conversation_id) for conversation_id in conversation_ids if conversation_id))
    if not unique_ids or not _table_exists(conn, "conversations") or not _table_exists(conn, "raw_conversations"):
        return set()
    placeholders = ", ".join("?" for _ in unique_ids)
    rows = conn.execute(
        f"""
        SELECT DISTINCT c.conversation_id, r.source_path
        FROM conversations AS c
        JOIN raw_conversations AS r ON r.raw_id = c.raw_id
        WHERE c.conversation_id IN ({placeholders})
          AND r.source_path IS NOT NULL
          AND r.source_path != ''
        ORDER BY c.conversation_id
        """,
        unique_ids,
    ).fetchall()
    current = time.time() if now is None else now
    hot: set[str] = set()
    for conversation_id, source_path in rows:
        if _source_path_is_hot_for_insights(Path(str(source_path)), now=current):
            hot.add(str(conversation_id))
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


def _stale_session_profile_ids(conn: sqlite3.Connection, conversation_ids: Sequence[str]) -> list[str]:
    unique_ids = tuple(dict.fromkeys(str(conversation_id) for conversation_id in conversation_ids if conversation_id))
    if not unique_ids or not _table_exists(conn, "conversations") or not _table_exists(conn, "session_profiles"):
        return []
    placeholders = ", ".join("?" for _ in unique_ids)
    rows = conn.execute(
        f"""
        SELECT c.conversation_id
        FROM conversations AS c
        LEFT JOIN session_profiles AS sp ON sp.conversation_id = c.conversation_id
        WHERE c.conversation_id IN ({placeholders})
          AND (
              sp.conversation_id IS NULL
              OR sp.materializer_version != ?
              OR (
                  c.sort_key IS NOT NULL
                  AND ABS(COALESCE(sp.source_sort_key, 0.0) - c.sort_key) > 0.000001
              )
              OR (
                  c.sort_key IS NULL
                  AND COALESCE(strftime('%s', sp.source_updated_at), sp.source_updated_at, '') !=
                      COALESCE(strftime('%s', c.updated_at), c.updated_at, '')
              )
          )
        ORDER BY c.conversation_id
        """,
        unique_ids + (SESSION_INSIGHT_MATERIALIZER_VERSION,),
    ).fetchall()
    return [str(row[0]) for row in rows]


__all__ = [
    "make_default_convergence_stages",
    "make_embed_stage",
    "make_fts_stage",
    "make_insights_stage",
]
