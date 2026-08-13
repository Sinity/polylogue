"""Convergence stage implementations for the daemon pipeline.

Each stage has a ``check`` that inspects current archive state and an
``execute`` that performs the missing work. The live watcher owns source
ingestion through daemon-side raw-record ingest; daemon convergence stages only
repair and refresh post-ingest archive state.

- fts: retry explicit session/global FTS debt; source-path foreground checks
  stay cheap because archive writes already repair newly changed rows
- embed: optional vectorization for changed sessions
- insights: refresh session profiles
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.config import load_polylogue_config
from polylogue.core.enums import Provider
from polylogue.core.raw_failure_evidence import (
    RAW_FAILURE_DEFERRED_SUPPORT_STATUS,
    RAW_FAILURE_REPLAY_AUTHORITY_EVIDENCE_KINDS,
)
from polylogue.daemon.convergence import ConvergenceStage, StageExecuteReturn, StageExecutionResult
from polylogue.daemon.convergence_standing_queries import make_standing_query_stage
from polylogue.logging import get_logger
from polylogue.operations.raw_authority_verdict_cache import (
    RawAuthorityVerdictCacheWork,
    find_raw_authority_verdict_cache_work,
    warm_raw_authority_verdict_cache,
)
from polylogue.sources.origin_specs import artifact_rule_for_path
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.insights.session.runtime import session_profile_stale_predicate
from polylogue.storage.introspection import table_exists as _table_exists
from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION
from polylogue.storage.source_sessions import (
    session_ids_for_source_path,
    session_ids_for_source_paths,
)
from polylogue.storage.sqlite.connection_profile import open_daemon_connection

if TYPE_CHECKING:
    from polylogue.sinex.service import PublicationService
    from polylogue.sinex.transport import SinexTransport

logger = get_logger(__name__)

_DAEMON_INSIGHT_REBUILD_PAGE_SIZE = 10
_HOT_INSIGHT_SOURCE_BYTES = 64 * 1024 * 1024
_HOT_INSIGHT_QUIET_SECONDS = 60.0
_DAEMON_EMBED_MAX_SESSIONS = 25
_DAEMON_EMBED_MAX_MESSAGES = 2_500
_DAEMON_EMBED_STOP_AFTER_SECONDS = 30
_DAEMON_EMBED_MAX_ERRORS = 3
_ARCHIVE_INSIGHT_WRITE_BUSY_TIMEOUT_MS = 120_000
_DAEMON_RAW_AUTHORITY_CACHE_MAX_COHORTS = 8


def _is_transient_sqlite_lock(exc: BaseException) -> bool:
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    message = str(exc).lower()
    return "database is locked" in message or "database table is locked" in message or "database is busy" in message


@dataclass(frozen=True, slots=True)
class FtsSurfaceRepairResult:
    """Outcome of one persisted FTS-surface repair attempt.

    A busy SQLite writer is deliberate backpressure and remains retryable. A
    repair exception or an exact-parity failure is a genuine failed attempt.
    Keeping that distinction beside the FTS repair route prevents the debt
    drain from reducing both outcomes to the same boolean.
    """

    success: bool
    deferred: bool = False
    detail: str | None = None

    def __bool__(self) -> bool:
        return self.success


def _open_archive_insight_write_connection(db_path: Path) -> sqlite3.Connection:
    conn = open_daemon_connection(db_path, timeout=_ARCHIVE_INSIGHT_WRITE_BUSY_TIMEOUT_MS / 1000)
    try:
        conn.execute(f"PRAGMA busy_timeout = {_ARCHIVE_INSIGHT_WRITE_BUSY_TIMEOUT_MS}")
    except BaseException:
        conn.close()
        raise
    return conn


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
            logger.warning("convergence freshness probe %s errored; treating as needs-work", "check", exc_info=True)
            return True

    def execute(path: Path) -> StageExecuteReturn:
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
            logger.warning(
                "convergence freshness probe %s errored; treating as needs-work", "check_many", exc_info=True
            )
            return set(paths)

    def execute_many(paths: Sequence[Path]) -> StageExecuteReturn:
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
            logger.warning(
                "convergence freshness probe %s errored; treating as needs-work", "check_sessions", exc_info=True
            )
            return set(session_ids)

    def execute_sessions(session_ids: Sequence[str]) -> StageExecuteReturn:
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
        false_means_pending=True,
    )


# ── Stage: embed ───────────────────────────────────────────────────


def make_embed_stage(db_path: Path, *, defer: Callable[[], bool] | None = None) -> ConvergenceStage:
    """Generate vector embeddings for changed sessions that need them.

    Before embedding, detects model/dimension config changes and marks
    affected rows for reindex. Enforces the configured cost cap during
    embedding.

    ``defer`` returning True (e.g. while watcher source catch-up is still
    running) converts every execute into a ``false_means_pending`` deferral:
    the work lands in convergence debt for the post-catch-up retry loops
    instead of paying serial network embedding inside each bulk ingest
    chunk. Embeddings are a rebuildable read-model — bulk backfill ordering
    is sessions first, vectors when quiet.
    """

    def _deferred() -> bool:
        if defer is not None and defer():
            logger.debug("embed: deferred to convergence debt (source catch-up in progress)")
            return True
        return False

    def check(path: Path) -> bool:
        if not _embedding_config_enabled():
            return False
        archive_db = _active_archive_index_path(db_path)
        return _archive_embed_check(archive_db, path, archive_root=db_path.parent) if archive_db is not None else False

    def execute(path: Path) -> StageExecuteReturn:
        if not _embedding_config_enabled():
            return True
        if _deferred():
            return False
        archive_db = _active_archive_index_path(db_path)
        return _archive_embed_execute(archive_db, path, archive_root=db_path.parent) if archive_db is not None else True

    def check_many(paths: Sequence[Path]) -> set[Path]:
        if not paths or not _embedding_config_enabled():
            return set()
        archive_db = _active_archive_index_path(db_path)
        return (
            _archive_embed_check_many(archive_db, paths, archive_root=db_path.parent)
            if archive_db is not None
            else set()
        )

    def execute_many(paths: Sequence[Path]) -> StageExecuteReturn:
        if not paths or not _embedding_config_enabled():
            return True
        if _deferred():
            return False
        archive_db = _active_archive_index_path(db_path)
        return (
            _archive_embed_execute_many(archive_db, paths, archive_root=db_path.parent)
            if archive_db is not None
            else True
        )

    def check_sessions(session_ids: Sequence[str]) -> set[str]:
        if not session_ids or not _embedding_config_enabled():
            return set()
        archive_db = _active_archive_index_path(db_path)
        return (
            _archive_embed_check_sessions(archive_db, session_ids, archive_root=db_path.parent)
            if archive_db is not None
            else set()
        )

    def execute_sessions(session_ids: Sequence[str]) -> StageExecuteReturn:
        if not session_ids or not _embedding_config_enabled():
            return True
        if _deferred():
            return False
        archive_db = _active_archive_index_path(db_path)
        return (
            _archive_embed_execute_sessions(archive_db, session_ids, archive_root=db_path.parent)
            if archive_db is not None
            else True
        )

    return ConvergenceStage(
        name="embed",
        description="Generate vector embeddings for changed sessions",
        check=check,
        execute=execute,
        check_many=check_many,
        execute_many=execute_many,
        check_sessions=check_sessions,
        execute_sessions=execute_sessions,
        false_means_pending=True,
    )


# ── Stage: Claude Workflow evidence ──────────────────────────────

_CLAUDE_WORKFLOW_RECORDED_GAP_LIMIT = 20


def _record_claude_workflow_stage_event(archive_root: Path, summary: object) -> None:
    """Persist the materialization summary so a readiness surface can read it.

    ``materialize_claude_workflow_archive`` returns a fresh
    ``ClaudeWorkflowMaterializationSummary`` every convergence pass; without
    this it was logged once and discarded. Recorded into the disposable
    ``ops.db`` tier via the existing generic ``daemon_stage_events`` table (no
    schema change) so ``polylogue doctor`` / archive readiness can report the
    current gap count instead of only a log line.
    """
    gaps = tuple(getattr(summary, "gaps", ()))
    payload: dict[str, object] = {
        "run_count": getattr(summary, "run_count", 0),
        "call_count": getattr(summary, "call_count", 0),
        "attempt_count": getattr(summary, "attempt_count", 0),
        "linked_session_count": getattr(summary, "linked_session_count", 0),
        "unresolved_call_count": getattr(summary, "unresolved_call_count", 0),
        "gap_count": len(gaps),
        "gaps": list(gaps[:_CLAUDE_WORKFLOW_RECORDED_GAP_LIMIT]),
    }
    status = "gaps" if gaps else "clean"
    try:
        from polylogue.storage.archive_readiness import CLAUDE_WORKFLOW_STAGE_NAME
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
        from polylogue.storage.sqlite.archive_tiers.ops_write import record_daemon_stage_event
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
        from polylogue.storage.sqlite.connection_profile import open_daemon_connection

        ops_db = archive_root / "ops.db"
        ops_db.parent.mkdir(parents=True, exist_ok=True)
        with open_daemon_connection(ops_db, timeout=30.0) as conn:
            initialize_archive_tier(conn, ArchiveTier.OPS)
            record_daemon_stage_event(
                conn,
                stage=CLAUDE_WORKFLOW_STAGE_NAME,
                status=status,
                observed_at_ms=int(time.time() * 1000),
                payload=payload,
            )
    except Exception:
        logger.warning("claude-workflow: failed to record materialization stage event", exc_info=True)


def make_claude_workflow_stage(db_path: Path) -> ConvergenceStage:
    """Rebuild Claude Workflow graphs after any admitted family member changes."""

    def archive_root() -> Path:
        active_index = _active_archive_index_path(db_path)
        return (active_index or db_path).parent

    def relevant(path: Path) -> bool:
        return artifact_rule_for_path(Provider.CLAUDE_CODE, str(path)) is not None

    def check(path: Path) -> bool:
        if not relevant(path):
            return False
        try:
            from polylogue.insights.claude_workflow_materializer import (
                claude_workflow_materialization_needed,
            )

            return claude_workflow_materialization_needed(archive_root())
        except FileNotFoundError:
            return False
        except Exception:
            logger.warning("claude-workflow freshness probe failed", exc_info=True)
            return True

    def execute(path: Path) -> StageExecuteReturn:
        if not relevant(path):
            return True
        try:
            from polylogue.insights.claude_workflow_materializer import materialize_claude_workflow_archive

            summary = materialize_claude_workflow_archive(archive_root())
            logger.info(
                "claude-workflow: materialized runs=%d calls=%d attempts=%d gaps=%d",
                summary.run_count,
                summary.call_count,
                summary.attempt_count,
                len(summary.gaps),
            )
            _record_claude_workflow_stage_event(archive_root(), summary)
            return True
        except Exception:
            logger.warning("claude-workflow: materialization failed", exc_info=True)
            return False

    def check_many(paths: Sequence[Path]) -> set[Path]:
        candidates = {path for path in paths if relevant(path)}
        if not candidates:
            return set()
        return candidates if check(next(iter(candidates))) else set()

    def execute_many(paths: Sequence[Path]) -> StageExecuteReturn:
        candidates = [path for path in paths if relevant(path)]
        return True if not candidates else execute(candidates[0])

    return ConvergenceStage(
        name="claude_workflow",
        description="Rebuild evidence-backed Claude Workflow topology from current raw authority",
        check=check,
        execute=execute,
        check_many=check_many,
        execute_many=execute_many,
    )


# ── Stage: insights ────────────────────────────────────────────────


def make_insights_stage(db_path: Path) -> ConvergenceStage:
    """Refresh session insights for sessions missing them."""

    def check(path: Path) -> bool:
        archive_db = _active_archive_index_path(db_path)
        if archive_db is not None:
            return _archive_insights_check(archive_db, path, archive_root=db_path.parent)
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
            logger.warning("convergence freshness probe %s errored; treating as needs-work", "check", exc_info=True)
            return True

    def execute(path: Path) -> StageExecuteReturn:
        archive_db = _active_archive_index_path(db_path)
        if archive_db is not None:
            return _archive_insights_execute(archive_db, path, archive_root=db_path.parent)
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
                    session_ids = [session_id for session_id in session_ids if session_id not in hot_ids]
                    if not session_ids:
                        return False
                counts = rebuild_session_insights_sync(
                    conn,
                    session_ids=session_ids,
                    page_size=_DAEMON_INSIGHT_REBUILD_PAGE_SIZE,
                )
                _record_fts_freshness_after_insights(conn)
                conn.commit()
                logger.info(
                    "insights: refreshed sessions=%d profiles=%d work_events=%d phases=%d threads=%d",
                    len(session_ids),
                    counts.profiles,
                    counts.work_events,
                    counts.phases,
                    counts.threads,
                )
                if hot_ids:
                    return False
            return True
        except Exception:
            logger.warning("insights: rebuild failed", exc_info=True)
            return False

    def check_many(paths: Sequence[Path]) -> set[Path]:
        archive_db = _active_archive_index_path(db_path)
        if archive_db is not None:
            return _archive_insights_check_many(archive_db, paths, archive_root=db_path.parent)
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
            logger.warning(
                "convergence freshness probe %s errored; treating as needs-work", "check_many", exc_info=True
            )
            return set(paths)

    def execute_many(paths: Sequence[Path]) -> StageExecuteReturn:
        archive_db = _active_archive_index_path(db_path)
        if archive_db is not None:
            return _archive_insights_execute_many(archive_db, paths, archive_root=db_path.parent)
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
                    session_ids = [session_id for session_id in session_ids if session_id not in hot_ids]
                    if not session_ids:
                        return False
                counts = rebuild_session_insights_sync(
                    conn,
                    session_ids=session_ids,
                    page_size=_DAEMON_INSIGHT_REBUILD_PAGE_SIZE,
                )
                _record_fts_freshness_after_insights(conn)
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
                if hot_ids:
                    return False
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
            logger.warning(
                "convergence freshness probe %s errored; treating as needs-work", "check_sessions", exc_info=True
            )
            return set(session_ids)

    def execute_sessions(session_ids: Sequence[str]) -> StageExecuteReturn:
        archive_db = _active_archive_index_path(db_path)
        if archive_db is not None:
            return _archive_insights_execute_sessions(archive_db, session_ids, archive_root=db_path.parent)
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
                    ids = [session_id for session_id in ids if session_id not in hot_ids]
                    if not ids:
                        return False
                counts = rebuild_session_insights_sync(
                    conn,
                    session_ids=ids,
                    page_size=_DAEMON_INSIGHT_REBUILD_PAGE_SIZE,
                )
                _record_fts_freshness_after_insights(conn)
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
                if hot_ids:
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
        false_means_pending=True,
    )


def _sinex_session_ids_for_paths(
    db_path: Path,
    paths: Sequence[Path],
) -> dict[Path, list[str]]:
    normalized = tuple(dict.fromkeys(Path(path) for path in paths))
    if not normalized:
        return {}
    lookup_db = _active_archive_index_path(db_path) or db_path
    if not lookup_db.exists():
        return {path: [] for path in normalized}
    conn = sqlite3.connect(f"file:{lookup_db}?mode=ro", uri=True, timeout=5.0)
    try:
        return _schema_archive_session_ids_for_source_paths(conn, normalized, archive_root=db_path.parent)
    finally:
        conn.close()


def make_sinex_publication_stage(
    db_path: Path,
    service: PublicationService,
) -> ConvergenceStage:
    """Drain the durable source-tier outbox before primary projections advance."""
    from polylogue.sinex.models import PublicationMode

    def ids_for_path(path: Path) -> list[str]:
        return _sinex_session_ids_for_paths(db_path, (path,)).get(path, [])

    def check(path: Path) -> bool:
        return bool(service.unresolved_object_ids(ids_for_path(path)))

    def execute(path: Path) -> StageExecuteReturn:
        session_ids = ids_for_path(path)
        if not session_ids:
            return True
        summary = service.drain_once(object_ids=session_ids, limit=service.max_batch)
        logger.info(
            "sinex_publication: drain attempted=%d confirmed=%d debt=%d rejected=%d "
            "transport_failures=%d payload_failures=%d remaining=%d",
            summary.attempted,
            summary.confirmed,
            summary.durable_debt,
            summary.rejected,
            summary.transport_failures,
            summary.payload_failures,
            summary.remaining_lag,
        )
        return not service.unresolved_object_ids(session_ids)

    def check_many(paths: Sequence[Path]) -> set[Path]:
        by_path = _sinex_session_ids_for_paths(db_path, paths)
        all_ids = tuple(dict.fromkeys(session_id for values in by_path.values() for session_id in values))
        unresolved = service.unresolved_object_ids(all_ids)
        return {path for path, values in by_path.items() if unresolved.intersection(values)}

    def execute_many(paths: Sequence[Path]) -> StageExecuteReturn:
        by_path = _sinex_session_ids_for_paths(db_path, paths)
        all_ids = tuple(dict.fromkeys(session_id for values in by_path.values() for session_id in values))
        if not all_ids:
            return True
        summary = service.drain_once(object_ids=all_ids, limit=service.max_batch)
        logger.info(
            "sinex_publication: batch drain subjects=%d attempted=%d confirmed=%d debt=%d rejected=%d "
            "transport_failures=%d payload_failures=%d remaining=%d",
            len(all_ids),
            summary.attempted,
            summary.confirmed,
            summary.durable_debt,
            summary.rejected,
            summary.transport_failures,
            summary.payload_failures,
            summary.remaining_lag,
        )
        return not service.unresolved_object_ids(all_ids)

    def check_sessions(session_ids: Sequence[str]) -> set[str]:
        return service.unresolved_object_ids(session_ids)

    def execute_sessions(session_ids: Sequence[str]) -> StageExecuteReturn:
        if not session_ids:
            return True
        summary = service.drain_once(object_ids=session_ids, limit=service.max_batch)
        logger.info(
            "sinex_publication: session drain subjects=%d attempted=%d confirmed=%d debt=%d rejected=%d "
            "transport_failures=%d payload_failures=%d remaining=%d",
            len(tuple(dict.fromkeys(session_ids))),
            summary.attempted,
            summary.confirmed,
            summary.durable_debt,
            summary.rejected,
            summary.transport_failures,
            summary.payload_failures,
            summary.remaining_lag,
        )
        return not service.unresolved_object_ids(session_ids)

    def barrier(path: Path) -> bool:
        return bool(service.blocking_object_ids(ids_for_path(path)))

    def barrier_many(paths: Sequence[Path]) -> set[Path]:
        by_path = _sinex_session_ids_for_paths(db_path, paths)
        all_ids = tuple(dict.fromkeys(session_id for values in by_path.values() for session_id in values))
        blocked = service.blocking_object_ids(all_ids)
        return {path for path, values in by_path.items() if blocked.intersection(values)}

    return ConvergenceStage(
        name="sinex_publication",
        description="Drain exact accepted revisions through the configured Sinex transport",
        check=check,
        execute=execute,
        check_many=check_many,
        execute_many=execute_many,
        check_sessions=check_sessions,
        execute_sessions=execute_sessions,
        false_means_pending=True,
        blocks_following_stages=service.mode is PublicationMode.PRIMARY,
        barrier_check=barrier,
        barrier_check_many=barrier_many,
        barrier_check_sessions=service.blocking_object_ids,
        status=lambda: service.status().as_dict(),
    )


_RAW_PARSE_RECOVERY_BATCH_LIMIT = 200
# Mirrors ``storage.repair.RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES`` as a
# local literal rather than importing it: new surface code (this stage lives
# in ``daemon/``) should not import substrate (``storage``) internals
# directly per this repo's layering ratchet, and ``repair_materialization``'s
# ``max_payload_bytes`` is a plain bound this stage can restate on its own.
_RAW_PARSE_RECOVERY_MAX_PAYLOAD_BYTES = 1024 * 1024 * 1024


def _raw_parse_recovery_pending_count(db_path: Path, path: Path, *, archive_root: Path | None = None) -> int:
    """Count raw rows under ``path`` acquired but never materialized.

    Mirrors the non-terminal branch of ``repair.py``'s candidate query at the
    cheap read-only level this stage's ``check`` needs: no materialized
    ``sessions`` row for the raw (by raw_id or native-id alias) and no
    terminal parse error recorded. It intentionally does not replicate the
    full authority/quarantine/byte-authority classification -- that
    refinement happens inside ``repair_raw_materialization`` itself during
    ``execute``; this is only a cheap "is there plausibly pending work here"
    probe so ``check`` stays fast and false positives just cost one wasted
    ``execute`` call rather than silently missing real backlog.
    """
    durable_root = archive_root or db_path.parent
    source_db = durable_root / "source.db"
    if not source_db.exists():
        # An uninitialized archive has neither durable nor derived tiers and
        # has no recovery work.  Once an index exists, however, a missing
        # source tier is an authority failure and must remain retryable.
        if db_path.exists() or (durable_root / ".index-active-pointer").exists():
            raise FileNotFoundError(f"durable source tier is missing: {source_db}")
        return 0
    index_db = ArchiveLocation.resolve(durable_root).active_index_path
    normalized_root = str(path).rstrip("/")
    replay_authority_placeholders = ", ".join("?" for _ in RAW_FAILURE_REPLAY_AUTHORITY_EVIDENCE_KINDS)
    try:
        conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error:
        logger.warning(
            "raw_parse_recovery: could not open source.db for %s; refusing to classify as no pending work",
            path,
            exc_info=True,
        )
        raise
    try:
        if index_db.exists():
            conn.execute("ATTACH DATABASE ? AS index_tier", (str(index_db),))
            materialized_join = """
                LEFT JOIN index_tier.sessions AS s_by_raw ON s_by_raw.raw_id = r.raw_id
                LEFT JOIN index_tier.sessions AS s_by_native
                  ON r.native_id IS NOT NULL
                 AND s_by_native.origin = r.origin
                 AND s_by_native.native_id = r.native_id
                LEFT JOIN raw_sessions AS existing_native_raw
                  ON existing_native_raw.raw_id = s_by_native.raw_id
            """
            materialized_where = """
              AND s_by_raw.raw_id IS NULL
              AND (s_by_native.native_id IS NULL OR existing_native_raw.raw_id IS NULL)
            """
        else:
            materialized_join = ""
            materialized_where = ""
        row = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM raw_sessions AS r
            {materialized_join}
            WHERE (r.source_path = ? OR r.source_path LIKE ?)
              AND NOT (
                COALESCE(r.validation_status, '') = 'failed'
                AND r.parsed_at_ms IS NULL
              )
              AND (
                (
                  r.parsed_at_ms IS NULL
                  AND (
                    r.parse_error IS NULL
                    OR r.parse_error = 'OperationalError: database is locked'
                    OR r.parse_error LIKE 'decode:%No such file or directory:%'
                    OR r.parse_error LIKE 'membership_replay_conflict:%'
                  )
                )
                OR EXISTS (
                    SELECT 1
                    FROM raw_artifacts AS failure_evidence
                    WHERE failure_evidence.raw_id IS r.raw_id
                      AND failure_evidence.origin IS r.origin
                      AND failure_evidence.source_path IS r.source_path
                      AND failure_evidence.source_index IS r.source_index
                      AND failure_evidence.artifact_kind IN ({replay_authority_placeholders})
                      AND failure_evidence.support_status = ?
                )
              )
              {materialized_where}
            """,
            (
                normalized_root,
                f"{normalized_root}/%",
                *sorted(RAW_FAILURE_REPLAY_AUTHORITY_EVIDENCE_KINDS),
                RAW_FAILURE_DEFERRED_SUPPORT_STATUS,
            ),
        ).fetchone()
        return int(row[0] or 0) if row is not None else 0
    except sqlite3.Error:
        logger.warning(
            "raw_parse_recovery: pending-count probe failed for %s; refusing to classify as no pending work",
            path,
            exc_info=True,
        )
        raise
    finally:
        conn.close()


def make_raw_parse_recovery_stage(db_path: Path, *, archive_root: Path | None = None) -> ConvergenceStage:
    """Requeue raw rows an interrupted ingest attempt never validated/parsed.

    polylogue-61jg: when the daemon stops mid-batch,
    ``CursorStore._mark_interrupted_ops_attempts`` stamps the dangling
    ``ingest_attempts`` row ``interrupted`` and records one
    ``raw_parse_recovery`` convergence-debt row per source path that attempt
    covered. This stage is what actually drains that debt: ``check`` reports
    whether raw rows under the path are still acquired but never
    materialized, and ``execute`` re-drives ``repair_raw_materialization``
    scoped to exactly that path via ``source_root`` -- the same replay engine
    the archive-wide trickle conveyor and manual ``ops maintenance
    rebuild-index`` already use, just requeued deterministically instead of
    waiting for an accidental future touch of the same path.
    """

    def check(path: Path) -> bool:
        return _raw_parse_recovery_pending_count(db_path, path, archive_root=archive_root) > 0

    def execute(path: Path) -> StageExecuteReturn:
        from polylogue.config import Config
        from polylogue.product.raw_authority import repair_materialization
        from polylogue.readiness.capability import raw_frontier_source_selection_block_reason

        configured_root = archive_root or db_path.parent
        if reason := raw_frontier_source_selection_block_reason(configured_root):
            logger.warning(
                "raw_parse_recovery: source-selection gate blocked for %s: %s",
                path,
                reason,
            )
            return False
        config = Config(archive_root=configured_root, render_root=configured_root, sources=[])
        try:
            repair_materialization(
                config,
                dry_run=False,
                raw_artifact_limit=_RAW_PARSE_RECOVERY_BATCH_LIMIT,
                max_payload_bytes=_RAW_PARSE_RECOVERY_MAX_PAYLOAD_BYTES,
                source_root=path,
            )
        except Exception:
            logger.warning("raw_parse_recovery: repair pass failed for %s", path, exc_info=True)
            return False
        return _raw_parse_recovery_pending_count(db_path, path, archive_root=configured_root) == 0

    return ConvergenceStage(
        name="raw_parse_recovery",
        description="Requeue raw rows an interrupted ingest attempt never validated/parsed",
        check=check,
        execute=execute,
        false_means_pending=True,
    )


def make_raw_authority_verdict_cache_stage(db_path: Path) -> ConvergenceStage:
    """Warm the content-keyed raw-authority verdict cache in bounded cohorts."""

    def work() -> RawAuthorityVerdictCacheWork | None:
        return find_raw_authority_verdict_cache_work(db_path.parent)

    def log_skipped_append_cohorts(discovered: RawAuthorityVerdictCacheWork) -> None:
        if discovered.skipped_append_cohorts:
            logger.info(
                "raw_authority_verdict_cache: skipped append cohorts=%d; append authority uses a separate proof",
                discovered.skipped_append_cohorts,
            )

    def check(_path: Path) -> bool:
        try:
            discovered = work()
        except Exception:
            logger.warning("raw_authority_verdict_cache: readiness probe failed", exc_info=True)
            return True
        if discovered is None:
            return False
        log_skipped_append_cohorts(discovered)
        return bool(discovered.pending_logical_source_keys)

    def check_many(paths: Sequence[Path]) -> set[Path]:
        if not paths:
            return set()
        try:
            discovered = work()
        except Exception:
            logger.warning("raw_authority_verdict_cache: batch readiness probe failed", exc_info=True)
            return set(paths)
        if discovered is None:
            return set()
        log_skipped_append_cohorts(discovered)
        return set(paths) if discovered.pending_logical_source_keys else set()

    def execute(_path: Path) -> StageExecuteReturn:
        return execute_many((_path,))

    def execute_many(_paths: Sequence[Path]) -> StageExecuteReturn:
        try:
            outcome = warm_raw_authority_verdict_cache(
                db_path.parent,
                max_cohorts=_DAEMON_RAW_AUTHORITY_CACHE_MAX_COHORTS,
                now_ms=int(time.time() * 1000),
            )
        except Exception:
            logger.warning("raw_authority_verdict_cache: bounded warmup failed", exc_info=True)
            return False
        logger.info(
            "raw_authority_verdict_cache: warmed cohorts=%d skipped_append=%d pending=%s",
            outcome.warmed_cohorts,
            outcome.skipped_append_cohorts,
            outcome.pending_cohorts,
        )
        return not outcome.pending_cohorts

    return ConvergenceStage(
        name="raw_authority_verdict_cache",
        description="Warm content-keyed raw-authority verdicts for full/unknown cohorts",
        check=check,
        execute=execute,
        check_many=check_many,
        execute_many=execute_many,
        false_means_pending=True,
    )


def make_default_convergence_stages(
    db_path: Path,
    *,
    sinex_transport: SinexTransport | None = None,
    embed_defer: Callable[[], bool] | None = None,
) -> tuple[ConvergenceStage, ...]:
    """Build daemon stages, failing explicitly when backed mode lacks transport."""
    from polylogue.archive.query.production_evaluator import ArchiveCanonicalPlanEvaluator
    from polylogue.paths import archive_root
    from polylogue.sinex.models import PublicationMode
    from polylogue.sinex.service import PublicationService
    from polylogue.sinex.transport import resolve_configured_transport
    from polylogue.storage.archive_identity import ArchiveLocation

    mode = PublicationMode.from_string(load_polylogue_config().sinex_mode)
    stages: list[ConvergenceStage] = []
    if mode is not PublicationMode.OFF:
        transport = sinex_transport if sinex_transport is not None else resolve_configured_transport()
        stages.append(
            make_sinex_publication_stage(
                db_path,
                PublicationService(
                    source_db_path=ArchiveLocation.resolve(archive_root()).configured_tier("source").configured_path,
                    mode=mode,
                    transport=transport,
                ),
            )
        )
    stages.extend(
        (
            make_raw_parse_recovery_stage(db_path, archive_root=archive_root()),
            make_raw_authority_verdict_cache_stage(db_path),
            make_fts_stage(db_path),
            make_embed_stage(db_path, defer=embed_defer),
            make_claude_workflow_stage(db_path),
            make_insights_stage(db_path),
            make_standing_query_stage(db_path, evaluator=ArchiveCanonicalPlanEvaluator(db_path)),
        )
    )
    return tuple(stages)


# ── Helpers ────────────────────────────────────────────────────────


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
    """Advance derivation generations when the configured recipe changes.

    ``embedding_status`` remains a compatibility projection. The generation/key
    transition is the monotonic authority: an older success or failure receipt
    can no longer clear the pending mark created here.
    """
    from polylogue.storage.embeddings.identity import EmbeddingRecipe, register_embedding_identity_sql
    from polylogue.storage.search_providers.sqlite_vec_runtime import _reconcile_vec0_dimension
    from polylogue.storage.search_providers.sqlite_vec_support import logger as vec_logger

    cfg = load_polylogue_config()
    configured_model = str(cfg.embedding_model)
    configured_dimension = int(cfg.embedding_dimension)
    recipe = EmbeddingRecipe.current(model=configured_model, dimensions=configured_dimension)

    if not _table_exists(conn, "message_embeddings_meta") or not _table_exists(conn, "embedding_status"):
        return

    meta_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(message_embeddings_meta)")}
    stored_models = conn.execute("SELECT DISTINCT model FROM message_embeddings_meta ORDER BY model").fetchall()
    stored_model = str(stored_models[0][0]) if stored_models else None
    model_changed = stored_model is not None and stored_model != configured_model

    dimension_changed = False
    if stored_model is not None:
        stored_dim_row = conn.execute("SELECT dimension FROM message_embeddings_meta LIMIT 1").fetchone()
        if stored_dim_row:
            dimension_changed = int(stored_dim_row[0]) != configured_dimension

    meta_recipe_changed = False
    if stored_model is not None and "recipe_hash" in meta_columns:
        meta_recipe_changed = (
            conn.execute(
                """
                SELECT 1
                FROM message_embeddings_meta
                WHERE recipe_hash IS NULL OR recipe_hash != ?
                LIMIT 1
                """,
                (recipe.recipe_hash,),
            ).fetchone()
            is not None
        )

    advanced_generations = 0
    advanced_session_ids: list[str] = []
    if _table_exists(conn, "embedding_derivation_state"):
        state_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(embedding_derivation_state)")}
        required = {
            "session_id",
            "generation",
            "derivation_key",
            "source_hash",
            "recipe_hash",
            "output_contract_hash",
            "attempt_state",
            "message_count",
            "updated_at_ms",
        }
        if required.issubset(state_columns):
            register_embedding_identity_sql(conn)
            now_ms = int(time.time() * 1000)
            rows = conn.execute(
                """
                UPDATE embedding_derivation_state
                SET generation = generation + 1,
                    derivation_key = polylogue_embedding_derivation_key(
                        session_id, source_hash, ?, ?
                    ),
                    recipe_hash = ?,
                    output_contract_hash = ?,
                    attempt_state = 'pending',
                    message_count = 0,
                    updated_at_ms = ?
                WHERE recipe_hash != ? OR output_contract_hash != ?
                RETURNING session_id
                """,
                (
                    recipe.recipe_hash,
                    recipe.output_contract_hash,
                    recipe.recipe_hash,
                    recipe.output_contract_hash,
                    now_ms,
                    recipe.recipe_hash,
                    recipe.output_contract_hash,
                ),
            ).fetchall()
            advanced_session_ids = [str(row[0]) for row in rows]
            advanced_generations = len(advanced_session_ids)

    # NOTE: model_changed deliberately does not participate in recipe_changed
    # below. EmbeddingRecipe.identity() hashes the model name as one of its
    # declared fields, so any genuine model change already flips recipe_hash
    # and is caught by meta_recipe_changed/advanced_generations. Keeping
    # model_changed out of the trigger avoids a permanent false-positive
    # archive-wide reindex for archives (e.g. the demo world) that carry a
    # deliberately labeled synthetic model alongside a matching recipe_hash.
    recipe_changed = dimension_changed or meta_recipe_changed or advanced_generations > 0
    if model_changed:
        vec_logger.info(
            "embedding model changed: stored=%s configured=%s",
            stored_model,
            configured_model,
        )
    if dimension_changed:
        vec_logger.info(
            "embedding dimension changed: stored=%d configured=%d — dropping vec0 + reindex",
            _stored_dim_from_meta(conn),
            configured_dimension,
        )
    if recipe_changed and not dimension_changed:
        vec_logger.info(
            "embedding recipe identity changed — marking %d session(s) for reindex (meta_recipe_changed=%s)",
            advanced_generations,
            meta_recipe_changed,
        )

    if recipe_changed:
        if dimension_changed:
            # A dimension change invalidates every stored vector archive-wide
            # (the vec0 table itself is dropped/recreated below), not just the
            # sessions whose derivation_state row was just advanced.
            conn.execute("UPDATE embedding_status SET needs_reindex = 1, error_message = NULL")
            _reconcile_vec0_dimension(conn, configured_dimension)
        elif advanced_session_ids:
            # Scope the bulk mark to the sessions whose derivation_state
            # generation actually advanced this pass. An archive larger than
            # one convergence window (_DAEMON_EMBED_MAX_SESSIONS) advances a
            # bounded subset per call; re-marking the *whole* table here would
            # re-flag sessions that already succeeded under the new
            # generation/key in an earlier pass, starving convergence on the
            # same first batch forever (polylogue PR #3067 review).
            placeholders = ", ".join("?" for _ in advanced_session_ids)
            conn.execute(
                f"""
                UPDATE embedding_status
                SET needs_reindex = 1, error_message = NULL
                WHERE session_id IN ({placeholders})
                """,
                advanced_session_ids,
            )


def _stored_dim_from_meta(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT dimension FROM message_embeddings_meta LIMIT 1").fetchone()
    return int(row[0]) if row else 0


def _reconcile_archive_embedding_config_change(index_db_path: Path, *, archive_root: Path | None = None) -> None:
    """Reconcile the recipe on the tier that owns embedding state.

    Active archives split ``index.db`` source facts from the rebuildable
    ``embeddings.db`` derivation ledger. Legacy monolithic fixtures keep the
    old same-file behavior. Loading sqlite-vec here keeps dimension-change
    reconciliation able to drop the virtual table on the real production
    route.

    ``archive_root`` names the durable-tier root explicitly when the caller
    already resolved it (e.g. through ``.index-active-pointer``-aware
    resolution, where ``index_db_path`` itself may point into a
    ``.index-generations/<gen>`` subdirectory rather than the archive root
    housing ``embeddings.db``); it defaults to ``index_db_path.parent`` for
    callers that never diverge from the plain convention.
    """

    from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

    root = archive_root if archive_root is not None else index_db_path.parent
    sibling = root / "embeddings.db"
    target = sibling if sibling.exists() else index_db_path
    conn = sqlite3.connect(target, timeout=5.0)
    try:
        try_load_sqlite_vec(conn)
        _reconcile_embedding_config_change(conn)
        conn.commit()
    finally:
        conn.close()


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
    """Record the real archive-wide ``messages_fts`` freshness state after a
    targeted (session-scoped) repair.

    A targeted repair only proves the sessions it touched are indexed -- it
    says nothing about the rest of the archive. This used to call
    ``message_fts_readiness_sync(conn, verify_total_rows=False)``, a cheap
    boolean that is really just "has the FTS index ever been populated at
    all AND does *any* indexable block exist" (``SELECT 1 ... LIMIT 1`` on
    each side), not "every indexable block is indexed". Because that boolean
    is true for almost any non-empty, previously-populated archive, every
    targeted repair call unconditionally wrote ``state=ready,
    missing_rows=0`` over the ledger's single global ``messages_fts`` row --
    discarding whatever accurate ``missing_rows`` an exact snapshot had
    previously recorded and asserting a global verdict a scoped repair never
    measured. Surface-coherence audit 2026-07-31 caught this live: 12,659
    blocks missing from the index while the ledger reported ``ready`` with
    ``missing_rows=0``.

    Fix: mirror ``session_work_events_fts``, which only ever gets its
    freshness row from an exact, archive-wide invariant snapshot
    (``fts_invariant_snapshot_sync``) -- never a scoped/cheap approximation.
    Reuse that same snapshot here for
    the messages surface; it runs the identical anti-join query the hourly
    ``fts_orphan_audit`` sweep already accepts at that cadence
    (``storage/fts/fts_lifecycle.py``'s ``_trigger_invariant_sync``), so its
    cost is already an accepted class for this surface.
    """
    from polylogue.storage.fts.freshness import READY, STALE, record_fts_surface_state_sync
    from polylogue.storage.fts.fts_lifecycle import fts_invariant_snapshot_sync

    surface = fts_invariant_snapshot_sync(conn).messages
    ready = bool(surface.ready)
    record_fts_surface_state_sync(
        conn,
        surface="messages_fts",
        state=READY if ready else STALE,
        source_rows=int(surface.source_rows),
        indexed_rows=int(surface.indexed_rows),
        missing_rows=int(surface.missing_rows),
        excess_rows=int(surface.excess_rows),
        duplicate_rows=int(surface.duplicate_rows),
        identity_mismatch_rows=int(surface.identity_mismatch_rows),
        detail=(
            "targeted changed-session repair complete"
            if ready
            else "targeted changed-session repair left an archive-wide FTS gap"
        ),
    )


def _record_fts_freshness_after_insights(conn: sqlite3.Connection) -> None:
    """Publish exact FTS readiness after insight rows have changed.

    The insights materializer writes ``session_work_events`` after the message
    FTS convergence stage has run. Its triggers keep the derived FTS rows in
    sync, but readiness consumers use the durable freshness snapshot rather
    than inferring health from trigger presence. Record the exact post-write
    invariant here so every production insights route owns the same final
    state; test harnesses must not repair the ledger themselves.
    """
    from polylogue.storage.fts.freshness import READY, STALE, record_fts_surface_state_sync
    from polylogue.storage.fts.fts_lifecycle import session_work_events_fts_invariant_sync

    surface = session_work_events_fts_invariant_sync(conn)
    record_fts_surface_state_sync(
        conn,
        surface=surface.name,
        state=READY if surface.ready else STALE,
        source_rows=surface.source_rows,
        indexed_rows=surface.indexed_rows,
        missing_rows=surface.missing_rows,
        excess_rows=surface.excess_rows,
        duplicate_rows=surface.duplicate_rows,
        identity_mismatch_rows=surface.identity_mismatch_rows,
        detail=None if surface.ready else "exact invariant failed after insights refresh",
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
    stale_predicate = session_profile_stale_predicate("c", "sp")
    rows = conn.execute(
        f"""
        SELECT c.session_id
        FROM sessions AS c
        LEFT JOIN session_profiles AS sp ON sp.session_id = c.session_id
        WHERE c.session_id IN ({placeholders})
          AND (
              sp.session_id IS NULL
              OR sp.materializer_version != ?
              OR {stale_predicate}
          )
        ORDER BY c.session_id
        """,
        unique_ids + (SESSION_INSIGHT_MATERIALIZER_VERSION,),
    ).fetchall()
    return [str(row[0]) for row in rows]


# ── Archive file-set helpers ─────────────────────────────────────


def _attached_source_db_path(conn: sqlite3.Connection, *, archive_root: Path | None = None) -> Path:
    if archive_root is not None:
        return archive_root / "source.db"
    for _, name, path in conn.execute("PRAGMA database_list").fetchall():
        if str(name) == "main" and path:
            return Path(str(path)).with_name("source.db")
    return Path("source.db")


def _ensure_source_tier_attached(conn: sqlite3.Connection, *, archive_root: Path | None = None) -> bool:
    for _, name, _path in conn.execute("PRAGMA database_list").fetchall():
        if str(name) == "source_tier":
            return True
    source_db = _attached_source_db_path(conn, archive_root=archive_root)
    if not source_db.exists():
        return False
    conn.execute("ATTACH DATABASE ? AS source_tier", (str(source_db),))
    return True


def _active_archive_index_path(db_path: Path) -> Path | None:
    """Resolve the active ``index.db`` for the archive rooted at ``db_path``'s directory.

    ``db_path`` always lives directly in the archive root (whether it names
    ``index.db``, ``source.db``, or another tier file), so ``db_path.parent``
    is the archive root -- this mirrors ``ArchiveLocation``'s own resolution
    instead of blindly renaming ``db_path`` to ``index.db`` in place, so an
    active ``.index-active-pointer`` generation is still followed correctly.
    """
    from polylogue.storage.archive_identity import ArchiveLocation

    index_db = ArchiveLocation.resolve(db_path.parent).active_index_path
    if not index_db.exists():
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


def _schema_archive_session_ids_for_source_path(
    conn: sqlite3.Connection, path: Path, *, archive_root: Path | None = None
) -> list[str]:
    return _schema_archive_session_ids_for_source_paths(conn, [path], archive_root=archive_root).get(path, [])


def _schema_archive_session_ids_for_source_paths(
    conn: sqlite3.Connection,
    paths: Sequence[Path],
    *,
    archive_root: Path | None = None,
) -> dict[Path, list[str]]:
    normalized_paths = tuple(dict.fromkeys(Path(path) for path in paths))
    if not normalized_paths or not _table_exists(conn, "sessions"):
        return {path: [] for path in normalized_paths}
    raw_table = "raw_sessions"
    if not _table_exists(conn, "raw_sessions"):
        raw_table = "source_tier.raw_sessions"
        if not _ensure_source_tier_attached(conn, archive_root=archive_root):
            return {path: [] for path in normalized_paths}
        # Deliberately let sqlite3.Error from the attach above propagate
        # instead of swallowing it into an empty result here (polylogue-co8b):
        # every caller of this helper (_archive_embed_check[_many],
        # _archive_insights_check[_many], _sinex_session_ids_for_paths) wraps
        # its own call in a broad try/except that fails OPEN -- "treating as
        # needs-work" -- matching every other freshness probe in this file.
        # Swallowing the error here instead made the outer probe see a clean
        # `{path: []}` result and conclude there was nothing to do, silently
        # disabling embed/insights convergence for the affected source paths
        # with no convergence_debt row and no counter, only a log line. The
        # existing false_means_pending -> convergence_debt retry path already
        # bounds the resulting "fires every tick" concern: a genuinely
        # persistent attach failure surfaces as repeated execute() failures,
        # which convergence_debt retries with its own backoff rather than
        # busy-looping here.
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
    return list(dict.fromkeys(str(row[0]) for row in rows))


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
    from polylogue.storage.fts.fts_lifecycle import reset_message_fts_index_sync

    reset_message_fts_index_sync(conn)


def _archive_repair_sessions_fts(conn: sqlite3.Connection, session_ids: Sequence[str]) -> None:
    """Repair ``messages_fts`` for just the changed sessions (#1851).

    The archive FTS convergence paths previously called the full
    ``_archive_rebuild_messages_fts`` (delete-all + re-insert every message row)
    on every batch, so a single small append re-indexed the entire corpus —
    ~14 MiB of writes regardless of payload size. When the source path resolves
    to known session ids we instead delete+reinsert only those sessions' FTS
    rows (the same targeted primitive the legacy monolith path used), and mark
    the surface ready. Unknown path scope is intentionally a no-op here:
    whole-archive FTS repair is a dedicated surface-debt operation, not a side
    effect of live path convergence.
    """
    ids = tuple(dict.fromkeys(str(session_id) for session_id in session_ids if session_id))
    if not ids:
        return
    if not _table_exists(conn, "messages_fts"):
        return
    from polylogue.storage.fts.fts_lifecycle import repair_message_fts_index_sync

    repair_message_fts_index_sync(conn, ids)
    _mark_message_fts_ready_after_targeted_repair(conn)


def _archive_fts_check(db_path: Path, path: Path) -> bool:
    del db_path, path
    return False


def _archive_fts_execute(db_path: Path, path: Path) -> bool:
    del db_path
    logger.info("fts: archive source-path foreground repair skipped path=%s", path)
    return True


def _archive_fts_check_many(db_path: Path, paths: Sequence[Path]) -> set[Path]:
    del db_path, paths
    return set()


def _archive_fts_execute_many(db_path: Path, paths: Sequence[Path]) -> bool:
    del db_path
    if paths:
        logger.info("fts: archive batch source-path foreground repair skipped paths=%d", len(paths))
    return True


def _archive_fts_check_sessions(db_path: Path, session_ids: Sequence[str]) -> set[str]:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        try:
            ids = _archive_existing_session_ids(conn, session_ids)
            return {session_id for session_id in ids if _archive_fts_needs_repair(conn, [session_id])}
        finally:
            conn.close()
    except Exception:
        logger.warning(
            "convergence freshness probe %s errored; treating as needs-work",
            "_archive_fts_check_sessions",
            exc_info=True,
        )
        return set(session_ids)


def _archive_fts_execute_sessions(db_path: Path, session_ids: Sequence[str]) -> bool:
    try:
        conn = _open_archive_insight_write_connection(db_path)
        try:
            ids = _archive_existing_session_ids(conn, session_ids)
            if not ids:
                return True
            _archive_repair_sessions_fts(conn, ids)
            conn.commit()
            logger.info("fts: archive repaired messages_fts session debt sessions=%d", len(ids))
            return not _archive_fts_needs_repair(conn, ids)
        finally:
            conn.close()
    except Exception as exc:
        if _is_transient_sqlite_lock(exc):
            logger.info("fts: archive session repair deferred because sqlite is busy: %s", exc)
            return False
        logger.warning("fts: archive session repair failed", exc_info=True)
        return False


def repair_messages_fts_surface_result(db_path: Path) -> FtsSurfaceRepairResult:
    """Repair the whole archive ``messages_fts`` surface after global drift."""
    archive_db = _active_archive_index_path(db_path) or db_path
    try:
        conn = _open_archive_insight_write_connection(archive_db)
        try:
            from polylogue.storage.fts.dangling_repair import configure_bounded_repair_connection
            from polylogue.storage.fts.freshness import READY, STALE, record_fts_surface_state_sync
            from polylogue.storage.fts.fts_lifecycle import (
                fts_invariant_snapshot_sync,
                reconcile_message_fts_rows_once_sync,
            )

            configure_bounded_repair_connection(conn)
            inserted_total, deleted_total = reconcile_message_fts_rows_once_sync(conn)
            surface = fts_invariant_snapshot_sync(conn).messages
            record_fts_surface_state_sync(
                conn,
                surface="messages_fts",
                state=READY if surface.ready else STALE,
                source_rows=surface.source_rows,
                indexed_rows=surface.indexed_rows,
                missing_rows=surface.missing_rows,
                excess_rows=surface.excess_rows,
                duplicate_rows=surface.duplicate_rows,
                identity_mismatch_rows=surface.identity_mismatch_rows,
                detail=None if surface.ready else "exact message FTS parity failed after repair",
            )
            conn.commit()
            logger.info(
                "fts: archive messages_fts surface repair complete ready=%s inserted=%d deleted=%d "
                "source_rows=%d indexed_rows=%d missing=%d excess=%d identity_mismatch=%d",
                surface.ready,
                inserted_total,
                deleted_total,
                surface.source_rows,
                surface.indexed_rows,
                surface.missing_rows,
                surface.excess_rows,
                surface.identity_mismatch_rows,
            )
            return FtsSurfaceRepairResult(
                success=surface.ready,
                detail=None if surface.ready else "exact message FTS parity failed after repair",
            )
        finally:
            conn.close()
    except Exception as exc:
        if _is_transient_sqlite_lock(exc):
            logger.info("fts: archive global messages_fts repair deferred because sqlite is busy: %s", exc)
            return FtsSurfaceRepairResult(success=False, deferred=True, detail="SQLite writer busy")
        logger.warning("fts: archive global messages_fts repair failed", exc_info=True)
        return FtsSurfaceRepairResult(success=False, detail=f"{type(exc).__name__}: {exc}")


def repair_messages_fts_surface(db_path: Path) -> bool:
    """Compatibility boolean for callers that only need repair success."""
    return bool(repair_messages_fts_surface_result(db_path))


def repair_fts_surface_result(db_path: Path, surface: str) -> FtsSurfaceRepairResult:
    """Repair a named archive FTS surface from daemon convergence debt."""
    if surface == "messages_fts":
        return repair_messages_fts_surface_result(db_path)
    if surface != "session_work_events_fts":
        logger.warning("fts: unsupported archive FTS surface debt surface=%s", surface)
        return FtsSurfaceRepairResult(success=False, detail=f"unsupported FTS surface: {surface}")
    archive_db = _active_archive_index_path(db_path) or db_path
    try:
        conn = _open_archive_insight_write_connection(archive_db)
        try:
            from polylogue.storage.fts.dangling_repair import (
                configure_bounded_repair_connection,
                repair_stale_fts_rows,
            )

            configure_bounded_repair_connection(conn)
            outcome = repair_stale_fts_rows(conn)
            conn.commit()
            if outcome.success:
                logger.info(
                    "fts: archive derived FTS surface repair completed surface=%s detail=%s", surface, outcome.detail
                )
                return FtsSurfaceRepairResult(success=True, detail=outcome.detail)
            logger.warning(
                "fts: archive derived FTS surface repair incomplete surface=%s detail=%s", surface, outcome.detail
            )
            return FtsSurfaceRepairResult(success=False, detail=outcome.detail)
        finally:
            conn.close()
    except Exception as exc:
        if _is_transient_sqlite_lock(exc):
            logger.info(
                "fts: archive derived FTS surface repair deferred surface=%s because sqlite is busy: %s", surface, exc
            )
            return FtsSurfaceRepairResult(success=False, deferred=True, detail="SQLite writer busy")
        logger.warning("fts: archive derived FTS surface repair failed surface=%s", surface, exc_info=True)
        return FtsSurfaceRepairResult(success=False, detail=f"{type(exc).__name__}: {exc}")


def repair_fts_surface(db_path: Path, surface: str) -> bool:
    """Compatibility boolean for named archive FTS-surface repair."""
    return bool(repair_fts_surface_result(db_path, surface))


def _archive_pending_embedding_session_ids(
    conn: sqlite3.Connection, session_ids: Sequence[str], *, archive_root: Path | None = None
) -> list[str]:
    from polylogue.storage.embeddings.materialization import select_pending_archive_session_window

    ids = tuple(dict.fromkeys(str(session_id) for session_id in session_ids if session_id))
    if not ids:
        return []
    if archive_root is not None:
        embeddings_db: Path | None = archive_root / "embeddings.db"
    else:
        db_row = conn.execute("PRAGMA database_list").fetchone()
        index_db = Path(str(db_row[2])) if db_row is not None and db_row[2] else None
        embeddings_db = index_db.with_name("embeddings.db") if index_db is not None else None
    status_table = None
    if embeddings_db is not None and embeddings_db.exists():
        attached = {str(row[1]) for row in conn.execute("PRAGMA database_list").fetchall() if len(row) > 1}
        if "embeddings" not in attached:
            conn.execute("ATTACH DATABASE ? AS embeddings", (str(embeddings_db),))
        status_table = "embeddings.embedding_status"
    return [
        item.session_id
        for item in select_pending_archive_session_window(
            conn,
            status_table=status_table,
            session_ids=ids,
            max_sessions=_DAEMON_EMBED_MAX_SESSIONS,
            max_messages=_DAEMON_EMBED_MAX_MESSAGES,
        )
    ]


def _archive_embed_check(db_path: Path, path: Path, *, archive_root: Path | None = None) -> bool:
    try:
        _reconcile_archive_embedding_config_change(db_path, archive_root=archive_root)
        conn = sqlite3.connect(db_path, timeout=5.0)
        try:
            session_ids = _schema_archive_session_ids_for_source_path(conn, path, archive_root=archive_root)
            return bool(_archive_pending_embedding_session_ids(conn, session_ids, archive_root=archive_root))
        finally:
            conn.close()
    except Exception:
        logger.warning(
            "convergence freshness probe %s errored; treating as needs-work", "_archive_embed_check", exc_info=True
        )
        return True


def _archive_embed_execute(db_path: Path, path: Path, *, archive_root: Path | None = None) -> bool:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        try:
            session_ids = _schema_archive_session_ids_for_source_path(conn, path, archive_root=archive_root)
            pending_ids = _archive_pending_embedding_session_ids(conn, session_ids, archive_root=archive_root)
        finally:
            conn.close()
        if not pending_ids:
            return True
        return _embed_archive_sessions_sync(
            db_path, pending_ids, archive_root=archive_root
        ) and not _archive_embedding_debt_remaining(db_path, session_ids, archive_root=archive_root)
    except Exception:
        logger.warning("embed: archive failed", exc_info=True)
        return False


def _archive_embed_check_many(db_path: Path, paths: Sequence[Path], *, archive_root: Path | None = None) -> set[Path]:
    try:
        _reconcile_archive_embedding_config_change(db_path, archive_root=archive_root)
        conn = sqlite3.connect(db_path, timeout=5.0)
        try:
            by_path = _schema_archive_session_ids_for_source_paths(conn, paths, archive_root=archive_root)
            all_ids = list(dict.fromkeys(session_id for ids in by_path.values() for session_id in ids))
            pending = set(_archive_pending_embedding_session_ids(conn, all_ids, archive_root=archive_root))
            return {path for path, ids in by_path.items() if any(session_id in pending for session_id in ids)}
        finally:
            conn.close()
    except Exception:
        logger.warning(
            "convergence freshness probe %s errored; treating as needs-work", "_archive_embed_check_many", exc_info=True
        )
        return set(paths)


def _archive_embed_execute_many(db_path: Path, paths: Sequence[Path], *, archive_root: Path | None = None) -> bool:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        try:
            by_path = _schema_archive_session_ids_for_source_paths(conn, paths, archive_root=archive_root)
            session_ids = list(dict.fromkeys(session_id for ids in by_path.values() for session_id in ids))
            pending_ids = _archive_pending_embedding_session_ids(conn, session_ids, archive_root=archive_root)
        finally:
            conn.close()
        if not pending_ids:
            return True
        return _embed_archive_sessions_sync(
            db_path, pending_ids, archive_root=archive_root
        ) and not _archive_embedding_debt_remaining(db_path, session_ids, archive_root=archive_root)
    except Exception:
        logger.warning("embed: archive batch failed", exc_info=True)
        return False


def _archive_embed_check_sessions(
    db_path: Path, session_ids: Sequence[str], *, archive_root: Path | None = None
) -> set[str]:
    try:
        _reconcile_archive_embedding_config_change(db_path, archive_root=archive_root)
        conn = sqlite3.connect(db_path, timeout=5.0)
        try:
            ids = _archive_existing_session_ids(conn, session_ids)
            return set(_archive_pending_embedding_session_ids(conn, ids, archive_root=archive_root))
        finally:
            conn.close()
    except Exception:
        logger.warning(
            "convergence freshness probe %s errored; treating as needs-work",
            "_archive_embed_check_sessions",
            exc_info=True,
        )
        return set(session_ids)


def _archive_embed_execute_sessions(
    db_path: Path, session_ids: Sequence[str], *, archive_root: Path | None = None
) -> bool:
    ids = tuple(dict.fromkeys(str(session_id) for session_id in session_ids if session_id))
    if not ids:
        return True
    ok = _embed_archive_sessions_sync(db_path, ids, archive_root=archive_root)
    return ok and not _archive_embedding_debt_remaining(db_path, ids, archive_root=archive_root)


def _archive_embedding_debt_remaining(
    db_path: Path, session_ids: Sequence[str], *, archive_root: Path | None = None
) -> bool:
    if not session_ids:
        return False
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        try:
            return bool(_archive_pending_embedding_session_ids(conn, session_ids, archive_root=archive_root))
        finally:
            conn.close()
    except Exception:
        logger.warning("embed: archive failed to check remaining debt", exc_info=True)
        return True


def _embed_archive_sessions_sync(
    db_path: Path, session_ids: Sequence[str], *, archive_root: Path | None = None
) -> bool:
    from polylogue.storage.embeddings.materialization import embed_archive_session_sync
    from polylogue.storage.search_providers import create_vector_provider

    cfg = load_polylogue_config()
    voyage_key = cfg.get("voyage_api_key")
    if not voyage_key:
        return True
    root = archive_root if archive_root is not None else db_path.parent
    embeddings_db = root / "embeddings.db"
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
    archive_root: Path | None = None,
) -> set[str]:
    unique_ids = tuple(dict.fromkeys(str(session_id) for session_id in session_ids if session_id))
    if not unique_ids or not _table_exists(conn, "sessions"):
        return set()
    raw_table = "raw_sessions"
    if not _table_exists(conn, "raw_sessions"):
        raw_table = "source_tier.raw_sessions"
        try:
            if not _ensure_source_tier_attached(conn, archive_root=archive_root):
                return set()
        except sqlite3.Error:
            logger.warning("archive convergence: failed to attach source tier", exc_info=True)
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
    stale_predicate = session_profile_stale_predicate("s", "sp")
    rows = conn.execute(
        f"""
        SELECT s.session_id
        FROM sessions AS s
        LEFT JOIN session_profiles AS sp ON sp.session_id = s.session_id
        LEFT JOIN insight_materialization AS im
          ON im.session_id = s.session_id AND im.insight_type = 'session_profile'
        LEFT JOIN insight_materialization AS pu
          ON pu.session_id = s.session_id AND pu.insight_type = 'provider_usage'
        WHERE s.session_id IN ({placeholders})
          AND (
              sp.session_id IS NULL
              OR sp.materializer_version != ?
              OR im.materializer_version != ?
              OR pu.session_id IS NULL
              OR pu.materializer_version != ?
              OR {stale_predicate}
          )
        ORDER BY s.session_id
        """,
        unique_ids
        + (
            SESSION_INSIGHT_MATERIALIZER_VERSION,
            SESSION_INSIGHT_MATERIALIZER_VERSION,
            SESSION_INSIGHT_MATERIALIZER_VERSION,
        ),
    ).fetchall()
    return [str(row[0]) for row in rows]


def _schema_archive_session_ids_missing_profiles(conn: sqlite3.Connection, *, limit: int | None = None) -> list[str]:
    if not _table_exists(conn, "sessions") or not _table_exists(conn, "session_profiles"):
        return []
    # The OR chain below also catches provider-usage staleness (polylogue-f2qv.5):
    # a session whose session_profile is fresh but whose session_model_usage
    # rollup predates a materializer-version bump (or was never stamped) still
    # needs a rebuild pass so it self-heals like every other session insight
    # instead of requiring a manual `ops reset --index`.
    stale_predicate = session_profile_stale_predicate("s", "sp")
    sql = f"""
        SELECT s.session_id
        FROM sessions AS s
        LEFT JOIN session_profiles AS sp ON sp.session_id = s.session_id
        LEFT JOIN insight_materialization AS im
          ON im.session_id = s.session_id AND im.insight_type = 'session_profile'
        LEFT JOIN insight_materialization AS pu
          ON pu.session_id = s.session_id AND pu.insight_type = 'provider_usage'
        WHERE
          sp.session_id IS NULL
          OR sp.materializer_version != ?
          OR im.materializer_version != ?
          OR pu.session_id IS NULL
          OR pu.materializer_version != ?
          OR {stale_predicate}
        ORDER BY s.session_id
    """
    params: tuple[object, ...] = (
        SESSION_INSIGHT_MATERIALIZER_VERSION,
        SESSION_INSIGHT_MATERIALIZER_VERSION,
        SESSION_INSIGHT_MATERIALIZER_VERSION,
    )
    if limit is not None:
        sql += " LIMIT ?"
        params = params + (max(0, int(limit)),)
    rows = conn.execute(sql, params).fetchall()
    return [str(row[0]) for row in rows]


def _archive_insights_check(db_path: Path, path: Path, *, archive_root: Path | None = None) -> bool:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        try:
            session_ids = _schema_archive_session_ids_for_source_path(conn, path, archive_root=archive_root)
            return bool(session_ids) and bool(_archive_stale_session_profile_ids(conn, session_ids))
        finally:
            conn.close()
    except Exception:
        logger.warning(
            "convergence freshness probe %s errored; treating as needs-work", "_archive_insights_check", exc_info=True
        )
        return True


def _archive_insights_execute(db_path: Path, path: Path, *, archive_root: Path | None = None) -> StageExecuteReturn:
    try:
        conn = _open_archive_insight_write_connection(db_path)
        try:
            session_ids = _schema_archive_session_ids_for_source_path(conn, path, archive_root=archive_root)
            if not session_ids:
                logger.info("insights: archive skipped path refresh with no resolved sessions path=%s", path)
                return True
            return _archive_insights_execute_ids(conn, session_ids, archive_root=archive_root)
        finally:
            conn.close()
    except Exception as exc:
        if _is_transient_sqlite_lock(exc):
            logger.info("insights: archive refresh deferred because sqlite is busy: %s", exc)
            return False
        logger.warning("insights: archive refresh failed", exc_info=True)
        return False


def _archive_insights_check_many(
    db_path: Path, paths: Sequence[Path], *, archive_root: Path | None = None
) -> set[Path]:
    if not paths:
        return set()
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        try:
            by_path = _schema_archive_session_ids_for_source_paths(conn, paths, archive_root=archive_root)
            result = {
                path
                for path, session_ids in by_path.items()
                if session_ids and _archive_stale_session_profile_ids(conn, session_ids)
            }
            return result
        finally:
            conn.close()
    except Exception:
        logger.warning(
            "convergence freshness probe %s errored; treating as needs-work",
            "_archive_insights_check_many",
            exc_info=True,
        )
        return set(paths)


def _archive_insights_execute_many(
    db_path: Path, paths: Sequence[Path], *, archive_root: Path | None = None
) -> StageExecuteReturn:
    try:
        conn = _open_archive_insight_write_connection(db_path)
        try:
            by_path = _schema_archive_session_ids_for_source_paths(conn, paths, archive_root=archive_root)
            session_ids = list(dict.fromkeys(session_id for ids in by_path.values() for session_id in ids))
            if not session_ids:
                logger.info(
                    "insights: archive skipped batch path refresh with no resolved sessions paths=%d", len(paths)
                )
                return True
            return _archive_insights_execute_ids(conn, session_ids, archive_root=archive_root)
        finally:
            conn.close()
    except Exception as exc:
        if _is_transient_sqlite_lock(exc):
            logger.info("insights: archive batch refresh deferred because sqlite is busy: %s", exc)
            return False
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
        logger.warning(
            "convergence freshness probe %s errored; treating as needs-work",
            "_archive_insights_check_sessions",
            exc_info=True,
        )
        return set(session_ids)


def _archive_insights_execute_sessions(
    db_path: Path, session_ids: Sequence[str], *, archive_root: Path | None = None
) -> StageExecuteReturn:
    try:
        conn = _open_archive_insight_write_connection(db_path)
        try:
            ids = _archive_existing_session_ids(conn, session_ids)
            return _archive_insights_execute_ids(conn, ids, archive_root=archive_root)
        finally:
            conn.close()
    except Exception as exc:
        if _is_transient_sqlite_lock(exc):
            logger.info("insights: archive session refresh deferred because sqlite is busy: %s", exc)
            return False
        logger.warning("insights: archive session refresh failed", exc_info=True)
        return False


def _archive_insights_execute_ids(
    conn: sqlite3.Connection, session_ids: Sequence[str], *, archive_root: Path | None = None
) -> StageExecuteReturn:
    from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync

    session_ids = list(dict.fromkeys(str(session_id) for session_id in session_ids if session_id))
    if not session_ids:
        return True
    hot_ids = _archive_hot_insight_session_ids(conn, session_ids, archive_root=archive_root)
    if hot_ids:
        logger.info(
            "insights: deferring hot archive source rebuild sessions=%d quiet_s=%.0f",
            len(hot_ids),
            _HOT_INSIGHT_QUIET_SECONDS,
        )
        session_ids = [session_id for session_id in session_ids if session_id not in hot_ids]
        if not session_ids:
            return False
    # The canonical rebuild function requires row-factory access on the
    # connection (name-based column reads throughout). The archive callers
    # use plain sqlite3.connect() without row_factory, so set it here.
    conn.row_factory = sqlite3.Row
    stage_timings_s: dict[str, float] = {}
    counts = rebuild_session_insights_sync(
        conn,
        session_ids=list(session_ids),
        page_size=_DAEMON_INSIGHT_REBUILD_PAGE_SIZE,
        stage_timings_s=stage_timings_s,
        stage_timing_prefix="insights",
    )
    # The rebuild commits its own rows. Publish and commit the final exact FTS
    # state in the same production stage before reporting success.
    _record_fts_freshness_after_insights(conn)
    conn.commit()
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
    return StageExecutionResult(success=not hot_ids and not remaining, stage_timings_s=stage_timings_s)


__all__ = [
    "make_claude_workflow_stage",
    "make_default_convergence_stages",
    "make_embed_stage",
    "make_fts_stage",
    "make_insights_stage",
    "make_raw_authority_verdict_cache_stage",
    "make_raw_parse_recovery_stage",
    "make_sinex_publication_stage",
    "make_standing_query_stage",
]
