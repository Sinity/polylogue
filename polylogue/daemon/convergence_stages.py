"""Convergence stage implementations for the daemon pipeline.

Each stage has a ``check`` that inspects current archive state and an
``execute`` that performs the missing work. The live watcher owns source
ingestion through daemon-side raw-record ingest; daemon convergence stages only
repair and refresh post-ingest archive state.

Raw, session, FTS and embedding outputs use their domain derivations.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.config import load_polylogue_config
from polylogue.core.enums import Provider
from polylogue.core.sqlite_introspection import table_exists as _table_exists
from polylogue.core.sqlite_locking import is_transient_sqlite_lock
from polylogue.daemon.convergence import ConvergenceStage, StageExecuteReturn
from polylogue.daemon.convergence_standing_queries import make_standing_query_stage
from polylogue.logging import INFO, WARNING, emit, span
from polylogue.operations.lineage_prefix_recompose import make_lineage_prefix_recompose_stage
from polylogue.operations.raw_authority_verdict_cache import (
    RawAuthorityVerdictCacheWork,
    find_raw_authority_verdict_cache_work,
    warm_raw_authority_verdict_cache,
)
from polylogue.sources.origin_specs import artifact_rule_for_path
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.sqlite.connection_profile import (
    open_daemon_connection,
    open_readonly_connection,
)

if TYPE_CHECKING:
    from polylogue.sinex.service import PublicationService
    from polylogue.sinex.transport import SinexTransport

_HOT_INSIGHT_SOURCE_BYTES = 64 * 1024 * 1024
_HOT_INSIGHT_QUIET_SECONDS = 60.0
_ARCHIVE_INSIGHT_WRITE_BUSY_TIMEOUT_MS = 120_000
_DAEMON_RAW_AUTHORITY_CACHE_MAX_COHORTS = 8


def _sinex_drain_reason(*, rejected: int, transport_failures: int, payload_failures: int) -> str:
    """Name the dominant failure lane so the reason token is not a catch-all."""
    if transport_failures:
        return "transport_failures"
    if payload_failures:
        return "payload_failures"
    if rejected:
        return "rejected_revisions"
    return "durable_debt"


def _emit_sinex_drain(scope: str, subjects: int, summary: object, *, path: Path | None = None) -> None:
    """Report one outbox drain with every failure lane kept distinct.

    Prose collapsed transport failures, payload failures and durable debt into
    one sentence; a reader of a long rebuild could not tell a transport outage
    from a malformed payload. Each stays its own counted field, and any
    non-zero failure lane makes the event a WARNING rather than an INFO.
    """
    attempted = int(getattr(summary, "attempted", 0))
    rejected = int(getattr(summary, "rejected", 0))
    transport_failures = int(getattr(summary, "transport_failures", 0))
    payload_failures = int(getattr(summary, "payload_failures", 0))
    debt = int(getattr(summary, "durable_debt", 0))
    remaining = int(getattr(summary, "remaining_lag", 0))
    failed = rejected + transport_failures + payload_failures
    if failed or debt:
        level = WARNING
        outcome = "degraded"
        reason = _sinex_drain_reason(
            rejected=rejected,
            transport_failures=transport_failures,
            payload_failures=payload_failures,
        )
    else:
        level = INFO
        outcome = "ok" if attempted else "empty"
        reason = "clean"
    # ``path`` is present only for the per-path drain; a null field on the
    # batch and session scopes would be noise in every rendered line.
    scoped: dict[str, object] = {} if path is None else {"path": path}
    emit(
        "daemon.stage.drained",
        level,
        outcome=outcome,
        reason=reason,
        stage="sinex_publication",
        action=scope,
        subjects=subjects,
        attempted=attempted,
        confirmed=int(getattr(summary, "confirmed", 0)),
        rejected=rejected,
        transport_failures=transport_failures,
        payload_failures=payload_failures,
        failed=failed,
        debt=debt,
        remaining=remaining,
        **scoped,
    )


def _is_transient_sqlite_lock(exc: BaseException) -> bool:
    """Defer to SQLite's result code; text alone misses SQLITE_LOCKED."""
    return is_transient_sqlite_lock(exc)


def _open_archive_insight_write_connection(db_path: Path, *, archive_root: Path) -> sqlite3.Connection:
    """Open an archive writer bound to the root admitted by its caller.

    ``db_path`` may be an active index generation outside the durable archive
    root.  The caller therefore supplies the admitted root rather than
    deriving it from the generation path.
    """
    conn = open_daemon_connection(
        db_path,
        timeout=_ARCHIVE_INSIGHT_WRITE_BUSY_TIMEOUT_MS / 1000,
        archive_root=archive_root,
    )
    try:
        conn.execute(f"PRAGMA busy_timeout = {_ARCHIVE_INSIGHT_WRITE_BUSY_TIMEOUT_MS}")
    except BaseException:
        conn.close()
        raise
    return conn


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
        from polylogue.storage.sqlite.archive_tiers.bootstrap import open_initialized_tier_connection
        from polylogue.storage.sqlite.archive_tiers.ops_write import record_daemon_stage_event
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        ops_db = archive_root / "ops.db"
        ops_db.parent.mkdir(parents=True, exist_ok=True)
        with open_initialized_tier_connection(ops_db, ArchiveTier.OPS) as conn:
            record_daemon_stage_event(
                conn,
                stage=CLAUDE_WORKFLOW_STAGE_NAME,
                status=status,
                observed_at_ms=int(time.time() * 1000),
                payload=payload,
                # A stable id makes this the current snapshot rather than an
                # append: every reader selects only the newest row for this
                # stage, and ``daemon_stage_events`` has no retention, so
                # letting the writer mint a fresh UUID each pass grew ops.db
                # without bound for a row nothing ever read again.
                event_id=f"{CLAUDE_WORKFLOW_STAGE_NAME}:current",
            )
    except Exception as exc:
        emit(
            "daemon.stage.event_record_failed",
            level=WARNING,
            stage="claude_workflow",
            outcome="degraded",
            reason="stage_event_not_recorded",
            status=status,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )


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
            from polylogue.analysis.claude_workflow_materializer import (
                claude_workflow_materialization_needed,
            )

            return claude_workflow_materialization_needed(archive_root())
        except FileNotFoundError:
            # No archive to materialize from is genuinely "no work", but it is the
            # one place in convergence where a swallowed exception still answers
            # "converged" -- say so rather than deciding it silently.
            emit(
                "daemon.stage.check_skipped",
                stage="claude_workflow",
                outcome="skipped",
                reason="no_archive",
                path=path,
            )
            return False

    def execute(path: Path) -> StageExecuteReturn:
        if not relevant(path):
            return True
        with span("daemon.stage.execute", stage="claude_workflow", path=path) as work:
            try:
                from polylogue.analysis.claude_workflow_materializer import materialize_claude_workflow_archive

                summary = materialize_claude_workflow_archive(archive_root())
            except Exception as exc:
                work.degraded(
                    "materialization_failed",
                    error_type=type(exc).__name__,
                    error_detail=str(exc),
                )
                return False
            gaps = len(summary.gaps)
            fields = {
                "runs": summary.run_count,
                "calls": summary.call_count,
                "attempts": summary.attempt_count,
                "gaps": gaps,
            }
            _record_claude_workflow_stage_event(archive_root(), summary)
            if gaps:
                work.degraded("unresolved_workflow_gaps", **fields)
            else:
                work.ok(**fields)
            return True

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
        whole_archive=True,
        writer_admission="bridged",
    )


# ── Stage: delegation work-evidence projection ───────────────────


def make_delegation_work_evidence_stage(db_path: Path) -> ConvergenceStage:
    """Project the canonical delegation view into the shared work graph."""

    def archive_root() -> Path:
        active_index = _active_archive_index_path(db_path)
        return (active_index or db_path).parent

    def check(path: Path) -> bool:
        del path
        try:
            from polylogue.analysis.delegation_work_evidence_materializer import (
                delegation_work_evidence_materialization_needed,
            )

            return delegation_work_evidence_materialization_needed(archive_root())
        except FileNotFoundError:
            return False
        except Exception as exc:
            # A probe that cannot answer (a locked or skewed archive, say) is
            # not evidence that there is no work: assume work and let execute
            # report the real terminal outcome, rather than propagating out of
            # the check and aborting the whole convergence pass.
            emit(
                "daemon.stage.check_failed",
                level=WARNING,
                outcome="degraded",
                stage="delegation_work_evidence",
                reason="probe_failed_assuming_work",
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            return True

    def execute(path: Path) -> StageExecuteReturn:
        del path
        with span("daemon.stage.execute", stage="delegation_work_evidence") as work:
            try:
                from polylogue.analysis.delegation_work_evidence_materializer import (
                    materialize_delegation_work_evidence_archive,
                )

                count = materialize_delegation_work_evidence_archive(archive_root())
            except Exception as exc:
                work.degraded(
                    "materialization_failed",
                    error_type=type(exc).__name__,
                    error_detail=str(exc),
                )
                return False
            if count:
                work.ok(rows=count)
            else:
                work.empty(rows=0)
            return True

    def check_many(paths: Sequence[Path]) -> set[Path]:
        return set(paths) if paths and check(next(iter(paths))) else set()

    def execute_many(paths: Sequence[Path]) -> StageExecuteReturn:
        return True if not paths else execute(paths[0])

    return ConvergenceStage(
        name="delegation_work_evidence",
        description="Project canonical delegation evidence into the generic work graph",
        check=check,
        execute=execute,
        check_many=check_many,
        execute_many=execute_many,
        whole_archive=True,
        writer_admission="bridged",
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
    conn = open_readonly_connection(lookup_db)
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
        _emit_sinex_drain("path", len(session_ids), summary, path=path)
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
        _emit_sinex_drain("batch", len(all_ids), summary)
        return not service.unresolved_object_ids(all_ids)

    def check_sessions(session_ids: Sequence[str]) -> set[str]:
        return service.unresolved_object_ids(session_ids)

    def execute_sessions(session_ids: Sequence[str]) -> StageExecuteReturn:
        if not session_ids:
            return True
        summary = service.drain_once(object_ids=session_ids, limit=service.max_batch)
        _emit_sinex_drain("sessions", len(tuple(dict.fromkeys(session_ids))), summary)
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
        writer_admission="bridged",
    )


def make_raw_authority_verdict_cache_stage(db_path: Path) -> ConvergenceStage:
    """Warm the content-keyed raw-authority verdict cache in bounded cohorts."""

    def work() -> RawAuthorityVerdictCacheWork | None:
        return find_raw_authority_verdict_cache_work(db_path.parent)

    def check(_path: Path) -> bool:
        discovered = work()
        if discovered is None:
            return False
        return bool(discovered.pending_logical_source_keys)

    def check_many(paths: Sequence[Path]) -> set[Path]:
        if not paths:
            return set()
        discovered = work()
        if discovered is None:
            return set()
        return set(paths) if discovered.pending_logical_source_keys else set()

    def execute(_path: Path) -> StageExecuteReturn:
        return execute_many((_path,))

    def execute_many(_paths: Sequence[Path]) -> StageExecuteReturn:
        with span("daemon.stage.execute", stage="raw_authority_verdict_cache", files=len(_paths)) as work:
            outcome = warm_raw_authority_verdict_cache(
                db_path.parent,
                max_cohorts=_DAEMON_RAW_AUTHORITY_CACHE_MAX_COHORTS,
                now_ms=int(time.time() * 1000),
            )
            pending = int(outcome.pending_cohorts or 0)
            if pending:
                # Backlog remains: the stage is not converged this pass, and
                # false_means_pending will schedule the retry.
                work.degraded("cohorts_still_pending", cohorts=outcome.warmed_cohorts, pending=pending)
            elif outcome.warmed_cohorts:
                work.ok(cohorts=outcome.warmed_cohorts, pending=0)
            else:
                work.empty(cohorts=0, pending=0)
            return not outcome.pending_cohorts

    return ConvergenceStage(
        name="raw_authority_verdict_cache",
        description="Warm content-keyed raw-authority verdicts for every revision cohort",
        check=check,
        execute=execute,
        check_many=check_many,
        execute_many=execute_many,
        false_means_pending=True,
        whole_archive=True,
    )


def make_fts_readiness_binding_stage(db_path: Path) -> ConvergenceStage:
    """Publish the message-FTS readiness binding once per quiet archive.

    polylogue-crwl6 AC6.  The five request paths (``/healthz``, ``/api/status``,
    ``/metrics``, ``health``, ``status_snapshot``) all reach FTS readiness
    through ``daemon.fts_status.fts_readiness_info``, which ran the
    archive-proportional global inspection on every call.  That inspection now
    has a domain-local binding to compare against -- but a binding has to be
    *published* by something holding the writer, and readiness probes are
    read-only by contract.

    This stage is that publisher, and it is deliberately the cheapest possible
    scheduler: ``check`` is a single indexed row read, so a bound archive costs
    nothing per pass.  The one authoritative inspection runs only when the
    binding is absent, which after ingest quiesces happens exactly once --
    every block write retires the binding again, so a burst does not run it
    repeatedly per row, it runs it once after the burst.
    """

    def _index_path() -> Path:
        return ArchiveLocation.resolve(db_path.parent).active_index_path

    def check(_path: Path) -> bool:
        from polylogue.operations.fts_derivation import fts_readiness_binding

        index_db = _index_path()
        if not index_db.exists():
            return False
        conn = open_readonly_connection(index_db, validate_schema=False)
        try:
            if not _table_exists(conn, "messages_fts_readiness_binding"):
                return False
            if not _table_exists(conn, "blocks") or not _table_exists(conn, "messages_fts"):
                return False
            return fts_readiness_binding(conn) is None
        finally:
            conn.close()

    def check_many(paths: Sequence[Path]) -> set[Path]:
        if not paths:
            return set()
        return set(paths) if check(paths[0]) else set()

    def execute(_path: Path) -> StageExecuteReturn:
        return execute_many((_path,))

    def execute_many(paths: Sequence[Path]) -> StageExecuteReturn:
        from polylogue.operations.fts_derivation import stamp_fts_readiness_binding

        with span("daemon.stage.execute", stage="fts_readiness_binding", files=len(paths)) as work:
            index_db = _index_path()
            if not index_db.exists():
                work.empty(bound=False, reason="no_index_tier")
                return True
            conn = open_daemon_connection(index_db, archive_root=db_path.parent)
            try:
                conn.execute("BEGIN IMMEDIATE")
                bound = stamp_fts_readiness_binding(conn)
                conn.execute("COMMIT" if bound else "ROLLBACK")
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()
            if bound:
                work.ok(bound=True)
            else:
                # Not a failure and not converged: the surface itself is not
                # valid right now, or a writer moved ``blocks`` underneath the
                # inspection. Either way readiness keeps answering from the
                # authoritative inspection and this stage retries.
                work.degraded("fts_surface_not_valid", bound=False)
            return bound

    return ConvergenceStage(
        name="fts_readiness_binding",
        description="Publish the message-FTS readiness binding from one authoritative inspection",
        check=check,
        execute=execute,
        check_many=check_many,
        execute_many=execute_many,
        false_means_pending=True,
        whole_archive=True,
    )


def make_hook_paste_enrichment_stage(db_path: Path) -> ConvergenceStage:
    """Retry hook-paste enrichment for the session subjects that recorded debt.

    The ordinary live-ingest path performs this enrichment after path
    convergence. This stage supplies the corresponding session-scoped retry
    route when that write raised. Session debt is the work predicate: replaying
    the idempotent enrichment is safe even if an earlier attempt committed
    before it failed to clear its debt row.
    """

    def check(_path: Path) -> bool:
        # Hook-paste retries are keyed by session id, not by source path.
        return False

    def execute(_path: Path) -> StageExecuteReturn:
        return True

    def check_sessions(session_ids: Sequence[str]) -> set[str]:
        return {str(session_id) for session_id in session_ids if session_id}

    def execute_sessions(session_ids: Sequence[str]) -> StageExecuteReturn:
        selected_ids = tuple(dict.fromkeys(str(session_id) for session_id in session_ids if session_id))
        if not selected_ids:
            return True
        from polylogue.operations.hook_paste_enrichment import retry_recorded_hook_paste

        # The stage engine admits this bounded, session-scoped write through
        # the daemon's writer bridge. A completed no-op is success: the hooks
        # may already have been applied or may no longer match a message.
        retry_recorded_hook_paste(db_path, selected_ids)
        return True

    return ConvergenceStage(
        name="hook_paste_enrichment",
        description="Apply durable hook paste evidence to the recorded sessions",
        check=check,
        execute=execute,
        check_sessions=check_sessions,
        execute_sessions=execute_sessions,
        whole_archive=False,
        writer_admission="whole_execute",
    )


def make_default_convergence_stages(
    db_path: Path,
    *,
    sinex_transport: SinexTransport | None = None,
) -> tuple[ConvergenceStage, ...]:
    """Build daemon stages, failing explicitly when backed mode lacks transport."""
    from polylogue.archive.query.production_evaluator import ArchiveCanonicalPlanEvaluator
    from polylogue.paths import archive_root
    from polylogue.sinex.models import PublicationMode
    from polylogue.sinex.service import PublicationService
    from polylogue.sinex.transport import resolve_configured_transport

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
            make_raw_authority_verdict_cache_stage(db_path),
            _make_attachment_bytes_stage(db_path, archive_root=archive_root()),
            make_claude_workflow_stage(db_path),
            make_delegation_work_evidence_stage(db_path),
            # The owner of the lineage-prefix losses ``archive_tiers/write.py``
            # records as convergence debt. Without it registered here the drain
            # skips every such row as an unimplemented stage and the backlog
            # never clears (polylogue-ia88n).
            make_lineage_prefix_recompose_stage(db_path),
            # polylogue-crwl6 AC6: the only production writer of the message-FTS
            # readiness binding the five status request paths compare against.
            make_fts_readiness_binding_stage(db_path),
            # Session-profile publication is no longer a generic stage.  The
            # daemon's typed session owner runs it through the derivation
            # kernel after ingest and from its no-hint periodic sweep.
            make_standing_query_stage(db_path, evaluator=ArchiveCanonicalPlanEvaluator(db_path)),
        )
    )
    return tuple(stages)


def _make_attachment_bytes_stage(db_path: Path, *, archive_root: Path) -> ConvergenceStage:
    """Construct the Drive attachment owner lazily with the normal config."""
    from polylogue.operations.attachment_convergence import make_configured_attachment_convergence_stage

    return make_configured_attachment_convergence_stage(
        db_path,
        archive_root=archive_root,
    )


# ── Helpers ────────────────────────────────────────────────────────


def _source_path_is_hot_for_insights(path: Path, *, now: float | None = None) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    if stat.st_size < _HOT_INSIGHT_SOURCE_BYTES:
        return False
    current = time.time() if now is None else now
    return current - stat.st_mtime < _HOT_INSIGHT_QUIET_SECONDS


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

    index_db = ArchiveLocation.resolve(db_path.parent).active_index_path
    if not index_db.exists():
        return None
    try:
        conn = open_readonly_connection(index_db)
        try:
            return index_db if _table_exists(conn, "sessions") else None
        finally:
            conn.close()
    except Exception as exc:
        emit(
            "daemon.archive.index_probe_failed",
            level=WARNING,
            outcome="degraded",
            reason="active_index_unreadable",
            path=index_db,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        return None


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
        # every caller of this helper (_archive_embed_check[_many] and
        # _sinex_session_ids_for_paths) wraps
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
        except sqlite3.Error as exc:
            emit(
                "daemon.archive.source_tier_attach_failed",
                level=WARNING,
                outcome="degraded",
                reason="hot_insight_probe_unavailable",
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
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


__all__ = [
    "make_claude_workflow_stage",
    "make_delegation_work_evidence_stage",
    "make_default_convergence_stages",
    "make_hook_paste_enrichment_stage",
    "make_lineage_prefix_recompose_stage",
    "make_raw_authority_verdict_cache_stage",
    "make_sinex_publication_stage",
    "make_standing_query_stage",
]
