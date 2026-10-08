"""Convergence stage implementations for the daemon pipeline.

Each stage has a ``check`` that inspects current archive state and an
``execute`` that performs the missing work. The live watcher owns source
ingestion through daemon-side raw-record ingest; daemon convergence stages only
derive and refresh post-ingest archive state.

Raw, session, FTS and embedding outputs use their domain derivations.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.config import load_polylogue_config
from polylogue.daemon.convergence import ConvergenceStage, StageExecuteReturn
from polylogue.daemon.convergence_standing_queries import make_standing_query_stage
from polylogue.logging import INFO, WARNING, emit, span
from polylogue.operations.claude_workflow_convergence import make_claude_workflow_stage
from polylogue.operations.lineage_prefix_recompose import make_lineage_prefix_recompose_stage
from polylogue.operations.raw_authority_verdict_cache import (
    RawAuthorityVerdictCacheWork,
    find_raw_authority_verdict_cache_work,
    warm_raw_authority_verdict_cache,
)
from polylogue.operations.raw_existence_journal import make_raw_existence_journal_prune_stage
from polylogue.operations.raw_frontier_inspection import make_raw_frontier_inspection_stage
from polylogue.operations.session_source_membership import session_ids_for_paths
from polylogue.operations.sinex_convergence import publication_service_for_archive

if TYPE_CHECKING:
    from polylogue.core.compute import BoundedComputeAdapter
    from polylogue.sinex.service import PublicationService
    from polylogue.sinex.transport import SinexTransport

_DAEMON_RAW_AUTHORITY_CACHE_MAX_COHORTS = 8
#: One stage execution keeps warming bounded cohort batches while each batch
#: makes progress, up to this much wall time. A single batch per execution
#: left a fresh archive's cohorts warming eight at a time behind the debt
#: backoff: hours for a full corpus.
_DAEMON_RAW_AUTHORITY_CACHE_PASS_SECONDS = 10.0


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


# ── Stage: delegation work-evidence projection ───────────────────


def make_delegation_work_evidence_stage(db_path: Path) -> ConvergenceStage:
    """Project the canonical delegation view into the shared work graph."""

    def archive_root() -> Path:
        # The stage anchor is rooted beside the durable tiers. Following the
        # active-index pointer here would mistake a promoted generation's
        # private directory for the archive root.
        return db_path.parent

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
        subject_independent=True,
        writer_admission="bridged",
    )


def make_sinex_publication_stage(
    db_path: Path,
    service: PublicationService,
) -> ConvergenceStage:
    """Drain the durable source-tier outbox before primary projections advance."""
    from polylogue.sinex.models import PublicationMode

    def ids_for_path(path: Path) -> list[str]:
        return session_ids_for_paths(db_path, (path,)).get(path, [])

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
        by_path = session_ids_for_paths(db_path, paths)
        all_ids = tuple(dict.fromkeys(session_id for values in by_path.values() for session_id in values))
        unresolved = service.unresolved_object_ids(all_ids)
        return {path for path, values in by_path.items() if unresolved.intersection(values)}

    def execute_many(paths: Sequence[Path]) -> StageExecuteReturn:
        by_path = session_ids_for_paths(db_path, paths)
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
        by_path = session_ids_for_paths(db_path, paths)
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
            deadline = time.monotonic() + _DAEMON_RAW_AUTHORITY_CACHE_PASS_SECONDS
            warmed = 0
            while True:
                outcome = warm_raw_authority_verdict_cache(
                    db_path.parent,
                    max_cohorts=_DAEMON_RAW_AUTHORITY_CACHE_MAX_COHORTS,
                    now_ms=int(time.time() * 1000),
                )
                warmed += outcome.warmed_cohorts
                # Stop on convergence, on a batch that warmed nothing (the
                # residue is not this pass's to clear), or at the budget.
                if not outcome.pending_cohorts or not outcome.warmed_cohorts or time.monotonic() >= deadline:
                    break
            pending = int(outcome.pending_cohorts or 0)
            if pending:
                # Backlog remains: the stage is not converged this pass, and
                # false_means_pending will schedule the retry.
                work.degraded("cohorts_still_pending", cohorts=warmed, pending=pending)
            elif warmed:
                work.ok(cohorts=warmed, pending=0)
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
        subject_independent=True,
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

    def check(_path: Path) -> bool:
        from polylogue.operations.fts_derivation import fts_readiness_binding_needed

        return fts_readiness_binding_needed(db_path.parent)

    def check_many(paths: Sequence[Path]) -> set[Path]:
        if not paths:
            return set()
        return set(paths) if check(paths[0]) else set()

    def execute(_path: Path) -> StageExecuteReturn:
        return execute_many((_path,))

    def execute_many(paths: Sequence[Path]) -> StageExecuteReturn:
        from polylogue.operations.fts_derivation import publish_fts_readiness_binding

        with span("daemon.stage.execute", stage="fts_readiness_binding", files=len(paths)) as work:
            bound = publish_fts_readiness_binding(db_path.parent)
            if bound is None:
                work.empty(bound=False, reason="no_index_tier")
                return True
            if bound:
                work.ok(bound=True)
            else:
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
        subject_independent=True,
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
    compute_adapter: BoundedComputeAdapter,
    sinex_transport: SinexTransport | None = None,
) -> tuple[ConvergenceStage, ...]:
    """Build daemon stages, failing explicitly when backed mode lacks transport."""
    from polylogue.archive.query.production_evaluator import ArchiveCanonicalPlanEvaluator
    from polylogue.paths import archive_root
    from polylogue.sinex.models import PublicationMode
    from polylogue.sinex.transport import resolve_configured_transport

    mode = PublicationMode.from_string(load_polylogue_config().sinex_mode)
    stages: list[ConvergenceStage] = []
    if mode is not PublicationMode.OFF:
        transport = sinex_transport if sinex_transport is not None else resolve_configured_transport()
        stages.append(
            make_sinex_publication_stage(
                db_path,
                publication_service_for_archive(
                    archive_root(),
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
            make_lineage_prefix_recompose_stage(db_path, compute_adapter=compute_adapter),
            # polylogue-crwl6 AC6: the only production writer of the message-FTS
            # readiness binding the five status request paths compare against.
            make_fts_readiness_binding_stage(db_path),
            make_raw_frontier_inspection_stage(db_path, compute_adapter=compute_adapter),
            make_raw_existence_journal_prune_stage(db_path, compute_adapter=compute_adapter),
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


__all__ = [
    "make_delegation_work_evidence_stage",
    "make_default_convergence_stages",
    "make_hook_paste_enrichment_stage",
    "make_lineage_prefix_recompose_stage",
    "make_raw_authority_verdict_cache_stage",
    "make_sinex_publication_stage",
    "make_standing_query_stage",
]
