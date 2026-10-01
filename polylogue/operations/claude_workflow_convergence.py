"""Claude workflow materialization and its current readiness receipt.

The product binds a pass's graph publication to a disposable ops receipt,
invalidates failed passes, and prevents an older failed attempt from replacing
a newer published result. Both writes retain the existing stage admission.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Sequence
from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.daemon.convergence import ConvergenceStage, StageExecuteReturn
from polylogue.logging import WARNING, emit, span
from polylogue.sources.origin_specs import artifact_rule_for_path

_CLAUDE_WORKFLOW_RECORDED_GAP_LIMIT = 20


def record_claude_workflow_stage_event(
    archive_root: Path, summary: object, *, started_at_ns: int | None = None
) -> None:
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
    _write_claude_workflow_stage_event(
        archive_root, status="gaps" if gaps else "clean", payload=payload, started_at_ns=started_at_ns
    )


def record_claude_workflow_failure_event(
    archive_root: Path, exc: BaseException, *, started_at_ns: int | None = None
) -> None:
    """Invalidate the recorded receipt when rematerialization itself failed.

    The receipt carries the stable id ``claude_workflow:current``, so a clean
    row from an earlier pass stays the latest event until something replaces
    it. Returning from the failure branch without writing therefore left
    ``_claude_workflow_materialization_check`` reporting OK on the strength of
    a receipt the current graph no longer matches -- the archive is failing to
    converge and readiness says it is healthy. Record the attempt's typed
    failure instead; the readiness check refuses a ``failed`` receipt rather
    than reading a ``gap_count`` that this pass never computed.
    """
    _write_claude_workflow_stage_event(
        archive_root,
        status="failed",
        payload={
            "error_type": type(exc).__name__,
            "error_detail": str(exc),
            # No gap tuple exists: the materialization that would have produced
            # one is the thing that failed. Declaring the absence keeps a reader
            # from treating a missing key as "zero gaps".
            "gap_count": None,
        },
        started_at_ns=started_at_ns,
    )


def _newer_claude_workflow_receipt(conn: sqlite3.Connection, event_id: str, attempt_started_at_ns: int) -> bool:
    """Whether the stored receipt comes from a pass that started no earlier than this one.

    Start times are nanosecond wall-clock readings; a tie counts as newer, so
    a failed pass never displaces a receipt it cannot prove it postdates.
    """
    from polylogue.core.json import loads

    row = conn.execute("SELECT payload_json FROM daemon_stage_events WHERE event_id = ?", (event_id,)).fetchone()
    if row is None or not row[0]:
        return False
    stored = loads(row[0])
    stored_started = stored.get("attempt_started_at_ns") if isinstance(stored, dict) else None
    return isinstance(stored_started, int) and stored_started >= attempt_started_at_ns


def _write_claude_workflow_stage_event(
    archive_root: Path, *, status: str, payload: dict[str, object], started_at_ns: int | None = None
) -> None:
    """Replace the claude_workflow stage receipt with this pass's outcome.

    The receipt is one row, and a write can be queued behind the writer lease
    while a later pass completes. Each receipt records when its pass started.
    A *failed* pass's write is dropped when the stored receipt comes from a
    pass that started later, so an older failure cannot overwrite a newer
    rematerialization. A successful pass always writes: its receipt follows
    its own publication, so the last published graph keeps the last word.
    """
    attempt_started_at_ns = time.time_ns() if started_at_ns is None else started_at_ns
    payload = {**payload, "attempt_started_at_ns": attempt_started_at_ns}
    try:
        from polylogue.core.stage_admission import admit_stage_write
        from polylogue.storage.archive_readiness import CLAUDE_WORKFLOW_STAGE_NAME
        from polylogue.storage.sqlite.archive_tiers.bootstrap import open_initialized_tier_connection
        from polylogue.storage.sqlite.archive_tiers.ops_write import record_daemon_stage_event
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        ops_db = archive_root / "ops.db"
        ops_db.parent.mkdir(parents=True, exist_ok=True)

        def record() -> None:
            with open_initialized_tier_connection(ops_db, ArchiveTier.OPS, archive_root=archive_root) as conn:
                if status == "failed" and _newer_claude_workflow_receipt(
                    conn, f"{CLAUDE_WORKFLOW_STAGE_NAME}:current", attempt_started_at_ns
                ):
                    return
                record_daemon_stage_event(
                    conn,
                    stage=CLAUDE_WORKFLOW_STAGE_NAME,
                    status=status,
                    observed_at_ms=int(time.time() * 1000),
                    payload=payload,
                    # A stable id makes this the current snapshot rather than
                    # an append: every reader selects only the newest row for
                    # this stage, and ``daemon_stage_events`` has no retention,
                    # so letting the writer mint a fresh UUID each pass grew
                    # ops.db without bound for a row nothing ever read again.
                    event_id=f"{CLAUDE_WORKFLOW_STAGE_NAME}:current",
                )

        # The stage is ``bridged``: its engine runs off the writer lease (the
        # convergence-debt retry calls it directly), so this ops write must be
        # admitted like the materializer's own publication.
        admit_stage_write("stage.claude_workflow.record", record)
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
        # The stage anchor names the configured root; an index generation
        # does not own durable source authority or disposable ops receipts.
        return db_path.parent

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
            started_at_ns = time.time_ns()
            try:
                from polylogue.analysis.claude_workflow_materializer import materialize_claude_workflow_archive

                summary = materialize_claude_workflow_archive(archive_root())
            except Exception as exc:
                # Invalidate the receipt before returning: an earlier clean row
                # is still the latest event otherwise, and readiness would keep
                # reporting OK while convergence fails (see
                # ``record_claude_workflow_failure_event``).
                record_claude_workflow_failure_event(archive_root(), exc, started_at_ns=started_at_ns)
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
            record_claude_workflow_stage_event(archive_root(), summary, started_at_ns=started_at_ns)
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
