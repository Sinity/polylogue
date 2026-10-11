"""One global Claude workflow demand and exact admitted retry ownership."""

from __future__ import annotations

import sqlite3
import time
import uuid
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from polylogue.analysis.claude_workflow_materializer import ClaudeWorkflowAuthority

from polylogue.core.compute import DaemonOperationCancelled
from polylogue.core.product_retry import ProductAttempt, ProductObservation, ProductOutcome
from polylogue.core.stage_admission import admit_stage_write
from polylogue.daemon.convergence import ConvergenceStage, StageExecutionResult
from polylogue.storage.sqlite.archive_tiers.bootstrap import open_initialized_tier_connection
from polylogue.storage.sqlite.archive_tiers.ops_write import record_daemon_stage_event
from polylogue.storage.sqlite.archive_tiers.product_retry import (
    admit_product_retry,
    observe_product_retry,
    product_attempt_current,
    settle_product_retry,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_T = TypeVar("_T")

_PRODUCT = "claude_workflow"
_TARGET = "current"
_OWNER = str(uuid.uuid4())
# Physical operation liveness only; the existing Ops debt row owns demand.
_ACTIVE_ATTEMPTS: set[str] = set()


class StaleWorkflowAttemptError(RuntimeError):
    """Prepared authority or the exact admitted Ops attempt was superseded."""


def _ops_write(root: Path, operation: Callable[[sqlite3.Connection], _T]) -> _T:
    def write() -> _T:
        with (
            closing(open_initialized_tier_connection(root / "ops.db", ArchiveTier.OPS, archive_root=root)) as conn,
            conn,
        ):
            return operation(conn)

    return admit_stage_write("stage.claude_workflow.retry", write)


def _receipt(conn: sqlite3.Connection, *, status: str, payload: dict[str, object], now_ms: int) -> None:
    record_daemon_stage_event(
        conn,
        stage=_PRODUCT,
        status=status,
        observed_at_ms=now_ms,
        payload=payload,
        event_id=f"{_PRODUCT}:current",
        commit=False,
    )


def _summary_payload(summary: object) -> dict[str, object]:
    gaps = tuple(getattr(summary, "gaps", ()))
    return {
        **{
            key: getattr(summary, key, 0)
            for key in ("run_count", "call_count", "attempt_count", "linked_session_count", "unresolved_call_count")
        },
        "gap_count": len(gaps),
        "gaps": list(gaps[:20]),
        "graph_ids": list(getattr(summary, "graph_ids", ())),
    }


def claude_workflow_receipt_current(root: Path, status: dict[str, object]) -> bool:
    """Compare exact current canonical inputs and the produced graph manifest."""
    from polylogue.analysis.claude_workflow_materializer import read_claude_workflow_authority

    authority = read_claude_workflow_authority(root, include_graph_refs=True)
    graph_ids = status.get("graph_ids")
    if not isinstance(graph_ids, list) or not all(isinstance(item, str) for item in graph_ids):
        return False
    identity = authority.corpus_snapshot_ref.format()
    return status.get("demand_identity") == identity and authority.existing_graph_refs == dict.fromkeys(
        graph_ids, identity
    )


def make_claude_workflow_stage(
    db_path: Path, *, now_ms: Callable[[], int] = lambda: int(time.time() * 1000), owner: str = _OWNER
) -> ConvergenceStage:
    """Inspect on every relevant pass; only an eligible global attempt starts."""
    root = db_path.parent

    def observe() -> tuple[ProductObservation, ClaudeWorkflowAuthority]:
        from polylogue.analysis.claude_workflow_materializer import read_claude_workflow_authority

        authority = read_claude_workflow_authority(root, include_graph_refs=True)
        identity = authority.corpus_snapshot_ref.format()

        def record(conn: sqlite3.Connection) -> ProductObservation:
            from polylogue.core.json import loads

            row = conn.execute(
                "SELECT payload_json,status FROM daemon_stage_events WHERE event_id=?", (f"{_PRODUCT}:current",)
            ).fetchone()
            payload = loads(row[0]) if row else {}
            graph_ids = payload.get("graph_ids") if isinstance(payload, dict) else None
            manifest = (
                dict.fromkeys(graph_ids, identity)
                if isinstance(graph_ids, list) and all(isinstance(item, str) for item in graph_ids)
                else None
            )
            certified = (
                bool(row and row[1] in {"clean", "gaps"})
                and isinstance(payload, dict)
                and payload.get("demand_identity") == identity
                and manifest is not None
                and manifest == authority.existing_graph_refs
            )
            current = ProductObservation(identity, not certified, certified and not manifest)
            observe_product_retry(conn, product=_PRODUCT, target=_TARGET, observation=current, now_ms=now_ms())
            return current

        return _ops_write(root, record), authority

    def check(_path: Path) -> bool:
        observation, _authority = observe()
        return observation.needs_work

    def execute(_path: Path) -> StageExecutionResult:
        from polylogue.analysis.claude_workflow_materializer import materialize_claude_workflow_archive

        observation, authority = observe()
        if not observation.needs_work:
            return StageExecutionResult(
                True, product_outcome=ProductOutcome.VALID_EMPTY if observation.valid_empty else ProductOutcome.CURRENT
            )

        def admit(conn: sqlite3.Connection) -> ProductAttempt | None:
            admitted = admit_product_retry(
                conn,
                product=_PRODUCT,
                target=_TARGET,
                demand_identity=observation.demand_identity,
                owner=owner,
                now_ms=now_ms(),
                active_attempts=frozenset(_ACTIVE_ATTEMPTS),
            )
            if admitted is not None:
                _ACTIVE_ATTEMPTS.add(admitted.attempt_id)
            return admitted

        attempt = _ops_write(root, admit)
        if attempt is None:
            return StageExecutionResult(False, product_outcome=ProductOutcome.BLOCKED)

        def guard(prepared_identity: str) -> None:
            from polylogue.analysis.claude_workflow_materializer import read_claude_workflow_authority

            if (
                prepared_identity != attempt.demand_identity
                or read_claude_workflow_authority(root).corpus_snapshot_ref.format() != prepared_identity
            ):
                raise StaleWorkflowAttemptError("workflow authority changed before publication")
            if not _ops_write(root, lambda conn: product_attempt_current(conn, attempt)):
                raise StaleWorkflowAttemptError("workflow retry attempt was superseded")

        def settle(
            outcome: ProductOutcome, *, summary: object | None = None, error: BaseException | None = None
        ) -> bool:
            def write(conn: sqlite3.Connection) -> bool:
                if not product_attempt_current(conn, attempt):
                    return False
                if summary is not None:
                    guard(attempt.demand_identity)
                    payload = {
                        **_summary_payload(summary),
                        "demand_identity": attempt.demand_identity,
                        "attempt_id": attempt.attempt_id,
                    }
                    _receipt(conn, status="gaps" if payload["gap_count"] else "clean", payload=payload, now_ms=now_ms())
                elif outcome is ProductOutcome.FAILED:
                    _receipt(
                        conn,
                        status="failed",
                        payload={
                            "error_type": type(error).__name__,
                            "error_detail": str(error),
                            "gap_count": None,
                            "demand_identity": attempt.demand_identity,
                            "attempt_id": attempt.attempt_id,
                        },
                        now_ms=now_ms(),
                    )
                return settle_product_retry(
                    conn, attempt, outcome, now_ms=now_ms(), error=str(error) if error else None
                )

            return bool(_ops_write(root, write))

        try:
            summary = materialize_claude_workflow_archive(root, publication_guard=guard, authority=authority)
            settled = settle(
                ProductOutcome.VALID_EMPTY if not getattr(summary, "graph_ids", ()) else ProductOutcome.CURRENT,
                summary=summary,
            )
            return StageExecutionResult(
                settled,
                stale_preparation=not settled,
                product_outcome=(
                    ProductOutcome.VALID_EMPTY if not getattr(summary, "graph_ids", ()) else ProductOutcome.CURRENT
                )
                if settled
                else ProductOutcome.STALE,
            )
        except StaleWorkflowAttemptError as exc:
            settle(ProductOutcome.STALE, error=exc)
            return StageExecutionResult(False, stale_preparation=True, product_outcome=ProductOutcome.STALE)
        except DaemonOperationCancelled as exc:
            settle(ProductOutcome.CANCELLED, error=exc)
            raise
        except Exception as exc:
            settle(ProductOutcome.FAILED, error=exc)
            return StageExecutionResult(False, product_outcome=ProductOutcome.FAILED)
        finally:
            _ACTIVE_ATTEMPTS.discard(attempt.attempt_id)

    return ConvergenceStage(
        name=_PRODUCT,
        description="Certify global Claude workflow topology from current authority",
        check=check,
        execute=execute,
        # Batch subjects receive one global verdict, never their own retry row.
        check_many=lambda paths: set(paths) if paths and check(db_path) else set(),
        execute_many=lambda _paths: execute(db_path),
        whole_archive=True,
        subject_independent=True,
        owns_retry=True,
        false_means_pending=True,
        writer_admission="bridged",
    )
