"""Run payload persistence and latest-run reads."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from uuid import uuid4

from typing_extensions import TypedDict

from polylogue.lib.json import JSONDocument
from polylogue.pipeline.run_support import write_run_json
from polylogue.storage.backends import create_backend
from polylogue.storage.backends.queries import runs as runs_q
from polylogue.storage.run_state import DriftBucketPayload, RunCountsPayload, RunDrift, RunDriftPayload, RunResult
from polylogue.storage.store import RunRecord

if TYPE_CHECKING:
    from polylogue.config import Config
    from polylogue.lib.metrics import PipelineMetrics
    from polylogue.pipeline.run_stages import IndexStageOutcome
    from polylogue.pipeline.run_state import RunExecutionState
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository
    from polylogue.storage.run_state import PlanResult


class RunPayload(TypedDict):
    run_id: str
    timestamp: int
    counts: RunCountsPayload
    drift: RunDriftPayload
    indexed: bool
    index_error: str | None
    duration_ms: int
    metrics: JSONDocument


def _counts_document(payload: RunCountsPayload) -> JSONDocument:
    document: JSONDocument = {}
    for key, value in payload.items():
        if isinstance(value, int):
            document[key] = value
    return document


def _drift_bucket_document(payload: DriftBucketPayload) -> JSONDocument:
    return {
        "conversations": payload["conversations"],
        "messages": payload["messages"],
        "attachments": payload["attachments"],
    }


def _drift_document(payload: RunDriftPayload) -> JSONDocument:
    return {
        "new": _drift_bucket_document(payload["new"]),
        "removed": _drift_bucket_document(payload["removed"]),
        "changed": _drift_bucket_document(payload["changed"]),
    }


def _run_payload_document(payload: RunPayload) -> JSONDocument:
    return {
        "run_id": payload["run_id"],
        "timestamp": payload["timestamp"],
        "counts": _counts_document(payload["counts"]),
        "drift": _drift_document(payload["drift"]),
        "indexed": payload["indexed"],
        "index_error": payload["index_error"],
        "duration_ms": payload["duration_ms"],
        "metrics": payload["metrics"],
    }


def build_run_payload(
    *,
    state: RunExecutionState,
    metrics: PipelineMetrics,
    index_outcome: IndexStageOutcome,
    duration_ms: int,
) -> RunPayload:
    """Create the canonical persisted payload for a completed pipeline run."""
    drift = state.finalize()
    return {
        "run_id": uuid4().hex,
        "timestamp": int(time.time()),
        "counts": state.counts.to_payload(),
        "drift": drift.to_payload(),
        "indexed": index_outcome.indexed,
        "index_error": index_outcome.error,
        "duration_ms": duration_ms,
        "metrics": metrics.to_summary(),
    }


async def persist_run_result(
    *,
    config: Config,
    repository: ConversationRepository,
    plan: PlanResult | None,
    state: RunExecutionState,
    metrics: PipelineMetrics,
    index_outcome: IndexStageOutcome,
    duration_ms: int,
) -> RunResult:
    """Write the completed run payload to JSON and the repository run ledger."""
    run_payload = build_run_payload(
        state=state,
        metrics=metrics,
        index_outcome=index_outcome,
        duration_ms=duration_ms,
    )
    await repository.record_run(
        RunRecord(
            run_id=str(run_payload["run_id"]),
            timestamp=str(run_payload["timestamp"]),
            plan_snapshot=plan.model_dump(mode="json") if plan else None,
            counts=run_payload["counts"],
            drift=dict(run_payload["drift"]),
            indexed=index_outcome.indexed,
            duration_ms=duration_ms,
        ),
    )
    run_path = write_run_json(config.archive_root, _run_payload_document(run_payload))
    return RunResult(
        run_id=str(run_payload["run_id"]),
        counts=state.counts.model_copy(deep=True),
        drift=RunDrift.model_validate(run_payload["drift"]),
        indexed=index_outcome.indexed,
        index_error=index_outcome.error,
        duration_ms=duration_ms,
        render_failures=state.render_failures,
        run_path=str(run_path),
    )


async def latest_run(backend: SQLiteBackend | None = None) -> RunRecord | None:
    """Fetch the most recent run record from the database asynchronously."""
    owns_backend = backend is None
    active_backend = backend or create_backend()
    try:
        async with active_backend.connection() as conn:
            return await runs_q.get_latest_run(conn)
    finally:
        if owns_backend:
            await active_backend.close()


__all__ = ["build_run_payload", "latest_run", "persist_run_result"]
