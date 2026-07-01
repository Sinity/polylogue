"""Idempotent and resumable replay execution for maintenance backfills.

This is the *execute* half of the maintenance planner (issue #1147).
It turns a :class:`~polylogue.maintenance.planner.BackfillOperation`
into a sequence of per-target executions that:

* converge — running the same operation twice in a row produces no
  additional changes after the first pass converges (the underlying
  repair functions are idempotent by construction; the loop adds the
  multi-target convergence guarantee);
* resume — an interrupted operation can be re-invoked with the same
  ``operation_id`` and will pick up at the first target it had not
  completed, skipping the targets already marked done;
* isolate failures — a target that raises is recorded as a bounded
  :class:`~polylogue.maintenance.planner.FailureSample` and the
  executor continues with the remaining targets instead of aborting
  the whole operation;
* report progress — every checkpoint reports
  ``operation_id``/current target/processed-vs-total/last cursor and
  in-flight failure count via the existing structured logger and via
  the returned :class:`BackfillOperation`.

The cursor is intentionally a small opaque string (``"target:N"``)
encoding the index of the *next* target to run. That gives us the AC
requirements (resume, no duplicate work, no skipped work) without
needing storage-layer schema changes — the cursor is kept either in
memory by the caller, or persisted to a small JSON state file under
the configured archive root.

The state file is the only durable resume substrate this module
introduces; it lives alongside the archive and is removed when the
operation completes successfully.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from polylogue.config import Config
from polylogue.core.json import JSONDocument, dumps, json_document, loads
from polylogue.logging import get_logger
from polylogue.maintenance.failure_routing import resolve_maintenance_failures, route_failure_sample
from polylogue.maintenance.invalidation import InvalidationReason
from polylogue.maintenance.planner import (
    BackfillKind,
    BackfillOperation,
    BackfillStatus,
    BoundedFailureSamples,
    FailureSample,
    MaintenanceScope,
)
from polylogue.maintenance.scope import MaintenanceScopeFilter
from polylogue.maintenance.targets import (
    MAINTENANCE_TARGET_NAMES,
    MaintenanceTargetSpec,
    build_maintenance_target_catalog,
)
from polylogue.storage.repair import (
    RepairResult,
    offline_maintenance_blockers,
    repair_dangling_fts,
    repair_empty_sessions,
    repair_message_embeddings,
    repair_message_type_backfill,
    repair_orphaned_attachments,
    repair_orphaned_blobs,
    repair_orphaned_messages,
    repair_raw_materialization,
    repair_session_insights,
    repair_wal_checkpoint,
)

logger = get_logger(__name__)

#: Sentinel cursor value meaning "operation completed; nothing left to do."
CURSOR_DONE: Final[str] = "done"

#: Cursor prefix for the typed target-index encoding. The opaque string
#: ``target:N`` means "the next target to run is index N in the resolved
#: target tuple". Plain integers are reserved for future per-target
#: progress (e.g. ``target:2:rowid:9182``).
_CURSOR_TARGET_PREFIX: Final[str] = "target:"

#: Subdirectory under :attr:`Config.archive_root` used for replay state
#: files. One JSON file per ``operation_id``.
_STATE_DIRNAME: Final[str] = ".maintenance-state"


# ---------------------------------------------------------------------------
# Target dispatch
# ---------------------------------------------------------------------------


#: Type alias for repair functions that take (config, dry_run) -> RepairResult.
_RepairFn = Callable[[Config, bool], RepairResult]


#: Concrete repair dispatch table. Targets listed here are the ones the
#: PR #1147 AC names explicitly: messages FTS and session insights.
#: ``message_type_backfill`` and ``wal_checkpoint`` are
#: included because the underlying repair functions are already
#: idempotent and the executor would otherwise refuse perfectly valid
#: target names that the catalog advertises.
#:
#: Targets that are not yet wired into this dispatch raise
#: :class:`UnsupportedReplayTargetError` when requested — the planner will
#: still preview them, but :func:`execute_replay` will record the
#: missing dispatch as a typed failure sample rather than silently
#: succeeding.
_REPLAY_DISPATCH: Final[dict[str, _RepairFn]] = {
    "session_insights": repair_session_insights,
    "dangling_fts": repair_dangling_fts,
    "message_type_backfill": repair_message_type_backfill,
    "raw_materialization": repair_raw_materialization,
    "message_embeddings": repair_message_embeddings,
    "wal_checkpoint": repair_wal_checkpoint,
    "orphaned_messages": repair_orphaned_messages,
    "empty_sessions": repair_empty_sessions,
    "orphaned_attachments": repair_orphaned_attachments,
    "orphaned_blobs": repair_orphaned_blobs,
}


def supported_replay_targets() -> tuple[str, ...]:
    """Names of targets the replay executor knows how to execute.

    Stable contract for callers and tests — adding a target to
    :data:`_REPLAY_DISPATCH` extends this set.
    """
    return tuple(_REPLAY_DISPATCH)


class UnsupportedReplayTargetError(RuntimeError):
    """Raised when a resolved target has no replay dispatch entry."""


# ---------------------------------------------------------------------------
# Cursor encoding
# ---------------------------------------------------------------------------


def _encode_cursor(next_target_index: int) -> str:
    """Encode the next target index as an opaque ``target:N`` string."""
    return f"{_CURSOR_TARGET_PREFIX}{next_target_index}"


def _decode_cursor(cursor: str | None, *, total_targets: int) -> int:
    """Decode a cursor back into a next-target index.

    Returns ``0`` for an absent or empty cursor (fresh run) and
    ``total_targets`` (i.e. "all done") for :data:`CURSOR_DONE`. Any
    malformed cursor falls back to ``0`` so a corrupt state file can
    never silently skip work.
    """
    if cursor is None or cursor == "":
        return 0
    if cursor == CURSOR_DONE:
        return total_targets
    if not cursor.startswith(_CURSOR_TARGET_PREFIX):
        logger.warning("replay_cursor_unrecognized", cursor=cursor)
        return 0
    suffix = cursor[len(_CURSOR_TARGET_PREFIX) :]
    head = suffix.split(":", 1)[0]
    try:
        index = int(head)
    except ValueError:
        logger.warning("replay_cursor_invalid_integer", cursor=cursor)
        return 0
    if index < 0:
        return 0
    if index > total_targets:
        return total_targets
    return index


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


def _state_dir(config: Config) -> Path:
    return Path(config.archive_root) / _STATE_DIRNAME


def state_path_for(config: Config, operation_id: str) -> Path:
    """Path of the JSON state file for ``operation_id``."""
    return _state_dir(config) / f"{operation_id}.json"


def _write_state(path: Path, payload: JSONDocument) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(dumps(payload))
    tmp.replace(path)


def load_state(config: Config, operation_id: str) -> JSONDocument | None:
    """Load a previously persisted operation state, or ``None``."""
    path = state_path_for(config, operation_id)
    if not path.exists():
        return None
    raw = loads(path.read_text())
    if not isinstance(raw, dict):
        logger.warning(
            "replay_state_unparseable",
            operation_id=operation_id,
            path=str(path),
        )
        return None
    return raw


def clear_state(config: Config, operation_id: str) -> None:
    """Best-effort removal of the on-disk state file for an operation."""
    path = state_path_for(config, operation_id)
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning(
            "replay_state_clear_failed",
            operation_id=operation_id,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayProgress:
    """One progress checkpoint emitted by :func:`execute_replay`.

    The shape is consumer-stable: CLI and daemon surfaces format the
    same fields. ``processed`` counts targets fully attempted (regardless
    of success); ``total`` is the resolved target count. ``cursor`` is
    the cursor the executor would persist *after* this checkpoint, so
    callers can crash between checkpoints and resume cleanly from the
    last one they observed.
    """

    operation_id: str
    target: str
    processed: int
    total: int
    cursor: str
    in_flight_failures: int
    progress_amount: int | None = None
    progress_desc: str | None = None

    def to_dict(self) -> JSONDocument:
        payload: dict[str, object] = {
            "operation_id": self.operation_id,
            "target": self.target,
            "processed": self.processed,
            "total": self.total,
            "cursor": self.cursor,
            "in_flight_failures": self.in_flight_failures,
        }
        if self.progress_amount is not None:
            payload["progress_amount"] = self.progress_amount
        if self.progress_desc is not None:
            payload["progress_desc"] = self.progress_desc
        return json_document(payload)


ProgressCallback = Callable[[ReplayProgress], None]


# ---------------------------------------------------------------------------
# Replay executor
# ---------------------------------------------------------------------------


@dataclass
class _ReplayState:
    """Mutable per-run state assembled during :func:`execute_replay`."""

    operation_id: str
    targets: tuple[str, ...]
    cursor: str
    results: list[JSONDocument] = field(default_factory=list)
    failures: list[FailureSample] = field(default_factory=list)
    repaired_total: int = 0

    def progress_for(self, target: str, processed: int) -> ReplayProgress:
        return ReplayProgress(
            operation_id=self.operation_id,
            target=target,
            processed=processed,
            total=len(self.targets),
            cursor=self.cursor,
            in_flight_failures=len(self.failures),
        )


def execute_replay(
    config: Config,
    targets: Iterable[str],
    *,
    operation_id: str | None = None,
    resume_cursor: str | None = None,
    dry_run: bool = False,
    persist_state: bool = True,
    progress_callback: ProgressCallback | None = None,
    scope_filter: MaintenanceScopeFilter | None = None,
) -> BackfillOperation:
    """Execute (or resume) a backfill replay against the configured archive.

    Parameters
    ----------
    config:
        Live runtime config. Threaded through to the underlying repair
        functions; ``config.archive_root`` is also where state files
        are written when ``persist_state=True``.
    targets:
        Target names to replay. Resolved against the canonical target
        catalog; unknown names produce a ``FAILED`` operation with an
        explanatory ``error`` and an empty :attr:`BackfillOperation.targets`
        (parity with :func:`~polylogue.maintenance.planner.execute_backfill`).
    operation_id:
        Stable operation identifier. Reuse across invocations to resume
        an interrupted run; omit to mint a fresh ``uuid4``.
    resume_cursor:
        Explicit resume cursor. When ``None`` and an on-disk state file
        exists for ``operation_id``, the cursor is loaded from disk so
        operators don't need to remember the last cursor value out of
        band. Pass an explicit value to override the persisted state.
    dry_run:
        Forwarded to the underlying repair functions.
    persist_state:
        When true (the default), each completed target advances the
        on-disk cursor file under ``<archive_root>/.maintenance-state/``.
        Disable for tests or for callers that own their own state
        substrate.
    progress_callback:
        Optional callback invoked after each per-target checkpoint with
        a :class:`ReplayProgress` snapshot.

    Returns
    -------
    BackfillOperation
        Status is ``COMPLETED`` only when every resolved target returned
        ``success=True``. Any failure (raised or repair-reported)
        downgrades the operation to ``FAILED`` while still recording the
        partial results so callers can resume from the cursor.
    """

    op_id = operation_id or str(uuid.uuid4())
    catalog = build_maintenance_target_catalog()
    resolved_specs = catalog.resolve(tuple(targets))
    resolved_names = tuple(spec.name for spec in resolved_specs)
    effective_filter = scope_filter or MaintenanceScopeFilter()

    if not resolved_names:
        return BackfillOperation(
            operation_id=op_id,
            kind=BackfillKind.DERIVED_REBUILD,
            targets=(),
            status=BackfillStatus.FAILED,
            error="No valid targets resolved from input",
            scope=MaintenanceScope(targets=(), filter=effective_filter),
        )

    blockers = offline_maintenance_blockers(
        config,
        repair=any(name in _REPLAY_DISPATCH for name in resolved_names),
        cleanup=False,
        dry_run=dry_run,
        targets=resolved_names,
    )
    if blockers:
        samples = tuple(
            FailureSample(
                kind="OfflineMaintenanceBlocked",
                locator=f"target:{result.name}",
                message=result.detail,
            )
            for result in blockers
        )
        started_at = datetime.now(timezone.utc).isoformat()
        return BackfillOperation(
            operation_id=op_id,
            kind=BackfillKind.DERIVED_REBUILD,
            targets=resolved_names,
            status=BackfillStatus.FAILED,
            progress=0.0,
            started_at=started_at,
            completed_at=started_at,
            affected_rows=0,
            results=[result.to_dict() for result in blockers],
            scope=MaintenanceScope(targets=resolved_names, filter=effective_filter),
            reason=InvalidationReason.UNKNOWN,
            failure_samples=BoundedFailureSamples.from_samples(samples),
            metrics={"repaired_count": 0.0},
        )

    if resume_cursor is None and persist_state:
        persisted = load_state(config, op_id)
        if persisted is not None:
            cursor_value = persisted.get("cursor")
            if isinstance(cursor_value, str):
                resume_cursor = cursor_value

    start_index = _decode_cursor(resume_cursor, total_targets=len(resolved_names))
    state = _ReplayState(
        operation_id=op_id,
        targets=resolved_names,
        cursor=_encode_cursor(start_index),
    )

    started_at = datetime.now(timezone.utc).isoformat()
    logger.info(
        "replay_starting",
        operation_id=op_id,
        targets=resolved_names,
        start_index=start_index,
        dry_run=dry_run,
    )

    for index in range(start_index, len(resolved_names)):
        spec = resolved_specs[index]
        target_name = spec.name
        _run_one_target(
            state,
            spec,
            config,
            dry_run=dry_run,
            scope_filter=effective_filter,
            progress_callback=progress_callback,
            target_total=len(resolved_names),
            processed_before_target=index - start_index,
        )
        # Advance the cursor *after* the target completes (success or
        # failure). On failure we still advance so the next resume does
        # not re-execute the same target and stack duplicate failure
        # samples; the bounded sample list already records the problem.
        next_index = index + 1
        state.cursor = CURSOR_DONE if next_index == len(resolved_names) else _encode_cursor(next_index)
        if persist_state:
            _checkpoint_state(
                config=config,
                operation_id=op_id,
                state=state,
                started_at=started_at,
                dry_run=dry_run,
            )
        if progress_callback is not None:
            progress_callback(state.progress_for(target_name, processed=next_index - start_index))

    completed_at = datetime.now(timezone.utc).isoformat()
    successful = not state.failures
    status = BackfillStatus.COMPLETED if successful else BackfillStatus.FAILED

    progress = (len(resolved_names) - start_index) / len(resolved_names)
    logger.info(
        "replay_completed",
        operation_id=op_id,
        targets=resolved_names,
        dry_run=dry_run,
        repaired_count=state.repaired_total,
        success=successful,
        failure_samples=len(state.failures),
    )

    final = BackfillOperation(
        operation_id=op_id,
        kind=BackfillKind.DERIVED_REBUILD,
        targets=resolved_names,
        status=status,
        progress=progress,
        started_at=started_at,
        completed_at=completed_at,
        affected_rows=state.repaired_total,
        results=state.results,
        scope=MaintenanceScope(targets=resolved_names, filter=effective_filter),
        reason=InvalidationReason.UNKNOWN if state.failures else None,
        resume_cursor=state.cursor,
        failure_samples=BoundedFailureSamples.from_samples(state.failures),
        metrics={"repaired_count": float(state.repaired_total)},
    )

    if persist_state:
        if successful:
            # Success path: drop the state file. The registry's
            # default TTL prune will never see this op_id again.
            clear_state(config, op_id)
        else:
            # Failure path: write the final snapshot through the
            # checkpoint so operators can inspect a failed run via
            # the registry surface without rerunning anything.
            _checkpoint_state(
                config=config,
                operation_id=op_id,
                state=state,
                started_at=started_at,
                dry_run=dry_run,
                operation_snapshot=final,
            )

    return final


def _record_failure(
    state: _ReplayState,
    sample: FailureSample,
    *,
    target: str,
    config: Config,
) -> None:
    """Append a failure sample and route it to the daemon raw-failure surface.

    The in-memory ``state.failures`` envelope continues to back the
    returned :class:`BackfillOperation`'s
    :class:`BoundedFailureSamples`, so existing callers see the same
    shape; the side effect of :func:`route_failure_sample` is the new
    daemon-visible JSONL append handled by
    :mod:`polylogue.maintenance.failure_routing`.
    """

    state.failures.append(sample)
    route_failure_sample(
        sample,
        operation_id=state.operation_id,
        archive_root=Path(config.archive_root),
        target=target,
    )


def _run_one_target(
    state: _ReplayState,
    spec: MaintenanceTargetSpec,
    config: Config,
    *,
    dry_run: bool,
    scope_filter: MaintenanceScopeFilter,
    progress_callback: ProgressCallback | None,
    target_total: int,
    processed_before_target: int,
) -> None:
    """Execute one target, recording success or a typed failure sample."""

    target_name = spec.name
    repair_fn = _REPLAY_DISPATCH.get(target_name)
    if repair_fn is None:
        sample = FailureSample(
            kind=UnsupportedReplayTargetError.__name__,
            locator=f"target:{target_name}",
            message=(
                f"Target {target_name!r} is not yet wired into the replay dispatch table. "
                "Add an entry to polylogue.maintenance.replay._REPLAY_DISPATCH to enable it."
            ),
        )
        _record_failure(state, sample, target=target_name, config=config)
        logger.warning(
            "replay_target_unsupported",
            operation_id=state.operation_id,
            target=target_name,
        )
        return

    def _emit_target_progress(amount: int, desc: str | None = None) -> None:
        if progress_callback is None:
            return
        progress_callback(
            ReplayProgress(
                operation_id=state.operation_id,
                target=target_name,
                processed=processed_before_target,
                total=target_total,
                cursor=state.cursor,
                in_flight_failures=len(state.failures),
                progress_amount=int(amount),
                progress_desc=desc,
            )
        )

    try:
        if target_name == "session_insights" and repair_fn is repair_session_insights:
            # The session-insights repair fn understands a narrowed
            # session-id scope directly; it also emits lower-level
            # materialization progress. Forward both so one large target
            # is not silent until the final per-target checkpoint.
            result = repair_session_insights(
                config,
                dry_run,
                session_ids=scope_filter.session_ids,
                progress_callback=_emit_target_progress,
            )
        elif target_name == "raw_materialization" and repair_fn is repair_raw_materialization:
            result = repair_raw_materialization(
                config,
                dry_run,
                raw_artifact_id=scope_filter.raw_artifact_id,
                provider=scope_filter.provider,
                source_family=scope_filter.source_family,
                source_root=scope_filter.source_root,
            )
        else:
            result = repair_fn(config, dry_run)
    except (RuntimeError, sqlite3.Error) as exc:
        # Per-AC: a single bad target must not abort the rest of the
        # operation. Convert the raised exception into a typed failure
        # sample so the caller can introspect it without unwinding.
        sample = FailureSample(
            kind=type(exc).__name__,
            locator=f"target:{target_name}",
            message=str(exc),
        )
        _record_failure(state, sample, target=target_name, config=config)
        logger.exception(
            "replay_target_failed",
            operation_id=state.operation_id,
            target=target_name,
            error=str(exc),
        )
        return

    state.results.append(result.to_dict())
    state.repaired_total += result.repaired_count
    if result.success:
        resolved_kinds = (UnsupportedReplayTargetError.__name__,) if dry_run else ()
        resolve_maintenance_failures(config.archive_root, target=target_name, kinds=resolved_kinds)
    if not result.success:
        # Repair functions can report failure without raising. Surface
        # that as a typed sample too so the FAILED state carries a
        # locator and a message instead of an empty samples list.
        _record_failure(
            state,
            FailureSample(
                kind="RepairReportedFailure",
                locator=f"target:{target_name}",
                message=result.detail or "Repair returned success=False",
            ),
            target=target_name,
            config=config,
        )


def _checkpoint_state(
    *,
    config: Config,
    operation_id: str,
    state: _ReplayState,
    started_at: str,
    dry_run: bool,
    operation_snapshot: BackfillOperation | None = None,
) -> None:
    """Persist the running operation state so a kill-mid-run can resume.

    The payload carries two layers:

    * legacy top-level fields (``operation_id``/``targets``/``cursor``/
      ``started_at``/``updated_at``/``dry_run``/``repaired_count``/
      ``failure_count``/``results``) so the existing resume path keeps
      working without conditionals;
    * a full :meth:`BackfillOperation.to_dict` snapshot under the
      ``operation`` key (issue #1197) so the
      :class:`~polylogue.maintenance.registry.MaintenanceOperationRegistry`
      can rehydrate the operation envelope without re-running anything.
    """

    snapshot = operation_snapshot or _build_in_progress_snapshot(
        operation_id=operation_id,
        state=state,
        started_at=started_at,
    )
    payload = json_document(
        {
            "operation_id": operation_id,
            "targets": list(state.targets),
            "cursor": state.cursor,
            "started_at": started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "dry_run": dry_run,
            "repaired_count": state.repaired_total,
            "failure_count": len(state.failures),
            "results": list(state.results),
            "operation": snapshot.to_dict(),
        }
    )
    _write_state(state_path_for(config, operation_id), payload)


def _build_in_progress_snapshot(
    *,
    operation_id: str,
    state: _ReplayState,
    started_at: str,
) -> BackfillOperation:
    """Project the in-flight :class:`_ReplayState` onto a :class:`BackfillOperation`.

    Used by :func:`_checkpoint_state` to make sure every state file
    carries a rehydratable snapshot even mid-run (before
    :func:`execute_replay` has assembled its final return value). The
    snapshot status is :data:`BackfillStatus.RUNNING` unless the
    executor has finished all targets, in which case it surfaces as
    :data:`BackfillStatus.COMPLETED` / :data:`BackfillStatus.FAILED`
    based on the in-flight failure count.
    """

    total = len(state.targets)
    cursor = state.cursor
    if cursor == CURSOR_DONE:
        status = BackfillStatus.FAILED if state.failures else BackfillStatus.COMPLETED
        progress = 1.0
        completed_at: str | None = datetime.now(timezone.utc).isoformat()
    else:
        status = BackfillStatus.RUNNING
        processed = _decode_cursor(cursor, total_targets=total)
        progress = processed / total if total > 0 else 0.0
        completed_at = None

    return BackfillOperation(
        operation_id=operation_id,
        kind=BackfillKind.DERIVED_REBUILD,
        targets=state.targets,
        status=status,
        progress=progress,
        started_at=started_at,
        completed_at=completed_at,
        affected_rows=state.repaired_total,
        results=list(state.results),
        scope=MaintenanceScope(targets=state.targets),
        resume_cursor=cursor,
        failure_samples=BoundedFailureSamples.from_samples(state.failures),
        metrics={"repaired_count": float(state.repaired_total)},
    )


__all__ = [
    "CURSOR_DONE",
    "MAINTENANCE_TARGET_NAMES",
    "MaintenanceScopeFilter",
    "ProgressCallback",
    "ReplayProgress",
    "UnsupportedReplayTargetError",
    "clear_state",
    "execute_replay",
    "load_state",
    "state_path_for",
    "supported_replay_targets",
]
