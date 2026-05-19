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
from polylogue.maintenance.failure_routing import route_failure_sample
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
from polylogue.maintenance.source_replay import (
    ArtifactFailure,
    repair_source_replay,
)
from polylogue.maintenance.targets import (
    MAINTENANCE_TARGET_NAMES,
    MaintenanceTargetSpec,
    build_maintenance_target_catalog,
)
from polylogue.storage.repair import (
    RepairResult,
    repair_action_event_read_model,
    repair_dangling_fts,
    repair_message_type_backfill,
    repair_session_insights,
    repair_wal_checkpoint,
)

#: Name of the source-replay target. Resolved via the catalog like any
#: other target, but its dispatch path is special-cased because it
#: needs (a) the scope filter to narrow which sources to re-acquire and
#: (b) a per-artifact resume cursor instead of the default per-target
#: cursor.
SOURCE_REPLAY_TARGET: Final[str] = "source_replay"

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
#: PR #1147 AC names explicitly: messages FTS, action-event read model,
#: session insights. ``message_type_backfill`` and ``wal_checkpoint`` are
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
    "action_event_read_model": repair_action_event_read_model,
    "dangling_fts": repair_dangling_fts,
    "message_type_backfill": repair_message_type_backfill,
    "wal_checkpoint": repair_wal_checkpoint,
    # ``source_replay`` is handled by a dedicated branch in
    # :func:`_run_one_target` because it needs the typed scope filter and
    # the per-artifact resume cursor. The sentinel value here only exists
    # so :func:`supported_replay_targets` advertises the target and so
    # the unsupported-target guard in :func:`_run_one_target` does not
    # fire. The value is never invoked.
    SOURCE_REPLAY_TARGET: repair_session_insights,
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


#: Sub-cursor segment marking the per-artifact resume index inside a
#: target-level cursor (AC: "Resume cursor advances per-artifact within
#: a source root, not just per-target"). A cursor of
#: ``"target:2:artifact:9182"`` means "resume at target index 2, and
#: inside that target, skip the first 9182 artifacts".
_CURSOR_ARTIFACT_SEGMENT: Final[str] = "artifact"


def _encode_cursor(next_target_index: int, *, artifact_index: int | None = None) -> str:
    """Encode the next target index (and optional artifact index) as an opaque string.

    The optional ``artifact_index`` is only attached when non-zero so
    targets without per-artifact iteration round-trip through the
    legacy ``target:N`` form (and existing state files keep parsing).
    """
    base = f"{_CURSOR_TARGET_PREFIX}{next_target_index}"
    if artifact_index is None or artifact_index <= 0:
        return base
    return f"{base}:{_CURSOR_ARTIFACT_SEGMENT}:{artifact_index}"


def _decode_cursor(cursor: str | None, *, total_targets: int) -> int:
    """Decode a cursor back into a next-target index.

    Returns ``0`` for an absent or empty cursor (fresh run) and
    ``total_targets`` (i.e. "all done") for :data:`CURSOR_DONE`. Any
    malformed cursor falls back to ``0`` so a corrupt state file can
    never silently skip work. Per-artifact sub-cursors are stripped
    here — use :func:`_decode_artifact_cursor` to recover them.
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


def _decode_artifact_cursor(cursor: str | None) -> int:
    """Return the per-artifact resume index encoded in ``cursor``, or 0.

    Recognized form: ``target:N:artifact:K``. Any other shape (including
    legacy ``target:N`` and :data:`CURSOR_DONE`) yields ``0``.
    """
    if cursor is None or cursor == "" or cursor == CURSOR_DONE:
        return 0
    if not cursor.startswith(_CURSOR_TARGET_PREFIX):
        return 0
    parts = cursor[len(_CURSOR_TARGET_PREFIX) :].split(":")
    # parts is ["<target_idx>", "artifact", "<K>"] when artifact set.
    if len(parts) < 3 or parts[1] != _CURSOR_ARTIFACT_SEGMENT:
        return 0
    try:
        artifact = int(parts[2])
    except ValueError:
        logger.warning("replay_cursor_invalid_artifact_index", cursor=cursor)
        return 0
    return max(0, artifact)


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

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "operation_id": self.operation_id,
                "target": self.target,
                "processed": self.processed,
                "total": self.total,
                "cursor": self.cursor,
                "in_flight_failures": self.in_flight_failures,
            }
        )


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
    #: Next per-artifact index to attempt for the currently-running
    #: target. Only meaningful when the target supports per-artifact
    #: iteration (currently :data:`SOURCE_REPLAY_TARGET`); other targets
    #: leave it at 0.
    artifact_resume_index: int = 0

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

    if resume_cursor is None and persist_state:
        persisted = load_state(config, op_id)
        if persisted is not None:
            cursor_value = persisted.get("cursor")
            if isinstance(cursor_value, str):
                resume_cursor = cursor_value

    start_index = _decode_cursor(resume_cursor, total_targets=len(resolved_names))
    start_artifact_index = _decode_artifact_cursor(resume_cursor)
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
        # The per-artifact cursor only applies to the first resumed
        # target. Once that target completes, the executor moves on to
        # the next target with a fresh 0 artifact cursor.
        artifact_resume = start_artifact_index if index == start_index else 0
        _run_one_target(
            state,
            spec,
            config,
            dry_run=dry_run,
            scope_filter=effective_filter,
            resume_artifact_index=artifact_resume,
        )
        # Advance the cursor *after* the target completes (success or
        # failure). On failure we still advance so the next resume does
        # not re-execute the same target and stack duplicate failure
        # samples; the bounded sample list already records the problem.
        next_index = index + 1
        state.cursor = CURSOR_DONE if next_index == len(resolved_names) else _encode_cursor(next_index)
        state.artifact_resume_index = 0
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

    if persist_state and successful:
        clear_state(config, op_id)

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

    kind = BackfillKind.SOURCE_REPLAY if SOURCE_REPLAY_TARGET in resolved_names else BackfillKind.DERIVED_REBUILD
    return BackfillOperation(
        operation_id=op_id,
        kind=kind,
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


def _artifact_failure_to_sample(failure: ArtifactFailure) -> FailureSample:
    """Project a per-artifact failure onto the planner's bounded sample envelope."""

    return FailureSample(
        kind=failure.kind,
        locator=(
            f"target:{SOURCE_REPLAY_TARGET}:source:{failure.source_name}:"
            f"artifact:{failure.artifact_index}:{failure.source_path}"
        ),
        message=failure.message,
    )


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
    resume_artifact_index: int = 0,
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

    try:
        if target_name == SOURCE_REPLAY_TARGET:
            # SOURCE_REPLAY has a dedicated execution path because it
            # consumes the typed scope filter to choose which sources to
            # iterate, supports per-artifact resume, and surfaces
            # per-artifact failures individually rather than aborting
            # the entire target on the first bad file.
            outcome = repair_source_replay(
                config,
                dry_run,
                scope_filter=scope_filter,
                resume_artifact_index=resume_artifact_index,
            )
            result = outcome.result
            for failure in outcome.failures:
                _record_failure(
                    state,
                    _artifact_failure_to_sample(failure),
                    target=target_name,
                    config=config,
                )
            if outcome.last_artifact_index >= 0:
                state.artifact_resume_index = outcome.last_artifact_index + 1
        elif target_name == "session_insights" and scope_filter.conversation_ids is not None:
            # The session-insights repair fn understands a narrowed
            # conversation-id scope natively; forward it so a one-session
            # plan only touches that one session's insights.
            result = repair_session_insights(
                config,
                dry_run,
                conversation_ids=scope_filter.conversation_ids,
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
) -> None:
    """Persist the running operation state so a kill-mid-run can resume."""

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
        }
    )
    _write_state(state_path_for(config, operation_id), payload)


__all__ = [
    "CURSOR_DONE",
    "MAINTENANCE_TARGET_NAMES",
    "SOURCE_REPLAY_TARGET",
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
