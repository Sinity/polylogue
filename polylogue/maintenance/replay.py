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

Persisted state records target identities in ``completed_targets``. New
checkpoints use those successful identities as the sole work coordinate and
retain ``cursor="target:0"`` only as a validated migration field for older
states. On resume, the executor derives pending work from completed identities
against the current catalog before any handler runs, so removing or reordering
a target cannot shift work onto an unrelated target. An unverifiable
historical state fails closed. The state is persisted to a small JSON file
under the configured archive root.

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
from typing import TYPE_CHECKING, Final, cast

from polylogue.config import Config
from polylogue.core.enums import OperationStatus
from polylogue.core.json import JSONDocument, dumps, json_document, loads
from polylogue.core.protocols import ProgressCallback as StageProgressCallback
from polylogue.logging import get_logger

if TYPE_CHECKING:
    from polylogue.sources.revision_backfill import RawParsePrefetchCache
from polylogue.maintenance.failure_routing import resolve_maintenance_failures, route_failure_sample
from polylogue.maintenance.invalidation import InvalidationReason
from polylogue.maintenance.planner import (
    BackfillKind,
    BackfillOperation,
    BoundedFailureSamples,
    FailureSample,
    MaintenanceScope,
)
from polylogue.maintenance.scope import MaintenanceScopeFilter
from polylogue.maintenance.targets import (
    CLEANUP_TARGETS,
    MAINTENANCE_TARGET_NAMES,
    SAFE_REPAIR_TARGETS,
    MaintenanceTargetSpec,
    build_maintenance_target_catalog,
)
from polylogue.storage import repair as _repair
from polylogue.storage.repair import (
    RepairResult,
    offline_maintenance_blockers,
    repair_session_insights,
)

logger = get_logger(__name__)

#: Sentinel cursor value meaning "operation completed; nothing left to do."
CURSOR_DONE: Final[str] = "done"

#: Cursor prefix retained for legacy state migration. New checkpoints write
#: ``target:0`` and derive pending work from successful target identities;
#: older ``target:N`` values are interpreted against their persisted target
#: tuple only after strict validation.
_CURSOR_TARGET_PREFIX: Final[str] = "target:"

#: Subdirectory under :attr:`Config.archive_root` used for replay state
#: files. One JSON file per ``operation_id``.
_STATE_DIRNAME: Final[str] = ".maintenance-state"

#: Spill-cache bound for the offline ``rebuild-index`` command's one-shot,
#: archive-wide census (``selected_raw_ids=None`` has no resource envelope,
#: so this is independent of any envelope block). It only avoids the
#: census-then-replay double parse for typical raws; oversized raws are
#: deliberately still not cached (see ``_ParsedSessionSpill``), so the cache
#: never becomes a second archive-wide materialization.
_REBUILD_CENSUS_SPILL_CACHE_BYTES: Final[int] = 512 * 1024 * 1024

#: Rebuild-scale commit batching (polylogue-amg1/oikv machinery; the rebuild
#: caller previously never opted in, paying one fsync'd commit per censused
#: raw and per replayed cohort -- thousands per page). A crash discards at
#: most one open batch and the resume reprocesses it from scratch (contract
#: pinned by test_backfill_resumes_after_replay_batch_crash_discards_whole_
#: batch_cleanly et al.), so the loss window is bounded and cheap.
_REBUILD_COMMIT_BATCH_UNITS: Final[int] = 200


# ---------------------------------------------------------------------------
# Target dispatch
# ---------------------------------------------------------------------------


#: Type alias for repair functions that take (config, dry_run) -> RepairResult.
_RepairFn = Callable[[Config, bool], RepairResult]


def _replay_handler_for(target_name: str, *, replayable: bool) -> _RepairFn | None:
    """Look up the concrete repair callable for one target.

    The catalog (:mod:`polylogue.maintenance.targets`) is the single
    source for target *identity* and replay *capability*
    (``MaintenanceTargetSpec.replayable``); the concrete handler
    implementation lives in :data:`polylogue.storage.repair.REPAIR_HANDLERS`
    (the same dict the non-resumable ``run_selected_maintenance`` path
    uses). There is deliberately no independent, hand-maintained replay
    dispatch table any more (polylogue-71ey) -- a target that is
    ``replayable`` in the catalog but missing from ``REPAIR_HANDLERS``
    (or vice versa) is a bug caught by
    ``tests/unit/maintenance/test_targets.py``'s catalog-equality test,
    not silently tolerated here.
    """
    if not replayable:
        return None
    return _repair.REPAIR_HANDLERS.get(target_name)


def supported_replay_targets() -> tuple[str, ...]:
    """Names of targets the replay executor knows how to execute.

    Derived from the catalog's declared ``replayable`` targets, filtered
    to those with a real handler in
    :data:`polylogue.storage.repair.REPAIR_HANDLERS`. Stable contract
    for callers and tests.
    """
    catalog = build_maintenance_target_catalog()
    return tuple(
        spec.name for spec in catalog.specs if _replay_handler_for(spec.name, replayable=spec.replayable) is not None
    )


async def rebuild_index_from_source(
    config: Config,
    *,
    raw_ids: list[str] | None,
    raw_batch_size: int,
    ingest_workers: int | None,
    materialize: bool,
    progress_callback: StageProgressCallback | None,
    owned_inactive_generation: tuple[str, str] | None = None,
    bulk_fts: bool = False,
    bulk_build: bool = False,
    prefetch_cache: RawParsePrefetchCache | None = None,
    deadline_check: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Replay retained bytes through typed revision authority.

    ``raw_ids`` is an initial scheduling hint, not an authority boundary: the
    replay expands to complete logical cohorts so a partial selection cannot
    make an older snapshot look newest.

    ``bulk_fts`` (polylogue-crd8, default ``False``) enables the guard-gated
    bulk FTS mode for whale prefix-sharing lineage cascades encountered during
    replay; see ``backfill_historical_revision_evidence``. The offline
    ``rebuild-index`` maintenance command passes ``True``.

    ``bulk_build`` (polylogue-v6i3, default ``False``) enables the broader
    bulk-generation-build lifecycle -- skip every per-session
    messages_fts/blocks_command_trigram/action_pairs/delegation_facts
    refresh during replay, deferred to one archive-wide repopulate at
    readiness. The offline ``rebuild-index`` maintenance command passes
    ``True``.

    ``prefetch_cache`` (polylogue-gd6v, default ``None``) lets a caller
    substitute parse output already computed off the writer hold (the
    daemon's ``DaemonParseStage``) for this pass's census phase; see
    ``backfill_historical_revision_evidence``.

    ``deadline_check`` (polylogue-uhgm, default ``None``) is forwarded
    unchanged to ``backfill_historical_revision_evidence``'s own parameter of
    the same name -- see its docstring for the exact interruption contract
    (checked between REPLAY cohorts, not only after this call returns).
    """
    if raw_batch_size <= 0:
        raise ValueError("raw_batch_size must be positive")
    # The caller owns page selection.  Do not widen its bounded page here:
    # revision backfill may still expand that page to a complete authority
    # cohort, which is required for correct newest-revision selection.
    del materialize
    import asyncio

    from polylogue.pipeline.services.process_pool import resolve_parse_worker_count
    from polylogue.sources.revision_backfill import (
        backfill_historical_revision_evidence,
        split_parse_and_apply_seconds,
    )

    resolved_ingest_workers = ingest_workers if ingest_workers is not None else resolve_parse_worker_count()
    if progress_callback is not None:
        progress_callback(0, "classifying retained raw revision cohorts")
    result = await asyncio.to_thread(
        backfill_historical_revision_evidence,
        Path(config.archive_root),
        selected_raw_ids=raw_ids,
        owned_inactive_generation=owned_inactive_generation,
        max_cached_payload_bytes=_REBUILD_CENSUS_SPILL_CACHE_BYTES,
        ingest_workers=resolved_ingest_workers,
        commit_batch_size=_REBUILD_COMMIT_BATCH_UNITS,
        # Replay-phase batching was previously pinned to 1 (per-cohort
        # commits) because a cohort with pending attachment blobs flushes
        # publication receipts on a SEPARATE source.db connection that waits
        # at BEGIN IMMEDIATE behind the batch's held write lock. Both apply
        # paths now commit the open batch before any NON-EMPTY flush
        # (``ArchiveBlobPublisher.has_pending``), so attachment-free cohorts
        # -- the overwhelming bulk of a rebuild -- share one commit per
        # batch and blob-carrying cohorts degrade to the old per-cohort
        # boundary instead of deadlocking. Measured: per-cohort commits were
        # ~38% of a rebuild pass's wall (4 transactions per cohort).
        replay_commit_batch_size=_REBUILD_COMMIT_BATCH_UNITS,
        bulk_fts=bulk_fts,
        bulk_build=bulk_build,
        prefetch_cache=prefetch_cache,
        deadline_check=deadline_check,
    )
    if progress_callback is not None:
        progress_callback(result.replayed_logical_sources, "revision replay complete")
    parse_s, apply_s = split_parse_and_apply_seconds(result.stage_timings_s)
    return {
        "scanned_raw_count": result.scanned,
        "classified_full_count": result.classified_full,
        "replayed_logical_source_count": result.replayed_logical_sources,
        "quarantined_raw_count": result.quarantined,
        "adoption_deferred_raw_count": result.adoption_deferred,
        "authority_selection_expanded": True,
        "scheduled_raw_count": len(raw_ids) if raw_ids is not None else None,
        "raw_batch_size": raw_batch_size,
        "ingest_workers": resolved_ingest_workers,
        # polylogue-623q: the parse-vs-apply split. ``parse_s`` is read-only
        # decode (census + spill-cache reload of already-parsed content),
        # embarrassingly parallel and scaling with ``ingest_workers``.
        # ``apply_s`` is everything charged to the single SQLite writer
        # (index/FTS/projection writes) -- see
        # ``revision_backfill.split_parse_and_apply_seconds``.
        "parse_s": round(parse_s, 6),
        "apply_s": round(apply_s, 6),
        "stage_timings_s": {key: round(value, 6) for key, value in result.stage_timings_s.items()},
    }


class UnsupportedReplayTargetError(RuntimeError):
    """Raised when a resolved target has no replay dispatch entry."""


class InvalidReplayStateError(RuntimeError):
    """Raised when persisted replay state cannot be trusted for resumption."""


class IncompatibleReplayStateError(RuntimeError):
    """Raised when persisted replay identities cannot map to current targets."""


def _failed_replay_state(
    *,
    operation_id: str,
    targets: tuple[str, ...],
    scope_filter: MaintenanceScopeFilter,
    message: str,
    kind: str,
) -> BackfillOperation:
    """Build a typed failure without invoking a maintenance handler."""
    started_at = datetime.now(timezone.utc).isoformat()
    sample = FailureSample(kind=kind, locator=f"operation:{operation_id}", message=message)
    return BackfillOperation(
        operation_id=operation_id,
        kind=BackfillKind.DERIVED_REBUILD,
        targets=targets,
        status=OperationStatus.FAILED,
        progress=0.0,
        started_at=started_at,
        completed_at=started_at,
        error=message,
        scope=MaintenanceScope(targets=targets, filter=scope_filter),
        reason=InvalidationReason.UNKNOWN,
        failure_samples=BoundedFailureSamples.from_samples((sample,)),
        metrics={"repaired_count": 0.0},
    )


# ---------------------------------------------------------------------------
# Cursor encoding
# ---------------------------------------------------------------------------


def _encode_cursor(next_target_index: int) -> str:
    """Encode the next target index as an opaque ``target:N`` string."""
    return f"{_CURSOR_TARGET_PREFIX}{next_target_index}"


def _strict_cursor(cursor: str | None, *, total_targets: int) -> tuple[int | None, str | None]:
    """Parse a cursor without converting corruption into a destructive run."""
    if cursor is None or cursor == "":
        return 0, None
    if cursor == CURSOR_DONE:
        return total_targets, None
    if not cursor.startswith(_CURSOR_TARGET_PREFIX):
        return None, "Persisted replay state has an invalid target cursor"
    head = cursor[len(_CURSOR_TARGET_PREFIX) :].split(":", 1)[0]
    try:
        index = int(head)
    except ValueError:
        return None, "Persisted replay state has an invalid target cursor"
    if index < 0 or index > total_targets:
        return None, "Persisted replay state has an incompatible target cursor"
    return index, None


def _resume_pending_specs(
    specs: tuple[MaintenanceTargetSpec, ...],
    persisted: JSONDocument,
    *,
    cursor_override: str | None = None,
) -> tuple[tuple[MaintenanceTargetSpec, ...] | None, tuple[str, ...], str | None]:
    """Map persisted completion identities onto the current target catalog."""
    raw_targets = persisted.get("targets")
    if not isinstance(raw_targets, list) or not all(isinstance(name, str) for name in raw_targets):
        return None, (), "Persisted replay state has no valid target identity list"
    old_targets = cast(tuple[str, ...], tuple(raw_targets))
    if len(set(old_targets)) != len(old_targets):
        return None, (), "Persisted replay state has duplicate target identities"

    has_completion_identities = "completed_targets" in persisted
    completed_raw = persisted.get("completed_targets")
    if has_completion_identities:
        if not isinstance(completed_raw, list) or not all(isinstance(name, str) for name in completed_raw):
            return None, (), "Persisted replay state has invalid completed target identities"
        completed_order = cast(tuple[str, ...], tuple(completed_raw))
        if len(set(completed_order)) != len(completed_order) or not set(completed_order) <= set(old_targets):
            return None, (), "Persisted replay state has incompatible completed target identities"
        completed = set(completed_order)
        legacy_pending = tuple(name for name in old_targets if name not in completed)
        cursor = cursor_override if cursor_override is not None else persisted.get("cursor")
        _, cursor_error = _strict_cursor(cursor if isinstance(cursor, str) else None, total_targets=len(legacy_pending))
        if cursor_error is not None:
            return None, (), cursor_error
        # New checkpoints use identities as the sole coordinate. The cursor is
        # retained only as a validated migration field and never advances more
        # identities after filtering.
        return tuple(spec for spec in specs if spec.name not in completed), completed_order, None

    cursor = cursor_override if cursor_override is not None else persisted.get("cursor")
    index, cursor_error = _strict_cursor(cursor if isinstance(cursor, str) else None, total_targets=len(old_targets))
    if cursor_error is not None or index is None:
        return None, (), cursor_error or "Persisted replay state has no valid target cursor"
    completed_order = old_targets[:index]
    completed = set(completed_order)
    return tuple(spec for spec in specs if spec.name not in completed), completed_order, None


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
    """Load a previously persisted operation state, or ``None``.

    A present but malformed state is distinct from an absent state. Callers
    must fail closed rather than treating corruption as a fresh operation.
    """
    path = state_path_for(config, operation_id)
    if not path.exists():
        return None
    try:
        raw = loads(path.read_text())
    except Exception as exc:
        logger.warning(
            "replay_state_unparseable",
            operation_id=operation_id,
            path=str(path),
            error=str(exc),
        )
        raise InvalidReplayStateError("Persisted replay state is not a JSON object") from exc
    if not isinstance(raw, dict):
        logger.warning(
            "replay_state_unparseable",
            operation_id=operation_id,
            path=str(path),
        )
        raise InvalidReplayStateError("Persisted replay state is not a JSON object")
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
    target_history: tuple[str, ...]
    cursor: str
    started_at: str
    completed_targets: list[str] = field(default_factory=list)
    results: list[JSONDocument] = field(default_factory=list)
    failures: list[FailureSample] = field(default_factory=list)
    repaired_total: int = 0
    metrics: dict[str, float] = field(default_factory=dict)

    def progress_for(self, target: str, processed: int) -> ReplayProgress:
        return ReplayProgress(
            operation_id=self.operation_id,
            target=target,
            processed=processed,
            total=len(self.targets),
            cursor=self.cursor,
            in_flight_failures=len(self.failures),
        )


def _numeric_metric(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _operation_metrics(state: _ReplayState) -> dict[str, float]:
    metrics = dict(state.metrics)
    metrics["repaired_count"] = float(state.repaired_total)
    for result in state.results:
        result_metrics = result.get("metrics")
        if not isinstance(result_metrics, dict):
            continue
        for key, value in result_metrics.items():
            metric_value = _numeric_metric(value)
            if metric_value is None:
                continue
            metric_key = str(key)
            if metric_key.endswith("_max_blob_bytes"):
                metrics[metric_key] = max(metrics.get(metric_key, 0.0), metric_value)
            else:
                metrics[metric_key] = metrics.get(metric_key, 0.0) + metric_value
    return metrics


def _hydrate_persisted_receipt(
    persisted: JSONDocument,
) -> tuple[str | None, list[JSONDocument], list[FailureSample], int, dict[str, float], str | None]:
    """Validate and hydrate cumulative receipt fields from a state file."""
    started_at = persisted.get("started_at")
    if started_at is not None and not isinstance(started_at, str):
        return None, [], [], 0, {}, "Persisted replay state has invalid started_at"

    raw_results = persisted.get("results", [])
    if not isinstance(raw_results, list) or not all(isinstance(item, dict) for item in raw_results):
        return None, [], [], 0, {}, "Persisted replay state has invalid results"
    results = cast(list[JSONDocument], raw_results)

    raw_failures = persisted.get("failure_samples", [])
    if not isinstance(raw_failures, list):
        return None, [], [], 0, {}, "Persisted replay state has invalid failure samples"
    failures: list[FailureSample] = []
    for item in raw_failures:
        if not isinstance(item, dict) or not all(
            isinstance(item.get(key), str) for key in ("kind", "locator", "message")
        ):
            return None, [], [], 0, {}, "Persisted replay state has invalid failure samples"
        failures.append(
            FailureSample(
                kind=cast(str, item["kind"]),
                locator=cast(str, item["locator"]),
                message=cast(str, item["message"]),
            )
        )

    raw_repaired = persisted.get("repaired_count", 0)
    if not isinstance(raw_repaired, (int, float)) or isinstance(raw_repaired, bool):
        return None, [], [], 0, {}, "Persisted replay state has invalid repaired count"
    raw_metrics = persisted.get("metrics", {})
    if not isinstance(raw_metrics, dict):
        return None, [], [], 0, {}, "Persisted replay state has invalid metrics"
    metrics: dict[str, float] = {}
    for key, value in raw_metrics.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None, [], [], 0, {}, "Persisted replay state has invalid metrics"
        metrics[str(key)] = float(value)
    return started_at, results, failures, int(raw_repaired), metrics, None


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
    # Empty ``targets`` means "no explicit scope" and expands to the
    # documented run-all set (every catalog target); an explicit but
    # unresolvable name still fails closed with an empty resolution
    # (polylogue-71ey bug 2: targetless ``maintenance run`` used to
    # resolve to zero targets and report ``status=failed``).
    resolved_specs = catalog.resolve_or_default(tuple(targets))
    resolved_names = tuple(spec.name for spec in resolved_specs)
    effective_filter = scope_filter or MaintenanceScopeFilter()

    if not resolved_names:
        return BackfillOperation(
            operation_id=op_id,
            kind=BackfillKind.DERIVED_REBUILD,
            targets=(),
            status=OperationStatus.FAILED,
            error="No valid targets resolved from input",
            scope=MaintenanceScope(targets=(), filter=effective_filter),
        )

    # Load persisted identity metadata before any execution gate. A cursor
    # written against a retired catalog must be remapped or rejected before a
    # handler can observe the request.
    explicit_resume = resume_cursor is not None
    persisted: JSONDocument | None = None
    if persist_state:
        try:
            persisted = load_state(config, op_id)
        except InvalidReplayStateError as exc:
            return _failed_replay_state(
                operation_id=op_id,
                targets=resolved_names,
                scope_filter=effective_filter,
                message=str(exc),
                kind="InvalidReplayState",
            )

    pending_specs = resolved_specs
    completed_targets: tuple[str, ...] = ()
    target_history = resolved_names
    receipt_started_at: str | None = None
    prior_results: list[JSONDocument] = []
    prior_failures: list[FailureSample] = []
    prior_repaired_total = 0
    prior_metrics: dict[str, float] = {}
    if persisted is not None:
        (
            receipt_started_at,
            prior_results,
            prior_failures,
            prior_repaired_total,
            prior_metrics,
            receipt_error,
        ) = _hydrate_persisted_receipt(persisted)
        if receipt_error is not None:
            return _failed_replay_state(
                operation_id=op_id,
                targets=resolved_names,
                scope_filter=effective_filter,
                message=receipt_error,
                kind="InvalidReplayState",
            )

    target_history = resolved_names
    if persisted is not None:
        if "targets" not in persisted:
            if not explicit_resume:
                return _failed_replay_state(
                    operation_id=op_id,
                    targets=resolved_names,
                    scope_filter=effective_filter,
                    message="Persisted replay state has no valid target identity list",
                    kind="IncompatibleReplayState",
                )
        else:
            raw_history = persisted.get("targets")
            if isinstance(raw_history, list) and all(isinstance(name, str) for name in raw_history):
                history_names = cast(list[str], raw_history)
                target_history = tuple(dict.fromkeys((*history_names, *resolved_names)))
            cursor_value = resume_cursor if explicit_resume else persisted.get("cursor")
            mapped_pending, completed_targets, resume_error = _resume_pending_specs(
                resolved_specs,
                persisted,
                cursor_override=cursor_value if isinstance(cursor_value, str) else None,
            )
            if resume_error is not None or mapped_pending is None:
                message = resume_error or "Persisted replay state is incompatible with the current target catalog"
                logger.error(
                    "replay_state_incompatible",
                    operation_id=op_id,
                    targets=resolved_names,
                    error=message,
                )
                return _failed_replay_state(
                    operation_id=op_id,
                    targets=resolved_names,
                    scope_filter=effective_filter,
                    message=message,
                    kind="IncompatibleReplayState",
                )
            assert mapped_pending is not None
            pending_specs = mapped_pending
            # The cursor has been translated by identity. The remaining tuple
            # is a fresh positional work list for this run.
            resume_cursor = _encode_cursor(0)
    elif not explicit_resume:
        resume_cursor = None

    blockers = offline_maintenance_blockers(
        config,
        repair=any(name in SAFE_REPAIR_TARGETS for name in resolved_names),
        cleanup=any(name in CLEANUP_TARGETS for name in resolved_names),
        dry_run=dry_run,
        targets=resolved_names,
    )
    if blockers and pending_specs:
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
            status=OperationStatus.FAILED,
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

    start_index, cursor_error = _strict_cursor(resume_cursor, total_targets=len(pending_specs))
    if cursor_error is not None or start_index is None:
        return _failed_replay_state(
            operation_id=op_id,
            targets=resolved_names,
            scope_filter=effective_filter,
            message=cursor_error or "Persisted replay state has an invalid target cursor",
            kind="InvalidReplayCursor",
        )
    if not completed_targets and start_index:
        completed_targets = tuple(spec.name for spec in pending_specs[:start_index])
    started_at = receipt_started_at or datetime.now(timezone.utc).isoformat()
    state = _ReplayState(
        operation_id=op_id,
        targets=resolved_names,
        target_history=target_history,
        cursor=_encode_cursor(0),
        started_at=started_at,
        completed_targets=list(completed_targets),
        results=prior_results,
        failures=prior_failures,
        repaired_total=prior_repaired_total,
        metrics=prior_metrics,
    )

    if not pending_specs:
        state.cursor = CURSOR_DONE
        completed_at = datetime.now(timezone.utc).isoformat()
        final = BackfillOperation(
            operation_id=op_id,
            kind=BackfillKind.DERIVED_REBUILD,
            targets=resolved_names,
            status=OperationStatus.COMPLETED,
            progress=1.0,
            started_at=started_at,
            completed_at=completed_at,
            affected_rows=state.repaired_total,
            results=state.results,
            scope=MaintenanceScope(targets=resolved_names, filter=effective_filter),
            resume_cursor=CURSOR_DONE,
            failure_samples=BoundedFailureSamples.from_samples(state.failures),
            metrics=_operation_metrics(state),
        )
        if persist_state:
            clear_state(config, op_id)
        return final

    logger.info(
        "replay_starting",
        operation_id=op_id,
        targets=resolved_names,
        start_index=start_index,
        dry_run=dry_run,
    )

    for index in range(start_index, len(pending_specs)):
        spec = pending_specs[index]
        target_name = spec.name
        succeeded = _run_one_target(
            state,
            spec,
            config,
            dry_run=dry_run,
            scope_filter=effective_filter,
            progress_callback=progress_callback,
            target_total=len(resolved_names),
            processed_before_target=index - start_index,
        )
        # Successful identities are authoritative. Failed targets remain in
        # the pending set for the next invocation under this operation id.
        if succeeded and target_name not in state.completed_targets:
            state.completed_targets.append(target_name)
        # New checkpoints always use identity completion as the coordinate.
        # target:0 is retained only as a validated legacy field.
        state.cursor = (
            CURSOR_DONE if all(name in state.completed_targets for name in resolved_names) else _encode_cursor(0)
        )
        if persist_state:
            _checkpoint_state(
                config=config,
                operation_id=op_id,
                state=state,
                started_at=started_at,
                dry_run=dry_run,
            )
        if progress_callback is not None:
            progress_callback(
                state.progress_for(
                    target_name,
                    processed=sum(name in state.completed_targets for name in resolved_names),
                )
            )

    completed_at = datetime.now(timezone.utc).isoformat()
    successful = all(name in state.completed_targets for name in resolved_names)
    status = OperationStatus.COMPLETED if successful else OperationStatus.FAILED

    completed_current = sum(name in state.completed_targets for name in resolved_names)
    progress = completed_current / len(resolved_names) if resolved_names else 1.0
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
        reason=InvalidationReason.UNKNOWN if not successful else None,
        resume_cursor=state.cursor,
        failure_samples=BoundedFailureSamples.from_samples(state.failures),
        metrics=_operation_metrics(state),
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
) -> bool:
    """Execute one target, recording success or a typed failure sample."""

    target_name = spec.name
    repair_fn = _replay_handler_for(target_name, replayable=spec.replayable)
    if repair_fn is None:
        reason = (
            spec.non_replayable_reason
            if not spec.replayable and spec.non_replayable_reason
            else (f"Target {target_name!r} is not yet wired into polylogue.storage.repair.REPAIR_HANDLERS.")
        )
        sample = FailureSample(
            kind=UnsupportedReplayTargetError.__name__,
            locator=f"target:{target_name}",
            message=reason,
        )
        _record_failure(state, sample, target=target_name, config=config)
        logger.warning(
            "replay_target_unsupported",
            operation_id=state.operation_id,
            target=target_name,
        )
        return False

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
        return False

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
    return result.success


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
            "targets": list(state.target_history),
            "resolved_targets": list(state.targets),
            "completed_targets": list(state.completed_targets),
            "cursor": state.cursor,
            "started_at": started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "dry_run": dry_run,
            "repaired_count": state.repaired_total,
            "failure_count": len(state.failures),
            "failure_samples": [sample.to_dict() for sample in state.failures],
            "results": list(state.results),
            "metrics": dict(state.metrics),
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
    snapshot status is :data:`OperationStatus.RUNNING` unless the
    executor has finished all targets, in which case it surfaces as
    :data:`OperationStatus.COMPLETED` / :data:`OperationStatus.FAILED`
    based on the in-flight failure count.
    """

    total = len(state.targets)
    cursor = state.cursor
    if cursor == CURSOR_DONE:
        all_completed = all(name in state.completed_targets for name in state.targets)
        status = OperationStatus.COMPLETED if all_completed else OperationStatus.FAILED
        progress = 1.0
        completed_at: str | None = datetime.now(timezone.utc).isoformat()
    else:
        status = OperationStatus.RUNNING
        processed = len(state.completed_targets)
        progress = min(processed / total, 1.0) if total > 0 else 0.0
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
        metrics=_operation_metrics(state),
    )


__all__ = [
    "CURSOR_DONE",
    "MAINTENANCE_TARGET_NAMES",
    "MaintenanceScopeFilter",
    "ProgressCallback",
    "ReplayProgress",
    "InvalidReplayStateError",
    "IncompatibleReplayStateError",
    "UnsupportedReplayTargetError",
    "clear_state",
    "execute_replay",
    "load_state",
    "rebuild_index_from_source",
    "state_path_for",
    "supported_replay_targets",
]
