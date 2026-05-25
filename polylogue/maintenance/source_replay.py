"""SOURCE_REPLAY backfill — re-acquire raw artifacts idempotently.

This module wires the :class:`~polylogue.maintenance.planner.BackfillKind.SOURCE_REPLAY`
target so a planner request like ``polylogue maintenance run --target
source-replay --source-root <path>`` actually re-runs the source-acquisition
path against a bounded scope and writes any new raw rows through the
content-hash-deduplicated :func:`persist_raw_record` helper.

The work is intentionally a thin orchestration layer:

* :func:`resolve_source_replay_sources` translates a typed
  :class:`MaintenanceScopeFilter` into the concrete set of
  :class:`Source` objects whose raw artifacts to re-acquire.
* :func:`repair_source_replay` is the per-target repair function. It
  iterates :func:`iter_source_raw_data` for each resolved source,
  converts each :class:`RawConversationData` to a
  :class:`RawConversationRecord`, and persists it through the existing
  async ingest path — which is already idempotent by content hash, so
  a second pass over an unchanged source root inserts zero new raw rows.

Per-artifact resume is supported via the ``resume_artifact_index``
parameter. The executor passes the decoded ``target:N:artifact:K``
cursor so an interrupted run picks up at artifact ``K`` of the source
scope rather than re-starting from artifact 0. The function reports the
last successfully-processed artifact index back through the returned
:class:`SourceReplayOutcome` so the executor can persist the per-target
cursor.

A single artifact's failure does not abort the rest of the scope — it
is recorded on :attr:`SourceReplayOutcome.failures` and the loop
continues. The outer :class:`BackfillOperation` aggregates those into
its bounded :class:`FailureSamples` envelope.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.config import Config, Source
from polylogue.logging import get_logger
from polylogue.maintenance.models import MaintenanceCategory
from polylogue.maintenance.scope import MaintenanceScopeFilter
from polylogue.pipeline.services.acquisition_persistence import persist_raw_record
from polylogue.pipeline.services.acquisition_records import make_raw_record
from polylogue.pipeline.stage_models import AcquireResult
from polylogue.sources.parsers.base import RawConversationData
from polylogue.sources.source_acquisition import iter_source_raw_data
from polylogue.storage.repair import RepairResult

if TYPE_CHECKING:
    from polylogue.storage.repository import ConversationRepository

logger = get_logger(__name__)


@dataclass
class ArtifactFailure:
    """One per-artifact failure observed during a source-replay scope."""

    source_name: str
    source_path: str
    artifact_index: int
    kind: str
    message: str


@dataclass
class SourceReplayOutcome:
    """Structured outcome of :func:`repair_source_replay`.

    Carries the :class:`RepairResult` summary expected by the replay
    dispatch contract plus per-artifact bookkeeping (failures, the last
    artifact index successfully attempted) that the executor uses to
    populate the per-artifact resume cursor and the bounded
    :class:`FailureSamples` envelope.
    """

    result: RepairResult
    acquired: int = 0
    skipped: int = 0
    failures: list[ArtifactFailure] = field(default_factory=list)
    last_artifact_index: int = -1
    total_artifacts_seen: int = 0


def _source_matches_filter(source: Source, scope_filter: MaintenanceScopeFilter) -> bool:
    """True when ``source`` is in scope according to ``scope_filter``.

    Only the source-shaped dimensions (``source_root``, ``source_family``,
    ``provider``) participate here; per-artifact dimensions
    (``raw_artifact_id``, ``time_range``) are evaluated inside the
    artifact loop because they require artifact-level data.
    """

    if scope_filter.source_root is not None:
        if source.path is None:
            return False
        source_path_resolved = _lexical_absolute_path(source.path)
        filter_path_resolved = _lexical_absolute_path(scope_filter.source_root)
        # Honor either equality or "this source is rooted under the
        # filter root" — operators frequently pass a parent directory.
        if source_path_resolved != filter_path_resolved:
            try:
                source_path_resolved.relative_to(filter_path_resolved)
            except ValueError:
                return False

    if scope_filter.provider is not None and source.name != scope_filter.provider:
        return False

    # ``source_family`` is the typed source-centered identifier from
    # ``polylogue.core.sources``. The Source dataclass does not yet
    # carry a typed family attribute (see "Dual Vocabulary Period" in
    # docs/architecture.md), so for now we match against ``source.name``
    # the same way ``provider`` does. When the family rename lands the
    # comparison should switch to ``source.family.value``.
    return not (scope_filter.source_family is not None and source.name != scope_filter.source_family)


def _lexical_absolute_path(path: Path) -> Path:
    """Normalize a path without touching the filesystem.

    Maintenance filters are applied to archived source paths. Some of those
    paths point at old removable mounts or disk-image locations; resolving them
    live can trigger autofs or block on unavailable media. For replay scoping,
    lexical absolute normalization is enough: comparisons only need stable path
    ancestry, not symlink truth in the current host namespace.
    """

    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return Path(str(expanded))


def resolve_source_replay_sources(
    config: Config,
    scope_filter: MaintenanceScopeFilter,
) -> list[Source]:
    """Return the subset of ``config.sources`` matching ``scope_filter``.

    An empty filter returns every source. The function never raises for
    an empty match — callers receive ``[]`` and surface that as a
    no-op repair with ``repaired_count=0``.
    """

    if scope_filter.is_empty():
        return list(config.sources)
    return [source for source in config.sources if _source_matches_filter(source, scope_filter)]


async def _persist_artifact(
    repository: ConversationRepository,
    raw_data: RawConversationData,
    result: AcquireResult,
) -> None:
    """Convert and persist one artifact via the standard ingest path."""

    record = make_raw_record(raw_data, source_name)
    await persist_raw_record(repository, record, result=result)


def _iter_artifacts(source: Source) -> Iterable[RawConversationData]:
    """Iterate raw artifacts of ``source`` without any cursor state.

    SOURCE_REPLAY intentionally ignores stored cursors / mtime hints so
    the operator's request to re-acquire actually re-acquires. The
    content-hash idempotency at :func:`persist_raw_record` guarantees
    no duplicate rows, so revisiting unchanged artifacts is cheap and
    correct.
    """

    yield from iter_source_raw_data(source, cursor_state=None, known_mtimes={}, known_cursors={})


def _artifact_matches_filter(
    raw_data: RawConversationData,
    scope_filter: MaintenanceScopeFilter,
) -> bool:
    """Per-artifact filter evaluation.

    Only the artifact-shaped dimensions participate here. Source-shaped
    dimensions are handled at the outer loop.
    """

    return not (scope_filter.raw_artifact_id is not None and raw_data.blob_hash != scope_filter.raw_artifact_id)


def repair_source_replay(
    config: Config,
    dry_run: bool = False,
    *,
    scope_filter: MaintenanceScopeFilter | None = None,
    resume_artifact_index: int = 0,
) -> SourceReplayOutcome:
    """Re-acquire raw artifacts for the resolved source scope.

    Parameters
    ----------
    config:
        Live runtime config — supplies ``config.sources`` and
        ``config.db_path``.
    dry_run:
        When ``True``, the function enumerates the artifacts that
        *would* be re-acquired but does not write any raw rows.
        Idempotency holds either way: a dry run never advances the
        archive.
    scope_filter:
        Typed scope filter from :mod:`polylogue.maintenance.scope`.
        ``None`` is equivalent to an empty filter — every configured
        source is replayed.
    resume_artifact_index:
        Skip the first ``N`` artifacts of the resolved scope. The
        executor passes the index decoded from the per-artifact cursor
        so an interrupted run continues at artifact ``N`` rather than
        re-doing artifacts already attempted in the prior run.

    Returns
    -------
    SourceReplayOutcome
        Carries the canonical :class:`RepairResult` (for compatibility
        with the rest of ``_REPLAY_DISPATCH``) plus per-artifact
        bookkeeping the executor uses for resume and failure
        reporting.
    """

    effective_filter = scope_filter or MaintenanceScopeFilter()
    sources = resolve_source_replay_sources(config, effective_filter)

    if not sources:
        return SourceReplayOutcome(
            result=RepairResult(
                name="source_replay",
                category=MaintenanceCategory.SOURCE_INGEST,
                destructive=False,
                repaired_count=0,
                success=True,
                detail="No sources matched the requested scope filter.",
            ),
        )

    outcome = SourceReplayOutcome(
        result=RepairResult(
            name="source_replay",
            category=MaintenanceCategory.SOURCE_INGEST,
            destructive=False,
            repaired_count=0,
            success=True,
            detail="",
        ),
    )

    if dry_run:
        # Dry run: enumerate the scope so the caller learns the
        # artifact count without writing anything. Per-artifact
        # filtering still applies so a narrowed dry run reports the
        # narrowed count, not the full source size.
        seen = 0
        for source in sources:
            for raw_data in _iter_artifacts(source):
                seen += 1
                if not _artifact_matches_filter(raw_data, effective_filter):
                    continue
        outcome.total_artifacts_seen = seen
        outcome.result = RepairResult(
            name="source_replay",
            category=MaintenanceCategory.SOURCE_INGEST,
            destructive=False,
            repaired_count=0,
            success=True,
            detail=f"Would attempt to re-acquire {seen} artifacts (dry run).",
        )
        return outcome

    # Async ingest path: we have to drive the existing async repository
    # from a sync repair function. ``asyncio.run`` is appropriate because
    # repair functions are invoked from sync orchestration code.
    asyncio.run(_run_async(config, sources, effective_filter, resume_artifact_index, outcome))

    if outcome.failures:
        outcome.result = RepairResult(
            name="source_replay",
            category=MaintenanceCategory.SOURCE_INGEST,
            destructive=False,
            repaired_count=outcome.acquired,
            success=False,
            detail=(
                f"Re-acquired {outcome.acquired} new raw rows, "
                f"skipped {outcome.skipped} unchanged, "
                f"{len(outcome.failures)} per-artifact failures."
            ),
        )
    else:
        outcome.result = RepairResult(
            name="source_replay",
            category=MaintenanceCategory.SOURCE_INGEST,
            destructive=False,
            repaired_count=outcome.acquired,
            success=True,
            detail=(f"Re-acquired {outcome.acquired} new raw rows, skipped {outcome.skipped} unchanged."),
        )
    return outcome


async def _run_async(
    config: Config,
    sources: list[Source],
    scope_filter: MaintenanceScopeFilter,
    resume_artifact_index: int,
    outcome: SourceReplayOutcome,
) -> None:
    """Async core: open a repository against ``config.db_path`` and replay."""

    # Local imports avoid module-load-time circular imports between the
    # maintenance package and storage backends.
    from polylogue.storage.repository import ConversationRepository
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

    backend = SQLiteBackend(config.db_path)
    repository = ConversationRepository(backend=backend)
    acquire_result = AcquireResult()
    artifact_index = 0
    async with backend.bulk_connection():
        for source in sources:
            for raw_data in _iter_artifacts(source):
                current_index = artifact_index
                artifact_index += 1
                if current_index < resume_artifact_index:
                    continue
                if not _artifact_matches_filter(raw_data, scope_filter):
                    continue
                outcome.total_artifacts_seen += 1
                try:
                    await _persist_artifact(repository, raw_data, source.name, acquire_result)
                    outcome.last_artifact_index = current_index
                except (RuntimeError, OSError, ValueError) as exc:
                    outcome.failures.append(
                        ArtifactFailure(
                            source_name=source.name,
                            source_path=raw_data.source_path,
                            artifact_index=current_index,
                            kind=type(exc).__name__,
                            message=str(exc),
                        )
                    )
    outcome.acquired = acquire_result.acquired
    outcome.skipped = acquire_result.skipped


__all__ = [
    "ArtifactFailure",
    "SourceReplayOutcome",
    "repair_source_replay",
    "resolve_source_replay_sources",
]
