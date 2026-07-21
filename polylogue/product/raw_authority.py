"""Product boundary for durable raw-authority maintenance.

The storage implementation deliberately owns the durable receipts and replay
algorithms.  CLI and daemon surfaces use this module so that they share one
typed product operation rather than importing storage internals directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from polylogue.config import Config
from polylogue.core.json import JSONDocument

if TYPE_CHECKING:
    from polylogue.sources.revision_backfill import RawParsePrefetchCache


@dataclass(frozen=True, slots=True)
class RawMaterializationCounts:
    """Separate units produced by one bounded maintenance pass.

    ``censused_components`` counts parser-census work performed toward a
    paused replay plan: no sessions were repaired yet, but the pass moved
    the backlog and the caller's backlog burst must continue instead of
    treating the census phase as quiescence.

    ``candidate_count`` and ``pending_blob_bytes`` describe the *whole*
    unbounded backlog the pass measured (not the bounded per-pass batch):
    ``repair_materialization`` enumerates every matching raw before
    applying ``raw_artifact_limit``, so these two fields are how a caller
    detects a bulk-scale backlog the trickle conveyor is not designed for
    (polylogue-m6tp) without re-querying storage itself.
    """

    repaired_sessions: int = 0
    executed_plans: int = 0
    remaining_candidates: int = 0
    censused_components: int = 0
    candidate_count: int = 0
    pending_blob_bytes: int = 0

    @property
    def made_progress(self) -> bool:
        return self.repaired_sessions > 0 or self.executed_plans > 0 or self.censused_components > 0


def inspect_frontier(config: Config) -> Any:
    from polylogue.storage.raw_reconciler import inspect_raw_authority_frontier

    return inspect_raw_authority_frontier(config)


def apply_frontier(config: Config, *, preview_census_id: str, selected_plan_ids: tuple[str, ...]) -> Any:
    from polylogue.storage.raw_reconciler import apply_raw_authority_frontier

    return apply_raw_authority_frontier(
        config,
        preview_census_id=preview_census_id,
        selected_plan_ids=selected_plan_ids,
    )


def recover_interrupted_frontier(config: Config) -> tuple[str, ...]:
    from polylogue.storage.raw_reconciler import recover_interrupted_raw_authority_frontier

    return recover_interrupted_raw_authority_frontier(config)


def repair_materialization(
    config: Config,
    *,
    dry_run: bool,
    raw_artifact_limit: int,
    max_payload_bytes: int,
    prefetch_cache: RawParsePrefetchCache | None = None,
    raw_artifact_id: str | None = None,
) -> Any:
    """Run one bounded raw source->index convergence pass.

    ``prefetch_cache`` (polylogue-m6tp phase (a), default ``None``) lets a
    caller substitute parse output already computed off the writer hold for
    this pass's census phase; see
    ``polylogue.sources.revision_backfill.RawParsePrefetchCache``.

    ``raw_artifact_id`` (polylogue-t93b, default ``None``) scopes the pass to
    the single logical authority component containing that raw -- the
    daemon's escalation-tier whale pass uses this to converge one
    resource-blocked component at a time under a widened ``max_payload_bytes``
    envelope instead of re-scanning the whole archive-wide backlog.
    """
    from polylogue.storage.repair import repair_raw_materialization

    return repair_raw_materialization(
        config,
        dry_run=dry_run,
        raw_artifact_limit=raw_artifact_limit,
        max_payload_bytes=max_payload_bytes,
        prefetch_cache=prefetch_cache,
        raw_artifact_id=raw_artifact_id,
    )


def whale_pass_candidate(
    config: Config,
    *,
    ordinary_max_payload_bytes: int,
    whale_max_payload_bytes: int,
) -> str | None:
    """Read-only: pick one resource-blocked, stream-safe component to escalate.

    polylogue-t93b. See
    ``polylogue.storage.repair.raw_materialization_whale_pass_candidate`` for
    the selection contract; safe to call without the writer hold.
    """
    from polylogue.storage.repair import raw_materialization_whale_pass_candidate

    return raw_materialization_whale_pass_candidate(
        config,
        ordinary_max_payload_bytes=ordinary_max_payload_bytes,
        whale_max_payload_bytes=whale_max_payload_bytes,
    )


def read_census(archive_root: Path, query_handle: str, *, limit: int, offset: int | None) -> JSONDocument:
    from polylogue.storage.raw_authority import read_raw_authority_census

    return read_raw_authority_census(archive_root, query_handle, limit=limit, offset=offset)


def read_detail(archive_root: Path, query_handle: str, *, chunk_chars: int, offset: int | None) -> JSONDocument:
    from polylogue.storage.raw_authority import read_raw_authority_detail

    return read_raw_authority_detail(archive_root, query_handle, chunk_chars=chunk_chars, offset=offset)


def resolve_blocker(
    archive_root: Path,
    blocker_id: str,
    *,
    resolution: str,
    assertion_id: str | None,
    judgment_disposition: str | None,
) -> JSONDocument:
    from polylogue.storage.raw_authority import resolve_raw_authority_blocker

    return resolve_raw_authority_blocker(
        archive_root,
        blocker_id,
        resolution=resolution,
        assertion_id=assertion_id,
        judgment_disposition=judgment_disposition,
    )


__all__ = [
    "RawMaterializationCounts",
    "apply_frontier",
    "inspect_frontier",
    "read_census",
    "read_detail",
    "recover_interrupted_frontier",
    "repair_materialization",
    "resolve_blocker",
]
