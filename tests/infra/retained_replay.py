"""Drive canonical retained preparation and record actual publication results."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.operations.raw_observation_derivation import make_raw_observation_derivation, raw_observation_frame
from polylogue.sources.revision_backfill import (
    PreparedRevisionReplayResult,
    RetainedPreparationRetryableError,
    RevisionCensusResult,
)
from polylogue.storage.index_generation import IndexGeneration
from polylogue.storage.sqlite.connection_profile import readonly_connection_context
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.live_ingest import prepared_live_convergence_owner

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polylogue.core.compute import BoundedComputeAdapter


@dataclass(frozen=True, slots=True)
class RetainedReplayRun:
    """Actual prepared apply receipts emitted during one synthetic replay."""

    receipts: tuple[PreparedRevisionReplayResult | RevisionCensusResult, ...]

    @property
    def scanned(self) -> int:
        return sum(receipt.scanned for receipt in self.receipts)

    @property
    def classified_full(self) -> int:
        return sum(receipt.classified_full for receipt in self.receipts)

    @property
    def replayed_logical_sources(self) -> int:
        return sum(
            receipt.replayed_logical_sources
            for receipt in self.receipts
            if isinstance(receipt, PreparedRevisionReplayResult)
        )

    @property
    def quarantined(self) -> int:
        return sum(receipt.quarantined for receipt in self.receipts)

    @property
    def adoption_deferred(self) -> int:
        return sum(
            receipt.adoption_deferred for receipt in self.receipts if isinstance(receipt, PreparedRevisionReplayResult)
        )


def _replay_on_creator(
    archive_root: Path,
    compute_adapter: BoundedComputeAdapter,
    seeds: tuple[str, ...],
    active_index_path: Path | None,
    owned_generation: IndexGeneration | None,
) -> RetainedReplayRun:
    adapter = make_raw_observation_derivation(
        archive_root,
        compute_adapter=compute_adapter,
        index_db_path=active_index_path,
        owned_generation=owned_generation,
    )
    frame = raw_observation_frame(archive_root, raw_ids=seeds, index_db_path=active_index_path)
    receipts: list[PreparedRevisionReplayResult | RevisionCensusResult] = []
    visited: set[str] = set()
    for raw_id in seeds:
        if raw_id in visited:
            continue
        while True:
            check_compute_cancelled()
            replacement = adapter.compute(frame, raw_id, replay_current=True)
            phases: list[str] = []

            def record(
                phase: Literal["census", "classification", "replay"],
                receipt: RevisionCensusResult | PreparedRevisionReplayResult,
                phases: list[str] = phases,
            ) -> None:
                phases.append(phase)
                receipts.append(receipt)

            with write_lease("synthetic-retained-replay", archive_root=archive_root):
                published = adapter.publish(frame, replacement, phase_receipt=record)
            if published:
                visited.update(replacement.raw_ids)
                break
            # A preparatory Source phase (census or classification) commits its
            # own receipt and changes the durable input binding; the next pass
            # prepares against it. A refusal that published no phase is surfaced.
            if not phases:
                raise RetainedPreparationRetryableError("canonical retained publication refused without progress")
    return RetainedReplayRun(tuple(receipts))


async def replay_retained_components_async(
    archive_root: Path,
    *,
    selected_raw_ids: Sequence[str] | None = None,
    active_index_path: Path | None = None,
    owned_generation: IndexGeneration | None = None,
) -> RetainedReplayRun:
    """Run the real captured preparation/publication route without fallback.

    Preparation runs on the daemon raw owner's admitted worker. A refused
    attempt with no progress is surfaced; this harness never adds a timeout,
    a retry count or a substitute result.
    """
    with readonly_connection_context(archive_root / "source.db") as source:
        retained = tuple(str(row[0]) for row in source.execute("SELECT raw_id FROM raw_sessions ORDER BY rowid"))
    selected = frozenset(selected_raw_ids) if selected_raw_ids is not None else None
    seeds = tuple(raw_id for raw_id in retained if selected is None or raw_id in selected)
    if owned_generation is not None:
        if active_index_path is not None and active_index_path.resolve() != Path(owned_generation.index_path).resolve():
            raise ValueError("retained fixture Index differs from its exact owned generation")
        active_index_path = Path(owned_generation.index_path)
    async with prepared_live_convergence_owner(archive_root) as owner:
        return await owner.run_convergence_sync(
            "test.retained-replay",
            _replay_on_creator,
            archive_root,
            owner._compute_adapter,
            seeds,
            active_index_path,
            owned_generation,
        )


def replay_retained_components(
    archive_root: Path,
    *,
    selected_raw_ids: Sequence[str] | None = None,
    active_index_path: Path | None = None,
    owned_generation: IndexGeneration | None = None,
) -> RetainedReplayRun:
    """Synchronous form of :func:`replay_retained_components_async`."""
    return asyncio.run(
        replay_retained_components_async(
            archive_root,
            selected_raw_ids=selected_raw_ids,
            active_index_path=active_index_path,
            owned_generation=owned_generation,
        )
    )
