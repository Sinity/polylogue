"""Drive canonical retained preparation and record actual publication results."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.enums import Provider, ValidationMode
from polylogue.operations.raw_observation_derivation import make_raw_observation_derivation, raw_observation_frame
from polylogue.sources.revision_backfill import (
    PreparedRevisionReplayResult,
    RetainedPreparationRetryableError,
    RevisionCensusResult,
)
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.connection_profile import readonly_connection_context
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from polylogue.core.compute import BoundedComputeAdapter
    from polylogue.storage.index_generation import IndexGeneration
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead


@dataclass(frozen=True, slots=True)
class RetainedReplayRun:
    """Actual prepared apply receipts emitted during one synthetic replay."""

    receipts: tuple[PreparedRevisionReplayResult | RevisionCensusResult, ...]
    #: The raw ids of each component the derivation published, in order.
    components: tuple[tuple[str, ...], ...] = ()

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


def _select(raw_ids: tuple[str, ...]) -> Callable[[PreparedSessionSourceRead], Sequence[str]]:
    def select(_reader: PreparedSessionSourceRead) -> Sequence[str]:
        return raw_ids

    return select


def _replay_on_creator(
    archive_root: Path,
    compute_adapter: BoundedComputeAdapter,
    seeds: tuple[str, ...],
    active_index_path: Path | None,
    owned_generation: IndexGeneration | None,
    validation_mode: ValidationMode,
) -> RetainedReplayRun:
    adapter = make_raw_observation_derivation(
        archive_root,
        compute_adapter=compute_adapter,
        index_db_path=active_index_path,
        owned_generation=owned_generation,
        validation_mode=validation_mode,
    )
    frame = raw_observation_frame(
        archive_root,
        raw_ids=seeds,
        index_db_path=active_index_path,
        validation_mode=validation_mode,
    )
    receipts: list[PreparedRevisionReplayResult | RevisionCensusResult] = []
    components: list[tuple[str, ...]] = []
    visited: set[str] = set()
    for raw_id in seeds:
        if raw_id in visited:
            continue
        # As the production owner does, a pass after a lineage deferral
        # re-prepares only the deferred children, not the keys it published.
        deferred_selection: tuple[str, ...] | None = None
        while True:
            check_compute_cancelled()
            replacement = adapter.compute(
                frame,
                raw_id,
                replay_current=True,
                select_retained_raw_ids=None if deferred_selection is None else _select(deferred_selection),
            )
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
                components.append(tuple(replacement.raw_ids))
                visited.update(replacement.raw_ids)
                break
            if "replay" in phases:
                # A lineage-deferral pass published its unit except the
                # deferred children, which this seed's next pass re-prepares.
                visited.update(set(replacement.raw_ids).difference(replacement.lineage_deferred_raw_ids))
                if replacement.lineage_deferred_raw_ids:
                    deferred_selection = replacement.lineage_deferred_raw_ids
            # A committed census, classification, byte restoration or deferred
            # parent publication is this key's own progress, which the adapter
            # reports exactly as it does to the derivation kernel; the next pass
            # prepares against it. A refusal that advanced nothing is surfaced.
            if not adapter.publication_advanced(replacement):
                raise RetainedPreparationRetryableError("canonical retained publication refused without progress")
    return RetainedReplayRun(tuple(receipts), tuple(components))


async def replay_retained_components_async(
    archive_root: Path,
    *,
    selected_raw_ids: Sequence[str] | None = None,
    active_index_path: Path | None = None,
    owned_generation: IndexGeneration | None = None,
    validation_mode: ValidationMode = ValidationMode.ADVISORY,
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
    from polylogue.sources.live.cold_build import active_cold_build_generation

    cold_build = active_cold_build_generation(archive_root)
    registered = None if cold_build is None else cold_build.generation
    if owned_generation is not None:
        if active_index_path is not None and active_index_path.resolve() != Path(owned_generation.index_path).resolve():
            raise ValueError("retained fixture Index differs from its exact owned generation")
        if registered is None or Path(registered.index_path).resolve() != Path(owned_generation.index_path).resolve():
            raise ValueError("retained fixture generation is not the registered cold-build destination")
        active_index_path = Path(owned_generation.index_path)
    elif active_index_path is not None:
        expected = (
            ArchiveLocation.resolve(archive_root).active_index_path
            if registered is None
            else Path(registered.index_path)
        )
        if active_index_path.resolve() != expected.resolve():
            raise ValueError("retained fixture Index is not the owner's actual destination")
    async with prepared_live_convergence_owner(archive_root, validation_mode=validation_mode) as owner:
        return await owner.run_convergence_sync(
            "test.retained-replay",
            _replay_on_creator,
            archive_root,
            owner._compute_adapter,
            seeds,
            active_index_path,
            owned_generation,
            validation_mode,
        )


def replay_retained_components(
    archive_root: Path,
    *,
    selected_raw_ids: Sequence[str] | None = None,
    active_index_path: Path | None = None,
    owned_generation: IndexGeneration | None = None,
    validation_mode: ValidationMode = ValidationMode.ADVISORY,
) -> RetainedReplayRun:
    """Synchronous form of :func:`replay_retained_components_async`."""
    return asyncio.run(
        replay_retained_components_async(
            archive_root,
            selected_raw_ids=selected_raw_ids,
            active_index_path=active_index_path,
            owned_generation=owned_generation,
            validation_mode=validation_mode,
        )
    )


async def publish_retained_payload(
    archive_root: Path,
    *,
    provider: Provider,
    payload: bytes,
    source_path: str,
    acquired_at_ms: int,
) -> tuple[str, tuple[str, ...]]:
    """Acquire real provider bytes, then publish them through the canonical owner.

    Returns the acquired raw ID and the session IDs its retained replay wrote.
    This replaces seeding a raw row beside an independently supplied parse: the
    indexed session is whatever the retained bytes actually parse to.
    """

    def acquire() -> str:
        bootstrap_archive_root(archive_root)
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=provider,
                payload=payload,
                source_path=source_path,
                canonical_source_path=source_path,
                acquired_at_ms=acquired_at_ms,
            )

    raw_id = await run_archive_fixture_write(archive_root, acquire)
    async with prepared_live_convergence_owner(archive_root) as owner:
        receipts = (await owner.replay_retained_raw_ids((raw_id,))).require_complete()
    written = tuple(sorted({key for receipt in receipts for key in receipt.written_session_ids}))
    return raw_id, written
