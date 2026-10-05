"""Drive canonical retained preparation and record actual publication results."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.sources.revision_backfill import PreparedRevisionReplayResult, RevisionCensusResult
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.index_generation import IndexGeneration
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.connection_profile import readonly_connection_context
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner


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


def replay_retained_components(
    archive_root: Path,
    *,
    selected_raw_ids: Sequence[str] | None = None,
    active_index_path: Path | None = None,
    owned_generation: IndexGeneration | None = None,
) -> RetainedReplayRun:
    """Run the real captured preparation/publication route without fallback.

    The canonical daemon owner settles preparatory Source phases and replays
    each selected retained component on its admitted compute creator. It binds
    to the registered cold-build generation itself; a fixture naming another
    destination is refused rather than redirected. A refused attempt with no
    progress surfaces the owner's typed error; this harness never adds a
    timeout, a retry count or a substitute result.
    """
    from polylogue.sources.live.cold_build import active_cold_build_generation

    with readonly_connection_context(archive_root / "source.db") as source:
        retained = tuple(str(row[0]) for row in source.execute("SELECT raw_id FROM raw_sessions ORDER BY rowid"))
    selected = frozenset(selected_raw_ids) if selected_raw_ids is not None else None
    seeds = tuple(raw_id for raw_id in retained if selected is None or raw_id in selected)
    cold_build = active_cold_build_generation(archive_root)
    registered = None if cold_build is None else cold_build.generation
    if owned_generation is not None:
        if active_index_path is not None and active_index_path.resolve() != Path(owned_generation.index_path).resolve():
            raise ValueError("retained fixture Index differs from its exact owned generation")
        if registered is None or Path(registered.index_path).resolve() != Path(owned_generation.index_path).resolve():
            raise ValueError("retained fixture generation is not the registered cold-build destination")
    elif active_index_path is not None:
        expected = (
            ArchiveLocation.resolve(archive_root).active_index_path
            if registered is None
            else Path(registered.index_path)
        )
        if active_index_path.resolve() != expected.resolve():
            raise ValueError("retained fixture Index is not the owner's actual destination")

    async def run() -> tuple[PreparedRevisionReplayResult, ...]:
        async with prepared_live_convergence_owner(archive_root) as owner:
            return await owner.replay_retained_raw_ids(seeds)

    receipts: tuple[PreparedRevisionReplayResult | RevisionCensusResult, ...] = tuple(asyncio.run(run()))
    return RetainedReplayRun(receipts)


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
                provider=provider, payload=payload, source_path=source_path, acquired_at_ms=acquired_at_ms
            )

    raw_id = await run_archive_fixture_write(archive_root, acquire)
    async with prepared_live_convergence_owner(archive_root) as owner:
        receipts = await owner.replay_retained_raw_ids((raw_id,))
    written = tuple(sorted({key for receipt in receipts for key in receipt.written_session_ids}))
    return raw_id, written
