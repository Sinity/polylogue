"""Drive the daemon's raw-observation owner from fixtures.

Append acquisition and retained replay publish only through
``RawObservationConvergenceOwner``. These helpers borrow that real owner (its
compute adapter, write coordinator and settlement) for one call; they add no
alternate preparation or publication path.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager, closing
from pathlib import Path
from typing import TYPE_CHECKING, Any

from polylogue.sources.live import WatchSource
from polylogue.sources.live.cold_build import (
    ColdBuildGeneration,
    clear_cold_build_generation,
    register_cold_build_generation,
)
from tests.infra.archive_templates import run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner

if TYPE_CHECKING:
    from polylogue.sources.live.batch_support import _AppendPlan, _AppendResult
    from polylogue.sources.revision_backfill import PreparedRevisionReplayResult


def _owner_archive_root(owner: Any) -> Path:
    return Path(getattr(owner._polylogue, "archive_root", owner._cursor._db_path.parent))


async def ingest_append_with_owner_async(owner: Any, plans: list[_AppendPlan]) -> _AppendResult:
    """Acquire, prepare and publish append plans on the actual raw owner."""
    async with prepared_live_convergence_owner(_owner_archive_root(owner)) as raw_owner:
        return await raw_owner.ingest_append_plans(owner, plans)


def ingest_append_with_owner(owner: Any, plans: list[_AppendPlan]) -> _AppendResult:
    """Synchronous form of :func:`ingest_append_with_owner_async`."""
    return asyncio.run(ingest_append_with_owner_async(owner, plans))


def retained_raw_ids(archive_root: Path) -> tuple[str, ...]:
    """Every retained raw, in acquisition order."""
    with closing(sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)) as conn:
        return tuple(str(row[0]) for row in conn.execute("SELECT raw_id FROM raw_sessions ORDER BY rowid"))


async def replay_retained_raws_async(
    archive_root: Path, raw_ids: Sequence[str] | None = None
) -> tuple[PreparedRevisionReplayResult, ...]:
    """Replay retained raws into the active Index through the owner's replay route."""
    selected = retained_raw_ids(archive_root) if raw_ids is None else tuple(raw_ids)
    async with prepared_live_convergence_owner(archive_root) as raw_owner:
        return await raw_owner.replay_retained_raw_ids(selected)


def replay_retained_raws(
    archive_root: Path, raw_ids: Sequence[str] | None = None
) -> tuple[PreparedRevisionReplayResult, ...]:
    """Synchronous form of :func:`replay_retained_raws_async`."""
    return asyncio.run(replay_retained_raws_async(archive_root, raw_ids))


@asynccontextmanager
async def cold_rebuilt_index(archive_root: Path) -> AsyncIterator[Path]:
    """Build a fresh Index generation from retained Source evidence alone.

    The owned generation receives every retained raw through the cold-build
    replay route; the caller reads its Index while it exists. The generation is
    discarded on exit and the active Index is never touched.
    """
    raws = retained_raw_ids(archive_root)

    def begin() -> ColdBuildGeneration:
        return ColdBuildGeneration.begin(
            archive_root, reason="test-retained-rebuild", sources=(WatchSource("fixture", archive_root / "absent"),)
        )

    generation = await run_archive_fixture_write(archive_root, begin)
    register_cold_build_generation(generation)
    try:
        async with prepared_live_convergence_owner(archive_root) as raw_owner:
            await raw_owner.replay_retained_raw_ids(raws)
        await run_archive_fixture_write(archive_root, generation.prepare_promotion_candidate)
        yield Path(generation.generation.index_path)
    finally:
        clear_cold_build_generation()
        await run_archive_fixture_write(archive_root, generation.discard)


__all__ = [
    "cold_rebuilt_index",
    "ingest_append_with_owner",
    "ingest_append_with_owner_async",
    "replay_retained_raws",
    "replay_retained_raws_async",
    "retained_raw_ids",
]
