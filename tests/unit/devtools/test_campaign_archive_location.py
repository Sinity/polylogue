"""Regression coverage for the phantom-``benchmark.db`` campaign bug (polylogue-ovme.3).

The historical bug: synthetic benchmark/scale campaigns generated a real
archive under an output directory, then handed a root-shaped sentinel path
(``archive_dir / "benchmark.db"``) to benchmark runners. Some runners
(``SQLiteBackend``) canonicalize any non-``index.db`` filename to
``<parent>/index.db`` before opening it; others
(``polylogue.storage.sqlite.connection.open_connection``, used directly by
``run_fts_rebuild_campaign`` and ``run_session_insight_materialization_campaign``)
open the literal path handed to them with no such canonicalization. Handing
the *same* sentinel to both kinds of consumer silently produced two SQLite
files: the real generated archive at ``index.db``, and an empty phantom
``benchmark.db`` that some runners quietly measured instead of the real
data.

``test_open_connection_on_raw_sentinel_reproduces_phantom_benchmark_db``
below reproduces that exact failure mode using the runner-facing
``open_connection`` path unchanged (it still takes a raw ``Path`` -- that is
correct; the fix is that campaigns must never hand it an invented sentinel).
``test_campaign_archive_location_prevents_phantom_benchmark_db`` proves the
fix: routing the same real generated archive through
``CampaignArchiveLocation.active_index_path`` never creates the phantom file
and always observes the real generated data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from devtools.campaign_archive_location import CampaignArchiveLocation
from devtools.large_archive_generator import ArchiveSpec, ScaleLevel, generate_archive
from devtools.synthetic_benchmark_runtime import _db_row_counts
from polylogue.storage.archive_identity import ArchiveOwnershipError
from polylogue.storage.sqlite.connection import open_connection


def _tiny_spec() -> ArchiveSpec:
    """A minimal ArchiveSpec producing a handful of real sessions/messages."""
    return ArchiveSpec(
        level=ScaleLevel.SMALL,
        provider_mix={"codex": 1.0},
        message_count=10,
        sessions=2,
        avg_messages_per_conv=4,
        content_blocks_ratio=0.0,
        seed=7,
    )


@pytest.mark.asyncio
async def test_generate_archive_writes_active_index_never_a_benchmark_db_sentinel(tmp_path: Path) -> None:
    """generate_archive must mutate the real active index.db, not invent a sentinel file."""
    archive_dir = tmp_path / "archive-small"

    with CampaignArchiveLocation.acquire(archive_dir) as location:
        metrics = await generate_archive(_tiny_spec(), archive_dir, location=location)

    assert metrics.session_count == 2
    assert metrics.message_count > 0
    assert (archive_dir / "index.db").exists()
    assert not (archive_dir / "benchmark.db").exists()

    real_stats = _db_row_counts(archive_dir / "index.db")
    assert real_stats["sessions_count"] == 2
    assert real_stats["messages_count"] == metrics.message_count


@pytest.mark.asyncio
async def test_open_connection_on_raw_sentinel_reproduces_phantom_benchmark_db(tmp_path: Path) -> None:
    """Reproduce the exact historical bug mechanism with today's runner-facing primitives.

    This is the ANTI-VACUITY canary for AC#4: it proves the failure this
    bead fixes is real by driving the same primitive
    (``polylogue.storage.sqlite.connection.open_connection``) that
    ``run_fts_rebuild_campaign``/``run_session_insight_materialization_campaign``
    use internally, against the literal ``archive_dir / "benchmark.db"``
    sentinel that campaigns used to hand them -- exactly what
    ``devtools/run_campaign.py`` and ``devtools/benchmark_campaigns.py`` did
    before this fix (see git history: both files previously derived
    ``db_path = archive_dir / "benchmark.db"`` and passed it straight
    through). It does NOT modify production code to demonstrate this: the
    ambiguity lives entirely in what path a *caller* constructs and passes
    in, which is exactly the boundary this bead's fix (routing every
    campaign call through ``CampaignArchiveLocation.active_index_path``
    instead) closes.
    """
    archive_dir = tmp_path / "archive-small"

    with CampaignArchiveLocation.acquire(archive_dir) as location:
        metrics = await generate_archive(_tiny_spec(), archive_dir, location=location)
    assert metrics.message_count > 0

    real_stats = _db_row_counts(archive_dir / "index.db")
    assert real_stats["sessions_count"] == 2
    assert real_stats["messages_count"] > 0

    # This is the old, buggy sentinel: a filename that was never part of the
    # real archive's tier set at all.
    phantom_sentinel = archive_dir / "benchmark.db"
    assert not phantom_sentinel.exists(), "phantom sentinel must not pre-exist before reproducing the bug"

    # `open_connection(db_path)` is exactly what `run_fts_rebuild_campaign`/
    # `run_session_insight_materialization_campaign` call internally, with no
    # canonicalization of a non-"index.db" filename (unlike SQLiteBackend).
    # Passing the literal sentinel here is what those runners received from
    # the pre-fix campaign wiring (`archive_dir / "benchmark.db"`).
    with open_connection(phantom_sentinel) as conn:
        conn.execute("SELECT 1")

    # The bug reproduced: a brand-new, empty SQLite file was silently
    # created at the sentinel path, completely disjoint from the real
    # generated archive -- exactly the "phantom benchmark.db" symptom.
    assert phantom_sentinel.exists(), "expected open_connection to have silently created the phantom sentinel file"
    phantom_stats = _db_row_counts(phantom_sentinel)
    assert phantom_stats.get("messages_count", 0) == 0
    assert phantom_stats.get("sessions_count", 0) == 0
    # And the phantom file is a genuinely distinct file from the real index.
    assert phantom_sentinel.resolve() != (archive_dir / "index.db").resolve()
    assert phantom_stats.get("db_size_bytes", 0) != real_stats.get("db_size_bytes", 0)


@pytest.mark.asyncio
async def test_campaign_archive_location_prevents_phantom_benchmark_db(tmp_path: Path) -> None:
    """The fix: routing through CampaignArchiveLocation never creates a phantom file."""
    archive_dir = tmp_path / "archive-small"

    with CampaignArchiveLocation.acquire(archive_dir) as location:
        metrics = await generate_archive(_tiny_spec(), archive_dir, location=location)

        # A campaign reopening the archive later in the same run (as
        # devtools/benchmark_campaigns.py::run_full_campaign now does for
        # every registered benchmark) must go through active_index_path.
        reopened_path = location.active_index_path
        assert reopened_path == archive_dir / "index.db"

        stats = _db_row_counts(reopened_path)
        assert stats["sessions_count"] == 2
        assert stats["messages_count"] == metrics.message_count

    # No phantom sentinel file was ever created anywhere in the archive dir.
    assert not (archive_dir / "benchmark.db").exists()
    on_disk = {path.name for path in archive_dir.iterdir() if path.is_file()}
    assert "benchmark.db" not in on_disk


@pytest.mark.asyncio
async def test_campaign_archive_location_acquire_fails_before_any_sqlite_open_when_owned(tmp_path: Path) -> None:
    """A second concurrent acquire over the same root must fail closed before opening SQLite."""
    archive_dir = tmp_path / "archive-small"
    archive_dir.mkdir(parents=True)

    first = CampaignArchiveLocation.acquire(archive_dir)
    try:
        with pytest.raises(ArchiveOwnershipError):
            CampaignArchiveLocation.acquire(archive_dir)
        # Ownership failed before any tier file was created.
        assert not (archive_dir / "index.db").exists()
    finally:
        first.release()


@pytest.mark.asyncio
async def test_active_index_path_fails_stale_after_concurrent_generation_promotion(tmp_path: Path) -> None:
    """active_index_path re-resolves on every call rather than caching the acquire-time value.

    A concurrent generation promotion (e.g. a b5l activation swapping the
    ``.index-active-pointer`` target mid-campaign) must be caught as a stale
    ownership proof, not silently followed to a path this campaign never
    proved it owns.
    """
    archive_dir = tmp_path / "archive-small"
    archive_dir.mkdir(parents=True)

    with CampaignArchiveLocation.acquire(archive_dir) as location:
        assert location.active_index_path == archive_dir / "index.db"

        # Promote a generation elsewhere and repoint the active-index pointer,
        # mirroring what a concurrent b5l activation would do.
        generation_dir = tmp_path / "generation-2"
        generation_dir.mkdir(parents=True)
        promoted = generation_dir / "index.db"
        promoted.write_bytes(b"")
        (archive_dir / ".index-active-pointer").write_text(str(promoted), encoding="utf-8")

        with pytest.raises(ArchiveOwnershipError):
            _ = location.active_index_path
