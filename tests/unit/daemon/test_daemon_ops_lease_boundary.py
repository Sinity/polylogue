"""The daemon's ingest path must not ask to be admitted as a writer to read.

polylogue-ol2fc. With the single-writer boundary armed -- which is exactly the
state a real ``polylogued run`` is in -- four sites on the live-ingest path
opened a write-mode archive connection from a context holding no lease, or
bound to the wrong archive root. Each one aborted the whole intake page, so a
daemon that started, bound its API and watched its sources ingested nothing at
all: ``daemon.intake.page_refused ... reason=page_admission_failed`` for every
file, forever.

Production dependencies exercised here, all through their real entry points:

* ``CursorStore.get_records`` and its sibling ops-tier readers
  (``sources/live/cursor.py``) -- a SELECT is not a write and must not demand
  the lease, while an ops-tier *write* still must.
* ``LiveBatchProcessor.ingest_files`` (``sources/live/batch.py``) -- the
  daemon's batch event is an ops-tier publication and takes the writer through
  the same ``_run_sync`` admission as every other one.
* ``polylogue.daemon.cli._drain_convergence_debt_once`` -- the maintenance
  drain reads the ledger and writes it back under ``admit_stage_write``; it
  must not bootstrap the ops tier outside a lease on the way in.
* ``ArchiveStore.open_cold_build_generation`` -> ``_ensure_source_conn``
  (``storage/sqlite/archive_tiers/archive.py``) -- the generation's
  ``source.db`` is a read-through symlink to the declared archive's durable
  tier, so its lease binding is the declared root, not the candidate directory.

Anti-vacuity: each test names the production edit whose revert makes it red,
and ``test_ops_tier_writes_still_require_the_write_lease`` pins the opposite
direction, so "stop enforcing the lease on ops.db" cannot pass this module.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.write_lease import (
    UnleasedWriteError,
    arm_write_lease_enforcement,
    require_write_lease,
    write_lease,
)


def _bootstrapped_root(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    return root


def test_ops_tier_reads_need_no_write_lease(tmp_path: Path) -> None:
    """Every read-only ``CursorStore`` query works with the boundary armed.

    Revert ``_connect_ops_read`` (route these back through ``_connect_ops``)
    and each call below raises ``UnleasedWriteError``; ``get_records`` is the
    one the daemon hit on every intake page.
    """
    root = _bootstrapped_root(tmp_path)
    store = CursorStore(root / "index.db")
    probe = root / "corpus" / "session.jsonl"

    with arm_write_lease_enforcement():
        assert store.get_records([probe]) == {}
        assert store.get_record(probe) is None
        assert store.list_excluded() == []
        assert store.list_failed_with_retry() == []
        assert store.list_convergence_debt(limit=5) == []
        assert store.recent_ingest_attempts(limit=5) == []
        assert store.open_whole_archive_convergence_pledges() == ()


def test_ops_tier_writes_still_require_the_write_lease(tmp_path: Path) -> None:
    """The opposite direction: making reads lease-free must not free writes.

    A blanket "ops.db no longer needs the lease" change would pass the read
    test above; it fails here.
    """
    root = _bootstrapped_root(tmp_path)
    store = CursorStore(root / "index.db")
    probe = root / "corpus" / "session.jsonl"

    with arm_write_lease_enforcement(), pytest.raises(UnleasedWriteError):
        store.set(probe, 128, record_count=1, source_name="claude-code")


@pytest.mark.asyncio
async def test_batch_event_is_published_through_the_daemon_writer(tmp_path: Path) -> None:
    """``ingest_files`` emits its batch event inside an admitted write section.

    The emitter here asserts the lease the daemon's real emitter needs to
    append to the ops-tier event ledger. Revert the ``_run_sync`` wrapper
    around ``self._event_emitter(...)`` and the emitter runs on the calling
    task with no lease held, so this raises ``UnleasedWriteError`` at the very
    end of an otherwise complete batch -- the shape that made the daemon
    report ``page_refused`` after doing the work.
    """
    from polylogue.sources.live.batch import LiveBatchProcessor

    root = _bootstrapped_root(tmp_path)
    store = CursorStore(root / "index.db")
    admitted_actors: list[str] = []
    emitted: list[str] = []

    async def sync_runner(actor: str, function: Any, /, *args: Any, **kwargs: Any) -> Any:
        admitted_actors.append(actor)
        with write_lease(actor, archive_root=root):
            return function(*args, **kwargs)

    def emitter(kind: str, payload: dict[str, object]) -> None:
        # Stands in for ``daemon.cli._emit_live_batch_event``, whose ops-tier
        # ledger append opens a write-mode connection.
        require_write_lease("live batch event ledger", archive_root=root)
        emitted.append(kind)

    class _StubPolylogue:
        archive_root = root
        backend: object | None = None
        config: object | None = None

    source_root = type("SourceRoot", (), {"name": "claude-code", "root": root / "corpus"})()
    processor = LiveBatchProcessor(
        _StubPolylogue(),
        [source_root],
        cursor=store,
        parser_fingerprint="test-fp",
        event_emitter=emitter,
        sync_runner=sync_runner,
    )

    with arm_write_lease_enforcement():
        await processor.ingest_files([])

    assert emitted == ["ingestion_batch"]
    assert "watcher.live_ingest.ops.batch_event" in admitted_actors


def test_convergence_debt_drain_reads_the_ledger_without_a_lease(tmp_path: Path) -> None:
    """The maintenance drain reads debt with the boundary armed.

    Revert ``CursorStore(db, initialize=False)`` in
    ``_drain_convergence_debt_once`` and constructing the store bootstraps the
    ops tier through an unleased ``sqlite3.connect``, so every debt pass dies
    with ``UnleasedWriteError`` before it reads a single row.
    """
    from polylogue.daemon.cli import _drain_convergence_debt_once

    root = _bootstrapped_root(tmp_path)

    with arm_write_lease_enforcement():
        assert _drain_convergence_debt_once(root / "index.db") == 0


def test_cold_build_generation_binds_source_writes_to_the_declared_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cold-build generation writes ``source.db`` under the archive's lease.

    The generation directory carries read-through symlinks to the durable
    tiers, so the physical file is the declared archive's own ``source.db``.
    Revert ``_ensure_source_conn`` to ``archive_root=self.archive_root`` and
    the lease check rejects it as "outside the archive bound to writer", which
    is what failed every live-ingest file once the page was admitted.
    """
    from polylogue.storage.index_generation import IndexGenerationStore
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = _bootstrapped_root(tmp_path)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))

    generations = IndexGenerationStore.for_archive_root(root)
    generation = generations.create(source_snapshot="ol2fc-snapshot")
    generation_root = Path(generation.index_path).parent
    assert (generation_root / "source.db").is_symlink()

    with (
        write_lease("test.cold_build", archive_root=root),
        arm_write_lease_enforcement(),
        ArchiveStore.open_cold_build_generation(
            generation_root,
            generation_id=generation.generation_id,
            owner_id=generation.owner_id,
        ) as archive,
    ):
        conn = archive._ensure_source_conn()
        assert isinstance(conn, sqlite3.Connection)
