"""Daemon-side wiring for the periodic blob-GC drain (automagic-invariants gap).

``polylogue ops maintenance blob-gc`` was, before this module existed, the
*only* path that reclaimed unreferenced blobs — every other mechanical,
non-judgment maintenance operation (FTS merge, WAL checkpoint, embedding
orphan reconcile) already has a daemon-owned periodic equivalent. These
tests exercise ``run_blob_gc_once`` — the bounded sync helper the periodic
daemon loop (``periodic_blob_gc_check``) invokes through the write
coordinator — against real on-disk blob-store fixtures, and prove the
production route (``DaemonWriteCoordinator.run_sync``) actually performs
the reclaim rather than a bypassed direct call.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sqlite3
from pathlib import Path

import pytest

from polylogue.daemon import blob_gc_periodic
from polylogue.daemon.blob_gc_periodic import run_blob_gc_once
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteEvent
from polylogue.storage.blob_store import BlobStore


def _make_source_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                blob_hash BLOB NOT NULL,
                blob_size INTEGER NOT NULL DEFAULT 0,
                acquired_at TEXT NOT NULL DEFAULT ''
            )"""
        )
        conn.execute(
            """CREATE TABLE blob_refs (
                blob_hash BLOB NOT NULL CHECK(length(blob_hash) = 32),
                ref_id TEXT NOT NULL,
                ref_type TEXT NOT NULL CHECK(ref_type IN ('raw_payload', 'attachment', 'sidecar')),
                source_path TEXT,
                size_bytes INTEGER NOT NULL DEFAULT 0 CHECK(size_bytes >= 0),
                acquired_at_ms INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (blob_hash, ref_type, ref_id)
            )"""
        )
        conn.execute(
            """CREATE TABLE gc_generations (
                generation_id   TEXT PRIMARY KEY,
                started_at_ms   INTEGER NOT NULL,
                completed_at_ms INTEGER,
                reclaimed_count INTEGER NOT NULL DEFAULT 0,
                reclaimed_bytes INTEGER NOT NULL DEFAULT 0
            )"""
        )
        conn.commit()
    finally:
        conn.close()


def _backdate(blob_store: BlobStore, blob_hash: str) -> None:
    """Backdate a blob's mtime well past MIN_AGE_S (60s) so it is GC-eligible.

    A fixed epoch (not a `time.time()`-relative offset) avoids any host-clock
    read in the test itself -- GC eligibility only needs "old enough", not a
    real elapsed duration.
    """
    os.utime(blob_store.blob_path(blob_hash), (0, 0))


def test_run_blob_gc_once_returns_none_when_blob_dir_absent(tmp_path: Path) -> None:
    db_path = tmp_path / "source.db"
    _make_source_db(db_path)
    result = run_blob_gc_once(db_path, tmp_path / "blob")
    assert result is None


def test_run_blob_gc_once_returns_none_when_source_db_absent(tmp_path: Path) -> None:
    blob_dir = tmp_path / "blob"
    blob_dir.mkdir()
    result = run_blob_gc_once(tmp_path / "source.db", blob_dir)
    assert result is None


def test_run_blob_gc_once_reclaims_unreferenced_aged_blob(tmp_path: Path) -> None:
    db_path = tmp_path / "source.db"
    _make_source_db(db_path)
    blob_dir = tmp_path / "blob"
    blob_store = BlobStore(blob_dir)

    blob_hash, _ = blob_store.write_from_bytes(b"orphan blob for daemon gc")
    _backdate(blob_store, blob_hash)

    result = run_blob_gc_once(db_path, blob_dir)

    assert result is not None
    assert result.deleted_count == 1
    assert not blob_store.blob_path(blob_hash).exists()


def test_daemon_coordinator_owns_real_blob_gc_mutation(tmp_path: Path) -> None:
    """Production-route proof; bypassing run_sync or its authority token makes this fail."""
    db_path = tmp_path / "source.db"
    _make_source_db(db_path)
    blob_dir = tmp_path / "blob"
    blob_store = BlobStore(blob_dir)

    blob_hash, _ = blob_store.write_from_bytes(b"orphan blob via coordinator")
    _backdate(blob_store, blob_hash)

    coordinator = DaemonWriteCoordinator()

    async def run() -> object:
        return await coordinator.run_sync("maintenance.blob_gc", run_blob_gc_once, db_path, blob_dir)

    result = asyncio.run(run())

    assert result is not None
    assert result.deleted_count == 1  # type: ignore[attr-defined]
    assert coordinator.snapshot().active_actor is None
    assert not blob_store.blob_path(blob_hash).exists()


def _make_publication_reconciliation_fixture(tmp_path: Path) -> tuple[Path, str]:
    from polylogue.core.enums import Origin
    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session_blob_ref

    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    source_db = archive_root / "source.db"
    store = BlobStore(archive_root / "blob")
    publisher = ArchiveBlobPublisher(source_db, store.root)
    missing_hash, _ = publisher.write_from_bytes(b"periodic-missing-terminal")
    referenced_hash, referenced_size = publisher.write_from_bytes(b"periodic-referenced-terminal")
    unresolved_hash, _ = publisher.write_from_bytes(b"periodic-unresolved")
    publisher.flush()
    store.blob_path(missing_hash).unlink()
    with sqlite3.connect(source_db) as conn:
        write_source_raw_session_blob_ref(
            conn,
            origin=Origin.CHATGPT_EXPORT,
            source_path="periodic-referenced.json",
            source_index=0,
            blob_hash=bytes.fromhex(referenced_hash),
            blob_size=referenced_size,
            acquired_at_ms=1,
            raw_id="periodic-referenced-raw",
        )
    return archive_root, unresolved_hash


def test_periodic_publication_reconciliation_repeats_safe_cleanup_and_retains_unresolved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The scheduled production route clears terminal rows on every tick only."""
    archive_root, unresolved_hash = _make_publication_reconciliation_fixture(tmp_path)
    source_db = archive_root / "source.db"
    coordinator_events: list[str] = []
    second_tick = asyncio.Event()

    def observe(event: DaemonWriteEvent) -> None:
        if event.phase == "acquired":
            coordinator_events.append(event.actor)
            if len(coordinator_events) == 2:
                second_tick.set()

    coordinator = DaemonWriteCoordinator(observer=observe)
    monkeypatch.setattr(blob_gc_periodic, "BLOB_PUBLICATION_RECONCILIATION_INTERVAL_SECONDS", 0)
    monkeypatch.setattr("polylogue.paths.archive_root", lambda: archive_root)
    monkeypatch.setattr("polylogue.paths.source_db_path", lambda: source_db)
    monkeypatch.setattr("polylogue.daemon.cli.daemon_write_coordinator", lambda: coordinator)

    async def exercise() -> None:
        task = asyncio.create_task(blob_gc_periodic.periodic_blob_publication_reconciliation_check())
        try:
            await asyncio.wait_for(second_tick.wait(), timeout=2.0)
        finally:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            assert await coordinator.shutdown(timeout=1.0)

    asyncio.run(exercise())

    assert coordinator_events == [
        "maintenance.blob_publication_reconciliation",
        "maintenance.blob_publication_reconciliation",
    ]
    assert coordinator.snapshot().active_actor is None
    with sqlite3.connect(source_db) as conn:
        remaining = conn.execute(
            "SELECT blob_hash FROM blob_publication_reservations ORDER BY publication_id"
        ).fetchall()
    assert [bytes(row[0]).hex() for row in remaining] == [unresolved_hash]


def test_periodic_publication_reconciliation_pages_past_unresolved_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A bounded pass advances past retained rows instead of starving later cleanup."""
    from polylogue.core.enums import Origin
    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
    from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session_blob_ref

    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    source_db = archive_root / "source.db"
    store = BlobStore(archive_root / "blob")
    publisher = ArchiveBlobPublisher(source_db, store.root)
    unresolved_a, _ = publisher.write_from_bytes(b"bounded-unresolved-a")
    unresolved_b, _ = publisher.write_from_bytes(b"bounded-unresolved-b")
    missing_hash, _ = publisher.write_from_bytes(b"bounded-missing")
    referenced_hash, referenced_size = publisher.write_from_bytes(b"bounded-referenced")
    receipts = publisher.flush()
    deterministic_ids = {
        unresolved_a: "publication-a",
        unresolved_b: "publication-b",
        missing_hash: "publication-c",
        referenced_hash: "publication-d",
    }
    store.blob_path(missing_hash).unlink()
    with sqlite3.connect(source_db) as conn:
        for receipt in receipts:
            conn.execute(
                "UPDATE blob_publication_reservations SET publication_id = ? WHERE publication_id = ?",
                (deterministic_ids[receipt.blob_hash], receipt.publication_id),
            )
        write_source_raw_session_blob_ref(
            conn,
            origin=Origin.CHATGPT_EXPORT,
            source_path="bounded-referenced.json",
            source_index=0,
            blob_hash=bytes.fromhex(referenced_hash),
            blob_size=referenced_size,
            acquired_at_ms=1,
            raw_id="bounded-referenced-raw",
        )

    acquired = asyncio.Event()
    coordinator_events: list[str] = []

    def observe(event: DaemonWriteEvent) -> None:
        if event.phase == "acquired":
            coordinator_events.append(event.actor)
            if len(coordinator_events) == 2:
                acquired.set()

    coordinator = DaemonWriteCoordinator(observer=observe)
    monkeypatch.setattr(blob_gc_periodic, "BLOB_PUBLICATION_RECONCILIATION_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(blob_gc_periodic, "BLOB_PUBLICATION_RECONCILIATION_MAX_BATCH", 2)
    monkeypatch.setattr("polylogue.paths.archive_root", lambda: archive_root)
    monkeypatch.setattr("polylogue.daemon.cli.daemon_write_coordinator", lambda: coordinator)

    async def exercise() -> None:
        task = asyncio.create_task(blob_gc_periodic.periodic_blob_publication_reconciliation_check())
        try:
            await asyncio.wait_for(acquired.wait(), timeout=2.0)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            assert await coordinator.shutdown(timeout=1.0)

    asyncio.run(exercise())

    assert coordinator_events == [
        "maintenance.blob_publication_reconciliation",
        "maintenance.blob_publication_reconciliation",
    ]
    with sqlite3.connect(source_db) as conn:
        remaining = conn.execute(
            "SELECT publication_id FROM blob_publication_reservations ORDER BY publication_id"
        ).fetchall()
    assert remaining == [("publication-a",), ("publication-b",)]


def test_blob_publication_reconciliation_reads_attachment_refs_from_active_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The daemon resolves the promoted generation before classifying receipts."""
    from polylogue.daemon.cli import _reconcile_blob_publications
    from polylogue.storage.blob_publication import ArchiveBlobPublisher
    from polylogue.storage.sqlite.archive_tiers.bootstrap import (
        initialize_active_archive_root,
        initialize_archive_database,
    )
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    source_db = archive_root / "source.db"
    store = BlobStore(archive_root / "blob")
    publisher = ArchiveBlobPublisher(source_db, store.root)
    blob_hash, size = publisher.write_from_bytes(b"active-generation-attachment")
    publisher.flush()

    active_index = archive_root / ".index-generations" / "gen-active" / "index.db"
    active_index.parent.mkdir(parents=True)
    initialize_archive_database(active_index, ArchiveTier.INDEX)
    with sqlite3.connect(active_index) as conn:
        conn.execute(
            """
            INSERT INTO attachments(attachment_id, blob_hash, byte_count, acquisition_status, ref_count)
            VALUES ('active-attachment', ?, ?, 'acquired', 1)
            """,
            (bytes.fromhex(blob_hash), size),
        )
        conn.commit()
    (archive_root / ".index-active-pointer").write_text(str(active_index.resolve()), encoding="utf-8")

    coordinator = DaemonWriteCoordinator()
    monkeypatch.setattr("polylogue.paths.archive_root", lambda: archive_root)
    monkeypatch.setattr("polylogue.daemon.cli.daemon_write_coordinator", lambda: coordinator)

    outcome = asyncio.run(_reconcile_blob_publications(actor="maintenance.blob_publication_reconciliation"))

    assert outcome is not None
    assert outcome.cleared_referenced == 1
    assert outcome.scanned == 1
    with sqlite3.connect(source_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 0
