"""``rebuild_index_from_source`` must prove archive-location ownership before
touching any generation directory or SQLite tier (polylogue-ovme.2 AC3).

An offline rebuild is exactly the maintenance/campaign writer
``OwnedArchiveLocation`` (polylogue-ovme.1, PR #3291) exists for. Before this
change, an offline rebuild never acquired that capability at all -- only
``RebuildLease`` (a rebuild-specific exclusion lock) guarded it, which does
not protect against a concurrent *different* maintenance/campaign writer
holding the general archive-location ownership lock.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.maintenance.rebuild_index import RebuildIndexRequest, rebuild_index_from_source_sync
from polylogue.storage.archive_identity import ArchiveLocation, ArchiveOwnershipError, OwnedArchiveLocation
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _init_empty_source(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    initialize_archive_database(root / "source.db", ArchiveTier.SOURCE)


def test_rebuild_refuses_when_archive_location_already_owned(tmp_path: Path) -> None:
    """A concurrent holder of the archive-location ownership lock must block
    an offline rebuild before any generation directory or SQLite tier is
    touched -- not merely before promotion, and not via ``RebuildLease``
    (a different, rebuild-specific lock) racing to the same conclusion.
    """
    root = tmp_path / "archive"
    _init_empty_source(root)
    location = ArchiveLocation.resolve(root)
    owned = OwnedArchiveLocation.acquire(location, owner_id="concurrent-campaign")
    try:
        with pytest.raises(ArchiveOwnershipError):
            rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root))
        # Failure happened before any generation bookkeeping was created.
        assert not (root / ".index-generations").exists()
        # The rebuild lease is now deliberately acquired before the general
        # archive-location ownership attempt.  Its released lock file may
        # remain as a diagnostic artifact, but no generation may be created.
    finally:
        owned.release()

    # Releasing the concurrent holder's ownership lets the rebuild proceed.
    receipt = rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root))
    assert receipt.status == "empty-source"


def test_rebuild_source_preflight_rejects_orphaned_blob_refs_before_generation_creation(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _init_empty_source(root)
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO blob_refs(blob_hash, ref_id, ref_type, size_bytes, acquired_at_ms)
            VALUES (?, 'raw-that-does-not-exist', 'raw_payload', 10, 100)
            """,
            (b"o" * 32,),
        )

    with pytest.raises(RuntimeError, match="reindex source preflight gate failed: blob-refs-liveness"):
        rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root))

    assert not (root / ".index-generations").exists()


def test_rebuild_source_preflight_rejects_unexplained_raw_failure(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _init_empty_source(root)
    with sqlite3.connect(root / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, blob_size,
                acquired_at_ms, parse_error
            ) VALUES ('raw-failed', 'codex-session', 'failed', '/x', ?, 10, 100, 'unexpected parser failure')
            """,
            (b"u" * 32,),
        )
        conn.commit()

    with pytest.raises(RuntimeError, match="raw-failure-lifecycle"):
        rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root))

    assert not (root / ".index-generations").exists()


def test_rebuild_preflight_exposes_unreconciled_source_ref_types(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    _init_empty_source(root)
    with sqlite3.connect(root / "source.db") as conn:
        conn.executemany(
            """
            INSERT INTO blob_refs(blob_hash, ref_id, ref_type, size_bytes, acquired_at_ms)
            VALUES (?, ?, ?, 10, 100)
            """,
            (
                (b"r" * 32, "raw-gone", "raw_payload"),
                (b"a" * 32, "attachment-gone", "attachment"),
                (b"h" * 32, "hook-gone", "hook_payload"),
            ),
        )

    with pytest.raises(RuntimeError) as exc_info:
        rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root))

    message = str(exc_info.value)
    assert "reindex source preflight gate failed: blob-refs-liveness" in message
    assert "raw_payload orphans=1" in message
    assert "attachment orphans=1" in message
    assert "hook_payload orphans=1" in message
    assert not (root / ".index-generations").exists()


def test_rebuild_releases_ownership_lock_after_completion(tmp_path: Path) -> None:
    """The ownership lock must not be left held after a rebuild returns, so a
    second rebuild (or any other maintenance/campaign writer) can acquire it.

    ``flock`` is scoped to the open file description, not the process, so a
    fresh ``acquire`` call from the *same* process only succeeds here if the
    first rebuild's ``owned.release()`` actually ran.
    """
    root = tmp_path / "archive"
    _init_empty_source(root)

    receipt = rebuild_index_from_source_sync(RebuildIndexRequest(archive_root=root))
    assert receipt.status == "empty-source"

    location = ArchiveLocation.resolve(root)
    owned = OwnedArchiveLocation.acquire(location, owner_id="post-rebuild-probe")
    try:
        assert (root / ".archive-ownership.lock").exists()
    finally:
        owned.release()
