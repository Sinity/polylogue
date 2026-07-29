"""polylogue-623q: the bulk-build pragma profile is scoped to owned inactive
index generations only -- it must never leak onto the live active-archive
writer connection.

Production dependencies exercised here: ``ArchiveStore.__init__`` (the real
constructor, not a reimplementation), ``ArchiveStore.open_owned_inactive_generation``,
``ArchiveStore.open_existing``, and ``IndexGenerationStore.create`` (the real
generation bootstrap the offline rebuild uses). Reverting the
``bulk_build_profile`` wiring in ``ArchiveStore._initialize_store`` (or
reverting ``BULK_BUILD_WRITE_CONNECTION_PROFILE`` back to the live profile)
makes ``test_owned_inactive_generation_uses_bulk_build_pragma_profile`` fail,
not merely produce a wrong number.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.storage.index_generation import IndexGenerationStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.connection_profile import (
    BULK_BUILD_CACHE_SIZE_KIB,
    WRITE_CACHE_SIZE_KIB,
)


def test_owned_inactive_generation_uses_bulk_build_pragma_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))

    store = IndexGenerationStore.for_archive_root(root)
    generation = store.create(source_snapshot="snapshot-a")
    generation_root = Path(generation.index_path).parent

    with ArchiveStore.open_owned_inactive_generation(
        generation_root, generation_id=generation.generation_id, owner_id=generation.owner_id
    ) as archive:
        journal_mode = archive._conn.execute("PRAGMA journal_mode").fetchone()[0]
        synchronous = archive._conn.execute("PRAGMA synchronous").fetchone()[0]
        cache_size = archive._conn.execute("PRAGMA cache_size").fetchone()[0]

    assert journal_mode.lower() == "memory"
    # PRAGMA synchronous readback is an integer: 0=OFF, 1=NORMAL, 2=FULL.
    assert synchronous == 0
    assert abs(cache_size) == BULK_BUILD_CACHE_SIZE_KIB


def test_live_active_archive_writer_keeps_wal_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact same constructor, without ``owned_inactive_generation``, must
    keep the live writer's crash-safe WAL/NORMAL profile unchanged."""
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))

    with ArchiveStore.open_existing(root, read_only=False) as archive:
        journal_mode = archive._conn.execute("PRAGMA journal_mode").fetchone()[0]
        synchronous = archive._conn.execute("PRAGMA synchronous").fetchone()[0]
        cache_size = archive._conn.execute("PRAGMA cache_size").fetchone()[0]

    assert journal_mode.lower() == "wal"
    assert synchronous == 1  # NORMAL
    assert abs(cache_size) == WRITE_CACHE_SIZE_KIB
