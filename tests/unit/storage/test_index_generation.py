from __future__ import annotations

import multiprocessing
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.storage.index_generation import (
    ActiveWriterLease,
    IndexGenerationStore,
    RebuildLease,
    RebuildLeaseUnavailableError,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _hold_lease(
    root: str, ready: multiprocessing.synchronize.Event, release: multiprocessing.synchronize.Event
) -> None:
    with RebuildLease(Path(root)):
        ready.set()
        release.wait(5)


def _archive(root: Path) -> None:
    for tier in (ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.EMBEDDINGS, ArchiveTier.OPS, ArchiveTier.INDEX):
        initialize_archive_database(root / f"{tier.value}.db", tier)


def test_rebuild_lease_excludes_competing_process(tmp_path: Path) -> None:
    ready = multiprocessing.Event()
    release = multiprocessing.Event()
    process = multiprocessing.Process(target=_hold_lease, args=(str(tmp_path), ready, release))
    process.start()
    assert ready.wait(5)
    try:
        with pytest.raises(RebuildLeaseUnavailableError):
            with RebuildLease(tmp_path):
                pass
    finally:
        release.set()
        process.join(5)
    assert process.exitcode == 0


def test_rebuild_lease_refuses_new_active_writer(tmp_path: Path) -> None:
    with RebuildLease(tmp_path):
        writer = ActiveWriterLease(tmp_path)
        with pytest.raises(RebuildLeaseUnavailableError):
            writer.acquire()


def test_generation_is_inactive_until_atomic_promotion(tmp_path: Path) -> None:
    _archive(tmp_path)
    original = (tmp_path / "index.db").resolve()
    original_inode = original.stat().st_ino
    store = IndexGenerationStore(tmp_path)
    generation = store.create(owner_id="operator", source_snapshot="snapshot-a")
    assert store.load(generation.generation_id).state == "inactive"
    assert (tmp_path / "index.db").resolve() == original

    promoted = store.promote(generation)
    assert promoted.state == "active"
    assert (tmp_path / "index.db").is_symlink()
    assert (tmp_path / "index.db").resolve() == Path(generation.index_path).resolve()
    retired = tuple(store.generations_root.glob("retired-*/index.db"))
    assert len(retired) == 1
    assert retired[0].stat().st_ino == original_inode


def test_stale_owner_cannot_checkpoint_or_promote(tmp_path: Path) -> None:
    _archive(tmp_path)
    store = IndexGenerationStore(tmp_path)
    generation = store.create(owner_id="operator", source_snapshot="snapshot-a")
    stale = replace(generation, owner_id="other")
    with pytest.raises(RuntimeError, match="owning inactive"):
        store.promote(stale)


def test_symlinked_configured_index_promotes_canonical_target(tmp_path: Path) -> None:
    configured = tmp_path / "configured"
    canonical = tmp_path / "canonical"
    configured.mkdir()
    canonical.mkdir()
    for tier in (ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.EMBEDDINGS, ArchiveTier.OPS):
        initialize_archive_database(canonical / f"{tier.value}.db", tier)
        (configured / f"{tier.value}.db").symlink_to(canonical / f"{tier.value}.db")
    initialize_archive_database(canonical / "index.db", ArchiveTier.INDEX)
    (configured / "index.db").symlink_to(canonical / "index.db")

    store = IndexGenerationStore(configured)
    generation = store.create(owner_id="operator", source_snapshot="snapshot-a")
    store.promote(generation)

    assert configured.joinpath("index.db").is_symlink()
    assert configured.joinpath("index.db").stat().st_ino == canonical.joinpath("index.db").stat().st_ino
    assert canonical.joinpath("index.db").resolve() == Path(generation.index_path).resolve()
    assert store.generations_root.parent == canonical

    second_store = IndexGenerationStore(configured)
    second = second_store.create(owner_id="operator-2", source_snapshot="snapshot-b")
    second_store.promote(second)
    assert second_store.active_pointer == canonical / "index.db"
    assert configured.joinpath("index.db").stat().st_ino == canonical.joinpath("index.db").stat().st_ino
    assert canonical.joinpath("index.db").resolve() == Path(second.index_path).resolve()
    assert len(tuple(store.generations_root.glob("retired-*/index.db"))) == 2
