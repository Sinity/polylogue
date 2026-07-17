from __future__ import annotations

import multiprocessing
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.storage.index_generation import (
    ActiveWriterLease,
    IndexGenerationStore,
    RebuildLease,
    RebuildLeaseUnavailableError,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
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


def test_promotion_removes_only_empty_active_sidecars(tmp_path: Path) -> None:
    _archive(tmp_path)
    store = IndexGenerationStore(tmp_path)
    generation = store.create(owner_id="operator", source_snapshot="snapshot-a")
    (tmp_path / "index.db-wal").touch()
    (tmp_path / "index.db-shm").touch()
    store.promote(generation)
    assert not (tmp_path / "index.db-wal").exists()
    assert not (tmp_path / "index.db-shm").exists()


def test_promotion_checkpoints_candidate_and_active_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _archive(tmp_path)
    store = IndexGenerationStore(tmp_path)
    generation = store.create(owner_id="operator", source_snapshot="snapshot-a")
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        "polylogue.storage.index_generation._checkpoint_truncate",
        lambda path, *, label: calls.append((path, label)),
    )

    store.promote(generation)

    assert calls == [(Path(generation.index_path).resolve(), "new index"), (tmp_path / "index.db", "active index")]


def test_recover_promotion_without_active_pointer_marks_inactive(tmp_path: Path) -> None:
    _archive(tmp_path)
    store = IndexGenerationStore(tmp_path)
    generation = store.create(owner_id="operator", source_snapshot="snapshot-a")
    store._write(replace(generation, state="promoting"))
    (tmp_path / "index.db").unlink()

    recovered = store.recover_promotion(generation.generation_id)

    assert recovered.state == "inactive"


def test_recover_promotion_after_pointer_swap_marks_active(tmp_path: Path) -> None:
    _archive(tmp_path)
    store = IndexGenerationStore(tmp_path)
    generation = store.create(owner_id="operator", source_snapshot="snapshot-a")
    store._write(replace(generation, state="promoting"))
    (tmp_path / "index.db").unlink()
    (tmp_path / "index.db").symlink_to(generation.index_path)

    recovered = store.recover_promotion(generation.generation_id)

    assert recovered.state == "active"


def test_archive_store_init_failure_releases_writer_lease(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("polylogue.paths.archive_root", lambda: tmp_path)
    monkeypatch.setattr(
        "polylogue.storage.sqlite.archive_tiers.archive.initialize_active_archive_root",
        lambda _root: (_ for _ in ()).throw(RuntimeError("bootstrap failed")),
    )
    with pytest.raises(RuntimeError, match="bootstrap failed"):
        ArchiveStore(tmp_path, read_only=False)

    with RebuildLease(tmp_path):
        pass


def test_failed_inactive_generation_is_discarded(tmp_path: Path) -> None:
    _archive(tmp_path)
    store = IndexGenerationStore(tmp_path)
    generation = store.create(owner_id="operator", source_snapshot="snapshot-a")

    assert store.discard_if_inactive(generation) is True
    assert not Path(generation.index_path).parent.exists()


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


def test_rebuild_transaction_persists_keyset_cursor_without_materializing_archive(tmp_path: Path) -> None:
    _archive(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        for raw_id, acquired_at_ms in (("raw-c", 30), ("raw-a", 10), ("raw-b", 10)):
            conn.execute(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, native_id, source_path, source_index, blob_hash,
                    blob_size, acquired_at_ms, validation_status
                ) VALUES (?, 'codex-session', ?, ?, 0, randomblob(32), 1, ?, 'passed')
                """,
                (raw_id, raw_id, f"/{raw_id}.jsonl", acquired_at_ms),
            )

    store = IndexGenerationStore(tmp_path)
    transaction = store.create_transaction(source_snapshot="source-v1", operation_id="resume-me")
    first_page = store.next_raw_page(transaction, limit=2)
    assert first_page == (("raw-a", 10), ("raw-b", 10))

    transaction = store.checkpoint_transaction(
        transaction,
        status="paused",
        last_acquired_at_ms=10,
        last_raw_id="raw-b",
        processed_raw_count=2,
    )
    assert transaction.cursor == "source:10:raw-b"
    assert store.load_transaction("resume-me") == transaction
    assert store.next_raw_page(transaction, limit=2) == (("raw-c", 30),)
