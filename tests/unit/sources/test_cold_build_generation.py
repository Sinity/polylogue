"""The daemon's cold build as an owned inactive generation (polylogue-b7dkb).

The active-generation cold-build shape (``test_live_cold_build_route.py``)
cannot take index deferral, ``journal_mode=MEMORY`` or ``locking_mode=
EXCLUSIVE``, because live readers hold the file it writes. These tests pin the
shape that can: the same dispatcher-fed ``LiveBatchProcessor.ingest_files``
pass, writing its index rows into a generation created by
``IndexGenerationStore`` and invisible to readers until ``promote()``.
"""

from __future__ import annotations

import asyncio
import errno
import json
import os
import sqlite3
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from polylogue.maintenance.candidate_capacity import (
    InsufficientCapacityError,
    read_capacity_receipts,
)
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchMetrics, LiveBatchProcessor
from polylogue.sources.live.cold_build import (
    ColdBuildGeneration,
    active_index_generation_is_empty,
    clear_cold_build_generation,
    register_cold_build_generation,
)
from polylogue.sources.source_layout import export_drop_layout
from polylogue.storage.archive_identity import GENERATIONS_DIRNAME, MAINTENANCE_STATE_DIRNAME
from polylogue.storage.index_generation import UnpublishedPromotionRecoveryError
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import run_archive_fixture_write
from tests.infra.live_batch import prepared_live_batch_processor


def _codex_session(native_id: str, text: str) -> bytes:
    return (
        f'{{"type":"session_meta","payload":{{"id":"{native_id}",'
        '"timestamp":"2026-06-02T00:00:00Z"}}\n'
        '{"type":"response_item","payload":{"type":"message","id":"message-0",'
        f'"role":"user","content":[{{"type":"input_text","text":"{text}"}}]}}}}\n'
    ).encode()


async def _ingest_paths(archive_root: Path, root: Path, paths: list[Path]) -> LiveBatchMetrics:
    async with prepared_live_batch_processor(
        archive_root,
        (WatchSource(name="codex", root=root, layout=export_drop_layout((".jsonl",))),),
        parser_fingerprint="test-parser",
    ) as processor:
        return await processor.ingest_files(paths, emit_event=False)


def _active_session_count(archive_root: Path) -> int:
    """Count sessions the way a reader resolving the active pointer sees them."""
    from polylogue.storage.archive_identity import resolve_active_index_path

    conn = sqlite3.connect(f"file:{resolve_active_index_path(archive_root)}?mode=ro", uri=True)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
    finally:
        conn.close()


def _candidate_index_names(generation: ColdBuildGeneration) -> set[str]:
    conn = sqlite3.connect(f"file:{generation.generation.index_path}?mode=ro", uri=True)
    try:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
    finally:
        conn.close()


@pytest.fixture
def cold_build(tmp_path: Path) -> Iterator[ColdBuildGeneration]:
    assert active_index_generation_is_empty(tmp_path) is True
    generation = ColdBuildGeneration.begin(
        tmp_path,
        reason="test",
        observed=ColdBuildGeneration.observe_source_baseline((WatchSource("fixture", tmp_path / "absent-source"),)),
    )
    register_cold_build_generation(generation)
    try:
        yield generation
    finally:
        clear_cold_build_generation()


def _ingest(archive_root: Path, root: Path, name: str, native_id: str) -> None:
    (root / name).write_bytes(_codex_session(native_id, native_id))
    metrics = asyncio.run(_ingest_paths(archive_root, root, [root / name]))
    assert metrics.succeeded_file_count == 1, metrics


@pytest.mark.parametrize("after_unlink", [False, True])
def test_receipt_tail_reconciles_active_generation_without_repromotion(
    tmp_path: Path, cold_build: ColdBuildGeneration, monkeypatch: pytest.MonkeyPatch, after_unlink: bool
) -> None:
    from polylogue.sources.live import production_baseline
    from polylogue.storage.index_generation import IndexGenerationStore

    root = tmp_path / "sessions"
    root.mkdir()
    _ingest(tmp_path, root, "one.jsonl", "receipt-tail")
    original_clear = production_baseline.clear_pending_production_baseline
    original_promote = IndexGenerationStore.promote
    promotions = 0
    clears = 0

    def count_promote(self: IndexGenerationStore, generation: Any, prepared: Any = None) -> Any:
        nonlocal promotions
        promotions += 1
        return original_promote(self, generation, prepared)

    def fail_clear_once(archive_root: Path, baseline: Any, *, allow_missing: bool = False) -> None:
        nonlocal clears
        clears += 1
        if clears == 1:
            if after_unlink:
                original_clear(archive_root, baseline)
            raise OSError(errno.EBUSY, "receipt cleanup interrupted")
        original_clear(archive_root, baseline, allow_missing=allow_missing)

    monkeypatch.setattr(IndexGenerationStore, "promote", count_promote)
    monkeypatch.setattr(production_baseline, "clear_pending_production_baseline", fail_clear_once)
    with pytest.raises(OSError, match="receipt cleanup interrupted"):
        cold_build.promote()
    assert cold_build.settled
    assert not cold_build.publication_complete
    assert _active_session_count(tmp_path) == 1
    assert cold_build.promote().generation_id == cold_build.generation_id
    assert cold_build.publication_complete
    assert promotions == 1
    assert clears == 2


@pytest.mark.parametrize("fault", ("busy", "wrapped_io"))
def test_pointer_swapped_before_metadata_failure_recovers_once(
    tmp_path: Path, cold_build: ColdBuildGeneration, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    from polylogue.storage.index_generation import IndexGenerationStore

    root = tmp_path / "sessions"
    root.mkdir()
    _ingest(tmp_path, root, "one.jsonl", "pointer-tail")
    original_write = IndexGenerationStore._write
    original_promote = IndexGenerationStore.promote
    writes_failed = 0
    promotions = 0

    def fail_active_write_once(self: IndexGenerationStore, generation: Any) -> None:
        nonlocal writes_failed
        if generation.generation_id == cold_build.generation_id and generation.state == "active" and not writes_failed:
            writes_failed += 1
            if fault == "wrapped_io":
                raise RuntimeError("cannot create active metadata temporary") from OSError(
                    errno.EIO, "active metadata I/O unavailable"
                )
            raise OSError(errno.EBUSY, "active metadata temporarily busy")
        original_write(self, generation)

    def count_promote(self: IndexGenerationStore, generation: Any, prepared: Any = None) -> Any:
        nonlocal promotions
        promotions += 1
        return original_promote(self, generation, prepared)

    monkeypatch.setattr(IndexGenerationStore, "_write", fail_active_write_once)
    monkeypatch.setattr(IndexGenerationStore, "promote", count_promote)
    assert cold_build.promote().state == "active"
    assert cold_build.publication_complete
    assert _active_session_count(tmp_path) == 1
    assert writes_failed == 1
    assert promotions == 1


def test_pointer_parent_fsync_must_succeed_before_promotion_tail(
    tmp_path: Path, cold_build: ColdBuildGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.sources.live.production_baseline import load_pending_production_baseline
    from polylogue.storage import index_generation

    root = tmp_path / "sessions"
    root.mkdir()
    _ingest(tmp_path, root, "one.jsonl", "pointer-fsync")
    original_fsync = index_generation._fsync_directory
    failures = 0

    def fail_pointer_parent_twice(path: Path) -> None:
        nonlocal failures
        if (
            path == tmp_path
            and (tmp_path / "index.db").resolve() == Path(cold_build.generation.index_path).resolve()
            and failures < 2
        ):
            failures += 1
            raise OSError(errno.ENOSPC, "pointer directory fsync unavailable")
        original_fsync(path)

    with monkeypatch.context() as patcher:
        patcher.setattr(index_generation, "_fsync_directory", fail_pointer_parent_twice)
        with pytest.raises(OSError) as failure:
            cold_build.promote()
    assert failure.value.errno == errno.ENOSPC
    assert failures == 2
    assert cold_build.promoted
    assert cold_build._store.load(cold_build.generation_id).state == "promoting"
    assert load_pending_production_baseline(tmp_path) is not None
    assert cold_build.promote().state == "active"
    assert cold_build.publication_complete
    assert load_pending_production_baseline(tmp_path) is None


def test_restart_completes_pointer_swapped_cold_promotion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.sources.live.production_baseline import load_pending_production_baseline
    from polylogue.storage.index_generation import IndexGenerationStore

    archive = _fresh_archive_root(tmp_path)
    source_root = tmp_path / "source"
    source_root.mkdir()
    member = source_root / "one.jsonl"
    member.write_bytes(_codex_session("one", "interrupted"))
    source = WatchSource("codex", source_root, layout=export_drop_layout((".jsonl",)), required=True)
    generation = ColdBuildGeneration.begin(
        archive, reason="first", observed=ColdBuildGeneration.observe_source_baseline((source,))
    )
    register_cold_build_generation(generation)
    try:
        assert asyncio.run(_ingest_paths(archive, source_root, [member])).succeeded_file_count == 1
        with generation.open_writer() as candidate:
            candidate.run_generation_readiness_pass()
        generation.source_baseline.verify(archive / "source.db")
        original_write = IndexGenerationStore._write

        def interrupt_active_metadata(self: IndexGenerationStore, row: Any) -> None:
            if row.generation_id == generation.generation_id and row.state == "active":
                raise OSError(errno.ENOSPC, "interrupted after pointer swap")
            original_write(self, row)

        with monkeypatch.context() as patcher:
            patcher.setattr(IndexGenerationStore, "_write", interrupt_active_metadata)
            with pytest.raises(OSError, match="interrupted after pointer swap"):
                generation._store.promote(generation.generation)
        assert generation._store.load(generation.generation_id).state == "promoting"
        assert (archive / "index.db").resolve() == Path(generation.generation.index_path).resolve()
        assert load_pending_production_baseline(archive) is not None
    finally:
        clear_cold_build_generation()
        generation._release_ops_checkpoint_holder()

    _reconcile_on_daemon_writer(archive)
    assert generation._store.load(generation.generation_id).state == "active"
    assert load_pending_production_baseline(archive) is None
    assert _active_session_count(archive) == 1
    _reconcile_on_daemon_writer(archive)


def test_restart_clears_matching_receipt_after_active_metadata(
    tmp_path: Path, cold_build: ColdBuildGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.sources.live import production_baseline

    root = tmp_path / "sessions"
    root.mkdir()
    _ingest(tmp_path, root, "one.jsonl", "receipt-tail")

    def interrupt_receipt_tail(*_args: Any, **_kwargs: Any) -> None:
        raise OSError(errno.ENOSPC, "receipt tail interrupted")

    with monkeypatch.context() as patcher:
        patcher.setattr(production_baseline, "clear_pending_production_baseline", interrupt_receipt_tail)
        with pytest.raises(OSError, match="receipt tail interrupted"):
            cold_build.promote()
    assert cold_build._store.load(cold_build.generation_id).state == "active"
    assert production_baseline.load_pending_production_baseline(tmp_path) is not None
    clear_cold_build_generation()
    _reconcile_on_daemon_writer(tmp_path)
    assert production_baseline.load_pending_production_baseline(tmp_path) is None


def test_pre_swap_storage_fault_restores_same_inactive_candidate(
    tmp_path: Path, cold_build: ColdBuildGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.index_generation import IndexGenerationStore

    pointer = tmp_path / "index.db"
    prior_pointer = pointer.lstat()
    original_symlink_to = Path.symlink_to

    def fail_candidate_pointer(path: Path, target: os.PathLike[str] | str, target_is_directory: bool = False) -> None:
        if path.name.startswith(".index.db.promote-"):
            raise OSError(errno.ENOSPC, "pointer filesystem full")
        original_symlink_to(path, target, target_is_directory=target_is_directory)

    with monkeypatch.context() as patcher:
        patcher.setattr(Path, "symlink_to", fail_candidate_pointer)
        with pytest.raises(OSError) as failure:
            cold_build.promote()
    assert failure.value.errno == errno.ENOSPC
    current_pointer = pointer.lstat()
    assert (current_pointer.st_dev, current_pointer.st_ino) == (prior_pointer.st_dev, prior_pointer.st_ino)
    assert IndexGenerationStore.for_archive_root(tmp_path).load(cold_build.generation_id).state == "inactive"
    assert not cold_build.settled
    assert cold_build.promote().generation_id == cold_build.generation_id


def test_repeated_pre_swap_fault_restores_predecessor_sidecars_and_marker(
    tmp_path: Path, cold_build: ColdBuildGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage import index_generation

    pointer = tmp_path / "index.db"
    prior_identity = (pointer.lstat().st_dev, pointer.lstat().st_ino)
    sidecars = tuple(pointer.with_name(pointer.name + suffix) for suffix in ("-wal", "-shm"))
    for sidecar in sidecars:
        sidecar.touch()
    sidecar_identities = tuple((path.stat().st_dev, path.stat().st_ino) for path in sidecars)
    original_replace = os.replace
    original_checkpoint = index_generation._checkpoint_truncate
    failures = 0

    def keep_zero_sidecars(path: Path, *, label: str, archive_root: Path) -> None:
        if label != "active index":
            original_checkpoint(path, label=label, archive_root=archive_root)

    def fail_pre_swap(source: os.PathLike[str] | str, target: os.PathLike[str] | str) -> None:
        nonlocal failures
        partial_sidecar_move = (
            Path(source).name == "index.db-shm" and Path(target).parent.name.startswith("retired-") and failures == 0
        )
        pointer_swap = Path(source).name.startswith(".index.db.promote-") and Path(target) == pointer
        if partial_sidecar_move or pointer_swap:
            failures += 1
            raise OSError(errno.EAGAIN, "pre-swap filesystem temporarily busy")
        original_replace(source, target)

    with monkeypatch.context() as patcher:
        patcher.setattr(os, "replace", fail_pre_swap)
        patcher.setattr(index_generation, "_checkpoint_truncate", keep_zero_sidecars)
        for _ in range(3):
            with pytest.raises(OSError) as failure:
                cold_build.promote()
            assert failure.value.errno == errno.EAGAIN
            assert (pointer.lstat().st_dev, pointer.lstat().st_ino) == prior_identity
            assert tuple((path.stat().st_dev, path.stat().st_ino) for path in sidecars) == sidecar_identities
            assert cold_build._store.load(cold_build.generation_id).state == "inactive"
            assert not tuple(cold_build._store.generations_root.glob("retired-*"))
    assert failures == 3
    assert cold_build.promote().state == "active"


def test_interrupted_pre_swap_with_moved_sidecars_restores_them_on_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage import index_generation

    archive = _fresh_archive_root(tmp_path)
    sources = (WatchSource("fixture", tmp_path / "absent-source"),)
    abandoned = ColdBuildGeneration.begin(
        archive, reason="first", observed=ColdBuildGeneration.observe_source_baseline(sources)
    )
    pointer = archive / "index.db"
    prior_identity = (pointer.lstat().st_dev, pointer.lstat().st_ino)
    sidecars = tuple(pointer.with_name(pointer.name + suffix) for suffix in ("-wal", "-shm"))
    for sidecar in sidecars:
        sidecar.touch()
    sidecar_identities = tuple((path.stat().st_dev, path.stat().st_ino) for path in sidecars)
    original_symlink_to = Path.symlink_to
    original_checkpoint = index_generation._checkpoint_truncate

    def keep_zero_sidecars(path: Path, *, label: str, archive_root: Path) -> None:
        if label != "active index":
            original_checkpoint(path, label=label, archive_root=archive_root)

    def interrupt_before_swap(path: Path, target: os.PathLike[str] | str, target_is_directory: bool = False) -> None:
        if path.name.startswith(".index.db.promote-"):
            raise KeyboardInterrupt("process stopped before pointer swap")
        original_symlink_to(path, target, target_is_directory=target_is_directory)

    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(Path, "symlink_to", interrupt_before_swap)
            patcher.setattr(index_generation, "_checkpoint_truncate", keep_zero_sidecars)
            with pytest.raises(KeyboardInterrupt):
                abandoned.promote()
        assert abandoned._store.load(abandoned.generation_id).state == "promoting"
        assert (pointer.lstat().st_dev, pointer.lstat().st_ino) == prior_identity
        assert all(not sidecar.exists() for sidecar in sidecars)
        assert len(tuple(abandoned._store.generations_root.glob("retired-*"))) == 1
    finally:
        abandoned._release_ops_checkpoint_holder()

    replacement = ColdBuildGeneration.begin(
        archive, reason="restart", observed=ColdBuildGeneration.observe_source_baseline(sources)
    )
    try:
        assert not abandoned.generation_root.exists()
        assert tuple((path.stat().st_dev, path.stat().st_ino) for path in sidecars) == sidecar_identities
        assert not tuple(replacement._store.generations_root.glob("retired-*"))
    finally:
        replacement.discard()


def test_pre_swap_sidecar_restore_fault_blocks_candidate_until_retry(
    tmp_path: Path, cold_build: ColdBuildGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage import index_generation

    pointer = tmp_path / "index.db"
    sidecar = tmp_path / "index.db-wal"
    sidecar.touch()
    original_inode = sidecar.stat().st_ino
    original_checkpoint = index_generation._checkpoint_truncate
    original_replace = os.replace
    original_link = os.link

    def keep_zero_sidecar(path: Path, *, label: str, archive_root: Path) -> None:
        if label != "active index":
            original_checkpoint(path, label=label, archive_root=archive_root)

    def fail_pointer_swap(source: os.PathLike[str] | str, target: os.PathLike[str] | str) -> None:
        if Path(source).name.startswith(".index.db.promote-") and Path(target) == pointer:
            raise OSError(errno.EAGAIN, "pointer swap busy")
        original_replace(source, target)

    def fail_sidecar_restore(
        source: os.PathLike[str] | str, target: os.PathLike[str] | str, *, follow_symlinks: bool = True
    ) -> None:
        if Path(source).parent.name.startswith("retired-") and Path(target) == sidecar:
            raise OSError(errno.EACCES, "sidecar restore unavailable")
        original_link(source, target, follow_symlinks=follow_symlinks)

    with monkeypatch.context() as patcher:
        patcher.setattr(index_generation, "_checkpoint_truncate", keep_zero_sidecar)
        patcher.setattr(os, "replace", fail_pointer_swap)
        patcher.setattr(os, "link", fail_sidecar_restore)
        with pytest.raises(OSError) as failure:
            cold_build.promote()
    assert failure.value.errno == errno.EAGAIN
    assert cold_build._store.load(cold_build.generation_id).state == "promoting"
    assert cold_build._store.unpublished_rollback_pending(cold_build.generation_id)
    assert not sidecar.exists()
    with cold_build.open_writer():
        pass
    assert cold_build._store.load(cold_build.generation_id).state == "inactive"
    assert sidecar.stat().st_ino == original_inode
    assert not tuple(cold_build._store.generations_root.glob("retired-*"))


def test_pre_swap_fault_survives_exhausted_rollback_and_retries_same_candidate(
    tmp_path: Path, cold_build: ColdBuildGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer = tmp_path / "index.db"
    prior_pointer = pointer.lstat()
    original_symlink_to = Path.symlink_to
    original_replace = os.replace

    def fail_candidate_pointer(path: Path, target: os.PathLike[str] | str, target_is_directory: bool = False) -> None:
        if path.name.startswith(".index.db.promote-"):
            raise OSError(errno.ENOSPC, "pointer filesystem full")
        original_symlink_to(path, target, target_is_directory=target_is_directory)

    def fail_rollback_replace(source: os.PathLike[str] | str, target: os.PathLike[str] | str) -> None:
        if Path(source).name == "generation.rollback.json":
            raise OSError(errno.ENOSPC, "rollback rename unavailable")
        original_replace(source, target)

    with monkeypatch.context() as patcher:
        patcher.setattr(Path, "symlink_to", fail_candidate_pointer)
        patcher.setattr(os, "replace", fail_rollback_replace)
        with pytest.raises(OSError) as failure:
            cold_build.promote()
    assert failure.value.errno == errno.ENOSPC
    assert cold_build._store.load(cold_build.generation_id).state == "promoting"
    assert cold_build._store.unpublished_rollback_pending(cold_build.generation_id)
    current_pointer = pointer.lstat()
    assert (current_pointer.st_dev, current_pointer.st_ino) == (prior_pointer.st_dev, prior_pointer.st_ino)
    with cold_build.open_writer():
        pass
    assert cold_build._store.load(cold_build.generation_id).state == "inactive"
    assert cold_build.promote().generation_id == cold_build.generation_id


def test_interrupted_pre_swap_promotion_reclaims_only_with_prior_pointer_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _fresh_archive_root(tmp_path)
    sources = (WatchSource("fixture", tmp_path / "absent-source"),)
    abandoned = ColdBuildGeneration.begin(
        archive, reason="first", observed=ColdBuildGeneration.observe_source_baseline(sources)
    )
    prior_pointer = (archive / "index.db").lstat()
    original_symlink_to = Path.symlink_to
    original_replace = os.replace

    def fail_candidate_pointer(path: Path, target: os.PathLike[str] | str, target_is_directory: bool = False) -> None:
        if path.name.startswith(".index.db.promote-"):
            raise OSError(errno.ENOSPC, "pointer filesystem full")
        original_symlink_to(path, target, target_is_directory=target_is_directory)

    def fail_rollback_replace(source: os.PathLike[str] | str, target: os.PathLike[str] | str) -> None:
        if Path(source).name == "generation.rollback.json":
            raise OSError(errno.ENOSPC, "rollback rename unavailable")
        original_replace(source, target)

    with monkeypatch.context() as patcher:
        patcher.setattr(Path, "symlink_to", fail_candidate_pointer)
        patcher.setattr(os, "replace", fail_rollback_replace)
        with pytest.raises(OSError, match="pointer filesystem full"):
            abandoned.promote()
    assert abandoned._store.load(abandoned.generation_id).state == "promoting"
    pointer = (archive / "index.db").lstat()
    assert (pointer.st_dev, pointer.st_ino) == (prior_pointer.st_dev, prior_pointer.st_ino)
    proof = abandoned.generation_root / "generation.rollback-pointer.json"
    proof_bytes = proof.read_bytes()
    proof.unlink()
    with pytest.raises(UnpublishedPromotionRecoveryError, match="proof is unavailable"):
        ColdBuildGeneration.begin(
            archive, reason="restart", observed=ColdBuildGeneration.observe_source_baseline(sources)
        )
    assert abandoned.generation_root.exists()
    proof.write_bytes(proof_bytes)

    replacement = ColdBuildGeneration.begin(
        archive, reason="restart", observed=ColdBuildGeneration.observe_source_baseline(sources)
    )
    try:
        assert replacement.generation_id != abandoned.generation_id
        assert not abandoned.generation_root.exists()
    finally:
        replacement.discard()


def test_rollback_preallocation_refusal_keeps_candidate_inactive(
    tmp_path: Path, cold_build: ColdBuildGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage import index_generation

    original_write = index_generation._atomic_json_write

    def refuse_rollback(path: Path, payload: dict[str, object], *, label: str) -> None:
        if path.name == "generation.rollback.json":
            raise OSError(errno.ENOSPC, "cannot reserve rollback metadata")
        original_write(path, payload, label=label)

    with monkeypatch.context() as patcher:
        patcher.setattr(index_generation, "_atomic_json_write", refuse_rollback)
        with pytest.raises(OSError) as failure:
            cold_build.promote()
    assert failure.value.errno == errno.ENOSPC
    assert cold_build._store.load(cold_build.generation_id).state == "inactive"
    assert not (cold_build.generation_root / "generation.rollback-pointer.json").exists()
    assert cold_build.promote().generation_id == cold_build.generation_id


def test_blocked_settlement_revision_tracks_receipt_and_source_evidence(
    tmp_path: Path, cold_build: ColdBuildGeneration
) -> None:
    from polylogue.storage.archive_identity import MAINTENANCE_STATE_DIRNAME

    initial = cold_build.settlement_evidence_revision()
    receipt = tmp_path / MAINTENANCE_STATE_DIRNAME / "production-source-baseline" / "pending.json"
    old_mtime = receipt.stat().st_mtime_ns
    os.utime(receipt, ns=(old_mtime, old_mtime + 1_000_000))
    changed = cold_build.settlement_evidence_revision()
    assert changed != initial
    source_db = tmp_path / "source.db"
    old_mtime = source_db.stat().st_mtime_ns
    os.utime(source_db, ns=(old_mtime, old_mtime + 1_000_000))
    source_changed = cold_build.settlement_evidence_revision()
    assert source_changed != changed
    binding = cold_build.generation_root / "source-baseline.json"
    mode = stat.S_IMODE(binding.stat().st_mode)
    external_before = cold_build.settlement_external_revision()
    try:
        binding.chmod(mode ^ stat.S_IRUSR)
        assert cold_build.settlement_evidence_revision() != source_changed
        assert cold_build.settlement_external_revision() != external_before
    finally:
        binding.chmod(mode)
    repaired_root = tmp_path / "repaired-source"
    source = WatchSource("repaired", repaired_root, required=True)
    missing = cold_build.settlement_evidence_revision((source,))
    repaired_root.mkdir()
    assert cold_build.settlement_evidence_revision((source,)) != missing


def test_blocked_settlement_revision_tracks_receipt_parent_permission_repair(
    tmp_path: Path, cold_build: ColdBuildGeneration
) -> None:
    directories = (
        tmp_path,
        tmp_path / MAINTENANCE_STATE_DIRNAME / "production-source-baseline",
        cold_build.generation_root,
    )
    for directory in directories:
        mode = stat.S_IMODE(directory.stat().st_mode)
        unavailable = cold_build.settlement_evidence_revision()
        try:
            directory.chmod(mode ^ stat.S_IWUSR)
            assert cold_build.settlement_evidence_revision() != unavailable
            unavailable = cold_build.settlement_evidence_revision()
        finally:
            directory.chmod(mode)
        assert cold_build.settlement_evidence_revision() != unavailable


def test_settlement_owned_directory_writes_do_not_change_external_revision(
    tmp_path: Path, cold_build: ColdBuildGeneration
) -> None:
    directories = (
        tmp_path / MAINTENANCE_STATE_DIRNAME / "production-source-baseline",
        cold_build.generation_root,
    )
    unchanged = cold_build.settlement_external_revision()
    for directory in directories:
        marker = directory / "owned-write.tmp"
        marker.write_bytes(b"settlement")
        assert cold_build.settlement_external_revision() == unchanged
        marker.unlink()
        assert cold_build.settlement_external_revision() == unchanged


def test_blocked_settlement_revision_survives_unavailable_evidence(
    tmp_path: Path, cold_build: ColdBuildGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.archive_identity import MAINTENANCE_STATE_DIRNAME

    receipt = tmp_path / MAINTENANCE_STATE_DIRNAME / "production-source-baseline" / "pending.json"
    original_stat = Path.stat
    error: OSError | None = PermissionError(errno.EACCES, "receipt unavailable")

    class OtherPermissionError(PermissionError):
        pass

    def stat_with_fault(path: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        if path == receipt and error is not None:
            raise error
        return original_stat(path, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(Path, "stat", stat_with_fault)
        unavailable = cold_build.settlement_evidence_revision()
        assert cold_build.settlement_evidence_revision() == unavailable
        error = OtherPermissionError(errno.EACCES, "receipt unavailable")
        assert cold_build.settlement_evidence_revision() != unavailable
        error = OSError(errno.EIO, "receipt unavailable")
        assert cold_build.settlement_evidence_revision() != unavailable
        error = None
        assert cold_build.settlement_evidence_revision() != unavailable

    cold_build.settlement_reason = "capacity_unavailable"
    original_statvfs = os.statvfs
    capacity_error: OSError | None = PermissionError(errno.EACCES, "capacity unavailable")

    def statvfs_with_fault(path: os.PathLike[str] | str) -> os.statvfs_result:
        if capacity_error is not None:
            raise capacity_error
        return original_statvfs(path)

    with monkeypatch.context() as patcher:
        patcher.setattr(os, "statvfs", statvfs_with_fault)
        unavailable = cold_build.settlement_evidence_revision()
        assert cold_build.settlement_evidence_revision() == unavailable
        capacity_error = OSError(errno.EIO, "capacity unavailable")
        assert cold_build.settlement_evidence_revision() != unavailable
        capacity_error = None
        assert cold_build.settlement_evidence_revision() != unavailable


def test_capacity_blocked_revision_tracks_candidate_filesystem_space(
    cold_build: ColdBuildGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    cold_build.settlement_reason = "capacity_unavailable"
    original_statvfs = os.statvfs
    candidate_space = 1000

    def split_statvfs(path: os.PathLike[str] | str) -> os.statvfs_result:
        statistics = list(original_statvfs(path))
        if Path(path) == cold_build.generation_root:
            statistics[4] = candidate_space
        return os.statvfs_result(statistics)

    monkeypatch.setattr(os, "statvfs", split_statvfs)
    blocked = cold_build.settlement_evidence_revision()
    candidate_space += 1000
    assert cold_build.settlement_evidence_revision() != blocked


def test_the_live_pass_writes_into_the_owned_generation_not_the_active_one(
    tmp_path: Path, cold_build: ColdBuildGeneration
) -> None:
    """Readers keep the previous active generation for the whole build.

    Anti-vacuity: making ``_open_archive_for_live_write`` ignore the
    registered generation (returning ``open_active_cold_build``) writes the
    rows into the active index and makes the during-build assertion red;
    deleting the ``promote()`` pointer swap makes the after-promotion one red.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    _ingest(tmp_path, root, "one.jsonl", "owned-one")

    # During the build the candidate holds the row and no reader can see it.
    assert cold_build.session_count() == 1
    assert _active_session_count(tmp_path) == 0
    with ArchiveStore.open_existing(tmp_path, read_only=True) as reader:
        assert reader._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0

    promoted = cold_build.promote()
    assert promoted.state == "active"
    assert _active_session_count(tmp_path) == 1
    with ArchiveStore.open_existing(tmp_path, read_only=True) as reader:
        assert reader._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


def test_the_ops_checkpoint_holder_spans_the_page_cursor_writes(
    tmp_path: Path, cold_build: ColdBuildGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The page's cursor publication still runs under the checkpoint holder.

    The archive pass closes inside the page, before its cursor, convergence
    and attempt writes; each of those publications closes its own ``ops.db``
    connection, which checkpoints unless the holder is still open.

    Anti-vacuity: release the holder on the archive pass's close again (drop
    the ``_ops_page_depth`` guard in ``open_writer``'s ``close_page`` or the
    ``begin_ops_page`` call in ``ingest_files``) and the holder is gone when
    the cursors are written.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    held_at_cursor_write: list[bool] = []
    record_full_cursors = LiveBatchProcessor._record_full_cursors

    def observed(self: LiveBatchProcessor, *args: Any, **kwargs: Any) -> Any:
        held_at_cursor_write.append(cold_build._ops_checkpoint_holder is not None)
        return record_full_cursors(self, *args, **kwargs)

    monkeypatch.setattr(LiveBatchProcessor, "_record_full_cursors", observed)
    _ingest(tmp_path, root, "one.jsonl", "page-holder")

    assert held_at_cursor_write == [True]
    # The page end still releases it: the window never spans pages.
    assert cold_build._ops_checkpoint_holder is None
    assert cold_build._ops_page_depth == 0


def test_a_file_intake_excludes_does_not_block_promotion(tmp_path: Path) -> None:
    """The baseline records intake's own exclusion, so the build promotes (polylogue-se08w).

    The source root holds one Codex session and one Codex JSONL that intake
    excludes before acquisition (it carries no conversational record). No raw
    row is ever written for it, so the baseline must not require one.

    Anti-vacuity: removing the ``classify_pre_acquisition`` step from
    ``capture_production_source_baseline`` records the sidecar as accepted,
    and ``promote()`` raises ``ProductionBaselineError`` naming one unretained
    revision.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    session = root / "one.jsonl"
    session.write_bytes(_codex_session("kept-session", "kept"))
    sidecar = root / "no-session.jsonl"
    sidecar.write_bytes(b'{"x":1}\n')
    generation = ColdBuildGeneration.begin(
        tmp_path,
        reason="test",
        observed=ColdBuildGeneration.observe_source_baseline(
            (WatchSource("codex", root, layout=export_drop_layout((".jsonl",))),)
        ),
    )
    register_cold_build_generation(generation)
    try:
        metrics = asyncio.run(_ingest_paths(tmp_path, root, [session, sidecar]))
        # Live acquisition is source-only: it retains both files' bytes and
        # leaves the sidecar's classification to retained replay, while the
        # baseline still records intake's own exclusion for it.
        assert metrics.failed_file_count == 0, metrics
        assert metrics.new_sessions == (("codex", "codex-session:kept-session"),), metrics
        decisions = {row.path: row for row in generation.source_baseline.decisions}
        assert decisions[str(sidecar)].disposition == "excluded"
        assert decisions[str(sidecar)].reason == "intake_excluded:declared artifact rule: not parsed as a session"
        assert decisions[str(session)].disposition == "accepted"

        assert generation.promote().state == "active"
        assert _active_session_count(tmp_path) == 1
    finally:
        clear_cold_build_generation()


def test_the_build_spans_passes_and_keeps_the_deferred_indexes_dropped(
    tmp_path: Path, cold_build: ColdBuildGeneration
) -> None:
    """A cold build is many dispatcher pages against one generation.

    The second page re-opens a generation that now has rows; the deferral must
    survive that rather than being refused for non-emptiness or silently
    recreated by the runtime-index ensure.

    Anti-vacuity: restoring the ``SELECT 1 FROM sessions`` refusal in the
    deferral branch makes the second ingest raise; letting
    ``ensure_runtime_indexes_sync`` run on a deferring open makes the
    ``idx_messages_role`` assertion red.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    _ingest(tmp_path, root, "one.jsonl", "owned-first")
    _ingest(tmp_path, root, "two.jsonl", "owned-second")

    assert cold_build.session_count() == 2
    assert "idx_messages_role" not in _candidate_index_names(cold_build)

    cold_build.promote()
    assert _active_session_count(tmp_path) == 2


def test_the_readiness_pass_restores_the_reader_shape_before_promotion(
    tmp_path: Path, cold_build: ColdBuildGeneration
) -> None:
    """One CREATE INDEX pass and one FTS build, then the generation is ordinary.

    A read-only open projects ``sqlite_master`` including indexes, so a
    promoted generation still missing its deferred indexes refuses with a
    schema mismatch -- which is exactly why the deferral needs the owned
    generation in the first place.

    Anti-vacuity: deleting ``restore_deferred_secondary_indexes_sync`` from
    ``run_generation_readiness_pass`` makes the read-only open raise; deleting
    the FTS rebuild makes the search assertion red.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    _ingest(tmp_path, root, "one.jsonl", "owned-ready")

    cold_build.promote()

    assert "idx_messages_role" in _candidate_index_names(cold_build)
    with ArchiveStore.open_existing(tmp_path, read_only=True) as reader:
        hits = reader._conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'owned'").fetchone()
        assert hits[0] >= 1


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_late_readiness_failure_preserves_restored_layout_for_next_canonical_preparation(
    tmp_path: Path, cold_build: ColdBuildGeneration, monkeypatch: pytest.MonkeyPatch, failure: type[BaseException]
) -> None:
    """A retry may prepare against restored indexes; its writer must retain them."""
    from polylogue.storage.fts import fts_lifecycle

    root = tmp_path / "sessions"
    root.mkdir()
    _ingest(tmp_path, root, "one.jsonl", "owned-first")
    original = fts_lifecycle.rebuild_fts_index_sync

    def fail_after_read_models(_conn: sqlite3.Connection) -> None:
        raise failure("interrupted before terminal FTS publication")

    with monkeypatch.context() as control:
        control.setattr(fts_lifecycle, "rebuild_fts_index_sync", fail_after_read_models)
        with pytest.raises(failure):
            cold_build.prepare_promotion_candidate()
    assert not cold_build._promotion_candidate_ready
    assert _active_session_count(tmp_path) == 0
    assert "idx_messages_role" in _candidate_index_names(cold_build)
    # This is actual acquisition, Source preparation and sealed Index replay,
    # after the failed pass changed the candidate's reader-index incarnation.
    _ingest(tmp_path, root, "two.jsonl", "owned-second")
    assert "idx_messages_role" in _candidate_index_names(cold_build)
    assert fts_lifecycle.rebuild_fts_index_sync is original
    cold_build.promote()
    assert _active_session_count(tmp_path) == 2
    with ArchiveStore.open_existing(tmp_path, read_only=True) as reader:
        assert (
            reader._conn.execute("SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'owned'").fetchone()[0]
            == 2
        )


def test_secondary_layout_preservation_requires_owned_writer_and_excludes_deferral(
    tmp_path: Path, cold_build: ColdBuildGeneration
) -> None:
    with pytest.raises(ValueError, match="owned inactive writable"):
        ArchiveStore(tmp_path, preserve_secondary_index_layout=True)
    with pytest.raises(ValueError, match="owned inactive writable"):
        ArchiveStore(tmp_path, read_only=True, preserve_secondary_index_layout=True)
    with pytest.raises(ValueError, match="deferred and preserved"):
        ArchiveStore.open_owned_inactive_generation(
            cold_build.generation_root,
            generation_id=cold_build.generation_id,
            owner_id=cold_build.generation.owner_id,
            defer_secondary_indexes=True,
            preserve_secondary_index_layout=True,
        )


def test_a_never_promoted_generation_is_discarded_and_leaves_readers_alone(
    tmp_path: Path, cold_build: ColdBuildGeneration
) -> None:
    """Crash semantics: the build simply never becomes visible.

    Anti-vacuity: making ``discard`` promote instead, or having the ingest
    pass write through to the active generation, makes the active count
    assertion red.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    _ingest(tmp_path, root, "one.jsonl", "owned-doomed")
    generation_root = cold_build.generation_root

    assert cold_build.discard() is True
    assert not generation_root.exists()
    assert _active_session_count(tmp_path) == 0


def test_acquisition_stays_on_the_real_durable_tiers(tmp_path: Path, cold_build: ColdBuildGeneration) -> None:
    """The candidate directory carries read-through symlinks, not a second archive.

    The cold build acquires and materializes in one pass, so the raw row and
    its blob must land in the archive's own ``source.db`` -- otherwise a
    discarded generation would take the durable evidence with it.

    Anti-vacuity: dropping ``durable_writer`` (so the store keeps the inactive
    candidate's refusing blob publisher) makes the ingest fail outright; a
    generation created before the durable tiers exist would grow its own
    ``source.db`` and make the symlink assertion red.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    _ingest(tmp_path, root, "one.jsonl", "owned-durable")

    assert (cold_build.generation_root / "source.db").is_symlink()
    conn = sqlite3.connect(f"file:{tmp_path / 'source.db'}?mode=ro", uri=True)
    try:
        assert int(conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0]) >= 1
    finally:
        conn.close()


def _free_space(monkeypatch: pytest.MonkeyPatch, available_bytes: int) -> None:
    """Pin what the filesystem reports as usable for the generations root."""
    monkeypatch.setattr(
        "polylogue.maintenance.candidate_capacity._available_bytes",
        lambda _path: available_bytes,
    )


def test_the_cold_build_refuses_before_it_allocates_a_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production build route is the one that has to refuse on free space.

    The guard used to sit only on the manual rebuild lifecycle, which has no
    production caller, so before this lift a real daemon cold build -- the
    route that writes a second whole index beside the one serving reads --
    allocated with no headroom check at all.

    Anti-vacuity: deleting the ``require_candidate_capacity`` call from
    ``ColdBuildGeneration.begin`` makes this red -- ``begin`` returns a live
    generation instead of raising, and a ``gen-*`` directory appears.
    """
    assert active_index_generation_is_empty(tmp_path) is True
    _free_space(monkeypatch, 0)

    with pytest.raises(InsufficientCapacityError) as refusal:
        ColdBuildGeneration.begin(
            tmp_path,
            reason="test",
            observed=ColdBuildGeneration.observe_source_baseline((WatchSource("fixture", tmp_path / "absent-source"),)),
        )

    assert refusal.value.projection.shortfall_bytes > 0
    assert list((tmp_path / GENERATIONS_DIRNAME).glob("gen-*")) == []
    # A refusal records its bound inputs and shortfall, but cannot calibrate
    # a later projection as if a candidate had been built.
    refused = read_capacity_receipts(tmp_path)
    assert len(refused) == 1
    assert refused[0].status == "refused"


def test_a_first_daemon_start_is_not_refused_by_the_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The preflight is unconditional, so it must clear a fresh root on its own.

    A freshly bootstrapped archive needs 276.8 MiB (the 256 MiB reserve floor
    plus the 16 MiB receipt floor dominate an evidence-proportional term of
    under 4 MiB), which is why this guard does not need an operator-intent
    gate to avoid blocking an ordinary first start.

    Anti-vacuity: raising ``RESERVE_FLOOR_BYTES`` above the supplied 512 MiB,
    or making the projection scale from the filesystem rather than the
    archive, makes this red independently of the runner's actual free space.
    """
    available_bytes = 512 * 1024 * 1024
    _free_space(monkeypatch, available_bytes)
    generation = ColdBuildGeneration.begin(
        tmp_path,
        reason="test",
        observed=ColdBuildGeneration.observe_source_baseline((WatchSource("fixture", tmp_path / "absent-source"),)),
    )
    try:
        receipts = read_capacity_receipts(tmp_path)
        assert [receipt.operation_id for receipt in receipts] == [generation.operation_id]
        receipt = receipts[0]
        assert receipt.required_free_bytes < available_bytes
        assert receipt.available_bytes_at_prediction == available_bytes
        assert receipt.available_bytes_at_prediction >= receipt.required_free_bytes
        assert receipt.final_candidate_allocated_bytes == 0
        assert receipt.baseline_digest == generation.source_baseline.digest
    finally:
        generation.discard()


def test_a_promoted_cold_build_calibrates_the_next_projection(tmp_path: Path, cold_build: ColdBuildGeneration) -> None:
    """Prediction without observation leaves ``calibrated_index_ratio`` at its default.

    ``record_capacity_observation`` used to have exactly one caller, on the
    manual rebuild lifecycle that had no production entry point, so every
    recorded receipt kept ``final_candidate_allocated_bytes == 0`` and every
    projection forever used the unmeasured 4.0 constant.

    Anti-vacuity: deleting the ``observe_candidate_capacity`` call from
    ``ColdBuildGeneration.promote`` leaves the peak at zero and the
    calibration source at ``default``, making both assertions red.
    """
    from polylogue.maintenance.candidate_capacity import calibrated_index_ratio

    root = tmp_path / "sessions"
    root.mkdir()
    _ingest(tmp_path, root, "one.jsonl", "owned-calibrated")
    assert calibrated_index_ratio(tmp_path) == (4.0, "default")

    cold_build.promote()

    receipt = read_capacity_receipts(tmp_path)[0]
    assert receipt.operation_id == cold_build.operation_id
    assert receipt.final_candidate_allocated_bytes > 0
    assert receipt.candidate_generation_id == cold_build.generation_id
    assert receipt.observations == 1
    ratio, source = calibrated_index_ratio(tmp_path)
    assert source == "recorded"
    assert ratio > 0


def test_fresh_capacity_uses_sealed_material_without_a_second_source_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.sources.live import production_baseline

    _free_space(monkeypatch, 1024**3)
    real_revision = production_baseline._revision
    reads = 0

    def measured_revision(
        path: Path, *, cancelled: Any = None, location: Any = None, source_binding: Any = None
    ) -> tuple[str, int]:
        nonlocal reads
        reads += 1
        digest, _size = real_revision(path, cancelled=cancelled, location=location, source_binding=source_binding)
        return digest, 2 * 1024**3

    monkeypatch.setattr(production_baseline, "_revision", measured_revision)
    large_root = tmp_path / "large"
    large_source = tmp_path / "large-source"
    large_source.mkdir()
    (large_source / "session.json").write_bytes(b"{}")
    assert active_index_generation_is_empty(large_root)
    with pytest.raises(InsufficientCapacityError):
        ColdBuildGeneration.begin(
            large_root,
            reason="test",
            observed=ColdBuildGeneration.observe_source_baseline(
                (WatchSource("fixture", large_source, layout=export_drop_layout((".json",))),)
            ),
        )
    assert reads == 1
    assert list((large_root / GENERATIONS_DIRNAME).glob("gen-*")) == []
    refusal = read_capacity_receipts(large_root)[0]
    assert refusal.prospective_material_bytes == 2 * 1024**3
    assert refusal.status == "refused"

    monkeypatch.setattr(production_baseline, "_revision", real_revision)
    small_root = tmp_path / "small"
    assert active_index_generation_is_empty(small_root)
    generation = ColdBuildGeneration.begin(
        small_root,
        reason="test",
        observed=ColdBuildGeneration.observe_source_baseline(
            (WatchSource("fixture", large_source, layout=export_drop_layout((".json",))),)
        ),
    )
    try:
        receipt = read_capacity_receipts(small_root)[0]
        assert receipt.status == "admitted"
        assert receipt.prospective_material_bytes == 2
        assert receipt.baseline_digest == generation.source_baseline.digest
    finally:
        generation.discard()


def test_a_failed_capacity_observation_does_not_block_promotion(
    tmp_path: Path, cold_build: ColdBuildGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calibration is evidence about the next build, never a gate on this one.

    Anti-vacuity: letting the observation error escape
    ``observe_candidate_capacity`` makes ``promote`` raise ``OSError`` here.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    _ingest(tmp_path, root, "one.jsonl", "owned-unmeasured")

    def explode(*_args: object, **_kwargs: object) -> None:
        raise OSError("receipt directory is gone")

    monkeypatch.setattr("polylogue.maintenance.candidate_capacity.record_capacity_observation", explode)

    assert cold_build.promote().state == "active"
    assert _active_session_count(tmp_path) == 1


def _declared_source_root(tmp_path: Path) -> Path:
    root = tmp_path / "declared"
    root.mkdir()
    (root / "one.jsonl").write_bytes(_codex_session("declared-one", "declared"))
    return root


def _fresh_archive_root(tmp_path: Path) -> Path:
    archive = tmp_path / "archive"
    archive.mkdir()
    return archive


def _reconcile_on_daemon_writer(archive: Path) -> None:
    """Reconcile interrupted promotions on the daemon writer, as daemon startup does."""
    asyncio.run(
        run_archive_fixture_write(archive, lambda: ColdBuildGeneration.reconcile_interrupted_promotions(archive))
    )


def test_cold_build_captures_effective_source_baseline(tmp_path: Path) -> None:
    """Removing the production source capture leaves the accepted revision unbound."""
    archive = _fresh_archive_root(tmp_path)
    source_root = _declared_source_root(tmp_path)
    source = WatchSource(name="codex", root=source_root, layout=export_drop_layout((".jsonl",)))
    generation = ColdBuildGeneration.begin(
        archive, reason="test", observed=ColdBuildGeneration.observe_source_baseline((source,))
    )
    try:
        assert len(generation.source_baseline.accepted) == 1
        assert generation.source_baseline.accepted[0].path == str(source_root / "one.jsonl")
        receipt = json.loads((generation.generation_root / "source-baseline.json").read_text())
        assert receipt["generation_id"] == generation.generation_id
    finally:
        generation.discard()


def test_required_missing_source_blocks_promotion(tmp_path: Path) -> None:
    """A configured root that vanished before capture cannot yield a clean promotion."""
    from polylogue.sources.live.production_baseline import ProductionBaselineError

    archive = _fresh_archive_root(tmp_path)
    source = WatchSource(
        name="account", root=tmp_path / "missing", layout=export_drop_layout((".json",)), required=True
    )
    generation = ColdBuildGeneration.begin(
        archive, reason="test", observed=ColdBuildGeneration.observe_source_baseline((source,))
    )
    try:
        assert generation.source_baseline.decisions[0].disposition == "fault"
        with pytest.raises(ProductionBaselineError, match="discovery fault"):
            generation.promote()
    finally:
        generation.discard()


def test_faulted_baseline_refresh_retains_prior_accepted_revisions(tmp_path: Path) -> None:
    from polylogue.sources.live.production_baseline import load_pending_production_baseline

    archive = _fresh_archive_root(tmp_path)
    first_root = tmp_path / "first"
    first_root.mkdir()
    first = first_root / "first.jsonl"
    first.write_bytes(_codex_session("first", "first"))
    second_root = tmp_path / "second"
    sources = (
        WatchSource("codex", first_root, layout=export_drop_layout((".jsonl",)), required=True),
        WatchSource("codex", second_root, layout=export_drop_layout((".jsonl",)), required=True),
    )
    generation = ColdBuildGeneration.begin(
        archive, reason="test", observed=ColdBuildGeneration.observe_source_baseline(sources)
    )
    register_cold_build_generation(generation)
    try:
        assert any(row.disposition == "fault" for row in generation.source_baseline.decisions)
        assert asyncio.run(_ingest_paths(archive, first_root, [first])).succeeded_file_count == 1
        generation.refresh_accepted_progress()
        assert generation.accepted_progress[:2] == (1, 1)
        first.unlink()
        second_root.mkdir()
        second = second_root / "second.jsonl"
        second.write_bytes(_codex_session("second", "second"))
        assert asyncio.run(_ingest_paths(archive, second_root, [second])).succeeded_file_count == 1
        generation.refresh_accepted_progress()
        assert generation.accepted_progress[:2] == (1, 1)

        observed_baseline = generation.observe_faulted_baseline(sources)
        assert observed_baseline is not None
        assert generation.refresh_faulted_baseline(observed_baseline)
        assert generation.accepted_progress[:2] == (2, 2)
        assert not generation.refresh_faulted_baseline(observed_baseline)
        assert not any(row.disposition == "fault" for row in generation.source_baseline.decisions)
        assert {row.path for row in generation.source_baseline.accepted} == {str(first), str(second)}
        pending = load_pending_production_baseline(archive)
        assert pending is not None and pending.digest == generation.source_baseline.digest
        bound = json.loads((generation.generation_root / "source-baseline.json").read_text())
        assert bound["baseline"]["digest"] == pending.digest
        assert generation.promote().state == "active"
        assert _active_session_count(archive) == 2
    finally:
        clear_cold_build_generation()
        if not generation.settled:
            generation.discard()


def test_faulted_baseline_refresh_reuses_candidate_and_retained_evidence_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.maintenance.candidate_capacity import (
        CandidateCapacityProjection,
        evidence_allocation_block_bytes,
        project_candidate_capacity,
    )
    from polylogue.sources.live.production_baseline import (
        capture_production_source_baseline,
        merge_pending_production_baseline,
    )

    archive = _fresh_archive_root(tmp_path)
    source_root = tmp_path / "later-source"
    sources = (WatchSource("codex", source_root, layout=export_drop_layout((".jsonl",)), required=True),)
    generation = ColdBuildGeneration.begin(
        archive, reason="test", observed=ColdBuildGeneration.observe_source_baseline(sources)
    )
    register_cold_build_generation(generation)
    try:
        source_root.mkdir()
        source = source_root / "one.jsonl"
        source.write_bytes(_codex_session("one", "one"))
        assert asyncio.run(_ingest_paths(archive, source_root, [source])).succeeded_file_count == 1
        assert generation.session_count() == 1
        observed = capture_production_source_baseline(sources, operation_id=generation.operation_id)
        merged = merge_pending_production_baseline(observed, generation.source_baseline)
        blob_block_bytes, source_db_block_bytes = evidence_allocation_block_bytes(archive)
        material_bytes = merged.prospective_material_bytes
        retained_bytes = merged.prospective_retained_allocation_bytes(blob_block_bytes)
        assert material_bytes is not None and retained_bytes is not None

        def projection(existing_candidate_generation_id: str | None = None) -> CandidateCapacityProjection:
            return project_candidate_capacity(
                archive,
                existing_candidate_generation_id=existing_candidate_generation_id,
                prospective_material_bytes=material_bytes,
                prospective_retained_allocation_bytes=retained_bytes,
                prospective_source_db_allocation_bytes=merged.prospective_source_db_allocation_bytes(
                    source_db_block_bytes
                ),
            )

        full = projection()
        reused = projection(generation.generation_id)
        retained = project_candidate_capacity(
            archive,
            existing_candidate_generation_id=generation.generation_id,
            prospective_material_bytes=0,
            prospective_retained_allocation_bytes=0,
            prospective_source_db_allocation_bytes=0,
        )
        assert reused.existing_candidate_index_allocated_bytes > 0
        assert reused.required_free_bytes < full.required_free_bytes
        assert retained.required_free_bytes < reused.required_free_bytes
        _free_space(monkeypatch, (retained.required_free_bytes + reused.required_free_bytes) // 2)
        assert not projection().sufficient
        assert not projection(generation.generation_id).sufficient
        observed_baseline = generation.observe_faulted_baseline(sources)
        assert observed_baseline is not None
        assert generation.refresh_faulted_baseline(observed_baseline)
        assert generation.generation_id == reused.inventory.generations[-1].generation_id
    finally:
        clear_cold_build_generation()
        if not generation.settled:
            generation.discard()


def test_next_build_reclaims_abandoned_inactive_candidate_before_capacity(tmp_path: Path) -> None:
    archive = _fresh_archive_root(tmp_path)
    sources = (WatchSource("fixture", tmp_path / "absent-source"),)
    abandoned = ColdBuildGeneration.begin(
        archive, reason="first", observed=ColdBuildGeneration.observe_source_baseline(sources)
    )
    abandoned_root = abandoned.generation_root
    assert abandoned_root.exists()

    replacement = ColdBuildGeneration.begin(
        archive, reason="restart", observed=ColdBuildGeneration.observe_source_baseline(sources)
    )
    try:
        assert not abandoned_root.exists()
        assert replacement.generation_root.exists()
        assert replacement.generation_id != abandoned.generation_id
    finally:
        replacement.discard()


def test_orphan_replacement_does_not_charge_already_retained_source_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.maintenance.candidate_capacity import require_candidate_capacity as original_require
    from polylogue.sources.live import cold_build as cold_build_module

    archive = _fresh_archive_root(tmp_path)
    source_root = tmp_path / "account"
    source_root.mkdir()
    member = source_root / "one.jsonl"
    member.write_bytes(_codex_session("one", "retained"))
    source = WatchSource("codex", source_root, layout=export_drop_layout((".jsonl",)), required=True)
    orphan = ColdBuildGeneration.begin(
        archive, reason="first", observed=ColdBuildGeneration.observe_source_baseline((source,))
    )
    register_cold_build_generation(orphan)
    try:
        assert asyncio.run(_ingest_paths(archive, source_root, [member])).succeeded_file_count == 1
    finally:
        clear_cold_build_generation()
    old_root = orphan.generation_root
    admissions = 0

    def require_only_new_evidence(*args: Any, **kwargs: Any) -> Any:
        nonlocal admissions
        admissions += 1
        assert kwargs["prospective_material_bytes"] == 0
        assert kwargs["prospective_retained_allocation_bytes"] == 0
        assert kwargs["prospective_source_db_allocation_bytes"] == 0
        return original_require(*args, **kwargs)

    monkeypatch.setattr(cold_build_module, "require_candidate_capacity", require_only_new_evidence)
    replacement = ColdBuildGeneration.begin(
        archive, reason="restart", observed=ColdBuildGeneration.observe_source_baseline((source,))
    )
    try:
        assert admissions == 1
        assert not old_root.exists()
        assert replacement.generation_id != orphan.generation_id
    finally:
        replacement.discard()


def test_discarded_generation_carries_deleted_source_into_retry(tmp_path: Path) -> None:
    """A crash after capture cannot shrink the next generation's denominator."""
    from polylogue.sources.live.production_baseline import (
        ProductionBaselineError,
        load_pending_production_baseline,
    )

    archive = _fresh_archive_root(tmp_path)
    source_root = tmp_path / "account"
    source_root.mkdir()
    member = source_root / "A.json"
    member.write_text('{"session":"A"}')
    source = WatchSource("account", source_root, layout=export_drop_layout((".json",)), required=True)
    first = ColdBuildGeneration.begin(
        archive, reason="first", observed=ColdBuildGeneration.observe_source_baseline((source,))
    )
    assert [row.path for row in first.source_baseline.accepted] == [str(member)]
    first.discard()
    member.unlink()

    retry = ColdBuildGeneration.begin(
        archive, reason="retry", observed=ColdBuildGeneration.observe_source_baseline((source,))
    )
    try:
        assert [row.path for row in retry.source_baseline.accepted] == [str(member)]
        pending = load_pending_production_baseline(archive)
        assert pending is not None
        assert pending.digest == retry.source_baseline.digest
        with pytest.raises(ProductionBaselineError, match="unretained revision"):
            retry.promote()
    finally:
        retry.discard()


@pytest.mark.parametrize("reason", ["explicit_cold_build", "empty_active_index_generation", "interrupted_promotion"])
def test_generation_lifecycle_preserves_stable_reason_in_production_events(tmp_path: Path, reason: str) -> None:
    """Generation lifecycle emits its declared startup/recovery reason without a drop."""
    from polylogue import logging as plog

    observed = ColdBuildGeneration.observe_source_baseline((WatchSource("fixture", tmp_path / "absent-source"),))
    previous_level = plog.set_level("info")
    try:
        with plog.capture() as records:
            generation = ColdBuildGeneration.begin(tmp_path, reason=reason, observed=observed)
            assert generation.discard() is True
        lifecycle_names = {"daemon.cold_build.generation_created", "daemon.cold_build.generation_discarded"}
        lifecycle = [record for record in records if record["event"] in lifecycle_names]
        assert {record["event"] for record in lifecycle} == lifecycle_names
        assert all(record["reason"] == reason for record in lifecycle)
        assert not any(
            record["event"] == "log.field_rejected" and record.get("source_event") in lifecycle_names
            for record in records
        )
    finally:
        plog.set_level(previous_level)


def test_generation_promotion_event_preserves_actual_predecessor(
    tmp_path: Path,
    cold_build: ColdBuildGeneration,
) -> None:
    from polylogue import logging as plog

    previous_level = plog.set_level("info")
    try:
        first = cold_build.promote()
        observed = ColdBuildGeneration.observe_source_baseline((WatchSource("fixture", tmp_path / "absent-source"),))
        second = ColdBuildGeneration.begin(tmp_path, reason="explicit_cold_build", observed=observed)
        with plog.capture() as records:
            promoted = second.promote()
        events = [record for record in records if record["event"] == "daemon.cold_build.generation_promoted"]
        assert len(events) == 1
        assert promoted.predecessor_generation_id == first.generation_id
        assert events[0]["predecessor"] == first.generation_id
        assert not any(
            record["event"] == "log.field_rejected"
            and record.get("source_event") == "daemon.cold_build.generation_promoted"
            for record in records
        )
    finally:
        plog.set_level(previous_level)
