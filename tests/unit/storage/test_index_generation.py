from __future__ import annotations

import fcntl
import json
import logging
import multiprocessing
import os
import sqlite3
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.index_generation import (
    RETENTION_RECEIPT_HISTORY,
    ActiveWriterLease,
    IndexGenerationStore,
    RebuildLease,
    RebuildLeaseUnavailableError,
    rebuild_lease_status,
    source_revision_snapshot,
)

# A pid guaranteed to never correspond to a running process: it exceeds any
# realistic pid_max (Linux defaults to <= 4194304 even with 64-bit pids).
_DEFINITELY_DEAD_PID = 2**31 - 1
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.archive_templates import clone_archive_template, finalize_archive_template

_ARCHIVE_TEMPLATE: Path | None = None


@pytest.fixture(scope="module", autouse=True)
def _archive_template(tmp_path_factory: pytest.TempPathFactory) -> Generator[None]:
    """Build the deterministic five-tier fixture once, then clone per test."""
    global _ARCHIVE_TEMPLATE
    template = tmp_path_factory.mktemp("index-generation-template") / "archive"
    for tier in (ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.EMBEDDINGS, ArchiveTier.OPS, ArchiveTier.INDEX):
        initialize_archive_database(template / f"{tier.value}.db", tier)
    finalize_archive_template(template)
    _ARCHIVE_TEMPLATE = template
    try:
        yield
    finally:
        _ARCHIVE_TEMPLATE = None


def _hold_lease(
    root: str, ready: multiprocessing.synchronize.Event, release: multiprocessing.synchronize.Event
) -> None:
    with RebuildLease(Path(root)):
        ready.set()
        release.wait(5)


def _archive(root: Path) -> None:
    assert _ARCHIVE_TEMPLATE is not None
    clone_archive_template(_ARCHIVE_TEMPLATE, root)


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


def test_rebuild_lease_reclaims_lock_held_by_dead_pid(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A lock file recorded as held by a pid that no longer exists is stale and reclaimable.

    Simulates the polylogue-k8kj live incident: a genuinely-locked inode
    (held here by a raw fd we keep open in-process, standing in for a
    crashed rebuild or an orphaned forked worker) whose lock file records a
    pid that is not actually running. A fresh ``RebuildLease`` acquisition
    must reclaim it -- not raise ``RebuildLeaseUnavailableError`` -- and log
    the reclamation loudly.
    """
    lock_path = tmp_path / ".index-rebuild.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    holder_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(holder_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.write(holder_fd, f"pid={_DEFINITELY_DEAD_PID} host=nowhere\n".encode())
    os.fsync(holder_fd)
    try:
        with caplog.at_level(logging.WARNING):
            with RebuildLease(tmp_path):
                pass
        assert "reclaiming stale index rebuild lease" in caplog.text
        assert str(_DEFINITELY_DEAD_PID) in caplog.text
    finally:
        fcntl.flock(holder_fd, fcntl.LOCK_UN)
        os.close(holder_fd)


def test_rebuild_lease_still_refuses_lock_held_by_live_pid(tmp_path: Path) -> None:
    """A lock recorded as held by a genuinely running process must still block.

    Complements the dead-pid reclaim test: recording *this* test process's
    own (very much alive) pid in the lock file must never be treated as
    stale, even though the mechanism for detecting staleness is the same
    read-the-file-then-check-liveness path.
    """
    lock_path = tmp_path / ".index-rebuild.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    holder_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(holder_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.write(holder_fd, f"pid={os.getpid()} host=here\n".encode())
    os.fsync(holder_fd)
    try:
        with pytest.raises(RebuildLeaseUnavailableError):
            with RebuildLease(tmp_path):
                pass
    finally:
        fcntl.flock(holder_fd, fcntl.LOCK_UN)
        os.close(holder_fd)


def test_bootstrap_writes_active_pointer_anchor_on_first_touch(tmp_path: Path) -> None:
    """polylogue-ovme.2.1: ``ArchiveLocation.resolve()`` is a pure read and
    deliberately never writes ``.index-active-pointer`` -- that first-touch
    bootstrap write must still happen, now performed by
    ``IndexGenerationStore`` itself from the resolved (but anchor-less)
    ``ArchiveLocation`` it is constructed from, not lost in the migration
    off a bare ``archive_root: Path``."""
    _archive(tmp_path)
    anchor = tmp_path / ".index-active-pointer"
    assert not anchor.exists()

    store = IndexGenerationStore.for_archive_root(tmp_path)

    assert anchor.exists()
    assert Path(anchor.read_text(encoding="utf-8").strip()) == (tmp_path / "index.db").absolute()
    assert store.active_pointer == tmp_path / "index.db"
    assert store.generations_root == tmp_path / ".index-generations"


def test_store_trusts_the_passed_location_instead_of_rereading_disk(tmp_path: Path) -> None:
    """polylogue-ovme.2.1 anti-regression: the retired constructor re-derived
    ``.index-active-pointer``/generation-root logic straight from disk on
    every construction, independently of any typed resolution a caller had
    already performed -- the exact "duplicate derivation" bug class named in
    this bead (mirrored by the real ``resolve_active_index_db_path`` bug
    ovme.2 found, which read its own module-level state instead of a
    caller-supplied override). Proves the new constructor is no longer
    capable of that: once an ``ArchiveLocation`` is resolved, a later,
    independent on-disk anchor mutation must NOT change what
    ``IndexGenerationStore`` derives from that already-resolved location."""
    _archive(tmp_path)
    other_root = tmp_path / "other-root"
    other_root.mkdir()
    _archive(other_root)

    # Bootstrap tmp_path's own anchor first so ArchiveLocation.resolve below
    # observes a real (non-bootstrapping) pointer read.
    IndexGenerationStore.for_archive_root(tmp_path)
    location = ArchiveLocation.resolve(tmp_path)
    assert location.active_pointer == tmp_path / "index.db"

    # Simulate a raced/foreign rewrite of the anchor file on disk AFTER the
    # location was resolved -- the old constructor re-read the anchor itself
    # and would have picked this up; the migrated one must not.
    (tmp_path / ".index-active-pointer").write_text(str((other_root / "index.db").absolute()), encoding="utf-8")

    store = IndexGenerationStore(location)

    assert store.active_pointer == tmp_path / "index.db"
    assert store.generations_root == tmp_path / ".index-generations"


def test_generation_is_inactive_until_atomic_promotion(tmp_path: Path) -> None:
    _archive(tmp_path)
    original = (tmp_path / "index.db").resolve()
    original_inode = original.stat().st_ino
    store = IndexGenerationStore.for_archive_root(tmp_path)
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
    store = IndexGenerationStore.for_archive_root(tmp_path)
    generation = store.create(owner_id="operator", source_snapshot="snapshot-a")
    stale = replace(generation, owner_id="other")
    with pytest.raises(RuntimeError, match="owning inactive"):
        store.promote(stale)


def test_promotion_removes_only_empty_active_sidecars(tmp_path: Path) -> None:
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    generation = store.create(owner_id="operator", source_snapshot="snapshot-a")
    (tmp_path / "index.db-wal").touch()
    (tmp_path / "index.db-shm").touch()
    store.promote(generation)
    assert not (tmp_path / "index.db-wal").exists()
    assert not (tmp_path / "index.db-shm").exists()


def test_promotion_checkpoints_candidate_and_active_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
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
    store = IndexGenerationStore.for_archive_root(tmp_path)
    generation = store.create(owner_id="operator", source_snapshot="snapshot-a")
    store._write(replace(generation, state="promoting"))
    (tmp_path / "index.db").unlink()

    recovered = store.recover_promotion(generation.generation_id)

    assert recovered.state == "inactive"


def test_recover_promotion_after_pointer_swap_does_not_mark_active(tmp_path: Path) -> None:
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    generation = store.create(owner_id="operator", source_snapshot="snapshot-a")
    store._write(replace(generation, state="promoting"))
    (tmp_path / "index.db").unlink()
    (tmp_path / "index.db").symlink_to(generation.index_path)

    recovered = store.recover_promotion(generation.generation_id)

    assert recovered.state == "promoting"
    assert store.load(generation.generation_id).state == "promoting"

    completed = store.complete_promotion_recovery(generation.generation_id)

    assert completed.state == "active"


def test_recovered_promotion_records_automatic_retention_and_reclamation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recovery completion must use the same retention lifecycle as promotion.

    Anti-vacuity: this exercises the production crash-recovery seam after a
    pointer swap. All three generations share a millisecond, while their UUID
    text is deliberately reverse-chronological. Removing recovery's retention
    collection leaves the third generation without a receipt; ordering by the
    old ``(created_at_ms, generation_id)`` tuple retains the wrong rollback
    target.
    """
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    monkeypatch.setattr("polylogue.storage.index_generation.time.time_ns", lambda: 1_000_000_000)
    uuid_hexes = iter(("ffffffff", "11111111", "22222222", "00000000", "33333333"))
    monkeypatch.setattr(
        "polylogue.storage.index_generation.uuid.uuid4",
        lambda: type("DeterministicUuid", (), {"hex": next(uuid_hexes)})(),
    )
    first = store.create(owner_id="build-1", source_snapshot="snapshot-1")
    store.promote(first)

    recovered_generations = []
    for index in (2, 3):
        generation = store.create(owner_id=f"build-{index}", source_snapshot=f"snapshot-{index}")
        store._write(replace(generation, state="promoting"))
        store.active_pointer.unlink()
        store.active_pointer.symlink_to(generation.index_path)
        recovered_generations.append(store.complete_promotion_recovery(generation.generation_id))

    receipt = store.load_retention_receipt(recovered_generations[-1].generation_id)

    assert receipt.states_by_generation_id == {
        recovered_generations[-1].generation_id: "active",
        recovered_generations[-2].generation_id: "retained",
        first.generation_id: "reclaimed",
    }
    assert Path(recovered_generations[-2].index_path).exists()
    assert not Path(first.index_path).parent.exists()
    assert {
        first.created_at_ms,
        recovered_generations[0].created_at_ms,
        recovered_generations[1].created_at_ms,
    } == {1_000}


def test_promotion_bounds_retention_receipt_history(tmp_path: Path) -> None:
    """Receipt evidence is automatic, but its bounded history cannot grow forever.

    Anti-vacuity: this performs four real promotions, then inspects the
    production receipt directory. Removing receipt pruning leaves all four
    receipt files behind instead of the active and immediately prior proofs.
    """
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)

    promoted = []
    for index in range(4):
        generation = store.create(owner_id=f"build-{index}", source_snapshot=f"snapshot-{index}")
        store.promote(generation)
        promoted.append(generation)

    receipts = {path.stem for path in (store.generations_root / "retention-receipts").glob("*.json")}

    assert RETENTION_RECEIPT_HISTORY == 2
    assert receipts == {promoted[-1].generation_id, promoted[-2].generation_id}


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
    store = IndexGenerationStore.for_archive_root(tmp_path)
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

    store = IndexGenerationStore.for_archive_root(configured)
    generation = store.create(owner_id="operator", source_snapshot="snapshot-a")
    store.promote(generation)

    assert configured.joinpath("index.db").is_symlink()
    assert configured.joinpath("index.db").stat().st_ino == canonical.joinpath("index.db").stat().st_ino
    assert canonical.joinpath("index.db").resolve() == Path(generation.index_path).resolve()
    assert store.generations_root.parent == canonical

    second_store = IndexGenerationStore.for_archive_root(configured)
    second = second_store.create(owner_id="operator-2", source_snapshot="snapshot-b")
    second_store.promote(second)
    assert second_store.active_pointer == canonical / "index.db"
    assert configured.joinpath("index.db").stat().st_ino == canonical.joinpath("index.db").stat().st_ino
    assert canonical.joinpath("index.db").resolve() == Path(second.index_path).resolve()
    # Both promotions retired the previous pointer; superseded-generation
    # retention (polylogue-wmft, keep=1) then prunes all but the newest marker.
    assert len(tuple(store.generations_root.glob("retired-*/index.db"))) == 1
    assert tuple(store.generations_root.glob("gen-*/generation.json")) != ()


def test_rebuild_transaction_persists_keyset_cursor_without_materializing_archive(tmp_path: Path) -> None:
    """polylogue-hord: paging orders by ``(blob_hash, raw_id)``, not acquisition
    time, so byte-identical duplicates land adjacently. ``raw-a``/``raw-b``
    deliberately share a ``blob_hash`` (proving the tie is broken by
    ``raw_id``, exactly as the old acquired_at-tie test proved) while
    ``raw-c`` gets a distinct, lexicographically-later hash -- acquired_at is
    set in the OPPOSITE order from hash order to prove paging no longer
    follows it.
    """
    _archive(tmp_path)
    hash_group_1 = b"\x01" * 32
    hash_group_2 = b"\x02" * 32
    with sqlite3.connect(tmp_path / "source.db") as conn:
        for raw_id, acquired_at_ms, blob_hash in (
            ("raw-c", 10, hash_group_2),
            ("raw-a", 30, hash_group_1),
            ("raw-b", 30, hash_group_1),
        ):
            conn.execute(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, native_id, source_path, source_index, blob_hash,
                    blob_size, acquired_at_ms, validation_status
                ) VALUES (?, 'codex-session', ?, ?, 0, ?, 1, ?, 'passed')
                """,
                (raw_id, raw_id, f"/{raw_id}.jsonl", blob_hash, acquired_at_ms),
            )

    store = IndexGenerationStore.for_archive_root(tmp_path)
    transaction = store.create_transaction(
        source_snapshot="source-v1", operation_id="resume-me", pass_byte_budget=2, pass_deadline_ms=5_000
    )
    first_page = store.next_raw_page(transaction, limit=2)
    assert first_page.rows == (("raw-a", hash_group_1.hex(), 1), ("raw-b", hash_group_1.hex(), 1))
    assert first_page.deferred_reason == "raw-batch"

    transaction = store.checkpoint_transaction(
        transaction,
        status="paused",
        last_blob_hash_hex=hash_group_1.hex(),
        last_raw_id="raw-b",
        processed_raw_count=2,
    )
    assert transaction.cursor == f"source:{hash_group_1.hex()}:raw-b"
    assert store.load_transaction("resume-me") == transaction
    assert store.next_raw_page(transaction, limit=2).rows == (("raw-c", hash_group_2.hex(), 1),)
    assert transaction.pass_byte_budget == 2


def test_rebuild_byte_budget_defers_without_excluding_an_oversized_first_raw(tmp_path: Path) -> None:
    _archive(tmp_path)
    hash_large = b"\x01" * 32
    hash_later = b"\x02" * 32
    with sqlite3.connect(tmp_path / "source.db") as conn:
        for raw_id, blob_hash, blob_size in (("large", hash_large, 100), ("later", hash_later, 1)):
            conn.execute(
                """INSERT INTO raw_sessions (raw_id, origin, native_id, source_path, source_index, blob_hash,
                   blob_size, acquired_at_ms, validation_status)
                   VALUES (?, 'codex-session', ?, ?, 0, ?, ?, 0, 'passed')""",
                (raw_id, raw_id, f"/{raw_id}", blob_hash, blob_size),
            )
    store = IndexGenerationStore.for_archive_root(tmp_path)
    transaction = store.create_transaction(source_snapshot="source-v1", pass_byte_budget=10)
    first = store.next_raw_page(transaction, limit=10)
    assert first.rows == (("large", hash_large.hex(), 100),)
    assert first.deferred_reason == "byte-budget"
    transaction = store.checkpoint_transaction(
        transaction,
        status="deferred",
        last_blob_hash_hex=hash_large.hex(),
        last_raw_id="large",
        processed_raw_count=1,
        processed_blob_bytes=100,
    )
    assert store.next_raw_page(transaction, limit=10).rows == (("later", hash_later.hex(), 1),)


def test_source_snapshot_changes_when_retained_blob_identity_changes(tmp_path: Path) -> None:
    _archive(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            """INSERT INTO raw_sessions (raw_id, origin, native_id, source_path, source_index, blob_hash,
               blob_size, acquired_at_ms, validation_status)
               VALUES ('raw-a', 'codex-session', 'raw-a', '/raw-a', 0, randomblob(32), 1, 1, 'passed')"""
        )
    before = source_revision_snapshot(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET blob_hash = randomblob(32), blob_size = 2 WHERE raw_id = 'raw-a'")
    assert source_revision_snapshot(tmp_path) != before


def test_derived_stores_cleared_defaults_false_and_round_trips_through_checkpoint(tmp_path: Path) -> None:
    """polylogue-v6i3: ``derived_stores_cleared`` is the transaction marker
    guarding the bulk-build "empty derived stores at resume" clear from
    re-firing on every subsequent page of the same operation. It must
    default False for a fresh transaction, persist True once checkpointed,
    and survive a reload via ``load_transaction``."""
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    transaction = store.create_transaction(source_snapshot="source-v1", operation_id="bulk-build-op")
    assert transaction.derived_stores_cleared is False

    transaction = store.checkpoint_transaction(transaction, status="running", derived_stores_cleared=True)
    assert transaction.derived_stores_cleared is True
    assert store.load_transaction("bulk-build-op").derived_stores_cleared is True

    # A later checkpoint that doesn't mention the field must not reset it.
    transaction = store.checkpoint_transaction(transaction, status="paused")
    assert transaction.derived_stores_cleared is True


def test_derived_stores_cleared_missing_from_persisted_json_defaults_false(tmp_path: Path) -> None:
    """A transaction persisted before this field existed (no key in its JSON)
    must load as ``False`` via the dataclass default, not raise or silently
    invent a different value."""
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    transaction = store.create_transaction(source_snapshot="source-v1", operation_id="pre-existing-op")
    path = store._transaction_path("pre-existing-op")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "derived_stores_cleared" in payload
    del payload["derived_stores_cleared"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    reloaded = store.load_transaction("pre-existing-op")
    assert reloaded.derived_stores_cleared is False
    assert reloaded.operation_id == transaction.operation_id


def test_promotion_prunes_superseded_generations(tmp_path: Path) -> None:
    """polylogue-wmft: a promoted generation is ~35 GB and nothing ever removed
    one. ``promote`` retired the *pointer* into a marker directory but left the
    superseded ``gen-*`` directory forever, and ``discard_if_inactive`` only
    disposes of candidates that were never promoted -- a live archive had
    accumulated nine dead generations, ~290 GB."""
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)

    promoted_ids = []
    for index in range(4):
        generation = store.create(owner_id="operator", source_snapshot=f"snapshot-{index}")
        store.promote(generation)
        promoted_ids.append(generation.generation_id)

    surviving = {path.parent.name for path in store.generations_root.glob("gen-*/generation.json")}
    active = Path(store.active_pointer).resolve(strict=True)

    # The active generation plus exactly one rollback target.
    assert surviving == {promoted_ids[-1], promoted_ids[-2]}
    assert active == Path(store.load(promoted_ids[-1]).index_path).resolve()
    # Markers follow the same retention rather than dangling at the removed ones.
    assert len(list(store.generations_root.glob("retired-*"))) == 1


def test_promotion_records_automatic_retention_and_reclamation(tmp_path: Path) -> None:
    """The real blue-green seam retains one rollback generation, then records GC.

    Anti-vacuity: this calls ``IndexGenerationStore.promote`` against actual
    SQLite index generations. Removing promotion's retention lifecycle call
    leaves the prior generation's metadata untouched and no durable receipt,
    so this test fails instead of merely checking a test-local planner.
    """
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)

    promoted = []
    for index in range(3):
        generation = store.create(owner_id=f"build-{index}", source_snapshot=f"snapshot-{index}")
        store.promote(generation)
        promoted.append(generation)

    receipt = store.load_retention_receipt(promoted[-1].generation_id)

    assert receipt.automatic is True
    assert receipt.retention_boundary == 1
    assert receipt.states_by_generation_id == {
        promoted[-1].generation_id: "active",
        promoted[-2].generation_id: "retained",
        promoted[-3].generation_id: "reclaimed",
    }
    assert receipt.eligible_generation_ids == (promoted[-3].generation_id,)
    assert receipt.owner_by_generation_id[promoted[-2].generation_id] == promoted[-1].generation_id
    assert receipt.owner_by_generation_id[promoted[-3].generation_id] == promoted[-1].generation_id
    assert Path(promoted[-2].index_path).exists(), "the rollback generation was reclaimed before its boundary"
    assert not Path(promoted[-3].index_path).parent.exists(), "eligible generation was not reclaimed automatically"


def test_promotion_retains_actual_predecessor_when_generation_ids_reverse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rollback retention follows the preceding pointer target, never UUID text.

    Anti-vacuity: this drives three production blue-green promotions with all
    generation timestamps in one millisecond. The first ID sorts after the
    actual predecessor, so the previous ``(created_at_ms, generation_id)``
    ordering retains the wrong generation after the third swap.
    """
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    monkeypatch.setattr("polylogue.storage.index_generation.time.time", lambda: 1.0)
    monkeypatch.setattr("polylogue.storage.index_generation.time.time_ns", lambda: 1_000_000_000)
    uuid_hexes = iter(
        ("ffffffff", "11111111", "22222222", "00000000", "33333333", "44444444", "55555555", "66666666", "77777777")
    )
    monkeypatch.setattr(
        "polylogue.storage.index_generation.uuid.uuid4",
        lambda: type("DeterministicUuid", (), {"hex": next(uuid_hexes)})(),
    )

    promoted = []
    for index in range(3):
        generation = store.create(owner_id=f"build-{index}", source_snapshot=f"snapshot-{index}")
        store.promote(generation)
        promoted.append(generation)

    receipt = store.load_retention_receipt(promoted[-1].generation_id)

    assert {generation.created_at_ms for generation in promoted} == {1_000}
    assert receipt.states_by_generation_id == {
        promoted[-1].generation_id: "active",
        promoted[-2].generation_id: "retained",
        promoted[-3].generation_id: "reclaimed",
    }


def test_promotion_refuses_ownerless_predecessor_before_pointer_swap(tmp_path: Path) -> None:
    """An ownerless predecessor cannot become an unaccountable GC candidate.

    The mutation that makes this fail is removing the promotion-time ownership
    preflight. The old promotion path accepted this corrupted predecessor,
    changed the active pointer, and left later GC unable to prove who owned
    the superseded generation.
    """
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    predecessor = store.create(owner_id="first-build", source_snapshot="snapshot-a")
    store.promote(predecessor)
    predecessor_metadata = Path(predecessor.index_path).with_name("generation.json")
    payload = json.loads(predecessor_metadata.read_text(encoding="utf-8"))
    payload["owner_id"] = ""
    predecessor_metadata.write_text(json.dumps(payload), encoding="utf-8")
    candidate = store.create(owner_id="second-build", source_snapshot="snapshot-b")

    with pytest.raises(RuntimeError, match="retention ownership"):
        store.promote(candidate)

    assert Path(store.active_pointer).resolve(strict=True) == Path(predecessor.index_path).resolve()
    assert store.load(candidate.generation_id).state == "inactive"


def test_pruning_never_removes_a_never_promoted_rebuild_candidate(tmp_path: Path) -> None:
    """An in-flight or paused rebuild candidate is `inactive` -- never promoted --
    and must survive an unrelated promotion's housekeeping.

    Treating every non-active generation as superseded let a promotion delete a
    rebuild in progress, and let a newer inactive candidate consume the single
    retained slot so the real rollback target was pruned instead. Never-promoted
    candidates belong to ``discard_if_inactive``, driven by their owner.
    """
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)

    first = store.create(owner_id="operator", source_snapshot="snapshot-a")
    store.promote(first)
    # A resumable rebuild candidate, created but never promoted.
    candidate = store.create(owner_id="rebuild", source_snapshot="snapshot-candidate")
    second = store.create(owner_id="operator", source_snapshot="snapshot-b")
    store.promote(second)

    assert store.load(candidate.generation_id).state == "inactive"
    assert Path(candidate.index_path).exists(), "an unrelated promotion deleted a live rebuild candidate"
    # The genuine rollback target -- the previously-active generation -- is what
    # the retained slot is for, not the inactive candidate.
    assert Path(first.index_path).exists()


def _seed_membership(
    source_db: Path,
    *,
    raw_id: str,
    logical_source_key: str,
    decision: str | None,
) -> None:
    with sqlite3.connect(source_db) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            INSERT INTO raw_session_memberships (
                raw_id, logical_source_key, provider_session_id, source_revision,
                normalized_content_hash, message_count, revision_authority, decision, decided_at_ms
            ) VALUES (?, ?, ?, ?, zeroblob(32), 1, 'byte_proven', ?, ?)
            """,
            (raw_id, logical_source_key, raw_id, raw_id, decision, 1 if decision is not None else None),
        )


def _seed_raw(conn: sqlite3.Connection, *, raw_id: str, blob_hash: bytes, acquired_at_ms: int) -> None:
    conn.execute(
        """INSERT INTO raw_sessions (raw_id, origin, native_id, source_path, source_index, blob_hash,
           blob_size, acquired_at_ms, validation_status)
           VALUES (?, 'codex-session', ?, ?, 0, ?, 1, ?, 'passed')""",
        (raw_id, raw_id, f"/{raw_id}.jsonl", blob_hash, acquired_at_ms),
    )


class TestNextRawPageExcludesSupersededResumeDebt:
    """polylogue-b5l.1 AC3: a raw whose every persisted membership decision is
    ``superseded_equivalent``/``superseded_prefix`` is resolved history, not
    resume debt -- it never gains its own ``index.sessions`` row (only its
    cohort's accepted head does), so scheduling it every pass wastes a full
    page slot re-parsing content a prior pass already resolved. A genuinely
    accepted-but-unindexed raw, or one never censused at all, must still be
    selected.
    """

    def test_fully_superseded_raw_is_excluded_from_the_page(self, tmp_path: Path) -> None:
        _archive(tmp_path)
        with sqlite3.connect(tmp_path / "source.db") as conn:
            _seed_raw(conn, raw_id="raw-superseded", blob_hash=b"\x01" * 32, acquired_at_ms=10)
            _seed_raw(conn, raw_id="raw-accepted", blob_hash=b"\x02" * 32, acquired_at_ms=20)
        _seed_membership(
            tmp_path / "source.db",
            raw_id="raw-superseded",
            logical_source_key="cohort-1",
            decision="superseded_prefix",
        )
        _seed_membership(
            tmp_path / "source.db",
            raw_id="raw-accepted",
            logical_source_key="cohort-2",
            decision="applied",
        )

        store = IndexGenerationStore.for_archive_root(tmp_path)
        transaction = store.create_transaction(source_snapshot="source-v1")
        page = store.next_raw_page(transaction, limit=10)

        raw_ids = [row[0] for row in page.rows]
        assert raw_ids == ["raw-accepted"]

    def test_never_censused_raw_remains_eligible(self, tmp_path: Path) -> None:
        """A raw with no membership row at all (never classified) must still
        be scheduled -- excluding it would silently drop genuinely novel,
        never-processed content."""
        _archive(tmp_path)
        with sqlite3.connect(tmp_path / "source.db") as conn:
            _seed_raw(conn, raw_id="raw-novel", blob_hash=b"\x03" * 32, acquired_at_ms=10)

        store = IndexGenerationStore.for_archive_root(tmp_path)
        transaction = store.create_transaction(source_snapshot="source-v1")
        page = store.next_raw_page(transaction, limit=10)

        assert [row[0] for row in page.rows] == ["raw-novel"]

    def test_raw_superseded_in_one_cohort_but_pending_in_another_remains_eligible(self, tmp_path: Path) -> None:
        """A multi-membership raw (e.g. a bundle member) is only resume-debt
        -free when EVERY known membership row is superseded; a mixed shape
        (superseded in one cohort, still ambiguous/pending in another) must
        remain eligible."""
        _archive(tmp_path)
        with sqlite3.connect(tmp_path / "source.db") as conn:
            _seed_raw(conn, raw_id="raw-mixed", blob_hash=b"\x04" * 32, acquired_at_ms=10)
        _seed_membership(
            tmp_path / "source.db", raw_id="raw-mixed", logical_source_key="cohort-a", decision="superseded_prefix"
        )
        _seed_membership(tmp_path / "source.db", raw_id="raw-mixed", logical_source_key="cohort-b", decision=None)

        store = IndexGenerationStore.for_archive_root(tmp_path)
        transaction = store.create_transaction(source_snapshot="source-v1")
        page = store.next_raw_page(transaction, limit=10)

        assert [row[0] for row in page.rows] == ["raw-mixed"]

    def test_exclusion_survives_the_keyset_cursor_across_pages(self, tmp_path: Path) -> None:
        """The superseded-exclusion filter is applied inside the same SQL
        query as the keyset cursor, so a superseded raw sitting between two
        eligible pages must never surface on a later page either."""
        _archive(tmp_path)
        with sqlite3.connect(tmp_path / "source.db") as conn:
            _seed_raw(conn, raw_id="raw-a", blob_hash=b"\x01" * 32, acquired_at_ms=10)
            _seed_raw(conn, raw_id="raw-superseded", blob_hash=b"\x02" * 32, acquired_at_ms=20)
            _seed_raw(conn, raw_id="raw-b", blob_hash=b"\x03" * 32, acquired_at_ms=30)
        _seed_membership(
            tmp_path / "source.db",
            raw_id="raw-superseded",
            logical_source_key="cohort-1",
            decision="superseded_equivalent",
        )

        store = IndexGenerationStore.for_archive_root(tmp_path)
        transaction = store.create_transaction(source_snapshot="source-v1")
        first_page = store.next_raw_page(transaction, limit=1)
        assert [row[0] for row in first_page.rows] == ["raw-a"]

        transaction = store.checkpoint_transaction(
            transaction,
            status="paused",
            last_blob_hash_hex=first_page.rows[0][1],
            last_raw_id=first_page.rows[0][0],
            processed_raw_count=1,
        )
        second_page = store.next_raw_page(transaction, limit=1)
        assert [row[0] for row in second_page.rows] == ["raw-b"]


class TestRebuildLeaseStatus:
    """polylogue-b5l.1 AC5: a read-only lease probe for status surfaces --
    must never block, never disturb a genuine holder, and must distinguish
    "not held" / "held by a live process" / "held but recorded pid is dead
    (reclaimable)"."""

    def test_reports_not_held_when_no_lock_file_exists(self, tmp_path: Path) -> None:
        status = rebuild_lease_status(tmp_path)
        assert status.held is False
        assert status.holder_pid is None
        assert status.stale is False

    def test_reports_not_held_after_a_lease_is_released(self, tmp_path: Path) -> None:
        with RebuildLease(tmp_path):
            pass
        status = rebuild_lease_status(tmp_path)
        assert status.held is False
        # The lock file's recorded pid/host from the released lease is still
        # readable (best-effort diagnosis), but "held" reflects reality now.
        assert status.holder_pid == os.getpid()

    def test_reports_held_by_this_process_while_a_lease_is_open(self, tmp_path: Path) -> None:
        with RebuildLease(tmp_path):
            status = rebuild_lease_status(tmp_path)
        assert status.held is True
        assert status.holder_pid == os.getpid()
        assert status.holder_alive is True
        assert status.stale is False

    def test_reports_held_by_a_separate_live_process(self, tmp_path: Path) -> None:
        ready = multiprocessing.Event()
        release = multiprocessing.Event()
        process = multiprocessing.Process(target=_hold_lease, args=(str(tmp_path), ready, release))
        process.start()
        assert ready.wait(5)
        try:
            status = rebuild_lease_status(tmp_path)
            assert status.held is True
            assert status.holder_pid == process.pid
            assert status.holder_alive is True
            assert status.stale is False
        finally:
            release.set()
            process.join(5)
        assert process.exitcode == 0

    def test_reports_stale_when_recorded_holder_pid_is_dead(self, tmp_path: Path) -> None:
        lock_path = tmp_path / ".index-rebuild.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        holder_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(holder_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(holder_fd, f"pid={_DEFINITELY_DEAD_PID} host=nowhere\n".encode())
        os.fsync(holder_fd)
        try:
            status = rebuild_lease_status(tmp_path)
            assert status.held is True
            assert status.holder_pid == _DEFINITELY_DEAD_PID
            assert status.holder_host == "nowhere"
            assert status.holder_alive is False
            assert status.stale is True
        finally:
            fcntl.flock(holder_fd, fcntl.LOCK_UN)
            os.close(holder_fd)

    def test_probe_never_blocks_or_disturbs_a_genuine_holder(self, tmp_path: Path) -> None:
        """Calling the probe repeatedly while a real lease is held must never
        raise, never remove the lock file, and never itself release the
        real holder's lock."""
        with RebuildLease(tmp_path):
            for _ in range(3):
                status = rebuild_lease_status(tmp_path)
                assert status.held is True
            # The real holder must still hold it after repeated probing.
            with pytest.raises(RebuildLeaseUnavailableError):
                with RebuildLease(tmp_path):
                    pass
