from __future__ import annotations

import fcntl
import json
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
    _open_source_snapshot,
    rebuild_lease_status,
    source_revision_snapshot,
)

# A pid guaranteed to never correspond to a running process: it exceeds any
# realistic pid_max (Linux defaults to <= 4194304 even with 64-bit pids).
_DEFINITELY_DEAD_PID = 2**31 - 1
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import (
    DEFAULT_ARCHIVE_PAGE_SIZE,
    initialize_archive_database,
)
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


# Bringing up the child process is not what these tests measure.
#
# Python 3.14 made ``forkserver`` the default start method on Linux, so the
# *first* ``Process.start()`` in a pytest session pays a full fresh-interpreter
# bring-up (measured here at 4.41s on an idle host, free-threaded build) before
# the child body runs at all. A 5s readiness deadline sat inside that cost and
# went red whenever the host was loaded; every later start in the same session
# reuses the running forkserver and returns in milliseconds, which is why only
# the first of these two tests to run used to fail.
#
# The deadline is a liveness bound, not a latency assertion: these tests assert
# that a lease held by a *separate process* excludes this one, and nothing here
# is made weaker by waiting longer for that process to exist. Reaching this
# deadline still fails the test.
#
# Anti-vacuity for the two tests that use these constants, verified by
# reverting: change the exclusive ``fcntl.flock(fd, lock_type | LOCK_NB)`` in
# ``index_generation._open_lock_fd`` to ``LOCK_SH`` and
# ``test_rebuild_lease_excludes_competing_process`` goes red -- no
# ``RebuildLeaseUnavailableError`` is raised while the child holds the lease.
# ``test_reports_held_by_a_separate_live_process`` reddens on the owner-text
# write instead: stop recording ``pid=`` in the lock file and its
# ``holder_pid == process.pid`` assertion fails.
_CHILD_READY_TIMEOUT_S = 60.0
# The child must outlive the parent's inspection of the lock, so its hold is
# bounded by the same budget rather than a shorter one.
_CHILD_HOLD_TIMEOUT_S = 60.0


def _hold_lease(
    root: str, ready: multiprocessing.synchronize.Event, release: multiprocessing.synchronize.Event
) -> None:
    with RebuildLease(Path(root)):
        ready.set()
        release.wait(_CHILD_HOLD_TIMEOUT_S)


def _archive(root: Path) -> None:
    assert _ARCHIVE_TEMPLATE is not None
    clone_archive_template(_ARCHIVE_TEMPLATE, root)


def test_rebuild_lease_excludes_competing_process(tmp_path: Path) -> None:
    ready = multiprocessing.Event()
    release = multiprocessing.Event()
    process = multiprocessing.Process(target=_hold_lease, args=(str(tmp_path), ready, release))
    process.start()
    assert ready.wait(_CHILD_READY_TIMEOUT_S)
    try:
        with pytest.raises(RebuildLeaseUnavailableError):
            with RebuildLease(tmp_path):
                pass
    finally:
        release.set()
        process.join(_CHILD_READY_TIMEOUT_S)
    assert process.exitcode == 0


def test_rebuild_lease_refuses_new_active_writer(tmp_path: Path) -> None:
    with RebuildLease(tmp_path):
        writer = ActiveWriterLease(tmp_path)
        with pytest.raises(RebuildLeaseUnavailableError):
            writer.acquire()


def test_rebuild_lease_does_not_bypass_lock_held_with_dead_pid(tmp_path: Path) -> None:
    """A live kernel lock remains authoritative when its diagnostic pid is stale.

    A forked worker can outlive the owner pid written by its parent.  Replacing
    that inode would let two exclusive writers proceed under different locks.
    """
    lock_path = tmp_path / ".index-rebuild.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    holder_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(holder_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.write(holder_fd, f"pid={_DEFINITELY_DEAD_PID} host=nowhere\n".encode())
    os.fsync(holder_fd)
    try:
        with pytest.raises(RebuildLeaseUnavailableError):
            with RebuildLease(tmp_path):
                pass
    finally:
        fcntl.flock(holder_fd, fcntl.LOCK_UN)
        os.close(holder_fd)


def test_rebuild_lease_does_not_bypass_active_writer_with_stale_owner_text(tmp_path: Path) -> None:
    """A shared daemon-writer lease and an exclusive rebuild never split lock domains."""
    lock_path = tmp_path / ".index-rebuild.lock"
    lock_path.write_text(f"pid={_DEFINITELY_DEAD_PID} host=old-owner\n", encoding="utf-8")
    writer = ActiveWriterLease(tmp_path)
    writer.acquire()
    try:
        with pytest.raises(RebuildLeaseUnavailableError):
            with RebuildLease(tmp_path):
                pass
    finally:
        writer.close()


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


def test_lifecycle_generation_root_symlink_is_rejected(tmp_path: Path) -> None:
    _archive(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / ".index-generations").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink"):
        IndexGenerationStore.for_archive_root(tmp_path)


def test_in_root_generation_alias_cannot_load_another_generation(tmp_path: Path) -> None:
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    generation = store.create(owner_id="operator", source_snapshot="snapshot-a")
    alias = store.generations_root / "gen-alias"
    alias.symlink_to(store.generations_root / generation.generation_id, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink|metadata"):
        store.load("gen-alias")


def test_generation_metadata_is_bound_to_its_identity_and_archive_root(tmp_path: Path) -> None:
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    generation = store.create(owner_id="operator", source_snapshot="snapshot-a")
    metadata_path = store._metadata_path(generation.generation_id)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["generation_id"] = "gen-other"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="generation metadata"):
        store.load(generation.generation_id)

    payload["generation_id"] = generation.generation_id
    payload["archive_root"] = str(tmp_path / "outside")
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="generation metadata"):
        store.load(generation.generation_id)

    payload["archive_root"] = str(tmp_path.resolve())
    payload["state"] = "untrusted"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="generation metadata"):
        store.load(generation.generation_id)


def test_recover_promotion_does_not_clear_other_active_generation(tmp_path: Path) -> None:
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    active = store.create(owner_id="active-owner", source_snapshot="snapshot-a")
    store.promote(active)
    candidate = store.create(owner_id="candidate-owner", source_snapshot="snapshot-b")
    store._write(replace(candidate, state="promoting"))

    recovered = store.recover_promotion(candidate.generation_id)

    assert recovered.state == "promoting"
    assert store.load(active.generation_id).state == "active"


def test_generation_metadata_index_path_cannot_escape_generation_root(tmp_path: Path) -> None:
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    generation = store.create(owner_id="operator", source_snapshot="snapshot-a")
    metadata_path = store._metadata_path(generation.generation_id)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["index_path"] = str(tmp_path / "outside-index.db")
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="generation metadata"):
        store.load(generation.generation_id)


def test_durable_tier_identity_change_during_linking_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _archive(tmp_path)
    source = tmp_path / "source.db"
    replacement = tmp_path / "replacement.db"
    replacement.write_bytes(source.read_bytes())
    original_resolve = Path.resolve
    switched = False

    def hostile_resolve(path: Path, *, strict: bool = False) -> Path:
        nonlocal switched
        if path == source and not switched:
            switched = True
            source.unlink()
            source.symlink_to(replacement)
        return original_resolve(path, strict=strict)

    store = IndexGenerationStore.for_archive_root(tmp_path)
    monkeypatch.setattr(Path, "resolve", hostile_resolve)
    with pytest.raises(RuntimeError, match="changed during identity capture"):
        store.create(owner_id="operator", source_snapshot="snapshot-a")


@pytest.mark.parametrize(
    "failure_target",
    ("initialize", "write"),
)
def test_create_removes_generation_root_when_materialization_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_target: str
) -> None:
    """A failed candidate build cannot leave an unenumerable generation directory.

    Anti-vacuity: each failure is injected after ``root.mkdir``. Removing the
    cleanup around the production create path leaves a ``gen-*`` directory
    without ``generation.json``, which later blocks archive-root relocation.
    """
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    if failure_target == "initialize":
        monkeypatch.setattr(
            "polylogue.storage.index_generation.initialize_archive_database",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("index unavailable")),
        )
    else:
        monkeypatch.setattr(
            store,
            "_write",
            lambda generation: (_ for _ in ()).throw(OSError("metadata unavailable")),
        )

    with pytest.raises(OSError):
        store.create(owner_id="operator", source_snapshot="snapshot-a")

    assert tuple(store.generations_root.glob("gen-*")) == ()


def test_capture_blob_directory_identity_is_stable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _archive(tmp_path)
    blob = tmp_path / "blob"
    blob.mkdir()
    replacement = tmp_path / "blob-replacement"
    replacement.mkdir()
    original_resolve = Path.resolve
    switched = False

    def hostile_resolve(path: Path, *, strict: bool = False) -> Path:
        nonlocal switched
        if path == blob and not switched:
            switched = True
            blob.rmdir()
            blob.symlink_to(replacement, target_is_directory=True)
        return original_resolve(path, strict=strict)

    store = IndexGenerationStore.for_archive_root(tmp_path)
    monkeypatch.setattr(Path, "resolve", hostile_resolve)
    with pytest.raises(RuntimeError, match="changed during identity capture"):
        store.create(owner_id="operator", source_snapshot="snapshot-a")


def test_absent_generation_metadata_preserves_file_not_found(tmp_path: Path) -> None:
    """A never-written generation reads as missing, not as corrupt metadata.

    Anti-vacuity: wrapping ``_read_json_nofollow``'s open in the same
    ``RuntimeError("invalid generation metadata")`` the decode failure raises
    makes an absent generation indistinguishable from a poisoned one.
    """
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)

    with pytest.raises(FileNotFoundError):
        store.load("gen-000000000000-absent0")


def test_metadata_tmp_symlink_is_not_followed(tmp_path: Path) -> None:
    """``_atomic_json_write`` must not write through a planted tmp symlink.

    Anti-vacuity: dropping ``O_EXCL``/``O_NOFOLLOW`` from the tmp open lets
    this overwrite ``external.json``.
    """
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    generation = store.create(owner_id="operator", source_snapshot="snapshot-a")
    path = store._metadata_path(generation.generation_id)
    path.unlink()
    external = tmp_path / "external.json"
    external.write_text("untouched", encoding="utf-8")
    path.with_suffix(".json.tmp").symlink_to(external)

    with pytest.raises((FileExistsError, RuntimeError, OSError)):
        store._write(generation)
    assert external.read_text(encoding="utf-8") == "untouched"


def test_check_to_use_replacement_cannot_redirect_metadata_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tmp file swapped for a symlink between write and replace is refused.

    Anti-vacuity: removing the post-write ``lstat`` re-check from
    ``_atomic_json_write`` lets the raced symlink become the metadata path.
    """
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    generation = store.create(owner_id="operator", source_snapshot="snapshot-a")
    path = store._metadata_path(generation.generation_id)
    external = tmp_path / "external.json"
    external.write_text("untouched", encoding="utf-8")
    real_replace = os.replace

    def replace_with_race(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        Path(source).unlink()
        Path(source).symlink_to(external)
        real_replace(source, target)

    monkeypatch.setattr("polylogue.storage.index_generation.os.replace", replace_with_race)
    with pytest.raises(RuntimeError, match="symlink"):
        store._write(generation)

    assert external.read_text(encoding="utf-8") == "untouched"
    assert not path.is_symlink()


def test_retention_receipt_payload_is_bound_to_requested_generation(tmp_path: Path) -> None:
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    generation = store.create(owner_id="operator", source_snapshot="snapshot-a")
    store.promote(generation)
    receipt_path = store._retention_receipt_path(generation.generation_id)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["promoted_generation_id"] = "gen-not-requested"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="retention receipt"):
        store.load_retention_receipt(generation.generation_id)


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


def test_stale_owner_cannot_promote(tmp_path: Path) -> None:
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
    calls: list[tuple[Path, str, Path]] = []
    monkeypatch.setattr(
        "polylogue.storage.index_generation._checkpoint_truncate",
        lambda path, *, label, archive_root: calls.append((path, label, archive_root)),
    )

    store.promote(generation)

    # Each checkpoint names the archive it is promoting into, which is what
    # makes its own ownership assertion archive-bound (polylogue-8qm4k AC1).
    assert calls == [
        (Path(generation.index_path).resolve(), "new index", store.archive_root),
        (tmp_path / "index.db", "active index", store.archive_root),
    ]


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


def test_source_snapshot_connection_stays_bound_across_path_replacement(tmp_path: Path) -> None:
    """A replacement after admission must not change the snapshot database."""
    _archive(tmp_path)
    source = tmp_path / "source.db"
    replacement = tmp_path / "replacement.db"
    with sqlite3.connect(replacement) as conn:
        conn.execute("CREATE TABLE raw_sessions (raw_id TEXT)")
        conn.execute("INSERT INTO raw_sessions VALUES ('replacement')")
    with _open_source_snapshot(tmp_path) as conn:
        original = conn.execute("SELECT count(*) FROM raw_sessions").fetchone()[0]
        source.replace(tmp_path / "source-original.db")
        replacement.replace(source)
        assert conn.execute("SELECT count(*) FROM raw_sessions").fetchone()[0] == original


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
    """An in-flight cold-build candidate is `inactive` -- never promoted --
    and must survive an unrelated promotion's housekeeping.

    Treating every non-active generation as superseded let a promotion delete a
    build in progress, and let a newer inactive candidate consume the single
    retained slot so the real rollback target was pruned instead. Never-promoted
    candidates belong to ``discard_if_inactive``, driven by their owner.
    """
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)

    first = store.create(owner_id="operator", source_snapshot="snapshot-a")
    store.promote(first)
    # A cold-build candidate, created but never promoted.
    candidate = store.create(owner_id="cold-build", source_snapshot="snapshot-candidate")
    second = store.create(owner_id="operator", source_snapshot="snapshot-b")
    store.promote(second)

    assert store.load(candidate.generation_id).state == "inactive"
    assert Path(candidate.index_path).exists(), "an unrelated promotion deleted a live build candidate"
    # The genuine rollback target -- the previously-active generation -- is what
    # the retained slot is for, not the inactive candidate.
    assert Path(first.index_path).exists()


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
        assert ready.wait(_CHILD_READY_TIMEOUT_S)
        try:
            status = rebuild_lease_status(tmp_path)
            assert status.held is True
            assert status.holder_pid == process.pid
            assert status.holder_alive is True
            assert status.stale is False
        finally:
            release.set()
            process.join(_CHILD_READY_TIMEOUT_S)
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


def test_source_snapshot_opens_through_the_validated_descriptor_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Descriptor opens use ``descriptor_alias_path``, not a hardcoded /proc/self/fd.

    ``descriptor_alias_path`` probes /dev/fd as well as /proc/self/fd and
    validates that the alias resolves to the same inode as the descriptor.
    Anti-vacuity: with the literal ``f"file:/proc/self/fd/{fd}?mode=ro"``
    restored, the recorder below is never called and the first assertion is
    red; making the helper return None must also refuse rather than fall back,
    which the second half asserts.
    """
    import polylogue.storage.index_generation as index_generation_module

    _archive(tmp_path)

    calls: list[int] = []
    from polylogue.storage.sqlite.connection_profile import descriptor_alias_path as real_alias

    def _recording_alias(fd: int) -> Path | None:
        calls.append(fd)
        return real_alias(fd)

    monkeypatch.setattr(index_generation_module, "descriptor_alias_path", _recording_alias)
    assert source_revision_snapshot(tmp_path)
    assert calls

    monkeypatch.setattr(index_generation_module, "descriptor_alias_path", lambda fd: None)
    with pytest.raises(RuntimeError, match="no validated descriptor alias"):
        source_revision_snapshot(tmp_path)


def _index_page_size(path: Path) -> int:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        return int(conn.execute("PRAGMA page_size").fetchone()[0])


def test_generation_index_is_created_at_the_declared_page_size_and_records_it(tmp_path: Path) -> None:
    """polylogue-kc8eq (4): page_size is a creation-time choice, so the
    generation is the only place that can make it, and the only place that can
    remember what was made.

    Anti-vacuity: before this, ``initialize_archive_database`` opened the file
    with a bare ``sqlite3.connect`` and never issued the pragma, so every
    generation silently got SQLite's compiled-in 4096. Dropping the
    ``page_size=`` argument from ``IndexGenerationStore.create`` makes the
    first assertion red; dropping the recorded field makes the second red.
    """
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)

    generation = store.create(owner_id="operator", source_snapshot="snapshot-a")

    assert _index_page_size(Path(generation.index_path)) == DEFAULT_ARCHIVE_PAGE_SIZE
    assert generation.page_size == DEFAULT_ARCHIVE_PAGE_SIZE
    assert store.load(generation.generation_id).page_size == DEFAULT_ARCHIVE_PAGE_SIZE


def test_two_page_sizes_in_one_process_do_not_share_a_tier_prototype(tmp_path: Path) -> None:
    """The prototype cache page-copies an empty tier with the SQLite backup
    API, which makes an empty destination adopt the *source's* page size. A
    cache keyed without page size would hand the second caller a database at
    the first caller's page size and report success.

    Anti-vacuity: removing ``page_size`` from ``_tier_prototype_key`` makes the
    second assertion red (both generations come back at 4096).
    """
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)

    small = store.create(owner_id="operator", source_snapshot="snapshot-a", page_size=4096)
    large = store.create(owner_id="operator", source_snapshot="snapshot-b", page_size=16384)

    assert _index_page_size(Path(small.index_path)) == 4096
    assert _index_page_size(Path(large.index_path)) == 16384


def test_page_size_refuses_a_value_sqlite_would_silently_ignore(tmp_path: Path) -> None:
    """``PRAGMA page_size`` accepts a non-power-of-two and then ignores it, so
    the refusal has to be ours.

    Anti-vacuity: deleting the ``_VALID_PAGE_SIZES`` check makes this green-by-
    accident case (a 4096 database claiming 5000) pass silently.
    """
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)

    with pytest.raises(ValueError, match="page_size"):
        store.create(owner_id="operator", source_snapshot="snapshot-a", page_size=5000)


def test_page_size_cannot_be_applied_to_an_existing_tier(tmp_path: Path) -> None:
    """A durable tier is opened, not created; asking for a page size there is a
    caller error, not a silent no-op.

    Anti-vacuity: dropping the ``allow_create=False`` guard lets the call
    succeed while changing nothing.
    """
    _archive(tmp_path)

    with pytest.raises(ValueError, match="creation-time"):
        initialize_archive_database(
            tmp_path / "source.db",
            ArchiveTier.SOURCE,
            allow_create=False,
            page_size=8192,
        )
