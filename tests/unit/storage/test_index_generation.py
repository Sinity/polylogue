from __future__ import annotations

import fcntl
import json
import multiprocessing
import os
import sqlite3
from builtins import BaseExceptionGroup
from collections.abc import Callable, Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.sources.parsers.base import ParsedSession
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
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.sqlite_cursor_settlement import (
    native_settlement_connections,  # noqa: F401  # Pytest fixture discovery.
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
from tests.infra.archive_custody_probe import archive_custody_available
from tests.infra.archive_templates import bootstrap_archive_root, clone_archive_template, finalize_archive_template
from tests.infra.index_writer import close_fixture_index_connection

_ARCHIVE_TEMPLATE: Path | None = None


@pytest.fixture(scope="module", autouse=True)
def _archive_template(tmp_path_factory: pytest.TempPathFactory) -> Generator[None]:
    """Build the canonical six-tier fixture once, then clone per test."""
    global _ARCHIVE_TEMPLATE
    template = tmp_path_factory.mktemp("index-generation-template") / "archive"
    bootstrap_archive_root(template)
    finalize_archive_template(template)
    _ARCHIVE_TEMPLATE = template
    try:
        yield
    finally:
        _ARCHIVE_TEMPLATE = None


@pytest.fixture(autouse=True)
def _state_home_outside_archive(
    _clear_polylogue_env: None, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep XDG state beside, not inside, the cloned archive root.

    These laws clone the archive into ``tmp_path`` itself, where the shared
    environment fixture also places XDG state. Population records backup
    attestation keys under XDG state, which the clone's file-set proof would
    otherwise count as an unowned archive file.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path_factory.mktemp("index-generation-state")))


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


def _write_prepared_session(archive: ArchiveStore, session: ParsedSession) -> str:
    """Prepare before the archive's Index transaction scope, then publish inside it."""
    from polylogue.storage.sqlite.archive_tiers.write import prepare_session_write
    from tests.infra.index_writer import write_fixture_index_session

    prepared = prepare_session_write(archive._conn, session, merge_append=False)
    with archive.index_mutation_scope():
        return write_fixture_index_session(
            archive._conn, session, prepared_write=prepared, content_hash=prepared.input_content_hash.hex()
        )


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


def test_create_failure_does_not_remove_replacement_generation_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup is bound to the created inode, even if its pathname is replaced.

    Anti-vacuity: pathname-only ``rmtree(root)`` deletes ``replacement`` after
    initialization moves the original directory aside and installs a new one.
    """
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    moved: list[Path] = []

    def replace_then_fail(_path: Path, *args: object, **kwargs: object) -> None:
        root = store.generations_root / next(p.name for p in store.generations_root.glob("gen-*"))
        destination = root.with_name(root.name + "-moved")
        root.rename(destination)
        root.mkdir()
        moved.append(destination)
        raise OSError("injected initialization failure")

    monkeypatch.setattr("polylogue.storage.index_generation.initialize_archive_database", replace_then_fail)
    with pytest.raises(OSError, match="injected"):
        store.create(owner_id="operator", source_snapshot="snapshot-a")

    assert len(moved) == 1 and moved[0].is_dir()
    replacements = tuple(path for path in store.generations_root.glob("gen-*") if path.name.endswith("-moved") is False)
    assert len(replacements) == 1 and replacements[0].is_dir()


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


@pytest.mark.parametrize("producer", ["metadata", "rollback", "pointer_proof", "retention_receipt"])
def test_promotion_preserves_regular_interrupted_temporary(tmp_path: Path, producer: str) -> None:
    """A prior write's private bytes do not own the next promotion attempt."""
    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    generation = store.create(owner_id="operator", source_snapshot="snapshot-a")
    target = {
        "metadata": store._metadata_path(generation.generation_id),
        "rollback": store._metadata_path(generation.generation_id).with_name("generation.rollback.json"),
        "pointer_proof": store._rollback_pointer_proof_path(generation.generation_id),
        "retention_receipt": store._retention_receipt_path(generation.generation_id),
    }[producer]
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(b"neutral interrupted lifecycle write")
    identity = temporary.stat()

    promoted = store.promote(generation)

    assert promoted.state == "active"
    assert (tmp_path / "index.db").resolve() == Path(generation.index_path).resolve()
    assert temporary.read_bytes() == b"neutral interrupted lifecycle write"
    assert (temporary.stat().st_dev, temporary.stat().st_ino) == (identity.st_dev, identity.st_ino)


def test_pointer_anchor_preserves_regular_interrupted_temporary(tmp_path: Path) -> None:
    _archive(tmp_path)
    temporary = tmp_path / ".index-active-pointer.tmp"
    temporary.write_bytes(b"neutral interrupted anchor write")
    identity = temporary.stat()

    store = IndexGenerationStore.for_archive_root(tmp_path)

    assert store.active_pointer == tmp_path / "index.db"
    assert (tmp_path / ".index-active-pointer").read_text() == str((tmp_path / "index.db").absolute())
    assert temporary.read_bytes() == b"neutral interrupted anchor write"
    assert temporary.stat().st_ino == identity.st_ino


def test_pointer_anchor_refuses_interrupted_temporary_symlink(tmp_path: Path) -> None:
    _archive(tmp_path)
    external = tmp_path / "external-anchor"
    external.write_text("untouched")
    (tmp_path / ".index-active-pointer.tmp").symlink_to(external)

    with pytest.raises(RuntimeError, match="symlink"):
        IndexGenerationStore.for_archive_root(tmp_path)

    assert external.read_text() == "untouched"


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


def _configured_symlink_archive(tmp_path: Path) -> tuple[Path, Path]:
    configured = tmp_path / "configured"
    canonical = tmp_path / "canonical"
    configured.mkdir()
    canonical.mkdir()
    _archive(canonical)
    for tier in ArchiveTier:
        (configured / f"{tier.value}.db").symlink_to(canonical / f"{tier.value}.db")
    return configured, canonical


def test_symlinked_configured_index_promotes_canonical_target(tmp_path: Path) -> None:
    configured, canonical = _configured_symlink_archive(tmp_path)

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
    with pytest.raises(RuntimeError):
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


@pytest.mark.parametrize("anchor", ["assertion", "annotation_prompt", "audit_preview"])
def test_promotion_refuses_candidate_that_orphans_a_resolved_durable_message_ref(tmp_path: Path, anchor: str) -> None:
    """Promotion preserves refs resolved by the previous active generation."""
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, Provider
    from polylogue.core.refs import ObjectRef
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.reference_seal import ReferenceSealError
    from polylogue.storage.sqlite.write_lease import write_lease
    from tests.infra.index_writer import write_fixture_index_session

    _archive(tmp_path)
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="durable-ref-target",
        title="durable-ref-target",
        messages=[
            ParsedMessage(
                provider_message_id="message-one",
                role=Role.USER,
                text="retained content",
                position=0,
                variant_index=0,
                is_active_path=True,
                is_active_leaf=True,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="retained content")],
            )
        ],
    )
    with write_lease("test.seed-promotion-reference", archive_root=tmp_path):
        # The fixture producer captures only a handle from the measured factory.
        conn = connect_measured(tmp_path / "index.db")
        conn.row_factory = sqlite3.Row
        try:
            session_id = write_fixture_index_session(conn, session)
            message_id = str(
                conn.execute("SELECT message_id FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0]
            )
            conn.commit()
        finally:
            close_fixture_index_connection(conn)
        target_ref = ObjectRef("message", message_id).format()
        if anchor == "assertion":
            from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

            with ArchiveStore(tmp_path, initialize=False) as archive:
                archive.save_annotation(
                    "annotation-preserve-message",
                    "message",
                    message_id,
                    "Synthetic retained note",
                    owner_session_id=session_id,
                )
                archive.commit()
        elif anchor == "annotation_prompt":
            from polylogue.annotations.batch import AnnotationBatch
            from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

            with ArchiveStore(tmp_path, initialize=False) as archive:
                archive.save_annotation_batch(
                    AnnotationBatch(
                        batch_id="annotation-preserve-prompt",
                        schema_id="delegation.discourse",
                        schema_version=1,
                        target_ref="delegation:neutral",
                        source_result_ref="result-set:neutral",
                        actor_ref="agent:neutral",
                        model_ref="agent:neutral",
                        prompt_ref=target_ref,
                        total_count=0,
                        valid_count=0,
                        invalid_count=0,
                        abstained_count=0,
                    )
                )
                archive.commit()
        else:
            with sqlite3.connect(tmp_path / "audit.db") as audit:
                audit.execute(
                    "INSERT INTO operation_previews(preview_id, operation_name, operation_version, "
                    "archive_instance_id, archive_identity_digest, plan_hash, parameter_digest, target_digest, "
                    "target_count, destructive_class, required_confirmation, required_capability_count, "
                    "principal_actor_ref, principal_surface, state, created_at_ms, expires_at_ms, plan_format, plan_json) "
                    "VALUES ('preview-reference', 'test.reference', 1, 'neutral', 'neutral', 'neutral', 'neutral', "
                    "'neutral', 1, 'additive', 'role_only', 0, 'user:local', 'internal', 'prepared', 0, 1, 'polylogue.mutation-plan/v1', '{}')"
                )
                audit.execute(
                    "INSERT INTO operation_preview_targets VALUES ('preview-reference', 0, 'message', ?, "
                    "'neutral', 'neutral', 'derived', 'rebuild')",
                    (target_ref,),
                )

    store = IndexGenerationStore.for_archive_root(tmp_path)
    active_before = Path(store.active_pointer).resolve(strict=True)
    candidate = store.create(owner_id="candidate-owner", source_snapshot="snapshot-b")
    with pytest.raises(ReferenceSealError, match="promotion would orphan"):
        store.prepare_promotion(candidate)

    assert Path(store.active_pointer).resolve(strict=True) == active_before
    assert store.load(candidate.generation_id).state == "inactive"

    preserving_candidate = store.create(owner_id="preserving-owner", source_snapshot="snapshot-c")
    with ArchiveStore.open_owned_inactive_generation(
        Path(preserving_candidate.index_path).parent,
        generation_id=preserving_candidate.generation_id,
        owner_id=preserving_candidate.owner_id,
    ) as candidate_archive:
        _write_prepared_session(candidate_archive, session)
    with store.prepare_promotion(preserving_candidate) as prepared:
        with write_lease("test.promote-preserving-candidate", archive_root=tmp_path):
            promoted = store.promote(preserving_candidate, prepared)
    assert promoted.state == "active"
    assert Path(store.active_pointer).resolve(strict=True) == Path(preserving_candidate.index_path).resolve()


def test_promotion_candidate_change_after_off_gate_proof_is_refused(tmp_path: Path) -> None:
    """The retained candidate observer detects writes before pointer admission."""
    from polylogue.storage.sqlite.reference_seal import ReferenceSealStaleError
    from polylogue.storage.sqlite.write_lease import write_lease

    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    active_before = Path(store.active_pointer).resolve(strict=True)
    candidate = store.create(owner_id="candidate-owner", source_snapshot="snapshot-b")
    with store.prepare_promotion(candidate) as prepared:
        changed = sqlite3.connect(candidate.index_path)
        try:
            changed.execute("CREATE TABLE promotion_race(value TEXT NOT NULL)")
            changed.commit()
        finally:
            changed.close()

        with write_lease("test.promote-changed-candidate", archive_root=tmp_path):
            with pytest.raises(ReferenceSealStaleError, match="candidate incarnation changed"):
                store.promote(candidate, prepared)

    assert Path(store.active_pointer).resolve(strict=True) == active_before
    assert store.load(candidate.generation_id).state == "inactive"


def test_promotion_preserves_same_composed_session_evidence_ref(tmp_path: Path) -> None:
    """A typed EvidenceRef remains reachable through its exact composed scope."""
    import json

    from polylogue.archive.message.roles import Role
    from polylogue.archive.session.branch_type import BranchType
    from polylogue.core.enums import BlockType, Provider
    from polylogue.core.refs import EvidenceRef, ObjectRef
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.write import read_archive_session_envelope
    from polylogue.storage.sqlite.reference_seal import ReferenceSealError
    from tests.infra.index_writer import write_fixture_index_session

    def parent_and_child(child_id: str) -> tuple[ParsedSession, ParsedSession]:
        parent = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="composed-parent",
            title="composed-parent",
            messages=[
                ParsedMessage(
                    provider_message_id="prefix",
                    role=Role.USER,
                    text="prefix",
                    position=0,
                    variant_index=0,
                    is_active_path=True,
                    is_active_leaf=False,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="prefix")],
                )
            ],
        )
        child = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id=child_id,
            title=child_id,
            parent_session_provider_id="composed-parent",
            branch_type=BranchType.FORK,
            messages=[
                ParsedMessage(
                    provider_message_id="prefix",
                    role=Role.USER,
                    text="prefix",
                    position=0,
                    variant_index=0,
                    is_active_path=True,
                    is_active_leaf=False,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="prefix")],
                ),
                ParsedMessage(
                    provider_message_id="child-tail",
                    role=Role.ASSISTANT,
                    text="tail",
                    position=1,
                    variant_index=0,
                    is_active_path=True,
                    is_active_leaf=True,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="tail")],
                ),
            ],
        )
        return parent, child

    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    active = connect_measured(tmp_path / "index.db")
    active.row_factory = sqlite3.Row
    try:
        parent, child = parent_and_child("composed-child")
        with write_lease("test.seed-composed-reference", archive_root=tmp_path):
            parent_id = write_fixture_index_session(active, parent)
            child_id = write_fixture_index_session(active, child)
            parent_message = active.execute(
                "SELECT message_id FROM messages WHERE session_id = ?", (parent_id,)
            ).fetchone()
            composed = read_archive_session_envelope(active, child_id)
            assert str(parent_message[0]) in {message.message_id for message in composed.messages}
            # This parent-owned message is visible only through the child's
            # prefix-sharing composed scope.
            evidence = EvidenceRef(session_id=child_id, message_id=str(parent_message[0])).format()
            active.commit()
            user = sqlite3.connect(tmp_path / "user.db")
            try:
                user.execute(
                    "INSERT INTO assertions(assertion_id, target_ref, evidence_refs_json, kind, "
                    "created_at_ms, updated_at_ms) VALUES (?, ?, ?, 'fact', 0, 0)",
                    (
                        "assertion-composed-evidence",
                        ObjectRef("message", str(parent_message[0])).format(),
                        json.dumps([evidence]),
                    ),
                )
                user.commit()
            finally:
                user.close()
    finally:
        close_fixture_index_connection(active)

    candidate = store.create(owner_id="candidate-owner", source_snapshot="snapshot-b")
    with write_lease("test.seed-composed-candidate", archive_root=tmp_path):
        with ArchiveStore.open_owned_inactive_generation(
            Path(candidate.index_path).parent,
            generation_id=candidate.generation_id,
            owner_id=candidate.owner_id,
        ) as candidate_archive:
            # The candidate has the same physical target row, but the child has no
            # composed prefix edge. A global message-id-only guard would accept it.
            child_without_parent = ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="composed-child",
                title="composed-child",
                messages=[
                    ParsedMessage(
                        provider_message_id="child-tail",
                        role=Role.ASSISTANT,
                        text="tail",
                        position=1,
                        variant_index=0,
                        is_active_path=True,
                        is_active_leaf=True,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="tail")],
                    )
                ],
            )
            # Parent's target is present in the candidate, but the child's scoped
            # EvidenceRef must remain composed through the child-parent edge.
            # Each session prepares before the generation's Index transaction scope.
            _write_prepared_session(candidate_archive, parent)
            _write_prepared_session(candidate_archive, child_without_parent)
    with pytest.raises(ReferenceSealError, match="promotion would orphan"):
        store.prepare_promotion(candidate)

    preserving = store.create(owner_id="preserving-owner", source_snapshot="snapshot-c")
    with write_lease("test.seed-preserving-composed-candidate", archive_root=tmp_path):
        with ArchiveStore.open_owned_inactive_generation(
            Path(preserving.index_path).parent,
            generation_id=preserving.generation_id,
            owner_id=preserving.owner_id,
        ) as preserving_archive:
            parent, child = parent_and_child("composed-child")
            _write_prepared_session(preserving_archive, parent)
            _write_prepared_session(preserving_archive, child)
    with store.prepare_promotion(preserving) as prepared:
        with write_lease("test.promote-composed-evidence", archive_root=tmp_path):
            promoted = store.promote(preserving, prepared)
    assert promoted.state == "active"


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


@pytest.mark.uses_real_clock("independent process probes custody between actual rebuild commits")
def test_rebuild_preparation_is_off_custody_between_committed_segments(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.connection_profile import open_isolated_write_connection
    from polylogue.storage.sqlite.write_lease import current_write_lease

    _archive(tmp_path)
    with RebuildLease(tmp_path) as rebuild:
        for _segment in range(2):
            assert current_write_lease() is None
            assert archive_custody_available(tmp_path)
            with rebuild.write_segment("test.rebuild.commit"):
                assert not archive_custody_available(tmp_path)
                connection = open_isolated_write_connection(
                    tmp_path / "index.db", purpose="test.rebuild.commit", archive_root=tmp_path
                )
                try:
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute("UPDATE sessions SET title = title WHERE 0")
                    assert connection.in_transaction
                    connection.commit()
                    assert not connection.in_transaction
                finally:
                    connection.close()
            assert current_write_lease() is None
            assert archive_custody_available(tmp_path)
        writer = ActiveWriterLease(tmp_path)
        with pytest.raises(RebuildLeaseUnavailableError):
            writer.acquire()
    writer.acquire()
    writer.close()


def _refused_rebuild_keeps_process_alive(
    root: str,
    refused: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
) -> None:
    try:
        with RebuildLease(Path(root)):
            raise AssertionError("the existing Store SH owner must exclude rebuild")
    except RebuildLeaseUnavailableError:
        refused.set()
        release.wait(_CHILD_HOLD_TIMEOUT_S)


@pytest.mark.uses_real_clock("two processes exercise SH ownership and prompt EX refusal")
def test_refused_rebuild_releases_custody_while_existing_sh_owner_remains(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.write_lease import write_lease

    writer = ActiveWriterLease(tmp_path)
    writer.acquire()
    refused = multiprocessing.Event()
    release = multiprocessing.Event()
    process = multiprocessing.Process(
        target=_refused_rebuild_keeps_process_alive, args=(str(tmp_path), refused, release)
    )
    process.start()
    try:
        assert refused.wait(_CHILD_READY_TIMEOUT_S)
        assert process.is_alive()
        with write_lease("test.existing-store.after-rebuild-refusal", archive_root=tmp_path):
            assert process.is_alive()
    finally:
        release.set()
        process.join(_CHILD_READY_TIMEOUT_S)
        if process.is_alive():
            process.terminate()
            process.join(_CHILD_READY_TIMEOUT_S)
        writer.close()
    assert process.exitcode == 0


@pytest.mark.uses_real_clock("independent process probes retained SQL after outer owner retires")
def test_outer_lease_retirement_keeps_store_sql_until_actual_commit(tmp_path: Path) -> None:
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.write_lease import UnleasedWriteError, current_write_lease, write_lease

    _archive(tmp_path)
    archive = ArchiveStore(tmp_path, initialize=False)
    try:
        with write_lease("test.store.outer", archive_root=tmp_path):
            archive._enter_mutation_lease()
            archive._conn.execute("BEGIN IMMEDIATE")
            archive._conn.execute("UPDATE sessions SET title = title WHERE 0")
        assert current_write_lease() is None
        assert archive._conn.in_transaction
        assert not archive_custody_available(tmp_path)
        with pytest.raises(UnleasedWriteError):
            archive.write_raw_payload(
                provider=Provider.CLAUDE_CODE,
                payload=b"{}",
                source_path="synthetic/retired-owner.jsonl",
                canonical_source_path="synthetic/retired-owner.jsonl",
                acquired_at_ms=1,
            )
        archive.commit()
        assert not archive._conn.in_transaction
        assert current_write_lease() is None
        assert archive_custody_available(tmp_path)
    finally:
        archive.close()


@pytest.mark.parametrize("pending", ["idle", "transaction", "failed_close"])
def test_promotion_settles_operation_cache_before_artifact_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pending: str,
) -> None:
    from polylogue.schemas.validation import artifacts
    from polylogue.schemas.validation.requests import ArtifactObservationQuery
    from polylogue.storage.sqlite import connection as cached
    from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError
    from polylogue.storage.sqlite.write_lease import write_lease
    from tests.infra.sqlite_cursor_settlement import SettlementConnection, arm_settlement

    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    generation = store.create(owner_id="operator", source_snapshot="snapshot-cache")
    from tests.infra.reference_sessions import reference_session

    with write_lease("test.seed-promotion-cache", archive_root=tmp_path):
        for inactive, marker in ((False, "old"), (True, "new")):
            archive = (
                ArchiveStore.open_owned_inactive_generation(
                    Path(generation.index_path).parent,
                    generation_id=generation.generation_id,
                    owner_id=generation.owner_id,
                )
                if inactive
                else ArchiveStore.open_existing(tmp_path, read_only=False)
            )
            with archive:
                _write_prepared_session(archive, reference_session(marker))
    seen: list[str] = []
    actual_materialize = cast(
        Callable[[sqlite3.Connection], object], vars(artifacts)["materialize_artifact_observations"]
    )

    def materialize(connection: sqlite3.Connection) -> object:
        # Hydration runs on the Source tier writer (581ba8b05e), never on a
        # cached Index handle the promotion must already have settled.
        seen.append(Path(next(row[2] for row in connection.execute("PRAGMA database_list") if row[1] == "main")).name)
        return actual_materialize(connection)

    monkeypatch.setattr(artifacts, "materialize_artifact_observations", materialize)
    with store.prepare_promotion(generation) as prepared:
        with write_lease("test.promotion_cache", archive_root=tmp_path):
            with cached.connection_context(tmp_path / "index.db") as old:
                assert old.execute("SELECT title FROM sessions").fetchone()[0] == "old"
            cache = cached._connection_cache.conns
            owner = cache[str(tmp_path / "index.db")]
            try:
                handle: SettlementConnection | None = None
                if pending == "transaction":
                    old.execute("BEGIN")
                    old.execute("SELECT * FROM sessions").fetchall()
                elif pending == "failed_close":
                    handle = arm_settlement(old)
                if pending != "idle":
                    with pytest.raises(NativeConnectionSettlementError):
                        store.promote(generation, prepared)
                    assert not (tmp_path / "index.db").is_symlink()
                    assert owner.connection is not None
                    assert not archive_custody_available(tmp_path)
                    if handle is not None:
                        handle.allow_cleanup.set()
                        owner.close()
                    else:
                        old.rollback()
                store.promote(generation, prepared)
                with pytest.raises(sqlite3.ProgrammingError):
                    old.execute("SELECT 1")
                assert (
                    artifacts.list_artifact_observation_rows(
                        db_path=tmp_path / "index.db",
                        request=ArtifactObservationQuery(),
                    )
                    == []
                )
                assert (
                    artifacts.list_artifact_cohort_rows(
                        db_path=tmp_path / "index.db",
                        request=ArtifactObservationQuery(),
                    )
                    == []
                )
                assert seen == ["source.db", "source.db"]
            finally:
                if isinstance(old, SettlementConnection):
                    old.allow_cleanup.set()
                owner.close()
    assert archive_custody_available(tmp_path)


@pytest.mark.parametrize("read_only", [True, False])
def test_async_promoted_index_keeps_configured_durable_siblings(tmp_path: Path, read_only: bool) -> None:
    import asyncio

    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    generation = store.create(owner_id="operator", source_snapshot="snapshot-async-siblings")
    store.promote(generation)
    backend = SQLiteBackend(tmp_path / "index.db")

    async def observe() -> None:
        try:
            context = backend.read_connection() if read_only else backend.connection()
            async with context as connection:
                cursor = await connection.execute("PRAGMA database_list")
                databases = {str(row[1]): Path(row[2]).resolve() for row in await cursor.fetchall()}
                assert databases["main"] == Path(generation.index_path).resolve()
                for alias, filename in (
                    ("source_tier", "source.db"),
                    ("user_tier", "user.db"),
                    ("ops_tier", "ops.db"),
                ):
                    assert databases[alias] == (tmp_path / filename).resolve()
                cursor = await connection.execute("SELECT COUNT(*) FROM source_tier.raw_sessions")
                row = await cursor.fetchone()
                assert row is not None and row[0] == 0
        finally:
            await backend.close()

    asyncio.run(observe())
    assert archive_custody_available(tmp_path)


def test_explicit_offline_generation_uses_owned_archive_siblings(tmp_path: Path) -> None:
    from polylogue.storage.sqlite import connection as cached
    from polylogue.storage.sqlite.connection_profile import open_connection
    from polylogue.storage.sqlite.write_lease import write_lease

    _archive(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    generation = store.create(owner_id="operator", source_snapshot="snapshot-offline-siblings")
    with write_lease("test.offline_generation", archive_root=tmp_path):
        connection = open_connection(generation.index_path, archive_root=tmp_path)
        try:
            assert connection.execute("SELECT COUNT(*) FROM source_tier.raw_sessions").fetchone()[0] == 0
        finally:
            connection.close()
        with cached.connection_context(Path(generation.index_path), archive_root=tmp_path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM source_tier.raw_sessions").fetchone()[0] == 0
    assert (Path(generation.index_path).parent / "source.db").is_symlink()
    assert (Path(generation.index_path).parent / "source.db").resolve() == (tmp_path / "source.db").resolve()
    assert archive_custody_available(tmp_path)


@pytest.mark.parametrize("route", ["checkpoint", "source_snapshot"])
def test_generation_native_failed_close_retains_selected_descriptor_and_sql(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
) -> None:
    from polylogue.storage import index_generation as generations
    from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError
    from polylogue.storage.sqlite.write_lease import write_lease
    from tests.infra.sqlite_cursor_settlement import SettlementConnection, arm_settlement

    _archive(tmp_path)
    handles: list[SettlementConnection] = []
    actual_connect = cast(Callable[..., sqlite3.Connection], sqlite3.connect)

    def connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        handle = arm_settlement(actual_connect(*args, **kwargs))
        handles.append(handle)
        return handle

    monkeypatch.setattr(sqlite3, "connect", connect)
    owner = None
    try:
        with write_lease("test.generation_anchor", archive_root=tmp_path):
            with pytest.raises(NativeConnectionSettlementError) as refused:
                if route == "checkpoint":
                    generations._checkpoint_truncate(tmp_path / "index.db", label="test index", archive_root=tmp_path)
                else:
                    with generations._open_source_snapshot(tmp_path) as connection:
                        assert connection.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == 0
            owner = refused.value.owner
            assert len(owner.anchored_descriptors) == 1
            metadata = os.fstat(owner.anchored_descriptors[0])
            selected = (tmp_path / ("index.db" if route == "checkpoint" else "source.db")).stat()
            assert (metadata.st_dev, metadata.st_ino) == (selected.st_dev, selected.st_ino)
        assert not archive_custody_available(tmp_path)
        assert owner.connection is not None
        for handle in handles:
            handle.allow_cleanup.set()
        owner.close()
        assert not owner.anchored_descriptors
        assert owner.connection is None
        assert archive_custody_available(tmp_path)
    finally:
        for handle in handles:
            handle.allow_cleanup.set()
        if owner is not None:
            owner.close()


def test_source_snapshot_failed_construction_exposes_ambiguous_descriptor_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage import index_generation as generations
    from polylogue.storage.sqlite import connection_profile as profiles
    from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError
    from tests.infra.descriptor_close_fault import DescriptorCloseFault

    _archive(tmp_path)
    monkeypatch.setattr(generations, "descriptor_alias_path", lambda descriptor: None)
    fault = DescriptorCloseFault(lambda descriptor: True)
    monkeypatch.setattr(profiles, "os", fault)
    with pytest.raises(NativeConnectionSettlementError) as refused:
        with generations._open_source_snapshot(tmp_path):
            raise AssertionError("failed admission cannot yield a source connection")
    owner = refused.value.owner
    assert isinstance(refused.value.__cause__, RuntimeError)
    assert owner.connection is None
    assert len(owner.anchored_descriptors) == 1
    descriptor = owner.anchored_descriptors[0]
    try:
        assert os.fstat(descriptor).st_ino == (tmp_path / "source.db").stat().st_ino
        with pytest.raises(NativeConnectionSettlementError):
            owner.close()
        assert fault.attempts == [descriptor]
    finally:
        os.close(descriptor)
        owner.close()


@pytest.mark.parametrize("failure", ["orphan", "enumeration", "cancelled", "constructor"])
def test_promotion_preparation_retains_primary_and_one_native_cleanup_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    import asyncio

    from polylogue.storage.sqlite import reference_seal
    from polylogue.storage.sqlite.connection_profile import (
        NativeConnectionSettlementError,
        native_sql_children,
        retained_native_settlement_owners_on_current_thread,
    )
    from tests.infra.archive_templates import bootstrap_archive_root
    from tests.infra.reference_sessions import reference_session
    from tests.infra.sqlite_cursor_settlement import ControlledCursor

    with write_lease("test.promotion-native-proof", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            session_id = _write_prepared_session(archive, reference_session("proof-target"))
            archive.save_annotation("proof-anchor", "session", session_id, "Retain the target")
            archive.commit()
    store = IndexGenerationStore.for_archive_root(tmp_path)
    candidate = store.create(owner_id="proof-owner", source_snapshot="proof-snapshot")
    active_before = Path(store.active_pointer).resolve(strict=True)
    captured: list[reference_seal.PreparedIndexMutation] = []
    cursors: list[ControlledCursor] = []
    original_open = reference_seal.PreparedIndexMutation._open_observer
    primary = (
        asyncio.CancelledError("synthetic proof cancellation")
        if failure == "cancelled"
        else ValueError("synthetic proof enumeration failure")
    )

    def open_observer(seal: reference_seal.PreparedIndexMutation, name: str, path: Path) -> sqlite3.Connection:
        connection = original_open(seal, name, path)
        selected = "user" if failure == "constructor" else "candidate"
        if name == selected:
            captured.append(seal)
            cursor = connection.cursor(factory=ControlledCursor)
            cursor.execute("SELECT 1 UNION ALL SELECT 2")
            assert next(cursor)[0] == 1
            cursor.allow_cleanup.clear()
            cursors.append(cursor)
            if failure == "constructor":
                raise primary
        return connection

    original_resolve = reference_seal._still_resolves

    def resolve(connection: sqlite3.Connection, ref: reference_seal._ResolvedReference) -> bool:
        if failure in {"enumeration", "cancelled"}:
            raise primary
        return original_resolve(connection, ref)

    monkeypatch.setattr(reference_seal.PreparedIndexMutation, "_open_observer", open_observer)
    monkeypatch.setattr(reference_seal, "_still_resolves", resolve)
    with pytest.raises(BaseExceptionGroup) as refused:
        store.prepare_promotion(candidate)
    assert len(captured) == len(cursors) == 1
    seal, cursor = captured[0], cursors[0]
    original_failure, cleanup = refused.value.exceptions
    if failure == "orphan":
        assert isinstance(original_failure, reference_seal.ReferenceSealError)
    else:
        assert original_failure is primary
    assert isinstance(cleanup, NativeConnectionSettlementError)
    assert cleanup.owner in native_sql_children(seal)
    assert cleanup.owner.connection is not None
    assert cursor.close_attempts == 1
    assert retained_native_settlement_owners_on_current_thread() == (seal,)
    assert Path(store.active_pointer).resolve(strict=True) == active_before
    assert store.load(candidate.generation_id).state == "inactive"
    assert not seal._closed
    cursor.allow_cleanup.set()
    seal.close()
    assert cursor.close_attempts == 2
    assert seal._closed
    assert native_sql_children(seal) == ()
    assert retained_native_settlement_owners_on_current_thread() == ()


@pytest.mark.parametrize("phase", ["constructor", "candidate"])
def test_proof_snapshot_failure_rolls_back_once_and_preserves_both_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    from polylogue.storage.sqlite import connection_profile, reference_seal
    from tests.infra.archive_templates import bootstrap_archive_root
    from tests.infra.reference_sessions import reference_session
    from tests.infra.sqlite_cursor_settlement import ControlledConnection

    with write_lease("test.promotion-snapshot-cleanup", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            session_id = _write_prepared_session(archive, reference_session("snapshot-target"))
            archive.save_annotation("snapshot-anchor", "session", session_id, "Retain the target")
            archive.commit()
    store = IndexGenerationStore.for_archive_root(tmp_path)
    candidate = store.create(owner_id="snapshot-owner", source_snapshot="snapshot")
    primary = ValueError("synthetic snapshot enumeration failure")
    rollback_failure = OSError("synthetic readonly rollback failure")
    captured: list[ControlledConnection] = []

    def connect(database: str | Path, **kwargs: Any) -> sqlite3.Connection:
        return sqlite3.connect(database, factory=ControlledConnection, **kwargs)

    def fail(connection: sqlite3.Connection) -> None:
        assert isinstance(connection, ControlledConnection)
        assert connection.in_transaction
        connection.rollback_failure = rollback_failure
        captured.append(connection)
        raise primary

    def user_references(connection: sqlite3.Connection) -> Generator[reference_seal._ReferenceAnchor]:
        fail(connection)
        yield reference_seal._ReferenceAnchor("unreachable")

    def resolve(connection: sqlite3.Connection, ref: reference_seal._ResolvedReference) -> bool:
        fail(connection)
        return False

    monkeypatch.setattr(connection_profile, "connect_measured", connect)
    if phase == "constructor":
        monkeypatch.setattr(reference_seal, "_references_from_user", user_references)
    else:
        monkeypatch.setattr(reference_seal, "_still_resolves", resolve)
    with pytest.raises(BaseExceptionGroup) as refused:
        store.prepare_promotion(candidate)
    assert refused.value.exceptions == (primary, rollback_failure)
    assert len(captured) == 1
    connection = captured[0]
    assert connection.rollback_attempts == connection.close_attempts == 1
    from polylogue.storage.sqlite.connection_profile import retained_native_settlement_owners_on_current_thread

    (seal,) = retained_native_settlement_owners_on_current_thread()
    seal.close()
    assert connection.rollback_attempts == connection.close_attempts == 1
    assert retained_native_settlement_owners_on_current_thread() == ()


@pytest.mark.parametrize("mutation_failure", [False, True])
def test_retained_promotion_proof_context_settles_its_creator_and_preserves_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation_failure: bool
) -> None:
    from polylogue.storage.sqlite import reference_seal
    from polylogue.storage.sqlite.connection_profile import (
        NativeConnectionSettlementError,
        retained_native_settlement_owners_on_current_thread,
    )
    from tests.infra.archive_templates import bootstrap_archive_root
    from tests.infra.sqlite_cursor_settlement import ControlledCursor

    with write_lease("test.promotion-context-cleanup", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
    store = IndexGenerationStore.for_archive_root(tmp_path)
    candidate = store.create(owner_id="context-owner", source_snapshot="context-snapshot")
    primary = ValueError("synthetic admitted promotion failure")
    cursors: list[ControlledCursor] = []
    original_open = reference_seal.PreparedIndexMutation._open_observer

    def open_observer(seal: reference_seal.PreparedIndexMutation, name: str, path: Path) -> sqlite3.Connection:
        connection = original_open(seal, name, path)
        if name == "candidate":
            cursor = connection.cursor(factory=ControlledCursor)
            cursor.execute("SELECT 1 UNION ALL SELECT 2")
            assert next(cursor)[0] == 1
            cursors.append(cursor)
        return connection

    monkeypatch.setattr(reference_seal.PreparedIndexMutation, "_open_observer", open_observer)
    prepared = store.prepare_promotion(candidate)
    (cursor,) = cursors
    if mutation_failure:
        with pytest.raises(BaseExceptionGroup) as refused:
            with prepared:
                cursor.allow_cleanup.clear()
                raise primary
        assert refused.value.exceptions[0] is primary
        cleanup = refused.value.exceptions[1]
        assert isinstance(cleanup, NativeConnectionSettlementError)
        assert cursor.close_attempts == 1
        assert retained_native_settlement_owners_on_current_thread() == (prepared.reference_seal,)
        cursor.allow_cleanup.set()
        prepared.close()
        assert cursor.close_attempts == 2
    else:
        with prepared:
            assert cursor.close_attempts == 0
        assert cursor.close_attempts == 1
    assert prepared.reference_seal._closed
    assert retained_native_settlement_owners_on_current_thread() == ()


@pytest.mark.parametrize("tier", ["source", "user", "audit", "index"])
@pytest.mark.parametrize("phase", ["open", "promotion"])
def test_configured_tier_retarget_cannot_inherit_prepared_promotion_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tier: str, phase: str
) -> None:
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation, ReferenceSealStaleError

    configured, canonical = _configured_symlink_archive(tmp_path)
    replacement = tmp_path / "replacement"
    _archive(replacement)
    store = IndexGenerationStore.for_archive_root(configured)
    generation = store.create(owner_id="retarget-control", source_snapshot="same-input")
    original_open = PreparedIndexMutation._open_observer

    def retarget() -> None:
        link = configured / f"{tier}.db"
        link.unlink()
        link.symlink_to(replacement / f"{tier}.db")

    def open_after_retarget(seal: PreparedIndexMutation, name: str, path: Path) -> sqlite3.Connection:
        if name == "index":
            retarget()
        return original_open(seal, name, path)

    if phase == "open":
        monkeypatch.setattr(PreparedIndexMutation, "_open_observer", open_after_retarget)
        with pytest.raises(ReferenceSealStaleError):
            store.prepare_promotion(generation)
    else:
        with store.prepare_promotion(generation) as prepared:
            retarget()
            with write_lease("test.retargeted-promotion", archive_root=configured):
                with pytest.raises(ReferenceSealStaleError):
                    store.promote(generation, prepared)
    assert store.load(generation.generation_id).state == "inactive"
    assert (canonical / "index.db").resolve() != Path(generation.index_path).resolve()


@pytest.mark.parametrize("tier", ["source", "user", "audit"])
def test_recreated_configured_link_to_same_leaf_invalidates_seal(tmp_path: Path, tier: str) -> None:
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation, ReferenceSealStaleError

    configured, canonical = _configured_symlink_archive(tmp_path)
    with PreparedIndexMutation(canonical / "index.db", archive_root=configured) as seal:
        link = configured / f"{tier}.db"
        # Keep the original incarnation allocated so this cannot pass through
        # immediate inode reuse while still resolving to the identical leaf.
        link.rename(configured / f"previous-{tier}.link")
        link.symlink_to(canonical / f"{tier}.db")
        with pytest.raises(ReferenceSealStaleError):
            seal.validate_observers_current()


def test_source_snapshot_rejects_recreated_configured_link_to_same_leaf(tmp_path: Path) -> None:
    configured, canonical = _configured_symlink_archive(tmp_path)
    with pytest.raises(RuntimeError):
        with _open_source_snapshot(configured) as connection:
            count = connection.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0]
            link = configured / "source.db"
            link.rename(configured / "previous-source.link")
            link.symlink_to(canonical / "source.db")
            assert connection.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()[0] == count
