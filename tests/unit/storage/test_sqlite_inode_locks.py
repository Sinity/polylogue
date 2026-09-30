"""Live inode verification and physical reads preserve SQLite's kernel locks."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import sqlite3
import stat
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.core.compute_cancel import compute_cancel
from polylogue.operations import archive_backup
from polylogue.storage.index_generation import _checkpoint_truncate, _open_source_snapshot
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite import connection_profile, lock_isolated_file_read
from polylogue.storage.sqlite.archive_tiers.schema_inventory import capture_schema_census
from polylogue.storage.sqlite.audit_leaf import (
    AuditLeafError,
    VerifiedAuditLeaf,
    open_verified_audit_connection,
    open_verified_audit_read_connection,
    open_verified_sqlite_read_connection,
    open_verified_sqlite_write_connection,
)
from polylogue.storage.sqlite.file_identity import SQLiteFileIdentity
from polylogue.storage.sqlite.migration_runner import _validate_live_source_fingerprint
from tests.infra import workload_artifacts
from tests.infra.sqlite_lock_probe import sqlite_lock_state

pytestmark = pytest.mark.uses_real_clock


@pytest.fixture(params=["linux_identity", "portable_custody"], autouse=True)
def identity_platform(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Removing O_PATH exercises custody throughout every production sibling."""
    if request.param == "portable_custody":
        monkeypatch.delattr(os, "O_PATH", raising=False)


def _database(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA user_version=1")
        connection.execute("CREATE TABLE raw_sessions(raw_id TEXT)")
        connection.execute("INSERT INTO raw_sessions VALUES ('neutral')")
        connection.commit()


@contextmanager
def _live_reader(path: Path) -> Iterator[sqlite3.Connection]:
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        assert connection.execute("SELECT raw_id FROM raw_sessions").fetchone() == ("neutral",)
        yield connection


def _assert_protected(path: Path) -> None:
    assert sqlite_lock_state(path) == {"main": "protected", "shm": "protected"}


@pytest.mark.parametrize("route", ["source_read", "audit_read", "source_write", "audit_write", "source_snapshot"])
def test_verified_sqlite_routes_preserve_other_live_readers_locks(tmp_path: Path, route: str) -> None:
    """Ordinary identity fd closes turn the inside/after lock probes red."""
    path = tmp_path / ("audit.db" if route.startswith("audit") else "source.db")
    _database(path)
    routes = {
        "source_read": lambda: open_verified_sqlite_read_connection(path),
        "audit_read": lambda: open_verified_audit_read_connection(path),
        "source_write": lambda: open_verified_sqlite_write_connection(path),
        "audit_write": lambda: open_verified_audit_connection(path),
        "source_snapshot": lambda: _open_source_snapshot(tmp_path),
    }
    with _live_reader(path):
        _assert_protected(path)
        with routes[route]() as connection:
            assert connection.execute("SELECT raw_id FROM raw_sessions").fetchone() == ("neutral",)
            _assert_protected(path)
        _assert_protected(path)


def test_repeated_leaf_namespace_checks_preserve_live_source_locks(tmp_path: Path) -> None:
    """Every metadata and sidecar check leaves the original WAL reader protected."""
    path = tmp_path / "source.db"
    _database(path)
    with _live_reader(path):
        with VerifiedAuditLeaf(tmp_path, filename=path.name) as leaf:
            leaf.assert_unchanged()
            _assert_protected(path)
            leaf.assert_unchanged()
            _assert_protected(path)
        _assert_protected(path)


def test_checkpoint_descriptor_admission_preserves_the_existing_reader(tmp_path: Path) -> None:
    """Checkpoint descriptor closes must not release the reader's main lock."""
    path = tmp_path / "index.db"
    _database(path)
    with _live_reader(path):
        _assert_protected(path)
        _checkpoint_truncate(path, label="neutral", archive_root=tmp_path)
        _assert_protected(path)


def test_isolated_physical_read_preserves_live_locks_and_exact_copy(tmp_path: Path) -> None:
    path = tmp_path / "source.db"
    destination = tmp_path / "snapshot.db"
    _database(path)
    expected = path.read_bytes()
    with _live_reader(path):
        _assert_protected(path)
        result = lock_isolated_file_read.read_sqlite_file_in_lock_isolated_process(path, copy_to=destination)
        _assert_protected(path)
    assert result.sha256 == hashlib.sha256(expected).hexdigest()
    assert result.size_bytes == len(expected)
    assert destination.read_bytes() == expected
    assert destination.stat().st_mode == result.metadata.st_mode
    assert destination.stat().st_mtime_ns == result.metadata.st_mtime_ns


def test_backup_physical_copy_preserves_its_transaction_locks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A parent-process hash or copy drops the BEGIN IMMEDIATE main lock."""
    path = tmp_path / "source.db"
    destination = tmp_path / "snapshot.db"
    _database(path)
    original = lock_isolated_file_read.read_sqlite_file_in_lock_isolated_process
    seen = []

    def checked_copy(source: Path, *, copy_to: Path | None = None) -> lock_isolated_file_read.SQLiteFileRead:
        _assert_protected(source)
        result = original(source, copy_to=copy_to)
        _assert_protected(source)
        seen.append(result)
        return result

    monkeypatch.setattr(archive_backup, "read_sqlite_file_in_lock_isolated_process", checked_copy)
    size, fingerprint = archive_backup._backup_sqlite(path, destination, archive_root_path=tmp_path)
    assert len(seen) == 1
    assert size == len(destination.read_bytes())
    assert fingerprint["sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()


def test_migration_physical_fingerprint_preserves_its_transaction_locks(tmp_path: Path) -> None:
    """Restoring the parent-process physical hash makes the final main probe red."""
    path = tmp_path / "source.db"
    _database(path)
    fingerprint = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "user_version": 1,
    }
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        _assert_protected(path)
        _validate_live_source_fingerprint(connection, {"source_fingerprint": fingerprint})
        _assert_protected(path)


def test_physical_reader_cancellation_reaps_the_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "source.db"
    _database(path)
    children: list[subprocess.Popen[bytes]] = []
    original = subprocess.Popen
    cancelled = threading.Event()
    token = compute_cancel.set(cancelled)

    def start_then_cancel(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        child = cast("subprocess.Popen[bytes]", original(*args, **kwargs))
        children.append(child)
        cancelled.set()
        return child

    monkeypatch.setattr(subprocess, "Popen", start_then_cancel)
    try:
        with pytest.raises(OSError) as error:
            lock_isolated_file_read.read_sqlite_file_in_lock_isolated_process(path)
        assert error.value.errno == errno.ECANCELED
    finally:
        compute_cancel.reset(token)
    assert len(children) == 1
    assert children[0].returncode is not None


def test_writer_lock_replacement_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "audit.db"
    _database(path)
    with VerifiedAuditLeaf(tmp_path, lock_writer=True) as leaf:
        filename = leaf._writer_lock_filename
        assert filename is not None
        lock = tmp_path / filename
        lock.rename(tmp_path / "displaced.lock")
        lock.touch(mode=0o600)
        with pytest.raises(AuditLeafError, match="writer lock changed"):
            leaf.assert_unchanged()


@pytest.mark.parametrize("directory_alias", [False, True])
def test_writer_lock_excludes_an_independent_writer_process(tmp_path: Path, directory_alias: bool) -> None:
    path = tmp_path / "audit.db"
    _database(path)
    alias_child = tmp_path / "directory"
    alias_child.mkdir()
    selected_directory = alias_child / ".." if directory_alias else tmp_path
    probe = """
import sys
from pathlib import Path
from polylogue.storage.sqlite.audit_leaf import AuditLeafError, VerifiedAuditLeaf
try:
    with VerifiedAuditLeaf(Path(sys.argv[1]), lock_writer=True):
        print("admitted")
except AuditLeafError:
    print("refused")
"""
    command = [sys.executable, "-c", probe, str(selected_directory)]
    with VerifiedAuditLeaf(tmp_path, lock_writer=True):
        assert subprocess.check_output(command, text=True).strip() == "refused"
    assert subprocess.check_output(command, text=True).strip() == "admitted"


@pytest.mark.parametrize("shape", ["symlink", "hardlink", "group_writable"])
def test_writer_lock_namespace_rejects_unowned_shapes(tmp_path: Path, shape: str) -> None:
    path = tmp_path / "audit.db"
    _database(path)
    metadata = path.stat()
    lock = tmp_path / f".audit.db.{metadata.st_dev}.{metadata.st_ino}.writer.lock"
    foreign = tmp_path / "foreign.lock"
    foreign.touch(mode=0o600)
    if shape == "symlink":
        lock.symlink_to(foreign)
    elif shape == "hardlink":
        lock.hardlink_to(foreign)
    else:
        lock.touch(mode=0o600)
        lock.chmod(0o660)
    with pytest.raises(AuditLeafError):
        with VerifiedAuditLeaf(tmp_path, lock_writer=True):
            pass


def test_main_replacement_during_writer_lock_admission_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "audit.db"
    replacement = tmp_path / "replacement.db"
    _database(path)
    _database(replacement)
    original = fcntl.flock

    def replace_after_lock(descriptor: int, operation: int) -> None:
        original(descriptor, operation)
        if operation == fcntl.LOCK_EX | fcntl.LOCK_NB:
            path.rename(tmp_path / "displaced.db")
            replacement.rename(path)

    monkeypatch.setattr(fcntl, "flock", replace_after_lock)
    with pytest.raises(AuditLeafError, match="leaf changed during writer admission"):
        with VerifiedAuditLeaf(tmp_path, lock_writer=True):
            pass


def test_snapshot_file_set_hashing_preserves_live_index_locks(tmp_path: Path) -> None:
    from devtools.index_snapshot import open_index_file_set, snapshot_index_file_set

    path = tmp_path / "index.db"
    _database(path)
    with _live_reader(path):
        with open_index_file_set(path) as files:
            result = snapshot_index_file_set(
                path, opened_main_identity=files.main_identity, opened_sidecar_identities=files.sidecar_identities
            )
            assert result["observation_complete"]
            _assert_protected(path)
        _assert_protected(path)


def test_fixture_inode_hashing_and_copy_preserve_live_source_locks(tmp_path: Path) -> None:
    source = tmp_path / "fixture"
    source.mkdir()
    path = source / "source.db"
    _database(path)
    destination = tmp_path / "clone"
    with _live_reader(path):
        descriptor = workload_artifacts._open_file_fd(path)
        try:
            digest = workload_artifacts._sha256_fd(descriptor)
            _assert_protected(path)
        finally:
            workload_artifacts._close_file(descriptor)
        _assert_protected(path)
        sidecar = Path(str(path) + "-shm")
        descriptor = workload_artifacts._open_file_fd(sidecar)
        try:
            assert workload_artifacts._sha256_fd(descriptor)
            _assert_protected(path)
        finally:
            workload_artifacts._close_file(descriptor)
        directory = os.open(source, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for leaf in (path, sidecar):
                workload_artifacts._chmod_at(directory, leaf.name, leaf.stat().st_mode & 0o777)
                _assert_protected(path)
        finally:
            os.close(directory)
        workload_artifacts._copy_tree(source, destination)
        _assert_protected(path)
    assert digest == hashlib.sha256((destination / path.name).read_bytes()).hexdigest()


def test_sqlite_descriptor_boundary_refuses_lock_releasing_file_handles(tmp_path: Path) -> None:
    path = tmp_path / "source.db"
    _database(path)
    with path.open("rb") as descriptor:
        with pytest.raises(ValueError, match="SQLiteFileIdentity"):
            connection_profile.open_readonly_connection(
                path, opened_main_identity=cast(SQLiteFileIdentity, descriptor.fileno()), validate_schema=False
            )


def test_schema_census_hash_preserves_an_existing_tier_readers_locks(tmp_path: Path) -> None:
    path = tmp_path / "source.db"
    _database(path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    with _live_reader(path):
        census = capture_schema_census(tmp_path, observed_at_ns=0, count_rows=False)
        _assert_protected(path)
    assert next(tier for tier in census.tiers if tier.tier.value == "source").file_sha256 == expected


def test_custody_keeps_selected_inode_pinned_and_refuses_replacement(tmp_path: Path) -> None:
    """Child custody survives rename; opening a replacement cannot reuse its identity."""
    path = tmp_path / "source.db"
    _database(path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = SQLiteFileIdentity(path, use_custodian=True)
    try:
        selected = identity.stat()
        path.rename(tmp_path / "displaced.db")
        _database(path)
        assert identity.stat().st_ino == selected.st_ino
        assert identity.physical_read(copy_to=None, copy_exclusive=False, copy_directory_fd=None)["sha256"] == expected
        with pytest.raises(OSError) as error:
            connection_profile.open_readonly_connection(path, opened_main_identity=identity, validate_schema=False)
        assert error.value.errno == errno.ESTALE
        assert identity._custodian is not None
        child = identity._custodian.child
        assert child.poll() is None
    finally:
        identity.close()
    assert child.returncode is not None


def test_custody_follows_pinned_directory_rename(tmp_path: Path) -> None:
    directory = tmp_path / "selected"
    directory.mkdir()
    path = directory / "source.db"
    _database(path)
    with SQLiteFileIdentity(path, use_custodian=True) as identity:
        directory.rename(tmp_path / "renamed")
        with closing(
            connection_profile.open_readonly_connection(path, opened_main_identity=identity, validate_schema=False)
        ) as reader:
            assert reader.execute("SELECT raw_id FROM raw_sessions").fetchone() == ("neutral",)


def test_custody_rechecks_identity_after_sqlite_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "source.db"
    _database(path)
    original = connect_measured
    connections = []

    def substitute_after_open(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = original(*args, **kwargs)
        connections.append(connection)
        path.rename(tmp_path / "displaced.db")
        _database(path)
        return connection

    with SQLiteFileIdentity(path, use_custodian=True) as identity:
        monkeypatch.setattr(connection_profile, "connect_measured", substitute_after_open)
        with pytest.raises(OSError) as error:
            connection_profile.open_readonly_connection(path, opened_main_identity=identity, validate_schema=False)
        assert error.value.errno == errno.ESTALE
        with pytest.raises(sqlite3.ProgrammingError):
            connections[0].execute("SELECT 1")


def test_custody_cancelled_copy_reaps_owner_and_preserves_parent_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "source.db"
    destination = tmp_path / "partial.db"
    _database(path)
    original_read = os.read
    cancelled = threading.Event()

    def cancel_at_progress(descriptor: int, size: int) -> bytes:
        chunk = original_read(descriptor, size)
        if b'"progress_bytes"' in chunk:
            cancelled.set()
        return chunk

    with _live_reader(path), SQLiteFileIdentity(path, use_custodian=True) as identity:
        assert identity._custodian is not None
        child = identity._custodian.child
        token = compute_cancel.set(cancelled)
        monkeypatch.setattr(os, "read", cancel_at_progress)
        try:
            with pytest.raises(OSError) as error:
                lock_isolated_file_read.read_sqlite_file_in_lock_isolated_process(
                    path, opened_identity=identity, copy_to=destination
                )
            assert error.value.errno == errno.ECANCELED
        finally:
            compute_cancel.reset(token)
        assert child.returncode is not None
        _assert_protected(path)
    # The owning operation, rather than the physical reader, removes partial output.
    destination.unlink(missing_ok=True)
    assert not destination.exists()


def test_physical_copy_preserves_metadata_and_supported_attributes(tmp_path: Path) -> None:
    """Native macOS flags and Linux xattrs survive the same production copy."""
    path = tmp_path / "source.db"
    destination = tmp_path / "snapshot.db"
    _database(path)
    path.chmod(0o640)
    attributes = hasattr(os, "setxattr")
    if attributes:
        try:
            os.setxattr(path, "user.polylogue-neutral", b"synthetic metadata")
        except OSError as exc:
            if exc.errno != errno.ENOTSUP:
                raise
            attributes = False
    if hasattr(os, "chflags"):
        os.chflags(path, stat.UF_NODUMP)
    with _live_reader(path):
        os.utime(path, ns=(1_600_000_000_123_456_789, 1_600_000_001_987_654_321))
        before = path.stat()
        result = lock_isolated_file_read.read_sqlite_file_in_lock_isolated_process(path, copy_to=destination)
        _assert_protected(path)
    after = destination.stat()
    assert after.st_mode == before.st_mode
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_atime_ns == before.st_atime_ns == result.metadata.st_atime_ns
    assert after.st_size == result.size_bytes
    if attributes:
        assert os.getxattr(destination, "user.polylogue-neutral") == b"synthetic metadata"
    flags = getattr(before, "st_flags", None)
    if flags is not None:
        assert getattr(after, "st_flags", None) == flags


@pytest.mark.parametrize("copy", [False, True])
def test_physical_read_metadata_precedes_streaming(tmp_path: Path, copy: bool) -> None:
    """A post-read stat disagrees with the copied and reported original atime."""
    path = tmp_path / "source.db"
    destination = tmp_path / "snapshot.db"
    _database(path)
    with _live_reader(path):
        # Set this after SQLite's own initial read, so the physical reader is
        # the operation that advances access time on a relatime filesystem.
        os.utime(path, ns=(1_600_000_000_123_456_789, 1_600_000_001_987_654_321))
        before = path.stat()
        result = lock_isolated_file_read.read_sqlite_file_in_lock_isolated_process(
            path, copy_to=destination if copy else None
        )
        _assert_protected(path)
        assert result.metadata.st_atime_ns == before.st_atime_ns
        assert result.metadata.st_mtime_ns == before.st_mtime_ns
        if copy:
            assert destination.stat().st_atime_ns == before.st_atime_ns
        if sys.platform == "linux":
            assert path.stat().st_atime_ns > before.st_atime_ns


def test_sqlite_open_refuses_native_directory_substitution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Checking only the pinned directory entry misses the actual SQLite pathname."""
    selected = tmp_path / "selected"
    selected.mkdir()
    path = selected / "source.db"
    _database(path)
    original = connect_measured
    connections = []

    def substitute_directory_at_open(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        selected.rename(tmp_path / "pinned-directory")
        selected.mkdir()
        _database(path)
        connection = original(*args, **kwargs)
        connections.append(connection)
        return connection

    with _live_reader(path), SQLiteFileIdentity(path, use_custodian=True) as identity:
        monkeypatch.setattr(connection_profile, "connect_measured", substitute_directory_at_open)
        with pytest.raises(OSError) as error:
            connection_profile.open_readonly_connection(path, opened_main_identity=identity, validate_schema=False)
        assert error.value.errno == errno.ESTALE
        _assert_protected(tmp_path / "pinned-directory" / "source.db")
        with pytest.raises(sqlite3.ProgrammingError):
            connections[0].execute("SELECT 1")


def test_checkpoint_refuses_disappeared_native_tier_without_creating_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restoring connect's create mode silently leaves a fresh tier after refusal."""
    directory = tmp_path / "archive"
    directory.mkdir()
    path = directory / "index.db"
    _database(path)
    original = sqlite3.connect

    def remove_directory_before_open(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        directory.rename(tmp_path / "pinned-archive")
        directory.mkdir()
        return cast(sqlite3.Connection, original(*args, **kwargs))

    with _live_reader(path):
        monkeypatch.setattr(sqlite3, "connect", remove_directory_before_open)
        with pytest.raises(sqlite3.OperationalError):
            _checkpoint_truncate(path, label="neutral", archive_root=directory)
        assert not path.exists()
        _assert_protected(tmp_path / "pinned-archive" / "index.db")
