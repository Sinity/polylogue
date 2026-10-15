"""Physical descriptor consumers preserve independent native SQLite locks."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import threading
from contextlib import closing
from pathlib import Path

import pytest

from devtools.index_snapshot import open_index_file_set, snapshot_index_file_set
from polylogue.core.compute import DaemonOperationCancelled
from polylogue.core.compute_cancel import compute_cancel
from polylogue.storage import index_generation
from polylogue.storage.index_generation import _open_source_snapshot
from polylogue.storage.sqlite import connection_profile, physical_file
from polylogue.storage.sqlite.connection_profile import (
    NativeConnectionSettlementError,
    retained_native_sql_owners_on_current_thread,
)
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra import workload_artifacts
from tests.infra.sqlite_lock_probe import retained_sqlite_namespace_locks, sqlite_lock_state

pytestmark = [
    pytest.mark.uses_real_clock,
    pytest.mark.skipif(sys.platform != "linux", reason="Linux identity capability"),
]


@pytest.mark.parametrize(
    "route", ["source_snapshot", "devtools_snapshot", "fixture_hash", "fixture_copy", "fixture_chmod"]
)
def test_physical_consumers_preserve_native_namespace_locks(tmp_path: Path, route: str) -> None:
    source = tmp_path / "source"
    source.mkdir()
    path = source / ("index.db" if route == "devtools_snapshot" else "source.db")
    all_names = route != "source_snapshot"
    expected = dict.fromkeys(("main", "wal", "shm", "journal") if all_names else ("main", "shm"), "protected")
    with retained_sqlite_namespace_locks(path):
        assert sqlite_lock_state(path, include_wal_journal=all_names) == expected

        if route == "source_snapshot":
            with _open_source_snapshot(source) as connection:
                with closing(connection.execute("SELECT value FROM evidence")) as rows:
                    assert rows.fetchone() == ("neutral",)
                assert sqlite_lock_state(path, include_wal_journal=all_names) == expected
        elif route == "devtools_snapshot":
            with open_index_file_set(path) as files:
                result = snapshot_index_file_set(
                    path, opened_main_fd=files.main_fd, opened_sidecar_fds=files.sidecar_fds
                )
                assert result["observation_complete"]
                assert sqlite_lock_state(path, include_wal_journal=all_names) == expected
        elif route == "fixture_hash":
            assert workload_artifacts._sha256(path)
        elif route == "fixture_copy":
            workload_artifacts._copy_tree(source, tmp_path / "copy")
            assert (tmp_path / "copy" / path.name).stat().st_size == path.stat().st_size
        else:
            directory = os.open(source, os.O_RDONLY | os.O_DIRECTORY)
            try:
                workload_artifacts._chmod_at(directory, path.name, path.stat().st_mode & 0o777)
            finally:
                os.close(directory)
        assert sqlite_lock_state(path, include_wal_journal=all_names) == expected


def test_checkpoint_admission_preserves_an_existing_native_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "index.db"
    with retained_sqlite_namespace_locks(path), write_lease("test.checkpoint-locks", archive_root=tmp_path):
        expected = {"main": "protected", "shm": "protected"}
        assert sqlite_lock_state(path) == expected
        original = index_generation.checkpoint_connection
        observed = []

        def checkpoint(*args, **kwargs):
            observed.append(sqlite_lock_state(path))
            return original(*args, **kwargs)

        monkeypatch.setattr(index_generation, "checkpoint_connection", checkpoint)
        # A pinned reader can make TRUNCATE busy. That explicit refusal must
        # still leave this independent reader's real kernel locks intact.
        try:
            index_generation._checkpoint_truncate(path, label="neutral", archive_root=tmp_path)
        except RuntimeError as error:
            assert "WAL checkpoint failed" in str(error)
        assert observed == [expected]
        assert sqlite_lock_state(path) == expected


@pytest.mark.parametrize("route", ["readonly", "checkpoint"])
@pytest.mark.parametrize("replacement", [False, True])
def test_native_open_refuses_substituted_parent_and_never_creates_missing_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str, replacement: bool
) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    path = selected / "index.db"
    displaced = tmp_path / "displaced"
    connections = []
    original = connection_profile.connect_measured

    def substitute(*args, **kwargs):
        selected.rename(displaced)
        selected.mkdir()
        if replacement:
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("CREATE TABLE unrelated(value TEXT)").close()
        connection = original(*args, **kwargs)
        connections.append(connection)
        return connection

    with retained_sqlite_namespace_locks(path):
        expected = {"main": "protected", "shm": "protected"}
        descriptor = physical_file.open_physical_identity(path)
        try:
            monkeypatch.setattr(connection_profile, "connect_measured", substitute)
            monkeypatch.setattr(index_generation, "connect_measured", substitute)
            with pytest.raises((OSError, sqlite3.OperationalError)):
                if route == "readonly":
                    connection_profile.open_readonly_connection(path, opened_main_fd=descriptor, validate_schema=False)
                else:
                    with write_lease("test.checkpoint-substitution", archive_root=tmp_path):
                        index_generation._checkpoint_truncate(path, label="neutral", archive_root=tmp_path)
            assert path.exists() is replacement
            assert sqlite_lock_state(displaced / path.name) == expected
            for connection in connections:
                with pytest.raises(sqlite3.ProgrammingError):
                    connection.execute("SELECT 1")
        finally:
            os.close(descriptor)


def test_descriptor_boundary_refuses_an_ordinary_parent_data_handle(tmp_path: Path) -> None:
    path = tmp_path / "source.db"
    with retained_sqlite_namespace_locks(path):
        with path.open("rb") as handle:
            with pytest.raises(ValueError, match="O_PATH"):
                connection_profile.open_readonly_connection(path, opened_main_fd=handle.fileno(), validate_schema=False)


def test_bound_open_failed_parent_close_keeps_its_actual_native_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "source.db"
    original_open = os.open
    original_close = os.close
    refused = []

    def opened(name, flags, *args, **kwargs):
        descriptor = original_open(name, flags, *args, **kwargs)
        if Path(name) == tmp_path and flags & os.O_DIRECTORY:
            refused.append(descriptor)
        return descriptor

    def close(descriptor):
        if descriptor in refused:
            raise OSError("synthetic parent close before effect")
        original_close(descriptor)

    with retained_sqlite_namespace_locks(path):
        descriptor = physical_file.open_physical_identity(path)
        owner = None
        try:
            with monkeypatch.context() as patch:
                patch.setattr(os, "open", opened)
                patch.setattr(os, "close", close)
                with pytest.raises(NativeConnectionSettlementError) as failed:
                    connection_profile.open_readonly_connection(path, opened_main_fd=descriptor, validate_schema=False)
                owner = failed.value.owner
                assert owner in retained_native_sql_owners_on_current_thread()
                assert owner.anchored_descriptors == tuple(refused)
                assert os.fstat(refused[0]).st_ino == tmp_path.stat().st_ino
                assert sqlite_lock_state(path) == {"main": "protected", "shm": "protected"}
        finally:
            for directory in refused:
                original_close(directory)
            if owner is not None:
                owner.close()
            original_close(descriptor)
        assert owner not in retained_native_sql_owners_on_current_thread()


@pytest.mark.parametrize("route", ["source_snapshot", "devtools_snapshot", "fixture_hash"])
def test_consumer_failed_identity_close_retains_current_native_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    path = tmp_path / ("index.db" if route == "devtools_snapshot" else "source.db")
    original_open = physical_file.open_physical_identity
    original_close = os.close
    refused = []

    def opened(*args, **kwargs):
        descriptor = original_open(*args, **kwargs)
        if os.fstat(descriptor).st_ino == path.stat().st_ino:
            refused.append(descriptor)
        return descriptor

    def close(descriptor):
        if descriptor in refused:
            raise OSError("synthetic identity close before effect")
        original_close(descriptor)

    with retained_sqlite_namespace_locks(path):
        expected = {"main": "protected", "shm": "protected"}
        owner = None
        try:
            with monkeypatch.context() as patch:
                import devtools.index_snapshot as snapshots

                patch.setattr(physical_file, "open_physical_identity", opened)
                patch.setattr(snapshots, "open_physical_identity", opened)
                patch.setattr(os, "close", close)
                with pytest.raises(NativeConnectionSettlementError) as failed:
                    if route == "source_snapshot":
                        with _open_source_snapshot(tmp_path) as connection:
                            connection.execute("SELECT value FROM evidence").close()
                    elif route == "devtools_snapshot":
                        with open_index_file_set(path):
                            pass
                    else:
                        workload_artifacts._sha256(path)
                owner = failed.value.owner
                assert owner in retained_native_sql_owners_on_current_thread()
                assert tuple(refused) == owner.anchored_descriptors
                assert physical_file.require_physical_identity(refused[0]).st_ino == path.stat().st_ino
                assert sqlite_lock_state(path) == expected
        finally:
            for descriptor in refused:
                original_close(descriptor)
            if owner is not None:
                owner.close()
        assert owner not in retained_native_sql_owners_on_current_thread()
        assert sqlite_lock_state(path) == expected


@pytest.mark.parametrize("copy", [False, True])
def test_descriptor_stream_cancellation_reaps_actual_child_and_preserves_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, copy: bool
) -> None:
    path = tmp_path / "source.db"
    cancelled = threading.Event()
    original = subprocess.Popen
    children = []

    def start(*args, **kwargs):
        process = original(*args, **kwargs)
        children.append(process)
        cancelled.set()
        return process

    with retained_sqlite_namespace_locks(path):
        descriptor = physical_file.open_physical_identity(path)
        directory = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
        token = compute_cancel.set(cancelled)
        try:
            monkeypatch.setattr(subprocess, "Popen", start)
            with pytest.raises(DaemonOperationCancelled):
                physical_file.physical_identity_operation(
                    descriptor, copy_directory_fd=directory if copy else None, copy_name="copy.db" if copy else None
                )
            assert len(children) == 1 and children[0].returncode is not None
        finally:
            compute_cancel.reset(token)
            os.close(descriptor)
            os.close(directory)
        assert sqlite_lock_state(path) == {"main": "protected", "shm": "protected"}


def test_fixture_failed_parent_close_releases_new_identity_pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "source.db"
    original_close = os.close
    original_parent = workload_artifacts._open_pinned_parent
    original_identity = physical_file.open_physical_identity
    parents = []
    pins = []

    def parent(*args, **kwargs):
        descriptor, leaf = original_parent(*args, **kwargs)
        parents.append(descriptor)
        return descriptor, leaf

    def identity(*args, **kwargs):
        descriptor = original_identity(*args, **kwargs)
        pins.append(descriptor)
        return descriptor

    def close(descriptor):
        if descriptor in parents:
            raise OSError("synthetic fixture parent close before effect")
        original_close(descriptor)

    with retained_sqlite_namespace_locks(path):
        owner = None
        try:
            with monkeypatch.context() as patch:
                patch.setattr(workload_artifacts, "_open_pinned_parent", parent)
                patch.setattr(physical_file, "open_physical_identity", identity)
                patch.setattr(os, "close", close)
                with pytest.raises(NativeConnectionSettlementError) as failed:
                    workload_artifacts._open_file_fd(path)
                owner = failed.value.owner
                assert owner.anchored_descriptors == tuple(parents)
                assert owner in retained_native_sql_owners_on_current_thread()
                assert len(pins) == 1
                with pytest.raises(OSError):
                    os.fstat(pins[0])
                assert sqlite_lock_state(path) == {"main": "protected", "shm": "protected"}
        finally:
            for descriptor in parents:
                original_close(descriptor)
            if owner is not None:
                owner.close()
        assert owner not in retained_native_sql_owners_on_current_thread()


def test_native_open_refuses_same_inode_substituted_leaf_symlink(tmp_path: Path) -> None:
    import errno

    path = tmp_path / "source.db"
    displaced = tmp_path / "selected.db"
    with retained_sqlite_namespace_locks(path):
        descriptor = physical_file.open_physical_identity(path)
        try:
            path.rename(displaced)
            path.symlink_to(displaced.name)
            assert path.stat().st_ino == os.fstat(descriptor).st_ino
            with pytest.raises(OSError) as refused:
                connection_profile.open_readonly_connection(path, opened_main_fd=descriptor, validate_schema=False)
            assert refused.value.errno == errno.ESTALE
            assert sqlite_lock_state(path) == {"main": "protected", "shm": "protected"}
        finally:
            os.close(descriptor)


@pytest.mark.parametrize("symlink", [False, True])
def test_child_copy_refuses_existing_destination_without_changing_either_file(tmp_path: Path, symlink: bool) -> None:
    source = tmp_path / "source.db"
    output = tmp_path / "output"
    output.mkdir()
    target = output / "predecessor"
    target.write_bytes(b"neutral predecessor")
    destination = output / "selected"
    if symlink:
        destination.symlink_to(target.name)
    else:
        destination.write_bytes(b"neutral existing leaf")
    before = destination.read_bytes()
    with retained_sqlite_namespace_locks(source):
        source_pin = physical_file.open_physical_identity(source)
        directory = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            with pytest.raises(OSError, match="physical file child failed"):
                physical_file.physical_identity_operation(source_pin, copy_directory_fd=directory, copy_name="selected")
            assert destination.is_symlink() is symlink
            assert destination.read_bytes() == before
            assert target.read_bytes() == b"neutral predecessor"
            assert sqlite_lock_state(source) == {"main": "protected", "shm": "protected"}
        finally:
            os.close(directory)
            os.close(source_pin)
