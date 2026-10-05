"""Real Linux SQLite locks and namespace refusal on verified product reads."""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, closing, contextmanager
from pathlib import Path

import pytest

from polylogue.maintenance.source_manifest_continuity import SourceDeclaration, SourceRole
from polylogue.sources.source_snapshot import execute_source_cut, preflight_source_cut
from polylogue.sources.sqlite_export import write_logical_export
from polylogue.storage.sqlite.audit_leaf import (
    AuditLeafError,
    VerifiedAuditLeaf,
    open_verified_audit_connection,
    open_verified_audit_read_connection,
    open_verified_sqlite_read_connection,
    open_verified_sqlite_write_connection,
)
from polylogue.storage.sqlite.connection_profile import (
    NativeConnectionSettlementError,
    NativeSQLCustodyOwner,
    retained_native_sql_owners_on_current_thread,
)
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.sqlite_lock_probe import (
    directory_flock_state,
    retained_sqlite_namespace_locks,
    sqlite_lock_state,
)

pytestmark = [pytest.mark.uses_real_clock, pytest.mark.skipif(sys.platform != "linux", reason="Linux component")]

ReadFactory = Callable[[Path], AbstractContextManager[sqlite3.Connection]]
_READS = [("source.db", open_verified_sqlite_read_connection), ("audit.db", open_verified_audit_read_connection)]


@pytest.mark.parametrize("filename,factory", _READS)
def test_actual_verified_read_preserves_native_sqlite_locks_before_during_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str, factory: ReadFactory
) -> None:
    """Closing an ordinary main/SHM identity descriptor makes this red."""
    path = tmp_path / filename
    with retained_sqlite_namespace_locks(path):
        expected = {"main": "protected", "shm": "protected"}
        observed = []
        actual = VerifiedAuditLeaf._assert_sidecar_namespace

        def inspect(leaf: VerifiedAuditLeaf) -> None:
            actual(leaf)
            observed.append(sqlite_lock_state(path))

        monkeypatch.setattr(VerifiedAuditLeaf, "_assert_sidecar_namespace", inspect)
        assert sqlite_lock_state(path) == expected
        with factory(path) as reader:
            with closing(reader.execute("SELECT value FROM evidence")) as rows:
                assert rows.fetchone() == ("neutral",)
            assert sqlite_lock_state(path) == expected
        assert observed and all(state == expected for state in observed)
        assert sqlite_lock_state(path) == expected


@pytest.mark.parametrize("filename", ["source.db", "audit.db"])
def test_actual_leaf_metadata_custody_preserves_each_namespace_inode_lock(tmp_path: Path, filename: str) -> None:
    """Ordinary identity close on any of the four names loses its real lock."""
    path = tmp_path / filename
    expected = dict.fromkeys(("main", "wal", "shm", "journal"), "protected")
    with retained_sqlite_namespace_locks(path):
        assert sqlite_lock_state(path, include_wal_journal=True) == expected
        with VerifiedAuditLeaf(tmp_path, filename=filename) as leaf:
            first_pins = dict(leaf._sidecar_fds)
            assert len(first_pins) == 3
            assert sqlite_lock_state(path, include_wal_journal=True) == expected
            leaf.assert_unchanged()
            leaf.identity_metadata()
            assert leaf._sidecar_fds == first_pins
            assert sqlite_lock_state(path, include_wal_journal=True) == expected
        assert sqlite_lock_state(path, include_wal_journal=True) == expected


def test_actual_source_export_and_snapshot_preserve_parent_process_namespace_locks(tmp_path: Path) -> None:
    """A parent-process byte or identity FD close on any source inode makes this red."""
    path = tmp_path / "state.sqlite"
    expected = dict.fromkeys(("main", "wal", "shm", "journal"), "protected")
    observed = []

    class Sink:
        def write(self, payload: bytes) -> int:
            observed.append(sqlite_lock_state(path, include_wal_journal=True))
            return len(payload)

    with retained_sqlite_namespace_locks(path):
        assert sqlite_lock_state(path, include_wal_journal=True) == expected
        write_logical_export(path, Sink())
        assert observed and all(state == expected for state in observed)
        assert sqlite_lock_state(path, include_wal_journal=True) == expected
        cut = preflight_source_cut([SourceDeclaration("neutral", SourceRole.MUTABLE_SQLITE, path, True)])
        result = execute_source_cut(cut, tmp_path / "cut")
        assert result.counts.conserved
        assert result.candidate_manifest.items[0].source_id == "neutral"
        assert sqlite_lock_state(path, include_wal_journal=True) == expected


def test_actual_audit_writer_directory_flock_excludes_nested_and_external_writers(tmp_path: Path) -> None:
    """Removing the anchored directory flock admits both contenders."""
    path = tmp_path / "audit.db"
    source = tmp_path / "source.db"
    with (
        retained_sqlite_namespace_locks(path),
        retained_sqlite_namespace_locks(source),
        write_lease("test.verified-audit-lock", archive_root=tmp_path),
    ):
        assert directory_flock_state(tmp_path) == "available"
        expected = {"main": "protected", "shm": "protected"}
        with open_verified_audit_connection(path) as writer:
            assert directory_flock_state(tmp_path) == "protected"
            assert sqlite_lock_state(path) == expected
            with pytest.raises(AuditLeafError):
                with open_verified_audit_connection(path):
                    pytest.fail("competing Audit writer was admitted")
            # Source/general verified writers do not take the Audit flock.
            with open_verified_sqlite_write_connection(source) as sibling:
                with closing(sibling.execute("SELECT value FROM evidence")) as rows:
                    assert rows.fetchone() == ("neutral",)
            with closing(writer.execute("SELECT value FROM evidence")) as rows:
                assert rows.fetchone() == ("neutral",)
            assert sqlite_lock_state(path) == expected
            assert sqlite_lock_state(source) == expected
        assert directory_flock_state(tmp_path) == "available"
        assert sqlite_lock_state(path) == expected


def test_actual_audit_writer_failed_directory_close_retains_kernel_flock_and_native_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Releasing the directory before its actual close makes this red."""
    path = tmp_path / "audit.db"
    real_close = os.close
    real_lock = VerifiedAuditLeaf._acquire_writer_lock
    captured: list[int] = []
    failed_owner: NativeSQLCustodyOwner | None = None

    def acquire(leaf: VerifiedAuditLeaf) -> None:
        real_lock(leaf)
        assert leaf._directory_fd is not None
        captured.append(leaf._directory_fd)

    def close(descriptor: int) -> None:
        if descriptor in captured:
            raise OSError("synthetic directory close before effect")
        real_close(descriptor)

    with retained_sqlite_namespace_locks(path), write_lease("test.verified-directory-close", archive_root=tmp_path):
        try:
            with monkeypatch.context() as patch:
                patch.setattr(VerifiedAuditLeaf, "_acquire_writer_lock", acquire)
                patch.setattr(os, "close", close)
                with pytest.raises(NativeConnectionSettlementError) as failed:
                    with open_verified_audit_connection(path) as writer:
                        with closing(writer.execute("SELECT value FROM evidence")) as rows:
                            assert rows.fetchone() == ("neutral",)
                failed_owner = failed.value.owner
                assert failed_owner in retained_native_sql_owners_on_current_thread()
                assert failed_owner.connection is None and failed_owner.leaf is not None
                terminal = failed_owner.leaf._terminal_owner
                assert terminal is not None and terminal.anchored_descriptors == tuple(captured)
                assert directory_flock_state(tmp_path) == "protected"
                assert sqlite_lock_state(path) == {"main": "protected", "shm": "protected"}
        finally:
            # Ambiguous substituted closes never authorize a blind retry.
            # This creator retires its still-bound actual descriptor once.
            for descriptor in captured:
                try:
                    os.fstat(descriptor)
                except OSError:
                    continue
                real_close(descriptor)
            if failed_owner is not None:
                failed_owner.close()
        assert failed_owner is not None
        assert failed_owner._settled
        assert directory_flock_state(tmp_path) == "available"


@contextmanager
def _replace_namespace(root: Path, filename: str, coordinate: str, change: str) -> Iterator[None]:
    selected = root if coordinate == "directory" else root / (filename + coordinate)
    retired = selected.with_name(selected.name + ".retained")
    selected.rename(retired)
    try:
        if change == "symlink":
            selected.symlink_to(retired, target_is_directory=coordinate == "directory")
        elif coordinate == "directory":
            shutil.copytree(retired, selected)
        else:
            shutil.copyfile(retired, selected)
            selected.chmod(0o600)
        assert selected.lstat().st_ino != retired.lstat().st_ino
        yield
    finally:
        if selected.is_symlink() or selected.is_file():
            selected.unlink()
        else:
            shutil.rmtree(selected)
        retired.rename(selected)


@pytest.mark.parametrize("filename,factory", _READS)
@pytest.mark.parametrize("coordinate", ["", "-wal", "-shm", "-journal", "directory"])
def test_verified_read_refuses_existing_redirected_namespace(
    tmp_path: Path, filename: str, factory: ReadFactory, coordinate: str
) -> None:
    path = tmp_path / filename
    with retained_sqlite_namespace_locks(path):
        with _replace_namespace(tmp_path, filename, coordinate, "symlink"), pytest.raises(AuditLeafError):
            with factory(path):
                pytest.fail("redirected original namespace was admitted")


@pytest.mark.parametrize("filename,factory", _READS)
@pytest.mark.parametrize("coordinate", ["", "-wal", "-shm", "-journal", "directory"])
@pytest.mark.parametrize("change", ["replacement", "symlink"])
def test_verified_read_refuses_namespace_changed_after_native_read(
    tmp_path: Path, filename: str, factory: ReadFactory, coordinate: str, change: str
) -> None:
    path = tmp_path / filename
    with retained_sqlite_namespace_locks(path):
        mutation = None
        try:
            with pytest.raises(AuditLeafError):
                with factory(path) as reader:
                    with closing(reader.execute("SELECT value FROM evidence")) as rows:
                        assert rows.fetchone() == ("neutral",)
                    mutation = _replace_namespace(tmp_path, filename, coordinate, change)
                    mutation.__enter__()
        finally:
            if mutation is not None:
                mutation.__exit__(None, None, None)
