"""Artifact lifetime stays with actual native owners after its preparation scope ends."""

import gc
import sqlite3
import tempfile
import weakref
from pathlib import Path
from typing import cast

import pytest

from polylogue.core.sql_settlement import current_native_sql_lifetimes, retain_native_sql_lifetimes
from polylogue.storage.sqlite.connection_profile import (
    NativeConnectionSettlementError,
    NativeSQLCustodyOwner,
    retained_native_sql_owners_for_lifetime,
)
from tests.infra.sqlite_settlement_handle import SettlementHandle


def test_scoped_artifact_survives_failed_close_and_context_reset(tmp_path: Path) -> None:
    scratch = tempfile.TemporaryDirectory(dir=tmp_path)
    directory = Path(scratch.name)
    reference = weakref.ref(scratch)
    with retain_native_sql_lifetimes(scratch):
        handle = SettlementHandle(sqlite3.connect(directory / "artifact.db"))
        owner = NativeSQLCustodyOwner(
            cast(sqlite3.Connection, handle), lifetime_dependencies=current_native_sql_lifetimes()
        )
        handle.connection.execute("CREATE TABLE evidence(value TEXT)")
    assert current_native_sql_lifetimes() == ()
    assert retained_native_sql_owners_for_lifetime(scratch) == (owner,)
    with pytest.raises(RuntimeError):
        owner.handoff()
    with pytest.raises(NativeConnectionSettlementError):
        owner.close()
    del scratch
    gc.collect()
    assert reference() is not None and directory.is_dir()
    handle.allow_cleanup.set()
    owner.close()
    retained = reference()
    if retained is not None:
        assert retained_native_sql_owners_for_lifetime(retained) == ()
        retained.cleanup()
    assert not directory.exists()


def test_scoped_artifact_survives_parent_bound_closed_child_until_parent_retirement(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.connection_profile import retire_native_sql_parent

    class Parent:
        def close(self) -> None:
            owner.close()
            retire_native_sql_parent(self)

    scratch = tempfile.TemporaryDirectory(dir=tmp_path)
    directory = Path(scratch.name)
    reference = weakref.ref(scratch)
    parent = Parent()
    with retain_native_sql_lifetimes(scratch):
        owner = NativeSQLCustodyOwner(
            sqlite3.connect(directory / "artifact.db"),
            terminal_parent=parent,
            lifetime_dependencies=current_native_sql_lifetimes(),
        )
    owner.close()
    assert owner.connection is None
    assert retained_native_sql_owners_for_lifetime(scratch) == (owner,)
    del scratch
    gc.collect()
    assert reference() is not None and directory.is_dir()
    parent.close()
    gc.collect()
    assert reference() is None and not directory.exists()


def test_failed_native_construction_keeps_scoped_artifact_after_context_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite import connection_profile

    scratch = tempfile.TemporaryDirectory(dir=tmp_path)
    directory = Path(scratch.name)
    reference = weakref.ref(scratch)
    handle = SettlementHandle(sqlite3.connect(directory / "artifact.db"))
    primary = ValueError("synthetic constructor custody refusal")

    def refuse_custody() -> None:
        raise primary

    monkeypatch.setattr(connection_profile, "current_sql_custody", refuse_custody)
    with retain_native_sql_lifetimes(scratch), pytest.raises(NativeConnectionSettlementError) as refused:
        NativeSQLCustodyOwner(cast(sqlite3.Connection, handle), lifetime_dependencies=current_native_sql_lifetimes())
    owner = refused.value.owner
    assert current_native_sql_lifetimes() == ()
    assert retained_native_sql_owners_for_lifetime(scratch) == (owner,)
    del scratch
    gc.collect()
    assert reference() is not None and directory.is_dir()
    handle.allow_cleanup.set()
    owner.close()
    retained = reference()
    if retained is not None:
        assert retained_native_sql_owners_for_lifetime(retained) == ()
        retained.cleanup()
    assert not directory.exists()


def test_ambiguous_descriptor_close_retains_creator_custody_and_artifact_without_numeric_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from polylogue.storage.sqlite import connection_profile as profiles
    from polylogue.storage.sqlite.write_lease import write_lease
    from tests.infra.archive_custody_probe import archive_custody_available
    from tests.infra.descriptor_close_fault import DescriptorCloseFault

    scratch = tempfile.TemporaryDirectory(dir=tmp_path)
    first = os.open(Path(scratch.name) / "first", os.O_CREAT | os.O_RDWR, 0o600)
    second = os.open(Path(scratch.name) / "second", os.O_CREAT | os.O_RDWR, 0o600)
    fault = DescriptorCloseFault(lambda descriptor: descriptor == first)
    with write_lease("test.ambiguous-descriptor", archive_root=tmp_path):
        owner = NativeSQLCustodyOwner(
            sqlite3.connect(":memory:"), anchored_descriptors=(first, second), scratch_directory=scratch
        )
        monkeypatch.setattr(profiles, "os", fault)
        with pytest.raises(NativeConnectionSettlementError) as refused:
            owner.close()
        assert refused.value.owner is owner
        assert owner.connection is None
        assert owner.anchored_descriptors == (first,)
        assert Path(scratch.name).exists()
        assert fault.attempts == [first, second]
    try:
        assert not archive_custody_available(tmp_path)
        with pytest.raises(NativeConnectionSettlementError):
            owner.close()
        assert fault.attempts == [first, second]
        # The fixture's trusted physical close establishes settlement. The
        # native owner then observes EBADF; it never reuses the numeric slot.
        os.close(first)
        owner.close()
        assert owner.anchored_descriptors == ()
        assert not Path(scratch.name).exists()
        assert archive_custody_available(tmp_path)
    finally:
        if owner.anchored_descriptors:
            os.close(first)
            owner.close()


def test_same_file_descriptor_reuse_is_not_an_open_file_description_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from polylogue.storage.sqlite import connection_profile as profiles
    from tests.infra.descriptor_close_fault import DescriptorCloseFault

    path = tmp_path / "same-file"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    owner = NativeSQLCustodyOwner(None, anchored_descriptors=(descriptor,))
    fault = DescriptorCloseFault(lambda selected: selected == descriptor)
    monkeypatch.setattr(profiles, "os", fault)
    try:
        with pytest.raises(NativeConnectionSettlementError):
            owner.close()
        os.close(descriptor)
        replacement = os.open(path, os.O_RDWR)
        assert replacement == descriptor
        with pytest.raises(NativeConnectionSettlementError):
            owner.close()
        assert fault.attempts == [descriptor]
        assert os.write(replacement, b"replacement remains owned by its opener") > 0
        os.close(replacement)
        owner.close()
        assert owner.anchored_descriptors == ()
    finally:
        if owner.anchored_descriptors:
            os.close(descriptor)
            owner.close()
