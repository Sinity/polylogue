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
