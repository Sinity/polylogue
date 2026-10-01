"""Artifact lifetime stays with actual native owners after its preparation scope ends."""

import gc
import sqlite3
import sys
import tempfile
import weakref
from contextlib import closing
from pathlib import Path
from typing import cast

import pytest

from polylogue.core.sql_settlement import current_native_sql_lifetimes, retain_native_sql_lifetimes
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.connection_profile import (
    NativeConnectionSettlementError,
    NativeSQLCustodyOwner,
    retained_native_sql_owners_for_lifetime,
)
from tests.infra.native_sql_descriptor_probe import selected_file_descriptors
from tests.infra.sqlite_cursor_settlement import (
    ControlledCursor,
    arm_settlement,
    native_settlement_connections,  # noqa: F401  # Pytest fixture discovery.
)


@pytest.mark.skipif(sys.platform != "linux", reason="physical descriptor observation uses Linux procfs")
@pytest.mark.parametrize("construction_failure", [False, True])
def test_unsettled_cursor_retains_native_owner_artifact_and_creator_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, construction_failure: bool
) -> None:
    from polylogue.storage.sqlite import connection_profile as profiles
    from polylogue.storage.sqlite.write_lease import write_lease

    scratch = tempfile.TemporaryDirectory(dir=tmp_path)
    directory = Path(scratch.name)
    path = directory / "artifact.db"
    with closing(connect_measured(path)) as seed:
        seed.execute("CREATE TABLE evidence(value INTEGER)")
        seed.executemany("INSERT INTO evidence VALUES (?)", ((1,), (2,), (3,)))
        seed.commit()
    metadata = path.stat()
    identity = metadata.st_dev, metadata.st_ino
    cursors: list[ControlledCursor] = []
    primary = ValueError("synthetic construction failure with a live statement")

    def retain_statement(connection: sqlite3.Connection) -> None:
        cursor = connection.cursor(factory=ControlledCursor)
        cursors.append(cursor)
        cursor.execute("SELECT value FROM evidence")
        assert next(cursor) == (1,)
        cursor.allow_cleanup.clear()

    def refuse_setup(connection: sqlite3.Connection, *args: object, **kwargs: object) -> None:
        retain_statement(connection)
        raise primary

    if construction_failure:
        monkeypatch.setattr(profiles, "_assert_schema_supported", refuse_setup)
    with write_lease("test.native-cursor-settlement", archive_root=tmp_path):
        with pytest.raises(NativeConnectionSettlementError) as failure:
            with profiles.readonly_connection_context(
                path, validate_schema=construction_failure, lifetime_dependencies=(scratch,)
            ) as connection:
                retain_statement(connection)
        owner = failure.value.owner
        if construction_failure:
            assert failure.value.__cause__ is primary
        assert owner.connection is not None
        assert owner.custody is not None
        assert directory.exists()
        assert retained_native_sql_owners_for_lifetime(scratch) == (owner,)
        assert selected_file_descriptors(identity)
        with pytest.raises(NativeConnectionSettlementError):
            owner.close()
        assert directory.exists()
        for cursor in cursors:
            cursor.allow_cleanup.set()
        owner.close()
        assert selected_file_descriptors(identity) == ()
        assert retained_native_sql_owners_for_lifetime(scratch) == ()
        assert owner.connection is None and owner.custody is None
        owner.close()
    scratch.cleanup()
    assert not directory.exists()


@pytest.mark.parametrize("construction_failure", [False, True])
def test_readonly_artifact_dependency_survives_constructor_or_reader_close_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, construction_failure: bool
) -> None:
    from polylogue.storage.sqlite import connection_profile as profiles

    scratch = tempfile.TemporaryDirectory(dir=tmp_path)
    directory = Path(scratch.name)
    path = directory / "artifact.db"
    with closing(connect_measured(path)) as seed:
        seed.execute("CREATE TABLE evidence(value TEXT)")
        seed.execute("INSERT INTO evidence VALUES ('retained')")
        seed.commit()
    handle = arm_settlement(connect_measured(f"{path.as_uri()}?mode=ro", uri=True))
    monkeypatch.setattr(profiles, "connect_measured", lambda *args, **kwargs: handle)
    primary = ValueError("synthetic read construction failure")

    def refuse_schema(*args: object, **kwargs: object) -> None:
        raise primary

    if construction_failure:
        monkeypatch.setattr(profiles, "_assert_schema_supported", refuse_schema)
    reference = weakref.ref(scratch)
    with pytest.raises(NativeConnectionSettlementError) as refused:
        with profiles.readonly_connection_context(
            path, validate_schema=construction_failure, lifetime_dependencies=(scratch,)
        ) as reader:
            assert not construction_failure
            assert reader.execute("SELECT value FROM evidence").fetchone()[0] == "retained"
    owner = refused.value.owner
    if construction_failure:
        assert refused.value.__cause__ is primary
    assert retained_native_sql_owners_for_lifetime(scratch) == (owner,)
    del scratch
    gc.collect()
    assert reference() is not None and directory.exists()
    handle.allow_cleanup.set()
    owner.close()
    retained = reference()
    if retained is not None:
        assert retained_native_sql_owners_for_lifetime(retained) == ()
        retained.cleanup()
    assert not directory.exists()


def test_healthy_readonly_artifact_dependency_retires_without_deleting_sealed_artifact(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.connection_profile import readonly_connection_context

    with tempfile.TemporaryDirectory(dir=tmp_path) as directory:
        path = Path(directory) / "artifact.db"
        with closing(connect_measured(path)) as seed:
            seed.execute("CREATE TABLE evidence(value TEXT)")
        lifetime = object()
        with readonly_connection_context(path, validate_schema=False, lifetime_dependencies=(lifetime,)) as reader:
            assert reader.execute("SELECT count(*) FROM evidence").fetchone()[0] == 0
            assert len(retained_native_sql_owners_for_lifetime(lifetime)) == 1
        assert retained_native_sql_owners_for_lifetime(lifetime) == ()
        assert path.is_file()


def test_scoped_artifact_survives_failed_close_and_context_reset(tmp_path: Path) -> None:
    scratch = tempfile.TemporaryDirectory(dir=tmp_path)
    directory = Path(scratch.name)
    reference = weakref.ref(scratch)
    with retain_native_sql_lifetimes(scratch):
        handle = arm_settlement(connect_measured(directory / "artifact.db"))
        owner = NativeSQLCustodyOwner(
            cast(sqlite3.Connection, handle), lifetime_dependencies=current_native_sql_lifetimes()
        )
        handle.execute("CREATE TABLE evidence(value TEXT)")
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
            connect_measured(directory / "artifact.db"),
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
    handle = arm_settlement(connect_measured(directory / "artifact.db"))
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
            connect_measured(":memory:"), anchored_descriptors=(first, second), scratch_directory=scratch
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
        assert not owner.anchored_descriptors
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
        assert not owner.anchored_descriptors
    finally:
        if owner.anchored_descriptors:
            os.close(descriptor)
            owner.close()


def test_nested_failed_new_child_preserves_entry_parent_until_parent_requests_cleanup(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.connection_profile import (
        native_sql_children,
        request_native_sql_parent_cleanup,
        retained_native_settlement_owners_on_current_thread,
        retire_native_sql_parent,
    )

    class Parent:
        def close(self) -> None:
            request_native_sql_parent_cleanup(self)
            for child in native_sql_children(self):
                child.close()
            retire_native_sql_parent(self)

    parent = Parent()
    first = NativeSQLCustodyOwner(connect_measured(tmp_path / "entry.db"), terminal_parent=parent)
    entry = (first,)
    second = NativeSQLCustodyOwner(connect_measured(tmp_path / "nested.db"), terminal_parent=parent)
    cursor = second.require_connection().cursor(factory=ControlledCursor)
    cursor.execute("SELECT 1 UNION ALL SELECT 2")
    next(cursor)
    cursor.allow_cleanup.clear()
    try:
        with pytest.raises(NativeConnectionSettlementError):
            second.close()
        assert retained_native_settlement_owners_on_current_thread(entry) == (second,)
        assert first.require_connection().execute("SELECT 1").fetchone()[0] == 1
        cursor.allow_cleanup.set()
        second.close()
        assert retained_native_settlement_owners_on_current_thread(entry) == ()
        request_native_sql_parent_cleanup(parent)
        assert retained_native_settlement_owners_on_current_thread(entry) == (parent,)
        with pytest.raises(RuntimeError):
            first.require_connection()
        with pytest.raises(RuntimeError):
            first.handoff()
        parent.close()
        assert not native_sql_children(parent)
    finally:
        cursor.allow_cleanup.set()
        parent.close()


def test_failed_cursor_close_cannot_handoff_its_creator_native_owner(tmp_path: Path) -> None:
    from polylogue.core.sql_settlement import retained_native_sql_owners

    owner = NativeSQLCustodyOwner(connect_measured(tmp_path / "handoff.db"))
    connection = owner.require_connection()
    cursor = connection.cursor(factory=ControlledCursor)
    cursor.execute("SELECT 1 UNION ALL SELECT 2")
    next(cursor)
    cursor.allow_cleanup.clear()
    try:
        with pytest.raises(NativeConnectionSettlementError):
            owner.close()
        assert cursor.close_attempts == 1
        with pytest.raises(RuntimeError):
            owner.handoff()
        with pytest.raises(RuntimeError):
            owner.require_connection()
        assert owner.connection is connection and owner in retained_native_sql_owners()
        assert cursor.close_attempts == 1
        cursor.allow_cleanup.set()
        owner.close()
        assert cursor.close_attempts == 2 and owner not in retained_native_sql_owners()
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
    finally:
        cursor.allow_cleanup.set()
        owner.close()


def test_healthy_native_construction_handoff_preserves_the_idle_connection(tmp_path: Path) -> None:
    from polylogue.core.sql_settlement import retained_native_sql_owners

    owner = NativeSQLCustodyOwner(connect_measured(tmp_path / "healthy-handoff.db"))
    connection = owner.handoff()
    try:
        assert owner not in retained_native_sql_owners()
        assert connection.execute("SELECT 1").fetchone() == (1,)
    finally:
        connection.close()
