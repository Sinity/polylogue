"""Canonical probe and prototype owners retain actual failed statements."""

import sqlite3
from builtins import BaseExceptionGroup
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.storage.sqlite import connection_profile as profiles
from polylogue.storage.sqlite.archive_tiers import bootstrap, schema_inventory
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.sqlite_cursor_settlement import BackupCursorFault, ControlledCursor


@pytest.mark.parametrize("fail_prepare", [False, True])
def test_canonical_probe_consumer_retains_actual_statement_until_creator_retry(
    monkeypatch: pytest.MonkeyPatch, fail_prepare: bool
) -> None:
    cursors: list[ControlledCursor] = []
    initialize = bootstrap.initialize_runtime_tier_probe

    def prepare(connection: sqlite3.Connection, tier: ArchiveTier) -> None:
        initialize(connection, tier)
        cursor = connection.cursor(factory=ControlledCursor)
        assert isinstance(cursor, ControlledCursor)
        cursor.execute("SELECT 1 UNION ALL SELECT 2")
        cursor.fetchone()
        cursor.allow_cleanup.clear()
        cursors.append(cursor)
        if fail_prepare:
            raise ValueError("synthetic canonical probe preparation fault")

    monkeypatch.setattr(bootstrap, "_TIER_PROTOTYPES", {})
    monkeypatch.setattr(bootstrap, "_record_tier_prototype", lambda *_args: None)
    monkeypatch.setattr(schema_inventory, "initialize_runtime_tier_probe", prepare)
    with pytest.raises(profiles.NativeConnectionSettlementError) as failed:
        schema_inventory.canonical_schema_objects(ArchiveTier.USER)
    owner = failed.value.owner
    try:
        assert owner.connection is not None and owner.scratch_directory is None
        assert cursors[0].close_attempts == 1
    finally:
        cursors[0].allow_cleanup.set()
        owner.close()
    assert cursors[0].close_attempts == 2 and owner.connection is None


@pytest.mark.parametrize("fail_copy", [False, True])
def test_prototype_copy_retains_staging_and_directory_until_native_cursor_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_copy: bool
) -> None:
    import sqlite3

    directory = tmp_path / "prototypes"
    directory.mkdir()
    monkeypatch.setattr(bootstrap, "_TIER_PROTOTYPE_DIR", directory)
    monkeypatch.setattr(bootstrap, "_TIER_PROTOTYPES", {})
    with closing(sqlite3.connect(":memory:", factory=BackupCursorFault)) as connection:
        connection.execute("CREATE TABLE evidence(value INTEGER)")
        connection.commit()
        assert isinstance(connection, BackupCursorFault)
        source = connection
        source.on_target = True
        source.fail_copy = fail_copy
        with pytest.raises(profiles.NativeConnectionSettlementError) as failed:
            bootstrap._record_tier_prototype(source, ArchiveTier.USER, 1)
        owner = failed.value.owner
        try:
            assert source.retained_cursor is not None and source.retained_cursor.close_attempts == 1
            assert list(directory.glob("*.tmp"))
            with pytest.raises(RuntimeError):
                bootstrap._cleanup_tier_prototype_dir(directory)
            assert directory.exists()
        finally:
            if source.retained_cursor is not None:
                source.retained_cursor.allow_cleanup.set()
            owner.close()
            bootstrap._cleanup_tier_prototype_dir(directory)
        assert source.retained_cursor is not None
        assert source.retained_cursor.close_attempts == 2
        assert not directory.exists()


def test_actual_failed_cursor_survives_discarded_error_and_creator_local_until_retry(tmp_path: Path) -> None:
    import gc
    import weakref

    from polylogue.storage.io_phase_metrics import connect_measured
    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner
    from tests.infra.native_sql_descriptor_probe import selected_file_descriptors
    from tests.infra.sqlite_cursor_settlement import ControlledCursor

    path = tmp_path / "cursor-control.db"
    connection = connect_measured(path)
    connection.execute("CREATE TABLE evidence(value INTEGER)")
    connection.executemany("INSERT INTO evidence VALUES (?)", [(1,), (2,)])
    connection.commit()
    metadata = path.stat()
    identity = metadata.st_dev, metadata.st_ino
    observe_descriptors = Path("/proc/self/fd").is_dir()
    if observe_descriptors:
        assert selected_file_descriptors(identity)
    owner = NativeSQLCustodyOwner(connection)
    cursor = connection.cursor(factory=ControlledCursor)
    assert isinstance(cursor, ControlledCursor)
    cursor.execute("SELECT value FROM evidence ORDER BY value")
    cursor.fetchone()
    cursor.allow_cleanup.clear()
    actual = weakref.ref(cursor)
    completions: list[str] = []
    owner.retain_settlement_callback(lambda: completions.append("complete"))

    def discard_close_error() -> None:
        try:
            owner.close()
        except profiles.NativeConnectionSettlementError:
            pass

    try:
        discard_close_error()
        del cursor
        gc.collect()
        retained = actual()
        assert retained is not None and retained.close_attempts == 1
        assert owner.connection is connection and completions == []
        if observe_descriptors:
            assert selected_file_descriptors(identity)
        retained.allow_cleanup.set()
        owner.close()
        assert retained.close_attempts == 2 and completions == ["complete"]
        if observe_descriptors:
            assert not selected_file_descriptors(identity)
        del retained
        gc.collect()
        assert actual() is None
    finally:
        if retained_cursor := actual():
            retained_cursor.allow_cleanup.set()
        owner.close()


def test_measured_connection_retains_failed_cursor_without_native_owner(tmp_path: Path) -> None:
    import gc
    import weakref

    from polylogue.storage.io_phase_metrics import connect_measured, live_connection_cursors
    from tests.infra.native_sql_descriptor_probe import selected_file_descriptors

    path = tmp_path / "primitive-control.db"
    connection = connect_measured(path)
    connection.execute("CREATE TABLE evidence(value INTEGER)")
    connection.executemany("INSERT INTO evidence VALUES (?)", [(1,), (2,)])
    connection.commit()
    metadata = path.stat()
    identity = metadata.st_dev, metadata.st_ino
    observe_descriptors = Path("/proc/self/fd").is_dir()
    cursor = connection.cursor(factory=ControlledCursor)
    assert isinstance(cursor, ControlledCursor)
    cursor.execute("SELECT value FROM evidence ORDER BY value")
    cursor.fetchone()
    cursor.allow_cleanup.clear()
    actual = weakref.ref(cursor)

    def discard_close_error() -> None:
        try:
            connection.close()
        except BaseExceptionGroup:
            pass

    try:
        discard_close_error()
        del cursor
        gc.collect()
        retained = actual()
        assert retained is not None and retained.close_attempts == 1
        assert live_connection_cursors(connection) == (retained,)
        if observe_descriptors:
            assert selected_file_descriptors(identity)
        retained.allow_cleanup.set()
        connection.close()
        assert retained.close_attempts == 2
        if observe_descriptors:
            assert not selected_file_descriptors(identity)
        del retained
        gc.collect()
        assert actual() is None
    finally:
        if retained_cursor := actual():
            retained_cursor.allow_cleanup.set()
        connection.close()


@pytest.mark.parametrize("route", ["initialize", "constructor", "parent_slot"])
def test_original_temp_archive_survives_pre_parent_or_terminal_close_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    import gc
    import tempfile
    import weakref

    from polylogue.core.sql_settlement import retain_native_sql_lifetimes
    from polylogue.storage.io_phase_metrics import _MeasuredConnection
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore, ArchiveStoreSettlementError
    from polylogue.storage.sqlite.connection_profile import retained_native_sql_owners_for_lifetime
    from polylogue.storage.sqlite.write_lease import write_lease
    from tests.infra.archive_templates import bootstrap_archive_root

    cursors: list[ControlledCursor] = []
    injected = OSError("synthetic original terminal close failure")
    initialize = bootstrap.initialize_archive_tier
    set_authorizer = _MeasuredConnection.set_authorizer
    blocked = True
    selected_file = None

    def block_statement(connection: sqlite3.Connection) -> None:
        cursor = connection.cursor(factory=ControlledCursor)
        assert isinstance(cursor, ControlledCursor)
        cursor.execute("SELECT 1 UNION ALL SELECT 2")
        cursor.fetchone()
        cursor.cleanup_failure = injected
        cursor.allow_cleanup.clear()
        cursors.append(cursor)

    def initialize_with_statement(connection: sqlite3.Connection, tier: ArchiveTier) -> None:
        initialize(connection, tier)
        block_statement(connection)

    def constructor_with_statement(
        connection: sqlite3.Connection,
        callback: Callable[[int, str | None, str | None, str | None, str | None], int] | None,
    ) -> None:
        set_authorizer(connection, callback)
        block_statement(connection)
        raise ValueError("synthetic failure before bootstrap receives its connection")

    def begin() -> tuple[
        Path, weakref.ReferenceType[tempfile.TemporaryDirectory[str]], profiles.NativeSQLCustodyOwner | ArchiveStore
    ]:
        nonlocal selected_file
        directory = tempfile.TemporaryDirectory(dir=tmp_path, prefix="original-archive-")
        root = Path(directory.name)
        reference = weakref.ref(directory)
        owner: profiles.NativeSQLCustodyOwner | ArchiveStore | None = None
        with write_lease("test.original-temp-archive", archive_root=root):
            with retain_native_sql_lifetimes(directory):
                if route == "parent_slot":
                    bootstrap_archive_root(root)
                    owner = ArchiveStore.open_existing(root, read_only=False)
                    owner._hold_replay_publisher_slot()
                    selected_file = owner._replay_publisher_lock_file
                    assert selected_file is not None
                    actual_close = selected_file.close

                    def close_file() -> None:
                        if blocked:
                            raise injected
                        actual_close()

                    monkeypatch.setattr(selected_file, "close", close_file)
                    with pytest.raises(ArchiveStoreSettlementError):
                        owner.close()
                else:
                    monkeypatch.setattr(bootstrap, "_TIER_PROTOTYPES", {})
                    monkeypatch.setattr(bootstrap, "_record_tier_prototype", lambda *_args: None)
                    if route == "initialize":
                        monkeypatch.setattr(bootstrap, "initialize_archive_tier", initialize_with_statement)
                        tier = ArchiveTier.INDEX
                    else:
                        monkeypatch.setattr(_MeasuredConnection, "set_authorizer", constructor_with_statement)
                        tier = ArchiveTier.SOURCE
                    try:
                        bootstrap.initialize_archive_database(root / f"{tier.value}.db", tier, expected_version=1)
                    except profiles.NativeConnectionSettlementError as failure:
                        owner = failure.owner
                    else:
                        pytest.fail("actual retained statement did not block physical cleanup")
        assert owner is not None
        return root, reference, owner

    root, reference, owner = begin()
    # Drop errors and contextmanager tracebacks before observing retention:
    # only the original native obligation may keep the directory alive.
    injected.__traceback__ = injected.__cause__ = injected.__context__ = None
    gc.collect()
    directory = reference()
    try:
        assert directory is not None and root.is_dir()
        assert retained_native_sql_owners_for_lifetime(directory)
        del directory
        if route == "parent_slot":
            assert isinstance(owner, ArchiveStore)
            assert selected_file is not None and not selected_file.closed
            assert owner._replay_publisher_lock_file is selected_file
        else:
            assert isinstance(owner, profiles.NativeSQLCustodyOwner)
            assert owner.connection is not None and cursors[0].close_attempts == 1
        blocked = False
        for cursor in cursors:
            cursor.allow_cleanup.set()
        owner.close()
        if route == "parent_slot":
            assert selected_file is not None and selected_file.closed
        else:
            assert isinstance(owner, profiles.NativeSQLCustodyOwner)
            assert owner.connection is None and cursors[0].close_attempts == 2
        gc.collect()
        assert reference() is None and not root.exists()
    finally:
        blocked = False
        for cursor in cursors:
            cursor.allow_cleanup.set()
        owner.close()
        if remaining := reference():
            remaining.cleanup()
