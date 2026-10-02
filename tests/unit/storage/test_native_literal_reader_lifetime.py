"""Actual incremental readers retire before their original SQL artifacts."""

import asyncio
import sqlite3
from builtins import BaseExceptionGroup
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.storage.sqlite.connection_profile import (
    NativeConnectionSettlementError,
    NativeSQLCustodyOwner,
    retained_native_sql_owners_on_current_thread,
    scratch_connection_context,
)
from tests.infra.native_sql_descriptor_probe import selected_file_descriptors


def _original_owner(connection: sqlite3.Connection) -> NativeSQLCustodyOwner:
    owners = tuple(owner for owner in retained_native_sql_owners_on_current_thread() if owner.connection is connection)
    assert len(owners) == 1
    return owners[0]


@pytest.mark.parametrize("literal", [b"", b"exact \xf0\x9f\x8c\x8d", b"\xff\xfe\x00"])
def test_original_native_text_reader_preserves_literal_bytes_and_refuses_writes(tmp_path: Path, literal: bytes) -> None:
    with scratch_connection_context(prefix="native-text-", filename="cells.db", directory=tmp_path) as connection:
        owner = _original_owner(connection)
        with closing(connection.execute("CREATE TABLE cells(value TEXT NOT NULL)")):
            pass
        with closing(connection.execute("INSERT INTO cells VALUES(CAST(? AS TEXT))", (literal,))):
            pass
        with owner.readonly_blob("cells", "value", 1) as blob:
            assert len(blob) == len(literal)
            assert blob.read() == literal
            blob.seek(0)
            with pytest.raises(sqlite3.OperationalError):
                blob.write(b"")
        assert not owner._incremental_blobs
    assert owner._settled
    assert owner.connection is None


@pytest.mark.parametrize("cancelled", [False, True])
def test_actual_blob_close_failure_retains_original_artifact_until_creator_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancelled: bool
) -> None:
    original_close = NativeSQLCustodyOwner.close_incremental_blob
    cleanup_allowed = False
    selected_owner: NativeSQLCustodyOwner | None = None
    selected_blob: sqlite3.Blob | None = None
    completed: list[str] = []

    def controlled_close(owner: NativeSQLCustodyOwner, blob: sqlite3.Blob) -> None:
        if owner is selected_owner and blob is selected_blob and not cleanup_allowed:
            raise OSError("synthetic actual incremental close fault")
        original_close(owner, blob)

    monkeypatch.setattr(NativeSQLCustodyOwner, "close_incremental_blob", controlled_close)
    try:
        with pytest.raises(NativeConnectionSettlementError):
            with scratch_connection_context(
                prefix="native-blob-fault-", filename="cells.db", directory=tmp_path
            ) as connection:
                selected_owner = _original_owner(connection)
                selected_owner.retain_settlement_callback(lambda: completed.append("settled"))
                with closing(connection.execute("PRAGMA database_list")) as cursor:
                    database = Path(next(row[2] for row in cursor if row[1] == "main"))
                metadata = database.stat()
                identity = metadata.st_dev, metadata.st_ino
                with closing(connection.execute("CREATE TABLE cells(value TEXT NOT NULL)")):
                    pass
                with closing(connection.execute("INSERT INTO cells VALUES('original bytes')")):
                    pass
                with selected_owner.readonly_blob("cells", "value", 1) as blob:
                    selected_blob = blob
                    assert blob.read(4) == b"orig"
                    if cancelled:
                        raise asyncio.CancelledError("synthetic original read cancellation")
        assert selected_owner is not None and selected_blob is not None
        assert database.exists() and database.parent.exists()
        assert selected_owner.connection is connection
        assert selected_owner.close_required and not selected_owner._settled
        assert selected_owner._incremental_blobs == [selected_blob]
        assert len(selected_blob) == len(b"original bytes")
        assert completed == []
        with pytest.raises(RuntimeError):
            selected_owner.require_connection()
        if Path("/proc/self/fd").is_dir():
            assert selected_file_descriptors(identity)
        cleanup_allowed = True
        selected_owner.close()
        assert selected_owner._settled and selected_owner.connection is None
        assert not selected_owner._incremental_blobs
        assert completed == ["settled"]
        assert not database.parent.exists()
        if Path("/proc/self/fd").is_dir():
            assert selected_file_descriptors(identity) == ()
    finally:
        cleanup_allowed = True
        if selected_owner is not None:
            selected_owner.close()


def test_successful_external_parent_close_retires_its_exact_original_blob(tmp_path: Path) -> None:
    completed: list[str] = []
    with scratch_connection_context(
        prefix="native-blob-parent-", filename="cells.db", directory=tmp_path
    ) as connection:
        owner = _original_owner(connection)
        owner.retain_settlement_callback(lambda: completed.append("settled"))
        with closing(connection.execute("PRAGMA database_list")) as cursor:
            database = Path(next(row[2] for row in cursor if row[1] == "main"))
        with closing(connection.execute("CREATE TABLE cells(value TEXT NOT NULL)")):
            pass
        with closing(connection.execute("INSERT INTO cells VALUES('actual parent')")):
            pass
        with owner.readonly_blob("cells", "value", 1) as blob:
            assert blob.read(6) == b"actual"
            connection.close()
            with pytest.raises(sqlite3.ProgrammingError):
                len(blob)
            assert owner._incremental_blobs == [blob]
            owner.close()
            assert owner._settled and owner.connection is None
            assert not owner._incremental_blobs
            assert completed == ["settled"]
        assert not database.parent.exists()
    assert completed == ["settled"]


@pytest.mark.parametrize("close_through_owner", [False, True])
def test_failed_external_parent_close_preserves_actual_native_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, close_through_owner: bool
) -> None:
    from typing import Any

    from polylogue.storage.sqlite import connection_profile
    from tests.infra.sqlite_cursor_settlement import ControlledConnection

    original_factory = connection_profile.connect_measured

    def controlled_factory(database: str | Path, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        return original_factory(database, *args, factory=ControlledConnection, **kwargs)

    monkeypatch.setattr(connection_profile, "connect_measured", controlled_factory)
    owner: NativeSQLCustodyOwner | None = None
    controlled: ControlledConnection | None = None
    completed: list[str] = []
    try:
        with pytest.raises(NativeConnectionSettlementError) as unsettled:
            with scratch_connection_context(
                prefix="native-blob-parent-fault-", filename="cells.db", directory=tmp_path
            ) as connection:
                assert isinstance(connection, ControlledConnection)
                controlled = connection
                owner = _original_owner(connection)
                owner.retain_settlement_callback(lambda: completed.append("settled"))
                with closing(connection.execute("PRAGMA database_list")) as cursor:
                    database = Path(next(row[2] for row in cursor if row[1] == "main"))
                with closing(connection.execute("CREATE TABLE cells(value TEXT NOT NULL)")):
                    pass
                with closing(connection.execute("INSERT INTO cells VALUES('still actual')")):
                    pass
                with owner.readonly_blob("cells", "value", 1) as blob:
                    failure = OSError("synthetic actual parent close fault")
                    controlled.close_failure = failure
                    if close_through_owner:
                        # The original child settles before the actual parent
                        # fails. Context exit must retain only that real failure.
                        owner.close()
                    else:
                        with pytest.raises(OSError) as external:
                            controlled.close()
                        assert external.value is failure
                        assert len(blob) == len(b"still actual")
                        assert blob.read() == b"still actual"
                        assert owner._incremental_blobs == [blob]
                        assert not controlled._native_closed
        assert owner is not None and controlled is not None

        def leaves(error: BaseException) -> list[BaseException]:
            if isinstance(error, BaseExceptionGroup):
                return [leaf for child in error.exceptions for leaf in leaves(child)]
            if isinstance(error, NativeConnectionSettlementError):
                return leaves(error.failure)
            return [error]

        assert leaves(unsettled.value) and all(error is failure for error in leaves(unsettled.value))
        assert owner.connection is controlled and owner.close_required
        assert not owner._incremental_blobs and not owner._settled
        assert completed == [] and database.exists()
        controlled.close_failure = None
        owner.close()
        assert owner._settled and owner.connection is None
        assert completed == ["settled"] and not database.parent.exists()
    finally:
        if controlled is not None:
            controlled.close_failure = None
        if owner is not None:
            owner.close()
