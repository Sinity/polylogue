"""Publication preserves the original lookup, not only its former row."""

import sqlite3
from builtins import BaseExceptionGroup
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.core.enums import AssertionKind, Provider
from polylogue.core.refs import EvidenceRef
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion
from polylogue.storage.sqlite.connection_profile import open_connection
from polylogue.storage.sqlite.reference_seal import ReferenceSealError
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.live_ingest import write_index_session
from tests.infra.reference_sessions import reference_session


@pytest.mark.parametrize("lookup", ["shared", "codex:shared", "codex-session:shared"])
def test_insert_preserves_original_session_alias_resolution(tmp_path: Path, lookup: str) -> None:
    with write_lease("test.reference-alias", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            original = write_index_session(archive, reference_session("shared"))
            archive.save_annotation("original-lookup", "session", lookup, "Retain the original lookup")
            archive.commit()
            if lookup == "shared":
                with pytest.raises(ReferenceSealError):
                    write_index_session(archive, reference_session("shared", provider=Provider.CHATGPT))
                assert archive.resolve_session_id(lookup) == original
                assert archive._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
            else:
                write_index_session(archive, reference_session("shared", provider=Provider.CHATGPT))
                assert archive.resolve_session_id(lookup) == original
            assert archive.get_annotation("original-lookup") is not None


@pytest.mark.parametrize("unrelated_anchors", [0, 96])
def test_batch_reuses_one_durable_reference_census(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unrelated_anchors: int
) -> None:
    from polylogue.storage.sqlite import reference_seal

    census_calls = 0
    assertion_queries: list[str] = []
    original_open = reference_seal.PreparedIndexMutation._open_observer
    original = reference_seal._references_from_user

    def observed(conn: sqlite3.Connection) -> Iterator[str]:
        nonlocal census_calls
        census_calls += 1
        yield from original(conn)

    def open_observer(seal: reference_seal.PreparedIndexMutation, name: str, path: Path) -> sqlite3.Connection:
        connection = original_open(seal, name, path)
        if name == "user":
            connection.set_trace_callback(
                lambda sql: assertion_queries.append(sql) if sql.startswith("SELECT scope_ref,") else None
            )
        return connection

    with write_lease("test.reference-batch", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            existing = write_index_session(archive, reference_session("unrelated"))
            for number in range(unrelated_anchors):
                archive.save_annotation(str(number), "session", existing, "Retained unrelated note")
            archive.commit()
            monkeypatch.setattr(reference_seal, "_references_from_user", observed)
            monkeypatch.setattr(reference_seal.PreparedIndexMutation, "_open_observer", open_observer)
            with archive.index_mutation_scope():
                for number in range(8):
                    write_index_session(archive, reference_session(f"batch-{number}"))
            assert census_calls == 1
            assert len(assertion_queries) == 1
            assert archive._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 9


@pytest.mark.parametrize("block_position", [None, 0])
def test_parent_replacement_preserves_evidence_scoped_to_a_composed_child(
    tmp_path: Path, block_position: int | None
) -> None:
    with write_lease("test.reference-descendant", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            parent = write_index_session(archive, reference_session("parent", messages=(("prefix", "prefix"),)))
            child = write_index_session(
                archive, reference_session("child", parent="parent", messages=(("prefix", "prefix"), ("tail", "tail")))
            )
            message_id = archive._conn.execute(
                "SELECT message_id FROM messages WHERE session_id = ?", (parent,)
            ).fetchone()[0]
            evidence = EvidenceRef(child, message_id, block_position).format()
            with closing(open_connection(tmp_path / "user.db", archive_root=tmp_path)) as user, user:
                upsert_assertion(
                    user,
                    assertion_id="inherited-evidence",
                    target_ref=f"session:{child}",
                    kind=AssertionKind.ANNOTATION,
                    key="inherited-evidence",
                    body_text="Retain the inherited locator",
                    author_kind="user",
                    evidence_refs=(evidence,),
                )
            with pytest.raises(ReferenceSealError):
                write_index_session(archive, reference_session("parent", messages=(("replacement", "replacement"),)))
            assert (
                archive._conn.execute("SELECT message_id FROM messages WHERE session_id = ?", (parent,)).fetchone()[0]
                == message_id
            )


def test_cancelled_active_mutation_rolls_back_and_original_owner_closes(tmp_path: Path) -> None:
    import asyncio
    import threading

    from polylogue.core.compute_cancel import compute_cancel
    from tests.infra.archive_custody_probe import archive_custody_available

    cancelled = threading.Event()
    token = compute_cancel.set(cancelled)
    archive = None
    try:
        with write_lease("test.cancelled-reference-scope", archive_root=tmp_path):
            bootstrap_archive_root(tmp_path)
            archive = ArchiveStore.open_existing(tmp_path, read_only=False)
            try:
                with pytest.raises(asyncio.CancelledError):
                    with archive.index_mutation_scope():
                        write_index_session(archive, reference_session("cancelled-pending"))
                        assert archive._conn.in_transaction
                        cancelled.set()
                        archive._conn.set_progress_handler(lambda: int(cancelled.is_set()), 1)
                        # Admission refuses another mutation. The scope must
                        # still roll back and close on this original owner.
                        write_index_session(archive, reference_session("must-not-start"))
                assert not archive._conn.in_transaction
                assert archive._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
                with pytest.raises(asyncio.CancelledError):
                    archive.delete_sessions(("cancelled-pending",))
            finally:
                archive.close()
                archive = None
        assert archive_custody_available(tmp_path)
    finally:
        compute_cancel.reset(token)
        if archive is not None:
            archive.close()


def test_active_seal_refuses_an_archive_shadow_index(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    with write_lease("test.reference-shadow", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        shadow = tmp_path / "shadow.db"
        with closing(sqlite3.connect(shadow)) as connection, connection:
            connection.execute("CREATE TABLE shadow(value INTEGER)")
        with pytest.raises(ReferenceSealError):
            PreparedIndexMutation(shadow, archive_root=tmp_path)


def test_active_suppression_uses_the_batch_user_observer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.storage.sqlite.archive_tiers import session_suppression

    def unexpected_reader(*args: object, **kwargs: object) -> None:
        raise AssertionError("a sealed batch must borrow its existing User observer")

    with write_lease("test.suppression-observer", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            monkeypatch.setattr(session_suppression, "readonly_connection_context", unexpected_reader)
            with archive.index_mutation_scope() as scope:
                assert scope is not None
                first = scope.suppression_reader()
                for number in range(3):
                    assert scope.suppression_reader() is first
                    write_index_session(archive, reference_session(f"suppression-{number}"))


@pytest.mark.parametrize("terminal", ["commit", "rollback", "close"])
def test_inactive_suppression_reader_retains_exact_scope_until_creator_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal: str
) -> None:
    from polylogue.storage.index_generation import IndexGenerationStore
    from polylogue.storage.io_phase_metrics import connect_measured
    from polylogue.storage.sqlite import reference_seal
    from polylogue.storage.sqlite.connection_profile import (
        NativeConnectionSettlementError,
        NativeSQLCustodyOwner,
        retained_native_sql_owners_for_lifetime,
    )
    from tests.infra.sqlite_cursor_settlement import ControlledCursor

    opened: list[NativeSQLCustodyOwner] = []
    cursors: list[ControlledCursor] = []
    original = reference_seal._open_readonly_owner

    def open_reader(path: Path, **kwargs: object) -> NativeSQLCustodyOwner:
        owner = original(path, **kwargs)  # type: ignore[arg-type]
        opened.append(owner)
        cursor = owner.require_connection().cursor(factory=ControlledCursor)
        cursor.execute("SELECT 1 UNION ALL SELECT 2")
        next(cursor)
        cursor.allow_cleanup.clear()
        cursors.append(cursor)
        return owner

    with write_lease("test.inactive-suppression-owner", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        generation = IndexGenerationStore.for_archive_root(tmp_path).create(source_snapshot="suppression-test")
        destination = reference_seal.IndexMutationDestination.owned_inactive(generation)
        with closing(connect_measured(generation.index_path)) as connection:
            monkeypatch.setattr(reference_seal, "_open_readonly_owner", open_reader)
            with pytest.raises(NativeConnectionSettlementError):
                with destination.mutation_scope(connection) as scope:
                    assert scope.suppression_reader() is scope.suppression_reader()
                    assert len(opened) == 1
                    import threading

                    failures: list[BaseException] = []

                    def foreign_close() -> None:
                        try:
                            scope.close()
                        except BaseException as error:
                            failures.append(error)

                    foreign = threading.Thread(target=foreign_close)
                    foreign.start()
                    foreign.join()
                    assert len(failures) == 1 and isinstance(failures[0], ReferenceSealError)
                    assert scope._active and cursors[0].close_attempts == 0
                    getattr(scope, terminal)()
            assert not scope._active
            assert not connection.in_transaction
            assert cursors[0].close_attempts == 1
            assert retained_native_sql_owners_for_lifetime(scope) == tuple(opened)
            assert opened[0].custody is not None
            cursors[0].allow_cleanup.set()
            scope.close()
            assert cursors[0].close_attempts == 2
            assert retained_native_sql_owners_for_lifetime(scope) == ()
            assert opened[0].connection is None
            assert opened[0].custody is None


def test_missing_canonical_user_cannot_disable_suppression(tmp_path: Path) -> None:
    from polylogue.storage.index_generation import IndexGenerationStore
    from polylogue.storage.io_phase_metrics import connect_measured
    from polylogue.storage.sqlite.archive_tiers.session_suppression import session_write_is_suppressed
    from polylogue.storage.sqlite.reference_seal import IndexMutationDestination

    with write_lease("test.suppression-required-user", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        generation = IndexGenerationStore.for_archive_root(tmp_path).create(source_snapshot="missing-user-test")
        destination = IndexMutationDestination.owned_inactive(generation)
        (tmp_path / "user.db").unlink()
        with closing(connect_measured(generation.index_path)) as connection:
            with pytest.raises(ReferenceSealError):
                with destination.mutation_scope(connection):
                    session_write_is_suppressed(connection, "codex-session:missing-user")


@pytest.mark.parametrize("user_failure", [False, True])
@pytest.mark.parametrize("terminal", ["rollback", "close", "body-failure"])
def test_scope_rollback_fault_still_attempts_user_cleanup_once_and_retains_exact_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, user_failure: bool, terminal: str
) -> None:
    from builtins import BaseExceptionGroup

    from polylogue.storage.index_generation import IndexGenerationStore
    from polylogue.storage.sqlite import reference_seal
    from polylogue.storage.sqlite.connection_profile import (
        NativeSQLCustodyOwner,
        retained_native_sql_owners_for_lifetime,
    )
    from tests.infra.sqlite_cursor_settlement import ControlledConnection, ControlledCursor

    cursors: list[ControlledCursor] = []
    original = reference_seal._open_readonly_owner

    def open_reader(path: Path, **kwargs: object) -> NativeSQLCustodyOwner:
        owner = original(path, **kwargs)  # type: ignore[arg-type]
        cursor = owner.require_connection().cursor(factory=ControlledCursor)
        cursor.execute("SELECT 1 UNION ALL SELECT 2")
        next(cursor)
        if user_failure:
            cursor.allow_cleanup.clear()
        cursors.append(cursor)
        return owner

    with write_lease("test.scope-rollback-fault", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        generation = IndexGenerationStore.for_archive_root(tmp_path).create(source_snapshot="rollback-fault")
        destination = reference_seal.IndexMutationDestination.owned_inactive(generation)
        with closing(sqlite3.connect(generation.index_path, factory=ControlledConnection)) as connection:
            primary = ValueError("synthetic mutation failure")
            rollback = OSError("synthetic actual writer rollback failure")
            monkeypatch.setattr(reference_seal, "_open_readonly_owner", open_reader)
            try:
                with pytest.raises((OSError, BaseExceptionGroup)) as caught:
                    with destination.mutation_scope(connection) as scope:
                        scope.suppression_reader()
                        connection.execute("CREATE TABLE uncommitted(value INTEGER)")
                        connection.rollback_failure = rollback
                        if terminal == "body-failure":
                            raise primary
                        getattr(scope, terminal)()
                assert not scope._active and connection.in_transaction
                assert connection.rollback_attempts == 1 and cursors[0].close_attempts == 1

                def contains(error: BaseException, target: BaseException) -> bool:
                    return error is target or (
                        isinstance(error, BaseExceptionGroup)
                        and any(contains(child, target) for child in error.exceptions)
                    )

                assert contains(caught.value, rollback)
                if terminal == "body-failure":
                    assert contains(caught.value, primary)
                assert bool(retained_native_sql_owners_for_lifetime(scope)) is user_failure
                connection.rollback_failure = None
                cursors[0].allow_cleanup.set()
                scope.close()
                assert not connection.in_transaction and connection.rollback_attempts == 2
                assert cursors[0].close_attempts == (2 if user_failure else 1)
                assert retained_native_sql_owners_for_lifetime(scope) == ()
            finally:
                connection.rollback_failure = None
                for cursor in cursors:
                    cursor.allow_cleanup.set()
                scope.close()


@pytest.mark.parametrize("terminal", ["rollback", "close"])
@pytest.mark.parametrize("user_failure", [False, True])
def test_archive_retains_failed_scope_and_retries_children_only_on_explicit_owner_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal: str, user_failure: bool
) -> None:
    from typing import Any

    from polylogue.storage.index_generation import IndexGenerationStore
    from polylogue.storage.sqlite import connection_profile, reference_seal
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStoreSettlementError
    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner
    from tests.infra.sqlite_cursor_settlement import ControlledConnection, ControlledCursor

    original_connect = connection_profile.connect_measured
    original_reader = reference_seal._open_readonly_owner
    cursors: list[ControlledCursor] = []

    def connect(database: str | Path, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        if str(database) == str(generation.index_path):
            return sqlite3.connect(database, *args, factory=ControlledConnection, **kwargs)
        return original_connect(database, *args, **kwargs)

    def reader(path: Path, **kwargs: Any) -> NativeSQLCustodyOwner:
        owner = original_reader(path, **kwargs)
        cursor = owner.require_connection().cursor(factory=ControlledCursor)
        cursor.execute("SELECT 1 UNION ALL SELECT 2")
        next(cursor)
        if user_failure:
            cursor.allow_cleanup.clear()
        cursors.append(cursor)
        return owner

    with write_lease("test.archive-scope-retirement", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        generation = IndexGenerationStore.for_archive_root(tmp_path).create(source_snapshot="scope-retirement")
        monkeypatch.setattr(connection_profile, "connect_measured", connect)
        monkeypatch.setattr(reference_seal, "_open_readonly_owner", reader)
        archive = generation.open_writer()
        connection = archive._conn
        assert isinstance(connection, ControlledConnection)
        failure = OSError("synthetic actual writer rollback failure")
        try:
            with pytest.raises(BaseExceptionGroup):
                with archive.index_mutation_scope() as scope:
                    scope.suppression_reader()
                    connection.execute("CREATE TABLE uncommitted(value INTEGER)")
                    connection.rollback_failure = failure
                    raise ValueError("synthetic mutation failure")
            assert archive._pending_index_mutation_scope is scope
            assert archive._has_pending_write_sql()
            assert connection.rollback_attempts == 1
            assert cursors[0].close_attempts == 1
            with pytest.raises((OSError, BaseExceptionGroup, ArchiveStoreSettlementError)):
                getattr(archive, terminal)()
            assert connection.rollback_attempts == 2
            assert cursors[0].close_attempts == (2 if user_failure else 1)
            assert archive._pending_index_mutation_scope is scope
            assert archive._owned_index_connection is connection
            connection.rollback_failure = None
            cursors[0].allow_cleanup.set()
            getattr(archive, terminal)()
            assert archive._pending_index_mutation_scope is None
            assert connection.rollback_attempts == 3
            assert cursors[0].close_attempts == (3 if user_failure else 1)
        finally:
            connection.rollback_failure = None
            for cursor in cursors:
                cursor.allow_cleanup.set()
            archive.close()
