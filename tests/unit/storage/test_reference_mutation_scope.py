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
from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError, open_connection
from polylogue.storage.sqlite.reference_seal import ReferenceSealError
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.live_ingest import write_index_session
from tests.infra.reference_sessions import reference_session


def test_prepared_inactive_scope_writes_only_its_captured_owned_generation(tmp_path: Path) -> None:
    from polylogue.storage.index_generation import IndexGenerationStore
    from polylogue.storage.sqlite.reference_seal import IndexMutationDestination, PreparedIndexMutation

    with write_lease("test.prepare-inactive-destination", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        generation = IndexGenerationStore.for_archive_root(tmp_path).create(source_snapshot="prepared-destination")
    destination = IndexMutationDestination.owned_inactive(generation)
    with PreparedIndexMutation(Path(generation.index_path), archive_root=tmp_path, destination=destination) as seal:
        with write_lease("test.publish-inactive-destination", archive_root=tmp_path):
            with ArchiveStore.open_owned_inactive_generation(
                Path(generation.index_path).parent,
                generation_id=generation.generation_id,
                owner_id=generation.owner_id,
            ) as archive:
                with archive.index_mutation_scope(prepared_seal=seal):
                    archive._conn.execute("CREATE TABLE captured_destination(value INTEGER)")
                    archive._conn.execute("INSERT INTO captured_destination VALUES (1)")
                assert archive._conn.execute("SELECT value FROM captured_destination").fetchone()[0] == 1
            with ArchiveStore.open_existing(tmp_path, read_only=False) as active:
                assert (
                    active._conn.execute("SELECT 1 FROM sqlite_schema WHERE name='captured_destination'").fetchone()
                    is None
                )
                with pytest.raises(ReferenceSealError):
                    with active.index_mutation_scope(prepared_seal=seal):
                        pytest.fail("an inactive seal admitted the active Index")


@pytest.mark.parametrize("failed_resource", ["payload", "cursor"])
def test_publication_exclusion_survives_failed_original_owner_cleanup(tmp_path: Path, failed_resource: str) -> None:
    import errno
    import fcntl
    import os

    from polylogue.storage.index_generation import ActiveWriterLease
    from polylogue.storage.sqlite.connection_profile import (
        native_sql_children,
        retained_native_settlement_owners_on_current_thread,
    )
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
    from tests.infra.sqlite_cursor_settlement import ControlledCursor

    with write_lease("test.publication-exclusion", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        seal = PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path)
        entry = tuple(native_sql_children(seal))
        exclusion = ActiveWriterLease(tmp_path)
        exclusion.acquire()
        cursor = seal.observer("user").cursor(factory=ControlledCursor)
        cursor.execute("SELECT 1 UNION ALL SELECT 2")
        next(cursor)
        fault = OSError(errno.EIO, "synthetic publication cleanup failure")
        payload_attempts = 0
        payload_fails = failed_resource == "payload"

        def close_payload() -> None:
            nonlocal payload_attempts
            payload_attempts += 1
            if payload_fails:
                raise fault

        if failed_resource == "cursor":
            cursor.cleanup_failure = fault
            cursor.allow_cleanup.clear()
        seal.retain_publication_lifetime(exclusion, close_payload)
        probe = os.open(exclusion.path, os.O_RDWR)
        try:
            with pytest.raises((OSError, NativeConnectionSettlementError)):
                seal.close()
            assert exclusion.held and not seal._closed and seal.publication_lifetime_bound
            assert retained_native_settlement_owners_on_current_thread(entry) == (seal,)
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            payload_fails = False
            cursor.allow_cleanup.set()
            seal.close()
            assert not exclusion.held and seal._closed and seal.publication_lifetime_bound
            assert payload_attempts == (2 if failed_resource == "payload" else 1)
            assert cursor.close_attempts == (2 if failed_resource == "cursor" else 1)
            assert retained_native_settlement_owners_on_current_thread(entry) == ()
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            payload_fails = False
            cursor.allow_cleanup.set()
            seal.close()
            os.close(probe)


@pytest.mark.parametrize("reuse_slot", [False, True])
def test_publication_parent_retains_uncertain_exclusion_close_without_retrying_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reuse_slot: bool
) -> None:
    import errno
    import fcntl
    import os

    from polylogue.storage.index_generation import ActiveWriterLease, ActiveWriterLeaseSettlementError
    from polylogue.storage.sqlite.connection_profile import (
        native_sql_children,
        retained_native_settlement_owners_on_current_thread,
    )
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    with write_lease("test.publication-fd-settlement", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        seal = PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path)
        entry = tuple(native_sql_children(seal))
        exclusion = ActiveWriterLease(tmp_path)
        exclusion.acquire()
        descriptor = exclusion._fd
        assert descriptor is not None
        original_close = os.close
        fault = OSError(errno.EIO, "synthetic pre-effect exclusion close failure")
        attempts = 0

        def close(fd: int) -> None:
            nonlocal attempts
            if fd == descriptor:
                attempts += 1
                raise fault
            original_close(fd)

        seal.retain_publication_lifetime(exclusion, lambda: None)
        probe = os.open(exclusion.path, os.O_RDWR)
        replacement_fd: int | None = None
        original_retired = False
        try:
            monkeypatch.setattr(os, "close", close)
            for _ in range(2):
                with pytest.raises(ActiveWriterLeaseSettlementError) as caught:
                    seal.close()
                assert caught.value.failure is fault
                assert exclusion.held and not seal._closed
                assert retained_native_settlement_owners_on_current_thread(entry) == (seal,)
                with pytest.raises(BlockingIOError):
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert attempts == 1
            # Explicitly settle the known original descriptor. A retry only
            # verifies its retirement; it never closes a recycled numeric slot.
            original_close(descriptor)
            original_retired = True
            if reuse_slot:
                replacement_fd = os.open(tmp_path / "other-description", os.O_CREAT | os.O_RDWR, 0o600)
                os.dup2(replacement_fd, descriptor)
            seal.close()
            assert seal._closed and not exclusion.held
            assert retained_native_settlement_owners_on_current_thread(entry) == ()
            if reuse_slot:
                assert replacement_fd is not None
                assert os.fstat(descriptor).st_ino == os.fstat(replacement_fd).st_ino
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            monkeypatch.setattr(os, "close", original_close)
            if exclusion.held and not original_retired:
                original_close(descriptor)
            seal.close()
            if replacement_fd is not None:
                original_close(descriptor)
                if replacement_fd != descriptor:
                    original_close(replacement_fd)
            original_close(probe)


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

    def observed(conn: sqlite3.Connection) -> Iterator[reference_seal._ReferenceAnchor]:
        nonlocal census_calls
        census_calls += 1
        yield from original(conn)

    def open_observer(seal: reference_seal.PreparedIndexMutation, name: str, path: Path) -> sqlite3.Connection:
        connection = original_open(seal, name, path)
        if name == "user":
            connection.set_trace_callback(
                lambda sql: (
                    assertion_queries.append(sql) if sql.startswith("SELECT assertion_id, kind, scope_ref,") else None
                )
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
    from polylogue.storage.sqlite.connection_profile import _open_readonly_owner

    original = _open_readonly_owner

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
    from polylogue.storage.sqlite.connection_profile import _open_readonly_owner

    original = _open_readonly_owner

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
    from polylogue.storage.sqlite import reference_seal
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStoreSettlementError
    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner, _open_readonly_owner
    from tests.infra.sqlite_cursor_settlement import ControlledConnection, ControlledCursor, control_archive_connections

    original_reader = _open_readonly_owner
    cursors: list[ControlledCursor] = []

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
        control_archive_connections(monkeypatch, generation.index_path)
        monkeypatch.setattr(reference_seal, "_open_readonly_owner", reader)
        archive = ArchiveStore.open_owned_inactive_generation(
            Path(generation.index_path).parent, generation_id=generation.generation_id, owner_id=generation.owner_id
        )
        connection = archive._conn
        failure = OSError("synthetic actual writer rollback failure")
        try:
            assert isinstance(connection, ControlledConnection)
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
            with pytest.raises(ArchiveStoreSettlementError):
                archive.commit()
            with pytest.raises(ArchiveStoreSettlementError):
                archive.write_raw_payload(
                    provider=Provider.CODEX, source_path="refused-new-work", acquired_at_ms=0, payload=b"new"
                )
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
            if isinstance(connection, ControlledConnection):
                connection.rollback_failure = None
            for cursor in cursors:
                cursor.allow_cleanup.set()
            archive.close()


def test_foreign_active_scope_cannot_be_settled_by_another_archive(tmp_path: Path) -> None:
    with write_lease("test.exact-scope-cleanup", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        first = ArchiveStore.open_existing(tmp_path, read_only=False)
        second = ArchiveStore.open_existing(tmp_path, read_only=False)
        try:
            with first.index_mutation_scope() as scope:
                first._conn.execute("CREATE TABLE exact_scope_owner(value INTEGER)")
                with pytest.raises(ReferenceSealError):
                    second.rollback()
                assert scope._active and first._conn.in_transaction
                assert first._pending_index_mutation_scope is scope
            assert first._conn.execute("SELECT COUNT(*) FROM exact_scope_owner").fetchone()[0] == 0
        finally:
            first.close()
            second.close()


@pytest.mark.parametrize("tier", ["source", "user", "vector"])
def test_explicit_archive_rollback_failure_is_not_repeated_by_scope_unwind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tier: str
) -> None:
    import errno

    from polylogue.core.storage_faults import StorageFaultKind, storage_fault_kind
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStoreSettlementError
    from tests.infra.sqlite_cursor_settlement import ControlledConnection, control_archive_connections

    with write_lease("test.archive-tier-rollback", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        control_archive_connections(monkeypatch, tmp_path / "index.db", tmp_path / "source.db", tmp_path / "user.db")
        archive = ArchiveStore.open_existing(tmp_path, read_only=False)
        child = None
        try:
            index = archive._conn
            assert isinstance(index, ControlledConnection)
            if tier == "source":
                child = archive.source_connection
            elif tier == "user":
                child = archive._open_user_write_connection()
            else:
                child = sqlite3.connect(":memory:", factory=ControlledConnection)
                archive.operation_vector_connection = child
            assert isinstance(child, ControlledConnection)
            child.execute("BEGIN")
            fault = OSError(errno.EIO, "synthetic actual tier rollback failure")
            with pytest.raises(OSError) as caught:
                with archive.index_mutation_scope() as scope:
                    index.execute("CREATE TABLE discarded_tier_batch(value INTEGER)")
                    child.rollback_failure = fault
                    archive.rollback()
            assert caught.value is fault
            assert child.rollback_attempts == 1 and index.rollback_attempts == 1
            assert child.in_transaction and archive._pending_index_mutation_scope is scope
            assert storage_fault_kind(caught.value) is StorageFaultKind.IO
            with pytest.raises(ArchiveStoreSettlementError):
                archive.commit()
            assert child.rollback_attempts == 1
            child.rollback_failure = None
            archive.rollback()
            assert child.rollback_attempts == 2 and index.rollback_attempts == 1
            assert not child.in_transaction and archive._pending_index_mutation_scope is None
        finally:
            if isinstance(child, ControlledConnection):
                child.rollback_failure = None
            archive.close()


@pytest.mark.parametrize("targets", [("user",), ("user", "scratch")])
def test_active_seal_attempts_each_native_child_once_and_keeps_typed_failures(
    tmp_path: Path, targets: tuple[str, ...]
) -> None:
    import errno

    from polylogue.core.storage_faults import StorageFaultKind, storage_fault_kind
    from polylogue.storage.sqlite.connection_profile import (
        native_sql_children,
        retained_native_settlement_owners_on_current_thread,
    )
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
    from tests.infra.sqlite_cursor_settlement import ControlledCursor

    with write_lease("test.seal-terminal-attempts", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        seal = PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path)
        entry = tuple(native_sql_children(seal))
        assert retained_native_settlement_owners_on_current_thread(entry) == ()
        cursors = []
        faults = []
        assert seal._scratch_directory is not None
        directory = Path(seal._scratch_directory.name)
        try:
            for name in targets:
                connection = seal._scratch if name == "scratch" else seal.observer(name)
                cursor = connection.cursor(factory=ControlledCursor)
                cursor.execute("SELECT 1 UNION ALL SELECT 2")
                next(cursor)
                fault = OSError(errno.EIO, f"synthetic {name} cursor settlement failure")
                cursor.cleanup_failure = fault
                cursor.allow_cleanup.clear()
                cursors.append(cursor)
                faults.append(fault)
            primary = ValueError("synthetic mutation failure")
            with pytest.raises(BaseExceptionGroup) as caught:
                with seal:
                    raise primary
            assert caught.value.exceptions[0] is primary
            assert storage_fault_kind(caught.value) is StorageFaultKind.IO
            cleanup = caught.value.exceptions[1]
            failures = cleanup.exceptions if isinstance(cleanup, BaseExceptionGroup) else (cleanup,)
            assert len(failures) == len(faults)
            for failure, fault in zip(failures, faults, strict=True):
                assert isinstance(failure, NativeConnectionSettlementError)
                assert failure.owner._terminal_parent is seal
                assert isinstance(failure.failure, BaseExceptionGroup)
                assert failure.failure.exceptions == (fault,)
            assert [cursor.close_attempts for cursor in cursors] == [1] * len(targets)
            assert not seal._closed
            with pytest.raises(ReferenceSealError):
                seal.observer("user")
            with pytest.raises(ReferenceSealError):
                seal.validate_observers_current()
            assert directory.exists() == ("scratch" in targets)
            # Even entry owners whose own close was never attempted select the
            # complete terminal parent after the first sibling fails.
            assert retained_native_settlement_owners_on_current_thread(entry) == (seal,)
            children = native_sql_children(seal)
            assert children and all(owner._parent_cleanup_requested for owner in children)
            for cursor in cursors:
                cursor.allow_cleanup.set()
            seal.close()
            assert [cursor.close_attempts for cursor in cursors] == [2] * len(targets)
            assert seal._closed and not directory.exists() and not native_sql_children(seal)
            assert retained_native_settlement_owners_on_current_thread(entry) == ()
        finally:
            for cursor in cursors:
                cursor.allow_cleanup.set()
            seal.close()


def test_archive_exit_keeps_mutation_and_native_cleanup_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno

    from polylogue.core.storage_faults import StorageFaultKind, storage_fault_kind
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStoreSettlementError
    from polylogue.storage.sqlite.connection_profile import native_sql_children
    from tests.infra.sqlite_cursor_settlement import ControlledConnection, control_archive_connections

    with write_lease("test.archive-exit-fault", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        control_archive_connections(monkeypatch, tmp_path / "index.db")
        archive = ArchiveStore.open_existing(tmp_path, read_only=False)
        connection = archive._conn
        try:
            assert isinstance(connection, ControlledConnection)
            primary = ValueError("synthetic mutation failure")
            fault = OSError(errno.EIO, "synthetic actual connection close failure")
            connection.close_failure = fault
            with pytest.raises(BaseExceptionGroup) as caught:
                with archive:
                    raise primary
            assert caught.value.exceptions[0] is primary
            cleanup = caught.value.exceptions[1]
            assert isinstance(cleanup, ArchiveStoreSettlementError) and cleanup.store is archive
            assert storage_fault_kind(caught.value) is StorageFaultKind.IO
            assert connection.close_attempts == 1 and native_sql_children(archive)
            connection.close_failure = None
            archive.close()
            assert connection.close_attempts == 2 and not native_sql_children(archive)
        finally:
            if isinstance(connection, ControlledConnection):
                connection.close_failure = None
            archive.close()


def test_archive_construction_failure_exposes_unsettled_store_and_original_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno
    from typing import Any

    from polylogue.core.storage_faults import StorageFaultKind, storage_fault_kind
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStoreSettlementError
    from tests.infra.sqlite_cursor_settlement import ControlledConnection, control_archive_connections

    with write_lease("test.archive-constructor-fault", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        control_archive_connections(monkeypatch, tmp_path / "index.db")
        primary = ValueError("synthetic initialization failure")
        fault = OSError(errno.EIO, "synthetic initialization cleanup failure")
        original = ArchiveStore._initialize_store
        stores = []

        def initialize(store: ArchiveStore, *args: Any, **kwargs: Any) -> None:
            original(store, *args, **kwargs)
            stores.append(store)
            assert isinstance(store._conn, ControlledConnection)
            store._conn.close_failure = fault
            raise primary

        monkeypatch.setattr(ArchiveStore, "_initialize_store", initialize)
        try:
            with pytest.raises(BaseExceptionGroup) as caught:
                ArchiveStore.open_existing(tmp_path, read_only=False)
            assert caught.value.exceptions[0] is primary
            cleanup = caught.value.exceptions[1]
            assert isinstance(cleanup, ArchiveStoreSettlementError) and cleanup.store is stores[0]
            assert isinstance(cleanup.failure, NativeConnectionSettlementError)
            assert cleanup.failure.failure is fault
            assert storage_fault_kind(caught.value) is StorageFaultKind.IO
            assert isinstance(stores[0]._conn, ControlledConnection)
            assert stores[0]._conn.close_attempts == 1
        finally:
            for store in stores:
                if isinstance(store._owned_index_connection, ControlledConnection):
                    store._owned_index_connection.close_failure = None
                store.close()


def test_seal_construction_failure_retains_primary_and_unsettled_native_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno

    from polylogue.core.storage_faults import StorageFaultKind, storage_fault_kind
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
    from tests.infra.sqlite_cursor_settlement import ControlledCursor

    with write_lease("test.seal-constructor-fault", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        primary = ValueError("synthetic reference preparation failure")
        fault = OSError(errno.EIO, "synthetic observer cleanup failure")
        original = PreparedIndexMutation._read_resolved_references
        seals = []
        cursors = []

        def read_references(seal: PreparedIndexMutation) -> None:
            original(seal)
            seals.append(seal)
            cursor = seal.observer("user").cursor(factory=ControlledCursor)
            cursors.append(cursor)
            cursor.execute("SELECT 1 UNION ALL SELECT 2")
            next(cursor)
            cursor.cleanup_failure = fault
            cursor.allow_cleanup.clear()
            raise primary

        monkeypatch.setattr(PreparedIndexMutation, "_read_resolved_references", read_references)
        try:
            with pytest.raises(BaseExceptionGroup) as caught:
                PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path)
            assert caught.value.exceptions[0] is primary
            cleanup = caught.value.exceptions[1]
            assert isinstance(cleanup, NativeConnectionSettlementError)
            assert cleanup.owner._terminal_parent is seals[0]
            assert isinstance(cleanup.failure, BaseExceptionGroup)
            assert cleanup.failure.exceptions == (fault,)
            assert storage_fault_kind(caught.value) is StorageFaultKind.IO
            assert cursors[0].close_attempts == 1 and not seals[0]._closed
        finally:
            for cursor in cursors:
                cursor.allow_cleanup.set()
            for seal in seals:
                seal.close()


@pytest.mark.parametrize("route", ["custom_cursor", "executemany", "executescript", "native_context"])
def test_known_source_permit_rejects_inherited_child_using_preexisting_handle(tmp_path: Path, route: str) -> None:
    import asyncio

    from polylogue.storage.sqlite.connection_profile import open_source_tier_write_connection
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
    from polylogue.storage.sqlite.write_lease import async_write_lease

    with write_lease("test.source-bootstrap", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        connection = open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)
        connection.execute("CREATE TABLE authority_control (key TEXT PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO authority_control VALUES ('retained', 'original')")
        connection.commit()

    async def scenario() -> None:
        async with async_write_lease("test.source-permit", archive_root=tmp_path):
            with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
                permit = seal.prepare_known_tier_mutation(
                    "authority_control", ("value",), (), tier="source", key_column="key"
                )
                with permit.hold_authority():

                    async def child() -> None:
                        if route == "custom_cursor":
                            with closing(connection.cursor(factory=sqlite3.Cursor)) as cursor:
                                cursor.execute("UPDATE authority_control SET value = 'wrong'")
                        elif route == "executemany":
                            connection.executemany("UPDATE authority_control SET value = ?", [("wrong",)])
                        elif route == "executescript":
                            connection.executescript("UPDATE authority_control SET value = 'wrong'; COMMIT;")
                        else:
                            with connection:
                                connection.execute("UPDATE authority_control SET value = 'wrong'")

                    with pytest.raises(sqlite3.DatabaseError):
                        await asyncio.create_task(child())
                    connection.rollback()
                    assert connection.execute("SELECT value FROM authority_control").fetchone()[0] == "original"

    try:
        asyncio.run(scenario())
        # The same persistent handle can belong to a later legitimate owner;
        # its constructor's retired lease is not permanent SQL authority.
        with write_lease("test.source-successor", archive_root=tmp_path):
            connection.execute("UPDATE authority_control SET value = 'successor'")
            connection.commit()
            assert connection.execute("SELECT value FROM authority_control").fetchone()[0] == "successor"
    finally:
        connection.close()


@pytest.mark.parametrize("persistent", [False, True])
def test_inactive_scope_reader_settles_once_through_its_actual_store_census(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, persistent: bool
) -> None:
    from typing import Any

    from polylogue.core.sql_settlement import retained_native_sql_owners
    from polylogue.storage.index_generation import IndexGenerationStore
    from polylogue.storage.sqlite import reference_seal
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStoreSettlementError
    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner, native_sql_children
    from polylogue.storage.sqlite.write_lease import current_sql_custody
    from tests.infra.sqlite_cursor_settlement import ControlledCursor

    class ReaderCursor(ControlledCursor):
        def close(self) -> None:
            try:
                super().close()
            finally:
                if not persistent:
                    self.allow_cleanup.set()

    from polylogue.storage.sqlite.connection_profile import _open_readonly_owner

    original = _open_readonly_owner
    cursors: list[ReaderCursor] = []

    def reader(path: str | Path, **kwargs: Any) -> NativeSQLCustodyOwner:
        owner = original(path, **kwargs)
        cursor = owner.require_connection().cursor(factory=ReaderCursor)
        cursor.execute("SELECT 1 UNION ALL SELECT 2")
        next(cursor)
        cursor.allow_cleanup.clear()
        cursors.append(cursor)
        return owner

    with write_lease("test.scope-reader-census", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        generation = IndexGenerationStore.for_archive_root(tmp_path).create(source_snapshot="reader-census")
        archive = ArchiveStore.open_owned_inactive_generation(
            Path(generation.index_path).parent, generation_id=generation.generation_id, owner_id=generation.owner_id
        )
        monkeypatch.setattr(reference_seal, "_open_readonly_owner", reader)
        try:
            with pytest.raises(BaseExceptionGroup):
                with archive.index_mutation_scope() as scope:
                    scope.suppression_reader()
                    owner = scope._user_owner
                    assert owner is not None and owner._terminal_parent is archive
                    assert owner.custody is None
                    assert archive._sql_custody is current_sql_custody()
                    raise ValueError("synthetic failure after acquiring actual User reader")
            assert cursors[0].close_attempts == 1
            captured = retained_native_sql_owners()
            assert captured == (archive,)
            if persistent:
                with pytest.raises(ArchiveStoreSettlementError):
                    captured[0].close()
                assert cursors[0].close_attempts == 2
                assert archive._pending_index_mutation_scope is scope
                assert retained_native_sql_owners() == (archive,)
                cursors[0].allow_cleanup.set()
            captured[0].close()
            assert cursors[0].close_attempts == (3 if persistent else 2)
            assert scope._user_owner is None
            assert not native_sql_children(archive)
            assert retained_native_sql_owners() == ()
        finally:
            for cursor in cursors:
                cursor.allow_cleanup.set()
            archive.close()


def test_healthy_inactive_scope_reader_retires_parent_and_lifetime_after_commit(tmp_path: Path) -> None:
    from polylogue.storage.index_generation import IndexGenerationStore
    from polylogue.storage.sqlite.connection_profile import native_sql_children, retained_native_sql_owners_for_lifetime

    with write_lease("test.scope-reader-retirement", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        generation = IndexGenerationStore.for_archive_root(tmp_path).create(source_snapshot="reader-retirement")
        archive = ArchiveStore.open_owned_inactive_generation(
            Path(generation.index_path).parent, generation_id=generation.generation_id, owner_id=generation.owner_id
        )
        try:
            with archive.index_mutation_scope() as scope:
                connection = scope.suppression_reader()
                assert connection is not None
                assert connection.execute("SELECT 1").fetchone()[0] == 1
                owner = scope._user_owner
                assert owner is not None and owner._terminal_parent is archive
                assert owner in native_sql_children(archive)
            assert owner._settled and owner._terminal_parent is None
            assert scope._user_owner is None and scope._user_admission_custody is None
            assert retained_native_sql_owners_for_lifetime(scope) == ()
            assert owner not in native_sql_children(archive)
        finally:
            archive.close()


def test_inactive_reader_initializer_failure_retains_original_store_before_sql(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typing import Any

    from polylogue.core.sql_settlement import retained_native_sql_owners
    from polylogue.storage.index_generation import IndexGenerationStore
    from polylogue.storage.sqlite import connection_profile
    from polylogue.storage.sqlite.write_lease import current_sql_custody
    from tests.infra.sqlite_cursor_settlement import ControlledConnection, ControlledCursor

    primary = ValueError("synthetic reader initializer failure")
    cursors: list[ControlledCursor] = []
    from polylogue.storage.io_phase_metrics import connect_measured

    original = connect_measured

    class InitializerConnection(ControlledConnection):
        def execute(self, sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
            assert archive._sql_custody is current_sql_custody()
            assert retained_native_sql_owners() == (archive,)
            cursor = self.cursor(factory=ControlledCursor)
            cursor.execute("SELECT 1 UNION ALL SELECT 2")
            next(cursor)
            cursor.allow_cleanup.clear()
            cursors.append(cursor)
            raise primary

    def connect(database: str | Path, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        if str(database).endswith("/user.db?mode=ro"):
            return sqlite3.connect(database, *args, factory=InitializerConnection, **kwargs)
        return original(database, *args, **kwargs)

    with write_lease("test.scope-reader-initializer", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        generation = IndexGenerationStore.for_archive_root(tmp_path).create(source_snapshot="reader-initializer")
        archive = ArchiveStore.open_owned_inactive_generation(
            Path(generation.index_path).parent, generation_id=generation.generation_id, owner_id=generation.owner_id
        )
        monkeypatch.setattr(connection_profile, "connect_measured", connect)
        try:
            with pytest.raises(NativeConnectionSettlementError) as caught:
                with archive.index_mutation_scope() as scope:
                    scope.suppression_reader()
            assert caught.value.__cause__ is primary
            assert caught.value.owner is scope._user_owner
            assert caught.value.owner._terminal_parent is archive
            assert archive._pending_index_mutation_scope is scope
            assert cursors[0].close_attempts == 1
            captured = retained_native_sql_owners()
            assert captured == (archive,)
            cursors[0].allow_cleanup.set()
            captured[0].close()
            assert cursors[0].close_attempts == 2
            assert scope._user_owner is None and retained_native_sql_owners() == ()
        finally:
            for cursor in cursors:
                cursor.allow_cleanup.set()
            archive.close()


def test_store_terminal_parent_request_refuses_work_before_first_native_close(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStoreSettlementError
    from polylogue.storage.sqlite.connection_profile import native_sql_children, request_native_sql_parent_cleanup

    with write_lease("test.store-terminal-admission", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        archive = ArchiveStore.open_existing(tmp_path, read_only=False)
        try:
            children = native_sql_children(archive)
            assert children and not any(child.close_required for child in children)
            request_native_sql_parent_cleanup(archive)
            with pytest.raises(ArchiveStoreSettlementError):
                archive.commit()
            with pytest.raises(ArchiveStoreSettlementError):
                archive.write_raw_payload(
                    provider=Provider.CODEX, source_path="refused-terminal-parent", acquired_at_ms=0, payload=b"new"
                )
            assert not any(child.close_required for child in children)
            archive.close()
            assert not native_sql_children(archive)
        finally:
            archive.close()


def test_owned_inactive_scope_without_store_parent_preserves_reader_custody(tmp_path: Path) -> None:
    from polylogue.storage.index_generation import IndexGenerationStore
    from polylogue.storage.sqlite.reference_seal import IndexMutationDestination
    from polylogue.storage.sqlite.write_lease import current_sql_custody

    with write_lease("test.scope-reader-no-store-parent", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        generation = IndexGenerationStore.for_archive_root(tmp_path).create(source_snapshot="reader-no-parent")
        destination = IndexMutationDestination.owned_inactive(generation)
        with closing(open_connection(generation.index_path)) as connection:
            with destination.mutation_scope(connection) as scope:
                reader = scope.suppression_reader()
                assert reader is not None and reader.execute("SELECT 1").fetchone()[0] == 1
                owner = scope._user_owner
                assert owner is not None and owner._terminal_parent is None
                assert owner.custody is current_sql_custody()
            assert owner._settled and owner.custody is None and scope._user_owner is None


@pytest.mark.parametrize(
    "lifecycle", [None, AssertionKind.SUPPRESSION, AssertionKind.EXCISION_RECORD, AssertionKind.EXCISION_REQUEST]
)
@pytest.mark.parametrize("ordinary_anchor", [False, True])
def test_bound_deletion_preserves_audit_identity_and_protects_ordinary_user_anchors(
    tmp_path: Path, lifecycle: AssertionKind | None, ordinary_anchor: bool
) -> None:
    from polylogue.operations.bindings import runtime_operation_binding
    from polylogue.operations.mutation_actuators import SessionDeleteActuator, SessionDeleteArgs
    from polylogue.operations.mutation_transaction import MutationPrincipal, OperationExecutor
    from polylogue.storage.sqlite.write_lease import permitted_session_removals

    with write_lease("test.bound-reference-removal", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            target = write_index_session(archive, reference_session("removal-target"))
            survivor = write_index_session(archive, reference_session("removal-survivor"))
            archive.commit()
            if lifecycle is not None:
                with closing(open_connection(tmp_path / "user.db")) as user:
                    upsert_assertion(
                        user,
                        assertion_id="lifecycle-removal",
                        target_ref=f"session:{target}",
                        kind=lifecycle,
                        value={},
                        author_ref="user:local",
                        author_kind="user",
                        now_ms=1,
                    )
                    user.commit()
            if ordinary_anchor:
                archive.save_annotation("protected-removal", "session", target, "Retained ordinary anchor")
                archive.commit()
            binding = runtime_operation_binding(SessionDeleteActuator())
            principal = MutationPrincipal("test", frozenset({"archive.delete_session"}), "api", "write")
            executor = OperationExecutor.for_archive_root(tmp_path)
            args = SessionDeleteArgs(archive=archive, session_ids=(target,))
            older = executor.prepare_bound_for_archive(binding, args, principal, archive_root=tmp_path)
            # The earlier preview remains historical and must retain its exact
            # target/digest even when a later authorized deletion applies.
            with closing(sqlite3.connect(tmp_path / "audit.db")) as audit:
                before = audit.execute(
                    "SELECT * FROM operation_preview_targets WHERE preview_id = ?", (older.preview_ref,)
                ).fetchall()
            current = executor.prepare_bound_for_archive(binding, args, principal, archive_root=tmp_path)
            authorization = executor.authorize_bound(binding, current, principal)
            if ordinary_anchor:
                with pytest.raises(ReferenceSealError):
                    executor.execute_bound(binding, current, authorization, args)
                assert archive.stored_session_ids((target,)) == (target,)
            else:
                receipt = executor.execute_bound(binding, current, authorization, args)
                assert receipt.status == "applied" and receipt.affected_count == 1
                assert archive.stored_session_ids((target,)) == ()
            assert archive.stored_session_ids((survivor,)) == (survivor,)
            assert permitted_session_removals(archive_root=tmp_path) == frozenset()
            with closing(sqlite3.connect(tmp_path / "audit.db")) as audit:
                assert (
                    audit.execute(
                        "SELECT * FROM operation_preview_targets WHERE preview_id = ?", (older.preview_ref,)
                    ).fetchall()
                    == before
                )
                assert audit.execute("SELECT target_ref FROM operation_targets").fetchall() == [(f"session:{target}",)]


@pytest.mark.asyncio
async def test_bound_removal_permission_stays_with_actual_apply_task_and_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio
    import threading

    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator
    from polylogue.operations.bindings import runtime_operation_binding
    from polylogue.operations.mutation_actuators import SessionDeleteActuator, SessionDeleteArgs
    from polylogue.operations.mutation_transaction import (
        MutationPlan,
        MutationPrincipal,
        MutationReceipt,
        OperationExecutor,
    )
    from polylogue.storage.sqlite.write_lease import (
        UnleasedWriteError,
        bind_write_lease_thread,
        grant_write_lease_thread,
        permitted_session_removals,
    )

    await asyncio.to_thread(bootstrap_archive_root, tmp_path)
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    observations: list[object] = []
    child_tasks: list[asyncio.Task[frozenset[str]]] = []
    original = SessionDeleteActuator.apply

    def apply(actuator: SessionDeleteActuator, plan: MutationPlan, args: SessionDeleteArgs) -> MutationReceipt:
        assert permitted_session_removals(archive_root=tmp_path) == frozenset(args.session_ids)
        grant = grant_write_lease_thread()

        def borrowed_thread() -> None:
            try:
                bind_write_lease_thread(grant)
                observations.append(permitted_session_removals(archive_root=tmp_path))
            finally:
                grant.complete()

        worker = threading.Thread(target=borrowed_thread)
        worker.start()
        worker.join()

        async def inherited_task() -> frozenset[str]:
            return permitted_session_removals(archive_root=tmp_path)

        child_tasks.append(asyncio.create_task(inherited_task()))
        return original(actuator, plan, args)

    monkeypatch.setattr(SessionDeleteActuator, "apply", apply)

    async def owner() -> None:
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            target = write_index_session(archive, reference_session("creator-removal"))
            archive.commit()
            args = SessionDeleteArgs(archive=archive, session_ids=(target,))
            binding = runtime_operation_binding(SessionDeleteActuator())
            principal = MutationPrincipal("test", frozenset({"archive.delete_session"}), "api", "write")
            executor = OperationExecutor.for_archive_root(tmp_path)
            preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=tmp_path)
            authorization = executor.authorize_bound(binding, preview, principal)
            assert executor.execute_bound(binding, preview, authorization, args).affected_count == 1
            assert permitted_session_removals(archive_root=tmp_path) == frozenset()
            with pytest.raises(UnleasedWriteError):
                await child_tasks[0]

    try:
        await coordinator.run("test.bound-removal", owner)
        assert observations == [frozenset()]
    finally:
        assert await coordinator.shutdown()


@pytest.mark.parametrize("commit_route", ["explicit", "native_context", "cancelled_after_commit"])
def test_known_source_receipt_accepts_only_its_declared_native_commit(tmp_path: Path, commit_route: str) -> None:
    from polylogue.storage.sqlite.connection_profile import open_source_tier_write_connection
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    with write_lease("test.source-receipt", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with closing(open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)) as setup:
            setup.execute("CREATE TABLE authority_control (key TEXT PRIMARY KEY, value TEXT)")
            setup.execute("INSERT INTO authority_control VALUES ('selected', 'original')")
            setup.commit()
        with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
            permit = seal.prepare_known_tier_mutation(
                "authority_control", ("value",), (("accepted", "selected"),), tier="source", key_column="key"
            )
            with permit.hold_authority(), permit.mutation_connection() as source:
                if commit_route == "native_context":
                    with source:
                        source.execute("BEGIN IMMEDIATE")
                        source.execute("UPDATE authority_control SET value = 'accepted' WHERE key = 'selected'")
                        permit.allow_commit(source)
                else:
                    source.execute("BEGIN IMMEDIATE")
                    source.execute("UPDATE authority_control SET value = 'accepted' WHERE key = 'selected'")
                    permit.allow_commit(source)
                    source.commit()
                if commit_route == "cancelled_after_commit":
                    import threading

                    from polylogue.core.compute_cancel import compute_cancel

                    cancelled = threading.Event()
                    token = compute_cancel.set(cancelled)
                    try:
                        for observer in (*seal._observers.values(), seal._scratch):
                            observer.set_progress_handler(lambda: int(cancelled.is_set()), 1)
                        cancelled.set()
                        seal.accept_known_tier_commit(permit.committed())
                    finally:
                        cancelled.clear()
                        compute_cancel.reset(token)
                else:
                    seal.accept_known_tier_commit(permit.committed())
            seal.validate_observers_current()
            assert seal.observer("source").execute("SELECT value FROM authority_control").fetchone()[0] == "accepted"


def test_known_source_rollback_cannot_become_a_successful_receipt(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.connection_profile import open_source_tier_write_connection
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    with write_lease("test.source-rollback", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with closing(open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)) as setup:
            setup.execute("CREATE TABLE authority_control (key TEXT PRIMARY KEY, value TEXT)")
            setup.execute("INSERT INTO authority_control VALUES ('selected', 'original')")
            setup.commit()
        with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
            permit = seal.prepare_known_tier_mutation(
                "authority_control", ("value",), (("accepted", "selected"),), tier="source", key_column="key"
            )
            with permit.hold_authority(), permit.mutation_connection() as source:
                source.execute("BEGIN IMMEDIATE")
                source.execute("UPDATE authority_control SET value = 'accepted' WHERE key = 'selected'")
                permit.allow_commit(source)
                source.rollback()
                with pytest.raises(ReferenceSealError):
                    permit.committed()
                assert source.execute("SELECT value FROM authority_control").fetchone()[0] == "original"


@pytest.mark.parametrize("route", ["connection", "custom_cursor", "executemany"])
def test_known_source_exact_row_guard_survives_factory_profile_setup(tmp_path: Path, route: str) -> None:
    from polylogue.storage.sqlite.connection_profile import open_source_tier_write_connection
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    with write_lease("test.source-exact-row", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with closing(open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)) as setup:
            setup.execute("CREATE TABLE authority_control (key TEXT PRIMARY KEY, value TEXT)")
            setup.executemany(
                "INSERT INTO authority_control VALUES (?, ?)", [("selected", "original"), ("other", "retained")]
            )
            setup.commit()
        with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
            permit = seal.prepare_known_tier_mutation(
                "authority_control", ("value",), (("accepted", "selected"),), tier="source", key_column="key"
            )
            with permit.hold_authority():
                with pytest.raises(ReferenceSealError):
                    with permit.mutation_connection() as source:
                        for setting in ("temp_store=DEFAULT", "foreign_keys=OFF", "synchronous=OFF"):
                            with pytest.raises(sqlite3.DatabaseError):
                                source.execute(f"PRAGMA {setting}")
                        source.execute("BEGIN IMMEDIATE")
                        sql = "UPDATE authority_control SET value = ? WHERE key = ?"
                        if route == "custom_cursor":
                            with closing(source.cursor(factory=sqlite3.Cursor)) as cursor:
                                cursor.execute(sql, ("accepted", "other"))
                        elif route == "executemany":
                            source.executemany(sql, [("accepted", "other")])
                        else:
                            source.execute(sql, ("accepted", "other"))
                assert (
                    seal.observer("source")
                    .execute("SELECT value FROM authority_control WHERE key='other'")
                    .fetchone()[0]
                    == "retained"
                )


def test_known_source_partial_upsert_preserves_undeclared_columns(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.connection_profile import open_source_tier_write_connection
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    with write_lease("test.source-partial-upsert", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with closing(open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)) as setup:
            setup.execute(
                "CREATE TABLE authority_control (key TEXT PRIMARY KEY, value TEXT, retained TEXT DEFAULT 'default')"
            )
            setup.execute("INSERT INTO authority_control VALUES ('selected', 'original', 'preserved')")
            setup.commit()
        with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
            permit = seal.prepare_known_tier_mutation(
                "authority_control", ("value",), (("accepted", "selected"),), tier="source", key_column="key"
            )
            with permit.hold_authority(), permit.mutation_connection() as source:
                source.execute("BEGIN IMMEDIATE")
                source.execute(
                    "INSERT INTO authority_control (key,value) VALUES ('selected','accepted') "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value"
                )
                permit.allow_commit(source)
                source.commit()
                seal.accept_known_tier_commit(permit.committed())
            assert tuple(seal.observer("source").execute("SELECT * FROM authority_control").fetchone()) == (
                "selected",
                "accepted",
                "preserved",
            )


@pytest.mark.parametrize("boundary", ["before_begin", "commit_gap"])
def test_known_source_refuses_foreign_unrelated_commit(tmp_path: Path, boundary: str) -> None:
    from polylogue.storage.sqlite.connection_profile import open_source_tier_write_connection
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation, ReferenceSealStaleError

    with write_lease("test.source-foreign-commit", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with closing(open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)) as setup:
            setup.execute("CREATE TABLE authority_control (key TEXT PRIMARY KEY, value TEXT)")
            setup.executemany(
                "INSERT INTO authority_control VALUES (?,?)", [("selected", "original"), ("other", "retained")]
            )
            setup.commit()
        with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
            permit = seal.prepare_known_tier_mutation(
                "authority_control", ("value",), (("accepted", "selected"),), tier="source", key_column="key"
            )
            original_version = seal.observer_version("source")
            with permit.hold_authority(), permit.mutation_connection() as source:
                with closing(sqlite3.connect(tmp_path / "source.db")) as foreign:
                    if boundary == "before_begin":
                        foreign.execute("UPDATE authority_control SET value='foreign' WHERE key='other'")
                        foreign.commit()
                    source.execute("BEGIN IMMEDIATE")
                    source.execute("UPDATE authority_control SET value='accepted' WHERE key='selected'")
                    if boundary == "before_begin":
                        with pytest.raises(ReferenceSealStaleError):
                            permit.allow_commit(source)
                        source.rollback()
                    else:
                        permit.allow_commit(source)
                        source.commit()
                        receipt = permit.committed()
                        foreign.execute("UPDATE authority_control SET value='foreign' WHERE key='other'")
                        foreign.commit()
                        with pytest.raises(ReferenceSealStaleError):
                            seal.accept_known_tier_commit(receipt)
                    assert seal.observer_version("source") == original_version
                    assert not source.in_transaction


def test_known_source_reserves_tier_during_original_observer_advancement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite.connection_profile import open_source_tier_write_connection
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    with write_lease("test.source-acceptance-reservation", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with closing(open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)) as setup:
            setup.execute("CREATE TABLE authority_control (key TEXT PRIMARY KEY, value TEXT)")
            setup.execute("INSERT INTO authority_control VALUES ('selected','original')")
            setup.commit()
        with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
            permit = seal.prepare_known_tier_mutation(
                "authority_control", ("value",), (("accepted", "selected"),), tier="source", key_column="key"
            )
            original_verify = seal._verify_known_tier_postimage
            attempted = False
            with closing(sqlite3.connect(tmp_path / "source.db", timeout=0)) as foreign:

                def verify_reserved(connection: sqlite3.Connection, tier: str) -> None:
                    nonlocal attempted
                    if connection is seal.observer("source"):
                        attempted = True
                        with pytest.raises(sqlite3.OperationalError) as refusal:
                            foreign.execute("UPDATE authority_control SET value='foreign' WHERE key='selected'")
                        assert refusal.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
                        foreign.rollback()
                    original_verify(connection, tier)

                monkeypatch.setattr(seal, "_verify_known_tier_postimage", verify_reserved)
                with permit.hold_authority(), permit.mutation_connection() as source:
                    source.execute("BEGIN IMMEDIATE")
                    source.execute("UPDATE authority_control SET value='accepted' WHERE key='selected'")
                    permit.allow_commit(source)
                    source.commit()
                    seal.accept_known_tier_commit(permit.committed())
                    assert not source.in_transaction
            assert attempted
            seal.validate_observers_current()


@pytest.mark.parametrize("include_cascade", [True, False])
def test_known_source_guards_declared_foreign_key_trigger_and_sequence_effects(
    tmp_path: Path, include_cascade: bool
) -> None:
    from polylogue.storage.sqlite.connection_profile import open_source_tier_write_connection
    from polylogue.storage.sqlite.reference_seal import KnownTierRowEffect, PreparedIndexMutation

    with write_lease("test.source-cascade-effects", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with closing(open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)) as setup:
            setup.executescript(
                "CREATE TABLE authority_parent (key TEXT PRIMARY KEY);"
                "CREATE TABLE authority_child (key TEXT PRIMARY KEY, parent TEXT REFERENCES authority_parent(key) ON DELETE CASCADE);"
                "CREATE TABLE authority_log (sequence INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT);"
                "CREATE TRIGGER authority_delete AFTER DELETE ON authority_parent BEGIN "
                "INSERT INTO authority_log(key) VALUES (OLD.key); END;"
                "INSERT INTO authority_parent VALUES ('selected');"
                "INSERT INTO authority_child VALUES ('child','selected');"
            )
        with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
            effects = [
                KnownTierRowEffect("authority_parent", ("key",), ("selected",), None),
                KnownTierRowEffect("authority_log", ("sequence", "key"), None, (1, "selected")),
                KnownTierRowEffect("sqlite_sequence", ("name", "seq"), None, ("authority_log", 1)),
            ]
            if include_cascade:
                effects.append(KnownTierRowEffect("authority_child", ("key", "parent"), ("child", "selected"), None))
            permit = seal.prepare_known_tier_mutation(tier="source", effects=effects)
            with permit.hold_authority():
                if include_cascade:
                    with permit.mutation_connection() as source:
                        source.execute("BEGIN IMMEDIATE")
                        source.execute("DELETE FROM authority_parent WHERE key='selected'")
                        permit.allow_commit(source)
                        source.commit()
                        seal.accept_known_tier_commit(permit.committed())
                    assert seal.observer("source").execute("SELECT * FROM authority_child").fetchone() is None
                    assert tuple(seal.observer("source").execute("SELECT * FROM authority_log").fetchone()) == (
                        1,
                        "selected",
                    )
                else:
                    with pytest.raises(sqlite3.DatabaseError):
                        with permit.mutation_connection() as source:
                            source.execute("BEGIN IMMEDIATE")
                            source.execute("DELETE FROM authority_parent WHERE key='selected'")
                    assert seal.observer("source").execute("SELECT * FROM authority_parent").fetchone() is not None
                    assert seal.observer("source").execute("SELECT * FROM authority_log").fetchone() is None


def test_native_settlement_callbacks_wait_for_actual_close_and_retry_once() -> None:
    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner
    from tests.infra.sqlite_cursor_settlement import ControlledConnection

    connection = sqlite3.connect(":memory:", factory=ControlledConnection)
    owner = NativeSQLCustodyOwner(connection)
    completions: list[str] = []
    callback_fails = True

    def final_callback() -> None:
        if callback_fails:
            raise ValueError("synthetic receipt completion fault")
        completions.append("final")

    owner.retain_settlement_callback(lambda: completions.append("first"))
    owner.retain_settlement_callback(final_callback)
    try:
        connection.close_failure = OSError("synthetic native close fault")
        with pytest.raises(NativeConnectionSettlementError):
            owner.close()
        assert completions == [] and not owner._settled
        connection.close_failure = None
        with pytest.raises(NativeConnectionSettlementError):
            owner.close()
        assert completions == ["first"] and owner.connection is None and not owner._settled
        callback_fails = False
        owner.close()
        assert completions == ["first", "final"] and owner._settled
        owner.close()
        assert completions == ["first", "final"]
    finally:
        connection.close_failure = None
        callback_fails = False
        owner.close()


def test_archive_settlement_callback_waits_for_all_original_read_children(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStoreSettlementError
    from tests.infra.sqlite_cursor_settlement import ControlledCursor

    with write_lease("test.archive-callback-bootstrap", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
    archive = ArchiveStore.open_existing(tmp_path, read_only=True)
    completions: list[str] = []
    archive.retain_settlement_callback(lambda: completions.append("complete"))
    reader = archive._open_read_connection(tmp_path / "user.db")
    cursor = reader.cursor(factory=ControlledCursor)
    cursor.execute("SELECT 1 UNION ALL SELECT 2")
    next(cursor)
    cursor.allow_cleanup.clear()
    try:
        with pytest.raises(ArchiveStoreSettlementError):
            archive.close()
        assert completions == [] and archive._owned_index_connection is None
        cursor.allow_cleanup.set()
        archive.close()
        assert completions == ["complete"]
        archive.close()
        assert completions == ["complete"]
    finally:
        cursor.allow_cleanup.set()
        archive.close()


@pytest.mark.parametrize("surviving_anchor", [False, True])
def test_original_excision_projection_survives_source_commit_and_protects_other_rows(
    tmp_path: Path, surviving_anchor: bool
) -> None:
    from polylogue.storage.sqlite.archive_tiers.archive import stage_index_session_deletions
    from polylogue.storage.sqlite.connection_profile import open_source_tier_write_connection
    from polylogue.storage.sqlite.reference_seal import KnownTierRowEffect, PreparedIndexMutation
    from polylogue.storage.sqlite.write_lease import authorized_session_removal

    with write_lease("test.original-excision-projection", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            target = write_index_session(archive, reference_session("projected-target"))
            survivor = write_index_session(archive, reference_session("projected-survivor"))
            archive.commit()
        with closing(open_connection(tmp_path / "user.db", archive_root=tmp_path)) as user:
            upsert_assertion(
                user, assertion_id="removable", target_ref=f"session:{target}", kind=AssertionKind.ANNOTATION, now_ms=1
            )
            upsert_assertion(
                user,
                assertion_id="request-history",
                target_ref=f"session:{target}",
                kind=AssertionKind.EXCISION_REQUEST,
                now_ms=1,
            )
            if surviving_anchor:
                upsert_assertion(
                    user,
                    assertion_id="retained",
                    target_ref=f"session:{survivor}",
                    kind=AssertionKind.ANNOTATION,
                    scope_ref=f"session:{target}",
                    evidence_refs=(f"session:{target}",),
                    now_ms=1,
                )
            user.commit()
        with closing(open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)) as setup:
            setup.execute("CREATE TABLE authority_control (key TEXT PRIMARY KEY, value TEXT)")
            setup.execute("INSERT INTO authority_control VALUES ('selected','original')")
            setup.commit()
        with (
            authorized_session_removal(
                archive_root=tmp_path, plan_hash="exact-projection", session_ids=(target,), excise_assertions=True
            ),
            PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal,
            closing(open_connection(tmp_path / "index.db", archive_root=tmp_path)) as index,
        ):
            user = seal.observer("user")
            columns = tuple(str(row[1]) for row in user.execute("PRAGMA table_info(assertions)"))
            old = tuple(user.execute("SELECT * FROM assertions WHERE assertion_id='removable'").fetchone())
            frame = tuple(user.execute("SELECT * FROM query_unit_frame_state").fetchone())
            user_permit = seal.prepare_known_tier_mutation(
                tier="user",
                effects=(
                    KnownTierRowEffect("assertions", columns, old, None),
                    KnownTierRowEffect("query_unit_frame_state", ("singleton", "epoch"), frame, (1, int(frame[1]) + 1)),
                ),
            )
            source_permit = seal.prepare_known_tier_mutation(
                "authority_control", ("value",), (("accepted", "selected"),), tier="source", key_column="key"
            )

            def apply_staged() -> None:
                with seal.mutation_scope(index) as scope:
                    scope.authorize_session_removal((target,))
                    assert stage_index_session_deletions(index, scope, (target,)) == (target,)
                    # Projected removal never changes the verified live proof.
                    with pytest.raises(ReferenceSealError):
                        scope.validate_reachability(index)
                    scope.preflight_reachability()
                    with source_permit.hold_authority(), source_permit.mutation_connection() as source:
                        source.execute("BEGIN IMMEDIATE")
                        source.execute("UPDATE authority_control SET value='accepted' WHERE key='selected'")
                        source_permit.allow_commit(source)
                        source.commit()
                        seal.accept_known_tier_commit(source_permit.committed())
                    with user_permit.hold_authority(), user_permit.mutation_connection() as writer:
                        assert writer.execute("PRAGMA foreign_keys").fetchone()[0] == 1
                        with pytest.raises(sqlite3.DatabaseError):
                            writer.execute("PRAGMA foreign_keys=OFF")
                        writer.execute("BEGIN IMMEDIATE")
                        writer.execute("DELETE FROM assertions WHERE assertion_id='removable'")
                        user_permit.allow_commit(writer)
                        writer.commit()
                        seal.accept_known_tier_commit(user_permit.committed())

            if surviving_anchor:
                with pytest.raises(ReferenceSealError):
                    apply_staged()
                assert index.execute("SELECT 1 FROM sessions WHERE session_id=?", (target,)).fetchone()
                assert user.execute("SELECT 1 FROM assertions WHERE assertion_id='removable'").fetchone()
                assert (
                    seal.observer("source").execute("SELECT value FROM authority_control").fetchone()[0] == "original"
                )
            else:
                apply_staged()
                assert index.execute("SELECT 1 FROM sessions WHERE session_id=?", (target,)).fetchone() is None
                assert user.execute("SELECT 1 FROM assertions WHERE assertion_id='removable'").fetchone() is None
                assert user.execute("SELECT 1 FROM assertions WHERE assertion_id='request-history'").fetchone()
