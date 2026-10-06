"""Publication preserves the original lookup, not only its former row."""

import sqlite3
from builtins import BaseExceptionGroup
from collections.abc import Callable, Iterator
from contextlib import closing
from pathlib import Path
from typing import Any, Literal

import pytest

from polylogue.core.enums import AssertionKind, Provider
from polylogue.core.refs import EvidenceRef
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.user_write import upsert_assertion
from polylogue.storage.sqlite.connection_profile import (
    NativeConnectionSettlementError,
    NativeSQLCustodyOwner,
    open_connection,
)
from polylogue.storage.sqlite.reference_seal import ReferenceSealError
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.index_writer import prepared_fixture_index_batch, write_fixture_index_session
from tests.infra.live_ingest import write_index_session
from tests.infra.reference_sessions import reference_session
from tests.infra.sqlite_cursor_settlement import settlement_owner_summary


def test_prepared_inactive_scope_writes_only_its_captured_owned_generation(tmp_path: Path) -> None:
    from polylogue.storage.index_generation import IndexGenerationStore
    from polylogue.storage.sqlite.reference_seal import IndexMutationDestination, PreparedIndexMutation

    # Bootstrap owns its own lease; the generation is created under this law's.
    bootstrap_archive_root(tmp_path)
    with write_lease("test.prepare-inactive-destination", archive_root=tmp_path):
        generation = IndexGenerationStore.for_archive_root(tmp_path).create(source_snapshot="prepared-destination")
    destination = IndexMutationDestination.owned_inactive(generation)
    with PreparedIndexMutation(Path(generation.index_path), archive_root=tmp_path, destination=destination) as seal:
        with write_lease("test.publish-inactive-destination", archive_root=tmp_path):
            with ArchiveStore.open_owned_inactive_generation(
                Path(generation.index_path).parent,
                generation_id=generation.generation_id,
                owner_id=generation.owner_id,
            ) as archive:
                with archive.index_mutation_scope(prepared_seal=seal) as scope:
                    archive._conn.execute("CREATE TABLE captured_destination(value INTEGER)")
                    archive._conn.execute("INSERT INTO captured_destination VALUES (1)")
                assert archive._conn.execute("SELECT value FROM captured_destination").fetchone()[0] == 1
                assert scope._committed and scope._commit_receipt is None and seal._cleanup_requested
                with pytest.raises(ReferenceSealError):
                    _ = scope.commit_receipt
                with pytest.raises(ReferenceSealError):
                    with seal.original_read_snapshot():
                        pytest.fail("terminal inactive seal admitted another original read")
                with pytest.raises(ReferenceSealError):
                    with seal.mutation_scope(archive._conn):
                        pytest.fail("terminal inactive seal admitted a second publication")
                writer_owner = scope._writer_owner
                connection = archive._conn
                assert writer_owner is not None and writer_owner.connection is connection
            assert writer_owner.connection is None and writer_owner._settled
            with pytest.raises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")
            with closing(sqlite3.connect(f"file:{generation.index_path}?mode=ro", uri=True)) as observed:
                with closing(observed.execute("SELECT value FROM captured_destination")) as rows:
                    assert rows.fetchone()[0] == 1
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
            assert retained_native_settlement_owners_on_current_thread(entry) == (seal,), settlement_owner_summary(
                retained_native_settlement_owners_on_current_thread(entry)
            )
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
                assert retained_native_settlement_owners_on_current_thread(entry) == (seal,), settlement_owner_summary(
                    retained_native_settlement_owners_on_current_thread(entry)
                )
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
                lambda sql: assertion_queries.append(sql) if sql == 'SELECT rowid FROM "assertions"' else None
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
            sessions = tuple(reference_session(f"batch-{number}") for number in range(8))
            with prepared_fixture_index_batch(archive._conn, sessions, archive_root=tmp_path) as (seal, prepared):
                with archive.index_mutation_scope(prepared_seal=seal):
                    for session, carrier in zip(sessions, prepared, strict=True):
                        write_fixture_index_session(
                            archive._conn,
                            session,
                            prepared_write=carrier,
                            content_hash=carrier.input_content_hash.hex(),
                            pending_input_content_hash=carrier.input_content_hash.hex(),
                        )
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
                sessions = (reference_session("cancelled-pending"), reference_session("must-not-start"))
                with prepared_fixture_index_batch(archive._conn, sessions, archive_root=tmp_path) as (seal, prepared):
                    with pytest.raises(asyncio.CancelledError):
                        with archive.index_mutation_scope(prepared_seal=seal):
                            write_fixture_index_session(
                                archive._conn,
                                sessions[0],
                                prepared_write=prepared[0],
                                content_hash=prepared[0].input_content_hash.hex(),
                                pending_input_content_hash=prepared[0].input_content_hash.hex(),
                            )
                            assert archive._conn.in_transaction
                            cancelled.set()
                            archive._conn.set_progress_handler(lambda: int(cancelled.is_set()), 1)
                            # Admission refuses another mutation. The scope must
                            # still roll back and close on this original owner.
                            write_fixture_index_session(
                                archive._conn,
                                sessions[1],
                                prepared_write=prepared[1],
                                content_hash=prepared[1].input_content_hash.hex(),
                                pending_input_content_hash=prepared[1].input_content_hash.hex(),
                            )
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
            sessions = tuple(reference_session(f"suppression-{number}") for number in range(3))
            with prepared_fixture_index_batch(archive._conn, sessions, archive_root=tmp_path) as (seal, prepared):
                with archive.index_mutation_scope(prepared_seal=seal) as scope:
                    assert scope is not None
                    first = scope.suppression_reader()
                    for session, carrier in zip(sessions, prepared, strict=True):
                        assert scope.suppression_reader() is first
                        write_fixture_index_session(
                            archive._conn,
                            session,
                            prepared_write=carrier,
                            content_hash=carrier.input_content_hash.hex(),
                            pending_input_content_hash=carrier.input_content_hash.hex(),
                        )


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
                    provider=Provider.CODEX,
                    source_path="refused-new-work",
                    canonical_source_path="refused-new-work",
                    acquired_at_ms=0,
                    payload=b"new",
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
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
    from tests.infra.sqlite_cursor_settlement import ControlledConnection, control_archive_connections

    with write_lease("test.archive-tier-rollback", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with monkeypatch.context() as factory_control:
            control_archive_connections(
                factory_control, tmp_path / "index.db", tmp_path / "source.db", tmp_path / "user.db"
            )
            archive = ArchiveStore.open_existing(tmp_path, read_only=False)
            if tier == "source":
                child = archive.source_connection
            elif tier == "user":
                child = archive._open_user_write_connection()
            else:
                child = sqlite3.connect(":memory:", factory=ControlledConnection)
                archive.operation_vector_connection = child
        seal = PreparedIndexMutation(archive.index_db_path, archive_root=tmp_path)
        try:
            index = archive._conn
            assert isinstance(index, ControlledConnection)
            assert isinstance(child, ControlledConnection)
            child.execute("BEGIN")
            fault = OSError(errno.EIO, "synthetic actual tier rollback failure")
            with pytest.raises(OSError) as caught:
                with archive.index_mutation_scope(prepared_seal=seal) as scope:
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
            seal.close()


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
            assert retained_native_settlement_owners_on_current_thread(entry) == (seal,), settlement_owner_summary(
                retained_native_settlement_owners_on_current_thread(entry)
            )
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
            assert captured == (archive,), settlement_owner_summary(captured)
            if persistent:
                with pytest.raises(ArchiveStoreSettlementError):
                    captured[0].close()
                assert cursors[0].close_attempts == 2
                assert archive._pending_index_mutation_scope is scope
                assert retained_native_sql_owners() == (archive,), settlement_owner_summary(
                    retained_native_sql_owners()
                )
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
            assert retained_native_sql_owners() == (archive,), settlement_owner_summary(retained_native_sql_owners())
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
            assert captured == (archive,), settlement_owner_summary(captured)
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
                    provider=Provider.CODEX,
                    source_path="refused-terminal-parent",
                    canonical_source_path="refused-terminal-parent",
                    acquired_at_ms=0,
                    payload=b"new",
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
def test_bound_deletion_preserves_audit_identity_and_keeps_ordinary_user_anchors(
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
            # An ordinary User anchor (an annotation) survives the authorized
            # delete unchanged and resolves again if the session is re-imported.
            receipt = executor.execute_bound(binding, current, authorization, args)
            assert receipt.status == "applied" and receipt.affected_count == 1
            assert archive.stored_session_ids((target,)) == ()
            if ordinary_anchor:
                with closing(sqlite3.connect(tmp_path / "user.db")) as user:
                    assert user.execute(
                        "SELECT COUNT(*) FROM assertions WHERE target_ref = ? AND kind = 'annotation'",
                        (f"session:{target}",),
                    ).fetchone() == (1,)
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

    def seed() -> str:
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            seeded = write_index_session(archive, reference_session("creator-removal"))
            archive.commit()
            return seeded

    # The fixture writer bootstraps under its own lease, so the session is
    # seeded before the coordinator's writer holds the archive.
    target = await asyncio.to_thread(seed)

    async def owner() -> None:
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
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
        assert await coordinator.shutdown(timeout=float("inf"))


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
                        for setting in (
                            "temp_store=DEFAULT",
                            "foreign_keys=OFF",
                            "recursive_triggers=OFF",
                            "synchronous=OFF",
                            "ignore_check_constraints=ON",
                            "defer_foreign_keys=ON",
                            "writable_schema=ON",
                            "trusted_schema=OFF",
                            "query_only=ON",
                            "legacy_alter_table=ON",
                            "wal_checkpoint",
                            "optimize",
                            "incremental_vacuum",
                        ):
                            with pytest.raises(sqlite3.DatabaseError):
                                source.execute(f"PRAGMA {setting}")
                        assert source.execute("PRAGMA table_info(authority_control)").fetchall()
                        with pytest.raises(sqlite3.DatabaseError):
                            source.blobopen("authority_control", "value", 1, readonly=False)
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
            original_observer = seal._require_unpinned_observer
            attempted = False
            with closing(sqlite3.connect(tmp_path / "source.db", timeout=0)) as foreign:

                def observe_reserved(tier: str) -> sqlite3.Connection:
                    nonlocal attempted
                    if tier == "source" and permit._acceptance_reservation:
                        attempted = True
                        with pytest.raises(sqlite3.OperationalError) as refusal:
                            foreign.execute("UPDATE authority_control SET value='foreign' WHERE key='selected'")
                        assert refusal.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
                        foreign.rollback()
                    return original_observer(tier)

                monkeypatch.setattr(seal, "_require_unpinned_observer", observe_reserved)
                with permit.hold_authority(), permit.mutation_connection() as source:
                    source.execute("BEGIN IMMEDIATE")
                    source.execute("UPDATE authority_control SET value='accepted' WHERE key='selected'")
                    permit.allow_commit(source)
                    source.commit()
                    seal.accept_known_tier_commit(permit.committed())
                    assert not source.in_transaction
            assert attempted
            seal.validate_observers_current()


def test_known_source_postimage_cursor_failure_refuses_commit_and_retains_actual_creator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite.connection_profile import native_sql_children, open_source_tier_write_connection
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
    from tests.infra.sqlite_cursor_settlement import ControlledCursor

    with write_lease("test.source-precommit-postimage-fault", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with closing(open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)) as setup:
            with closing(setup.execute("CREATE TABLE authority_control (key TEXT PRIMARY KEY, value TEXT)")):
                pass
            with closing(setup.execute("INSERT INTO authority_control VALUES ('selected','original')")):
                pass
            setup.commit()
        with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
            permit = seal.prepare_known_tier_mutation(
                "authority_control", ("value",), (("accepted", "selected"),), tier="source", key_column="key"
            )
            original_version = seal.observer_version("source")
            failed_cursors: list[ControlledCursor] = []

            class PostimageCursor(ControlledCursor):
                def execute(self, sql: str, parameters: Any = (), /) -> "PostimageCursor":
                    super().execute(sql, parameters)
                    if sql.startswith("SELECT") and 'FROM "authority_control"' in sql:
                        self.allow_cleanup.clear()
                        failed_cursors.append(self)
                    return self

            source: sqlite3.Connection | None = None
            try:
                with pytest.raises(NativeConnectionSettlementError):
                    with permit.hold_authority(), permit.mutation_connection() as source:
                        with closing(source.execute("BEGIN IMMEDIATE")):
                            pass
                        with closing(
                            source.execute("UPDATE authority_control SET value='accepted' WHERE key='selected'")
                        ):
                            pass
                        original_cursor = source.cursor

                        def postimage_cursor() -> sqlite3.Cursor:
                            return original_cursor(factory=PostimageCursor)

                        monkeypatch.setattr(source, "cursor", postimage_cursor)
                        permit.allow_commit(source)
                assert source is not None and failed_cursors
                owner = next(child for child in native_sql_children(seal) if child.connection is source)
                assert owner.close_required and not permit._commit_allowed
                assert seal._versions["source"] == original_version
                with pytest.raises(ReferenceSealError):
                    seal.observer_version("source")
                with pytest.raises(ReferenceSealError):
                    permit.committed()
                for failed_cursor in failed_cursors:
                    failed_cursor.allow_cleanup.set()
                owner.close()
                assert owner.connection is None
                with pytest.raises(ReferenceSealError):
                    seal.observer("source")
                with closing(open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)) as check:
                    with closing(check.execute("SELECT value FROM authority_control")) as cursor:
                        assert cursor.fetchone()[0] == "original"
            finally:
                for failed_cursor in failed_cursors:
                    failed_cursor.allow_cleanup.set()
                if source is not None:
                    for child in native_sql_children(seal):
                        if child.connection is source:
                            child.close()


@pytest.mark.parametrize("include_cascade", [True, False])
def test_known_source_guards_declared_foreign_key_trigger_and_sequence_effects(
    tmp_path: Path, include_cascade: bool
) -> None:
    from polylogue.storage.sqlite.connection_profile import open_source_tier_write_connection
    from polylogue.storage.sqlite.reference_seal import KnownTierRowEffect, KnownTierRowImage, PreparedIndexMutation

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
            with seal.original_read_snapshot():
                parent = seal.retain_tier_row("source", "authority_parent", 1)
                child = seal.retain_tier_row("source", "authority_child", 1)
            assert parent is not None and child is not None
            log = KnownTierRowImage(
                seal,
                "authority_log",
                ("sequence", "key"),
                1,
                (seal.retain_literal_scalar(1), parent.cells[0]),
            )
            sequence = KnownTierRowImage(
                seal,
                "sqlite_sequence",
                ("name", "seq"),
                1,
                (seal.retain_literal_scalar("authority_log"), seal.retain_literal_scalar(1)),
            )
            effects = [
                KnownTierRowEffect("authority_parent", parent.columns, parent, None),
                KnownTierRowEffect("authority_log", log.columns, None, log),
                KnownTierRowEffect("sqlite_sequence", sequence.columns, None, sequence),
            ]
            if include_cascade:
                effects.append(KnownTierRowEffect("authority_child", child.columns, child, None))
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
def test_original_excision_projection_survives_source_commit_and_keeps_other_rows(
    tmp_path: Path, surviving_anchor: bool
) -> None:
    from dataclasses import replace

    from polylogue.storage.sqlite.archive_tiers.archive import stage_index_session_deletions
    from polylogue.storage.sqlite.connection_profile import (
        ISOLATED_TIER_WRITE_PROFILE,
        open_isolated_write_connection,
        open_source_tier_write_connection,
    )
    from polylogue.storage.sqlite.reference_seal import KnownTierRowEffect, KnownTierRowImage, PreparedIndexMutation
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
            closing(
                NativeSQLCustodyOwner(
                    open_isolated_write_connection(
                        tmp_path / "index.db",
                        purpose="test.excision-index-stage",
                        archive_root=tmp_path,
                        profile=replace(ISOLATED_TIER_WRITE_PROFILE, foreign_keys=True),
                    )
                )
            ) as index_owner,
        ):
            index = index_owner.require_connection()
            with closing(index.execute("PRAGMA foreign_keys")) as cursor:
                assert cursor.fetchone()[0] == 1
            with closing(index.execute("SELECT message_id FROM messages WHERE session_id=?", (target,))) as cursor:
                target_messages = tuple(row[0] for row in cursor)
            with closing(
                index.execute(
                    "SELECT b.block_id FROM blocks b JOIN messages m ON m.message_id=b.message_id WHERE m.session_id=?",
                    (target,),
                )
            ) as cursor:
                target_blocks = tuple(row[0] for row in cursor)
            assert target_messages and target_blocks
            with closing(index.execute("SELECT * FROM messages WHERE session_id=?", (survivor,))) as cursor:
                survivor_messages = tuple(cursor)
            with closing(
                index.execute(
                    "SELECT b.* FROM blocks b JOIN messages m ON m.message_id=b.message_id WHERE m.session_id=?",
                    (survivor,),
                )
            ) as cursor:
                survivor_blocks = tuple(cursor)
            assert survivor_messages and survivor_blocks
            user = seal.observer("user")
            with seal.original_read_snapshot():
                with seal.original_rows(
                    "user", "SELECT rowid FROM assertions WHERE assertion_id='removable'"
                ) as cursor:
                    assertion_rowid = cursor.fetchone()[0]
                old = seal.retain_tier_row("user", "assertions", assertion_rowid)
                frame = seal.retain_tier_row("user", "query_unit_frame_state", 1)
            assert old is not None and frame is not None
            columns = old.columns
            with seal._owned_cursor(user, "SELECT epoch FROM query_unit_frame_state WHERE singleton=1") as cursor:
                next_epoch = cursor.fetchone()[0] + 1
            advanced_frame = KnownTierRowImage(
                seal,
                frame.table,
                frame.columns,
                frame.rowid,
                (frame.cells[0], seal.retain_literal_scalar(next_epoch)),
            )
            user_permit = seal.prepare_known_tier_mutation(
                tier="user",
                effects=(
                    KnownTierRowEffect("assertions", columns, old, None),
                    KnownTierRowEffect(
                        "query_unit_frame_state",
                        frame.columns,
                        frame,
                        advanced_frame,
                        _trigger_parent_ordinal=1,
                        _canonical_trigger="query_unit_frame_assertions_delete",
                    ),
                ),
            )
            source_permit = seal.prepare_known_tier_mutation(
                "authority_control", ("value",), (("accepted", "selected"),), tier="source", key_column="key"
            )

            def apply_staged() -> None:
                with seal.mutation_scope(index) as scope:
                    scope.authorize_session_removal((target,))
                    assert stage_index_session_deletions(index, scope, (target,)) == (target,)
                    # An authorized removal may leave User anchors naming the
                    # removed session, live or projected.
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

            apply_staged()
            if surviving_anchor:
                # A surviving row's references to the excised session stay,
                # unchanged, beside the removal (user.db is durable).
                assert user.execute("SELECT scope_ref FROM assertions WHERE assertion_id='retained'").fetchone()[0] == (
                    f"session:{target}"
                )
            assert index.execute("SELECT 1 FROM sessions WHERE session_id=?", (target,)).fetchone() is None
            assert user.execute("SELECT 1 FROM assertions WHERE assertion_id='removable'").fetchone() is None
            assert user.execute("SELECT 1 FROM assertions WHERE assertion_id='request-history'").fetchone()
            for message_id in target_messages:
                with closing(index.execute("SELECT 1 FROM messages WHERE message_id=?", (message_id,))) as cursor:
                    assert cursor.fetchone() is None
            for block_id in target_blocks:
                with closing(index.execute("SELECT 1 FROM blocks WHERE block_id=?", (block_id,))) as cursor:
                    assert cursor.fetchone() is None
            with closing(index.execute("SELECT * FROM messages WHERE session_id=?", (survivor,))) as cursor:
                assert tuple(cursor) == survivor_messages
            with closing(
                index.execute(
                    "SELECT b.* FROM blocks b JOIN messages m ON m.message_id=b.message_id WHERE m.session_id=?",
                    (survivor,),
                )
            ) as cursor:
                assert tuple(cursor) == survivor_blocks
            with closing(index.execute("PRAGMA foreign_key_check")) as cursor:
                assert cursor.fetchone() is None


@pytest.mark.parametrize("include_replace_delete", [False, True])
def test_known_source_replace_requires_its_implicit_delete_effect(tmp_path: Path, include_replace_delete: bool) -> None:
    from polylogue.storage.sqlite.connection_profile import open_source_tier_write_connection
    from polylogue.storage.sqlite.reference_seal import KnownTierRowEffect, KnownTierRowImage, PreparedIndexMutation

    with write_lease("test.source-replace-effects", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with closing(open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)) as setup:
            setup.execute("CREATE TABLE replace_control (key TEXT PRIMARY KEY, value TEXT UNIQUE)")
            setup.executemany(
                "INSERT INTO replace_control VALUES (?,?)", (("selected", "original"), ("other", "collision"))
            )
            setup.commit()
        with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
            with seal.original_read_snapshot():
                selected = seal.retain_tier_row("source", "replace_control", 1)
                collision = seal.retain_tier_row("source", "replace_control", 2)
            assert selected is not None and collision is not None
            replacement = KnownTierRowImage(
                seal,
                selected.table,
                selected.columns,
                selected.rowid,
                (selected.cells[0], collision.cells[1]),
            )
            effects = [KnownTierRowEffect("replace_control", selected.columns, selected, replacement)]
            if include_replace_delete:
                effects.append(KnownTierRowEffect("replace_control", collision.columns, collision, None))
            permit = seal.prepare_known_tier_mutation(tier="source", effects=effects)
            with permit.hold_authority():

                def replace() -> None:
                    with permit.mutation_connection() as source:
                        assert source.execute("PRAGMA recursive_triggers").fetchone()[0] == 1
                        source.execute("BEGIN IMMEDIATE")
                        source.execute("UPDATE OR REPLACE replace_control SET value='collision' WHERE key='selected'")
                        permit.allow_commit(source)
                        source.commit()
                        seal.accept_known_tier_commit(permit.committed())

                if include_replace_delete:
                    replace()
                    assert tuple(
                        tuple(row) for row in seal.observer("source").execute("SELECT * FROM replace_control")
                    ) == (("selected", "collision"),)
                else:
                    with pytest.raises(ReferenceSealError):
                        replace()
                    assert tuple(
                        tuple(row)
                        for row in seal.observer("source").execute("SELECT * FROM replace_control ORDER BY key")
                    ) == (
                        ("other", "collision"),
                        ("selected", "original"),
                    )


def test_native_incremental_handle_close_failure_retains_actual_owner_until_creator_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.io_phase_metrics import connect_measured
    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner

    connection = connect_measured(":memory:")
    connection.execute("CREATE TABLE literal_control(key INTEGER PRIMARY KEY, value TEXT)")
    connection.execute("INSERT INTO literal_control VALUES (1,'retained literal')")
    connection.commit()
    owner = NativeSQLCustodyOwner(connection)
    completions: list[str] = []
    owner.retain_settlement_callback(lambda: completions.append("complete"))
    close_blob = NativeSQLCustodyOwner.close_incremental_blob
    blocked = True
    selected_blob: sqlite3.Blob | None = None

    def guarded_close(selected: NativeSQLCustodyOwner, blob: sqlite3.Blob) -> None:
        if selected is owner and blocked:
            raise OSError("synthetic actual incremental close fault")
        close_blob(selected, blob)

    monkeypatch.setattr(NativeSQLCustodyOwner, "close_incremental_blob", guarded_close)
    try:
        with pytest.raises(NativeConnectionSettlementError):
            with owner.readonly_blob("literal_control", "value", 1) as blob:
                selected_blob = blob
                assert blob.read() == b"retained literal"
        assert owner._incremental_blobs == [selected_blob]
        assert owner.connection is connection and not owner._settled and completions == []
        with pytest.raises(NativeConnectionSettlementError):
            owner.close()
        assert selected_blob is not None and len(selected_blob) == len(b"retained literal")
        blocked = False
        owner.close()
        assert owner._settled and completions == ["complete"] and owner._incremental_blobs == []
        with pytest.raises(sqlite3.ProgrammingError):
            len(selected_blob)
    finally:
        blocked = False
        owner.close()


@pytest.mark.parametrize("native_owner", [False, True])
def test_settlement_callbacks_wait_for_original_custody_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_owner: bool
) -> None:
    from polylogue.storage.io_phase_metrics import connect_measured
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStoreSettlementError
    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner
    from polylogue.storage.sqlite.write_lease import ArchiveWriteCustody

    with write_lease("test.callback-custody-release", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        selected: NativeSQLCustodyOwner | ArchiveStore
        error_type: type[NativeConnectionSettlementError] | type[ArchiveStoreSettlementError]
        if native_owner:
            selected = NativeSQLCustodyOwner(connect_measured(":memory:"))
            custody = selected.custody
            error_type = NativeConnectionSettlementError
        else:
            selected = ArchiveStore.open_existing(tmp_path, read_only=False)
            selected._enter_mutation_lease()
            custody = selected._sql_custody
            error_type = ArchiveStoreSettlementError
        assert custody is not None
        completions: list[str] = []
        selected.retain_settlement_callback(lambda: completions.append("complete"))
        release = ArchiveWriteCustody.release_sql_owner
        blocked = True

        def guarded_release(original: ArchiveWriteCustody, owner: object) -> None:
            if original is custody and owner is selected and blocked:
                raise OSError("synthetic original physical custody release fault")
            release(original, owner)

        monkeypatch.setattr(ArchiveWriteCustody, "release_sql_owner", guarded_release)
        try:
            with pytest.raises(error_type):
                selected.close()
            assert completions == [] and selected in custody.retained_sql_owners_on_current_thread()
            assert (
                selected.custody if isinstance(selected, NativeSQLCustodyOwner) else selected._sql_custody
            ) is custody
            blocked = False
            selected.close()
            assert completions == ["complete"] and selected not in custody.retained_sql_owners_on_current_thread()
            selected.close()
            assert completions == ["complete"]
        finally:
            blocked = False
            selected.close()


@pytest.mark.parametrize("failed_resource", ["file", "sql"])
def test_archive_retains_actual_replay_slot_file_until_close_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_resource: str
) -> None:
    import fcntl

    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStoreSettlementError
    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner, native_sql_children

    with write_lease("test.replay-slot-close", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        archive = ArchiveStore.open_existing(tmp_path, read_only=False)
        archive._hold_replay_publisher_slot()
        lock_file = archive._replay_publisher_lock_file
        assert lock_file is not None
        close_file = lock_file.close
        close_sql = NativeSQLCustodyOwner.close
        selected_sql = next(child for child in native_sql_children(archive) if child.connection is archive._conn)
        blocked = True
        completions: list[str] = []
        archive.retain_settlement_callback(lambda: completions.append("complete"))

        def guarded_close() -> None:
            if blocked and failed_resource == "file":
                raise OSError("synthetic actual replay file close fault")
            close_file()

        def guarded_sql_close(owner: NativeSQLCustodyOwner) -> None:
            if owner is selected_sql and blocked and failed_resource == "sql":
                raise NativeConnectionSettlementError(owner, OSError("synthetic original SQL close fault"))
            close_sql(owner)

        monkeypatch.setattr(lock_file, "close", guarded_close)
        monkeypatch.setattr(NativeSQLCustodyOwner, "close", guarded_sql_close)
        try:
            with pytest.raises(ArchiveStoreSettlementError):
                archive.close()
            assert archive._replay_publisher_lock_file is lock_file and not lock_file.closed and completions == []
            with open(lock_file.name, "a+b") as contender:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            blocked = False
            archive.close()
            assert lock_file.closed and archive._replay_publisher_slot is None and completions == ["complete"]
        finally:
            blocked = False
            archive.close()


@pytest.mark.parametrize("snapshot", ["cursor", "transaction", "consumed_cursor"])
@pytest.mark.parametrize("boundary", ["observers", "writer"])
def test_original_observer_currency_refuses_pinned_wal_read_before_foreign_commit_acceptance(
    tmp_path: Path, snapshot: str, boundary: str
) -> None:
    from polylogue.storage.sqlite.connection_profile import (
        open_isolated_write_connection,
        open_source_tier_write_connection,
    )
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation, ReferenceSealStaleError

    with write_lease("test.original-observer-pinned", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with closing(open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)) as setup:
            setup.execute("PRAGMA journal_mode=WAL")
            setup.execute("CREATE TABLE authority_control(key TEXT PRIMARY KEY,value TEXT)")
            setup.executemany(
                "INSERT INTO authority_control VALUES (?,?)", [("first", "original"), ("second", "retained")]
            )
            setup.commit()
        with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
            observer = seal.observer("source")
            with closing(
                open_isolated_write_connection(
                    tmp_path / "index.db", archive_root=tmp_path, purpose="test observer currency"
                )
            ) as writer:
                validate = (
                    seal.validate_observers_current
                    if boundary == "observers"
                    else lambda: seal.validate_for_writer(writer)
                )
                if snapshot == "transaction":
                    observer.execute("BEGIN")
                cursor = observer.execute("SELECT value FROM authority_control ORDER BY key")
                try:
                    assert cursor.fetchone()[0] == "original"
                    if snapshot == "consumed_cursor":
                        cursor.fetchall()
                    with closing(
                        open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)
                    ) as foreign:
                        foreign.execute("UPDATE authority_control SET value='foreign' WHERE key='second'")
                        foreign.commit()
                    with pytest.raises(ReferenceSealError):
                        validate()
                finally:
                    cursor.close()
                    if observer.in_transaction:
                        observer.rollback()
                with pytest.raises(ReferenceSealStaleError):
                    validate()


def test_original_reference_readers_settle_retained_native_cursors_before_currency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from collections.abc import Callable

    from polylogue.archive.context_models import ContextImage, ContextSpec
    from polylogue.context.compiler import context_snapshot_record_from_image
    from polylogue.storage.block_anchor import BlockAnchor, resolve_block_anchor
    from polylogue.storage.io_phase_metrics import live_connection_cursors
    from polylogue.storage.sqlite.archive_tiers.context_delivery_write import write_context_delivery
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    retained: list[tuple[sqlite3.Connection, sqlite3.Cursor]] = []
    open_observer = PreparedIndexMutation._open_observer

    def sticky_observer(seal: PreparedIndexMutation, tier: str, path: Path) -> sqlite3.Connection:
        connection = open_observer(seal, tier, path)
        make_cursor = connection.cursor

        def sticky_cursor(factory: Callable[[sqlite3.Connection], sqlite3.Cursor] | None = None) -> sqlite3.Cursor:
            cursor = make_cursor() if factory is None else make_cursor(factory)
            retained.append((connection, cursor))
            return cursor

        monkeypatch.setattr(connection, "cursor", sticky_cursor)
        return connection

    with write_lease("test.original-reference-readers", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            parent = write_index_session(archive, reference_session("parent", messages=(("prefix", "prefix"),)))
            child = write_index_session(
                archive, reference_session("child", parent="parent", messages=(("prefix", "prefix"), ("tail", "tail")))
            )
            with closing(
                archive._conn.execute(
                    "SELECT m.message_id,b.content_hash FROM messages m JOIN blocks b USING(message_id) "
                    "WHERE m.session_id=? ORDER BY b.position LIMIT 1",
                    (parent,),
                )
            ) as cursor:
                message_id, content_hash = cursor.fetchone()
            evidence = EvidenceRef(child, message_id)
            image = ContextImage(
                spec=ContextSpec(seed_refs=(f"session:{child}",), read_views=()), segments=(), evidence_refs=(evidence,)
            )
            with closing(open_connection(tmp_path / "user.db", archive_root=tmp_path)) as user, user:
                upsert_assertion(
                    user,
                    assertion_id="inherited-reader",
                    target_ref="session:child",
                    kind=AssertionKind.ANNOTATION,
                    author_kind="user",
                    evidence_refs=(evidence.format(),),
                )
                write_context_delivery(
                    user,
                    image=image,
                    record=context_snapshot_record_from_image(image, boundary="session-start"),
                    recipient_ref="agent:reader",
                    delivered_by_ref="user:local",
                    delivered_at_ms=1,
                )
            monkeypatch.setattr(PreparedIndexMutation, "_open_observer", sticky_observer)
            with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
                with seal.original_read_snapshot():
                    result = resolve_block_anchor(
                        seal.observer("index"), BlockAnchor(parent, message_id, bytes(content_hash).hex())
                    )
                    assert result.state == "ok"
                seal.validate_observers_current()
                assert retained
                for connection, cursor in retained:
                    assert live_connection_cursors(connection) == ()
                    with pytest.raises(sqlite3.ProgrammingError):
                        cursor.fetchone()
                # The observers remain open and usable. Closing the parent
                # cannot conceal a still-live child statement in this control.
                with seal.original_read_snapshot():
                    with seal.original_rows("index", "SELECT count(*) FROM sessions") as cursor:
                        assert cursor.fetchone()[0] == 2


@pytest.mark.parametrize("tier", ["source", "user"])
@pytest.mark.parametrize("setup_effect", ["declared", "different_value", "different_pragma", "failed_close"])
def test_known_tier_profile_setup_is_exact_and_retires_before_guard_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tier: str, setup_effect: str
) -> None:
    from typing import Any, Literal, cast

    from polylogue.storage.io_phase_metrics import _MeasuredCursor
    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner, native_sql_children
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
    from tests.infra.sqlite_cursor_settlement import ControlledCursor

    actual_execute = _MeasuredCursor.execute
    selected: NativeSQLCustodyOwner | None = None
    blocked_cursor: ControlledCursor | None = None

    with write_lease("test.exact-profile-setup", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
            permit = seal.prepare_known_tier_mutation(tier=cast(Literal["source", "user"], tier), effects=())

            def execute(cursor: _MeasuredCursor, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
                nonlocal selected, blocked_cursor
                connection = cursor.connection
                if sql.startswith("PRAGMA busy_timeout =") and permit._setup_connection is connection:
                    selected = next(child for child in native_sql_children(seal) if child.connection is connection)
                    if setup_effect == "different_value":
                        sql = "PRAGMA busy_timeout = 1"
                    elif setup_effect == "different_pragma":
                        sql = "PRAGMA foreign_keys = OFF"
                    elif setup_effect == "failed_close":
                        blocked_cursor = connection.cursor(factory=ControlledCursor)
                        assert isinstance(blocked_cursor, ControlledCursor)
                        blocked_cursor.execute("SELECT 1 UNION ALL SELECT 2")
                        blocked_cursor.fetchone()
                        blocked_cursor.allow_cleanup.clear()
                        raise OSError("synthetic failure inside exact profile setup")
                return actual_execute(cursor, sql, parameters)

            monkeypatch.setattr(_MeasuredCursor, "execute", execute)
            try:
                with permit.hold_authority():
                    if setup_effect == "declared":
                        with permit.mutation_connection() as connection:
                            assert permit._setup_connection is None and permit._setup_pragma is None
                            with pytest.raises(ReferenceSealError):
                                permit.configure_mutation_connection(connection, ("PRAGMA foreign_keys = OFF",))
                            for pragma in ("journal_size_limit = 1", "foreign_keys = OFF", "recursive_triggers = OFF"):
                                with pytest.raises(sqlite3.DatabaseError):
                                    with closing(connection.execute("PRAGMA " + pragma)):
                                        pass
                    else:
                        expected: type[BaseException] = (
                            NativeConnectionSettlementError if setup_effect == "failed_close" else sqlite3.DatabaseError
                        )
                        with pytest.raises(expected):
                            with permit.mutation_connection():
                                pytest.fail("a forged or failed profile reached guarded producer SQL")
                assert permit._setup_connection is None and permit._setup_pragma is None
                assert selected is not None
                if setup_effect == "failed_close":
                    assert selected.connection is not None and selected.close_required
                    assert blocked_cursor is not None and blocked_cursor.close_attempts == 1
                    blocked_cursor.allow_cleanup.set()
                    selected.close()
                    assert blocked_cursor.close_attempts == 2
                assert selected.connection is None
            finally:
                if blocked_cursor is not None:
                    blocked_cursor.allow_cleanup.set()
                if selected is not None:
                    selected.close()


@pytest.mark.parametrize(
    "storage_class,literal",
    [
        ("blob", b""),
        ("text", b""),
        ("blob", bytes(range(256)) * 2344),
        ("text", b"\xff\x00\xc0\x80" * 150000),
        ("text", ("雪\u0000é" * 100000).encode("utf-8")),
    ],
)
def test_original_native_literal_slot_preserves_bytes_below_sqlite_value_limit(
    tmp_path: Path, storage_class: Literal["text", "blob"], literal: bytes
) -> None:
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    with write_lease("test.native-literal-slot", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
            seal._scratch.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1000000)
            # The hex predecessor exceeded this native value limit for the
            # same valid literal. The production slot retains its bytes once.
            cell = seal.retain_literal_stream(
                storage_class,
                len(literal),
                (literal[offset : offset + 32768] for offset in range(0, len(literal), 32768)),
            )
            expression, arguments = seal.source_literal_expression(cell)
            with closing(
                seal._scratch.execute(
                    f"SELECT typeof({expression}),length(CAST({expression} AS BLOB)),CAST({expression} AS BLOB)",
                    arguments * 3,
                )
            ) as cursor:
                result = cursor.fetchone()
            assert result == (storage_class, len(literal), literal)
            assert b"".join(seal._literal_cell_chunks(cell)) == literal
            with closing(seal._scratch.execute("SELECT count(*) FROM known_tier_literals")) as cursor:
                assert cursor.fetchone()[0] == 1
            null = seal.retain_literal_scalar(None)
            null_expression, null_arguments = seal.source_literal_expression(null)
            with closing(seal._scratch.execute(f"SELECT typeof({null_expression})", null_arguments)) as cursor:
                assert cursor.fetchone()[0] == "null"


def test_original_literal_preparation_retains_failed_actual_blob_and_directory_until_creator_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno

    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner, native_sql_children
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    with write_lease("test.literal-blob-retry", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        seal = PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path)
        original_close = NativeSQLCustodyOwner.close_incremental_blob
        scratch_owner = next(child for child in native_sql_children(seal) if child.connection is seal._scratch)
        witness = seal._original_witness_path()
        failure = OSError(errno.EIO, "synthetic original literal Blob close failure")
        blocked = True

        def close(owner: NativeSQLCustodyOwner, blob: sqlite3.Blob) -> None:
            if owner is scratch_owner and blocked:
                raise failure
            original_close(owner, blob)

        monkeypatch.setattr(NativeSQLCustodyOwner, "close_incremental_blob", close)
        try:
            with pytest.raises(NativeConnectionSettlementError) as caught:
                seal.retain_literal_stream("blob", 3, (b"abc",))
            assert caught.value.owner is scratch_owner and caught.value.failure is failure
            assert len(scratch_owner._incremental_blobs) == 1
            actual_blob = scratch_owner._incremental_blobs[0]
            assert len(actual_blob) == 3
            with pytest.raises(NativeConnectionSettlementError) as refused:
                seal.retain_literal_scalar(1)
            assert refused.value.owner is scratch_owner
            with pytest.raises(sqlite3.DatabaseError):
                with closing(seal._scratch.execute("DELETE FROM known_tier_literals")):
                    pass
            with pytest.raises(NativeConnectionSettlementError):
                seal.close()
            assert witness.exists() and scratch_owner.connection is seal._scratch
            assert actual_blob is scratch_owner._incremental_blobs[0]
            blocked = False
            seal.close()
            assert not witness.exists() and scratch_owner._settled
            assert not scratch_owner._incremental_blobs
            with pytest.raises(sqlite3.ProgrammingError):
                len(actual_blob)
        finally:
            blocked = False
            seal.close()


@pytest.mark.parametrize("failed_close", [False, True])
def test_literal_attachment_freezes_all_main_writes_until_actual_dedicated_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_close: bool
) -> None:
    import errno

    from polylogue.storage.sqlite.connection_profile import native_sql_children, open_source_tier_write_connection
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
    from tests.infra.sqlite_cursor_settlement import ControlledConnection, control_archive_connections

    with write_lease("test.literal-phase-gate", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with closing(open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)) as setup:
            with closing(setup.execute("CREATE TABLE authority_control(key TEXT PRIMARY KEY,value TEXT)")):
                pass
            with closing(setup.execute("INSERT INTO authority_control VALUES ('selected','original')")):
                pass
            setup.commit()
        with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
            cell = seal.retain_literal_stream("blob", 3, (b"abc",))
            cached_write = "UPDATE known_tier_literal_cells SET byte_length=byte_length WHERE cell_id=?"
            with closing(seal._scratch.execute(cached_write, (cell._cell_id,))):
                pass
            seal._scratch.commit()
            permit = seal.prepare_known_tier_mutation(
                "authority_control", ("value",), (("accepted", "selected"),), tier="source", key_column="key"
            )
            control_archive_connections(monkeypatch, tmp_path / "source.db")
            with permit.hold_authority():
                connection_context = permit.mutation_connection()
                source = connection_context.__enter__()
                assert isinstance(source, ControlledConnection)
                owner = next(child for child in native_sql_children(seal) if child.connection is source)
                witness = seal._original_witness_path()
                try:
                    expression, arguments = seal.source_literal_expression(cell)
                    with closing(source.execute(f"SELECT {expression}", arguments)) as cursor:
                        assert cursor.fetchone()[0] == b"abc"
                    with pytest.raises(ReferenceSealError):
                        seal.retain_literal_stream("blob", 1, (b"x",))
                    for statement in (cached_write, "DELETE FROM known_tier_literals WHERE rowid=?"):
                        with pytest.raises(sqlite3.DatabaseError):
                            with closing(seal._scratch.execute(statement, (cell._cell_id,))):
                                pass
                    with pytest.raises(sqlite3.DatabaseError):
                        with closing(seal._scratch.execute("CREATE TABLE unrelated_main_write(value BLOB)")):
                            pass
                    with pytest.raises(sqlite3.DatabaseError):
                        seal._scratch.blobopen("known_tier_literals", "literal", cell._cell_id, readonly=False)
                    with closing(
                        seal._scratch.execute("INSERT INTO temp.authorized_removals VALUES ('exact-temporary')")
                    ):
                        pass
                    seal._scratch.commit()
                    assert b"".join(seal._literal_cell_chunks(cell)) == b"abc"
                    if failed_close:
                        source.close_failure = OSError(errno.EIO, "synthetic actual attached writer close failure")
                        with pytest.raises(NativeConnectionSettlementError):
                            connection_context.__exit__(None, None, None)
                        assert witness.exists() and seal._live_literal_readers == {owner: permit}
                        assert owner.connection is source and owner.close_required
                        with pytest.raises(ReferenceSealError):
                            seal.retain_literal_scalar(1)
                        source.close_failure = None
                        owner.close()
                    else:
                        connection_context.__exit__(None, None, None)
                    assert not seal._live_literal_readers and owner.connection is None
                    assert seal.retain_literal_scalar(1)._seal is seal
                finally:
                    source.close_failure = None
                    owner.close()


def test_native_row_capture_reuses_exact_unchanged_cells_and_prepared_native_slots(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.connection_profile import open_source_tier_write_connection
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    value = b"retained\x00literal" * 40000
    with write_lease("test.literal-row-reuse", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with closing(open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)) as source:
            with closing(source.execute("CREATE TABLE authority_control(key TEXT PRIMARY KEY,value BLOB) STRICT")):
                pass
            with closing(source.execute("INSERT INTO authority_control VALUES ('selected',?)", (value,))):
                pass
            source.commit()
        with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
            with seal.original_read_snapshot():
                original = seal.retain_tier_row("source", "authority_control", 1)
                assert original is not None
                prepared = seal.retain_literal_stream(
                    "blob", len(value), (value[offset : offset + 32768] for offset in range(0, len(value), 32768))
                )
                reused = seal._retain_native_row(
                    seal.observer("source"),
                    "authority_control",
                    original.columns,
                    1,
                    reuse=original,
                    prepared_cells={"value": prepared},
                )
                assert reused is not None
                assert reused.cells[0] == original.cells[0] and reused.cells[1] == prepared
                wrong = seal.retain_literal_stream(
                    "blob",
                    len(value),
                    (b"x" * min(32768, len(value) - offset) for offset in range(0, len(value), 32768)),
                )
                unchanged = seal._retain_native_row(
                    seal.observer("source"),
                    "authority_control",
                    original.columns,
                    1,
                    reuse=original,
                    prepared_cells={"value": wrong},
                )
                assert unchanged is not None and unchanged.cells == original.cells
                # Original key/value plus the two actual prepared values.
                # Both native captures reuse locators rather than copying cells.
                with closing(seal._scratch.execute("SELECT count(*) FROM known_tier_literals")) as cursor:
                    assert cursor.fetchone()[0] == 4


@pytest.mark.parametrize("marker_first", [True, False])
def test_selected_source_hydration_preserves_actual_global_sequence_inputs_and_journal(
    tmp_path: Path, marker_first: bool
) -> None:
    import hashlib

    from polylogue.storage.sqlite.connection_profile import open_source_tier_write_connection
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    sequence = (1 << 62) + 17
    with write_lease("test.selected-source-baseline", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with closing(open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)) as source:
            statements = (
                (
                    "INSERT INTO accepted_marker_inputs(sequence,identity,raw_id,payload,payload_sha256) "
                    "VALUES (?,?,?,?,?)",
                    (
                        sequence + 10,
                        "a" * 64,
                        "selected",
                        b"neutral marker",
                        hashlib.sha256(b"neutral marker").hexdigest(),
                    ),
                ),
                ("INSERT INTO raw_existence_changes(sequence,raw_id) VALUES (?,?)", (sequence, "retained")),
            )
            for sql, parameters in statements if marker_first else reversed(statements):
                with closing(source.execute(sql, parameters)):
                    pass
            with closing(
                source.execute(
                    "INSERT INTO raw_sessions(raw_id,origin,source_path,blob_hash,blob_size,acquired_at_ms) "
                    "VALUES ('selected','codex','neutral-source',?,0,0)",
                    (bytes(32),),
                )
            ):
                pass
            source.commit()
        with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
            with seal.original_read_snapshot():
                with seal.original_rows(
                    "source", "SELECT rowid,name,seq FROM sqlite_sequence ORDER BY rowid"
                ) as cursor:
                    original_sequence = tuple(tuple(row) for row in cursor)
                with seal.original_rows("source", "SELECT rowid FROM raw_sessions WHERE raw_id='selected'") as cursor:
                    raw_rowid = cursor.fetchone()[0]
                with seal.original_rows(
                    "source", "SELECT sequence,raw_id FROM raw_existence_changes ORDER BY sequence"
                ) as cursor:
                    original_journal = tuple(tuple(row) for row in cursor)
                assert (sequence, "retained") in original_journal
                journal_inputs = tuple(
                    seal.retain_tier_row("source", "raw_existence_changes", row[0]) for row in original_journal
                )
                assert all(row is not None for row in journal_inputs)
                image = seal.retain_tier_row("source", "raw_sessions", raw_rowid)
                assert image is not None
                seal._provision_source_stage()
                for journal_input in journal_inputs:
                    assert journal_input is not None
                    seal._load_source_row(journal_input)
                seal._source_allocation_dependencies("raw_existence_changes")
                seal._source_allocation_dependencies("accepted_marker_inputs")
                assert seal._load_source_row(image)
                assert not seal._load_source_row(image)
                with closing(
                    seal._scratch.execute("SELECT rowid,name,seq FROM sqlite_sequence ORDER BY rowid")
                ) as cursor:
                    assert tuple(tuple(row) for row in cursor) == original_sequence
                with closing(
                    seal._scratch.execute("SELECT sequence,raw_id FROM raw_existence_changes ORDER BY sequence")
                ) as cursor:
                    assert tuple(tuple(row) for row in cursor) == original_journal
                with closing(
                    seal._scratch.execute("SELECT retained_floor FROM raw_existence_journal_control")
                ) as cursor:
                    assert cursor.fetchone()[0] == 0
                assert seal._scratch.getconfig(sqlite3.SQLITE_DBCONFIG_ENABLE_TRIGGER)
                with closing(seal._scratch.execute("PRAGMA foreign_key_check")) as cursor:
                    assert cursor.fetchone() is None
            seal.validate_observers_current()


def test_selected_source_loads_a_long_canonical_supersession_chain_without_python_recursion(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.connection_profile import open_source_tier_write_connection
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    chain_length = 1200
    with write_lease("test.selected-source-material-chain", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with closing(open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)) as source:
            for position in range(chain_length):
                with closing(
                    source.execute(
                        "INSERT INTO material_observations("
                        "material_id,referrer_ref,source_uri,acquisition_state,retryable,supersedes_material_id,"
                        "custody,privacy_classification,acquired_at_ms,created_at_ms) "
                        "VALUES (?,?,'https://example.invalid/material','claimed',0,?,'claimed','synthetic',0,0)",
                        (
                            f"material-{position}",
                            "session:neutral",
                            None if position == 0 else f"material-{position - 1}",
                        ),
                    )
                ):
                    pass
            source.commit()
        with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
            with seal.original_read_snapshot():
                with seal.original_rows(
                    "source",
                    "SELECT rowid FROM material_observations WHERE material_id=?",
                    (f"material-{chain_length - 1}",),
                ) as cursor:
                    rowid = cursor.fetchone()[0]
                image = seal.retain_tier_row("source", "material_observations", rowid)
                assert image is not None
                seal._provision_source_stage()
                assert seal._load_source_row(image)
                with closing(seal._scratch.execute("SELECT count(*) FROM material_observations")) as cursor:
                    assert cursor.fetchone()[0] == chain_length
                with closing(
                    seal._scratch.execute(
                        "SELECT count(*) FROM temp.polylogue_source_stage_rows "
                        "WHERE table_name='material_observations' AND load_state!=2"
                    )
                ) as cursor:
                    assert cursor.fetchone()[0] == 0
                with closing(seal._scratch.execute("PRAGMA foreign_key_check")) as cursor:
                    assert cursor.fetchone() is None
                assert seal._scratch.getconfig(sqlite3.SQLITE_DBCONFIG_ENABLE_TRIGGER)
            seal.validate_observers_current()


@pytest.mark.parametrize("index_state", ["absent", "unreadable"])
def test_source_only_witness_never_resolves_or_opens_missing_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index_state: str
) -> None:
    from polylogue.storage import archive_identity
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    with write_lease("test.source-only-capability", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
    # Existing Source remains authoritative even when the optional derived
    # tier cannot be read. Resolution itself must not occur on this route.
    selected = archive_identity.resolve_active_index_path(tmp_path)
    selected.unlink()
    if index_state == "unreadable":
        selected.mkdir()
    opened: list[str] = []
    original_open = PreparedIndexMutation._open_observer

    def forbidden_resolution(root: Path) -> Path:
        pytest.fail("Source-only capability attempted active Index resolution")

    def recorded_open(owner: PreparedIndexMutation, tier: str, path: Path) -> sqlite3.Connection:
        opened.append(tier)
        return original_open(owner, tier, path)

    monkeypatch.setattr(archive_identity, "resolve_active_index_path", forbidden_resolution)
    monkeypatch.setattr(PreparedIndexMutation, "_open_observer", recorded_open)
    with PreparedIndexMutation.source_only(archive_root=tmp_path) as seal:
        assert opened == ["source"]
        assert set(seal._observers) == {"source"}
        with seal.original_read_snapshot():
            with seal.original_rows("source", "SELECT 1") as rows:
                assert rows.fetchone()[0] == 1
        for tier in ("index", "user", "audit"):
            with pytest.raises(ReferenceSealError):
                seal.observer(tier)
        with pytest.raises(ReferenceSealError):
            seal.note_session_namespace_change()
        with pytest.raises(ReferenceSealError):
            seal.authorize_session_removal(("unproven-session",))
        with pytest.raises(ReferenceSealError):
            seal._retire_user_fields(projected=True)
        with pytest.raises(ReferenceSealError):
            _ = seal.index_identity
    assert opened == ["source"]


def test_source_only_witness_refuses_foreign_source_commit_and_literal_owner(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation, ReferenceSealStaleError

    with write_lease("test.source-only-currency", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
    with PreparedIndexMutation.source_only(archive_root=tmp_path) as first:
        with PreparedIndexMutation.source_only(archive_root=tmp_path) as second:
            cell = first.retain_literal_scalar("original owner")
            with pytest.raises(ReferenceSealError):
                second.source_literal_expression(cell)
        with closing(sqlite3.connect(tmp_path / "source.db")) as foreign:
            with closing(
                foreign.execute(
                    "INSERT INTO raw_sessions(raw_id,origin,source_path,blob_hash,blob_size,acquired_at_ms) "
                    "VALUES('foreign-source-only','unknown-export','synthetic/foreign',?,1,1)",
                    (b"f" * 32,),
                )
            ):
                pass
            foreign.commit()
        with pytest.raises(ReferenceSealStaleError):
            first.validate_observers_current()


def test_user_reference_array_census_does_not_fetch_complete_json_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typing import Any

    from polylogue.storage.sqlite import reference_seal
    from tests.infra.sqlite_cursor_settlement import ControlledCursor

    array_fields = frozenset({"evidence_refs_json", "assertion_refs_json", "model_refs_json"})
    charges: list[int] = []

    class ArrayFetchGuard(ControlledCursor):
        def execute(self, sql: str, parameters: Any = (), /) -> "ArrayFetchGuard":
            if sql.startswith("SELECT rowid AS physical_rowid,") and 'FROM "assertions" WHERE rowid=' in sql:
                assert charges, "constructor reference fields hydrated before demand registration"
            result = super().execute(sql, parameters)
            assert self.description is None or not any(column[0] in array_fields for column in self.description), (
                "reference census transferred a complete durable JSON array"
            )
            return result

    original_open = reference_seal.PreparedIndexMutation._open_observer

    def open_observer(seal: reference_seal.PreparedIndexMutation, name: str, path: Path) -> sqlite3.Connection:
        connection = original_open(seal, name, path)
        if name == "user":
            original_cursor = connection.cursor

            def guarded_cursor() -> sqlite3.Cursor:
                return original_cursor(factory=ArrayFetchGuard)

            monkeypatch.setattr(connection, "cursor", guarded_cursor)
        return connection

    with write_lease("test.native-user-reference-arrays", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with closing(open_connection(tmp_path / "user.db", archive_root=tmp_path)) as user:
            upsert_assertion(
                user,
                assertion_id="large-reference-array",
                target_ref="user:local",
                kind=AssertionKind.ANNOTATION,
                evidence_refs=("user:local",) * 20000,
                now_ms=1,
            )
            user.commit()
        monkeypatch.setattr(reference_seal.PreparedIndexMutation, "_open_observer", open_observer)
        with reference_seal.PreparedIndexMutation(
            tmp_path / "index.db", archive_root=tmp_path, input_demand=charges.append
        ) as seal:
            assert sum(charges) > 200000
            initial = tuple(charges)
            with seal.original_read_snapshot():
                with closing(reference_seal._references_from_user(seal.observer("user"))) as anchors:
                    assert sum(anchor.field == "evidence_refs_json" for anchor in anchors) == 20000
            assert tuple(charges) == initial


@pytest.mark.parametrize("reference_kind", ["session", "run", "observed-event", "context-snapshot"])
def test_constructor_accounts_original_index_identity_before_resolving_long_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference_kind: str
) -> None:
    from typing import Any

    from polylogue.storage.sqlite import reference_seal
    from tests.infra.sqlite_cursor_settlement import ControlledCursor

    charges: list[int] = []
    original_open = reference_seal.PreparedIndexMutation._open_observer
    checked: list[int] = []

    def open_observer(seal: reference_seal.PreparedIndexMutation, name: str, path: Path) -> sqlite3.Connection:
        connection = original_open(seal, name, path)
        if name == "index":
            original_cursor = connection.cursor

            class IdentityFetchGuard(ControlledCursor):
                def execute(self, sql: str, parameters: Any = (), /) -> "IdentityFetchGuard":
                    if sql.startswith(
                        (
                            "SELECT session_id FROM sessions WHERE rowid=",
                            'SELECT "session_id" FROM "sessions" WHERE rowid=',
                        )
                    ):
                        assert isinstance(parameters, tuple)
                        with seal._owned_cursor(
                            seal._scratch,
                            "SELECT byte_length FROM temp.original_input_fields WHERE tier='index' "
                            "AND table_name='sessions' AND column_name='session_id' AND row_address=?",
                            parameters,
                        ) as metadata:
                            row = metadata.fetchone()
                        assert row is not None and row[0] > 20000
                        checked.append(row[0])
                    return super().execute(sql, parameters)

            def guarded_cursor() -> sqlite3.Cursor:
                return original_cursor(factory=IdentityFetchGuard)

            monkeypatch.setattr(connection, "cursor", guarded_cursor)
        return connection

    with write_lease("test.constructor-index-input", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            session_id = write_index_session(archive, reference_session("long-" + "x" * 20000))
            archive.commit()
        target_ref = f"{reference_kind}:{session_id}"
        if reference_kind == "observed-event":
            target_ref += ":session_started"
        elif reference_kind == "context-snapshot":
            target_ref += ":session_start"
        with closing(open_connection(tmp_path / "user.db", archive_root=tmp_path)) as user:
            upsert_assertion(
                user,
                assertion_id="long-session-reference",
                target_ref=target_ref,
                kind=AssertionKind.ANNOTATION,
                now_ms=1,
            )
            user.commit()
        monkeypatch.setattr(reference_seal.PreparedIndexMutation, "_open_observer", open_observer)
        with reference_seal.PreparedIndexMutation(
            tmp_path / "index.db", archive_root=tmp_path, input_demand=charges.append
        ) as seal:
            assert checked and sum(charges) >= checked[0]
            initial = tuple(charges)
            with seal.original_read_snapshot():
                seal.before_index_input(
                    "sessions", ("session_id",), "SELECT rowid FROM sessions WHERE session_id=?", (session_id,)
                )
            assert tuple(charges) == initial


def test_partial_update_factory_retains_paid_original_cells_outside_effect_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite.connection_profile import open_source_tier_write_connection
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    value = b"original-cell" * 100000
    charges: list[int] = []
    failure = OSError("synthetic terminal preparation verification failure")
    with write_lease("test.partial-update-original-demand", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with closing(open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)) as source:
            with closing(source.execute("CREATE TABLE authority_control(key TEXT PRIMARY KEY,value BLOB)")):
                pass
            with closing(source.execute("INSERT INTO authority_control VALUES ('selected',?)", (value,))):
                pass
            source.commit()
        with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path, input_demand=charges.append) as seal:
            original_verify = seal._verify_implicit_sequence_effects
            injected = False

            def verify(tier: str) -> None:
                nonlocal injected
                original_verify(tier)
                if not injected:
                    injected = True
                    raise failure

            monkeypatch.setattr(seal, "_verify_implicit_sequence_effects", verify)
            with pytest.raises(OSError) as caught:
                seal.prepare_known_tier_mutation(
                    "authority_control", ("value",), (("accepted", "selected"),), tier="source", key_column="key"
                )
            assert caught.value is failure
            paid = tuple(charges)
            with seal._owned_cursor(
                seal._scratch,
                "SELECT column_name,byte_length FROM temp.original_input_fields "
                "WHERE tier='source' AND table_name='authority_control' AND row_address=1",
            ) as cursor:
                assert dict(cursor) == {"key": len("selected"), "value": len(value)}
            with seal._owned_cursor(seal._scratch, "SELECT count(*) FROM temp.known_tier_effects") as cursor:
                assert cursor.fetchone()[0] == 0
            assert "source" not in seal._pending_tier_permits
            permit = seal.prepare_known_tier_mutation(
                "authority_control", ("value",), (("accepted", "selected"),), tier="source", key_column="key"
            )
            assert tuple(charges) == paid
            with permit.hold_authority(), permit.mutation_connection() as source:
                with closing(source.execute("BEGIN IMMEDIATE")):
                    pass
                with closing(source.execute("UPDATE authority_control SET value='accepted' WHERE key='selected'")):
                    pass
                permit.allow_commit(source)
                source.commit()
                seal.accept_known_tier_commit(permit.committed())
            seal.validate_observers_current()


def test_partial_update_factory_rolls_back_effects_after_original_window_currency_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite.connection_profile import open_source_tier_write_connection
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation, ReferenceSealStaleError

    with write_lease("test.partial-update-original-window", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with closing(open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)) as source:
            with closing(source.execute("CREATE TABLE authority_control(key TEXT PRIMARY KEY,value TEXT)")):
                pass
            with closing(source.execute("INSERT INTO authority_control VALUES ('selected','original')")):
                pass
            source.commit()
        with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
            original_verify = seal._verify_implicit_sequence_effects

            def commit_between_collection_and_currency(tier: str) -> None:
                original_verify(tier)
                assert seal._original_reads_active and seal.observer("source").in_transaction
                with closing(sqlite3.connect(tmp_path / "source.db")) as foreign:
                    with closing(foreign.execute("INSERT INTO authority_control VALUES ('unrelated','foreign')")):
                        pass
                    foreign.commit()

            monkeypatch.setattr(seal, "_verify_implicit_sequence_effects", commit_between_collection_and_currency)
            with pytest.raises(ReferenceSealStaleError):
                seal.prepare_known_tier_mutation(
                    "authority_control", ("value",), (("accepted", "selected"),), tier="source", key_column="key"
                )
            assert not seal._original_reads_active and not seal.observer("source").in_transaction
            assert "source" not in seal._pending_tier_permits
            with seal._owned_cursor(seal._scratch, "SELECT count(*) FROM temp.known_tier_effects") as cursor:
                assert cursor.fetchone()[0] == 0
            with seal._owned_cursor(
                seal.observer("source"), "SELECT value FROM authority_control WHERE key='selected'"
            ) as cursor:
                assert cursor.fetchone()[0] == "original"


def test_partial_update_factory_rollback_close_failure_retains_primary_and_original_creator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite.connection_profile import native_sql_children, open_source_tier_write_connection
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
    from tests.infra.sqlite_cursor_settlement import ControlledCursor

    primary = OSError("synthetic exact effect verification fault")
    blocked: list[ControlledCursor] = []
    with write_lease("test.partial-update-rollback-close", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with closing(open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)) as source:
            with closing(source.execute("CREATE TABLE authority_control(key TEXT PRIMARY KEY,value TEXT)")):
                pass
            with closing(source.execute("INSERT INTO authority_control VALUES ('selected','original')")):
                pass
            source.commit()
        seal = PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path)
        owner = next(child for child in native_sql_children(seal) if child.connection is seal._scratch)
        assert seal._scratch_directory is not None
        witness = Path(seal._scratch_directory.name)
        original_cursor = seal._scratch.cursor

        class RollbackCursor(ControlledCursor):
            def execute(self, sql: str, parameters: Any = (), /) -> "RollbackCursor":
                super().execute(sql, parameters)
                if sql == "ROLLBACK TO prepare_known_tier_effects":
                    self.allow_cleanup.clear()
                    blocked.append(self)
                return self

        def cursor() -> sqlite3.Cursor:
            return original_cursor(factory=RollbackCursor)

        def refuse(tier: str) -> None:
            raise primary

        monkeypatch.setattr(seal._scratch, "cursor", cursor)
        monkeypatch.setattr(seal, "_verify_implicit_sequence_effects", refuse)
        try:
            with pytest.raises(BaseExceptionGroup) as caught:
                seal.prepare_known_tier_mutation(
                    "authority_control", ("value",), (("accepted", "selected"),), tier="source", key_column="key"
                )

            def includes(failure: BaseException, target: BaseException) -> bool:
                if failure is target:
                    return True
                if isinstance(failure, BaseExceptionGroup):
                    return any(includes(child, target) for child in failure.exceptions)
                return False

            assert includes(caught.value, primary) and blocked
            assert owner.connection is seal._scratch and owner.close_required
            assert witness.exists() and "source" not in seal._pending_tier_permits
            with pytest.raises(ReferenceSealError):
                seal.prepare_known_tier_mutation(tier="source", effects=())
            with pytest.raises(NativeConnectionSettlementError) as close_failure:
                seal.close()
            assert close_failure.value.owner is owner
            assert owner.connection is seal._scratch
            assert witness.exists()
            for selected in blocked:
                selected.allow_cleanup.set()
            seal.close()
            assert owner.connection is None and not witness.exists()
        finally:
            for selected in blocked:
                selected.allow_cleanup.set()
            seal.close()


@pytest.mark.parametrize("first_tier", ["source", "user"])
@pytest.mark.parametrize("settlement_failure", ["none", "close", "authorizer"])
def test_overlapping_literal_children_freeze_main_until_last_physical_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_tier: str, settlement_failure: str
) -> None:
    import errno

    from polylogue.storage.sqlite.connection_profile import native_sql_children
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
    from polylogue.storage.sqlite.write_lease import ArchiveWriteCustody
    from tests.infra.sqlite_cursor_settlement import ControlledConnection, control_archive_connections

    released: list[object] = []
    actual_release = ArchiveWriteCustody.release_sql_owner
    selected_seal = None

    def release(custody: ArchiveWriteCustody, owner: object) -> None:
        if owner is selected_seal:
            released.append(owner)
        return actual_release(custody, owner)

    monkeypatch.setattr(ArchiveWriteCustody, "release_sql_owner", release)
    with write_lease("test.overlapping-literal-children", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
            selected_seal = seal
            cell = seal.retain_literal_stream("blob", 3, (b"abc",))
            source_permit = seal.prepare_known_tier_mutation(tier="source", effects=())
            user_permit = seal.prepare_known_tier_mutation(tier="user", effects=())
            # Construct the foreign original observer before the controlled
            # writable factory selects existing inode identities.
            foreign = PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path)
            control_archive_connections(monkeypatch, tmp_path / "source.db", tmp_path / "user.db")
            with pytest.raises(ReferenceSealError):
                seal._settle_frozen_literal_bookkeeping()
            contexts = {}
            owners = {}
            try:
                for tier, permit in (("user", user_permit), ("source", source_permit)):
                    with permit.hold_authority():
                        context = permit.mutation_connection()
                        connection = context.__enter__()
                        contexts[tier] = context
                        assert isinstance(connection, ControlledConnection)
                        owners[tier] = next(
                            child for child in native_sql_children(seal) if child.connection is connection
                        )
                        with closing(connection.execute("BEGIN IMMEDIATE")):
                            pass
                        expression, bindings = seal.source_literal_expression(cell)
                        with closing(connection.execute(f"SELECT {expression}", bindings)) as cursor:
                            assert cursor.fetchone()[0] == b"abc"
                        if tier == "source":
                            permit.allow_commit(connection)
                            connection.commit()
                            seal.accept_known_tier_commit(permit.committed())
                    # Only TEMP bookkeeping can change while either literal
                    # reader retains the same immutable original MAIN.
                    with closing(
                        seal._scratch.execute("INSERT OR IGNORE INTO temp.authorized_removals VALUES (?)", (tier,))
                    ):
                        pass
                    with closing(seal._scratch.execute("SELECT 1 UNION ALL SELECT 2")) as live:
                        assert live.fetchone()[0] == 1
                        with pytest.raises(ReferenceSealError):
                            seal._settle_frozen_literal_bookkeeping()
                    scratch_owner = next(
                        child for child in native_sql_children(seal) if child.connection is seal._scratch
                    )
                    with scratch_owner.readonly_blob("known_tier_literals", "literal", cell._cell_id):
                        with pytest.raises(ReferenceSealError):
                            seal._settle_frozen_literal_bookkeeping()
                    seal._settle_frozen_literal_bookkeeping()
                assert owners["user"].require_connection().in_transaction
                assert seal._live_literal_readers == {owners["user"]: user_permit, owners["source"]: source_permit}
                with pytest.raises(ReferenceSealError):
                    seal._retain_live_literal_reader(owners["user"], user_permit)
                with pytest.raises(ReferenceSealError):
                    foreign._retain_live_literal_reader(owners["user"], user_permit)
                for tier in (first_tier, "user" if first_tier == "source" else "source"):
                    owner = owners[tier]
                    connection = owner.require_connection()
                    assert isinstance(connection, ControlledConnection)
                    if settlement_failure != "none" and tier == first_tier:
                        actual_authorizer = seal._scratch.set_authorizer
                        if settlement_failure == "close":
                            connection.close_failure = OSError(errno.EIO, "exact overlapping reader close failure")
                        else:

                            def fail_authorizer(
                                callback: Callable[[int, str | None, str | None, str | None, str | None], int] | None,
                            ) -> None:
                                raise OSError(errno.EIO, "exact frozen authorizer settlement failure")

                            monkeypatch.setattr(seal._scratch, "set_authorizer", fail_authorizer)
                        with pytest.raises(NativeConnectionSettlementError):
                            contexts[tier].__exit__(None, None, None)
                        assert owner in seal._live_literal_readers
                        assert owner.connection is (connection if settlement_failure == "close" else None)
                        assert owner.close_required
                        assert released == []
                        with pytest.raises(sqlite3.DatabaseError):
                            with closing(seal._scratch.execute("DELETE FROM known_tier_literals")):
                                pass
                        connection.close_failure = None
                        monkeypatch.setattr(seal._scratch, "set_authorizer", actual_authorizer)
                        owner.close()
                    else:
                        contexts[tier].__exit__(None, None, None)
                    assert owner not in seal._live_literal_readers and owner.connection is None
                    if seal._live_literal_readers:
                        assert released == [] and seal._mutation_custody is not None
                        with pytest.raises(ReferenceSealError):
                            seal.retain_literal_scalar(1)
                        with pytest.raises(sqlite3.DatabaseError):
                            with closing(seal._scratch.execute("DELETE FROM known_tier_literals")):
                                pass
                        with pytest.raises(sqlite3.DatabaseError):
                            seal._scratch.blobopen("known_tier_literals", "literal", cell._cell_id, readonly=False)
                assert released == [seal] and seal._mutation_custody is None
                assert not seal._literal_custody_release_pending
                assert seal.retain_literal_scalar(1)._seal is seal
            finally:
                for owner in owners.values():
                    if owner.connection is not None:
                        assert isinstance(owner.connection, ControlledConnection)
                        owner.connection.close_failure = None
                    owner.close()
                foreign.close()


@pytest.mark.parametrize("cancel_site", ["source_admission", "after_source_close"])
def test_overlapping_literal_cancellation_preserves_exact_native_obligations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_site: str
) -> None:
    import asyncio
    import threading

    from polylogue.core.compute_cancel import compute_cancel
    from polylogue.storage.sqlite.connection_profile import native_sql_children, open_source_tier_write_connection
    from polylogue.storage.sqlite.reference_seal import KnownTierMutationPermit, PreparedIndexMutation

    cancelled = threading.Event()
    token = compute_cancel.set(cancelled)
    try:
        with write_lease("test.overlapping-literal-cancellation", archive_root=tmp_path):
            bootstrap_archive_root(tmp_path)
            with closing(open_source_tier_write_connection(tmp_path / "source.db", archive_root=tmp_path)) as setup:
                with closing(setup.execute("CREATE TABLE authority_control(key TEXT PRIMARY KEY,value TEXT)")):
                    pass
                with closing(setup.execute("INSERT INTO authority_control VALUES('selected','original')")):
                    pass
                setup.commit()
            with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
                user_permit = seal.prepare_known_tier_mutation(tier="user", effects=())
                source_permit = seal.prepare_known_tier_mutation(
                    "authority_control", ("value",), (("accepted", "selected"),), tier="source", key_column="key"
                )
                with user_permit.hold_authority():
                    user_context = user_permit.mutation_connection()
                    user = user_context.__enter__()
                    owner = next(child for child in native_sql_children(seal) if child.connection is user)
                    with closing(user.execute("BEGIN IMMEDIATE")):
                        pass
                actual_configure = KnownTierMutationPermit.configure_mutation_connection
                reached = []

                def configure(
                    permit: KnownTierMutationPermit, connection: sqlite3.Connection, statements: tuple[str, ...]
                ) -> None:
                    if permit is source_permit and cancel_site == "source_admission":
                        reached.append(True)
                        cancelled.set()
                    return actual_configure(permit, connection, statements)

                monkeypatch.setattr(KnownTierMutationPermit, "configure_mutation_connection", configure)
                try:
                    with pytest.raises(asyncio.CancelledError):
                        with source_permit.hold_authority(), source_permit.mutation_connection() as source:
                            with closing(source.execute("BEGIN IMMEDIATE")):
                                pass
                            with closing(
                                source.execute("UPDATE authority_control SET value='accepted' WHERE key='selected'")
                            ):
                                pass
                            source_permit.allow_commit(source)
                            source.commit()
                            seal.accept_known_tier_commit(source_permit.committed())
                        reached.append(True)
                        cancelled.set()
                        seal._require_new_work()
                    assert reached == [True]
                    assert seal._live_literal_readers == {owner: user_permit}
                    assert user.in_transaction and seal._mutation_custody is not None
                    # Cancellation never substitutes a logical close for this
                    # actual original User transaction's rollback/settlement.
                    user_context.__exit__(None, None, None)
                    assert owner.connection is None and not seal._live_literal_readers
                    assert not seal._literal_custody_release_pending
                    if cancel_site == "after_source_close":
                        assert seal._mutation_custody is None
                finally:
                    owner.close()
                cancelled.clear()
            with closing(sqlite3.connect(tmp_path / "source.db")) as observed:
                with closing(observed.execute("SELECT value FROM authority_control WHERE key='selected'")) as rows:
                    assert rows.fetchone()[0] == ("accepted" if cancel_site == "after_source_close" else "original")
    finally:
        cancelled.clear()
        compute_cancel.reset(token)


@pytest.mark.parametrize("effect", ["ddl", "dml", "noop"])
async def test_exact_active_index_commit_accepts_own_state_without_change_count_authority(
    tmp_path: Path, effect: str
) -> None:
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
    from tests.infra.archive_templates import run_archive_fixture_write

    def run() -> None:
        with write_lease("test.index-own-state", archive_root=tmp_path):
            bootstrap_archive_root(tmp_path)
            with closing(ArchiveStore.open_existing(tmp_path, read_only=False)) as store:
                with closing(store._conn.execute("CREATE TABLE acceptance_control(value INTEGER)")):
                    pass
                with closing(store._conn.execute("INSERT INTO acceptance_control VALUES(1)")):
                    pass
                store._conn.commit()
                with PreparedIndexMutation(store.index_db_path, archive_root=tmp_path) as seal:
                    prior = seal._versions["index"]
                    with seal.mutation_scope(store._conn) as scope:
                        sql = {
                            "ddl": "CREATE TABLE accepted_ddl(value INTEGER)",
                            "dml": "UPDATE acceptance_control SET value=2",
                            "noop": "UPDATE acceptance_control SET value=2 WHERE 0",
                        }[effect]
                        with closing(store._conn.execute(sql)):
                            pass
                        scope.commit()
                    receipt = scope.commit_receipt
                    seal.require_index_commit_receipt(receipt)
                    assert receipt._writer is store._conn and receipt._scope is scope and receipt._seal is seal
                    assert receipt._effect_count == int(effect == "dml")
                    assert scope._writer_owner is not None
                    assert scope._writer_owner.connection is store._conn and not store._conn.in_transaction
                    assert (seal._versions["index"] != prior) == (effect != "noop")
                    with pytest.raises(ReferenceSealError):
                        with seal.mutation_scope(store._conn):
                            pytest.fail("accepted active seal admitted a second Index publication")
                    if effect == "ddl":
                        assert store._conn.execute("SELECT count(*) FROM accepted_ddl").fetchone()[0] == 0
                    else:
                        assert store._conn.execute("SELECT value FROM acceptance_control").fetchone()[0] == (
                            2 if effect == "dml" else 1
                        )

    await run_archive_fixture_write(tmp_path, run)


async def test_foreign_ddl_in_exact_index_commit_gap_refuses_even_with_zero_total_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite.connection_profile import open_isolated_write_connection
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation, ReferenceSealStaleError
    from tests.infra.archive_templates import run_archive_fixture_write

    def run() -> None:
        with write_lease("test.index-foreign-ddl-gap", archive_root=tmp_path):
            bootstrap_archive_root(tmp_path)
            with closing(ArchiveStore.open_existing(tmp_path, read_only=False)) as store:
                with PreparedIndexMutation(store.index_db_path, archive_root=tmp_path) as seal:
                    prior = seal._versions["index"]
                    actual_commit = store._conn.commit
                    foreign = []

                    def commit_then_foreign() -> None:
                        actual_commit()
                        with closing(
                            open_isolated_write_connection(
                                store.index_db_path, archive_root=tmp_path, purpose="test.foreign-ddl"
                            )
                        ) as other:
                            with closing(other.execute("CREATE TABLE foreign_gap_ddl(value INTEGER)")):
                                pass
                            assert other.total_changes == 0
                            other.commit()
                            foreign.append(True)

                    with monkeypatch.context() as patch:
                        patch.setattr(store._conn, "commit", commit_then_foreign)
                        with pytest.raises(ReferenceSealStaleError):
                            with seal.mutation_scope(store._conn) as scope:
                                original_changes = store._conn.total_changes
                                with closing(store._conn.execute("CREATE TABLE own_gap_ddl(value INTEGER)")):
                                    pass
                                assert store._conn.total_changes == original_changes
                                scope.commit()
                    assert foreign == [True] and scope._committed and not scope._index_accepted
                    assert seal._versions["index"] == prior and seal._accepted_index_commit is None
                    with pytest.raises(ReferenceSealError):
                        _ = scope.commit_receipt
                    for name in ("own_gap_ddl", "foreign_gap_ddl"):
                        assert (
                            store._conn.execute("SELECT name FROM sqlite_schema WHERE name=?", (name,)).fetchone()[0]
                            == name
                        )

    await run_archive_fixture_write(tmp_path, run)


@pytest.mark.parametrize("state", ["prepared", "unprepared", "source-only", "foreign", "stale", "cursor", "snapshot"])
def test_candidate_observer_requires_its_original_index_promotion_proof(tmp_path: Path, state: str) -> None:
    from polylogue.storage.index_generation import IndexGenerationStore
    from polylogue.storage.io_phase_metrics import connect_measured, connection_cursor
    from polylogue.storage.sqlite.connection_profile import native_sql_children
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation, ReferenceSealStaleError

    with write_lease("test.candidate-role-fixture", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        generations = IndexGenerationStore.for_archive_root(tmp_path)
        candidate = generations.create(source_snapshot="candidate-role")
        foreign = generations.create(source_snapshot="foreign-candidate-role")
    path = Path(candidate.index_path)
    seal = (
        PreparedIndexMutation.source_only(archive_root=tmp_path)
        if state == "source-only"
        else PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path)
    )
    with seal:
        if state == "source-only":
            with pytest.raises(ReferenceSealError):
                seal.prepare_candidate_reachability(path)
            with pytest.raises(ReferenceSealError):
                seal._require_unpinned_observer("candidate")
            assert "candidate" not in seal._observers
        elif state == "unprepared":
            with pytest.raises(ReferenceSealError):
                seal._require_unpinned_observer("candidate")
            assert "candidate" not in seal._observers
        else:
            identity = seal.prepare_candidate_reachability(path)
            observer = seal._observers["candidate"]
            assert any(owner.connection is observer for owner in native_sql_children(seal))
            # The private promotion role does not become a public storage tier.
            with pytest.raises(ReferenceSealError):
                seal.observer("candidate")
            if state == "foreign":
                with pytest.raises(ReferenceSealStaleError):
                    seal.validate_candidate_current(Path(foreign.index_path))
            elif state == "stale":
                with write_lease("test.foreign-candidate-commit", archive_root=tmp_path):
                    writer = connect_measured(path)
                    owner = NativeSQLCustodyOwner(writer)
                    try:
                        with connection_cursor(writer, "CREATE TABLE foreign_candidate_effect(value INTEGER)"):
                            pass
                        writer.commit()
                    finally:
                        owner.close()
                with pytest.raises(ReferenceSealStaleError):
                    seal.validate_candidate_current(path)
                assert seal._candidate_identity == identity
            elif state == "cursor":
                with connection_cursor(observer, "SELECT 1"):
                    with pytest.raises(ReferenceSealError):
                        seal.validate_candidate_current(path)
                assert seal.validate_candidate_current(path) == identity
            elif state == "snapshot":
                with connection_cursor(observer, "BEGIN"):
                    pass
                try:
                    with pytest.raises(ReferenceSealError):
                        seal.validate_candidate_current(path)
                finally:
                    observer.rollback()
                assert seal.validate_candidate_current(path) == identity
            else:
                assert seal.validate_candidate_current(path) == identity
    assert native_sql_children(seal) == ()


@pytest.mark.parametrize("failed_resource", ["payload", "cursor"])
@pytest.mark.parametrize("terminal_reason", ["cancelled", "namespace_changed", "source_producer_failed"])
def test_preparation_payload_waits_for_original_sql_and_retries_failed_cleanup(
    tmp_path: Path, failed_resource: str, terminal_reason: str
) -> None:
    import asyncio
    import threading
    from tempfile import TemporaryDirectory

    from polylogue.core.compute_cancel import compute_cancel
    from polylogue.core.sql_settlement import retain_native_sql_lifetimes
    from polylogue.storage.derived.raw import _cleanup_scratch
    from polylogue.storage.sqlite.connection_profile import (
        native_sql_children,
        retained_native_sql_owners_for_lifetime,
    )
    from polylogue.storage.sqlite.reference_seal import (
        PreparedIndexMutation,
        ReferenceSealStaleError,
        _check_reference_cancellation,
    )
    from tests.infra.sqlite_cursor_settlement import ControlledCursor

    with write_lease("test.preparation-payload", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
        scratch = TemporaryDirectory(dir=tmp_path)
        seal = PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path)
        cursor = None

        def open_pending_cursor() -> ControlledCursor:
            with retain_native_sql_lifetimes(scratch):
                pending = seal.observer("user").cursor(factory=ControlledCursor)
                pending.execute("SELECT 1 UNION ALL SELECT 2")
                next(pending)
            if failed_resource == "cursor":
                pending.cleanup_failure = fault
                pending.allow_cleanup.clear()
            return pending

        fault = OSError("synthetic original preparation cleanup failure")
        attempts = 0
        payload_fails = failed_resource == "payload"

        def close_payload() -> None:
            nonlocal attempts
            attempts += 1
            assert all(owner._settled for owner in native_sql_children(seal))
            assert not retained_native_sql_owners_for_lifetime(scratch)
            if payload_fails:
                raise fault
            _cleanup_scratch(scratch)

        cancellation_token = None
        displaced_index = tmp_path / "displaced-index.db"
        try:
            if terminal_reason != "source_producer_failed":
                cursor = open_pending_cursor()
            if terminal_reason == "cancelled":
                cancellation_flag = threading.Event()
                cancellation_flag.set()
                cancellation_token = compute_cancel.set(cancellation_flag)
                with pytest.raises(asyncio.CancelledError) as cancelled:
                    _check_reference_cancellation()
                primary: BaseException = cancelled.value
            elif terminal_reason == "namespace_changed":
                (tmp_path / "index.db").rename(displaced_index)
                with pytest.raises(ReferenceSealStaleError) as stale:
                    seal._assert_configured_namespace()
                primary = stale.value
            else:
                producer_failure = RuntimeError("synthetic original Source producer failure")
                with pytest.raises(BaseExceptionGroup) as failed:
                    with seal.original_read_snapshot(), seal.source_producer():
                        cursor = open_pending_cursor()
                        raise producer_failure
                assert failed.value.exceptions[0] is producer_failure
                assert any(isinstance(error, NativeConnectionSettlementError) for error in failed.value.exceptions[1:])
                assert seal._cleanup_requested
                primary = failed.value
            try:
                raise primary
            except BaseException as original:
                assert original is primary
                seal.retain_preparation_payload(close_payload)
            with pytest.raises((OSError, NativeConnectionSettlementError)) as cleanup:
                seal.close()
            if isinstance(cleanup.value, NativeConnectionSettlementError):
                assert isinstance(cleanup.value.failure, BaseExceptionGroup)
                assert cleanup.value.failure.exceptions == (fault,)
            else:
                assert cleanup.value is fault
            assert not seal._closed and Path(scratch.name).exists()
            assert attempts == (1 if failed_resource == "payload" else 0)
            payload_fails = False
            if cursor is not None:
                cursor.allow_cleanup.set()
            seal.close()
            assert seal._closed and not Path(scratch.name).exists()
            assert attempts == (2 if failed_resource == "payload" else 1)
            assert not native_sql_children(seal)
        finally:
            payload_fails = False
            if cursor is not None:
                cursor.allow_cleanup.set()
            try:
                seal.close()
            finally:
                if cancellation_token is not None:
                    compute_cancel.reset(cancellation_token)
                if displaced_index.exists():
                    displaced_index.rename(tmp_path / "index.db")
