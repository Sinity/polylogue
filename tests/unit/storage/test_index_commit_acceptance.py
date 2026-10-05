"""Exact original Index commit acceptance on an admitted physical writer."""

import asyncio
import sqlite3
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from polylogue.core.compute_cancel import compute_cancel
from polylogue.storage.io_phase_metrics import connect_measured, connection_cursor
from polylogue.storage.sqlite import reference_seal
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.source_items import publish_source_generation
from polylogue.storage.sqlite.connection_profile import (
    NativeConnectionSettlementError,
    native_sql_children,
    native_sql_owner_for_connection,
    open_isolated_write_connection,
)
from polylogue.storage.sqlite.reference_seal import (
    IndexCommitReceipt,
    PreparedIndexMutation,
    ReferenceSealError,
    ReferenceSealStaleError,
)
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.storage_records import SessionBuilder


async def test_handed_off_index_writer_is_captured_once_and_physically_closed(tmp_path: Path) -> None:
    def run() -> None:
        with write_lease("test.index-handed-off", archive_root=tmp_path):
            _seed(tmp_path)
            writer = open_isolated_write_connection(
                tmp_path / "index.db", purpose="synthetic handed-off writer", archive_root=tmp_path
            )
            assert native_sql_owner_for_connection(writer) is None
            with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
                with seal.mutation_scope(writer) as scope:
                    _update(writer)
                    scope.commit()
                    owner = native_sql_owner_for_connection(writer)
                    assert owner is not None
                    assert owner is scope._writer_owner and owner._terminal_parent is None
                    assert owner.connection is writer and not owner._settled
                seal.require_index_commit_receipt(scope.commit_receipt)
                assert native_sql_owner_for_connection(writer) is owner
                writer.close()
                owner.close()
                assert owner._settled and owner.connection is None

    await run_archive_fixture_write(tmp_path, run)


async def test_handed_off_foreign_creator_is_not_recaptured(tmp_path: Path) -> None:
    def run() -> None:
        with write_lease("test.index-foreign-creator", archive_root=tmp_path):
            _seed(tmp_path)
            with ThreadPoolExecutor(max_workers=1) as creator:
                writer = creator.submit(connect_measured, tmp_path / "index.db", check_same_thread=False).result()
                try:
                    with PreparedIndexMutation(tmp_path / "index.db", archive_root=tmp_path) as seal:
                        with pytest.raises(ReferenceSealError, match="original measured physical creator"):
                            with seal.mutation_scope(writer):
                                pytest.fail("foreign creator entered original Index producer")
                        assert native_sql_owner_for_connection(writer) is None
                        assert not writer.in_transaction
                finally:
                    creator.submit(writer.close).result()

    await run_archive_fixture_write(tmp_path, run)


def _seed(root: Path) -> None:
    bootstrap_archive_root(root)
    SessionBuilder(root / "index.db", "index-acceptance").provider("codex").add_message(text="Neutral").save()


def _title(connection: sqlite3.Connection) -> str | None:
    with connection_cursor(connection, "SELECT title FROM sessions") as rows:
        title: str | None = rows.fetchone()[0]
        return title


def _update(connection: sqlite3.Connection) -> None:
    with connection_cursor(connection, "UPDATE sessions SET title='accepted-index'") as rows:
        assert rows.rowcount == 1


async def test_exact_index_commit_accepts_original_observer_for_source_only(tmp_path: Path) -> None:
    def run() -> None:
        with write_lease("test.index-acceptance", archive_root=tmp_path):
            _seed(tmp_path)
            with closing(ArchiveStore.open_existing(tmp_path, read_only=False)) as store:
                with PreparedIndexMutation(store.index_db_path, archive_root=tmp_path) as seal:
                    prior = seal.observer_version("index")
                    with seal.mutation_scope(store._conn) as scope:
                        _update(store._conn)
                        scope.commit()
                        receipt = scope.commit_receipt
                        assert receipt._writer is store._conn and receipt._seal is seal
                    seal.require_index_commit_receipt(receipt)
                    assert seal.observer_version("index") != prior
                    assert seal._original_input_epochs["index"] == 1
                    with seal.original_read_snapshot():
                        seal.require_index_commit_receipt(receipt)
                        assert _title(seal.observer("index")) == "accepted-index"
                        with pytest.raises(ReferenceSealError):
                            seal.require_index_commit_receipt(replace(receipt, _writer=seal.observer("source")))
                    with pytest.raises(ReferenceSealError):
                        seal.accept_index_commit(receipt)
                    with pytest.raises(ReferenceSealError):
                        with seal.mutation_scope(store._conn):
                            pytest.fail("accepted seal published Index again")

    await run_archive_fixture_write(tmp_path, run)


async def test_index_rollback_has_no_receipt_or_observer_advance(tmp_path: Path) -> None:
    class InjectedRollbackError(Exception):
        pass

    injected = InjectedRollbackError()

    def run() -> None:
        with write_lease("test.index-rollback", archive_root=tmp_path):
            _seed(tmp_path)
            with closing(ArchiveStore.open_existing(tmp_path, read_only=False)) as store:
                with PreparedIndexMutation(store.index_db_path, archive_root=tmp_path) as seal:
                    original = _title(store._conn)
                    prior = seal.observer_version("index")
                    with pytest.raises(InjectedRollbackError) as caught:
                        with seal.mutation_scope(store._conn) as scope:
                            _update(store._conn)
                            assert _title(store._conn) == "accepted-index"
                            raise injected
                    assert caught.value is injected and not scope._committed
                    with pytest.raises(ReferenceSealError):
                        _ = scope.commit_receipt
                    assert seal._accepted_index_commit is None
                    assert seal.observer_version("index") == prior
                    assert _title(store._conn) == original
                    seal.validate_observers_current()

    await run_archive_fixture_write(tmp_path, run)


async def test_foreign_index_commit_in_acceptance_gap_refuses_original_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run() -> None:
        with write_lease("test.index-foreign-gap", archive_root=tmp_path):
            _seed(tmp_path)
            with closing(ArchiveStore.open_existing(tmp_path, read_only=False)) as store:
                with PreparedIndexMutation(store.index_db_path, archive_root=tmp_path) as seal:
                    prior = seal._versions["index"]
                    actual_commit = store._conn.commit
                    foreign = []

                    def commit_then_foreign() -> None:
                        actual_commit()
                        with closing(
                            open_isolated_write_connection(
                                store.index_db_path,
                                archive_root=tmp_path,
                                purpose="test.index-foreign-gap",
                            )
                        ) as other:
                            with connection_cursor(other, "UPDATE sessions SET title='foreign-index'") as rows:
                                assert rows.rowcount == 1
                            other.commit()
                            foreign.append(other)

                    with monkeypatch.context() as patch:
                        patch.setattr(store._conn, "commit", commit_then_foreign)
                        with pytest.raises(
                            ReferenceSealStaleError, match="another writer entered Index commit acceptance"
                        ):
                            with seal.mutation_scope(store._conn) as scope:
                                _update(store._conn)
                                scope.commit()
                    assert len(foreign) == 1 and scope._committed and not scope._index_accepted
                    assert seal._versions["index"] == prior and seal._accepted_index_commit is None
                    assert _title(store._conn) == "foreign-index"
                    with pytest.raises(ReferenceSealError):
                        _ = scope.commit_receipt

    await run_archive_fixture_write(tmp_path, run)


@pytest.mark.parametrize("binding", ["writer", "parent"])
async def test_index_receipt_refuses_wrong_binding_before_exact_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding: str,
) -> None:
    def run() -> None:
        with write_lease("test.index-receipt-binding", archive_root=tmp_path):
            _seed(tmp_path)
            with closing(ArchiveStore.open_existing(tmp_path, read_only=False)) as store:
                with PreparedIndexMutation(store.index_db_path, archive_root=tmp_path) as seal:
                    with PreparedIndexMutation(store.index_db_path, archive_root=tmp_path) as other_seal:
                        actual_accept = seal.accept_index_commit
                        refused = []

                        def check_binding(receipt: IndexCommitReceipt) -> None:
                            prior = seal._versions["index"]
                            if binding == "writer":
                                altered = replace(receipt, _writer=seal.observer("index"))
                                with pytest.raises(ReferenceSealError):
                                    actual_accept(altered)
                            else:
                                with pytest.raises(ReferenceSealError):
                                    other_seal.accept_index_commit(receipt)
                            assert seal._versions["index"] == prior and seal._accepted_index_commit is None
                            refused.append(binding)
                            actual_accept(receipt)

                        with monkeypatch.context() as patch:
                            patch.setattr(seal, "accept_index_commit", check_binding)
                            with seal.mutation_scope(store._conn) as scope:
                                _update(store._conn)
                                scope.commit()
                        assert refused == [binding] and scope.commit_receipt._seal is seal

    await run_archive_fixture_write(tmp_path, run)


@pytest.mark.parametrize("timing", ["before_commit", "after_commit"])
async def test_index_commit_cancellation_preserves_truthful_receipt_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timing: str,
) -> None:
    def run() -> None:
        with write_lease("test.index-cancellation", archive_root=tmp_path):
            _seed(tmp_path)
            with closing(ArchiveStore.open_existing(tmp_path, read_only=False)) as store:
                with PreparedIndexMutation(store.index_db_path, archive_root=tmp_path) as seal:
                    original = _title(store._conn)
                    cancellation = threading.Event()
                    token = compute_cancel.set(cancellation)
                    actual_commit = store._conn.commit

                    def commit_then_cancel() -> None:
                        actual_commit()
                        cancellation.set()

                    try:
                        with monkeypatch.context() as patch:
                            if timing == "after_commit":
                                patch.setattr(store._conn, "commit", commit_then_cancel)
                                with seal.mutation_scope(store._conn) as scope:
                                    _update(store._conn)
                                    scope.commit()
                                assert scope._committed and scope._index_accepted
                                assert seal._accepted_index_commit is scope._commit_receipt
                            else:
                                with pytest.raises(asyncio.CancelledError):
                                    with seal.mutation_scope(store._conn) as scope:
                                        _update(store._conn)
                                        cancellation.set()
                                        scope.commit()
                                assert not scope._committed and scope._commit_receipt is None
                    finally:
                        cancellation.clear()
                        compute_cancel.reset(token)
                    assert _title(store._conn) == ("accepted-index" if timing == "after_commit" else original)
                    seal.validate_observers_current()

    await run_archive_fixture_write(tmp_path, run)


async def test_index_failed_close_retains_original_writer_and_refuses_source_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run() -> None:
        with write_lease("test.index-failed-close", archive_root=tmp_path):
            _seed(tmp_path)
            with closing(ArchiveStore.open_existing(tmp_path, read_only=False)) as store:
                with PreparedIndexMutation(store.index_db_path, archive_root=tmp_path) as seal:
                    with seal.mutation_scope(store._conn) as scope:
                        _update(store._conn)
                        scope.commit()
                    receipt = scope.commit_receipt
                    owner = next(owner for owner in native_sql_children(store) if owner.connection is store._conn)
                    actual_close = store._conn.close
                    blocked = [True]

                    def close() -> None:
                        if blocked[0]:
                            raise OSError("synthetic exact Index writer close fault")
                        actual_close()

                    try:
                        with monkeypatch.context() as patch:
                            patch.setattr(store._conn, "close", close)
                            with pytest.raises(NativeConnectionSettlementError) as caught:
                                owner.close()
                            assert caught.value.owner is owner
                            assert owner.connection is store._conn and not owner._settled
                            assert seal in owner._lifetime_dependencies and scope in owner._lifetime_dependencies
                            with pytest.raises(ReferenceSealError):
                                seal.require_index_commit_receipt(receipt)
                            assert seal._scratch_directory is not None
                            witness = Path(seal._scratch_directory.name)
                            with pytest.raises(NativeConnectionSettlementError) as retained:
                                seal.close()
                            assert retained.value.owner is owner and witness.exists()
                            assert seal._mutation_custody is not None
                            blocked[0] = False
                            owner.close()
                            assert owner.connection is None and owner._settled
                    finally:
                        blocked[0] = False

    await run_archive_fixture_write(tmp_path, run)


@pytest.mark.parametrize("window", ["initial_reservation", "acceptance_reservation"])
async def test_index_reservation_checks_every_original_observer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    window: str,
) -> None:
    def run() -> None:
        with write_lease("test.index-sibling-gap", archive_root=tmp_path):
            _seed(tmp_path)
            with closing(ArchiveStore.open_existing(tmp_path, read_only=False)) as store:
                with PreparedIndexMutation(store.index_db_path, archive_root=tmp_path) as seal:
                    actual_cursor = connection_cursor
                    actual_commit = store._conn.commit
                    changed = []

                    def change_user() -> None:
                        with closing(
                            open_isolated_write_connection(
                                tmp_path / "user.db",
                                archive_root=tmp_path,
                                purpose="test.index-sibling-gap",
                            )
                        ) as user:
                            with connection_cursor(
                                user, "INSERT INTO user_settings VALUES ('synthetic','1',1,'user:local')"
                            ):
                                pass
                            user.commit()
                            changed.append(True)

                    @contextmanager
                    def reserve_after_foreign(
                        connection: sqlite3.Connection, sql: str, *args: Any, **kwargs: Any
                    ) -> Iterator[sqlite3.Cursor]:
                        if connection is store._conn and sql == "BEGIN IMMEDIATE" and not changed:
                            assert not connection.in_transaction
                            change_user()
                        with actual_cursor(connection, sql, *args, **kwargs) as cursor:
                            yield cursor

                    def commit_then_foreign() -> None:
                        actual_commit()
                        change_user()

                    with monkeypatch.context() as patch:
                        if window == "initial_reservation":
                            patch.setattr(reference_seal, "connection_cursor", reserve_after_foreign)
                        else:
                            patch.setattr(store._conn, "commit", commit_then_foreign)
                        with pytest.raises(ReferenceSealStaleError):
                            with seal.mutation_scope(store._conn) as scope:
                                _update(store._conn)
                                scope.commit()
                    assert changed == [True] and seal._accepted_index_commit is None
                    if window == "initial_reservation":
                        assert _title(store._conn) != "accepted-index"
                    else:
                        assert _title(store._conn) == "accepted-index"

    await run_archive_fixture_write(tmp_path, run)


async def test_original_window_receipt_does_not_adopt_foreign_source_commit(tmp_path: Path) -> None:
    def run() -> None:
        with write_lease("test.index-receipt-original-window", archive_root=tmp_path):
            _seed(tmp_path)
            with (
                closing(ArchiveStore.open_existing(tmp_path, read_only=False)) as store,
                closing(
                    open_isolated_write_connection(
                        tmp_path / "source.db", purpose="synthetic source arrival", archive_root=tmp_path
                    )
                ) as source,
            ):
                with PreparedIndexMutation(store.index_db_path, archive_root=tmp_path) as seal:
                    with seal.mutation_scope(store._conn) as scope:
                        _update(store._conn)
                        scope.commit()
                    receipt = scope.commit_receipt
                    source_version = seal._versions["source"]
                    source_epoch = seal._original_input_epochs["source"]
                    reached = []
                    with pytest.raises(ReferenceSealStaleError):
                        with seal.original_read_snapshot():
                            seal.require_index_commit_receipt(receipt)
                            publish_source_generation(
                                source,
                                source_generation_id="synthetic-late-generation",
                                manifest_digest="e" * 64,
                                addressing_mode="physical-file-v1",
                                coordinates=("neutral.json",),
                                observed_at_ms=1,
                            )
                            assert not source.in_transaction
                            reached.append(True)
                            # This guard preserves the original bracket; it
                            # cannot adopt the foreign commit into its proof.
                            seal.require_index_commit_receipt(receipt)
                    assert reached == [True]
                    assert seal._versions["source"] == source_version
                    assert seal._original_input_epochs["source"] == source_epoch

    await run_archive_fixture_write(tmp_path, run)
