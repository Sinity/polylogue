"""A writable connection is obtainable only from a held lease.

polylogue-8qm4k: the daemon claims to be the sole SQLite writer, but that was a
convention. The gate is advisory, holds were unbounded, and dispatch to the
coordinator fell open on a duck-typing miss, so a writer off the gate exhausted
the busy timeout while a holder ran for hours (a measured
``maintenance.drive_catchup`` hold of 18,623 s) and the live catch-up chunk died
with ``database is locked``.

These laws make the boundary structural: where enforcement is armed, opening a
write-mode connection without the lease raises instead of contending.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import select
import sqlite3
import threading
from builtins import BaseExceptionGroup
from collections.abc import Callable, Iterator
from contextlib import closing
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStoreSettlementError
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import (
    open_connection,
    open_daemon_connection,
    open_isolated_write_connection,
)
from polylogue.storage.sqlite.write_lease import (
    UnleasedWriteError,
    WriteLeaseThreadGrant,
    adopt_write_lease,
    arm_write_lease_enforcement,
    async_write_lease,
    bind_write_lease_thread,
    current_write_lease,
    delegate_write_lease,
    grant_write_lease_thread,
    require_write_lease,
    write_lease,
    write_lease_enforced,
)
from tests.infra.sqlite_cursor_settlement import (
    native_settlement_connections,  # noqa: F401  # Pytest fixture discovery.
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "tier.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.commit()
    return path


def test_an_unleased_write_open_is_refused_where_enforcement_is_armed(db_path: Path) -> None:
    """The anti-vacuity case named on the bead: a writer with no lease is a bug.

    Anti-vacuity: delete the ``require_write_lease`` call from
    ``open_connection`` and this goes green while an unserialized writer opens a
    write-mode connection exactly as before.
    """
    with arm_write_lease_enforcement():
        with pytest.raises(UnleasedWriteError):
            open_connection(db_path, validate_schema=False)
        with pytest.raises(UnleasedWriteError):
            open_daemon_connection(db_path, validate_schema=False)
        with pytest.raises(UnleasedWriteError):
            open_isolated_write_connection(db_path, purpose="probe")


def test_overlapping_process_wide_arming_remains_until_last_exit() -> None:
    """One overlapping invocation cannot disarm another active invocation.

    Anti-vacuity: restore a saved process-global boolean and exit the first
    context before the second; enforcement becomes false while still held.
    """
    was_enforced = write_lease_enforced()
    first = arm_write_lease_enforcement(process_wide=True)
    second = arm_write_lease_enforcement(process_wide=True)
    first.__enter__()
    second.__enter__()
    try:
        first.__exit__(None, None, None)
        assert write_lease_enforced()
    finally:
        second.__exit__(None, None, None)
    assert write_lease_enforced() is was_enforced


def test_a_leased_write_open_succeeds(db_path: Path) -> None:
    """The lease authorizes; it does not merely record."""
    with arm_write_lease_enforcement(), write_lease("test.writer", archive_root=db_path.parent):
        with closing(open_connection(db_path, validate_schema=False)) as conn:
            conn.execute("INSERT INTO t VALUES (1)")
            conn.commit()
    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute("SELECT count(*) FROM t").fetchone()[0] == 1


def test_archive_insight_writer_refuses_a_different_archive_root(tmp_path: Path) -> None:
    """A generation path cannot turn one archive's lease into another's writer.

    Anti-vacuity: omitting the explicit ``archive_root`` from the convergence
    writer lets this open succeed because the factory has no root to compare.
    """
    from polylogue.operations.session_profile_convergence import make_session_profile_derivation

    owner_root = tmp_path / "owner"
    target_root = tmp_path / "target"
    owner_root.mkdir()
    target_root.mkdir()
    target_db = target_root / "index.db"
    initialize_archive_database(target_db, ArchiveTier.INDEX)

    with (
        write_lease("test.owner", archive_root=owner_root),
        pytest.raises(UnleasedWriteError),
    ):
        make_session_profile_derivation(target_db, archive_root=target_root, now=lambda: 0.0)._write_connection()


def test_checkpoint_writer_refuses_a_different_archive_root(tmp_path: Path) -> None:
    """The periodic checkpoint route carries the root it was admitted for.

    Anti-vacuity: dropping ``archive_root=archive_root`` from
    ``checkpoint_archive_wals`` reopens the target database under this lease.
    """
    from polylogue.storage.sqlite.wal_checkpoint import checkpoint_archive_wals

    owner_root = tmp_path / "owner"
    target_root = tmp_path / "target"
    owner_root.mkdir()
    target_root.mkdir()
    initialize_archive_database(target_root / "index.db", ArchiveTier.INDEX)

    with (
        write_lease("test.owner", archive_root=owner_root),
        pytest.raises(UnleasedWriteError),
    ):
        checkpoint_archive_wals(target_root, reason="test", warn_bytes=0)


def test_index_generation_bootstrap_requires_the_archive_bound_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A generation's direct writable index open cannot bypass admission.

    Anti-vacuity: removing the lease assertion from ``IndexGenerationStore``
    lets this production bootstrap create an archive-tier ``index.db`` while
    the daemon's connection guard is armed but no writer owns the archive.
    """
    from polylogue.storage.index_generation import IndexGenerationStore

    root = tmp_path / "archive"
    root.mkdir()
    initialize_active_archive_root(root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    store = IndexGenerationStore.for_archive_root(root)

    with arm_write_lease_enforcement(), pytest.raises(UnleasedWriteError):
        store.create(source_snapshot="snapshot-unleased")

    with arm_write_lease_enforcement(), write_lease("test.generation", archive_root=root):
        generation = store.create(source_snapshot="snapshot-leased")
    assert Path(generation.index_path).is_file()

    with arm_write_lease_enforcement(), pytest.raises(UnleasedWriteError):
        store.promote(generation)


def test_index_generation_lifecycle_receipts_and_recovery_require_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Filesystem-only generation mutations cannot bypass archive admission.

    The connection guard cannot see JSON receipt writes or pointer-recovery
    metadata.  Anti-vacuity: removing the store-bound check lets these routes
    mutate an archive while the daemon's process-wide enforcement is armed.
    """
    from polylogue.storage.index_generation import IndexGenerationStore

    root = tmp_path / "archive"
    root.mkdir()
    initialize_active_archive_root(root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    store = IndexGenerationStore.for_archive_root(root)

    with arm_write_lease_enforcement(), write_lease("test.generation", archive_root=root):
        generation = store.create(source_snapshot="snapshot-leased")

    with arm_write_lease_enforcement(), pytest.raises(UnleasedWriteError):
        store.recover_promotion(generation.generation_id)
    with arm_write_lease_enforcement(), pytest.raises(UnleasedWriteError):
        store.complete_promotion_recovery(generation.generation_id)


def test_generation_checkpoint_binds_to_its_archive_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Descriptor-bound checkpoint construction admits its configured archive.

    Removing the root admission permits both unleased and wrong-root writers.
    """
    from polylogue.storage.index_generation import _checkpoint_truncate

    root = tmp_path / "archive"
    root.mkdir()
    initialize_active_archive_root(root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    index_db = root / "index.db"

    with arm_write_lease_enforcement(), pytest.raises(UnleasedWriteError):
        _checkpoint_truncate(index_db, label="active index", archive_root=root)

    other_root = tmp_path / "other"
    other_root.mkdir()
    with (
        arm_write_lease_enforcement(),
        write_lease("test.generation", archive_root=other_root),
        pytest.raises(UnleasedWriteError, match="outside the archive"),
    ):
        _checkpoint_truncate(index_db, label="active index", archive_root=root)

    # The opposite direction: a properly owned checkpoint still runs, so
    # "refuse every checkpoint" cannot pass as a fix.
    with arm_write_lease_enforcement(), write_lease("test.generation", archive_root=root):
        _checkpoint_truncate(index_db, label="active index", archive_root=root)
    assert index_db.is_file()


def test_cold_generation_open_binds_to_the_declared_archive_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The candidate path is not a substitute for its archive authority."""
    from polylogue.storage.index_generation import IndexGenerationStore
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = tmp_path / "archive"
    root.mkdir()
    initialize_active_archive_root(root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    store = IndexGenerationStore.for_archive_root(root)
    with write_lease("test.generation", archive_root=root):
        generation = store.create(source_snapshot="snapshot-leased")
        candidate = Path(generation.index_path).parent
        with ArchiveStore.open_cold_build_generation(
            candidate,
            generation_id=generation.generation_id,
            owner_id=generation.owner_id,
        ) as archive:
            assert archive.archive_root == Path(generation.index_path).parent

    wrong_root = tmp_path / "other"
    wrong_root.mkdir()
    with (
        arm_write_lease_enforcement(),
        write_lease("test.wrong-generation", archive_root=wrong_root),
        pytest.raises(UnleasedWriteError, match="outside the archive"),
    ):
        ArchiveStore.open_cold_build_generation(
            candidate,
            generation_id=generation.generation_id,
            owner_id=generation.owner_id,
        )


def test_writable_archive_store_releases_open_custody_and_gates_each_mutation(tmp_path: Path) -> None:
    """A persistent SQLite handle does not hold physical custody between writes."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = tmp_path / "archive"
    root.mkdir()
    with arm_write_lease_enforcement(), write_lease("test.archive.open", archive_root=root):
        initialize_active_archive_root(root)
        archive = ArchiveStore(root, initialize=False, read_only=False)
    assert current_write_lease() is None

    # A second archive owner can acquire custody while the first store's
    # persistent handles remain open; the active-store SH lease is a
    # separate lifecycle lock and is intentionally retained.
    with write_lease("test.archive.between-writes", archive_root=root):
        assert current_write_lease() is not None

    try:
        raw_id = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=b"{}",
            source_path="synthetic/session.jsonl",
            canonical_source_path="synthetic/session.jsonl",
            acquired_at_ms=1,
        )
        assert raw_id
        assert current_write_lease() is None
    finally:
        archive.close()


def test_archive_store_close_settles_sqlite_before_releasing_its_mutation_lease(tmp_path: Path) -> None:
    """An early handle-close failure cannot leave an index transaction live."""
    from polylogue.storage.io_phase_metrics import connect_measured
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError
    from tests.infra.sqlite_cursor_settlement import arm_settlement

    root = tmp_path / "archive"
    root.mkdir()
    with write_lease("test.archive.open", archive_root=root):
        initialize_active_archive_root(root)
        archive = ArchiveStore(root, initialize=False, read_only=False)
    archive._enter_mutation_lease()
    archive._conn.execute("BEGIN IMMEDIATE")
    archive._conn.execute("CREATE TABLE close_probe (value INTEGER)")
    vector = arm_settlement(connect_measured(":memory:"))
    vector.execute("BEGIN")
    archive.operation_vector_connection = vector
    try:
        with pytest.raises(ArchiveStoreSettlementError) as failure:
            archive.close()
        assert failure.value.store is archive
        assert isinstance(failure.value.failure, NativeConnectionSettlementError)
        assert isinstance(failure.value.failure.failure, BaseExceptionGroup)
        assert len(failure.value.failure.failure.exceptions) == 2
        assert all(isinstance(error, OSError) for error in failure.value.failure.failure.exceptions)
        assert current_write_lease() is not None
        vector.allow_cleanup.set()
        archive.close()
        assert current_write_lease() is None
        with write_lease("test.archive.after-close-failure", archive_root=root):
            with closing(connect_measured(root / "index.db")) as conn:
                assert (
                    conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='close_probe'").fetchone()
                    is None
                )
        archive.close()
    finally:
        vector.allow_cleanup.set()
        archive.close()


@pytest.mark.uses_real_clock("a competing physical writer waits for actual SQLite settlement")
def test_archive_store_retains_custody_when_sqlite_transaction_cannot_be_settled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed rollback and close keep the live writer behind its physical gate."""
    import threading

    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.sqlite_cursor_settlement import ControlledConnection, control_archive_connections

    root = tmp_path / "archive"
    root.mkdir()
    with write_lease("test.archive.open", archive_root=root):
        initialize_active_archive_root(root)
        control_archive_connections(monkeypatch, root / "index.db")
        archive = ArchiveStore(root, initialize=False, read_only=False)
    connection = archive._conn
    contender: threading.Thread | None = None
    try:
        assert isinstance(connection, ControlledConnection)
        archive._enter_mutation_lease()
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("CREATE TABLE unsettled_probe (value INTEGER)")
        connection.rollback_failure = OSError("synthetic rollback failure")
        connection.close_failure = OSError("synthetic close failure")

        with pytest.raises(ArchiveStoreSettlementError) as failure:
            archive.close()

        assert failure.value.store is archive
        assert connection.in_transaction
        assert archive._conn is connection
        assert current_write_lease() is not None
        acquired = threading.Event()
        finished = threading.Event()
        contender_failures: list[BaseException] = []

        def competing_writer() -> None:
            try:
                with write_lease("test.archive.waiting-writer", archive_root=root):
                    acquired.set()
            except BaseException as error:
                contender_failures.append(error)
            finally:
                finished.set()

        contender = threading.Thread(target=contextvars.Context().run, args=(competing_writer,))
        contender.start()
        assert not acquired.wait(0.05)
        assert not contender_failures

        connection.rollback_failure = None
        connection.close_failure = None
        archive.close()
        assert acquired.wait(2)
        contender.join(timeout=2)
        assert finished.is_set()
        assert not contender_failures
        assert current_write_lease() is None
        with sqlite3.connect(root / "index.db") as conn:
            assert (
                conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='unsettled_probe'").fetchone()
                is None
            )
    finally:
        if isinstance(connection, ControlledConnection):
            connection.rollback_failure = None
            connection.close_failure = None
        archive.close()
        if contender is not None:
            contender.join(timeout=2)


@pytest.mark.uses_real_clock("foreign-thread refusal precedes native SQLite cleanup")
def test_archive_store_wrong_thread_close_preserves_owner_recovery(tmp_path: Path) -> None:
    """A foreign thread cannot strand a thread-affine SQLite transaction."""
    import threading

    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = tmp_path / "archive"
    root.mkdir()
    with write_lease("test.archive.open", archive_root=root):
        initialize_active_archive_root(root)
        archive = ArchiveStore(root, initialize=False, read_only=False)
    archive._enter_mutation_lease()
    archive._conn.execute("BEGIN IMMEDIATE")
    archive._conn.execute("CREATE TABLE owner_thread_probe (value INTEGER)")
    failures: list[BaseException] = []

    def foreign_close() -> None:
        try:
            archive.close()
        except BaseException as exc:
            failures.append(exc)

    closer = threading.Thread(target=foreign_close)
    closer.start()
    closer.join(timeout=2)
    assert not closer.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    assert archive._conn.in_transaction
    assert current_write_lease() is not None

    archive.close()
    assert current_write_lease() is None
    with sqlite3.connect(root / "index.db") as conn:
        assert (
            conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='owner_thread_probe'").fetchone()
            is None
        )


def test_archive_insight_rebuild_uses_store_mutation_admission(tmp_path: Path) -> None:
    """The retained adapter cannot bypass daemon admission with Store._conn."""
    from polylogue.storage.derived.session.rebuild import rebuild_archive_session_insights
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    archive = ArchiveStore(root, initialize=False)
    observed_writes: list[tuple[str, bool]] = []

    def observe(sql: str) -> None:
        if sql.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE", "REPLACE")):
            observed_writes.append((sql, current_write_lease() is not None))

    try:
        archive._conn.set_trace_callback(observe)
        rebuild_archive_session_insights(archive)
        assert observed_writes
        assert all(admitted for _sql, admitted in observed_writes)
        assert current_write_lease() is None
        with arm_write_lease_enforcement(), pytest.raises(UnleasedWriteError):
            rebuild_archive_session_insights(archive)
    finally:
        archive._conn.set_trace_callback(None)
        archive.close()


@pytest.mark.uses_real_clock("raw connection cleanup runs on its actual aiosqlite worker")
def test_async_writer_grant_is_retained_until_worker_connection_closes(
    workspace_env: dict[str, Path],
) -> None:
    """The actual worker and grant stay owned through native close failure."""
    from polylogue.storage.sqlite import async_sqlite
    from polylogue.storage.sqlite.write_lease import async_write_lease
    from tests.infra.sqlite_cursor_settlement import SettlementConnection, arm_settlement

    async def scenario() -> None:
        root = workspace_env["archive_root"]
        backend = async_sqlite.SQLiteBackend(root / "index.db")
        async with async_write_lease("test.worker-close", archive_root=root):
            connection = await async_sqlite._open_configured_backend_connection(backend, read_only=True)
            entry = async_sqlite._BACKEND_CONNECTIONS[id(connection)]
            grant = entry.grant
            assert grant is not None
            raw = cast(
                SettlementConnection,
                await connection._execute(lambda: arm_settlement(connection._conn)),  # type: ignore[no-untyped-call]
            )
            try:
                with pytest.raises(BaseExceptionGroup) as refused:
                    await async_sqlite._close_backend_connection(connection, rollback=True)
                assert len(refused.value.exceptions) == 2
                assert all(isinstance(error, OSError) for error in refused.value.exceptions)
                assert id(connection) in async_sqlite._BACKEND_CONNECTIONS
                assert not grant.custody_retired
                assert connection._connection is raw and connection._running
                assert connection._thread.is_alive()
                raw.allow_cleanup.set()
                await async_sqlite._close_backend_connection(connection, rollback=True)
                assert id(connection) not in async_sqlite._BACKEND_CONNECTIONS
                assert grant.custody_retired
                assert connection._connection is None and not connection._thread.is_alive()
                assert all(thread is raw.owner for _operation, thread in raw.calls)
            finally:
                raw.allow_cleanup.set()
                await backend.close()

    asyncio.run(scenario())


def test_archive_close_can_settle_a_finished_task_only_on_its_original_thread(tmp_path: Path) -> None:
    """A completed task leaves cleanup custody, never mutation authority."""
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = tmp_path / "archive"
    initialize_active_archive_root(root)

    async def scenario() -> None:
        admitted = asyncio.Event()
        finish = asyncio.Event()
        stores: list[ArchiveStore] = []

        async def owner() -> None:
            async with async_write_lease("test.store.owner", archive_root=root):
                store = ArchiveStore(root, initialize=False)
                stores.append(store)
                store._enter_mutation_lease()
                store._conn.execute("BEGIN IMMEDIATE")
                store._conn.execute("CREATE TABLE finished_task_probe (value INTEGER)")
                admitted.set()
                await finish.wait()

        task = asyncio.create_task(owner())
        await admitted.wait()
        store = stores[0]
        try:
            with pytest.raises(RuntimeError):
                store.close()
            finish.set()
            await task
            with pytest.raises(RuntimeError):
                store.commit()
            store.close()
            assert store._owned_index_connection is None
            assert current_write_lease() is None
            async with async_write_lease("test.store.successor", archive_root=root):
                with sqlite3.connect(root / "index.db") as conn:
                    assert (
                        conn.execute("SELECT 1 FROM sqlite_master WHERE name='finished_task_probe'").fetchone() is None
                    )
        finally:
            finish.set()
            await task
            store.close()

    asyncio.run(scenario())


@pytest.mark.uses_real_clock("terminal cleanup drains the actual original SQLite worker")
def test_async_manual_cleanup_retires_a_finished_task_without_borrowing_live_authority(
    tmp_path: Path,
) -> None:
    """Only terminal cleanup may settle another task's retained real worker."""
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    backend = SQLiteBackend(db_path=root / "index.db")

    async def scenario() -> None:
        admitted = asyncio.Event()
        finish = asyncio.Event()

        async def owner() -> None:
            await backend.begin()
            admitted.set()
            await finish.wait()
            # Deliberately leave an unsettled manual transaction, as after a
            # cleanup failure. The backend retains its real connection/lease.

        task = asyncio.create_task(owner())
        await admitted.wait()
        connection = backend._txn_conn
        assert connection is not None
        try:
            with pytest.raises(UnleasedWriteError):
                await backend.close()
            with pytest.raises(UnleasedWriteError):
                async with backend.transaction():
                    pytest.fail("another task entered the live manual transaction")
            with pytest.raises(UnleasedWriteError):
                async with backend.bulk_connection():
                    pytest.fail("another task acquired custody ahead of owner validation")
            assert backend._txn_conn is connection
            assert connection._thread.is_alive()
            finish.set()
            await task
            await backend.close()
            assert backend._txn_conn is None
            assert backend._manual_lease_cm is None
            assert current_write_lease() is None
            connection._thread.join()
            assert not connection._thread.is_alive()
            async with async_write_lease("test.successor", archive_root=root):
                assert current_write_lease() is not None
        finally:
            finish.set()
            await task
            await backend.close()

    asyncio.run(scenario())


def test_cold_generation_discard_requires_the_archive_bound_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty-build teardown cannot remove a candidate outside writer admission.

    Anti-vacuity: before ``discard_if_inactive`` asserted its archive-root
    lease, this production ``ColdBuildGeneration.discard`` route succeeded
    while enforcement was armed and removed the candidate's writable
    ``index.db`` concurrently with any admitted writer.
    """
    from polylogue.sources.live.cold_build import ColdBuildGeneration
    from polylogue.sources.live.watcher import WatchSource

    root = tmp_path / "archive"
    root.mkdir()
    initialize_active_archive_root(root)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(root))
    with write_lease("test.generation", archive_root=root):
        generation = ColdBuildGeneration.begin(
            root,
            reason="test",
            observed=ColdBuildGeneration.observe_source_baseline((WatchSource("fixture", root / "absent-source"),)),
        )

    with arm_write_lease_enforcement(), pytest.raises(UnleasedWriteError):
        generation.discard()

    assert generation.generation_root.is_dir()
    with arm_write_lease_enforcement(), write_lease("test.generation", archive_root=root):
        assert generation.discard() is True
    assert generation.settled
    assert not generation.generation_root.exists()


def test_embedding_failure_resolution_refuses_an_unleased_writer(tmp_path: Path) -> None:
    """The CLI failure-resolution path cannot bypass archive-bound admission.

    Anti-vacuity: restoring the generic ``sqlite_connection`` open makes this
    mutation run despite the daemon's armed process-wide writer boundary.
    """
    from polylogue.storage.embeddings.materialization import resolve_embedding_failure_with_lifecycle

    embeddings_db = tmp_path / "embeddings.db"
    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)

    with arm_write_lease_enforcement(), pytest.raises(UnleasedWriteError):
        resolve_embedding_failure_with_lifecycle(
            embeddings_db,
            failure_id="failure-does-not-matter-before-admission",
            action="acknowledge",
        )


def test_enforcement_is_off_by_default_so_one_shot_writers_are_unaffected(db_path: Path) -> None:
    """A CLI or API process is its own single writer and has no gate to be outside of."""
    assert write_lease_enforced() is False
    with closing(open_connection(db_path, validate_schema=False)) as conn:
        conn.execute("INSERT INTO t VALUES (2)")
        conn.commit()


def test_the_lease_does_not_leak_past_its_block(db_path: Path) -> None:
    with arm_write_lease_enforcement():
        with write_lease("test.writer", archive_root=db_path.parent):
            assert current_write_lease() is not None
        assert current_write_lease() is None
        with pytest.raises(UnleasedWriteError):
            open_connection(db_path, validate_schema=False)


def test_enforcement_does_not_leak_past_its_block(db_path: Path) -> None:
    with arm_write_lease_enforcement():
        assert write_lease_enforced() is True
    assert write_lease_enforced() is False


def test_a_nested_acquisition_returns_the_outer_lease(db_path: Path) -> None:
    """A publish inside a batch must not reset the outer hold's budget.

    Anti-vacuity: make the inner acquisition a second lease with its own clock
    and a long outer hold stops being reported, which is the exact blindness the
    budget exists to remove.
    """
    with write_lease("outer", max_hold_seconds=10.0, archive_root=db_path.parent) as outer:
        with write_lease("inner", max_hold_seconds=0.001, archive_root=db_path.parent) as inner:
            assert inner is outer
            assert inner.actor == "outer"


def test_a_nested_acquisition_inherits_the_outer_archive_identity(tmp_path: Path) -> None:
    """Re-entry under an archive-bound lease names no new archive.

    A derivation publish takes ``write_lease(actor)`` without a root while the
    daemon's archive-bound lease is held. Anti-vacuity: check re-entry with
    ``require_write_lease`` and no archive root, as #5727 left it, and the
    first nested acquisition raises "omitted archive identity".
    """
    with arm_write_lease_enforcement(), write_lease("outer", archive_root=tmp_path) as outer:
        with write_lease("derivation.publish") as inner:
            assert inner is outer
        with write_lease("same-root", archive_root=tmp_path) as same:
            assert same is outer
        with pytest.raises(UnleasedWriteError, match="outside the archive bound"):
            with write_lease("other-root", archive_root=tmp_path / "other"):
                pass


def test_a_slow_valid_hold_is_measured_without_changing_its_outcome(tmp_path: Path) -> None:
    """A duration budget is telemetry and cannot turn a completed write into failure."""
    with write_lease("test.slow", max_hold_seconds=0.0, archive_root=tmp_path) as lease:
        assert lease.over_budget is True


def test_a_hold_within_its_budget_is_silent(tmp_path: Path) -> None:
    with write_lease("test.fast", max_hold_seconds=60.0, archive_root=tmp_path) as lease:
        assert lease.over_budget is False


def test_unused_thread_grant_cannot_bind_after_its_owner_releases(tmp_path: Path) -> None:
    with write_lease("test.owner", archive_root=tmp_path):
        grant = grant_write_lease_thread()

    observed: list[BaseException] = []

    def bind_late() -> None:
        try:
            bind_write_lease_thread(grant)
        except BaseException as exc:
            observed.append(exc)

    worker = threading.Thread(target=lambda: contextvars.Context().run(bind_late))
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert len(observed) == 1
    assert isinstance(observed[0], UnleasedWriteError)


@pytest.mark.uses_real_clock("coordinates a real forked process and a queued flock owner")
@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is unavailable on this platform")
def test_forked_child_cannot_reuse_or_unlock_parent_custody(tmp_path: Path) -> None:
    """A child drops inherited authority without unlocking its parent's flock."""
    root = tmp_path / "archive"
    root.mkdir()
    read_fd, write_fd = os.pipe()
    queued = threading.Event()
    acquired = threading.Event()

    def queued_writer() -> None:
        def run() -> None:
            queued.set()
            with write_lease("test.queued_writer", archive_root=root):
                acquired.set()

        contextvars.Context().run(run)

    worker = threading.Thread(target=queued_writer, name="queued-archive-writer")
    with write_lease("test.parent", archive_root=root):
        grant = grant_write_lease_thread()
        worker.start()
        assert queued.wait(timeout=2)
        child_pid = os.fork()
        if child_pid == 0:  # pragma: no cover - assertions are reported to parent by pipe
            os.close(read_fd)
            try:
                try:
                    bind_write_lease_thread(grant)
                except UnleasedWriteError:
                    os.write(write_fd, b"R")
                else:
                    os.write(write_fd, b"X")
                with write_lease("test.forked_child", archive_root=root):
                    os.write(write_fd, b"A")
            except BaseException:
                os.write(write_fd, b"E")
            finally:
                os._exit(0)
        os.close(write_fd)
        assert os.read(read_fd, 1) == b"R"
        assert select.select((read_fd,), (), (), 0.05)[0] == []
        assert not acquired.is_set()

    worker.join(timeout=5)
    assert not worker.is_alive()
    assert acquired.is_set()
    assert os.read(read_fd, 1) == b"A"
    _, status = os.waitpid(child_pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
    os.close(read_fd)


def test_require_write_lease_returns_the_lease_for_an_authorized_caller(tmp_path: Path) -> None:
    with write_lease("test.writer", archive_root=tmp_path) as lease:
        assert require_write_lease("probe", archive_root=tmp_path) is lease


def test_a_child_task_cannot_inherit_its_parents_write_lease(db_path: Path) -> None:
    """A copied context marker is not authority to start another writer.

    Anti-vacuity: omitting the task-owner check in ``require_write_lease`` lets
    ``create_task`` inherit the context variable and open an overlapping
    writer while the owning task still holds the coordinator lease.
    """

    async def scenario() -> None:
        async with async_write_lease("test.owner", archive_root=db_path.parent):
            with arm_write_lease_enforcement():

                async def child_writer() -> None:
                    with closing(open_connection(db_path, validate_schema=False)):
                        pass

                child = asyncio.create_task(child_writer())
                with pytest.raises(UnleasedWriteError, match="inherited by a child task"):
                    await child

                # The owning task retains its own authority after refusing the
                # inherited child context.
                with closing(open_connection(db_path, validate_schema=False)):
                    pass

    asyncio.run(scenario())


def test_two_child_tasks_cannot_borrow_offline_archive_custody(tmp_path: Path) -> None:
    """Ambient offline custody is not a way for child tasks to mint writers."""
    from polylogue.storage.sqlite.write_lease import ArchiveWriteCustody, archive_write_custody

    async def scenario(custody: ArchiveWriteCustody) -> None:
        async def child_writer() -> None:
            with write_lease(
                "test.inherited_offline_custody",
                archive_root=tmp_path,
                _custody=custody,
            ):
                pytest.fail("a child task borrowed its parent's physical archive custody")

        tasks = (asyncio.create_task(child_writer()), asyncio.create_task(child_writer()))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert len(results) == 2
        assert all(isinstance(result, UnleasedWriteError) for result in results)

    # The synchronous owner scope deliberately survives into asyncio.run so
    # both tasks see the same thread-local carrier. Its exact owner task is
    # still ``None``; neither new task is allowed to turn that carrier into
    # separate authorities over the same flock descriptor.
    with archive_write_custody(tmp_path) as custody:
        asyncio.run(scenario(custody))


@pytest.mark.uses_real_clock("settles an adopted writer while the rebuild exclusion remains held")
def test_rebuild_exclusion_waits_for_an_adopted_writer(tmp_path: Path) -> None:
    from polylogue.storage.index_generation import ActiveWriterLease, RebuildLease, RebuildLeaseUnavailableError

    root = tmp_path / "archive"
    root.mkdir()
    started = threading.Event()
    release = threading.Event()
    observed_rebuild_exclusion: list[bool] = []

    with RebuildLease(root) as rebuild:
        with rebuild.write_segment("test.rebuild-exclusion"):
            delegation = delegate_write_lease()

            def adopted_writer() -> None:
                with adopt_write_lease(delegation):
                    started.set()
                    assert release.wait(timeout=5)
                    active = ActiveWriterLease(root)
                    try:
                        active.acquire()
                    except RebuildLeaseUnavailableError:
                        observed_rebuild_exclusion.append(True)
                    else:
                        active.close()
                        observed_rebuild_exclusion.append(False)

            worker = threading.Thread(target=lambda: contextvars.Context().run(adopted_writer))
            worker.start()
            assert started.wait(timeout=5)
            # The declared writer segment drains its adopted writer before the
            # rebuild owner can release EX exclusion after this probe.
            release.set()

    worker.join(timeout=5)
    assert not worker.is_alive()
    assert observed_rebuild_exclusion == [True]


def test_require_write_lease_is_permissive_when_unarmed() -> None:
    assert require_write_lease("probe") is None


def test_a_readonly_open_never_needs_a_lease(db_path: Path) -> None:
    """The boundary is about writers; refusing readers would be overreach."""
    with arm_write_lease_enforcement():
        with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
            assert conn.execute("SELECT count(*) FROM t").fetchone()[0] >= 0


def test_every_write_mode_factory_in_storage_routes_through_the_lease() -> None:
    """The census the bead asks for: no second door into a writable tier.

    A new write-mode factory that forgets ``require_write_lease`` reintroduces
    exactly the defect this closes -- an in-process writer outside the gate,
    exhausting the busy timeout of whoever holds it. Enumerating the factories
    is what makes that a test failure rather than a review miss.

    Anti-vacuity: drop the ``require_write_lease`` call from any listed factory
    and this fails naming it.
    """
    import ast
    from pathlib import Path as _Path

    #: Every function in ``polylogue/storage`` that opens a write-mode SQLite
    #: connection to an archive tier. Read-only opens are deliberately absent.
    write_mode_factories = {
        "polylogue/storage/sqlite/connection_profile.py": {
            "open_connection",
            "open_daemon_connection",
            "open_isolated_write_connection",
        },
        "polylogue/storage/sqlite/connection.py": {"_get_cached_connection"},
        "polylogue/storage/sqlite/audit_leaf.py": {
            "open_verified_audit_connection",
            "open_verified_sqlite_write_connection",
        },
    }

    repo_root = _Path(__file__).resolve().parents[3]
    unguarded: list[str] = []
    for relative, functions in write_mode_factories.items():
        module = ast.parse((repo_root / relative).read_text(encoding="utf-8"))
        found = {
            node.name: node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name in functions
        }
        assert set(found) == functions, f"{relative}: {functions - set(found)} no longer exist; update the census"
        for name, node in found.items():
            calls = {
                call.func.id
                for call in ast.walk(node)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            }
            if "require_write_lease" not in calls:
                unguarded.append(f"{relative}:{name}")
    assert unguarded == [], f"write-mode factories that do not take the lease: {unguarded}"


def test_an_unleased_backend_admits_an_established_archive_read_only(tmp_path: Path) -> None:
    """A daemon-armed reader may construct a backend over an established archive.

    The live batch probes ``Polylogue.backend`` outside the writer lease. The
    backend must admit an established root read-only -- no filesystem
    mutation -- and still refuse an index whose derived identity this runtime
    cannot serve, and a root whose format marker is gone. Anti-vacuity: route
    that construction through ``initialize_active_archive_root`` again and it
    raises ``UnleasedWriteError`` ("active archive bootstrap requires the
    daemon write lease"); chmod the index on this path and the mode check
    fails; drop the validating index open and the stale identity is served.
    """
    import os
    import stat

    from polylogue.core.errors import SchemaSkew
    from polylogue.storage.sqlite.archive_tiers.archive_plan import archive_format_marker_path
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

    root = tmp_path / "archive"
    root.mkdir()
    index_db = root / "index.db"
    initialize_active_archive_root(root)
    os.chmod(index_db, 0o644)
    with arm_write_lease_enforcement():
        assert current_write_lease() is None
        backend = SQLiteBackend(db_path=index_db)
        assert backend.db_path == index_db
        assert stat.S_IMODE(index_db.stat().st_mode) == 0o644

        with closing(sqlite3.connect(index_db)) as conn:
            conn.execute("UPDATE schema_identity SET identity = 'stale' WHERE tier = 'index'")
            conn.commit()
        with pytest.raises(SchemaSkew):
            SQLiteBackend(db_path=index_db)

        archive_format_marker_path(root).unlink()
        with pytest.raises(RuntimeError, match="archive format marker is missing"):
            SQLiteBackend(db_path=index_db)


def test_the_cached_write_connection_is_refused_without_a_lease(tmp_path: Path) -> None:
    """The async runtime's thread-local writer is on the lease like any other."""
    from polylogue.storage.sqlite.connection import open_connection as cached_write_connection

    with arm_write_lease_enforcement():
        with pytest.raises(UnleasedWriteError):
            with cached_write_connection(tmp_path / "cached.db"):
                pass


def test_an_over_budget_hold_does_not_displace_its_own_failure(tmp_path: Path) -> None:
    """A failing over-budget hold reports its own error, not the budget breach.

    Determinism: ``max_hold_seconds=0.0`` puts the hold over budget on every
    run without a sleep, since ``held_seconds`` is strictly positive by the
    time release is reached. No timing window is raced.

    Anti-vacuity: raising a timing error from the release
    ``finally`` turns this red. That masking is not cosmetic:
    ``_publication_commit_known`` in ``daemon/convergence.py``
    recovers a partial-write fact by ``isinstance`` on the raised exception, so
    a displaced error makes a committed index replacement with an unlowered
    marker report as an ordinary failure carrying no committed fact.
    """

    class PartialWriteError(RuntimeError):
        def __init__(self) -> None:
            super().__init__("index committed, marker lowering failed")
            self.index_family_committed = True

    with pytest.raises(PartialWriteError) as caught:
        with write_lease("test.failing_slow_writer", max_hold_seconds=0.0, archive_root=tmp_path):
            raise PartialWriteError

    # The typed fact an ``except`` clause needs survives the over-budget release.
    assert caught.value.index_family_committed is True


def test_a_failing_over_budget_hold_still_releases_the_lease(tmp_path: Path) -> None:
    """The contextvar is restored on the failing path, not only the clean one.

    Deterministic for the same reason as above; a leaked lease would let the
    next unrelated caller in this context open a write connection unleased.
    """

    class BoomError(RuntimeError):
        pass

    with pytest.raises(BoomError):
        with write_lease("test.failing_slow_writer", max_hold_seconds=0.0, archive_root=tmp_path):
            raise BoomError

    assert current_write_lease() is None


def test_an_unbound_thread_that_inherits_the_lease_is_still_refused(tmp_path: Path) -> None:
    """A spawned thread must not write on a lease it merely inherited.

    This interpreter is a free-threading (no-GIL) CPython build, and on it a
    new ``threading.Thread`` starts from a *copy* of the creating thread's
    context rather than an empty one. So a thread spawned while the daemon
    holds the write lease observes that lease in ``_ACTIVE`` -- the
    ContextVar does not isolate it. What actually keeps the single-writer
    boundary is the bound-thread check: the inherited lease names the owner's
    thread id, and an unbound thread is refused.

    Determinism: no sleeps. The worker is joined before the assertion, so the
    result is observed after the thread has certainly finished, and the lease
    is held for the whole join. Nothing races.

    Anti-vacuity: deleting the ``bound_thread_ids`` check in
    ``require_write_lease`` turns this green-to-red -- the inherited lease
    would authorize an arbitrary thread to open a durable write connection
    while the daemon believes it is the sole writer. It is red today only by
    that check, not by contextvar isolation.
    """
    observed: dict[str, object] = {}

    def worker() -> None:
        observed["inherited_lease"] = current_write_lease()
        try:
            require_write_lease("worker durable write", archive_root=tmp_path)
        except UnleasedWriteError as exc:
            observed["outcome"] = f"refused: {exc}"
        else:
            observed["outcome"] = "allowed"

    with arm_write_lease_enforcement(), write_lease("daemon.writer", archive_root=tmp_path) as lease:
        thread = threading.Thread(target=worker, name="inheriting-worker")
        thread.start()
        thread.join()

    # The inheritance itself is real: the guard, not isolation, is the defense.
    assert observed["inherited_lease"] is lease
    assert str(observed["outcome"]).startswith("refused: ")
    assert "unauthorized thread" in str(observed["outcome"])
    # The worker must not have smuggled itself into the owner's bound set.
    assert lease.bound_thread_ids == {lease.owner_thread_id}


def test_delegation_authorizes_a_foreign_thread_and_loop_but_nothing_else(tmp_path: Path) -> None:
    """Ownership travels as a value, so a hand-off survives thread + loop changes.

    Anti-vacuity: drop the ``adopt_write_lease`` block from ``worker`` and the
    adopted probe raises ``UnleasedWriteError`` -- the ambient lease does not
    reach a worker thread running its own event loop, which is exactly the
    daemon HTTP write gate's shape (polylogue-h5l6i).
    """
    seen: list[str] = []

    with arm_write_lease_enforcement():
        with write_lease("owner", archive_root=tmp_path):
            delegation = delegate_write_lease()

            def worker() -> None:
                async def body() -> None:
                    with adopt_write_lease(delegation):
                        lease = require_write_lease("user.db write", archive_root=tmp_path)
                        seen.append("adopted" if lease is not None else "unleased")

                asyncio.run(body())

            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()

    assert seen == ["adopted"]


def test_a_thread_without_the_delegation_is_still_refused_inside_the_hold(tmp_path: Path) -> None:
    """The load-bearing negative: admission is not ambient authorization.

    Anti-vacuity: authorize the rogue thread ambiently -- call
    ``bind_write_lease_thread()`` inside ``rogue`` -- and the assertion below
    fails, because the lease's ContextVar *is* inherited by threads on this
    build. Only the explicit hand-off keeps it out.
    """
    refused: list[BaseException | None] = []

    with arm_write_lease_enforcement():
        with write_lease("owner", archive_root=tmp_path):
            delegate_write_lease()

            def rogue() -> None:
                try:
                    require_write_lease("user.db write", archive_root=tmp_path)
                except UnleasedWriteError as exc:
                    refused.append(exc)
                else:
                    refused.append(None)

            thread = threading.Thread(target=rogue)
            thread.start()
            thread.join()

    assert len(refused) == 1
    assert isinstance(refused[0], UnleasedWriteError)


def test_delegation_admits_one_writer_at_a_time(tmp_path: Path) -> None:
    """One admission cannot fan out into concurrent writers.

    Anti-vacuity: remove the ``_adopted_by`` guard in ``adopt_write_lease``
    and the second adoption succeeds while the first still holds it.
    """
    entered = threading.Event()
    release = threading.Event()

    with arm_write_lease_enforcement():
        with write_lease("owner", archive_root=tmp_path):
            delegation = delegate_write_lease()

            def first() -> None:
                with adopt_write_lease(delegation):
                    entered.set()
                    release.wait(timeout=5.0)

            thread = threading.Thread(target=first)
            thread.start()
            assert entered.wait(timeout=5.0)
            try:
                with pytest.raises(UnleasedWriteError, match="already executing"):
                    with adopt_write_lease(delegation):
                        pass
            finally:
                release.set()
                thread.join()


def test_delegation_is_revoked_when_its_lease_is_released(tmp_path: Path) -> None:
    """A stashed grant authorizes nothing once the admission is over.

    Anti-vacuity: delete the revoke loop in ``write_lease``'s finally block and
    this adoption succeeds outside any admission.
    """
    with arm_write_lease_enforcement():
        with write_lease("owner", archive_root=tmp_path):
            delegation = delegate_write_lease()
        assert not delegation.live
        with pytest.raises(UnleasedWriteError, match="revoked"):
            with adopt_write_lease(delegation):
                pass


def test_delegation_cannot_be_minted_without_holding_the_lease() -> None:
    """Delegation is a hand-off, never an escalation.

    Anti-vacuity: mint a ``WriteLeaseDelegation`` directly instead of routing
    through ``require_write_lease`` and an unleased caller gains authority.
    """
    with arm_write_lease_enforcement():
        with pytest.raises(UnleasedWriteError):
            delegate_write_lease()


def test_delegation_is_revoked_when_its_lease_fails(tmp_path: Path) -> None:
    """A failing hold releases the lease, so it must revoke the same grants.

    The delegation contract is that a stashed delegation authorizes nothing
    once its lease is gone. A hold that raises releases the lease exactly as a
    successful one does.

    Anti-vacuity: delete the revoke loop from the ``except BaseException``
    branch in ``write_lease`` and this adoption succeeds outside any
    admission, with ``delegation.live`` still True.
    """
    with arm_write_lease_enforcement():
        delegation = None
        with pytest.raises(RuntimeError, match="hold failed"):
            with write_lease("owner", archive_root=tmp_path):
                delegation = delegate_write_lease()
                raise RuntimeError("hold failed")
        assert delegation is not None
        assert not delegation.live
        with pytest.raises(UnleasedWriteError, match="revoked"):
            with adopt_write_lease(delegation):
                pass


def test_an_inheriting_thread_cannot_bind_itself_into_the_live_lease(tmp_path: Path) -> None:
    """polylogue-1oa7o residual 1: binding must be delegated, not self-served.

    ``bind_write_lease_thread()`` used to read the ambient lease and add the
    calling thread's id to it. On this free-threading build *every* thread
    spawned during a hold inherits that lease, so any of them could join the
    daemon's live authority by calling it. Binding now requires a single-use
    grant the owner minted before the thread existed.

    Anti-vacuity: restore the no-argument self-binding form and the worker
    below joins ``bound_thread_ids``, so both assertions go red. The
    inheritance itself is real (asserted first), so this is not vacuous.
    """
    from polylogue.storage.sqlite.write_lease import bind_write_lease_thread

    observed: dict[str, object] = {}

    def worker() -> None:
        observed["inherited_lease"] = current_write_lease()
        try:
            bind_write_lease_thread(None)  # type: ignore[arg-type]
        except (UnleasedWriteError, AttributeError) as exc:
            observed["outcome"] = f"refused: {type(exc).__name__}"
        else:
            observed["outcome"] = "bound"

    with arm_write_lease_enforcement(), write_lease("daemon.writer", archive_root=tmp_path) as lease:
        thread = threading.Thread(target=worker, name="self-binding-worker")
        thread.start()
        thread.join()

        assert observed["inherited_lease"] is lease
        assert str(observed["outcome"]).startswith("refused: ")
        assert lease.authorized_threads() == frozenset({lease.owner_thread_id})


def test_a_granted_thread_may_bind_and_write(tmp_path: Path) -> None:
    """The owner-minted grant is the admitted path the coordinator uses."""
    from polylogue.storage.sqlite.write_lease import bind_write_lease_thread, grant_write_lease_thread

    observed: dict[str, object] = {}

    with arm_write_lease_enforcement(), write_lease("daemon.writer", archive_root=tmp_path) as lease:
        grant = grant_write_lease_thread()

        def worker() -> None:
            try:
                bind_write_lease_thread(grant)
                observed["lease"] = require_write_lease("granted worker write", archive_root=tmp_path)
            finally:
                grant.complete()

        thread = threading.Thread(target=worker, name="granted-worker")
        thread.start()
        thread.join()

        assert observed["lease"] is lease
        assert threading.get_ident() in lease.authorized_threads()


def test_bound_thread_grant_keeps_an_already_open_writer_valid_until_settled(tmp_path: Path) -> None:
    """Owner retirement revokes future grants while an admitted worker settles."""
    from polylogue.storage.sqlite.write_lease import bind_write_lease_thread, grant_write_lease_thread

    ready = threading.Event()
    finish = threading.Event()
    observed: dict[str, object] = {}

    with arm_write_lease_enforcement(), write_lease("daemon.writer", archive_root=tmp_path) as lease:
        grant = grant_write_lease_thread()

        def worker() -> None:
            try:
                bind_write_lease_thread(grant)
                ready.set()
                assert finish.wait(timeout=5)
                observed["lease"] = require_write_lease("settling an admitted SQLite writer", archive_root=tmp_path)
            except BaseException as exc:
                observed["error"] = exc
            finally:
                grant.complete()

        thread = threading.Thread(target=worker, name="settling-granted-writer")
        thread.start()
        assert ready.wait(timeout=5)

    finish.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert observed.get("error") is None
    assert observed["lease"] is lease
    assert lease.custody is not None and not lease.custody.held


def test_unused_thread_grant_cannot_bind_after_owner_release(tmp_path: Path) -> None:
    """A grant cannot become a late writer after the physical owner exits."""
    from polylogue.storage.sqlite.write_lease import bind_write_lease_thread, grant_write_lease_thread

    with arm_write_lease_enforcement(), write_lease("daemon.writer", archive_root=tmp_path):
        grant = grant_write_lease_thread()

    outcome: list[str] = []

    def worker() -> None:
        try:
            bind_write_lease_thread(grant)
        except UnleasedWriteError:
            outcome.append("refused")
        else:
            outcome.append("bound")
        finally:
            grant.complete()

    thread = threading.Thread(target=worker, name="late-granted-writer")
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert outcome == ["refused"]


def test_repeated_cancellation_does_not_abandon_settling_sqlite_work() -> None:
    """The queued operation settles before its original cancellation escapes."""
    import asyncio

    from polylogue.storage.sqlite.async_sqlite import _await_settled

    async def scenario() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        settled = asyncio.Event()

        async def queued_work() -> None:
            started.set()
            await release.wait()
            settled.set()

        task = asyncio.create_task(_await_settled(queued_work()))
        await started.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("caller cancellation was not propagated")
        assert settled.is_set(), "queued SQLite work was abandoned before its operation settled"

    asyncio.run(scenario())


def test_archive_bound_lease_can_grant_a_thread_without_dropping_identity(tmp_path: Path) -> None:
    """Thread handoff preserves the archive identity already held by the owner.

    Anti-vacuity: omit ``archive_root`` in ``grant_write_lease_thread``'s
    admission check and this archive-bound lease is refused before minting.
    """
    from polylogue.storage.sqlite.write_lease import grant_write_lease_thread

    with arm_write_lease_enforcement(), write_lease("daemon.writer", archive_root=tmp_path) as lease:
        grant = grant_write_lease_thread()

    assert grant.lease is lease


def test_archive_bound_lease_can_delegate_without_dropping_identity(tmp_path: Path) -> None:
    """Delegation must carry the archive identity checked by the lease.

    Anti-vacuity: omitting ``archive_root`` from the delegation admission makes
    this raise ``UnleasedWriteError`` under process-wide lease enforcement.
    """
    from polylogue.storage.sqlite.write_lease import adopt_write_lease, delegate_write_lease

    with arm_write_lease_enforcement(process_wide=True), write_lease("archive-writer", archive_root=tmp_path) as lease:
        delegation = delegate_write_lease()
        with adopt_write_lease(delegation) as adopted:
            # Adoption binds a per-thread view of the minting lease, not the
            # lease object itself; the identity it carries is the contract.
            assert adopted.archive_root == lease.archive_root
            assert require_write_lease("adopted archive writer", archive_root=tmp_path) is adopted


def test_a_reused_thread_ident_does_not_inherit_a_retired_workers_authority(tmp_path: Path) -> None:
    """Authority belongs to the bound thread object, not its reusable ident.

    OS thread idents are recycled once a thread exits. A granted worker that
    finishes during a long hold leaves its ident in ``bound_thread_ids``; a
    later inheriting thread that receives the same ident used to pass
    ``require_write_lease`` without any grant (polylogue-1oa7o residual 3).

    Anti-vacuity: restore the ident-only membership check in
    ``require_write_lease`` and the impostor below is admitted.
    """
    from types import SimpleNamespace
    from unittest.mock import patch

    from polylogue.storage.sqlite import write_lease as lease_module
    from polylogue.storage.sqlite.write_lease import bind_write_lease_thread, grant_write_lease_thread

    observed: dict[str, object] = {}

    with arm_write_lease_enforcement(), write_lease("daemon.writer", archive_root=tmp_path) as lease:
        grant = grant_write_lease_thread()

        def granted() -> None:
            try:
                bind_write_lease_thread(grant)
                observed["retired_ident"] = threading.get_ident()
            finally:
                grant.complete()

        retired = threading.Thread(target=granted, name="retired-granted-worker")
        retired.start()
        retired.join()
        retired_ident = observed["retired_ident"]
        assert retired_ident in lease.authorized_threads()

        def impostor() -> None:
            threading_view = SimpleNamespace(**vars(threading))
            threading_view.get_ident = lambda: retired_ident
            with patch.object(lease_module, "threading", threading_view):
                try:
                    require_write_lease("write from a thread that reused a retired ident", archive_root=tmp_path)
                except UnleasedWriteError:
                    observed["outcome"] = "refused"
                else:
                    observed["outcome"] = "admitted"

        thread = threading.Thread(target=impostor, name="ident-reuse-impostor")
        thread.start()
        thread.join()

    assert observed["outcome"] == "refused"


def test_a_grant_authorizes_exactly_one_thread(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.write_lease import bind_write_lease_thread, grant_write_lease_thread

    outcomes: list[str] = []

    with arm_write_lease_enforcement(), write_lease("daemon.writer", archive_root=tmp_path):
        grant = grant_write_lease_thread()

        def worker() -> None:
            try:
                bind_write_lease_thread(grant)
            except UnleasedWriteError:
                outcomes.append("refused")
            else:
                outcomes.append("bound")
            finally:
                grant.complete()

        first = threading.Thread(target=worker)
        first.start()
        first.join()
        second = threading.Thread(target=worker)
        second.start()
        second.join()

    assert outcomes == ["bound", "refused"]


@pytest.mark.parametrize("thread_context", ["inherited", "empty"])
def test_two_granted_threads_bind_concurrently_in_either_context_mode(tmp_path: Path, thread_context: str) -> None:
    """Two owner grants bind two threads at once, whatever the thread context.

    ``thread_context`` emulates both interpreter modes on any build:
    ``inherited`` starts each worker with a copy of the owner's context (the
    free-threading default), ``empty`` with a fresh one (the GIL-build
    default). Both workers bind behind a barrier, so the two
    ``authorize_thread`` calls race on ``bound_thread_ids``.

    Anti-vacuity: make ``bind_write_lease_thread`` install the lease only when
    the context already carries it and the ``empty`` case fails its write;
    drop ``_bind_guard`` and a lost update can leave one worker unbound.
    """
    from polylogue.storage.sqlite.write_lease import bind_write_lease_thread, grant_write_lease_thread

    barrier = threading.Barrier(2)
    observed: dict[str, object] = {}
    errors: list[BaseException] = []

    with arm_write_lease_enforcement(), write_lease("daemon.writer", archive_root=tmp_path) as lease:
        grants = [grant_write_lease_thread(), grant_write_lease_thread()]

        def worker(name: str, grant: WriteLeaseThreadGrant) -> None:
            try:
                observed[f"{name}:ambient"] = current_write_lease()
                barrier.wait()
                bind_write_lease_thread(grant)
                observed[name] = require_write_lease(f"{name} write", archive_root=tmp_path)
                observed[f"{name}:ident"] = threading.get_ident()
                barrier.wait()
            except BaseException as exc:
                errors.append(exc)
                barrier.abort()
            finally:
                grant.complete()

        threads = [
            threading.Thread(
                target=worker,
                args=(f"worker-{index}", grant),
                context=contextvars.copy_context() if thread_context == "inherited" else contextvars.Context(),
            )
            for index, grant in enumerate(grants)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        expected_ambient = lease if thread_context == "inherited" else None
        assert observed["worker-0:ambient"] is expected_ambient
        assert observed["worker-1:ambient"] is expected_ambient
        assert observed["worker-0"] is lease
        assert observed["worker-1"] is lease
        assert {observed["worker-0:ident"], observed["worker-1:ident"], lease.owner_thread_id} <= set(
            lease.authorized_threads()
        )


def test_a_nested_lease_in_an_inheriting_thread_is_refused(tmp_path: Path) -> None:
    """polylogue-1oa7o residual 2: the re-entrant branch had no thread check.

    An inheriting thread asking for ``write_lease(...)`` took the re-entrant
    path and was handed the parent's lease, creating neither its own authority
    nor a refusal.

    Anti-vacuity: drop the ``require_write_lease`` call from the re-entrant
    branch of ``write_lease`` and the worker below reports "granted".
    """
    observed: dict[str, object] = {}

    def worker() -> None:
        try:
            with write_lease("nested.worker", archive_root=tmp_path):
                observed["outcome"] = "granted"
        except UnleasedWriteError as exc:
            observed["outcome"] = f"refused: {exc}"

    with arm_write_lease_enforcement(), write_lease("daemon.writer", archive_root=tmp_path) as lease:
        assert lease is not None
        thread = threading.Thread(target=worker, name="nested-inheriting-worker")
        thread.start()
        thread.join()

    assert str(observed["outcome"]).startswith("refused: ")
    assert "unauthorized thread" in str(observed["outcome"])


def test_persistent_store_refuses_replaced_archive_directory_before_sql(tmp_path: Path) -> None:
    """A new directory's custody cannot authorize old SQLite handles."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    archive = ArchiveStore(root, initialize=False)
    parked = tmp_path / "parked"
    root.rename(parked)
    root.mkdir()
    try:
        with pytest.raises(UnleasedWriteError):
            archive.write_raw_payload(
                provider=Provider.CLAUDE_CODE,
                payload=b"{}",
                source_path="synthetic/directory-binding.jsonl",
                canonical_source_path="synthetic/directory-binding.jsonl",
                acquired_at_ms=1,
            )
        assert current_write_lease() is None
        from polylogue.storage.sqlite.write_lease import ARCHIVE_WRITE_CUSTODY_LOCK_NAME

        assert tuple(root.iterdir()) == (root / ARCHIVE_WRITE_CUSTODY_LOCK_NAME,)
    finally:
        replacement = tmp_path / "replacement"
        root.rename(replacement)
        parked.rename(root)
        archive.close()


def test_direct_blackboard_writer_acquires_custody_and_retires_temporary_user_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite.archive_tiers import archive as archive_module

    initialize_active_archive_root(tmp_path)
    store = archive_module.ArchiveStore(tmp_path, initialize=False)
    real_open = cast(Callable[..., sqlite3.Connection], vars(archive_module)["open_connection"])
    admissions: list[bool] = []

    def observe_open(path: Path, *args: object, **kwargs: object) -> sqlite3.Connection:
        if path.name == "user.db":
            lease = current_write_lease()
            admissions.append(lease is not None and lease.custody is not None and lease.custody.held)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(archive_module, "open_connection", observe_open)
    try:
        note = store.post_blackboard_note("synthetic note")
        assert note.body == "synthetic note"
        assert admissions == [True]
        assert not store._user_write_connections
        assert store._sql_custody is None
        assert current_write_lease() is None
    finally:
        store.close()


def test_store_successful_sql_settlement_retires_context_before_late_custody_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    initialize_active_archive_root(tmp_path)
    store = ArchiveStore(tmp_path, initialize=False)
    original_close = os.close
    store._enter_mutation_lease()
    store._conn.execute("BEGIN IMMEDIATE")
    store._conn.execute("CREATE TABLE terminal_scope_probe (value INTEGER)")
    assert store._sql_custody is not None
    selected_fd = store._sql_custody._fd
    injected = False

    def late_close(fd: int) -> None:
        nonlocal injected
        original_close(fd)
        if fd == selected_fd and not injected:
            injected = True
            raise OSError("synthetic late close error")

    monkeypatch.setattr(os, "close", late_close)
    try:
        with pytest.raises(OSError):
            store.commit()
        assert injected
        assert store._pending_archive_mutation_lease_context is None
        assert store._sql_custody is None
        assert current_write_lease() is None
        store.commit()
    finally:
        store.close()


def test_late_custody_close_error_never_closes_a_reused_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A close error is diagnostic; the kernel may already have retired its FD."""
    original_close = os.close
    replacement_fd: list[int] = []
    selected_fd = -1
    attempts: list[int] = []

    def close_then_report_error(fd: int) -> None:
        original_close(fd)
        if fd == selected_fd:
            attempts.append(fd)
            replacement = os.open(tmp_path / "replacement-file", os.O_CREAT | os.O_RDWR, 0o600)
            assert replacement == selected_fd
            replacement_fd.append(replacement)
            raise OSError("synthetic late close error")

    context = write_lease("test.custody.close-once", archive_root=tmp_path)
    lease = context.__enter__()
    custody = lease.custody
    assert custody is not None
    selected_fd = custody._fd
    monkeypatch.setattr(os, "close", close_then_report_error)
    try:
        with pytest.raises(OSError):
            context.__exit__(None, None, None)
        assert current_write_lease() is None
        assert not custody.held
        custody.close_owner()
        with write_lease("test.custody.after-late-close-error", archive_root=tmp_path):
            assert current_write_lease() is not None
        assert attempts == [selected_fd]
        assert os.write(replacement_fd[0], b"still owned by its new opener") > 0
    finally:
        for fd in replacement_fd:
            original_close(fd)


def test_cached_handle_reuses_only_within_one_physical_operation(tmp_path: Path) -> None:
    from polylogue.storage.sqlite import connection as cached
    from tests.infra.archive_custody_probe import archive_custody_available

    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    connections: list[sqlite3.Connection] = []
    try:
        with arm_write_lease_enforcement():
            for actor in ("test.cached_first", "test.cached_second"):
                with write_lease(actor, archive_root=root):
                    with cached.connection_context(root / "index.db") as connection:
                        connections.append(connection)
                        connection.execute("BEGIN IMMEDIATE")
                    connection.commit()
                    with cached.connection_context(root / "index.db") as reused:
                        assert reused is connection
                        assert reused.execute("SELECT 1").fetchone()[0] == 1
                    assert not archive_custody_available(root)
                assert archive_custody_available(root)
                with pytest.raises(sqlite3.ProgrammingError):
                    connection.execute("SELECT 1")
            assert connections[0] is not connections[1]
            with pytest.raises(UnleasedWriteError):
                with cached.connection_context(root / "index.db"):
                    pytest.fail("cached reuse bypassed current write authority")
    finally:
        cached._clear_connection_cache()


@pytest.mark.uses_real_clock("retirement/adoption ordering exercises actual physical custody")
def test_delegation_cannot_adopt_after_source_retirement_before_revoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite.write_lease import WriteLeaseDelegation
    from tests.infra.archive_custody_probe import archive_custody_available

    minted = threading.Event()
    finish_body = threading.Event()
    retirement_paused = threading.Event()
    resume_retirement = threading.Event()
    delegations: list[WriteLeaseDelegation] = []
    failures: list[BaseException] = []
    real_revoke = WriteLeaseDelegation.revoke

    def paused_revoke(delegation: WriteLeaseDelegation) -> None:
        retirement_paused.set()
        resume_retirement.wait()
        real_revoke(delegation)

    monkeypatch.setattr(WriteLeaseDelegation, "revoke", paused_revoke)

    def owner() -> None:
        try:
            with write_lease("test.retirement_owner", archive_root=tmp_path):
                delegations.append(delegate_write_lease())
                minted.set()
                finish_body.wait()
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=owner)
    thread.start()
    ran = False
    try:
        assert minted.wait(timeout=30)
        delegation = delegations[0]
        finish_body.set()
        assert retirement_paused.wait(timeout=30)
        assert not delegation.lease.active
        assert not delegation.live
        with pytest.raises(UnleasedWriteError):
            with adopt_write_lease(delegation):
                ran = True
        assert not ran
        assert delegation.settled
        assert not delegation.adopted
        assert not archive_custody_available(tmp_path)
    finally:
        finish_body.set()
        resume_retirement.set()
        thread.join(timeout=30)
    assert not thread.is_alive()
    assert failures == []
    assert archive_custody_available(tmp_path)


def test_adoption_cleanup_retires_every_grant_and_preserves_body_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite.write_lease import WriteLeaseThreadGrant
    from tests.infra.archive_custody_probe import archive_custody_available

    grants: list[WriteLeaseThreadGrant] = []
    real_revoke = WriteLeaseThreadGrant.revoke

    def fail_after_first_retirement(grant: WriteLeaseThreadGrant) -> None:
        real_revoke(grant)
        if grant is grants[0]:
            raise OSError("synthetic late grant-release failure")

    with write_lease("test.adoption_cleanup", archive_root=tmp_path):
        delegation = delegate_write_lease()
        monkeypatch.setattr(WriteLeaseThreadGrant, "revoke", fail_after_first_retirement)
        primary = ValueError("synthetic body failure")
        with pytest.raises(BaseExceptionGroup) as caught:
            with adopt_write_lease(delegation):
                grants.extend([grant_write_lease_thread(), grant_write_lease_thread()])
                raise primary
        assert len(grants) == 2
        assert all(not grant._custody_held for grant in grants)
        assert all(grant._revoked for grant in grants)
        assert delegation.settled
        assert not delegation.adopted
        assert caught.value.exceptions[0] is primary
        assert isinstance(caught.value.exceptions[1], OSError)
        assert not archive_custody_available(tmp_path)
    assert archive_custody_available(tmp_path)


def test_native_anchor_cleanup_attempts_all_and_never_recloses_reused_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner

    actual_close = os.close
    first = os.open(tmp_path / "first-anchor", os.O_CREAT | os.O_RDWR, 0o600)
    second = os.open(tmp_path / "second-anchor", os.O_CREAT | os.O_RDWR, 0o600)
    replacement: list[int] = []
    attempts: list[int] = []
    from polylogue.storage.io_phase_metrics import connect_measured

    connection = connect_measured(tmp_path / "anchor-probe.db")

    def close(descriptor: int) -> None:
        actual_close(descriptor)
        if descriptor in (first, second):
            attempts.append(descriptor)
            with pytest.raises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")
        if descriptor == first and not replacement:
            replacement.append(os.open(tmp_path / "replacement-anchor", os.O_CREAT | os.O_RDWR, 0o600))
            assert replacement[0] == first
            raise OSError("synthetic terminal descriptor close error")

    with write_lease("test.anchor_cleanup", archive_root=tmp_path):
        owner = NativeSQLCustodyOwner(connection, anchored_descriptors=(first, second))
        monkeypatch.setattr(os, "close", close)
        try:
            with pytest.raises(OSError):
                owner.close()
            assert owner.connection is None
            assert owner.anchored_descriptors == ()
            assert attempts == [first, second]
            owner.close()
            assert os.write(replacement[0], b"still owned by replacement") > 0
        finally:
            for descriptor in replacement:
                actual_close(descriptor)


def test_initialized_tier_further_schema_sql_retains_failed_actual_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite import connection_profile as profiles
    from polylogue.storage.sqlite.archive_tiers import bootstrap
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
    from tests.infra.archive_custody_probe import archive_custody_available
    from tests.infra.sqlite_cursor_settlement import SettlementConnection, arm_settlement

    initialize_active_archive_root(tmp_path)
    actual_open = profiles.open_daemon_connection
    handles: list[SettlementConnection] = []

    def open_connection(*args: object, **kwargs: object) -> sqlite3.Connection:
        handle = arm_settlement(actual_open(*args, **kwargs))  # type: ignore[arg-type]
        handles.append(handle)
        return handle

    def materialize(connection: sqlite3.Connection, tier: ArchiveTier) -> None:
        assert tier is ArchiveTier.OPS
        connection.execute("BEGIN IMMEDIATE")
        raise ValueError("synthetic higher factory schema failure")

    monkeypatch.setattr(profiles, "open_daemon_connection", open_connection)
    monkeypatch.setattr(bootstrap, "converge_same_version_tier", materialize)
    owner = None
    try:
        with write_lease("test.initialized_tier_cleanup", archive_root=tmp_path):
            with pytest.raises(profiles.NativeConnectionSettlementError) as refused:
                bootstrap.open_initialized_tier_connection(tmp_path / "ops.db", ArchiveTier.OPS, archive_root=tmp_path)
            owner = refused.value.owner
            assert cast(object, owner.connection) is handles[0]
            assert handles[0].in_transaction
        assert not archive_custody_available(tmp_path)
        handles[0].allow_cleanup.set()
        owner.close()
        assert archive_custody_available(tmp_path)
    finally:
        for handle in handles:
            handle.allow_cleanup.set()
        if owner is not None:
            owner.close()


@pytest.mark.asyncio
async def test_coordinator_lease_observation_rejects_inherited_child_task(tmp_path: Path) -> None:
    """Only the actual admitted execution unit observes coordinator authority."""
    from polylogue.core.write_lease import coordinator_write_lease_active
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator

    assert not coordinator_write_lease_active()

    def offline() -> bool:
        with write_lease("test.offline", archive_root=tmp_path):
            return coordinator_write_lease_active()

    assert not await asyncio.to_thread(offline)
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)

    async def admitted() -> None:
        assert coordinator_write_lease_active()

        async def child() -> bool:
            return coordinator_write_lease_active()

        assert not await asyncio.create_task(child())
        assert coordinator_write_lease_active()

    try:
        await coordinator.run("test.coordinator.observation", admitted)
    finally:
        await coordinator.shutdown(timeout=1.0)
    assert not coordinator_write_lease_active()


@pytest.mark.parametrize("failed_binding", ["lock", "directory", "both"])
def test_custody_ambiguous_close_retains_exact_binding_without_numeric_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_binding: str
) -> None:
    from polylogue.core.sql_settlement import retained_native_sql_owners
    from polylogue.storage.sqlite import connection_profile  # noqa: F401 - registers the existing census
    from polylogue.storage.sqlite.write_lease import ArchiveCustodySettlementError
    from tests.infra.archive_custody_probe import archive_custody_available

    real_close = os.close
    attempts: list[int] = []
    primary = ValueError("synthetic body failure")
    with pytest.raises(BaseExceptionGroup) as caught:
        with write_lease("test.ambiguous_custody", archive_root=tmp_path) as lease:
            custody = lease.custody
            assert custody is not None
            lock, directory = custody._fd, custody._directory_fd
            failing = (
                {lock}
                if failed_binding == "lock"
                else {directory}
                if failed_binding == "directory"
                else {lock, directory}
            )

            def fail_before_close(descriptor: int) -> None:
                if descriptor in (lock, directory):
                    attempts.append(descriptor)
                if descriptor in failing:
                    raise OSError("synthetic close before effect")
                real_close(descriptor)

            monkeypatch.setattr(os, "close", fail_before_close)
            raise primary
    try:
        assert caught.value.exceptions[0] is primary
        assert attempts == [lock, directory]
        assert custody in retained_native_sql_owners()
        assert custody.held == (lock in failing)
        if lock in failing:
            assert not archive_custody_available(tmp_path)
        before_retry = list(attempts)
        with pytest.raises((ArchiveCustodySettlementError, BaseExceptionGroup)):
            custody.close()
        assert attempts == before_retry
        with pytest.raises(UnleasedWriteError):
            with write_lease("test.must_not_restart", archive_root=tmp_path):
                pytest.fail("unresolved custody admitted new work")
    finally:
        # Controlled failures occurred before effect. Settle those actual
        # retained bindings even if a behavioral assertion failed.
        monkeypatch.setattr(os, "close", real_close)
        assert custody is not None
        for descriptor in tuple(custody._pending_descriptor_closes):
            real_close(descriptor)
        custody.close()
    assert custody not in retained_native_sql_owners()
    assert archive_custody_available(tmp_path)


def test_custody_acquisition_failure_retains_partial_directory_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite import write_lease as lease_module
    from polylogue.storage.sqlite.write_lease import ArchiveCustodySettlementError

    real_open, real_close = os.open, os.close
    selected: list[int] = []
    primary = OSError("synthetic lock open failure")

    def open_descriptor(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if path == lease_module.ARCHIVE_WRITE_CUSTODY_LOCK_NAME:
            raise primary
        descriptor = real_open(path, flags, *args, **kwargs)
        selected.append(descriptor)
        return descriptor

    def fail_close(descriptor: int) -> None:
        if descriptor in selected:
            raise OSError("synthetic directory close before effect")
        real_close(descriptor)

    monkeypatch.setattr(os, "open", open_descriptor)
    monkeypatch.setattr(os, "close", fail_close)
    try:
        with pytest.raises(BaseExceptionGroup) as caught:
            lease_module._acquire_archive_write_custody(tmp_path)
        assert caught.value.exceptions[0] is primary
        cleanup = caught.value.exceptions[1]
        assert isinstance(cleanup, ArchiveCustodySettlementError)
        owner = cleanup.owner
        assert owner._fd == -1
        assert owner._directory_fd == selected[0]
    finally:
        monkeypatch.setattr(os, "close", real_close)
        with lease_module._CUSTODY_REGISTRY_LOCK:
            retained = tuple(custody for custody in lease_module._CUSTODIES if custody.archive_root == tmp_path)
        for custody in retained:
            for descriptor in tuple(custody._pending_descriptor_closes):
                real_close(descriptor)
            custody.close()
    assert owner._directory_fd == -1


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("constructor_fault", [False, True])
async def test_failed_async_acquisition_retains_original_worker_until_exact_retirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancelled: bool, constructor_fault: bool
) -> None:
    from polylogue.storage.sqlite import write_lease as leases

    real_open, real_close, real_fstat = os.open, os.close, os.fstat
    selected: list[int] = []
    primary = OSError("synthetic async custody construction failure")
    construction_failed = False
    cleanup_attempted = threading.Event()
    attempts: list[threading.Thread] = []

    def open_descriptor(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if not constructor_fault and path == leases.ARCHIVE_WRITE_CUSTODY_LOCK_NAME:
            raise primary
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == tmp_path:
            selected.append(descriptor)
        return descriptor

    def close_before_effect(descriptor: int) -> None:
        if descriptor in selected:
            attempts.append(threading.current_thread())
            cleanup_attempted.set()
            raise OSError("synthetic async directory close before effect")
        real_close(descriptor)

    def fail_initial_fstat(descriptor: int) -> os.stat_result:
        nonlocal construction_failed
        if constructor_fault and descriptor in selected and not construction_failed:
            construction_failed = True
            raise primary
        return real_fstat(descriptor)

    monkeypatch.setattr(os, "open", open_descriptor)
    monkeypatch.setattr(os, "close", close_before_effect)
    monkeypatch.setattr(os, "fstat", fail_initial_fstat)

    async def acquire() -> None:
        async with async_write_lease("test.async_failed_acquisition", archive_root=tmp_path):
            pytest.fail("failed acquisition admitted a writer")

    task = asyncio.create_task(acquire())
    owner = None
    try:
        assert await asyncio.to_thread(cleanup_attempted.wait, 30)
        with leases._CUSTODY_REGISTRY_LOCK:
            owner = next(custody for custody in leases._CUSTODIES if custody.archive_root == tmp_path)
        original_worker = attempts[0]
        assert owner._descriptor_cleanup_thread is original_worker
        assert not task.done()
        if cancelled:
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
    finally:
        monkeypatch.setattr(os, "close", real_close)
        with leases._CUSTODY_REGISTRY_LOCK:
            retained = tuple(custody for custody in leases._CUSTODIES if custody.archive_root == tmp_path)
        for custody in retained:
            for descriptor in tuple(custody._pending_descriptor_closes):
                real_close(descriptor)
            custody.request_sql_settlement()
        outcome = await asyncio.gather(task, return_exceptions=True)
    assert owner is not None
    assert isinstance(outcome[0], BaseExceptionGroup)
    assert original_worker is owner._descriptor_cleanup_thread or owner._descriptor_cleanup_thread is None
    assert attempts == [original_worker]
    assert owner._directory_fd == -1
    assert owner not in leases._CUSTODIES
    assert primary in tuple(_exception_graph(outcome[0]))


def _exception_graph(error: BaseException) -> Iterator[BaseException]:
    yield error
    if isinstance(error, BaseExceptionGroup):
        for child in error.exceptions:
            yield from _exception_graph(child)


@pytest.mark.asyncio
async def test_transferred_async_custody_cleanup_keeps_original_loop_task_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.infra.archive_custody_probe import archive_custody_available

    actual_close = os.close
    cleanup_started = asyncio.Event()
    owners = []
    attempts: list[threading.Thread] = []
    primary = ValueError("synthetic async body failure")

    async def operation() -> None:
        async with async_write_lease("test.loop_cleanup", archive_root=tmp_path) as lease:
            owner = lease.custody
            assert owner is not None
            owners.append(owner)
            descriptor = owner._fd

            def fail_close(value: int) -> None:
                if value == descriptor:
                    attempts.append(threading.current_thread())
                    cleanup_started.set()
                    raise OSError("synthetic loop-owned close before effect")
                actual_close(value)

            monkeypatch.setattr(os, "close", fail_close)
            raise primary

    task = asyncio.create_task(operation())
    try:
        await asyncio.wait_for(cleanup_started.wait(), 30)
        owner = owners[0]
        assert owner.owner_task is task
        assert owner._descriptor_cleanup_task is task
        assert not task.done()
        assert not archive_custody_available(tmp_path)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        monkeypatch.setattr(os, "close", actual_close)
        for owner in owners:
            for descriptor in tuple(owner._pending_descriptor_closes):
                actual_close(descriptor)
            owner.request_sql_settlement()
        outcome = await asyncio.gather(task, return_exceptions=True)
    assert isinstance(outcome[0], BaseExceptionGroup)
    assert primary in tuple(_exception_graph(outcome[0]))
    assert any(isinstance(error, asyncio.CancelledError) for error in _exception_graph(outcome[0]))
    assert attempts == [threading.current_thread()]
    assert owner._fd == -1
    assert archive_custody_available(tmp_path)


@pytest.mark.asyncio
async def test_daemon_acquisition_physical_drain_keeps_gate_before_next_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator
    from polylogue.storage.sqlite import write_lease as leases

    actual_open, actual_close = os.open, os.close
    failed_cleanup = threading.Event()
    queued = asyncio.Event()
    selected: list[int] = []
    first = True

    def open_descriptor(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal first
        if first and path == leases.ARCHIVE_WRITE_CUSTODY_LOCK_NAME:
            first = False
            raise OSError("synthetic initial daemon lock opening failure")
        descriptor = actual_open(path, flags, *args, **kwargs)
        if path == tmp_path and first:
            selected.append(descriptor)
        return descriptor

    def close_before_effect(descriptor: int) -> None:
        if descriptor in selected:
            failed_cleanup.set()
            raise OSError("synthetic daemon directory close before effect")
        actual_close(descriptor)

    def observed(event: Any) -> None:
        if event.actor == "test.daemon.next" and event.phase == "queued":
            queued.set()

    coordinator = DaemonWriteCoordinator(archive_root=tmp_path, observer=observed)
    monkeypatch.setattr(os, "open", open_descriptor)
    monkeypatch.setattr(os, "close", close_before_effect)
    ran: list[str] = []

    async def refused_operation() -> None:
        ran.append("refused")

    async def next_operation() -> None:
        ran.append("next")

    initial = asyncio.create_task(coordinator.run("test.daemon.initial", refused_operation))
    successor = None
    try:
        assert await asyncio.to_thread(failed_cleanup.wait, 30)
        with leases._CUSTODY_REGISTRY_LOCK:
            owner = next(custody for custody in leases._CUSTODIES if custody.archive_root == tmp_path)
        successor = asyncio.create_task(coordinator.run("test.daemon.next", next_operation))
        await asyncio.wait_for(queued.wait(), 30)
        assert not initial.done()
        assert not successor.done()
        assert coordinator.snapshot().active_actor == "test.daemon.initial"
        assert ran == []
    finally:
        monkeypatch.setattr(os, "close", actual_close)
        with leases._CUSTODY_REGISTRY_LOCK:
            retained = tuple(custody for custody in leases._CUSTODIES if custody.archive_root == tmp_path)
        for custody in retained:
            for descriptor in tuple(custody._pending_descriptor_closes):
                actual_close(descriptor)
            custody.request_sql_settlement()
        outcome = await asyncio.gather(initial, return_exceptions=True)
        if successor is not None:
            await successor
        assert await coordinator.shutdown(timeout=1.0)
    assert isinstance(outcome[0], BaseExceptionGroup)
    assert ran == ["next"]
    assert owner not in leases._CUSTODIES
