"""Physical worker completion includes exact creator-thread SQL settlement."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import sqlite3
import sys
import threading
import time
from builtins import BaseExceptionGroup
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from polylogue.core.compute import BoundedComputeAdapter, RetainedSQLSettlement
from polylogue.core.sql_settlement import NativeSQLSettlementEvidence, SQLCustodyOwner, retained_native_sql_owners
from polylogue.pipeline import ids
from polylogue.sources.prepared_message_sink import SqliteMessageStore
from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.native_sql_descriptor_probe import selected_file_descriptors
from tests.infra.reference_sessions import reference_session
from tests.infra.sqlite_cursor_settlement import ControlledConnection, ControlledCursor

pytestmark = pytest.mark.uses_real_clock(
    "Native cleanup tests coordinate actual executor workers and cancelled asyncio wrappers."
)


class WorkerSettlementConnection(ControlledConnection):
    """Inject close failure on the actual measured connection owned by Native."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.owner = self.creator
        self.allow_cleanup = threading.Event()
        self.calls: list[tuple[str, threading.Thread]] = []

    def close(self) -> None:
        assert threading.current_thread() is self.owner
        self.calls.append(("close", threading.current_thread()))
        if not self.allow_cleanup.is_set():
            raise OSError("synthetic native connection close remains unsettled")
        super().close()


def _projection(path: Path) -> ids.SessionRevisionProjection:
    source = SqliteMessageStore(path)
    try:
        session = reference_session("worker-artifact")
        sink = source.new_sink()
        sink.extend(session.messages)
        result = ids.session_revision_projection(session.model_copy(update={"messages": sink}))
    finally:
        source.close()
    assert retained_native_sql_owners() == ()
    return result


async def _pending(inventory: Callable[[], tuple[object, ...]]) -> None:
    deadline = time.monotonic() + 5
    while not inventory() and time.monotonic() < deadline:
        await asyncio.sleep(0.005)
    assert inventory()


@pytest.mark.skipif(sys.platform != "linux", reason="physical descriptor observation uses Linux procfs")
async def test_native_cursor_failure_retains_physical_task_and_artifact_until_exact_retry(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.connection_profile import scratch_connection_context

    cursors: list[ControlledCursor] = []
    artifacts: list[Path] = []
    identities: list[tuple[int, int]] = []

    def operation() -> None:
        with scratch_connection_context(prefix="worker-cursor-", filename="artifact.db", directory=tmp_path) as conn:
            path = Path(conn.execute("PRAGMA database_list").fetchone()[2])
            conn.execute("CREATE TABLE evidence(value INTEGER)")
            conn.executemany("INSERT INTO evidence VALUES (?)", ((1,), (2,), (3,)))
            conn.commit()
            metadata = path.stat()
            identities.append((metadata.st_dev, metadata.st_ino))
            artifacts.append(path)
            cursor = conn.cursor(factory=ControlledCursor)
            cursor.execute("SELECT value FROM evidence")
            next(cursor)
            cursor.allow_cleanup.clear()
            cursors.append(cursor)

    adapter = BoundedComputeAdapter(max_workers=1)
    submitted = adapter.submit(operation)
    wrapper = asyncio.ensure_future(asyncio.wrap_future(submitted.future))
    try:
        await _pending(adapter.retained_sql_settlements)
        assert artifacts[0].is_file() and selected_file_descriptors(identities[0])
        assert not submitted.future.done() and adapter.snapshot().active_units == 1
        wrapper.cancel()
        with pytest.raises(asyncio.CancelledError):
            await wrapper
        assert not submitted.future.done()
        assert artifacts[0].is_file() and selected_file_descriptors(identities[0])
        cursors[0].allow_cleanup.set()
        submitted.retry_sql_settlement()
        with pytest.raises(NativeConnectionSettlementError):
            await asyncio.wrap_future(submitted.future)
        assert adapter.snapshot().active_units == 0
        assert not artifacts[0].exists()
        assert selected_file_descriptors(identities[0]) == ()
        assert adapter.retained_sql_settlements() == ()
    finally:
        for cursor in cursors:
            cursor.allow_cleanup.set()
        submitted.retry_sql_settlement()
        assert adapter.close(join_timeout_s=5) == ()


@pytest.mark.parametrize("projection_phase", ["writer", "readonly-page"])
@pytest.mark.parametrize("await_physical", [False, True])
async def test_failed_projection_close_retains_physical_future_context_and_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, projection_phase: str, await_physical: bool
) -> None:
    context = contextvars.ContextVar("settlement_test_context", default="absent")
    token = context.set("submitted-context")
    handles: list[WorkerSettlementConnection] = []
    artifacts: list[Path] = []
    observed_contexts: list[str] = []
    actual_connect = sqlite3.connect
    actual_close = WorkerSettlementConnection.close
    projection: ids.SessionRevisionProjection | None = None
    if projection_phase == "readonly-page":
        with ThreadPoolExecutor(max_workers=1) as builder:
            projection = builder.submit(_projection, tmp_path / "prepared.db").result()

        def operation() -> object:
            assert projection is not None
            return next(iter(projection.message_hashes))
    else:

        def operation() -> object:
            return _projection(tmp_path / "prepared.db")

    def connect(database: Any, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        name = str(database)
        is_projection = "projection.db" in name
        is_readonly = name.startswith("file:") and "mode=ro" in name
        if is_projection and is_readonly == (projection_phase == "readonly-page"):
            kwargs["factory"] = WorkerSettlementConnection
            handle = actual_connect(database, *args, **kwargs)
            assert isinstance(handle, WorkerSettlementConnection)
            handles.append(handle)
            artifacts.append(Path(name.removeprefix("file:").split("?", 1)[0]).parent)
            return handle
        connection: sqlite3.Connection = actual_connect(database, *args, **kwargs)
        return connection

    def close(handle: WorkerSettlementConnection) -> None:
        observed_contexts.append(context.get())
        actual_close(handle)

    monkeypatch.setattr(sqlite3, "connect", connect)
    monkeypatch.setattr(WorkerSettlementConnection, "close", close)
    compute = BoundedComputeAdapter(max_workers=1)
    completed = threading.Event()
    physical = None
    inventory: Callable[[], tuple[RetainedSQLSettlement, ...]]
    submitted = compute.submit(operation)
    physical = submitted.future
    physical.add_done_callback(lambda _future: completed.set())
    wrapper = asyncio.ensure_future(submitted.wait() if await_physical else asyncio.wrap_future(physical))
    inventory = compute.retained_sql_settlements
    retry = compute.retry_sql_settlement
    try:
        await _pending(inventory)
        assert handles and artifacts[0].exists()
        assert not completed.is_set()
        if physical is not None:
            assert not physical.done()
            assert compute.snapshot().active_units == 1
        wrapper.cancel()
        if await_physical:
            await asyncio.sleep(0)
            assert not wrapper.done()
        else:
            with pytest.raises(asyncio.CancelledError):
                await wrapper
        assert not completed.is_set()
        assert inventory()[0].owner_count == 1
        assert artifacts[0].exists()
        assert observed_contexts and set(observed_contexts) == {"submitted-context"}
        assert all((thread is handles[0].owner for _name, thread in handles[0].calls))
        for handle in handles:
            handle.allow_cleanup.set()
        retry()
        assert await asyncio.to_thread(completed.wait, 5)
        assert inventory() == ()
        if await_physical:
            with pytest.raises(BaseExceptionGroup) as caught:
                await wrapper
            assert any(isinstance(item, asyncio.CancelledError) for item in caught.value.exceptions)
            assert any(isinstance(item, NativeConnectionSettlementError) for item in caught.value.exceptions)
        if physical is not None:
            with pytest.raises(NativeConnectionSettlementError):
                physical.result()
            assert compute.snapshot().active_units == 0
        assert all((thread is handles[0].owner for _name, thread in handles[0].calls))
        assert set(observed_contexts) == {"submitted-context"}
    finally:
        for handle in handles:
            handle.allow_cleanup.set()
        retry()
        assert compute.close(join_timeout_s=5) == ()
        if projection_phase == "readonly-page":
            assert projection is not None
            projection.close()
        context.reset(token)


def test_retry_request_after_wake_survives_failed_following_close(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.core.sql_settlement import SQLSettlementRetry, settle_native_sql
    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner

    retry = SQLSettlementRetry()
    first_pending = threading.Event()
    first_wake = threading.Event()
    release_wake = threading.Event()
    actual_wait = SQLSettlementRetry.wait_after
    actual_close = WorkerSettlementConnection.close
    handles: list[WorkerSettlementConnection] = []
    attempts = 0

    def wait_after(notification: SQLSettlementRetry, observed: int) -> int:
        generation = actual_wait(notification, observed)
        if observed == 0:
            first_wake.set()
            assert release_wake.wait(5)
        return generation

    def close(handle: WorkerSettlementConnection) -> None:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise OSError("synthetic close has not settled")
        actual_close(handle)

    def worker() -> BaseException | None:
        handle = sqlite3.connect(":memory:", factory=WorkerSettlementConnection)
        handle.allow_cleanup.set()
        handles.append(handle)
        NativeSQLCustodyOwner(handle)
        return settle_native_sql(retry=retry, on_pending=lambda _evidence: first_pending.set(), on_settled=lambda: None)

    monkeypatch.setattr(SQLSettlementRetry, "wait_after", wait_after)
    monkeypatch.setattr(WorkerSettlementConnection, "close", close)
    with ThreadPoolExecutor(max_workers=1) as executor:
        physical = executor.submit(worker)
        try:
            assert first_pending.wait(5)
            retry.request()
            assert first_wake.wait(5)
            # This request arrives after wait returned its first generation,
            # exactly where clearing an Event would discard the second retry.
            retry.request()
            release_wake.set()
            assert isinstance(physical.result(timeout=5), NativeConnectionSettlementError)
            assert attempts == 3
            with pytest.raises(sqlite3.ProgrammingError):
                handles[0].execute("SELECT 1")
        finally:
            release_wake.set()
            retry.request()


@pytest.mark.parametrize("parent_kind", ["archive-reader", "construction-reader", "reference-seal", "shard-builder"])
async def test_parent_native_close_retains_worker_until_all_parent_obligations_settle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parent_kind: str
) -> None:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore, ArchiveStoreSettlementError
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    root = tmp_path / "archive"
    root.mkdir()
    await run_archive_fixture_write(root, lambda: bootstrap_archive_root(root))
    handles: list[WorkerSettlementConnection] = []
    parents: list[SQLCustodyOwner] = []
    actual_connect = sqlite3.connect

    def connect(database: Any, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        is_reader = str(database).startswith("file:") and "mode=ro" in str(database)
        is_shard = str(database).endswith("worker-shard.db")
        if (is_shard if parent_kind == "shard-builder" else is_reader) and (not handles):
            kwargs["factory"] = WorkerSettlementConnection
            handle = actual_connect(database, *args, **kwargs)
            assert isinstance(handle, WorkerSettlementConnection)
            handles.append(handle)
            return handle
        connection: sqlite3.Connection = actual_connect(database, *args, **kwargs)
        return connection

    construction_primary = LookupError("synthetic constructor failure after native ownership")
    if parent_kind == "construction-reader":

        def fail_attachment(_archive: ArchiveStore) -> None:
            raise construction_primary

        monkeypatch.setattr(ArchiveStore, "_attach_user_tier_if_present", fail_attachment)

    def operation() -> None:
        from polylogue.storage.sqlite.session_shard import SessionShardBuilder

        try:
            parent = (
                SessionShardBuilder(root / "worker-shard.db")
                if parent_kind == "shard-builder"
                else PreparedIndexMutation(root / "index.db", archive_root=root)
                if parent_kind == "reference-seal"
                else ArchiveStore(root, read_only=True)
            )
        except BaseException:
            parents.extend(retained_native_sql_owners())
            raise
        parents.append(parent)
        assert retained_native_sql_owners() == (parent,)
        parent.close()

    monkeypatch.setattr(sqlite3, "connect", connect)
    compute = BoundedComputeAdapter(max_workers=1)
    completed = threading.Event()
    physical = None
    inventory: Callable[[], tuple[RetainedSQLSettlement, ...]]
    physical = compute.submit(operation).future
    physical.add_done_callback(lambda _future: completed.set())
    wrapper = asyncio.ensure_future(asyncio.wrap_future(physical))
    inventory = compute.retained_sql_settlements
    retry = compute.retry_sql_settlement
    try:
        await _pending(inventory)
        assert parents and handles and (not completed.is_set())
        if parent_kind == "shard-builder":
            assert (root / "worker-shard.db").exists()
        assert inventory()[0].owner_count == 1
        if isinstance(parents[0], PreparedIndexMutation):
            assert parents[0]._scratch_directory is not None or parents[0]._observers
        if physical is not None:
            assert not physical.done()
            assert compute.snapshot().active_units == 1
        wrapper.cancel()
        with pytest.raises(asyncio.CancelledError):
            await wrapper
        assert not completed.is_set()
        handles[0].allow_cleanup.set()
        retry()
        assert await asyncio.to_thread(completed.wait, 5)
        assert inventory() == ()
        if parent_kind == "shard-builder":
            assert not (root / "worker-shard.db").exists()
        assert all((thread is handles[0].owner for _name, thread in handles[0].calls))
        if physical is not None:
            expected_error = (
                BaseExceptionGroup
                if parent_kind == "construction-reader"
                else ArchiveStoreSettlementError
                if parent_kind == "archive-reader"
                else NativeConnectionSettlementError
            )
            with pytest.raises(expected_error) as caught:
                physical.result()
            if parent_kind == "construction-reader":
                assert isinstance(caught.value, BaseExceptionGroup)
                assert caught.value.exceptions[0] is construction_primary
                assert isinstance(caught.value.exceptions[1], ArchiveStoreSettlementError)
            assert compute.snapshot().active_units == 0
    finally:
        for handle in handles:
            handle.allow_cleanup.set()
        retry()
        assert compute.close(join_timeout_s=5) == ()


def test_nested_native_settlement_preserves_outer_sql_but_closes_new_parent_child(tmp_path: Path) -> None:
    from polylogue.core.sql_settlement import SQLSettlementRetry, capture_native_sql_owners, settle_native_sql
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.write_lease import write_lease

    with write_lease("test.fixture.archive", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
    archive = ArchiveStore(tmp_path, read_only=True)
    try:
        entry = capture_native_sql_owners()
        reader = archive._open_read_connection(tmp_path / "user.db")
        assert len(capture_native_sql_owners()) == len(entry) + 1
        pending: list[NativeSQLSettlementEvidence] = []
        failure = settle_native_sql(
            retry=SQLSettlementRetry(),
            preserved_native_owners=entry,
            on_pending=pending.append,
            on_settled=lambda: None,
        )
        assert failure is None and pending == []
        assert archive._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        with pytest.raises(sqlite3.ProgrammingError):
            reader.execute("SELECT 1")
        assert retained_native_sql_owners() == (archive,)
        # Its closed binding remains available to the actual outer parent.
        archive._close_owned_read_connection(reader)
    finally:
        archive.close()
    assert retained_native_sql_owners() == ()


async def test_nested_compute_future_retains_parent_reservation_until_its_new_reader_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    await run_archive_fixture_write(tmp_path, lambda: bootstrap_archive_root(tmp_path))
    actual_connect = sqlite3.connect
    handles: list[WorkerSettlementConnection] = []
    parents: list[ArchiveStore] = []
    nested_returned = threading.Event()
    parent_verified = threading.Event()
    nested_reader_requested = threading.Event()

    def connect(database: Any, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        if nested_reader_requested.is_set() and "/user.db?" in str(database) and not handles:
            kwargs["factory"] = WorkerSettlementConnection
            handle = actual_connect(database, *args, **kwargs)
            assert isinstance(handle, WorkerSettlementConnection)
            handles.append(handle)
            return handle
        connection: sqlite3.Connection = actual_connect(database, *args, **kwargs)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    compute = BoundedComputeAdapter(max_workers=1)

    def operation() -> str:
        archive = ArchiveStore(tmp_path, read_only=True)
        parents.append(archive)
        reader: sqlite3.Connection | None = None

        def nested() -> None:
            nonlocal reader
            nested_reader_requested.set()
            reader = archive._open_read_connection(tmp_path / "user.db")
            reader.execute("SELECT COUNT(*) FROM assertions").fetchone()

        try:
            submitted = compute.submit(nested)
            nested_returned.set()
            with pytest.raises(NativeConnectionSettlementError):
                submitted.future.result()
            assert reader is not None
            archive._close_owned_read_connection(reader)
            assert archive._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
            parent_verified.set()
            return "parent completed after nested settlement"
        finally:
            archive.close()

    physical = compute.submit(operation).future
    try:
        await _pending(compute.retained_sql_settlements)
        assert parents and handles and not physical.done()
        assert not nested_returned.is_set()
        assert not parent_verified.is_set()
        assert compute.snapshot().active_units == 1
        assert compute.retained_sql_settlements()[0].owner_count == 1
        handles[0].allow_cleanup.set()
        compute.retry_sql_settlement()
        assert await asyncio.to_thread(physical.result, 5) == "parent completed after nested settlement"
        assert parent_verified.is_set()
        assert compute.retained_sql_settlements() == ()
        assert compute.snapshot().active_units == 0
        assert all(thread is handles[0].owner for _name, thread in handles[0].calls)
    finally:
        for handle in handles:
            handle.allow_cleanup.set()
        compute.retry_sql_settlement()
        assert compute.close(join_timeout_s=5) == ()


def test_submitted_retry_before_first_close_is_consumed_by_same_physical_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite.connection_profile import NativeSQLCustodyOwner

    registered = threading.Event()
    finish_work = threading.Event()
    first_close = threading.Event()
    finish_close = threading.Event()
    handles: list[WorkerSettlementConnection] = []
    close_threads: list[threading.Thread] = []
    actual_close = WorkerSettlementConnection.close

    def close(handle: WorkerSettlementConnection) -> None:
        close_threads.append(threading.current_thread())
        if len(close_threads) == 1:
            first_close.set()
            assert finish_close.wait(5)
            raise OSError("synthetic first close remains unsettled")
        actual_close(handle)

    def work() -> str:
        handle = sqlite3.connect(":memory:", factory=WorkerSettlementConnection)
        handle.allow_cleanup.set()
        handles.append(handle)
        NativeSQLCustodyOwner(handle)
        registered.set()
        assert finish_work.wait(5)
        return "prepared"

    monkeypatch.setattr(WorkerSettlementConnection, "close", close)
    compute = BoundedComputeAdapter(max_workers=1)
    operation = compute.submit(work)
    try:
        assert registered.wait(5)
        # The request precedes settlement entry, after this physical task began.
        operation.retry_sql_settlement()
        finish_work.set()
        assert first_close.wait(5)
        assert not operation.future.done()
        assert compute.snapshot().active_units == 1
        finish_close.set()
        with pytest.raises(NativeConnectionSettlementError):
            operation.future.result(timeout=5)
        assert len(close_threads) == 2
        assert all(thread is handles[0].owner for thread in close_threads)
        assert compute.retained_sql_settlements() == ()
        assert compute.snapshot().active_units == 0
    finally:
        finish_work.set()
        finish_close.set()
        operation.retry_sql_settlement()
        assert compute.close(join_timeout_s=5) == ()


@pytest.mark.skipif(sys.platform != "linux", reason="publication exclusion uses Linux flock")
@pytest.mark.parametrize("failed_resource", ["payload", "cursor"])
async def test_actual_raw_publication_keeps_exclusion_and_physical_reservation_until_creator_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_resource: str
) -> None:
    import fcntl
    import os
    from dataclasses import replace

    from polylogue.core.enums import Provider
    from polylogue.operations.raw_observation_derivation import raw_observation_frame
    from polylogue.storage.derived.raw import RawObservationDerivation, RawObservationReplacement
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.write_lease import write_lease

    await run_archive_fixture_write(tmp_path, lambda: bootstrap_archive_root(tmp_path))
    # Source admission happens on the physical worker too. The event loop
    # cannot grant a synchronous mutation lease or transfer SQL ownership.
    payload = (
        b'[{"id":"raw-lifetime","create_time":1,"current_node":"m","mapping":{"m":'
        b'{"id":"m","parent":null,"children":[],"message":{"id":"m","author":{"role":"user"},'
        b'"create_time":1,"content":{"content_type":"text","parts":["retained"]}}}}}]'
    )
    prepared: list[RawObservationReplacement] = []
    cursors: list[ControlledCursor] = []
    ready = threading.Event()
    allow_payload = threading.Event()
    payload_attempts: list[threading.Thread] = []
    original_cleanup = RawObservationReplacement._close_prepared_payload

    def close_payload(replacement: RawObservationReplacement) -> None:
        if prepared and replacement is prepared[0]:
            payload_attempts.append(threading.current_thread())
            if failed_resource == "payload" and not allow_payload.is_set():
                raise OSError("synthetic actual Raw payload cleanup remains unsettled")
        original_cleanup(replacement)

    def work() -> bool:
        with write_lease("synthetic-raw-lifetime-admission", archive_root=tmp_path):
            with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
                raw_id = archive.write_raw_payload(
                    provider=Provider.CHATGPT,
                    payload=payload,
                    source_path="raw-lifetime.json",
                    canonical_source_path="raw-lifetime.json",
                    acquired_at_ms=1,
                )
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute_adapter)
        frame = raw_observation_frame(tmp_path, raw_ids=(raw_id,))
        replacement = adapter.compute(frame, raw_id)
        prepared.append(replacement)
        assert replacement.reference_seal is not None
        assert replacement.scratch_directory is not None
        if failed_resource == "cursor":
            cursor = replacement.reference_seal.observer("user").cursor(factory=ControlledCursor)
            cursor.execute("SELECT 1 UNION ALL SELECT 2")
            next(cursor)
            cursor.allow_cleanup.clear()
            cursors.append(cursor)
        ready.set()
        # The real Raw publisher binds the exclusion before refusing this
        # moved frame. Its own terminal finally must retain failed cleanup.
        moved = replace(frame, source_revision=str(tmp_path / "other-index.db"))
        with write_lease("synthetic-raw-lifetime-publication", archive_root=tmp_path):
            return adapter.publish(moved, replacement)

    monkeypatch.setattr(RawObservationReplacement, "_close_prepared_payload", close_payload)
    compute_adapter = BoundedComputeAdapter(max_workers=1)
    adapter = compute_adapter
    operation = adapter.submit(work, exclusive_bytes=True)
    probe: int | None = None
    try:
        await _pending(adapter.retained_sql_settlements)
        assert ready.is_set() and len(prepared) == 1
        replacement = prepared[0]
        seal = replacement.reference_seal
        assert seal is not None and seal.publication_lifetime_bound
        exclusion = seal._publication_exclusion
        assert exclusion is not None and exclusion.held
        assert not operation.future.done() and adapter.snapshot().active_units == 1
        assert replacement.scratch_directory is not None
        # The prepared artifact stays on disk while any of its native owners
        # (the payload carrier or a seal child cursor) remains unsettled.
        assert replacement.scratch_directory.exists()
        probe = os.open(exclusion.path, os.O_RDWR)
        with pytest.raises(BlockingIOError):
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        allow_payload.set()
        for cursor in cursors:
            cursor.allow_cleanup.set()
        operation.retry_sql_settlement()
        # A lone failed payload close surfaces as itself; an unsettled native
        # child surfaces as the settlement error or its group.
        expected_failure: tuple[type[BaseException], ...] = (
            (OSError,) if failed_resource == "payload" else (BaseExceptionGroup, NativeConnectionSettlementError)
        )
        with pytest.raises(expected_failure):
            await asyncio.wrap_future(operation.future)
        assert adapter.snapshot().active_units == 0
        assert adapter.retained_sql_settlements() == ()
        assert not replacement.scratch_directory.exists()
        assert not exclusion.held
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert payload_attempts and all(thread is payload_attempts[0] for thread in payload_attempts)
        if failed_resource == "cursor":
            assert len(payload_attempts) == 1
            assert cursors[0].creator is payload_attempts[0]
    finally:
        allow_payload.set()
        for cursor in cursors:
            cursor.allow_cleanup.set()
        operation.retry_sql_settlement()
        assert adapter.close(join_timeout_s=5) == ()
        if probe is not None:
            os.close(probe)


@pytest.mark.parametrize("exclusive_bytes", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_operation_prepared_phase_retains_original_creator_and_refuses_successors_until_native_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancelled: bool, exclusive_bytes: bool
) -> None:
    from polylogue.core.stage_admission import admit_stage_write
    from polylogue.core.write_lease import current_write_lease
    from polylogue.daemon.operation_runtime import DaemonOperationRuntime
    from polylogue.daemon.write_coordinator import (
        DaemonWriteCoordinator,
        DaemonWriterSettlementError,
        DaemonWriteThreadBridge,
    )
    from polylogue.storage.sqlite.connection_profile import scratch_connection_context

    actual_connect = sqlite3.connect
    handles: list[WorkerSettlementConnection] = []
    creators: list[threading.Thread] = []
    entered_publication = threading.Event()
    artifact_paths: list[Path] = []

    def connect(database: Any, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        if str(database).endswith("publication.db"):
            kwargs["factory"] = WorkerSettlementConnection
            handle = actual_connect(database, *args, **kwargs)
            assert isinstance(handle, WorkerSettlementConnection)
            handles.append(handle)
            return handle
        connection: sqlite3.Connection = actual_connect(database, *args, **kwargs)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    adapter = BoundedComputeAdapter(max_workers=1, queue_units=1, queue_bytes=100)
    coordinator = DaemonWriteCoordinator(archive_root=tmp_path)
    bridge = DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop())
    from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner

    runtime = DaemonOperationRuntime(
        tmp_path,
        write_bridge=bridge,
        execution_kernel=adapter,
        raw_observation_owner=RawObservationConvergenceOwner(
            tmp_path, compute_adapter=adapter, write_bridge=bridge, write_coordinator=bridge.coordinator
        ),
    )

    with pytest.raises(RuntimeError):
        runtime.prepared_compute_adapter()

    def admitted_without_publication_admission() -> None:
        from polylogue.core.stage_admission import stage_write_admission_bound

        adapter.require_current_creator()
        assert not stage_write_admission_bound()
        with pytest.raises(RuntimeError):
            runtime.prepared_compute_adapter()

    await asyncio.wrap_future(adapter.submit(admitted_without_publication_admission).future)

    release_publication = threading.Event()

    def operation() -> int:
        adapter.require_current_creator()
        assert runtime.prepared_compute_adapter() is adapter
        if exclusive_bytes:
            runtime.prepared_compute_adapter().amend_current_input_demand(200)
        else:
            with pytest.raises(RuntimeError):
                runtime.prepared_compute_adapter().amend_current_input_demand(200)
        assert current_write_lease() is None
        creators.append(threading.current_thread())
        with scratch_connection_context(prefix="operation-prepared-", filename="phase.db", directory=tmp_path) as conn:
            with contextlib.closing(conn.execute("CREATE TABLE prepared(value INTEGER)")):
                pass
            conn.commit()

            def publish() -> int:
                lease = current_write_lease()
                assert lease is not None and lease.coordinator is coordinator
                adapter.require_current_creator()
                assert threading.current_thread() is creators[0]
                with contextlib.closing(conn.execute("SELECT COUNT(*) FROM prepared")) as cursor:
                    assert cursor.fetchone()[0] == 0
                with scratch_connection_context(
                    prefix="operation-publication-", filename="publication.db", directory=tmp_path
                ) as publication:
                    with contextlib.closing(publication.execute("PRAGMA database_list")) as cursor:
                        artifact_paths.append(Path(cursor.fetchone()[2]))
                    with contextlib.closing(publication.execute("CREATE TABLE effect(value INTEGER)")):
                        pass
                    with contextlib.closing(publication.execute("INSERT INTO effect VALUES (7)")):
                        pass
                    publication.commit()
                    entered_publication.set()
                    release_publication.wait()
                    # The publication child's close fails on leaving this block.
                    return 7

            return admit_stage_write("operation.actual-prepared-publication", publish)

    phase = asyncio.create_task(
        runtime.prepared_phase("native-lifetime", operation, estimated_bytes=7, exclusive_bytes=exclusive_bytes)
    )
    try:
        deadline = time.monotonic() + 5
        while not entered_publication.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        assert entered_publication.is_set()
        # The admitted body holds the single-writer gate while it is inside it.
        assert coordinator.snapshot().active_actor == "operation.actual-prepared-publication"
        if cancelled:
            # Cancellation never abandons the physical worker.
            phase.cancel()
            await asyncio.sleep(0)
            assert not phase.done()
        release_publication.set()

        await _pending(adapter.retained_sql_settlements)
        # Shipped contract (01c1f38193): the prepared phase delivers the
        # writer's typed refusal while its compute slot stays occupied by the
        # original creator, which still owns the unsettled publication SQL.
        with pytest.raises((DaemonWriterSettlementError, BaseExceptionGroup)) as refused:
            await phase
        failures = refused.value.exceptions if isinstance(refused.value, BaseExceptionGroup) else (refused.value,)
        assert any(
            isinstance(failure, DaemonWriterSettlementError) and failure.code == "writer_sql_unsettled"
            for failure in failures
        )
        assert cancelled == any(isinstance(failure, asyncio.CancelledError) for failure in failures)
        assert adapter.snapshot().active_units == 1
        assert adapter.snapshot().active_input_bytes == (207 if exclusive_bytes else 7)
        assert adapter.snapshot().used_bytes == (100 if exclusive_bytes else 7)
        assert coordinator.snapshot().unsettled_writer_workers == 1
        assert artifact_paths[0].exists()

        # No second writer is admitted while that SQL is unsettled.
        with pytest.raises(DaemonWriterSettlementError):
            await coordinator.run_sync("test.unsettled_successor", lambda: None)
        assert artifact_paths[0].exists()

        # A later admission retries cleanup on the original creator.
        handles[0].allow_cleanup.set()
        await coordinator.run_sync("test.settled_successor", lambda: None)
        assert coordinator.snapshot().unsettled_writer_workers == 0
        deadline = time.monotonic() + 5
        while adapter.snapshot().active_units and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        assert adapter.snapshot().active_units == 0
        assert adapter.snapshot().active_input_bytes == adapter.snapshot().exclusive_byte_units == 0
        assert adapter.retained_sql_settlements() == ()
        assert coordinator.snapshot().active_actor is None
        assert not artifact_paths[0].exists()
        assert all(thread is creators[0] for _name, thread in handles[0].calls)
    finally:
        release_publication.set()
        for handle in handles:
            handle.allow_cleanup.set()
        adapter.retry_sql_settlement()
        if not phase.done():
            with contextlib.suppress(BaseException):
                await phase
        adapter.shutdown(wait=True)
        assert await coordinator.shutdown(timeout=float("inf"))
