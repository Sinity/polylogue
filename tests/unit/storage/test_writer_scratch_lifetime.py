"""Prepared writer artifacts transport rows without transporting native SQL."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.archive.message.roles import Role
from polylogue.sources.parsers.base import ParsedMessage
from polylogue.storage.sqlite.archive_tiers import write
from polylogue.storage.sqlite.connection_profile import (
    NativeConnectionSettlementError,
    retained_native_sql_owners_for_lifetime,
    retained_native_sql_owners_on_current_thread,
)
from tests.infra.sqlite_cursor_settlement import (
    SettlementConnection,
    arm_settlement,
    native_settlement_connections,  # noqa: F401  # Pytest fixture discovery.
    settle_fault_connections,
)


@pytest.fixture(params=["duplicates", "prefix", "signatures", "union"])
def artifact(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    result: Any
    if request.param == "duplicates":
        result = write._DiskDuplicateNativeIds(
            (
                ParsedMessage(provider_message_id=f"id-{position:04d}", role=Role.USER, text="neutral")
                for position in range(700)
                for _occurrence in range(2)
            ),
            tmp_path,
        )
    elif request.param == "prefix":
        result = write._DiskSourceMessageIds(tmp_path)
        for position in range(700):
            result[f"id-{position:04d}"] = f"message-{position}"
        result.finish()
    elif request.param == "signatures":
        result = write._DiskSignatureSequence(tmp_path)
        for position in range(700):
            result.append(f"id-{position:04d}", f"digest-{position}")
        result.finish()
    else:
        result = write._UnionScratch(tmp_path)
        for position in range(700):
            result.put("merged_message", position, (f"id-{position:04d}",), key=f"id-{position:04d}")
        result.finish()
    yield result
    result.close()


def test_prepared_scratch_iterator_can_move_and_be_abandoned_after_a_closed_page(artifact: Any) -> None:
    rows = artifact.rows("merged_message") if isinstance(artifact, write._UnionScratch) else artifact
    iterator = iter(rows)
    first = next(iterator)
    if isinstance(artifact, write._UnionScratch):
        assert first == ("id-0000",)
    else:
        assert first
    assert retained_native_sql_owners_for_lifetime(artifact) == ()
    with ThreadPoolExecutor(max_workers=1) as consumer:
        remaining, native = consumer.submit(
            lambda: (len(list(iterator)), retained_native_sql_owners_on_current_thread())
        ).result()
        assert remaining == 699 and native == ()
        abandoned = iter(rows)
        next(abandoned)
        assert retained_native_sql_owners_for_lifetime(artifact) == ()
        consumer.submit(cast(Generator[Any, None, None], abandoned).close).result()
        consumer.submit(artifact.close).result()
    assert not Path(artifact._scratch.name).exists()


def test_failed_readonly_page_close_retains_artifact_until_original_owner_retries(
    artifact: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual_connect = sqlite3.connect
    handles: list[SettlementConnection] = []

    def fail_reader_close(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = actual_connect(*args, **kwargs)
        if kwargs.get("uri") and "mode=ro" in str(args[0]):
            handle = arm_settlement(connection)
            handles.append(handle)
            return cast(sqlite3.Connection, handle)
        return cast(sqlite3.Connection, connection)

    monkeypatch.setattr(sqlite3, "connect", fail_reader_close)
    rows = artifact.rows("merged_message") if isinstance(artifact, write._UnionScratch) else artifact
    try:
        with pytest.raises(NativeConnectionSettlementError) as failure:
            next(iter(rows))
        assert len(handles) == 1
        directory = Path(artifact._scratch.name)
        assert retained_native_sql_owners_for_lifetime(artifact) == (failure.value.owner,)
        with pytest.raises(NativeConnectionSettlementError):
            artifact.close()
        assert directory.is_dir()
        handles[0].allow_cleanup.set()
        failure.value.owner.close()
        artifact.close()
        assert retained_native_sql_owners_for_lifetime(artifact) == ()
        assert not directory.exists()
    finally:
        settle_fault_connections(handles)


def test_borrowed_artifact_reader_refuses_an_inherited_task(artifact: Any) -> None:
    import asyncio

    async def run() -> None:
        scope = artifact.access() if isinstance(artifact, write._UnionScratch) else artifact.reader()
        rows = artifact.rows("merged_message") if isinstance(artifact, write._UnionScratch) else artifact
        with scope:

            async def inherited() -> None:
                with pytest.raises(RuntimeError):
                    next(iter(rows))

            await asyncio.create_task(inherited())
            assert len(rows) == 700
        assert retained_native_sql_owners_for_lifetime(artifact) == ()

    asyncio.run(run())


def test_cancelled_artifact_reader_stops_work_but_settles_its_creator_handle(artifact: Any) -> None:
    import asyncio
    import threading

    from polylogue.core.compute_cancel import compute_cancel

    cancellation = threading.Event()
    token = compute_cancel.set(cancellation)
    rows = artifact.rows("merged_message") if isinstance(artifact, write._UnionScratch) else artifact
    scope = artifact.access() if isinstance(artifact, write._UnionScratch) else artifact.reader()
    try:
        with scope:
            cancellation.set()
            with pytest.raises(asyncio.CancelledError):
                next(iter(rows))
        assert retained_native_sql_owners_for_lifetime(artifact) == ()
    finally:
        compute_cancel.reset(token)
    artifact.close()
    assert not Path(artifact._scratch.name).exists()


def test_scratch_constructor_registers_before_first_pragma_and_retains_failed_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite.connection_profile import open_scratch_connection

    class ConstructionHandle(SettlementConnection):
        def execute(self, sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
            assert retained_native_sql_owners_for_lifetime(tmp_path)
            raise LookupError("synthetic pragma failure")

    handle = arm_settlement(sqlite3.connect(tmp_path / "constructor.db", factory=ConstructionHandle))
    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: cast(sqlite3.Connection, handle))
    try:
        with pytest.raises(NativeConnectionSettlementError) as failure:
            open_scratch_connection(tmp_path / "constructor.db", lifetime_dependencies=(tmp_path,))
        assert isinstance(failure.value.__cause__, LookupError)
        assert isinstance(failure.value.failure, OSError)
        assert retained_native_sql_owners_for_lifetime(tmp_path) == (failure.value.owner,)
        handle.allow_cleanup.set()
        failure.value.owner.close()
        assert retained_native_sql_owners_for_lifetime(tmp_path) == ()
    finally:
        settle_fault_connections([handle])


@pytest.mark.parametrize("kind", ["prefix", "signatures", "union"])
def test_settled_failed_writer_can_release_its_artifact_on_the_consumer(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual_connect = sqlite3.connect
    handles: list[SettlementConnection] = []

    def failed_close(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        handle = arm_settlement(actual_connect(*args, **kwargs))
        handles.append(handle)
        return cast(sqlite3.Connection, handle)

    monkeypatch.setattr(sqlite3, "connect", failed_close)
    artifact: Any
    if kind == "prefix":
        artifact = write._DiskSourceMessageIds(tmp_path)
        artifact["native"] = "message"
    elif kind == "signatures":
        artifact = write._DiskSignatureSequence(tmp_path)
        artifact.append("message", "digest")
    else:
        artifact = write._UnionScratch(tmp_path)
        artifact.put("merged_message", 0, ("message",))
    directory = Path(artifact._scratch.name)
    try:
        with pytest.raises(NativeConnectionSettlementError) as failure:
            artifact.finish()
        assert directory.is_dir()
        assert retained_native_sql_owners_for_lifetime(artifact) == (failure.value.owner,)
        handles[0].allow_cleanup.set()
        failure.value.owner.close()
        with ThreadPoolExecutor(max_workers=1) as consumer:
            consumer.submit(artifact.close).result()
        assert not directory.exists()
        assert retained_native_sql_owners_for_lifetime(artifact) == ()
    finally:
        settle_fault_connections(handles)


@pytest.mark.parametrize(
    "kind", ["prefix", "signatures", "union", "event", "duplicates", "projection", "prepared", "shard"]
)
@pytest.mark.parametrize("failed_close", [False, True])
def test_artifact_constructor_ddl_failure_settles_or_exposes_its_actual_owner(
    kind: str, failed_close: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile

    from polylogue.core.sql_settlement import retain_native_sql_lifetimes
    from polylogue.pipeline.ids import _DiskRevisionStore
    from polylogue.sources.prepared_message_sink import SqliteMessageStore
    from polylogue.storage.sqlite.session_shard import SessionShardBuilder

    class DDLHandle(SettlementConnection):
        def execute(self, sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
            if sql.lstrip().startswith("CREATE"):
                raise LookupError("synthetic artifact DDL refusal")
            return super().execute(sql, *args, **kwargs)

    actual_connect = sqlite3.connect
    handles: list[SettlementConnection] = []

    def constructor_connection(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        handle = arm_settlement(actual_connect(*args, **{**kwargs, "factory": DDLHandle}))
        if not failed_close:
            handle.allow_cleanup.set()
        handles.append(handle)
        return cast(sqlite3.Connection, handle)

    monkeypatch.setattr(sqlite3, "connect", constructor_connection)
    scratch = tempfile.TemporaryDirectory(dir=tmp_path)
    directory = Path(scratch.name)
    constructors: dict[str, Callable[[], object]] = {
        "prefix": lambda: write._DiskSourceMessageIds(directory),
        "signatures": lambda: write._DiskSignatureSequence(directory),
        "union": lambda: write._UnionScratch(directory),
        "event": lambda: write._DiskMessageEventIndex(directory),
        "duplicates": lambda: write._DiskDuplicateNativeIds((), directory),
        "projection": lambda: _DiskRevisionStore(directory),
        "prepared": lambda: SqliteMessageStore(directory / "prepared.db"),
        "shard": lambda: SessionShardBuilder(directory / "shard.db"),
    }
    try:
        with retain_native_sql_lifetimes(scratch):
            if failed_close:
                with pytest.raises(NativeConnectionSettlementError) as failure:
                    constructors[kind]()
                owner = failure.value.owner
                assert isinstance(failure.value.__cause__, LookupError)
                assert retained_native_sql_owners_for_lifetime(scratch) == (owner,)
                assert directory.is_dir()
                handles[0].allow_cleanup.set()
                if owner._terminal_parent is not None:
                    owner._terminal_parent.close()
                else:
                    owner.close()
            else:
                with pytest.raises(LookupError):
                    constructors[kind]()
        assert len(handles) == 1
        with pytest.raises(sqlite3.ProgrammingError):
            handles[0].execute("SELECT 1")
        assert retained_native_sql_owners_for_lifetime(scratch) == ()
        scratch.cleanup()
        assert not directory.exists()
    finally:
        settle_fault_connections(handles)


@pytest.mark.parametrize("kind", ["duplicates", "shard_population", "shard_seal"])
@pytest.mark.parametrize("failed_close", [False, True])
def test_population_and_sealing_failure_settles_or_exposes_the_actual_scratch_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, failed_close: bool
) -> None:
    from polylogue.storage.sqlite.session_shard import build_session_shard

    class SealHandle(SettlementConnection):
        def execute(self, sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
            if kind == "shard_seal" and sql.startswith("INSERT INTO shard_seal"):
                raise LookupError("synthetic seal refusal")
            return super().execute(sql, *args, **kwargs)

    actual_connect = sqlite3.connect
    handles: list[SettlementConnection] = []

    def connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        handle = arm_settlement(actual_connect(*args, **{**kwargs, "factory": SealHandle}))
        if not failed_close:
            handle.allow_cleanup.set()
        handles.append(handle)
        return cast(sqlite3.Connection, handle)

    monkeypatch.setattr(sqlite3, "connect", connect)

    def broken_messages() -> Iterator[ParsedMessage]:
        yield ParsedMessage(provider_message_id="one", role=Role.USER, text="neutral")
        raise LookupError("synthetic population refusal")

    def build() -> None:
        if kind == "duplicates":
            write._DiskDuplicateNativeIds(broken_messages(), tmp_path)
        elif kind == "shard_population":
            build_session_shard(tmp_path, [object()])
        else:
            build_session_shard(tmp_path, [])

    try:
        primary = AttributeError if kind == "shard_population" else LookupError
        if failed_close:
            with pytest.raises(NativeConnectionSettlementError) as failure:
                build()
            owner = failure.value.owner
            assert isinstance(failure.value.__cause__, primary)
            assert owner in retained_native_sql_owners_on_current_thread()
            handles[0].allow_cleanup.set()
            if owner._terminal_parent is not None:
                owner._terminal_parent.close()
            else:
                owner.close()
        else:
            with pytest.raises(primary):
                build()
        assert retained_native_sql_owners_on_current_thread() == ()
        assert list(tmp_path.iterdir()) == []
        with pytest.raises(sqlite3.ProgrammingError):
            handles[0].execute("SELECT 1")
    finally:
        settle_fault_connections(handles)


@pytest.mark.parametrize("failure_kind", ["locator", "flush"])
def test_session_event_population_failure_closes_the_disk_owner_index(tmp_path: Path, failure_kind: str) -> None:
    from polylogue.pipeline.ids import disk_message_owner_resolution, message_content_identities
    from polylogue.sources.parsers.base import ParsedSessionEvent
    from polylogue.sources.prepared_message_sink import SqliteMessageSink, SqliteMessageStore

    store = SqliteMessageStore(tmp_path / "messages.db")
    sink = store.new_sink()
    sink.append(ParsedMessage(provider_message_id="one", role=Role.USER, text="neutral"))
    store.conn.commit()
    store.close()
    messages = SqliteMessageSink(store.path, 0, count=1)
    destination = sqlite3.connect(":memory:")
    try:
        destination.execute(
            "CREATE TABLE session_agent_policies (session_id TEXT, position INTEGER, "
            "approval_policy TEXT, sandbox_policy TEXT, network_policy TEXT)"
        )
        event = ParsedSessionEvent(event_type="claude_tool_result_sidecar", payload={"tool_use_id": "one"})
        locators = {"one": {"undeclared": "value"}} if failure_kind == "locator" else None
        expected = ValueError if failure_kind == "locator" else sqlite3.OperationalError
        with disk_message_owner_resolution(messages) as owners, pytest.raises(expected):
            write._write_session_events(
                destination,
                "session",
                messages,
                [event],
                content_identities=message_content_identities(messages),
                owner_resolution=owners,
                sidecar_blob_locators=locators,
            )
        assert retained_native_sql_owners_on_current_thread() == ()
        assert list(tmp_path.glob("polylogue-event-owners-*")) == []
    finally:
        destination.close()


def test_file_edit_iteration_transfers_closed_pages_and_settles_abandoned_artifact(tmp_path: Path) -> None:
    from polylogue.core.enums import BlockType, ToolOutcome
    from polylogue.pipeline.ids import message_content_identities
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedFileEdit
    from polylogue.sources.prepared_message_sink import SqliteMessageSink, SqliteMessageStore

    store = SqliteMessageStore(tmp_path / "messages.db")
    sink = store.new_sink()
    for ordinal in range(700):
        tool_id = f"edit-{ordinal}"
        sink.append(
            ParsedMessage(
                provider_message_id=f"message-{ordinal}",
                role=Role.ASSISTANT,
                blocks=[
                    ParsedContentBlock(type=BlockType.TOOL_USE, tool_id=tool_id, tool_name="Edit", tool_input={}),
                    ParsedContentBlock(
                        type=BlockType.TOOL_RESULT,
                        tool_outcome=ToolOutcome.OK,
                        is_error=False,
                        tool_id=tool_id,
                        file_edit=ParsedFileEdit(file_path="neutral.py", old_string="before", new_string="after"),
                    ),
                ],
            )
        )
    store.conn.commit()
    store.close()
    messages = SqliteMessageSink(store.path, 0, count=700)
    identities = message_content_identities(messages)
    rows = write._iter_file_edit_rows("session", messages, content_identities=identities)
    assert next(rows)[3] == "neutral.py"
    assert retained_native_sql_owners_on_current_thread() == ()
    with ThreadPoolExecutor(max_workers=1) as consumer:
        assert consumer.submit(lambda: len(list(rows))).result() == 699
        abandoned = write._iter_file_edit_rows("session", messages, content_identities=identities)
        assert next(abandoned)[3] == "neutral.py"
        assert retained_native_sql_owners_on_current_thread() == ()
        consumer.submit(cast(Generator[Any, None, None], abandoned).close).result()
    assert list(tmp_path.glob("polylogue-prefix-refs-*")) == []
