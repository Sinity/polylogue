"""Prepared writer artifacts transport rows without transporting native SQL."""

from __future__ import annotations

import sqlite3
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
from tests.infra.sqlite_settlement_handle import SettlementHandle


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
        consumer.submit(abandoned.close).result()
        consumer.submit(artifact.close).result()
    assert not Path(artifact._scratch.name).exists()


def test_failed_readonly_page_close_retains_artifact_until_original_owner_retries(
    artifact: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual_connect = sqlite3.connect
    handles: list[SettlementHandle] = []

    def fail_reader_close(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        connection = actual_connect(*args, **kwargs)
        if kwargs.get("uri") and "mode=ro" in str(args[0]):
            handle = SettlementHandle(connection)
            handles.append(handle)
            return cast(sqlite3.Connection, handle)
        return connection

    monkeypatch.setattr(sqlite3, "connect", fail_reader_close)
    rows = artifact.rows("merged_message") if isinstance(artifact, write._UnionScratch) else artifact
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


def test_borrowed_artifact_reader_refuses_an_inherited_task(artifact: Any) -> None:
    import asyncio

    async def run() -> None:
        scope = artifact.access() if isinstance(artifact, write._UnionScratch) else artifact.reader()
        rows = artifact.rows("merged_message") if isinstance(artifact, write._UnionScratch) else artifact
        with scope:

            async def inherited() -> None:
                with pytest.raises(RuntimeError):
                    len(rows)

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
                len(rows)
        assert retained_native_sql_owners_for_lifetime(artifact) == ()
    finally:
        compute_cancel.reset(token)
    artifact.close()
    assert not Path(artifact._scratch.name).exists()


def test_scratch_constructor_registers_before_first_pragma_and_retains_failed_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.core.sql_settlement import retain_native_sql_lifetimes
    from polylogue.storage.sqlite.connection_profile import open_scratch_connection

    class ConstructionHandle(SettlementHandle):
        def execute(self, sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
            assert retained_native_sql_owners_for_lifetime(tmp_path)
            raise LookupError("synthetic pragma failure")

    handle = ConstructionHandle(sqlite3.connect(tmp_path / "constructor.db"))
    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: cast(sqlite3.Connection, handle))
    with retain_native_sql_lifetimes(tmp_path), pytest.raises(NativeConnectionSettlementError) as failure:
        open_scratch_connection(tmp_path / "constructor.db")
    assert isinstance(failure.value.__cause__, LookupError)
    assert isinstance(failure.value.failure, OSError)
    assert retained_native_sql_owners_for_lifetime(tmp_path) == (failure.value.owner,)
    handle.allow_cleanup.set()
    failure.value.owner.close()
    assert retained_native_sql_owners_for_lifetime(tmp_path) == ()


@pytest.mark.parametrize("kind", ["prefix", "signatures", "union"])
def test_settled_failed_writer_can_release_its_artifact_on_the_consumer(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actual_connect = sqlite3.connect
    handles: list[SettlementHandle] = []

    def failed_close(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        handle = SettlementHandle(actual_connect(*args, **kwargs))
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
