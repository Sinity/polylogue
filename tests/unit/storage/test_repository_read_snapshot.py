"""Composed repository reads own a single snapshot and connection lifetime."""

from __future__ import annotations

import asyncio
import json
from contextvars import ContextVar
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.storage.repository import SessionRepository
from polylogue.storage.sqlite import async_sqlite
from polylogue.storage.sqlite.query_store import SQLiteQueryStore
from polylogue.storage.sqlite.query_store_archive import messages_q
from tests.infra.retained_replay import publish_retained_payload


def _payload(extended: bool) -> bytes:
    rows = [
        {"type": "session_meta", "payload": {"id": "neutral-snapshot", "timestamp": "2026-06-02T00:00:00Z"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "one",
                "role": "user",
                "content": [{"type": "input_text", "text": "needle"}],
            },
        },
    ]
    if extended:
        rows.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": "two",
                    "role": "assistant",
                    "timestamp": "2026-06-02T00:00:01Z",
                    "content": [{"type": "output_text", "text": "second"}],
                },
            }
        )
    return ("\n".join(json.dumps(row) for row in rows) + "\n").encode()


async def _publish(root: Path, extended: bool = False) -> None:
    await publish_retained_payload(
        root,
        provider=Provider.CODEX,
        payload=_payload(extended),
        source_path="/neutral/session.jsonl",
        acquired_at_ms=2 if extended else 1,
    )


@pytest.mark.parametrize("route", ["summary", "full"])
def test_canonical_append_cannot_tear_repository_hydration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    async def exercise() -> None:
        root = tmp_path / "archive"
        await _publish(root)
        async with SessionRepository(db_path=root / "index.db") as repo:
            before = (await repo.search_summaries("needle"))[0]
            method = "get_message_counts_batch" if route == "summary" else "get_messages_batch"
            original = getattr(SQLiteQueryStore, method)
            written = False

            async def interleave(store, ids, **kwargs):
                nonlocal written
                if not written:
                    written = True
                    await _publish(root, True)
                return await original(store, ids, **kwargs)

            monkeypatch.setattr(SQLiteQueryStore, method, interleave)
            current = (await (repo.search_summaries("needle") if route == "summary" else repo.search("needle")))[0]
            count = current.message_count if route == "summary" else len(current.messages)
            assert count == 1
            assert current.updated_at == before.updated_at
            after = (await repo.search_summaries("needle"))[0]
            assert after.message_count == 2
            assert after.updated_at != before.updated_at

    asyncio.run(exercise())


@pytest.mark.parametrize("cancel_first", [False, True], ids=["finish", "cancel"])
def test_overlapping_searches_keep_their_own_hydration_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_first: bool
) -> None:
    owner = ContextVar("neutral_read_owner", default="")

    async def exercise() -> None:
        root = tmp_path / "archive"
        await _publish(root)
        b_ready, a_ready, allow_b, b_borrowed, release_b = (asyncio.Event() for _ in range(5))
        original_front = SQLiteQueryStore.get_messages_batch
        original_rows = messages_q.get_messages_batch

        async def front(store, ids, **kwargs):
            if owner.get() == "B":
                b_ready.set()
                await allow_b.wait()
            if owner.get() == "A":
                a_ready.set()
                await b_borrowed.wait()
                if cancel_first:
                    await asyncio.Event().wait()
            return await original_front(store, ids, **kwargs)

        async def rows(conn, ids, **kwargs):
            result = await original_rows(conn, ids, **kwargs)
            if owner.get() == "B":
                b_borrowed.set()
                await release_b.wait()
            return result

        monkeypatch.setattr(SQLiteQueryStore, "get_messages_batch", front)
        monkeypatch.setattr(messages_q, "get_messages_batch", rows)
        async with SessionRepository(db_path=root / "index.db") as repo:

            async def read(name):
                token = owner.set(name)
                try:
                    return await repo.search("needle")
                finally:
                    owner.reset(token)

            b = asyncio.create_task(read("B"))
            a = None
            try:
                await asyncio.wait_for(b_ready.wait(), 10)
                a = asyncio.create_task(read("A"))
                await asyncio.wait_for(a_ready.wait(), 10)
                allow_b.set()
                await asyncio.wait_for(b_borrowed.wait(), 10)
                if cancel_first:
                    a.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await a
                else:
                    assert len(await asyncio.wait_for(a, 10)) == 1
                release_b.set()
                result = await asyncio.wait_for(b, 10)
                assert len(result) == 1 and len(result[0].messages) == 1
                assert not any(entry.backend is repo.backend for entry in async_sqlite._BACKEND_CONNECTIONS.values())
            finally:
                for task in (a, b):
                    if task is not None and not task.done():
                        task.cancel()
                await asyncio.gather(*(task for task in (a, b) if task is not None), return_exceptions=True)

    asyncio.run(exercise())


def test_snapshot_store_refuses_other_task_and_use_after_scope(tmp_path: Path) -> None:
    async def exercise() -> None:
        root = tmp_path / "archive"
        await _publish(root)
        async with SessionRepository(db_path=root / "index.db") as repo:
            async with repo.queries.read_snapshot() as queries:
                assert await queries.search_session_hits("needle")
                with pytest.raises(RuntimeError, match="active operation owner"):
                    await asyncio.create_task(queries.search_session_hits("needle"))
            with pytest.raises(RuntimeError, match="active operation owner"):
                await queries.search_session_hits("needle")

    asyncio.run(exercise())


def test_search_failed_native_close_retains_exact_owner_until_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aiosqlite

    async def exercise() -> None:
        root = tmp_path / "archive"
        await _publish(root)
        repo = SessionRepository(db_path=root / "index.db")
        captured = []
        original_rows = messages_q.get_messages_batch
        execute = aiosqlite.Connection._execute
        refuse_close = True

        async def rows(conn, ids, **kwargs):
            captured.append(conn)
            return await original_rows(conn, ids, **kwargs)

        async def execute_with_fault(conn, function, *args, **kwargs):
            if refuse_close and conn in captured and getattr(function, "__name__", None) == "close_raw":
                raise OSError("neutral native close refusal")
            return await execute(conn, function, *args, **kwargs)

        monkeypatch.setattr(messages_q, "get_messages_batch", rows)
        monkeypatch.setattr(aiosqlite.Connection, "_execute", execute_with_fault)
        try:
            with pytest.raises(OSError, match="native close refusal"):
                await repo.search("needle")
            assert len(captured) == 1
            conn = captured[0]
            assert conn._connection is not None and conn._thread.is_alive()
            assert async_sqlite._BACKEND_CONNECTIONS[id(conn)].backend is repo.backend
        finally:
            refuse_close = False
            await repo.close()
        assert conn._connection is None and not conn._thread.is_alive()
        assert not any(entry.backend is repo.backend for entry in async_sqlite._BACKEND_CONNECTIONS.values())

    asyncio.run(exercise())
