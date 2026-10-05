"""Tests for pipeline/services/indexing.py - IndexService."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from builtins import BaseExceptionGroup
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.config import Config
from polylogue.pipeline.services.indexing import IndexService
from polylogue.storage.fts import fts_lifecycle
from polylogue.storage.fts.fts_lifecycle import ensure_fts_index_async
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.storage_records import make_content_block, make_message, make_session, save_session_to_archive


def _config() -> Config:
    return Config(
        archive_root=Path("/tmp"),
        render_root=Path("/tmp/render"),
        sources=[],
    )


class TestIndexService:
    """Test IndexService functionality."""

    async def test_update_index_empty_list(self, sqlite_backend: SQLiteBackend) -> None:
        """Update index with empty session list."""
        service = IndexService(_config(), backend=sqlite_backend)

        result = await service.update_index([])
        assert result is True

    async def test_update_index_with_sessions(self, sqlite_backend: SQLiteBackend) -> None:
        """Update index with actual sessions."""

        # Create test data using backend-compatible records
        conv = make_session(
            session_id="conv1",
            source_name="chatgpt",
            title="Test",
            content_hash="hash123",
        )
        msg = make_message(
            message_id="msg1",
            session_id="conv1",
            role="user",
            text="Hello world",
            content_hash="msghash1",
        )
        # Insert using backend API
        await save_session_to_archive(sqlite_backend, session=conv, messages=[msg])

        service = IndexService(_config(), backend=sqlite_backend)

        result = await service.update_index(["conv1"])
        assert result is True

    async def test_update_index_accepts_async_iterable(self, sqlite_backend: SQLiteBackend) -> None:
        """Streaming session IDs can be indexed without prebuilding a list."""

        await save_session_to_archive(
            sqlite_backend,
            session=make_session(
                session_id="conv-stream",
                source_name="chatgpt",
                title="Stream Test",
                content_hash="hash-stream",
            ),
            messages=[
                make_message(
                    message_id="msg-stream",
                    session_id="conv-stream",
                    role="user",
                    text="hello world",
                    content_hash="msghash-stream",
                )
            ],
        )

        service = IndexService(_config(), backend=sqlite_backend)

        async def session_ids() -> AsyncIterator[str]:
            yield "conv-stream"

        result = await service.update_index(session_ids())

        assert result is True
        status = await service.get_index_status()
        assert status["exists"] is True
        assert status["count"] >= 1

    async def test_rebuild_index_success(self, sqlite_backend: SQLiteBackend) -> None:
        """Rebuild index from scratch."""
        service = IndexService(_config(), backend=sqlite_backend)

        result = await service.rebuild_index()
        assert result is True

    @pytest.mark.parametrize("terminal", ["complete", "callback_error", "cancel"])
    async def test_rebuild_index_reports_chunk_progress(
        self, sqlite_backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch, terminal: str
    ) -> None:
        """The actual full route reports settled pages and retains worker custody."""
        monkeypatch.setattr(fts_lifecycle, "FTS_REBUILD_SESSION_PAGE_SIZE", 1)

        for index in range(3):
            session_id = f"conv-progress-{index}"
            await save_session_to_archive(
                sqlite_backend,
                session=make_session(
                    session_id=session_id,
                    source_name="chatgpt",
                    title=f"Session {index}",
                    content_hash=f"hash-{index}",
                ),
                messages=[
                    make_message(
                        message_id=f"msg-progress-{index}",
                        session_id=session_id,
                        role="user",
                        text=f"Hello from session {index}",
                        content_hash=f"message-hash-{index}",
                    )
                ],
            )

        service = IndexService(_config(), backend=sqlite_backend)
        progress_events: list[tuple[int, str | None]] = []

        observer_error = RuntimeError("synthetic progress observer refusal")

        def capture(amount: int, desc: str | None = None) -> None:
            progress_events.append((amount, desc))
            if desc == "Indexing: full-text search 1/3":
                if terminal == "callback_error":
                    raise observer_error
                if terminal == "cancel":
                    rebuild.cancel()

        rebuild = asyncio.create_task(service.rebuild_index(progress_callback=capture))
        if terminal == "complete":
            assert await rebuild is True
            assert progress_events == [
                (0, "Indexing: full-text search 0/3"),
                (1, "Indexing: full-text search 1/3"),
                (1, "Indexing: full-text search 2/3"),
                (1, "Indexing: full-text search 3/3"),
            ]
        else:
            error = RuntimeError if terminal == "callback_error" else asyncio.CancelledError
            with pytest.raises(error) as caught:
                await rebuild
            if terminal == "callback_error":
                assert caught.value is observer_error
            assert progress_events == [
                (0, "Indexing: full-text search 0/3"),
                (1, "Indexing: full-text search 1/3"),
            ]
            await asyncio.sleep(0)
            assert len(progress_events) == 2
        async with sqlite_backend.connection() as conn:
            async with conn.execute("SELECT COUNT(*) FROM messages_fts_docsize") as cursor:
                row = await cursor.fetchone()
                assert row is not None and row[0] == 3
            async with conn.execute("SELECT COUNT(*) FROM messages_fts_identity") as cursor:
                row = await cursor.fetchone()
                assert row is not None and row[0] == 3

    async def test_rebuild_index_reports_empty_completion(self, sqlite_backend: SQLiteBackend) -> None:
        events: list[tuple[int, str | None]] = []
        assert (
            await IndexService(_config(), backend=sqlite_backend).rebuild_index(
                progress_callback=lambda amount, desc=None: events.append((amount, desc))
            )
            is True
        )
        assert events == [(0, "Indexing: full-text search 0/0")]

    async def test_rebuild_index_skips_action_phase_for_tool_use_blocks(
        self,
        sqlite_backend: SQLiteBackend,
    ) -> None:
        """Full rebuild reports FTS progress only; the action-event phase was removed (#1743)."""

        await save_session_to_archive(
            sqlite_backend,
            session=make_session(
                session_id="conv-plain",
                source_name="chatgpt",
                title="Plain Session",
                content_hash="hash-plain",
            ),
            messages=[
                make_message(
                    message_id="msg-plain",
                    session_id="conv-plain",
                    role="user",
                    text="No tool use here",
                    sort_key=0.5,
                    content_hash="message-hash-plain",
                )
            ],
        )
        # Content block folded into message — no separate save_content_blocks step.
        await save_session_to_archive(
            sqlite_backend,
            session=make_session(
                session_id="conv-action",
                source_name="chatgpt",
                title="Action Session",
                content_hash="hash-action",
            ),
            messages=[
                make_message(
                    message_id="msg-action",
                    session_id="conv-action",
                    role="assistant",
                    text="Ran rg",
                    sort_key=1.0,
                    content_hash="message-hash-action",
                    blocks=[
                        make_content_block(
                            message_id="msg-action",
                            session_id="conv-action",
                            block_index=0,
                            block_type="tool_use",
                            tool_name="exec_command",
                            tool_id="tool-1",
                            tool_input='{"cmd":"rg -n actions"}',
                        )
                    ],
                )
            ],
        )

        service = IndexService(_config(), backend=sqlite_backend)
        progress_events: list[tuple[int, str | None]] = []

        def capture(amount: int, desc: str | None = None) -> None:
            progress_events.append((amount, desc))

        result = await service.rebuild_index(progress_callback=capture)

        assert result is True
        descriptions = [desc for _, desc in progress_events if desc is not None]
        assert all(desc.startswith("Indexing: full-text search ") for desc in descriptions)
        assert descriptions[0] == "Indexing: full-text search 0/2"
        assert descriptions[-1] == "Indexing: full-text search 2/2"

    async def test_get_index_status(self, sqlite_backend: SQLiteBackend) -> None:
        """Get index status."""
        service = IndexService(_config(), backend=sqlite_backend)

        status = await service.get_index_status()
        assert isinstance(status, dict)
        assert "exists" in status
        assert "count" in status

    async def test_get_index_status_uses_service_connection(self, sqlite_backend: SQLiteBackend) -> None:
        """Regression: get_index_status must use the service's backend, not open_connection(None)."""
        service = IndexService(_config(), backend=sqlite_backend)

        # Ensure FTS table exists via this backend
        async with sqlite_backend.connection() as conn:
            await ensure_fts_index_async(conn)

        # get_index_status should use the same backend and find the table
        status = await service.get_index_status()
        assert status["exists"] is True
        assert isinstance(status["count"], int)

    async def test_get_index_status_after_schema_init(self, sqlite_backend: SQLiteBackend) -> None:
        """Status after schema init shows index exists with zero-or-more entries."""
        service = IndexService(_config(), backend=sqlite_backend)

        from polylogue.storage.query_models import SessionRecordQuery

        await sqlite_backend.queries.list_sessions(SessionRecordQuery())

        status = await service.get_index_status()
        assert status["exists"] is True
        assert isinstance(status["count"], int)


# --- Merged from test_supplementary_coverage.py ---


class TestIndexServiceErrors:
    """Tests for IndexService error handling paths."""

    async def test_update_index_failure(self) -> None:
        """update_index should return False on exception."""
        config = MagicMock()
        service = IndexService(config=config, backend=MagicMock())

        with patch(
            "polylogue.pipeline.services.indexing.update_index_for_sessions",
            new_callable=AsyncMock,
            side_effect=sqlite3.DatabaseError("db locked"),
        ):
            result = await service.update_index(["conv1", "conv2"])
            assert result is False

    async def test_rebuild_index_failure(self) -> None:
        """rebuild_index should return False on exception."""
        config = MagicMock()
        service = IndexService(config=config, backend=MagicMock())

        with patch(
            "polylogue.pipeline.services.indexing.rebuild_index",
            new_callable=AsyncMock,
            side_effect=sqlite3.DatabaseError("disk full"),
        ):
            result = await service.rebuild_index()
            assert result is False

    async def test_get_index_status_failure(self) -> None:
        """get_index_status should return fallback on exception."""
        config = MagicMock()
        service = IndexService(config=config, backend=MagicMock())

        with patch(
            "polylogue.pipeline.services.indexing.index_status",
            new_callable=AsyncMock,
            side_effect=sqlite3.DatabaseError("no such table"),
        ):
            result = await service.get_index_status()
            assert result == {"exists": False, "count": 0}

    async def test_update_index_no_backend(self) -> None:
        """Update without backend returns False."""
        config = MagicMock()
        service = IndexService(config=config, backend=None)

        result = await service.update_index(["conv-1"])

        assert result is False

    async def test_rebuild_no_backend(self) -> None:
        """Rebuild without backend returns False."""
        config = MagicMock()
        service = IndexService(config=config, backend=None)

        result = await service.rebuild_index()

        assert result is False

    async def test_get_index_status_no_backend(self) -> None:
        """Status without backend returns defaults."""
        config = MagicMock()
        service = IndexService(config=config, backend=None)

        status = await service.get_index_status()

        assert status == {"exists": False, "count": 0}

    async def test_update_index_empty_ids_ensures_index(self) -> None:
        """update_index always delegates to the canonical storage helper."""
        config = MagicMock()
        mock_backend = MagicMock()
        service = IndexService(config=config, backend=mock_backend)

        with patch(
            "polylogue.pipeline.services.indexing.update_index_for_sessions", new_callable=AsyncMock
        ) as mock_update:
            result = await service.update_index([])
            assert result is True
            mock_update.assert_called_once_with([], mock_backend)


@pytest.mark.uses_real_clock
@pytest.mark.parametrize("distinct_failure", [False, True])
async def test_borrowed_full_rebuild_settles_cancelled_worker_and_preserves_distinct_failure(
    sqlite_backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch, distinct_failure: bool
) -> None:
    """Cancellation waits for native exit and retains the actual observer failure."""
    stopping = threading.Event()
    allow_exit = threading.Event()
    exited = threading.Event()
    original_rebuild = fts_lifecycle.rebuild_fts_index_sync
    observer_error = RuntimeError("synthetic observer failure racing cancellation")
    callback_threads: list[int] = []
    loop_thread = threading.get_ident()
    cancel_token = "synthetic full rebuild cancellation"

    def retained_rebuild(conn: sqlite3.Connection, **kwargs: Any) -> None:
        try:
            original_rebuild(conn, **kwargs)
        finally:
            stopping.set()
            try:
                assert allow_exit.wait(5)
            finally:
                exited.set()

    monkeypatch.setattr(fts_lifecycle, "rebuild_fts_index_sync", retained_rebuild)

    def observe(amount: int, desc: str | None = None) -> None:
        callback_threads.append(threading.get_ident())
        rebuild.cancel(cancel_token)
        if distinct_failure:
            raise observer_error

    async with sqlite_backend.connection() as conn:
        rebuild = asyncio.create_task(fts_lifecycle.rebuild_fts_index_async(conn, progress_callback=observe))
        failure: BaseException | None = None
        try:
            assert await asyncio.to_thread(stopping.wait, 5)
            assert not rebuild.done(), "logical completion released a physically retained worker"
        finally:
            allow_exit.set()
            try:
                await rebuild
            except BaseException as caught:
                failure = caught
        assert exited.is_set()
        assert callback_threads == [loop_thread]
        if distinct_failure:
            assert isinstance(failure, BaseExceptionGroup)
            cancellation, actual_failure = failure.exceptions
            assert isinstance(cancellation, asyncio.CancelledError)
            assert cancellation.args == (cancel_token,)
            assert actual_failure is observer_error
        else:
            assert isinstance(failure, asyncio.CancelledError)
            assert failure.args == (cancel_token,)
        async with conn.execute("SELECT 1") as cursor:
            row = await cursor.fetchone()
            assert row is not None and row[0] == 1
