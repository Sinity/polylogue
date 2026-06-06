"""Focused tests for low-level async index helpers."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from polylogue.storage.query_models import SessionRecordQuery
from polylogue.storage.repository import SessionRepository
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.storage_records import make_message, make_session


class TestAsyncEnsureIndex:
    """Tests for ensure_index."""

    @pytest.mark.asyncio
    async def test_creates_fts_table(self) -> None:
        from polylogue.pipeline.services.indexing import ensure_index, index_status

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            await backend.queries.list_sessions(SessionRecordQuery())
            await ensure_index(backend)
            status = await index_status(backend)
            assert status["exists"] is True
            await backend.close()

    @pytest.mark.asyncio
    async def test_idempotent(self) -> None:
        from polylogue.pipeline.services.indexing import ensure_index

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            await backend.queries.list_sessions(SessionRecordQuery())
            await ensure_index(backend)
            await ensure_index(backend)
            await backend.close()


class TestAsyncRebuildIndex:
    """Tests for rebuild_index."""

    @pytest.mark.asyncio
    async def test_populates_from_messages(self) -> None:
        from polylogue.pipeline.services.indexing import index_status, rebuild_index

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            repo = SessionRepository(backend=backend)
            now = datetime.now(timezone.utc).isoformat()
            session = make_session(
                session_id="test:rebuild",
                source_name="test",
                provider_session_id="ext-1",
                title="Rebuild Test",
                created_at=now,
                updated_at=now,
                content_hash=uuid4().hex,
            )
            messages = [
                make_message(
                    message_id=f"m{i}",
                    session_id="test:rebuild",
                    role="user",
                    text=f"Message {i} about testing",
                    timestamp=now,
                    content_hash=uuid4().hex[:16],
                )
                for i in range(5)
            ]
            await repo.save_session(session, messages, [])
            await rebuild_index(backend)
            status = await index_status(backend)
            assert status["exists"] is True
            assert status["count"] == 5
            await backend.close()

    @pytest.mark.asyncio
    async def test_rebuild_clears_stale_entries(self) -> None:
        from polylogue.pipeline.services.indexing import index_status, rebuild_index

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            repo = SessionRepository(backend=backend)
            now = datetime.now(timezone.utc).isoformat()
            session = make_session(
                session_id="test:stale",
                source_name="test",
                provider_session_id="ext-stale",
                title="Stale Test",
                created_at=now,
                updated_at=now,
                content_hash=uuid4().hex,
            )
            messages = [
                make_message(
                    message_id=f"stale-m{i}",
                    session_id="test:stale",
                    role="user",
                    text=f"Stale message {i}",
                    timestamp=now,
                    content_hash=uuid4().hex[:16],
                )
                for i in range(3)
            ]
            await repo.save_session(session, messages, [])
            await rebuild_index(backend)
            status_before = await index_status(backend)
            assert status_before["count"] == 3
            await repo.delete_session("test:stale")
            await rebuild_index(backend)
            status_after = await index_status(backend)
            assert status_after["count"] == 0
            await backend.close()


class TestAsyncUpdateIndex:
    """Tests for update_index_for_sessions."""

    @pytest.mark.asyncio
    async def test_incremental_update(self) -> None:
        from polylogue.pipeline.services.indexing import index_status, update_index_for_sessions

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            repo = SessionRepository(backend=backend)
            now = datetime.now(timezone.utc).isoformat()
            for session_id in ["test:a", "test:b"]:
                session = make_session(
                    session_id=session_id,
                    source_name="test",
                    provider_session_id=session_id.split(":")[1],
                    title=f"Conv {session_id}",
                    created_at=now,
                    updated_at=now,
                    content_hash=uuid4().hex,
                )
                messages = [
                    make_message(
                        message_id=f"{session_id}-m1",
                        session_id=session_id,
                        role="user",
                        text=f"Message for {session_id}",
                        timestamp=now,
                        content_hash=uuid4().hex[:16],
                    )
                ]
                await repo.save_session(session, messages, [])

            status = await index_status(backend)
            assert status["count"] == 2
            await update_index_for_sessions(["test:a"], backend)
            status = await index_status(backend)
            assert status["count"] == 2
            await backend.close()

    @pytest.mark.asyncio
    async def test_update_empty_list(self) -> None:
        from polylogue.pipeline.services.indexing import update_index_for_sessions

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            await backend.queries.list_sessions(SessionRecordQuery())
            await update_index_for_sessions([], backend)
            await backend.close()


class TestAsyncIndexStatus:
    """Tests for index_status."""

    @pytest.mark.asyncio
    async def test_reports_exists_and_count(self) -> None:
        from polylogue.pipeline.services.indexing import index_status

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = SQLiteBackend(db_path=Path(tmpdir) / "test.db")
            await backend.queries.list_sessions(SessionRecordQuery())
            status = await index_status(backend)
            assert status["exists"] is True
            assert status["count"] == 0
            await backend.close()
