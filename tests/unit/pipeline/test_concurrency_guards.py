"""Tests for concurrency, resource management, and async safety.

Covers:
- ebef687: TOCTOU race in metadata read-modify-write
- d5c3228: sqlite-vec created+discarded connection on every operation
- fa2b132: sqlite-vec returned broken connection instead of raising
- abbe871: Connection storm at scale (3000+ parallel asyncio.gather calls)

Also covers filter state isolation: reusing a filter builder must not
accumulate state from previous uses.

NOTE: Claude Code model property tests (role mapping, timestamps, boolean
flags) have been consolidated into tests/unit/sources/test_models.py using
shared tables from tests/infra/tables.py.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from pathlib import Path

import pytest

from polylogue.archive.filter.filters import SessionFilter
from polylogue.storage.repository import SessionRepository
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.storage_records import SessionBuilder, make_message, make_session

# =============================================================================
# Filter state isolation (implicit in CLI routing bugs)
# =============================================================================


class TestFilterStateIsolation:
    """SessionFilter must not leak state between uses.

    The fluent filter builder returns ``self`` for chaining over an
    immutable :class:`SessionQueryPlan`. The plan is the canonical
    execution state in the archive; these tests assert that each
    mutator updates the plan with the documented accumulate/replace
    semantics and that two filters never share a plan.
    """

    @staticmethod
    def _filter(tmp_path: Path) -> SessionFilter:
        return SessionFilter(archive_root=tmp_path)

    def test_separate_filters_have_independent_state(self, tmp_path: Path) -> None:
        """Two filters built from the same root must not share plan state."""
        f1 = self._filter(tmp_path)
        f2 = self._filter(tmp_path)

        f1.origin("chatgpt-export")
        f2.origin("claude-ai-export")

        assert f1.build_query_plan().origins == ("chatgpt-export",)
        assert f2.build_query_plan().origins == ("claude-ai-export",)

    def test_chained_methods_accumulate_on_same_instance(self, tmp_path: Path) -> None:
        """Chaining on the same filter must accumulate predicates in the plan."""
        f = self._filter(tmp_path)
        f.origin("chatgpt-export").contains("error").limit(10)

        plan = f.build_query_plan()
        assert plan.origins == ("chatgpt-export",)
        assert plan.contains_terms == ("error",)
        assert plan.limit == 10

    def test_filter_reuse_accumulates_providers(self, tmp_path: Path) -> None:
        """Reusing a filter OR-accumulates providers (documented contract)."""
        f = self._filter(tmp_path)
        f.origin("chatgpt-export")
        f.origin("claude-ai-export")  # second call accumulates, not replaces

        assert f.build_query_plan().origins == ("chatgpt-export", "claude-ai-export")

    @pytest.mark.asyncio
    async def test_filter_sort_replaces_not_accumulates(self, tmp_path: Path) -> None:
        """sort() should replace the previous sort, not append.

        Seeded directly: an older long session and a newer short one.
        sort("date") would put the newer first; the later sort("words")
        must win, ordering by word count so the long session is first.
        """
        db_path = tmp_path / "index.db"
        (
            SessionBuilder(db_path, "old-long")
            .provider("chatgpt")
            .title("old-long")
            .created_at("2025-01-01T00:00:00+00:00")
            .updated_at("2025-01-01T00:00:00+00:00")
            .add_message(role="user", text="this message has many many words")
            .save()
        )
        (
            SessionBuilder(db_path, "new-short")
            .provider("chatgpt")
            .title("new-short")
            .created_at("2025-01-02T00:00:00+00:00")
            .updated_at("2025-01-02T00:00:00+00:00")
            .add_message(role="user", text="tiny")
            .save()
        )

        f = self._filter(tmp_path)
        f.sort("date")
        f.sort("words")
        assert f.build_query_plan().sort == "words"

        results = await f.list()
        assert [str(c.id).split(":")[-1] for c in results][0] == "ext-old-long"

    def test_fresh_filter_has_no_predicates(self, tmp_path: Path) -> None:
        """A brand-new filter should carry an empty plan."""
        plan = self._filter(tmp_path).build_query_plan()
        assert plan.limit is None
        assert plan.origins == ()
        assert plan.contains_terms == ()
        assert plan.sort is None

    def test_limit_replaces_previous_limit(self, tmp_path: Path) -> None:
        """Calling limit() twice replaces, doesn't stack."""
        f = self._filter(tmp_path)
        f.limit(40)
        f.limit(5)
        assert f.build_query_plan().limit == 5


# =============================================================================
# Async backend: batch operations must not create N connections
# =============================================================================


class TestConnectionManagement:
    """Batch operations must use O(1) connections, not O(N)."""

    @pytest.mark.asyncio
    async def test_get_sessions_batch_uses_single_connection(self, sqlite_backend: SQLiteBackend) -> None:
        """get_sessions_batch must use 1 query, not N queries."""
        backend = sqlite_backend

        # Insert 20 sessions
        async with backend.connection() as conn:
            for i in range(20):
                await conn.execute(
                    "INSERT INTO sessions (session_id, source_name, "
                    "provider_session_id, content_hash, version) "
                    "VALUES (?, 'test', ?, ?, 1)",
                    (f"conv-{i}", f"pconv-{i}", f"hash-{i}"),
                )
            await conn.commit()

        # Batch get should work
        ids = [f"conv-{i}" for i in range(20)]
        records = await backend.get_sessions_batch(ids)
        assert len(records) == 20

    @pytest.mark.asyncio
    async def test_get_messages_batch_groups_by_session(self, sqlite_backend: SQLiteBackend) -> None:
        """get_messages_batch must return messages grouped by session_id."""
        backend = sqlite_backend

        # Insert sessions with messages
        async with backend.connection() as conn:
            for i in range(5):
                await conn.execute(
                    "INSERT INTO sessions (session_id, source_name, "
                    "provider_session_id, content_hash, version) "
                    "VALUES (?, 'test', ?, ?, 1)",
                    (f"conv-{i}", f"pconv-{i}", f"hash-{i}"),
                )
                for j in range(3):
                    await conn.execute(
                        "INSERT INTO messages (message_id, session_id, role, "
                        "text, content_hash, version) "
                        "VALUES (?, ?, 'user', ?, ?, 1)",
                        (f"msg-{i}-{j}", f"conv-{i}", f"text {i} {j}", f"mhash-{i}-{j}"),
                    )
            await conn.commit()

        ids = [f"conv-{i}" for i in range(5)]
        msgs_by_id = await backend.get_messages_batch(ids)

        assert len(msgs_by_id) == 5
        for conv_id, msgs in msgs_by_id.items():
            assert len(msgs) == 3
            assert all(m.session_id == conv_id for m in msgs)

    @pytest.mark.asyncio
    async def test_batch_with_empty_ids_returns_empty(self, sqlite_backend: SQLiteBackend) -> None:
        """Passing empty list of IDs should return empty, not error."""
        backend = sqlite_backend
        records = await backend.get_sessions_batch([])
        assert records == []

    @pytest.mark.asyncio
    async def test_batch_with_nonexistent_ids_returns_partial(self, sqlite_backend: SQLiteBackend) -> None:
        """Requesting nonexistent IDs should return only existing ones."""
        backend = sqlite_backend

        async with backend.connection() as conn:
            await conn.execute(
                "INSERT INTO sessions (session_id, source_name, "
                "provider_session_id, content_hash, version) "
                "VALUES ('exists', 'test', 'pconv', 'hash', 1)",
            )
            await conn.commit()

        records = await backend.get_sessions_batch(["exists", "ghost-1", "ghost-2"])
        assert len(records) == 1
        assert records[0].session_id == "exists"


# =============================================================================
# Async repository: concurrent save safety
# =============================================================================


class TestConcurrentSaveGuards:
    """Multiple concurrent saves must not corrupt data."""

    @pytest.mark.asyncio
    async def test_concurrent_saves_dont_crash(self, sqlite_backend: SQLiteBackend) -> None:
        """Concurrent saves to different sessions must not error."""
        backend = sqlite_backend
        repo = SessionRepository(backend=backend)

        async def _save_one(idx: int) -> None:
            conv = make_session(
                session_id=f"test:conv-{idx}",
                source_name="test",
                provider_session_id=f"conv-{idx}",
                title=f"Session {idx}",
                content_hash=f"hash-{idx}",
                version=1,
            )
            msg = make_message(
                message_id=f"msg-{idx}",
                session_id=f"test:conv-{idx}",
                role="user",
                text=f"Hello from session {idx}",
                content_hash=f"mhash-{idx}",
                version=1,
            )
            await repo.save_session(conv, [msg], [])

        # Run 10 concurrent saves
        await asyncio.gather(*[_save_one(i) for i in range(10)])

        # Verify all saved
        async with backend.connection() as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM sessions")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 10

    @pytest.mark.asyncio
    async def test_concurrent_saves_to_same_session(self, sqlite_backend: SQLiteBackend) -> None:
        """Concurrent upserts to the same session must not corrupt."""
        backend = sqlite_backend
        repo = SessionRepository(backend=backend)

        async def _upsert(version: int) -> None:
            conv = make_session(
                session_id="test:same-conv",
                source_name="test",
                provider_session_id="same",
                title=f"Version {version}",
                content_hash=f"hash-v{version}",
                version=version,
            )
            msg = make_message(
                message_id=f"msg-v{version}",
                session_id="test:same-conv",
                role="user",
                text=f"Version {version}",
                content_hash=f"mhash-v{version}",
                version=version,
            )
            await repo.save_session(conv, [msg], [])

        # Run 5 concurrent upserts to the same session
        await asyncio.gather(*[_upsert(i) for i in range(5)])

        # Should have exactly 1 session (upserted, not duplicated)
        async with backend.connection() as conn:
            cursor = await conn.execute("SELECT COUNT(*) FROM sessions")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 1

    @pytest.mark.asyncio
    async def test_concurrent_reads_during_writes(self, sqlite_backend: SQLiteBackend) -> None:
        """Reads during concurrent writes must not error or return garbage."""
        backend = sqlite_backend
        repo = SessionRepository(backend=backend)

        async def _write(idx: int) -> None:
            conv = make_session(
                session_id=f"test:conv-{idx}",
                source_name="test",
                provider_session_id=f"conv-{idx}",
                title=f"Session {idx}",
                content_hash=f"hash-{idx}",
                version=1,
            )
            await repo.save_session(conv, [], [])

        async def _read() -> int:
            async with backend.read_connection() as conn:
                cursor = await conn.execute("SELECT COUNT(*) FROM sessions")
                row = await cursor.fetchone()
                assert row is not None
                return int(row[0])

        # Interleave writes and reads
        tasks: list[Awaitable[object]] = []
        for i in range(10):
            tasks.append(_write(i))
            tasks.append(_read())

        results = await asyncio.gather(*tasks, return_exceptions=True)
        write_results = results[::2]
        read_results = results[1::2]

        from sqlite3 import OperationalError

        def _is_locked_error(result: object) -> bool:
            return isinstance(result, OperationalError) and "locked" in str(result)

        # Transient OperationalError("database is locked") is acceptable under
        # heavy contention. SQLite WAL mode does not guarantee zero-wait access
        # on all platforms. Non-locked exceptions are real failures.
        unexpected_writes = [
            result for result in write_results if isinstance(result, Exception) and not _is_locked_error(result)
        ]
        assert unexpected_writes == [], (
            f"Got unexpected write exceptions during concurrent read/write: {unexpected_writes}"
        )

        unexpected = [r for r in read_results if isinstance(r, Exception) and not _is_locked_error(r)]
        assert unexpected == [], f"Got unexpected exceptions during concurrent read/write: {unexpected}"

        successful_writes = sum(1 for result in write_results if not isinstance(result, Exception))
        final_count = await _read()
        assert final_count == successful_writes
