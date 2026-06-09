"""Native convergence laws: ingest → steady state, delete → clean, re-ingest → idempotent.

The legacy single-file store needed an explicit repair pass to reconcile orphan
messages and empty sessions (artifacts of a schema where messages and
sessions were independently writable). The archive makes
those debt classes structurally impossible: messages and blocks are owned by a
session and cascade-delete with it, and re-ingest is content-hash idempotent.

These tests pin the resulting convergence invariants directly against the
archive store and the async facade — there is no debt to repair because the
write path never produces it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.api import Polylogue
from tests.infra.storage_records import SessionBuilder, db_setup


def _archive_counts(db_path: Path) -> tuple[int, int, int]:
    """Return (sessions, messages, blocks) row counts from the index.db."""
    with sqlite3.connect(db_path) as conn:
        return (
            int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]),
            int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]),
            int(conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]),
        )


def _orphan_message_count(db_path: Path) -> int:
    """Messages whose owning session row is absent (must always be zero)."""
    with sqlite3.connect(db_path) as conn:
        return int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM messages m
                WHERE NOT EXISTS (
                    SELECT 1 FROM sessions s WHERE s.origin || ':' || s.native_id = m.session_id
                )
                """
            ).fetchone()[0]
        )


def _empty_session_count(db_path: Path) -> int:
    """Sessions with zero messages (the archive write path never produces these)."""
    with sqlite3.connect(db_path) as conn:
        return int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM sessions s
                WHERE NOT EXISTS (
                    SELECT 1 FROM messages m WHERE m.session_id = s.origin || ':' || s.native_id
                )
                """
            ).fetchone()[0]
        )


def _seed_healthy(workspace_env: dict[str, Path]) -> Path:
    db_path = db_setup(workspace_env)
    SessionBuilder(db_path, "healthy-1").provider("chatgpt").title("Healthy").add_message(
        role="user", text="A valid message"
    ).add_message(role="assistant", text="A valid reply").save()
    SessionBuilder(db_path, "healthy-2").provider("claude-code").title("Also healthy").add_message(
        role="user", text="Another message"
    ).save()
    return db_path


class TestNativeStoreHasNoOrphanDebt:
    """The archive store cannot hold orphaned messages or empty sessions."""

    def test_freshly_ingested_archive_has_no_orphan_or_empty_debt(self, workspace_env: dict[str, Path]) -> None:
        db_path = _seed_healthy(workspace_env)
        sessions, messages, blocks = _archive_counts(db_path)
        assert sessions == 2
        assert messages == 3
        assert blocks == 3
        assert _orphan_message_count(db_path) == 0
        assert _empty_session_count(db_path) == 0


class TestDeleteCascadeConvergence:
    """Deleting a session converges its owned rows to zero (no orphans)."""

    @pytest.mark.asyncio
    async def test_delete_cascades_messages_and_blocks(self, workspace_env: dict[str, Path]) -> None:
        db_path = db_setup(workspace_env)
        builder = (
            SessionBuilder(db_path, "to-delete")
            .provider("chatgpt")
            .add_message(role="user", text="first")
            .add_message(role="assistant", text="second")
        )
        builder.save()
        session_id = builder.native_session_id()

        assert _archive_counts(db_path) == (1, 2, 2)

        async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
            assert await poly.delete_session(session_id) is True

        # Cascade leaves no rows and, critically, no orphaned messages/blocks.
        assert _archive_counts(db_path) == (0, 0, 0)
        assert _orphan_message_count(db_path) == 0

    @pytest.mark.asyncio
    async def test_delete_leaves_neighbors_intact(self, workspace_env: dict[str, Path]) -> None:
        db_path = _seed_healthy(workspace_env)
        target = SessionBuilder(db_path, "transient").provider("codex").add_message(role="user", text="ephemeral")
        target.save()
        target_id = target.native_session_id()

        async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
            assert await poly.delete_session(target_id) is True
            remaining = {str(c.id) for c in await poly.list_sessions(limit=100)}

        assert target_id not in remaining
        assert len(remaining) == 2
        assert _orphan_message_count(db_path) == 0
        assert _empty_session_count(db_path) == 0


class TestReingestIdempotencyConvergence:
    """Re-ingesting identical content converges to the same steady state."""

    def test_reingest_does_not_grow_the_archive(self, workspace_env: dict[str, Path]) -> None:
        db_path = db_setup(workspace_env)
        builder = (
            SessionBuilder(db_path, "stable")
            .provider("chatgpt")
            .add_message(role="user", text="hi")
            .add_message(role="assistant", text="yo")
        )
        builder.save()
        before = _archive_counts(db_path)

        # Re-ingest the identical session multiple times.
        builder.save()
        builder.save()
        after = _archive_counts(db_path)

        assert after == before == (1, 2, 2)
        assert _orphan_message_count(db_path) == 0
        assert _empty_session_count(db_path) == 0

    @pytest.mark.asyncio
    async def test_reingest_then_facade_sees_single_session(self, workspace_env: dict[str, Path]) -> None:
        db_path = db_setup(workspace_env)
        builder = SessionBuilder(db_path, "stable").provider("claude-code").add_message(role="user", text="hello")
        builder.save()
        builder.save()
        session_id = builder.native_session_id()

        async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
            convos = await poly.list_sessions(limit=100)
            assert [str(c.id) for c in convos] == [session_id]
            session = await poly.get_session(session_id)
            assert session is not None
            assert len(session.messages) == 1


class TestDeleteThenReingestConvergence:
    """delete → re-ingest reaches the same healthy state (deterministic id)."""

    @pytest.mark.asyncio
    async def test_delete_then_reingest_restores_single_session(self, workspace_env: dict[str, Path]) -> None:
        db_path = db_setup(workspace_env)
        builder = (
            SessionBuilder(db_path, "cycle")
            .provider("chatgpt")
            .add_message(role="user", text="q")
            .add_message(role="assistant", text="a")
        )
        builder.save()
        session_id = builder.native_session_id()

        async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
            assert await poly.delete_session(session_id) is True

        assert _archive_counts(db_path) == (0, 0, 0)

        # Re-ingest the same logical session; the deterministic id returns.
        SessionBuilder(db_path, "cycle").provider("chatgpt").add_message(role="user", text="q").add_message(
            role="assistant", text="a"
        ).save()

        assert _archive_counts(db_path) == (1, 2, 2)
        assert _orphan_message_count(db_path) == 0

        async with Polylogue(db_path=db_path, archive_root=workspace_env["archive_root"]) as poly:
            convos = await poly.list_sessions(limit=100)
            assert [str(c.id) for c in convos] == [session_id]
