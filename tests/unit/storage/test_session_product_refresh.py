from __future__ import annotations

import pytest

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.session_product_refresh import (
    _apply_session_product_conversation_updates_async,
)
from tests.infra.storage_records import make_conversation, make_message, store_records


@pytest.mark.asyncio
async def test_apply_session_product_conversation_updates_async_batches_hydrated_conversations(
    tmp_path,
) -> None:
    db_path = tmp_path / "refresh.db"
    with open_connection(db_path) as conn:
        store_records(
            conversation=make_conversation("conv-refresh", title="Refresh Test"),
            messages=[
                make_message("conv-refresh:msg-1", "conv-refresh", text="Need help with batching"),
                make_message(
                    "conv-refresh:msg-2",
                    "conv-refresh",
                    role="assistant",
                    text="Let's batch the refresh path.",
                ),
            ],
            attachments=[],
            conn=conn,
        )
        conn.commit()

    backend = SQLiteBackend(db_path=db_path)
    async with backend.connection() as conn:
        update = await _apply_session_product_conversation_updates_async(
            conn,
            ["conv-refresh"],
            transaction_depth=1,
            page_size=10,
        )

    assert update.counts["profiles"] == 1
    assert update.thread_root_ids == {"conv-refresh"}
    assert update.affected_groups
    assert len(update.chunk_observations) == 1
    assert float(update.chunk_observations[0]["load_ms"]) >= 0.0
    assert float(update.chunk_observations[0]["hydrate_ms"]) >= 0.0
    assert float(update.chunk_observations[0]["build_ms"]) >= 0.0
    assert float(update.chunk_observations[0]["write_ms"]) >= 0.0


@pytest.mark.asyncio
async def test_apply_session_product_conversation_updates_async_preserves_thread_roots_for_children(
    tmp_path,
) -> None:
    db_path = tmp_path / "refresh-thread.db"
    with open_connection(db_path) as conn:
        store_records(
            conversation=make_conversation("conv-root", title="Root"),
            messages=[make_message("conv-root:msg-1", "conv-root", text="Root message")],
            attachments=[],
            conn=conn,
        )
        store_records(
            conversation=make_conversation(
                "conv-child",
                title="Child",
                parent_conversation_id="conv-root",
            ),
            messages=[make_message("conv-child:msg-1", "conv-child", text="Child message")],
            attachments=[],
            conn=conn,
        )
        conn.commit()

    backend = SQLiteBackend(db_path=db_path)
    async with backend.connection() as conn:
        update = await _apply_session_product_conversation_updates_async(
            conn,
            ["conv-root", "conv-child"],
            transaction_depth=1,
            page_size=10,
        )

    assert update.counts["profiles"] == 2
    assert update.thread_root_ids == {"conv-root"}
    assert len(update.chunk_observations) == 1


@pytest.mark.asyncio
async def test_apply_session_product_conversation_updates_async_clears_deleted_conversations(
    tmp_path,
) -> None:
    db_path = tmp_path / "refresh-delete.db"
    with open_connection(db_path) as conn:
        store_records(
            conversation=make_conversation("conv-stale", title="Stale"),
            messages=[make_message("conv-stale:msg-1", "conv-stale", text="Before delete")],
            attachments=[],
            conn=conn,
        )
        conn.commit()

    backend = SQLiteBackend(db_path=db_path)
    async with backend.connection() as conn:
        first_update = await _apply_session_product_conversation_updates_async(
            conn,
            ["conv-stale"],
            transaction_depth=1,
            page_size=10,
        )
        await conn.commit()

    assert first_update.counts["profiles"] == 1

    with open_connection(db_path) as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", ("conv-stale",))
        conn.execute("DELETE FROM conversations WHERE conversation_id = ?", ("conv-stale",))
        conn.commit()

    async with backend.connection() as conn:
        second_update = await _apply_session_product_conversation_updates_async(
            conn,
            ["conv-stale"],
            transaction_depth=1,
            page_size=10,
        )
        await conn.commit()

    assert second_update.counts["profiles"] == 0
    assert len(second_update.chunk_observations) == 1

    with open_connection(db_path) as conn:
        profile_count = conn.execute(
            "SELECT COUNT(*) FROM session_profiles WHERE conversation_id = ?",
            ("conv-stale",),
        ).fetchone()[0]
        work_event_count = conn.execute(
            "SELECT COUNT(*) FROM session_work_events WHERE conversation_id = ?",
            ("conv-stale",),
        ).fetchone()[0]
        phase_count = conn.execute(
            "SELECT COUNT(*) FROM session_phases WHERE conversation_id = ?",
            ("conv-stale",),
        ).fetchone()[0]

    assert profile_count == 0
    assert work_event_count == 0
    assert phase_count == 0
