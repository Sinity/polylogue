from __future__ import annotations

import pytest

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.session_product_aggregates import refresh_async_provider_day_aggregates
from polylogue.storage.session_product_refresh import (
    _apply_session_product_conversation_updates_async,
    _refresh_thread_roots_async,
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
async def test_apply_session_product_conversation_updates_async_uses_small_default_chunks(
    tmp_path,
) -> None:
    db_path = tmp_path / "refresh-default-chunks.db"
    conversation_ids: list[str] = []
    with open_connection(db_path) as conn:
        for index in range(11):
            conversation_id = f"conv-{index:02d}"
            conversation_ids.append(conversation_id)
            store_records(
                conversation=make_conversation(conversation_id, title=f"Conversation {index:02d}"),
                messages=[
                    make_message(
                        f"{conversation_id}:msg-1",
                        conversation_id,
                        text=f"Message for {conversation_id}",
                    )
                ],
                attachments=[],
                conn=conn,
            )
        conn.commit()

    backend = SQLiteBackend(db_path=db_path)
    async with backend.connection() as conn:
        update = await _apply_session_product_conversation_updates_async(
            conn,
            conversation_ids,
            transaction_depth=1,
        )

    assert update.counts["profiles"] == 11
    assert len(update.chunk_observations) == 2
    assert update.chunk_observations[0]["conversation_count"] == 10
    assert update.chunk_observations[1]["conversation_count"] == 1


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


@pytest.mark.asyncio
async def test_refresh_thread_roots_async_batches_root_rebuilds(
    tmp_path,
) -> None:
    db_path = tmp_path / "refresh-thread-roots.db"
    with open_connection(db_path) as conn:
        store_records(
            conversation=make_conversation("conv-root-a", title="Root A"),
            messages=[make_message("conv-root-a:msg-1", "conv-root-a", text="Root A message")],
            attachments=[],
            conn=conn,
        )
        store_records(
            conversation=make_conversation(
                "conv-child-a",
                title="Child A",
                parent_conversation_id="conv-root-a",
            ),
            messages=[make_message("conv-child-a:msg-1", "conv-child-a", text="Child A message")],
            attachments=[],
            conn=conn,
        )
        store_records(
            conversation=make_conversation("conv-root-b", title="Root B"),
            messages=[make_message("conv-root-b:msg-1", "conv-root-b", text="Root B message")],
            attachments=[],
            conn=conn,
        )
        store_records(
            conversation=make_conversation(
                "conv-child-b",
                title="Child B",
                parent_conversation_id="conv-root-b",
            ),
            messages=[make_message("conv-child-b:msg-1", "conv-child-b", text="Child B message")],
            attachments=[],
            conn=conn,
        )
        conn.commit()

    backend = SQLiteBackend(db_path=db_path)
    async with backend.connection() as conn:
        update = await _apply_session_product_conversation_updates_async(
            conn,
            ["conv-root-a", "conv-child-a", "conv-root-b", "conv-child-b"],
            transaction_depth=1,
            page_size=10,
        )
        refreshed = await _refresh_thread_roots_async(
            conn,
            sorted(update.thread_root_ids),
            transaction_depth=1,
        )
        await conn.commit()

    assert refreshed == 2

    with open_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT thread_id, root_id, session_count
            FROM work_threads
            ORDER BY thread_id
            """
        ).fetchall()

    assert [(row["thread_id"], row["root_id"], row["session_count"]) for row in rows] == [
        ("conv-root-a", "conv-root-a", 2),
        ("conv-root-b", "conv-root-b", 2),
    ]


@pytest.mark.asyncio
async def test_refresh_async_provider_day_aggregates_batches_multiple_groups(
    tmp_path,
) -> None:
    db_path = tmp_path / "refresh-provider-day-groups.db"
    with open_connection(db_path) as conn:
        store_records(
            conversation=make_conversation(
                "conv-chatgpt-a",
                provider_name="chatgpt",
                title="ChatGPT A",
                created_at="2026-04-02T10:00:00+00:00",
                updated_at="2026-04-02T10:05:00+00:00",
            ),
            messages=[
                make_message(
                    "conv-chatgpt-a:msg-1",
                    "conv-chatgpt-a",
                    text="ChatGPT A message",
                    timestamp="2026-04-02T10:00:00+00:00",
                )
            ],
            attachments=[],
            conn=conn,
        )
        store_records(
            conversation=make_conversation(
                "conv-chatgpt-b",
                provider_name="chatgpt",
                title="ChatGPT B",
                created_at="2026-04-02T11:00:00+00:00",
                updated_at="2026-04-02T11:05:00+00:00",
            ),
            messages=[
                make_message(
                    "conv-chatgpt-b:msg-1",
                    "conv-chatgpt-b",
                    text="ChatGPT B message",
                    timestamp="2026-04-02T11:00:00+00:00",
                )
            ],
            attachments=[],
            conn=conn,
        )
        store_records(
            conversation=make_conversation(
                "conv-claude-a",
                provider_name="claude-ai",
                title="Claude A",
                created_at="2026-04-03T09:00:00+00:00",
                updated_at="2026-04-03T09:05:00+00:00",
            ),
            messages=[
                make_message(
                    "conv-claude-a:msg-1",
                    "conv-claude-a",
                    text="Claude A message",
                    timestamp="2026-04-03T09:00:00+00:00",
                )
            ],
            attachments=[],
            conn=conn,
        )
        conn.commit()

    backend = SQLiteBackend(db_path=db_path)
    async with backend.connection() as conn:
        update = await _apply_session_product_conversation_updates_async(
            conn,
            ["conv-chatgpt-a", "conv-chatgpt-b", "conv-claude-a"],
            transaction_depth=1,
            page_size=10,
        )
        await refresh_async_provider_day_aggregates(
            conn,
            update.affected_groups,
            transaction_depth=1,
        )
        await conn.commit()

    with open_connection(db_path) as conn:
        day_rows = conn.execute(
            """
            SELECT provider_name, day, conversation_count
            FROM day_session_summaries
            ORDER BY provider_name, day
            """
        ).fetchall()

    assert [(row["provider_name"], row["day"], row["conversation_count"]) for row in day_rows] == [
        ("chatgpt", "2026-04-02", 2),
        ("claude-ai", "2026-04-03", 1),
    ]
