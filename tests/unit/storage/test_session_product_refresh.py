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
