from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.queries.message_query_reads import (
    _post_filter_by_message_type,
    get_messages,
    get_messages_batch,
    get_messages_paginated,
    iter_messages,
)
from tests.infra.storage_records import make_conversation, make_message


@pytest.mark.asyncio
async def test_message_query_reads_cover_type_filters_batches_and_stream_limits(tmp_path: Path) -> None:
    backend = SQLiteBackend(db_path=tmp_path / "messages.db")
    conv = make_conversation("conv-message-reads", title="Message Reads")
    messages = [
        make_message(
            "msg-summary",
            "conv-message-reads",
            role="system",
            text="summary",
            message_type="summary",
            content_blocks=[{"type": "text", "text": "summary"}],
        ),
        make_message(
            "msg-tool",
            "conv-message-reads",
            role="tool",
            text="tool result",
            message_type="tool_result",
            content_blocks=[{"type": "tool_result", "text": "tool result"}],
        ),
        make_message(
            "msg-user",
            "conv-message-reads",
            role="user",
            text="user",
            message_type="message",
        ),
        make_message(
            "msg-assistant",
            "conv-message-reads",
            role="assistant",
            text="assistant",
            message_type="message",
        ),
    ]

    async with backend.transaction():
        await backend.save_conversation_record(conv)
        await backend.save_messages(messages)

    async with backend.connection() as conn:
        await conn.executemany(
            """
            INSERT OR REPLACE INTO content_blocks
            (block_id, message_id, conversation_id, block_index, type, text)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                ("blk-summary", "msg-summary", "conv-message-reads", 0, "text", "summary"),
                ("blk-tool", "msg-tool", "conv-message-reads", 0, "tool_result", "tool result"),
            ],
        )
        await conn.commit()

        assert await get_messages_batch(conn, []) == ({}, [])

        by_conversation, all_messages = await get_messages_batch(conn, ["conv-message-reads", "missing"])
        assert [message.message_id for message in by_conversation["conv-message-reads"]] == [
            "msg-summary",
            "msg-tool",
            "msg-user",
            "msg-assistant",
        ]
        assert [message.message_id for message in all_messages] == [
            "msg-summary",
            "msg-tool",
            "msg-user",
            "msg-assistant",
        ]

        paginated, total = await get_messages_paginated(
            conn,
            "conv-message-reads",
            message_type="summary",
            limit=1,
            offset=0,
        )
        assert total == 1
        assert [message.message_id for message in paginated] == ["msg-summary"]

        hydrated = await get_messages(conn, "conv-message-reads")
        assert [
            message.message_id for message in await _post_filter_by_message_type(conn, hydrated, "tool_result")
        ] == ["msg-tool"]
        assert [message.message_id for message in await _post_filter_by_message_type(conn, hydrated, "summary")] == [
            "msg-summary"
        ]
        assert await _post_filter_by_message_type(conn, hydrated, "thinking") == hydrated

        assert [message async for message in iter_messages(conn, "conv-message-reads", limit=0)] == []
        assert [message.message_id async for message in iter_messages(conn, "missing")] == []
        assert [
            message.message_id
            async for message in iter_messages(conn, "conv-message-reads", dialogue_only=True, chunk_size=1, limit=1)
        ] == ["msg-user"]

    await backend.close()
