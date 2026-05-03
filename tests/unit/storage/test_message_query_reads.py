from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.queries.message_query_reads import (
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
            timestamp="2026-01-01T00:00:00Z",
            message_type="summary",
            content_blocks=[{"type": "text", "text": "summary"}],
        ),
        make_message(
            "msg-summary-2",
            "conv-message-reads",
            role="system",
            text="summary two",
            timestamp="2026-01-01T00:00:01Z",
            message_type="summary",
            content_blocks=[{"type": "text", "text": "summary two"}],
        ),
        make_message(
            "msg-tool",
            "conv-message-reads",
            role="tool",
            text="tool result",
            timestamp="2026-01-01T00:00:02Z",
            message_type="tool_result",
            content_blocks=[{"type": "tool_result", "text": "tool result"}],
        ),
        make_message(
            "msg-user",
            "conv-message-reads",
            role="user",
            text="user",
            timestamp="2026-01-01T00:00:03Z",
            message_type="message",
        ),
        make_message(
            "msg-assistant",
            "conv-message-reads",
            role="assistant",
            text="assistant",
            timestamp="2026-01-01T00:00:04Z",
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
                ("blk-summary-2", "msg-summary-2", "conv-message-reads", 0, "text", "summary two"),
                ("blk-tool", "msg-tool", "conv-message-reads", 0, "tool_result", "tool result"),
            ],
        )
        await conn.commit()

        assert await get_messages_batch(conn, []) == ({}, [])

        by_conversation, all_messages = await get_messages_batch(conn, ["conv-message-reads", "missing"])
        assert [message.message_id for message in by_conversation["conv-message-reads"]] == [
            "msg-summary",
            "msg-summary-2",
            "msg-tool",
            "msg-user",
            "msg-assistant",
        ]
        assert [message.message_id for message in all_messages] == [
            "msg-summary",
            "msg-summary-2",
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
        assert total == 2
        assert [message.message_id for message in paginated] == ["msg-summary"]

        paginated_with_offset, offset_total = await get_messages_paginated(
            conn,
            "conv-message-reads",
            message_type="summary",
            limit=1,
            offset=1,
        )
        assert offset_total == 2
        assert [message.message_id for message in paginated_with_offset] == ["msg-summary-2"]

        tool_messages, tool_total = await get_messages_paginated(
            conn,
            "conv-message-reads",
            message_type="tool_result",
            limit=10,
            offset=0,
        )
        assert tool_total == 1
        assert [message.message_id for message in tool_messages] == ["msg-tool"]

        user_messages, user_total = await get_messages_paginated(
            conn,
            "conv-message-reads",
            message_type="message",
            limit=10,
            offset=0,
        )
        assert user_total == 2
        assert [message.message_id for message in user_messages] == ["msg-user", "msg-assistant"]

        with pytest.raises(ValueError, match="Unknown message type"):
            await get_messages_paginated(
                conn,
                "conv-message-reads",
                message_type="summmary",  # type: ignore[arg-type]
                limit=10,
                offset=0,
            )

        hydrated = await get_messages(conn, "conv-message-reads")
        assert len(hydrated) == 5

        assert [message async for message in iter_messages(conn, "conv-message-reads", limit=0)] == []
        assert [message.message_id async for message in iter_messages(conn, "missing")] == []
        assert [
            message.message_id
            async for message in iter_messages(conn, "conv-message-reads", dialogue_only=True, chunk_size=1, limit=1)
        ] == ["msg-user"]

    await backend.close()
