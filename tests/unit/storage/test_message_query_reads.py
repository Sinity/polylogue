from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.timestamps import parse_timestamp
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.queries.message_query_reads import (
    get_messages,
    get_messages_batch,
    get_messages_paginated,
    iter_messages,
)
from tests.infra.storage_records import make_message, make_session, save_session_to_archive


@pytest.mark.asyncio
async def test_message_query_reads_cover_type_filters_batches_and_stream_limits(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    backend = SQLiteBackend(db_path=tmp_path / "index.db")
    current_session_id = "unknown-export:conv-message-reads"
    expected_message_ids = [
        f"{current_session_id}:msg-summary",
        f"{current_session_id}:msg-summary-2",
        f"{current_session_id}:msg-tool",
        f"{current_session_id}:msg-user",
        f"{current_session_id}:msg-assistant",
    ]
    conv = make_session("conv-message-reads", title="Message Reads")
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

    await save_session_to_archive(backend, session=conv, messages=messages)

    async with backend.connection() as conn:
        assert await get_messages_batch(conn, []) == ({}, [])

        by_session, all_messages = await get_messages_batch(conn, [current_session_id, "missing"])
        assert [message.message_id for message in by_session[current_session_id]] == expected_message_ids

        since = parse_timestamp("2026-01-01T00:00:03Z")
        assert since is not None
        filtered_by_session, filtered_messages = await get_messages_batch(
            conn,
            [current_session_id, "missing"],
            sort_key_since=since.timestamp(),
            message_role=(Role.USER,),
        )
        assert [message.message_id for message in filtered_by_session[current_session_id]] == [
            f"{current_session_id}:msg-user"
        ]
        assert [message.message_id for message in filtered_messages] == [f"{current_session_id}:msg-user"]
        assert [message.message_id for message in all_messages] == expected_message_ids

        paginated, total = await get_messages_paginated(
            conn,
            current_session_id,
            message_type="summary",
            limit=1,
            offset=0,
        )
        assert total == 2
        assert [message.message_id for message in paginated] == [f"{current_session_id}:msg-summary"]

        paginated_with_offset, offset_total = await get_messages_paginated(
            conn,
            current_session_id,
            message_type="summary",
            limit=1,
            offset=1,
        )
        assert offset_total == 2
        assert [message.message_id for message in paginated_with_offset] == [f"{current_session_id}:msg-summary-2"]

        tool_messages, tool_total = await get_messages_paginated(
            conn,
            current_session_id,
            message_type="tool_result",
            limit=10,
            offset=0,
        )
        assert tool_total == 1
        assert [message.message_id for message in tool_messages] == [f"{current_session_id}:msg-tool"]

        user_messages, user_total = await get_messages_paginated(
            conn,
            current_session_id,
            message_type="message",
            limit=10,
            offset=0,
        )
        assert user_total == 2
        assert [message.message_id for message in user_messages] == [
            f"{current_session_id}:msg-user",
            f"{current_session_id}:msg-assistant",
        ]

        with pytest.raises(ValueError, match="Unknown message type"):
            await get_messages_paginated(
                conn,
                current_session_id,
                message_type="summmary",  # type: ignore[arg-type]
                limit=10,
                offset=0,
            )

        hydrated = await get_messages(conn, current_session_id)
        assert len(hydrated) == 5

        assert [message async for message in iter_messages(conn, current_session_id, limit=0)] == []
        assert [message.message_id async for message in iter_messages(conn, "missing")] == []
        assert [
            message.message_id
            async for message in iter_messages(conn, current_session_id, dialogue_only=True, chunk_size=1, limit=1)
        ] == [f"{current_session_id}:msg-user"]

    await backend.close()
