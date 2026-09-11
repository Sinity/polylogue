from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import MaterialOrigin
from polylogue.core.timestamps import parse_timestamp
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.queries.message_query_reads import (
    get_message_edge_windows,
    get_messages,
    get_messages_batch,
    get_messages_paginated,
    iter_messages,
)
from tests.infra.identity import archive_message_id
from tests.infra.storage_records import make_message, make_session, save_session_to_archive


@pytest.mark.asyncio
async def test_message_query_reads_cover_type_filters_batches_and_stream_limits(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    backend = SQLiteBackend(db_path=tmp_path / "index.db")
    current_session_id = "unknown-export:conv-message-reads"
    expected_message_ids = [
        archive_message_id(current_session_id, "msg-summary", position=0),
        archive_message_id(current_session_id, "msg-summary-2", position=1),
        archive_message_id(current_session_id, "msg-tool", position=2),
        archive_message_id(current_session_id, "msg-user", position=3),
        archive_message_id(current_session_id, "msg-protocol", position=4),
        archive_message_id(current_session_id, "msg-assistant", position=5),
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
            blocks=[{"type": "text", "text": "summary"}],
        ),
        make_message(
            "msg-summary-2",
            "conv-message-reads",
            role="system",
            text="summary two",
            timestamp="2026-01-01T00:00:01Z",
            message_type="summary",
            blocks=[{"type": "text", "text": "summary two"}],
        ),
        make_message(
            "msg-tool",
            "conv-message-reads",
            role="tool",
            text="tool result",
            timestamp="2026-01-01T00:00:02Z",
            message_type="tool_result",
            blocks=[{"type": "tool_result", "text": "tool result", "is_error": False}],
        ),
        make_message(
            "msg-user",
            "conv-message-reads",
            role="user",
            text="user",
            timestamp="2026-01-01T00:00:03Z",
            message_type="message",
            material_origin="human_authored",
        ),
        make_message(
            "msg-protocol",
            "conv-message-reads",
            role="assistant",
            text='{"queries":["find context"]}',
            timestamp="2026-01-01T00:00:03.500000Z",
            message_type="message",
            material_origin="runtime_protocol",
        ),
        make_message(
            "msg-assistant",
            "conv-message-reads",
            role="assistant",
            text="assistant",
            timestamp="2026-01-01T00:00:04Z",
            message_type="message",
            material_origin="assistant_authored",
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
            archive_message_id(current_session_id, "msg-user", position=3)
        ]
        assert [message.message_id for message in filtered_messages] == [
            archive_message_id(current_session_id, "msg-user", position=3)
        ]
        assert [message.message_id for message in all_messages] == expected_message_ids

        traced_sql: list[str] = []
        await conn.set_trace_callback(traced_sql.append)
        try:
            first_edge, last_edge, edge_total = await get_message_edge_windows(
                conn,
                current_session_id,
                message_role=(Role.USER, Role.ASSISTANT),
                message_type="message",
                edge_limit=1,
            )
        finally:
            await conn.set_trace_callback(lambda _statement: None)
        assert edge_total == 2
        assert [message.message_id for message in first_edge] == [
            archive_message_id(current_session_id, "msg-user", position=3)
        ]
        assert [message.message_id for message in last_edge] == [
            archive_message_id(current_session_id, "msg-assistant", position=5)
        ]
        assert any("COUNT(*) FROM messages INDEXED BY idx_messages_session_position" in sql for sql in traced_sql)

        authored_first, authored_last, authored_total = await get_message_edge_windows(
            conn,
            current_session_id,
            message_role=(Role.USER, Role.ASSISTANT),
            message_type="message",
            material_origin=(MaterialOrigin.HUMAN_AUTHORED, MaterialOrigin.ASSISTANT_AUTHORED),
            edge_limit=2,
        )
        assert authored_total == 2
        assert [message.message_id for message in authored_first] == [
            archive_message_id(current_session_id, "msg-user", position=3),
            archive_message_id(current_session_id, "msg-assistant", position=5),
        ]
        assert authored_last == []

        paginated, total, paginated_completeness = await get_messages_paginated(
            conn,
            current_session_id,
            message_type="summary",
            limit=1,
            offset=0,
        )
        assert total == 2
        assert [message.message_id for message in paginated] == [
            archive_message_id(current_session_id, "msg-summary", position=0)
        ]
        assert paginated_completeness.complete is True

        paginated_with_offset, offset_total, _offset_completeness = await get_messages_paginated(
            conn,
            current_session_id,
            message_type="summary",
            limit=1,
            offset=1,
        )
        assert offset_total == 2
        assert [message.message_id for message in paginated_with_offset] == [
            archive_message_id(current_session_id, "msg-summary-2", position=1)
        ]

        tool_messages, tool_total, _tool_completeness = await get_messages_paginated(
            conn,
            current_session_id,
            message_type="tool_result",
            limit=10,
            offset=0,
        )
        assert tool_total == 1
        assert [message.message_id for message in tool_messages] == [
            archive_message_id(current_session_id, "msg-tool", position=2)
        ]

        user_messages, user_total, _user_completeness = await get_messages_paginated(
            conn,
            current_session_id,
            message_type="message",
            limit=10,
            offset=0,
        )
        assert user_total == 2
        assert [message.message_id for message in user_messages] == [
            archive_message_id(current_session_id, "msg-user", position=3),
            archive_message_id(current_session_id, "msg-assistant", position=5),
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
        assert len(hydrated) == 6

        assert [message async for message in iter_messages(conn, current_session_id, limit=0)] == []
        assert [message.message_id async for message in iter_messages(conn, "missing")] == []
        assert [
            message.message_id
            async for message in iter_messages(
                conn,
                current_session_id,
                message_roles=(Role.USER, Role.ASSISTANT),
                chunk_size=1,
                limit=1,
            )
        ] == [archive_message_id(current_session_id, "msg-user", position=3)]

    await backend.close()


@pytest.mark.asyncio
async def test_transcript_read_routes_agree_on_one_order(tmp_path: Path) -> None:
    """Every route that states a session's message order returns one sequence.

    The fixture's clock runs backwards against its content positions, which is
    ordinary: 719 sessions across all seven origins are non-monotonic this way.

    Anti-vacuity: ordering any single route by `(occurred_at_ms IS NULL),
    occurred_at_ms, message_id` -- what ``iter_messages``,
    ``get_messages_batch`` and the markdown export each did -- reverses that
    route alone and turns the comparison red.
    """
    initialize_active_archive_root(tmp_path)
    backend = SQLiteBackend(db_path=tmp_path / "index.db")
    session_id = "unknown-export:conv-order-parity"
    conv = make_session("conv-order-parity", title="Order Parity")
    # position N carries timestamp 00:00:(9-N): content order and clock order are
    # exact reverses of one another.
    messages = [
        make_message(
            f"msg-{position}",
            "conv-order-parity",
            role="user" if position % 2 == 0 else "assistant",
            text=f"body {position}",
            timestamp=f"2026-01-01T00:00:{9 - position:02d}Z",
            blocks=[{"type": "text", "text": f"body {position}"}],
        )
        for position in range(6)
    ]
    await save_session_to_archive(backend, session=conv, messages=messages)

    expected = [archive_message_id(session_id, f"msg-{position}", position=position) for position in range(6)]

    async with backend.connection() as conn:
        composed = [message.message_id for message in await get_messages(conn, session_id)]
        assert composed == expected

        paginated, total, _completeness = await get_messages_paginated(conn, session_id, limit=100, offset=0)
        assert [message.message_id for message in paginated] == expected
        assert total == len(expected)

        for chunk_size in (1, 2, 3, 5, 6, 100):
            streamed = [message.message_id async for message in iter_messages(conn, session_id, chunk_size=chunk_size)]
            assert streamed == expected, f"chunk_size={chunk_size}"

        batched, _all_messages = await get_messages_batch(conn, [session_id])
        assert [message.message_id for message in batched[session_id]] == expected

        first, last, edge_total = await get_message_edge_windows(conn, session_id, edge_limit=2)
        assert [message.message_id for message in first] == expected[:2]
        assert [message.message_id for message in last] == expected[-2:]
        assert edge_total == len(expected)

        # `polylogue read --format ndjson` takes its total from
        # get_messages_paginated and then slices the iter_messages stream by
        # offset. The two must page the same sequence.
        for offset in range(len(expected) + 1):
            page, page_total, _ = await get_messages_paginated(conn, session_id, limit=2, offset=offset)
            streamed_page = [message.message_id async for message in iter_messages(conn, session_id, limit=offset + 2)][
                offset : offset + 2
            ]
            assert [message.message_id for message in page] == streamed_page, f"offset={offset}"
            assert page_total == total

    await backend.close()


@pytest.mark.asyncio
async def test_transcript_reads_use_position_index_without_temp_btree(tmp_path: Path) -> None:
    """`idx_messages_session_position` must satisfy the `position, variant_index`
    ordering every transcript read uses, so the keyset stream does not fall back
    to a per-session temp B-tree sort on every chunk (#2475)."""
    import sqlite3

    initialize_active_archive_root(tmp_path)
    backend = SQLiteBackend(db_path=tmp_path / "index.db")
    conv = make_session("conv-eqp", title="EQP")
    messages = [
        make_message(f"msg-{i}", "conv-eqp", role="user", text=f"t{i}", timestamp=f"2026-01-01T00:00:{i:02d}Z")
        for i in range(20)
    ]
    await save_session_to_archive(backend, session=conv, messages=messages)
    await backend.close()

    conn = sqlite3.connect(tmp_path / "index.db")
    try:
        sid = conn.execute("SELECT session_id FROM sessions LIMIT 1").fetchone()[0]
        query = (
            "SELECT m.message_id FROM messages m JOIN sessions s ON s.session_id = m.session_id "
            "WHERE m.session_id = ? "
            "AND (m.position > ? OR (m.position = ? AND m.variant_index > ?)) "
            "ORDER BY m.position, m.variant_index LIMIT 50"
        )
        plan_rows = conn.execute(f"EXPLAIN QUERY PLAN {query}", (sid, 0, 0, 0)).fetchall()
        plan = " | ".join(str(r[3]) for r in plan_rows)
    finally:
        conn.close()

    assert "TEMP B-TREE" not in plan.upper(), f"unexpected temp sort in plan: {plan}"
    assert "idx_messages_session_position" in plan, f"position index not used: {plan}"
