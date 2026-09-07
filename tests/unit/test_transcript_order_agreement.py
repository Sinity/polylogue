"""Every read surface states one transcript order for one session.

A session's transcript order is content position. The fixture here carries a
clock that runs exactly backwards against its positions -- the ordinary shape,
not a pathology -- so any route that keys on ``occurred_at_ms`` returns the
reverse of every other route and the comparison below goes red.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from polylogue import Polylogue
from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.cli.read_views.streaming_markdown import stream_exact_session_markdown
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from polylogue.storage.sqlite.queries.message_query_reads import (
    get_message_edge_windows,
    get_messages,
    get_messages_batch,
    get_messages_paginated,
    iter_messages,
)
from tests.infra.identity import archive_message_id
from tests.infra.live_ingest import write_index_session
from tests.infra.mcp import ALL_CAPABILITIES, MCPServerUnderTest, invoke_surface_async

_PARENT_NATIVE_ID = "order-parity"
_CHILD_NATIVE_ID = "order-parity-fork"
_PARENT_LENGTH = 6
_BRANCH_LENGTH = 4
_TAIL_LENGTH = 2


@pytest.fixture
def mcp_server() -> MCPServerUnderTest:
    from polylogue.mcp.server import build_server

    return cast(MCPServerUnderTest, build_server(capabilities=ALL_CAPABILITIES))


def _body(position: int) -> str:
    return f"body {position}"


def _message(native_id: str, position: int, text: str) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=native_id,
        role=Role.USER if position % 2 == 0 else Role.ASSISTANT,
        text=text,
        position=position,
        variant_index=0,
        is_active_path=True,
        is_active_leaf=False,
        # Position N carries second 9-N: content order and clock order are exact
        # reverses of one another.
        timestamp=f"2026-01-01T00:00:{9 - position:02d}Z",
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )


def _seed(root: Path) -> tuple[str, str]:
    """Write an adversarially clocked session and a prefix-sharing fork of it."""
    with ArchiveStore(root) as store:
        parent_id = write_index_session(
            store,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=_PARENT_NATIVE_ID,
                title="Order parity",
                messages=[_message(f"p{position}", position, _body(position)) for position in range(_PARENT_LENGTH)],
            ),
        )
        child_id = write_index_session(
            store,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=_CHILD_NATIVE_ID,
                title="Order parity fork",
                parent_session_provider_id=_PARENT_NATIVE_ID,
                branch_type=BranchType.FORK,
                messages=[
                    *(_message(f"p{position}", position, _body(position)) for position in range(_BRANCH_LENGTH)),
                    *(_message(f"c{index}", _BRANCH_LENGTH + index, f"tail {index}") for index in range(_TAIL_LENGTH)),
                ],
            ),
        )
    return parent_id, child_id


def _streamed_message_order(root: Path, session_id: str, expected: list[str]) -> list[str]:
    """Map the markdown export back to message ids through each body's text.

    The export renders blocks, not identities, so the fixture gives every
    message a unique body and the rendered offsets recover the order the
    export emitted them in.
    """
    out = root / "export.md"
    assert stream_exact_session_markdown(root, session_id, out, prose_only=False)
    text = out.read_text(encoding="utf-8")
    offsets = {}
    for position, message_id in enumerate(expected):
        body = _body(position)
        assert body in text, f"{body} missing from the export"
        offsets[message_id] = text.index(body)
    return sorted(offsets, key=lambda message_id: offsets[message_id])


@pytest.mark.asyncio
async def test_every_read_route_returns_one_message_order(
    tmp_path: Path,
    mcp_server: MCPServerUnderTest,
) -> None:
    parent_id, child_id = _seed(tmp_path)
    expected = [archive_message_id(parent_id, f"p{position}", position=position) for position in range(_PARENT_LENGTH)]
    # The fork stores only its divergent tail; its composed transcript is the
    # parent's prefix in the very same order, then that tail.
    expected_composed = expected[:_BRANCH_LENGTH] + [
        archive_message_id(child_id, f"c{index}", position=_BRANCH_LENGTH + index) for index in range(_TAIL_LENGTH)
    ]

    backend = SQLiteBackend(db_path=tmp_path / "index.db")
    try:
        async with backend.connection() as conn:
            assert [message.message_id for message in await get_messages(conn, parent_id)] == expected

            paginated, total, _completeness = await get_messages_paginated(conn, parent_id, limit=100)
            assert [message.message_id for message in paginated] == expected
            assert total == _PARENT_LENGTH

            for chunk_size in (1, 4, 100):
                streamed = [
                    message.message_id async for message in iter_messages(conn, parent_id, chunk_size=chunk_size)
                ]
                assert streamed == expected, f"chunk_size={chunk_size}"

            batched, _all_messages = await get_messages_batch(conn, [parent_id])
            assert [message.message_id for message in batched[parent_id]] == expected

            first, last, edge_total = await get_message_edge_windows(conn, parent_id, edge_limit=2)
            assert [message.message_id for message in first] == expected[:2]
            assert [message.message_id for message in last] == expected[-2:]
            assert edge_total == _PARENT_LENGTH

            # Lineage composition splices the parent prefix back in; the shared
            # prefix must be the identical sequence, not a re-sorted one.
            composed_child = [message.message_id for message in await get_messages(conn, child_id)]
            assert composed_child == expected_composed
    finally:
        await backend.close()

    assert _streamed_message_order(tmp_path, parent_id, expected) == expected
    # The export declines a prefix-sharing child rather than emitting a
    # tail-only transcript, so the eager composed path serves it.
    assert not stream_exact_session_markdown(tmp_path, child_id, tmp_path / "child.md", prose_only=False)

    archive = Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")
    try:
        api_messages, api_total, _completeness = await archive.get_messages_paginated(parent_id, limit=100)
        assert [str(message.id) for message in api_messages] == expected
        assert api_total == _PARENT_LENGTH

        api_child, _child_total, _child_completeness = await archive.get_messages_paginated(child_id, limit=100)
        assert [str(message.id) for message in api_child] == expected_composed

        with patch("polylogue.mcp.server._get_polylogue", return_value=archive):
            raw_parent = await invoke_surface_async(
                mcp_server._tool_manager._tools["read"].fn,
                ref=parent_id,
                view="messages",
                limit=100,
            )
            raw_child = await invoke_surface_async(
                mcp_server._tool_manager._tools["read"].fn,
                ref=child_id,
                view="messages",
                limit=100,
            )
    finally:
        await archive.close()

    assert [message["id"] for message in json.loads(raw_parent)["messages"]] == expected
    assert [message["id"] for message in json.loads(raw_child)["messages"]] == expected_composed
