"""polylogue-9layg: MCP `read` must serve the messages view, not refuse it.

`read` accepted only the summary and topology views, so a caller asking for
messages — the view the CLI serves — got `unsupported read view` rather than
the session's messages.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from polylogue import Polylogue
from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.runtime import LineageCompleteness
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.identity import archive_message_id
from tests.infra.live_ingest import write_index_session
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async, make_polylogue_mock

_SESSION_ID = "codex-session:native-1"


async def _invoke_read(mcp_server: MCPServerUnderTest, *, view: str) -> dict[str, object]:
    with patch("polylogue.mcp.server._get_polylogue") as get_polylogue:
        poly = make_polylogue_mock(resolved_id=_SESSION_ID)
        poly.get_messages_paginated = AsyncMock(
            return_value=((), 0, LineageCompleteness(complete=True, truncation_reason=None))
        )
        get_polylogue.return_value = poly
        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["read"].fn,
            ref=_SESSION_ID,
            view=view,
        )
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
    return parsed


@pytest.mark.asyncio
async def test_read_serves_the_messages_view(mcp_server: MCPServerUnderTest) -> None:
    """Anti-vacuity: drop the messages branch and this returns the
    `unsupported read view` error instead of a messages payload."""
    payload = await _invoke_read(mcp_server, view="messages")

    assert "unsupported read view" not in json.dumps(payload)
    assert payload.get("session_id") == _SESSION_ID
    assert "messages" in payload


@pytest.mark.asyncio
async def test_read_still_refuses_a_view_it_does_not_serve(mcp_server: MCPServerUnderTest) -> None:
    """Widening must not turn the view name into a free-for-all."""
    payload = await _invoke_read(mcp_server, view="not-a-view")

    assert "unsupported read view" in json.dumps(payload)


@pytest.mark.asyncio
async def test_read_messages_matches_python_api_positional_order(
    mcp_server: MCPServerUnderTest,
    tmp_path: Path,
) -> None:
    root = tmp_path
    with ArchiveStore(root) as store:
        session_id = write_index_session(
            store,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="order-parity",
                title="Order parity",
                messages=[
                    ParsedMessage(
                        provider_message_id=f"message-{position}",
                        role=Role.USER if position % 2 == 0 else Role.ASSISTANT,
                        text=f"body {position}",
                        timestamp=f"2026-01-01T00:00:{9 - position:02d}Z",
                        blocks=[],
                    )
                    for position in range(6)
                ],
            ),
        )

    archive = Polylogue(archive_root=root, db_path=root / "index.db")
    try:
        api_messages, total, _completeness = await archive.get_messages_paginated(session_id, limit=100)
        with patch("polylogue.mcp.server._get_polylogue", return_value=archive):
            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["read"].fn,
                ref=session_id,
                view="messages",
                limit=100,
            )
    finally:
        await archive.close()

    expected = [archive_message_id(session_id, f"message-{position}", position=position) for position in range(6)]
    payload = json.loads(raw)
    assert total == len(expected)
    assert [str(message.id) for message in api_messages] == expected
    assert [message["id"] for message in payload["messages"]] == expected
