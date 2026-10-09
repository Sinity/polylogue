from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from polylogue import Polylogue
from polylogue.archive.message.roles import Role
from polylogue.archive.query.transaction import QueryContinuation
from polylogue.core.enums import Provider
from polylogue.mcp.payloads import MCPMessageFragmentPayload
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.live_ingest import write_index_session
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async


@pytest.mark.asyncio
@pytest.mark.parametrize("projection", ["timeline", "sessions", "session-operations"])
async def test_registered_typed_pages_keep_budget_continuation(
    tmp_path: Path, mcp_server: MCPServerUnderTest, projection: str
) -> None:
    def seed() -> None:
        with ArchiveStore(tmp_path) as archive:
            for index in range(20):
                write_index_session(
                    archive,
                    ParsedSession(
                        source_name=Provider.CODEX,
                        provider_session_id=f"timeline-{index}",
                        title="t" * 2000,
                        messages=[
                            ParsedMessage(
                                provider_message_id="message",
                                role=Role.USER,
                                text="x" * 2000,
                                timestamp=f"2026-01-01T00:00:{index:02d}Z",
                                blocks=[],
                            )
                        ],
                    ),
                )

    run_off_event_loop(seed)
    api = Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")
    fn = mcp_server._tool_manager._tools["query"].fn
    args: dict[str, object] = {"projection": projection, "limit": 20}
    if projection == "session-operations":
        args = {"projection": projection, "session_operation": {"operation": "sessions.timeline", "limit": 20}}
    ids = []
    try:
        with (
            patch("polylogue.mcp.server._get_polylogue", return_value=api),
            patch("polylogue.mcp.server_support._record_mcp_call_log"),
        ):
            for _ in range(30):
                raw = await invoke_surface_async(fn, **args)
                assert len(raw.encode()) <= 25_000
                body = json.loads(raw)
                page = body.get("page", body)
                ids.extend(item.get("reference", item.get("id")) for item in page["items"])
                descriptor = body.get("continuation")
                if isinstance(descriptor, dict):
                    args = descriptor["arguments"]
                    token = args.get("continuation") or args["session_operation"]["continuation"]
                    assert QueryContinuation.decode(token).request.offset == len(ids)
                elif descriptor:
                    args = {"projection": projection, "continuation": descriptor}
                    if projection == "session-operations":
                        args = {
                            "projection": projection,
                            "session_operation": {"operation": "sessions.timeline", "continuation": descriptor},
                        }
                else:
                    break
            else:
                pytest.fail("continuation did not terminate")
    finally:
        await api.close()
    assert len(ids) == 20
    assert len(set(ids)) == 20


@pytest.mark.asyncio
async def test_registered_read_reassembles_oversized_unicode_message(
    tmp_path: Path, mcp_server: MCPServerUnderTest
) -> None:
    text = 'neutral é U0001f642 \\ "\n' * 4000

    def seed() -> str:
        with ArchiveStore(tmp_path) as archive:
            return write_index_session(
                archive,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="fragment",
                    messages=[
                        ParsedMessage(provider_message_id="large", role=Role.USER, text=text, blocks=[]),
                        ParsedMessage(provider_message_id="later", role=Role.ASSISTANT, text="later", blocks=[]),
                    ],
                ),
            )

    session_id = run_off_event_loop(seed)
    api = Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")
    fn = mcp_server._tool_manager._tools["read"].fn
    args: dict[str, object] = {"ref": f"session:{session_id}", "view": "messages", "limit": 1}
    parts = []
    offset = 0
    try:
        with (
            patch("polylogue.mcp.server._get_polylogue", return_value=api),
            patch("polylogue.mcp.server_support._record_mcp_call_log"),
        ):
            for _ in range(100):
                raw = await invoke_surface_async(fn, **args)
                assert len(raw.encode()) <= 25_000
                fragment = MCPMessageFragmentPayload.model_validate_json(raw)
                assert fragment.offset == offset
                assert fragment.session_ref == f"session:{session_id}"
                assert fragment.json_fragment.isascii()
                parts.append(fragment.json_fragment)
                offset += len(fragment.json_fragment)
                if fragment.next_fragment_offset is None:
                    assert offset == fragment.total_bytes
                    descriptor = fragment.continuation
                    assert descriptor is not None
                    assert QueryContinuation.decode(descriptor["arguments"]["continuation"]).request.offset == 1
                    later = json.loads(await invoke_surface_async(fn, **descriptor["arguments"]))
                    assert [row["text"] for row in later["messages"]] == ["later"]
                    break
                assert fragment.continuation is not None
                args = fragment.continuation["arguments"]
                # Same bound window and fragment offset is an idempotent retry.
                repeated = await invoke_surface_async(fn, **args)
                assert json.loads(repeated)["offset"] == offset
            else:
                pytest.fail("fragment walk did not terminate")
    finally:
        await api.close()
    row = json.loads("".join(parts))
    assert row["text"] == text
    assert row["id"] == fragment.message_id


@pytest.mark.asyncio
async def test_typed_session_error_is_logged_as_failure(mcp_server: MCPServerUnderTest) -> None:
    from tests.infra.mcp import make_polylogue_mock

    with (
        patch("polylogue.mcp.server._get_polylogue", return_value=make_polylogue_mock()),
        patch("polylogue.mcp.server_support._record_mcp_call_log") as record,
    ):
        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["query"].fn,
            projection="session-operations",
            session_operation={"operation": "sessions.timeline", "since": "2026-10-09"},
        )
    assert json.loads(raw)["outcome"] == "error"
    assert record.call_args.kwargs["success"] is False
    assert record.call_args.kwargs["error_detail"] == "session_read_failed"


@pytest.mark.asyncio
async def test_registered_blackboard_can_read_after_millionth_note(mcp_server: MCPServerUnderTest) -> None:
    from unittest.mock import AsyncMock

    from polylogue.archive.blackboard import BlackboardNote, BlackboardPage
    from tests.infra.mcp import make_polylogue_mock

    poly = make_polylogue_mock()
    poly.read_blackboard_page = AsyncMock(
        return_value=BlackboardPage(
            items=(
                BlackboardNote(
                    note_id="after-prefix",
                    kind="finding",
                    title="neutral",
                    content="neutral",
                    scope_repo=None,
                    target_type=None,
                    target_id=None,
                    created_at_ms=1,
                    updated_at_ms=1,
                ),
            ),
            total=1_000_001,
            limit=1,
            offset=1_000_000,
        )
    )
    with (
        patch("polylogue.mcp.server._get_polylogue", return_value=poly),
        patch("polylogue.mcp.server_support._record_mcp_call_log"),
    ):
        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["query"].fn, projection="blackboard", limit=1, offset=1_000_000
        )
    body = json.loads(raw)
    assert body["total"] == 1_000_001
    assert body["items"][0]["note_id"] == "after-prefix"
    assert body["next_offset"] is None
    poly.read_blackboard_page.assert_awaited_once_with(limit=1, offset=1_000_000)
