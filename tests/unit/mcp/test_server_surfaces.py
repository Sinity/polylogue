"""Production contracts for the six-tool MCP read surface."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.mcp import MCP_TOOL_NAME_BASELINE, MCPServerUnderTest, invoke_surface_async
from tests.infra.storage_records import SessionBuilder


def _write_message(archive_root: Path, native_id: str, text: str) -> None:
    SessionBuilder(archive_root / "index.db", native_id).provider("codex-session").add_message(
        role="user", text=text
    ).save()


@pytest.fixture
def mcp_server() -> MCPServerUnderTest:
    from polylogue.mcp.server import build_server

    return cast(MCPServerUnderTest, build_server())


def test_default_read_discovery_has_no_retired_tools(mcp_server: MCPServerUnderTest) -> None:
    assert set(mcp_server._tool_manager._tools) == MCP_TOOL_NAME_BASELINE
    assert "query_units" not in mcp_server._tool_manager._tools
    assert "search" not in mcp_server._tool_manager._tools


@pytest.mark.asyncio
async def test_query_drains_real_archive_rows_with_continuation_only(
    mcp_server: MCPServerUnderTest, tmp_path: Path
) -> None:
    archive_root = tmp_path / "archive"
    with ArchiveStore(archive_root):
        _write_message(archive_root, "cutover-one", "cutover needle one")
        _write_message(archive_root, "cutover-two", "cutover needle two")

    from polylogue import Polylogue

    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=archive_root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=Polylogue(archive_root=archive_root)),
    ):
        first = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["query"].fn,
                expression="messages where text:needle",
                limit=1,
            )
        )
        second = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["query"].fn,
                continuation=first["continuation"],
            )
        )

    assert first["query_ref"] == second["query_ref"]
    assert first["result_ref"] == second["result_ref"]
    assert first["continuation"].startswith("q2.")
    assert {item["message_id"] for item in (*first["items"], *second["items"])}


@pytest.mark.asyncio
async def test_query_transaction_certifies_twenty_large_messages_across_api_and_mcp(
    mcp_server: MCPServerUnderTest, tmp_path: Path
) -> None:
    """A terminal storage page that exceeds MCP bytes still drains losslessly.

    Production dependencies exercised: ``Polylogue.query_units`` creates the
    framed q2 result, while the registered MCP ``query`` handler must rebase
    the terminal storage page from that frame instead of replacing it with a
    metadata-only overflow response.  Removing that rebasing loses the suffix
    or makes the returned continuation point past unseen rows.
    """
    archive_root = tmp_path / "archive"
    body = "transaction-certification " + ("x" * 3_500)
    with ArchiveStore(archive_root):
        for number in range(20):
            _write_message(archive_root, f"certification-{number:02d}", f"{body} {number:02d}")

    from polylogue import Polylogue

    archive = Polylogue(archive_root=archive_root)
    api_page = await archive.query_units("messages where text:transaction-certification", limit=20)
    assert len(api_page.items) == 20
    assert api_page.continuation is None

    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=archive_root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=archive),
    ):
        response = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["query"].fn,
                expression="messages where text:transaction-certification",
                limit=20,
            )
        )
        assert response["status"] == "response_budget_exceeded"
        assert response["original_bytes"] > 25_000
        pages: list[dict[str, object]] = [cast(dict[str, object], response["page"])]
        continuation_arguments = cast(dict[str, object], cast(dict[str, object], response["continuation"])["arguments"])
        while continuation_arguments:
            page_response = json.loads(
                await invoke_surface_async(mcp_server._tool_manager._tools["query"].fn, **continuation_arguments)
            )
            if page_response.get("status") == "response_budget_exceeded":
                pages.append(cast(dict[str, object], page_response["page"]))
                continuation_arguments = cast(
                    dict[str, object], cast(dict[str, object], page_response["continuation"])["arguments"]
                )
            else:
                pages.append(page_response)
                continuation = cast(str | None, page_response.get("continuation"))
                continuation_arguments = {"continuation": continuation} if continuation is not None else {}

    returned_ids = {item["message_id"] for page in pages for item in cast(list[dict[str, str]], page["items"])}
    assert len(returned_ids) == 20


@pytest.mark.asyncio
async def test_query_rejects_resume_parameter_overrides(mcp_server: MCPServerUnderTest, tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    with ArchiveStore(archive_root):
        _write_message(archive_root, "cutover-one", "cutover override needle")
        _write_message(archive_root, "cutover-two", "cutover override needle")

    from polylogue import Polylogue

    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=archive_root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=Polylogue(archive_root=archive_root)),
    ):
        first = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["query"].fn,
                expression="messages where text:needle",
                limit=1,
            )
        )
        rejected = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["query"].fn,
                expression="messages where text:other",
                continuation=first["continuation"],
            )
        )

    assert rejected["code"] == "invalid_continuation"


@pytest.mark.asyncio
async def test_query_rejects_epoch_stale_resume(mcp_server: MCPServerUnderTest, tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    with ArchiveStore(archive_root):
        _write_message(archive_root, "cutover-one", "cutover epoch needle")
        _write_message(archive_root, "cutover-two", "cutover epoch needle")

    from polylogue import Polylogue

    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=archive_root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=Polylogue(archive_root=archive_root)),
    ):
        first = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["query"].fn,
                expression="messages where text:needle",
                limit=1,
            )
        )
        with ArchiveStore(archive_root):
            _write_message(archive_root, "cutover-three", "cutover epoch needle")
        stale = json.loads(
            await invoke_surface_async(mcp_server._tool_manager._tools["query"].fn, continuation=first["continuation"])
        )

    assert stale["code"] == "query_continuation_stale"


@pytest.mark.asyncio
async def test_read_and_get_accept_stable_session_uris(mcp_server: MCPServerUnderTest, tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    with ArchiveStore(archive_root):
        _write_message(archive_root, "cutover-ref", "stable ref")

    uri = "polylogue://session/codex-session:cutover-ref"
    from polylogue import Polylogue

    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=archive_root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=Polylogue(archive_root=archive_root)),
    ):
        read = json.loads(await invoke_surface_async(mcp_server._tool_manager._tools["read"].fn, ref=uri))
        exact = json.loads(await invoke_surface_async(mcp_server._tool_manager._tools["get"].fn, ref=uri))

    assert read == exact


@pytest.mark.asyncio
async def test_get_projection_events_surfaces_session_timeline_evidence(
    mcp_server: MCPServerUnderTest, tmp_path: Path
) -> None:
    """``get(ref, projection="events")`` reaches ``session_events`` -- the
    evidence axis Codex ``world_state``/``agent_policy`` facts, Claude Code
    sidecar events, and Hermes tool-availability spans all ride, and which
    no MCP tool could reach before this projection.
    """
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import Provider
    from polylogue.sources.parsers.base import ParsedMessage, ParsedSession, ParsedSessionEvent

    archive_root = tmp_path / "archive"
    with ArchiveStore(archive_root) as archive_db:
        parsed = ParsedSession(
            source_name=Provider.from_string("codex"),
            provider_session_id="mcp-events-ref",
            title="MCP events projection",
            messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="hello")],
            session_events=[
                ParsedSessionEvent(
                    event_type="world_state",
                    payload={"cwd": "/repo"},
                ),
            ],
        )
        archive_db.write_raw_and_parsed(
            parsed,
            payload=b'{"raw": "codex payload"}',
            source_path="/tmp/raw.jsonl",
            acquired_at_ms=1735689600000,
        )

    uri = "polylogue://session/codex-session:mcp-events-ref"
    from polylogue import Polylogue

    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=archive_root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=Polylogue(archive_root=archive_root)),
    ):
        events_payload = json.loads(
            await invoke_surface_async(mcp_server._tool_manager._tools["get"].fn, ref=uri, projection="events")
        )
        default_payload = json.loads(await invoke_surface_async(mcp_server._tool_manager._tools["get"].fn, ref=uri))

    assert events_payload["total"] == 1
    assert events_payload["events"][0]["event_type"] == "world_state"
    assert events_payload["events"][0]["payload"]["cwd"] == "/repo"
    # The default (no projection) path still resolves the ordinary session
    # summary, unaffected by the new projection branch.
    assert "events" not in default_payload

    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=archive_root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=Polylogue(archive_root=archive_root)),
    ):
        missing = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["get"].fn,
                ref="polylogue://session/codex-session:does-not-exist",
                projection="events",
            )
        )
    assert missing["code"] == "not_found"


@pytest.mark.asyncio
async def test_get_projection_file_edits_surfaces_structured_patch_evidence(
    mcp_server: MCPServerUnderTest, tmp_path: Path
) -> None:
    """``get(ref, projection="file-edits")`` reaches the ``file_edits`` table
    (polylogue-nua7/polylogue-cgfy): structured unified diffs, pre-edit file
    content, and old/new string pairs captured on Edit/Write tool calls, but
    unreachable from any surface before this projection existed -- the exact
    "what did this session change" evidence a postmortem report needs.
    """
    from polylogue.core.enums import BlockType, Provider, Role
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedFileEdit, ParsedMessage, ParsedSession

    archive_root = tmp_path / "archive"
    with ArchiveStore(archive_root) as archive_db:
        parsed = ParsedSession(
            source_name=Provider.CLAUDE_CODE,
            provider_session_id="mcp-file-edits-ref",
            title="MCP file-edits projection",
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.ASSISTANT,
                    position=0,
                    blocks=[
                        ParsedContentBlock(
                            type=BlockType.TOOL_USE,
                            tool_name="Edit",
                            tool_id="edit-tool-1",
                            tool_input={"file_path": "/tmp/foo.py"},
                        ),
                    ],
                ),
                ParsedMessage(
                    provider_message_id="m2",
                    role=Role.USER,
                    position=1,
                    blocks=[
                        ParsedContentBlock(
                            type=BlockType.TOOL_RESULT,
                            tool_id="edit-tool-1",
                            text="applied",
                            file_edit=ParsedFileEdit(
                                file_path="/tmp/foo.py",
                                structured_patch=[
                                    {"oldStart": 1, "oldLines": 1, "newStart": 1, "newLines": 2, "lines": ["+x"]}
                                ],
                                original_file="old contents\n",
                                old_string="old",
                                new_string="new",
                                replace_all=False,
                                user_modified=True,
                            ),
                        ),
                    ],
                ),
            ],
        )
        archive_db.write_raw_and_parsed(
            parsed,
            payload=b'{"raw": "claude payload"}',
            source_path="/tmp/raw.jsonl",
            acquired_at_ms=1735689600000,
        )

    uri = "polylogue://session/claude-code-session:mcp-file-edits-ref"
    from polylogue import Polylogue

    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=archive_root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=Polylogue(archive_root=archive_root)),
    ):
        payload = json.loads(
            await invoke_surface_async(mcp_server._tool_manager._tools["get"].fn, ref=uri, projection="file-edits")
        )
        default_payload = json.loads(await invoke_surface_async(mcp_server._tool_manager._tools["get"].fn, ref=uri))

    assert payload["total"] == 1
    edit = payload["file_edits"][0]
    assert edit["file_path"] == "/tmp/foo.py"
    assert edit["original_file"] == "old contents\n"
    assert edit["old_string"] == "old"
    assert edit["new_string"] == "new"
    assert edit["structured_patch"] == [{"oldStart": 1, "oldLines": 1, "newStart": 1, "newLines": 2, "lines": ["+x"]}]
    assert "file_edits" not in default_payload

    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=archive_root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=Polylogue(archive_root=archive_root)),
    ):
        missing = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["get"].fn,
                ref="polylogue://session/claude-code-session:does-not-exist",
                projection="file-edits",
            )
        )
    assert missing["code"] == "not_found"


@pytest.mark.asyncio
async def test_get_projection_agent_policies_surfaces_sandbox_facts(
    mcp_server: MCPServerUnderTest, tmp_path: Path
) -> None:
    """``get(ref, projection="agent-policies")`` reaches the dedicated
    ``session_agent_policies`` table (polylogue-nua7) -- Codex sandbox/
    approval/network policy facts the writer diverts out of
    ``session_events`` for zero-loss re-derivation, but which had zero
    surface consumers before this projection.
    """
    from polylogue.core.enums import BlockType, Provider, Role
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent

    archive_root = tmp_path / "archive"
    with ArchiveStore(archive_root) as archive_db:
        parsed = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="mcp-agent-policies-ref",
            title="MCP agent-policies projection",
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.USER,
                    text="run it",
                    position=0,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="run it")],
                ),
            ],
            session_events=[
                ParsedSessionEvent(
                    event_type="agent_policy",
                    timestamp="2026-01-01T00:00:01+00:00",
                    payload={
                        "approval_policy": "never",
                        "sandbox_policy": "danger-full-access",
                        "network_policy": "true",
                    },
                ),
            ],
        )
        archive_db.write_raw_and_parsed(
            parsed,
            payload=b'{"raw": "codex payload"}',
            source_path="/tmp/raw.jsonl",
            acquired_at_ms=1735689600000,
        )

    uri = "polylogue://session/codex-session:mcp-agent-policies-ref"
    from polylogue import Polylogue

    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=archive_root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=Polylogue(archive_root=archive_root)),
    ):
        payload = json.loads(
            await invoke_surface_async(mcp_server._tool_manager._tools["get"].fn, ref=uri, projection="agent-policies")
        )
        default_payload = json.loads(await invoke_surface_async(mcp_server._tool_manager._tools["get"].fn, ref=uri))

    assert payload["total"] == 1
    policy = payload["agent_policies"][0]
    assert policy["approval_policy"] == "never"
    assert policy["sandbox_policy"] == "danger-full-access"
    assert policy["network_policy"] == "true"
    assert "agent_policies" not in default_payload

    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=archive_root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=Polylogue(archive_root=archive_root)),
    ):
        missing = json.loads(
            await invoke_surface_async(
                mcp_server._tool_manager._tools["get"].fn,
                ref="polylogue://session/codex-session:does-not-exist",
                projection="agent-policies",
            )
        )
    assert missing["code"] == "not_found"


@pytest.mark.asyncio
async def test_get_default_projection_surfaces_display_name_when_title_absent(
    mcp_server: MCPServerUnderTest, tmp_path: Path
) -> None:
    """``get(ref)`` (no projection) reaches ``sessions.display_name``
    (polylogue-cgfy): a session with no title-worthy sidecar evidence (the
    common Claude Code subagent case) now surfaces its provider-assigned
    slug as the title instead of a raw session id -- read through the real
    ``ArchiveStore``-backed summary path (``_resolve_session_object_ref`` ->
    ``_archive_summary_to_domain`` -> ``SessionSummaryPayload``), not the
    storage row in isolation.
    """
    from polylogue.core.enums import BlockType, Provider, Role
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession

    archive_root = tmp_path / "archive"
    with ArchiveStore(archive_root) as archive_db:
        parsed = ParsedSession(
            source_name=Provider.CLAUDE_CODE,
            provider_session_id="mcp-slug-only-ref",
            title=None,
            display_name="greedy-squishing-hamming",
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.ASSISTANT,
                    position=0,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hi")],
                ),
            ],
        )
        archive_db.write_raw_and_parsed(
            parsed,
            payload=b'{"raw": "claude payload"}',
            source_path="/tmp/raw.jsonl",
            acquired_at_ms=1735689600000,
        )

    uri = "polylogue://session/claude-code-session:mcp-slug-only-ref"
    from polylogue import Polylogue

    with (
        patch("polylogue.mcp.server._get_config", return_value=SimpleNamespace(archive_root=archive_root)),
        patch("polylogue.mcp.server._get_polylogue", return_value=Polylogue(archive_root=archive_root)),
    ):
        payload = json.loads(await invoke_surface_async(mcp_server._tool_manager._tools["get"].fn, ref=uri))

    assert payload["title"] == "greedy-squishing-hamming"
