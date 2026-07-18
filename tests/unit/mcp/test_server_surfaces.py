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

    return cast(MCPServerUnderTest, build_server(role="read"))


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
