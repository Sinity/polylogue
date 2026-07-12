"""MCP parity contract for candidate capture."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from polylogue.api import Polylogue
from polylogue.cli import cli
from polylogue.core.enums import AssertionKind, AssertionStatus
from polylogue.surfaces.payloads import AssertionClaimPayload
from tests.infra.mcp import MCPServerUnderTest, invoke_surface, make_polylogue_mock
from tests.infra.storage_records import SessionBuilder


def test_capture_candidate_returns_the_shared_assertion_payload(mcp_server: MCPServerUnderTest) -> None:
    payload = AssertionClaimPayload(
        assertion_id="assertion-terminal-note:mcp",
        target_ref="session:codex-session:demo",
        kind=AssertionKind.LESSON,
        body_text="lesson from MCP",
        evidence_refs=("session:codex-session:demo",),
        status=AssertionStatus.CANDIDATE,
        context_policy={"inject": False, "promotion_required": True},
        created_at_ms=1,
        updated_at_ms=1,
    )
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        poly = make_polylogue_mock()
        poly.capture_assertion_candidate = AsyncMock(return_value=payload)
        mock_get_polylogue.return_value = poly
        raw = invoke_surface(
            mcp_server._tool_manager._tools["capture_assertion_candidate"].fn,
            body_text="lesson from MCP",
            kind="lesson",
            refs=["session:codex-session:demo"],
            scope_refs=["repo:polylogue"],
        )

    assert json.loads(raw) == payload.model_dump(mode="json")
    poly.capture_assertion_candidate.assert_awaited_once_with(
        body_text="lesson from MCP",
        kind=AssertionKind.LESSON,
        refs=("session:codex-session:demo",),
        scope_refs=("repo:polylogue",),
        cwd=None,
    )


def test_capture_candidate_mcp_and_cli_share_the_real_pending_queue(
    mcp_server: MCPServerUnderTest,
    cli_workspace: dict[str, Path],
) -> None:
    """Both public surfaces write equivalent candidates through the same gate.

    This exercises the root CLI, registered MCP tool, real user tier, and
    pending-candidate reader. Replacing the MCP tool's shared facade call with
    a mock-only path, or letting either writer bypass candidate status, makes
    the queue assertions fail.
    """

    session = SessionBuilder(cli_workspace["db_path"], "terminal-note-mcp-parity").provider("codex")
    session.save()
    session_ref = f"session:{session.native_session_id()}"
    cli_result = CliRunner().invoke(
        cli,
        [
            "--plain",
            "note",
            "same capture through both surfaces",
            "--ref",
            session_ref,
            "--repo",
            "polylogue",
            "--topic",
            "sqlite",
            "--kind",
            "lesson",
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert cli_result.exit_code == 0, cli_result.output
    cli_payload = cast(dict[str, object], json.loads(cli_result.output))

    poly = Polylogue(archive_root=cli_workspace["archive_root"])
    with patch("polylogue.mcp.server._get_polylogue", return_value=poly):
        raw_mcp_payload = invoke_surface(
            mcp_server._tool_manager._tools["capture_assertion_candidate"].fn,
            body_text="same capture through both surfaces",
            kind="lesson",
            refs=[session_ref],
            scope_refs=["repo:polylogue", "insight:sqlite"],
        )
    mcp_payload = cast(dict[str, object], json.loads(raw_mcp_payload))

    compared_fields = (
        "target_ref",
        "scope_ref",
        "kind",
        "body_text",
        "evidence_refs",
        "status",
        "context_policy",
        "value",
    )
    assert {field: cli_payload[field] for field in compared_fields} == {
        field: mcp_payload[field] for field in compared_fields
    }

    async def list_pending() -> list[str]:
        async with Polylogue(archive_root=cli_workspace["archive_root"]) as reader:
            reviews = await reader.list_assertion_candidate_reviews()
        return [review.candidate.assertion_id for review in reviews.items]

    assert set(asyncio.run(list_pending())) == {cli_payload["assertion_id"], mcp_payload["assertion_id"]}
