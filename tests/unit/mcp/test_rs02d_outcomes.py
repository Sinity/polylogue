"""Production MCP boundaries preserve partial reads and uncertain writes (rs02d)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from polylogue.analysis.pathology import PathologyFinding, PathologyReport
from polylogue.analysis.postmortem import PostmortemScope, compile_postmortem_bundle
from polylogue.api import Polylogue
from polylogue.daemon_client import DaemonClient
from polylogue.mcp.declarations.models import MCPCapabilities
from polylogue.operations.daemon_errors import DaemonMutationIndeterminateError
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.mcp import build_tools, installed_runtime_services, invoke_surface_async


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["lost_receipt", "indeterminate_receipt", "absent"])
async def test_registered_maintenance_preserves_submitted_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """08.F045: one submit, no replay, and an actionable id for uncertain writes."""
    tools = build_tools(MCPCapabilities(maintenance=True))
    monkeypatch.setattr("polylogue.cli.read_dispatch.daemon_route_disabled", lambda **_: False)
    submitted: list[str] = []

    def transport(
        self: object,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        mutation: bool = False,
        **kwargs: object,
    ) -> dict[str, Any] | None:
        assert body is not None
        assert method == "POST" and path == "/api/operation"
        assert mutation is True
        assert body["operation"] == "maintenance.insights.rebuild"
        submitted.append(body["request_id"])
        if mode == "absent":
            return None
        raise DaemonMutationIndeterminateError(method=method, path=path, request_id=body["request_id"])

    def indeterminate_operation(
        self: object, operation: str, payload: object, *, request_id: str, **kwargs: object
    ) -> dict[str, Any]:
        assert operation == "maintenance.insights.rebuild"
        submitted.append(request_id)
        return {"operation": operation, "request_id": request_id, "outcome": "indeterminate"}

    if mode == "indeterminate_receipt":
        monkeypatch.setattr(DaemonClient, "operation", indeterminate_operation)
    else:
        # Keep the real DaemonClient operation/request construction, failing only
        # at the transport seam after the precise mutation identity is known.
        monkeypatch.setattr(DaemonClient, "_request_json_response", transport)
    with installed_runtime_services(tmp_path / "archive"):
        result = json.loads(
            await invoke_surface_async(tools["maintenance"], operation="rebuild_insights", confirm=True)
        )
    assert len(submitted) == 1
    assert result["is_error"] is True
    if mode == "absent":
        assert result["code"] == "daemon_required"
        assert "request_id" not in result
    else:
        assert result["code"] == "indeterminate"
        assert result["request_id"] == submitted[0]
        assert result["retryable"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["cancelled", "timed-out", "degraded", "disconnected-before-acceptance"])
async def test_registered_maintenance_refuses_unfinished_envelopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """An envelope that did not complete is an error even when it carries a body."""
    tools = build_tools(MCPCapabilities(maintenance=True))
    monkeypatch.setattr("polylogue.cli.read_dispatch.daemon_route_disabled", lambda **_: False)

    def unfinished_operation(
        self: object, operation: str, payload: object, *, request_id: str, **kwargs: object
    ) -> dict[str, Any]:
        return {"operation": operation, "request_id": request_id, "outcome": outcome, "result": {"rebuilt": 0}}

    monkeypatch.setattr(DaemonClient, "operation", unfinished_operation)
    with installed_runtime_services(tmp_path / "archive"):
        result = json.loads(
            await invoke_surface_async(tools["maintenance"], operation="rebuild_insights", confirm=True)
        )
    assert result["is_error"] is True, result
    assert result["code"] == outcome
    assert "rebuilt" not in result


def test_shared_exception_boundary_preserves_identity_without_internal_paths() -> None:
    from polylogue.mcp.server_support import _exception_to_error_json

    error = DaemonMutationIndeterminateError(
        method="POST", path="/internal/secret-location", request_id="synthetic-accepted-write"
    )
    result = json.loads(_exception_to_error_json("write", error))
    assert result["request_id"] == "synthetic-accepted-write"
    assert result["code"] == "indeterminate"
    assert result["retryable"] is False
    assert "secret-location" not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("report", "state", "gaps"),
    [
        (PathologyReport(matched_session_count=0, analyzed_session_count=0), "empty", set()),
        (PathologyReport(matched_session_count=2, analyzed_session_count=2), "empty", set()),
        (
            PathologyReport(matched_session_count=2, analyzed_session_count=1, truncated=True, dropped_session_count=1),
            "degraded",
            {"match_cap_exceeded"},
        ),
        (
            PathologyReport(matched_session_count=1, analyzed_session_count=0, failed_session_count=1),
            "degraded",
            {"session_digest_unavailable"},
        ),
        (
            PathologyReport(
                findings=(PathologyFinding(kind="wasted_loop", session_id="synthetic", severity="low", detail="loop"),),
                matched_session_count=1,
                analyzed_session_count=1,
            ),
            "ok",
            set(),
        ),
    ],
)
async def test_registered_pathology_projection_reports_terminal_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, report: PathologyReport, state: str, gaps: set[str]
) -> None:
    """08.F043: zero findings is not evidence of a complete analysis."""
    tools = build_tools()
    query = AsyncMock(return_value=report)
    monkeypatch.setattr(Polylogue, "pathology_report", query)
    with installed_runtime_services(tmp_path / "archive"):
        result = json.loads(await invoke_surface_async(tools["query"], projection="pathologies"))
    query.assert_awaited_once()
    assert result["findings"] == report.model_dump(mode="json")["findings"]
    assert result["outcome"]["state"] == state
    assert set(result["outcome"].get("detail", {}).get("gaps", [])) == gaps


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "missing_digests", "expected_gaps"),
    [
        (PostmortemScope(), 0, set()),
        (
            PostmortemScope(matched_session_count=2, analyzed_session_count=1, truncated=True, dropped_session_count=1),
            0,
            {"match_cap_exceeded", "longest_tool_gap_unavailable"},
        ),
        (
            PostmortemScope(matched_session_count=1, analyzed_session_count=0),
            0,
            {"session_profile_unavailable", "longest_tool_gap_unavailable"},
        ),
        (
            PostmortemScope(matched_session_count=1, analyzed_session_count=1),
            1,
            {"session_digest_unavailable", "longest_tool_gap_unavailable"},
        ),
    ],
)
async def test_registered_postmortem_projection_preserves_coverage_gaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope: PostmortemScope,
    missing_digests: int,
    expected_gaps: set[str],
) -> None:
    tools = build_tools()
    bundle = compile_postmortem_bundle([], {}, scope=scope)
    if missing_digests:
        # The operation boundary receives typed coverage receipts from the
        # analysis owner; it must not discard a missing-digest receipt.
        field = bundle.failure_mode.model_copy(update={"missing_digest_count": missing_digests})
        bundle = bundle.model_copy(update={"failure_mode": field})
    query = AsyncMock(return_value=bundle)
    monkeypatch.setattr(Polylogue, "postmortem_bundle", query)
    with installed_runtime_services(tmp_path / "archive"):
        result = json.loads(await invoke_surface_async(tools["query"], projection="postmortem"))
    query.assert_awaited_once()
    assert result["scope"] == scope.model_dump(mode="json", exclude_none=True)
    assert result["outcome"]["state"] == ("degraded" if expected_gaps else "empty")
    assert set(result["outcome"].get("detail", {}).get("gaps", [])) == expected_gaps


@pytest.mark.asyncio
async def test_real_archive_no_match_projections_are_empty(tmp_path: Path) -> None:
    """Keep the full registered tool -> facade -> archive path under test too."""
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.live_ingest import write_index_session

    root = tmp_path / "archive"

    def seed() -> None:
        with ArchiveStore(root) as archive:
            write_index_session(
                archive,
                ParsedSession(
                    source_name=Provider.CHATGPT,
                    provider_session_id="synthetic-outcome",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m1",
                            role=Role.USER,
                            text="synthetic outcome evidence",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="synthetic outcome evidence")],
                        )
                    ],
                ),
            )

    run_off_event_loop(seed)

    tools = build_tools()
    with installed_runtime_services(root):
        for projection in ("postmortem", "pathologies"):
            result = json.loads(
                await invoke_surface_async(tools["query"], projection=projection, repo="nonexistent-synthetic-repo")
            )
            assert result.get("is_error") is not True, result
            assert result["outcome"]["state"] == "empty", result


@pytest.mark.asyncio
async def test_registered_delete_prepares_then_consumes_the_callers_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """08.F022: the real MCP/daemon route never manufactures consent at apply."""
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, Provider
    from polylogue.daemon.socket_path import daemon_socket_path
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.daemon_operations import running_daemon_operations
    from tests.infra.live_ingest import write_index_session

    root = tmp_path / "archive"
    ids: list[str] = []

    def seed(archive_root: Path) -> None:
        with ArchiveStore(archive_root) as archive:
            ids.append(
                write_index_session(
                    archive,
                    ParsedSession(
                        source_name=Provider.CHATGPT,
                        provider_session_id="synthetic-delete-preview",
                        messages=[
                            ParsedMessage(
                                provider_message_id="m1",
                                role=Role.USER,
                                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="synthetic delete target")],
                            )
                        ],
                    ),
                )
            )

    monkeypatch.setattr("polylogue.daemon.api_auth.resolve_api_auth_token", lambda *_args, **_kwargs: None)
    with running_daemon_operations(root, seed_archive=seed, socket_path=daemon_socket_path(root)):
        tools = build_tools(MCPCapabilities(write=True))
        with installed_runtime_services(root):
            bare = json.loads(
                await invoke_surface_async(tools["write"], operation="delete_session", session_id=ids[0], confirm=True)
            )
            assert bare["code"] == "invalid_argument", bare
            prepared = json.loads(
                await invoke_surface_async(tools["write"], operation="prepare_delete_session", session_id=ids[0])
            )
            assert prepared["outcome"] == "prepared", prepared
            assert prepared["session_id"] == ids[0]
            deleted = json.loads(
                await invoke_surface_async(
                    tools["write"],
                    operation="delete_session",
                    session_id=ids[0],
                    confirm=True,
                    fields={"preview_ref": prepared["preview_ref"]},
                )
            )
            assert deleted["status"] == "deleted", deleted
            absent = json.loads(
                await invoke_surface_async(tools["write"], operation="prepare_delete_session", session_id=ids[0])
            )
            assert absent["outcome"] == "not_found", absent
            assert "preview_ref" not in absent
