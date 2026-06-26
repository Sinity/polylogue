"""Tests for the distilled-bundle MCP tools (#2436).

`get_postmortem_bundle` (read) and `export_sanitized` (write) delegate to the
#2380/#2381 substrate. The sanitized surface is redacted-only by construction:
it has no redaction-disable affordance and returns a typed refusal — never a
partial bundle — when the fail-closed gate trips.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.export.sanitize import SanitizedExportError, SanitizedExportResult
from polylogue.insights.postmortem import PostmortemScope, compile_postmortem_bundle
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async, make_polylogue_mock


@pytest.mark.asyncio
async def test_get_postmortem_bundle_delegates_and_serializes(mcp_server: MCPServerUnderTest) -> None:
    bundle = compile_postmortem_bundle([], {}, scope=PostmortemScope(matched_session_count=0))
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.postmortem_bundle = AsyncMock(return_value=bundle)
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["get_postmortem_bundle"].fn,
            since="2026-01-01",
        )

    payload = json.loads(raw)
    assert "scope" in payload
    assert "schema_version" in payload
    assert mock_poly.postmortem_bundle.await_count == 1


@pytest.mark.asyncio
async def test_export_sanitized_refuses_on_gate_failure(mcp_server: MCPServerUnderTest, tmp_path: Path) -> None:
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.sanitized_export = AsyncMock(side_effect=SanitizedExportError("absolute path survived the gate"))
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["export_sanitized"].fn,
            output_path=str(tmp_path / "bundle"),
        )

    payload = json.loads(raw)
    # Typed refusal — nothing published, no raised exception.
    assert payload.get("code") == "sanitized_export_refused"
    assert "message" in payload


@pytest.mark.asyncio
async def test_export_sanitized_returns_result_on_success(mcp_server: MCPServerUnderTest, tmp_path: Path) -> None:
    out = tmp_path / "bundle"
    result = SanitizedExportResult(
        output_path=out,
        dataset_path=out / "dataset.jsonl",
        manifest_path=out / "redaction-manifest.json",
        readme_path=out / "README.md",
        row_count=2,
        total_included=2,
        verify_ok=True,
    )
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.sanitized_export = AsyncMock(return_value=result)
        mock_get_polylogue.return_value = mock_poly

        raw = await invoke_surface_async(
            mcp_server._tool_manager._tools["export_sanitized"].fn,
            output_path=str(out),
        )

    payload = json.loads(raw)
    assert payload["row_count"] == 2
    assert payload["verify_ok"] is True


@pytest.mark.asyncio
async def test_export_sanitized_forces_redaction(mcp_server: MCPServerUnderTest, tmp_path: Path) -> None:
    out = tmp_path / "bundle"
    result = SanitizedExportResult(
        output_path=out,
        dataset_path=out / "dataset.jsonl",
        manifest_path=out / "redaction-manifest.json",
        readme_path=out / "README.md",
    )
    with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
        mock_poly = make_polylogue_mock()
        mock_poly.sanitized_export = AsyncMock(return_value=result)
        mock_get_polylogue.return_value = mock_poly

        await invoke_surface_async(
            mcp_server._tool_manager._tools["export_sanitized"].fn,
            output_path=str(out),
        )

    # The request handed to the substrate must be redacted-only.
    request = mock_poly.sanitized_export.await_args.args[1]
    assert request.redact is True
    assert request.acknowledge_unredacted is False


def test_export_sanitized_has_no_redaction_disable_param(mcp_server: MCPServerUnderTest) -> None:
    """AC3: the MCP sanitized path exposes no redaction-disable affordance."""
    params = set(inspect.signature(mcp_server._tool_manager._tools["export_sanitized"].fn).parameters)
    assert "redact" not in params
    assert "acknowledge_unredacted" not in params
    assert "no_redact" not in params
