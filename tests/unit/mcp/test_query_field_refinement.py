"""The MCP ``query`` tool teaches a misspelt field without guessing (polylogue-z9gh.3.3)."""

from __future__ import annotations

import json
from pathlib import Path

from tests.infra.mcp import MCPServerUnderTest, invoke_surface
from tests.unit.mcp.test_contract_evidence import _seeded_runtime_services
from tests.unit.mcp.test_reference_query_pipeline import _seed_archive


def test_the_mcp_query_tool_returns_the_refinement_and_runs_nothing(
    mcp_server: MCPServerUnderTest, tmp_path: Path
) -> None:
    """The declaration-to-wire-to-consumer path for the selected family.

    A cold caller's first try misspells a field. The error names the field,
    carries the declared candidates and the corrected expression, and returns
    no rows; the corrected expression, sent back verbatim, is an ordinary
    successful read. Anti-vacuity: drop the MCP boundary's
    ``propose_field_correction`` call and ``corrected_expression`` is null.
    """
    archive_root = tmp_path / "archive"
    _seed_archive(archive_root)
    with _seeded_runtime_services(archive_root):
        tool = mcp_server._tool_manager._tools["query"].fn
        refused = json.loads(invoke_surface(tool, expression="messages where rol:user"))
        retried = json.loads(invoke_surface(tool, expression=refused["corrected_expression"]))

    assert refused["code"] == "invalid_query"
    assert refused["field"] == "rol"
    assert refused["candidates"] == ["role"]
    assert refused["corrected_expression"] == "messages where role:user"
    assert "rows" not in refused
    assert retried.get("is_error") is not True, retried
