"""MCP ``get`` tool resolves ``metric:<hash-or-name>`` refs (polylogue-rxdo.9.1).

``MetricDefinition``/``MetricRegistry`` (PR #2888, merged) had zero
production callers -- the corrective AC's second consumer path depends on
``polylogue-9l5.7``'s statistics registry, which remains unstarted (that
whole registry/composition epic is out of scope for this wiring pass). This
test proves the bounded, honest slice that IS wired: the process-wide
``DEFAULT_METRIC_REGISTRY`` (``polylogue/insights/measurement/
registered_metrics.py``) resolves through the real MCP ``get`` tool, not
just its own unit tests.
"""

from __future__ import annotations

import json
from pathlib import Path

from polylogue.insights.measurement.registered_metrics import SESSION_COST_USD_METRIC
from tests.infra.mcp import MCPServerUnderTest, invoke_surface
from tests.unit.mcp.test_contract_evidence import _seeded_runtime_services


def test_mcp_get_resolves_metric_ref_by_registered_name(mcp_server: MCPServerUnderTest, tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    with _seeded_runtime_services(archive_root):
        result = invoke_surface(mcp_server._tool_manager._tools["get"].fn, ref="metric:session_cost_usd")

    body = json.loads(result)
    assert body["ref"] == SESSION_COST_USD_METRIC.ref
    assert body["definition"]["construct"] == SESSION_COST_USD_METRIC.construct


def test_mcp_get_resolves_metric_ref_by_content_hash(mcp_server: MCPServerUnderTest, tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    with _seeded_runtime_services(archive_root):
        result = invoke_surface(mcp_server._tool_manager._tools["get"].fn, ref=SESSION_COST_USD_METRIC.ref)

    body = json.loads(result)
    assert body["definition"]["output_schema"] == "usd:float"


def test_mcp_get_unknown_metric_ref_returns_typed_not_found(mcp_server: MCPServerUnderTest, tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    with _seeded_runtime_services(archive_root):
        result = invoke_surface(mcp_server._tool_manager._tools["get"].fn, ref="metric:no-such-metric")

    body = json.loads(result)
    assert body.get("code") == "not_found", body
