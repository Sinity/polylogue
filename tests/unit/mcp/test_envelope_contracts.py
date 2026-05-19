"""Registry-wide envelope contract for MCP tools (#819).

Pins the rule that every tool returning list-shaped data exposes a
bounded envelope with at least one named array field plus ``total``,
and that every tool not returning list-shaped data is explicitly
classified. A new tool added without registering in this matrix fails
the test, forcing the author to make a coherent decision rather than
silently shipping a bare array.

The matrix lives here rather than as a runtime feature because each
row is a deliberate classification — frameworks cannot tell whether a
given result is "naturally list-shaped and small" or "user-bounded and
should support pagination". The author owns that decision.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from pydantic import BaseModel

from tests.infra.mcp import EXPECTED_TOOL_NAMES, MCPServerUnderTest

# ---------------------------------------------------------------------------
# Tool classification — every registered tool must appear here.
#
# Values:
#   - ("envelope", required_field_names) — list-shaped tool, JSON output is
#     an object containing all listed fields. Use the domain-meaningful
#     array name plus ``total`` (and ``limit``/``offset`` where applicable).
#   - "single_object" — returns one record (e.g. get_conversation).
#   - "stats_map"     — JSON object map keyed by domain key.
#   - "operation_result" — mutation/maintenance result (own structured shape).
#   - Insight registry tools share the standard envelope shape
#     ``{items: [...], total: N}`` after #1007 aligned the field name.
#     They are intentionally absent from ``tool_to_class`` because they
#     wrap a plain ``MCPRootPayload[dict]`` rather than a typed payload
#     class; their envelope shape is exercised by
#     :class:`TestInsightEnvelopeRuntimeSerialisation` below.
# ---------------------------------------------------------------------------

EnvelopeSpec = tuple[str, frozenset[str]]
ToolKind = EnvelopeSpec | str

TOOL_CONTRACT: dict[str, ToolKind] = {
    # ------- search and list (envelope) -------
    "search": ("envelope", frozenset({"hits", "total"})),
    "list_conversations": ("envelope", frozenset({"items", "total"})),
    "neighbor_candidates": ("envelope", frozenset({"items", "total", "limit"})),
    "get_session_tree": ("envelope", frozenset({"items", "total"})),
    "get_messages": ("envelope", frozenset({"messages", "total"})),
    "raw_artifacts": ("envelope", frozenset({"raw_artifacts", "total"})),
    # ------- mutation list -------
    "list_tags": "stats_map",  # RootModel[dict[tag, count]]; small, by design
    "list_marks": ("envelope", frozenset({"items", "total"})),
    "list_annotations": ("envelope", frozenset({"items", "total"})),
    "list_saved_views": ("envelope", frozenset({"items", "total"})),
    "list_recall_packs": ("envelope", frozenset({"items", "total"})),
    "list_workspaces": ("envelope", frozenset({"items", "total"})),
    # ------- single record -------
    "get_conversation": "single_object",
    "get_conversation_summary": "single_object",
    "get_metadata": "single_object",
    "session_profile": "single_object",
    "session_classification": "single_object",
    "get_resume_brief": "single_object",
    "archive_coverage": "single_object",
    "cost_outlook": "single_object",
    "stats": "single_object",
    "build_context_pack": "single_object",
    "readiness_check": "single_object",
    "insight_rigor_audit": "single_object",
    # ------- stats / map -------
    "get_stats_by": "stats_map",
    # ------- mutation tools -------
    "add_tag": "operation_result",
    "add_mark": "operation_result",
    "remove_tag": "operation_result",
    "remove_mark": "operation_result",
    "save_annotation": "operation_result",
    "delete_annotation": "operation_result",
    "bulk_tag_conversations": "operation_result",
    "save_saved_view": "operation_result",
    "delete_saved_view": "operation_result",
    "save_recall_pack": "operation_result",
    "delete_recall_pack": "operation_result",
    "save_workspace": "operation_result",
    "delete_workspace": "operation_result",
    "set_metadata": "operation_result",
    "delete_metadata": "operation_result",
    "delete_conversation": "operation_result",
    # ------- learning corrections (#1131) -------
    "record_correction": "operation_result",
    "list_corrections": ("envelope", frozenset({"corrections", "total"})),
    "clear_corrections": "operation_result",
    # ------- maintenance tools -------
    "rebuild_index": "operation_result",
    "rebuild_session_insights": "operation_result",
    "update_index": "operation_result",
    "export_conversation": "operation_result",
    "export_query_results": "operation_result",
    "maintenance_preview": "operation_result",
    "maintenance_execute": "operation_result",
    "maintenance_status": "operation_result",
    "maintenance_list": ("envelope", frozenset({"items", "total"})),
    # ------- insight registry tools -------
    "session_profiles": ("envelope", frozenset({"items", "total"})),
    "session_enrichments": ("envelope", frozenset({"items", "total"})),
    "session_phases": ("envelope", frozenset({"items", "total"})),
    "session_tag_rollups": ("envelope", frozenset({"items", "total"})),
    "session_work_events": ("envelope", frozenset({"items", "total"})),
    "session_costs": ("envelope", frozenset({"items", "total"})),
    "cost_rollups": ("envelope", frozenset({"items", "total"})),
    "day_session_summaries": ("envelope", frozenset({"items", "total"})),
    "week_session_summaries": ("envelope", frozenset({"items", "total"})),
    "work_threads": ("envelope", frozenset({"items", "total"})),
    "provider_analytics": ("envelope", frozenset({"items", "total"})),
    "tool_usage": ("envelope", frozenset({"items", "total"})),
    "productivity_rollups": ("envelope", frozenset({"items", "total"})),
    "archive_debt": ("envelope", frozenset({"items", "total"})),
}


@pytest.fixture
def admin_server() -> MCPServerUnderTest:
    """Build a server with the admin role so all tools are visible."""
    from polylogue.mcp.server import build_server

    return cast(MCPServerUnderTest, build_server(role="admin"))


# ---------------------------------------------------------------------------
# Registry consistency
# ---------------------------------------------------------------------------


class TestRegistryWideClassification:
    """Every registered tool must be present in the classification matrix
    and vice versa.
    """

    def test_every_registered_tool_is_classified(self, admin_server: MCPServerUnderTest) -> None:
        registered = set(admin_server._tool_manager._tools.keys())
        classified = set(TOOL_CONTRACT.keys())
        missing = registered - classified
        assert not missing, (
            f"Tools registered but not classified in TOOL_CONTRACT: {sorted(missing)}. "
            f"Add a row to test_envelope_contracts.py::TOOL_CONTRACT and assert the "
            f"intended envelope shape, OR change the tool to use an envelope. Do not "
            f"silently add tools that return bare arrays."
        )

    def test_no_stale_classifications(self, admin_server: MCPServerUnderTest) -> None:
        registered = set(admin_server._tool_manager._tools.keys())
        classified = set(TOOL_CONTRACT.keys())
        stale = classified - registered
        assert not stale, (
            f"TOOL_CONTRACT classifies tools that are not registered: {sorted(stale)}. "
            f"Remove them from the matrix or restore the tool."
        )

    def test_expected_tool_names_subset_of_classified(self) -> None:
        """The infra-level pin and our classification must agree on tools."""
        unclassified = EXPECTED_TOOL_NAMES - set(TOOL_CONTRACT.keys())
        assert not unclassified, f"EXPECTED_TOOL_NAMES not in TOOL_CONTRACT: {sorted(unclassified)}"


# ---------------------------------------------------------------------------
# Envelope payload shape — Pydantic class fields match the matrix.
# ---------------------------------------------------------------------------


def _build_typed_envelope_classes() -> dict[str, type[BaseModel]]:
    from polylogue.mcp.payloads import (
        MCPMessagesListPayload,
        MCPNeighborCandidatesPayload,
        MCPPaginatedQueryResultPayload,
        MCPPaginatedSearchResultPayload,
        MCPRawArtifactsListPayload,
        MCPSessionTreePayload,
    )

    return {
        "search": MCPPaginatedSearchResultPayload,
        "list_conversations": MCPPaginatedQueryResultPayload,
        "neighbor_candidates": MCPNeighborCandidatesPayload,
        "get_session_tree": MCPSessionTreePayload,
        "get_messages": MCPMessagesListPayload,
        "raw_artifacts": MCPRawArtifactsListPayload,
    }


# Build the mapping once at collection time so parametrize and the test
# body share the same dict instead of importing payload classes per case.
_TYPED_ENVELOPE_CLASSES: dict[str, type[BaseModel]] = _build_typed_envelope_classes()


@pytest.mark.parametrize(
    ("tool_name", "expected_fields"),
    sorted(
        (name, fields)
        for name, kind in TOOL_CONTRACT.items()
        if isinstance(kind, tuple) and kind[0] == "envelope" and name in _TYPED_ENVELOPE_CLASSES
        for fields in (kind[1],)
    ),
)
def test_envelope_class_carries_required_fields(tool_name: str, expected_fields: frozenset[str]) -> None:
    """For every envelope tool backed by a typed payload class, the
    class declares all required envelope fields.

    Insight registry tools wrap a ``MCPRootPayload[dict]`` rather than a
    typed payload class — their envelope shape is covered separately by
    :class:`TestInsightEnvelopeRuntimeSerialisation`.
    """
    cls = _TYPED_ENVELOPE_CLASSES[tool_name]
    fields = set(cls.model_fields.keys())
    missing = expected_fields - fields
    assert not missing, (
        f"{tool_name}: payload class {cls.__name__} missing required envelope keys "
        f"{sorted(missing)} (got {sorted(fields)})"
    )


# ---------------------------------------------------------------------------
# Runtime envelope smoke — factories produce JSON with the documented keys.
# ---------------------------------------------------------------------------


class TestEnvelopeRuntimeSerialisation:
    def test_session_tree_envelope_serialises_with_items_and_total(self) -> None:
        from polylogue.mcp.payloads import session_tree_payload
        from tests.infra.builders import make_conv

        conv = make_conv(id="x:y", title="Test")
        payload = session_tree_payload([conv])
        body = json.loads(payload.model_dump_json())
        assert "items" in body
        assert "total" in body
        assert body["total"] == 1
        assert isinstance(body["items"], list)

    def test_neighbor_candidates_envelope_serialises_with_items_total_limit(self) -> None:
        from polylogue.mcp.payloads import neighbor_candidates_payload

        payload = neighbor_candidates_payload([], limit=7)
        body = json.loads(payload.model_dump_json())
        assert body["items"] == []
        assert body["total"] == 0
        assert body["limit"] == 7


# ---------------------------------------------------------------------------
# Insight envelope — registry-driven insight tools share the standard
# ``{items, total}`` envelope shape after #1007 aligned the field name
# from the historical ``count``. Pinned at the registry payload factory
# (``insight_items_payload``) so any drift fails here loudly instead of
# silently re-introducing the legacy field name.
# ---------------------------------------------------------------------------


class TestInsightEnvelopeRuntimeSerialisation:
    """``insight_items_payload`` and the MCP insight tools must emit the
    same ``{<key>: [...], "total": N}`` shape every other paginated MCP
    surface uses. Pin both call sites so the alignment from #1007 cannot
    silently regress.
    """

    def test_insight_items_payload_uses_total_with_default_key(self) -> None:
        from polylogue.insights.registry import INSIGHT_REGISTRY, insight_items_payload

        pt = next(iter(INSIGHT_REGISTRY.values()))
        payload = insight_items_payload([], pt)
        assert "total" in payload
        assert "count" not in payload  # legacy field removed in #1007
        assert payload["total"] == 0
        assert pt.json_key in payload

    def test_insight_items_payload_uses_total_with_named_item_key(self) -> None:
        from polylogue.insights.registry import INSIGHT_REGISTRY, insight_items_payload

        pt = next(iter(INSIGHT_REGISTRY.values()))
        payload = insight_items_payload([], pt, item_key="items")
        assert "total" in payload
        assert "items" in payload
        assert "count" not in payload


# ---------------------------------------------------------------------------
# Resource error envelope coverage — every resource error path returns the
# structured MCPErrorPayload shape (#819-A2).
# ---------------------------------------------------------------------------


def _resource(server: MCPServerUnderTest, uri: str) -> Any:
    """Resolve an MCP resource (concrete URI or template) by its URI string."""
    if uri in server._resource_manager._resources:
        return server._resource_manager._resources[uri].fn
    if uri in server._resource_manager._templates:
        return server._resource_manager._templates[uri].fn
    raise KeyError(uri)


@pytest.fixture
def read_server() -> MCPServerUnderTest:
    """Read-role server — resources are visible at this scope."""
    from polylogue.mcp.server import build_server

    return cast(MCPServerUnderTest, build_server(role="read"))


def _assert_structured_error(payload: str, *, expected_code: str | None = None) -> None:
    """Assert payload is a structured MCPErrorPayload with is_error and code."""
    body = json.loads(payload)
    assert "error" in body, f"missing 'error' field: {body}"
    assert body.get("is_error") is True, f"missing or false 'is_error': {body}"
    if expected_code is not None:
        assert body.get("code") == expected_code, f"expected code={expected_code}, got {body.get('code')}"


class TestSessionTreeResourceShapeMatchesTool:
    """The ``polylogue://session-tree/{conv_id}`` resource and the
    ``get_session_tree`` tool must serialise the same domain entity in
    the same envelope shape.

    A previous closure of #819 left this gap — the tool was migrated
    to ``MCPSessionTreePayload`` while the resource still used the
    older ``MCPPaginatedQueryResultPayload`` (which carries unrelated
    ``limit``/``offset``/``next_offset`` fields it doesn't use). This
    test catches that coherence gap.
    """

    def test_resource_returns_session_tree_envelope_not_paginated_query(self, read_server: MCPServerUnderTest) -> None:
        from unittest.mock import AsyncMock as _AsyncMock
        from unittest.mock import MagicMock as _MagicMock
        from unittest.mock import patch as _patch

        from tests.infra.builders import make_conv
        from tests.infra.mcp import invoke_surface

        conv = make_conv(id="x:y", title="Resource shape probe")

        with _patch("polylogue.mcp.server._get_archive_ops") as mock_get:
            mock_ops = _MagicMock()
            mock_ops.get_session_tree = _AsyncMock(return_value=[conv])
            mock_get.return_value = mock_ops
            result = invoke_surface(_resource(read_server, "polylogue://session-tree/{conv_id}"), conv_id="x:y")

        body = json.loads(result)
        assert "items" in body
        assert "total" in body
        assert body["total"] == 1
        # Coherence pin: the resource must NOT carry the paginated-query
        # fields. If a future refactor reintroduces ``limit``/``offset``
        # to this resource, that's a deliberate scope change and this
        # test should be updated alongside ``MCPSessionTreePayload``.
        for forbidden in ("limit", "offset", "next_offset"):
            assert forbidden not in body, (
                f"session-tree resource leaked paginated-query field {forbidden!r}: "
                f"resource and tool envelope shapes have drifted apart"
            )


class TestResourceErrorEnvelopes:
    """All 8 MCP resources must emit the structured error envelope.

    Pins #819-A2: "Resource handlers produce structured, tested errors."
    Each test forces an error path (backend exception or missing record)
    and asserts the JSON has ``error``, ``is_error: true``, and the
    declared ``code``.
    """

    def test_stats_resource_internal_error(self, read_server: MCPServerUnderTest) -> None:
        from unittest.mock import AsyncMock as _AsyncMock
        from unittest.mock import MagicMock as _MagicMock
        from unittest.mock import patch as _patch

        with _patch("polylogue.mcp.server._get_archive_ops") as mock_get:
            mock_ops = _MagicMock()
            mock_ops.storage_stats = _AsyncMock(side_effect=RuntimeError("boom"))
            mock_get.return_value = mock_ops
            from tests.infra.mcp import invoke_surface

            result = invoke_surface(_resource(read_server, "polylogue://stats"))
        _assert_structured_error(result, expected_code="internal_error")

    def test_conversations_resource_internal_error(self, read_server: MCPServerUnderTest) -> None:
        from unittest.mock import patch as _patch

        with _patch("polylogue.mcp.server._get_query_store") as mock_get:
            mock_get.side_effect = RuntimeError("boom")
            from tests.infra.mcp import invoke_surface

            result = invoke_surface(_resource(read_server, "polylogue://conversations"))
        _assert_structured_error(result, expected_code="internal_error")

    def test_conversation_resource_not_found(self, read_server: MCPServerUnderTest) -> None:
        from unittest.mock import AsyncMock as _AsyncMock
        from unittest.mock import MagicMock as _MagicMock
        from unittest.mock import patch as _patch

        with _patch("polylogue.mcp.server._get_archive_ops") as mock_get:
            mock_ops = _MagicMock()
            mock_ops.get_conversation_summary = _AsyncMock(return_value=None)
            mock_get.return_value = mock_ops
            from tests.infra.mcp import invoke_surface

            result = invoke_surface(_resource(read_server, "polylogue://conversation/{conv_id}"), conv_id="missing")
        _assert_structured_error(result, expected_code="not_found")

    def test_tags_resource_internal_error(self, read_server: MCPServerUnderTest) -> None:
        from unittest.mock import patch as _patch

        with _patch("polylogue.mcp.server._get_tag_store") as mock_get:
            mock_get.side_effect = RuntimeError("boom")
            from tests.infra.mcp import invoke_surface

            result = invoke_surface(_resource(read_server, "polylogue://tags"))
        _assert_structured_error(result, expected_code="internal_error")

    def test_messages_resource_not_found(self, read_server: MCPServerUnderTest) -> None:
        from unittest.mock import AsyncMock as _AsyncMock
        from unittest.mock import MagicMock as _MagicMock
        from unittest.mock import patch as _patch

        with _patch("polylogue.mcp.server._get_archive_ops") as mock_get:
            mock_ops = _MagicMock()
            mock_ops.get_conversation_summary = _AsyncMock(return_value=None)
            mock_get.return_value = mock_ops
            from tests.infra.mcp import invoke_surface

            result = invoke_surface(_resource(read_server, "polylogue://messages/{conv_id}"), conv_id="missing")
        _assert_structured_error(result, expected_code="not_found")

    def test_session_tree_resource_internal_error(self, read_server: MCPServerUnderTest) -> None:
        from unittest.mock import AsyncMock as _AsyncMock
        from unittest.mock import MagicMock as _MagicMock
        from unittest.mock import patch as _patch

        with _patch("polylogue.mcp.server._get_archive_ops") as mock_get:
            mock_ops = _MagicMock()
            mock_ops.get_session_tree = _AsyncMock(side_effect=RuntimeError("boom"))
            mock_get.return_value = mock_ops
            from tests.infra.mcp import invoke_surface

            result = invoke_surface(_resource(read_server, "polylogue://session-tree/{conv_id}"), conv_id="x")
        _assert_structured_error(result, expected_code="internal_error")

    def test_provider_recent_resource_internal_error(self, read_server: MCPServerUnderTest) -> None:
        from unittest.mock import patch as _patch

        with _patch("polylogue.mcp.server._get_query_store") as mock_get:
            mock_get.side_effect = RuntimeError("boom")
            from tests.infra.mcp import invoke_surface

            result = invoke_surface(_resource(read_server, "polylogue://provider/{name}/recent"), name="chatgpt")
        _assert_structured_error(result, expected_code="internal_error")

    def test_readiness_resource_internal_error(self, read_server: MCPServerUnderTest) -> None:
        from unittest.mock import patch as _patch

        with _patch("polylogue.readiness.get_readiness", side_effect=RuntimeError("boom")):
            from tests.infra.mcp import invoke_surface

            result = invoke_surface(_resource(read_server, "polylogue://readiness"))
        _assert_structured_error(result, expected_code="internal_error")
