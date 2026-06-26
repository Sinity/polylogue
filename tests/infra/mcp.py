"""Shared MCP test helpers and surface contracts."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable
from unittest.mock import AsyncMock, MagicMock

from polylogue.archive.models import Session
from polylogue.core.enums import Provider
from tests.infra.builders import make_conv, make_msg

EXPECTED_TOOL_NAMES = {
    "search",
    "query_units",
    "resolve_ref",
    "explain_import",
    "list_sessions",
    "compile_context",
    "build_context_pack",
    "blackboard_list",
    "blackboard_post",
    "list_assertion_claims",
    "get_session",
    "neighbor_candidates",
    "stats",
    "embedding_status",
    "embedding_preflight",
    "facets",
    "add_tag",
    "remove_tag",
    "bulk_tag_sessions",
    "list_tags",
    "list_marks",
    "add_mark",
    "remove_mark",
    "list_annotations",
    "save_annotation",
    "delete_annotation",
    "list_saved_views",
    "save_saved_view",
    "delete_saved_view",
    "get_metadata",
    "set_metadata",
    "delete_metadata",
    "delete_session",
    "get_session_summary",
    "get_session_tree",
    "get_session_topology",
    "get_logical_session",
    "get_stats_by",
    "list_read_view_profiles",
    "get_recovery_report",
    "get_recovery_work_packet",
    "explain_query_expression",
    "query_completions",
    "readiness_check",
    "rebuild_index",
    "update_index",
    "export_session",
    "export_query_results",
    "export_sanitized",
    "get_postmortem_bundle",
    "get_pathologies",
    "rebuild_session_insights",
    "maintenance_preview",
    "maintenance_execute",
    "maintenance_status",
    "maintenance_list",
    "session_profile",
    "session_latency_profile",
    "tool_call_latency_distribution",
    "find_stuck_sessions",
    "workflow_shape_distribution",
    "find_abandoned_sessions",
    "get_resume_brief",
    "find_resume_candidates",
    "aggregate_sessions",
    "compare_sessions",
    "find_similar_sessions",
    "correlate_session",
    "correlate_sessions",
    "session_tool_timing",
    "cost_outlook",
    "archive_debt",
    "session_profiles",
    "session_work_events",
    "session_phases",
    "session_tag_rollups",
    "threads",
    "archive_coverage",
}

EXPECTED_RESOURCE_URIS = {
    "polylogue://stats",
    "polylogue://sessions",
    "polylogue://tags",
    "polylogue://readiness",
}

EXPECTED_RESOURCE_TEMPLATE_URIS = {
    "polylogue://session/{conv_id}",
}

EXPECTED_PROMPT_NAMES = {
    "analyze_errors",
    "summarize_week",
    "extract_code",
    "compare_sessions",
    "extract_patterns",
}

SurfaceResult = TypeVar("SurfaceResult")
MCPSurfaceHandler: TypeAlias = Callable[..., str | Awaitable[str]]


class RegisteredMCPSurface(Protocol):
    fn: MCPSurfaceHandler


class MCPToolManager(Protocol):
    _tools: Mapping[str, RegisteredMCPSurface]


class MCPResourceManager(Protocol):
    _resources: Mapping[str, RegisteredMCPSurface]
    _templates: Mapping[str, RegisteredMCPSurface]


class MCPPromptManager(Protocol):
    _prompts: Mapping[str, RegisteredMCPSurface]


@runtime_checkable
class MCPServerUnderTest(Protocol):
    _tool_manager: MCPToolManager
    _resource_manager: MCPResourceManager
    _prompt_manager: MCPPromptManager


async def _await_surface(result: Awaitable[SurfaceResult]) -> SurfaceResult:
    return await result


def invoke_surface(
    fn: Callable[..., SurfaceResult | Awaitable[SurfaceResult]],
    /,
    *args: object,
    **kwargs: object,
) -> SurfaceResult:
    """Call an MCP surface whether it is sync or async."""
    result = fn(*args, **kwargs)
    if isinstance(result, Awaitable):
        surface: SurfaceResult = asyncio.run(_await_surface(result))
        return surface
    return result


async def invoke_surface_async(
    fn: Callable[..., SurfaceResult | Awaitable[SurfaceResult]],
    /,
    *args: object,
    **kwargs: object,
) -> SurfaceResult:
    """Await an MCP surface from async tests."""
    result = fn(*args, **kwargs)
    if isinstance(result, Awaitable):
        return await result
    return result


def make_polylogue_mock(*, resolved_id: str | None = None) -> MagicMock:
    """Create a Polylogue facade mock matching the current MCP tool surface.

    Pass ``resolved_id`` to make ``get_session_summary`` return a summary
    whose ``.id`` is that value — the MCP mutation/read tools resolve a
    session by calling ``get_session_summary`` and reading ``.id``
    (the archive canonical-id seam). Default ``None`` is the "not found" path.
    """
    poly = MagicMock()
    poly.add_tag = AsyncMock()
    poly.remove_tag = AsyncMock()
    poly.list_tags = AsyncMock(return_value={})
    poly.get_metadata = AsyncMock(return_value={})
    poly.update_metadata = AsyncMock()
    poly.delete_session = AsyncMock(return_value=False)
    poly.list_marks = AsyncMock(return_value=[])
    poly.add_mark = AsyncMock(return_value=False)
    poly.remove_mark = AsyncMock(return_value=False)
    poly.list_annotations = AsyncMock(return_value=[])
    poly.save_annotation = AsyncMock(return_value=False)
    poly.delete_annotation = AsyncMock(return_value=False)
    poly.list_views = AsyncMock(return_value=[])
    poly.save_view = AsyncMock(return_value=False)
    poly.delete_view = AsyncMock(return_value=False)
    poly.get_session_summary = AsyncMock(return_value=(MagicMock(id=resolved_id) if resolved_id is not None else None))
    poly.get_session_stats = AsyncMock(return_value={})
    poly.get_stats_by = AsyncMock(return_value={})
    poly.get_session_tree = AsyncMock(return_value=[])
    poly.rebuild_index = AsyncMock(return_value=True)
    poly.update_index = AsyncMock(return_value=True)
    poly.get_index_status = AsyncMock(return_value={"exists": True, "count": 0})
    poly.get_raw_artifacts_for_session = AsyncMock(return_value=([], 0))
    poly.get_session_topology = AsyncMock(return_value=None)
    poly.get_logical_session = AsyncMock(return_value=None)
    poly.list_read_view_profiles = AsyncMock(return_value=[])
    poly.list_assertion_claims = AsyncMock(return_value=[])
    poly.list_assertion_claim_payloads = AsyncMock(return_value=[])
    poly.recovery_report = AsyncMock(return_value=None)
    poly.recovery_work_packet = AsyncMock(return_value=None)
    poly.compile_context = AsyncMock(return_value=None)
    poly.explain_import = AsyncMock(return_value=None)
    poly.explain_query_expression = AsyncMock(return_value={})
    poly.query_completions = AsyncMock(return_value={})
    poly.get_messages_paginated = AsyncMock(return_value=([], 0))
    poly.get_session = AsyncMock(return_value=None)
    poly.get_session_profile_insight = AsyncMock(return_value=None)
    poly.resume_brief = AsyncMock(return_value=None)
    poly.find_resume_candidates = AsyncMock(return_value=())
    poly.neighbor_candidates = AsyncMock(return_value=[])
    poly.rebuild_insights = AsyncMock(
        return_value=MagicMock(to_dict=MagicMock(return_value={}), total=MagicMock(return_value=0))
    )
    poly.list_session_profile_insights = AsyncMock(return_value=[])
    poly.list_session_work_event_insights = AsyncMock(return_value=[])
    poly.list_session_phase_insights = AsyncMock(return_value=[])
    poly.list_session_tag_rollup_insights = AsyncMock(return_value=[])
    poly.list_thread_insights = AsyncMock(return_value=[])
    poly.list_archive_coverage_insights = AsyncMock(return_value=[])
    poly.list_session_cost_insights = AsyncMock(return_value=[])
    poly.list_cost_rollup_insights = AsyncMock(return_value=[])
    poly.list_archive_debt_insights = AsyncMock(return_value=[])
    from polylogue.surfaces.payloads import ArchiveDebtListPayload, ArchiveDebtTotalsPayload

    poly.archive_debt = AsyncMock(
        return_value=ArchiveDebtListPayload(
            generated_at="2026-06-20T00:00:00+00:00",
            archive_root="/tmp/polylogue-test",
            rows=(),
            totals=ArchiveDebtTotalsPayload(),
        )
    )
    poly.cost_outlook = AsyncMock(return_value=None)
    # Typed mutation entrypoints (#862).
    from polylogue.surfaces.payloads import (
        BulkTagMutationResult,
        DeleteSessionResult,
        MetadataMutationResult,
    )

    poly.set_metadata = AsyncMock(return_value=MetadataMutationResult(outcome="set", session_id="", key=""))
    poly.delete_metadata = AsyncMock(return_value=MetadataMutationResult(outcome="deleted", session_id="", key=""))
    poly.delete_session_safe = AsyncMock(return_value=DeleteSessionResult(outcome="deleted", session_id=""))
    poly.bulk_tag_sessions = AsyncMock(
        return_value=BulkTagMutationResult(session_count=0, tag_count=0, affected_count=0, skipped_count=0)
    )
    return poly


def make_mock_filter(results: Sequence[object] | None = None, **method_overrides: object) -> MagicMock:
    """Create a chaining-capable SessionFilter mock."""
    filt = MagicMock()
    for method in (
        "contains",
        "exclude_text",
        "provider",
        "exclude_provider",
        "tag",
        "exclude_tag",
        "has",
        "title",
        "id",
        "since",
        "until",
        "sort",
        "reverse",
        "limit",
        "sample",
        "after",
        "before",
        "tags",
    ):
        getattr(filt, method).return_value = filt
    filt.list = AsyncMock(return_value=results or [])
    filt.count = AsyncMock(return_value=len(results or []))
    filt.delete = AsyncMock(return_value=0)
    for method_name, override_value in method_overrides.items():
        method = getattr(filt, method_name)
        if isinstance(override_value, Exception):
            method.side_effect = override_value
        else:
            method.return_value = override_value
    return filt


def make_simple_session() -> Session:
    """Return a representative session for MCP surface tests."""
    return make_conv(
        id="test:conv-123",
        provider=Provider.CHATGPT,
        title="Test Session",
        messages=[
            make_msg(
                id="msg-1",
                role="user",
                text="Hello, how are you?",
                timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            ),
            make_msg(
                id="msg-2",
                role="assistant",
                text="I'm doing well, thank you!",
                timestamp=datetime(2024, 1, 15, 10, 30, 30, tzinfo=timezone.utc),
            ),
        ],
        created_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        updated_at=datetime(2024, 1, 15, 10, 31, 0, tzinfo=timezone.utc),
    )
