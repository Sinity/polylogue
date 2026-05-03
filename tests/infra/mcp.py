"""Shared MCP test helpers and surface contracts."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable
from unittest.mock import AsyncMock, MagicMock

from polylogue.archive.models import Conversation
from polylogue.types import Provider
from tests.infra.builders import make_conv, make_msg

EXPECTED_TOOL_NAMES = {
    "search",
    "list_conversations",
    "get_conversation",
    "neighbor_candidates",
    "stats",
    "add_tag",
    "remove_tag",
    "bulk_tag_conversations",
    "list_tags",
    "get_metadata",
    "set_metadata",
    "delete_metadata",
    "delete_conversation",
    "get_conversation_summary",
    "get_session_tree",
    "get_stats_by",
    "readiness_check",
    "rebuild_index",
    "update_index",
    "export_conversation",
    "export_query_results",
    "rebuild_session_insights",
    "session_profile",
    "archive_debt",
    "session_profiles",
    "session_enrichments",
    "session_work_events",
    "session_phases",
    "session_tag_rollups",
    "work_threads",
    "day_session_summaries",
    "week_session_summaries",
}

EXPECTED_RESOURCE_URIS = {
    "polylogue://stats",
    "polylogue://conversations",
    "polylogue://tags",
    "polylogue://readiness",
}

EXPECTED_RESOURCE_TEMPLATE_URIS = {
    "polylogue://conversation/{conv_id}",
}

EXPECTED_PROMPT_NAMES = {
    "analyze_errors",
    "summarize_week",
    "extract_code",
    "compare_conversations",
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


def make_query_store_mock(*, resolved_id: str | None = None) -> MagicMock:
    """Create a query-store mock matching the current MCP read/query seam.

    Pass ``resolved_id`` to make ``resolve_id`` return that value (the realistic
    "conversation found" path used by mutation tools that gate on resolution).
    Default ``None`` matches the unresolved path.
    """
    store = MagicMock()
    store.list = AsyncMock(return_value=[])
    store.list_summaries = AsyncMock(return_value=[])
    store.list_summaries_by_query = AsyncMock(return_value=[])
    store.search = AsyncMock(return_value=[])
    store.search_summaries = AsyncMock(return_value=[])
    store.view = AsyncMock(return_value=None)
    store.get = AsyncMock(return_value=None)
    store.get_eager = AsyncMock(return_value=None)
    store.resolve_id = AsyncMock(return_value=resolved_id)
    store.delete_conversation = AsyncMock(return_value=False)
    return store


def make_tag_store_mock() -> MagicMock:
    """Create a tag/metadata store mock matching the current MCP mutation seam."""
    store = MagicMock()
    store.add_tag = AsyncMock(return_value=None)
    store.remove_tag = AsyncMock(return_value=None)
    store.bulk_add_tags = AsyncMock(return_value=0)
    store.list_tags = AsyncMock(return_value={})
    store.get_metadata = AsyncMock(return_value={})
    store.update_metadata = AsyncMock(return_value=None)
    store.delete_metadata = AsyncMock(return_value=None)
    return store


def make_archive_ops_mock() -> MagicMock:
    """Create an archive-operations mock matching the current MCP read seam."""
    operations = MagicMock()
    operations.get_conversation = AsyncMock(return_value=None)
    operations.get_conversation_summary = AsyncMock(return_value=None)
    operations.get_conversation_stats = AsyncMock(return_value={})
    operations.get_session_tree = AsyncMock(return_value=[])
    operations.get_stats_by = AsyncMock(return_value={})
    operations.neighbor_candidates = AsyncMock(return_value=[])
    operations.storage_stats = AsyncMock(return_value=MagicMock())
    return operations


def make_polylogue_mock() -> MagicMock:
    """Create a Polylogue facade mock matching the current MCP tool surface."""
    poly = MagicMock()
    poly.add_tag = AsyncMock()
    poly.remove_tag = AsyncMock()
    poly.list_tags = AsyncMock(return_value={})
    poly.get_metadata = AsyncMock(return_value={})
    poly.update_metadata = AsyncMock()
    poly.delete_conversation = AsyncMock(return_value=False)
    poly.get_conversation_summary = AsyncMock(return_value=None)
    poly.get_conversation_stats = AsyncMock(return_value={})
    poly.get_session_tree = AsyncMock(return_value=[])
    poly.get_messages_paginated = AsyncMock(return_value=([], 0))
    poly.get_conversation = AsyncMock(return_value=None)
    poly.get_session_profile_insight = AsyncMock(return_value=None)
    poly.neighbor_candidates = AsyncMock(return_value=[])
    poly.rebuild_insights = AsyncMock(
        return_value=MagicMock(to_dict=MagicMock(return_value={}), total=MagicMock(return_value=0))
    )
    poly.list_session_profile_insights = AsyncMock(return_value=[])
    poly.list_session_enrichment_insights = AsyncMock(return_value=[])
    poly.list_session_work_event_insights = AsyncMock(return_value=[])
    poly.list_session_phase_insights = AsyncMock(return_value=[])
    poly.list_session_tag_rollup_insights = AsyncMock(return_value=[])
    poly.list_work_thread_insights = AsyncMock(return_value=[])
    poly.list_day_session_summary_insights = AsyncMock(return_value=[])
    poly.list_week_session_summary_insights = AsyncMock(return_value=[])
    poly.list_provider_analytics_insights = AsyncMock(return_value=[])
    poly.list_session_cost_insights = AsyncMock(return_value=[])
    poly.list_cost_rollup_insights = AsyncMock(return_value=[])
    poly.list_archive_debt_insights = AsyncMock(return_value=[])
    return poly


def make_mock_filter(results: Sequence[object] | None = None, **method_overrides: object) -> MagicMock:
    """Create a chaining-capable ConversationFilter mock."""
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
    for method_name, override_value in method_overrides.items():
        method = getattr(filt, method_name)
        if isinstance(override_value, Exception):
            method.side_effect = override_value
        else:
            method.return_value = override_value
    return filt


def make_simple_conversation() -> Conversation:
    """Return a representative conversation for MCP surface tests."""
    return make_conv(
        id="test:conv-123",
        provider=Provider.CHATGPT,
        title="Test Conversation",
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
