"""Shared MCP test helpers and surface contracts."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from datetime import datetime, timezone
from typing import Any, TypeVar, cast
from unittest.mock import AsyncMock, MagicMock

from polylogue.lib.models import Conversation
from polylogue.types import Provider
from tests.infra.builders import make_conv, make_msg

EXPECTED_TOOL_NAMES = {
    "search",
    "list_conversations",
    "get_conversation",
    "stats",
    "add_tag",
    "remove_tag",
    "list_tags",
    "get_metadata",
    "set_metadata",
    "delete_metadata",
    "delete_conversation",
    "get_conversation_summary",
    "get_session_tree",
    "get_stats_by",
    "health_check",
    "rebuild_index",
    "update_index",
    "export_conversation",
    "session_profile",
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
    "polylogue://health",
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


def invoke_surface(
    fn: Callable[..., SurfaceResult | Awaitable[SurfaceResult]],
    /,
    *args: object,
    **kwargs: object,
) -> SurfaceResult:
    """Call an MCP surface whether it is sync or async."""
    result = fn(*args, **kwargs)
    if inspect.iscoroutine(result):
        return asyncio.run(cast(Coroutine[Any, Any, SurfaceResult], result))
    return cast(SurfaceResult, result)


async def invoke_surface_async(
    fn: Callable[..., SurfaceResult | Awaitable[SurfaceResult]],
    /,
    *args: object,
    **kwargs: object,
) -> SurfaceResult:
    """Await an MCP surface from async tests."""
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        return await cast(Awaitable[SurfaceResult], result)
    return result


def make_query_store_mock() -> MagicMock:
    """Create a query-store mock matching the current MCP read/query seam."""
    store = MagicMock()
    store.list = AsyncMock(return_value=[])
    store.list_summaries = AsyncMock(return_value=[])
    store.list_summaries_by_query = AsyncMock(return_value=[])
    store.search = AsyncMock(return_value=[])
    store.search_summaries = AsyncMock(return_value=[])
    store.view = AsyncMock(return_value=None)
    store.get = AsyncMock(return_value=None)
    store.get_eager = AsyncMock(return_value=None)
    store.resolve_id = AsyncMock(return_value=None)
    store.delete_conversation = AsyncMock(return_value=False)
    return store


def make_tag_store_mock() -> MagicMock:
    """Create a tag/metadata store mock matching the current MCP mutation seam."""
    store = MagicMock()
    store.add_tag = AsyncMock(return_value=None)
    store.remove_tag = AsyncMock(return_value=None)
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
    operations.storage_stats = AsyncMock(return_value=MagicMock())
    return operations


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
