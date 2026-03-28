"""Shared MCP test helpers and surface contracts."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from polylogue.lib.models import Conversation, Message

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
<<<<<<< HEAD
||||||| parent of c5d6c6a9 (refactor: narrow governance/health/repair (27 files deleted))
    "session_profile",
    "session_profiles",
    "session_enrichments",
    "session_work_events",
    "session_phases",
    "session_tag_rollups",
    "work_threads",
    "day_session_summaries",
    "week_session_summaries",
    "maintenance_runs",
=======
    "session_profile",
    "session_profiles",
    "session_enrichments",
    "session_work_events",
    "session_phases",
    "session_tag_rollups",
    "work_threads",
    "day_session_summaries",
    "week_session_summaries",
>>>>>>> c5d6c6a9 (refactor: narrow governance/health/repair (27 files deleted))
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


def invoke_surface(fn, /, *args, **kwargs):
    """Call an MCP surface whether it is sync or async."""
    result = fn(*args, **kwargs)
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result


async def invoke_surface_async(fn, /, *args, **kwargs):
    """Await an MCP surface from async tests."""
    result = fn(*args, **kwargs)
    if asyncio.iscoroutine(result):
        return await result
    return result


def make_repo_mock() -> MagicMock:
    """Create a repository mock with async methods used by MCP surfaces."""
    repo = MagicMock()
    repo.list = AsyncMock(return_value=[])
    repo.search = AsyncMock(return_value=[])
    repo.view = AsyncMock(return_value=None)
    repo.get = AsyncMock(return_value=None)
    repo.resolve_id = AsyncMock(return_value=None)
    repo.get_archive_stats = AsyncMock(return_value=MagicMock())
    repo.get_summary = AsyncMock(return_value=None)
    repo.get_session_tree = AsyncMock(return_value=[])
    repo.add_tag = AsyncMock(return_value=None)
    repo.remove_tag = AsyncMock(return_value=None)
    repo.list_tags = AsyncMock(return_value={})
    repo.get_metadata = AsyncMock(return_value={})
    repo.update_metadata = AsyncMock(return_value=None)
    repo.delete_metadata = AsyncMock(return_value=None)
    repo.delete_conversation = AsyncMock(return_value=False)
    repo.backend = MagicMock()
    repo.queries = MagicMock()
    repo.queries.get_conversation_stats = AsyncMock(return_value={})
    repo.queries.get_stats_by = AsyncMock(return_value={})
    return repo


def make_mock_filter(results=None, **method_overrides):
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
    return Conversation(
        id="test:conv-123",
        provider="chatgpt",
        title="Test Conversation",
        messages=[
            Message(
                id="msg-1",
                role="user",
                text="Hello, how are you?",
                timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            ),
            Message(
                id="msg-2",
                role="assistant",
                text="I'm doing well, thank you!",
                timestamp=datetime(2024, 1, 15, 10, 30, 30, tzinfo=timezone.utc),
            ),
        ],
        created_at=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        updated_at=datetime(2024, 1, 15, 10, 31, 0, tzinfo=timezone.utc),
    )
