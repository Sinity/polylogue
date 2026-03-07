"""Shared fixtures for integration tests."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from polylogue.lib.models import Conversation, Message


@pytest.fixture
def mock_repo():
    """Create a mock ConversationRepository."""
    repo = MagicMock()
    repo.list = AsyncMock(return_value=[])
    repo.search = AsyncMock(return_value=[])
    repo.view = AsyncMock(return_value=None)
    repo.get = AsyncMock(return_value=None)
    repo.resolve_id = AsyncMock(return_value=None)
    repo.get_archive_stats = AsyncMock(return_value=MagicMock())
    repo.get_summary = AsyncMock(return_value=None)
    repo.get_session_tree = AsyncMock(return_value=[])
    repo.get_stats_by = AsyncMock(return_value={})
    repo.add_tag = AsyncMock(return_value=None)
    repo.remove_tag = AsyncMock(return_value=None)
    repo.list_tags = AsyncMock(return_value={})
    repo.get_metadata = AsyncMock(return_value={})
    repo.update_metadata = AsyncMock(return_value=None)
    repo.delete_metadata = AsyncMock(return_value=None)
    repo.delete_conversation = AsyncMock(return_value=False)
    return repo


def make_mock_filter(results=None, **method_overrides):
    """Create a pre-configured mock ConversationFilter.

    Args:
        results: List of conversations to return from .list()
        **method_overrides: Set side_effect for any method

    Returns:
        Configured MagicMock filter instance with chaining support.
    """
    f = MagicMock()
    for method in ("provider", "contains", "after", "before", "tags", "title", "since", "limit", "tag"):
        getattr(f, method).return_value = f
    f.list = AsyncMock(return_value=results or [])
    for method_name, override_value in method_overrides.items():
        method = getattr(f, method_name)
        if isinstance(override_value, Exception):
            method.side_effect = override_value
        else:
            method.return_value = override_value
    return f


@pytest.fixture
def simple_conversation():
    """Create a simple conversation for testing."""
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
