"""Fixtures for storage tests.

Extracted from the monolithic test_search_index.py to support module-level
fixtures used across storage test files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.sources.parsers.base import ParsedConversation, ParsedMessage
from polylogue.sources.parsers.claude import SessionIndexEntry


@pytest.fixture
def archive_root(tmp_path):
    """Create an archive root directory with render subdirectory."""
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "render").mkdir()
    return archive


@pytest.fixture
def sample_sessions_index(tmp_path):
    """Create a sample sessions-index.json file."""
    index_data = {
        "version": 1,
        "entries": [
            {
                "sessionId": "abc123-def456",
                "fullPath": str(tmp_path / "abc123-def456.jsonl"),
                "fileMtime": 1700000000000,
                "firstPrompt": "How do I fix the bug in auth?",
                "summary": "Fixed authentication bug in login flow",
                "messageCount": 12,
                "created": "2024-01-15T10:30:00.000Z",
                "modified": "2024-01-15T11:45:00.000Z",
                "gitBranch": "feature/auth-fix",
                "projectPath": "/home/user/myproject",
                "isSidechain": False,
            },
            {
                "sessionId": "ghi789-jkl012",
                "fullPath": str(tmp_path / "ghi789-jkl012.jsonl"),
                "firstPrompt": "No prompt",
                "summary": "User Exits CLI Session",
                "messageCount": 2,
                "created": "2024-01-14T08:00:00.000Z",
                "modified": "2024-01-14T08:01:00.000Z",
                "gitBranch": "main",
                "projectPath": "/home/user/myproject",
                "isSidechain": False,
            },
            {
                "sessionId": "sidechain-test",
                "fullPath": str(tmp_path / "sidechain-test.jsonl"),
                "firstPrompt": "Analyze this code",
                "summary": "Sidechain analysis task",
                "messageCount": 5,
                "created": "2024-01-16T14:00:00.000Z",
                "modified": "2024-01-16T14:30:00.000Z",
                "gitBranch": "main",
                "projectPath": "/home/user/myproject",
                "isSidechain": True,
            },
        ],
    }

    index_path = tmp_path / "sessions-index.json"
    index_path.write_text(json.dumps(index_data))
    return index_path


@pytest.fixture
def sample_conversation():
    """Create a sample parsed conversation."""
    return ParsedConversation(
        provider_name="claude-code",
        provider_conversation_id="abc123-def456",
        title="abc123-def456",  # Default title is session ID
        created_at=None,
        updated_at=None,
        messages=[
            ParsedMessage(
                provider_message_id="msg-1",
                role="user",
                text="How do I fix the bug?",
                timestamp="1700000000",
            ),
            ParsedMessage(
                provider_message_id="msg-2",
                role="assistant",
                text="Let me help you fix that bug.",
                timestamp="1700000001",
            ),
        ],
    )
