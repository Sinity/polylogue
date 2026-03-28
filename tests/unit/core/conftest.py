"""Fixtures for core module tests.

Module-level fixtures for test consolidation:
- populated_db: Database with test conversations
- mock_schema_dir: Mock schema directory for validation tests
- make_filter_repo: Factory fixture for building a ConversationRepository with custom conversations
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.index import rebuild_index
from polylogue.storage.repository import ConversationRepository
from tests.infra.helpers import ConversationBuilder


@pytest.fixture
def populated_db(db_path: Path) -> Path:
    """Provide a database populated with test data."""
    (ConversationBuilder(db_path, "conv-1")
     .provider("chatgpt")
     .title("First Conversation")
     .add_message("msg-1-1", role="user", text="Hello")
     .add_message("msg-1-2", role="assistant", text="Hi there")
     .save())

    (ConversationBuilder(db_path, "conv-2")
     .provider("claude")
     .title("With Attachments")
     .add_message("msg-2-1", role="user", text="Here is an image")
     .add_attachment("att-1", message_id="msg-2-1", mime_type="image/png", size_bytes=1024)
     .save())

    return db_path


@pytest.fixture
def mock_schema_dir(tmp_path):
    """Create a mock schema directory with test schemas."""
    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()

    test_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "count": {"type": "integer"},
            "meta": {"type": "object", "properties": {"source": {"type": "string"}}, "additionalProperties": False},
        },
        "required": ["id"],
        "additionalProperties": False,
    }

    (schema_dir / "test-provider.schema.json").write_text(json.dumps(test_schema), encoding="utf-8")

    open_schema = {"type": "object", "properties": {"id": {"type": "string"}}, "additionalProperties": {}}
    (schema_dir / "open-provider.schema.json").write_text(json.dumps(open_schema), encoding="utf-8")

    return schema_dir


@pytest.fixture
def make_filter_repo(tmp_path):
    """Factory fixture: build a ConversationRepository with custom conversations.

    Usage::

        async def test_something(make_filter_repo):
            repo = make_filter_repo([
                {"id": "c1", "provider": "claude", "title": "Test Conv",
                 "messages": [{"id": "m1", "role": "user", "text": "hello"}]},
            ])
            result = await ConversationFilter(repo).provider("claude").list()

    Each conversation dict accepts:
        id (str): required
        provider (str): required
        title (str): optional
        messages (list[dict]): each with id, role, text (all optional except id)
        metadata (dict): optional
        created_at (str): optional ISO timestamp
        branch_type (str): optional
        parent_conversation (str): optional
    """

    def _factory(conversations: list[dict]) -> ConversationRepository:
        db_path = tmp_path / "test.db"
        with open_connection(db_path) as conn:
            rebuild_index(conn)

        for spec in conversations:
            cid = spec["id"]
            provider = spec.get("provider", "test")
            builder = ConversationBuilder(db_path, cid).provider(provider)

            if title := spec.get("title"):
                builder = builder.title(title)
            if metadata := spec.get("metadata"):
                builder = builder.metadata(metadata)
            if created_at := spec.get("created_at"):
                builder = builder.created_at(created_at)
            if branch_type := spec.get("branch_type"):
                builder = builder.branch_type(branch_type)
            if parent := spec.get("parent_conversation"):
                builder = builder.parent_conversation(parent)

            for msg in spec.get("messages", []):
                builder = builder.add_message(
                    msg["id"],
                    role=msg.get("role", "user"),
                    text=msg.get("text", ""),
                )

            builder.save()

        return ConversationRepository(SQLiteBackend(db_path))

    return _factory
