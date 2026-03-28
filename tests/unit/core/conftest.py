"""Fixtures for core module tests.

Module-level fixtures for test consolidation:
- populated_db: Database with test conversations
- mock_schema_dir: Mock schema directory for validation tests
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
