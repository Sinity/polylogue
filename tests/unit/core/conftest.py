"""Fixtures for core module tests.

Module-level fixtures for test consolidation:
- populated_db: Database with test sessions
- mock_schema_dir: Mock schema directory for validation tests
- make_filter_repo: Factory fixture for building a SessionRepository with custom sessions
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from polylogue.schemas.registry import SchemaRegistry
from tests.infra.storage_records import SessionBuilder


def _metadata_payload(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return {str(key): item for key, item in value.items()}


def _message_specs(value: object) -> list[dict[str, object]]:
    if value is None:
        return []
    assert isinstance(value, list)
    messages: list[dict[str, object]] = []
    for item in value:
        assert isinstance(item, dict)
        messages.append({str(key): field for key, field in item.items()})
    return messages


@pytest.fixture
def populated_db(db_path: Path) -> Path:
    """Provide a database populated with test data."""
    (
        SessionBuilder(db_path, "conv-1")
        .provider("chatgpt")
        .title("First Session")
        .add_message("msg-1-1", role="user", text="Hello")
        .add_message("msg-1-2", role="assistant", text="Hi there")
        .save()
    )

    (
        SessionBuilder(db_path, "conv-2")
        .provider("claude-ai")
        .title("With Attachments")
        .add_message("msg-2-1", role="user", text="Here is an image")
        .add_attachment("att-1", message_id="msg-2-1", mime_type="image/png", size_bytes=1024)
        .save()
    )

    return db_path


@pytest.fixture
def mock_schema_dir(tmp_path: Path) -> Path:
    """Create a mock schema directory with test schemas."""
    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()
    registry = SchemaRegistry(storage_root=schema_dir)

    test_schema: dict[str, object] = {
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

    open_schema: dict[str, object] = {
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "additionalProperties": {},
    }
    registry.write_schema_version("chatgpt", "v1", test_schema)
    registry.write_schema_version("codex", "v1", open_schema)

    return schema_dir


@pytest.fixture
def make_filter_repo(tmp_path: Path) -> Callable[[list[dict[str, object]]], Path]:
    """Factory fixture: build a archive with custom sessions.

    Returns the archive root, which is what the native ``SessionFilter``
    reads from.

    Usage::

        async def test_something(make_filter_repo):
            root = make_filter_repo([
                {"id": "c1", "provider": "claude-ai", "title": "Test Conv",
                 "messages": [{"id": "m1", "role": "user", "text": "hello"}]},
            ])
            result = await SessionFilter(archive_root=root).origin("claude-ai-export").list()

    Each session dict accepts:
        id (str): required
        provider (str): required
        title (str): optional
        messages (list[dict]): each with id, role, text (all optional except id)
        metadata (dict): optional
        created_at (str): optional ISO timestamp
        branch_type (str): optional
        parent_session (str): optional
    """

    def _factory(sessions: list[dict[str, object]]) -> Path:
        root = tmp_path / "make_filter_archive"
        root.mkdir(exist_ok=True)
        db_path = root / "index.db"

        for spec in sessions:
            cid = str(spec["id"])
            provider = str(spec.get("provider", "test"))
            builder = SessionBuilder(db_path, cid).provider(provider)

            if title := spec.get("title"):
                builder = builder.title(str(title))
            if metadata := spec.get("metadata"):
                builder = builder.metadata(_metadata_payload(metadata))
            if created_at := spec.get("created_at"):
                builder = builder.created_at(str(created_at))
            if branch_type := spec.get("branch_type"):
                builder = builder.branch_type(str(branch_type))
            if parent := spec.get("parent_session"):
                builder = builder.parent_session(str(parent))

            for msg in _message_specs(spec.get("messages")):
                builder = builder.add_message(
                    str(msg["id"]),
                    role=str(msg.get("role", "user")),
                    text=str(msg.get("text", "")),
                )

            builder.save()

        return root

    return _factory
