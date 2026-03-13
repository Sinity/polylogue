"""Pinned parsing-service regressions for raw-record dispatch."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from polylogue.config import Config
from polylogue.pipeline.services.parsing import ParsingService
from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.backends.connection import open_connection
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.store import RawConversationRecord


@pytest.fixture
def backend(tmp_path: Path) -> SQLiteBackend:
    return SQLiteBackend(db_path=tmp_path / "test.db")


@pytest.fixture
def repository(backend: SQLiteBackend) -> ConversationRepository:
    return ConversationRepository(backend=backend)


@pytest.fixture
def parsing_service(tmp_path: Path, repository: ConversationRepository) -> ParsingService:
    config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
    return ParsingService(repository=repository, archive_root=tmp_path / "archive", config=config)


def _raw_record(*, raw_id: str, provider: str, path: str, raw_content: bytes, source_name: str = "exports", source_index: int | None = None) -> RawConversationRecord:
    return RawConversationRecord(
        raw_id=raw_id,
        provider_name=provider,
        source_name=source_name,
        source_path=path,
        source_index=source_index,
        raw_content=raw_content,
        acquired_at=datetime.now(timezone.utc).isoformat(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("record", "expected_count", "expected_provider", "expected_ids"),
    [
        (
            _raw_record(
                raw_id="chatgpt-single-json",
                provider="chatgpt",
                path="/exports/conversations.json",
                source_index=0,
                raw_content=b'''{
    "title": "Test Conversation",
    "mapping": {
        "node1": {"message": {"id": "msg-1", "author": {"role": "user"}, "content": {"parts": ["Hello"], "content_type": "text"}, "create_time": 1700000000}, "parent": "root", "children": ["node2"]},
        "node2": {"message": {"id": "msg-2", "author": {"role": "assistant"}, "content": {"parts": ["Hi"], "content_type": "text"}, "create_time": 1700000001}, "parent": "node1", "children": []}
    },
    "create_time": 1700000000,
    "update_time": 1700000001
}''',
            ),
            1,
            "chatgpt",
            {"chatgpt-single-json"},
        ),
        (
            _raw_record(
                raw_id="chatgpt-bundle-json",
                provider="chatgpt",
                path="/exports/conversations.json",
                raw_content=b'''[
  {"id": "conv-1", "title": "Conversation One", "mapping": {"m1": {"id": "m1", "message": {"id": "m1", "author": {"role": "user"}, "content": {"parts": ["Hello 1"], "content_type": "text"}, "create_time": 1700000000}, "children": []}}},
  {"id": "conv-2", "title": "Conversation Two", "mapping": {"m2": {"id": "m2", "message": {"id": "m2", "author": {"role": "user"}, "content": {"parts": ["Hello 2"], "content_type": "text"}, "create_time": 1700000100}, "children": []}}}
]''',
            ),
            2,
            "chatgpt",
            {"conv-1", "conv-2"},
        ),
        (
            _raw_record(
                raw_id="claude-code-jsonl",
                provider="claude-code",
                path="/exports/session.jsonl",
                raw_content=b'{"parentUuid":null,"isSidechain":false,"cwd":"/","sessionId":"test-session-1","version":"1.0.30","type":"user","message":{"role":"user","content":"Hello world"},"uuid":"msg-1","timestamp":"2025-06-20T11:34:16.232Z"}\n{"parentUuid":"msg-1","isSidechain":false,"cwd":"/","sessionId":"test-session-1","version":"1.0.30","type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Hi there!"}]},"uuid":"msg-2","timestamp":"2025-06-20T11:34:20.000Z"}',
            ),
            1,
            "claude-code",
            {"test-session-1"},
        ),
    ],
    ids=["chatgpt-single", "chatgpt-bundle", "claude-code-jsonl"],
)
async def test_parse_raw_record_dispatch_contract(
    parsing_service: ParsingService,
    record: RawConversationRecord,
    expected_count: int,
    expected_provider: str,
    expected_ids: set[str],
) -> None:
    parsed = await parsing_service._parse_raw_record(record)
    assert len(parsed) == expected_count
    assert {conversation.provider_name for conversation in parsed} == {expected_provider}
    actual_ids = {conversation.provider_conversation_id for conversation in parsed}
    assert actual_ids == expected_ids
    assert all(conversation.messages for conversation in parsed)


@pytest.mark.asyncio
async def test_parse_raw_record_infers_provider_from_generic_bundle_source(parsing_service: ParsingService) -> None:
    record = _raw_record(
        raw_id="chatgpt-bundle-generic",
        provider="test-inbox",
        source_name="test-inbox",
        path="/exports/conversations.json",
        raw_content=b'''[{"id": "conv-generic-1", "title": "Generic One", "mapping": {"m1": {"id": "m1", "message": {"id": "m1", "author": {"role": "user"}, "content": {"parts": ["hello"], "content_type": "text"}}, "children": []}}}]''',
    )

    parsed = await parsing_service._parse_raw_record(record)
    assert len(parsed) == 1
    assert record.payload_provider == "chatgpt"
    assert parsed[0].provider_name == "chatgpt"
    assert parsed[0].provider_conversation_id == "conv-generic-1"


@pytest.mark.asyncio
async def test_parse_raw_record_jsonl_skips_invalid_lines(parsing_service: ParsingService) -> None:
    record = _raw_record(
        raw_id="claude-code-mixed",
        provider="claude-code",
        path="/exports/session-with-errors.jsonl",
        raw_content=b'{"parentUuid":null,"type":"user","message":{"role":"user","content":"Valid line 1"},"uuid":"msg-1","timestamp":"2025-06-20T11:34:16Z"}\nThis is not JSON at all, should be skipped\n{"parentUuid":"msg-1","type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Valid line 2"}]},"uuid":"msg-2","timestamp":"2025-06-20T11:34:20Z"}\n{"malformed": "json"\n{"parentUuid":"msg-2","type":"user","message":{"role":"user","content":"Valid line 3"},"uuid":"msg-3","timestamp":"2025-06-20T11:34:25Z"}',
    )

    parsed = await parsing_service._parse_raw_record(record)
    assert len(parsed) == 1
    assert parsed[0].provider_name == "claude-code"
    assert len(parsed[0].messages) >= 2


@pytest.mark.asyncio
async def test_parse_from_raw_reparses_orphaned_raw_records(
    backend: SQLiteBackend,
    repository: ConversationRepository,
    tmp_path: Path,
) -> None:
    config = Config(sources=[], archive_root=tmp_path / "archive", render_root=tmp_path / "render")
    parsing_service = ParsingService(repository=repository, archive_root=tmp_path / "archive", config=config)
    raw_record = _raw_record(
        raw_id="orphaned-raw-001",
        provider="claude-code",
        source_name="orphaned_exports",
        path="/exports/orphaned.jsonl",
        raw_content=b'{"parentUuid":null,"type":"user","message":{"role":"user","content":"Orphaned message"},"uuid":"msg-1","timestamp":"2025-06-20T11:34:16Z"}\n{"parentUuid":"msg-1","type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Response"}]},"uuid":"msg-2","timestamp":"2025-06-20T11:34:20Z"}',
    )

    await backend.save_raw_conversation(raw_record)

    with open_connection(backend.db_path) as conn:
        orphaned_rows = conn.execute(
            """
            SELECT r.raw_id
            FROM raw_conversations r
            LEFT JOIN conversations c ON r.raw_id = c.raw_id
            WHERE c.conversation_id IS NULL
            """
        ).fetchall()
    assert [row["raw_id"] for row in orphaned_rows] == ["orphaned-raw-001"]

    result = await parsing_service.parse_from_raw(raw_ids=["orphaned-raw-001"])
    assert result.counts["conversations"] > 0 or result.counts["messages"] > 0

    with open_connection(backend.db_path) as conn:
        linked = conn.execute(
            "SELECT conversation_id FROM conversations WHERE raw_id = ?",
            ("orphaned-raw-001",),
        ).fetchall()
    assert linked
