"""Tests for JSONL parsing in IngestionService._parse_raw_record.

Tests the fix for the bug where _parse_raw_record used json.loads() on raw
content, which failed for JSONL (multi-line newline-delimited JSON). The fix
makes it fall back to line-by-line parsing when single JSON parsing fails.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from polylogue.config import Config
from polylogue.pipeline.services.ingestion import IngestionService
from polylogue.storage.backends.sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.store import RawConversationRecord


class TestParseRawRecordJsonl:
    """Tests for IngestionService._parse_raw_record with JSONL and JSON inputs."""

    @pytest.fixture
    def backend(self, tmp_path: Path) -> SQLiteBackend:
        """Create a SQLiteBackend with a temp database."""
        db_path = tmp_path / "test.db"
        return SQLiteBackend(db_path=db_path)

    @pytest.fixture
    def repository(self, backend: SQLiteBackend) -> ConversationRepository:
        """Create a ConversationRepository with the test backend."""
        return ConversationRepository(backend=backend)

    @pytest.fixture
    def ingestion_service(
        self, tmp_path: Path, repository: ConversationRepository
    ) -> IngestionService:
        """Create an IngestionService for testing."""
        config = Config(
            sources=[],
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
        )
        return IngestionService(
            repository=repository,
            archive_root=tmp_path / "archive",
            config=config,
            drive_client_factory=None,
        )

    def test_parse_raw_record_single_json(
        self, ingestion_service: IngestionService
    ) -> None:
        """Single JSON document (ChatGPT format) parses correctly."""
        # ChatGPT export format: single JSON with title and mapping
        raw_content = b"""{
    "title": "Test Conversation",
    "mapping": {
        "node1": {
            "message": {
                "id": "msg-1",
                "author": {"role": "user"},
                "content": {"parts": ["Hello"], "content_type": "text"},
                "create_time": 1700000000
            },
            "parent": "root",
            "children": ["node2"]
        },
        "node2": {
            "message": {
                "id": "msg-2",
                "author": {"role": "assistant"},
                "content": {"parts": ["Hi"], "content_type": "text"},
                "create_time": 1700000001
            },
            "parent": "node1",
            "children": []
        }
    },
    "create_time": 1700000000,
    "update_time": 1700000001
}"""
        raw_record = RawConversationRecord(
            raw_id="chatgpt-single-json",
            provider_name="chatgpt",
            source_name="exports",
            source_path="/exports/conversations.json",
            source_index=0,
            raw_content=raw_content,
            acquired_at=datetime.now(timezone.utc).isoformat(),
        )

        # Should parse without error and return a conversation
        parsed = ingestion_service._parse_raw_record(raw_record)

        assert len(parsed) > 0
        assert parsed[0].provider_name == "chatgpt"
        assert parsed[0].title == "Test Conversation"
        # ChatGPT parser extracts messages from the mapping
        assert len(parsed[0].messages) == 2

    def test_parse_raw_record_jsonl(self, ingestion_service: IngestionService) -> None:
        """Multi-line JSONL (claude-code format) parses correctly and produces messages."""
        # Claude Code format: JSONL with messages as separate lines
        raw_content = b"""{"parentUuid":null,"isSidechain":false,"cwd":"/","sessionId":"test-session-1","version":"1.0.30","type":"user","message":{"role":"user","content":"Hello world"},"uuid":"msg-1","timestamp":"2025-06-20T11:34:16.232Z"}
{"parentUuid":"msg-1","isSidechain":false,"cwd":"/","sessionId":"test-session-1","version":"1.0.30","type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Hi there!"}]},"uuid":"msg-2","timestamp":"2025-06-20T11:34:20.000Z"}"""

        raw_record = RawConversationRecord(
            raw_id="claude-code-jsonl",
            provider_name="claude-code",
            source_name="claude_code_exports",
            source_path="/exports/session.jsonl",
            source_index=None,
            raw_content=raw_content,
            acquired_at=datetime.now(timezone.utc).isoformat(),
        )

        # Should parse without error and return a conversation
        parsed = ingestion_service._parse_raw_record(raw_record)

        assert len(parsed) > 0
        assert parsed[0].provider_name == "claude-code"
        # Claude Code parser groups all JSONL lines into one conversation
        # with messages from the payload
        assert len(parsed[0].messages) == 2
        # First message should be user
        assert parsed[0].messages[0].role == "user"
        assert "Hello" in parsed[0].messages[0].text
        # Second message should be assistant
        assert parsed[0].messages[1].role == "assistant"

    def test_parse_raw_record_jsonl_with_invalid_lines(
        self, ingestion_service: IngestionService
    ) -> None:
        """JSONL with some invalid lines skips them gracefully."""
        # Mix of valid and invalid JSON lines
        raw_content = b"""{"parentUuid":null,"type":"user","message":{"role":"user","content":"Valid line 1"},"uuid":"msg-1","timestamp":"2025-06-20T11:34:16Z"}
This is not JSON at all, should be skipped
{"parentUuid":"msg-1","type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Valid line 2"}]},"uuid":"msg-2","timestamp":"2025-06-20T11:34:20Z"}
{"malformed": "json"
{"parentUuid":"msg-2","type":"user","message":{"role":"user","content":"Valid line 3"},"uuid":"msg-3","timestamp":"2025-06-20T11:34:25Z"}"""

        raw_record = RawConversationRecord(
            raw_id="claude-code-mixed",
            provider_name="claude-code",
            source_name="claude_code_exports",
            source_path="/exports/session-with-errors.jsonl",
            source_index=None,
            raw_content=raw_content,
            acquired_at=datetime.now(timezone.utc).isoformat(),
        )

        # Should parse without error, skipping invalid lines
        parsed = ingestion_service._parse_raw_record(raw_record)

        assert len(parsed) > 0
        # Should have extracted the 3 valid lines (invalid ones skipped)
        # The claude-code parser groups them into one conversation
        assert parsed[0].provider_name == "claude-code"
        # Should have at least 2-3 messages from the valid lines
        assert len(parsed[0].messages) >= 2

    def test_orphan_raw_records_reparsed(
        self, backend: SQLiteBackend, repository: ConversationRepository, tmp_path: Path
    ) -> None:
        """Raw records without conversations are detected and re-parsed.

        This tests the orphaned raw records scenario from ingest_sources():
        When a raw record exists but the corresponding conversation was deleted
        or never parsed, it should be re-parsed.
        """
        config = Config(
            sources=[],
            archive_root=tmp_path / "archive",
            render_root=tmp_path / "render",
        )
        ingestion_service = IngestionService(
            repository=repository,
            archive_root=tmp_path / "archive",
            config=config,
            drive_client_factory=None,
        )

        # Store a raw record without a corresponding conversation
        raw_content = b"""{"parentUuid":null,"type":"user","message":{"role":"user","content":"Orphaned message"},"uuid":"msg-1","timestamp":"2025-06-20T11:34:16Z"}
{"parentUuid":"msg-1","type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Response"}]},"uuid":"msg-2","timestamp":"2025-06-20T11:34:20Z"}"""

        raw_record = RawConversationRecord(
            raw_id="orphaned-raw-001",
            provider_name="claude-code",
            source_name="orphaned_exports",
            source_path="/exports/orphaned.jsonl",
            source_index=None,
            raw_content=raw_content,
            acquired_at=datetime.now(timezone.utc).isoformat(),
        )

        # Save the raw record
        backend.save_raw_conversation(raw_record)

        # Verify it's stored
        stored_raw = backend.get_raw_conversation("orphaned-raw-001")
        assert stored_raw is not None
        assert stored_raw.provider_name == "claude-code"

        # Query for orphaned raw records (without conversations)
        # This is the pattern from ingest_sources()
        conn = backend._get_connection()
        orphaned_rows = conn.execute(
            """
            SELECT r.raw_id
            FROM raw_conversations r
            LEFT JOIN conversations c ON r.raw_id = c.raw_id
            WHERE c.conversation_id IS NULL
        """
        ).fetchall()

        # Should find the orphaned record
        orphaned_ids = [row["raw_id"] for row in orphaned_rows]
        assert "orphaned-raw-001" in orphaned_ids

        # Now parse it using ingest_from_raw with the orphaned ID
        result = ingestion_service.ingest_from_raw(raw_ids=["orphaned-raw-001"])

        # Should successfully parse and create a conversation
        assert result.counts["conversations"] > 0 or result.counts["messages"] > 0
        # Verify the conversation was created with raw_id link
        # Query directly for conversations with raw_id
        conn = backend._get_connection()
        linked_convos = conn.execute(
            """
            SELECT conversation_id, raw_id
            FROM conversations
            WHERE raw_id = ?
        """,
            ("orphaned-raw-001",),
        ).fetchall()
        assert (
            len(linked_convos) > 0
        ), "Orphaned raw record should be linked to created conversation"
