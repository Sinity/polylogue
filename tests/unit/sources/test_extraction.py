"""Focused regression contracts for unified extraction bridges and stored viewport recovery."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from polylogue.schemas.unified import (
    bulk_harmonize,
    harmonize_parsed_message,
)


class TestHarmonizeParsedMessage:
    """Direct contracts for ParsedMessage -> HarmonizedMessage bridging."""

    def test_returns_none_for_missing_provider_meta(self) -> None:
        assert harmonize_parsed_message("claude-ai", None) is None

    def test_skips_non_message_claude_code_records(self) -> None:
        assert harmonize_parsed_message("claude-code", {"type": "progress"}) is None

    def test_overlays_database_fields_for_raw_payloads(self) -> None:
        provider_meta = {"raw": {"sender": "human", "text": ""}}

        msg = harmonize_parsed_message(
            "claude-ai",
            provider_meta,
            message_id="db-id",
            role="user",
            text="db text",
            timestamp="2024-01-15T10:30:00Z",
        )

        assert msg is not None
        assert msg.id == "db-id"
        assert msg.role == "user"
        assert msg.text == "db text"
        assert msg.timestamp is not None
        assert msg.provider == "claude"

    def test_falls_back_to_provider_meta_harmonization_for_malformed_payloads(self) -> None:
        provider_meta = {
            "sender": "assistant",
            "text": "Fallback text",
            "created_at": "2024-01-15T10:30:00Z",
        }

        msg = harmonize_parsed_message("claude-ai", provider_meta)

        assert msg is not None
        assert msg.role == "assistant"
        assert msg.text == "Fallback text"
        assert msg.timestamp is not None


class TestBulkHarmonize:
    """Contracts for bulk ParsedMessage harmonization."""

    def test_skips_entries_without_provider_meta_or_non_messages(self) -> None:
        parsed_messages = [
            SimpleNamespace(provider_meta=None),
            SimpleNamespace(provider_meta={"type": "progress"}),
            SimpleNamespace(
                provider_meta={"sender": "assistant", "text": "Kept"},
                provider_message_id="kept-id",
                role="assistant",
                text="Kept",
                timestamp="2024-01-15T10:30:00Z",
            ),
        ]

        result = bulk_harmonize("claude-code", parsed_messages)

        assert len(result) == 1
        assert result[0].id == "kept-id"
        assert result[0].role == "assistant"
        assert result[0].text == "Kept"

    def test_passes_database_context_through_to_harmonized_messages(self) -> None:
        parsed_messages = [
            SimpleNamespace(
                provider_meta={"sender": "human", "text": ""},
                provider_message_id="db-id",
                role="user",
                text="db text",
                timestamp="2024-01-15T10:30:00Z",
            )
        ]

        result = bulk_harmonize("claude-ai", parsed_messages)

        assert len(result) == 1
        assert result[0].id == "db-id"
        assert result[0].role == "user"
        assert result[0].text == "db text"
        assert result[0].timestamp is not None


class TestDatabaseIntegration:
    """Integration checks using the seeded synthetic database."""

    @pytest.mark.parametrize("provider", ["claude-code", "chatgpt", "codex"])
    def test_messages_have_content_blocks(self, seeded_db, provider: str) -> None:
        conn = sqlite3.connect(seeded_db)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT m.message_id, m.role, m.text
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.conversation_id
            WHERE c.provider_name = ?
            LIMIT 20
            """,
            (provider,),
        )
        rows = cur.fetchall()
        conn.close()

        assert rows, f"No {provider} messages in seeded database"
        for msg_id, role, text in rows:
            assert role in ("user", "assistant", "system", "tool"), (
                f"Unexpected role {role!r} in {provider} message {msg_id}"
            )
            assert isinstance(text, (str, type(None)))

    def test_claude_code_has_tool_use_blocks(self, seeded_db) -> None:
        conn = sqlite3.connect(seeded_db)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT cb.type, cb.tool_name, cb.semantic_type
            FROM content_blocks cb
            JOIN messages m ON cb.message_id = m.message_id
            JOIN conversations c ON m.conversation_id = c.conversation_id
            WHERE c.provider_name = 'claude-code'
              AND cb.type = 'tool_use'
            LIMIT 100
            """
        )
        rows = cur.fetchall()
        conn.close()

        assert rows
        for block_type, tool_name, semantic_type in rows:
            assert block_type == "tool_use"
            assert tool_name is not None
            assert semantic_type is None or isinstance(semantic_type, str)


class TestExtractedProviderMetaFallbackText:
    """Database-backed content block semantics still deserve a seeded regression check."""

    def test_content_blocks_have_semantic_types(self, seeded_db) -> None:
        conn = sqlite3.connect(seeded_db)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT semantic_type, COUNT(*) as cnt
            FROM content_blocks
            WHERE semantic_type IS NOT NULL
            GROUP BY semantic_type
            ORDER BY cnt DESC
            """
        )
        rows = cur.fetchall()
        conn.close()

        assert rows
        known_types = {
            "file_read",
            "file_write",
            "file_edit",
            "shell",
            "git",
            "search",
            "web",
            "agent",
            "subagent",
            "thinking",
            "code",
            "other",
        }
        for sem_type, _count in rows:
            assert sem_type in known_types, f"Unknown semantic_type {sem_type!r}"
