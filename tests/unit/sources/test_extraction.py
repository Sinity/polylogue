"""Seeded-db regressions for unified extraction output."""

from __future__ import annotations

import sqlite3

import pytest


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
