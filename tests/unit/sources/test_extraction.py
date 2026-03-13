"""Seeded database regressions for extracted semantic storage."""

from __future__ import annotations

import sqlite3

import pytest


@pytest.mark.parametrize("provider", ["claude-code", "chatgpt", "codex"])
def test_seeded_messages_have_expected_role_and_text_shapes(seeded_db, provider: str) -> None:
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
    assert all(role in ("user", "assistant", "system", "tool") for _msg_id, role, _text in rows)
    assert all(isinstance(text, (str, type(None))) for _msg_id, _role, text in rows)


def test_seeded_claude_code_tool_use_blocks_have_names(seeded_db) -> None:
    conn = sqlite3.connect(seeded_db)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT cb.type, cb.tool_name, cb.semantic_type
        FROM content_blocks cb
        JOIN messages m ON cb.message_id = m.message_id
        JOIN conversations c ON m.conversation_id = c.conversation_id
        WHERE c.provider_name = 'claude-code' AND cb.type = 'tool_use'
        LIMIT 100
        """
    )
    rows = cur.fetchall()
    conn.close()

    assert rows
    assert all(block_type == "tool_use" and tool_name for block_type, tool_name, _semantic_type in rows)
    assert all(semantic_type is None or isinstance(semantic_type, str) for _block_type, _tool_name, semantic_type in rows)


def test_seeded_content_blocks_use_only_known_semantic_types(seeded_db) -> None:
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
    assert rows
    assert {semantic_type for semantic_type, _count in rows} <= known_types
