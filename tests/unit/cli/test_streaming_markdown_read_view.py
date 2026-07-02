from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.cli.read_views.streaming_markdown import stream_exact_session_markdown


def _seed_minimal_index(root: Path) -> None:
    conn = sqlite3.connect(root / "index.db")
    conn.executescript(
        """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            native_id TEXT,
            origin TEXT,
            title TEXT,
            updated_at_ms INTEGER
        );
        CREATE TABLE session_links (
            src_session_id TEXT,
            dst_origin TEXT,
            dst_native_id TEXT,
            link_type TEXT,
            resolved_dst_session_id TEXT,
            branch_point_message_id TEXT,
            inheritance TEXT
        );
        CREATE TABLE messages (
            message_id TEXT PRIMARY KEY,
            session_id TEXT,
            role TEXT,
            occurred_at_ms INTEGER
        );
        CREATE TABLE blocks (
            block_id TEXT PRIMARY KEY,
            message_id TEXT,
            position INTEGER,
            block_type TEXT,
            text TEXT,
            tool_name TEXT,
            tool_id TEXT,
            tool_input TEXT,
            language TEXT,
            semantic_type TEXT,
            tool_result_is_error INTEGER,
            tool_result_exit_code INTEGER
        );
        INSERT INTO sessions VALUES ('codex-session:abc', 'abc', 'codex-session', 'Large export', 2);
        INSERT INTO messages VALUES ('m1', 'codex-session:abc', 'user', 1000);
        INSERT INTO messages VALUES ('m2', 'codex-session:abc', 'assistant', 2000);
        INSERT INTO messages VALUES ('m3', 'codex-session:abc', 'assistant', 3000);
        INSERT INTO blocks VALUES ('b1', 'm1', 1, 'text', 'hello', NULL, NULL, NULL, NULL, NULL, NULL, NULL);
        INSERT INTO blocks VALUES ('b2', 'm2', 1, 'tool_use', NULL, 'shell', 'call-1', '{"command":"pytest"}', NULL, NULL, NULL, NULL);
        INSERT INTO blocks VALUES ('b3', 'm3', 1, 'tool_result', '1 passed', NULL, 'call-1', NULL, NULL, NULL, 0, 0);
        """
    )
    conn.commit()
    conn.close()


def test_stream_exact_session_markdown_writes_full_file(tmp_path: Path) -> None:
    _seed_minimal_index(tmp_path)
    out = tmp_path / "out.md"

    assert stream_exact_session_markdown(tmp_path, "codex-session:abc", out, prose_only=False)

    text = out.read_text(encoding="utf-8")
    assert "# Large export" in text
    assert "## user" in text
    assert "hello" in text
    assert "**Tool: shell**" in text
    assert "1 passed" in text


def test_stream_exact_session_markdown_prose_only_omits_tools(tmp_path: Path) -> None:
    _seed_minimal_index(tmp_path)
    out = tmp_path / "dialogue.md"

    assert stream_exact_session_markdown(tmp_path, "abc", out, prose_only=True)

    text = out.read_text(encoding="utf-8")
    assert "hello" in text
    assert "Tool: shell" not in text
    assert "1 passed" not in text


def test_stream_exact_session_markdown_defers_lineage_composition(tmp_path: Path) -> None:
    _seed_minimal_index(tmp_path)
    conn = sqlite3.connect(tmp_path / "index.db")
    conn.execute(
        """
        INSERT INTO session_links VALUES (
            'codex-session:abc',
            'codex-session',
            'parent',
            'branch',
            'codex-session:parent',
            'parent-message',
            'prefix-sharing'
        )
        """
    )
    conn.commit()
    conn.close()

    assert not stream_exact_session_markdown(tmp_path, "abc", tmp_path / "out.md", prose_only=False)


def test_stream_exact_session_markdown_without_session_links_streams(tmp_path: Path) -> None:
    _seed_minimal_index(tmp_path)
    conn = sqlite3.connect(tmp_path / "index.db")
    conn.execute("DROP TABLE session_links")
    conn.commit()
    conn.close()

    out = tmp_path / "out.md"

    assert stream_exact_session_markdown(tmp_path, "abc", out, prose_only=False)
