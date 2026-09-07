"""Compaction boundary columns survive parse and write (polylogue-4ts.5).

The boundary range and the materialized summary are only useful if they reach
``session_events``. These tests run the real Claude Code and Codex parsers over
the record shapes those providers actually write, hand the result to the
production writer, and read the stored columns back.

Anti-vacuity: the Claude case uses the ``{type, summary, leafUuid}`` record --
the shape with the text at the top level. A parser reading only
``message.content`` produces an empty summary, which leaves
``boundary_message_position`` unset, which stores ``boundary_message_id`` NULL:
the exact chain that made effective-context reads fall back to the full
transcript for the origin that dominates the archive.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.sources.parsers.claude.code_parser import parse_code
from polylogue.sources.parsers.codex import parse as parse_codex
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _stored_compactions(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT session_id, boundary_start_position, boundary_end_position, boundary_message_id
        FROM session_events
        WHERE event_type = 'compaction'
        ORDER BY position
        """
    ).fetchall()


def test_claude_code_compaction_stores_the_boundary_range_and_summary(tmp_path: Path) -> None:
    payload: list[object] = [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": "boundary-claude",
            "timestamp": "2026-01-01T10:00:00Z",
            "message": {"role": "user", "content": "first ask"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "sessionId": "boundary-claude",
            "timestamp": "2026-01-01T10:00:01Z",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "first answer"}]},
        },
        # The record Claude Code actually writes: text at the top level.
        {
            "type": "summary",
            "summary": "compacted the conversation so far",
            "leafUuid": "8f2c1d04-0000-4000-8000-000000000001",
        },
        {
            "type": "user",
            "uuid": "u2",
            "sessionId": "boundary-claude",
            "timestamp": "2026-01-01T10:00:03Z",
            "message": {"role": "user", "content": "post-compaction ask"},
        },
    ]

    parsed = parse_code(payload, "boundary-claude")
    assert [event.boundary_start_position for event in parsed.session_events] == [0]
    assert [event.boundary_end_position for event in parsed.session_events] == [1]

    conn = _connect(tmp_path / "index.db")
    session_id = write_parsed_session_to_archive(conn, parsed)
    conn.commit()

    rows = _stored_compactions(conn)
    assert len(rows) == 1
    row = rows[0]
    assert row["session_id"] == session_id
    assert row["boundary_start_position"] == 0
    assert row["boundary_end_position"] == 1
    assert row["boundary_message_id"] is not None
    summary_text = conn.execute(
        "SELECT text FROM blocks WHERE message_id = ? ORDER BY position", (row["boundary_message_id"],)
    ).fetchall()
    assert [block["text"] for block in summary_text] == ["compacted the conversation so far"]


def test_codex_compaction_stores_the_boundary_range_and_summary(tmp_path: Path) -> None:
    payload: list[object] = [
        {
            "timestamp": "2026-01-01T10:00:00Z",
            "type": "response_item",
            "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "first ask"}]},
        },
        {
            "timestamp": "2026-01-01T10:00:01Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "first answer"}],
            },
        },
        {
            "timestamp": "2026-01-01T10:00:02Z",
            "type": "compacted",
            "payload": {"message": "compacted the conversation so far"},
        },
        {
            "timestamp": "2026-01-01T10:00:03Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "post-compaction ask"}],
            },
        },
    ]

    parsed = parse_codex(payload, "boundary-codex")
    compactions = [event for event in parsed.session_events if event.event_type == "compaction"]
    assert compactions, "codex parser produced no compaction event"
    assert compactions[0].boundary_start_position is not None
    assert compactions[0].boundary_end_position is not None

    conn = _connect(tmp_path / "index.db")
    write_parsed_session_to_archive(conn, parsed)
    conn.commit()

    rows = _stored_compactions(conn)
    assert len(rows) == 1
    assert rows[0]["boundary_start_position"] is not None
    assert rows[0]["boundary_end_position"] is not None
    assert rows[0]["boundary_message_id"] is not None
