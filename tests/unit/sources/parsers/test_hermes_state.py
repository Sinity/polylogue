"""Hermes ``state.db`` tool-outcome extraction contracts.

Hermes stores tool results as a JSON envelope in ``messages.content``
(``{"output": ...}`` plus ``exit_code`` / ``success`` / ``error`` depending on
tool family) rather than dedicated outcome columns. These tests pin the
mapping from that envelope onto ``ParsedContentBlock.is_error`` /
``exit_code``, verified against real shapes observed in a live Hermes
``state.db`` (see ``polylogue-uwlu``).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.core.enums import BlockType, TitleSource
from polylogue.sources.parsers.base import ParsedContentBlock
from polylogue.sources.parsers.hermes_state import parse_state_db


def _write_state_db(path: Path, *, tool_contents: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version(version) VALUES (16);
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT,
                model TEXT,
                model_config TEXT,
                parent_session_id TEXT,
                started_at REAL,
                ended_at REAL,
                title TEXT
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_name TEXT,
                tool_calls TEXT,
                timestamp REAL NOT NULL,
                observed INTEGER DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1,
                compacted INTEGER NOT NULL DEFAULT 0
            );
            INSERT INTO sessions (id, source, model, model_config, parent_session_id, started_at, ended_at, title)
            VALUES ('s1', 'hermes', 'test-model', '{}', NULL, 1775000000.0, 1775000010.0, 'Outcome fixture');
            """
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, 'user', 'go', ?)",
            ("s1", 1775000001.0),
        )
        for index, content in enumerate(tool_contents):
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, tool_call_id, tool_name, timestamp)
                VALUES ('s1', 'tool', ?, ?, 'shell', ?)
                """,
                (content, f"call-{index}", 1775000002.0 + index),
            )
        conn.commit()


def _tool_result_blocks(path: Path, *, tool_contents: list[str]) -> list[ParsedContentBlock]:
    _write_state_db(path, tool_contents=tool_contents)
    sessions = parse_state_db(path)
    assert len(sessions) == 1
    return [
        block for message in sessions[0].messages for block in message.blocks if block.type is BlockType.TOOL_RESULT
    ]


def test_exit_code_zero_is_not_an_error(tmp_path: Path) -> None:
    blocks = _tool_result_blocks(tmp_path / "state.db", tool_contents=[json.dumps({"output": "ok", "exit_code": 0})])
    assert blocks[0].is_error is False
    assert blocks[0].exit_code == 0


def test_state_db_explicit_title_is_provider_provenance(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    _write_state_db(path, tool_contents=[json.dumps({"output": "ok", "exit_code": 0})])

    sessions = parse_state_db(path)

    assert sessions[0].title == "Outcome fixture"
    assert sessions[0].title_source is TitleSource.ORIGIN


def test_nonzero_exit_code_is_an_error(tmp_path: Path) -> None:
    blocks = _tool_result_blocks(
        tmp_path / "state.db", tool_contents=[json.dumps({"output": "boom", "exit_code": 127})]
    )
    assert blocks[0].is_error is True
    assert blocks[0].exit_code == 127


def test_success_true_is_not_an_error(tmp_path: Path) -> None:
    blocks = _tool_result_blocks(tmp_path / "state.db", tool_contents=[json.dumps({"output": "done", "success": True})])
    assert blocks[0].is_error is False
    assert blocks[0].exit_code is None


def test_success_false_with_error_message_is_an_error(tmp_path: Path) -> None:
    blocks = _tool_result_blocks(
        tmp_path / "state.db",
        tool_contents=[json.dumps({"output": "", "success": False, "error": "not found"})],
    )
    assert blocks[0].is_error is True
    assert blocks[0].exit_code is None


def test_bare_error_message_with_no_exit_code_or_success_is_an_error(tmp_path: Path) -> None:
    blocks = _tool_result_blocks(
        tmp_path / "state.db", tool_contents=[json.dumps({"output": None, "error": "invalid sort: recent"})]
    )
    assert blocks[0].is_error is True
    assert blocks[0].exit_code is None


def test_plain_output_with_no_outcome_signal_is_unknown_not_guessed(tmp_path: Path) -> None:
    blocks = _tool_result_blocks(tmp_path / "state.db", tool_contents=[json.dumps({"output": "just text"})])
    assert blocks[0].is_error is None
    assert blocks[0].exit_code is None


def test_exit_code_and_error_together_prefer_error_but_keep_exit_code(tmp_path: Path) -> None:
    blocks = _tool_result_blocks(
        tmp_path / "state.db",
        tool_contents=[json.dumps({"output": "boom", "exit_code": 1, "error": "denied"})],
    )
    assert blocks[0].is_error is True
    assert blocks[0].exit_code == 1


def _write_reasoning_state_db(path: Path) -> None:
    """A minimal state.db with a reasoning-bearing assistant message.

    ``reasoning_details``/``codex_reasoning_items``/``codex_message_items``
    are Hermes's captured Codex-native reasoning-trace evidence
    (bd polylogue-9x22): the ``blocks`` table has no metadata column, so this
    would be silently dropped at write time if it only reached
    ``ParsedContentBlock.metadata``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version(version) VALUES (16);
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT,
                model TEXT,
                model_config TEXT,
                parent_session_id TEXT,
                started_at REAL,
                ended_at REAL,
                title TEXT
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_name TEXT,
                tool_calls TEXT,
                reasoning_content TEXT,
                reasoning_details TEXT,
                codex_reasoning_items TEXT,
                codex_message_items TEXT,
                timestamp REAL NOT NULL,
                observed INTEGER DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1,
                compacted INTEGER NOT NULL DEFAULT 0
            );
            INSERT INTO sessions (id, source, model, model_config, parent_session_id, started_at, ended_at, title)
            VALUES ('s1', 'hermes', 'test-model', '{}', NULL, 1775000000.0, 1775000010.0, 'Reasoning fixture');
            """
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, 'user', 'solve it', ?)",
            ("s1", 1775000001.0),
        )
        conn.execute(
            """
            INSERT INTO messages (
                session_id, role, content, reasoning_content, reasoning_details,
                codex_reasoning_items, codex_message_items, timestamp
            )
            VALUES ('s1', 'assistant', 'done', 'thinking it through', ?, ?, ?, 1775000002.0)
            """,
            (
                json.dumps([{"type": "text", "text": "step one"}]),
                json.dumps(["reasoning-item-1"]),
                json.dumps(["message-item-1"]),
            ),
        )
        conn.commit()


def test_reasoning_evidence_routes_to_session_events_not_only_block_metadata(tmp_path: Path) -> None:
    """Production route: sqlite state.db row -> ``parse_state_db`` -> ``session_events``.

    This fails if ``_reasoning_evidence_events`` (or its wiring into
    ``_parse_session_row``) is removed, since the reasoning payload would
    then only live on ``ParsedContentBlock.metadata`` -- a field the archive
    write path never persists (only ``language`` is read back out of it).
    """
    path = tmp_path / "state.db"
    _write_reasoning_state_db(path)
    sessions = parse_state_db(path)
    assert len(sessions) == 1
    session = sessions[0]

    thinking_blocks = [
        block for message in session.messages for block in message.blocks if block.type is BlockType.THINKING
    ]
    assert len(thinking_blocks) == 1
    assert thinking_blocks[0].text == "thinking it through"

    reasoning_events = [event for event in session.session_events if event.event_type == "hermes_reasoning_evidence"]
    assert len(reasoning_events) == 1
    event = reasoning_events[0]
    assert event.payload == {
        "reasoning_details": [{"type": "text", "text": "step one"}],
        "codex_reasoning_items": ["reasoning-item-1"],
        "codex_message_items": ["message-item-1"],
    }
    assistant_message = next(message for message in session.messages if message.role.value == "assistant")
    assert event.source_message_provider_id == assistant_message.provider_message_id
