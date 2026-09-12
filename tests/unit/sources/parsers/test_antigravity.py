from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider, TitleSource
from polylogue.pipeline.ids import session_revision_projection
from polylogue.sources.parsers.antigravity import (
    AntigravitySessionSummary,
    _mark_active_leaf,
    looks_like_trajectory_db_path,
    parse_markdown_export,
    parse_trajectory_db,
)
from polylogue.sources.parsers.base import ParsedMessage


def test_parse_markdown_export_splits_known_sections() -> None:
    markdown = """# Chat Session

Note: _This is purely the output of the chat session._

### User Input

Run pytest.

### Planner Response

The focused checks passed.
"""
    summary = AntigravitySessionSummary(
        cascade_id="cascade-1",
        title="Focused checks",
        workspace_name="polylogue",
        snippet="Run pytest.",
        last_modified_time="2026-03-05T04:21:34Z",
    )

    session = parse_markdown_export(markdown, summary)

    assert session.source_name is Provider.ANTIGRAVITY
    assert session.provider_session_id == "cascade-1"
    assert session.title == "Focused checks"
    assert session.title_source is TitleSource.ORIGIN
    assert session.updated_at == "2026-03-05T04:21:34Z"
    assert [message.role for message in session.messages] == [Role.USER, Role.ASSISTANT]
    assert [message.text for message in session.messages] == [
        "Run pytest.",
        "The focused checks passed.",
    ]
    assert session.messages[0].blocks[0].type == BlockType.TEXT
    assert session.messages[0].provider_message_id.startswith("synthetic-")
    assert session.messages[1].provider_message_id.startswith("synthetic-")
    assert session.messages[0].provider_message_id != session.messages[1].provider_message_id
    assert [message.position for message in session.messages] == [0, 1]
    assert [message.variant_index for message in session.messages] == [0, 0]
    assert [message.is_active_path for message in session.messages] == [True, True]
    assert [message.is_active_leaf for message in session.messages] == [False, True]
    assert session.active_leaf_message_provider_id == session.messages[1].provider_message_id


def test_parse_markdown_export_reordering_keeps_synthetic_revision_identity() -> None:
    summary = AntigravitySessionSummary(cascade_id="cascade-order")
    forward = parse_markdown_export(
        "### User Input\n\nQuestion\n\n### Planner Response\n\nAnswer\n",
        summary,
    )
    reordered = parse_markdown_export(
        "### Planner Response\n\nAnswer\n\n### User Input\n\nQuestion\n",
        summary,
    )

    assert (
        session_revision_projection(forward).message_contents == session_revision_projection(reordered).message_contents
    )


def test_mark_active_leaf_flags_exactly_one_message_with_duplicate_ids() -> None:
    """bd polylogue-2hwl: a duplicate ``provider_message_id`` (a retried or
    regenerated section reusing the same id) must not produce more than one
    ``is_active_leaf=True`` message -- the pre-fix comparison
    (``message.provider_message_id == active_leaf_message_provider_id``)
    flagged every position sharing that id, not just the true final one.
    """
    messages = [
        ParsedMessage(provider_message_id="dup", role=Role.USER, text="first", position=0, is_active_path=True),
        ParsedMessage(provider_message_id="other", role=Role.ASSISTANT, text="middle", position=1, is_active_path=True),
        ParsedMessage(provider_message_id="dup", role=Role.ASSISTANT, text="final", position=2, is_active_path=True),
    ]

    marked = _mark_active_leaf(messages)

    leaves = [message for message in marked if message.is_active_leaf]
    assert len(leaves) == 1
    assert leaves[0].text == "final"


def test_parse_markdown_export_falls_back_to_single_export_message() -> None:
    markdown = """# Chat Session

Note: generated export

Unstructured transcript body.
"""
    summary = AntigravitySessionSummary(cascade_id="cascade-2")

    session = parse_markdown_export(markdown, summary)

    assert [message.role for message in session.messages] == [Role.ASSISTANT]
    assert session.messages[0].provider_message_id.startswith("synthetic-")
    assert session.messages[0].text == "Unstructured transcript body."
    assert session.messages[0].position == 0
    assert session.messages[0].is_active_leaf is True
    assert session.active_leaf_message_provider_id == session.messages[0].provider_message_id
    assert session.title_source is None


def test_parse_markdown_export_has_no_degraded_flag() -> None:
    """Language-server export sessions are whole transcripts — not fragmented.

    Markdown export sessions must NOT carry the brain-metadata fragment flag;
    they represent complete work sessions and must not be excluded from primary
    views by the same filter that suppresses brain-metadata fragments.
    """
    markdown = "### User Input\n\nRun checks.\n\n### Planner Response\n\nDone.\n"
    summary = AntigravitySessionSummary(
        cascade_id="cascade-clean",
        title="Clean session",
        last_modified_time="2026-04-01T12:00:00Z",
    )

    session = parse_markdown_export(markdown, summary)

    assert session.ingest_flags == []


def _trajectory_db(path: Path, *, malformed: bool = False) -> Path:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE trajectory_meta (trajectory_id TEXT, cascade_id TEXT);
        CREATE TABLE steps (
            idx INTEGER, step_type TEXT, step_format TEXT, step_payload TEXT,
            status TEXT, error_details TEXT
        );
        CREATE TABLE conversation_summaries (cascade_id TEXT, title TEXT, last_modified_time TEXT);
        CREATE TABLE parent_references (cascade_id TEXT, parent_id TEXT);
        """
    )
    connection.execute("INSERT INTO trajectory_meta VALUES (?, ?)", ("trajectory-1", "cascade-1"))
    connection.execute(
        "INSERT INTO conversation_summaries VALUES (?, ?, ?)",
        ("cascade-1", "Trajectory title", "2026-03-05T04:21:34Z"),
    )
    connection.execute("INSERT INTO parent_references VALUES (?, ?)", ("cascade-1", "parent-1"))
    connection.executemany(
        "INSERT INTO steps VALUES (?, ?, ?, ?, ?, ?)",
        [
            (0, "message", "v1", '{"role":"user","text":"hello"}', None, None),
            (1, "terminal_command", "v1", '{"tool_name":"shell","command":"printf hi"}', None, None),
            (2, "tool_result", "v1", '{"tool_name":"shell","output":"hi","status":"success"}', None, None),
            (3, "file_edit", "v1", '{"path":"README.md","old_string":"old","new_string":"new"}', None, None),
            (4, "future_step", "future", '{"opaque":true}' if not malformed else "not-json", None, None),
        ],
    )
    connection.commit()
    connection.close()
    return path


def test_trajectory_sqlite_parser_preserves_identity_order_tools_and_summary(tmp_path: Path) -> None:
    path = _trajectory_db(tmp_path / "conversation.db")

    assert looks_like_trajectory_db_path(path)
    sessions = list(parse_trajectory_db(path))

    assert len(sessions) == 1
    session = sessions[0]
    assert session.provider_session_id == "trajectory-1"
    assert session.provider_session_aliases == ["cascade-1"]
    assert session.title == "Trajectory title"
    assert [message.position for message in session.messages] == [0, 1, 2, 3]
    assert session.messages[1].blocks[0].tool_name == "shell"
    assert session.messages[1].blocks[0].tool_input == {"command": "printf hi"}
    result = session.messages[2].blocks[0]
    assert result.text == "hi"
    assert result.is_error is False
    assert session.messages[3].blocks[0].file_edit is not None
    assert any(event.event_type == "antigravity_parent_reference" for event in session.session_events)
    assert any(event.event_type == "antigravity_unmatched_parent_reference" for event in session.session_events)
    assert session.parent_session_provider_id == "parent-1"


def test_trajectory_sqlite_parser_refuses_known_text_with_unknown_step_format(tmp_path: Path) -> None:
    path = _trajectory_db(tmp_path / "conversation.db")
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE steps SET step_format = 'future-v9' WHERE idx = 0")

    session = list(parse_trajectory_db(path))[0]

    assert [message.text for message in session.messages] == [None, "hi", None]
    unsupported = [event for event in session.session_events if event.event_type == "antigravity_unsupported_step"]
    assert unsupported
    assert unsupported[0].payload["reason"] == "unsupported_step_format_or_type"
    assert "degraded:unsupported-trajectory-steps" in session.ingest_flags


def test_trajectory_sqlite_parser_empty_schema_is_attributable(tmp_path: Path) -> None:
    path = tmp_path / "empty.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE trajectory_meta (trajectory_id TEXT, cascade_id TEXT);
            CREATE TABLE steps (idx INTEGER, step_type TEXT, step_format TEXT, step_payload TEXT);
            """
        )

    sessions = list(parse_trajectory_db(path, fallback_id="empty"))

    assert len(sessions) == 1
    assert sessions[0].messages == []
    assert any(event.event_type == "antigravity_trajectory_empty" for event in sessions[0].session_events)


def test_trajectory_sqlite_parser_does_not_merge_unkeyed_multiple_trajectories(tmp_path: Path) -> None:
    path = _trajectory_db(tmp_path / "conversation.db")
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO trajectory_meta VALUES (?, ?)", ("trajectory-2", "cascade-2"))

    sessions = list(parse_trajectory_db(path))

    assert [session.provider_session_id for session in sessions] == ["trajectory-1", "trajectory-2"]
    assert all(session.messages == [] for session in sessions)
    assert all(
        any(event.event_type == "antigravity_unattributed_steps" for event in session.session_events)
        for session in sessions
    )


def test_trajectory_sqlite_parser_retains_unmatched_summary_as_metadata(tmp_path: Path) -> None:
    path = _trajectory_db(tmp_path / "conversation.db")
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO conversation_summaries VALUES (?, ?, ?)",
            ("orphan-cascade", "Orphan title", "2026-03-06T04:21:34Z"),
        )

    sessions = list(parse_trajectory_db(path))

    orphan = next(session for session in sessions if session.provider_session_id == "orphan-cascade")
    assert orphan.messages == []
    assert orphan.ingest_flags == ["degraded:unmatched-trajectory-summary"]
    event = next(event for event in orphan.session_events if event.event_type == "antigravity_unmatched_summary")
    assert event.payload["title"] == "Orphan title"


def test_trajectory_sqlite_parser_refuses_malformed_step_without_fabricating_text(tmp_path: Path) -> None:
    path = _trajectory_db(tmp_path / "conversation.db", malformed=True)

    session = list(parse_trajectory_db(path))[0]

    assert len(session.messages) == 4
    assert any(event.event_type == "antigravity_unsupported_step" for event in session.session_events)
    assert "degraded:unsupported-trajectory-steps" in session.ingest_flags
