"""1vpm.1: the `delegations` view composes a parent-dispatched subagent edge
from session_links + the parent's Task dispatch action (`actions`) +
session_profiles (orchestrator/subagent model identity, cost, terminal
state). Not every subagent link is a delegation -- a link with no parent
Task action (e.g. a Codex async subagent with no dispatch action to join)
must surface with result_status='unknown', never a guessed ok/error."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_HASH = b"x" * 32


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _insert_session(
    conn: sqlite3.Connection,
    *,
    native_id: str,
    origin: str = "claude-code-session",
    created_at_ms: int = 1_767_225_600_000,
) -> str:
    conn.execute(
        """
        INSERT INTO sessions (
            native_id, origin, title, content_hash, created_at_ms, updated_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (native_id, origin, f"session {native_id}", _HASH, created_at_ms, created_at_ms + 1000),
    )
    return str(
        conn.execute(
            "SELECT session_id FROM sessions WHERE native_id = ? AND origin = ?", (native_id, origin)
        ).fetchone()["session_id"]
    )


def _insert_message(conn: sqlite3.Connection, *, session_id: str, native_id: str, position: int) -> str:
    conn.execute(
        """
        INSERT INTO messages (
            session_id, native_id, position, role, message_type, content_hash, occurred_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (session_id, native_id, position, "assistant", "message", _HASH, 1_767_225_600_000 + position),
    )
    return str(
        conn.execute(
            "SELECT message_id FROM messages WHERE session_id = ? AND native_id = ?", (session_id, native_id)
        ).fetchone()["message_id"]
    )


def _insert_session_profile(conn: sqlite3.Connection, *, session_id: str, **overrides: object) -> None:
    columns = {
        "session_id": session_id,
        **overrides,
    }
    keys = list(columns.keys())
    placeholders = ", ".join("?" for _ in keys)
    conn.execute(
        f"INSERT INTO session_profiles ({', '.join(keys)}) VALUES ({placeholders})",
        tuple(columns.values()),
    )


def _insert_session_link(
    conn: sqlite3.Connection,
    *,
    src_session_id: str,
    dst_origin: str,
    dst_native_id: str,
    resolved_dst_session_id: str | None,
    branch_point_message_id: str | None,
    link_type: str = "subagent",
    status: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO session_links (
            src_session_id, dst_origin, dst_native_id, link_type,
            resolved_dst_session_id, branch_point_message_id, status, observed_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            src_session_id,
            dst_origin,
            dst_native_id,
            link_type,
            resolved_dst_session_id,
            branch_point_message_id,
            status,
            1_767_225_600_000,
        ),
    )


def test_delegation_composes_dispatch_action_and_profiles(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")

    parent_id = _insert_session(conn, native_id="parent")
    child_id = _insert_session(conn, native_id="child")
    dispatch_message_id = _insert_message(conn, session_id=parent_id, native_id="dispatch", position=0)

    # The parent's Task dispatch action: tool_use (semantic_type=subagent) +
    # its paired tool_result, at the dispatch message.
    conn.execute(
        """
        INSERT INTO blocks (
            message_id, session_id, position, block_type, tool_name, tool_id, semantic_type, tool_input
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (dispatch_message_id, parent_id, 0, "tool_use", "Task", "task-1", "subagent", '{"prompt": "audit the thing"}'),
    )
    conn.execute(
        """
        INSERT INTO blocks (
            message_id, session_id, position, block_type, text, tool_id, tool_result_is_error, tool_result_exit_code
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (dispatch_message_id, parent_id, 1, "tool_result", "3 gaps found", "task-1", 0, 0),
    )

    _insert_session_profile(
        conn,
        session_id=parent_id,
        primary_model_name="claude-opus-4-8",
        primary_model_family="anthropic",
        terminal_state="clean_finish",
    )
    _insert_session_profile(
        conn,
        session_id=child_id,
        primary_model_name="claude-haiku-4-5",
        primary_model_family="anthropic",
        total_cost_usd=0.42,
        total_input_tokens=1000,
        total_output_tokens=500,
        total_cache_read_tokens=200,
        total_cache_write_tokens=50,
        wall_duration_ms=44_100,
        terminal_state="clean_finish",
    )

    _insert_session_link(
        conn,
        src_session_id=parent_id,
        dst_origin="claude-code-session",
        dst_native_id="child",
        resolved_dst_session_id=child_id,
        branch_point_message_id=dispatch_message_id,
    )

    row = conn.execute("SELECT * FROM delegations WHERE parent_session_id = ?", (parent_id,)).fetchone()
    assert row is not None
    assert row["child_session_id"] == child_id
    assert row["orchestrator_model"] == "claude-opus-4-8"
    assert row["orchestrator_model_family"] == "anthropic"
    assert row["orchestrator_origin"] == "claude-code-session"
    assert row["parent_terminal_state"] == "clean_finish"
    assert row["subagent_model"] == "claude-haiku-4-5"
    assert row["subagent_model_family"] == "anthropic"
    assert row["child_cost_usd"] == pytest.approx(0.42)
    assert row["child_tokens"] == 1000 + 500 + 200 + 50
    assert row["child_wall_ms"] == 44_100
    assert row["child_terminal_state"] == "clean_finish"
    assert row["instruction_payload"] == '{"prompt": "audit the thing"}'
    assert row["artifact_text"] == "3 gaps found"
    assert row["result_is_error"] == 0
    assert row["result_exit_code"] == 0
    assert row["result_status"] == "ok"


def test_delegation_result_status_error_when_dispatch_action_reports_error(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    parent_id = _insert_session(conn, native_id="parent")
    child_id = _insert_session(conn, native_id="child")
    dispatch_message_id = _insert_message(conn, session_id=parent_id, native_id="dispatch", position=0)
    conn.execute(
        """
        INSERT INTO blocks (
            message_id, session_id, position, block_type, tool_name, tool_id, semantic_type, tool_input
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (dispatch_message_id, parent_id, 0, "tool_use", "Task", "task-1", "subagent", "{}"),
    )
    conn.execute(
        """
        INSERT INTO blocks (
            message_id, session_id, position, block_type, text, tool_id, tool_result_is_error, tool_result_exit_code
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (dispatch_message_id, parent_id, 1, "tool_result", "boom", "task-1", 1, 1),
    )
    _insert_session_link(
        conn,
        src_session_id=parent_id,
        dst_origin="claude-code-session",
        dst_native_id="child",
        resolved_dst_session_id=child_id,
        branch_point_message_id=dispatch_message_id,
    )

    row = conn.execute("SELECT result_status FROM delegations WHERE parent_session_id = ?", (parent_id,)).fetchone()
    assert row["result_status"] == "error"


def test_delegation_result_status_unknown_when_no_dispatch_action(tmp_path: Path) -> None:
    """Codex async subagents/sidechains with no parent Task action: counted,
    never guessed -- result_status must be 'unknown', not silently 'ok'."""
    conn = _connect(tmp_path / "index.db")
    parent_id = _insert_session(conn, native_id="parent", origin="codex-session")
    child_id = _insert_session(conn, native_id="child", origin="codex-session")
    # No dispatch action block exists at all -- e.g. a Codex link with no
    # discoverable parent-side Task/subagent tool call.
    ghost_message_id = f"{parent_id}:no-such-message"

    _insert_session_link(
        conn,
        src_session_id=parent_id,
        dst_origin="codex-session",
        dst_native_id="child",
        resolved_dst_session_id=child_id,
        branch_point_message_id=ghost_message_id,
    )

    row = conn.execute(
        "SELECT result_status, instruction_block_id FROM delegations WHERE parent_session_id = ?", (parent_id,)
    ).fetchone()
    assert row is not None
    assert row["result_status"] == "unknown"
    assert row["instruction_block_id"] is None


def test_delegation_excludes_quarantined_links(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "index.db")
    parent_id = _insert_session(conn, native_id="parent")
    child_id = _insert_session(conn, native_id="child")
    ghost_message_id = f"{parent_id}:no-such-message"

    _insert_session_link(
        conn,
        src_session_id=parent_id,
        dst_origin="claude-code-session",
        dst_native_id="child",
        resolved_dst_session_id=child_id,
        branch_point_message_id=ghost_message_id,
        status="quarantined",
    )

    rows = conn.execute("SELECT * FROM delegations WHERE parent_session_id = ?", (parent_id,)).fetchall()
    assert rows == []


def test_delegation_excludes_non_subagent_link_types(tmp_path: Path) -> None:
    """A prefix-sharing continuation/fork link is not a delegation."""
    conn = _connect(tmp_path / "index.db")
    parent_id = _insert_session(conn, native_id="parent")
    child_id = _insert_session(conn, native_id="child")

    _insert_session_link(
        conn,
        src_session_id=parent_id,
        dst_origin="claude-code-session",
        dst_native_id="child",
        resolved_dst_session_id=child_id,
        branch_point_message_id=None,
        link_type="continuation",
    )

    rows = conn.execute("SELECT * FROM delegations WHERE parent_session_id = ?", (parent_id,)).fetchall()
    assert rows == []
